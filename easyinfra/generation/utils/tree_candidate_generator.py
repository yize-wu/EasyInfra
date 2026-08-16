import torch
from typing import TYPE_CHECKING, Dict, Optional, Tuple, Any
import copy

from transformers.generation.logits_process import LogitsProcessorList

# from ..cache_utils import TreeDynamicCache

from .candidate_generator import (
    AssistedCandidateGenerator,
    _prepare_2d_attention_mask,
    get_prefill_candidate_refill,
    _crop_past_key_values,
)

from ...generation.utils.utils import GenerateDecoderOnlyTreeOutput
from .configuration_utils import GenerationConfig
from ...generation.parallel.parallel_utils import get_global_rank
from ...utils import rank0_print
    
class TreeAssistedCandidateGenerator(AssistedCandidateGenerator):
    def __init__(
        self,
        assistant_model,
        candidate_refill_policy,
        generation_config: GenerationConfig,
        model_kwargs: Dict,
        logits_processor = None,
        num_assistant_tokens: Optional[int] = None,
        **kwargs
    ):
        super().__init__(
            assistant_model,
            candidate_refill_policy,
            generation_config,
            model_kwargs,
            logits_processor,
            num_assistant_tokens,
            **kwargs
        )
        self.last_candidate_tree_depth = 0
    
    def get_candidates(self, 
                       input_ids: torch.LongTensor,
                       valid_retrieve_indices: Optional[torch.LongTensor],
                       ) -> Tuple[torch.LongTensor, Optional[torch.FloatTensor]]:
        """
            Returns candidate_flat_ids, retrieve_indices, candidate_flat_probs, run_candidate_once.
                input_ids: [bsz, num_input_ids]
                valid_retrieve_indices: for cache cropping
        """
        # tree candidator needs to record last time input length, for cache cropping
        # Don't generate more than `max_length - 1` candidates since the target model generates one extra token.
        
        new_cur_len = input_ids.shape[-1]
        max_new_tokens, min_new_tokens = self._get_max_min_new_tokens(new_cur_len)
        run_candidate_once = False
        if max_new_tokens == 0:
            return input_ids, None, None, run_candidate_once

        run_candidate_once = True
        if not self.assistant_model.parallel_config.in_participant:
            return None, None, None, run_candidate_once
        
        # 1. If it is not the first round of candidate generation, prepare the inputs based on the input_ids length
        # (which implicitly contains the number of accepted candidates from the previous round)
        has_past_key_values = self.assistant_kwargs.get("past_key_values", None) is not None
             
        if has_past_key_values:
            if valid_retrieve_indices is None:
                raise ValueError
            # cache size has no last-layer tokens' kv, so we should ensure the crop depth= tree_depth - 1
            _valid_path_length = max(0, min(self.last_candidate_tree_depth-1, valid_retrieve_indices.shape[-1]-1))
            # rank0_print(f"valid_path length: {_valid_path_length}")
            valid_retrieve_indices = valid_retrieve_indices.narrow(dim=-1, start=0, length=_valid_path_length)
            
            # compute new cache size
            if self.next_candidate_refill:
                # TODO: it is a buggy code when no bound layers and EVERY_MISMATCH
                seq_cache_length = self.assistant_kwargs["past_key_values"].get_seq_length()
                new_cache_size = seq_cache_length - self.non_leaf_flat_candidate_length - self.fuzzy_length
                # create a None indices to avoid indexing from cropped part
                valid_retrieve_indices = None
                self.fuzzy_length = 0
            else:
                self.fuzzy_length += _valid_path_length + 2 # bonus token and last-layer candidate token
                new_cache_size = new_cur_len - 2 # without bonus token and last-layer candidate token
                # if get_global_rank() == 0:
                #     rank0_print(f"new cache size: {new_cache_size}")

            # For processes who are not participant of assistant model, it will not go here,
            self.assistant_kwargs["past_key_values"] = _crop_past_key_values(
                self.assistant_kwargs["past_key_values"], new_cache_size, valid_retrieve_indices
            )
            self.assistant_kwargs["attention_mask_2d"] = self.assistant_kwargs["attention_mask"] # tree usage
            self.assistant_kwargs = _prepare_2d_attention_mask(
                self.assistant_kwargs, new_cur_len
            )
            # self.assistant_kwargs = _prepare_token_type_ids(self.assistant_kwargs, new_cur_len)

        # 2. Forecast next N tokens using the assistant model.
        assistant_generation_kwargs = {
            self.input_ids_key: input_ids,
            "min_new_tokens": min_new_tokens,
            "max_new_tokens": max_new_tokens,
            "generation_config": self.generation_config,
            "logits_processor": self.logits_processor,
            "candidate_refill": self.next_candidate_refill,
        }

        assistant_output: GenerateDecoderOnlyTreeOutput = self.assistant_model.generate(**assistant_generation_kwargs, **self.assistant_kwargs)


        # if assistant_output is None, then you must not be the participant of the draft model
        if assistant_output is None:
            raise ValueError("participant return None for draft model.")
        
        candidate_ids = assistant_output.tree_draft_tokens
        retrieve_indices = assistant_output.retrieve_indices
        
        if assistant_output.scores is not None and assistant_output.scores[0] is not None:
            candidate_flat_probs = torch.cat(assistant_output.scores, dim=1)
        else:
            candidate_flat_probs = None
        self.assistant_kwargs["past_key_values"] = assistant_output.past_key_values
        
        leaf_candidate_length = self.generation_config.tree_attention_config.top_k # TODO: if tree pruning?
        self.non_leaf_flat_candidate_length = assistant_output.tree_draft_tokens.shape[-1] - input_ids.shape[-1] - leaf_candidate_length
        self.last_candidate_tree_depth = retrieve_indices.shape[-1]
        
        return candidate_ids, retrieve_indices, candidate_flat_probs, run_candidate_once
    
