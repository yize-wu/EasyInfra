import torch
from typing import TYPE_CHECKING, Dict, Optional, Tuple, Any
import copy
import torch.distributed as dist

from ...generation.utils.logits_process import LogitsProcessorList
from ..cache_utils import TreeDynamicCache
from .configuration_utils import GenerationConfig
# from ..cache_utils import TreeDynamicCache

from ...generation.parallel.parallel_utils import get_global_rank

def _crop_past_key_values(past_key_values: TreeDynamicCache, max_length: int, valid_retrieve_indices: Optional[torch.Tensor] = None):
    """Crops the past key values up to a certain maximum length."""
    if isinstance(past_key_values, TreeDynamicCache):
        past_key_values.crop(max_length, valid_retrieve_indices)
    else:
        raise NotImplementedError
    return past_key_values

def _prepare_2d_attention_mask(
    model_kwargs: Dict[str, Any], 
    new_length: int, 
) -> Dict[str, Any]:
    """Expands or crops the model's mask for decoding purposes, to the defined length"""

    enable_tree_attention = model_kwargs["enable_tree_attention"]
    old_mask_key = "attention_mask" if not enable_tree_attention else "attention_mask_2d"
    new_mask_key = "attention_mask"
    if old_mask_key not in model_kwargs:
        raise ValueError(f"must have attention_mask")

    mask: torch.LongTensor = model_kwargs[old_mask_key]
    
    mask_length_diff = new_length - mask.shape[1]
    if mask_length_diff < 0:
        model_kwargs[new_mask_key] = mask[:, :mask_length_diff]
    elif mask_length_diff > 0:
        model_kwargs[new_mask_key] = torch.cat([mask, mask.new_ones((mask.shape[0], mask_length_diff))], dim=-1)
        
    return model_kwargs


def _prepare_token_type_ids(model_kwargs: Dict[str, Any], new_length: int) -> Dict[str, Any]:
    """Expands or crops the model's token_type_ids for decoding purposes, to the defined length"""
    if "token_type_ids" not in model_kwargs or model_kwargs["token_type_ids"] is None:
        return model_kwargs

    raise NotImplementedError
    token_type_ids = model_kwargs["token_type_ids"]
    final_token_type = token_type_ids[:, -1].unsqueeze(-1)
    type_length_diff = new_length - token_type_ids.shape[1]

    if type_length_diff < 0:
        token_type_ids = token_type_ids[:, :type_length_diff]
    elif type_length_diff > 0:
        token_type_copies = final_token_type.repeat(1, type_length_diff)
        model_kwargs["token_type_ids"] = torch.cat([model_kwargs["token_type_ids"], token_type_copies], dim=-1)
    return model_kwargs

from enum import Enum
class CandidateRefillPolicy(Enum):
    NEVER = 0,
    PREFILL_ONLY = 1,
    EVERY_ROUND = 2,
    EVERY_MISMATCH = 3
_DEFAULT_CANDIDATE_REFILL_POLICY = CandidateRefillPolicy.EVERY_MISMATCH

def get_prefill_candidate_refill(candidate_refill_policy:CandidateRefillPolicy):
    # return candidate_refill under the circumstance that it is a prefilling
    return candidate_refill_policy != CandidateRefillPolicy.NEVER
def get_verified_candidate_refill(candidate_refill_policy:CandidateRefillPolicy, all_accepted:bool):
    # return candidate_refill under the circumstance that it is a updating after the verification
    if (
        candidate_refill_policy == CandidateRefillPolicy.NEVER 
        or candidate_refill_policy == CandidateRefillPolicy.PREFILL_ONLY
        or (candidate_refill_policy == CandidateRefillPolicy.EVERY_MISMATCH and all_accepted is True)
    ):
        return False
    else:
        return True
        

class AssistedCandidateGenerator:
    def __init__(
        self,
        assistant_model,
        candidate_refill_policy: CandidateRefillPolicy,
        generation_config: GenerationConfig,
        model_kwargs: Dict,
        logits_processor = None,
        num_assistant_tokens: Optional[int] = None,
        assistant_confidence_threshold: Optional[float] = None,
        **kwargs
    ):
        # Prepare the assistant and the starting number of candidate tokens
        self.assistant_model = assistant_model
        self.num_assistant_tokens = num_assistant_tokens if num_assistant_tokens is not None else assistant_model.generation_config.num_assistant_tokens

        # Set eos in assistant same as in target model
        self.assistant_model.generation_config.eos_token_id = generation_config.eos_token_id

        # Prepare the kwargs for the assistant model
        assistant_kwargs = {}
        for key, value in model_kwargs.items():  # deepcopy crashes if we attempt to copy encoder outputs with grads
            # cache should not be passed to assistant model, as it IS the main model's cache.
            if key not in ("past_key_values"):
                assistant_kwargs[key] = (
                    value if isinstance(value, torch.Tensor) else copy.deepcopy(value)
                )
                
        self.assistant_kwargs = assistant_kwargs

        # both are decoder-only
        self.input_ids_key = "input_ids"

        # Prepare generation-related options.
        self.logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        
        self.generation_config = copy.deepcopy(generation_config)
        # change some assistant-specific items
        self.generation_config.is_assistant = True
        self.generation_config.return_dict_in_generate = True
        self.generation_config.output_scores = True

        # Disable sampling -- this implementation of assisted generation/speculative decoding uses the assistant
        # greedily to maximize matches. Disables sampling-related flags to prevent warnings
        self.generation_config.do_sample = False
        for attr in ("temperature", "top_p", "min_p", "typical_p", "top_k", "epsilon_cutoff", "eta_cutoff"):
            setattr(self.generation_config, attr, None)

        # avoid unnecessary warnings that min_length is larger than max_new_tokens
        # remove the `MinLengthLogitsProcessor` if exists (NOTE: no need to check for `MinNewTokensLogitsProcessor`)
        self.main_model_min_length = self.generation_config.min_length
        self.generation_config.min_length = 0
        self.generation_config.min_new_tokens = None

        # We need to roll back the cache in assisted generation, only DynamicCache is supported
        self.generation_config.cache_implementation = None
        
        # add confidence threshold if needed
        if assistant_confidence_threshold is not None:
            if not isinstance(assistant_confidence_threshold, float):
                raise ValueError
            self.generation_config.assistant_confidence_threshold = assistant_confidence_threshold
            
        self.candidate_refill_policy = candidate_refill_policy
        self.next_candidate_refill = get_prefill_candidate_refill(candidate_refill_policy)
        # record fuzzy length for later refill
        self.fuzzy_length = 0

    # def set_candidate_refill_policy(self, refill_policy: CandidateRefillPolicy = None):
    #     if refill_policy is not None:
    #         self.candidate_refill_policy = refill_policy
    #     else:
    #         self.candidate_refill_policy = self.default_candidate_refill_policy
    #     self.next_candidate_refill = get_prefill_candidate_refill(self.candidate_refill_policy)
    
    def get_candidates(self, input_ids: torch.LongTensor, last_candidate_length: int) -> Tuple[torch.LongTensor, Optional[torch.FloatTensor]]:

        # Don't generate more than `max_length - 1` candidates since the target model generates one extra token.
        new_cur_len = input_ids.shape[-1]
        max_new_tokens, min_new_tokens = self._get_max_min_new_tokens(new_cur_len)
        if max_new_tokens == 0:
            return input_ids, None

        # 1. If it is not the first round of candidate generation, prepare the inputs based on the input_ids length
        # (which implicitly contains the number of accepted candidates from the previous round)
        has_past_key_values = self.assistant_kwargs.get("past_key_values", None) is not None
             
        if has_past_key_values:
            # TODO: need fix
            # Modify new_cur_len here will ignore the first prefilling, but ignoring it is ok.
            if self.next_candidate_refill:
                # TODO: it is a buggy code when no bound layers and EVERY_MISMATCH
                min_cache_length = self.assistant_kwargs["past_key_values"].get_min_length()
                seq_cache_length = self.assistant_kwargs["past_key_values"].get_seq_length()
                if min_cache_length < seq_cache_length:
                    # min cache length is a good indicator of non-tree length
                    new_cache_size = min_cache_length + 1
                else:
                    # we have no way to ensure non-tree length, so we have to use last flat_candidate_length
                    new_cache_size = seq_cache_length - last_candidate_length + 1
            else:
                new_cache_size = new_cur_len - 1

            # why crop:max_length = (new_cache_size-1)? because the last kv item is either non-exist or invalid
            self.assistant_kwargs["past_key_values"] = _crop_past_key_values(
                self.assistant_kwargs["past_key_values"], new_cache_size - 1
            )  # the assistant does not have the token after the last match, hence the -1

            self.assistant_kwargs = _prepare_2d_attention_mask(
                self.assistant_kwargs, new_cur_len
            )

        # 2. Forecast next N tokens using the assistant model.
        assistant_generation_kwargs = {
            self.input_ids_key: input_ids,
            "min_new_tokens": min_new_tokens,
            "max_new_tokens": max_new_tokens,
            "generation_config": self.generation_config,
            "logits_processor": self.logits_processor,
            "candidate_refill": self.next_candidate_refill,
        }

        assistant_output = self.assistant_model.generate(**assistant_generation_kwargs, **self.assistant_kwargs)
        
        # if assistant_output is None, then you must not be the participant of the draft model
        if assistant_output is None:
            if self.assistant_model.parallel_config.in_participant:
                raise ValueError("participant return None for draft model.")
            candidate_ids = None
            candidate_logits = None
        else:
            # participants must have candidate_ids != None, while logits are not guaranteed
            candidate_ids = assistant_output.sequences
            # Update variables for the next round of candidate generation
            self.assistant_kwargs["past_key_values"] = assistant_output.past_key_values
            # Prepare variables for output
            
            if assistant_output.scores is None or assistant_output.scores[0] is None:
                if get_global_rank() == self.assistant_model.parallel_config.driver:
                    # driver must have prob
                    raise ValueError(f"driver must have prob")
                candidate_logits = None
            else:
                candidate_logits = torch.stack(assistant_output.scores, dim=1)
        
        return candidate_ids, candidate_logits

    def update_candidate_strategy(self, candidate_length:int, num_matches: int):
        if isinstance(num_matches, torch.Tensor):
            num_matches = num_matches.item()
        all_accepted = (num_matches == candidate_length)
        self.next_candidate_refill = get_verified_candidate_refill(self.candidate_refill_policy, all_accepted)
        return
        # Adjust the max number of assistant tokens to use in the next iteration. This is a simple heuristic,
        if all_accepted:
            self.num_assistant_tokens += 2
        else:
            self.num_assistant_tokens = max(1, self.num_assistant_tokens // 2)
        self.assistant_model.generation_config.num_assistant_tokens = self.num_assistant_tokens

    def _get_max_min_new_tokens(self, new_cur_len: int):
        max_new_tokens = min(int(self.num_assistant_tokens), self.generation_config.max_length - new_cur_len - 1)
        min_new_tokens = max(min(max_new_tokens, self.main_model_min_length - new_cur_len), 0)
        return max_new_tokens, min_new_tokens
        
