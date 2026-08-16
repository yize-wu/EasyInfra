import torch
from typing import Dict, Optional, List
from ..generation.utils.base_parallel import BaseParallelGenerationMixin
from ..generation.utils.configuration_utils import GenerationConfig
from ..generation.cache_utils import MultiRequestDynamicCache
from .base_parallel import BaseParallelPreTrainedModel
from ..generation.utils.tree_attention import (
    _update_tree_causal_mask_from_retrieve_indices,
)
from ..generation.utils.tree_attention import (
    _prepare_tree_verification_causal_mask
)
from ..generation.utils.attention_mask import (
    _prepare_4d_causal_attention_mask_with_cache_position    
)
    
from .utils import _prepare_position_id_from_2d_attention_mask
from ..generation.functions import (
    exclusive_cumsum,
)
from ..schedule.chunk import ChunkBlockList
import math
from ..utils.stats import show_rank_print
from ..generation.cache_utils import TreeDynamicCache, MultiRequestDynamicCache

def flatten_input(t: torch.Tensor, chunk_list: ChunkBlockList, request_length: torch.Tensor, request_offset: torch.Tensor,):
    # t: [bsz, seq_len]
    flat_t = t.new_empty([request_length.sum()]) # [flat_len]
    for chunk in chunk_list:
        request_id = chunk.request_id
        flat_t[ chunk.global_span[0] : chunk.global_span[1] ] = t[request_id, :]
    return flat_t

from ..schedule.request import Request, RequestHub
from ..schedule.scheduler import MoeLayerScheduler

_ATTN_MASK_DIM = 4
class CausalLMGenerationMixin(BaseParallelGenerationMixin):
    is_causal_model = True
    def is_training(self):
        return self.training
    
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past_key_values,
        attention_mask: torch.Tensor,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        enable_tree_attention=False,
        enable_scheduler=False,
        enable_seq_split=False,
        min_split_length=None,
        request_hub: RequestHub = None,
        scheduler: MoeLayerScheduler = None,
        **kwargs,
    ):
        model_inputs: Dict = {
            "enable_scheduler": enable_scheduler,
        }
        
        if not enable_scheduler:
            assert request_hub is None
            if past_key_values is None:
                raise ValueError(f"Causal Language Model needs cache.")
            if attention_mask is None:
                raise ValueError(f"Causal Language Model needs an attention mask, as it is a causal model.")
            elif attention_mask.dim() != 2 and attention_mask.dim() != _ATTN_MASK_DIM:
                raise ValueError(f"Causal Language Model needs a 2d or {_ATTN_MASK_DIM}d attention mask, but {attention_mask.dim()}.")
            if inputs_embeds is not None:
                raise NotImplementedError(f"No support for input_embeds for now.")

            
            # choose input_ids for the generation
            if input_ids.shape[1] != cache_position.shape[0]:
                # target_length = input_ids.shape[-1]
                input_ids = input_ids[:, cache_position]

            if position_ids is None:
                position_ids = _prepare_position_id_from_2d_attention_mask(attention_mask)
                if past_key_values:
                    position_ids = position_ids[:, -input_ids.shape[1]:]
                    # This `clone` call is needed to avoid recapturing cuda graphs with `torch.compile`'s  `mode="reduce-overhead`, as otherwise the input `position_ids` would have various stride during the decoding. Here, simply using `.contiguous()` is not sufficient as in the batch size = 1 case, `position_ids` is already contiguous but with varying stride which retriggers a capture.
                    position_ids = position_ids.clone(memory_format=torch.contiguous_format)

            if hasattr(self.config, "sliding_window") and self.config.sliding_window is not None:
                raise NotImplementedError
            attention_mask_4d = _prepare_4d_causal_attention_mask_with_cache_position(
                attention_mask,
                sequence_length=input_ids.shape[-1],
                target_length=attention_mask.shape[-1],
                dtype=self.parallel_config.dtype,
                device=self.parallel_config.device,
                cache_position=cache_position,
                batch_size=input_ids.shape[0],
            )

            model_inputs.update(
                {
                    "input_ids": input_ids.clone(memory_format=torch.contiguous_format), # The clone here is for the same reason as for `position_ids`.
                    # "inputs_embeds": None,
                    "position_ids": position_ids,
                    "cache_position": cache_position,
                    "past_key_values": past_key_values,
                    "attention_mask": attention_mask_4d,
                }
            )
        
        else:
            assert request_hub is not None
            pad_side = "left"
            if not pad_side == "left":
                raise NotImplementedError
            
            # Layer Scheduler
            # The layout of requests has been decided, now just chunk
            max_chunk_size = kwargs.pop("max_chunk_size", -1)
            if max_chunk_size == -1:
                max_chunk_size = 1e9
                            
            # create chunk list
            chunk_list = ChunkBlockList(self, max_chunk_size, enable_seq_split=enable_seq_split, min_split_length=min_split_length)
            # partition inputs
            chunk_list.partition_input_into_chunks(request_hub)
            scheduler.update_chunks(chunk_list)
            
            # input and position ids
            input_ids = [chunk.input_ids.clone(memory_format=torch.contiguous_format) for chunk in chunk_list.chunk_list]
            position_ids = [chunk.position_ids for chunk in chunk_list.chunk_list]
            model_inputs.update({
                "input_ids": input_ids,
                "position_ids": position_ids,
                "request_hub": request_hub,
                "scheduler": scheduler,
            })
            
        return model_inputs
    
    def check_model_inputs(self, model_inputs: Dict):
        if any(key not in model_inputs for key in (
            "input_ids",
            "position_ids",
        )):
            raise ValueError(f"input_ids and/or position_ids is missing for causal LM.")
            
        if "scheduler" in model_inputs:
            if any(key in model_inputs for key in (
                "cache_position",
                "past_key_values",
                "attention_mask",
            )):
                raise ValueError(f"With scheduler, no specified arguments of cache_position/past_key_values/attention_mask is needed.")
    
    def prepare_outputs_for_update(
        self,
        outputs,
        model_inputs,
        **kwargs,
    ):
        enable_scheduler = model_inputs.get("enable_scheduler", False)
        if not enable_scheduler:
            return outputs
                
        logits: torch.Tensor = outputs.logits
        # if logits is not None:
        #     request_length: torch.LongTensor = model_inputs.get("request_length")
        #     # change output logits to batched form
        #     output_logits_positions = request_length.cumsum(dim=-1) - 1
        #     logits = logits[0,output_logits_positions,:].reshape(output_logits_positions.shape[-1],1,-1)
        #     outputs.logits = logits
            
        return outputs
        
    
    def _update_model_kwargs_for_generation(
        self,
        outputs,
        generation_config: GenerationConfig,
        model_kwargs: Dict,
        model_inputs: Dict,
        num_new_tokens: int = 1,
    ):
        if getattr(outputs, "state", None) is not None:
            raise ValueError
        if "token_type_ids" in model_kwargs:
            raise ValueError
        
        # update past_key_values keeping its naming used in model code
        cache_name, cache = "past_key_values", outputs.past_key_values
        
        enable_tree_attention = model_kwargs.get("enable_tree_attention")
        last_kv_len = model_kwargs["attention_mask"].shape[-1] # it is valid in both 2d and 4d case
        
        # if model_kwargs["enable_scheduler"] == True:
        #     model_kwargs["enable_scheduler"] = False
        #     request_ids = model_inputs["request_ids"]
        #     # kv cache
        #     base_cache = TreeDynamicCache()
        #     cache: MultiRequestDynamicCache
            
        #     sorted_cache = [None] * request_ids.shape[-1]
        #     for i,j in enumerate(request_ids.tolist()):
        #         sorted_cache[j] = cache.caches[i]
                
        #     for tree_cache in sorted_cache:
        #         key_cache, value_cache = tree_cache.key_cache, tree_cache.value_cache
        #         for layer_idx in key_cache:
        #             keys = key_cache[layer_idx]
        #             values = value_cache[layer_idx]
        #             if layer_idx not in base_cache.key_cache:
        #                 base_cache.update(keys, values, layer_idx)
        #             else:
        #                 base_cache.key_cache[layer_idx] = torch.cat([base_cache.key_cache[layer_idx], keys], dim=0)
        #                 base_cache.value_cache[layer_idx] = torch.cat([base_cache.value_cache[layer_idx], values], dim=0)
        #     cache = base_cache
        model_kwargs[cache_name] = cache
        
        if not enable_tree_attention:
            pass
            # update attention mask
            # attention_mask: torch.Tensor = model_kwargs["attention_mask"]
            # model_kwargs["attention_mask"] = torch.cat(
            #     [attention_mask, attention_mask.new_ones((attention_mask.shape[0], num_new_tokens))], dim=-1
            # )
            # # update cache position
            # model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + num_new_tokens
        else:
            if generation_config.is_assistant:
                attention_mask = model_kwargs["attention_mask"]
                # it is a tree-sampling draft model
                # get retrieve_indices from model_kwargs
                topk_retrieve_indices: torch.LongTensor = model_kwargs["topk_retrieve_indices"]
                top_k = topk_retrieve_indices.shape[1]
                
                before_this_time_flat_tree_length = top_k * (topk_retrieve_indices.shape[2] - 1)
                non_tree_length = attention_mask.shape[-1] - before_this_time_flat_tree_length # it should be a constant
                
                # update attention mask
                model_kwargs["attention_mask"] = _update_tree_causal_mask_from_retrieve_indices(
                    attention_mask,
                    topk_retrieve_indices=topk_retrieve_indices,
                    non_tree_length=non_tree_length,
                    dtype=self.parallel_config.dtype,
                    device=self.parallel_config.device,
                )
                # update 2d attention mask
                attention_mask_2d: torch.LongTensor = model_kwargs["attention_mask_2d"]
                model_kwargs["attention_mask_2d"] = torch.cat(
                    [attention_mask_2d, attention_mask_2d.new_ones((attention_mask_2d.shape[0], num_new_tokens))], dim=-1
                )
                model_kwargs["cache_position"] = torch.arange(last_kv_len, last_kv_len + top_k, device=model_kwargs["cache_position"].device, dtype=model_kwargs["cache_position"].dtype)
                model_kwargs["position_ids"] = (model_kwargs["position_ids"][:,-1:] + 1).expand(model_kwargs["position_ids"].shape[0], top_k)
            else:
                # tree verification
                attention_mask_2d: torch.LongTensor = model_kwargs["attention_mask_2d"]
                # update attention mask
                model_kwargs["attention_mask_2d"] = torch.cat(
                    [attention_mask_2d, attention_mask_2d.new_ones((attention_mask_2d.shape[0], num_new_tokens))], dim=-1
                )
                model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + num_new_tokens
            
        return model_kwargs
    
    def _prepare_tree_verification_attention_mask(
        self,
        attention_mask_2d: torch.LongTensor,
        candidate_input_ids_length: int,
        flat_candidate_length: int,
        cache_position: torch.LongTensor,
        retrieve_indices: torch.Tensor,
        past_kv_length:int, 
        dtype: torch.dtype,
        device: torch.device,
    ):
        return _prepare_tree_verification_causal_mask(
            attention_mask_2d,
            candidate_input_ids_length=candidate_input_ids_length, 
            flat_candidate_length=flat_candidate_length,
            cache_position=cache_position,
            retrieve_indices=retrieve_indices, 
            past_kv_length=past_kv_length,
            dtype=dtype,
            device=device,
        )
        
    def prepare_chunk_schedule_stages(self) -> tuple[tuple]:
        stages = []
        stage_has_communication = []
        for decoder_layer in self.model.layers[:self.config.num_hidden_layers]:
            stages += [_ for _ in decoder_layer.self_attn.stages] + [_ for _ in decoder_layer.mlp.stages]
            stage_has_communication += (
                [_ for _ in decoder_layer.self_attn.stage_has_communication] + 
                [_ for _ in decoder_layer.mlp.stage_has_communication]
            )
        return stages, stage_has_communication

class CausalLMPretrainedModel(BaseParallelPreTrainedModel, CausalLMGenerationMixin):
    _support_tree_attention = True
    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx.to(past_state.device))
                    for past_state in layer_past
                ),
            )
        return reordered_past
    
    def need_load_weight(self, param_name:str, module_name:str):
        if 'rotary_emb.inv_freq' in module_name:
            # no need to load inv_freq
            return False
        else:
            return True
        

from ..generation.parallel.parallel_configuration import MoeParallelConfig
from ..generation.parallel.parallel_utils import get_world_size
from easyinfra.generation.blocks.moe import logical_to_physical_expert_ids

class CausalLMMoePretrainedModel(CausalLMPretrainedModel):
    @classmethod
    def make_parallel_config(cls, config, **kwargs):
        local_world_size = kwargs.pop("local_world_size")
        ep_size = kwargs.pop("ep_size")
        attn_tp_size = kwargs.pop("attn_tp_size")
        shared_expert_tp_size = kwargs.pop("shared_expert_tp_size")
        use_eplb = kwargs.pop("use_eplb", None)
        num_global_experts = kwargs.pop("num_global_experts", None)
        hierarchical_all2all_structure = kwargs.pop("hierarchical_all2all_structure", None)
        
        all_ranks = [_ for _ in range(get_world_size())]
        parallel_config = MoeParallelConfig(
            all_ranks, 
            ep_size=ep_size, 
            attn_tp_size=attn_tp_size,
            shared_expert_tp_size=shared_expert_tp_size,
            use_eplb=use_eplb,
            num_global_experts=num_global_experts,
            hierarchical_all2all_structure=hierarchical_all2all_structure,
        )
        
        return parallel_config, kwargs
    
    @classmethod
    def get_physical_expert_module_names(cls, module, param_name: str) -> List[str]:
        '''
            Return None if the expert should not be placed on this device.
        '''
        # get logical expert id
        splits = param_name.split("experts.experts.")
        assert len(splits) == 2
        logical_expert_id = int(splits[1].split(".")[0])
        # get phy2log expert map and ep_rank
        splits = param_name.split(".")
        for split_i, split in enumerate(splits[:-1]):
            new_module = getattr(module, split, None)
            if split == "experts":
                phy2log_expert_map = module.phy2log_expert_map_cpu
                ep_rank = module.ep_rank
                break
            elif new_module is None:
                raise ValueError(f"{module} has no attribute {split}.")
            module = new_module
                            
        physical_expert_ids = logical_to_physical_expert_ids(logical_expert_id, 
                                                            phy2log_expert_map[ep_rank])
        module_names = []
        EXPERT_ID_OFFSET_TO_SPLIT = 2 # modify the pos of id
        for physical_expert_id in physical_expert_ids:
            splits[split_i+EXPERT_ID_OFFSET_TO_SPLIT] = str(physical_expert_id)
            module_names.append(".".join(splits))
                    
        # if [], this expert should not be load to this device
        return module_names
