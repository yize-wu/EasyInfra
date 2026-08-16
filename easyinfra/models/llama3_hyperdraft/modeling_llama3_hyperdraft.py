from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.cache_utils import Cache

import torch
from torch import nn
import torch.distributed as dist

from typing import List, Optional, Tuple, Union, Dict

import math
import time
import inspect
import warnings
from enum import Enum

from ...generation.cache_utils import (
    TreeDynamicCache,
)

from ...utils import record_time_sync, rank0_print
from ...modeling_utils.causal_lm import CausalLMPretrainedModel

from ...generation.modules.linear import BaseLinear
from ...generation.parallel.parallel_state import GroupCommunicator       
from ...generation.modules.rmsnorm import BaseRMSNorm      
from ...generation.modules.activation.base_activation import SiLU
from ...generation.utils.attention_mask import (
    _prepare_4d_causal_attention_mask_with_cache_position
)
from ...generation.modules.word_embed.embedding import Embedding
from ...generation.modules.rmsnorm import BaseRMSNorm        
from ...generation.modules.position_embed import BaseRotaryEmbedding       
from ...generation.modules.logits_head import BaseLogitsHead

from ..auto.hyperdraft_parallel_config import Layer2LayerPolicy, LayerOutputPolicy
from ..auto.hyperdraft_parallel_config import (
    HyperDraftModelParallelConfig,
    get_world_size,
    get_global_rank,
)

def remove_llama_attn_or_mlp_name(param_name: str):
    module_name = param_name.split('.')
    if len(module_name) >= 4 and module_name[3] in ['self_attn', 'mlp']:
        module_name.pop(3) # 'self_attn' or 'mlp' is removed
    module_name = '.'.join(module_name)
    return module_name

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def _apply_rotary_pos_emb(hidden_states, cos, sin, **kwargs) -> torch.Tensor:
    return (hidden_states * cos) + (rotate_half(hidden_states) * sin)

def apply_rotary_pos_emb(q, k, cos, sin, **kwargs) -> torch.Tensor:
    return _apply_rotary_pos_emb(q, cos, sin), _apply_rotary_pos_emb(k, cos, sin, **kwargs)


class HyperDraftLlamaPreTrainedModel(CausalLMPretrainedModel):
    _model_type = 'llama'
    config_class = LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["MyLlamaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
    
from ..auto.hyperdraft_parallel_config import LayerParallelPolicy
from .llama_decoder_impl import LPPOLICY_2_DECODER, LlamaDecoder, all_gather_attn_output    
    
class LlamaHyperDecoderLayer(nn.Module):
    """
        We only do colwise and rowwise tensor parallelism to reduce communication overhead.
    """
    def __init__(
        self, 
        config: LlamaConfig, 
        hyper_layer_idx: int, 
        lp_group: GroupCommunicator,
        lp_policy: LayerParallelPolicy,
        layer_indices: List[int] = None,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.hyper_layer_idx = hyper_layer_idx
        
        self.lp_group: GroupCommunicator = lp_group
        self.lp_policy: LayerParallelPolicy = lp_policy
        self.lp_group_size: int = lp_group.group_size
        
        # choose different implementations of inner decoder layer
        decoder_class: LlamaDecoder = LPPOLICY_2_DECODER[self.lp_policy]
        self.layers = nn.ModuleList(
            [decoder_class(config, layer_idx, self.lp_group) for idx, layer_idx in enumerate(layer_indices)]
        )
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: TreeDynamicCache = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
        candidate_refill: bool = True,
        **kwargs,
    )-> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
            Here the runner must be a participant of this hyper layer.
        """
        # start = record_time_sync()
        if candidate_refill:
            for inner_layer_idx, decoder_layer in enumerate(self.layers):
                block_output = decoder_layer.run_attn_block(
                    hidden_states,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    position_embeddings=position_embeddings
                )
                hidden_states += block_output
                block_output = decoder_layer.run_mlp_block(hidden_states)
                hidden_states += block_output
        else:
            my_decoder_layer: LlamaDecoder = self.layers[self.lp_group.local_rank]
            if self.lp_policy == LayerParallelPolicy.ATTN_ONLY:
                attn_output = my_decoder_layer.run_attn_block(
                    hidden_states,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    position_embeddings=position_embeddings
                )
                attn_output_all_layers = all_gather_attn_output(attn_output, self.lp_group)
                for inner_layer_idx, decoder_layer in enumerate(self.layers):
                    hidden_states = hidden_states + attn_output_all_layers[inner_layer_idx]
                    mlp_output = decoder_layer.run_mlp_block(hidden_states)
                    hidden_states += mlp_output
            else:
                # Run your layer
                layer_outputs = my_decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    position_embeddings=position_embeddings,
                )
                # adder should be the same for all participants
                adder = layer_outputs[0]
                hidden_states += adder # TODO: Is it safe?
            
        # print(self.hyper_layer_idx, record_time_sync()-start)
                        
        outputs = (hidden_states,)
        return outputs



        
class LlamaModel(HyperDraftLlamaPreTrainedModel):
    def __init__(self, config: LlamaConfig, parallel_config: HyperDraftModelParallelConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        
        self.need_lm_head = parallel_config.need_lm_head
        
        # self.layer_output_policies = parallel_config.layer_output_policies
        self.l2l_groups = parallel_config.l2l_groups
        self.l2l_policies = parallel_config.l2l_policies
        
        self.global_rank = get_global_rank()
        
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        # module_list = []
        # for idx, layer_indices in enumerate(parallel_config.bound_strategy):
        #     module_list.append(LlamaHyperDecoderLayer(config, idx, parallel_config.lp_groups[idx], layer_indices))
        # self.layers = nn.ModuleList(module_list)
        self.layers = nn.ModuleList(
            [LlamaHyperDecoderLayer(config, idx, parallel_config.lp_groups[idx], parallel_config.lp_policies[idx], layer_indices) for idx, layer_indices in enumerate(parallel_config.bound_strategy)]
        )
        self.norm = BaseRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = BaseRotaryEmbedding(config=config)

        # Initialize weights and apply final processing
        self.post_init()
        
    # def get_input_embeddings(self):
    #     return self.embed_tokens

    # def set_input_embeddings(self, value):
    #     self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        cache_position: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        candidate_refill: bool = False,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            raise ValueError
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            raise ValueError
            position_ids = cache_position.unsqueeze(0)
        
        hidden_states = inputs_embeds
        
        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        
        # print(hidden_states.shape)
        # Refill: run all layers sequentially.
        # Speculative decoding: run layer parallel.
        # exit_layer = 4
        # skip_layers = [4,8,12,16,20,24]
        for hyper_layer_idx, decoder_layer in enumerate(self.layers):
            # if hyper_layer_idx in skip_layers:
            #     continue
            if candidate_refill:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    candidate_refill=candidate_refill,
                )
                hidden_states = layer_outputs[0]
            else:
                lp_group: GroupCommunicator = decoder_layer.lp_group
                if self.global_rank in lp_group.all_ranks:
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                        candidate_refill=candidate_refill,
                    )
                    hidden_states = layer_outputs[0]
                    # # After decoder layer, hidden_states is output if not need to reduce; or hidden_states + adder is output
                    # # If the adder needs reduction, we must use layer_added to get the right result
                    # hidden_states, adder = layer_outputs[0], layer_outputs[1]
                    
                    # layer_output_policy = self.layer_output_policies[hyper_layer_idx]
                    # if layer_output_policy == LayerOutputPolicy.ALL_REDUCE:
                    #     lp_group.all_reduce(adder)
                    #     hidden_states = hidden_states + adder
                else:
                    pass
                    # prepare a zero(why not empty? to avoid inf bugging) tensor for possibly broadcasting
                    # hidden_states = torch.zeros_like(hidden_states) 
                
                # This could be a l2l, or it also could be a last layer -> lm_head
                to_next_layer_group = self.l2l_groups[hyper_layer_idx]
                if self.global_rank in to_next_layer_group.all_ranks:
                    to_next_layer_policy = self.l2l_policies[hyper_layer_idx]
                    if to_next_layer_policy == Layer2LayerPolicy.BROADCAST:
                        to_next_layer_group.broadcast(hidden_states, src=lp_group.driver)
        
        if self.need_lm_head:                                                            
            hidden_states = self.norm(hidden_states)
        
        # Note that the hidden_states could be not the last hidden_states, 
        # if process is not participant of the last layer
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values
        )

def arg_from_config_or_kwargs(config, keyword_arg_dict: Dict, keyword: str, default_value=None):
    arg_from_kwargs = keyword_arg_dict.pop(keyword, default_value)     
    arg_from_config = getattr(config, keyword, default_value)   
    if arg_from_kwargs != arg_from_config and arg_from_config != default_value and arg_from_kwargs != default_value:
        raise ValueError(f"keyword {keyword} argument are specified in config and kwargs, but they are different.")
    if arg_from_kwargs != default_value:
        return arg_from_kwargs
    else:
        # arg from config == default_value or not doesn't matter
        return arg_from_config

class LlamaForCausalLM(HyperDraftLlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]
    
    @classmethod
    def make_parallel_config(
        cls, 
        config: LlamaConfig, 
        **kwargs
    ):
        bound_strategy = arg_from_config_or_kwargs(config, kwargs, keyword="bound_strategy")
        if bound_strategy is None:
            raise ValueError("Both config and kwargs has no arg named bound_strategy.")
        lp_policy = arg_from_config_or_kwargs(config, kwargs, keyword="layer_parallel_policy")
        base_model_driver = kwargs.pop("base_model_driver", 0)
        
        hyperdraft_parallel_config = HyperDraftModelParallelConfig.make_parallel_config(bound_strategy, lp_policy, base_model_driver)
        return hyperdraft_parallel_config, kwargs
    
    def param_name_to_module_kwargs(self, param_name: str):
        module_name = remove_llama_attn_or_mlp_name(param_name)
        if 'layers' in module_name: # not a layer param
            module_name_split = module_name.split('.')
            layer_idx = int(module_name_split[2])
            hyper_idx = [i for i,l in enumerate(self.bound_strategy) if layer_idx in l][0]
            inner_layer_idx = self.bound_strategy[hyper_idx].index(layer_idx)
            # modify out layer idx to hyper_idx
            module_name_split[2] = str(hyper_idx)
            # add inner layer idx
            module_name_split.insert(3, 'layers')
            module_name_split.insert(4, str(inner_layer_idx))
            module_name = '.'.join(module_name_split)
            
        return ((module_name, {}),)

    def need_load_weight(self, param_name, module_name):
        if 'rotary_emb.inv_freq' in module_name:
            # no need to load inv_freq
            return False
        elif 'lm_head' in module_name:
            return self.need_lm_head
        else:
            return self.need_model
    
    def _update_model_kwargs_for_generation(
        self,
        outputs,
        generation_config,
        model_kwargs,
        model_inputs,
        num_new_tokens: int = 1,
        force_candidate_refill: bool = False,
    ):
        # Normally, after one run, do not refill
        # But it can be forced.
        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs,
            generation_config,
            model_kwargs,
            num_new_tokens,
        )
        if force_candidate_refill:
            raise NotImplementedError
        model_kwargs["candidate_refill"] = force_candidate_refill
        return model_kwargs
    
    def __init__(self, config, parallel_config: HyperDraftModelParallelConfig,):
        super().__init__(config)
        
        self.bound_strategy = parallel_config.bound_strategy
        self.hidden_size = config.hidden_size
        self.parallel_config = parallel_config
        self.need_model = parallel_config.in_participant
        self.need_lm_head = parallel_config.need_lm_head
        
        if self.need_model:
            self.model = LlamaModel(config, parallel_config)
        
        # All process on base model must have lm_head for logits forward.
        # This is to avoid logits broadcasting (broadcast the last hidden_states instead)
        if self.need_lm_head:
            self.vocab_size = config.vocab_size
            self.lm_head = BaseLogitsHead(config.hidden_size, config.vocab_size, bias=False)
            
        # Initialize weights and apply final processing
        self.post_init()
            
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        return_dict: bool = None,
        candidate_refill: bool = True,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # print("is_refill", candidate_refill)
        if self.need_model:
            # start = record_time_sync()
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                inputs_embeds=inputs_embeds,
                candidate_refill=candidate_refill,
            )
            hidden_states : torch.Tensor = outputs[0]
            # model_end = record_time_sync()
            # print(f"rank {self.parallel_config.all_ranks_group.local_rank} model end time {model_end}")
            # print("model forward time", model_end - start)
            
            if self.need_lm_head:
                # start = record_time_sync()
                logits = self.lm_head(hidden_states)
                logits = logits.float()
                # print("lm head forward time", record_time_sync() - start)
            else:
                logits = None
            past_key_values = outputs.past_key_values
            
        else:
            logits = None
            past_key_values = None
        
        if not return_dict:
            output = (logits,) + outputs[1:]
            return output
        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=past_key_values,
        )
        
    def prepare_inputs_for_generation(
        self,
        input_ids,
        candidate_refill=True,
        **kwargs,
    ):
        # hyperdraft model need candidate_refill as model_inputs
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            **kwargs
        )
        model_inputs.update(
            {
                "candidate_refill": candidate_refill,
            }
        )
        return model_inputs
