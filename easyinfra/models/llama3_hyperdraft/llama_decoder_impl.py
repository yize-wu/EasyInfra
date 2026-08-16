from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.models.llama.configuration_llama import LlamaConfig

import torch
from torch import nn
import torch.distributed as dist

from typing import List, Optional, Tuple, Union

import math
import time
import inspect
import warnings
from enum import Enum

from ...generation.cache_utils import (
    TreeDynamicCache,
)

from ...utils import record_time_sync

from ...generation.modules.linear import BaseLinear
from ...generation.parallel.parallel_state import GroupCommunicator       
from ...generation.modules.rmsnorm import BaseRMSNorm      
from ...generation.modules.activation.base_activation import ACT2FN
from ...generation.modules.attention.sdpa import (
    SdpaAttention,
    flatten_attn_output
)
from ...generation.modules.rmsnorm import BaseRMSNorm        


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def _apply_rotary_pos_emb(hidden_states, cos, sin, **kwargs) -> torch.Tensor:
    return (hidden_states * cos) + (rotate_half(hidden_states) * sin)

def apply_rotary_pos_emb(q, k, cos, sin, **kwargs) -> torch.Tensor:
    return _apply_rotary_pos_emb(q, cos, sin), _apply_rotary_pos_emb(k, cos, sin, **kwargs)

from ..auto.hyperdraft_parallel_config import LayerParallelPolicy

def all_gather_attn_output(attn_output: torch.Tensor, lp_group: GroupCommunicator):
    all_layers_attn_output_shape = list(attn_output.shape)
    all_layers_attn_output_shape.insert(0, lp_group.group_size)
    attn_output_all_layers = torch.empty(all_layers_attn_output_shape, device=attn_output.device, dtype=attn_output.dtype)
    lp_group.all_gather_into_tensor(attn_output_all_layers, attn_output)
    # all_layers_attn_output_list = [torch.empty_like(attn_output) for _ in range(lp_group.group_size)]
    # lp_group.all_gather(all_layers_attn_output_list, attn_output)
    return attn_output_all_layers
 
def all_reduce_adder(attn_output:torch.Tensor, mlp_output:torch.Tensor, lp_group: GroupCommunicator):
    output = attn_output + mlp_output
    lp_group.all_reduce(output)
    return output
    
class LlamaDecoder(nn.Module):
    def __init__(
        self, 
        config: LlamaConfig, 
        layer_idx: int, 
        lp_group: GroupCommunicator,
    ):
        super().__init__()
        
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        
        self.intermediate_size = config.intermediate_size
                
        self.input_layernorm = BaseRMSNorm(config.hidden_size, config.rms_norm_eps)

        self.q_proj = BaseLinear(self.hidden_size, self.num_heads*self.head_dim, bias=config.attention_bias)
        self.k_proj = BaseLinear(self.hidden_size, self.num_key_value_heads*self.head_dim, bias=config.attention_bias)
        self.v_proj = BaseLinear(self.hidden_size, self.num_key_value_heads*self.head_dim, bias=config.attention_bias)
        self.attn = SdpaAttention()
        self.o_proj = BaseLinear(self.num_heads*self.head_dim, self.hidden_size, bias=config.o_proj_bias)
        
        self.post_attention_layernorm = BaseRMSNorm(config.hidden_size, config.rms_norm_eps)
        
        self.up_proj = BaseLinear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.gate_proj = BaseLinear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]
        self.down_proj = BaseLinear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
    
    def run_attn_block(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: TreeDynamicCache = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
    ):
        """
            Compute the attn_output.
        """
        bsz, q_len = hidden_states.shape[:2]
        hidden_states = self.input_layernorm(hidden_states)
        # attn_start = record_time_sync()
        query_states: torch.Tensor = self.q_proj(hidden_states)
        key_states: torch.Tensor = self.k_proj(hidden_states)
        value_states: torch.Tensor = self.v_proj(hidden_states)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(-2,-3)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(-2,-3)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(-2,-3)
        
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
        
        hidden_states = self.attn(query_states, key_states, value_states, attention_mask=attention_mask)
        hidden_states = flatten_attn_output(hidden_states)
        hidden_states = self.o_proj(hidden_states)
        return hidden_states
    
    def run_mlp_block(
        self,
        hidden_states: torch.Tensor,
    ):
        """
            Compute the mlp_output.
        """
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.down_proj(self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))
        return hidden_states
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: TreeDynamicCache = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
        **kwargs,
    )-> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
            ALL sequential.
        """
        # start = record_time_sync()
        residual = hidden_states
        hidden_states = self.run_attn_block(hidden_states, attention_mask=attention_mask, past_key_values=past_key_values, position_embeddings=position_embeddings)
        # print(record_time_sync() - attn_start)
        hidden_states = residual + hidden_states
        
        # start = record_time_sync()
        residual = hidden_states
        hidden_states = self.run_mlp_block(hidden_states)
        hidden_states = residual + hidden_states
        # print(record_time_sync()-start)
                        
        outputs = (hidden_states,)
        return outputs

class LlamaIndirectMlpParallelDecoder(LlamaDecoder):
    def __init__(
        self, 
        config: LlamaConfig, 
        layer_idx: int, 
        lp_group: GroupCommunicator,
    ):
        """
            Must make sure the relative_layer_idx equals to local_rank
        """
        super().__init__(config, layer_idx, lp_group)
        self.lp_group = lp_group

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: TreeDynamicCache = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
        **kwargs,
    )-> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
            Attn cumsum and mlp.
        """
        # start = record_time_sync()
        residual = hidden_states
        attn_output = self.run_attn_block(hidden_states, attention_mask=attention_mask, past_key_values=past_key_values, position_embeddings=position_embeddings)
        # print(record_time_sync() - attn_start)
        
        # attn-only decoding policy:
        # Gather attn output of each layer, then cumsum them as needed
        if self.lp_group.group_size > 1:
            # cumsum attn_output
            attn_output_all_layers = all_gather_attn_output(attn_output, self.lp_group)
            attn_output = attn_output_all_layers[:self.lp_group.local_rank].sum(dim=0)
        hidden_states = residual + attn_output
        
        # start = record_time_sync()
        residual = hidden_states
        mlp_output = self.run_mlp_block(hidden_states)
        # print(record_time_sync()-start)
        
        adder = all_reduce_adder(attn_output, mlp_output, self.lp_group)
        outputs = (adder,)
        return outputs
    
class LlamaDirectMlpParallelDecoder(LlamaDecoder):
    def __init__(
        self, 
        config: LlamaConfig, 
        layer_idx: int, 
        lp_group: GroupCommunicator,
    ):
        """
            Must make sure the relative_layer_idx equals to local_rank
        """
        super().__init__(config, layer_idx, lp_group)
        self.lp_group = lp_group

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: TreeDynamicCache = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
        **kwargs,
    )-> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
            Full layer's parallel.
        """
        # start = record_time_sync()
        residual = hidden_states
        attn_output = self.run_attn_block(hidden_states, attention_mask=attention_mask, past_key_values=past_key_values, position_embeddings=position_embeddings)
        # print(record_time_sync() - attn_start)
        hidden_states = residual + attn_output
        
        # start = record_time_sync()
        residual = hidden_states
        mlp_output = self.run_mlp_block(hidden_states)
        # print(record_time_sync()-start)
        adder = all_reduce_adder(attn_output, mlp_output, self.lp_group)
        outputs = (adder,)
        return outputs
    
LPPOLICY_2_DECODER = {
    LayerParallelPolicy.ATTN_ONLY: LlamaDecoder,
    LayerParallelPolicy.DIRECT_MLP: LlamaDirectMlpParallelDecoder,
    LayerParallelPolicy.INDIRECT_MLP: LlamaIndirectMlpParallelDecoder,
}

