from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from easyinfra.schedule.chunk import ChunkBlock


import time
import torch
from torch import nn
from typing import List, Tuple, Optional

from ...generation.parallel.parallel_configuration import MoeParallelConfig
from ...generation.parallel.parallel_utils import get_device

from ...generation.modules.attention import SdpaAttention, Attention
from ...generation.cache_utils import Cache

from ...utils.stats import record_time_sync, show_rank_print, stage_rank_print

from ..modules.attention.sdpa import apply_rotary_pos_emb
from easyinfra.generation.utils.compute_utils import _add_to_residual
from easyinfra.generation.utils.perf import class_with_timing
from enum import Enum
from easyinfra.generation.blocks.stage_name import AttentionStageName

@class_with_timing
class AttentionBlock(nn.Module):
    def __init__(
        self,
        config, 
        parallel_config: MoeParallelConfig,
        layer_idx: int,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.scaling = None
        self.attention_dropout = config.attention_dropout
        self.sliding_window = getattr(config, "sliding_window", None)
        assert self.sliding_window is None
        self.is_causal = True
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        
        self.num_attention_heads = config.num_attention_heads
        self.parallel_config = parallel_config
        self.tp_group = parallel_config.attn_tp_group
        self.tp_size = self.tp_group.group_size
        self.tp_num_attention_heads = config.num_attention_heads // self.tp_size
        self.tp_num_key_value_heads = config.num_key_value_heads // self.tp_size
        if self.tp_num_attention_heads * self.tp_size != config.num_attention_heads:
            raise ValueError(f"Attention tp size = {self.tp_size}, but number of attention heads is {config.num_attention_heads}, cannot devide")
        elif self.tp_num_key_value_heads * self.tp_size != config.num_key_value_heads:
            raise ValueError(f"Attention tp size = {self.tp_size}, but number of kv heads is {config.num_key_value_heads}, cannot devide")
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        
        self.input_layernorm: nn.Module = None
        
        self.attn: Attention
        
        self.stages = ((AttentionStageName.ATTENTION, self.compute, self.communicate),)
        self.stage_has_communication = (self.parallel_config.attn_tp_group.group_size > 1,)
        # self.stage_has_communication = (True,)

        self.attn_compute_time = 0.0
        self.recv_token_time = 0.0
        
    def compute(
        self,
        chunk: ChunkBlock,
    ):
        stage_rank_print(f"chunk{chunk.chunk_id} in attention compute", 0)
        # time0 = time.time()
        # watch out for cold start
        if chunk.is_first_step():
            residual = chunk.active_variables[0]
        else:
            # no matter from mlp or moe, the logic is the same
            (residual, hidden_states) = chunk.active_variables
            residual = _add_to_residual(residual, hidden_states)
            hidden_states.record_stream(chunk.compute_cuda_stream)
            del hidden_states
            
        # time1 = time.time()
        this_rank_tp_attn_output = self._attn_compute(
            self.input_layernorm(residual), 
            chunk.position_embeddings, 
            chunk.attention_masks, 
            chunk.kv_caches,
            chunk.seqlens_q,
            chunk.cu_seqlens_q,
            chunk.cu_seqlens_k,
        )
        # time2 = time.time()
        # show_rank_print(f"{time1 - time0}, {time2 - time1}", 0)
        chunk.active_variables = (residual, this_rank_tp_attn_output,)
        
    def communicate(
        self,
        chunk: ChunkBlock,
    ):
        stage_rank_print(f"chunk{chunk.chunk_id} in attention communicate", 0)
        # time1 = time.time()
        residual, this_rank_tp_attn_output = chunk.active_variables
        this_rank_tp_attn_output: torch.Tensor
        # if self.parallel_config.attn_tp_group.group_size == 1:
        #     chunk.active_variables = (residual, this_rank_tp_attn_output)
        # else:
        reduced_attn_output, handler = self._communicate(this_rank_tp_attn_output, async_op=True)
        chunk.comm_handler = handler
        this_rank_tp_attn_output.record_stream(chunk.comm_cuda_stream)
        chunk.active_variables = (residual, reduced_attn_output)
        
    def _attn_compute(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_masks: Tuple[torch.Tensor],
        kv_caches: Tuple[Cache],
        seqlens_q: List[int],
        cu_seqlens_q: List[int],
        cu_seqlens_k: List[int],
        **kwargs,
    ):
        '''
            This is for self-defined models to modify.
        '''
        raise NotImplementedError
        input_dim = hidden_states.dim()
        if input_dim == 2:
            # for flash attention convinience
            hidden_states = hidden_states.unsqueeze(0)
        
        input_shape = hidden_states.shape[:-1]
                
        hidden_shape = (*input_shape, -1, self.head_dim)
        
        query_states, key_states, value_states = self.qkv_proj(hidden_states).split(self.qkv_proj.split_sizes, dim=-1)
        query_states: torch.Tensor = self.q_norm(query_states.view(hidden_shape)).transpose(-2, -3)
        key_states: torch.Tensor = self.k_norm(key_states.view(hidden_shape)).transpose(-2, -3)
        value_states: torch.Tensor = value_states.view(hidden_shape).transpose(-2, -3)
        
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        output_shape = (*input_shape, self.hidden_size)
        
        hidden_states = self.attn(
            query_states, key_states, value_states, attention_masks, kv_caches,
            req_offsets, req_lengths, output_shape,
            self.layer_idx,
        )
        hidden_states = self.o_proj(hidden_states, force_reduce=False)

        if input_dim == 2:
            hidden_states = hidden_states.squeeze(0)
        return hidden_states
    
    def _communicate(
        self,
        hidden_states: torch.Tensor,
        async_op: bool = False,
    ):
        # show_rank_print(f"reach attn comm")
        # time1 = time.time()
        if self.parallel_config.attn_tp_group.group_size > 1:
            work = self.parallel_config.attn_tp_group.all_reduce(hidden_states, async_op=async_op)
            work.wait()
            output = work.output
            handler = work.handler
        else:
            output = hidden_states
            handler = None
        # show_rank_print(f"{time.time() - time1} {async_op}")
        # show_rank_print("finish attn comm")
        return output, handler
    
