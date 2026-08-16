# coding=utf-8
# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable, Optional, Union, List, Tuple

import torch
from torch import nn

from ...generation import ACT2FN
from ...generation.cache_utils import Cache
from transformers.modeling_layers import (
    GradientCheckpointingLayer,
)
from transformers.modeling_outputs import MoeCausalLMOutputWithPast, MoeModelOutputWithPast
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, can_return_tuple
from transformers.utils.generic import OutputRecorder
from .configuration_qwen3_moe import Qwen3MoeConfig

import time

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin,):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


from easyinfra.generation.parallel.parallel_state import GroupCommunicator       
from easyinfra.generation.modules.linear import RowParallelLinear, ColumnParallelLinear, MergedColumnParallelLinear
from easyinfra.generation.modules.attention.sdpa import (
    SdpaAttention,
)
from easyinfra.generation.modules.attention.attention import Attention
from easyinfra.generation.parallel.parallel_configuration import MoeParallelConfig
from easyinfra.schedule.request import RequestHub

from easyinfra.generation.blocks import AttentionBlock
from easyinfra import envs

class Qwen3MoeAttention(AttentionBlock):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, 
        config: Qwen3MoeConfig, 
        parallel_config: MoeParallelConfig,
        layer_idx: int,
    ):
        super().__init__(config, parallel_config, layer_idx)
        
        self.scaling = self.head_dim**-0.5
        
        self.input_layernorm = BaseRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.qkv_proj = MergedColumnParallelLinear(
            config.hidden_size, 
            (config.num_attention_heads * self.head_dim, 
             config.num_key_value_heads * self.head_dim, 
             config.num_key_value_heads * self.head_dim), 
            param_names=("q_proj","k_proj","v_proj"),
            bias=config.attention_bias, 
            tp_group=self.tp_group
        )
        self.attn = Attention(
            self.tp_num_attention_heads,
            self.head_dim,
            num_kv_heads=self.tp_num_key_value_heads,
            scale=self.scaling,
            dropout=0.0 if not self.training else self.attention_dropout,
        )
        self.o_proj = RowParallelLinear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias, tp_group=self.tp_group)
        
        self.q_norm = BaseRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = BaseRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        # self.stages = ((self.compute, self.communicate, self.compute_after_comm),)

    # @torch.compile(dynamic=True)    
    def _attn_compute(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_masks: Tuple[torch.Tensor],
        kv_caches: Tuple[Cache],
        seqlens_q: List[int],
        cu_seqlens_q: List[int],
        cu_seqlens_k: List[int],
    ):
        # time0 = time.time()
        # return hidden_states
        
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        
        query_states, key_states, value_states = self.qkv_proj(hidden_states).split(self.qkv_proj.split_sizes, dim=-1)
        query_states: torch.Tensor = self.q_norm(query_states.view(hidden_shape))
        key_states: torch.Tensor = self.k_norm(key_states.view(hidden_shape))
        value_states: torch.Tensor = value_states.view(hidden_shape)
        
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin) # TODO Attention
        
        # output_shape = (*input_shape, self.o_proj.tp_in_features)
        
        ### 
        hidden_states = self.attn(
            query_states, key_states, value_states, 
            attention_masks, kv_caches,
            seqlens_q, cu_seqlens_q, cu_seqlens_k, 
            self.layer_idx,
        )
        hidden_states = self.o_proj(hidden_states, force_reduce=False)

        return hidden_states
    
    def _communicate(
        self,
        hidden_states: torch.Tensor,
        async_op: bool = False,
    ):
        # show_rank_print(f"reach attn comm")
        # time1 = time.time()
        work = self.parallel_config.attn_tp_group.all_reduce(hidden_states, async_op=async_op)
        work.wait()
        # show_rank_print(f"{time.time() - time1} {async_op}")
        # show_rank_print("finish attn comm")
        return work.output, work.handler
    
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        _start = 0.0
        _compute_end = 0.0
        _end = 0.0
        
        _start = record_time_sync()
        hidden_states = self._attn_compute(
            hidden_states,
            position_embeddings,
            attention_mask,
            past_key_values,
        )
        _compute_end = record_time_sync()
        
        # self.parallel_config.attn_tp_group.barrier()
        
        _communicate_start = record_time_sync()
        hidden_states, _ = self._communicate(hidden_states)
        _end = record_time_sync()
        
        self.attn_compute_time += _compute_end - _start
        self.recv_token_time += _end - _communicate_start
        return hidden_states, None

from ...generation.modules.linear import BaseLinear
class Qwen3MoeMLP(nn.Module):
    def __init__(self, config: Qwen3MoeConfig, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size if intermediate_size is not None else config.intermediate_size
        # self.gate_proj = BaseLinear(self.hidden_size, self.intermediate_size, bias=False)
        # self.up_proj = BaseLinear(self.hidden_size, self.intermediate_size, bias=False)
        self.gate_up_proj = MergedColumnParallelLinear(
            self.hidden_size, 
            (self.intermediate_size, self.intermediate_size), 
            param_names=("gate_proj","up_proj"),
            bias=False,
        )
        self.down_proj = RowParallelLinear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        # return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        gate_states, up_states = self.gate_up_proj(x).split(self.gate_up_proj.split_sizes, dim=-1)
        return self.down_proj(self.act_fn(gate_states) * up_states)

from ...generation.parallel.parallel_utils import get_device
from ...utils import record_time_sync, show_rank_print
import math

from ...generation.functions import exclusive_cumsum


from ...generation.blocks import MoeBlock

class Qwen3MoeSparseMoeBlock(MoeBlock):
    def __init__(self, config: Qwen3MoeConfig, parallel_config: MoeParallelConfig, layer_idx: int):
        super().__init__(
            layer_idx=layer_idx,
            hidden_size=config.hidden_size,
            moe_intermediate_size=config.moe_intermediate_size,
            activation_key=config.hidden_act,
            gate_up_bias=False,
            down_bias=False,
            num_logical_experts=config.num_experts,
            num_experts_per_tok=config.num_experts_per_tok,
            norm_topk_prob=config.norm_topk_prob,
            use_eplb=parallel_config.use_eplb,
            has_shared=False,
            shared_expert_intermediate_size=None,
            chunk_routing_group=parallel_config.attn_tp_group,
            shared_expert_tp_group=None,
            ep_group=parallel_config.ep_group,
            moe_tp_group=parallel_config.moe_tp_group,
            num_global_experts=parallel_config.num_global_experts,
            parallel_config=parallel_config,
        )
        # gating
        self.post_attention_layernorm = BaseRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # We do not partition gate
        self.gate = BaseLinear(config.hidden_size, config.num_experts, bias=False)

    def _routing(self, hidden_states: torch.Tensor,):
        '''
            Returns: routing_weights, selected_experts, (something else)
        '''
        # The naive routing
        routing_weights = nn.functional.softmax(self.gate(hidden_states), dim=-1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.num_experts_per_tok, dim=-1)
        if self.norm_topk_prob:
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        # back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)
        return routing_weights, selected_experts, None
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
        raise NotImplementedError
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        timer = SyncTimeRecorder()
        timestamp_start = timer.record_time_sync()
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        
        hidden_states = hidden_states.view(-1, hidden_dim)
        # start_hidden_states = hidden_states.clone()
        
        batch_sequence_length = hidden_states.shape[0]
                        
        (
            hidden_states,
            selected_experts, 
            routing_weights, 
            router_logits,
            chunk_size,
        ) = self._chunk_routing_compute(hidden_states)
        
        all_selected_experts, _ = self._expert_data_communicate(selected_experts)
        
        (
            local_expert_mask, 
            device_expert_mask,
            expert_count, 
            expert_hit
        ) = self._all2all_expert_data_compute(selected_experts, all_selected_experts)
        
        (
            send_token_split, 
            recv_token_split, 
            send_indices, 
            send_weight_top_k, 
            recv_hidden_length,
        ) = self._all2all_prepare_token_communicate(hidden_states, local_expert_mask, device_expert_mask)
                
        # show_rank_print(f"expert count: {expert_count}")
        routing_weights, hidden_states, _ = self._all2all_token_communicate(
            hidden_states, 
            routing_weights, 
            send_token_split, 
            recv_token_split, 
            send_indices, 
            send_weight_top_k, 
            recv_hidden_length,
        )
        
        
        hidden_states = self._all2all_expert_compute(
            hidden_states,
            device_expert_mask,
            expert_hit,
        )
        
        timestamp6 = timer.record_time_sync()
        # self.parallel_config.ep_group.barrier()
        timestamp10 = timer.record_time_sync()
        # send back
        hidden_states, _ = self._all2all_expert_output_communicate(hidden_states, send_token_split, recv_token_split)
        timestamp7 = timer.record_time_sync()
        # add to construct final output
        hidden_states = self._all2all_chunk_output_compute(chunk_size, send_indices, hidden_states)
        timestamp8 = timer.record_time_sync()
        hidden_states, _ = self._all2all_gather_chunk_output_communicate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_dim)[:batch_sequence_length,:].view(batch_size,sequence_length,hidden_dim)
        timestamp9 = timer.record_time_sync()
        
        ####################################################
        
        return hidden_states, router_logits


# @use_kernel_forward_from_hub("RMSNorm")

from ...generation.modules.rmsnorm import BaseRMSNorm
class Qwen3MoeDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3MoeConfig, parallel_config: MoeParallelConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        
        self.self_attn = Qwen3MoeAttention(config, parallel_config, layer_idx)
        self.mlp = Qwen3MoeSparseMoeBlock(config, parallel_config, layer_idx)
        
        # stats
        self.attn_time = 0.0
        self.mlp_time = 0.0
        
        self.attn_compute_time = 0.0
        self.attn_recv_time = 0.0
        
        self.expert_compute_time = 0.0
        self.expert_loop_out_prepare_time = 0.0
        self.expert_loop_in_prepare_time = 0.0
        self.expert_compute_to_barrier_time = 0.0
        self.expert_pre_compute_time = 0.0
        self.expert_post_recv_all2all_time = 0.0
        self.send_token_time = 0.0
        self.recv_token_time = 0.0
        self.barrier_wait_time = 0.0
        self.cp_gather_time = 0.0

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        **kwargs,
    ) -> torch.FloatTensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_router_logits (`bool`, *optional*):
                Whether or not to return the logits of all the routers. They are useful for computing the router loss,
                and should not be returned during inference.
            past_key_values (`Cache`, *optional*): cached past key and value projection states
            position_embeddings (`tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """
        residual = hidden_states

        # Self Attention
        time0 = record_time_sync()
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            **kwargs,
        )
        hidden_states = residual + hidden_states
        
        # time1 = record_time_sync()

        # Fully Connected
        residual = hidden_states
        hidden_states = self.mlp(hidden_states)
        # For the MoE layers, we need to unpack
        if isinstance(hidden_states, tuple):
            hidden_states, _ = hidden_states
        hidden_states = residual + hidden_states
        
        # time2 = record_time_sync()
                
        # this_attn_time = time1 - time0
        # this_mlp_time = time2 - time1
        # self.attn_time += this_attn_time
        # self.mlp_time += this_mlp_time
        
        # attn_compute_time, attn_recv_time = attn_time_stats
        # self.attn_compute_time += attn_compute_time
        # self.attn_recv_time += attn_recv_time
        
        # self.expert_compute_time += self.mlp.expert_compute_time
        # self.expert_loop_out_prepare_time += self.mlp.expert_loop_out_prepare_time
        # self.expert_loop_in_prepare_time += self.mlp.expert_loop_in_prepare_time
        # self.expert_compute_to_barrier_time += self.mlp.expert_compute_to_barrier_time
        # self.expert_pre_compute_time += self.mlp.expert_pre_compute_time
        # self.expert_post_recv_all2all_time += self.mlp.expert_post_recv_all2all_time
        # self.send_token_time += self.mlp.send_token_time
        # self.recv_token_time += self.mlp.recv_token_time
        # self.barrier_wait_time += self.mlp.barrier_wait_time
        
        # show_rank_print(f"attn time: {time1 - time0}, mlp time: {time2 - time1}")
        # show_rank_print(f"mlp time: {time2 - time1}")

        return hidden_states

from ...modeling_utils.causal_lm import CausalLMMoePretrainedModel

class Qwen3MoePreTrainedModel(CausalLMMoePretrainedModel):
    config: Qwen3MoeConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3MoeDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    # _supports_flash_attn = True
    _supports_sdpa = True
    # _supports_flex_attn = True
    _can_compile_fullgraph = False  # MoE models don't work with torch.compile (`torch.where(condition)` not supported)
    _supports_attention_backend = True
    _can_record_outputs = {
        "router_logits": OutputRecorder(Qwen3MoeSparseMoeBlock, index=1),
        "hidden_states": Qwen3MoeDecoderLayer,
        "attentions": Qwen3MoeAttention,
    }

from ...generation.modules.word_embed.embedding import Embedding
from ...generation.modules.position_embed import BaseRotaryEmbedding
from ...schedule.scheduler import MoeLayerScheduler
class Qwen3MoeModel(Qwen3MoePreTrainedModel):
    def __init__(self, config: Qwen3MoeConfig, parallel_config: MoeParallelConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.rotary_emb = BaseRotaryEmbedding(config=config)
        # self.first_k_dense_replace = 0
        self.layers = nn.ModuleList(
            [Qwen3MoeDecoderLayer(config, parallel_config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = BaseRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        
        self.config = config
        self.parallel_config: MoeParallelConfig = parallel_config
        
        # Initialize weights and apply final processing
        self.post_init()

    # @check_model_inputs()
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        scheduler: Optional[MoeLayerScheduler] = None,
        enable_scheduler: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeModelOutputWithPast:
        if not enable_scheduler:
            if past_key_values is None:
                raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
            
        hidden_states: Union[torch.Tensor, List[torch.Tensor]] = self.embed_tokens(input_ids)
        h_dtype, h_device_type = (hidden_states.dtype, hidden_states.device.type) \
            if isinstance(hidden_states, torch.Tensor) else \
                (hidden_states[0].dtype, hidden_states[0].device.type)
        position_embeddings = self.rotary_emb(position_ids, dtype=h_dtype, device_type=h_device_type)

        executed_decoder_layers = self.layers[: self.config.num_hidden_layers]
        
        torch.cuda.synchronize()
        time1 = time.time()
        if not enable_scheduler:
            for decoder_layer in executed_decoder_layers:
                hidden_states = decoder_layer(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    **kwargs,
                )
        else:
            # scheduler
            scheduler.update_tensors(hidden_states, position_embeddings)
            scheduler.run()
            hidden_states = scheduler.get_output_hidden_states()
            
        hidden_states = self.norm(hidden_states)
        
        torch.cuda.synchronize()
        time2 = time.time()
        self.layers_forward_time += time2 - time1
        
        self.attn_full_time = sum([layer.attn_time for layer in executed_decoder_layers])
        self.mlp_full_time = sum([layer.mlp_time for layer in executed_decoder_layers])
        
        self.attn_compute_full_time = sum([layer.self_attn.attn_compute_time for layer in executed_decoder_layers])
        self.attn_recv_full_time = sum([layer.self_attn.recv_token_time for layer in executed_decoder_layers])
        
        self.expert_compute_full_time = sum([layer.mlp.expert_compute_time for layer in executed_decoder_layers])
        self.expert_loop_out_prepare_full_time = sum([layer.mlp.expert_loop_out_prepare_time for layer in executed_decoder_layers])
        self.expert_loop_in_prepare_full_time = sum([layer.mlp.expert_loop_in_prepare_time for layer in executed_decoder_layers])
        self.expert_pre_compute_full_time = sum([layer.mlp.expert_pre_compute_time for layer in executed_decoder_layers])
        self.expert_compute_to_barrier_full_time = sum([layer.mlp.expert_compute_to_barrier_time for layer in executed_decoder_layers])
        self.barrier_wait_full_time = sum([layer.mlp.barrier_wait_time for layer in executed_decoder_layers])
        self.recv_token_full_time = sum([layer.mlp.recv_token_time for layer in executed_decoder_layers])
        self.expert_post_recv_all2all_full_time = sum([layer.mlp.expert_post_recv_all2all_time for layer in executed_decoder_layers])
        
        self.send_token_full_time = sum([layer.mlp.send_token_time for layer in executed_decoder_layers])
        # self.cp_gather_full_time = sum([layer.cp_gather_time for layer in executed_decoder_layers])
        # show_rank_print(f"attn full time: {self.attn_full_time}, mlp full time: {self.mlp_full_time}")
        # show_rank_print(f"attn: compute: {self.attn_compute_full_time}, recv: {self.attn_recv_full_time}")
        # show_rank_print(f"expert: pre_compute: loop_out:{self.expert_loop_out_prepare_full_time}/{self.expert_pre_compute_full_time}. compute2barrier: {self.expert_compute_full_time}+{self.expert_loop_in_prepare_full_time}/{self.expert_compute_to_barrier_full_time}, wait: {self.barrier_wait_full_time}, recv: {self.recv_token_full_time}, post_recv: {self.expert_post_recv_all2all_full_time}")
        # show_rank_print(f"send token time: {self.send_token_full_time}, recv token time: {self.recv_token_full_time}, compute: {self.expert_compute_full_time}, wait ep time: {self.barrier_wait_full_time}")
        # show_rank_print(f"wait ep time: {self.barrier_wait_full_time}")
        # show_rank_print(f"expert: compute: {self.expert_compute_full_time}, loop out: {self.expert_loop_out_prepare_full_time}, loop in: {self.expert_loop_in_prepare_full_time}")
        

        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )
    
from ...generation.parallel.parallel_configuration import MoeParallelConfig
class Qwen3MoeForCausalLM(Qwen3MoePreTrainedModel):
    # _tied_weights_keys = ["lm_head.weight"]
    def param_name_to_module_with_kwargs(self, param_name: str):
        
        module_name = param_name
        if "mlp.experts" in param_name:  
            # double "experts"
            module_name = module_name.replace("experts", "experts.experts")
            if not envs.MOE_IMPL == "fused":
                if "gate_proj" in module_name:
                    module_name = module_name.replace("gate_proj", "gate_up_proj")
                    module_kwargs = {"param_name": "gate_proj"}
                elif "up_proj" in module_name:
                    module_name = module_name.replace("up_proj", "gate_up_proj")
                    module_kwargs = {"param_name": "up_proj"}
                else:
                    module_kwargs = {}
            ## get physical id
            ## module_names could be empty, and could be multiple under EPLB
            module_names = self.get_physical_expert_module_names(self, module_name)  
            if envs.MOE_IMPL == "fused":
                res = []
                for name in module_names:
                    assert any(_name in module_name for _name in ("gate_proj", "up_proj", "down_proj"))
                    module_name_split = name.split("experts.experts.")
                    assert len(module_name_split) == 2
                    res.append((module_name_split[0] + "experts.experts", {"param_name": module_name_split[1]}))
            else:
                res = tuple((name, module_kwargs) for name in module_names)
            return res
        
        if "q_proj" in module_name:
            return ((module_name.replace("q_proj", "qkv_proj"), {"param_name": "q_proj"}),)
        elif "k_proj" in module_name:
            return ((module_name.replace("k_proj", "qkv_proj"), {"param_name": "k_proj"}),)
        elif "v_proj" in module_name:
            return ((module_name.replace("v_proj", "qkv_proj"), {"param_name": "v_proj"}),)
            
        elif "input_layernorm" in module_name:
            return ((module_name.replace("input_layernorm", "self_attn.input_layernorm"),{}),)
        elif "post_attention_layernorm" in module_name:
            return ((module_name.replace("post_attention_layernorm", "mlp.post_attention_layernorm"),{}),)
        else:
            return ((module_name,{}),)
            
    def get_head_dim(self) -> int:
        return self.model.layers[0].self_attn.head_dim
    
    def __init__(self, config: Qwen3MoeConfig, parallel_config: MoeParallelConfig):
        super().__init__(config)
        self.parallel_config = parallel_config
        self.model = Qwen3MoeModel(config, parallel_config)
        self.vocab_size = config.vocab_size
        self.lm_head = BaseLinear(config.hidden_size, config.vocab_size, bias=False)
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.first_k_dense_replace = 0

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        labels: Optional[torch.LongTensor] = None,
        output_router_logits: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = None,
        scheduler: Optional[MoeLayerScheduler] = None,
        **kwargs,
    ) -> MoeCausalLMOutputWithPast:
        r"""
        ```"""
        config_output_router_logits = getattr(self.config, "output_router_logits", False)
        output_router_logits = (
            output_router_logits if output_router_logits is not None else config_output_router_logits
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: MoeModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_router_logits=output_router_logits,
            scheduler=scheduler,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        if logits_to_keep is not None:
            hidden_states = hidden_states[logits_to_keep, :]
        logits = self.lm_head(hidden_states)

        # loss = None
        # if labels is not None:
        #     loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)

        # aux_loss = None
        # if output_router_logits:
        #     aux_loss = load_balancing_loss_func(
        #         outputs.router_logits,
        #         self.num_experts,
        #         self.num_experts_per_tok,
        #         attention_mask,
        #     )
        #     if labels is not None:
        #         loss += self.router_aux_loss_coef * aux_loss.to(loss.device)  # make sure to reside in the same device

        return MoeCausalLMOutputWithPast(
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )



__all__ = [
    "Qwen3MoeForCausalLM",
    "Qwen3MoeModel",
    "Qwen3MoePreTrainedModel",
]
