# coding=utf-8
# Copy from https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/modeling_deepseek.py
# Copyright 2023 DeepSeek-AI and The HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
""" PyTorch DeepSeek model."""
import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
import time

from ...generation import ACT2FN
from ...generation.cache_utils import Cache
from transformers.modeling_layers import (
    GradientCheckpointingLayer,
)
from transformers.modeling_outputs import (
    MoeCausalLMOutputWithPast, 
    MoeModelOutputWithPast
)
from .configuration_deepseek_v3 import DeepseekV3Config
from easyinfra import envs

def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(
        torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0)
    )
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


from ...generation.modules import BaseRMSNorm
from ...generation.parallel.communicator import GroupCommunicator
from ...generation.modules.linear import RowParallelLinear, ColumnParallelLinear, BaseLinear, MergedColumnParallelLinear
from ...generation.parallel.parallel_configuration import MoeParallelConfig
from ...generation.modules.attention import SdpaAttention
from ...generation.modules.attention import Attention
from ...generation.modules.position_embed import BaseRotaryEmbedding
from ...utils.stats import show_rank_print

from ...schedule.request import RequestHub


from ...generation.blocks.mlp import MlpBlock
# class DeepseekV3ExpertMLP(MlpBlock):
#     def __init__(
#         self, 
#         config: DeepseekV3Config, 
#         layer_idx: int,
#         tp_group: GroupCommunicator,
#         hidden_size=None, 
#         intermediate_size=None,
#     ):
#         hidden_size = config.hidden_size if hidden_size is None else hidden_size
#         intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size
            
#         super().__init__(
#             layer_idx=layer_idx,
#             hidden_size=hidden_size, 
#             intermediate_size=intermediate_size, 
#             tp_group=tp_group,
#         )

#         # self.gate_up_proj = MergedColumnParallelLinear(
#         #     self.hidden_size, 
#         #     (self.intermediate_size, self.intermediate_size), 
#         #     param_names=("gate_proj","up_proj"),
#         #     bias=False,
#         #     tp_group=self.tp_group,
#         # )
#         # self.down_proj = RowParallelLinear(self.intermediate_size, self.hidden_size, bias=False, tp_group=self.tp_group)
#         # self.act_fn = ACT2FN[config.hidden_act]
    
#     def forward(self, x):
#         gate_states, up_states = self.gate_up_proj(x).split(self.gate_up_proj.split_sizes, dim=-1)
#         x = self.down_proj(self.act_fn(gate_states) * up_states)
#         x_work = self.tp_group.all_reduce(x, async_op=False)
#         return x


class DeepseekV3DenseMLP(MlpBlock):
    def __init__(
        self, 
        config: DeepseekV3Config, 
        layer_idx: int,
        hidden_size: int, 
        intermediate_size: int,
        tp_group: GroupCommunicator,
    ):
        super().__init__(
            layer_idx=layer_idx,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            activation_key=config.hidden_act,
            gate_up_bias=False,
            down_bias=False,
            tp_group=tp_group,
        )
        self.post_attention_layernorm = BaseRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

from ...generation.modules.weight_load_utils import BaseWeightLoader
class DeepseekV3MoEGate(nn.Module, BaseWeightLoader):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.scoring_func = config.scoring_func
        self.seq_aux = config.seq_aux
        self.topk_method = config.topk_method
        self.n_group = config.n_group
        self.topk_group = config.topk_group

        # topk selection algorithm
        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(
            torch.empty((self.n_routed_experts, self.gating_dim))
        )
        if self.topk_method == "noaux_tc":
            self.e_score_correction_bias = nn.Parameter(
                torch.empty((self.n_routed_experts))
            )
    
    def forward(self, hidden_states: torch.Tensor):
        seq_len, _ = hidden_states.shape
        
        ### scores
        logits = F.linear(
            hidden_states.type(torch.float32), self.weight.type(torch.float32), None
        )
        if self.scoring_func == "sigmoid":
            scores = logits.sigmoid()
        else:
            raise NotImplementedError(
                f"insupportable scoring function for MoE gating: {self.scoring_func}"
            )

        ### select top-k experts
        if self.topk_method == "noaux_tc":
            assert not self.training
            scores_for_choice = scores.view(seq_len, -1) + self.e_score_correction_bias.unsqueeze(0)
            group_scores = (
                scores_for_choice.view(seq_len, self.n_group, -1).topk(2, dim=-1)[0].sum(dim = -1)
            )  # [n, n_group]
            group_idx = torch.topk(
                group_scores, k=self.topk_group, dim=-1, sorted=False
            )[
                1
            ]  # [n, top_k_group]
            group_mask = torch.zeros_like(group_scores)  # [n, n_group]
            group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
            score_mask = (
                group_mask.unsqueeze(-1)
                .expand(
                    seq_len, self.n_group, self.n_routed_experts // self.n_group
                )
                .reshape(seq_len, -1)
            )  # [n, e]
            tmp_scores = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)  # [n, e]
            _, topk_idx = torch.topk(
                tmp_scores, k=self.num_experts_per_tok, dim=-1, sorted=False
            )
            topk_weight = scores.gather(1, topk_idx)
        else:
            raise NotImplementedError(
                f"insupportable TopK function for MoE gating: {self.topk_method}"
            )

        ### norm gate to sum 1
        if self.num_experts_per_tok > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator
        topk_weight = topk_weight * self.routed_scaling_factor # must multiply the scaling factor
        topk_weight = topk_weight.to(hidden_states.dtype)

        return topk_weight, topk_idx, None
        

from ...generation.blocks import MoeBlock
class DeepseekV3MoE(MoeBlock):
    """
    A mixed expert module containing shared experts.
    """

    def __init__(self, config: DeepseekV3Config, parallel_config: MoeParallelConfig, layer_idx: int):
        shared_expert_intermediate_size = config.moe_intermediate_size * config.n_shared_experts
        super().__init__(
            layer_idx=layer_idx, 
            hidden_size=config.hidden_size, 
            moe_intermediate_size=config.moe_intermediate_size,
            activation_key=config.hidden_act,
            gate_up_bias=False,
            down_bias=False,
            num_logical_experts=config.n_routed_experts,
            num_experts_per_tok=config.num_experts_per_tok,
            norm_topk_prob=config.norm_topk_prob,
            use_eplb=parallel_config.use_eplb,
            num_first_k_dense_layers=config.first_k_dense_replace,
            chunk_routing_group=parallel_config.attn_tp_group,
            ep_group=parallel_config.ep_group,
            moe_tp_group=parallel_config.moe_tp_group,
            has_shared=True,
            shared_expert_intermediate_size=shared_expert_intermediate_size,
            shared_activation_key=config.hidden_act,
            shared_gate_up_bias=False,
            shared_down_bias=False,
            shared_expert_tp_group=parallel_config.shared_expert_tp_group,
            num_global_experts=parallel_config.num_global_experts,
            parallel_config=parallel_config,
        )
        self.post_attention_layernorm = BaseRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gate = DeepseekV3MoEGate(config)
        # if config.n_shared_experts is not None:
        #     intermediate_size = config.moe_intermediate_size * config.n_shared_experts
        #     self.shared_experts = DeepseekV3ExpertMLP(
        #         config, layer_idx, tp_group=parallel_config.shared_expert_tp_group, intermediate_size=intermediate_size,
        #     )
        # else:
        #     self.shared_experts = None
            
            
    def _routing(self, hidden_states: torch.Tensor):
        return self.gate(hidden_states)

    def forward(self, hidden_states):
        raise NotImplementedError
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

        y = self.moe_infer(hidden_states, topk_idx, topk_weight).view(*orig_shape)
        if self.config.n_shared_experts is not None:
            y = y + self.shared_experts(identity)
        return y


from ..qwen3_moe.modeling_qwen3_moe import rotate_half, apply_rotary_pos_emb

def head_dim_inverse_interleave(t: torch.Tensor):
    other_dims, d = t.shape[:-1], t.shape[-1]
    return t.view(*other_dims, d // 2, 2).transpose(-1, -2).reshape(*other_dims, d)
def apply_rotary_pos_emb_interleave(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
    layout: str = "shd",
):
    r"""
    TODO let's just use the original freqcis computation to not have the view
    transpose + reshape! This is not optimized!
    Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
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
    
    q = head_dim_inverse_interleave(q)
    k = head_dim_inverse_interleave(k)
    # s, h, d = q.shape
    # q = q.view(s, h, d // 2, 2).transpose(-1, -2).reshape(s, h, d)

    # s, h, d = k.shape
    # k = k.view(s, h, d // 2, 2).transpose(-1, -2).reshape(s, h, d)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


from ...modeling_utils.causal_lm import CausalLMMoePretrainedModel
from ...generation.blocks.attention import AttentionBlock

class DeepseekV3Attention(AttentionBlock):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self, 
        config: DeepseekV3Config, 
        parallel_config: MoeParallelConfig,
        layer_idx: int
    ):
        super().__init__(config, parallel_config, layer_idx)
        self.config = config
        self.layer_idx = layer_idx
        self.parallel_config = parallel_config
    
        self.rope_theta = config.rope_theta
        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim

        self.softmax_scale = self.qk_head_dim ** (-0.5)
        
        self.input_layernorm = BaseRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        if self.q_lora_rank is None:
            self.q_proj = ColumnParallelLinear(
                self.hidden_size, self.num_attention_heads * self.qk_head_dim, bias=False, tp_group=self.parallel_config.attn_tp_group,
            )
        else:
            self.q_a_proj = BaseLinear(
                self.hidden_size, config.q_lora_rank, bias=config.attention_bias
            )
            self.q_a_layernorm = BaseRMSNorm(config.q_lora_rank)
            self.q_b_proj = ColumnParallelLinear(
                config.q_lora_rank, self.num_attention_heads * self.qk_head_dim, bias=False, tp_group=self.parallel_config.attn_tp_group,
            )

        # kv_a_proj_with_mqa cannot be merged with q_proj
        self.kv_a_proj_with_mqa = BaseLinear(
            self.hidden_size,
            config.kv_lora_rank + config.qk_rope_head_dim,
            bias=config.attention_bias,
        )
        self.kv_a_layernorm = BaseRMSNorm(config.kv_lora_rank)
        self.kv_b_proj = ColumnParallelLinear(
            config.kv_lora_rank,
            self.num_attention_heads
            * (self.qk_head_dim - self.qk_rope_head_dim + self.v_head_dim),
            bias=False,
            tp_group=self.parallel_config.attn_tp_group,
        )
        
        self.attn = Attention(
            self.tp_num_attention_heads,
            self.qk_head_dim,
            v_head_dim=self.v_head_dim,
            num_kv_heads=self.tp_num_key_value_heads,
            scale=self.softmax_scale,
            dropout=0.0 if not self.training else self.attention_dropout,
        )

        self.o_proj = RowParallelLinear(
            self.num_attention_heads * self.v_head_dim,
            self.hidden_size,
            bias=config.attention_bias,
            tp_group=self.parallel_config.attn_tp_group,
        )
        # self._init_rope()

        if self.config.rope_scaling is not None:
            raise NotImplementedError
            mscale_all_dim = self.config.rope_scaling.get("mscale_all_dim", 0)
            scaling_factor = self.config.rope_scaling["factor"]
            if mscale_all_dim:
                mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
                self.softmax_scale = self.softmax_scale * mscale * mscale

    def _attn_compute(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_masks: Optional[torch.Tensor],
        kv_caches: Tuple[Cache],
        seqlens_q: List[int],
        cu_seqlens_q: List[int],
        cu_seqlens_k: List[int],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
            
        q_len, _ = hidden_states.size()

        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q: torch.Tensor
        
        q = q.view(q_len, self.tp_num_attention_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_pe = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        k_pe = k_pe.view(q_len, 1, self.qk_rope_head_dim)
        kv = (
            self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
            .view(q_len, self.tp_num_attention_heads, self.qk_nope_head_dim + self.v_head_dim)
        )

        k_nope, value_states = torch.split(
            kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )

        cos, sin = position_embeddings

        if self.config.rope_interleave:  # support using interleaved weights for efficiency
            q_pe, k_pe = apply_rotary_pos_emb_interleave(q_pe, k_pe, cos, sin)
        else:
            q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin)

        query_states = torch.cat((q_nope, q_pe), dim=-1)

        key_states = torch.cat((k_nope, k_pe.expand((q_len, self.tp_num_attention_heads, self.qk_rope_head_dim))), dim=-1)
                
        hidden_states = self.attn(
            query_states, key_states, value_states, 
            attention_masks, kv_caches,
            seqlens_q, cu_seqlens_q, cu_seqlens_k,
            self.layer_idx,
        )
        hidden_states = self.o_proj(hidden_states)

        return hidden_states





class DeepseekV3DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: DeepseekV3Config, parallel_config: MoeParallelConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = DeepseekV3Attention(config, parallel_config, layer_idx)
        self.mlp = (
            DeepseekV3MoE(config, parallel_config, layer_idx)
            if (
                config.n_routed_experts is not None
                and layer_idx >= config.first_k_dense_replace
                and layer_idx % config.moe_layer_freq == 0
            )
            else DeepseekV3DenseMLP(
                config, 
                layer_idx, 
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                tp_group=parallel_config.mlp_tp_group,
            )
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        residual = hidden_states

        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        return outputs




class DeepseekV3PreTrainedModel(CausalLMMoePretrainedModel):
    config: DeepseekV3Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    # _supports_flash_attn = True
    _supports_sdpa = True
    # _supports_flex_attn = True
    _can_compile_fullgraph = False  # MoE models don't work with torch.compile (`torch.where(condition)` not supported)
    _supports_attention_backend = True


from ...generation.modules.word_embed.embedding import Embedding
from ...generation.modules.position_embed import BaseRotaryEmbedding
from ...schedule.scheduler import MoeLayerScheduler
from ...schedule.chunk import ChunkBlockList

class DeepseekV3Model(DeepseekV3PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`DeepseekV3DecoderLayer`]

    Args:
        config: DeepseekV3Config
    """
    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = BaseRotaryEmbedding(
                config=self.config
            )
        else:
            raise NotImplementedError
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = DeepseekV3LinearScalingRotaryEmbedding(
                    self.qk_rope_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = DeepseekV3DynamicNTKScalingRotaryEmbedding(
                    self.qk_rope_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "yarn":
                kwargs = {
                    key: self.config.rope_scaling[key]
                    for key in [
                        "original_max_position_embeddings",
                        "beta_fast",
                        "beta_slow",
                        "mscale",
                        "mscale_all_dim",
                    ]
                    if key in self.config.rope_scaling
                }
                self.rotary_emb = DeepseekV3YarnRotaryEmbedding(
                    self.qk_rope_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                    **kwargs,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")
            
    def __init__(self, config: DeepseekV3Config, parallel_config: MoeParallelConfig):
        super().__init__(config)
        self.parallel_config: MoeParallelConfig = parallel_config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self._init_rope()
        self.first_k_dense_replace = config.first_k_dense_replace
        self.layers = nn.ModuleList(
            [
                DeepseekV3DecoderLayer(config, parallel_config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        if self.config.rope_scaling is None:
            self.rotary_emb = BaseRotaryEmbedding(
                config=self.config
            )
        else:
            raise NotImplementedError
        self.norm = BaseRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        scheduler: MoeLayerScheduler = None,
        enable_scheduler: bool = False,
        **kwargs,
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
            scheduler.update_tensors(hidden_states, position_embeddings)
            scheduler.run()
            hidden_states = scheduler.get_output_hidden_states()
            
        hidden_states = self.norm(hidden_states)
        
        torch.cuda.synchronize()
        time2 = time.time()
        self.layers_forward_time += time2 - time1
        
        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class DeepseekV3ForCausalLM(DeepseekV3PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]
    def param_name_to_module_with_kwargs(self, param_name: str):
        module_name = param_name
        if "mlp.experts" in param_name:    
            # check a long sequence of "mlp.experts" to avoid mismatch of "shared_experts"
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
        ## gate and up should be handled there, for shared experts
        elif "gate_proj" in module_name:
            return ((module_name.replace("gate_proj", "gate_up_proj"),{"param_name": "gate_proj"}),)
        elif "up_proj" in module_name:
            return ((module_name.replace("up_proj", "gate_up_proj"),{"param_name": "up_proj"}),)
        elif "input_layernorm" in module_name:
            return ((module_name.replace("input_layernorm", "self_attn.input_layernorm"),{}),)
        elif "post_attention_layernorm" in module_name:
            return ((module_name.replace("post_attention_layernorm", "mlp.post_attention_layernorm"),{}),)
        else:
            return ((module_name,{}),)

    def __init__(self, config: DeepseekV3Config, parallel_config: MoeParallelConfig):
        super().__init__(config)

        self.parallel_config = parallel_config
        
        self.first_k_dense_replace = config.first_k_dense_replace
        
        self.model = DeepseekV3Model(config, parallel_config)
        self.vocab_size = config.vocab_size
        self.lm_head = BaseLinear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

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
        **kwargs
    ) -> MoeCausalLMOutputWithPast:
        config_output_router_logits = getattr(self.config, "output_router_logits", False)
        output_router_logits = (
            output_router_logits if output_router_logits is not None else config_output_router_logits
        )
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

        return MoeCausalLMOutputWithPast(
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )

