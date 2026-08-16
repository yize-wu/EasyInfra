from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..generation.cache_utils import TreeDynamicCache
    from ..generation.utils.configuration_utils import MoeConfig
    from ..generation.utils.configuration_utils import PretrainedConfig
    from ..generation.parallel.parallel_configuration import MoeParallelConfig

import torch
from torch import nn

from typing import Optional, Iterable

from .base_parallel import BaseParallelPreTrainedModel

from ..generation.modules import (
    Embedding,
    BaseRMSNorm,
)

from ..modeling_layers import GradientCheckpointingLayer


from ..utils.stats import record_time_sync

class MoeDecoderLayer(GradientCheckpointingLayer):
    def __init__(
        self, 
        config: MoeConfig, 
        parallel_config: MoeParallelConfig, 
        layer_idx: int,
        **moe_decoder_layer_kwargs
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        # attention
        # attention_proj_norm = moe_decoder_layer_kwargs.pop("attention_proj_norm", None)
        # if attention_proj_norm is None or attention_proj_norm == "":
        #     self.self_attn = QKVOAttention(config, parallel_config, layer_idx)
        # else:
        #     self.self_attn = QKVONormAttention(config, parallel_config, layer_idx)
        
        # mlp
        # self.mlp = SparseMoeBlock(config, parallel_config, layer_idx)

        self.input_layernorm = BaseRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = BaseRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # stats
        self.attn_time = 0.0
        self.mlp_time = 0.0
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
        past_key_values: Optional[TreeDynamicCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> torch.FloatTensor:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        time0 = record_time_sync()
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states
        time1 = record_time_sync()

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        # For the MoE layers, we need to unpack
        if isinstance(hidden_states, tuple):
            hidden_states, _, expert_time_stats = hidden_states
        hidden_states = residual + hidden_states
        time2 = record_time_sync()
        
        this_attn_time = time1 - time0
        this_mlp_time = time2 - time1
        self.attn_time += this_attn_time
        self.mlp_time += this_mlp_time
        
        expert_compute_time, expert_loop_out_prepare_time, expert_loop_in_prepare_time, expert_compute_to_barrier_time, expert_pre_compute_time, expert_post_recv_all2all_time, send_token_time, recv_token_time, barrier_wait_time, cp_gather_time = expert_time_stats
        self.expert_compute_time += expert_compute_time
        self.expert_loop_out_prepare_time += expert_loop_out_prepare_time
        self.expert_loop_in_prepare_time += expert_loop_in_prepare_time
        self.expert_compute_to_barrier_time += expert_compute_to_barrier_time
        self.expert_pre_compute_time += expert_pre_compute_time
        self.expert_post_recv_all2all_time += expert_post_recv_all2all_time
        self.send_token_time += send_token_time
        self.recv_token_time += recv_token_time
        self.barrier_wait_time += barrier_wait_time
        self.cp_gather_time += cp_gather_time
        # show_rank_print(f"attn time: {time1 - time0}, mlp time: {time2 - time1}")
        # show_rank_print(f"mlp time: {time2 - time1}")

        return hidden_states
    
class MoeLayers(nn.ModuleList):
    def __init__(self, modules: Iterable):
        for i,module in enumerate(modules):
            if not isinstance(module, MoeDecoderLayer):
                raise ValueError(f"The {i}-th module class is {type(module)}, not MoeDecoderLayer")
        super().__init__(modules)

class MoePretrainedModel(BaseParallelPreTrainedModel):
    def __init__(self, config: PretrainedConfig, parallel_config: MoeParallelConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [MoeDecoderLayer(config, parallel_config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = BaseRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # self.rotary_emb = BaseRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[TreeDynamicCache] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs
    ):
        if (input_ids is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if use_cache and past_key_values is None:
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + input_ids.shape[1], device=input_ids.device
            )
        if position_ids is None:
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
            position_ids = cache_position.unsqueeze(0)
        
        inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds
        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        executed_decoder_layers = self.layers[: self.config.num_hidden_layers]
        for decoder_layer in executed_decoder_layers:
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )
        
        self.attn_full_time = sum([layer.attn_time for layer in executed_decoder_layers])
        self.mlp_full_time = sum([layer.mlp_time for layer in executed_decoder_layers])
        self.expert_compute_full_time = sum([layer.expert_compute_time for layer in executed_decoder_layers])
        self.expert_loop_out_prepare_full_time = sum([layer.expert_loop_out_prepare_time for layer in executed_decoder_layers])
        self.expert_loop_in_prepare_full_time = sum([layer.expert_loop_in_prepare_time for layer in executed_decoder_layers])
        self.expert_pre_compute_full_time = sum([layer.expert_pre_compute_time for layer in executed_decoder_layers])
        self.expert_compute_to_barrier_full_time = sum([layer.expert_compute_to_barrier_time for layer in executed_decoder_layers])
        self.barrier_wait_time_full_time = sum([layer.barrier_wait_time for layer in executed_decoder_layers])
        self.recv_token_full_time = sum([layer.recv_token_time for layer in executed_decoder_layers])
        self.expert_post_recv_all2all_full_time = sum([layer.expert_post_recv_all2all_time for layer in executed_decoder_layers])
        
        self.send_token_full_time = sum([layer.send_token_time for layer in executed_decoder_layers])
        self.cp_gather_full_time = sum([layer.cp_gather_time for layer in executed_decoder_layers])
        # show_rank_print(f"attn full time: {self.attn_full_time}, mlp full time: {self.mlp_full_time}")
        show_rank_print(f"expert: pre_compute: loop_out:{self.expert_loop_out_prepare_full_time}/{self.expert_pre_compute_full_time}. compute2barrier: {self.expert_compute_full_time}+{self.expert_loop_in_prepare_full_time}/{self.expert_compute_to_barrier_full_time}, wait: {self.barrier_wait_time_full_time}, recv: {self.recv_token_full_time}, post_recv: {self.expert_post_recv_all2all_full_time}")
        # show_rank_print(f"send token time: {self.send_token_full_time}, recv token time: {self.recv_token_full_time}, compute: {self.expert_compute_full_time}, wait ep time: {self.barrier_wait_time_full_time}")
        # show_rank_print(f"expert: compute: {self.expert_compute_full_time}, loop out: {self.expert_loop_out_prepare_full_time}, loop in: {self.expert_loop_in_prepare_full_time}")
        
        
        hidden_states = self.norm(hidden_states)

        return MoeModelOutputWithPast(  # only diff with Mistral is the output type, we need MoE
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )