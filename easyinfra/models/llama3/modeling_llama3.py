from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.models.llama.configuration_llama import LlamaConfig

import torch
import torch.distributed as dist
import torch._dynamo.config
import torch._inductor.config
from torch.nn.attention.flex_attention import BlockMask, create_block_mask

torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
# Experimental features to reduce compilation times, will be on by default in future
torch._inductor.config.fx_graph_cache = True 
torch._functorch.config.enable_autograd_cache = True
torch._inductor.config.triton.cudagraph_trees = False
torch._dynamo.config.cache_size_limit = 10000
torch._dynamo.config.suppress_errors = True
torch.set_float32_matmul_precision('high')

from torch import nn
from typing import List, Optional, Tuple, Union

import time
import warnings
from enum import Enum

from ...generation.cache_utils import (
    TreeDynamicCache,
)

from ...utils import rank0_print, record_time_sync
from ...modeling_utils.causal_lm import CausalLMPretrainedModel

from ...generation.modules.linear import RowParallelLinear, ColumnParallelLinear
from ...generation.parallel.parallel_state import GroupCommunicator       
from ...generation.modules.rmsnorm import BaseRMSNorm      
from ...generation.modules.activation.base_activation import ACT2FN
from ...generation.modules.attention.sdpa import (
    SdpaAttention,
    flatten_attn_output,
)
from ...generation.utils.attention_mask import (
    _prepare_4d_causal_attention_mask_with_cache_position
)
from ...generation.modules.word_embed.embedding import Embedding
from ...generation.modules.position_embed import BaseRotaryEmbedding       
from ...generation.modules.logits_head.tp_logits_head import TPLogitsHead
from ...generation.parallel.parallel_configuration import TPParallelConfig
from ...generation.parallel.parallel_state import (
    get_world_size
)

HEAD_DIM = -3
HIDDEN_SIZE_DIM = -1
HIDDEN_LENGTH_DIM = -2

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def _apply_rotary_pos_emb(hidden_states, cos, sin) -> torch.Tensor:
    # torch.compiler.cudagraph_mark_step_begin()
    h = (hidden_states * cos) + (rotate_half(hidden_states) * sin)
    return h
    # return (hidden_states * cos) + (rotate_half(hidden_states) * sin)

# @torch.compiler.disable(recursive=True)
def apply_rotary_pos_emb(q, k, cos, sin) -> torch.Tensor:
    # q = _apply_rotary_pos_emb(q, cos, sin)
    # k = _apply_rotary_pos_emb(k, cos, sin)
    # for h in (q,k):
    #     torch.compiler.cudagraph_mark_step_begin()
    #     q = (q * cos) + (rotate_half(q) * sin)
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k
       
 
class LlamaDecoderLayer(nn.Module):
    """
        We only do colwise and rowwise tensor parallelism to reduce communication overhead.
    """
    def __init__(self, config: LlamaConfig, layer_idx: int, tp_group: GroupCommunicator):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.layer_idx: int = layer_idx
        
        self.tp_group = tp_group
        self.tp_size = tp_group.group_size
        
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        
        self.tp_hidden_size = config.hidden_size
        self.tp_intermediate_size = config.intermediate_size // self.tp_size
        self.tp_num_heads = self.num_heads // self.tp_size
        self.tp_num_key_value_heads = self.num_key_value_heads // self.tp_size
        if self.tp_num_heads * self.tp_size != self.num_heads:
            raise ValueError
        
        self.input_layernorm = BaseRMSNorm(config.hidden_size, config.rms_norm_eps, )

        self.q_proj = ColumnParallelLinear(self.hidden_size, self.num_heads*self.head_dim, bias=config.attention_bias, tp_group=tp_group)
        self.k_proj = ColumnParallelLinear(self.hidden_size, self.num_key_value_heads*self.head_dim, bias=config.attention_bias, tp_group=tp_group)
        self.v_proj = ColumnParallelLinear(self.hidden_size, self.num_key_value_heads*self.head_dim, bias=config.attention_bias, tp_group=tp_group)
        self.attn = SdpaAttention()
        self.o_proj = RowParallelLinear(self.hidden_size, self.num_heads*self.head_dim, bias=config.o_proj_bias, tp_group=tp_group)
        
        self.post_attention_layernorm = BaseRMSNorm(config.hidden_size, config.rms_norm_eps, )
        
        self.up_proj = ColumnParallelLinear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias, tp_group=tp_group,)
        self.gate_proj = ColumnParallelLinear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias, tp_group=tp_group,)
        self.act_fn = ACT2FN[config.hidden_act]
        self.down_proj = RowParallelLinear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias, tp_group=tp_group,)
        
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
        bsz, q_len = hidden_states.shape[:2]
        residual = hidden_states
        
        hidden_states = self.input_layernorm(hidden_states)
        # attn_start = record_time_sync()
        query_states: torch.Tensor = self.q_proj(hidden_states)
        key_states: torch.Tensor = self.k_proj(hidden_states)
        value_states: torch.Tensor = self.v_proj(hidden_states)
        query_states = query_states.view(bsz, q_len, self.tp_num_heads, self.head_dim).transpose(-2,-3)
        key_states = key_states.view(bsz, q_len, self.tp_num_key_value_heads, self.head_dim).transpose(-2,-3)
        value_states = value_states.view(bsz, q_len, self.tp_num_key_value_heads, self.head_dim).transpose(-2,-3)
        
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
        
        hidden_states = self.attn(query_states, key_states, value_states, attention_mask=attention_mask)
        hidden_states = flatten_attn_output(hidden_states)
        hidden_states = self.o_proj(hidden_states, force_reduce=True)
        # print(record_time_sync() - attn_start)
        
        hidden_states = residual + hidden_states
        # print(record_time_sync()-start)
        # start = record_time_sync()
        
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.down_proj(self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states), force_reduce=True)        
        hidden_states = residual + hidden_states
        # print(record_time_sync()-start)
                        
        outputs = (hidden_states,)
        return outputs
                                            
    
class LlamaPreTrainedModel(CausalLMPretrainedModel):
    _model_type = 'llama'
    config_class = LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["MyLlamaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_sdpa = True
    _supports_cache_class = True

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

        
class LlamaModel(LlamaPreTrainedModel):
    def __init__(self, config: LlamaConfig, tp_group):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, idx, tp_group) for idx in range(config.num_hidden_layers)]
        )
        self.norm = BaseRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = BaseRotaryEmbedding(config=config)

        # Initialize weights and apply final processing
        self.post_init()
        
    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[TreeDynamicCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
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
        # print(position_ids)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers

        for decoder_layer in self.layers:
            kwargs = {}
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_embeddings=position_embeddings,
                **kwargs
            )
            hidden_states = layer_outputs[0]
            
        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=None,
            attentions=None,
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: TreeDynamicCache,
    ):
        raise NotImplementedError

        # if self.config._attn_implementation == "flash_attention_2":
        #     if attention_mask is not None and 0.0 in attention_mask:
        #         return attention_mask
        #     return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = False

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        # if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
        #     if AttentionMaskConverter._ignore_causal_mask_sdpa(
        #         attention_mask,
        #         inputs_embeds=input_tensor,
        #         past_key_values_length=past_seen_tokens,
        #         is_training=self.training,
        #     ):
        #         return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_length()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = _prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            min_dtype=min_dtype,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask
        

class LlamaForCausalLM(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    @classmethod
    def make_parallel_config(cls, config, **kwargs):
        all_ranks = [_ for _ in range(get_world_size())]
        parallel_config = TPParallelConfig(all_ranks)
        return parallel_config, kwargs

    def param_name_to_module_with_kwargs(self, param_name: str):
        module_name = param_name.split('.')
        if len(module_name) >= 4 and module_name[3] in ['self_attn', 'mlp']:
            module_name.pop(3) # 'self_attn' or 'mlp' is removed
        module_name = '.'.join(module_name)
        module_kwargs = {}
        return ((module_name,module_kwargs),)
    
    def need_load_weight(self, param_name:str, module_name:str):
        if 'rotary_emb.inv_freq' in module_name:
            # no need to load inv_freq
            return False
        else:
            return True
    
    def __init__(self, config, parallel_config):
        super().__init__(config)
        
        parallel_config: TPParallelConfig = parallel_config
        self.parallel_config = parallel_config
        self.tp_group = parallel_config.tp_group
        
        self.model = LlamaModel(config, self.tp_group)
        self.vocab_size = config.vocab_size
        self.lm_head = TPLogitsHead(config.hidden_size, config.vocab_size, bias=False, tp_group=self.tp_group, reduce=True)

        # Initialize weights and apply final processing
        self.post_init()
            
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[TreeDynamicCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        # outputs = self.model(
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     position_ids=position_ids,
        #     past_key_values=past_key_values,
        #     cache_position=cache_position,
        # )
        
        # hidden_states : torch.Tensor = outputs[0]
        
        # # Now, do not use need_reduce because we only need rank 0 to sample
        # logits = self.lm_head(hidden_states)
        # logits = logits.float()
        
        outputs, logits = model_forward(
            self.model, 
            self.lm_head, 
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
        )
                                         
        if not return_dict:
            output = (logits,) + outputs[1:]
            return output
        
        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

# @torch.compile(mode="reduce-overhead", fullgraph=True, dynamic=True)
def model_forward(model, lm_head, **kwargs):
    input_ids = kwargs.pop("input_ids")
    attention_mask = kwargs.pop("attention_mask")
    position_ids = kwargs.pop("position_ids")
    past_key_values = kwargs.pop("past_key_values")
    cache_position = kwargs.pop("cache_position")
    
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        cache_position=cache_position,
    )
    
    hidden_states : torch.Tensor = outputs[0]
    
    # Now, do not use need_reduce because we only need rank 0 to sample
    logits = lm_head(hidden_states)
    logits = logits.float()
    
    return outputs, logits
