from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from easyinfra.generation.cache_utils import Cache
import torch
from torch import nn
import torch.nn.functional as F
import math

from easyinfra.generation.utils.version import is_version_greater_or_equal
from ....utils import record_time_sync, show_rank_print

from typing import Optional, List, Tuple

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    b, h, s, d = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(b, h, n_rep, s, d)
    return hidden_states.reshape(b, h * n_rep, s, d)

def repeat_kv_3d(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    h, s, d = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, None, :, :].expand(h, n_rep, s, d)
    return hidden_states.reshape(h * n_rep, s, d)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=0):
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


def compute_attn_output(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                        attention_mask: torch.Tensor, dropout: float, scaling: Optional[float], is_causal: bool):
    '''
        Require qkv layout to be `BHSD`
    '''
    
    ### Note: enable_gqa=True can make it slow, as SDPA will choose the math-attention implementation, 
    # rather than mem-efficient ones like flash attention.
    # So, we manually repeat kv here, rather than `_sdpa_kwargs["enable_gqa"] = True`
    # See https://github.com/pytorch/pytorch/pull/191085
    n_rep = q.shape[-3] // k.shape[-3]
    k = repeat_kv(k, n_rep)
    v = repeat_kv(v, n_rep)
    
    _sdpa_kwargs = {}
    ## Must name causal if no attention mask
    # Explicit attention mask is slower than attention_causal_bias
    # Configure `need_attention_causal_bias` if needed
    if attention_mask is None:
        raise ValueError(f"SDPA must have attention bias.")
        _sdpa_kwargs["is_causal"] = is_causal
        
    attn_output = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attention_mask,
        dropout_p=dropout,
        scale=scaling,
        **_sdpa_kwargs
    )
    return attn_output

def compute_attn_output_3d(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                           attention_mask: torch.Tensor, dropout: float, scaling: Optional[float], is_causal: bool
                           ):
    n_rep = q.shape[-3] // k.shape[-3]
    k = repeat_kv_3d(k, n_rep)
    v = repeat_kv_3d(v, n_rep)
    
    _sdpa_kwargs = {}
    if attention_mask is None:
        ## must name causal if no attention mask
        _sdpa_kwargs["is_causal"] = is_causal
    else:
        show_rank_print(f"Is attention mask really needed?")
        
    attn_output = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attention_mask,
        dropout_p=dropout,
        scale=scaling,
        **_sdpa_kwargs
    )
    return attn_output



def compute_attn_output_naive(
    q: torch.Tensor, 
    k: torch.Tensor, 
    v: torch.Tensor, 
    attention_mask: torch.Tensor,
    dropout: float, 
    scaling: Optional[float] = None,
    output_tensor: Optional[torch.Tensor] = None,
):
    
    n_rep = q.shape[-3] // k.shape[-3]
    k = repeat_kv(k, n_rep)
    v = repeat_kv(v, n_rep)
    
    if scaling is None:
        scaling = 1 / math.sqrt(q.shape[-1])
    attn_weight = (q @ k.transpose(-1,-2)) * scaling
    attn_output = (attn_weight + attention_mask).softmax(-1, dtype=torch.float32).to(q.dtype) @ v
    return attn_output

class SdpaAttention(nn.Module):
    def __init__(
        self,
        num_q_heads: int,
        qk_head_dim: int,
        v_head_dim: int,
        num_kv_heads: int,
        causal: bool,
        scale: float,
        dropout: float,
        sliding_window: Optional[int] = None,
        explicit_attn_kwargs: Optional[int] = None,
    ):
        super().__init__()
        self.method = compute_attn_output
        self.num_q_heads = num_q_heads
        self.qk_head_dim = qk_head_dim
        self.v_head_dim = v_head_dim
        self.num_kv_heads = num_kv_heads
        self.causal = causal
        self.scale = scale
        self.dropout = dropout
        self.sliding_window = sliding_window
        self.explicit_attn_kwargs = explicit_attn_kwargs
        
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        kv_caches: List[Cache],
        
        seqlens_q: List[int],
        cu_seqlens_q: List[int],
        layer_idx: int,
        **sdpa_kwargs,
    ):
        q_splits = query.split(seqlens_q, dim=0)
        k_splits = key.split(seqlens_q, dim=0)
        v_splits = value.split(seqlens_q, dim=0)
        
        if len(seqlens_q) > 1:
            attn_outputs = query.new_empty((query.shape[0], self.num_q_heads * self.v_head_dim))
        for i, (q, k, v) in enumerate(zip(q_splits, k_splits, v_splits)):
            
            k, v = kv_caches[i].update(k, v, layer_idx,)
            
            q = q.transpose(-2, -3).unsqueeze(0)
            k = k.transpose(-2, -3).unsqueeze(0)
            v = v.transpose(-2, -3).unsqueeze(0)
            output: torch.Tensor = self.method(
                q,
                k,
                v,
                attention_mask=attention_mask[i],
                dropout=self.dropout,
                scaling=self.scale,
                is_causal=self.causal,
            )

            ### TODO: how to zero-copy
            output = output.transpose(-2,-3).reshape(seqlens_q[i], -1)
            if len(seqlens_q) > 1:
                attn_outputs[cu_seqlens_q[i]:cu_seqlens_q[i+1],:] = output
            else:
                attn_outputs = output
                
            # time3 = time.time()
            # show_rank_print(f"{time1 - time0}, {time2 - time1}, {time3 - time2}", 0)
        
        return attn_outputs
    
