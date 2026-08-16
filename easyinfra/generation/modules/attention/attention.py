import time
import torch
from torch import nn
from typing import Tuple, List, Optional, Dict

from .sdpa import SdpaAttention
from .flash_attention2 import FlashAttention2

from easyinfra.generation.cache_utils import Cache
from easyinfra.utils import show_rank_print
from easyinfra import envs
# from EasyInfra.utils.tensors import _to_cpu_list

def get_attn_backend(attn_backend_label: str):
    if attn_backend_label == "sdpa":
        return SdpaAttention
    elif attn_backend_label == "flash-attn":
        return FlashAttention2
    else:
        raise ValueError(f"unsupported attention implementation label {attn_backend_label}")

class Attention(nn.Module):
    def __init__(
        self,
        num_q_heads: int,
        qk_head_dim: int,
        v_head_dim: Optional[int] = None,
        num_kv_heads: Optional[int] = None,
        causal: bool = True,
        scale: Optional[float] = None,
        dropout: float = 0.0,
        sliding_window: Optional[int] = None,
        attn_backend: Optional[object] = None,
        explicit_attn_kwargs: Optional[int] = None,
        **kwargs
    ):
        super().__init__()
        if v_head_dim is None:
            v_head_dim = qk_head_dim # default kv head_dim same
        if num_kv_heads is None:
            num_kv_heads = num_q_heads
        if attn_backend is None:
            attn_backend = get_attn_backend(envs.ATTENTION_BACKEND)
        
        self.attn_backend = attn_backend(
            num_q_heads=num_q_heads,
            qk_head_dim=qk_head_dim,
            v_head_dim=v_head_dim,
            num_kv_heads=num_kv_heads,
            causal=causal,
            scale=scale,
            dropout=dropout,
            sliding_window=sliding_window,
            explicit_attn_kwargs=explicit_attn_kwargs,
        )    
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_masks: Tuple[torch.Tensor],
        kv_caches: Tuple[Cache],
        seqlens_q: List[int],
        cu_seqlens_q: List[int],
        cu_seqlens_k: List[int],
        layer_idx: int,
        **kwargs
    ):
        '''
            Return flattened attn outputs.
            Input:
                q,k,v: [seqlen, num_head, head_dim]
        '''
        return self.attn_backend(
            query,
            key,
            value,
            attention_masks,
            kv_caches,
            seqlens_q=seqlens_q,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            layer_idx=layer_idx,
        )
        
