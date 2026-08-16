from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

import torch
from torch import nn

from typing import Optional, Tuple, List, Union
from .rope_utils import ROPE_INIT_FUNCTIONS

from ....generation.parallel.parallel_utils import get_device

class BaseRotaryEmbedding(nn.Module):
    
    def __init__(
        self,
        dim=None,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        rope_type="default",
        config = None,
        qk_layout = 'shd',
    ):
        super().__init__()
        # TODO (joao): remove the `if` below, only used for BC
        self.rope_kwargs = {}
        if config is None:
            # logger.warning_once(
            #     "`LlamaRotaryEmbedding` can now be fully parameterized by passing the model config through the "
            #     "`config` argument. All other arguments will be removed in v4.46"
            # )
            self.rope_kwargs = {
                "rope_type": rope_type,
                "factor": scaling_factor,
                "dim": dim,
                "base": base,
                "max_position_embeddings": max_position_embeddings,
            }
            self.rope_type = rope_type
            self.max_seq_len_cached = max_position_embeddings
            self.original_max_seq_len = max_position_embeddings
        else:
            # BC: "rope_type" was originally "type"
            if config.rope_scaling is not None:
                self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
            else:
                self.rope_type = "default"
            self.max_seq_len_cached = config.max_position_embeddings
            self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, **self.rope_kwargs)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # Move to device
        self.inv_freq = self.inv_freq.to(device=get_device())
        self.original_inv_freq = self.inv_freq
        self.qk_layout = qk_layout # for pos embed
    
    def _pos_embed(self, pos_ids: torch.Tensor, dtype, device_type):
        '''
            Output: cos, sin
        '''
        
        inv_freq_expanded = self.inv_freq[:, None].float()
        position_ids_expanded = pos_ids[None, :].float()

        # device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            # freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            freqs = (inv_freq_expanded @ position_ids_expanded).transpose(0, 1)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
            # print(freqs.shape, cos.shape)

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling
        
        cos = cos.to(dtype=dtype)
        sin = sin.to(dtype=dtype)
        if self.qk_layout == 'shd':
            unsqueeze_dim = -2
        elif self.qk_layout == 'hsd':
            unsqueeze_dim = -3
            
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        
        return cos, sin
    
    def forward(
        self, 
        position_ids: Union[torch.LongTensor, List[torch.LongTensor]], 
        dtype: torch.dtype = None,
        device_type: torch.DeviceObjType = None,
        position_embeddings_tensors: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        """
            Assume position_ids are the same for all batches.
        """
        if isinstance(position_ids, torch.Tensor):
            cos, sin = self._pos_embed(position_ids, dtype=dtype, device_type=device_type)
            if position_embeddings_tensors is not None:
                position_embeddings_tensors[0].copy_(cos)
                position_embeddings_tensors[1].copy_(sin)
                cos, sin = position_embeddings_tensors
            return cos, sin
        elif isinstance(position_ids, list):
            cos_list, sin_list = [], []
            for pos_id in position_ids:
                cos, sin = self._pos_embed(pos_id, dtype=dtype, device_type=device_type)
                cos_list.append(cos)
                sin_list.append(sin)
            return cos_list, sin_list
        else:
            raise ValueError(f"Unsupported input for rotary embed")                

    
    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(
                self.config, device, seq_len=seq_len, **self.rope_kwargs
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len
