import torch
from torch import nn
from typing import Union, List, Iterable
from ..weight_load_utils import create_param_tensor_on_device

def _rmsnorm(t : torch.Tensor, weight: torch.Tensor, variance_epsilon: float):
    input_dtype = t.dtype
    t = t.to(torch.float32)
    variance = t.pow(2).mean(-1, keepdim=True)
    t = t * torch.rsqrt(variance + variance_epsilon)
    t = weight * t.to(input_dtype)
    return t

class BaseRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
            BaseRMSNorm does not use tensor parallel. It can be run on multiple devices if tp is on.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        
    def forward(self, hidden_states : Union[torch.Tensor, List[torch.Tensor]], ):
        if isinstance(hidden_states, torch.Tensor):
            return _rmsnorm(hidden_states, self.weight, self.variance_epsilon)
        else:
            if not isinstance(hidden_states, Iterable):
                raise ValueError
            return [_rmsnorm(t, self.weight, self.variance_epsilon) for t in hidden_states]

    def weight_loader(self, module_name:str, value:torch.Tensor, dtype, **kwargs):
        old_value = getattr(self, module_name)
        new_value = create_param_tensor_on_device(old_value, value, dtype)
        setattr(self, module_name, new_value)

class FusedToLinearRMSNorm(nn.Module):
    """
        Fusing weight to the linears after this norm may reduce computation.
    """
    def __init__(self, hidden_size, eps=1e-6):
        """
            No tensor parallel. It can be run on multiple devices if tp is on.
        """
        super().__init__()
        self.variance_epsilon = eps
        
    def forward(self, hidden_states : torch.Tensor, ):
        # start = record_time_sync()
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = hidden_states.to(input_dtype)
        # print(record_time_sync() - start)
        return hidden_states

    def weight_loader(self, *args, **kwargs):
        # it should never be called
        raise ValueError("Use fused rmsnorm only after weight loading.")
