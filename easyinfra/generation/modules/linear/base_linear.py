import torch
from torch import nn
from torch.nn.parameter import Parameter
from typing import List, Optional, Tuple, Union, Dict
import torch.nn.functional as F

import math
import time

from ..tp_module import TPModule
from ..weight_load_utils import create_param_tensor_on_device

class BaseLinear(nn.Module):
    """
        This class is merely a copy of nn.Linear for now.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        factory_kwargs = {'device': None, 'dtype': None}
        self.weight = Parameter(torch.empty(out_features, in_features, **factory_kwargs))
        if bias == True:
            self.bias = Parameter(torch.empty((self.out_features), **factory_kwargs))
        else:
            self.register_parameter('bias', None)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
            hidden_states: [bsz, q_len, in_features]
            out: [bsz, q_len, out_features]
        """
        return F.linear(hidden_states, self.weight, self.bias)  
    
    def weight_loader(self, module_name:str, value:torch.Tensor, dtype:torch.dtype, **kwargs):
        old_value: torch.Tensor = getattr(self, module_name)
        # if module_name == 'bias':
        #     print(1)
        # load weight to device
        new_value = create_param_tensor_on_device(old_value, value, dtype)
        setattr(self, module_name, new_value)