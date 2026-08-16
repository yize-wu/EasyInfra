import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from typing import Optional

from ...parallel.parallel_state import GroupCommunicator
from ..tp_module import TPModule
from ..weight_load_utils import create_param_tensor_on_device


class TPLogitsHead(TPModule):
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 bias: bool = False, 
                 tp_group: Optional[GroupCommunicator] = None,
                 reduce: bool = True):
        """
            default_reduce manages the forward way.
            if True, the logit head will do a tp linear forward and reduce, otherwise it do a normal forward at multiple gpus.
        """
        
        if tp_group is None:
            raise ValueError
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        factory_kwargs = {'device': None, 'dtype': None}
        
        self.reduce = reduce
        if reduce:
            self.tp_group = tp_group
            self.tp_size = self.tp_group.group_size
            self.tp_in_features = in_features // self.tp_size
            self.slice_start = self.tp_group.local_rank * self.tp_in_features
            self.slice_end = self.slice_start + self.tp_in_features
            
            if self.tp_in_features * self.tp_size != in_features:
                raise ValueError
            in_features = self.tp_in_features
            
        self.weight = Parameter(torch.empty(out_features, in_features, **factory_kwargs))
        if bias == True:
            self.bias = Parameter(torch.empty((out_features), **factory_kwargs))
        else:
            self.register_parameter('bias', None)
            
    
    def forward(
        self,
        hidden_states: torch.Tensor,
    ):
        if self.reduce:
            hidden_states = hidden_states.narrow(dim=-1, start=self.slice_start, length=self.tp_in_features)
            hidden_states = F.linear(hidden_states, self.weight, None)
            self.tp_group.all_reduce(hidden_states)
            if self.bias is not None:
                hidden_states += self.bias # no parallel for bias
        else:
            hidden_states = F.linear(hidden_states, self.weight, self.bias)
        return hidden_states
    
    def weight_loader(self, module_name:str, value:torch.Tensor, dtype:torch.dtype, **kwargs):
        """
            value should be a safetensor slice.
            if reduce=True, load a tp slice of weight; otherwise, load a full matrix weight.
        """
        old_value: torch.Tensor = getattr(self, module_name)
        is_bias = module_name == 'bias'
        # load weight to device
        if not self.reduce or is_bias:
            # load the full matrix weight
            value = value[:]
        else:
            # load a slice
            value = value[:, self.slice_start:self.slice_end]
            
        new_value = create_param_tensor_on_device(old_value, value, dtype)
        setattr(self, module_name, new_value)