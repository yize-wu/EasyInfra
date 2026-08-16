import torch
from torch import nn
from torch.nn.parameter import Parameter
from typing import List, Optional, Tuple, Union, Dict
import torch.nn.functional as F

from ..tp_module import TPModule
from ..weight_load_utils import create_param_tensor_on_device
from ...parallel.parallel_state import GroupCommunicator
from easyinfra.generation.functions import exclusive_cumsum

class ColumnParallelLinear(TPModule):
    """
        (n*m, l) -> n*(m, l).
    """
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 bias: bool = False, 
                 tp_group: Optional[GroupCommunicator] = None,
                 default_gather: bool = False):
        
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        factory_kwargs = {'device': None, 'dtype': None}
        
        self.tp_group = tp_group
        self.tp_size = self.tp_group.group_size if tp_group is not None else 1
        self.local_rank = self.tp_group.local_rank if tp_group is not None else 0
        self.default_gather = default_gather
        
        self.tp_out_features = out_features // self.tp_size
        if self.tp_out_features * self.tp_size != out_features:
            raise ValueError
        
        self.output_dim = 0
        self.weight = Parameter(torch.empty(self.tp_out_features, in_features, **factory_kwargs))
        if bias == True:
            self.bias = Parameter(torch.empty((self.tp_out_features), **factory_kwargs))
        else:
            self.register_parameter('bias', None)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        force_gather: bool = False
    ) -> torch.Tensor:
        # bias is of tp_out_features size, so just forward is ok
        hidden_states = F.linear(hidden_states, self.weight, self.bias)  
        if force_gather or self.default_gather:
            raise NotImplementedError
            # output_shape = list(hidden_states.shape)
            # output_shape[-1] *= self.tp_size
            # new_hidden_states = torch.empty(output_shape, dtype=hidden_states.dtype, device=hidden_states.device)
        return hidden_states
    
    def weight_loader(self, module_name:str, value:torch.Tensor, dtype:torch.dtype, **kwargs):
        """
            Load weight tensor to self.weight
        """
        old_value: torch.Tensor = getattr(self, module_name)
        # slice the value tensor
        slice_start = self.local_rank * self.tp_out_features
        slice_end = slice_start + self.tp_out_features
        value = value[slice_start:slice_end, ...]
        # create parameter (rather than assign the value to the old torch.Parameter)
        new_value = create_param_tensor_on_device(old_value, value, dtype)
        setattr(self, module_name, new_value)

class MergedColumnParallelLinear(ColumnParallelLinear):
    def __init__(self, in_features: int, out_feature_sizes: List[int], param_names: List[str],
                 bias = False, tp_group = None, default_gather = False):
        '''
            out_feature_sizes: []
        '''
        assert bias == False or bias == None
        out_features = sum(out_feature_sizes)
        super().__init__(in_features, out_features, bias, tp_group, default_gather)
        
        self.tp_out_feature_sizes = [_sizes // self.tp_size for _sizes in out_feature_sizes]
        self.tp_out_feature_offsets = [0]
        for i,_tp_out_feature_size in enumerate(self.tp_out_feature_sizes):
            if _tp_out_feature_size * self.tp_size != out_feature_sizes[i]:
                raise ValueError(f"The {i}-th out feature size is {_tp_out_feature_size}, not equal to {out_feature_sizes[i]}//{self.tp_group.group_size}")
            if i > 0:
                self.tp_out_feature_offsets.append(self.tp_out_feature_offsets[-1] + self.tp_out_feature_sizes[i-1])
        self.tp_out_feature_offsets = exclusive_cumsum(torch.tensor(self.tp_out_feature_sizes, device="cpu")).tolist()
        self.split_sizes = self.tp_out_feature_sizes
        self.param_names = param_names[:]
        self.num_loaded_values = 0
        self.tmp_tensor: torch.Tensor = None
        
    def weight_loader(self, module_name:str, value:torch.Tensor, dtype:torch.dtype, param_name: str, **kwargs):
        param_name_index = self.param_names.index(param_name)
        
        slice_start = self.local_rank * self.tp_out_feature_sizes[param_name_index]
        slice_end = slice_start + self.tp_out_feature_sizes[param_name_index]
        value = value[slice_start:slice_end, ...]
        
        # torch.cat on device is faster than on host, so move to device first
        if self.num_loaded_values == 0:
            self.tmp_tensor = torch.empty((self.tp_out_features, self.in_features), dtype=dtype, device=torch.cuda.current_device())
        self.tmp_tensor.narrow(self.output_dim, self.tp_out_feature_offsets[param_name_index], self.tp_out_feature_sizes[param_name_index]).copy_(value)
        self.num_loaded_values += 1
        if self.num_loaded_values == len(self.param_names):
            old_value: torch.Tensor = getattr(self, module_name)
            new_value = create_param_tensor_on_device(old_value, self.tmp_tensor, dtype)
            
            del self.tmp_tensor # release memory
            delattr(self, module_name)
            
            setattr(self, module_name, new_value)
        
        return
        
   
class RowParallelLinear(TPModule):
    """
        (m, n*l) -> n*(m, l).
    """
    _support_safetensor_slice = True
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 bias: bool = False, 
                 tp_group: Optional[GroupCommunicator] = None,
                 default_reduce: bool = True):
        
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        factory_kwargs = {'device': None, 'dtype': None}
        
        self.tp_group = tp_group
        self.tp_size = self.tp_group.group_size if tp_group is not None else 1
        self.local_rank = self.tp_group.local_rank if tp_group is not None else 0
        self.default_reduce = default_reduce
        
        self.tp_in_features = in_features // self.tp_size
        if self.tp_in_features * self.tp_size != in_features:
            raise ValueError
        
        self.weight = Parameter(torch.empty(self.out_features, self.tp_in_features, **factory_kwargs))
        if bias == True:
            self.bias = Parameter(torch.empty((self.out_features), **factory_kwargs))
        else:
            self.register_parameter('bias', None)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        force_reduce: bool = False,
    ) -> torch.Tensor:
        # matrix multiplication is tp, and bias is not
        hidden_states = F.linear(hidden_states, self.weight)  
        if force_reduce:
            self.tp_group.all_reduce(hidden_states)
            if self.bias is not None:
                hidden_states += self.bias
        # if not reduced, the bias will not be added
        return hidden_states
    
    def weight_loader(self, module_name:str, value, dtype:torch.dtype, **kwargs):
        """
            value should be a safetensor slice.
        """
        old_value: torch.Tensor = getattr(self, module_name)
        if module_name == 'bias':
            raise NotImplementedError
        # load weight to device
        
        slice_start = self.local_rank * self.tp_in_features
        slice_end = slice_start + self.tp_in_features
        # load tensor on
        value = value[:, slice_start:slice_end]
        new_value = create_param_tensor_on_device(old_value, value, dtype)
        setattr(self, module_name, new_value)