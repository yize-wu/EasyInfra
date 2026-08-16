import torch
from torch import nn
from ..weight_load_utils import create_param_tensor_on_device
from typing import List, Union
from collections.abc import Iterable

class Embedding(nn.Embedding):
    def weight_loader(self, module_name, value, dtype, **kwargs):
        old_value = getattr(self, module_name)
        new_value = create_param_tensor_on_device(old_value, value, dtype)
        setattr(self, module_name, new_value)
    
    def _embedding(self, input_ids: torch.Tensor, out=None):
        if out is None:
            return super().forward(input_ids)
        else:
            output_shape = (*input_ids.shape, -1)
            return torch.index_select(self.weight, dim=0, index=input_ids.view(-1), out=out).view(output_shape)
    
    def forward(self, input_ids: Union[torch.Tensor, List[torch.Tensor]], out=None):
        if isinstance(input_ids, torch.Tensor):
            return self._embedding(input_ids, out)
        else:
            if not isinstance(input_ids, Iterable):
                raise ValueError
            return [self._embedding(input, out) for input in input_ids]