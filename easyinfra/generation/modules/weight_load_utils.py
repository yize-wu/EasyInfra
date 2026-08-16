import torch
from torch import nn
from typing import Optional, Union, Dict, List

def check_shape(
    old_value: torch.Tensor,
    value: torch.Tensor,
):
    if tuple(old_value.shape) != tuple(value.shape):
        raise ValueError(f"old value shape is {tuple(old_value.shape)}, new shape is {tuple(value.shape)}")

def create_param_tensor_on_device(
    old_value: torch.Tensor,
    value: torch.Tensor,
    dtype: Optional[Union[str, torch.dtype]] = None,
    device: int = None,
    fp16_statistics: Optional[torch.HalfTensor] = None,
    tied_params_map: Optional[Dict[int, Dict[torch.device, torch.Tensor]]] = None,
    **kwargs
):
    check_shape(old_value, value)
    if device is None:
        device = torch.cuda.current_device()
    param_cls = type(old_value)
    # get dtype from old_value if not specified
    if dtype is None:
        value = value.to(old_value.dtype)
    elif not str(value.dtype).startswith(("torch.uint", "torch.int", "torch.bool")):
        ## cast the new value to the dtype
        value = value.to(dtype)

    with torch.no_grad():
        require_grad = old_value.requires_grad
        new_value = value.to(torch.cuda.current_device()) # default device should be set
        new_value = param_cls(new_value, requires_grad=require_grad)
            
    # clean pre and post forward hook
    torch.cuda.empty_cache()
    return new_value


def create_or_fill_sharded_param_tensor_on_device(
    old_value: torch.Tensor,
    value: torch.Tensor,
    dtype: Optional[Union[str, torch.dtype]] = None,
    device: int = None,
    fp16_statistics: Optional[torch.HalfTensor] = None,
    tied_params_map: Optional[Dict[int, Dict[torch.device, torch.Tensor]]] = None,
    **kwargs
):
    if device is None:
        device = torch.cuda.current_device()
    
    # At first, the old_value is not on device. We need to create an empty on-device parameter tensor.
    if old_value.device != device:
        new_value
    param_cls = type(old_value)
    # get dtype from old_value if not specified
    if dtype is None:
        value = value.to(old_value.dtype)
    elif not str(value.dtype).startswith(("torch.uint", "torch.int", "torch.bool")):
        value = value.to(dtype)

    with torch.no_grad():
        require_grad = old_value.requires_grad
        new_value = value.to(torch.cuda.current_device()) # default device should be set
        new_value = param_cls(new_value, requires_grad=require_grad)
            
    # clean pre and post forward hook
    torch.cuda.empty_cache()
    return new_value

class BaseWeightLoader:
    def weight_loader(self, module_name:str, value:torch.Tensor, dtype:torch.dtype, **kwargs):
        old_value: torch.Tensor = getattr(self, module_name)
        new_value = create_param_tensor_on_device(old_value, value, dtype)
        setattr(self, module_name, new_value)