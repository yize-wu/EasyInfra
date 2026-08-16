from transformers.modeling_utils import (
    nn, ModuleUtilsMixin, PushToHubMixin, PeftAdapterMixin, is_fsdp_enabled,
    is_peft_available, is_deepspeed_zero3_enabled,
    is_accelerate_available, is_offline_mode,
    _add_variant,
    ADAPTER_SAFE_WEIGHTS_NAME,
    ADAPTER_WEIGHTS_NAME,
    CONFIG_NAME,
    DUMMY_INPUTS,
    FLAX_WEIGHTS_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    TF2_WEIGHTS_NAME,
    TF_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    ContextManagers,
    PushToHubMixin,
    cached_file,
    download_url,
    extract_commit_hash,
    has_file,
    is_accelerate_available,
    is_flash_attn_2_available,
    is_offline_mode,
    is_optimum_available,
    is_peft_available,
    is_remote_url,
    is_torch_xla_available,
    logging,
    get_checkpoint_shard_files,
    no_init_weights,
    init_empty_weights,
    get_balanced_memory,
    get_max_memory,
    find_tied_parameters,
    load_state_dict,
    dispatch_model,
    check_tied_parameters_on_same_device
)

from ..utils.import_utils import (
    is_safetensors_available,
    is_torch_sdpa_available,
)

from accelerate.hooks import (
    add_hook_to_module,
    attach_align_device_hook_on_blocks,
)
from accelerate.utils import (
    check_tied_parameters_on_same_device,
    extract_model_from_parallel,
    find_tied_parameters,
    get_balanced_memory,
    get_max_memory,
    load_offloaded_weights,
    offload_weight,
    save_offload_index,
)
from accelerate.utils.modeling import (
    compute_module_sizes,
    check_tied_parameters_in_config,
    get_max_layer_size,
    compute_module_total_buffer_size,
    clean_device_map,
    retie_parameters,
)

from transformers.configuration_utils import PretrainedConfig
from transformers.generation import GenerationConfig

from typing import List, Optional, Tuple, Union, Dict, Callable, OrderedDict


import torch
from torch.utils.checkpoint import checkpoint

import collections
import copy
import functools
import gc
import importlib.metadata
import inspect
import itertools
import json
import os
import re
import shutil
import tempfile
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial, wraps
from threading import Thread
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from zipfile import is_zipfile

import torch
from huggingface_hub import split_torch_state_dict_into_shards
from packaging import version
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss, Identity
from torch.utils.checkpoint import checkpoint

import torch.distributed as dist
import torch.multiprocessing as mp

from ..device_map import (
    infer_auto_device_map,
    infer_multi_gpu_device_map,
)

_init_weights = True
logger = logging.get_logger(__name__)

from safetensors import safe_open
def load_state_dict_keys(
    checkpoint_file: str,
):
    with safe_open(checkpoint_file, framework="pt") as f:
        metadata = f.metadata()
        state_dict_keys = f.keys()
    if metadata is None:
        pass
        # warnings.warn(f"No metadata in file {checkpoint_file}")
    elif metadata.get("format") not in ["pt"]:
        raise OSError("support pt safetensor only.")
    return state_dict_keys

def load_state_dict_all_slices(checkpoint_file: str,):
    result = {}
    with safe_open(checkpoint_file, framework="pt") as f:
        metadata = f.metadata()
        for key in f.keys():
            result[key] = f.get_slice(key)
    if metadata.get("format") not in ["pt"]:
        raise OSError("support pt safetensor only.")
    return result

def load_state_dict_slice(checkpoint_file: str, param_name: str):
    # get the slice, leave real slicing work to weight loader
    with safe_open(checkpoint_file, framework='pt') as f:
        slice = f.get_slice(param_name)
    return slice
def load_state_dict_tensor(checkpoint_file: str, param_name: str):
    # get the real value
    with safe_open(checkpoint_file, framework='pt') as f:
        t = f.get_tensor(param_name)
    return t

# def _load_state_dict_into_meta_model(
#     model,
#     state_dict,
#     loaded_state_dict_keys,
#     start_prefix,
#     expected_keys,
#     device_map=None,
#     offload_folder=None,
#     offload_index=None,
#     state_dict_folder=None,
#     state_dict_index=None,
#     dtype=None,
#     hf_quantizer=None,
#     is_safetensors=False,
#     keep_in_fp32_modules=None,
#     unexpected_keys=None,  # passing `unexpected` for cleanup from quantization items
#     pretrained_model_name_or_path=None,  # for flagging the user when the model contains renamed keys
#     is_batch_accelerate=False,
#     is_multi_gpu_accelerate=False,
#     bound_strategy=None,
# ):

#     is_quantized = hf_quantizer is not None
#     warning_msg = f"This model {type(model)}"

#     is_torch_e4m3fn_available = hasattr(torch, "float8_e4m3fn")

#     for param_name, param in state_dict.items():

#         if param_name.startswith(start_prefix):
#             param_name = param_name[len(start_prefix) :]
            
#         set_module_kwargs = {}
        
#         # now consider layers
#         sub_strategies = bound_strategy
        
#         set_module_kwargs["is_batch_accelerate"] = is_batch_accelerate
#         set_module_kwargs["is_multi_gpu_accelerate"] = is_multi_gpu_accelerate
        
#         # param name is the file state dict name
#         # module name is the name of model module where the param belongs
#         if is_batch_accelerate:
#             if sub_strategies is None or 'layers' not in param_name: # not a layer param
#                 module_name = param_name
#             else:
#                 layer_idx = int(param_name.split('.')[2])
                
#                 hyper_idx = [i for i,l in enumerate(sub_strategies) if layer_idx in l]
#                 if len(hyper_idx) > 1:
#                     raise ValueError
#                 hyper_idx = hyper_idx[0]
#                 in_layer_offset = sub_strategies[hyper_idx].index(layer_idx)
                
#                 module_name = param_name.split('.')
#                 module_name[2] = str(hyper_idx)
#                 if module_name[3] in ['self_attn', 'mlp']:
#                     module_name.pop(3) # 'self_attn' or 'mlp'
#                 module_name = '.'.join(module_name)
                
#                 set_module_kwargs["in_layer_offset"] = in_layer_offset
                
#         elif is_multi_gpu_accelerate:
            
#             if 'layers' not in param_name:
#                 module_name = param_name
#             else:    
#                 module_name = param_name.split('.')
#                 if module_name[3] in ['self_attn', 'mlp']:
#                     module_name.pop(3) # 'self_attn' or 'mlp'
#                 module_name = '.'.join(module_name)
        
#         else:
#             module_name = param_name
        
#         set_module_kwargs["value"] = param
#         set_module_kwargs["dtype"] = dtype

#         # For compatibility with PyTorch load_state_dict which converts state dict dtype to existing dtype in model, and which
#         # uses `param.copy_(input_param)` that preserves the contiguity of the parameter in the model.
#         # Reference: https://github.com/pytorch/pytorch/blob/db79ceb110f6646523019a59bbd7b838f43d4a86/torch/nn/modules/module.py#L2040C29-L2040C29
#         # old_param = model
#         # splits = param_name.split(".")
#         # for split in splits:
#         #     old_param = getattr(old_param, split)
#         #     if old_param is None:
#         #         break
#         # if old_param is not None:
#         #     if dtype is None:
#         #         param = param.to(old_param.dtype)

#         #     if old_param.is_contiguous():
#         #         param = param.contiguous()
#         param = param.contiguous()

#         # find next higher level module that is defined in device_map:
#         # bert.lm_head.weight -> bert.lm_head -> bert -> ''
#         device_module_name = module_name[:]
#         while len(device_module_name) > 0 and device_module_name not in device_map:
#             device_module_name = ".".join(device_module_name.split(".")[:-1])
#         if device_module_name == "" and "" not in device_map:
#             # TODO: group all errors and raise at the end.
#             raise ValueError(f"{module_name} doesn't have any device set.")
        
#         param_device = device_map[device_module_name]

#         # find leave of module and name
#         module = model
#         if "." in module_name:
#             splits = module_name.split(".")
#             for split in splits[:-1]:
#                 new_module = getattr(module, split)
#                 if new_module is None:
#                     raise ValueError(f"{module} has no attribute {split}.")
#                 module = new_module
#             module_name = splits[-1]
#         is_batch_param = ("in_layer_offset" in set_module_kwargs)
#         is_slice_param = False
#         old_value = getattr(module, module_name, None)
#         if old_value is None and is_multi_gpu_accelerate:
#             module_name = module_name+"_slices"
#             old_value = getattr(module, module_name, None)
#             is_slice_param = True
#         # if old_value is None and module_name not in module._buffers:
#         #         raise ValueError(f"{module} does not have a parameter or a buffer named {module_name}.")
#         # if old_value.device == torch.device("meta") and device not in ["meta", torch.device("meta")] and value is None:
#         #     raise ValueError(f"{module_name} is on the meta device, we need a `value` to put in on {device}.")
               
                
#         # For backward compatibility with older versions of `accelerate` and for non-quantized params
#         set_module_kwargs["is_slice_param"] = is_slice_param
#         set_module_kwargs["is_batch_param"] = is_batch_param
#         if is_slice_param:
#             set_module_kwargs["slices_device_list"] = [0,0,0,0]
#         set_module_tensor_to_device(module, module_name, param_device, **set_module_kwargs)

#     return [], offload_index, state_dict_index

def set_module_tensor_to_device(
    module: nn.Module,
    module_name: str,
    device: Union[int, str, torch.device],
    value: Optional[torch.Tensor] = None,
    dtype: Optional[Union[str, torch.dtype]] = None,
    fp16_statistics: Optional[torch.HalfTensor] = None,
    tied_params_map: Optional[Dict[int, Dict[torch.device, torch.Tensor]]] = None,
    **kwargs
):
    
    is_batch_accelerate = kwargs.pop("is_batch_accelerate", False)
    is_multi_gpu_accelerate = kwargs.pop("is_multi_gpu_accelerate", False)
    is_batch_param = kwargs.pop("is_batch_param", False)
    is_slice_param = kwargs.pop("is_slice_param", False)
    
    # print(module._get_name(),module_name)
    # is_buffer = module_name in module._buffers

    old_value = getattr(module, module_name)
    if is_slice_param:
        assert type(old_value) == nn.ParameterList
        old_value = old_value[0]
    param_cls = type(old_value)

    if value is not None:
        # We can expect mismatches when using bnb 4bit since Params4bit will reshape and pack the weights.
        # In other cases, we want to make sure we're not loading checkpoints that do not match the config.
        # if old_value.shape != value.shape and param_cls.__name__ != "Params4bit" and in_layer_offset is None:
        #     raise ValueError(
        #         f'Trying to set a tensor of shape {value.shape} in "{module_name}" (which has shape {old_value.shape}), this looks incorrect.'
        #     )

        if dtype is None:
            # For compatibility with PyTorch load_state_dict which converts state dict dtype to existing dtype in model
            value = value.to(old_value.dtype)
        elif not str(value.dtype).startswith(("torch.uint", "torch.int", "torch.bool")):
            value = value.to(dtype)

    # device_quantization = None
    with torch.no_grad():
        
        require_grad = old_value.requires_grad
        # require_grad = False
        if is_batch_accelerate:
            in_layer_offset = kwargs.pop("in_layer_offset", None)
            if in_layer_offset is None:
                new_value = value.to(device)
                new_value = param_cls(new_value, requires_grad=require_grad).to(device)
                module._parameters[module_name] = new_value
            else:
                # Can we make sure the offset dim is always 0?
                if module_name != 'weight':
                    raise ValueError
                if module.bound_dim != 0:
                    raise ValueError
                if module._parameters[module_name].dtype != value.dtype:
                    raise ValueError
                if in_layer_offset == 0:
                    module._parameters[module_name] = param_cls(torch.empty(module._parameters[module_name].shape, requires_grad=require_grad, dtype=dtype, device=device))
                
                module._parameters[module_name][in_layer_offset,...] = value.to(device)
        elif is_multi_gpu_accelerate and is_slice_param:
            
            tp_size = module.tp_size
            param_slices = getattr(module, module_name)
            if len(param_slices) != tp_size:
                raise ValueError(f"Slices number {len(param_slices)} must be equal to tensor parallel size {tp_size}.")
            
            # get device list
            device_list = kwargs.pop("slices_device_list")
            # determine the split shape
            split_dim = getattr(module, "split_dims")[0]
            each_split_shape = param_slices[0].shape
            if split_dim != -1 and split_dim != -2:
                raise ValueError
            split_step = each_split_shape[split_dim]
            
            # load each slice
            for tp_rank in range(tp_size):
                this_slice_device = device_list[tp_rank]
                start, end = split_step*tp_rank, split_step*(tp_rank+1)
                if split_dim == -1:
                    new_value = value[..., start:end].to(this_slice_device)
                elif split_dim == -2:
                    new_value = value[..., start:end, :].to(this_slice_device)
                new_value = param_cls(new_value, requires_grad=require_grad).to(this_slice_device)
                if param_slices[tp_rank].shape != new_value.shape:
                    raise ValueError
                param_slices[tp_rank] = new_value
        else:
            # normal
            new_value = value.to(device)
            new_value = param_cls(new_value, requires_grad=require_grad).to(device)
            module._parameters[module_name] = new_value
            
    # clean pre and post foward hook
    torch.cuda.empty_cache()

def set_module_tensor_to_device_multi_gpu_acc(
    module: nn.Module,
    module_name: str,
    device: Union[int, str, torch.device],
    value: Optional[torch.Tensor] = None,
    dtype: Optional[Union[str, torch.dtype]] = None,
    fp16_statistics: Optional[torch.HalfTensor] = None,
    tied_params_map: Optional[Dict[int, Dict[torch.device, torch.Tensor]]] = None,
    **kwargs
):
    raise NotImplementedError
    # print(module._get_name(),module_name)
    # Recurse if needed
    if "." in module_name:
        splits = module_name.split(".")
        for split in splits[:-1]:
            new_module = getattr(module, split)
            if new_module is None:
                raise ValueError(f"{module} has no attribute {split}.")
            module = new_module
        module_name = splits[-1]

    
    # is_buffer = module_name in module._buffers
    old_value = getattr(module, module_name, None)
    if old_value is None:
        old_value = getattr(module, module_name+"_slices", None)
        
    if old_value is None and module_name not in module._buffers:
            raise ValueError(f"{module} does not have a parameter or a buffer named {module_name}.")
        
    if old_value.device == torch.device("meta") and device not in ["meta", torch.device("meta")] and value is None:
        raise ValueError(f"{module_name} is on the meta device, we need a `value` to put in on {device}.")

    param_cls = type(old_value[0])

    if value is not None:
        # We can expect mismatches when using bnb 4bit since Params4bit will reshape and pack the weights.
        # In other cases, we want to make sure we're not loading checkpoints that do not match the config.
        if old_value.shape != value.shape and param_cls.__name__ != "Params4bit" and in_layer_offset is None:
            raise ValueError(
                f'Trying to set a tensor of shape {value.shape} in "{module_name}" (which has shape {old_value.shape}), this looks incorrect.'
            )

        if dtype is None:
            # For compatibility with PyTorch load_state_dict which converts state dict dtype to existing dtype in model
            value = value.to(old_value.dtype)
        elif not str(value.dtype).startswith(("torch.uint", "torch.int", "torch.bool")):
            value = value.to(dtype)

    # device_quantization = None
    with torch.no_grad():
        
        require_grad = old_value.requires_grad
        # require_grad = False
        if in_layer_offset is None:
            new_value = value.to(device)
            new_value = param_cls(new_value, requires_grad=require_grad).to(device)
            module._parameters[module_name] = new_value
        else:
            # Can we make sure the offset dim is always 0?
            if module_name != 'weight':
                raise ValueError
            if module.bound_dim != 0:
                raise ValueError
            if module._parameters[module_name].dtype != value.dtype:
                raise ValueError
            if in_layer_offset == 0:
                module._parameters[module_name] = param_cls(torch.empty(module._parameters[module_name].shape, requires_grad=require_grad, dtype=dtype, device=device))
            
            module._parameters[module_name][in_layer_offset,...] = value.to(device)
            
    # clean pre and post foward hook
    torch.cuda.empty_cache()

    
def set_module_tensor_to_device_batch_acc(
    module: nn.Module,
    module_name: str,
    device: Union[int, str, torch.device],
    value: Optional[torch.Tensor] = None,
    dtype: Optional[Union[str, torch.dtype]] = None,
    fp16_statistics: Optional[torch.HalfTensor] = None,
    tied_params_map: Optional[Dict[int, Dict[torch.device, torch.Tensor]]] = None,
    **kwargs
):
    # print(module._get_name(),module_name)
    # Recurse if needed
    if "." in module_name:
        splits = module_name.split(".")
        for split in splits[:-1]:
            new_module = getattr(module, split)
            if new_module is None:
                raise ValueError(f"{module} has no attribute {split}.")
            module = new_module
        module_name = splits[-1]

    if getattr(module, module_name, None) is None:
        if module_name not in module._buffers:
            raise ValueError(f"{module} does not have a parameter or a buffer named {module_name}.")
    
    # is_buffer = module_name in module._buffers
    old_value = getattr(module, module_name)

    # Treat the case where old_value (or a custom `value`, typically offloaded to RAM/disk) belongs to a tied group, and one of the weight
    # in the tied group has already been dispatched to the device, by avoiding reallocating memory on the device and just copying the pointer.
    if (
        value is not None
        and tied_params_map is not None
        and value.data_ptr() in tied_params_map
        and device in tied_params_map[value.data_ptr()]
    ):
        raise ValueError
        module._parameters[module_name] = tied_params_map[value.data_ptr()][device]
        return
    elif (
        tied_params_map is not None
        and old_value.data_ptr() in tied_params_map
        and device in tied_params_map[old_value.data_ptr()]
    ):
        raise ValueError
        module._parameters[module_name] = tied_params_map[old_value.data_ptr()][device]
        return

    if old_value.device == torch.device("meta") and device not in ["meta", torch.device("meta")] and value is None:
        raise ValueError(f"{module_name} is on the meta device, we need a `value` to put in on {device}.")

    # param = module._parameters[module_name] if module_name in module._parameters else None
    # param_cls = type(param)
    param_cls = type(module._parameters[module_name])
    in_layer_offset = kwargs.pop('in_layer_offset', None)

    if value is not None:
        # We can expect mismatches when using bnb 4bit since Params4bit will reshape and pack the weights.
        # In other cases, we want to make sure we're not loading checkpoints that do not match the config.
        if old_value.shape != value.shape and param_cls.__name__ != "Params4bit" and in_layer_offset is None:
            raise ValueError(
                f'Trying to set a tensor of shape {value.shape} in "{module_name}" (which has shape {old_value.shape}), this looks incorrect.'
            )

        if dtype is None:
            # For compatibility with PyTorch load_state_dict which converts state dict dtype to existing dtype in model
            value = value.to(old_value.dtype)
        elif not str(value.dtype).startswith(("torch.uint", "torch.int", "torch.bool")):
            value = value.to(dtype)

    # device_quantization = None
    with torch.no_grad():
        # leave it on cpu first before moving them to cuda
        # # fix the case where the device is meta, we don't want to put it on cpu because there is no data =0
        # if (
        #     param is not None
        #     and param.device.type != "cuda"
        #     and torch.device(device).type == "cuda"
        #     and param_cls.__name__ in ["Int8Params", "FP4Params", "Params4bit"]
        # ):
        #     device_quantization = device
        #     device = "cpu"
        # `torch.Tensor.to(<int num>)` is not supported by `torch_npu` (see this [issue](https://github.com/Ascend/pytorch/issues/16)).
        # if isinstance(device, int):
        #     if is_npu_available():
        #         device = f"npu:{device}"
        #     elif is_mlu_available():
        #         device = f"mlu:{device}"
        #     elif is_musa_available():
        #         device = f"musa:{device}"
        #     elif is_xpu_available():
        #         device = f"xpu:{device}"
        # if "xpu" in str(device) and not is_xpu_available():
        #     raise ValueError(f'{device} is not available, you should use device="cpu" instead')
        # if value is None:
        #     new_value = old_value.to(device)
        #     if dtype is not None and device in ["meta", torch.device("meta")]:
        #         if not str(old_value.dtype).startswith(("torch.uint", "torch.int", "torch.bool")):
        #             new_value = new_value.to(dtype)

        #         if not is_buffer:
        #             module._parameters[tensor_name] = param_cls(new_value, requires_grad=old_value.requires_grad)
        # elif isinstance(value, torch.Tensor):
        #     new_value = value.to(device)
        # else:
        #     new_value = torch.tensor(value, device=device)
        # if device_quantization is not None:
        #     device = device_quantization
        # if is_buffer:
        #     module._buffers[tensor_name] = new_value
        # elif value is not None or not check_device_same(torch.device(device), module._parameters[tensor_name].device):
        
        require_grad = old_value.requires_grad
        # require_grad = False
        if in_layer_offset is None:
            new_value = value.to(device)
            new_value = param_cls(new_value, requires_grad=require_grad).to(device)
            module._parameters[module_name] = new_value
        else:
            # Can we make sure the offset dim is always 0?
            if module_name != 'weight':
                raise ValueError
            if module.bound_dim != 0:
                raise ValueError
            if module._parameters[module_name].dtype != value.dtype:
                raise ValueError
            if in_layer_offset == 0:
                module._parameters[module_name] = param_cls(torch.empty(module._parameters[module_name].shape, requires_grad=require_grad, dtype=dtype, device=device))
            
            module._parameters[module_name][in_layer_offset,...] = value.to(device)
            
    # clean pre and post foward hook
    torch.cuda.empty_cache()

    # When handling tied weights, we update tied_params_map to keep track of the tied weights that have already been allocated on the device in
    # order to avoid duplicating memory, see above.
    # if (
    #     tied_params_map is not None
    #     and old_value.data_ptr() in tied_params_map
    #     and device not in tied_params_map[old_value.data_ptr()]
    # ):
    #     tied_params_map[old_value.data_ptr()][device] = new_value
    # elif (
    #     value is not None
    #     and tied_params_map is not None
    #     and value.data_ptr() in tied_params_map
    #     and device not in tied_params_map[value.data_ptr()]
    # ):
    #     tied_params_map[value.data_ptr()][device] = new_value

def check_device_map(model: nn.Module, device_map: Dict[str, Union[int, str, torch.device]]):
    """
    Checks a device map covers everything in a given model.

    Args:
        model (`torch.nn.Module`): The model to check the device map against.
        device_map (`Dict[str, Union[int, str, torch.device]]`): The device map to check.
    """
    all_model_tensors = [name for name, _ in model.state_dict().items()]
    for module_name in device_map.keys():
        if module_name == "":
            all_model_tensors.clear()
            break
        else:
            all_model_tensors = [
                name
                for name in all_model_tensors
                if not name == module_name and not name.startswith(module_name + ".")
            ]
    if len(all_model_tensors) > 0:
        non_covered_params = ", ".join(all_model_tensors)
        raise ValueError(
            f"The device_map provided does not give any device for the following parameters: {non_covered_params}"
        )

def dispatch_model(
    model: nn.Module,
    device_map: Dict[str, Union[str, int, torch.device]],
    main_device: Optional[torch.device] = None,
    state_dict: Optional[Dict[str, torch.Tensor]] = None,
    offload_dir: Optional[Union[str, os.PathLike]] = None,
    offload_index: Optional[Dict[str, str]] = None,
    offload_buffers: bool = False,
    skip_keys: Optional[Union[str, List[str]]] = None,
    preload_module_classes: Optional[List[str]] = None,
    force_hooks: bool = False,
):
    """
    Dispatches a model according to a given device map. Layers of the model might be spread across GPUs, offloaded on
    the CPU or even the disk.

    Args:
        model (`torch.nn.Module`):
            The model to dispatch.
        device_map (`Dict[str, Union[str, int, torch.device]]`):
            A dictionary mapping module names in the models `state_dict` to the device they should go to. Note that
            `"disk"` is accepted even if it's not a proper value for `torch.device`.
        main_device (`str`, `int` or `torch.device`, *optional*):
            The main execution device. Will default to the first device in the `device_map` different from `"cpu"` or
            `"disk"`.
        state_dict (`Dict[str, torch.Tensor]`, *optional*):
            The state dict of the part of the model that will be kept on CPU.
        offload_dir (`str` or `os.PathLike`):
            The folder in which to offload the model weights (or where the model weights are already offloaded).
        offload_index (`Dict`, *optional*):
            A dictionary from weight name to their information (`dtype`/ `shape` or safetensors filename). Will default
            to the index saved in `save_folder`.
        offload_buffers (`bool`, *optional*, defaults to `False`):
            Whether or not to offload the buffers with the model parameters.
        skip_keys (`str` or `List[str]`, *optional*):
            A list of keys to ignore when moving inputs or outputs between devices.
        preload_module_classes (`List[str]`, *optional*):
            A list of classes whose instances should load all their weights (even in the submodules) at the beginning
            of the forward. This should only be used for classes that have submodules which are registered but not
            called directly during the forward, for instance if a `dense` linear layer is registered, but at forward,
            `dense.weight` and `dense.bias` are used in some operations instead of calling `dense` directly.
        force_hooks (`bool`, *optional*, defaults to `False`):
            Whether or not to force device hooks to be attached to the model even if all layers are dispatched to a
            single device.
    """
    # Error early if the device map is incomplete.
    check_device_map(model, device_map)

    # for backward compatibility
    is_bnb_quantized = (
        getattr(model, "is_quantized", False) or getattr(model, "is_loaded_in_8bit", False)
    ) and getattr(model, "quantization_method", "bitsandbytes") == "bitsandbytes"

    # We attach hooks if the device_map has at least 2 different devices or if
    # force_hooks is set to `True`. Otherwise, the model in already loaded
    # in the unique device and the user can decide where to dispatch the model.
    # If the model is quantized, we always force-dispatch the model
    if (len(set(device_map.values())) > 1) or is_bnb_quantized or force_hooks:
        if main_device is None:
            if set(device_map.values()) == {"cpu"} or set(device_map.values()) == {"cpu", "disk"}:
                main_device = "cpu"
            else:
                main_device = [d for d in device_map.values() if d not in ["cpu", "disk"]][0]

        if main_device != "cpu":
            cpu_modules = [name for name, device in device_map.items() if device == "cpu"]
            if state_dict is None and len(cpu_modules) > 0:
                state_dict = extract_submodules_state_dict(model.state_dict(), cpu_modules)

        disk_modules = [name for name, device in device_map.items() if device == "disk"]
        if offload_dir is None and offload_index is None and len(disk_modules) > 0:
            raise ValueError(
                "We need an `offload_dir` to dispatch this model according to this `device_map`, the following submodules "
                f"need to be offloaded: {', '.join(disk_modules)}."
            )
        if (
            len(disk_modules) > 0
            and offload_index is None
            and (not os.path.isdir(offload_dir) or not os.path.isfile(os.path.join(offload_dir, "index.json")))
        ):
            disk_state_dict = extract_submodules_state_dict(model.state_dict(), disk_modules)
            offload_state_dict(offload_dir, disk_state_dict)

        execution_device = {
            name: main_device if device in ["cpu", "disk"] else device for name, device in device_map.items()
        }
        execution_device[""] = main_device
        offloaded_devices = ["disk"] if main_device == "cpu" or main_device == "mps" else ["cpu", "disk"]
        offload = {name: device in offloaded_devices for name, device in device_map.items()}
        save_folder = offload_dir if len(disk_modules) > 0 else None
        if state_dict is not None or save_folder is not None or offload_index is not None:
            device = main_device if offload_index is not None else None
            weights_map = OffloadedWeightsLoader(
                state_dict=state_dict, save_folder=save_folder, index=offload_index, device=device
            )
        else:
            weights_map = None

        # When dispatching the model's parameters to the devices specified in device_map, we want to avoid allocating memory several times for the
        # tied parameters. The dictionary tied_params_map keeps track of the already allocated data for a given tied parameter (represented by its
        # original pointer) on each devices.
        tied_params = find_tied_parameters(model)

        tied_params_map = {}
        for group in tied_params:
            for param_name in group:
                # data_ptr() is enough here, as `find_tied_parameters` finds tied params simply by comparing `param1 is param2`, so we don't need
                # to care about views of tensors through storage_offset.
                data_ptr = recursive_getattr(model, param_name).data_ptr()
                tied_params_map[data_ptr] = {}

                # Note: To handle the disk offloading case, we can not simply use weights_map[param_name].data_ptr() as the reference pointer,
                # as we have no guarantee that safetensors' `file.get_tensor()` will always give the same pointer.

        attach_align_device_hook_on_blocks(
            model,
            execution_device=execution_device,
            offload=offload,
            offload_buffers=offload_buffers,
            weights_map=weights_map,
            skip_keys=skip_keys,
            preload_module_classes=preload_module_classes,
            tied_params_map=tied_params_map,
        )

        # warn if there is any params on the meta device
        offloaded_devices_str = " and ".join(
            [device for device in set(device_map.values()) if device in ("cpu", "disk")]
        )
        if len(offloaded_devices_str) > 0:
            logger.warning(
                f"Some parameters are on the meta device device because they were offloaded to the {offloaded_devices_str}."
            )

        # Attaching the hook may break tied weights, so we retie them
        retie_parameters(model, tied_params)

        # add warning to cuda and to method
        def add_warning(fn, model):
            @wraps(fn)
            def wrapper(*args, **kwargs):
                warning_msg = "You shouldn't move a model that is dispatched using accelerate hooks."
                if str(fn.__name__) == "to":
                    to_device = torch._C._nn._parse_to(*args, **kwargs)[0]
                    if to_device is not None:
                        logger.warning(warning_msg)
                else:
                    logger.warning(warning_msg)
                for param in model.parameters():
                    if param.device == torch.device("meta"):
                        raise RuntimeError("You can't move a model that has some modules offloaded to cpu or disk.")
                return fn(*args, **kwargs)

            return wrapper

        # Make sure to update _accelerate_added_attributes in hooks.py if you add any hook
        model.to = add_warning(model.to, model)
        # if is_npu_available():
        #     model.npu = add_warning(model.npu, model)
        # elif is_mlu_available():
        #     model.mlu = add_warning(model.mlu, model)
        # elif is_musa_available():
        #     model.musa = add_warning(model.musa, model)
        # elif is_xpu_available():
        #     model.xpu = add_warning(model.xpu, model)
        # else:
        #     model.cuda = add_warning(model.cuda, model)
        model.cuda = add_warning(model.cuda, model)

        # Check if we are using multi-gpus with RTX 4000 series
        use_multi_gpu = len([device for device in set(device_map.values()) if device not in ("cpu", "disk")]) > 1
        # if use_multi_gpu and not check_cuda_p2p_ib_support():
        #     logger.warning(
        #         "We've detected an older driver with an RTX 4000 series GPU. These drivers have issues with P2P. "
        #         "This can affect the multi-gpu inference when using accelerate device_map."
        #         "Please make sure to update your driver to the latest version which resolves this."
        #     )
    else:
        device = list(device_map.values())[0]
        # `torch.Tensor.to(<int num>)` is not supported by `torch_npu` (see this [issue](https://github.com/Ascend/pytorch/issues/16)).
        # if is_npu_available() and isinstance(device, int):
        #     device = f"npu:{device}"
        # elif is_mlu_available() and isinstance(device, int):
        #     device = f"mlu:{device}"
        # elif is_musa_available() and isinstance(device, int):
        #     device = f"musa:{device}"
        # elif is_xpu_available() and isinstance(device, int):
        #     device = f"xpu:{device}"
        if device != "disk":
            model.to(device)
        else:
            raise ValueError(
                "You are trying to offload the whole model to the disk. Please use the `disk_offload` function instead."
            )
    # Convert OrderedDict back to dict for easier usage
    model.hf_device_map = dict(device_map)
    return model


