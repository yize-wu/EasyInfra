from transformers.modeling_utils import (
    get_max_memory,
    find_tied_parameters,
)
from accelerate.utils import (
    find_tied_parameters,
    get_max_memory,
)
from accelerate.utils.modeling import (
    compute_module_sizes,
    check_tied_parameters_in_config,
    get_max_layer_size,
    compute_module_total_buffer_size,
    clean_device_map,
)

from typing import List, Optional, Tuple, Union, Dict, Callable, OrderedDict

import warnings
import torch
from torch import nn
import torch.distributed as dist

def infer_auto_device_map(
    model: nn.Module,
    max_memory: Optional[Dict[Union[int, str], Union[int, str]]] = None,
    no_split_module_classes: Optional[List[str]] = None,
    dtype: Optional[Union[str, torch.dtype]] = None,
    special_dtypes: Optional[Dict[str, Union[str, torch.dtype]]] = None,
    verbose: bool = False,
    clean_result: bool = True,
    offload_buffers: bool = False,
):
    # Get default / clean up max_memory
    max_memory = get_max_memory(max_memory)
    if no_split_module_classes is None:
        no_split_module_classes = []
    elif not isinstance(no_split_module_classes, (list, tuple)):
        no_split_module_classes = [no_split_module_classes]

    devices = list(max_memory.keys())
    if "disk" not in devices:
        devices.append("disk")
    gpus = [device for device in devices if device not in ["cpu", "disk"]]

    # Devices that need to keep space for a potential offloaded layer.
    if "mps" in gpus:
        main_devices = ["mps"]
    elif len(gpus) > 0:
        main_devices = [gpus[0], "cpu"]
    else:
        main_devices = ["cpu"]

    module_sizes = compute_module_sizes(model, dtype=dtype, special_dtypes=special_dtypes)
    tied_parameters = find_tied_parameters(model)

    if check_tied_parameters_in_config(model) and len(tied_parameters) == 0:
        logger.warn(
            "The model weights are not tied. Please use the `tie_weights` method before using the `infer_auto_device` function."
        )

    device_map = OrderedDict()
    current_device = 0
    current_memory_used = 0
    device_memory_used = {}
    device_buffer_sizes = {}

    # Direct submodules and parameters
    modules_to_treat = (
        list(model.named_parameters(recurse=False))
        + list(model.named_children())
        + list(model.named_buffers(recurse=False))
    )
    # Initialize maximum largest layer, to know which space to keep in memory
    max_layer_size, max_layer_names = get_max_layer_size(modules_to_treat, module_sizes, no_split_module_classes)

    # Ready ? This is going to be a bit messy.
    while len(modules_to_treat) > 0:
        name, module = modules_to_treat.pop(0)
        if verbose:
            print(f"\nTreating module {name}.")
        # Max size in the remaining layers may have changed since we took one, so we maybe update it.
        max_layer_names = [n for n in max_layer_names if n != name and not n.startswith(name + ".")]
        if len(max_layer_names) == 0:
            max_layer_size, max_layer_names = get_max_layer_size(
                [(n, m) for n, m in modules_to_treat if isinstance(m, torch.nn.Module)],
                module_sizes,
                no_split_module_classes,
            )
        # Assess size needed
        module_size = module_sizes[name]

        # We keep relevant tied parameters only: one of the tied parameters in the group is inside the current module
        # and the other is not.
        # Note: If we are currently processing the name `compute.weight`, an other parameter named e.g. `compute.weight_submodule.parameter`
        # needs to be considered outside the current module, hence the check with additional dots.
        tied_param_goups = [
            tied_group
            for tied_group in tied_parameters
            if any(name + "." in k + "." for k in tied_group) and not all(name + "." in k + "." for k in tied_group)
        ]

        if verbose and len(tied_param_goups) > 0:
            print(f"  Found the relevant tied param groups {tied_param_goups}")

        # Then we keep track of all the parameters that are tied to the current module, but not in the current module
        tied_params = sum(
            [[p for p in tied_group if name + "." not in p + "."] for tied_group in tied_param_goups], []
        )

        if verbose and len(tied_params) > 0:
            print(f"  So those parameters need to be taken into account {tied_params}")

        device = devices[current_device]
        current_max_size = max_memory[device] if device != "disk" else None
        current_memory_reserved = 0
        # Reduce max size available by the largest layer.
        if devices[current_device] in main_devices:
            current_max_size = current_max_size - max_layer_size
            current_memory_reserved = max_layer_size
        # Case 1 -> We're too big!
        if current_max_size is not None and current_memory_used + module_size > current_max_size:
            # Split or not split?
            modules_children = (
                []
                if isinstance(module, nn.Parameter) or isinstance(module, torch.Tensor)
                else list(module.named_children())
            )
            if verbose:
                print(
                    f"Not enough space on {devices[current_device]} to put {name} (space available "
                    f"{current_max_size - current_memory_used}, module size {module_size})."
                )
            if len(modules_children) == 0 or module.__class__.__name__ in no_split_module_classes:
                # -> no split, we go to the next device
                if verbose:
                    print("This module cannot be split, going to the next device.")

                device_memory_used[device] = current_memory_used + current_memory_reserved
                current_device += 1
                modules_to_treat = [(name, module)] + modules_to_treat
                current_memory_used = 0
            else:
                # -> split, we replace the module studied by its children + parameters
                if verbose:
                    print(f"Splitting {name}.")
                modules_children = list(module.named_parameters(recurse=False)) + modules_children
                modules_to_treat = [(f"{name}.{n}", v) for n, v in modules_children] + modules_to_treat
                # Update the max layer size.
                max_layer_size, max_layer_names = get_max_layer_size(
                    [(n, m) for n, m in modules_to_treat if isinstance(m, torch.nn.Module)],
                    module_sizes,
                    no_split_module_classes,
                )

        # Case 2, it fits! We're not entirely out of the wood though, because we may have some tied parameters.
        elif len(tied_params) > 0:
            # First locate all tied modules
            tied_module_names = []
            tied_modules = []
            for tied_param in tied_params:
                tied_module_index = [i for i, (n, _) in enumerate(modules_to_treat) if n in tied_param][0]
                tied_module_names.append(modules_to_treat[tied_module_index][0])
                tied_modules.append(modules_to_treat[tied_module_index][1])
            if verbose:
                print(
                    f"  It looks like {name} is going to fit on {devices[current_device]} but we have tied "
                    f"parameters to account for.\n  - Names {tied_params}\n  - Module names {tied_module_names}"
                )

            # Let's see if it all fits first
            module_size_with_ties = module_size
            for tied_param, tied_module_name in zip(tied_params, tied_module_names):
                module_size_with_ties += module_sizes[tied_module_name] - module_sizes[tied_param]

            if current_max_size is None or current_memory_used + module_size_with_ties <= current_max_size:
                # We really really fit!
                if verbose:
                    print(f"Putting {name} and {tied_module_names} on {devices[current_device]}.")
                current_memory_used += module_size_with_ties
                device_map[name] = devices[current_device]
                for tied_module_name in tied_module_names:
                    if tied_module_name in [m[0] for m in modules_to_treat]:
                        # The module may have been removed by a previous iteration of this loop.
                        tied_module_index = [i for i, (n, _) in enumerate(modules_to_treat) if n == tied_module_name][
                            0
                        ]
                        modules_to_treat.pop(tied_module_index)
                    device_map[tied_module_name] = devices[current_device]

                if not offload_buffers and isinstance(module, nn.Module):
                    current_buffer_size = compute_module_total_buffer_size(
                        module, dtype=dtype, special_dtypes=special_dtypes
                    )
                    device_buffer_sizes[device] = device_buffer_sizes.get(device, 0) + current_buffer_size

            else:
                # We don't fit with the tied modules. Next question is: can we split one of the tied modules to make it
                # smaller or do we need to go on the next device?
                if verbose:
                    print(
                        f"Not enough space on {devices[current_device]} to put {name} and {tied_module_names} (space "
                        f"available {current_max_size - current_memory_used}, needed size {module_size_with_ties})."
                    )
                split_happened = False
                for tied_module_name, tied_module in zip(tied_module_names, tied_modules):
                    tied_module_children = list(tied_module.named_children())
                    if len(tied_module_children) == 0 or tied_module.__class__.__name__ in no_split_module_classes:
                        # can't break this one.
                        continue

                    if verbose:
                        print(f"Splitting {tied_module_name}.")
                    tied_module_children = list(tied_module.named_parameters(recurse=False)) + tied_module_children
                    tied_module_children = [(f"{tied_module_name}.{n}", v) for n, v in tied_module_children]
                    tied_module_index = [i for i, (n, _) in enumerate(modules_to_treat) if n == tied_module_name][0]

                    modules_to_treat = (
                        [(name, module)]
                        + modules_to_treat[:tied_module_index]
                        + tied_module_children
                        + modules_to_treat[tied_module_index + 1 :]
                    )
                    # Update the max layer size.
                    max_layer_size, max_layer_names = get_max_layer_size(
                        [(n, m) for n, m in modules_to_treat if isinstance(m, torch.nn.Module)],
                        module_sizes,
                        no_split_module_classes,
                    )
                    split_happened = True
                    break

                if not split_happened:
                    # If the tied module is not split, we go to the next device
                    if verbose:
                        print("None of the tied module can be split, going to the next device.")

                    device_memory_used[device] = current_memory_used + current_memory_reserved
                    current_device += 1
                    modules_to_treat = [(name, module)] + modules_to_treat
                    current_memory_used = 0

        else:
            if verbose:
                if current_max_size is None:
                    print(f"Putting {name} (size={module_size}) on {devices[current_device]}.")
                else:
                    print(
                        f"Putting {name} (size={module_size}) on {devices[current_device]} "
                        f"(available={current_max_size - current_memory_used})."
                    )
            current_memory_used += module_size
            device_memory_used[device] = current_memory_used + current_memory_reserved
            device_map[name] = devices[current_device]

            if not offload_buffers and isinstance(module, nn.Module):
                current_buffer_size = compute_module_total_buffer_size(
                    module, dtype=dtype, special_dtypes=special_dtypes
                )
                device_buffer_sizes[device] = device_buffer_sizes.get(device, 0) + current_buffer_size

    if clean_result:
        device_map = clean_device_map(device_map)

    non_gpu_buffer_size = device_buffer_sizes.get("cpu", 0) + device_buffer_sizes.get("disk", 0)
    if non_gpu_buffer_size > 0 and not offload_buffers:
        is_buffer_fit_any_gpu = False
        for gpu_device, gpu_max_memory in max_memory.items():
            if gpu_device == "cpu" or gpu_device == "disk":
                continue

            if not is_buffer_fit_any_gpu:
                gpu_memory_used = device_memory_used.get(gpu_device, 0)

                if gpu_max_memory >= non_gpu_buffer_size + gpu_memory_used:
                    is_buffer_fit_any_gpu = True

        if len(gpus) > 0 and not is_buffer_fit_any_gpu:
            warnings.warn(
                f"Current model requires {non_gpu_buffer_size} bytes of buffer for offloaded layers, which seems does "
                f"not fit any GPU's remaining memory. If you are experiencing a OOM later, please consider using "
                f"offload_buffers=True."
            )

    return device_map

def is_sorted_list(l: List[int]):
    for i in range(len(l)-1):
        if l[i] >= l[i+1]:
            return False
    return True

def named_module_tensors(
    module: nn.Module, include_buffers: bool = True, recurse: bool = False, remove_non_persistent: bool = False
):
    yield from module.named_parameters(recurse=recurse)

    # if include_buffers:
    #     non_persistent_buffers = set()
    #     if remove_non_persistent:
    #         non_persistent_buffers = get_non_persistent_buffers(module, recurse=recurse)
    #     for named_buffer in module.named_buffers(recurse=recurse):
    #         name, _ = named_buffer
    #         if name not in non_persistent_buffers:
    #             yield named_buffer

def get_module_names(
    model: nn.Module,
    buffers_only: bool = False,
):
    """
    Compute the size of each submodule of a given model.
    """
    module_list = named_module_tensors(model, recurse=True)
    module_names = []
    for name, tensor in module_list:
        module_names.append(name)
    return module_names

def infer_multi_gpu_device_map(
    model: nn.Module,
    tensor_parallel_group,
    tensor_parallel_size: int,
):
    module_names = get_module_names(model)
    tp_rank = dist.get_rank(tensor_parallel_group)
    
    tp_group_ranks = dist.get_process_group_ranks(tensor_parallel_group)
    if not is_sorted_list(tp_group_ranks):
        raise ValueError
    
    group_largest_global_rank = max(tp_group_ranks)
    tp_group_id = (group_largest_global_rank + 1) // tensor_parallel_size - 1
    
    device = tp_group_id * tensor_parallel_size + tp_rank
    # device is the global device idx
    # device_map = {name:device for name in module_names}
    
    global_rank = dist.get_rank()
    device_map = {name:global_rank for name in module_names}
    return device_map