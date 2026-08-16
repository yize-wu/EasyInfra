from typing import List, Optional, Tuple, Union, Dict
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed import ReduceOp as TorchReduceOp

from enum import Enum

from .communicator import (
    GroupCommunicator,
    Layer2LayerGroupCommunicator,
    ReduceOp,
)
from .parallel_configuration import BaseParallelConfig
from .parallel_utils import (
    get_global_rank,
    get_world_size,
)

_PARALLEL_CONFIG: List[BaseParallelConfig] = []   

def get_parallel_config(idx) -> BaseParallelConfig:
    return _PARALLEL_CONFIG[idx]

def register_parallel_config(info: BaseParallelConfig):
    idx = len(_PARALLEL_CONFIG)
    _PARALLEL_CONFIG.append(info)
    return idx
            
