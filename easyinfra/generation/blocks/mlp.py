from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from easyinfra.schedule.chunk import ChunkBlock

import torch
from torch import nn
import math
from typing import Optional
import torch.nn.functional as F
from typing import Union, List


from ...generation.parallel.parallel_configuration import MoeParallelConfig
from ...generation.parallel.communicator import GroupCommunicator
from ...generation.parallel.parallel_utils import get_device
from easyinfra.generation.modules.activation import ACT2FN
from easyinfra.generation.modules.linear import MergedColumnParallelLinear, RowParallelLinear

from ...utils.stats import show_rank_print, stage_rank_print
from easyinfra.generation.utils.perf import class_with_timing

@class_with_timing
class MlpBlock(nn.Module):
    def __init__(
        self, 
        layer_idx: int,
        hidden_size: int,
        intermediate_size: int,
        activation_key: str,
        gate_up_bias: bool,
        down_bias: bool,
        tp_group: GroupCommunicator,
    ):
        super().__init__()
        assert tp_group is not None
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.tp_group = tp_group
        self.tp_size = self.tp_group.group_size
        self.tp_rank = self.tp_group.local_rank
        
        self.gate_up_proj = MergedColumnParallelLinear(
            self.hidden_size, 
            (self.intermediate_size, self.intermediate_size), 
            param_names=("gate_proj","up_proj"),
            bias=gate_up_bias,
            tp_group=self.tp_group,
        )
        self.down_proj = RowParallelLinear(self.intermediate_size, self.hidden_size, bias=down_bias, tp_group=self.tp_group)
        self.act_fn = ACT2FN[activation_key]
        
        self.stages = (("mlp", self.compute, self.communicate),)
        self.stage_has_communication = (self.tp_size > 1,)
        
    def compute(
        self,
        chunk: ChunkBlock,
        # hidden_states: torch.Tensor
    ):
        stage_rank_print(f"chunk{chunk.chunk_id} in mlp communicate", 0)
        # residual add
        (residual, hidden_states) = chunk.active_variables
        residual += hidden_states # must be in-place
        
        hidden_states = self._compute(self.post_attention_layernorm(residual))
        
        chunk.active_variables = (residual, hidden_states)
    
    def communicate(self, chunk: ChunkBlock):
        stage_rank_print(f"chunk{chunk.chunk_id} in mlp communicate", 0)
        (residual, hidden_states) = chunk.active_variables
        hidden_states, handler = self._communicate(hidden_states)
        chunk.comm_handler = handler
        chunk.active_variables = (residual, hidden_states)
    
    def _compute(self, x: torch.Tensor):
        gate_states, up_states = self.gate_up_proj(x).split(self.gate_up_proj.split_sizes, dim=-1)
        x = self.down_proj(self.act_fn(gate_states) * up_states)
        return x
    
    def _communicate(self, hidden_states: torch.Tensor):
        hidden_states_work = self.tp_group.all_reduce(hidden_states, async_op=True)
        hidden_states_work.wait()
        handler = hidden_states_work.handler
        return hidden_states, handler
    
    def forward(self, hidden_states):
        raise NotImplementedError
        
