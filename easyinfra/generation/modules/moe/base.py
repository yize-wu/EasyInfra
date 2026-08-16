from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from easyinfra.generation.parallel.parallel_state import GroupCommunicator

import torch
from torch import nn

from easyinfra import envs

from easyinfra.generation.modules.linear import MergedColumnParallelLinear, RowParallelLinear
from easyinfra.generation.modules.activation import ACT2FN
from easyinfra.utils import record_time_sync

class MLP(nn.Module):
    def __init__(
        self, 
        hidden_size: int, 
        intermediate_size: int, 
        activation_key: str,
        gate_up_bias: bool,
        down_bias: bool,
        moe_tp_group: GroupCommunicator,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        # self.gate_proj = BaseLinear(self.hidden_size, self.intermediate_size, bias=False)
        # self.up_proj = BaseLinear(self.hidden_size, self.intermediate_size, bias=False)
        self.gate_up_proj = MergedColumnParallelLinear(
            self.hidden_size, 
            (self.intermediate_size, self.intermediate_size), 
            param_names=("gate_proj","up_proj"),
            bias=gate_up_bias,
            tp_group=moe_tp_group,
        )
        self.down_proj = RowParallelLinear(self.intermediate_size, self.hidden_size, bias=down_bias, tp_group=moe_tp_group,)
        self.act_fn = ACT2FN[activation_key]

    def forward(self, x):
        # return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        gate_states, up_states = self.gate_up_proj(x).split(self.gate_up_proj.split_sizes, dim=-1)
        return self.down_proj(self.act_fn(gate_states) * up_states)

class _MoENoExperts(nn.Module):
    def __init__(
        self, 
        layer_idx: int,
        moe_intermediate_size: int,
        num_logical_experts: int,
        ep_group: GroupCommunicator,
        moe_tp_group: GroupCommunicator,
        phy2log_expert_map: torch.Tensor,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        
        self.num_logical_experts = num_logical_experts
        self.ep_group = ep_group
        self.moe_tp_group = moe_tp_group


        self.ep_size = ep_group.group_size
        self.ep_rank = ep_group.local_rank
        self.moe_tp_size = moe_tp_group.group_size
        self.moe_tp_rank = moe_tp_group.local_rank
        self.moe_tp_intermediate_size = moe_intermediate_size // self.moe_tp_size
        if self.moe_tp_intermediate_size * self.moe_tp_size != moe_intermediate_size:
            raise ValueError(f"MoE intermediate size {moe_intermediate_size} cannot divide moe_tp size {self.moe_tp_size}")
        self.num_physical_experts = phy2log_expert_map.shape[-1]
        self.num_global_experts = self.num_physical_experts * self.ep_size
        
        if ep_group.group_size * (self.num_logical_experts // self.ep_size) != self.num_logical_experts:
            raise ValueError(f"Num logical experts cannot be devided by EP size: ep size is {self.ep_size}, but the number of experts is {self.num_logical_experts}. "
                             f"This should be raised before!")
        
        # permute-unpermute indices
        self.permute_indices = torch.arange(self.num_global_experts).view(self.ep_size, -1).transpose(0,1).reshape(-1) # [0, num_experts, ne*2, ... ne*(ep-1), 1, ...]
        if self.ep_size > 1 and not hasattr(self, "forward_ep"):
            raise NotImplementedError
        if self.ep_size == 1 and not hasattr(self, "forward_noep"):
            raise NotImplementedError

        if envs.ENABLE_MOE_PERMUTE_UNPERMUTE:
            self.forward = self.forward_ep_permute_unpermute if self.ep_size > 1 else self.forward_noep
        else:
            self.forward = self.forward_ep if self.ep_size > 1 else self.forward_noep
        
        self.expert_compute_time = 0.0