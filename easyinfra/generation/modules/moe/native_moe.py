import torch
from torch import nn

from typing import List

import easyinfra
from easyinfra.generation.parallel.parallel_configuration import MoeParallelConfig
from easyinfra.generation.parallel.parallel_state import GroupCommunicator
from easyinfra.generation.parallel.parallel_utils import get_device

from easyinfra.generation.modules.linear import MergedColumnParallelLinear, RowParallelLinear
from easyinfra.generation.modules.activation import ACT2FN
from easyinfra.utils import record_time_sync
from easyinfra.utils.tensors import _to_cpu_list
from .base import _MoENoExperts, MLP

class NativeMoE(_MoENoExperts):
    def __init__(
        self, 
        layer_idx:int, 
        hidden_size: int, 
        moe_intermediate_size: int,
        activation_key: str,
        gate_up_bias: bool,
        down_bias: bool,
        num_logical_experts: int,
        ep_group: GroupCommunicator,
        moe_tp_group: GroupCommunicator,
        phy2log_expert_map: torch.Tensor,
        **kwargs,
    ):
        
        super().__init__(
            layer_idx,
            moe_intermediate_size,
            num_logical_experts,
            ep_group,
            moe_tp_group,
            phy2log_expert_map,
        )

        self.experts = nn.ModuleList(
            [MLP(hidden_size, moe_intermediate_size, activation_key, gate_up_bias, down_bias, moe_tp_group) 
             for _ in range(self.num_physical_experts)]
        )
        
        
    
    def forward_noep(
        self,
        hidden_states: torch.Tensor, 
        sorted_selected_experts: torch.Tensor, 
    ):
        raise NotImplementedError
    
    def forward_ep(
        self,
        recv_hidden_states: torch.Tensor, 
        expert_count: List[int], 
    ):
        # return recv_hidden_states
        # show_rank_print
        
        expert_count: torch.Tensor = torch.tensor(expert_count, device='cpu')
        if expert_count.numel() == self.ep_size * self.num_physical_experts:
            expert_count = expert_count.view(self.ep_size, self.num_physical_experts)
            expert_hit = torch.greater(expert_count.sum(0), 0).nonzero().view(-1)
            hit_expert_num = expert_hit.shape[0]
            if hit_expert_num == 0:
                # no need to compute
                assert recv_hidden_states.numel() == 0
                return torch.empty_like(recv_hidden_states)
            
            ep_size = self.ep_size
            # [ep_size, num_local_e]
            hit_expert_count = expert_count[:,expert_hit].view(-1) # [ep_size * num_hit_experts]
            assert not hit_expert_count.is_cuda # Avoid CUDA Sync
            hit_expert_count = _to_cpu_list(hit_expert_count)
            
            recv_hidden_state_splits = recv_hidden_states.split(hit_expert_count) # [ep_size * num_hit_experts]

            # expert compute
            # _expert_compute_start = record_time_sync()
            _outputs = []
            for i,physical_expert_id in enumerate(expert_hit):
                expert_layer = self.experts[physical_expert_id]
                current_states = torch.cat([t for t in recv_hidden_state_splits[i::hit_expert_num]])
                current_states: torch.Tensor = expert_layer(current_states)
                _outputs.extend(current_states.split(hit_expert_count[i::hit_expert_num]))
            
            # re-permute outputs
            _index = torch.arange(len(hit_expert_count), device='cpu').view(ep_size, hit_expert_num).transpose(0,1).reshape(-1)
            permuted_outputs = [None] * len(hit_expert_count)
            for i,j in enumerate(_index):
                permuted_outputs[j] = _outputs[i]
            
            del _outputs
            
            permuted_outputs = torch.cat(permuted_outputs, dim=0)
            # _expert_compute_end = record_time_sync()
            # self.expert_compute_time += _expert_compute_end - _expert_compute_start
                    
            return permuted_outputs
        elif expert_count.numel() == self.num_physical_experts:
            raise NotImplementedError
            expert_hit = torch.greater(expert_count, 0).nonzero().view(-1)
            hit_expert_num = expert_hit.shape[0]
            if hit_expert_num == 0:
                assert recv_hidden_states.numel() == 0
                return torch.empty_like(recv_hidden_states)
            hit_expert_count = expert_count[expert_hit] # [num_hit_experts]
            recv_hidden_state_splits = recv_hidden_states.split(hit_expert_count) # [num_hit_experts]
            
            _outputs = []
            for i,physical_expert_id in enumerate(expert_hit):
                expert_layer = self.experts[physical_expert_id]
                current_states = recv_hidden_state_splits[i]
                current_states: torch.Tensor = expert_layer(current_states)
                _outputs.append(current_states)
            
            _outputs = torch.cat(_outputs, dim=0)
                    
            return _outputs
        else:
            raise ValueError(f"expert_count list length is {expert_count.numel()} while ep={self.ep_size} with {self.num_physical_experts} physical experts each.")
    
        
