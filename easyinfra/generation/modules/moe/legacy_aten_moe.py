from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from easyinfra.generation.parallel.parallel_state import GroupCommunicator

import torch
from torch import nn

from typing import List


from easyinfra.generation.modules.activation import ACT2FN
from easyinfra.utils.tensors import _to_cpu_list
from .native_moe import _MoENoExperts, MLP

class ATenMoE(_MoENoExperts):
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
        self.act_fn = ACT2FN[activation_key]        
    
    def forward_noep(
        self,
        hidden_states: torch.Tensor, 
        sorted_selected_experts: torch.Tensor, 
    ):
        raise NotImplementedError
    
    def forward_ep(
        self,
        hidden_states: torch.Tensor, 
        block_sizes: List[int], 
    ):

        M = sum(block_sizes)
        if M == 0:
            # no need to compute
            assert hidden_states.numel() == 0
            return torch.empty_like(hidden_states)
        
        ## GATE-UP                
        gate_up_weights = [e.gate_up_proj.weight for e in self.experts]
        # the input could be multiple contiguous blocks of hidden states
        num_weight_rep = len(block_sizes) // len(gate_up_weights)
        if len(gate_up_weights) * num_weight_rep != len(block_sizes):
            raise ValueError(f"we have {len(gate_up_weights)} physical experts, but block size is of length {len(block_sizes)}.")
        assert num_weight_rep in (1, self.ep_size) # 1 for gather_reduce, ep_size for all2all
        gate_up_weights *= num_weight_rep
        hidden_states = torch.ops.ops.aten_grouped_gemm_single(hidden_states, gate_up_weights, block_sizes)
        gate_states, up_states = hidden_states.split((self.moe_tp_intermediate_size, self.moe_tp_intermediate_size), dim=-1)  
              
        ## ACT                
        hidden_states = self.act_fn(gate_states) * up_states
        del gate_states, up_states
        
        ## DOWN                
        down_weights = [e.down_proj.weight for e in self.experts] * num_weight_rep
        hidden_states = torch.ops.ops.aten_grouped_gemm_single(hidden_states, down_weights, block_sizes)



        return hidden_states

    def forward_ep_permute_unpermute(
        self,
        hidden_states: torch.Tensor, 
        expert_count: List[int], 
    ):
        M = sum(expert_count)
        if M == 0:
            # no need to compute
            assert hidden_states.numel() == 0
            return torch.empty_like(hidden_states)
        
        ### Permute
        hidden_state_splits = hidden_states.split(expert_count) # [ep_size * num_local_experts]
        _index = torch.arange(len(expert_count)).view(self.ep_size, -1).transpose(0,1).reshape(-1) # [0, num_experts, ne*2, ... ne*(ep-1), 1, ...]
        permuted_tensors = [None] * len(hidden_state_splits)
        for i,j in enumerate(_index):
            permuted_tensors[i] = hidden_state_splits[j]
        del hidden_state_splits
        hidden_states = torch.cat(permuted_tensors, dim=0)
        del permuted_tensors
        
        ### Compute
        expert_count_T = torch.tensor(expert_count).view(self.ep_size, -1).transpose(0,1) # [num_experts, ep_rank]
        block_sizes = _to_cpu_list(expert_count_T.sum(-1)) # [num_experts]
        gate_up_weights = [e.gate_up_proj.weight for e in self.experts]
        hidden_states = torch.ops.ops.aten_grouped_gemm_single(hidden_states, gate_up_weights, block_sizes)
        gate_states, up_states = hidden_states.split((self.moe_tp_intermediate_size, self.moe_tp_intermediate_size), dim=-1)   
        hidden_states = self.act_fn(gate_states) * up_states
        down_weights = [e.down_proj.weight for e in self.experts]
        hidden_states = torch.ops.ops.aten_grouped_gemm_single(hidden_states, down_weights, block_sizes)

        ### Unpermute
        hidden_state_splits = hidden_states.split(_to_cpu_list(expert_count_T.reshape(-1))) # [ep_size * num_local_experts]
        _index = _index
        permuted_tensors = [None] * len(hidden_state_splits)
        for i,j in enumerate(_index):
            permuted_tensors[j] = hidden_state_splits[i]
        del hidden_state_splits
        hidden_states = torch.cat(permuted_tensors, dim=0)
        del permuted_tensors
        
        return hidden_states
