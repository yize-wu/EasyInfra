from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from easyinfra.generation.parallel.parallel_state import GroupCommunicator

import torch
from torch import nn

from typing import List

from easyinfra.utils.tensors import _to_cpu_list

from easyinfra.generation.modules.activation import ACT2FN
from easyinfra.utils import record_time_sync, show_rank_print

from easyinfra.generation.modules.moe.base import _MoENoExperts, MLP
from easyinfra import envs

import time

class SingleGroupGEMMMoE(_MoENoExperts):
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
        '''
            For backends that use X_grouped_gemm_single(A, B_list, block_sizes, C_option)
        '''
        super().__init__(
            layer_idx=layer_idx,
            moe_intermediate_size=moe_intermediate_size,
            num_logical_experts=num_logical_experts,
            ep_group=ep_group,
            moe_tp_group=moe_tp_group,
            phy2log_expert_map=phy2log_expert_map,
        )

        self.experts = nn.ModuleList(
            [MLP(hidden_size, moe_intermediate_size, activation_key, gate_up_bias, down_bias, moe_tp_group) 
             for _ in range(self.num_physical_experts)]
        )
        self.act_fn = ACT2FN[activation_key]

        ## Single GroupGEMM implementation
        if envs.MOE_IMPL == "cutlass":
            self.grouped_gemm_single_impl = torch.ops.ops.cutlass_grouped_gemm_single
        elif envs.MOE_IMPL == "aten":
            self.grouped_gemm_single_impl = torch.ops.ops.aten_grouped_gemm_single
        else:
            moe_impl_str = envs.MOE_IMPL if envs.MOE_IMPL else "not set in envs."
            raise NotImplementedError(f"Not supported MoE_IMPL={moe_impl_str}")
            
    def forward_ep(
        self,
        hidden_states: torch.Tensor, 
        block_sizes: List[int], 
    ):
        # return hidden_states
                        
        M = sum(block_sizes)
        if M == 0:
            # no need to compute
            assert hidden_states.numel() == 0
            return torch.empty_like(hidden_states)
        
        ## GATE-UP                
        gate_up_weights = [e.gate_up_proj.weight for e in self.experts]
        # the input could be multiple contiguous blocks of hidden states
        num_weight_rep = len(block_sizes) // len(gate_up_weights)
        assert num_weight_rep in (1, self.ep_size) # 1 for gather_reduce, ep_size for all2all
        gate_up_weights *= num_weight_rep
        hidden_states = self.grouped_gemm_single_impl(hidden_states, gate_up_weights, block_sizes)
        gate_states, up_states = hidden_states.split((self.moe_tp_intermediate_size, self.moe_tp_intermediate_size), dim=-1)  
              
        ## ACT                
        hidden_states = self.act_fn(gate_states) * up_states
        del gate_states, up_states
        
        ## DOWN                
        down_weights = [e.down_proj.weight for e in self.experts] * num_weight_rep
        hidden_states = self.grouped_gemm_single_impl(hidden_states, down_weights, block_sizes)
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
        permuted_tensors = [None] * len(hidden_state_splits)
        for i,j in enumerate(self.permute_indices):
            permuted_tensors[i] = hidden_state_splits[j]
        del hidden_state_splits
        hidden_states = torch.cat(permuted_tensors, dim=0)
        del permuted_tensors
        
        ### Compute
        expert_count_T = torch.tensor(expert_count).view(self.ep_size, -1).transpose(0,1) # [num_experts, ep_rank]
        block_sizes = _to_cpu_list(expert_count_T.sum(-1)) # [num_experts]
        gate_up_weights = [e.gate_up_proj.weight for e in self.experts]
        hidden_states = self.grouped_gemm_single_impl(hidden_states, gate_up_weights, block_sizes)
        gate_states, up_states = hidden_states.split((self.moe_tp_intermediate_size, self.moe_tp_intermediate_size), dim=-1)   
        hidden_states = self.act_fn(gate_states) * up_states
        down_weights = [e.down_proj.weight for e in self.experts]
        hidden_states = self.grouped_gemm_single_impl(hidden_states, down_weights, block_sizes)

        ### Unpermute
        hidden_state_splits = hidden_states.split(_to_cpu_list(expert_count_T.reshape(-1))) # [ep_size * num_local_experts]
        permuted_tensors = [None] * len(hidden_state_splits)
        for i,j in enumerate(self.permute_indices):
            permuted_tensors[j] = hidden_state_splits[i]
        del hidden_state_splits
        hidden_states = torch.cat(permuted_tensors, dim=0)
        del permuted_tensors
        
        # torch.cuda.current_stream().synchronize()
        # time1 = time.time()
        # self.expert_compute_time += time1 - time0
        # show_rank_print(f"expert_compute_time: {time1 - time0}", 0)
        return hidden_states
    
        
