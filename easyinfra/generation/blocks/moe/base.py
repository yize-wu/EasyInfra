import torch
import time
import os
import json
from torch import nn
import math
from typing import Optional
import torch.nn.functional as F
from typing import Union, List

from easyinfra.generation.blocks.mlp import MlpBlock
from easyinfra.generation.parallel.parallel_utils import get_device
from easyinfra.generation.parallel.communicator import GroupCommunicator
from easyinfra.generation.parallel.parallel_configuration import MoeParallelConfig

from easyinfra.generation.modules.moe import (
    get_moe_impl,
    is_chunk_routing,
)
from easyinfra.generation.utils.compute_utils import _add_to_residual
from easyinfra.utils.stats import show_rank_print
from easyinfra.utils.tensors import _to_cpu_list
from easyinfra import envs

def logical_to_physical_expert_ids(
    logical_expert_id: int,
    logical_experts: torch.LongTensor,
)-> List[int]:
    # check if it is unique
    logical_expert_list = _to_cpu_list(logical_experts)
    # if logical_expert_id in logical_expert_list:
    #     # if not is_unique(logical_expert_id, logical_experts):
    #     #     raise ValueError(f"Multiple id in map:\nlogid {logical_expert_id} in {logical_experts}")
    #     ### indexing
    physical_expert_ids = [i for i, x in enumerate(logical_expert_list) if x == logical_expert_id]
    # else:
    #     physical_expert_ids = None
    return physical_expert_ids

def eplb_map_to_physical_and_record(
    selected_experts: torch.Tensor,
    logical_to_physical_expert_map: torch.Tensor,
    logical_replica_count: torch.Tensor,
):
    '''
        Sourced from vllm/model_executor/layers/fused_moe/router/base_router.py
        logical_to_physical_expert_map is from this rank.
    '''
    selected_experts_long = selected_experts.long()
    # Use (token position) modulo (replica count)
    # to deterministically choose a replica
    replica_count = logical_replica_count[selected_experts_long]
    # Flatten-position based index, reshaped back to `topk_ids` shape
    pos_indices = torch.arange(
        selected_experts.numel(), device=selected_experts.device, dtype=torch.long
    ).reshape_as(selected_experts)
    # Compute pseudo-random indices by modulo
    replica_indices = (pos_indices % replica_count).unsqueeze(-1)
    physical_ids = (
        logical_to_physical_expert_map[selected_experts_long]
        .gather(-1, replica_indices)
        .squeeze(-1)
    )
    return physical_ids

from easyinfra.generation.blocks.stage_name import MoeStageName

class _MoeBlock(nn.Module):
    
    def __init__(
        self, 
        layer_idx: int,
        hidden_size: int,
        # MoE
        moe_intermediate_size: int,
        activation_key: str,
        gate_up_bias: bool,
        down_bias: bool,
        num_logical_experts: int,
        num_experts_per_tok: int,
        norm_topk_prob: bool,
        use_eplb: bool,
        # communicator
        chunk_routing_group: GroupCommunicator,
        ep_group: GroupCommunicator,
        moe_tp_group: GroupCommunicator,
        # Shared
        has_shared: bool,
        shared_expert_intermediate_size: Optional[int] = None,
        shared_activation_key: Optional[str] = None,
        shared_gate_up_bias: Optional[bool] = None,
        shared_down_bias: Optional[bool] = None,
        shared_expert_tp_group: Optional[GroupCommunicator] = None,
        # EPLB
        num_global_experts: Optional[int] = None,
        num_first_k_dense_layers: Optional[int] = None,
        # Config
        parallel_config: Optional[MoeParallelConfig] = None
    ):
        '''
            num_global_experts: physical number of global experts
        '''
        
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.parallel_config = parallel_config
        
        self.num_experts_per_tok = num_experts_per_tok
        self.norm_topk_prob = norm_topk_prob
        self.num_logical_experts = num_logical_experts
        
        self.use_chunk_routing: bool = is_chunk_routing()
        self.chunk_routing_group = chunk_routing_group
        if chunk_routing_group is not None:
            self.chunk_routing_size = chunk_routing_group.group_size
            self.chunk_routing_rank = chunk_routing_group.local_rank
        else:
            self.chunk_routing_size = 1
            self.chunk_routing_rank = 0
        self.shared_expert_tp_group = shared_expert_tp_group # could be None for non-shared layer
        if shared_expert_tp_group is not None and not has_shared:
            raise ValueError(f"No Shared Expert Found, but shared_expert_tp_group is concrete.")
        self.shared_expert_tp_size = shared_expert_tp_group.group_size if shared_expert_tp_group is not None else 1
        self.ep_group = ep_group
        self.ep_data_group = parallel_config.ep_data_group
        self.ep_size = ep_group.group_size
        self.ep_rank = ep_group.local_rank
        assert self.ep_data_group.group_size == self.ep_size
        assert self.ep_data_group.local_rank == self.ep_rank
        
        # EPLB
        self.use_eplb = use_eplb
        if num_global_experts is None:
            if self.use_eplb is True:
                raise ValueError(f"EPLB must specify the global expert number.")
            num_global_experts = num_logical_experts
        else:
            if self.use_eplb is False:
                raise ValueError(f"EPLB is disabled but the global expert number is set.")
            else:
                if num_global_experts < num_logical_experts:
                    raise ValueError(f"EPLB global expert number {num_global_experts} must be greater than logical {num_logical_experts}.")
                
                if num_global_experts % self.ep_size != 0:
                    raise ValueError(f"EPLB only support divided global number, but {num_global_experts} with ep={self.ep_size}")
                if num_global_experts > num_logical_experts * self.ep_size:
                    raise ValueError(f"Too large global {num_global_experts} for ep={self.ep_size}. Logical experts are {num_logical_experts}.")
        self.num_first_k_dense_layers = num_first_k_dense_layers
        self.num_global_experts = num_global_experts
        self.num_physical_experts = num_global_experts // self.ep_size
        
        ## physical to logical expert map
        ## For all ranks at the same layer, the map is uniform
        self.enable_eplb_config_read = self.use_eplb
        if self.enable_eplb_config_read is True and self.use_eplb is False:
            raise ValueError(f"EPLB config can only be read when using EPLB."
                             " Set enable_eplb_config_read to False, or use_eplb to True.")
        if self.enable_eplb_config_read is True:
            phy2log_expert_map = self.load_eplb_config()
        else:
            ## default replication
            num_base_experts = num_logical_experts // self.ep_size
            assert num_logical_experts == num_base_experts * self.ep_size
            base = torch.arange(num_base_experts * self.ep_size).view(self.ep_size, num_base_experts)
            tail = torch.arange(1, self.num_physical_experts - num_base_experts + 1)[None,:] + base[:,-1:]
            phy2log_expert_map = (torch.cat([base, tail], dim=-1) % num_logical_experts)
        self.phy2log_expert_map_cpu = phy2log_expert_map.pin_memory()
        self.phy2log_expert_map = self.phy2log_expert_map_cpu.to(get_device())

        ## logical replica count
        _p2l_flat = self.phy2log_expert_map.reshape(-1)
        self.logical_replica_count = torch.bincount(_p2l_flat, minlength=num_logical_experts)
        ## logical to physical expert map
        max_replica_count = self.logical_replica_count.max().item()
        logical_to_physical_map = torch.full((num_logical_experts, max_replica_count), -1,
            dtype=torch.long, device=self.phy2log_expert_map.device
        )
        # fill
        offset = torch.zeros(num_logical_experts, dtype=torch.long, device=self.phy2log_expert_map.device)
        for physical_id in range(self.num_global_experts):
            logical_id = _p2l_flat[physical_id]
            logical_to_physical_map[logical_id, offset[logical_id]] = physical_id
            offset[logical_id] += 1
        self.log2phy_expert_map = logical_to_physical_map # [num_logical_experts, max_replica]
        ## expert mask retrieve
        self.local_expert_mask_retriever = torch.arange(self.num_global_experts, device=get_device()).view(self.ep_size, self.num_physical_experts)
        
        self.post_attention_layernorm: nn.Module = None
        self.gate: nn.Module = None
        
        moe_implementation = get_moe_impl()
        self.experts = moe_implementation(
            layer_idx,
            hidden_size,
            moe_intermediate_size,
            activation_key,
            gate_up_bias,
            down_bias,
            num_logical_experts,
            ep_group,
            moe_tp_group,
            self.phy2log_expert_map,
        )
        
        self.has_shared = has_shared
        if not self.has_shared:
            self.shared_experts: nn.Module = None
        else:
            self.shared_experts: nn.Module = MlpBlock(
                layer_idx=layer_idx, 
                hidden_size=hidden_size, 
                intermediate_size=shared_expert_intermediate_size, 
                activation_key=shared_activation_key, 
                gate_up_bias=shared_gate_up_bias,
                down_bias=shared_down_bias,
                tp_group=shared_expert_tp_group,
            )

        self.print_moe_compute_time = envs.PRINT_MOE_COMPUTE_TIME
        
        self.expert_compute_time = 0.0
        self.expert_loop_in_prepare_time = 0.0
        self.expert_loop_out_prepare_time = 0.0
        self.expert_pre_compute_time = 0.0
        self.expert_compute_to_barrier_time = 0.0
        self.barrier_wait_time = 0.0
        self.recv_token_time = 0.0
        self.expert_post_recv_all2all_time = 0.0
        self.expert_loop_out_prepare_time = 0.0
        self.send_token_time = 0.0
        
        # self._all2all_expert_compute = self._all2all_expert_compute_torch_compile
        # self._all2all_expert_compute = self._all2all_expert_compute_eager
    
    def load_eplb_config(self) -> torch.Tensor:
        eplb_config_path = envs.EPLB_CONFIG
        if not os.path.isfile(eplb_config_path):
            if os.path.isdir(eplb_config_path):
                raise ValueError(f"Set `EPLB_CONFIG` in environment. Currently it is a directory: {eplb_config_path}.\nMaybe the config file is in this directory?")
            else:
                raise ValueError(f"`EPLB_CONFIG` points to non-exist file. Set it correctly.\nCurrently it is: {eplb_config_path}.")
                
        with open(eplb_config_path, "r") as f:
            for line in f:
                all_layer_phy2log_expert_map = json.loads(line.strip())["phy2log"]
                # TODO: not only read the first line?
                break
        moe_layer_idx_start = self.num_first_k_dense_layers if self.num_first_k_dense_layers else 0
        phy2log_expert_map = all_layer_phy2log_expert_map[self.layer_idx - moe_layer_idx_start]
        assert isinstance(phy2log_expert_map, list)
        if len(phy2log_expert_map) != self.ep_size * self.num_physical_experts:
            raise ValueError(f"Read EPLB config from {eplb_config_path}.\n"
                             f"At Layer {self.layer_idx}, config # physical experts is {len(phy2log_expert_map)}, "
                             f"while we need {self.ep_size}*{self.num_physical_experts}={self.ep_size*self.num_physical_expertsu}")
        return torch.tensor(phy2log_expert_map, device="cpu").view(self.ep_size, self.num_physical_experts)
    
    def get_stages(self):
        '''Output: stages'''
        return self.stages
    def get_stage_has_communication(self):
        '''Output: stage_has_communication'''
        return self.stage_has_communication
    
    def get_shared_experts(self):
        shared_experts = getattr(self, "shared_experts", None)
        assert (shared_experts is None and self.has_shared is False) or (shared_experts is not None and self.has_shared is True)
        return shared_experts
    
    def _routing(self, hidden_states: torch.Tensor,):
        '''
            Returns: routing_weights, selected_experts, (something else)
        '''
        raise NotImplementedError
    
    
    def _chunk_routing_compute(self, hidden_states: torch.Tensor, compute_expert_load: bool = False):
        '''
            Padding the hidden_states at the padding_side, if not divisible by chunk_tp_size.
            Returns: 
                hidden_states,
                routing_weights, 
                selected_experts, 
                router_logits,
                chunk_size,

        '''
        # context parallel across tp group
        # time0 = time.time()
        hidden_dim = hidden_states.shape[-1]
        factory_kwargs = {"dtype": hidden_states.dtype, "device": hidden_states.device}
        chunk_routing_size = self.chunk_routing_size
        h_length = hidden_states.shape[0]
        chunk_size = math.ceil(h_length / chunk_routing_size) if (h_length % chunk_routing_size) != 0 else (h_length // chunk_routing_size)
        sequence_start = chunk_size * self.chunk_routing_rank
        sequence_end = min(sequence_start + chunk_size, h_length)
        # show_rank_print(f"sequence start: {sequence_start}, end: {sequence_end}, chunk: {h_length}/{chunk_routing_size}={chunk_size}")
        # pad for the same size
        if sequence_end <= sequence_start:
            hidden_states = torch.rand((chunk_size, hidden_dim), **factory_kwargs)
        elif sequence_start != 0 or sequence_end != h_length:
            hidden_states = hidden_states[sequence_start:sequence_end,:]
            if sequence_end - sequence_start < chunk_size:
                hidden_states = torch.cat([hidden_states, torch.rand((sequence_start + chunk_size - sequence_end, hidden_dim), **factory_kwargs)], dim=0)
        
        # time1 = time.time()
        # routing
        # hidden_states = torch.rand_like(hidden_states)
        # show_rank_print(f"h_sum: {hidden_states.sum().item()}")
        # time2 = time.time()
        routing_weights, selected_experts, router_logits = self._routing(hidden_states) # [t,k]
        # Consider EPLB
        if self.use_eplb:
            # should return physical id
            selected_experts = eplb_map_to_physical_and_record(
                selected_experts, self.log2phy_expert_map, self.logical_replica_count)
        # time3 = time.time()
        # show_rank_print(f"{time1 - time0}, {time2 - time1}, {time3 - time2}", 0)
        
        # show_rank_print("careful! running with manipulated routing results", 0)
        # selected_experts = torch.randint_like(selected_experts, self.num_global_experts)
        # ratio = 0.5
        # part = int(hidden_states.shape[0] * ratio)
        # device_0_part = torch.randint(0, self.num_physical_experts, [part,8], dtype=torch.int64, device=get_device())
        # other_part = torch.randint(0, self.num_global_experts - self.num_physical_experts, 
        #                            [hidden_states.shape[0] - part, self.num_experts_per_tok], 
        #                            dtype=torch.int64, device=get_device()) + self.num_physical_experts
        # required_shape = selected_experts.shape
        # selected_experts = torch.cat([device_0_part, other_part], dim=0)
        # assert required_shape == selected_experts.shape
        
        # router_logits = None
        # routing_weights = torch.zeros([hidden_states.shape[0],8], dtype=torch.bfloat16, device=get_device())
        # selected_experts = torch.zeros([hidden_states.shape[0],8], dtype=torch.int64, device=get_device())
        return (
                hidden_states,
                routing_weights, 
                selected_experts, 
                router_logits,
                chunk_size,
        )
    
    def _all_expert_count_compute(
        self,
        selected_experts: torch.LongTensor,
    ):
        '''
        Output:
            local_tok_all_expert_count
        '''
        
        ### selected_experts is physical ID
        # local_tok_all_expert_count = local_tok_all_expert_mask.sum((-1,-2)) # [ep_rank, num_local_experts] from expert map
        # time0 = time.time()
        ids = selected_experts.view(-1)
        # No Use of torch.bincount, as it causes CPU sync somehow
        local_tok_all_expert_count_2d = torch.zeros(self.num_global_experts, device=ids.device, dtype=torch.int64)
        ones = torch.ones_like(ids, dtype=local_tok_all_expert_count_2d.dtype)
        local_tok_all_expert_count_2d.scatter_add_(0, ids, ones)
        # time1 = time.time()
        local_tok_all_expert_count_2d = local_tok_all_expert_count_2d[self.local_expert_mask_retriever]
        # time2 = time.time()
        # show_rank_print(f"{time1 - time0}, {time2 - time1}", 0)
        return local_tok_all_expert_count_2d
    
    def _all_expert_mask_compute(
        self,
        selected_experts: torch.LongTensor,
    ):
        ### expert mask: one_hot of [ep_rank, local_expert_logic_id, k, local_t]
        local_tok_all_expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_global_experts).transpose(0, 2)[self.local_expert_mask_retriever,:,:] 
        return local_tok_all_expert_mask