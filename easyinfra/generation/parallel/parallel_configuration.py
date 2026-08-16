import torch
from typing import Optional, List

from .communicator import GroupCommunicator, HierarchicalAll2AllGroupCommunicator
from .parallel_utils import (
    get_global_rank, get_local_world_size, get_world_size,
    get_local_rank,
)
from easyinfra.utils import show_rank_print
from easyinfra import envs

def ready_device_list() -> List[int]:
    device_num = torch.cuda.device_count()
    # find ids of all visible devices
    device_list = []
    for i in range(device_num):
        try:
            _ = torch.tensor([0], device=i)
            device_list.append(i)
        except Exception:
            print(f"device {i} not working.")
            continue
    return device_list

_DEVICE_LIST = None

class _BaseParallelConfig:
    def __init__(self, all_ranks: List[int], driver=None, device=None, dtype=None):
        if driver is not None and driver not in all_ranks:
            raise ValueError(f"driver {driver} is not in all ranks {all_ranks}")
        self.all_ranks = sorted(all_ranks) # make sure it is ordered
        self.in_participant = get_global_rank() in self.all_ranks
        if device is not None:
            if _DEVICE_LIST is None:
                _DEVICE_LIST = ready_device_list() 
            if device not in _DEVICE_LIST:
                raise ValueError(f"device is specified as {device}, while active devices are {_DEVICE_LIST}")
        self.device = device if device is not None else torch.cuda.current_device()
        self.dtype = dtype if dtype is not None else torch.get_default_dtype()
        self.logits_dtype = torch.float32

class BaseParallelConfig(_BaseParallelConfig):
    """
        It has a group communicator.
    """
    def __init__(self, all_ranks, driver=None, device=None,):
        super().__init__(all_ranks, driver=driver, device=device)
        self.all_ranks_group = GroupCommunicator(all_ranks, driver=driver)
        self.dp_size = 1
        self.driver = self.all_ranks_group.driver
        self.need_broadcast_inputs = False
                
class TPParallelConfig(BaseParallelConfig):
    def __init__(self, all_ranks, driver=None, device=None):
        super().__init__(all_ranks, driver=driver, device=device)
        # tp info does not need driver
        self.tp_group: GroupCommunicator = self.all_ranks_group

class SpeculativeDecodingParallelConfig(BaseParallelConfig):
    """
        SD info needs a group communicator, so use BaseParallelConfig instead of _BaseParallelConfig.
    """
    def __init__(self, draft_parallel_config:BaseParallelConfig, verification_parallel_config:BaseParallelConfig, device=None):
        if draft_parallel_config.driver != verification_parallel_config.driver:
            raise ValueError(f"Draft and verification model must have the same driver, but {draft_parallel_config.driver} and {verification_parallel_config.driver}")
        all_ranks = sorted(list(set(draft_parallel_config.all_ranks) | set(verification_parallel_config.all_ranks)))
        # If verification model all_ranks is a subset of draft model's, then no need to broadcast candidate
        self.need_broadcast_candidate_from_draft = (set(verification_parallel_config.all_ranks) - set(draft_parallel_config.all_ranks) != set())
        # self.need_broadcast_candidate_from_draft = True # TODO: maybe no need to broadcast?
        super().__init__(all_ranks, driver=draft_parallel_config.driver, device=device)

from enum import Enum
class MoeCommMode(Enum):
    DEFAULT = "all2all"
    ALL2ALL = "all2all"
    HIERARCHICAL_ALL2ALL = "hierarchical"
    HYBRID = "hybrid"
    INTRA_GATHER = "intra_gather"
    GATHER_REDUCE_SCATTER = "gather_reduce_scatter"
    GATHER_All2All = "gather_all2all"

class MoeParallelConfig(BaseParallelConfig):
    def __init__(self, 
        all_ranks: List[int], 
        ep_size: int,
        attn_tp_size: int, 
        shared_expert_tp_size: Optional[int] = None,
        use_eplb: Optional[bool] = None,
        num_global_experts: Optional[int] = None,
        comm_mode: Optional[str] = None,
        hierarchical_all2all_structure: Optional[List[int]] = None,
        driver=None, 
        device=None
    ):
        '''
            ep_size: size of one expert-parallel group
                
            tp_size: size of one attention-tp group
        '''
        # TP is dynamically applciated to each dp groups
        super().__init__(all_ranks, driver, device)
        
        all_rank_group_size = self.all_ranks_group.group_size
        self.ep_size = ep_size
        ### Must specify num of global experts
        self.num_global_experts = num_global_experts
        
        ### it is a MoE config, so care about ep first
        if use_eplb is None:
            show_rank_print("use_eplb not specified. Default is False.")
            use_eplb = False
        self.use_eplb = use_eplb
        self.comm_mode = comm_mode if comm_mode is not None else MoeCommMode.DEFAULT
            
        # self.comm_mode = MoeCommMode.INTRA_GATHER
        if self.ep_size == all_rank_group_size:
            if (self.comm_mode == MoeCommMode.ALL2ALL
                or self.comm_mode == MoeCommMode.GATHER_REDUCE_SCATTER
                or self.comm_mode == MoeCommMode.GATHER_All2All
                ):
                ## faster init, no group creation again
                self.ep_group = self.all_ranks_group
            elif self.comm_mode == MoeCommMode.INTRA_GATHER:
                self.ep_group = self.all_ranks_group
                
                local_world_size = get_local_world_size()
                world_size = get_world_size()
                global_rank = get_global_rank()
                local_rank = get_local_rank()
                local_rank0 = global_rank - global_rank % local_world_size
                show_rank_print(f"local rank0: {local_rank0}")
                self.intra_node_group = GroupCommunicator([_ for _ in range(local_rank, world_size, local_world_size)])
                self.inter_node_group = GroupCommunicator([_ for _ in range(local_rank0, local_rank0+local_world_size)])
            else:
                self.ep_group = HierarchicalAll2AllGroupCommunicator([_ for _ in range(all_rank_group_size)], hierarchical_all2all_structure)
            
            ## Create a seperated group for expert data communiction
            # self.ep_data_group = GroupCommunicator(self.ep_group.all_ranks)    
            self.ep_data_group = self.ep_group 
        # elif all_rank_group_size % ep_size != 0:
        #     print(f"All rank size is {all_rank_group_size} but ep_size is {ep_size}, some ranks are not in expert parallelism")
        #     exit(1)
        else:
            raise NotImplementedError
            # we need a ep-dp driver
            self.ep_group = None
        self.ep_rank = self.all_ranks_group.local_rank
            
        if attn_tp_size == all_rank_group_size:
            self.attn_tp_group = self.all_ranks_group
            self.attn_dp_head_group = GroupCommunicator([self.attn_tp_group.driver])
        elif all_rank_group_size % attn_tp_size != 0:
            raise ValueError(f"All rank size is {all_rank_group_size} but attn_tp_size is {attn_tp_size}, some ranks are not in tensor parallelism")
        else:
            # we need a attn-dp driver            
            attn_tp_group_start = attn_tp_size * (get_global_rank() // attn_tp_size)
            self.attn_tp_group = GroupCommunicator([_ for _ in range(attn_tp_group_start, attn_tp_group_start+attn_tp_size)], driver=attn_tp_group_start)
            # decide a dp_head group for transmission
            if attn_tp_size == 1:
                self.attn_dp_head_group = self.all_ranks_group
            else:
                self.attn_dp_head_group = GroupCommunicator([_ for _ in range(0, all_rank_group_size, attn_tp_size)])
            
        self.attn_dp_head_size = self.attn_dp_head_group.group_size
        self.dp_size = self.attn_dp_head_size
        self.dp_rank = get_global_rank() // attn_tp_size
        
        self.mlp_tp_group = self.attn_tp_group # TODO: diverse mlp tp group?
        
        if all_rank_group_size % shared_expert_tp_size != 0:
            raise ValueError(f"All rank size is {all_rank_group_size} but attn_tp_size is {attn_tp_size}, some ranks are not in tensor parallelism")
        shared_expert_tp_group_start = shared_expert_tp_size * (get_global_rank() // shared_expert_tp_size)
        self.shared_expert_tp_group = GroupCommunicator([_ for _ in range(shared_expert_tp_group_start, shared_expert_tp_group_start+shared_expert_tp_size)], driver=shared_expert_tp_group_start)

        if self.ep_size > 1:
            # non-shared expert adopts no tp under ep
            self.moe_tp_group = GroupCommunicator([get_global_rank()])