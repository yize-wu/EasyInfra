import copy
import torch
from typing import List
from enum import Enum
from ...generation.parallel.parallel_configuration import TPParallelConfig
from ...generation.parallel.parallel_state import (
    get_world_size,
    get_global_rank
)
from ...generation.parallel.communicator import GroupCommunicator

def check_validate_bound_strategy(bound_strategy: List[List[int]]):
    # check if bound strategy is a valid strategy:
    # start from 0
    # no duplication
    all_layers = []
    for hyper_layer_indices in bound_strategy:
        for hyper_layer_idx in hyper_layer_indices:
            if hyper_layer_idx in all_layers:
                raise ValueError(f"{hyper_layer_idx} is duplicated in {bound_strategy}")
            all_layers.append(hyper_layer_idx)
    all_layers.sort()
    if all_layers[0] != 0:
        raise ValueError

def modify_strategy(bound_strategy: List[List[int]], support_batch_layer = False):
    check_validate_bound_strategy(bound_strategy)
    
    world_size = get_world_size()     
    # If an hyper layer bounds layers of more than device_num, we should modify the bound strategy,
    # to make sure each hyper layer can be executed in parallel.
    new_bound_strategy = []
    for hyper_layer_indices in bound_strategy:
        if len(hyper_layer_indices) > world_size:
            if not support_batch_layer:
                # the last slice could be shorter than others
                split_strategy = [hyper_layer_indices[i:i+world_size] for i in range(0,len(hyper_layer_indices),world_size)]
            else:
                # form batch layer, while the modulo is split to other layers.
                modulo = len(hyper_layer_indices) % world_size
                # we are pretty sure that modulo > 0, but check
                if modulo > 0:
                    split_strategy = [hyper_layer_indices[:-modulo], hyper_layer_indices[-modulo:]]
                else:
                    split_strategy = [hyper_layer_indices]
            print(f"layer {hyper_layer_indices} is too big to fit, while world size is {world_size}. We will split it to {split_strategy}.")
            new_bound_strategy.extend(split_strategy)
        else:
            new_bound_strategy.append(hyper_layer_indices)
    bound_strategy = new_bound_strategy
                
    print(f"final bound strategy: {bound_strategy}")
    print(f"world size: {world_size}")
    
    return bound_strategy            

def choose_layer_participants(layer_parallel_size: int):
    # choose which process to participate in this layer
    if layer_parallel_size > get_world_size():# this shoule never happen
        raise ValueError
    return [_ for _ in range(layer_parallel_size)]

class LayerParallelPolicy(Enum):
    ATTN_ONLY = 0,
    INDIRECT_MLP = 1,
    DIRECT_MLP = 2
_DEFAULT_LAYER_PARALLEL_POLICY = LayerParallelPolicy.ATTN_ONLY

class LayerOutputPolicy(Enum):
    ALL_REDUCE = 0,
    NO_ACT = 1,

class Layer2LayerPolicy(Enum):
    NO_ACT = 0,
    BROADCAST = 1,

class HyperDraftModelParallelConfig(TPParallelConfig):
    def __init__(
        self, 
        draft_all_ranks, 
        bound_strategy, 
        lp_groups, 
        lp_policies,
        # layer_output_policies, 
        l2l_groups, 
        l2l_policies, 
        lm_head_ranks, 
        driver=None,
        device=None, 
    ):
        super().__init__(draft_all_ranks, driver=driver, device=device)
        if device is None:
            self.device = torch.cuda.current_device()
        
        self.in_participant = get_global_rank() in self.all_ranks
        
        self.bound_strategy = bound_strategy
        
        self.lp_groups: List[GroupCommunicator] = lp_groups
        self.lp_policies: List[LayerParallelPolicy] = lp_policies
        # self.layer_output_policies: List[LayerOutputPolicy] = layer_output_policies
        self.l2l_groups: List[GroupCommunicator] = l2l_groups
        self.l2l_policies: List[Layer2LayerPolicy] = l2l_policies

        self.need_lm_head: List[int] = (get_global_rank() in lm_head_ranks)
        # if the last l2l group has the same members as full-view, then no need to broadcast input_ids
        self.need_broadcast_inputs = (self.l2l_groups[-1].all_ranks != self.all_ranks)
        
    """
        Example:
        bound_strategy: [[0],[1],[2],[3],[4,5],[6,7],[8,9],[10,11],[12,13],[14,15],[16,17],[18,19],
            [20,21],[22,23],[24,25],[26,27],[28],[29],[30],[31]]
    """
    @classmethod
    def make_parallel_config(cls, bound_strategy, lp_policy=None, base_model_driver=None):
        world_size = get_world_size()
        if base_model_driver is None:
            base_model_driver = 0 # TODO
        
        old_bound_strategy = copy.deepcopy(bound_strategy)    
        bound_strategy = modify_strategy(bound_strategy, support_batch_layer=False)
        if old_bound_strategy != bound_strategy and lp_policy is not None:
            raise ValueError(f"modify boundary but policies are specific")
        elif lp_policy is None:
            lp_policy = _DEFAULT_LAYER_PARALLEL_POLICY

        lp_policies = [lp_policy for _ in range(len(bound_strategy))]
        
        # Note: all processes must go through new_group construction, even if some are not in this group
        lp_groups = []
        draft_all_ranks = []
        # for each hyper layer, infer an execution strategy    
        for hyper_layer_idx in range(len(bound_strategy)):
            layer_parallel_size = len(bound_strategy[hyper_layer_idx])
            # if layer_parallel_size > world_size, then it is a batch layer, otherwise it is not
            if layer_parallel_size <= world_size:                
                # establish groups
                this_layer_participants = choose_layer_participants(layer_parallel_size)
                lp_groups.append(GroupCommunicator(this_layer_participants))
                draft_all_ranks = sorted(list(set(draft_all_ranks) | set(this_layer_participants)))
            else:
                raise NotImplementedError
                # batched layers need to be executed on the same device
                    
        # handle layer-to-layer group
        # each group is for the end of this layer, e.g. group[0] is for layer0 -> layer1
        # group[-1] is for layer-1 -> logit head
        l2l_groups = []
        l2l_policies = []
        for hyper_layer_idx, lp_group in enumerate(lp_groups):
            this_layer_driver = get_layer_driver(lp_groups, hyper_layer_idx)
            this_layer_participants = get_layer_participants(lp_groups, hyper_layer_idx)
            
            if hyper_layer_idx == len(lp_groups) - 1:
                # handle lm_head
                lm_head_group_ranks = [base_model_driver]
                # We silumate a group communicator here, to figure out who is the driver of base model
                if this_layer_driver not in lm_head_group_ranks:
                    lm_head_group_ranks.append(this_layer_driver)
                # if the same, just all reduce; otherwise, broadcast
                if set(lm_head_group_ranks) == set(this_layer_participants):
                    comm_policy = Layer2LayerPolicy.NO_ACT
                    l2l_group = lp_group
                else:
                    comm_policy = Layer2LayerPolicy.BROADCAST
                    l2l_group = GroupCommunicator(lm_head_group_ranks)
                
            else:
                next_layer_participants = get_layer_participants(lp_groups, hyper_layer_idx+1)
                # If next layer's participant group is a subset of this layer's, no action is needed
                # Else, use broadcast
                if (set(next_layer_participants) - set(this_layer_participants)) == set():
                    comm_policy = Layer2LayerPolicy.NO_ACT
                    l2l_group = lp_group
                else:
                    comm_policy = Layer2LayerPolicy.BROADCAST
                    group_ranks = next_layer_participants[:]
                    if this_layer_driver not in group_ranks:
                        group_ranks.append(this_layer_driver)
                    l2l_group = GroupCommunicator(group_ranks)
            
            l2l_groups.append(l2l_group)        
            l2l_policies.append(comm_policy)
        
        parallel_config = cls(
            draft_all_ranks=draft_all_ranks, 
            bound_strategy=bound_strategy, 
            lp_groups=lp_groups, 
            lp_policies=lp_policies,
            l2l_groups=l2l_groups, 
            l2l_policies=l2l_policies, 
            lm_head_ranks=[base_model_driver]
        )
        return parallel_config                    

def get_layer_driver(lp_groups: List[GroupCommunicator], hyper_layer_idx: int) -> int:
    return lp_groups[hyper_layer_idx].driver

def get_layer_participants(lp_groups: HyperDraftModelParallelConfig, hyper_layer_idx: int) -> List[int]:
    return lp_groups[hyper_layer_idx].all_ranks
    