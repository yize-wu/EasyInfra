import torch
import torch.distributed as dist
from torch.distributed import ReduceOp as TorchReduceOp
from .parallel_utils import (
    get_global_rank, 
    get_group_rank,
    get_device,
    get_local_world_size,
    get_world_size,
)
from ...utils.stats import show_rank_print

from enum import Enum

import time
from typing import Optional, List

from easyinfra import envs

from easyinfra.generation.utils.version import is_version_greater_or_equal

if not is_version_greater_or_equal("torch", "2.6.0"):
    TORCH_DIST_ALL_GATHER_SINGLE_API = dist.all_gather_into_tensor
else:
    TORCH_DIST_ALL_GATHER_SINGLE_API = dist.all_gather_single

class ReduceOp(Enum):
    SUM = "sum"
    AVG = "avg"
    MIN = "min"

class _TorchGroupCommunicator:
    _comm_implementation = "torch"
    @staticmethod
    def get_op(op: ReduceOp):
        if op == ReduceOp.AVG:
            _op = TorchReduceOp.AVG
        elif op == ReduceOp.SUM:
            _op = TorchReduceOp.SUM
        elif op == ReduceOp.MIN:
            _op = TorchReduceOp.MIN
        else:
            _op = TorchReduceOp.SUM
        return _op

class DummyGroupCommunicator(_TorchGroupCommunicator):
    def __init__(self, *args, **kwargs):
        self.group_size = 0
        self.all_ranks = []
        self.in_participant = False
        self.driver = None
        self.group = None
        self.local_rank = 1 # to raise exception if used

class WorkHandler:
    def __init__(self, output_obj, dist_work = None):
        self.output = output_obj
        self.handler = dist_work
    def wait(self):
        if self.handler is not None:
            return self.handler.wait()

class GroupCommunicator(_TorchGroupCommunicator):
    
    def __init__(self, group_ranks: List[int], driver=None, warmup=True):
        '''
            For 1-rank group, local_rank = 0
        '''
        
        if self._comm_implementation != "torch":
            raise NotImplementedError
        if not isinstance(group_ranks, list):
            raise ValueError
        if len(group_ranks) == 0:
            raise ValueError("Cannot create a group of no rank.")
        
        self.group_size = len(group_ranks)
        self.all_ranks = sorted(group_ranks) # guaranteed to be ordered
        # self.rank = get_global_rank()
        # self.in_participant = self.rank in self.all_ranks
        self.in_participant = get_global_rank() in self.all_ranks
        
        if driver is not None:
            if driver not in self.all_ranks:
                raise ValueError(f"all ranks are {self.all_ranks}, while driver is specific {driver} and not in.")
        else:
            # smallest rank as driver
            driver = self.all_ranks[0]
        self.driver = driver
        
        
        self.record_comm_time = envs.PRINT_COMM_TIME
        
        self.group = dist.new_group(self.all_ranks, use_local_synchronization=True)
        if self.group is None:
            if self.in_participant is True:
                # 1-process group
                print("You create a group communicator, but size is 1")
                # raise ValueError("You create a group communicator, but size is 1")
                self.local_rank = 0
            else:
                self.local_rank = self.group_size # to raise exception if used
        else:
            # warmup with one comm to avoid first-time latency of nccl
            if warmup:
                self.warmup()
            self.local_rank = get_group_rank(self.group)
        
    
    def warmup(self):

        if self.group is None:
            raise ValueError

        # object
        obj = None
        obj_list = [obj]
        self.broadcast_object_list(obj_list, src=self.driver)
        # all to all
        send_t = torch.tensor([0.0]*self.group_size, dtype=torch.bfloat16, device=get_device())
        recv_t = torch.empty_like(send_t)
        self.all_to_all_single(recv_t, send_t)
        # self.all_reduce(recv_t)
        # show_rank_print(f"times: {time1-time0}, {time2-time1},")
    
    def ignore_comm_op_or_raise(self):
        if self.group is None:
            return True
            if self.in_participant is True:
                # it is a one-process group
                return True
            else:
                raise ValueError()
        else:
            return False
    
    def is_driver(self):
        return get_global_rank() == self.driver
    
    def all_reduce(self, t: torch.Tensor, op: Optional[ReduceOp] = None, async_op: bool = False):
        # show_rank_print(f"launch all reduce")
        # time1 = time.time()
        if self.ignore_comm_op_or_raise() or self.group_size <= 1:
            dist_work = None
        else:
            dist_work = dist.all_reduce(t, self.get_op(op), self.group, async_op=async_op)
        # show_rank_print(f"{time.time() - time1} {async_op}")
        return WorkHandler(t, dist_work)
    
    def broadcast(self, t: torch.Tensor, src: Optional[int] = None, async_op: bool = False):
        if self.ignore_comm_op_or_raise():
            return WorkHandler(None)
        else:
            if src is None:
                src = self.driver
            return WorkHandler(None, dist.broadcast(t, src=src, group=self.group, async_op=async_op))
    
    def broadcast_object_list(self, obj_list, src:int=None, async_op:bool=False):
        if self.ignore_comm_op_or_raise():
            return
        if not isinstance(obj_list, list):
            raise ValueError
        if src is None:
            src = self.driver
        dist.broadcast_object_list(obj_list, src=src, group=self.group)
    
    def all_gather_into_tensor(self, tensor_out:torch.Tensor, tensor_in:torch.Tensor, dim: int = 0, async_op:bool=False):
        if self.ignore_comm_op_or_raise() or self.group_size <= 1:
            tensor_out = tensor_in.unsqueeze(dim)
            work = WorkHandler(tensor_out)
            # tensor_out.record_stream(torch.cuda.current_stream())
            return work
        else:
            if self.record_comm_time:
                self.barrier()
                torch.cuda.synchronize()
                time0 = time.time()
            work = WorkHandler(tensor_out, TORCH_DIST_ALL_GATHER_SINGLE_API(tensor_out, tensor_in, self.group, async_op=async_op))
            if self.record_comm_time:
                torch.cuda.synchronize()
                time1 = time.time()
                show_rank_print(f"all gather into tensor time: {time1-time0}, out shape: {tensor_out.shape}", 0)
        return work
                
    def barrier(self, async_op:bool=False):
        if self.ignore_comm_op_or_raise():
            return WorkHandler(None)
        return WorkHandler(None, dist.barrier(self.group, async_op=async_op))
    
    def all_gather_into_tensor_auto(self, tensor_in:torch.Tensor, dim: int = 0, async_op:bool=False):
        # in and out could be the same
        if self.ignore_comm_op_or_raise() or self.group_size <= 1:
            tensor_out = tensor_in.unsqueeze(dim)
            work = WorkHandler(tensor_out)
        else:
            if self.record_comm_time:
                self.barrier()
                torch.cuda.synchronize()
                time0 = time.time()
            # create output tensor
            tensor_out_shape = list(tensor_in.shape)
            tensor_out_shape.insert(dim, self.group_size)
            tensor_out = tensor_in.new_empty(tensor_out_shape)
            # do communication
            work = WorkHandler(tensor_out, TORCH_DIST_ALL_GATHER_SINGLE_API(tensor_out, tensor_in, self.group, async_op=async_op))
            if self.record_comm_time:
                torch.cuda.synchronize()
                time1 = time.time()
                show_rank_print(f"all gather into tensor time: {time1-time0}, out shape: {tensor_out.shape}", 0)
        
        # tensor_out.record_stream(torch.cuda.current_stream()) ## tensor_out is created rather than input
        return work
    
    def all_gather(self, tensor_list: List[torch.Tensor], t:torch.Tensor, async_op:bool=False):
        if self.ignore_comm_op_or_raise():
            return
        dist.all_gather(tensor_list, t, group=self.group, async_op=async_op)
        
    def reduce_scatter_tensor(self, tensor_out: torch.Tensor, tensor_in: torch.Tensor, op: ReduceOp = None, async_op: bool = False):
        if self.ignore_comm_op_or_raise():
            return
        dist.reduce_scatter_tensor(tensor_out, tensor_in, self.get_op(op), self.group, async_op=async_op)

    def reduce_scatter_tensor_auto(self, tensor_in: torch.Tensor, dim: int = 0, op: ReduceOp = None, async_op: bool = False) -> WorkHandler:
        tensor_out_shape = list(tensor_in.shape)
        tensor_out_shape.pop(dim)
        tensor_out = torch.empty(tensor_out_shape, dtype=tensor_in.dtype, device=tensor_in.device)
        if self.group is None:
            if self.in_participant:
                tensor_out = tensor_in.squeeze(dim)
            return WorkHandler(tensor_out) # empty tensor
        else:
            # do
            if self.record_comm_time:
                self.barrier()
                torch.cuda.synchronize()
                time0 = time.time()
            work = WorkHandler(tensor_out, dist.reduce_scatter_tensor(tensor_out, tensor_in, self.get_op(op), self.group, async_op=async_op))
            if self.record_comm_time:
                torch.cuda.synchronize()
                time1 = time.time()
                show_rank_print(f"reduce scatter tensor time: {time1-time0}", 0)
            return work
    
    def all_to_all(self, 
                   output_tensor_list: List[torch.Tensor],
                   input_tensor_list: List[torch.Tensor],
                   output_split_sizes: Optional[List[int]] = None, 
                   input_split_sizes: Optional[List[int]] = None, 
                   async_op: bool = False,
    ):
        if self.ignore_comm_op_or_raise():
            return WorkHandler(input_tensor_list)
        # check the input format
        if output_split_sizes is not None:
            if len(output_split_sizes) == 0:
                raise ValueError(f"empty output sizes") 
            if not isinstance(output_split_sizes, list) or not isinstance(output_split_sizes[0], int):
                raise ValueError(f"output split sizes must be List[int], but {type(output_split_sizes)}")
        if input_split_sizes is not None:
            if len(input_split_sizes) == 0:
                raise ValueError(f"empty output sizes") 
            if not isinstance(input_split_sizes, list) or not isinstance(input_split_sizes[0], int):
                raise ValueError(f"output split sizes must be List[int], but {type(output_split_sizes)}")
        
        work = WorkHandler(output_tensor_list, dist.all_to_all(output_tensor_list, input_tensor_list, self.group, async_op))
        return work
        
    def all_to_all_single(self, tensor_out: torch.Tensor, tensor_in: torch.Tensor, output_split_sizes: Optional[List[int]] = None, input_split_sizes: Optional[List[int]] = None, async_op: bool = False):
        if self.ignore_comm_op_or_raise() or self.group_size <= 1:
            return WorkHandler(tensor_in)
        # check the input format
        if output_split_sizes is not None:
            if len(output_split_sizes) == 0:
                raise ValueError(f"empty output sizes") 
            if not isinstance(output_split_sizes, list) or not isinstance(output_split_sizes[0], int):
                raise ValueError(f"output split sizes must be List[int], but {type(output_split_sizes)} with 1st element {type(output_split_sizes[0])}, value is {output_split_sizes}")
        if input_split_sizes is not None:
            if len(input_split_sizes) == 0:
                raise ValueError(f"empty output sizes") 
            if not isinstance(input_split_sizes, list) or not isinstance(input_split_sizes[0], int):
                raise ValueError(f"output split sizes must be List[int], but type is {type(input_split_sizes)} with 1st element {type(input_split_sizes[0])}, value is {input_split_sizes}")
        # do
        if self.record_comm_time:
            self.barrier()
            torch.cuda.synchronize()
            time0 = time.time()
        work = WorkHandler(tensor_out, dist.all_to_all_single(tensor_out, tensor_in, output_split_sizes, input_split_sizes, self.group, async_op=async_op))
        if self.record_comm_time:
            torch.cuda.synchronize()
            time1 = time.time()
            show_rank_print(f"all2all time: {time1-time0}", 0)
        return work
        
class Layer2LayerGroupCommunicator(GroupCommunicator):
    def __init__(self, dist_group, driver):
        super().__init__(dist_group)
        # Layer2Layer driver should be specified
        self.driver = driver
    
class HierarchicalAll2AllGroupCommunicator(GroupCommunicator):
    def __init__(self, group_ranks: List[int], structure: List[int] = [], driver=None):
        super().__init__(group_ranks, driver)
        ## Check structure.
        ## structure is [2,2,2], group_size = 8
        for depth_i, subgroup_size in enumerate(structure):
            if depth_i > 0 and subgroup_size % structure[depth_i-1] != 0:
                raise ValueError("Hierarchical All2All requires divided subgroup sizes, "
                                 f"but depth-{depth_i-1}-to-{depth_i} are {structure[depth_i-1]} and {subgroup_size}.")
            if depth_i == len(structure-1) and subgroup_size % self.group_size != 0:
                raise ValueError(f"Hierarchical All2All requires divided subgroup sizes, "
                                 f"but final depth-{depth_i} is {subgroup_size} and whole-group size is {self.group_size}.")
        ## construct sub-groups
        self.structure = structure
        self.subgroups = []
        for depth_i, subgroup_size in enumerate(structure):
            next_subgroup_size = structure[depth_i+1] if depth_i < len(structure)-1 else self.group_size
            this_comm_size = next_subgroup_size // subgroup_size
            this_comm_ranks = (self.local_rank % subgroup_size) + [_*subgroup_size for _ in range(this_comm_size)]
            assert get_group_rank(self) in this_comm_ranks
            this_comm_group = GroupCommunicator(this_comm_ranks)
            self.subgroups.append(this_comm_group)
    
    def all_to_all_single(self, tensor_out: torch.Tensor, tensor_in: torch.Tensor, output_split_sizes: Optional[List[int]] = None, input_split_sizes: Optional[List[int]] = None, async_op: bool = False):
        if len(self.structure) == 0:
            return super().all_to_all_single(tensor_out, tensor_in, output_split_sizes, input_split_sizes, async_op)
        
        if self.ignore_comm_op_or_raise():
            return WorkHandler(tensor_in)
        # check the input format
        if output_split_sizes is not None:
            if len(output_split_sizes) == 0:
                raise ValueError(f"empty output sizes") 
            if not isinstance(output_split_sizes, list) or not isinstance(output_split_sizes[0], int):
                raise ValueError(f"output split sizes must be List[int], but {type(output_split_sizes)}")
        if input_split_sizes is not None:
            if len(input_split_sizes) == 0:
                raise ValueError(f"empty output sizes") 
            if not isinstance(input_split_sizes, list) or not isinstance(input_split_sizes[0], int):
                raise ValueError(f"output split sizes must be List[int], but {type(output_split_sizes)}")
        
        # Structured
        for subgroup in self.subgroups:
            # inside-group distribution
            pass
            # intra-group comm
