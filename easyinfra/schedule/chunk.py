from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .request import RequestHub
    from easyinfra.generation.cache_utils import Cache

import torch
import itertools
from enum import Enum

from typing import Optional, List, Dict, Tuple

from easyinfra.generation.parallel.communicator import GroupCommunicator, ReduceOp
from easyinfra.generation.parallel.parallel_utils import get_device
from easyinfra.generation.functions import exclusive_cumsum, cumsum
from easyinfra.generation.blocks.stage_name import MoeStageName

from easyinfra.utils.stats import show_rank_print, event_rank_print
import time

from easyinfra import envs
from easyinfra.generation.utils.compute_utils import _add_to_residual
from easyinfra.utils.tensors import _to_cpu_list
from .stream import get_current_stream, create_overlap_stream

class ChunkRunType(Enum):
    ALL = 0
    COMPUTE = 0
    COMM = 0
    
# Use current stream for all computations
_COMPUTE_CUDA_STREAM = None
def _init_compute_stream():
    global _COMPUTE_CUDA_STREAM
    if _COMPUTE_CUDA_STREAM is None:
        _COMPUTE_CUDA_STREAM = get_current_stream()
    return _COMPUTE_CUDA_STREAM
# Use a single stream for all communications except expert data
_COMM_CUDA_STREAM = None
def _init_comm_stream():
    global _COMM_CUDA_STREAM
    if _COMM_CUDA_STREAM is None:
        _COMM_CUDA_STREAM = create_overlap_stream()
    return _COMM_CUDA_STREAM
# Use a stream for expert-data communications
_EXPERT_DATA_COMM_CUDA_STREAM = None
def _init_expert_data_comm_stream():
    global _EXPERT_DATA_COMM_CUDA_STREAM
    if _EXPERT_DATA_COMM_CUDA_STREAM is None:
        _EXPERT_DATA_COMM_CUDA_STREAM = create_overlap_stream()
    return _EXPERT_DATA_COMM_CUDA_STREAM


def _None_default(variable, default_value):
    if variable is None:
        return default_value
    else:
        return variable

class ChunkBlock:
    support_multi_stage_forward = True       
     
    def __init__(
        self,
        chunk_id: int, 
        request_ids: List[int], 
        seqlens_q: List[int],
        cu_seqlens_q: List[int],
        input_ids: torch.Tensor, 
        pos_ids: torch.Tensor, 
        attention_masks: Tuple[torch.Tensor],
        chunk_kv_caches: Tuple[Cache],
        chunk_inference_logits_to_keep_indices_list,
        stages,
        stage_has_communication,
        producer_chunk: "ChunkBlock",
        extra_cuda_event_num: int = 1,
    ):
        self.chunk_id = chunk_id
        self.request_ids = request_ids
        self.seqlens_q = seqlens_q
        self.cu_seqlens_q = cu_seqlens_q
        self.inference_logits_to_keep_indices = chunk_inference_logits_to_keep_indices_list
        self.input_ids = input_ids
        self.position_ids = pos_ids
        self.attention_masks = attention_masks
        self.kv_caches = chunk_kv_caches
        self.producer_chunk = producer_chunk
        
        self.seqlens_k = [cache.next_step_expected_length for cache in self.kv_caches]
        self.cu_seqlens_k = [0] + cumsum(self.seqlens_k)
        self.max_seqlen_q = max(self.seqlens_q)
        self.max_seqlen_k = max(self.seqlens_k)
        # show_rank_print(f"chunk{self.chunk_id}: cu: q{self.cu_seqlens_q} k{self.cu_seqlens_k}")
        
        # run-time usage        
        self.active_variables = ()
        self.position_embeddings = None
        
        # Routing stats
        self.require_global_device_count_on_cpu: bool = None
        self.global_tok_device_count: torch.Tensor = None # 1D for device
        self.global_tok_device_count_cpu: torch.Tensor = None # 1D for device
        self.global_tok_local_ep_rank_count: torch.Tensor = None
        
        ## For MoE Kernel
        self.global_tok_all_expert_count_2d: torch.Tensor = None # [ep_rank, device_id, expert_id]
        self.global_tok_all_expert_count_2d_cpu: torch.Tensor = None # [ep_rank, device_id, expert_id]
        self.global_tok_local_expert_count_2d_flat_list: List[int] = None
        self.send_token_split: List[int] = None
        self.recv_token_split: List[int] = None
        
        self.num_finished_pool_steps = 0
        self.num_finished_compute_steps = 0
        self.num_finished_comm_steps = 0
        self.compute_cuda_stream = _init_compute_stream()
        self.comm_cuda_stream = _init_comm_stream()
        # self.expert_data_comm_cuda_stream = _init_expert_data_comm_stream()
        # self.is_separated_streams_expert_data_and_dispatch = (self.comm_cuda_stream != self.expert_data_comm_cuda_stream)
        self.compute_event = torch.cuda.Event()
        self.comm_event = torch.cuda.Event()
        self.comm_handler: torch.distributed.Work = None
        self.compute_wait_comm = False # the 1st compute waits no communication
        self.record_compute = True
        self.extra_cuda_events = [torch.cuda.Event() for _ in range(extra_cuda_event_num)]
        
        self.num_deferred_steps = 0
        self.peak_device: int = None
        self.util: float = None
        self.max_min_gap: float = None
        
        # functions for stages
        self.stages = stages
        self.stage_has_communication = stage_has_communication
        self.num_stages = len(self.stages)
        if self.num_stages != len(stage_has_communication):
            raise ValueError
        self.this_stage_has_comm = True
        self.last_stage_has_comm = True
        self.has_skipped_valid_comm = False

        # whether this chunk will be scheduled next (a prediction)
        self.will_be_scheduled: bool = None
        
        # output
        self.final_output_at = 0
        
        
        # stats
        self.comm_time = 0.0
        self.global_device_history: List[torch.Tensor] = []
        self.global_expert_history: List[torch.Tensor] = []
            
    def is_ready(
        self,
        wait_producer: bool,
    ):
        # local_ready = 1
        # if self.comm_event is not None:
        #     # if self.comm_event.query() is False:
        #     if self.comm_handler is None:
        #         assert self.num_finished_pool_steps == -1
        #     if self.comm_handler is not None and self.comm_handler.is_completed() is False:
        #         # not finished
        #         local_ready = 0
        #     elif wait_producer and (self.producer_chunk is not None) and not self.producer_chunk.num_finished_pool_steps > self.num_finished_pool_steps:
        #         local_ready = 0
        
        local_ready = 1
        if wait_producer and (self.producer_chunk is not None) and not self.producer_chunk.num_finished_compute_steps > self.num_finished_compute_steps:
            local_ready = 0
        return local_ready == 1
    
    # def is_global_ready(
    #     self,
    #     wait_producer: bool,
    # ):
    #     local_ready = self.is_ready(wait_producer=wait_producer)
    #     ready = torch.tensor([local_ready], device=get_device())
    #     # use this to ensure global consistency
    #     if self.comm_group is not None:
    #         self.comm_group.all_reduce(ready, op=ReduceOp.MIN)
    #     return ready.item() == 1
    
    def is_finished(self):
        return self.is_compute_finished() and self.is_comm_finished()
        # if self.num_finished_pool_steps > self.num_stages:
        #     raise ValueError(f"chunk {self.chunk_id} exceed layer num")
        # return self.num_finished_pool_steps == self.num_stages
    def is_compute_finished(self):
        if self.num_finished_compute_steps > self.num_stages:
            raise ValueError(f"chunk {self.chunk_id} exceed layer num")
        return self.num_finished_compute_steps == self.num_stages
    def is_comm_finished(self):
        if self.num_finished_comm_steps > self.num_stages:
            raise ValueError(f"chunk {self.chunk_id} exceed layer num")
        return self.num_finished_comm_steps == self.num_stages
    # def is_last_step(self):
    #     assert self.num_finished_pool_steps < self.num_stages
    #     return (self.num_stages - self.num_finished_pool_steps) == 1
    def is_first_step(self):
        assert self.num_finished_compute_steps >= 0
        return self.num_finished_compute_steps == 0
    
    def run_compute(self):
        stage_name, compute_func, _ = self.stages[self.num_finished_compute_steps]
        this_stage_has_comm = self.stage_has_communication[self.num_finished_compute_steps]
        last_stage_has_comm = False if self.num_finished_compute_steps == 0 else self.stage_has_communication[self.num_finished_compute_steps-1]
        
        # show_rank_print(f"query in run_compute(): {self.compute_event.query()}", 0)
        with torch.cuda.stream(self.compute_cuda_stream):        
            if self.compute_wait_comm is True:
                # self.compute_cuda_stream.wait_event(self.comm_event)
                if last_stage_has_comm or self.has_skipped_valid_comm:
                    event_rank_print(f"chunk{self.chunk_id} wait comm.", 0)
                    self.compute_cuda_stream.wait_event(self.comm_event)
                    self.has_skipped_valid_comm = False
                # Else, Do not wait event, as it has been waited before
            else:
                if last_stage_has_comm:
                    # you have skipped this comm, but it should be waited next
                    self.has_skipped_valid_comm = True
                    # self.compute_cuda_stream.wait_event(self.comm_event)
                    
            compute_func(self)
            
            # The Event may not be recorded, due to fast forward
            # If DEBUG, remember to record it
            
            # self.compute_event.record(self.compute_cuda_stream)
            # time0 = time.time()
            # self.compute_event.synchronize()
            if (self.record_compute and 
                (not self.support_multi_stage_forward or this_stage_has_comm)
            ):
                # self.compute_event.record(self.compute_cuda_stream)
                event_rank_print(f"chunk{self.chunk_id} record compute.", 0)
                self.compute_event.record()
        # show_rank_print(f"query in run_compute2(): {self.compute_event.query()}", 0)
        
        # Default it to True, change it in the compute func if no wait for last communication
        self.record_compute = True
        self.compute_wait_comm = True 
        # Increase step
        self.num_finished_compute_steps += 1
        
    def run_communicate(self):
        stage_name, _, communicate_func = self.stages[self.num_finished_comm_steps]
        this_stage_has_comm = self.stage_has_communication[self.num_finished_comm_steps]
        
        ## Expert data should use a different stream
        comm_cuda_stream = self.comm_cuda_stream
        with torch.cuda.stream(comm_cuda_stream):
            # must wait for the last compute. If no need, then the communication should be put in front of this computation
            # self.comm_cuda_stream.wait_event(self.compute_event)
            if this_stage_has_comm:
                event_rank_print(f"chunk{self.chunk_id} wait compute.", 0)
                comm_cuda_stream.wait_event(self.compute_event)
            ### If there is CPU sync in routing, no need to wait event here
            # if stage_name == MoeStageName.DISPATCH and self.is_separated_streams_expert_data_and_dispatch:
            #     comm_cuda_stream.wait_event(self.comm_event)
            communicate_func(self) # can change self.compute_wait_comm
            # self.comm_event.record(self.comm_cuda_stream)
            if this_stage_has_comm:
                # self.comm_event.record(self.comm_cuda_stream)
                event_rank_print(f"chunk{self.chunk_id} record comm.", 0)
                self.comm_event.record()
                
        # Increase step
        self.num_finished_comm_steps += 1
        
    def run(
        self,
        force_disable_multi_stage_forward: Optional[bool] = None,
        get_over_next_comm: Optional[bool] = None,
        defer_next_valid_comm: Optional[bool] = None,
        run_last_communication: Optional[bool] = None,
    ):
        '''
            force_multi_stage_forward: run until
        '''
        
        # No computation or synchronization should occur
        _is_finished = self.is_finished()
        if _is_finished:
            ## It is possible that fast-forward has reached the finish line
            if (self.num_finished_compute_steps == self.num_finished_pool_steps 
                and self.num_finished_comm_steps == self.num_finished_pool_steps):
                raise ValueError
        elif (self.num_finished_compute_steps < self.num_finished_pool_steps
              or self.num_finished_comm_steps + 1 < self.num_finished_pool_steps):
            raise ValueError
        # show_rank_print(f"enter run: chunk {self.chunk_id} num finished steps: {self.num_finished_pool_steps}", 0)
        
        # TODO: better cpu overhead: avoid extra loop in while, if steps allow
        
        ### For fast-forward, just reduce the number is enough
        # if self.num_finished_compute_steps == self.num_finished_pool_steps:
        #     self.num_past_fast_forward_stages -= 1
        #     return
        
        assert force_disable_multi_stage_forward is None or isinstance(force_disable_multi_stage_forward, bool)
        assert get_over_next_comm is None or isinstance(get_over_next_comm, bool)
        assert defer_next_valid_comm is None or isinstance(defer_next_valid_comm, bool)
        force_disable_multi_stage_forward = _None_default(force_disable_multi_stage_forward, False)
        get_over_next_comm = _None_default(get_over_next_comm, False)
        defer_next_valid_comm = _None_default(defer_next_valid_comm, False)
        run_last_communication = _None_default(run_last_communication, False)
        
        has_compute_to_run = self.num_finished_compute_steps <= self.num_finished_pool_steps
        has_comm_to_run = (self.num_finished_comm_steps <= self.num_finished_pool_steps) ### equivatent to == or -1 ==
        
        stop_because_comm = False
        
        # if self.chunk_id == 0:
        #     show_rank_print(f"{self.num_finished_compute_steps}, {self.num_finished_comm_steps}, {self.num_finished_pool_steps}, {get_over_next_comm}", 0)
        #     show_rank_print(f"{force_disable_multi_stage_forward}, {defer_next_valid_comm}", 0)
        #     show_rank_print(f"compute? {has_compute_to_run}, comm? {has_comm_to_run}", 0)
        ### If the stage does not need communication, try to skip it for longer computation (and therefore better overlapping)
        while (not _is_finished 
               and not stop_because_comm):
            ## last stage has comm: just passed it
            ## this stage has comm: defered

            ### Check if last comm is deferred, and run it if so
            if run_last_communication and self.num_finished_compute_steps > self.num_finished_comm_steps:
                if self.num_finished_compute_steps - self.num_finished_comm_steps != 1: # must only defer 1 comm
                    raise ValueError
                # show_rank_print(f"compute step: {self.num_finished_compute_steps}, comm step: {self.num_finished_comm_steps}", 0)
                self.run_communicate()

            if has_compute_to_run:
                if self.num_finished_compute_steps > self.num_finished_comm_steps:
                    raise ValueError
                self.run_compute()
                
            #### Communicate
            ### Can defer comm to LAST COMPUTE
            ### Run comm only when has run compute. do not run comm single. if not including 'has_compute_to_run', could single run
            ### single run breaks comm defer
            this_stage_has_comm = self.stage_has_communication[self.num_finished_comm_steps]
            did_defer_comm = (defer_next_valid_comm and this_stage_has_comm)
            if has_compute_to_run and has_comm_to_run and not did_defer_comm:
                self.run_communicate()
            last_stage_has_comm = self.stage_has_communication[self.num_finished_comm_steps-1]
            
            if force_disable_multi_stage_forward or not self.support_multi_stage_forward:
                break
            elif did_defer_comm:
                ## This stage comm is defered but valid, so break
                break
            
            _is_finished = self.is_finished()
            ## If last_stage_has_comm is True, either has run a comm, or has go over a comm but not run (impossible)
            stop_because_comm = last_stage_has_comm or this_stage_has_comm
            ### get over, but do not get over agin
            if last_stage_has_comm and get_over_next_comm:
                get_over_next_comm = False
                stop_because_comm = False
            
        self.num_finished_pool_steps += 1
    
    def finalize(self):
        (residual, hidden_states) = self.active_variables
        with torch.cuda.stream(self.compute_cuda_stream):
            residual = _add_to_residual(residual, hidden_states)
        # show_rank_print(f"residual: {residual.sum()}, hidden_states: {hidden_states.sum()}", 0)
        torch.cuda.current_stream().wait_stream(self.compute_cuda_stream)
        self.active_variables = (residual,)

    def get_global_tok_device_count(self):
        has_gpu_value = not (self.global_tok_device_count is None)
        has_cpu_value = not (self.global_tok_device_count_cpu is None)
        if has_gpu_value:
            return self.global_tok_device_count
        elif has_cpu_value:
            return self.global_tok_device_count_cpu
        raise ValueError(f"Not having both gpu and cpu global_tok_device_count")



class ChunkBlockList:
    def __init__(self,
                 wrapped_model,
                 max_chunk_size: int,
                 enable_seq_split: bool = False,
                 min_split_length: Optional[int] = None,
    ):
        self.max_chunk_size = max_chunk_size
        self.enable_seq_split = enable_seq_split
        if self.enable_seq_split:
            if min_split_length is None:
                raise ValueError(f"Must have min_split_length with seq split")
            elif min_split_length > self.max_chunk_size:
                raise ValueError(f"a split must fit in a chunk, but min_split_length vs max_chunk_size: {min_split_length}, {self.max_chunk_size}")
            elif min_split_length > self.max_chunk_size * 0.5:
                show_rank_print(f"min_split_length can cause current chunk too short: min_split{min_split_length}, max_chunk{self.max_chunk_size}")
            self.min_split_length = min_split_length

        self.chunk_list: List[ChunkBlock] = []
        # prepare stage functions for chunk scheduling
        self.stages, self.stage_has_communication = wrapped_model.prepare_chunk_schedule_stages()
        self.parallel_config = wrapped_model.parallel_config
        
        self._chunk_id_generator = itertools.count(start=0)

    def new_chunk_id(self) -> int:
        return next(self._chunk_id_generator)
        
    def add_chunk(self,
                req_ids, 
                seqlens_q,
                cu_seqlens_q,
                chunk_input_ids, 
                chunk_pos_ids, 
                chunk_attention_masks,
                chunk_kv_caches,
                chunk_inference_logits_to_keep_indices_list,
                producer_chunk: Optional[ChunkBlock] = None,
    ):
        new_chunk_id = self.new_chunk_id()
        new_chunk = ChunkBlock(
            new_chunk_id, 
            req_ids, 
            seqlens_q,
            cu_seqlens_q,
            chunk_input_ids, 
            chunk_pos_ids, 
            chunk_attention_masks,
            chunk_kv_caches,
            chunk_inference_logits_to_keep_indices_list,
            stages=self.stages,
            stage_has_communication=self.stage_has_communication,
            producer_chunk=producer_chunk,
        )
        self.chunk_list.append(new_chunk)
        return new_chunk.chunk_id
    
    def partition_input_into_chunks(
        self,
        request_hub: RequestHub,
    ):

        req_logical_ids, req_lengths = self._get_req_ids_and_lengths(request_hub)
        assert len(req_logical_ids) > 0 # must be non-empty

        # partitioning
        chunk_request_map_list: List[Dict] = [] # the i-th item is chunk i, which has ({req_id0: (offset, span_len)}, {req_id1: (offset, span_len)}, ...)
        curr_chunk_request_map, curr_chunk_offsets = {}, []
        curr_chunk_length = 0 ## curr_chunk_length: the sum length of this chunk
        def _append_chunk_metadata():
            nonlocal curr_chunk_request_map, curr_chunk_offsets, curr_chunk_length
            assert len(curr_chunk_request_map) > 0
            # add this new chunk into map
            if curr_chunk_request_map is not None:
                chunk_request_map_list.append(curr_chunk_request_map)
            curr_chunk_request_map, curr_chunk_offsets = {}, []
            curr_chunk_length = 0

        
        for (logic_req_id, this_req_len) in zip(req_logical_ids, req_lengths):
            curr_req_offset = 0 ## curr_req_offset: the offset of this request

            while curr_req_offset < this_req_len:
                req_remaining_length = this_req_len - curr_req_offset
                chunk_remaining_length = self.max_chunk_size - curr_chunk_length

                if self.enable_seq_split:
                    put_in_size = min(chunk_remaining_length, req_remaining_length)
                    req_after_put_in_length = req_remaining_length - put_in_size
                    if req_after_put_in_length > 0 and req_after_put_in_length < self.min_split_length:
                        # maybe we put too long here
                        put_in_size -= (self.min_split_length - req_after_put_in_length)

                    if curr_chunk_length == 0:
                        # must put in
                        do_put_in = True
                    else:
                        do_put_in = (
                            put_in_size <= chunk_remaining_length 
                            and put_in_size >= self.min_split_length
                        )
                    
                else:
                    put_in_size = req_remaining_length
                    do_put_in = req_remaining_length <= chunk_remaining_length

                if do_put_in:
                    curr_chunk_request_map.update({logic_req_id: (curr_req_offset, put_in_size)})
                    curr_chunk_offsets.append(curr_chunk_length)
                    curr_chunk_length += put_in_size
                    curr_req_offset += put_in_size
                else:
                    # this chunk is full
                    _append_chunk_metadata()
            
        ## Append the last                
        _append_chunk_metadata()
        
        # add actual chunks to the chunk_list
        for chunk_req_map in chunk_request_map_list:
            seqlens_q = [v[1] for v in chunk_req_map.values()]
            cu_seqlens_q = [0] + cumsum(seqlens_q)
            
            (
                chunk_input_ids, 
                chunk_pos_ids, 
                chunk_attention_masks, 
                chunk_kv_caches,
                chunk_inference_logits_to_keep_indices_list
            ) = request_hub.get_inputs_from_req_metadata(chunk_req_map)
            
            _chunk_id = self.add_chunk(
                chunk_req_map, 
                seqlens_q,
                cu_seqlens_q,
                chunk_input_ids, 
                chunk_pos_ids, 
                chunk_attention_masks,
                chunk_kv_caches,
                chunk_inference_logits_to_keep_indices_list,
            )
            
            show_rank_print(f"chunk{_chunk_id}, request{chunk_req_map}", 0)

    def _get_req_ids_and_lengths(self, request_hub: RequestHub):
        '''
            Return permuted (next_step_logical_ids, next_step_req_lengths)
        '''
        next_step_logical_ids = request_hub.next_step_logical_ids
        next_step_req_lengths = [request_hub.next_step_logical_ids_and_lengths[logic_req_id] for logic_req_id in next_step_logical_ids]
        assert all(req_len > 0 for req_len in next_step_req_lengths)
        if not self.enable_seq_split:
            for req_id, req_len in zip(next_step_logical_ids, next_step_req_lengths):
                if req_len > self.max_chunk_size:
                    raise ValueError(f"request{req_id} is too long for a chunk: {req_len} with chunk max {self.max_chunk_size}. "
                                        "Enable seq split with `enable_seq_split=True`")
                
        ## In what sequence Should we split the chunks?
        ## Now we use long-to-short
        sorted_indices = sorted(range(len(next_step_req_lengths)), key=lambda i: next_step_req_lengths[i])
        next_step_logical_ids = [next_step_logical_ids[i] for i in sorted_indices]
        next_step_req_lengths = [next_step_req_lengths[i] for i in sorted_indices]

        return next_step_logical_ids, next_step_req_lengths
        
    def __getitem__(self, chunk_id: int):
        return self.chunk_list[chunk_id]
    
    def get_length(self):
        return len(self.chunk_list)
    
    def slice_output_hidden_states(self):
        """
            Slice the output h of each chunk, from req_lengths.
        """
        output_hidden_states = []
        for chunk in self.chunk_list:
            logits_to_keep = chunk.inference_logits_to_keep_indices
            this_chunk_output = chunk.active_variables[chunk.final_output_at] # make sure the residual is at 0
            output_hidden_states.append(this_chunk_output[logits_to_keep, :])
        return torch.cat(output_hidden_states, dim=0)