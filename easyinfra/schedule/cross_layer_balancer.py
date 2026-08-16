from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from easyinfra.schedule.chunk import ChunkBlock
import torch
import time
from typing import List, Dict
from easyinfra.utils.tensors import _to_cpu_list
from easyinfra.generation.parallel.parallel_utils import get_device
from .stream import create_overlap_stream, get_current_stream
from enum import Enum
from .utils import ids_of_chunk_list, pop_elements_of_list
from easyinfra.utils.stats import show_rank_print

class MoeBalanceStrategyType(Enum):
    SYNC = "sync" # decide after all chunks synced
    ASYNC = "async" # decide right after each chunk arrived

# Use a single stream for all communications except expert data
_SCHEDULE_CUDA_STREAM = None
def _init_schedule_stream():
    global _SCHEDULE_CUDA_STREAM
    if _SCHEDULE_CUDA_STREAM is None:
        _SCHEDULE_CUDA_STREAM = create_overlap_stream()
    return _SCHEDULE_CUDA_STREAM

DEFAULT_HEAVIEST_DEVICE = -1
class MoeCrossLayerBalancer:
    def __init__(self,
                 num_min_run_chunks: int,
                 num_max_deferred_steps: int,
                 cross_layer_scheduling_strategy,
                 record_decide_time: bool = False,
        ):
        self.record_decide_time = record_decide_time
        self.decide_time = 0.0
        self.reverse_schedule = False
        
        self.num_chunks: int = -1
        self.num_min_run_chunks = num_min_run_chunks
        self.num_max_deferred_steps = num_max_deferred_steps
        self.enable_reactive_max_deferred_chunk = (self.num_max_deferred_steps > 0)
        self.cross_layer_scheduling_strategy = cross_layer_scheduling_strategy
        if self.cross_layer_scheduling_strategy == "random":
            self.gen = torch.Generator()
            self.gen.manual_seed(42) # for random uniform
        
        if (
            self.cross_layer_scheduling_strategy.startswith("async")
        ):
            self.strategy_type = MoeBalanceStrategyType.ASYNC
        else:
            self.strategy_type = MoeBalanceStrategyType.SYNC
        
                
        
        self.device = get_device()
        # self.device = "cpu"
        
        if self.device != "cpu":
            self.scheduling_cuda_stream = _init_schedule_stream()
        else:
            self.scheduling_cuda_stream = get_current_stream()
            
        self.chunk_device_counts = []
        self.chunk_num_steps = []
        self.ready_chunks: List[ChunkBlock] = []
        self.deferred_chunks: List[ChunkBlock] = None
        self.last_active_and_ready_chunks: List[ChunkBlock] = None
        
        self.scheduled_chunks: List[ChunkBlock] = []
        self.scheduled_chunk_device_count: torch.Tensor = None
        self.heaviest_device_dict: Dict = {}
        self.scheduled_chunk_heaviest_device: int = DEFAULT_HEAVIEST_DEVICE
        
        ## Strategy that do not ensure the max-deferred chunks are appended
        self.do_append_max_deferred_at_end = self.cross_layer_scheduling_strategy in (
            "enumerate_max_util", )
        ## Strategy that do not ensure enough number of chunks has been appended
        self.do_append_to_threshold_at_end = self.cross_layer_scheduling_strategy in (
            "enumerate_max_util", )
        
        self.num_steps = 0
    
    def prepare(self, num_chunks: int,):
        self.num_chunks = num_chunks
        if (isinstance(self.cross_layer_scheduling_strategy, str) and
            self.cross_layer_scheduling_strategy.startswith("enumerate")):
            self._prepare_mask()
    
    def prepare_each_step(
        self,
        deferred_chunks: List[ChunkBlock],
        last_active_and_ready_chunks: List[ChunkBlock],
    ):
        self
        self.deferred_chunks = deferred_chunks
        self.last_active_and_ready_chunks = last_active_and_ready_chunks
        self.num_chunks_before_anything = len(self.deferred_chunks) + len(self.last_active_and_ready_chunks)
    
    def finalize_each_step(self,):
        self.num_steps += 1
        
        ### Evict scheduled chunks from self.ready_chunks
        _pop_indices = []
        for i,chunk in enumerate(self.ready_chunks):
            if chunk in self.scheduled_chunks:
                _pop_indices.append(i)
        pop_elements_of_list(self.ready_chunks, _pop_indices)
        
        ### Update deferred steps
        for chunk in self.scheduled_chunks:
            chunk.num_deferred_steps = 0
        for chunk in self.ready_chunks:
            chunk.num_deferred_steps += 1
            
        
        self.ready_chunks = []
        self.scheduled_chunks = []
        self.scheduled_chunk_device_count = None
        self.scheduled_chunk_heaviest_device = DEFAULT_HEAVIEST_DEVICE
        self.heaviest_device_dict = {}
    
    def add_ready_chunk(self, chunk: ChunkBlock):
        self.ready_chunks.append(chunk)
    
    def async_decide_which_run(
        self,
    ):
        if self.record_decide_time:
            torch.cuda.synchronize()
            _decide_start = time.time()
            
        num_launched_chunks = len(self.scheduled_chunks)
        num_ready_chunks = len(self.ready_chunks)
        num_max_possible_future_chunks = self.num_chunks_before_anything - num_launched_chunks - num_ready_chunks
        # show_rank_print(f"now decide chunk{self.ready_chunks[-1].chunk_id}")
        
        if num_max_possible_future_chunks == 0:
            ## This step at least launch one
            num_this_step_min_run_chunks = self.num_min_run_chunks - num_launched_chunks
        else:
            num_this_step_min_run_chunks = 0
        
        if num_this_step_min_run_chunks == num_ready_chunks:
            res = [_ for _ in range(num_ready_chunks)]
        else:
            with torch.cuda.stream(self.scheduling_cuda_stream):
                if self.cross_layer_scheduling_strategy == "async_none":
                    res = [_ for _ in range(num_ready_chunks)]
                elif self.cross_layer_scheduling_strategy == "async_max_util":
                    res = []
                    for i,ready_chunk in enumerate(self.ready_chunks[::-1]):
                        if self.scheduled_chunk_device_count is None:
                            res.append(num_ready_chunks - i - 1)
                            self.scheduled_chunk_device_count = ready_chunk.global_tok_device_count
                        else:
                            old_util = self.scheduled_chunk_device_count.sum() / self.scheduled_chunk_device_count.max()
                            new_chunk_device_count = self.scheduled_chunk_device_count + ready_chunk.global_tok_device_count
                            new_util = new_chunk_device_count.sum() / new_chunk_device_count.max()
                            if new_util.item() > old_util.item():
                                res.append(num_ready_chunks - i - 1)
                                self.scheduled_chunk_device_count = self.scheduled_chunk_device_count + ready_chunk.global_tok_device_count
                                
                    # if last step, must reach the num_min_run_chunk requirement
                    if num_max_possible_future_chunks == 0 and num_launched_chunks + len(res) < self.num_min_run_chunks:
                        # run all previous
                        for i,ready_chunk in enumerate(self.ready_chunks):
                            if i not in res:
                                res.append(i)
                            if num_launched_chunks + len(res) == self.num_min_run_chunks:
                                break
                        
                elif self.cross_layer_scheduling_strategy == "async_diff_peak":
                    res = []
                    for i,ready_chunk in enumerate(self.ready_chunks[::-1]):
                        do_activate = self.scheduled_chunk_device_count is None
                        if do_activate:
                            assert self.scheduled_chunk_heaviest_device == DEFAULT_HEAVIEST_DEVICE
                            # the 1st activate                            
                        else:
                            assert self.scheduled_chunk_heaviest_device != DEFAULT_HEAVIEST_DEVICE
                            if ready_chunk.chunk_id not in self.heaviest_device_dict:
                                self.heaviest_device_dict[ready_chunk.chunk_id] = ready_chunk.global_tok_device_count_cpu.argmax(-1).item()
                            do_activate = (self.heaviest_device_dict[ready_chunk.chunk_id] != self.scheduled_chunk_heaviest_device)
                            
                        if do_activate:
                            res.append(num_ready_chunks - i - 1)
                            if self.scheduled_chunk_device_count is None:
                                self.scheduled_chunk_device_count = ready_chunk.global_tok_device_count_cpu
                            else:
                                self.scheduled_chunk_device_count = self.scheduled_chunk_device_count + ready_chunk.global_tok_device_count_cpu
                            self.scheduled_chunk_heaviest_device = self.scheduled_chunk_device_count.argmax(-1).item()
                                
                    # if last step, must reach the num_min_run_chunk requirement
                    if num_max_possible_future_chunks == 0 and num_launched_chunks + len(res) < self.num_min_run_chunks:
                        # run all previous
                        for i,ready_chunk in enumerate(self.ready_chunks):
                            if i not in res:
                                res.append(i)
                            if num_launched_chunks + len(res) == self.num_min_run_chunks:
                                break
                elif self.cross_layer_scheduling_strategy == "async_group":
                    if self.num_steps == 0:
                        ## The first step must come from inner loop schedule
                        res = [0]
                            
                        if num_max_possible_future_chunks == 0:
                            ## The last chunk
                            all_chunks = self.scheduled_chunks + self.ready_chunks
                            num_all_chunks = len(all_chunks)
                            self.num_all_chunks = num_all_chunks
                            all_chunks_heaviest_device = [chunk.global_tok_device_count_cpu.argmax(-1).item() for chunk in all_chunks]
                            ## Divide into groups
                            self.groups = [[c for c in all_chunks[:(num_all_chunks // 2)]], [c for c in all_chunks[(num_all_chunks // 2):]]]
                            self.start = True
                    else:
                        if self.start:
                            if self.num_chunks_before_anything - num_max_possible_future_chunks <= self.num_all_chunks // 2:
                                res = [0]
                            else:
                                res = []
                            if num_max_possible_future_chunks == 0:
                                self.start = False
                        else:
                            if num_launched_chunks == 0:
                                res = [_ for _ in range(num_ready_chunks)]
                            else:
                                res = [0]
                else:
                    raise ValueError(f"unsupported cross-layer strategy: {self.cross_layer_scheduling_strategy}")        

        # sorted
        res = sorted(res)
        active_chunks = [self.ready_chunks[run_chunk_i] for run_chunk_i in res]
        pop_elements_of_list(self.ready_chunks, res)
        self.scheduled_chunks += active_chunks
        # show_rank_print(f"scheduled chunks: {ids_of_chunk_list(self.scheduled_chunks)}")        
        
        if self.record_decide_time:
            torch.cuda.synchronize()
            _decide_end = time.time()
            self.decide_time += _decide_end - _decide_start
        
        return active_chunks
                
    def sync_decide_which_run(
        self, 
    ):
        '''
            Returns: scheduled indices of next-step chunks.
            The operations can be done on GPU, as the scheduling operation happens after the sync of the last chunk.
            Sorted.
        '''
        if self.record_decide_time:
            torch.cuda.synchronize()
            _decide_start = time.time()
        ### Min run number
        num_deferred_chunks = len(self.deferred_chunks)
        num_last_active_and_ready_chunks = len(self.last_active_and_ready_chunks)
        input_num_ready_chunks = num_deferred_chunks + num_last_active_and_ready_chunks
        # self.num_deferred_chunks = num_deferred_chunks
        # self.num_last_active_and_ready_chunks = num_last_active_and_ready_chunks
        # self.num_ready_chunks = num_ready_chunks

        if len(self.ready_chunks) != input_num_ready_chunks or input_num_ready_chunks != self.num_chunks_before_anything:
            raise ValueError(f"ready chunks are {ids_of_chunk_list(self.ready_chunks)}, but input_num_ready_chunks is {input_num_ready_chunks} and num_chunks_before_anything is {self.num_chunks_before_anything}")
        
        if (
            input_num_ready_chunks <= self.num_min_run_chunks 
            or self.cross_layer_scheduling_strategy == "none"
        ):
            ## Run all chunks
            active_chunks = self.ready_chunks[:]
            # res = [i for i in range(input_num_ready_chunks)]
            # show_rank_print(f"deferred steps: {[chunk.num_deferred_steps for chunk in self.ready_chunks]}")
        else:
            num_active_chunks = 0
            active_chunks: List[ChunkBlock] = []
            ### add max_deferred chunks
            ready_chunks = self.ready_chunks[:]
            # show_rank_print(f"deferred steps: {[chunk.num_deferred_steps for chunk in ready_chunks]}")
            if self.enable_reactive_max_deferred_chunk:
                ## `enumerate_max_util` will do the append afterwards
                if self.cross_layer_scheduling_strategy != "enumerate_max_util":
                    _pop_indices = []
                    for i,chunk in enumerate(ready_chunks):
                        if chunk.num_deferred_steps == self.num_max_deferred_steps:
                            active_chunks.append(chunk)
                            _pop_indices.append(i)
                            num_active_chunks += 1
                    pop_elements_of_list(ready_chunks, _pop_indices)
                    num_ready_chunks = len(ready_chunks)
                
            if num_active_chunks < self.num_min_run_chunks:
                num_still_need_chunks = self.num_min_run_chunks - num_active_chunks
                
                with torch.cuda.stream(self.scheduling_cuda_stream):
                    if self.cross_layer_scheduling_strategy == "random":
                        res = _to_cpu_list(torch.randperm(num_ready_chunks, generator=self.gen)[:num_still_need_chunks])
                        active_chunks += [ready_chunks[i] for i in res]
                    elif self.cross_layer_scheduling_strategy == "cumsum_max_util":
                        if num_active_chunks > 0:
                            scheduled_chunk_device_counts: torch.IntTensor = torch.stack([chunk.global_tok_device_count for chunk in active_chunks]).sum(0)   
                            old_util = (scheduled_chunk_device_counts.sum() / scheduled_chunk_device_counts.max()).item()
                        else:
                            scheduled_chunk_device_counts = None                     
                        _pop_indices = []
                        for i,chunk in enumerate(ready_chunks):
                            if scheduled_chunk_device_counts is None: ## no active chunk yet, run the 1st
                                new_counts = chunk.global_tok_device_count
                                scheduled_chunk_device_counts = new_counts
                                old_util = (new_counts.sum() / new_counts.max()).item()
                                active_chunks.append(chunk)
                                _pop_indices.append(i)
                            else:
                                new_counts = scheduled_chunk_device_counts + chunk.global_tok_device_count
                                new_util = (new_counts.sum() / new_counts.max()).item()
                                if new_util >= old_util:
                                    scheduled_chunk_device_counts = new_counts
                                    old_util = new_util
                                    active_chunks.append(chunk)
                                    _pop_indices.append(i)
                        pop_elements_of_list(ready_chunks, _pop_indices)
                        active_chunks = self._sync_append_to_threshold(active_chunks, ready_chunks)                
                    else:
                        ###list of 'num_chunks' elements, each element is a [num_devices] tensor
                        if self.cross_layer_scheduling_strategy == "enumerate_max_util":
                            chunk_device_counts: torch.Tensor = torch.stack([chunk.global_tok_device_count for chunk in self.ready_chunks]) # [num_chunks, num_devices]
                            ## maximum util of the subset
                            res = self._enumerate_max_util(chunk_device_counts, self.num_min_run_chunks)
                            tmp_active_chunks = [self.ready_chunks[i] for i in res]
                            ## Add max_defered chunks
                            if self.enable_reactive_max_deferred_chunk:
                                for chunk in self.ready_chunks:
                                    if (
                                        chunk.num_deferred_steps == self.num_max_deferred_steps 
                                        and chunk not in tmp_active_chunks
                                    ):
                                        tmp_active_chunks.append(chunk)
                            ## Permute for better overlap
                            tmp_active_last_defer_chunks = []
                            tmp_active_last_active_chunks = []
                            for chunk in tmp_active_chunks:
                                if chunk.num_deferred_steps > 0:
                                    tmp_active_last_defer_chunks.append(chunk)
                                else:
                                    tmp_active_last_active_chunks.append(chunk)
                            active_chunks = tmp_active_last_defer_chunks + tmp_active_last_active_chunks
                                    
                        elif self.cross_layer_scheduling_strategy == "each_diff_peak":
                            last_active_chunks = [chunk for chunk in ready_chunks if chunk.num_deferred_steps == 0]
                            peak_devices = _to_cpu_list(torch.stack([chunk.global_tok_device_count for chunk in last_active_chunks]).max(dim=-1).indices)
                            # peak_devices = [chunk.global_tok_device_count.cpu().max().item() for chunk in last_active_chunks]
                            for chunk, peak_device in zip(last_active_chunks, peak_devices):
                                chunk.peak_device = peak_device
                                
                            chosen_devices = set()
                            ## append earlies
                            for active_chunk in active_chunks:
                                chosen_devices.add(active_chunk.peak_device)
                            _pop_indices = []    
                            for i,chunk in enumerate(ready_chunks):
                                if chunk.peak_device not in chosen_devices:
                                    active_chunks.append(chunk)
                                    chosen_devices.add(chunk.peak_device)
                                    _pop_indices.append(i)
                            pop_elements_of_list(ready_chunks, _pop_indices)
                            active_chunks = self._sync_append_to_threshold(active_chunks, ready_chunks)
                        # can have stragglers (one chunk is too imbalance)
                        elif self.cross_layer_scheduling_strategy == "first_k_util": 
                            # the top-k utilization
                            chunk_device_counts = torch.stack([chunk.global_tok_device_count for chunk in ready_chunks])
                            utils = _to_cpu_list(chunk_device_counts.sum(dim=-1) / chunk_device_counts.max(dim=-1).values)
                            for chunk, util in zip(ready_chunks, utils):
                                chunk.util = util
                            res = sorted(range(num_ready_chunks), key=lambda i: ready_chunks[i].util)[:num_still_need_chunks]
                            for i in sorted(res): # sort the res for order
                                active_chunks.append(ready_chunks[i])
                        elif self.cross_layer_scheduling_strategy == "bottom_k_max_minor_min":
                            # the bottom-k max-minor-min diff
                            chunk_device_counts = torch.stack([chunk.global_tok_device_count for chunk in ready_chunks])
                            gaps = (chunk_device_counts.max(dim=-1).values - chunk_device_counts.min(dim=-1).values) / chunk_device_counts.sum(dim=-1) # [num_chunks]
                            for chunk, gap in zip(ready_chunks, gaps):
                                chunk.max_min_gap = gap
                            res = sorted(range(num_ready_chunks), key=lambda i: ready_chunks[i].max_min_gap)[:num_still_need_chunks]
                            for i in sorted(res): # sort the res for order
                                active_chunks.append(ready_chunks[i])
                        else:
                            raise NotImplementedError
        
        self.scheduled_chunks = active_chunks[:]
        # show_rank_print(f"### inside decider chunks: {ids_of_chunk_list(active_chunks)}", 0)
                    
        if self.record_decide_time:
            torch.cuda.synchronize()
            _decide_end = time.time()
            self.decide_time += _decide_end - _decide_start
        # return active_chunks, (_shared_expert_pool_order,)
        return active_chunks, None
    
    def _sync_append_to_threshold(
        self,
        active_chunks: List[ChunkBlock],
        ready_chunks: List[ChunkBlock],
    ):
        gap = self.num_min_run_chunks - len(active_chunks)
        if gap > 0:
            ## append early chunks
            active_chunks += ready_chunks[:gap]
        return active_chunks
        
    
    def _prepare_mask(self, ):
        '''
            (2^N, N) binary mask.
        '''
        # Shape: (2^N,)
        ids = torch.arange(2**self.num_chunks, device=self.device)
        # Shape: (N,)
        bit_positions = torch.arange(self.num_chunks, device=self.device)
        # Broadcasted bit test
        masks = ((ids[:, None] >> bit_positions) & 1)
                
        self.masks: List[torch.Tensor] = []
        for num_chunks in range(self.num_min_run_chunks + 1, self.num_chunks + 1):
            this_mask = masks[:2**num_chunks][1:, :num_chunks] # [2^num_chunks-1, num_chunks]
            this_mask = this_mask[this_mask.sum(-1) >= self.num_min_run_chunks,:]
            self.masks.append(this_mask)    
            
    def _enumerate_min_diff(self, chunk_device_counts: torch.Tensor,):
        # TODO: expert map
        num_chunks = chunk_device_counts.shape[0]

        # create mask
        mask = self.masks[num_chunks - self.num_min_run_chunks - 1]
        # mask = self.masks[:2**num_chunks][1:, :num_chunks] # [2^num_chunks-1, num_chunks]
        # mask = mask[mask.sum(-1) >= self.num_min_run_chunks,:]
        if mask.numel() == 0:
            return [_ for _ in range(num_chunks)]
        # multiply mask and add
        grouped_results = ((chunk_device_counts[None,:,:] * mask[:,:,None])).sum(1) # [num_mask, num_devices]
        
        # avoid inconsistency if some results are equal
        grouped_diff = grouped_results.max(dim=-1).values - grouped_results.min(dim=-1).values
        # decide which best, and get corresponding indices 
        greatest_combination_value = grouped_diff.min(dim=-1).values # smallest of [num_mask]
        if self.reverse_schedule:
            greatest_combination_value = grouped_diff.max(dim=-1).values # smallest of [num_mask]
        greatest_combination_i = torch.where(grouped_diff == greatest_combination_value)[0][0] # the first
        res = _to_cpu_list(torch.where(mask[greatest_combination_i])[0])
        return res
    
    def _enumerate_min_abs_unused(self, chunk_device_counts: torch.Tensor,):
        num_chunks = chunk_device_counts.shape[0]

        # must have num_chunks > self.num_min_run_chunks
        assert num_chunks > self.num_min_run_chunks
        mask = self.masks[num_chunks - self.num_min_run_chunks - 1]
        
        # multiply mask and add
        grouped_results = ((chunk_device_counts[None,:,:] * mask[:,:,None])).sum(1) # [num_mask, num_devices]
        
        # avoid inconsistency if some results are equal
        grouped_abs_unused = (grouped_results.max(dim=-1, keepdim=True).values - grouped_results).sum(-1) # [num_mask]
        # decide which best, and get corresponding indices 
        if self.reverse_schedule:
            greatest_combination_value = grouped_abs_unused.max(dim=-1).values
        else:
            greatest_combination_value = grouped_abs_unused.min(dim=-1).values
        greatest_combination_i = torch.where(grouped_abs_unused == greatest_combination_value)[0][0] # the first
        res = _to_cpu_list(torch.where(mask[greatest_combination_i])[0])
        return res
    
    def _enumerate_max_util(
        self, 
        chunk_device_counts: torch.Tensor,
        num_min_active_chunks: int,
    ) -> List[int]:
        '''
            Return indices of active chunks, according to chunk_device_counts.
        '''
        
        num_chunks = chunk_device_counts.shape[0]

        # must have num_chunks > self.num_min_run_chunks
        assert num_chunks > num_min_active_chunks
        mask = self.masks[num_chunks - num_min_active_chunks - 1]
        
        grouped_results = ((chunk_device_counts[None,:,:] * mask[:,:,None])).sum(1) # [num_mask, num_devices]
        
        grouped_util = grouped_results.sum(-1) / grouped_results.max(dim=-1).values
        # show_rank_print(grouped_util, 0)    
        
        # Decide which best, and get corresponding indices 
        if self.reverse_schedule:
            raise ValueError
            # Just for ablation, should never used
            greatest_combination_value = grouped_util.min(dim=-1).values # smallest of [num_mask]
        else:
            # Largest of [num_mask]
            # greatest_combination_value = grouped_util.max(dim=-1).values
            greatest_combination_i = grouped_util.argmax(dim=-1)
        
        # greatest_combination_i = torch.where(grouped_util == greatest_combination_value)[0][0] # the first
        res = _to_cpu_list(torch.where(mask[greatest_combination_i])[0])
        return res