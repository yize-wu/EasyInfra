from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .chunk import (
        ChunkBlock,
        ChunkBlockList,
    )
    from ..generation.parallel.communicator import GroupCommunicator

import torch
from typing import List, Optional, Callable, Tuple, Dict

from easyinfra.generation.blocks.stage_name import MoeStageName
from easyinfra.schedule.cross_layer_balancer import (
    MoeCrossLayerBalancer,
    MoeBalanceStrategyType,
)
from easyinfra.utils import show_rank_print, record_time_sync
from easyinfra.generation.parallel.parallel_utils import get_device, get_world_size
import time
import math
import random
from easyinfra import envs
from easyinfra.utils.tensors import _to_cpu_list
from easyinfra.utils.generic import lst_permute
from .utils import ids_of_chunk_list

_SYNC_TIME = 0.0    


class BaseStagePool:
    wait_producer: bool = False
    def __init__(self, chunk_list: List[ChunkBlock]):
        self.chunk_list: List[ChunkBlock] = chunk_list
        self.pool: List[ChunkBlock] = []
        
    def _add(
        self,
        chunk: ChunkBlock,
    ):
        if chunk.is_finished():
            raise ValueError(f"chunk{chunk.chunk_id} is finished, but applied add")
        self.pool.append(chunk)
    
    def add_or_finalize(
        self,
        chunk: ChunkBlock,
    ):
        if chunk.is_finished():
            chunk.finalize()
        else:
            self._add(chunk)
    
    def delete(
        self,
        chunk: ChunkBlock,
    ):
        for i,c in enumerate(self.pool):
            if c.chunk_id == chunk.chunk_id:
                self.pool.pop(i)
                return
        raise ValueError(f"chunk{chunk.chunk_id} is not in this pool: {[c.chunk_id for c in self.pool]}")
        
    def is_empty(self) -> bool:
        return len(self.pool) == 0
    
    def permute(self, indices: List[int]):
        self.pool = lst_permute(self.pool, indices)
    
    def __getitem__(self, index: int):
        return self.pool[index]
        
    # def find_and_evict_step_finished_chunk(self):
    #     '''
    #         Return a ready chunk and [all finished chunks] in the pool.
    #         If none of chunks is ready, return None.
    #     '''
    #     chunk = None
    #     evicted_idx = []
    #     for chunk_i, chunk in enumerate(self.pool):
    #         if chunk.is_global_ready(self.wait_producer):
    #             # show_rank_print(f"global ready: {chunk.chunk_id}")
    #             # the reverse order can make sure that pop() is correctly used
    #             evicted_idx.insert(0, chunk_i)
    #             chunk.prepare_next_step()
    #             # a finished chunk should not be returned as ready chunk
    #             if not chunk.is_finished():
    #                 break
    #         else:
    #             # show_rank_print(f"global not ready: {chunk.chunk_id}")
    #             pass
                
    #     # evict ready chunk and finished chunks
    #     if evicted_idx == []:
    #         return None, []
    #     else:
    #         finished_chunks = []
    #         # evict finished chunks, but no need to return them as ready
    #         for evict_i in evicted_idx[:-1]:
    #             finished_chunks.append(self.pool.pop(evict_i))
            
    #         return self.pool.pop(evicted_idx[-1]), finished_chunks
    

class WaitProducerStagePool(BaseStagePool):
    wait_producer: bool = True

class MoeStagePool(BaseStagePool):
    pass


class MoeLayerScheduler:
    def __init__(
        self,
        # chunk_list: ChunkBlockList,
        model,
        cross_layer_scheduling_strategy: Optional[str],
        num_min_run_chunks: int = -1,
        num_max_deferred_steps: int = -1,
        enable_token_count_stats = False,
        enable_device_history_stats = False,
        enable_expert_history_stats = False,
        enable_trajectory_history_stats = False,
        record_decide_time = False,
    ):
        self.first_k_dense_replace = getattr(model, "first_k_dense_replace", 0)
        self.ep_group: GroupCommunicator = model.parallel_config.ep_group
        self.attn_tp_group: GroupCommunicator = model.parallel_config.attn_tp_group
        self.mlp_tp_group: GroupCommunicator = model.parallel_config.mlp_tp_group
        self.moe_comm_mode = envs.MOE_COMM_MODE
        
        self.cross_layer_balancer = MoeCrossLayerBalancer(
            num_min_run_chunks=num_min_run_chunks,
            num_max_deferred_steps=num_max_deferred_steps,
            cross_layer_scheduling_strategy=cross_layer_scheduling_strategy,
            record_decide_time=record_decide_time,
        )
        self.cross_layer_scheduling_strategy = cross_layer_scheduling_strategy
        self.num_min_run_chunks = num_min_run_chunks
        self.num_max_deferred_steps = num_max_deferred_steps
        self.enable_token_count_stats = enable_token_count_stats
        self.enable_device_history_stats = enable_device_history_stats
        self.enable_expert_history_stats = enable_expert_history_stats
        self.enable_trajectory_history_stats = enable_trajectory_history_stats
        if self.cross_layer_scheduling_strategy == "random":
            self.gen = torch.Generator()
            self.gen.manual_seed(42) # for random uniform
        self.cross_layer_scheduling_cuda_stream = torch.cuda.Stream()
        
        # self.cpu_cuda_stream = torch.cuda.Stream()
        # self.cpu_cuda_stream = torch.cuda.current_stream()
        
        self.all_workload = 0
        self.sum_each_gap: int = 0
        self.sum_gap: int = 0
        
        self.shortage_of_all_devices = 0
        self.effective_workload = 0
        
        # self.load_stats: List[torch.Tensor] = []
        
        self.decide_time = 0.0
        self.reverse_schedule = False
        self.trajectory_history: List[List[int]] = []
    
    def show_pool_and_chunks(self):
        all_chunk_ids = []
        for i, pool in enumerate(self.all_pools):
            all_chunk_ids.append([chunk.chunk_id for chunk in pool.pool])
        show_rank_print(f"{all_chunk_ids}", 0)
        
    
    def is_finished(self):
        return all(pool.is_empty() for pool in self.all_pools)   
    
    

    def update_chunks(self, chunk_list: ChunkBlockList):
        if self.num_min_run_chunks == -1:
            self.num_min_run_chunks = max(math.ceil(len(chunk_list.chunk_list) * 0.75), 3)
            show_rank_print(f"num min run chunk auto set to {self.num_min_run_chunks}")
        
        self.cross_layer_balancer.prepare(num_chunks=len(chunk_list.chunk_list))
        
        # scheduler
        self.attention_stage_pool = WaitProducerStagePool(chunk_list)
        # if self.first_k_dense_replace > 0:
        #     self.first_k_dense_replace_pool = BaseStagePool(chunk_list)
        self.first_k_dense_replace_pool = BaseStagePool(chunk_list)
        self.routing_stage_pool = BaseStagePool(chunk_list)
        self.shared_expert_stage_pool = BaseStagePool(chunk_list)
        self.moe_compute_combine_stage_pool = MoeStagePool(chunk_list)
        self.moe_epilogue_stage_pool = MoeStagePool(chunk_list)
        
        self.all_pools: List[BaseStagePool] = [
            self.attention_stage_pool, 
            self.first_k_dense_replace_pool,
            self.routing_stage_pool, 
            self.shared_expert_stage_pool, 
            self.moe_compute_combine_stage_pool,
            self.moe_epilogue_stage_pool,
        ]
                
        # add attention compute to list
        for chunk in chunk_list:
            self.attention_stage_pool.add_or_finalize(chunk) # for cold start
        
        self.chunk_list = chunk_list
    
    

    
    
    def update_tensors(
        self,
        hidden_states: List[torch.Tensor],
        position_embeddings: Tuple[List[torch.Tensor], List[torch.Tensor]] = None,
    ):
        """
            Update hidden_states and pos_embeddings
        """
        num_chunks = self.chunk_list.get_length()
        if len(hidden_states) != num_chunks or (position_embeddings is not None and len(position_embeddings[0]) != num_chunks):
            raise ValueError(f"num of chunks is {num_chunks}, but h is {len(hidden_states)} and pos is {len(position_embeddings[0])}")
        
        for i, chunk in enumerate(self.chunk_list.chunk_list):
            chunk.active_variables = (hidden_states[i],) + chunk.active_variables
            if position_embeddings is not None:
                chunk.position_embeddings = (position_embeddings[0][i], position_embeddings[1][i]) # cos, sin
    
    def pool_step(
        self,
        pool: BaseStagePool,
        next_pool: BaseStagePool,
        selected_chunks: Optional[List[ChunkBlock]] = None,
        prehook_for_ready_chunk: Callable = None,
        posthook_for_ready_chunk: Callable = None,
        force_disable_multi_stage_forward: Optional[bool] = None,
        run_last_communication: Optional[bool] = None,
        get_over_next_comm: Optional[bool] = None,
        defer_next_valid_comm: Optional[bool] = None,
        do_early_launch_moe: bool = False,
        do_moe_schedule: bool = False,
        **run_kwargs
    ):
        # time1 = time.time()
        # show_rank_print(f"sync time {time.time() - time1}")
        # self.show_pool_and_chunks()
        
        if selected_chunks is None:
            ## do not compromise the original pool
            selected_chunks = [chunk for chunk in pool] 
            
        # show_rank_print(f"run chunks: {[chunk.chunk_id for chunk in selected_chunks]}", 0)
        for i,chunk in enumerate(selected_chunks):
            assert chunk.is_ready(pool.wait_producer)

            time0 = time.time()
            # run pre hook
            if prehook_for_ready_chunk is not None:
                prehook_for_ready_chunk(chunk)
            
            # time1 = time.time()            
            if do_moe_schedule:
                self.cross_layer_balancer.add_ready_chunk(chunk)
            ## Run chunk
            chunk.run(force_disable_multi_stage_forward=force_disable_multi_stage_forward,
                        get_over_next_comm=get_over_next_comm,
                        defer_next_valid_comm=defer_next_valid_comm,
                        run_last_communication=run_last_communication,
                        **run_kwargs)
            
            ## If early launch moe, we need to know which to run, with async method
            if do_moe_schedule and self.cross_layer_balancer.strategy_type == MoeBalanceStrategyType.ASYNC:
                active_chunks = self.cross_layer_balancer.async_decide_which_run()
                if do_early_launch_moe:
                    for early_active_moe_chunk in active_chunks:
                        early_active_moe_chunk.run(defer_next_valid_comm=True) # defer moe comm, TODO what if no moe comm (EP=1)?
                
            # run post hook
            # time2 = time.time()
            if posthook_for_ready_chunk is not None:
                posthook_for_ready_chunk(chunk)
                
            pool.delete(chunk)
            # important: do not add finished chunk to pool
            next_pool.add_or_finalize(chunk)
            # time3 = time.time()
            # show_rank_print(f"pool_step chunk[{chunk.chunk_id}] time: {time1 - time0}, {time2 - time1}, {time3 - time2} id: {chunk.chunk_id}", 0)
        
        ## It is possible that there is no still-active chunks. Remedy here.
        if (
            do_moe_schedule 
            and self.cross_layer_balancer.strategy_type == MoeBalanceStrategyType.ASYNC 
            and len(selected_chunks) == 0
        ):
            active_chunks = self.cross_layer_balancer.async_decide_which_run()
            if do_early_launch_moe:
                for early_active_moe_chunk in active_chunks:
                    early_active_moe_chunk.run(defer_next_valid_comm=True)
        return
                    
    def run(
        self,
    ):
        for chunk in self.attention_stage_pool:
            chunk.require_global_device_count_on_cpu = (
                self.cross_layer_balancer.strategy_type == MoeBalanceStrategyType.ASYNC
            )
        
        num_moe_steps = 0
        num_scheduling_steps = 0
        
        while not self.is_finished():

            # attention
            if not self.attention_stage_pool.is_empty() and self.attention_stage_pool.pool[0].num_finished_pool_steps < self.first_k_dense_replace * 2:
                # first k layers are dense MLP
                defer_dispatching = (self.attn_tp_group.group_size == 1 and self.mlp_tp_group.group_size == 1 and self.ep_group.group_size > 1)
                self.pool_step(self.attention_stage_pool, next_pool=self.first_k_dense_replace_pool,
                               defer_next_valid_comm = defer_dispatching)
                self.pool_step(self.first_k_dense_replace_pool, next_pool=self.attention_stage_pool,
                               )
            else:
                is_first_moe_step = (num_moe_steps == 0 and num_scheduling_steps == self.first_k_dense_replace)
                defer_dispatching = is_first_moe_step and self.attn_tp_group.group_size == 1 and self.ep_group.group_size > 1
                self.pool_step(self.attention_stage_pool, next_pool=self.routing_stage_pool,
                            defer_next_valid_comm = defer_dispatching)            
                
            ### routing_stage_pool empty because: 1. first_k, 2. all chunks exit (max layer)
            ### if 1. then shared pool must be empty; if 2. and shared_expert_stage_pool non-empty, then should continue
            if (self.routing_stage_pool.is_empty() 
                and self.shared_expert_stage_pool.is_empty() 
            ):
                num_scheduling_steps += 1
                continue
            
            
            # token transmission
            deferred_chunks = self.shared_expert_stage_pool.pool
            last_active_and_ready_chunks = self.routing_stage_pool.pool
            self.cross_layer_balancer.prepare_each_step(deferred_chunks, last_active_and_ready_chunks)
            # show_rank_print(f"deferred chunks: {ids_of_chunk_list(deferred_chunks)}", 0)
            # show_rank_print(f"last active and ready chunks: {ids_of_chunk_list(last_active_and_ready_chunks)}", 0)
            
            for chunk in deferred_chunks:
                self.cross_layer_balancer.add_ready_chunk(chunk)
            
            # show_rank_print("routing", 0)
            do_early_launch_moe = (
                self.cross_layer_balancer.strategy_type == MoeBalanceStrategyType.ASYNC
                and self.ep_group.group_size > 1
            )
            run_last_communication = (self.attn_tp_group.group_size == 1 and self.ep_group.group_size > 1)
            self.pool_step(self.routing_stage_pool, next_pool=self.shared_expert_stage_pool,
                           run_last_communication = run_last_communication,
                           do_moe_schedule = True,
                           do_early_launch_moe = do_early_launch_moe,
                           )
                        
            if self.cross_layer_balancer.strategy_type == MoeBalanceStrategyType.SYNC:
                next_active_chunks, balancer_decide_side_effects = self.cross_layer_balancer.sync_decide_which_run()
                ### Add the new_run chunk to the start of the pool, so that they finish first
            else:
                next_active_chunks = self.cross_layer_balancer.scheduled_chunks[:]
            
            # show_rank_print(f"### decider chunks (id): {ids_of_chunk_list(next_active_chunks)}", 0)
            if not next_active_chunks:
                raise ValueError
            self.maybe_count_routing_stats()
            self.cross_layer_balancer.finalize_each_step()
            
            
            ### Shared Experts. Indices are just a range(), as the pool has been permuted
            self.pool_step(self.shared_expert_stage_pool, next_pool=self.moe_compute_combine_stage_pool,
                           force_disable_multi_stage_forward=True,
                           selected_chunks=next_active_chunks)
                        
                                    
            ### MoE Stage can be a little different.            
            ### If no compute-comm overlap, run all compute_func, then all comm_func, to reduce syncs!!
            ### Otherwise, chunk-by-chunk would be the same.
            for chunk in self.moe_compute_combine_stage_pool:
                # time0 = time.time()
                if (chunk.num_finished_compute_steps <= chunk.num_finished_pool_steps # no fast forward
                    and chunk.num_finished_compute_steps <= chunk.num_finished_comm_steps # no do extra moe
                ):
                    chunk.run_compute()
                if envs.ENABLE_COMPUTE_COMM_OVERLAP is True:
                    if chunk.num_finished_comm_steps <= chunk.num_finished_pool_steps:
                        chunk.run_communicate()
                
            if False:
                # TODO: run condition is not good
                N = 4
                # pool_len = len(self.moe_compute_combine_stage_pool.pool)
                for chunk_i, chunk in enumerate(self.moe_compute_combine_stage_pool.pool):
                    # time0 = time.time()
                    if chunk.num_finished_comm_steps <= chunk.num_finished_pool_steps:
                        chunk.run_communicate()
                    if not do_early_launch_moe:
                        chunk.num_finished_pool_steps += 1
                    if chunk_i >= N:
                        self.pool_step(self.moe_epilogue_stage_pool, next_pool=self.attention_stage_pool,
                            # defer_next_valid_comm=True
                            get_over_next_comm=True
                            )
                    self.moe_epilogue_stage_pool.add_or_finalize(chunk)
                        
                    # time1 = time.time()
                    # show_rank_print(f"pool_step chunk time: {time1 - time0}, id: {chunk.chunk_id}", 0)
                # show_rank_print(f"compute query after moe: {[chunk.compute_event.query() for chunk in self.moe_epilogue_stage_pool.pool]}", 0)
                # for i in sorted(real_next_moe_compute_i)[::-1]:
                for i in range(len(self.moe_compute_combine_stage_pool.pool))[::-1]:
                    self.moe_compute_combine_stage_pool.pool.pop(i)
                
                self.pool_step(self.moe_epilogue_stage_pool, next_pool=self.attention_stage_pool,
                            # defer_next_valid_comm=True
                            get_over_next_comm=True
                            )
            else:
                for chunk in self.moe_compute_combine_stage_pool:
                    # time0 = time.time()
                    if envs.ENABLE_COMPUTE_COMM_OVERLAP is False:
                        if chunk.num_finished_comm_steps <= chunk.num_finished_pool_steps:
                            chunk.run_communicate()
                    if not do_early_launch_moe:
                        chunk.num_finished_pool_steps += 1
                    self.moe_epilogue_stage_pool.add_or_finalize(chunk)
                    # time1 = time.time()
                    # show_rank_print(f"pool_step chunk time: {time1 - time0}, id: {chunk.chunk_id}", 0)
                # for i in sorted(real_next_moe_compute_i)[::-1]:
                for i in range(len(self.moe_compute_combine_stage_pool.pool))[::-1]:
                    self.moe_compute_combine_stage_pool.pool.pop(i)
                
                _next_pool = self.attention_stage_pool
                # allocated = torch.cuda.memory_allocated()/1024**3
                # reserved = torch.cuda.memory_reserved()/1024**3
                # show_rank_print(f"allocated: {allocated}, reserved: {reserved}", 0)
                self.pool_step(self.moe_epilogue_stage_pool, next_pool=_next_pool,
                            defer_next_valid_comm=True
                            )
                # show_rank_print(f"compute query after epilogue: {[chunk.compute_event.query() for chunk in _next_pool.pool]}", 0)
            
            num_moe_steps += 1
            num_scheduling_steps += 1
            
        # run last prepare op
        for chunk in self.chunk_list:
            # the communication is perhaps running
            if chunk.compute_wait_comm:
                torch.cuda.current_stream().wait_event(chunk.comm_event)

        if self.cross_layer_balancer.record_decide_time:
            show_rank_print(f"decide time: {self.cross_layer_balancer.decide_time}/{self.cross_layer_balancer.num_steps}={self.cross_layer_balancer.decide_time / self.cross_layer_balancer.num_steps}", 0)
        
        global _SYNC_TIME
        _SYNC_TIME = 0.0
        return
    
    def maybe_count_routing_stats(self,
                                  ):
        # stats
        if self.enable_token_count_stats:
            _each_num_list = torch.stack([active_chunk.get_global_tok_device_count() for active_chunk in self.cross_layer_balancer.scheduled_chunks]) # [num_run_chunk, num_devices]
            _active_device_count_list = _each_num_list.sum(0) # [num_devices]
            # self.sum_gap += (_num_list.max() - _num_list.min()).item()
            this_step_all_workload = _active_device_count_list.sum().item()
            self.all_workload += this_step_all_workload
            self.effective_workload += _active_device_count_list.max().item() * _active_device_count_list.shape[-1]
            self.shortage_of_all_devices += (_active_device_count_list.max() - _active_device_count_list).sum().item()
            # show_rank_print(f"under-util: {self.shortage_of_all_devices}/{self.effective_workload}", 0)
            
            # each_diff = _each_num_list.max(dim=1).values - _each_num_list.min(dim=1).values
            # self.sum_each_gap += each_diff.sum().item()
            # show_rank_print(f"each diff sum: {each_diff.sum().item()}")
            # show_rank_print(f"this_step_all_workload: {this_step_all_workload}:{_each_num_list.sum(-1).tolist()}", 0)
            # show_rank_print(f"real diff: {(_active_device_count_list.max() - _active_device_count_list.min()).item()}", 0)
        if self.enable_device_history_stats:
            for active_chunk in self.cross_layer_balancer.scheduled_chunks:
                ## global_expert_count_2d: [ep_rank_from, (ep_rank_to, num_local_experts)], (ep_rank_to, num_local_experts) is from expert map
                active_chunk.global_device_history.append(_to_cpu_list(active_chunk.get_global_tok_device_count()))
        if self.enable_expert_history_stats:
            for active_chunk in self.cross_layer_balancer.scheduled_chunks:
                active_chunk.global_expert_history.append(_to_cpu_list(active_chunk.global_tok_all_expert_count_2d.sum(0).reshape(-1)))
        if self.enable_trajectory_history_stats:
            self.trajectory_history.append([active_chunk.chunk_id for active_chunk in self.cross_layer_balancer.scheduled_chunks])
    
    def get_output_hidden_states(self):
        """
            Output the output hidden tensors of all chunks
        """
        return self.chunk_list.slice_output_hidden_states()
    
    def get_hidden_states(self):
        """
            Output the listed hidden tensors of all chunks
        """
        return [chunk.active_variables[0] for chunk in self.chunk_list]
    
    def get_flat_hidden_states(self):
        """
            Output the flattened hidden tensors of all chunks
        """
        return torch.cat([chunk.active_variables[0] for chunk in self.chunk_list], dim=0)
    
