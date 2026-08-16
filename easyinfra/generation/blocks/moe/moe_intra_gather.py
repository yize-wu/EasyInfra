import torch
import torch._dynamo
import time
import json
from torch import nn
import math
from typing import Optional
import torch.nn.functional as F
from typing import Union, List

from easyinfra.schedule.chunk import ChunkBlock
from easyinfra.generation.blocks.mlp import MlpBlock
from easyinfra.generation.parallel.parallel_utils import get_device
from easyinfra.generation.parallel.communicator import GroupCommunicator
from easyinfra.generation.parallel.parallel_configuration import MoeParallelConfig

from easyinfra.generation.modules.moe import (
    get_moe_impl,
    is_chunk_routing,
)
from easyinfra.utils.stats import show_rank_print
from easyinfra import envs

from .moe_all2all import MoeBlock

class IntraGatherMoeBlock(MoeBlock):
    
    def __init__(
        self,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        self.stages = ((self.chunk_routing_compute, self.expert_data_communicate),
                       (self.all2all_prepare_token_communicate, self.all2all_token_communicate),
                       (self.all2all_hit_and_weight_slicing_and_shared_experts, self.shared_expert_result_communicate),
                       (self.all2all_expert_compute, self.all2all_expert_output_communicate),
                       (self.all2all_chunk_output_compute, self.all2all_gather_chunk_output_communicate),
                       )   
             
    def _expert_data_communicate(
        self,
        local_token_all_expert_count: torch.Tensor, 
    ):
        '''
            local_expert_count: [ep_rank, num_local_experts]
        '''
        # send expert data
        global_expert_count_work = self.ep_group.all_gather_into_tensor_auto(local_token_all_expert_count, async_op=True)
        global_expert_count = global_expert_count_work.output
        
        expert_count = global_expert_count[:,self.parallel_config.intra_node_group.all_ranks,:] # [ep_rank_from, (ep_rank_to, num_local_experts)]
        global_expert_count_work.wait()
                
        return global_expert_count, expert_count, global_expert_count_work.handler 
    
    def _intra_gather_communicate(
        self,
        hidden_states: torch.Tensor, 
    ):
        '''
            Intra-node gather.
        '''
        hidden_size = hidden_states.shape[-1]
        work = self.parallel_config.intra_node_group.all_gather_into_tensor_auto(hidden_states, async_op=True)
        work.wait()
        # return a flatten view
        return work.output.view(-1, hidden_size), work.handler
                
    def _inter_all2all_prepare_token_communicate(
        self, 
        hidden_states: torch.Tensor, 
        local_expert_mask: torch.Tensor,
        local_token_all_expert_count:torch.Tensor, 
        expert_count:torch.Tensor,
    ):
        '''
            return:
                send_hidden_states,
                recv_hidden_states,
                send_token_split, 
                recv_token_split, 
                send_indices, 
                send_weight_top_k, 
        '''         
        send_token_split = local_token_all_expert_count.sum(-1) # [ep_rank]
        _,_, send_weight_top_k, send_indices = torch.where(local_expert_mask)
        send_hidden_states = hidden_states[send_indices, :] # [num_send, hidden_size]
        
        recv_token_split = expert_count.sum(-1) # [ep_rank]
        # recv_hidden_length = recv_token_split.sum().item()
        
        send_token_split = send_token_split.tolist()
        recv_token_split = recv_token_split.tolist()
        recv_hidden_length = sum(recv_token_split)
        recv_hidden_states = torch.empty((recv_hidden_length, hidden_states.shape[-1]), dtype=hidden_states.dtype, device=hidden_states.device) # [sum_length, hidden_dim]
        
        return (
            send_hidden_states,
            recv_hidden_states,
            send_token_split, 
            recv_token_split, 
            send_indices, 
            send_weight_top_k, 
        )
    
    def _all2all_token_communicate(
        self, 
        send_hidden_states: torch.Tensor, 
        recv_hidden_states: torch.Tensor, 
        send_token_split: List[int], 
        recv_token_split: List[int], 
    ):
        '''
            Returns: 
            recv_routing_weights, [recv_hidden_length, k]
            recv_hidden_states,  [recv_hidden_length, hidden_dim]
            all_hidden_states_work.handler
        '''
        recv_hidden_states_work = self.ep_group.all_to_all_single(recv_hidden_states, send_hidden_states, recv_token_split, send_token_split, async_op=True)
        recv_hidden_states_work.wait()
        
        return recv_hidden_states_work.output, recv_hidden_states_work.handler
    
    def _all2all_hit_and_weight_slicing_and_shared_experts(
        self,
        hidden_states: Optional[torch.Tensor],
        routing_weights: torch.Tensor,
        send_indices: torch.Tensor,
        send_weight_top_k: torch.Tensor,
    ):
        # shared expert compute
        shared_experts = getattr(self, "shared_experts", None)
        assert (shared_experts is None and self.has_shared is False) or (shared_experts is not None and self.has_shared is True)
        if shared_experts is not None:
            hidden_states = shared_experts._compute(hidden_states)
        elif hidden_states is not None:
            raise ValueError("hidden states as a real parameter, but no shared experts")
            
        sliced_routing_weights = routing_weights[send_indices, send_weight_top_k] # [num_send]
        return sliced_routing_weights, hidden_states
    
    def _shared_expert_result_communicate(
        self, hidden_states: Optional[torch.Tensor]
    ):
        if not self.has_shared:
            return None, None
        work = self.shared_expert_tp_group.all_reduce(hidden_states, async_op=True)
        work.wait()
        return hidden_states, work.handler
        
    
    def _all2all_expert_compute(
        self, 
        recv_hidden_states: torch.Tensor, 
        expert_count: torch.Tensor, 
    ):
        '''
            device_expert_mask: [num_local_experts, k, all_t]
        '''
        if self.print_moe_compute_time:
            torch.cuda.synchronize()
            time0 = time.time()
        result = self.experts(recv_hidden_states, expert_count)
        if self.print_moe_compute_time:
            torch.cuda.synchronize()
            time1 = time.time()
            show_rank_print(f"moe compute time: {time1-time0}", 0)
        return result
            
    def _all2all_expert_output_communicate(
        self,
        output_tensor: torch.Tensor, 
        send_token_split: List[int], 
        recv_token_split: List[int],
    ):
        recv_tensor = output_tensor.new_empty((sum(send_token_split), output_tensor.shape[-1]))
        # time0 = time.time()
        # torch.cuda.current_stream().synchronize()
        # time1 = time.time()
        # show_rank_print(f"sync expert_output_communicate time: {time1 - time0}")
        # _after_compute = record_time_sync()
        # self.ep_group.barrier()
        # _after_barrier = record_time_sync()
        # self.barrier_wait_time += _after_barrier - _after_compute
        expert_output_work = self.ep_group.all_to_all_single(recv_tensor, output_tensor, send_token_split, recv_token_split, async_op=True)
        expert_output_work.wait()
        return expert_output_work.output, expert_output_work.handler
    
    def _all2all_chunk_output_compute(self, chunk_size, send_indices, recv_result_hidden_states:torch.Tensor, sliced_routing_weights: torch.Tensor):
        chunk_outputs = recv_result_hidden_states.new_zeros((chunk_size, self.hidden_size)) # [chunk, hidden_dim]
        recv_result_hidden_states *= sliced_routing_weights[:,None]
        chunk_outputs.index_add_(0, send_indices, recv_result_hidden_states)
        return chunk_outputs
    
    def _all2all_gather_chunk_output_communicate(self, chunk_outputs: torch.Tensor):
        final_outputs_work = self.chunk_routing_group.all_gather_into_tensor_auto(chunk_outputs, async_op=True)
        final_outputs_work.wait()
        return final_outputs_work.output, final_outputs_work.handler
    
        
    
    
    def chunk_routing_compute(self, chunk: ChunkBlock):
        # residual add
        (residual, hidden_states) = chunk.active_variables
        residual += hidden_states # must be in-place
        
        # prepare for routing
        hidden_states = residual
        normed_hidden_states = self.post_attention_layernorm(hidden_states).view(-1, self.hidden_size)
        chunked_hidden_states, local_expert_mask, local_token_all_expert_count, routing_weights, router_logits, chunk_size = self._chunk_routing_compute(normed_hidden_states)
        
        # show_rank_print("test", 0)
        # self.parallel_config.intra_node_group.all_gather_into_tensor_auto(normed_hidden_states)
        
        if self.shared_experts is None:
            normed_hidden_states = None # no need to keep the normed tensor

        chunk.active_variables = (
            chunked_hidden_states, 
            local_expert_mask, 
            local_token_all_expert_count, 
            chunk_size, 
            routing_weights, 
            normed_hidden_states, 
            residual
        )
        
    def expert_data_communicate(self, chunk: ChunkBlock):
        (
            chunked_hidden_states, 
            local_expert_mask, 
            local_token_all_expert_count, 
            chunk_size, 
            routing_weights, 
            normed_hidden_states, 
            residual
        ) = chunk.active_variables
        
        global_expert_count, expert_count, handler = self._expert_data_communicate(local_token_all_expert_count)
        chunk.comm_handler = handler
        ### The expert count has not been synced, so do not use the value now
        chunk.global_expert_count = global_expert_count
        chunk.expert_count = expert_count
        # chunk.expert_count_flat_list = chunk.expert_count.reshape(-1).tolist()
        
        chunk.active_variables = (
            chunked_hidden_states, 
            routing_weights, 
            local_expert_mask, 
            local_token_all_expert_count, 
            expert_count, 
            chunk_size, 
            normed_hidden_states, 
            residual
        )
        
    def all2all_prepare_token_communicate(self, chunk: ChunkBlock):
        (
            chunked_hidden_states, 
            routing_weights, 
            local_expert_mask, 
            local_token_all_expert_count, 
            expert_count, 
            chunk_size, 
            normed_hidden_states,
            residual
        ) = chunk.active_variables
        
        (
            send_hidden_states,
            recv_hidden_states,
            send_token_split, 
            recv_token_split, 
            send_indices, 
            send_weight_top_k, 
        ) = self._all2all_prepare_token_communicate(
            chunked_hidden_states,
            local_expert_mask,
            local_token_all_expert_count,
            expert_count,
        )
        # chunk.send_weight_top_k = send_weight_top_k
        # chunk.send_indices = send_indices
        # chunk.send_token_split = send_token_split
        # chunk.recv_token_split = recv_token_split
        
        try_send = False
        if try_send:
            send_hidden_state_splits = send_hidden_states.split(send_token_split)
            torch.cuda.synchronize()
            time0 = time.time()
            send_hidden_states = torch.cat(send_hidden_state_splits, dim=0)
            torch.cuda.synchronize()
            time1 = time.time()
            show_rank_print(f"tensor permute time: {time1-time0}", 0)
        
        chunk.active_variables = (
            routing_weights,
            expert_count, 
            send_hidden_states,
            recv_hidden_states,
            send_token_split, 
            recv_token_split, 
            send_indices, 
            send_weight_top_k, 
            chunk_size,
            normed_hidden_states,
            residual,
        )
        
    
    def all2all_token_communicate(self, chunk: ChunkBlock):
        (
            routing_weights,
            expert_count, 
            send_hidden_states,
            recv_hidden_states,
            send_token_split, 
            recv_token_split, 
            send_indices, 
            send_weight_top_k, 
            chunk_size,
            normed_hidden_states,
            residual,
        ) = chunk.active_variables
                
        recv_hidden_states, handler = self._all2all_token_communicate(
            send_hidden_states, recv_hidden_states, send_token_split, recv_token_split
        )
        chunk.comm_handler = handler
        chunk.compute_wait_comm = False # no need to wait
        
        chunk.active_variables = (
            recv_hidden_states,
            routing_weights,
            expert_count,
            send_token_split, 
            recv_token_split, 
            send_indices, 
            send_weight_top_k, 
            chunk_size,
            normed_hidden_states,
            residual,
        )
    
    def all2all_hit_and_weight_slicing_and_shared_experts(self, chunk: ChunkBlock):
        (
            recv_hidden_states,
            routing_weights,
            expert_count,
            send_token_split, 
            recv_token_split, 
            send_indices, 
            send_weight_top_k, 
            chunk_size,
            hidden_states_for_shared_expert_compute,
            residual,
        ) = chunk.active_variables
        
        # add to residual if needed
        (
            sliced_routing_weights, 
            hidden_states_for_shared_expert_compute
        ) = self._all2all_hit_and_weight_slicing_and_shared_experts(
            hidden_states_for_shared_expert_compute,
            routing_weights,
            send_indices,
            send_weight_top_k,
        )
        
        chunk.active_variables = (
            recv_hidden_states,
            sliced_routing_weights,
            send_token_split, 
            recv_token_split, 
            send_indices, 
            expert_count,
            hidden_states_for_shared_expert_compute,
            chunk_size,
            residual,
        )
    
    def shared_expert_result_communicate(self, chunk: ChunkBlock):
        (
            recv_hidden_states,
            sliced_routing_weights,
            send_token_split, 
            recv_token_split, 
            send_indices, 
            expert_count,
            hidden_states_for_shared_expert_compute,
            chunk_size,
            residual,
        ) = chunk.active_variables
        
        hidden_states_for_shared_expert_compute, handler = self._shared_expert_result_communicate(hidden_states_for_shared_expert_compute)
        ## if no shared_expert, the handler will be from the previous stage
        if self.has_shared:
            chunk.comm_handler = handler
        # add to residual to save memory
        if hidden_states_for_shared_expert_compute is not None:
            residual += hidden_states_for_shared_expert_compute
        
        chunk.active_variables = (
            recv_hidden_states,
            sliced_routing_weights,
            send_token_split, 
            recv_token_split, 
            send_indices, 
            expert_count,
            chunk_size,
            residual,
        )
    
    def all2all_expert_compute(self, chunk: ChunkBlock):
        (
            recv_hidden_states,
            sliced_routing_weights,
            send_token_split, 
            recv_token_split, 
            send_indices, 
            expert_count,
            chunk_size,
            residual,
        ) = chunk.active_variables
        
        # torch.cuda.synchronize()
        # time0 = time.time()
        recv_hidden_states = self._all2all_expert_compute(
            recv_hidden_states,
            # expert_count,
            chunk.expert_count_flat_list,
        )
        # torch.cuda.synchronize()
        # time1 = time.time()
        # show_rank_print(f"expert compute time: {time1 - time0}", 0)
        
        chunk.active_variables = (
            recv_hidden_states, 
            sliced_routing_weights,
            send_token_split, 
            recv_token_split,
            send_indices, 
            chunk_size,
            residual,
        )
    
    def all2all_expert_output_communicate(self, chunk: ChunkBlock):
        (
            hidden_states, # recv_hidden_states
            sliced_routing_weights,
            send_token_split, 
            recv_token_split,
            send_indices, 
            chunk_size,
            residual,
        ) = chunk.active_variables
        
        # torch.cuda.synchronize()
        # time0 = time.time()
        hidden_states, handler = self._all2all_expert_output_communicate(hidden_states, send_token_split, recv_token_split)
        chunk.comm_handler = handler
        # torch.cuda.synchronize()
        # time1 = time.time()
        # show_rank_print(f"expert comm time: {time1 - time0}", 0)
        
        output_recv_hidden_states = hidden_states
        
        chunk.active_variables = (
            output_recv_hidden_states,
            sliced_routing_weights,
            send_indices, 
            chunk_size,
            residual,
        )
    
    def all2all_chunk_output_compute(self, chunk: ChunkBlock):
        (
            hidden_states, # output_recv_hidden_states
            sliced_routing_weights,
            send_indices, 
            chunk_size,
            residual,
        ) = chunk.active_variables
        hidden_states = self._all2all_chunk_output_compute(chunk_size, send_indices, hidden_states, sliced_routing_weights)
        chunk.active_variables = (residual, hidden_states)

    def all2all_gather_chunk_output_communicate(self, chunk: ChunkBlock):
        (residual, hidden_states) = chunk.active_variables
        # show_rank_print(f"residual: {residual.sum()}, hidden_states: {hidden_states.sum()}", 0)
        hidden_states, handler = self._all2all_gather_chunk_output_communicate(hidden_states)
        chunk.comm_handler = handler
        chunk.active_variables = (residual, hidden_states)
        
