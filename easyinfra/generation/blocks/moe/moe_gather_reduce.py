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
from easyinfra.generation.utils.compute_utils import _add_to_residual

from easyinfra.generation.modules.moe import (
    get_moe_impl,
    is_chunk_routing,
)
from easyinfra.utils.stats import show_rank_print
from easyinfra import envs

from .moe_all2all import MoeAll2AllBlock
from easyinfra.generation.utils.perf import class_with_timing

@class_with_timing
class MoeGatherReduceBlock(MoeAll2AllBlock):
    
    def __init__(
        self,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        self.stages = ((self.gather_reduce_chunk_routing_compute, self.gather_reduce_expert_data_communicate),
                       (self.gather_reduce_prepare_token_communicate, self.gather_reduce_token_communicate),
                       (self.gather_reduce_slicing_and_shared_experts, self.shared_expert_result_communicate),
                       (self.gather_reduce_expert_compute, self.gather_reduce_expert_output_communicate),
                       (self.gather_reduce_chunk_output_compute, self.gather_reduce_chunk_output_communicate),
                       )   
    
    def _cast_tensor_to_flat_view(self, t: torch.Tensor, size_dim: int = -1):
        assert size_dim == -1
        t_dim = t.dim()
        if t_dim == 2:
            # already flat
            return t
        elif t_dim == 3:
            return t.view(-1, t.shape[size_dim])
        else:
            raise ValueError(f"casted tensor has dim={t_dim}, while require 2 or 3")
        
    def _cast_tensor_to_gathered_view(self, t: torch.Tensor, group_size: int, size_dim: int = -1, ):
        assert size_dim == -1
        t_dim = t.dim()
        if t_dim == 3:
            # already in gathering form
            return t
        elif t_dim == 2:
            return t.view(group_size, -1, t.shape[size_dim])
        else:
            raise ValueError(f"casted tensor has dim={t_dim}, while require 2 or 3")        
             
    def _gather_reduce_expert_data_communicate(
        self,
        local_tok_all_expert_count_2d: torch.LongTensor,
        selected_experts,
    ):
        '''
            routing_weights: [local_t, k]
            selected_experts: [local_t, k]
        '''
        ### transfer expert_count
        global_tok_all_expert_count_work = self.ep_group.all_gather_into_tensor_auto(local_tok_all_expert_count_2d, async_op=True)
        global_tok_all_expert_count_work.wait()
        global_tok_all_expert_count_2d = global_tok_all_expert_count_work.output # [ep_rank_from, (ep_rank_to, num_local_experts)]
        global_tok_local_expert_count_2d = global_tok_all_expert_count_2d[:,self.ep_rank,:] # [ep_rank_from, num_local_experts]
        ### transfer selected_experts
        all_selected_experts_work = self.ep_group.all_gather_into_tensor_auto(selected_experts, async_op=True)
        all_selected_experts_work.wait()
        all_selected_experts = all_selected_experts_work.output
                
        return (
            global_tok_all_expert_count_2d,
            global_tok_local_expert_count_2d,
            all_selected_experts,
            global_tok_all_expert_count_work.handler,
        )
    
                
    def _gather_reduce_prepare_token_communicate(
        self, 
        selected_experts: torch.Tensor,
        local_tok_all_expert_count_2d: torch.Tensor,
        global_tok_local_expert_count_2d: torch.Tensor,
    ):
        '''
        '''
        send_token_split = local_tok_all_expert_count_2d.sum(-1) # [ep_rank]
        recv_token_split = global_tok_local_expert_count_2d.sum(-1) # [ep_rank]
        send_token_split = send_token_split.tolist()
        recv_token_split = recv_token_split.tolist()
        recv_length = sum(recv_token_split)
        ### compute expert mask
        local_tok_all_expert_mask = self._all_expert_mask_compute(selected_experts)
        _, _, send_weight_top_k, send_token_indices = torch.where(local_tok_all_expert_mask)
        ### create recv tensors
        recv_weight_top_k = send_weight_top_k.new_empty((recv_length))         
        recv_token_indices = send_token_indices.new_empty((recv_length))         
        
        return (
            send_token_split,
            recv_token_split,
            send_weight_top_k, 
            send_token_indices, 
            recv_weight_top_k,
            recv_token_indices,
        )
        
    def _gather_reduce_token_communicate(
        self,
        hidden_states: torch.Tensor, 
        routing_weights: torch.Tensor, 
        send_split: List[int],
        recv_split: List[int],
        send_weight_top_k: torch.Tensor, 
        send_token_indices: torch.Tensor, 
        recv_weight_top_k: torch.Tensor,
        recv_token_indices: torch.Tensor,
        metadata_comm_event: torch.cuda.Event,
    ):
        '''
            All gather.
            Output: hidden_states: [-1, hidden_size]
        '''
        ## communicate meta data with all2all still (metadata is small)
        work = self.parallel_config.ep_group.all_to_all_single(recv_weight_top_k, send_weight_top_k, recv_split, send_split)
        work.wait()
        work = self.parallel_config.ep_group.all_to_all_single(recv_token_indices, send_token_indices, recv_split, send_split)
        work.wait()
        metadata_comm_event.record(torch.cuda.current_stream())
        
        # all_hidden_states
        work = self.parallel_config.ep_group.all_gather_into_tensor_auto(hidden_states, async_op=True)
        work.wait()
        hidden_states = self._cast_tensor_to_flat_view(work.output)
        # all_routing_weights
        work = self.parallel_config.ep_group.all_gather_into_tensor_auto(routing_weights, async_op=True)
        work.wait()
        routing_weights = self._cast_tensor_to_flat_view(work.output)
                
        return hidden_states, routing_weights, work.handler # return a flatten view
    
    def _gather_reduce_manage_recv_indices(
        self,
        recv_indices: torch.Tensor,
        recv_split: List[int],
        chunk_size: int,
    ):
        recv_split: torch.Tensor = recv_indices.new_tensor(recv_split)
        offset = torch.repeat_interleave(recv_split.new_tensor([_ for _ in range(self.ep_size)]) * chunk_size, recv_split)
        recv_indices = recv_indices + offset
        return recv_indices
        
    def _gather_reduce_slicing_and_shared_experts(
        self,
        hidden_states_for_shared_expert_compute: Optional[torch.Tensor],
        all_hidden_states: torch.Tensor,
        all_routing_weights: torch.Tensor,
        weight_top_k: torch.LongTensor,
        token_indices: torch.LongTensor,
    ):
        '''
            return:
            hidden_states_for_shared_expert_compute, 
            sliced_routing_weights, 
            sliced_hidden_states
        '''
        shared_experts = self.get_shared_experts()
        if shared_experts is not None:
            hidden_states_for_shared_expert_compute = shared_experts._compute(hidden_states_for_shared_expert_compute)
        elif hidden_states_for_shared_expert_compute is not None:
            raise ValueError("hidden states for shared experts is a real parameter, but no shared experts")
            
        sliced_routing_weights = all_routing_weights[token_indices, weight_top_k] # [num_compute]
        sliced_hidden_states = all_hidden_states[token_indices, :]
        return hidden_states_for_shared_expert_compute, sliced_routing_weights, sliced_hidden_states        
    
    def _gather_reduce_expert_compute(
        self, 
        hidden_states: torch.Tensor, 
        block_sizes: Union[torch.Tensor, List[int]], 
        token_indices: torch.Tensor,
        sliced_routing_weights: torch.Tensor, 
        chunk_size: int,
    ):
        '''
            1. compute moe
            2. write to gathered tensor
        '''
        if self.print_moe_compute_time:
            torch.cuda.synchronize()
            time0 = time.time()
        hidden_states = self.experts(hidden_states, block_sizes)
        if self.print_moe_compute_time:
            torch.cuda.synchronize()
            time1 = time.time()
            show_rank_print(f"moe compute time: {time1-time0}", 0)
        
        hidden_states = sliced_routing_weights.unsqueeze(-1) * hidden_states
        
        hidden_size = hidden_states.shape[-1]
        output_hidden_states = hidden_states.new_zeros((chunk_size*self.ep_size, hidden_size))
        output_hidden_states.index_add_(0, token_indices, hidden_states)
        return output_hidden_states
            
    def _gather_reduce_expert_output_communicate(
        self,
        expert_output_tensor: torch.Tensor, 
    ):
        expert_output_work = self.ep_group.reduce_scatter_tensor_auto(expert_output_tensor, async_op=True)
        expert_output_work.wait()
        return expert_output_work.output, expert_output_work.handler    
    
    def gather_reduce_chunk_routing_compute(self, chunk: ChunkBlock):
        # residual add
        (residual, hidden_states) = chunk.active_variables
        residual += hidden_states # must be in-place
        
        # prepare for routing
        hidden_states = residual
        normed_hidden_states = self.post_attention_layernorm(hidden_states).view(-1, self.hidden_size)
        (chunked_hidden_states, routing_weights, selected_experts, router_logits, chunk_size
         ) = self._chunk_routing_compute(normed_hidden_states)
        local_tok_all_expert_count_2d = self._all_expert_count_compute(selected_experts)
        
        if self.shared_experts is None:
            normed_hidden_states = None # no need to keep the normed tensor

        chunk.active_variables = (
            chunked_hidden_states, 
            chunk_size, 
            routing_weights, 
            selected_experts,
            local_tok_all_expert_count_2d,
            normed_hidden_states, 
            residual
        )
        
    def gather_reduce_expert_data_communicate(self, chunk: ChunkBlock):
        (
            chunked_hidden_states, 
            chunk_size, 
            routing_weights, 
            selected_experts,
            local_tok_all_expert_count_2d,
            normed_hidden_states, 
            residual
        ) = chunk.active_variables
        
        # all_xxx: [ep_size, local_t, k]
        (
            global_tok_all_expert_count_2d, global_tok_local_expert_count_2d, all_selected_experts, handler
        ) = self._gather_reduce_expert_data_communicate(local_tok_all_expert_count_2d, selected_experts)
        DEBUG_GATHER_REDUCE = False
        if DEBUG_GATHER_REDUCE:
            all_selected_experts = self._cast_tensor_to_flat_view(all_selected_experts)
            global_expert_mask = torch.nn.functional.one_hot(all_selected_experts, num_classes=self.num_global_experts).transpose(0, 2)[self.local_expert_mask_retriever,:,:] # [ep_rank, local_expert_logic_id, k, ep_size*local_t]
            ### retrieve local mask and count
            local_alltok_expert_mask = global_expert_mask[self.ep_rank, ...] # [local_expert_id, k, ep_size*local_t]
            _, weight_top_k, token_indices = torch.where(local_alltok_expert_mask) 
            chunk.weight_top_k = weight_top_k
            chunk.token_indices = token_indices
        
        chunk.comm_handler = handler
        chunk.global_tok_all_expert_count_2d = global_tok_all_expert_count_2d
        chunk.global_tok_local_expert_count_2d = global_tok_local_expert_count_2d
        ### The expert count has not been synced, so do not use the value now
        
        chunk.active_variables = (
            chunked_hidden_states, 
            routing_weights, 
            selected_experts,
            local_tok_all_expert_count_2d,
            chunk_size, 
            normed_hidden_states, 
            residual
        )
        
    def gather_reduce_prepare_token_communicate(self, chunk: ChunkBlock):
        (
            chunked_hidden_states, 
            routing_weights, 
            selected_experts,
            local_tok_all_expert_count_2d,
            chunk_size, 
            normed_hidden_states, 
            residual
        ) = chunk.active_variables
        
        (
            send_token_split,
            recv_token_split,
            send_weight_top_k, 
            send_token_indices, 
            recv_weight_top_k,
            recv_token_indices,
        ) = self._gather_reduce_prepare_token_communicate(
            selected_experts,
            local_tok_all_expert_count_2d,
            chunk.global_tok_local_expert_count_2d,
        )
        
        # chunk.send_weight_top_k = send_weight_top_k
        # chunk.send_indices = send_indices
        # chunk.send_token_split = send_token_split
        # chunk.recv_token_split = recv_token_split
                
        chunk.active_variables = (
            chunked_hidden_states,
            routing_weights,
            send_token_split,
            recv_token_split,
            send_weight_top_k, 
            send_token_indices, 
            recv_weight_top_k,
            recv_token_indices,
            chunk_size,
            normed_hidden_states,
            residual,
        )
        
    
    def gather_reduce_token_communicate(self, chunk: ChunkBlock):
        (
            hidden_states,
            routing_weights,
            send_token_split,
            recv_token_split,
            send_weight_top_k, 
            send_token_indices, 
            recv_weight_top_k,
            recv_token_indices,
            chunk_size,
            normed_hidden_states,
            residual,
        ) = chunk.active_variables
                
        hidden_states, routing_weights, handler = self._gather_reduce_token_communicate(
            hidden_states, 
            routing_weights,
            send_token_split,
            recv_token_split,
            send_weight_top_k, 
            send_token_indices, 
            recv_weight_top_k,
            recv_token_indices,
            chunk.extra_cuda_events[0],
        )
        chunk.comm_handler = handler
        chunk.compute_wait_comm = False # no need to wait
        
        chunk.active_variables = (
            hidden_states,
            routing_weights,
            recv_weight_top_k,
            recv_token_indices, 
            recv_token_split,
            chunk_size,
            normed_hidden_states,
            residual,
        )
    
    def gather_reduce_slicing_and_shared_experts(self, chunk: ChunkBlock):
        (
            all_hidden_states,
            all_routing_weights,
            recv_weight_top_k,
            recv_token_indices, 
            recv_token_split,
            chunk_size,
            hidden_states_for_shared_expert_compute,
            residual,
        ) = chunk.active_variables
        
        torch.cuda.current_stream().wait_event(chunk.extra_cuda_events[0])
        # manage token indices, as it is communicated from a local perspective while token is gathered as global
        recv_token_indices = self._gather_reduce_manage_recv_indices(recv_token_indices, recv_token_split, chunk_size)
        
        # add to residual if needed
        (
            hidden_states_for_shared_expert_compute,
            sliced_routing_weights, 
            sliced_hidden_states, 
        ) = self._gather_reduce_slicing_and_shared_experts(
            hidden_states_for_shared_expert_compute,
            all_hidden_states,
            all_routing_weights,
            recv_weight_top_k,
            recv_token_indices,
        )
        
        chunk.active_variables = (
            hidden_states_for_shared_expert_compute,
            sliced_routing_weights, 
            sliced_hidden_states, 
            recv_token_indices,
            chunk_size,
            residual,
        )
    
    def shared_expert_result_communicate(self, chunk: ChunkBlock):
        (
            hidden_states_for_shared_expert_compute,
            sliced_routing_weights, 
            sliced_hidden_states, 
            recv_token_indices,
            chunk_size,
            residual,
        ) = chunk.active_variables
        
        hidden_states_for_shared_expert_compute, handler = self._shared_expert_result_communicate(hidden_states_for_shared_expert_compute)
        ## if no shared_expert, the handler will be from the previous stage
        if self.has_shared:
            chunk.comm_handler = handler
        else:
            chunk.comm_handler = None
        # add to residual
        if hidden_states_for_shared_expert_compute is not None:
            residual = _add_to_residual(residual, hidden_states_for_shared_expert_compute)
        
        chunk.active_variables = (
            sliced_routing_weights, 
            sliced_hidden_states, 
            recv_token_indices,
            chunk_size,
            residual,
        )
    
    def gather_reduce_expert_compute(self, chunk: ChunkBlock):
        (
            sliced_routing_weights, 
            hidden_states, 
            recv_token_indices,
            chunk_size,
            residual,
        ) = chunk.active_variables
        
        hidden_states = self._gather_reduce_expert_compute(
            hidden_states,
            chunk.global_tok_local_expert_count_2d_flat_list,
            recv_token_indices,
            sliced_routing_weights, 
            chunk_size,
        )
        hidden_states = self._cast_tensor_to_gathered_view(hidden_states, self.ep_size)
        ## Now hidden_states is in gathered form
        
        chunk.active_variables = (
            hidden_states, 
            residual,
        )
    
    def gather_reduce_expert_output_communicate(self, chunk: ChunkBlock):
        (
            hidden_states, 
            residual,
        ) = chunk.active_variables
        
        # torch.cuda.synchronize()
        # time0 = time.time()
        hidden_states, handler = self._gather_reduce_expert_output_communicate(hidden_states,)
        chunk.comm_handler = handler
        # torch.cuda.synchronize()
        # time1 = time.time()
        # show_rank_print(f"expert comm time: {time1 - time0}", 0)
                
        chunk.active_variables = (
            residual,
            hidden_states,
        )
    
    def gather_reduce_chunk_output_compute(self, chunk: ChunkBlock):
        pass

    def gather_reduce_chunk_output_communicate(self, chunk: ChunkBlock):
        """
            Residual addition if it is the last step.
        """
        self.all2all_gather_chunk_output_communicate(chunk)
        return
        (residual, hidden_states) = chunk.active_variables
        # show_rank_print(f"residual: {residual.sum()}, hidden_states: {hidden_states.sum()}", 0)
        hidden_states, handler = self._gather_chunk_output_communicate(hidden_states)
        chunk.comm_handler = handler
        chunk.active_variables = (residual, hidden_states)
        
