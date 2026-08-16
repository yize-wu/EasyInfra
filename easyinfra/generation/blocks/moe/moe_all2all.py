from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from easyinfra.schedule.chunk import ChunkBlock
    
import torch
import time
import json
from torch import nn
import math
from typing import Optional
from typing import Union, List


from easyinfra.generation.parallel.communicator import GroupCommunicator, WorkHandler
from easyinfra.generation.parallel.parallel_configuration import MoeParallelConfig

from easyinfra.generation.utils.compute_utils import _add_to_residual
from easyinfra.utils.stats import show_rank_print, stage_rank_print
from easyinfra.utils.tensors import _to_cpu_list
from easyinfra import envs

from .base import _MoeBlock, MoeStageName
from easyinfra.generation.utils.perf import class_with_timing

@class_with_timing
class MoeAll2AllBlock(_MoeBlock):
    
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
        super().__init__(
            layer_idx=layer_idx,
            hidden_size=hidden_size,
            moe_intermediate_size=moe_intermediate_size,
            activation_key=activation_key,
            gate_up_bias=gate_up_bias,
            down_bias=down_bias,
            num_logical_experts=num_logical_experts,
            num_experts_per_tok=num_experts_per_tok,
            norm_topk_prob=norm_topk_prob,
            use_eplb=use_eplb,
            num_first_k_dense_layers=num_first_k_dense_layers,
            chunk_routing_group=chunk_routing_group,
            ep_group=ep_group,
            moe_tp_group=moe_tp_group,
            has_shared=has_shared,
            shared_expert_intermediate_size=shared_expert_intermediate_size,
            shared_activation_key=shared_activation_key,
            shared_gate_up_bias=shared_gate_up_bias,
            shared_down_bias=shared_down_bias,
            shared_expert_tp_group=shared_expert_tp_group,
            num_global_experts=num_global_experts,
            parallel_config=parallel_config,
        )
        
        self.transfer_routing_weights = False
        self.stages = ((MoeStageName.ROUTING_DISPATCH, self.all2all_chunk_routing_compute, self.all2all_moe_dispatch_communicate),
                       (MoeStageName.SHARED_EXPERT, self.all2all_slicing_weight_and_shared_experts, self.shared_expert_result_communicate),
                       (MoeStageName.MOE, self.all2all_expert_compute, self.all2all_moe_combine_communicate),
                       (MoeStageName.GATHER, self.all2all_chunk_output_compute, self.all2all_gather_chunk_output_communicate),
                       )       
        self.stage_has_communication = (
            self.ep_size > 1,
            self.shared_expert_tp_size > 1,
            self.ep_size > 1,
            self.chunk_routing_size > 1,
        )             
                
    def _expert_data_communicate(
        self,
        global_tok_all_expert_count_2d: torch.Tensor,
        local_tok_all_expert_count: torch.Tensor, 
        # local_tok_device_count: torch.Tensor,
    ):
        '''
            local_expert_count: [ep_rank, num_local_experts]
        '''
        # send expert data
        # global_tok_device_count_work = self.ep_data_group.all_reduce(local_tok_device_count, async_op=True)
        # global_tok_device_count_work.wait()
        # handler = global_tok_device_count_work.handler
        global_tok_all_expert_count_work = self.ep_data_group.all_gather_into_tensor(global_tok_all_expert_count_2d, local_tok_all_expert_count, async_op=True)
        global_tok_all_expert_count_work.wait()
        handler = global_tok_all_expert_count_work.handler

                
        return None, handler

        
    def _all2all_prepare_token_dispatch(
        self, 
        send_hidden_states: torch.Tensor, 
        routing_weights: torch.Tensor,
        recv_length: int, 
    ):
        '''
            return:
                send_hidden_states,
                recv_hidden_states,
                send_indices, 
                send_weight_top_k, 
        '''                 
        # dev = selected_experts.device
        # selected_experts_sort_idx = selected_experts.view(-1).sort().indices 
        # send_indices = selected_experts_sort_idx // self.num_experts_per_tok
        # _topk_aux = torch.arange(self.num_experts_per_tok, device=dev).view(1,self.num_experts_per_tok).expand(chunk_size,self.num_experts_per_tok) 
        # send_weight_top_k = _topk_aux.reshape(-1)[selected_experts_sort_idx]
        
        # # slice hidden states
        # send_hidden_states = hidden_states[send_indices, :] # [num_send, hidden_size]
        # # slice routing_weights if needed
        # if self.transfer_routing_weights:
        #     send_routing_weights = routing_weights[send_indices, send_weight_top_k]
        # else:
        #     send_routing_weights = None
        
        recv_hidden_states = send_hidden_states.new_empty((recv_length, self.hidden_size)) # [sum_length, hidden_dim]
        if self.transfer_routing_weights:
            recv_routing_weights = routing_weights.new_empty((recv_length,))
        else:
            recv_routing_weights = None
        return (
            recv_hidden_states,
            recv_routing_weights,
        )
    
    def _all2all_moe_dispatch_communicate(
        self, 
        send_hidden_states: torch.Tensor, 
        # recv_hidden_states: torch.Tensor, 
        send_token_split: List[int], 
        recv_token_split: List[int], 
    ):
        '''
            Returns: 
            recv_routing_weights, [recv_length, k]
            recv_hidden_states,  [recv_length, hidden_dim]
            all_hidden_states_work.handler
        '''
        recv_length = sum(recv_token_split)
        recv_hidden_states = send_hidden_states.new_empty((recv_length, self.hidden_size))
        recv_hidden_states_work = self.ep_group.all_to_all_single(recv_hidden_states, send_hidden_states, recv_token_split, send_token_split, async_op=True)
        recv_hidden_states_work.wait()
        
        return recv_hidden_states_work.output, recv_hidden_states_work.handler
    
    def _all2all_slicing_weight_and_shared_experts(
        self,
        hidden_states: Optional[torch.Tensor],
        routing_weights: torch.Tensor,
        send_indices: torch.Tensor,
        send_weight_top_k: torch.Tensor,
    ):
        # time0 = time.time()
        # shared expert compute
        shared_experts = self.get_shared_experts()
        if shared_experts is not None:
            hidden_states = shared_experts._compute(hidden_states)
        elif hidden_states is not None:
            raise ValueError("hidden states as a real parameter, but no shared experts")
        
        # time1 = time.time()
        ## for local-computed routing weights, slicing is here
        if not self.transfer_routing_weights:    
            sliced_routing_weights = routing_weights[send_indices, send_weight_top_k] # [num_send]
        else:
            sliced_routing_weights = None
        # time2 = time.time()
        # show_rank_print(f"{time1 - time0}, {time2 - time1}", 0)
        
        return sliced_routing_weights, hidden_states
    
    def _shared_expert_result_communicate(
        self, hidden_states: Optional[torch.Tensor]
    ):
        if not self.has_shared:
            return None, None
        else:
            work = self.shared_expert_tp_group.all_reduce(hidden_states, async_op=True)
            work.wait()
            return hidden_states, work.handler
    
    # @torch.compile(mode="default")
    def _all2all_expert_compute_torch_compile(
        self, 
        recv_hidden_states: torch.Tensor, 
        expert_count: torch.Tensor, 
    ):
        '''
            device_expert_mask: [num_local_experts, k, all_t]
        '''
        
        ep_size = self.ep_size
        num_physical_experts = expert_count.shape[-1]
        
        # the offset in the recv_hidden_state
        flat_expert_count = expert_count.reshape(-1)
        ends = flat_expert_count.cumsum(-1)
        starts = ends - flat_expert_count
        
        output_tensor = recv_hidden_states
        for i,physical_expert_id in enumerate(self.physical_experts):
            expert_layer = self.experts[physical_expert_id]
            # find starts and ends
            flat_indices = i + torch.arange(ep_size, device=expert_count.device) * num_physical_experts
            this_ends = ends[flat_indices]
            this_starts = starts[flat_indices]
            
            # form inputs
            current_states = torch.cat([recv_hidden_states[s:e] for s, e in zip(this_starts, this_ends)], dim=0)
            if current_states.shape[0] > 0:
                current_states: torch.Tensor = expert_layer(current_states)
                # write back
                local_ends = (this_ends - this_starts).cumsum(-1)
                local_starts = local_ends - (this_ends - this_starts)
                for j, (s, e) in enumerate(zip(this_starts, this_ends)):
                    output_tensor[s:e] = current_states[local_starts[j]:local_ends[j]]
        
        return output_tensor
    
    
    def _all2all_expert_compute(
        self, 
        hidden_states: torch.Tensor, 
        routing_weights: Optional[torch.Tensor],
        expert_count: torch.Tensor, 
    ):
        '''
            device_expert_mask: [num_local_experts, k, all_t]
        '''
        # if self.print_moe_compute_time:
        #     torch.cuda.synchronize()
        #     time0 = time.time()
        hidden_states = self.experts(hidden_states, expert_count)
        # if self.print_moe_compute_time:
        #     torch.cuda.synchronize()
        #     time1 = time.time()
        #     show_rank_print(f"moe compute time: {time1-time0}", 0)
        if self.transfer_routing_weights:
            assert routing_weights is not None
            hidden_states = hidden_states * routing_weights.unsqueeze(-1)
        return hidden_states
            
    def _all2all_moe_combine_communicate(
        self,
        output_tensor: torch.Tensor, 
        recv_tensor: torch.Tensor,
        send_token_split: List[int], 
        recv_token_split: List[int],
    ):
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
    
    def _all2all_chunk_output_compute(
        self, 
        send_indices: torch.Tensor, 
        chunk_size: int,
        recv_result_hidden_states: torch.Tensor, 
        sliced_routing_weights: torch.Tensor
    ):
        # time0 = time.time()
        chunk_outputs = recv_result_hidden_states.new_zeros((chunk_size, self.hidden_size)) # [chunk, hidden_dim]
        if not self.transfer_routing_weights:
            assert sliced_routing_weights is not None
            recv_result_hidden_states = recv_result_hidden_states * sliced_routing_weights[:,None]
        # time1 = time.time()
        chunk_outputs.index_add_(0, send_indices, recv_result_hidden_states)
        # time2 = time.time()
        # show_rank_print(f"epilogue compute: {time1 - time0}, {time2 - time1}", 0)
        return chunk_outputs
    
    def _gather_chunk_output_communicate(self, chunk_outputs: torch.Tensor):
        if self.chunk_routing_group.group_size > 1:
            final_outputs_work = self.chunk_routing_group.all_gather_into_tensor_auto(chunk_outputs, async_op=True)
            final_outputs_work.wait()
            ## flat
            final_outputs = final_outputs_work.output.view(-1, final_outputs_work.output.shape[-1])
            handler = final_outputs_work.handler
        else:
            final_outputs = chunk_outputs
            handler = WorkHandler(chunk_outputs)
        
        return final_outputs, handler
    
        
    
    
    def all2all_chunk_routing_compute(self, chunk: ChunkBlock):
        stage_rank_print(f"chunk{chunk.chunk_id} in chunk_routing_compute", 0)
        # residual add
        # time0 = time.time()
        (residual, hidden_states) = chunk.active_variables
        chunk.active_variables = ()
        residual = _add_to_residual(residual, hidden_states)
        
        # prepare for routing
        hidden_states = residual
        normed_hidden_states = self.post_attention_layernorm(hidden_states).view(-1, self.hidden_size)
        chunked_hidden_states, routing_weights, selected_experts, router_logits, chunk_size = self._chunk_routing_compute(normed_hidden_states)
        
        # time1 = time.time()
        local_tok_all_expert_count_2d = self._all_expert_count_compute(selected_experts)
        local_tok_device_count = local_tok_all_expert_count_2d.sum(-1)
        # time2 = time.time()
        if self.shared_experts is None:
            normed_hidden_states = None # no need to keep the normed tensor

        dev = selected_experts.device
        selected_experts_sort_idx = selected_experts.view(-1).sort().indices 
        send_indices = selected_experts_sort_idx // self.num_experts_per_tok
        send_weight_top_k = torch.arange(self.num_experts_per_tok, device=dev).view(1,self.num_experts_per_tok).expand(chunk_size, self.num_experts_per_tok).reshape(-1)[selected_experts_sort_idx]
        
        # slice hidden states
        send_hidden_states = chunked_hidden_states[send_indices, :] # [num_send, hidden_size]
        # slice routing_weights if needed
        if self.transfer_routing_weights:
            send_routing_weights = routing_weights[send_indices, send_weight_top_k]
        else:
            send_routing_weights = None

        global_tok_all_expert_count_2d = local_tok_all_expert_count_2d.new_empty((self.ep_size, self.ep_size, self.num_physical_experts))
        chunk.active_variables = (
            local_tok_device_count, 
            local_tok_all_expert_count_2d,
            global_tok_all_expert_count_2d,
            chunk_size, 
            routing_weights, 
            normed_hidden_states, 
            send_hidden_states,
            send_routing_weights,
            send_indices, 
            send_weight_top_k, 
            residual
        )
        # time3 = time.time()
        # show_rank_print(f"{time1 - time0}, {time2 - time1}, {time3 - time2}", 0)
        
    def all2all_moe_dispatch_communicate(self, chunk: ChunkBlock):
        stage_rank_print(f"chunk{chunk.chunk_id} in moe_dispatch_communicate", 0)
        (
            local_tok_device_count, 
            local_tok_all_expert_count_2d,
            global_tok_all_expert_count_2d,
            chunk_size, 
            routing_weights, 
            normed_hidden_states, 
            send_hidden_states,
            send_routing_weights,
            send_indices, 
            send_weight_top_k, 
            residual
        ) = chunk.active_variables
        local_tok_all_expert_count_2d: torch.Tensor
        
        # # show_rank_print(f"layer{self.layer_idx} chunk{chunk.chunk_id}: {residual.sum()}")
        # # show_rank_print(f"layer{self.layer_idx} chunk{chunk.chunk_id}: {local_tok_all_expert_count_2d.tolist()}")
        # # show_rank_print(f"layer{self.layer_idx} chunk{chunk.chunk_id}: {chunk.send_token_split.tolist()}")
        
        global_tok_all_expert_count_2d: torch.Tensor
        _, handler = self._expert_data_communicate(
            global_tok_all_expert_count_2d, 
            local_tok_all_expert_count_2d,
        )
        chunk.comm_handler = handler
        
        ### The expert count has not been synced, so do not use the value now
        device = global_tok_all_expert_count_2d.device
        chunk.global_tok_all_expert_count_2d = global_tok_all_expert_count_2d
        global_tok_all_expert_count_2d.record_stream(chunk.comm_cuda_stream)
                
        ## GPU for sync scheduling, CPU for better overlap
        chunk.global_tok_all_expert_count_2d = global_tok_all_expert_count_2d
        global_tok_all_expert_count_2d = global_tok_all_expert_count_2d.cpu()
        chunk.global_tok_all_expert_count_2d_cpu = global_tok_all_expert_count_2d
        
        chunk.global_tok_device_count_cpu = global_tok_all_expert_count_2d.sum((0,-1))
        global_tok_local_expert_count_2d = global_tok_all_expert_count_2d[:,self.ep_rank,:]
        chunk.global_tok_local_ep_rank_count = global_tok_local_expert_count_2d.sum(-1)
        global_tok_local_expert_count_2d_flat = global_tok_local_expert_count_2d.reshape(-1)            
        
        ### You MUST make sure that all future-required tensors are synchronized here
        chunk.global_tok_local_expert_count_2d_flat_list = _to_cpu_list(global_tok_local_expert_count_2d_flat)
        chunk.send_token_split = _to_cpu_list(local_tok_device_count)
        chunk.recv_token_split = _to_cpu_list(chunk.global_tok_local_ep_rank_count) # [ep_rank]  
        # show_rank_print(f"layer{self.layer_idx} chunk{chunk.chunk_id}: {chunk.global_tok_device_count.tolist()}")

        chunk.global_tok_device_count = chunk.global_tok_device_count_cpu.to(device, non_blocking=True)
        
        local_tok_device_count.record_stream(chunk.comm_cuda_stream)
        local_tok_all_expert_count_2d.record_stream(chunk.comm_cuda_stream)


        send_hidden_states: torch.Tensor
        send_routing_weights: torch.Tensor
        
        if self.transfer_routing_weights:
            recv_routing_weights, handler = self._all2all_moe_dispatch_communicate(
                # send_routing_weights, recv_routing_weights, chunk.send_token_split, chunk.recv_token_split
                send_routing_weights, chunk.send_token_split, chunk.recv_token_split
            )
            send_routing_weights.record_stream(chunk.comm_cuda_stream)
        else:
            recv_routing_weights = None        
        moe_hidden_states, handler = self._all2all_moe_dispatch_communicate(
            # send_hidden_states, moe_hidden_states, chunk.send_token_split, chunk.recv_token_split
            send_hidden_states, chunk.send_token_split, chunk.recv_token_split
        )
        send_hidden_states.record_stream(chunk.comm_cuda_stream)
        
        chunk.comm_handler = handler
        chunk.compute_wait_comm = False # no need to wait
        
        chunk.active_variables = (
            moe_hidden_states,
            recv_routing_weights,
            routing_weights,
            send_indices, 
            send_weight_top_k, 
            chunk_size,
            normed_hidden_states,
            residual,
        )        
        
    def all2all_slicing_weight_and_shared_experts(self, chunk: ChunkBlock):
        stage_rank_print(f"chunk{chunk.chunk_id} in shared_experts_compute", 0)
        (
            moe_hidden_states,
            recv_routing_weights,
            routing_weights,
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
        ) = self._all2all_slicing_weight_and_shared_experts(
            hidden_states_for_shared_expert_compute,
            routing_weights,
            send_indices,
            send_weight_top_k,
        )
        
        chunk.active_variables = (
            moe_hidden_states,
            sliced_routing_weights,
            recv_routing_weights,
            send_indices, 
            hidden_states_for_shared_expert_compute,
            chunk_size,
            residual,
        )
    
    def shared_expert_result_communicate(self, chunk: ChunkBlock):
        stage_rank_print(f"chunk{chunk.chunk_id} in shared_experts_communicate", 0)
        (
            moe_hidden_states,
            sliced_routing_weights,
            recv_routing_weights,
            send_indices, 
            hidden_states_for_shared_expert_compute,
            chunk_size,
            residual,
        ) = chunk.active_variables
        
        shared_expert_output_tensor, handler = self._shared_expert_result_communicate(hidden_states_for_shared_expert_compute)
        ## if no shared_expert, the handler will be from the previous stage
        if self.has_shared:
            chunk.comm_handler = handler
            hidden_states_for_shared_expert_compute.record_stream(chunk.comm_cuda_stream)
        
        chunk.active_variables = (
            moe_hidden_states,
            sliced_routing_weights,
            recv_routing_weights,
            send_indices, 
            shared_expert_output_tensor,
            chunk_size,
            residual,
        )
    
    def all2all_expert_compute(self, chunk: ChunkBlock):
        stage_rank_print(f"chunk{chunk.chunk_id} in expert_compute", 0)
        (
            moe_hidden_states,
            sliced_routing_weights,
            recv_routing_weights,
            send_indices, 
            shared_expert_output_tensor,
            chunk_size,
            residual,
        ) = chunk.active_variables
        
        # add to residual
        if shared_expert_output_tensor is not None:
            residual = _add_to_residual(residual, shared_expert_output_tensor)
        # torch.cuda.synchronize()
        # time0 = time.time()
        moe_hidden_states.record_stream(chunk.compute_cuda_stream) ## it might be created on comm stream
        moe_hidden_states = self._all2all_expert_compute(
            moe_hidden_states,
            recv_routing_weights,
            chunk.global_tok_local_expert_count_2d_flat_list,
        )
        moe_comm_recv_tensor = moe_hidden_states.new_empty((sum(chunk.send_token_split), self.hidden_size))
        # torch.cuda.synchronize()
        # time1 = time.time()
        # show_rank_print(f"expert compute time: {time1 - time0}", 0)
        
        chunk.active_variables = (
            moe_hidden_states, 
            moe_comm_recv_tensor,
            sliced_routing_weights,
            send_indices, 
            chunk_size,
            residual,
        )
    
    def all2all_moe_combine_communicate(self, chunk: ChunkBlock):
        stage_rank_print(f"chunk{chunk.chunk_id} in expert_output_communicate", 0)
        (
            moe_hidden_states, 
            moe_comm_recv_tensor,
            sliced_routing_weights,
            send_indices, 
            chunk_size,
            residual,
        ) = chunk.active_variables
        moe_comm_recv_tensor: torch.Tensor
        
        # torch.cuda.synchronize()
        # time0 = time.time()
        _, handler = self._all2all_moe_combine_communicate(
            moe_hidden_states, 
            moe_comm_recv_tensor,
            chunk.send_token_split, 
            chunk.recv_token_split
        )
        chunk.comm_handler = handler
        ## Record stream
        moe_hidden_states.record_stream(chunk.comm_cuda_stream)
        moe_comm_recv_tensor.record_stream(chunk.comm_cuda_stream)
                
        chunk.active_variables = (
            moe_comm_recv_tensor,
            sliced_routing_weights,
            send_indices, 
            chunk_size,
            residual,
        )
    
    def all2all_chunk_output_compute(self, chunk: ChunkBlock):
        stage_rank_print(f"chunk{chunk.chunk_id} in chunk_output_compute", 0)
        (
            hidden_states, # moe_comm_recv_tensor
            sliced_routing_weights,
            send_indices, 
            chunk_size,
            residual,
        ) = chunk.active_variables
        
        hidden_states = self._all2all_chunk_output_compute(send_indices, chunk_size, hidden_states, sliced_routing_weights)
        chunk.active_variables = (residual, hidden_states)

    def all2all_gather_chunk_output_communicate(self, chunk: ChunkBlock):
        stage_rank_print(f"chunk{chunk.chunk_id} in chunk_output_communicate", 0)
        (residual, hidden_states) = chunk.active_variables
        # show_rank_print(f"residual: {residual.sum()}, hidden_states: {hidden_states.sum()}", 0)
        hidden_states, handler = self._gather_chunk_output_communicate(hidden_states)
        chunk.comm_handler = handler
        chunk.active_variables = (residual, hidden_states)
        
