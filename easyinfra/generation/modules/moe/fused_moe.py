import torch
from torch.nn.parameter import Parameter
from torch import nn
import torch.nn.functional as F
import triton
import triton.language as tl

from easyinfra.generation.parallel.communicator import GroupCommunicator
from easyinfra.generation.modules.moe.native_moe import _MoENoExperts
from easyinfra.generation.modules.weight_load_utils import create_param_tensor_on_device
from easyinfra.generation.modules.activation import ACT2FN
from easyinfra.generation.functions import exclusive_cumsum
from easyinfra.utils.tensors import _to_cpu_list

class FusedMoE(_MoENoExperts):
    '''
        Fused MoE.
    '''
    def __init__(
        self, 
        layer_idx:int, 
        hidden_size: int, 
        moe_intermediate_size: int,
        activation_key: str,
        gate_up_bias: bool,
        down_bias: bool,
        num_logical_experts: int,
        ep_group: GroupCommunicator,
        moe_tp_group: GroupCommunicator,
        phy2log_expert_map: torch.Tensor,
        **kwargs,
    ):
        
        super().__init__(
            layer_idx,
            moe_intermediate_size,
            num_logical_experts,
            ep_group,
            moe_tp_group,
            phy2log_expert_map,
        )

        self.activation_key = activation_key
        self.act_fn = ACT2FN[activation_key]
        factory_kwargs = {'device': None, 'dtype': None}
        self.experts_gate_up_weight = Parameter(torch.empty((self.num_physical_experts, self.moe_tp_intermediate_size*2, hidden_size,), **factory_kwargs))
        self.experts_down_weight = Parameter(torch.empty((self.num_physical_experts, hidden_size, self.moe_tp_intermediate_size,), **factory_kwargs))
        self.has_gate_up_bias = gate_up_bias
        self.has_down_bias = down_bias
        if self.has_gate_up_bias:
            self.experts_gate_up_bias = Parameter(torch.empty((self.num_physical_experts, self.moe_tp_intermediate_size*2,), **factory_kwargs))
        else:
            self.experts_gate_up_bias = None
        if self.has_down_bias:
            self.experts_down_bias = Parameter(torch.empty((self.num_physical_experts, hidden_size,), **factory_kwargs))
        else:
            self.experts_down_bias = None
        
        self.method = TritonExperts(activation_key)
        self.do_permute_unpermute = False
            
        self.loaded_values = {
            "gate_up": [None] * self.num_physical_experts * 2,
            "down": [None] * self.num_physical_experts,
        }
        self.output_dim = 0
        self.proj_names_and_offsets = {
            "gate": ("gate_up", 0),
            "up": ("gate_up", 1),
            "down": ("down", 0),
        }
        self.proj_names_and_block_size = {
            "gate_up": 2,
            "down": 1,
        }

    def weight_loader(self, module_name:str, value:torch.Tensor, dtype:torch.dtype, param_name:str, **kwargs):
        '''
            param_name: "physical_id.{gate|up|down}_proj.weight"
                e.g. "0.gate_proj.weight"
        '''
        param_name_split = param_name.split(".")
        physical_expert_id = int(param_name_split[0])
        _proj_name = param_name_split[1][:-5] # remove '_proj', leave only prefix
        
        proj_name, offset = self.proj_names_and_offsets[_proj_name]
        
        if proj_name not in self.loaded_values:
            raise ValueError(f"project name {proj_name} (from {_proj_name}) not in {list(self.loaded_values.keys())}")
        
        # The memory layout would be: [gate0, up0, gate1, up1, ...]
        block_size = self.proj_names_and_block_size[proj_name]
        
        self.loaded_values[proj_name][block_size * physical_expert_id + offset] = value.to(torch.cuda.current_device())
        if not any(v is None for v in self.loaded_values[proj_name]):
            old_value_name = "experts_" + proj_name + "_" + param_name_split[2] # e.g. experts_gate_up_weight
            old_value: torch.Tensor = getattr(self, old_value_name)
            _reshaped_loaded_values = torch.stack(self.loaded_values[proj_name], dim=0).view(old_value.shape)
            new_value = create_param_tensor_on_device(old_value, _reshaped_loaded_values, dtype)
            del self.loaded_values[proj_name] # release memory
            setattr(self, old_value_name, new_value)
        
        if len(self.loaded_values) == 0:
            del self.loaded_values
        
    def forward_ep(
        self,
        # recv_hidden_states, from all2all
        hidden_states: torch.Tensor, 
        expert_count: torch.Tensor, 
    ):
        # return hidden_states
        
        if hidden_states.shape[0] == 0:
            return hidden_states
        # permute
        if self.do_permute_unpermute:
            ep_size, num_local_experts = expert_count.shape
            if isinstance(hidden_states, torch.Tensor):
                hidden_states_split = hidden_states.split(_to_cpu_list(expert_count.flatten()))
                _index = torch.arange(ep_size*num_local_experts).view(ep_size, num_local_experts).transpose(0,1).reshape(-1)
                permuted_splits = [None] * ep_size * num_local_experts
                for i,j in enumerate(_index):
                    permuted_splits[i] = hidden_states_split[j]
                hidden_states = torch.cat(permuted_splits, dim=0)
                del hidden_states_split, permuted_splits
            else:
                raise NotImplementedError
            
            used_expert_count = expert_count.sum(0, keepdim=True) # [1,num_local_e]
        
        else:
            used_expert_count = expert_count
        # compute
        hidden_states = self.method.forward_ep(
            hidden_states,
            self.experts_gate_up_weight.transpose(1,2),
            self.experts_down_weight.transpose(1,2),
            self.experts_gate_up_bias,
            self.experts_down_bias,
            used_expert_count,
        )
        # unpermute
        if self.do_permute_unpermute:
            if isinstance(hidden_states, torch.Tensor):
                hidden_states_split = hidden_states.split(_to_cpu_list(expert_count.transpose(0,1).flatten()))
                _index = torch.arange(ep_size*num_local_experts).view(ep_size, num_local_experts).transpose(0,1).reshape(-1)
                permuted_splits = [None] * ep_size * num_local_experts
                for i,j in enumerate(_index):
                    permuted_splits[j] = hidden_states_split[i]
                hidden_states = torch.cat(permuted_splits, dim=0)
                del hidden_states_split, permuted_splits
            else:
                raise NotImplementedError
        
        return hidden_states

    def forward_ep_native(
        self,
        recv_hidden_states: torch.Tensor, 
        expert_count: torch.Tensor, 
    ):
        expert_hit = torch.greater(expert_count.sum(0), 0).nonzero().view(-1)
        hit_expert_num = expert_hit.shape[0]
        if hit_expert_num == 0:
            # no need to compute
            assert recv_hidden_states.numel() == 0
            return torch.empty_like(recv_hidden_states)
        
        ep_size = self.ep_size
        # [ep_size, num_local_e]
        hit_expert_count = expert_count[:,expert_hit].view(-1) # [ep_size * num_hit_experts]
        
        recv_hidden_state_splits = recv_hidden_states.split(_to_cpu_list(hit_expert_count)) # [ep_size * num_hit_experts]

        # expert compute
        # _expert_compute_start = record_time_sync()
        _outputs = []
        for i,physical_expert_id in enumerate(expert_hit):
            current_states = torch.cat([t for t in recv_hidden_state_splits[i::hit_expert_num]])
            expert_gate_up = self.experts_gate_up_weight[physical_expert_id]
            expert_down = self.experts_down_weight[physical_expert_id]
            current_states = F.linear(current_states, expert_gate_up).split((self.moe_tp_intermediate_size, self.moe_tp_intermediate_size), dim=-1)
            current_states = F.linear(self.act_fn(current_states[0]) * current_states[1], expert_down)
            _outputs.extend(current_states.split(_to_cpu_list(hit_expert_count)[i::hit_expert_num]))
        
        # re-permute outputs
        _index = torch.arange(len(hit_expert_count)).view(ep_size, hit_expert_num).transpose(0,1).reshape(-1)
        permuted_outputs = [None] * len(hit_expert_count)
        for i,j in enumerate(_index):
            permuted_outputs[j] = _outputs[i]
        
        del _outputs
        
        permuted_outputs = torch.cat(permuted_outputs, dim=0)
        # _expert_compute_end = record_time_sync()
        # self.expert_compute_time += _expert_compute_end - _expert_compute_start
        return permuted_outputs
    
    
def get_config(num_tokens, hidden_size, moe_intermediate_size):
    config = {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 8,
        "SPLIT_K": 1,
        "num_warps": 2,
        "num_stages": 2,
    }
    return config

@triton.jit
def fused_moe_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    b_bias_ptr,
    expert_id_ptr,
    start_ptr,
    span_end_ptr,
    # variables
    expanded_num_tokens,
    N,
    K,
    # strides
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_bbe,  # bias expert stride
    stride_bbn,  # bias N stride
    # constexpr
    HAS_BIAS: tl.constexpr,
    compute_type: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(expanded_num_tokens, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    off_experts = tl.load(expert_id_ptr + pid_m).to(tl.int64)
    
    token_start = tl.load(start_ptr + pid_m).to(tl.int64)
    token_end = tl.load(span_end_ptr + pid_m).to(tl.int64)
    
    offs_token = token_start + tl.arange(0, BLOCK_SIZE_M)
    offs_token_mask = offs_token < token_end
    
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    # offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64))
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    a_ptrs = a_ptr + (
        offs_token[:, None] * stride_am + offs_k[None, :] * stride_ak
    )
    b_ptrs = (
        b_ptr
        + off_experts * stride_be
        + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    )
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the
        # K dimension.
        a = tl.load(
            a_ptrs,
            mask=offs_token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
            other=0.0,
        )
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk    
        
    accumulator = accumulator.to(compute_type)
    
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = offs_token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def invoke_fused_moe_triton_kernel(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    output_hidden_states: torch.Tensor,
    starts: torch.IntTensor,
    ends: torch.IntTensor,
    physical_expert_ids: torch.IntTensor,
    num_m_blocks: int,
    num_valid_tokens: int,
    config,
):
    # stride_am = hidden_states.shape[0] * hidden_states.shape[1]
    # stride_ak = hidden_states.shape[1]
    # stride_be = weight.shape[0] * weight.shape[1] * weight.shape[2]
    # stride_bn = weight.shape[1] * weight.shape[2]
    # stride_bk = weight.shape[2]
    # stride_bbe = bias.shape[0] * bias.shape[1]
    # stride_bbn = bias.shape[1]
    # stride_cm = output_hidden_states.shape[0] * output_hidden_states.shape[1]
    # stride_cn = output_hidden_states.shape[1]
    
    expanded_num_tokens = num_m_blocks * config["BLOCK_SIZE_M"]
    
    K = weight.shape[1]
    N = weight.shape[2]
    if hidden_states.dtype == torch.bfloat16:
        compute_type = tl.bfloat16
    elif hidden_states.dtype == torch.float16:
        compute_type = tl.float16
    elif hidden_states.dtype == torch.float32:
        compute_type = tl.float32
    elif (
        hidden_states.dtype == torch.float8_e4m3fn
        or hidden_states.dtype == torch.float8_e4m3fnuz
    ):
        compute_type = tl.bfloat16
    else:
        raise ValueError(f"Unsupported compute_type: {hidden_states.dtype}")

    
    grid = lambda META: (
        num_m_blocks * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    
    fused_moe_kernel[grid](
        hidden_states,
        weight,
        output_hidden_states,
        bias,
        physical_expert_ids,
        starts,
        ends,
        expanded_num_tokens,
        N,
        K,
        hidden_states.stride(0),
        hidden_states.stride(1),
        weight.stride(0),
        weight.stride(1),
        weight.stride(2),
        output_hidden_states.stride(0),
        output_hidden_states.stride(1),
        bias.stride(0) if bias else 0,  # bias expert stride
        bias.stride(1) if bias else 0,  # bias stride
        bias is not None,
        compute_type,
        **config,
    )
    
    

        
class TritonExperts(nn.Module):
    def __init__(self,
                 activation_key: str):
        super().__init__()
        self.activation = ACT2FN[activation_key]
    
    
    def forward_ep(
        self,
        hidden_states: torch.Tensor,
        experts_gate_up_weight: torch.Tensor,
        experts_down_weight: torch.Tensor,
        experts_gate_up_bias: torch.Tensor | None,
        experts_down_bias: torch.Tensor | None,
        expert_count: torch.Tensor,
    ):
        '''
            We do not care if hidden_states has been permuted.
            hidden_states: (M, hidden_size)
            experts_gate_up_weight: (E, hidden_size, 2*moe_intermediate_size)
            experts_down_weight: : (E, moe_intermediate_size, hidden_size)
        '''
        # get configurations
        expert_count_shape = expert_count.shape
        expert_count_flat = expert_count.reshape(-1)
        # expert_count_transpose = expert_count.transpose(0,1)
        
        num_tokens = expert_count.sum()
        hidden_size = hidden_states.size(-1)
        moe_intermediate_size = experts_down_weight.size(1)
        
        config = get_config(
            num_tokens, hidden_size, moe_intermediate_size
        )
        
        # # [ep_size, num_local_e]
        # end = expert_count_flat.cumsum(-1)
        # start = end - expert_count_flat.view(-1)
        # end = end.view(expert_count_shape).transpose(0,1).reshape(-1) # [num_local_e, ep_size]
        # seg_start = start.view(expert_count_shape).transpose(0,1).reshape(-1) # [num_local_e, ep_size]
        
        BLOCK_SIZE_M = config["BLOCK_SIZE_M"]
        each_seg_num_m_blocks = (expert_count_flat + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M # [ep_size * num_local_e]
        total_num_m_blocks = each_seg_num_m_blocks.sum()
        
        # start
        starts = torch.arange(total_num_m_blocks, dtype=torch.int64, device=expert_count_flat.device) * BLOCK_SIZE_M
        # exclusive_cumsum_num_m_blocks = exclusive_cumsum(each_seg_num_m_blocks)
        # ecumsum_expert_count_flat = exclusive_cumsum(expert_count_flat)
        shorter = torch.empty_like(expert_count_flat)
        shorter[1:] = (-expert_count_flat % BLOCK_SIZE_M)[:-1]
        shorter[0] = 0
        starts = starts - shorter.cumsum(-1).repeat_interleave(each_seg_num_m_blocks)
        # starts = starts - exclusive_cumsum(ecumsum_expert_count_flat % BLOCK_SIZE_M).repeat_interleave(each_seg_num_m_blocks)
        ends = torch.empty_like(starts)
        ends[:-1] = starts[1:]
        ends[-1] = num_tokens
        # (expert_count_flat.cumsum(-1) - expert_count_flat).repeat_interleave(each_seg_num_m_blocks)
        # (exclusive_cumsum_num_m_blocks * BLOCK_SIZE_M).repeat_interleave(each_seg_num_m_blocks)
        
        # starts = []
        # for each_block_num in each_num_m_blocks:
        #     this_range = seg_start + torch.arange(each_block_num, device=expert_count.device, dtype=expert_count.dtype) * config["BLOCK_SIZE_M"]
        #     starts.append(this_range)
        # starts: torch.Tensor = torch.cat(start, dim=-1)
        
        # expanded_num_tokens = each_num_m_blocks.sum() * config["BLOCK_SIZE_M"]
        physical_expert_ids = torch.arange(expert_count_shape[1], device=expert_count.device, dtype=expert_count.dtype)[None,:].expand(expert_count_shape).reshape(-1).repeat_interleave(each_seg_num_m_blocks)
        
        num_valid_tokens = hidden_states.shape[0]
        
        # gate_up
        output_hidden_states = hidden_states.new_empty((num_valid_tokens, moe_intermediate_size*2))
        invoke_fused_moe_triton_kernel(
            hidden_states,
            experts_gate_up_weight,
            experts_gate_up_bias,
            output_hidden_states,
            starts,
            ends,
            physical_expert_ids,
            starts.size(-1),
            num_valid_tokens,
            config,
        )
        # activation
        output_hidden_states_splits = output_hidden_states.split((moe_intermediate_size, moe_intermediate_size), 1)
        output_hidden_states = self.activation(output_hidden_states_splits[0]) * output_hidden_states_splits[1]
        del output_hidden_states_splits
        
        # down
        hidden_states = output_hidden_states
        output_hidden_states = hidden_states.new_empty((num_valid_tokens, hidden_size))
        invoke_fused_moe_triton_kernel(
            hidden_states,
            experts_down_weight,
            experts_down_bias,
            output_hidden_states,
            starts,
            ends,
            physical_expert_ids,
            starts.size(-1),
            num_valid_tokens,
            config,
        )
        
        return output_hidden_states
