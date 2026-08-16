import torch
import itertools
from torch.nn.attention.bias import causal_lower_right

from typing import Dict, List, OrderedDict, Tuple

from ..generation.cache_utils import TreeDynamicCache
from ..generation.utils.attention_mask import _prepare_4d_causal_attention_mask_with_cache_position
from ..modeling_utils.utils import _prepare_1d_position_id_from_2d_attention_mask
from ..generation.parallel.parallel_utils import get_device, get_global_rank

class Request:
    def __init__(
        self,
        request_id: int,
        token_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        offload_kv: bool = False,
    ):
        self.request_id = request_id
        self.token_ids = token_ids
        assert attention_mask.dim() == 2
        self.attention_mask_2d = attention_mask
        
        self.offload_kv = offload_kv
        
        self.input_length = token_ids.shape[-1]
        
        self.finished_request_length = 0
        self.next_request_length = -1
        self.attention_causal_bias = None
        self.cache_position = None
        self.kv_cache = None
        self.next_input_ids = None
        self.next_position_ids = None
        

    
    def is_before_prefilling(
        self,
    ):
        return self.kv_cache is None

    def prepare_next_step(
        self,
        next_request_length: int,
        need_attention_causal_bias: bool,
    ):
        self.next_request_length = next_request_length
        new_length = self.finished_request_length + self.next_request_length
        if self.kv_cache is None:
            self.kv_cache = TreeDynamicCache(offload_kv=self.offload_kv)
            self.cache_position = torch.arange(self.finished_request_length, new_length, device=get_device(), dtype=torch.int64)
            self.kv_cache.prepare_next_step(self.cache_position[self.finished_request_length:])
        
        # self.attention_mask_4d = _prepare_4d_causal_attention_mask_with_cache_position(
        #     self.attention_mask_2d,
        #     sequence_length=self.next_request_length,
        #     target_length=new_length,
        #     cache_position=self.cache_position,
        #     batch_size=1,
        # ) if need_attention_causal_bias else None
        self.attention_causal_bias = causal_lower_right(self.next_request_length, new_length) if need_attention_causal_bias else None

        self.next_input_ids = self.token_ids[self.finished_request_length: new_length] # 1-d
        self.next_position_ids = _prepare_1d_position_id_from_2d_attention_mask(self.attention_mask_2d)
        assert self.next_input_ids.dim() == 1
        
        
class RequestHub:
    def __init__(
        self,
        model_config,
        offload_kv: bool = False,
        raw_input_pad_side = 'left',
    ):
        assert raw_input_pad_side == 'left'
        self.logical_requests: Dict[int, Request] = {}
        self._id_generator = itertools.count(start=0)
        self.hidden_size = model_config.hidden_size
        self.head_dim = getattr(model_config, "head_dim", model_config.hidden_size // model_config.num_attention_heads)
        
        self.offload_kv = offload_kv
                
    def __getitem__(self, req_id):
        return self.logical_requests[req_id]
    
    def get_new_id(self) -> int:
        return next(self._id_generator)
    
    def append_new_inputs(
        self,
        raw_input_ids: torch.LongTensor,
        raw_attention_mask_2d: torch.LongTensor,
    ):
        assert raw_input_ids.dim() == 2 and raw_attention_mask_2d.dim() == 2
        _batch_size, _seq_len = raw_input_ids.shape
        assert _batch_size == raw_attention_mask_2d.shape[0] and _seq_len == raw_attention_mask_2d.shape[-1]
        
        for _batch_i in range(_batch_size):
            _attention_mask = raw_attention_mask_2d[_batch_i]
            _effective_length = _attention_mask.sum() # TODO: more flexible
            token_ids = raw_input_ids[_batch_i, -_effective_length:]
            _attention_mask = _attention_mask[None,-_effective_length:]
                        
            new_request = Request(
                self.get_new_id(),
                token_ids, # 1d token
                _attention_mask, # 2d, to prepare 4d attention mask
                offload_kv=self.offload_kv,
            )
            
            self.logical_requests.update({new_request.request_id: new_request})
    
    def step(
        self,
        need_attention_causal_bias: bool = False,
    ):
        # TODO: schedule requests
        self.next_step_logical_ids_and_lengths = OrderedDict({req_id:req.input_length for req_id,req in self.logical_requests.items()})
        # sort by next length
        
        self.next_step_logical_ids = list(self.next_step_logical_ids_and_lengths.keys())
        self.next_step_input_length = torch.tensor(list(self.next_step_logical_ids_and_lengths.values()), dtype=torch.int64, device=get_device())
        self.next_step_input_sum_length = self.next_step_input_length.sum()
        
        # input_id_list = []
        # pos_id_list = []
        for request_id in self.next_step_logical_ids_and_lengths:
            request = self.logical_requests[request_id]
            
            if request.is_before_prefilling():
                # prefill
                request.prepare_next_step(self.next_step_logical_ids_and_lengths[request_id],
                                          need_attention_causal_bias=need_attention_causal_bias)
            else:
                # decode
                raise NotImplementedError
    
    def get_inputs_from_req_metadata(self, req_metadata: Dict[int, Tuple[int, int]]):
        '''
            Return the inputs of requests in the req_metadata.
            Tokens and pos_ids are concatenated.
        '''
        input_id_list = []
        pos_id_list = []
        attention_causal_bias_list = []
        kv_cache_list = []
        inference_logits_to_keep_indices_list = []

        curr_chunk_len = 0
        for request_id, (req_offset, span) in req_metadata.items():
            request = self.logical_requests[request_id]
            input_id_list.append(request.next_input_ids.narrow(0, req_offset, span))
            pos_id_list.append(request.next_position_ids.narrow(0, req_offset, span))
            attention_causal_bias_list.append(
                # request.attention_mask_4d[:,:, req_offset:(req_offset+span), :(req_offset+span)]
                request.attention_causal_bias if request.attention_causal_bias is not None else None
            )
            kv_cache_list.append(request.kv_cache)
            request_len = request.next_request_length
            if req_offset + span == request_len:
                inference_logits_to_keep_indices_list.append(curr_chunk_len + span - 1)
            curr_chunk_len += span
        ## Concat the input tensors
        input_ids = torch.cat(input_id_list)
        position_ids = torch.cat(pos_id_list)
        return input_ids, position_ids, attention_causal_bias_list, kv_cache_list, inference_logits_to_keep_indices_list
        