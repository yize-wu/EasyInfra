import torch
import torch.distributed as dist
from typing import  Optional, Tuple, List, Dict, Any
from .configuration_utils import (
    TreeAttentionConfig,
    _DEFAULT_TREE_ATTENTION_CONFIG_DICT,
)
from .attention_mask import (
    _prepare_4d_causal_attention_mask_with_cache_position
)


def _check_user_tree_attention_config(key: str, value: Any):
    if key == "depth" and value <= 0:
        raise ValueError(f"tree depth must > 0, but {value}")
    if key == "top_k" and value <= 0:
        raise ValueError(f"tree top_k must > 0, but {value}")

def _pop_tree_attention_config(model_kwargs: Dict):
    """
        Get tree draft config, and return model_kwargs
    """
    if model_kwargs.get("enable_tree_attention", False) != True:
        raise ValueError
    tree_attention_config_dict = {}
    for key in _DEFAULT_TREE_ATTENTION_CONFIG_DICT:
        value_from_model_kwargs = model_kwargs.pop(key, None)
        if value_from_model_kwargs is None:
            value_from_model_kwargs = _DEFAULT_TREE_ATTENTION_CONFIG_DICT[key]
            print(f"rank {dist.get_rank()} use default tree config: {key}={value_from_model_kwargs}")
        tree_attention_config_dict[key] = value_from_model_kwargs
        _check_user_tree_attention_config(key, tree_attention_config_dict[key])
        
    tree_attention_config = TreeAttentionConfig(**tree_attention_config_dict)
    
    return tree_attention_config


def _prepare_tree_verification_position_ids(
    candidate_input_ids_length: int,
    flat_candidate_length: int,
    past_kv_length:int, 
    candidate_length:int,
    top_k:int,
    batch_size: int,
    device: torch.device,
):
    factory_kwargs = {"dtype":torch.long, "device":device}
    # sequence_length = prefill_length + flat_candidate_length
    # candidate_length = cache_length + sequence_length = cache_length + prefill_length + flat_candidate_length
    sequence_length = candidate_input_ids_length - past_kv_length
    prefill_length = sequence_length - flat_candidate_length
    
    # from past_kv_length to prefill is sequential, after prefill is tree
    position_ids_prefill = torch.arange(past_kv_length, past_kv_length + prefill_length, **factory_kwargs)
    position_ids_tree = torch.arange(past_kv_length + prefill_length, past_kv_length + prefill_length + candidate_length, **factory_kwargs)
    position_ids_tree = position_ids_tree[:,None].expand(candidate_length, top_k).flatten()
    position_ids = torch.cat([position_ids_prefill, position_ids_tree], dim=-1).view(1,sequence_length).expand(batch_size, sequence_length)
    return position_ids

def _prepare_tree_verification_cache_position(
    candidate_input_ids_length: int,
    past_kv_length:int, 
    device: torch.device,
):
    # all sequential
    factory_kwargs = {"dtype":torch.long, "device":device}
    return torch.arange(past_kv_length, candidate_input_ids_length, **factory_kwargs)

def _prepare_tree_verification_causal_mask(
    attention_mask_2d: torch.LongTensor,
    candidate_input_ids_length: int,
    flat_candidate_length: int,
    cache_position: torch.LongTensor,
    retrieve_indices: torch.Tensor,
    past_kv_length:int, 
    dtype: torch.dtype,
    device: torch.device,
    min_dtype: Optional[float] = None,
):
    min_dtype = torch.finfo(dtype).min if min_dtype is None else min_dtype
    factory_kwargs = {"dtype":dtype, "device":device}
    
    batch_size = retrieve_indices.shape[0]
    # sequence_length = prefill_length + flat_candidate_length
    # candidate_length = cache_length + sequence_length = cache_length + prefill_length + flat_candidate_length
    sequence_length = candidate_input_ids_length - past_kv_length
    prefill_length = sequence_length - flat_candidate_length

    # cache_attention_mask is zeros, [sequence_length, past_kv_length]
    # TODO: use 2d mask to get real cache_mask and prefill_mask
    # cache_attention_mask = torch.zeros((batch_size, 1, sequence_length, past_kv_length), **factory_kwargs)
    
    # # prefill_attention_mask is diagnol, 
    # prefill_attention_mask = torch.full((batch_size, 1, prefill_length, prefill_length), fill_value=min_dtype, **factory_kwargs).triu(diagonal=1)
    # prefill_attention_mask2 = torch.zeros((batch_size, 1, flat_candidate_length, prefill_length), **factory_kwargs)
    # prefill_attention_mask = torch.cat([prefill_attention_mask, prefill_attention_mask2], dim=-2)
    
    # [sequence_length, non_candidate_length]
    non_candidate_attention_mask = _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask=attention_mask_2d,
        sequence_length=sequence_length,
        target_length=candidate_input_ids_length-flat_candidate_length,
        dtype=dtype,
        device=device,
        cache_position=cache_position,
        batch_size=attention_mask_2d.shape[0],
    )
    
    # tree_attention_mask is from retrieve indices, 
    # [sequence_length, flat_candidate_length] -> [prefill_length+flat_candidate_length, flat_candidate_length]
    # [prefill_length, flat_candidate_length], prefill ids no look to candidate ids
    candidate_attention_mask0 = torch.full((batch_size, 1, prefill_length, flat_candidate_length), fill_value=min_dtype, **factory_kwargs)
    
    # [flat_candidate_length, flat_candidate_length]
    # retrieve_indices: [bsz, flat_candidate_length, depth] and candidate tokens are 1-1
    candidate_attention_mask = torch.full((batch_size, 1, flat_candidate_length, flat_candidate_length), fill_value=min_dtype, **factory_kwargs)
    candidate_attention_mask.scatter_(dim=-1, index=retrieve_indices.unsqueeze(1), value=0.0)
    # retrieve_indices does not consider triu, do it
    upper_tri = torch.triu(torch.full((batch_size,1,flat_candidate_length,flat_candidate_length), fill_value=True, dtype=torch.bool, device=device), diagonal=1)
    candidate_attention_mask.masked_fill_(upper_tri, value=min_dtype)
    candidate_attention_mask = torch.cat([candidate_attention_mask0, candidate_attention_mask], dim=-2)
    
    attention_mask = torch.cat([non_candidate_attention_mask, candidate_attention_mask], dim=-1)
    return attention_mask

def _update_tree_causal_mask_from_retrieve_indices(
    attention_mask: torch.Tensor,
    topk_retrieve_indices: torch.Tensor,
    non_tree_length: int,
    dtype: torch.dtype,
    device: torch.device,
    batch_size: int = None,
    min_dtype: Optional[float] = None,
):
    """
        retrieve_indices: [bsz, top_k, cur_depth]
    """
    # if attention_mask.shape[-2] != top_k and attention_mask.shape[-1] != non_tree_length + 1:
    #     raise ValueError
    if batch_size is None:
        batch_size = attention_mask.shape[0]
    elif batch_size != attention_mask.shape[0]:
        raise ValueError
    
    if min_dtype is None:
        min_dtype = torch.finfo(dtype).min

    top_k = topk_retrieve_indices.shape[1]
    # copy non_tree_mask from the last token is always safe
    non_tree_mask = attention_mask[:,:,-1:,:non_tree_length].expand((batch_size,1,top_k,non_tree_length))
    
    last_length = attention_mask.shape[-1]
    upper_layer_tree_length = last_length - non_tree_length
    if upper_layer_tree_length < 0:
        raise ValueError
    # upper_layer_tree_mask: [bsz, 1, top_k, upper_layer_tree_length]
    upper_layer_tree_mask = torch.full((batch_size,1,top_k,upper_layer_tree_length), fill_value=min_dtype, dtype=dtype, device=device)
    if upper_layer_tree_length > 0:
        # topk_retrieve_indices is 1-ahead
        # why not later update topk_retrieve_indices? it will cause 2-reference to this tensor, 1 from model_kwargs, 1 from variable,
        # which consumes more memory when updating (the old one cannot be released)
        upper_layer_tree_mask.scatter_(dim=-1, index=topk_retrieve_indices[:,:,:-1].unsqueeze(1), value=0.0)

    this_layer_tree_mask = torch.full((top_k,top_k), fill_value=min_dtype, dtype=dtype, device=device)
    this_layer_tree_mask = this_layer_tree_mask.fill_diagonal_(fill_value=0.0)[None,None,:,:].expand(batch_size, 1, top_k, top_k)
    
    attention_mask = torch.cat([non_tree_mask, upper_layer_tree_mask, this_layer_tree_mask], dim=-1)
    return attention_mask

def _retrieve_from_flat(
    flat_tensor: torch.Tensor,
    retrieve_indices: torch.Tensor,
    is_vrfy: bool,
):
    """
        flat_tensor -> retrieved tensor
        Input:
            flat_tensor: [bsz, flat_len (+ 1(root)), ...]
            retrieve_indices: [bsz, choose_num, depth], each element is in [0, flat_len]
            
    """
    if is_vrfy:
        flat_tensor_length = flat_tensor.shape[-1]
        flat_tensor_root, flat_tensor = flat_tensor.split((1,flat_tensor_length-1), dim=-1)
    
    # retrieve
    # prepare flat tensor
    expanded_flat_tensor_shape = list(flat_tensor.shape)
    expanded_flat_tensor_shape.insert(1, retrieve_indices.shape[1]) # [bsz, choose_num, flat_len, ...]
    flat_tensor = flat_tensor.unsqueeze(1).expand(expanded_flat_tensor_shape)
    # prepare retrive indices
    retrieve_indices_unsqueeze_num = len(flat_tensor.shape) - len(retrieve_indices.shape)
    for i in range(retrieve_indices_unsqueeze_num):
        retrieve_indices = retrieve_indices.unsqueeze(-1) # [bsz, choose_num, depth-1, 1,1,...]
        _expand_shape = list(retrieve_indices.shape)
        _expand_shape[-1] = flat_tensor.shape[-(retrieve_indices_unsqueeze_num-i)]
        retrieve_indices = retrieve_indices.expand(_expand_shape)
    # retrieve 
    # TODO: if bsz==1, this could be optimized   
    retrieved_tensor = flat_tensor.gather(dim=2, index=retrieve_indices) # [bsz, choose_num, depth-1,...]
    
    # add root to the retrieved result
    if is_vrfy:
        expanded_flat_tensor_shape[2] = 1
        flat_tensor_root = flat_tensor_root.unsqueeze(1).expand(expanded_flat_tensor_shape) # [bsz, choose_num, 1, ...]
        retrieved_tensor = torch.cat([flat_tensor_root, retrieved_tensor], dim=2)
        
    return retrieved_tensor
    
    
def _tree_token_verification(
    candidate_retrieved_new_tokens: torch.LongTensor, 
    vrfy_retrieved_selected_tokens: torch.LongTensor,
    vrfy_flat_selected_tokens: torch.LongTensor,
    retrieve_indices: torch.LongTensor,
    eos_token_id: Optional[torch.LongTensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """ There might be some invalid tokens in retrieved result, as padding are enabled when forming retrieve_indices
        retrieved_tokens: [bsz, choose_num, depth]
        eos_token_id: [eos_token_num], each is an eos-token id
        candidate is 1-token behind the verification, see _retrieve_from_flat.add_root
    """
    if vrfy_retrieved_selected_tokens.shape[-1] != candidate_retrieved_new_tokens.shape[-1] + 1:
        raise ValueError
    
    # create a valid mask to prevent padding
    batch_size = retrieve_indices.shape[0]
    candidate_length = candidate_retrieved_new_tokens.shape[-1]
    flat_candidate_length = candidate_retrieved_new_tokens.shape[1]
    top_k = candidate_retrieved_new_tokens.shape[1] // candidate_length
    pos = torch.arange(0, candidate_length, dtype=retrieve_indices.dtype, device=retrieve_indices.device).view(candidate_length,1).expand(candidate_length,top_k).flatten().view(1,flat_candidate_length,1).expand(batch_size,flat_candidate_length,candidate_length)
    stop_pos = torch.arange(0, candidate_length, dtype=retrieve_indices.dtype, device=retrieve_indices.device)
    valid_mask = (pos >= stop_pos)  
    
    is_accepted = valid_mask & (candidate_retrieved_new_tokens == vrfy_retrieved_selected_tokens[..., :-1])
    # is_accepted = (candidate_retrieved_new_tokens == vrfy_retrieved_selected_tokens[..., :-1])
    # prevent eos from being overly accepted
    if eos_token_id is not None:
        not_eos_or_after = (torch.isin(candidate_retrieved_new_tokens, eos_token_id, invert=True).cumprod(dim=-1) > 0)
        is_accepted = is_accepted & not_eos_or_after # (is_accepted + not_eos_or_after) > 0
        
    n_matches, best_candidate = ((~is_accepted).cumsum(-1) < 1).sum(-1).max(dim=-1) # [bsz, choose_num]
    n_matches = n_matches[0].view(())
    best_candidate = best_candidate[0].view(())
    # slice valid tokens
    if n_matches.item() > 0:
        # if best_candidate // top_k + 1 > n_matches:
        #     print(1)
        valid_tokens = vrfy_retrieved_selected_tokens[:,best_candidate,:n_matches] # [bsz, n_matches]
        # add bonus token to valid_tokens by indexing
        last_valid_token_idx = retrieve_indices[0,best_candidate,n_matches-1]
        bonus_token = vrfy_flat_selected_tokens[:,last_valid_token_idx+1]
        valid_tokens = torch.cat([valid_tokens, bonus_token.view(valid_tokens.shape[0],1)], dim=-1)
    else:
        # just the 0 token
        valid_tokens = vrfy_flat_selected_tokens[:,:1].contiguous()
    
    return best_candidate, n_matches, valid_tokens
    
    
    
    