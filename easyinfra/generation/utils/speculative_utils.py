import torch
from typing import Optional

def _speculative_sampling(
    candidate_input_ids: torch.LongTensor,
    candidate_logits: torch.FloatTensor,
    candidate_length: int,
    new_logits: torch.FloatTensor,
    is_done_candidate: bool,
):
    # raise NotImplementedError
    new_candidate_input_ids = candidate_input_ids[:, -candidate_length:]
    # Gets the probabilities from the logits. q_i and p_i denote the assistant and model probabilities of the tokens
    # selected by the assistant, respectively.
    q = candidate_logits.softmax(dim=-1)
    q_i = q[:, torch.arange(candidate_length), new_candidate_input_ids].squeeze(0, 1)
    p = new_logits.softmax(dim=-1)
    p_i = p[:, torch.arange(candidate_length), new_candidate_input_ids].squeeze(0, 1)
    probability_ratio = p_i / q_i

    # When probability_ratio > 1 (i.e. q_i(x) < p_i(x), or "assistant probability of the candidate token is smaller
    # than the model probability for the same token"), keep the token. Otherwise reject with p = 1 - probability_ratio
    # (= keep with p = probability_ratio). Keep all the tokens until the first rejection
    r_i = torch.rand_like(probability_ratio)
    is_accepted = r_i <= probability_ratio
    n_matches = ((~is_accepted).cumsum(dim=-1) < 1).sum()  # this is `n` in algorithm 1

    # Ensure we don't generate beyond max_len or an EOS token (not in algorithm 1, but needed for correct behavior)
    if is_done_candidate and n_matches == candidate_length:
        # Output length is assumed to be `n_matches + 1`. Since we won't generate another token with the target model
        # due to acceptance on EOS we fix `n_matches`
        n_matches -= 1
        valid_tokens = new_candidate_input_ids[:, : n_matches + 1]
    else:
        # Next token selection: if there is a rejection, adjust the distribution from the main model before sampling.
        gamma = candidate_logits.shape[1]
        p_n_plus_1 = p[:, n_matches, :]
        if n_matches < gamma:
            q_n_plus_1 = q[:, n_matches, :]
            p_prime = torch.clamp((p_n_plus_1 - q_n_plus_1), min=0)
            p_prime.div_(p_prime.sum())
        else:
            p_prime = p_n_plus_1
        t = torch.multinomial(p_prime, num_samples=1).squeeze(1)[None, :]

        # The selected tokens include the matches (if any) plus the next sampled tokens
        if n_matches > 0:
            valid_tokens = torch.cat((new_candidate_input_ids[:, :n_matches], t), dim=-1)
        else:
            valid_tokens = t

    return valid_tokens, n_matches

from .tree_attention import _retrieve_from_flat

def _retrieve_prob(
    flat_probs: torch.Tensor,
    retrieve_indices: torch.Tensor,
    candidate_retrieved_new_tokens: torch.LongTensor,
):
    """
        flat_tensor -> retrieved tensor
        Input:
            flat_tensor: [bsz, flat_len (+ 1(root)), ...]
            retrieve_indices: [bsz, choose_num, depth], each element is in [0, flat_len]
        Return:
            [bsz, choose_num, depth] of retrieved probs
    """
    # to support batch, add a batch dim

    choose_dim_indices = retrieve_indices.new_zeros(1,1,1).expand(retrieve_indices.shape[1], retrieve_indices.shape[2], 1)
    flat_dim_indices = retrieve_indices[0].unsqueeze(-1)
    prob_dim_indices = candidate_retrieved_new_tokens[0].unsqueeze(-1)
    # [bsz, 1, flat_len, logits_dim] -> [bsz, choose_num, flat_len, logits_dim]
    retrieved_prob = flat_probs[:, None][:, choose_dim_indices, flat_dim_indices, prob_dim_indices]
    # retrieved_prob = retrieved_prob.permute(3,0,1,2)
    
    return retrieved_prob.squeeze(-1)
    


def _tree_speculative_sampling(
    candidate_flat_input_ids: torch.LongTensor,
    candidate_flat_probs: torch.FloatTensor,
    retrieve_indices: torch.LongTensor,
    new_logits: torch.FloatTensor,
    eos_token_id: Optional[torch.LongTensor],
):
    vrfy_flat_probs = new_logits.softmax(dim=-1)
    
    # retrieve probs 
    batch_size = retrieve_indices.shape[0]
    real_retrieve_indices = torch.cat([retrieve_indices.new_zeros(batch_size, retrieve_indices.shape[1], 1), (retrieve_indices + 1)], dim=-1)
    
    # [bsz, choose_num, depth]
    candidate_retrieved_new_tokens = _retrieve_from_flat(candidate_flat_input_ids, retrieve_indices, is_vrfy=False)
    
    # candidate_retrieved_probs = _retrieve_from_flat(candidate_flat_probs, real_retrieve_indices[:,:,:-1], is_vrfy=False)
    # vrfy_retrieved_probs = _retrieve_from_flat(vrfy_flat_probs, real_retrieve_indices[:,:,:-1], is_vrfy=False)   
    q_probs = _retrieve_prob(candidate_flat_probs, real_retrieve_indices[:,:,:-1], candidate_retrieved_new_tokens)
    p_probs = _retrieve_prob(vrfy_flat_probs, real_retrieve_indices[:,:,:-1], candidate_retrieved_new_tokens)
    
    
    # q_probs = candidate_retrieved_probs.gather(dim=-1, index=candidate_retrieved_new_tokens.unsqueeze(-1)).squeeze(-1)
    # p_probs = vrfy_retrieved_probs.gather(dim=-1, index=candidate_retrieved_new_tokens.unsqueeze(-1)).squeeze(-1)
    
    # no need for valid mask like token-retrieve, as invalid part of path_prob is set to 0.0
    # create a valid mask to prevent padding
    candidate_length = candidate_retrieved_new_tokens.shape[-1]
    flat_candidate_length = candidate_retrieved_new_tokens.shape[1]
    top_k = candidate_retrieved_new_tokens.shape[1] // candidate_length
    pos = torch.arange(0, candidate_length, dtype=retrieve_indices.dtype, device=retrieve_indices.device).view(candidate_length,1).expand(candidate_length,top_k).flatten().view(1,flat_candidate_length,1).expand(batch_size,flat_candidate_length,candidate_length)
    stop_pos = torch.arange(0, candidate_length, dtype=retrieve_indices.dtype, device=retrieve_indices.device)
    valid_mask = (pos >= stop_pos)  
    
    probability_ratio = p_probs / (q_probs + 1e-12)
    r_i = torch.rand_like(probability_ratio)
    is_accepted = valid_mask & (r_i <= probability_ratio)
    # prevent eos from being overly accepted
    if eos_token_id is not None:
        not_eos_or_after = (torch.isin(candidate_retrieved_new_tokens, eos_token_id, invert=True).cumprod(dim=-1) > 0)
        is_accepted = is_accepted & not_eos_or_after # (is_accepted + not_eos_or_after) > 0
            
    n_matches, best_candidate = ((~is_accepted).cumsum(-1) < 1).sum(-1).max(dim=-1) # [bsz, choose_num]
    n_matches = n_matches[0].view(())
    best_candidate = best_candidate[0].view(())
    
    # slice valid tokens
    bonus_pos = real_retrieve_indices[0,best_candidate,n_matches]
    p_n_plus_1 = vrfy_flat_probs[:,bonus_pos,:]
    
    # bonus token
    if n_matches.item() < candidate_length:
        # there could be a overflow, when p_n_plus_1 is a spike 1.0 and q_n_plus_1 is also a spike 1.0
        if torch.any(p_n_plus_1 == 1.0):
            # clamp(p - q) is a 1.0 spike tensor at the original position of 1.0, so just use p_n_plus_1 is ok
            p_prime = p_n_plus_1
        else:
            q_n_plus_1 = candidate_flat_probs[:, bonus_pos, :]
            p_prime = torch.clamp((p_n_plus_1 - q_n_plus_1), min=0)
            p_prime.div_(p_prime.sum())
    else:
        p_prime = p_n_plus_1
        
    # add bonus token to valid_tokens by indexing
    bonus_token = torch.multinomial(p_prime, num_samples=1).view(batch_size,1)
    
    # output
    if n_matches.item() > 0:
        valid_tokens = candidate_retrieved_new_tokens[:,best_candidate,:n_matches] # [bsz, n_matches]
        valid_tokens = torch.cat([valid_tokens, bonus_token], dim=-1)
    else:
        # just the 0 token
        valid_tokens = bonus_token
    
    return best_candidate, n_matches, valid_tokens
