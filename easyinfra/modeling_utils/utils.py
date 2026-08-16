import torch

def _prepare_position_id_from_2d_attention_mask(
    attention_mask: torch.Tensor,
) -> torch.LongTensor:
    """
        Return the position_ids inferred from 2d attention mask.
    """
    if attention_mask.dim() != 2:
        raise ValueError(f"preparing position id from attention mask, but attention mask dim is {attention_mask.dim()}")
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    return position_ids

def _prepare_1d_position_id_from_2d_attention_mask(
    attention_mask: torch.Tensor,
) -> torch.LongTensor:
    """
        Return the position_ids inferred from 2d attention mask.
    """
    if attention_mask.dim() != 2 or attention_mask.shape[0] != 1:
        raise ValueError(f"preparing position id from attention mask, but attention mask shape is {attention_mask.size()}")
    attention_mask = attention_mask.squeeze(0)
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    return position_ids
