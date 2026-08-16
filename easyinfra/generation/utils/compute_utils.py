import torch

def _add_to_residual(residual: torch.Tensor, output_tensor: torch.Tensor, padding_side = "right"):
    '''
        Assume the padding side is RIGHT.
    '''
    sequence_length, hidden_size = residual.shape[-2], residual.shape[-1]
    if padding_side == "right":
        output_tensor = output_tensor.view(-1, hidden_size)[:sequence_length,:]
    elif padding_side == "left":
        output_tensor = output_tensor.view(-1, hidden_size)[-sequence_length:,:]
    else:
        raise ValueError
    residual = residual + output_tensor
    return residual