import torch
from typing import Union, List

def _tensor_exclusive_cumsum(each_num: Union[torch.Tensor, List], dim: int):
    return each_num.cumsum(dim=dim) - each_num
def _tensor_cumsum(each_num: Union[torch.Tensor, List], dim: int):
    return each_num.cumsum(dim=dim)

def exclusive_cumsum(each_num: Union[torch.Tensor, List], dim: int = -1):
    '''
        Exclusive Cumsum. Will reserve the type of input.
    '''
    if isinstance(each_num, torch.Tensor):
        return _tensor_exclusive_cumsum(each_num, dim)
    elif isinstance(each_num, List):
        each_num = torch.tensor(each_num, device='cpu')
        return _tensor_exclusive_cumsum(each_num, dim).tolist()
    else:
        raise ValueError(f"exclusive sum must each_num as tensor or list, but {type(each_num)}")

def cumsum(each_num: Union[torch.Tensor, List], dim: int = -1):
    '''
        Exclusive Cumsum. Will reserve the type of input.
    '''
    if isinstance(each_num, torch.Tensor):
        return _tensor_cumsum(each_num, dim)
    elif isinstance(each_num, List):
        each_num = torch.tensor(each_num, device='cpu')
        return _tensor_cumsum(each_num, dim).tolist()
    else:
        raise ValueError(f"exclusive sum must each_num as tensor or list, but {type(each_num)}")

def segment_reduction(t: torch.Tensor, seg_sizes: torch.LongTensor, dim: int = -1):
    assert seg_sizes.dim() == 1
    segs_end_indices = seg_sizes.cumsum(-1) - 1
    t = t.cumsum(dim=dim).index_select(dim, segs_end_indices)
    t[1:] = t[1:] - t[:-1]
    return t
