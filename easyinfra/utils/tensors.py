import torch
import time
from easyinfra.utils.stats import show_rank_print

def is_unique(value, t: torch.Tensor,):
    return (t == value).sum() == 1

# Avoid synchronizing the whole device. Do a non-blocking copy to CPU
# and only synchronize the current CUDA stream for that device before
# converting to Python list.
def _to_cpu_list(t: torch.Tensor):
    return t.tolist()
    if not t.is_cuda:
        return t.tolist()
    # async copy to CPU (requires pinned memory to be truly non-blocking)
    # time0 = time.time()
    pinned_cpu_t = torch.empty(t.size(), device='cpu', dtype=t.dtype, pin_memory=True)
    # time1 = time.time()
    pinned_cpu_t.copy_(t, non_blocking=True)
    # time2 = time.time()
    torch.cuda.current_stream().synchronize()
    # time3 = time.time()
    # show_rank_print(f"tocpu time: {time1-time0}, {time2-time1}, {time3-time2}", 0)
    return pinned_cpu_t.tolist()
