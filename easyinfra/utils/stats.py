import torch
import time
import torch.distributed as dist
from easyinfra.generation.parallel.parallel_utils import get_local_rank, get_global_rank

def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
        Compute the last 1-D's cosine similarity.
    """
    return (a * b).sum(dim=-1) / (a.norm(dim=-1) * b.norm(dim=-1))

def synchronize_device():
    torch.cuda.synchronize()

def record_time_sync():
    # torch.cuda.current_stream().synchronize() # current stream does not include NCCL kernel
    # synchronize_device()
    return time.time()

def rank0_print(obj):
    if dist.get_rank() == 0:
        print(obj)

def stage_rank_print(obj, local_rank=0):
    # show_rank_print(obj, local_rank)
    return
def event_rank_print(obj, local_rank=0):
    # show_rank_print(obj, local_rank)
    return

def show_rank_print(obj, local_rank=0):
    if local_rank is not None and local_rank != get_local_rank():
        return
    print(f"rank: {get_global_rank()}:: {obj}")
    
def show_global_rank_print(obj, global_rank=None):
    if global_rank is not None and global_rank != get_global_rank():
        return
    print(f"rank: {dist.get_rank()}:: {obj}")

def print_and_record(fd, s:str):
    if fd is not None:
        fd.write(s + '\n')
    print(s)



def round_sum(l, round_dim=4):
    return round(sum(l), round_dim)
