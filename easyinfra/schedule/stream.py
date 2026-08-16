import torch
from easyinfra import envs
from easyinfra.utils import show_rank_print

def get_current_stream():
    return torch.cuda.current_stream()
def create_overlap_stream():
    stream = torch.cuda.Stream() if envs.ENABLE_COMPUTE_COMM_OVERLAP else torch.cuda.current_stream()
    if envs.ENABLE_COMPUTE_COMM_OVERLAP is False:
        show_rank_print(f"debug no compute-comm overlap", 0)
    return stream