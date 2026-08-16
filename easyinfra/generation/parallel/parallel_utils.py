import torch.distributed as dist
import os

_LOCAL_RANK = None

def get_global_rank():
    return dist.get_rank()

def get_local_rank():
    global _LOCAL_RANK
    if _LOCAL_RANK is None:
        _LOCAL_RANK = int(os.environ.get("LOCAL_RANK"))
    return _LOCAL_RANK

def get_world_size(group=None):
    return dist.get_world_size(group)

def get_local_world_size():
    local_world_size = os.environ.get("EASYINFRA_LOCAL_WORLD_SIZE")
    return int(local_world_size)

def get_node_rank():
    node_rank = os.environ.get("EASYINFRA_NODE_RANK")
    return int(node_rank)

def get_device():
    return get_local_rank()

def get_group_rank(group):
    return dist.get_group_rank(group, get_global_rank())
