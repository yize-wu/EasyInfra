import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import os
import time

from transformers import AutoTokenizer

from easyinfra import envs
from ..auto.modeling_auto_distributed import AutoDistributedModelForCausalLM
from ...generation.parallel.communicator import GroupCommunicator
from ...generation.parallel.parallel_utils import get_device
from ...utils import record_time_sync, show_rank_print

EASYINFRA_PORT = 49521

def get_device_count():
    return torch.cuda.device_count()

def get_world_nodes_ip(node_config_path: str):
    '''
        Return all IP of nodes in the config.
    '''
    nodes_ip = []
    with open(node_config_path, "r") as f:
        for line in f:
            nodes_ip.append(line.strip()) # IP address
    return nodes_ip
    
import socket

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Doesn't actually connect, just picks the right interface
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    finally:
        s.close()
    return ip


def get_visible_device_list(local_world_size, trust_device_count=True):
    device_num = get_device_count()
    if local_world_size > device_num:
        raise ValueError(f"local_world_size must be less or equal than # of current visible devices, but {local_world_size} vs {device_num}.")
    if trust_device_count:
        device_list = [_ for _ in range(device_num)]
    else:
        # find ids of all visible devices
        device_list = []
        for i in range(device_num):
            try:
                _ = torch.tensor([0], device=i)
                device_list.append(i)
                if len(device_list) == local_world_size:
                    break
            except Exception:
                print(f"device {i} not working.")
                continue
    return device_list

_WORKER_INFO = None
class WorkerInfo:
    def __init__(self, world_size: int, local_world_size: int):
        # force driver to be 0
        self.tensor_group = GroupCommunicator([_ for _ in range(world_size)], driver=0, warmup=False) # a group for all processes
        self.group = self.tensor_group
        # self.non_tensor_group = dist.new_group([_ for _ in range(world_size)], backend='nccl')
        # test = [None]
        # dist.broadcast_object_list(test, src=0, group=self.non_tensor_group)
        # dist.barrier(self.non_tensor_group)

def enable_rdma():
    # os.environ["NCCL_DEBUG"] = 'TRACE'
    os.environ["NCCL_IB_DISABLE"] = '0'
    os.environ["NCCL_SOCKET_IFNAME"] = '^lo,docker'
    # os.environ["NCCL_IB_HCA"] = 'mlx5_1'
    # os.environ["NCCL_IB_GID_INDEX"] = '1'

def init_env(global_rank, local_rank, world_size, local_world_size, node_rank, master_ip, device):
    """
        For distributed environment, must set some environ variables.
    """
    os.environ['MASTER_ADDR'] = master_ip
    os.environ['MASTER_PORT'] = str(EASYINFRA_PORT)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(global_rank)
    os.environ['LOCAL_RANK'] = str(local_rank)
    torch.cuda.set_device(device)
    
    if local_world_size < world_size:
        ### Enable RDMA
        enable_rdma()

    ### distributed env does not need them, but EasyInfra does
    os.environ['EASYINFRA_LOCAL_WORLD_SIZE'] = str(local_world_size)
    os.environ['EASYINFRA_NODE_RANK'] = str(node_rank)
    
def init_dist(global_rank, local_rank, world_size, local_world_size, node_rank, master_ip, device):
    """
        Set up torch.distributed environment.
    """
    init_env(global_rank, local_rank, world_size, local_world_size, node_rank, master_ip, device)
    dist.init_process_group(
        backend="nccl", 
        rank=global_rank, 
        device_id=torch.device(f"cuda:{device}"),
        world_size=world_size
    )
    global _WORKER_INFO
    _WORKER_INFO = WorkerInfo(world_size=world_size, local_world_size=local_world_size)

def broadcast_objects(src=0, group=None, objects=None):
    if not isinstance(objects, list):
        raise ValueError
    _WORKER_INFO.group.broadcast_object_list(objects, src=src)
    # dist.broadcast_object_list(objects, src=src, group=_WORKER_INFO.non_tensor_group)
    return objects

def broadcast_generation_args(generation_args, generation_kwargs):
    # start = record_time_sync()
    # print(f"rank {global_rank} start time: {start}")
        
    generation_args, = broadcast_objects(0, objects=[generation_args])
    # start = record_time_sync()
    # print(f"rank {global_rank} args time: {start}")
    generation_kwargs, = broadcast_objects(0, objects=[generation_kwargs])
    # start = record_time_sync()
    # print(f"rank {global_rank} kwargs time: {start}")
    
    return generation_args, generation_kwargs

def set_seed(seed:int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def create_shared_memory(max_toks: int = 4096*4096):
    # Create shared memory block
    
    # input_ids and attention masks
    ### Pin the memory to avoid re-copy when moving it to device RAM
    shared_input_ids = torch.empty((max_toks,), dtype=torch.int64, device="cpu", pin_memory=True)
    shared_output_ids = torch.empty((max_toks,), dtype=torch.int64, device="cpu", pin_memory=True)
    shared_attention_masks = torch.empty((max_toks,), dtype=torch.int64, device="cpu", pin_memory=True)
    shared_shape = torch.empty((2,), dtype=torch.int64, device="cpu", pin_memory=True)
    shared_input_ids.share_memory_()
    shared_attention_masks.share_memory_()
    shared_shape.share_memory_()
    
    return {
        "input_ids": shared_input_ids, 
        "output_ids": shared_output_ids, 
        "attention_mask": shared_attention_masks,
        "shape": shared_shape,
    }

def worker_func(
    global_rank, 
    local_rank, 
    world_size, 
    local_world_size,
    node_rank,
    master_ip,
    device,
    shared_mem, 
    worker_pipe, 
):
    # import torch
    
    init_dist(global_rank, local_rank, world_size, local_world_size, node_rank, master_ip, device)    
    # batch group?
    print(f"rank {global_rank} finished worker initialization.")
    
    # rank 0 should communicate with host
    if local_rank == 0:
        from_pretrained_args, from_pretrained_kwargs = worker_pipe.recv()
    else:
        from_pretrained_args = None
        from_pretrained_kwargs = None
    
    set_seed(42+global_rank)
    objects = [from_pretrained_args, from_pretrained_kwargs]
    _WORKER_INFO.group.broadcast_object_list(objects, src=0)
    from_pretrained_args, from_pretrained_kwargs = objects
    
    # if assistant model is used, pop the args and kwargs
    assistant_model_path = from_pretrained_kwargs.pop("assistant_model_path", None)
    load_assistant_model = assistant_model_path is not None
    if load_assistant_model:
        assistant_from_pretrained_args = from_pretrained_kwargs.pop("assistant_from_pretrained_args", None)
        assistant_from_pretrained_kwargs = from_pretrained_kwargs.pop("assistant_from_pretrained_kwargs", None)
        # If there is no specific assistant args, use base model args
        if assistant_from_pretrained_args is None:
            assistant_from_pretrained_args = list(from_pretrained_args)
            assistant_from_pretrained_args[0] = assistant_model_path
        if assistant_from_pretrained_kwargs is None:
            assistant_from_pretrained_kwargs = from_pretrained_kwargs
            assistant_from_pretrained_kwargs["use_hyperdraft"] = True
    
    # load model
    model = AutoDistributedModelForCausalLM.from_pretrained(
        *from_pretrained_args, **from_pretrained_kwargs
    )
    if load_assistant_model:
        # assistant_tokenizer = AutoTokenizer.from_pretrained(assistant_model_path)
        assistant_model = AutoDistributedModelForCausalLM.from_pretrained(
            assistant_model_path, **assistant_from_pretrained_kwargs
        )
    else:
        # assistant_tokenizer = None
        assistant_model = None
        
    model.prepare_for_assistant(assistant_model)
    # # compile
    # if os.environ.get("TORCH_COMPILE_BACKEND") == "NATIVE":
    #     import torch._dynamo
    #     torch.set_float32_matmul_precision('high')
    #     torch._dynamo.config.cache_size_limit = 64
    #     # model = torch.compile(model, mode="reduce-overhead", dynamic=True)
    #     # model.forward = torch.compile(model.forward, mode="reduce-overhead", dynamic=True)
    #     # if assistant_model is not None:
    #     #     assistant_model = torch.compile(assistant_model, mode="reduce-overhead", dynamic=True)
    
    _WORKER_INFO.tensor_group.barrier()
    del from_pretrained_args, from_pretrained_kwargs
    
    if local_rank == 0:
        worker_pipe.send("finish loading model.")
    
    my_generation_time = 0.0
    num_all_generated_tokens = 0
    # generate
    res = None
    while True:
        if local_rank == 0:
            generation_args, generation_kwargs = worker_pipe.recv()
        else:
            generation_args, generation_kwargs = None, None
        
        generation_args, generation_kwargs = broadcast_generation_args(generation_args, generation_kwargs)
        if generation_args is None and generation_kwargs is None:
            # time to exit
            break
        
        # add assistant model into kwargs if needed
        use_assistant_model = generation_kwargs.get("use_assistant_model", False)
        if use_assistant_model is True:
            if assistant_model is None:
                raise ValueError(f"Cannot use assistant model. Not loaded.")
            generation_kwargs.update({"assistant_model": assistant_model})
        
        input_shape: torch.Tensor = tuple(shared_mem["shape"].tolist())
        num_toks = input_shape[0] * input_shape[1]
        inputs = {
            "input_ids": shared_mem["input_ids"][:num_toks].to("cuda").view(input_shape),
            "attention_mask": shared_mem["attention_mask"][:num_toks].to("cuda").view(input_shape),
        }
        show_rank_print(f"attention sum: {inputs["attention_mask"].sum()}/{inputs["attention_mask"].numel()}", 0)
        
        _WORKER_INFO.tensor_group.barrier()
        
        start = time.time()
        # show_rank_print(f"start generate time: {start}")
        
        # add inputs to kwargs
        generation_kwargs.update(inputs)
        # add tokenizer to kwargs
        # generation_kwargs["tokenizer"] = tokenizer
        
        # generate    
        res = model.generate(*generation_args, **generation_kwargs)
        
        num_this_generated_tokens = res.shape[-1] - input_shape[-1]
        num_all_generated_tokens += num_this_generated_tokens
            
        # print(f"rank {global_rank} end generate.")
        
        end = time.time()
        my_generation_time += end - start
        
        # send result back to host
        if local_rank == 0:
            num_toks = res.shape[0] * num_this_generated_tokens
            shared_mem["output_ids"][:num_toks] = res[:,-num_this_generated_tokens:].reshape(-1).to('cpu')
            shared_mem["shape"][0] = res.shape[0]
            shared_mem["shape"][1] = num_this_generated_tokens
            res_dict = {
                "total_run_time": model.total_run_time,
                "this_run_time": model.this_run_time,
                "this_num_accepted_tokens": model.this_num_accepted_tokens,
                "this_num_candidate_tokens": model.this_num_candidate_tokens,
                "this_assistant_runtime": model.this_assistant_runtime,
                "num_accepted_tokens": model.num_accepted_tokens,
                "num_candidate_tokens": model.num_candidate_tokens,
                "assistant_runtime": model.assistant_runtime,
            }
            worker_pipe.send(res_dict)
    
    full_generation_time = torch.tensor(my_generation_time, dtype=torch.float32, device=get_device())
    # print(f"rank {global_rank} generation time: {my_generation_time}")
    assistant_model_run_time = model.assistant_runtime
    base_model_run_time = my_generation_time - model.assistant_runtime
    # print(f"rank {global_rank} assistant run time: {assistant_model_run_time}")
    # print(f"rank {global_rank} base model raw run time: {base_model_run_time}")
    # dist.reduce(full_generation_time, dst=0, op=dist.ReduceOp.SUM)
    if local_rank == 0:
        worker_pipe.send([
            my_generation_time, 
            num_all_generated_tokens, 
            model.num_accepted_tokens, 
            model.num_candidate_tokens, 
            assistant_model_run_time
        ])
    
    del generation_args, generation_kwargs, model, res
    if local_rank == 0:
        worker_pipe.close()
    
    print(f"rank {global_rank} destroying process group")
    dist.destroy_process_group()

def init_workers(world_size: int):
    
    print("start initializing worker")
    
    local_world_size = os.environ.get("EASYINFRA_LOCAL_WORLD_SIZE", get_device_count())
    ### If world size > local_world_size, we must use multi-node
    if world_size > local_world_size:
        ### Must ensure each node has equal number of devices
        if world_size % local_world_size != 0:
            raise ValueError(f"world_size should be divided by local_world_size, but {world_size}/{local_world_size}")
        ### read all nodes IP
        node_config_path = os.path.join(envs.NODE_CONFIG, "node_config.txt")
        if not os.path.isfile(node_config_path):
            raise ValueError(f"The node config path not exist: {node_config_path}")
        all_nodes_ip = get_world_nodes_ip(node_config_path)
        ### Find all active nodes
        my_ip = get_local_ip()
        if my_ip not in all_nodes_ip:
            raise ValueError(f"my ip {my_ip} not in all_nodes_ip list {all_nodes_ip}")
        active_nodes_config = {ip: local_world_size for ip in all_nodes_ip} # TODO
    else:
        my_ip = '127.0.0.1'
        local_world_size = world_size
        active_nodes_config = {my_ip: local_world_size}
        
    active_nodes_ip = list(active_nodes_config.keys())
    ### node rank
    node_rank = active_nodes_ip.index(my_ip)
    ### master
    master_ip = active_nodes_ip[0]
    print(f"world size: {world_size}, node: {node_rank}/{len(active_nodes_config)}")

    mp.set_start_method("spawn", force=True) # spawn as start method
    host_pipe, worker_pipe = mp.Pipe() # comm pipe with host
    device_list = get_visible_device_list(local_world_size) # device list
    shared_mem = create_shared_memory()
    workers = []
    for local_rank, global_rank in enumerate(range(node_rank*local_world_size, (node_rank+1)*local_world_size)):
        p = mp.Process(target=worker_func, args=(
            global_rank, local_rank, world_size, local_world_size, node_rank,
            master_ip, device_list[local_rank], 
            shared_mem, worker_pipe,))
        workers.append(p)
    
    for worker in workers:
        worker.start()
        
    # remember to put worker in model.workers
    return workers, shared_mem, host_pipe

def destroy_workers(workers, host_pipe):
    for worker in workers:
        worker.join()
    host_pipe.close()
