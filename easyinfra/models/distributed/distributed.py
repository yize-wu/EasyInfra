from .workers import init_workers, destroy_workers

class DistributedInferenceEngine:
    
    def __init__(self, base_model_path, world_size, **kwargs):
        
        from_pretrained_args = (base_model_path, world_size)
        from_pretrained_kwargs = kwargs
                
        self.workers, self.shared_mem, self.host_pipe = init_workers(world_size)
        
        # Now you are the host.
        # send args to workers. Workers will start to load model.
        self.host_pipe.send((from_pretrained_args, from_pretrained_kwargs))
        self.host_pipe.recv() # sync for finishing load model 
    
    @classmethod
    def from_pretrained(
        cls,
        base_model_path: str,
        world_size: int,
        **kwargs
    ):
        if world_size < 1:
            raise ValueError(f"world size must > 0, yours is {world_size}")
        model = cls(base_model_path, world_size, **kwargs)
        return model
    
    def generate(self, *args, **kwargs):
        
        generate_args, generate_kwargs = args, kwargs
        self.host_pipe.send((generate_args, generate_kwargs))
        res = self.host_pipe.recv()
        
        return res
    
    def get_generation_time(self):
        # stop generate: args and kwargs are None
        self.host_pipe.send([None, None])
        rank0_generation_time, sum_generation_tokens, accept_num, candidate_num, assistant_model_run_time = self.host_pipe.recv()
        return rank0_generation_time, sum_generation_tokens, accept_num, candidate_num, assistant_model_run_time
    
    def destroy(self):
        
        destroy_workers(self.workers, self.host_pipe)