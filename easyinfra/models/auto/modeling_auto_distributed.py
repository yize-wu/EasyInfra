from transformers import AutoConfig
from ...parallel_configurations import AttnParallelPolicy, ExpertParallelPolicy

class AutoDistributedModelForCausalLM:
    """
        This class is for distributed worker to load model from auto.
    """
    @classmethod
    def from_pretrained(cls, base_model_path, world_size, **kwargs):
        # get config
        config = AutoConfig.from_pretrained(
            base_model_path,
        )
        # get type
        Type = config.architectures[0]
                
        from_pretrained_args = (base_model_path,)
        from_pretrained_kwargs = kwargs
        from_pretrained_kwargs.update({"config": config})
        
        use_tp = kwargs.pop("use_tp", False)
        use_hyperdraft = kwargs.pop("use_hyperdraft", False)
                        
        if Type == 'LlamaForCausalLM':
            if use_tp:
                from ..llama3.modeling_llama3 import LlamaForCausalLM
            elif use_hyperdraft:
                from ..llama3_hyperdraft.modeling_llama3_hyperdraft import LlamaForCausalLM
            else:
                raise NotImplementedError
            config.o_proj_bias = False
            model_cls = LlamaForCausalLM
        elif Type == "Qwen2ForCausalLM":
            if use_tp:
                from ..llama3.modeling_llama3 import LlamaForCausalLM
            elif use_hyperdraft:
                from ..llama3_hyperdraft.modeling_llama3_hyperdraft import LlamaForCausalLM
            else:
                raise NotImplementedError
            config.o_proj_bias = False
            config.attention_bias = True
            config.mlp_bias = False
            model_cls = LlamaForCausalLM
        elif Type == "Qwen3MoeForCausalLM":
            from ..qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM
            model_cls = Qwen3MoeForCausalLM
        elif Type == "DeepseekV3ForCausalLM":
            from ..deepseek_v3.modeling_deepseek_v3 import DeepseekV3ForCausalLM
            model_cls = DeepseekV3ForCausalLM
        else:
            raise NotImplementedError
        
        base_model = model_cls.from_pretrained(
            *from_pretrained_args, 
            **from_pretrained_kwargs
        )
        return base_model
