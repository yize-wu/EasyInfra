from transformers.modeling_utils import (
    SAFE_WEIGHTS_INDEX_NAME,
    get_checkpoint_shard_files,
    init_empty_weights,
    find_tied_parameters,
)
from ..modeling_utils import (
    ContextManagers,
)

from transformers.configuration_utils import PretrainedConfig
from ..generation.utils.configuration_utils import GenerationConfig
from ..generation.utils.base_parallel import BaseParallelGenerationMixin
from ..modeling_utils.context import no_init_weights

import torch
from torch import nn

import copy
import os
from ..utils import my_logging
from functools import partial, wraps
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union, Iterable
from zipfile import is_zipfile

import torch
from torch import nn

from ..generation.parallel.communicator import GroupCommunicator
from ..generation.modules.logits_head.tp_logits_head import TPLogitsHead

from .auto import (
    load_state_dict_keys, 
    load_state_dict_slice, 
    load_state_dict_tensor,
)

logger = my_logging.get_logger(__name__)


def _load_state_dict_into_meta_model(
    model: "BaseParallelPreTrainedModel",
    shard_file: str,
    state_dict_keys: List[str],
    start_prefix: str = "",
    dtype=None,
):
    # return
    for param_name in state_dict_keys:

        if param_name.startswith(start_prefix):
            param_name = param_name[len(start_prefix) :]
                    
        # get module name
        module_with_kwargs = model.param_name_to_module_with_kwargs(param_name)
        if not isinstance(module_with_kwargs, Iterable):
            raise ValueError(f"non supported module name list {module_with_kwargs}")
        
        for (module_name, module_kwargs) in module_with_kwargs:
            if module_name is None:
                raise ValueError
            if hasattr(model, "need_load_weight") and model.need_load_weight(param_name, module_name) is False:
                continue
            if 'inv_freq' in module_name:
                # print('inv_freq')
                continue
            
            # find leave of module and name
            module, module_name = model.find_leaf(module_name)
            # some modules may not belong to this device
            if module is None:
                continue
                        
            if getattr(module, "_" + "support_safetensor_slice", False) == True:
                param = load_state_dict_slice(shard_file, param_name)
            else:
                param = load_state_dict_tensor(shard_file, param_name)
            
            module_kwargs["value"] = param
            if "dtype" not in module_kwargs:
                module_kwargs["dtype"] = dtype

            module.weight_loader(module_name, **module_kwargs)    

from ..generation.parallel.parallel_utils import get_global_rank

class BaseParallelPreTrainedModel(nn.Module, BaseParallelGenerationMixin,):
    config_class = None
    base_model_prefix = ""
    main_input_name = "input_ids"
    model_tags = None
    _auto_class = None
    _no_split_modules = None
    _skip_keys_device_placement = None
    _keep_in_fp32_modules = None
    # a list of `re` patterns of `state_dict` keys that should be removed from the list of missing
    # keys we find (keys inside the model but not in the checkpoint) and avoid unnecessary warnings.
    _keys_to_ignore_on_load_missing = None
    # a list of `re` patterns of `state_dict` keys that should be removed from the list of
    # unexpected keys we find (keys inside the checkpoint but not the model) and avoid unnecessary
    # warnings.
    _keys_to_ignore_on_load_unexpected = None
    # a list of `state_dict` keys to ignore when saving the model (useful for keys that aren't
    # trained, but which are either deterministic or tied variables)
    _keys_to_ignore_on_save = None
    # a list of `state_dict` keys that are potentially tied to another key in the state_dict.
    _tied_weights_keys = None
    is_parallelizable = False
    supports_gradient_checkpointing = False
    _is_stateful = False
    # Flash Attention 2 support
    _supports_flash_attn_2 = False
    # SDPA support
    _supports_sdpa = False
    # Has support for a `Cache` instance as `past_key_values`? Does it support a `StaticCache`?
    _supports_cache_class = False
    _supports_static_cache = False
    # Has support for a `QuantoQuantizedCache` instance as `past_key_values`
    _supports_quantized_cache = False

    @property
    def dummy_inputs(self) -> Dict[str, torch.Tensor]:
        """
        `Dict[str, torch.Tensor]`: Dummy inputs to do a forward pass in the network.
        """
        raise NotImplementedError
        return {"input_ids": torch.tensor([1,0,0,0,0])}

    @property
    def framework(self) -> str:
        """
        :str: Identifies that this is a PyTorch model.
        """
        raise NotImplementedError
        return "pt"

    def __init__(self, config: PretrainedConfig, *inputs, **kwargs):
        super().__init__()
        if not isinstance(config, PretrainedConfig):
            raise ValueError(
                f"Parameter config in `{self.__class__.__name__}(config)` should be an instance of class "
                "`PretrainedConfig`. To create a model from a pretrained model use "
                f"`model = {self.__class__.__name__}.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        # Save config and origin of the pretrained weights if given in model
        self.config = config
        self.name_or_path = config.name_or_path
        self.warnings_issued = {}
        self.generation_config = GenerationConfig.from_model_config(config)
        # Overwrite the class attribute to make it an instance attribute, so models like
        # `InstructBlipForConditionalGeneration` can dynamically update it without modifying the class attribute
        # when a different component (e.g. language_model) is used.
        self._keep_in_fp32_modules = copy.copy(self.__class__._keep_in_fp32_modules)
        
        # stats
        self.layers_forward_time = 0.0
        self.model_forward_time = 0.0
        self.effective_workload = 0
        self.shortage_of_all_devices = 0
        self.all_workload = 0

    def post_init(self):
        """
        A method executed at the end of each Transformer model initialization, to execute code that needs the model's
        modules properly initialized (such as weight initialization).
        """
        return
        self.init_weights()
    
    def param_name_to_module_with_kwargs(self, param_name: str) -> Dict:
        '''
            From param_name, get module_kwargs for module weight loading.
        '''
        return {"module_name_list": (param_name,)}

    def find_leaf(self, module_name: str):
        module = self
        if "." in module_name:
            splits = module_name.split(".")
            for split in splits[:-1]:
                new_module = getattr(module, split, None)
                if new_module is None:
                    _ori_module_name = ".".join(splits)
                    raise ValueError(f"{module} has no attribute {split}. Full name: {_ori_module_name}")
                module = new_module
            module_name = splits[-1]
        return (module, module_name)
    

    @classmethod
    def _set_default_torch_dtype(cls, dtype: torch.dtype) -> torch.dtype:
        """
        Change the default dtype and return the previous one. This is needed when wanting to instantiate the model
        under specific dtype.

        Args:
            dtype (`torch.dtype`):
                a floating dtype to set to.

        Returns:
            `torch.dtype`: the original `dtype` that can be used to restore `torch.set_default_dtype(dtype)` if it was
            modified. If it wasn't, returns `None`.

        Note `set_default_dtype` currently only works with floating-point types and asserts if for example,
        `torch.int64` is passed. So if a non-float `dtype` is passed this functions will throw an exception.
        """
        if not dtype.is_floating_point:
            raise ValueError(
                f"Can't instantiate {cls.__name__} model under dtype={dtype} since it is not a floating point dtype"
            )

        logger.info(f"Instantiating {cls.__name__} model under default dtype {dtype}.")
        dtype_orig = torch.get_default_dtype()
        torch.set_default_dtype(dtype)
        return dtype_orig

    @wraps(torch.nn.Module.to)
    def to(self, *args, **kwargs):
        return super().to(*args, **kwargs)

    def can_generate(self):
        return True
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args,
        config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        **kwargs,
    ):
        _fast_init = kwargs.pop("_fast_init", True)
        torch_dtype = kwargs.pop("torch_dtype", None)
        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", False)
        from_auto_class = kwargs.pop("_from_auto", False)
        from_pipeline = kwargs.pop("_from_pipeline", None)
        resume_download=None
        proxies=None
        _commit_hash=None
        
        from_pt = True
        # load pt weights early so that we know which dtype to init the model under
        if from_pt:
            dtype_orig = None
            if torch_dtype is not None:
                if isinstance(torch_dtype, str):
                    if torch_dtype == "auto":
                        torch_dtype = torch.get_default_dtype()
                    elif hasattr(torch, torch_dtype):
                        torch_dtype = getattr(torch, torch_dtype)
                    else:
                        raise ValueError(
                            f'`torch_dtype` can be one of: `torch.dtype`, `"auto"` or a string of a valid `torch.dtype`, but received {torch_dtype}'
                        )
                dtype_orig = cls._set_default_torch_dtype(torch_dtype)
        
        # At the very beginning, set up parallel config
        # cls to pop args, so model_kwargs needs to be returned
        parallel_config, model_kwargs = cls.make_parallel_config(config, **kwargs)
        device = parallel_config.device        
        
        user_agent = {"file_type": "model", "framework": "pytorch", "from_auto_class": from_auto_class}
        
        # avoid modifying original config-oriented objects    
        config = copy.deepcopy(config)

        if not isinstance(pretrained_model_name_or_path, str):
            raise ValueError(f"incorrect model dir: {pretrained_model_name_or_path}")
                    
        # Load from a sharded safetensors checkpoint
        subfolder = ''
        if os.path.isfile(os.path.join(pretrained_model_name_or_path, SAFE_WEIGHTS_INDEX_NAME)):
            resolved_archive_file = os.path.join(pretrained_model_name_or_path, subfolder, SAFE_WEIGHTS_INDEX_NAME)
            # resolved_archive_file becomes a list of files that point to the different checkpoint shards in this case.
            resolved_archive_file, sharded_metadata = get_checkpoint_shard_files(
                pretrained_model_name_or_path,
                resolved_archive_file,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                local_files_only=local_files_only,
                token=token,
                user_agent=user_agent,
                revision=revision,
                subfolder=subfolder,
                _commit_hash=_commit_hash,
            )
        else:
            resolved_archive_file = [os.path.join(pretrained_model_name_or_path, "model.safetensors")]
            

        config.name_or_path = pretrained_model_name_or_path

        # Instantiate model.
        init_contexts = [no_init_weights()]
        if low_cpu_mem_usage:
            init_contexts.append(init_empty_weights())

        config = copy.deepcopy(config)  # We do not want to modify the config inplace in from_pretrained.
        model_kwargs.update({"parallel_config": parallel_config})

        with ContextManagers(init_contexts):
            # Let's make sure we don't run the init function of buffer modules
            model = cls(config, *model_args, **model_kwargs)

        # make sure we use the model's config since the __init__ call might have copied it
        config = model.config
        
        model = cls._load_pretrained_model(
            model,
            resolved_archive_file,
            dtype=torch_dtype,
        )

        # make sure token embedding weights are still tied if needed
        model.tie_weights()

        # Set model in evaluation mode to deactivate DropOut modules by default
        model.eval()

        # If it is a model with generation capabilities, attempt to load the generation config
        model.generation_config = GenerationConfig.from_pretrained(
            pretrained_model_name_or_path,
            cache_dir=cache_dir,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            subfolder=subfolder,
            _from_auto=from_auto_class,
            _from_pipeline=from_pipeline,
            **kwargs,
        )
        
        model.to(device)
        
        return model

    def tie_weights(
        self
    ):
        if (getattr(self.config, "tie_word_embeddings", False) is True) and (hasattr(self, "lm_head")):
            assert self.lm_head.bias is None
            # print(self.model.embed_tokens.weight.shape, self.lm_head.weight.shape)
            if isinstance(self.lm_head, TPLogitsHead):
                self.lm_head: TPLogitsHead
                assert self.model.embed_tokens.weight.shape[-1] == self.lm_head.weight.shape[-1] * self.lm_head.tp_size
                self.lm_head.weight = nn.Parameter(self.model.embed_tokens.weight[:,self.lm_head.slice_start:self.lm_head.slice_end].clone())
            else:
                assert self.model.embed_tokens.weight.shape == self.lm_head.weight.shape
                self.lm_head.weight = nn.Parameter(self.model.embed_tokens.weight.clone())
                
    
    @classmethod
    def _load_pretrained_model(
        cls,
        model,
        resolved_archive_file,
        dtype=None,
    ):
        if find_tied_parameters(model):
            raise NotImplementedError(f"No support for tied parameters.")

        tqdm_for_one = False # the bar only show the progress of process-0
        if len(resolved_archive_file) > 1:
            if (not tqdm_for_one) or get_global_rank() == 0:
                resolved_archive_file = my_logging.tqdm(resolved_archive_file, desc="Loading checkpoint shards")
        for shard_file in resolved_archive_file:
            state_dict_keys = load_state_dict_keys(shard_file)
            _load_state_dict_into_meta_model(
                model,
                shard_file,
                state_dict_keys,
                dtype=dtype,
            )
                        
        return model