from transformers.generation.utils import (
    GenerateOutput,
    Callable,
    GenerateNonBeamOutput, 
    GenerateDecoderOnlyOutput,
    MaxTimeCriteria,
    # StopStringCriteria,
)

import torch
import contextlib
from typing import List, Optional, Tuple, Union, Dict, Any

import inspect
import warnings
import copy
import time
import json

from transformers.generation.utils import GenerationMixin

from ..cache_utils import (
    TreeDynamicCache,
)
from ..parallel.parallel_utils import get_global_rank, get_local_rank
from ..parallel.parallel_configuration import (
    SpeculativeDecodingParallelConfig, 
    MoeParallelConfig,
    BaseParallelConfig,
)
from ...utils import record_time_sync, show_rank_print
from .configuration_utils import (
    TreeAttentionConfig,
    GenerationConfig,
)
from ..generation_mode import GenerationMode
from .candidate_generator import (
    AssistedCandidateGenerator,
    _prepare_2d_attention_mask,
    _crop_past_key_values,
    _prepare_token_type_ids,
    CandidateRefillPolicy,
    _DEFAULT_CANDIDATE_REFILL_POLICY,
)
from .tree_candidate_generator import (
    TreeAssistedCandidateGenerator,
    LogitsProcessorList,
)

from .stopping_criteria import (
    MaxLengthCriteria,
    EosTokenCriteria,
    ConfidenceCriteria,
    StoppingCriteriaList,
)

from .logits_process import (
    TemperatureLogitsWarper
)

from .tree_attention import (
    _prepare_tree_verification_cache_position,
    _prepare_tree_verification_position_ids,
    _retrieve_from_flat,
    _tree_token_verification,
    _pop_tree_attention_config,
)

from .utils import (
    GenerateDecoderOnlyTreeOutput,
)    

from .speculative_utils import _speculative_sampling, _tree_speculative_sampling
from ...generation.functions import exclusive_cumsum

from ...schedule.request import RequestHub
from ...schedule.scheduler import MoeLayerScheduler
from easyinfra import envs
from easyinfra.schedule.chunk import _init_compute_stream, _init_comm_stream

class BaseParallelGenerationMixin(GenerationMixin):
    is_causal_model: bool
    parallel_config: Union[BaseParallelConfig, MoeParallelConfig] = None
    spec_parallel_config: Optional[SpeculativeDecodingParallelConfig] = None
    
    def init_statistics(self):
        self.total_run_time = 0.0
        self.this_run_time = 0.0
        
        self.assistant_runtime = 0.0
        self.num_accepted_tokens = 0
        self.num_candidate_tokens = 0
        self.this_assistant_runtime = 0.0
        self.this_num_accepted_tokens = 0
        self.this_num_candidate_tokens = 0
    
    def prepare_for_assistant(self, assistant_model):
        self.init_statistics()
        if assistant_model is not None:
            # create a spec parallel_config for this generation
            self.spec_parallel_config = SpeculativeDecodingParallelConfig(assistant_model.parallel_config, self.parallel_config)
            assistant_model.init_statistics()
    
    def prepare_chunk_schedule_stages(self):
        raise NotImplementedError
    
    def _prepare_generation_config(
        self, generation_config: Optional[GenerationConfig], **kwargs: Dict
    ) -> Tuple[GenerationConfig, Dict]:
        """
            Return: generation_config, model_kwargs
        """
        # priority: `generation_config` argument > `model.generation_config` (the default generation config)
        using_model_generation_config = False
        if generation_config is None:
            new_generation_config = GenerationConfig.from_model_config(self.config)
            # eos id should be from self.generation_config???
            new_generation_config.eos_token_id = self.generation_config.eos_token_id
            # if new_generation_config != self.generation_config:  # 4)
            self.generation_config = new_generation_config
            generation_config = self.generation_config
            using_model_generation_config = True

        generation_config = copy.deepcopy(generation_config)
        model_kwargs = generation_config.update(**kwargs) # Not a dict update, but config self-defined update
        # If `generation_config` is provided, let's fallback ALL special tokens to the default values for the model
        if not using_model_generation_config:
            if generation_config.bos_token_id is None:
                generation_config.bos_token_id = self.generation_config.bos_token_id
            if generation_config.eos_token_id is None:
                generation_config.eos_token_id = self.generation_config.eos_token_id
            if generation_config.pad_token_id is None:
                generation_config.pad_token_id = self.generation_config.pad_token_id
            if generation_config.decoder_start_token_id is None:
                generation_config.decoder_start_token_id = self.generation_config.decoder_start_token_id

        # pass tree attention config in generation stage, not in model loading stage
        if kwargs.get("enable_tree_attention") is True and using_model_generation_config:
            generation_config.tree_attention_config = _pop_tree_attention_config(model_kwargs)
            
        return generation_config, model_kwargs
    
    def _prepare_special_tokens(
        self,
        generation_config: GenerationConfig,
        kwargs_has_attention_mask: Optional[bool] = None,
        device: Optional[Union[torch.device, str]] = None,
    ):
        """
        Prepares the special tokens for generation, overwriting the generation config with their processed versions
        converted to tensor.

        Note that `generation_config` is changed in place and stops being serializable after this method is called.
        That is no problem if called within `generate` (`generation_config` is a local copy that doesn't leave the
        function). However, if called outside `generate`, consider creating a copy of `generation_config` first.
        """

        # Convert special tokens to tensors
        def _tensor_or_none(token, device=None):
            if token is None:
                return token

            device = device if device is not None else self.device
            if isinstance(token, torch.Tensor):
                return token.to(device)
            return torch.tensor(token, device=device, dtype=torch.long)

        bos_token_tensor = _tensor_or_none(generation_config.bos_token_id, device=device)
        eos_token_tensor = _tensor_or_none(generation_config.eos_token_id, device=device)
        pad_token_tensor = _tensor_or_none(generation_config.pad_token_id, device=device)
        decoder_start_token_tensor = _tensor_or_none(generation_config.decoder_start_token_id, device=device)

        # for BC we also try to get `decoder_start_token_id` or `bos_token_id` (#30892)
        if self.config.is_encoder_decoder:
            decoder_start_token_tensor = (
                decoder_start_token_tensor if decoder_start_token_tensor is not None else bos_token_tensor
            )

        # We can have more than one eos token. Always treat it as a 1D tensor (when it exists).
        if eos_token_tensor is not None and eos_token_tensor.ndim == 0:
            eos_token_tensor = eos_token_tensor.unsqueeze(0)

        # Set pad token if unset (and there are conditions to do so)
        if pad_token_tensor is None and eos_token_tensor is not None:
            # if kwargs_has_attention_mask is not None and not kwargs_has_attention_mask:
                # logger.warning(
                #     "The attention mask and the pad token id were not set. As a consequence, you may observe "
                #     "unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results."
                # )
            pad_token_tensor = eos_token_tensor[0]
            # logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{pad_token_tensor} for open-end generation.")

        # Sanity checks/warnings
        if self.config.is_encoder_decoder and decoder_start_token_tensor is None:
            raise ValueError(
                "`decoder_start_token_id` or `bos_token_id` has to be defined for encoder-decoder generation."
            )
        # if (
        #     eos_token_tensor is not None
        #     and isin_mps_friendly(elements=eos_token_tensor, test_elements=pad_token_tensor).any()
        # ):
        #     if kwargs_has_attention_mask is not None and not kwargs_has_attention_mask:
        #         logger.warning_once(
        #             "The attention mask is not set and cannot be inferred from input because pad token is same as "
        #             "eos token. As a consequence, you may observe unexpected behavior. Please pass your input's "
        #             "`attention_mask` to obtain reliable results."
        #         )
        # if eos_token_tensor is not None and (
        #     torch.is_floating_point(eos_token_tensor) or (eos_token_tensor < 0).any()
        # ):
        #     logger.warning(
        #         f"`eos_token_id` should consist of positive integers, but is {eos_token_tensor}. Your generation "
        #         "will not stop until the maximum length is reached. Depending on other flags, it may even crash."
        #     )

        # Update generation config with the updated special tokens tensors
        # NOTE: this must be written into a different attribute name than the one holding the original special tokens
        generation_config._bos_token_tensor = bos_token_tensor
        generation_config._eos_token_tensor = eos_token_tensor
        generation_config._pad_token_tensor = pad_token_tensor
        generation_config._decoder_start_token_tensor = decoder_start_token_tensor
    
    def _prepare_attention_mask_for_generation(
        self,
        inputs_tensor: torch.Tensor,
        generation_config: GenerationConfig,
        model_kwargs: dict[str, Any],
    ) -> torch.LongTensor:
        pad_token_id = generation_config._pad_token_tensor
        eos_token_id = generation_config._eos_token_tensor

        # `input_ids` may be present in the model kwargs, instead of being the main input (e.g. multimodal model)
        if "input_ids" in model_kwargs and model_kwargs["input_ids"].shape[1] > 0:
            inputs_tensor = model_kwargs["input_ids"]

        # No information for attention mask inference -> return default attention mask
        default_attention_mask = torch.ones(inputs_tensor.shape[:2], dtype=torch.long, device=inputs_tensor.device)
        if pad_token_id is None:
            return default_attention_mask

        is_input_ids = len(inputs_tensor.shape) == 2 and inputs_tensor.dtype in [torch.int, torch.long]
        if not is_input_ids:
            return default_attention_mask

        is_pad_token_in_inputs = (pad_token_id is not None) and (
            torch.isin(elements=inputs_tensor, test_elements=pad_token_id).any()
        )
        is_pad_token_not_equal_to_eos_token_id = (eos_token_id is None) or ~(
            torch.isin(elements=eos_token_id, test_elements=pad_token_id).any()
        )
        can_infer_attention_mask = is_pad_token_in_inputs * is_pad_token_not_equal_to_eos_token_id
        attention_mask_from_padding = inputs_tensor.ne(pad_token_id).long()

        attention_mask = (
            attention_mask_from_padding * can_infer_attention_mask + default_attention_mask * ~can_infer_attention_mask
        )
        return attention_mask
    
    def prepare_outputs_for_update(self, outputs, model_inputs, **kwargs):
        raise NotImplementedError
        
    @torch.no_grad()
    def generate(
        self,
        assistant_model = None,
        use_assistant_model: bool = False,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Optional[Union[GenerateOutput, torch.LongTensor]]:
        # statistics
        self.this_assistant_runtime = 0.0
        self.this_num_accepted_tokens = 0
        self.this_num_candidate_tokens = 0
        all_start = record_time_sync()
        # 0. If it is not a participant, just return a None
        parallel_config = self.parallel_config if not use_assistant_model else self.spec_parallel_config
        if not parallel_config.in_participant:
            # print(f"rank {get_global_rank()} is not a participant of this generation.")
            return None
        
        # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
        self._validate_model_class()
        tokenizer = kwargs.pop("tokenizer", None)  # Pull this out first, we only use it for stopping criteria
        generation_config, model_kwargs = self._prepare_generation_config(generation_config, **kwargs)
        # self._validate_model_kwargs(model_kwargs.copy())
        self._validate_assistant(assistant_model)
        
        # pop candidate refill policy
        if use_assistant_model:
            candidate_refill_policy = kwargs.pop("candidate_refill_policy", _DEFAULT_CANDIDATE_REFILL_POLICY)
            num_assistant_tokens = kwargs.pop("num_assistant_tokens", None)
            assistant_confidence_threshold = kwargs.pop("assistant_confidence_threshold", None)
        # push enable_tree_attention back to leave it for further use    
        enable_tree_attention = kwargs.get("enable_tree_attention", False)
        if (
            enable_tree_attention and 
            not generation_config.is_assistant and 
            not use_assistant_model):
            warnings.warn(f"tree attention can only be used in assistant decoding. Disable tree attention.")
            enable_tree_attention = False
            kwargs["enable_tree_attention"] = enable_tree_attention
            model_kwargs["enable_tree_attention"] = enable_tree_attention

        # 2. Set generation parameters if not already defined
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        accepts_attention_mask = "attention_mask" in set(inspect.signature(self.forward).parameters.keys())
        requires_attention_mask = "encoder_outputs" not in model_kwargs
        kwargs_has_attention_mask = model_kwargs.get("attention_mask", None) is not None

        # 3. Define model inputs
        input_ids, model_input_name, model_kwargs = self._prepare_model_inputs(
            model_kwargs, generation_config
        )
        input_ids: torch.LongTensor
        model_kwargs: Dict
        
        # min_dtype = torch.finfo(dtype).min
        device = input_ids.device
        self._prepare_special_tokens(generation_config, kwargs_has_attention_mask, device=device)
        model_kwargs["use_cache"] = True

        if not kwargs_has_attention_mask and requires_attention_mask and accepts_attention_mask:
            model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
                input_ids, generation_config, model_kwargs
            )


        if generation_config.token_healing:
            raise NotImplementedError
            # input_ids = self.heal_tokens(input_ids, tokenizer)
            
        # 6. Prepare `max_length` depending on other stopping criteria.
        input_ids_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        has_default_min_length = kwargs.get("min_length") is None and generation_config.min_length is not None
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            has_default_min_length=has_default_min_length,
            model_input_name=model_input_name,
            inputs_tensor=input_ids,
            input_ids_length=input_ids_length,
        )

        # 7. handle cache and determine generation mode
        cache_name = "past_key_values"
        requires_cross_attention_cache = (
            self.config.is_encoder_decoder or model_kwargs.get("encoder_outputs") is not None
        )
        if requires_cross_attention_cache:
            raise NotImplementedError("Cross attention cache")
        # If no past cache is specified, create a DynamicCache
        past_cache = model_kwargs.get(cache_name, None)
        if past_cache is None:
            # TODO: chunk
            model_kwargs[cache_name] = TreeDynamicCache()

        self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)

        if not enable_tree_attention:
            generation_mode = GenerationMode.SAMPLE if not use_assistant_model else GenerationMode.ASSISTED_DECODING
        else:
            generation_mode = GenerationMode.TREE_SAMPLE if not use_assistant_model else GenerationMode.TREE_ASSISTED_DECODING
            
        # 8. prepare distribution pre_processing samplers
        prepared_logits_processor = self._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_length,
            encoder_input_ids=input_ids,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            logits_processor=logits_processor,
            device=input_ids.device,
            model_kwargs=model_kwargs,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
        )

        # 9. prepare stopping criteria
        prepared_stopping_criteria = self._get_stopping_criteria(
            generation_config=generation_config, stopping_criteria=stopping_criteria, tokenizer=tokenizer, **kwargs
        )
                
        if model_kwargs["enable_scheduler"] == True and generation_mode != GenerationMode.SAMPLE:
            raise NotImplementedError
        # before_run = record_time_sync()    
        if generation_mode == GenerationMode.SAMPLE:
            # torch.cuda.current_stream().synchronize()
            # before_sampling = time.time()
            # print("before sampling time:", before_sampling - all_start)
            # 13. run sample (it degenerates to greedy search when `generation_config.do_sample=False`)
            use_scheduler = envs.ENABLE_MOE_SCHEDULER
            if use_scheduler:
                result = self._sample2(
                    input_ids,
                    logits_processor=prepared_logits_processor,
                    stopping_criteria=prepared_stopping_criteria,
                    generation_config=generation_config,
                    **model_kwargs,
                )
            else:
                result = self._sample(
                    input_ids,
                    logits_processor=prepared_logits_processor,
                    stopping_criteria=prepared_stopping_criteria,
                    generation_config=generation_config,
                    **model_kwargs,
                )
        elif generation_mode == GenerationMode.TREE_SAMPLE:
            result = self._tree_sample(
                input_ids,
                logits_processor=prepared_logits_processor,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                **model_kwargs,
            )
        elif generation_mode == GenerationMode.ASSISTED_DECODING or generation_mode == GenerationMode.TREE_ASSISTED_DECODING:
            # 12. Get the candidate generator, given the parameterization
            candidate_generator = self._get_candidate_generator(
                generation_config=generation_config,
                assistant_model=assistant_model,
                candidate_refill_policy=candidate_refill_policy,
                logits_processor=logits_processor,
                num_assistant_tokens=num_assistant_tokens,
                assistant_confidence_threshold=assistant_confidence_threshold,
                model_kwargs=model_kwargs,
            )
            if generation_mode == GenerationMode.ASSISTED_DECODING:
                # 13. run assisted generate
                result = self._assisted_decoding(
                    input_ids,
                    candidate_generator=candidate_generator,
                    logits_processor=prepared_logits_processor,
                    stopping_criteria=prepared_stopping_criteria,
                    generation_config=generation_config,
                    **model_kwargs,
                )
            else:
                # 13. run assisted generate
                result = self._tree_assisted_decoding(
                    input_ids,
                    candidate_generator=candidate_generator,
                    logits_processor=prepared_logits_processor,
                    stopping_criteria=prepared_stopping_criteria,
                    generation_config=generation_config,
                    **model_kwargs,
                )
        else:
            raise NotImplementedError

        all_end = record_time_sync()
        # rank0_print(f"before run: {before_run-all_start}, real run: {after_run - before_run}")
        self.this_run_time = all_end - all_start
        self.total_run_time += self.this_run_time
        self.assistant_runtime += self.this_assistant_runtime
        self.num_accepted_tokens += self.this_num_accepted_tokens
        self.num_candidate_tokens += self.this_num_candidate_tokens
        
        return result

    
    def _sample2(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        # init values
        pad_token_id = generation_config._pad_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        max_length = generation_config.max_length
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        do_sample = generation_config.do_sample # It is crucial for assistant model not to use sampling.

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # keep track of which sequences are already finished
        batch_size, cur_len = input_ids.shape
        finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        # model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

        # record_info
        record_info: Dict = model_kwargs.get("record_info", None)
        # dp
        dp_head_group = self.parallel_config.attn_dp_head_group
        dp_size = self.parallel_config.dp_size
        dp_rank = self.parallel_config.dp_rank
        # attn tp
        attn_tp_group = self.parallel_config.attn_tp_group
        # request hub
        offload_kv = model_kwargs.get("offload_kv", False)
        request_hub = RequestHub(model_config=self.config, offload_kv=offload_kv)
        # scheduler
        cross_layer_scheduling_strategy = model_kwargs.get("cross_layer_scheduling_strategy", False)
        enable_token_count_stats = model_kwargs.get("enable_token_count_stats", False)
        # enable_seq_split = model_kwargs.get("enable_seq_split", False)
        enable_runtime_stats = model_kwargs.get("enable_runtime_stats", False)
        enable_device_history_stats = model_kwargs.get("enable_device_history_stats", False)
        enable_expert_history_stats = model_kwargs.get("enable_expert_history_stats", False)
        enable_trajectory_history_stats = model_kwargs.get("enable_trajectory_history_stats", False)
        enable_schedule_overhead_stats = model_kwargs.get("enable_schedule_overhead_stats", False)
        enable_profiler_recording = model_kwargs.get("enable_profiler_recording", False)
        num_min_run_chunks = model_kwargs.get("num_min_run_chunks", -1)
        num_max_deferred_steps = model_kwargs.get("num_max_deferred_steps", -1)
        if record_info is None:
            if (enable_token_count_stats or enable_runtime_stats or enable_device_history_stats or enable_expert_history_stats or enable_schedule_overhead_stats):
                raise ValueError
        if "num_max_deferred_steps" in model_kwargs and num_max_deferred_steps < 0:
            raise ValueError(f"specify `num_max_deferred_steps` with < 0 value: {num_max_deferred_steps}")
        
        scheduler_enable_token_count = enable_token_count_stats or envs.TOKEN_COUNT_DEBUG
        scheduler_enable_trajectory_history = enable_trajectory_history_stats or envs.TRAJECTORY_DEBUG
        scheduler = MoeLayerScheduler(
            self, 
            cross_layer_scheduling_strategy=cross_layer_scheduling_strategy,
            num_min_run_chunks=num_min_run_chunks,
            num_max_deferred_steps=num_max_deferred_steps,
            enable_token_count_stats=scheduler_enable_token_count,
            enable_device_history_stats=enable_device_history_stats,
            enable_expert_history_stats=enable_expert_history_stats,
            enable_trajectory_history_stats=scheduler_enable_trajectory_history,
            record_decide_time=enable_schedule_overhead_stats,
        )
        
        model_kwargs.update({
            "request_hub": request_hub,
            "scheduler": scheduler,
        })
        need_attention_causal_bias = self.is_causal_model ## Causal model must use attention_causal_bias
        
        # _init_compute_stream()
        # _init_comm_stream()
        if enable_profiler_recording:
            # torch.profiler._utils._init_for_cuda_graphs()
            # prof = torch.profiler.profile()
            prof = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
            )
        else:
            prof = contextlib.nullcontext()
            
        while not finished:
            
            batch_size, cur_len = input_ids.shape
            if batch_size % dp_size != 0:
                raise ValueError
            dp_batch_size = batch_size // dp_size
            
            dp_input_ids = input_ids[dp_batch_size * dp_rank: dp_batch_size * (dp_rank+1),:]
            dp_attention_masks = model_kwargs["attention_mask"][dp_batch_size * dp_rank: dp_batch_size * (dp_rank+1),:]
            # show_rank_print(f"dp_batch_size: {dp_batch_size}, dp_rank: {dp_rank}")
            request_hub.append_new_inputs(dp_input_ids, dp_attention_masks)
            request_hub.step(need_attention_causal_bias=need_attention_causal_bias)
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(dp_input_ids, **model_kwargs)
            # check the integrity of inputs
            self.check_model_inputs(model_inputs)
            
                
            # prepare time: 0.01-0.1
            # forward pass to get next token
            torch.cuda.synchronize()
            _before_model_run = time.time()
            with prof:
                outputs = self(**model_inputs, return_dict=True)
                if hasattr(prof, "step"):
                    prof.step()
            torch.cuda.synchronize()
            _after_model_run = time.time()
            
            self.model_forward_time += _after_model_run - _before_model_run
            self.effective_workload += scheduler.effective_workload
            self.shortage_of_all_devices += scheduler.shortage_of_all_devices
            self.all_workload += scheduler.all_workload
                
            show_rank_print(f"############# model time: {self.model_forward_time}", 0)
            # if envs.TOKEN_COUNT_DEBUG:
            #     show_rank_print(f"######## TOKEN_COUNT_DEBUG", 0)
            # if envs.TRAJECTORY_DEBUG:
            #     show_rank_print(f"######## TRAJECTORY_DEBUG", 0)
                
            if record_info is not None and record_info["batch_i"] == record_info["gen_num"] - 1:
                if enable_runtime_stats:
                    if get_global_rank() == 0:
                        with open(record_info["run_time_path"], "w", encoding="utf-8") as f:
                            info_dict = {
                                "batch_i": record_info["batch_i"],
                                "model_time": self.model_forward_time,
                                "layer_time": self.model.layers_forward_time,
                            }
                            f.write(json.dumps(info_dict, ensure_ascii=False) + "\n") 
                
                if hasattr(prof, "export_chrome_trace"):
                    prof.export_chrome_trace(f"{envs.OUTER_ROOT_DIR}/rank_{get_global_rank()}.json")
                
                if scheduler.enable_token_count_stats:
                    show_rank_print(f"######## shortage/effective_workload: {self.shortage_of_all_devices}/{self.effective_workload}", 0)
                    if get_global_rank() == 0 and not envs.TOKEN_COUNT_DEBUG:
                        with open(record_info["token_count_path"], "w", encoding="utf-8") as f:
                            info_dict = {
                                "shortage": self.shortage_of_all_devices,
                                "workload": self.effective_workload,
                                "all_workload": self.all_workload,
                            }
                            f.write(json.dumps(info_dict, ensure_ascii=False) + "\n") 
                            
                # reset time and token count if recorded          
                self.model_forward_time = 0.0
                self.model.layers_forward_time = 0.0
                self.effective_workload = 0
                self.shortage_of_all_devices = 0
                self.all_workload = 0
                
                if enable_device_history_stats:
                    if get_global_rank() == 0:
                        with open(record_info["d_history_path"], "w", encoding="utf-8") as f:
                            for chunk in scheduler.chunk_list:
                                info_dict = {
                                    "chunk_id": chunk.chunk_id,
                                    "history": chunk.global_device_history,
                                }
                                # show_rank_print(f"####chunk {chunk.chunk_id} device history: {chunk.global_device_history}")
                                f.write(json.dumps(info_dict, ensure_ascii=False) + "\n") 
                if enable_expert_history_stats:
                    if get_global_rank() == 0:
                        with open(record_info["e_history_path"], "w", encoding="utf-8") as f:
                            for chunk in scheduler.chunk_list:
                                info_dict = {
                                    "chunk_id": chunk.chunk_id,
                                    "history": chunk.global_expert_history,
                                }
                                # show_rank_print(f"####chunk {chunk.chunk_id} expert history: {chunk.global_expert_history}")
                                f.write(json.dumps(info_dict, ensure_ascii=False) + "\n") 
                if scheduler.enable_trajectory_history_stats:
                    show_rank_print(f"#### Trajectory: {scheduler.trajectory_history}")
                    if get_global_rank() == 0 and not envs.TRAJECTORY_DEBUG:
                        with open(record_info["trajectory_history_path"], "w", encoding="utf-8") as f:
                            info_dict = {
                                "history": scheduler.trajectory_history,
                            }
                            f.write(json.dumps(info_dict, ensure_ascii=False) + "\n") 
                
                
            self.model.layers[-1].self_attn.show_timing_results()
            self.model.layers[-1].self_attn.init_clear_timing_results()
            self.model.layers[-1].mlp.show_timing_results()
            self.model.layers[-1].mlp.init_clear_timing_results()
            # self.expert_compute_full_time = sum([layer.mlp.expert_compute_time for layer in self.model.layers])
            # self.barrier_wait_full_time = sum([layer.mlp.barrier_wait_time for layer in self.model.layers if hasattr(layer.mlp, "barrier_wait_time")])
            # show_rank_print(f"expert compute time: {self.expert_compute_full_time}", 0)
            # show_rank_print(f"wait ep time: {self.barrier_wait_full_time}")
                                
            outputs = self.prepare_outputs_for_update(outputs, model_inputs, **model_kwargs)
            
            # If logits is None, then the model must have parallel_config
            if outputs.logits is None:
                if not hasattr(self, "parallel_config"):
                    raise ValueError("logits is None, but the model doesn't have parallel_config")
                if get_global_rank() == self.parallel_config.all_ranks_group.driver:
                    raise ValueError
                # wait for drivers to broadcast input_ids
                next_tokens = torch.empty([1], device=dp_input_ids.device, dtype=dp_input_ids.dtype)
                next_token_logits = None
                next_token_scores = None    
                selected_token_probs = torch.empty([dp_input_ids.shape[0],1], device=dp_input_ids.device, dtype=self.parallel_config.dtype)
            else:    
                next_token_logits = outputs.logits
                next_token_scores = logits_processor(dp_input_ids, next_token_logits)
                # TODO: maybe no need to compute prob and select
                probs = torch.nn.functional.softmax(next_token_scores, dim=-1)
                        
                if do_sample:
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                else:
                    # argmax is safe
                    next_tokens = torch.argmax(next_token_scores, dim=-1)
                
                selected_token_probs = torch.gather(probs, dim=-1, index=next_tokens.unsqueeze(-1)) if probs is not None else None
            
            # start = record_time_sync()
            need_broadcast_inputs = self.parallel_config.need_broadcast_inputs or (do_sample)
            if need_broadcast_inputs:
                self.parallel_config.all_ranks_group.broadcast(next_tokens, src=self.parallel_config.all_ranks_group.driver)
                if generation_config.is_assistant:
                    # condifence criteria need selected token probs
                    self.parallel_config.all_ranks_group.broadcast(selected_token_probs, src=self.parallel_config.all_ranks_group.driver)
                        
            ## gather outputs from dp
            ## first gather among dp-group drivers
            if attn_tp_group.is_driver():
                next_tokens = dp_head_group.all_gather_into_tensor_auto(next_tokens).output.view(batch_size)
            else:
                next_tokens = input_ids.new_empty(batch_size)
            ## then broadcast to each tp-group
            attn_tp_group.broadcast(next_tokens)
            
            ## update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1) # TODO: if not the same order?

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, selected_token_probs)
            finished = unfinished_sequences.max() == 0
            num_new_tokens = 1
                                
            cur_len += num_new_tokens
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                generation_config,
                model_kwargs,
                model_inputs,
                num_new_tokens,
            )
            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs

        if return_dict_in_generate:
            return GenerateDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores, # tuple, remember to stack
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        else:
            return input_ids
                    
    def _sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        # init values
        pad_token_id = generation_config._pad_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        max_length = generation_config.max_length
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        do_sample = generation_config.do_sample # It is crucial for assistant model not to use sampling.

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # keep track of which sequences are already finished
        batch_size, cur_len = input_ids.shape
        finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

        if model_kwargs["enable_profiler_recording"] is True:
            torch.profiler._utils._init_for_cuda_graphs()
            prof = torch.profiler.profile(activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ])
        else:
            prof = contextlib.nullcontext()
            
        while not finished:
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            
            # forward pass to get next token
            with prof:
                outputs = self(**model_inputs, return_dict=True)
            if hasattr(prof, "export_chrome_trace"):
                prof.export_chrome_trace(f"~/rank_{get_global_rank()}.json")
            # forward_end = record_time_sync()
            # print(f"rank {global_rank} forcausalLM forward", forward_end - start)
            
            outputs = self.prepare_outputs_for_update(outputs, model_inputs, **model_kwargs)
            
            # If logits is None, then the model must have parallel_config
            if outputs.logits is None:
                if not hasattr(self, "parallel_config"):
                    raise ValueError("logits is None, but the model doesn't have parallel_config")
                if get_global_rank() == self.parallel_config.all_ranks_group.driver:
                    raise ValueError
                # wait for drivers to broadcast input_ids
                next_tokens = torch.empty([1], device=input_ids.device, dtype=input_ids.dtype)
                next_token_logits = None
                next_token_scores = None    
                selected_token_probs = torch.empty([input_ids.shape[0],1], device=input_ids.device, dtype=self.parallel_config.dtype)
            else:    
                next_token_logits = outputs.logits[:, -1, :].clone()
                next_token_scores = logits_processor(input_ids, next_token_logits)
                # TODO: maybe no need to compute prob and select
                probs = torch.nn.functional.softmax(next_token_scores, dim=-1)
                        
                if do_sample:
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                else:
                    # argmax is safe
                    next_tokens = torch.argmax(next_token_scores, dim=-1)
                
                selected_token_probs = torch.gather(probs, dim=-1, index=next_tokens.unsqueeze(-1)) if probs is not None else None
            
            # start = record_time_sync()
            need_broadcast_inputs = self.parallel_config.need_broadcast_inputs or (do_sample)
            if need_broadcast_inputs:
                self.parallel_config.all_ranks_group.broadcast(next_tokens, src=self.parallel_config.all_ranks_group.driver)
                if generation_config.is_assistant:
                    # condifence criteria need selected token probs
                    self.parallel_config.all_ranks_group.broadcast(selected_token_probs, src=self.parallel_config.all_ranks_group.driver)
                        
            # finished sentences should have their next token be a padding token
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, selected_token_probs)
            finished = unfinished_sequences.max() == 0
            num_new_tokens = 1
                                
            cur_len += num_new_tokens
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                generation_config,
                model_kwargs,
                model_inputs,
                num_new_tokens,
            )
            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs

        if return_dict_in_generate:
            return GenerateDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores, # tuple, remember to stack
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        else:
            return input_ids
        
        
    def _check_same_driver(
        self,
        candidate_generator: AssistedCandidateGenerator
    ):
        # Draft model and verification model must have the same driver
        assistant_model = candidate_generator.assistant_model
        if self.parallel_config.all_ranks_group.driver != assistant_model.parallel_config.all_ranks_group.driver:
            raise ValueError(f"Draft model and verification model must have the same driver.")
        
    def _broadcast_candidate_result(self, input_ids:torch.Tensor, candidate_input_ids:Optional[torch.Tensor]):
        """
            Return: candidate_input_ids, candidate_length
        """
        # In assisted decoding, some process may not run on draft model.
        # Then there is no need to broadcast candidate ids and length.
        if self.spec_parallel_config.need_broadcast_candidate_from_draft:
            global_rank = get_global_rank()
            if global_rank == self.parallel_config.driver:
                if candidate_input_ids is None:
                    raise ValueError
                candidate_length = candidate_input_ids.shape[1] - input_ids.shape[1]
            else:
                candidate_length = 0
            
            # start = record_time_sync()
            candidate_length = torch.tensor([candidate_length], dtype=torch.long, device=self.parallel_config.device)
            self.parallel_config.all_ranks_group.broadcast(candidate_length, src=self.parallel_config.driver)
            candidate_length = candidate_length.item()
            # print(f"rank {global_rank} length broadcast", record_time_sync() - start)
            
            # Caution: candidate length could be 0
            if candidate_length > 0:
                if global_rank == self.parallel_config.driver:
                    new_candidate_input_ids = candidate_input_ids[:,-candidate_length:]
                else:
                    new_candidate_input_ids = torch.zeros((input_ids.shape[0], candidate_length), dtype=input_ids.dtype, device=input_ids.device)
                # start = record_time_sync()
                self.parallel_config.all_ranks_group.broadcast(new_candidate_input_ids, src=self.parallel_config.driver)
                candidate_input_ids = torch.cat([input_ids, new_candidate_input_ids], dim=-1)
                # print(f"rank {global_rank} new_candidate broadcast", record_time_sync() - start)
            else:
                candidate_input_ids = input_ids
        else:
            candidate_length = candidate_input_ids.shape[1] - input_ids.shape[1]
        
        return candidate_input_ids, candidate_length
    
    def _broadcast_verification_result(self, n_matches:torch.Tensor, valid_tokens:Optional[torch.Tensor], input_ids:torch.Tensor):
        """
            Return: n_matches, valid_tokens
        """
        global_rank = get_global_rank()
        # start = record_time_sync()
        # from driver to all specs
        self.spec_parallel_config.all_ranks_group.broadcast(n_matches, src=self.spec_parallel_config.driver)
        if global_rank != self.spec_parallel_config.driver:
            # need +1 for bonus token?
            valid_tokens = torch.empty((input_ids.shape[0], n_matches+1), device=input_ids.device, dtype=input_ids.dtype)
        self.spec_parallel_config.all_ranks_group.broadcast(valid_tokens, src=self.spec_parallel_config.driver)
        # print(f"rank {global_rank} valid tokens broadcast", record_time_sync() - start)
        return n_matches, valid_tokens
    
    def _broadcast_tree_verification_result(
        self, 
        valid_tokens:Optional[torch.Tensor], 
        valid_retrieve_indices:Optional[torch.Tensor],
        n_matches:torch.Tensor,
        input_ids:torch.Tensor
    ):
        """
            Return: valid_tokens, valid_retrieve_indices, n_matches
        """
        global_rank = get_global_rank()
        # start = record_time_sync()
        # from driver to all specs
        self.spec_parallel_config.all_ranks_group.broadcast(n_matches, src=self.spec_parallel_config.driver)
        if global_rank != self.spec_parallel_config.driver:
            # need +1 for bonus token?
            valid_tokens = torch.empty((input_ids.shape[0], n_matches+1), device=input_ids.device, dtype=input_ids.dtype)
            valid_retrieve_indices = torch.empty((input_ids.shape[0], n_matches), device=input_ids.device, dtype=input_ids.dtype)
        
        # if not valid_tokens.is_contiguous():
        #     rank0_print(f"n_match: {n_matches} shape: {valid_tokens.shape}")
        self.spec_parallel_config.all_ranks_group.broadcast(valid_tokens, src=self.spec_parallel_config.driver)
        self.spec_parallel_config.all_ranks_group.broadcast(valid_retrieve_indices, src=self.spec_parallel_config.driver)
        # print(f"rank {global_rank} valid tokens broadcast", record_time_sync() - start)
        return valid_tokens, valid_retrieve_indices, n_matches
    
    def _get_candidate_generator(
        self,
        generation_config: GenerationConfig,
        assistant_model,
        candidate_refill_policy: CandidateRefillPolicy,
        logits_processor: LogitsProcessorList,
        num_assistant_tokens: Optional[int],
        assistant_confidence_threshold: Optional[float],
        model_kwargs: Dict,
    ):
        candidate_generator_class = AssistedCandidateGenerator if model_kwargs["enable_tree_attention"] is False else TreeAssistedCandidateGenerator
        candidate_generator = candidate_generator_class(
            assistant_model=assistant_model,
            candidate_refill_policy=candidate_refill_policy,
            generation_config=generation_config,
            model_kwargs=model_kwargs,
            logits_processor=logits_processor,
            num_assistant_tokens=num_assistant_tokens,
            assistant_confidence_threshold=assistant_confidence_threshold,
        )
        return candidate_generator
            
    def _assisted_decoding(
        self,
        input_ids: torch.LongTensor,
        candidate_generator: AssistedCandidateGenerator,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        
        self.this_assistant_runtime = 0.0
        self.this_num_accepted_tokens = 0
        self.this_num_candidate_tokens = 0
        
        self._check_same_driver(candidate_generator)
        
        # init values
        do_sample = generation_config.do_sample
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # keep track of which sequences are already finished
        batch_size = input_ids.shape[0]
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)
        
        if hasattr(self, "num_assistant_tokens"):
            candidate_generator.num_assistant_tokens = self.num_assistant_tokens
            candidate_generator.assistant_model.generation_config.num_assistant_tokens = self.num_assistant_tokens
        if hasattr(self, "num_assistant_tokens_schedule"):
            candidate_generator.assistant_model.generation_config.num_assistant_tokens_schedule = self.num_assistant_tokens_schedule
            
        finished = False
        candidate_length = 0
        while not finished:
            cur_len = input_ids.shape[-1]

            #  1. Fetch candidate sequences from a `CandidateGenerator`
            start = record_time_sync()
            candidate_input_ids, candidate_logits = candidate_generator.get_candidates(input_ids, candidate_length)
            self.this_assistant_runtime += record_time_sync() - start
            
            global_rank = get_global_rank()
            # print(f"rank {global_rank} sd start time: {record_time_sync()}")
            # Here, if you are not a participant of the verification model, you must be participant of draft model,
            # otherwise, you will be blocked from generate() at the beginning
            # so, just wait for verification.
            if not self.parallel_config.in_participant:
                n_matches = torch.zeros((), dtype=torch.long, device=self.spec_parallel_config.device)
            else:
                # Broadcast candidate_input_ids from driver to all ranks on verification model
                candidate_input_ids, candidate_length = self._broadcast_candidate_result(input_ids, candidate_input_ids)
                
                # Prepare verification model inputs
                is_done_candidate = stopping_criteria(candidate_input_ids, None)
                candidate_kwargs = copy.copy(model_kwargs)
                candidate_kwargs = _prepare_2d_attention_mask(
                    candidate_kwargs, candidate_input_ids.shape[1]
                )
                candidate_kwargs = _prepare_token_type_ids(candidate_kwargs, candidate_input_ids.shape[1])
                if "cache_position" in candidate_kwargs:
                    candidate_kwargs["cache_position"] = torch.cat(
                        (
                            candidate_kwargs["cache_position"],
                            torch.arange(cur_len, cur_len + candidate_length, device=input_ids.device, dtype=torch.long),
                        ),
                        dim=0,
                    )
                model_inputs = self.prepare_inputs_for_generation(candidate_input_ids, **candidate_kwargs)
                model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
                model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

                # Run a forward pass of verification on the candidate sequence
                outputs = self(**model_inputs)
                
                # It is verification model driver's job to evaluate candidates, others should just wait
                # Why? The sampling procedure could give different outputs on each process, which causes trouble
                if global_rank == self.parallel_config.driver:
                    # sample_start = record_time_sync()
                    new_logits = outputs.logits[:, -candidate_length - 1 :]  # excludes the input prompt if present
                    if return_dict_in_generate and output_logits:
                        next_token_logits = new_logits.clone()
                    if len(logits_processor) > 0:
                        for i in range(candidate_length + 1):
                            new_logits[:, i, :] = logits_processor(candidate_input_ids[:, : cur_len + i], new_logits[:, i, :])

                    # Select the accepted tokens
                    if do_sample and candidate_logits is not None:
                        valid_tokens, n_matches = _speculative_sampling(
                            candidate_input_ids,
                            candidate_logits,
                            candidate_length,
                            new_logits,
                            is_done_candidate,
                        )
                    else:
                        if do_sample:
                            probs = new_logits.softmax(dim=-1)
                            selected_tokens = torch.multinomial(probs[0, :, :], num_samples=1).squeeze(1)[None, :]
                        else:
                            selected_tokens = new_logits.argmax(dim=-1)
                        candidate_new_tokens = candidate_input_ids[:, cur_len:]
                        n_matches = ((~(candidate_new_tokens == selected_tokens[:, :-1])).cumsum(dim=-1) < 1).sum()                       
                        # Ensure we don't generate beyond max_len or an EOS token
                        if is_done_candidate and n_matches == candidate_length:
                            n_matches -= 1
                        # n_matches can be -1, to get valid tokens right from selected_tokens
                        valid_tokens = selected_tokens[:, : n_matches + 1]
                    # sample_end = record_time_sync()
                    # print(f"sample need: {sample_end - sample_start}")
                    
                    self.this_num_accepted_tokens += n_matches
                    self.this_num_candidate_tokens += candidate_length
                    # rank0_print(f"accept: {n_matches}/{candidate_length} {self.this_num_accepted_tokens}/{self.this_num_candidate_tokens}")
                else:
                    next_token_logits = None
                    new_logits = None
                    n_matches = torch.zeros((), device=global_rank, dtype=torch.int64)
                    valid_tokens = None
                    
            # n_matches and valid_tokens should be on both draft model and verification model
            # now only driver has the correct n_matches and valid_tokens, because it is a sampling result
            n_matches, valid_tokens = self._broadcast_verification_result(n_matches, valid_tokens, input_ids)

            # 4. Update variables according to the number of matching assistant tokens. Remember: the token generated
            # by the model after the last candidate match is also valid, as it is generated from a correct sequence.
            # Because of this last token, assisted generation search reduces to a normal greedy search/sample if there
            # is no match.
            # print(f"rank {global_rank} sd end time: {record_time_sync()}")

            # Get the valid continuation, after the matching tokens
            # rank0_print(valid_tokens)
            # if new_logits is not None:
            #     rank0_print(new_logits.max(dim=-1).values)
            input_ids = torch.cat((input_ids, valid_tokens), dim=-1)
            # if streamer is not None:
            #     streamer.put(valid_tokens.cpu())
            
            new_cur_len = input_ids.shape[-1]
            # Discard past key values relative to unused assistant tokens
            new_cache_size = new_cur_len - 1
            
            outputs.past_key_values = _crop_past_key_values(outputs.past_key_values, new_cache_size)
            
            # Update the candidate generation strategy if needed
            # Note: update candidate_refill
            candidate_generator.update_candidate_strategy(candidate_length, n_matches)

            # Store scores, attentions and hidden_states when required
            # Assistant: modified to append one tuple element per token, as in the other generation methods.
            if return_dict_in_generate:
                if output_scores:
                    scores += tuple(new_logits[:, i, :] for i in range(n_matches + 1)) if new_logits else None
                if output_logits:
                    raw_logits += (next_token_logits,)

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                generation_config,
                model_kwargs,
                model_inputs,
                num_new_tokens=n_matches + 1,
            )
            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            # print(input_ids.shape[-1], generation_config.max_length)
            finished = unfinished_sequences.max() == 0

        if return_dict_in_generate:
            return GenerateDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        else:
            return input_ids
        
    def _broadcast_tree_candidate_result(self, 
                                  candidate_input_ids: Optional[torch.Tensor], 
                                  retrieve_indices: Optional[torch.Tensor],  
                                  tree_attention_config: TreeAttentionConfig,
                                  input_ids:torch.Tensor,
                                  ):
        # In assisted decoding, some process may not run on draft model.
        # Then there is no need to broadcast candidate ids and length.
        if self.spec_parallel_config.need_broadcast_candidate_from_draft:
            # determine the flat candidate length
            global_rank = get_global_rank()
            if global_rank == self.parallel_config.driver:
                if candidate_input_ids is None:
                    raise ValueError
                flat_candidate_length = candidate_input_ids.shape[1] - input_ids.shape[1]
            else:
                flat_candidate_length = 0
                
            flat_candidate_length = torch.tensor([flat_candidate_length], dtype=torch.long, device=self.parallel_config.device)
            self.parallel_config.all_ranks_group.broadcast(flat_candidate_length, src=self.parallel_config.driver)
            flat_candidate_length = flat_candidate_length.item()
                
            # Caution: flat candidate length could be 0
            if flat_candidate_length > 0:
                retrieve_indices_shape = (input_ids.shape[0], flat_candidate_length, flat_candidate_length // tree_attention_config.top_k, )
                if global_rank == self.parallel_config.driver:
                    if retrieve_indices_shape != tuple(retrieve_indices.shape):
                        raise ValueError(f"retrieve_indices_shape: {retrieve_indices_shape}, .shape: {retrieve_indices.shape}")
                    new_candidate_input_ids = candidate_input_ids[:,-flat_candidate_length:]
                else:
                    new_candidate_input_ids = torch.zeros((input_ids.shape[0], flat_candidate_length), dtype=input_ids.dtype, device=input_ids.device)
                    retrieve_indices = torch.empty(retrieve_indices_shape, dtype=input_ids.dtype, device=input_ids.device)
                # start = record_time_sync()

                self.parallel_config.all_ranks_group.broadcast(new_candidate_input_ids, src=self.parallel_config.driver)
                self.parallel_config.all_ranks_group.broadcast(retrieve_indices, src=self.parallel_config.driver)
                
                candidate_input_ids = torch.cat([input_ids, new_candidate_input_ids], dim=-1)
                # print(f"rank {global_rank} new_candidate broadcast", record_time_sync() - start)
            else:
                candidate_input_ids = input_ids
        else:
            flat_candidate_length = candidate_input_ids.shape[1] - input_ids.shape[1]
        
        return candidate_input_ids, retrieve_indices, flat_candidate_length
    
    def _prepare_tree_verification_attention_mask(
        candidate_input_ids_length: int,
        flat_candidate_length: int,
        attention_mask_2d: torch.LongTensor,
        cache_position: torch.LongTensor,
        retrieve_indices: torch.Tensor,
        past_kv_length:int, 
        dtype: torch.dtype,
        device: torch.device,
    ):
        raise NotImplementedError
      
    def _tree_assisted_decoding(
        self,
        input_ids: torch.LongTensor,
        candidate_generator: TreeAssistedCandidateGenerator,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        
        # rank0_print(f"input len: {input_ids.shape[-1]}")
        self._check_same_driver(candidate_generator)
        
        # init values
        do_sample = generation_config.do_sample
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None
        
        # tree use
        eos_stop_criteria = [sc for sc in stopping_criteria if isinstance(sc, EosTokenCriteria)]
        eos_token_id: torch.Tensor = None if eos_stop_criteria == [] else eos_stop_criteria[0].eos_token_id
        valid_retrieve_indices = None # for the 1-st candidator cache cropping

        # keep track of which sequences are already finished
        batch_size = input_ids.shape[0]
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)
        model_kwargs["attention_mask_2d"] = model_kwargs["attention_mask"]
        
        finished = False
        
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
        # Run a forward pass of verification on the candidate sequence
        outputs = self(**model_inputs)
        
        new_logits = outputs.logits[:, -1:, :]
        next_tokens = new_logits.argmax(-1)
        input_ids = torch.cat([input_ids, next_tokens], dim=-1)
        
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs,
            generation_config,
            model_kwargs,
            model_inputs,
            num_new_tokens=1,
        )
        unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, None)
        finished = unfinished_sequences.max() == 0
        
        candidate_generator.assistant_kwargs["attention_mask"] = model_kwargs["attention_mask_2d"]
        
        if hasattr(self, "num_assistant_tokens"):
            candidate_generator.num_assistant_tokens = self.num_assistant_tokens
            candidate_generator.assistant_model.generation_config.num_assistant_tokens = self.num_assistant_tokens
        if hasattr(self, "num_assistant_tokens_schedule"):
            candidate_generator.assistant_model.generation_config.num_assistant_tokens_schedule = self.num_assistant_tokens_schedule
        
        flat_candidate_length = 0 # any value is ok, won't be used until second iter
        while not finished:
            cur_len = input_ids.shape[-1]
            #  1. Fetch candidate sequences from a `CandidateGenerator`
            candidate_start = record_time_sync()
            candidate_input_ids, retrieve_indices, candidate_flat_probs, run_candidate_once = candidate_generator.get_candidates(
                input_ids, valid_retrieve_indices
            )
            candidate_end = record_time_sync()
            self.this_assistant_runtime += candidate_end - candidate_start
            # rank0_print(f"candidate time: {candidate_end - candidate_start}")
            
            global_rank = get_global_rank()

            # Here, if you are not a participant of the verification model, you must be participant of draft model,
            # otherwise, you will be blocked from generate() at the beginning
            # so, if not a participant of verification model, just wait for another draft.
            if not self.parallel_config.in_participant:
                raise NotImplementedError
                # flat_candidate_length = candidate_input_ids.shape[-1] - input_ids.shape[-1]
                # valid_tokens = torch.empty([generation_config.tree_attention_config.depth], dtype=torch.int64, device=self.spec_parallel_config.device)
            else:
                # maybe reach candidate max_length (real max length - 1)
                if run_candidate_once is False:
                    candidate_input_ids = input_ids
                    retrieve_indices = torch.zeros((input_ids.shape[0], 0), dtype=torch.long, device=self.parallel_config.device)
                    flat_candidate_length = 0
                    candidate_length = 0
                else:
                    # Broadcast candidate_input_ids from driver to all ranks on verification model
                    candidate_input_ids, retrieve_indices, flat_candidate_length = self._broadcast_tree_candidate_result(
                        candidate_input_ids, retrieve_indices, generation_config.tree_attention_config, input_ids
                    )
                    candidate_length = retrieve_indices.shape[-1]
                    
                # Prepare verification model inputs
                # is_done_candidate = stopping_criteria(candidate_input_ids, None)
                verification_kwargs = copy.copy(model_kwargs) # shallow copy
                past_kv_length = verification_kwargs.get("past_key_values").get_seq_length()
                verification_kwargs["cache_position"] = _prepare_tree_verification_cache_position(
                    candidate_input_ids.shape[1], past_kv_length, self.parallel_config.device
                )
                verification_kwargs["attention_mask"] = self._prepare_tree_verification_attention_mask(
                    verification_kwargs["attention_mask_2d"],
                    candidate_input_ids_length=candidate_input_ids.shape[1], 
                    flat_candidate_length=flat_candidate_length,
                    cache_position=verification_kwargs["cache_position"],
                    retrieve_indices=retrieve_indices, 
                    past_kv_length=past_kv_length,
                    dtype=self.parallel_config.dtype,
                    device=self.parallel_config.device,
                )
                verification_kwargs["position_ids"] = _prepare_tree_verification_position_ids(
                    candidate_input_ids.shape[1], 
                    flat_candidate_length,
                    past_kv_length, 
                    candidate_length,
                    generation_config.tree_attention_config.top_k,
                    input_ids.shape[0],
                    self.parallel_config.device,
                )
                model_inputs = self.prepare_inputs_for_generation(candidate_input_ids, **verification_kwargs)
                model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
                model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

                # Run a forward pass of verification on the candidate sequence
                # verification_start = record_time_sync()
                outputs = self(**model_inputs)
                # verification_end = record_time_sync()
                # rank0_print(f"verification time: {verification_end-candidate_end}")
                
                # It is verification model driver's job to evaluate candidates, others should just wait
                # Why? The sampling procedure could give different outputs on each process, which causes trouble
                if global_rank == self.parallel_config.driver:
                    # sample_start = record_time_sync()
                    new_logits = outputs.logits[:, -flat_candidate_length - 1 :]
                    if return_dict_in_generate and output_logits:
                        next_token_logits = new_logits.clone()
                    if len(logits_processor) > 0: # no logits processor needs input_ids for now
                        # new_logits = logits_processor(candidate_input_ids[:, : cur_len], new_logits)
                        new_logits = logits_processor(None, new_logits)

                    # Select the accepted tokens
                    if do_sample:
                        if candidate_length == 0:
                            valid_tokens = torch.multinomial(new_logits.softmax(dim=-1)[0, :, :], num_samples=1).squeeze(1)[None, :] # TODO: batch
                            n_matches = torch.zeros((), dtype=torch.long, device=self.parallel_config.device)
                            valid_retrieve_indices = torch.zeros((batch_size,0), dtype=torch.long, device=self.parallel_config.device)
                        else:
                            if candidate_flat_probs is None:
                                raise ValueError
                            # tree_verify_start = record_time_sync()
                            # rank0_print(f"logit process time: {tree_verify_start - verification_end}")
                            best_candidate, n_matches, valid_tokens = _tree_speculative_sampling(
                                candidate_flat_input_ids=candidate_input_ids[:, cur_len:],
                                candidate_flat_probs=candidate_flat_probs,
                                retrieve_indices=retrieve_indices,
                                new_logits=new_logits,
                                eos_token_id=eos_token_id,
                            )
                            # tree_verify_end = record_time_sync()
                            # rank0_print(f"tree verify time: {tree_verify_end - tree_verify_start}")
                            if n_matches.item() > 0:
                                valid_retrieve_indices = retrieve_indices[:,best_candidate,:n_matches].contiguous()
                            else:
                                valid_retrieve_indices = torch.empty((batch_size,0), dtype=torch.long, device=self.parallel_config.device)
                    else:
                        if do_sample:
                            # do_sample but path probs is None, not supported
                            raise NotImplementedError(f"do_sample but path probs is None, not supported")
                            # vrfy_flat_selected_tokens = torch.multinomial(new_logits.softmax(dim=-1)[0, :, :], num_samples=1).squeeze(1)[None, :]
                        else:
                            vrfy_flat_selected_tokens = new_logits.argmax(dim=-1)
                    
                        # retrieve selected_tokens from vrfy_flat_selected_tokens
                        if candidate_length > 0:
                            # tree_verify_start = record_time_sync()
                            # rank0_print(f"logit process time: {tree_verify_start - verification_end}")
                            vrfy_retrieved_selected_tokens = _retrieve_from_flat(vrfy_flat_selected_tokens, retrieve_indices, is_vrfy=True)        
                            candidate_retrieved_new_tokens = _retrieve_from_flat(candidate_input_ids[:, cur_len:], retrieve_indices, is_vrfy=False)
                            best_candidate, n_matches, valid_tokens = _tree_token_verification(
                                candidate_retrieved_new_tokens, 
                                vrfy_retrieved_selected_tokens, 
                                vrfy_flat_selected_tokens, 
                                retrieve_indices,
                                eos_token_id, 
                            )
                            # tree_verify_end = record_time_sync()
                            # rank0_print(f"tree verify time: {tree_verify_end - tree_verify_start}")
                            # print(valid_tokens)
                            if n_matches.item() > 0:
                                valid_retrieve_indices = retrieve_indices[:,best_candidate,:n_matches].contiguous()
                            else:
                                valid_retrieve_indices = torch.empty((batch_size,0), dtype=torch.long, device=self.parallel_config.device)
                        else:
                            valid_tokens = vrfy_flat_selected_tokens
                            n_matches = torch.zeros((), dtype=torch.long, device=self.parallel_config.device)
                            valid_retrieve_indices = torch.zeros((batch_size,0), dtype=torch.long, device=self.parallel_config.device)
                    # sample_end = record_time_sync()
                    # print(f"sample need: {sample_end - sample_start}")
                    self.this_num_accepted_tokens += n_matches
                    self.this_num_candidate_tokens += candidate_length
                    # rank0_print(f"accept: {n_matches}/{candidate_length} {self.this_num_accepted_tokens}/{self.this_num_candidate_tokens}")
                else:
                    next_token_logits = None
                    new_logits = None
                    
                    n_matches = torch.zeros((), device=self.parallel_config.device, dtype=torch.int64)
                    valid_tokens = None
                    valid_retrieve_indices = None # as we don't know the shape
                    
            # n_matches and valid_tokens should be on both draft model and verification model
            # now only driver has the correct n_matches and valid_tokens, because it is a sampling result
            valid_tokens, valid_retrieve_indices, n_matches = self._broadcast_tree_verification_result(
                valid_tokens, valid_retrieve_indices, n_matches, input_ids
            )

            # print(f"rank {global_rank} sd end time: {record_time_sync()}")

            # Get the valid continuation, after the matching tokens
            # print(valid_tokens)
            input_ids = torch.cat((input_ids, valid_tokens), dim=-1)
            new_cur_len = input_ids.shape[-1]
            new_cache_size = new_cur_len - 1
            
            # Discard past key values relative to unused assistant tokens
            outputs.past_key_values = _crop_past_key_values(outputs.past_key_values, new_cache_size, valid_retrieve_indices)
            
            # 5. Update the candidate generation strategy if needed
            # Note: update candidate_refill
            candidate_generator.update_candidate_strategy(candidate_length, n_matches)

            # Store scores, attentions and hidden_states when required
            # Assistant: modified to append one tuple element per token, as in the other generation methods.
            if return_dict_in_generate:
                if output_scores:
                    scores += tuple(new_logits[:, i, :] for i in range(n_matches + 1)) if new_logits else None
                if output_logits:
                    raw_logits += (next_token_logits,)

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                generation_config,
                model_kwargs,
                model_inputs,
                num_new_tokens=n_matches + 1,
            )
            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            finished = unfinished_sequences.max() == 0

            
        if return_dict_in_generate:
            return GenerateDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        else:
            return input_ids
        
    
    def _tree_sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        # init values
        # if dist.get_rank() == 0:
        #     print(f"sample input len: {input_ids.shape[-1]}")
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        
        scores = ()
        
        do_sample = generation_config.do_sample # It is crucial for assistant model not to use sampling.

        # keep track of which sequences are already finished
        token_ids_factory = {"dtype": input_ids.dtype, "device": input_ids.device}
        # retrieve_indices is a relative index matrix
        tree_attention_config: TreeAttentionConfig = generation_config.tree_attention_config
        # get max depth for generation
        max_depth = min(generation_config.max_length - input_ids.shape[-1], tree_attention_config.depth)
        top_k = tree_attention_config.top_k
        
        batch_size = input_ids.shape[0]
        start_len = input_ids.shape[1]
        
        # each process should have:
        retrieve_indices = None
        new_input_ids_retrieve_indices = torch.arange(0,top_k, **token_ids_factory).view(1,top_k,1).expand(batch_size, top_k, 1)
        pad_retrieve_indices = torch.full((batch_size,top_k,1), fill_value=0, dtype=torch.long, device=self.parallel_config.device)
        
        path_probs = None
        
        # broadcast:
        topk_from_old_indices = new_input_ids_retrieve_indices.clone().squeeze(-1)
        topk_leaf_probs = torch.empty((batch_size,top_k),dtype=self.parallel_config.dtype,device=self.parallel_config.device)
        next_tokens = torch.empty([batch_size,top_k], **token_ids_factory)
                    
        # record some useful information    
        total_new_tokens = 0
        # prepare model kwargs
        model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)
        # tree-specific: attention mask will be 4d, so save 2d
        model_kwargs["attention_mask_2d"] = model_kwargs["attention_mask"]
        model_kwargs["topk_retrieve_indices"] = new_input_ids_retrieve_indices
            
        finished = False
        while not finished:
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            # forward pass to get next token
            # start = record_time_sync()
            outputs = self(**model_inputs, return_dict=True)
            total_new_tokens += 1
            # forward_end = record_time_sync()
            # rank0_print(f"tree sample length: {top_k}, time: {forward_end - start}")
            
            topk_retrieve_indices = model_kwargs.pop("topk_retrieve_indices") # pop
            if total_new_tokens == 1:
                # the 1st generation need manual init
                model_kwargs["attention_mask"] = model_inputs["attention_mask"] # change 2d to 4d
                model_kwargs["position_ids"] = model_inputs["position_ids"] # record 1-st position ids
                
            # Here, you must be the draft model, as verification sampling is in assisted_decoding
            # If logits is None, then the model must have parallel_config
            if outputs.logits is None:
                if not hasattr(self, "parallel_config"):
                    raise ValueError("logits is None, but the model doesn't have parallel_config")
                if get_global_rank() == self.parallel_config.all_ranks_group.driver:
                    raise ValueError
                next_token_probs = None
                # wait for drivers to broadcast
            else:
                # get next_tokens, topk_leaf_probs, topk_from_old_indices(if>1)
                if total_new_tokens == 1:
                    # the 1st generation is different, as we need to set up the tree
                    # path probs is from the 1st logits
                    next_token_logits = outputs.logits[:, -1, :].clone()
                    next_token_scores = logits_processor(input_ids, next_token_logits)
                    next_token_probs = next_token_scores.softmax(dim=-1)
                    topk_next_token_probs, next_tokens = next_token_probs.topk(top_k, dim=-1) # [bsz, top_k]
                    topk_leaf_probs = topk_next_token_probs # [bsz, top_k]
                else:                    
                    next_token_logits = outputs.logits[:, -top_k:, :].clone()
                    next_token_scores = logits_processor(input_ids, next_token_logits)
                    # need prob to choose top_k
                    next_token_probs = next_token_scores.softmax(dim=-1) # [bsz, top_k, logits_dim]
                    # get draft trees, next_token_probs, next_tokens: [bsz, top_k, top_k]
                    topk_next_token_probs, next_tokens = next_token_probs.topk(top_k, dim=-1)
                    
                    # update retrieve_indices and path_prob: find the top-k largest
                    # probs and id must be a 1-1
                    leaf_probs: torch.Tensor = (topk_leaf_probs[:,:,None].expand(batch_size, top_k, top_k) * topk_next_token_probs).view(batch_size,top_k**2)
                    # we use sort here, but actually only need to know who is or is not topk
                    # [bsz, top_k]
                    topk_leaf_probs, topk_leaf_indices = leaf_probs.topk(top_k, dim=-1)
                    # choose next tokens, [bsz, top_k]
                    next_tokens = next_tokens.view(batch_size,top_k**2).gather(dim=-1, index=topk_leaf_indices)
                    # new_indices: [bsz, top_k], each element is in [0,top_k**2]
                    # now, divide new indices with top_k to choose past retrieve indices and path probs
                    topk_from_old_indices = topk_leaf_indices // top_k
                    
                    # TODO: tree pruning?
                    # TODO: eos id early stopping?
                
            if self.parallel_config.need_broadcast_inputs:
                self.parallel_config.all_ranks_group.broadcast(next_tokens, src=self.parallel_config.all_ranks_group.driver)
                self.parallel_config.all_ranks_group.broadcast(topk_leaf_probs, src=self.parallel_config.all_ranks_group.driver)
                if total_new_tokens > 1:
                    # 1-st generation does not need this item, as it is default to be range(0,top_k)    
                    self.parallel_config.all_ranks_group.broadcast(topk_from_old_indices, src=self.parallel_config.all_ranks_group.driver)
                        
            if total_new_tokens == 1:
                # topk_retrieve_indices
                retrieve_indices = new_input_ids_retrieve_indices.clone()
                topk_path_probs = topk_leaf_probs.unsqueeze(-1)
                path_probs = topk_path_probs
            else:
                # form retrieve indices
                new_input_ids_retrieve_indices = torch.arange((total_new_tokens-1)*top_k,total_new_tokens*top_k, **token_ids_factory).view(1,top_k,1).expand(batch_size, top_k, 1)
                topk_retrieve_indices = torch.cat([topk_retrieve_indices.gather(dim=-2, index=topk_from_old_indices.unsqueeze(-1).expand_as(topk_retrieve_indices)), new_input_ids_retrieve_indices], dim=-1)
                # retrieve_indices = torch.cat([torch.cat([retrieve_indices, pad_retrieve_indices.repeat(1,total_new_tokens-1,1)], dim=-1), topk_retrieve_indices], dim=-2)
                retrieve_indices = torch.cat([torch.cat([retrieve_indices, new_input_ids_retrieve_indices.repeat(1,total_new_tokens-1,1)], dim=-1), topk_retrieve_indices], dim=-2)
                # form path_probs
                topk_path_probs = torch.cat([topk_path_probs.gather(dim=-2, index=topk_from_old_indices.unsqueeze(-1).expand_as(topk_path_probs)), topk_leaf_probs.unsqueeze(-1)], dim=-1)
                # for pruned path, the prob is set to 0.0, to reduce work when do sample verification
                # if you wanna do reranking, you will need a prob retrieve
                path_probs = torch.cat([torch.cat([path_probs, path_probs.new_zeros(batch_size,path_probs.shape[1],1)], dim=-1), topk_path_probs], dim=-2)
            
            input_ids = torch.cat([input_ids, next_tokens], dim=-1)
            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    # scores will always be returned.
                    if next_token_probs is not None and next_token_probs.dim() == 2:
                        # it is the 1st generation
                        next_token_probs = next_token_probs.unsqueeze(1) # [bsz, 1, logits_dim]
                    scores += (next_token_probs,)
                if output_logits:
                    raise NotImplementedError

            # update generated ids, model inputs, and length for next step
            # encounter_eos = (torch.isin(stopping_criteria[1].eos_token_id, next_tokens).max() > 0)
            # finished = encounter_eos | (total_new_tokens >= max_depth)
            finished = (total_new_tokens >= max_depth)

            if (
                generation_config.assistant_confidence_threshold is not None
                and topk_leaf_probs.max() < generation_config.assistant_confidence_threshold
            ):
                finished = True
                # print(topk_path_probs)
                # print(total_new_tokens)
                
            
            model_kwargs["topk_retrieve_indices"] = topk_retrieve_indices # push back
            # num_new_token is set to 1    
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                generation_config=generation_config,
                model_kwargs=model_kwargs,
                model_inputs=model_inputs,
                num_new_tokens=1, # for attention_mask_2d
            )
            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            # all_end = record_time_sync()
            # rank0_print(f"rank {get_global_rank()} handle args forward: {all_end - forward_end}")
            del outputs

        # rank0_print(input_ids[:,start_len:])
        # rank0_print(retrieve_indices)
        return GenerateDecoderOnlyTreeOutput(
            tree_draft_tokens=input_ids,
            retrieve_indices=retrieve_indices,
            scores=scores,
            past_key_values=model_kwargs.get("past_key_values"),
        )       
        
    def _get_stopping_criteria(
        self,
        generation_config: GenerationConfig,
        stopping_criteria: Optional[StoppingCriteriaList],
        tokenizer: Optional["PreTrainedTokenizerBase"] = None,
        **kwargs,
    ) -> StoppingCriteriaList:
        criteria = StoppingCriteriaList()
        if generation_config.max_length is not None:
            max_position_embeddings = getattr(self.config, "max_position_embeddings", None)
            criteria.append(
                MaxLengthCriteria(
                    max_length=generation_config.max_length,
                    max_position_embeddings=max_position_embeddings,
                )
            )
        if generation_config.max_time is not None:
            criteria.append(MaxTimeCriteria(max_time=generation_config.max_time))
        if generation_config.stop_strings is not None:
            if tokenizer is None:
                raise ValueError(
                    "There are one or more stop strings, either in the arguments to `generate` or in the "
                    "model's generation config, but we could not locate a tokenizer. When generating with "
                    "stop strings, you must pass the model's tokenizer to the `tokenizer` argument of `generate`."
                )
            pass
            # criteria.append(StopStringCriteria(stop_strings=generation_config.stop_strings, tokenizer=tokenizer))
        if generation_config._eos_token_tensor is not None:
            criteria.append(EosTokenCriteria(eos_token_id=generation_config._eos_token_tensor))
        if (
            generation_config.is_assistant
            and generation_config.assistant_confidence_threshold is not None
            and generation_config.assistant_confidence_threshold > 0
            and kwargs["enable_tree_attention"] is False
        ):
            pass
            # print(f"condifence threshold is set: {generation_config.assistant_confidence_threshold}")
            # criteria.append(
            #     ConfidenceCriteria(assistant_confidence_threshold=generation_config.assistant_confidence_threshold)
            # )
        criteria = self._merge_criteria_processor_list(criteria, stopping_criteria)
        return criteria
    
    def _get_logits_processor(
        self,
        generation_config: GenerationConfig,
        input_ids_seq_length: int,
        encoder_input_ids: torch.LongTensor,
        prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], List[int]],
        logits_processor: Optional[LogitsProcessorList],
        device: str = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
    ) -> LogitsProcessorList:
        
        processors = LogitsProcessorList() if logits_processor is None else logits_processor
        # only supports temperature
        # TODO: top_p, top_k
        if generation_config.do_sample:
            if generation_config.num_beams > 1:
                raise ValueError(f"no support for beam search.")
            min_tokens_to_keep = 1
            if generation_config.temperature is not None and generation_config.temperature != 1.0:
                processors.append(TemperatureLogitsWarper(generation_config.temperature))
            # if generation_config.top_k is not None and generation_config.top_k != 0:
            #     processors.append(
            #         TopKLogitsWarper(top_k=generation_config.top_k, min_tokens_to_keep=min_tokens_to_keep)
            #     )
            # if generation_config.top_p is not None and generation_config.top_p < 1.0:
            #     processors.append(
            #         TopPLogitsWarper(top_p=generation_config.top_p, min_tokens_to_keep=min_tokens_to_keep)
            #     )
        return processors
                
    def _update_model_kwargs_for_generation(
        self,
        outputs,
        generation_config,
        model_kwargs,
        model_inputs,
        num_new_tokens: int = 1,
    ):
        raise NotImplementedError
        # update past_key_values keeping its naming used in model code
        cache_name, cache = self._extract_past_from_model_output(outputs)
        model_kwargs[cache_name] = cache
        if getattr(outputs, "state", None) is not None:
            model_kwargs["state"] = outputs.state

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

        # update attention mask
        if "attention_mask" in model_kwargs:
            attention_mask: torch.Tensor = model_kwargs["attention_mask"]
            if attention_mask.dim() != 2:
                raise ValueError("Custom attention mask is used, but model._update_model_kwargs_for_generation is still from BaseGenerationMixin.")
            model_kwargs["attention_mask"] = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )

        if model_kwargs.get("use_cache", True):
            model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + num_new_tokens
        else:
            past_positions = model_kwargs.pop("cache_position")
            new_positions = torch.arange(
                past_positions[-1] + 1, past_positions[-1] + num_new_tokens + 1, dtype=past_positions.dtype
            ).to(past_positions.device)
            model_kwargs["cache_position"] = torch.cat((past_positions, new_positions))
        return model_kwargs
        
    def _get_initial_cache_position(self, input_ids, model_kwargs):
        """Calculates `cache_position` for the pre-fill stage based on `input_ids` and optionally past length"""
        # `torch.compile`-friendly `torch.arange` from a shape -- the lines below are equivalent to `torch.arange`
        if "inputs_embeds" in model_kwargs:
            raise NotImplementedError
            cache_position = torch.ones_like(model_kwargs["inputs_embeds"][0, :, 0], dtype=torch.int64).cumsum(0) - 1
        else:
            cache_position = torch.ones_like(input_ids[0, :], dtype=torch.int64).cumsum(0) - 1

        past_length = 0
        if model_kwargs.get("past_key_values") is not None:
            cache = model_kwargs["past_key_values"]
            past_length = cache.get_seq_length()
            cache_position = cache_position[past_length:]

        model_kwargs["cache_position"] = cache_position
        return model_kwargs

    def _prepare_model_inputs(self, model_kwargs, generation_config):
        input_name = self.main_input_name
        input_ids = model_kwargs.pop("input_ids")
        return input_ids, input_name, model_kwargs

    def _validate_model_class(self):
        """
        Confirms that the model class is compatible with generation. If not, raises an exception that points to the
        right class to use.
        """
        return True

    def _validate_assistant(self, assistant_model):
        if assistant_model is None:
            return
        # if not self.config.vocab_size == assistant_model.config.vocab_size:
        #     raise ValueError("Make sure the main and assistant model use the same tokenizer")
        # if not self.config.get_text_config().vocab_size == assistant_model.config.get_text_config().vocab_size:
        #     raise ValueError("Make sure the main and assistant model use the same tokenizer")

    def _validate_model_kwargs(self, model_kwargs: Dict[str, Any]):
        """Validates model kwargs for generation. Generate argument typos will also be caught here."""
        # If a `Cache` instance is passed, checks whether the model is compatible with it
        if isinstance(model_kwargs.get("past_key_values", None), Cache) and not self._supports_cache_class:
            raise ValueError(
                f"{self.__class__.__name__} does not support an instance of `Cache` as `past_key_values`. Please "
                "check the model documentation for supported cache formats."
            )

        # Excludes arguments that are handled before calling any model function
        if self.config.is_encoder_decoder:
            for key in ["decoder_input_ids"]:
                model_kwargs.pop(key, None)

        unused_model_args = []
        model_args = set(inspect.signature(self.prepare_inputs_for_generation).parameters)
        # `kwargs`/`model_kwargs` is often used to handle optional forward pass inputs like `attention_mask`. If
        # `prepare_inputs_for_generation` doesn't accept them, then a stricter check can be made ;)
        if "kwargs" in model_args or "model_kwargs" in model_args:
            model_args |= set(inspect.signature(self.forward).parameters)

        # Encoder-Decoder models may also need Encoder arguments from `model_kwargs`
        if self.config.is_encoder_decoder:
            base_model = getattr(self, self.base_model_prefix, None)

            # allow encoder kwargs
            encoder = getattr(self, "encoder", None)
            # `MusicgenForConditionalGeneration` has `text_encoder` and `audio_encoder`.
            # Also, it has `base_model_prefix = "encoder_decoder"` but there is no `self.encoder_decoder`
            # TODO: A better way to handle this.
            if encoder is None and base_model is not None:
                encoder = getattr(base_model, "encoder", None)

            if encoder is not None:
                encoder_model_args = set(inspect.signature(encoder.forward).parameters)
                model_args |= encoder_model_args

            # allow decoder kwargs
            decoder = getattr(self, "decoder", None)
            if decoder is None and base_model is not None:
                decoder = getattr(base_model, "decoder", None)

            if decoder is not None:
                decoder_model_args = set(inspect.signature(decoder.forward).parameters)
                model_args |= {f"decoder_{x}" for x in decoder_model_args}

            # allow assistant_encoder_outputs to be passed if we're doing assisted generating
            if "assistant_encoder_outputs" in model_kwargs:
                model_args |= {"assistant_encoder_outputs"}

        for key, value in model_kwargs.items():
            if value is not None and key not in model_args:
                unused_model_args.append(key)

        if unused_model_args:
            raise ValueError(
                f"The following `model_kwargs` are not used by the model: {unused_model_args} (note: typos in the"
                " generate arguments will also show up in this list)"
            )
    
