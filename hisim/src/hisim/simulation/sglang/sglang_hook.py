import torch
import time
from dataclasses import asdict
from collections import defaultdict
import json
import os
import heapq
import importlib
import threading
from queue import Empty

from hisim.utils import get_logger
from hisim.spec import ModelInfo, AcceleratorInfo
from hisim.hook import BaseHook
from hisim.simulation.types import (
    MockSimulationMode,
    RequestStats,
)
from hisim.hook.utils import get_obj_from_args
from hisim.utils.json import CustomJsonEncoder
from hisim.simulation.manager import StateManager, ConfigManager, Envs
from hisim.time_predictor import (
    InferTimePredictor,
    FakeRequest,
    ScheduleBatch as HisimScheduleBatch,
)
from hisim.simulation.sglang.sglang_mock_class import (
    MockReqToTokenPool,
    MockTokenToKVPool,
    MockTokenToKVPoolAllocator,
    MockPagedTokenToKVPoolAllocator,
    MockTokenToKVPoolHost,
    MockHiCacheStorage,
)
from hisim.simulation.utils import (
    calc_metrics,
    estimate_kv_cache_pool_capacity,
)
from hisim.simulation.sglang.version import VersionDispatcher


logger = get_logger("hisim")

# Initialize call counters for diagnostics - must be declared at module level
PROCESS_BATCH_CALLS = 0
GET_NEW_BATCH_PREFILL_CALLS = 0
RUN_BATCH_CALLS = 0


class C_EngineHook(BaseHook):
    HOOK_CLASS_NAME = "Engine"
    HOOK_MODULE_NAME = "sglang.srt.entrypoints.engine"

    @classmethod
    def hook(cls, target):
        def hook_clear_hicache_storage(self):
            return self.loop.run_until_complete(
                self.tokenizer_manager.clear_hicache_storage()
            )

        target.clear_hicache_storage = hook_clear_hicache_storage
        return target


class C_TokenizerManagerHook(BaseHook):
    HOOK_CLASS_NAME = "TokenizerManager"
    HOOK_MODULE_NAME = "sglang.srt.managers.tokenizer_manager"

    @classmethod
    def hook(cls, target):
        original_send_one_request = target._send_one_request

        # Check the signature of the original _send_one_request to handle SGLang version differences
        import inspect
        sig = inspect.signature(original_send_one_request)
        params = list(sig.parameters.keys())

        if len(params) == 2:  # SGLang 0.5.13+: (self, tokenized_obj)
            def wrapped_send_one_request(self, tokenized_obj):
                # Populate REQUEST_STATS with request information for HTTP requests
                # HTTP requests bypass recv_requests, so we need to populate REQUEST_STATS here
                if hasattr(tokenized_obj, 'rid') and hasattr(tokenized_obj, 'input_ids'):
                    req_stats = C_SchedulerHook.REQUEST_STATS[tokenized_obj.rid]
                    req_stats.rid = tokenized_obj.rid
                    req_stats.input_length = len(tokenized_obj.input_ids)
                    if hasattr(tokenized_obj.sampling_params, 'max_new_tokens'):
                        req_stats.output_length = tokenized_obj.sampling_params.max_new_tokens

                    # Set server_created_time and queue_start for tracking
                    now = time.time()
                    if tokenized_obj.sampling_params.custom_params is not None and "simulation" in tokenized_obj.sampling_params.custom_params:
                        simulation_args = tokenized_obj.sampling_params.custom_params.get("simulation", {})

                        # Set server_created_time
                        tokenized_obj.sampling_params.custom_params["simulation"]["server_created_time"] = now
                        req_stats.created_time = simulation_args.get("created_time", now)

                        # Fix: Use global clock for last_event_time in OFFLINE mode to avoid timing mismatch
                        sim_mode = getattr(C_SchedulerHook, 'SIM_MODE', None)
                        if sim_mode == MockSimulationMode.OFFLINE:
                            req_stats.absolute_created_time = req_stats.created_time
                            try:
                                req_stats.last_event_time = StateManager.get_global_clock()
                            except:
                                req_stats.last_event_time = 0  # StateManager may not be initialized yet
                        else:
                            req_stats.last_event_time = req_stats.created_time

                        # Set queue_start (HTTP requests start queueing when they are sent)
                        req_stats.queue_start = now

                        # Handle queue_start from simulation args if provided
                        queue_start = simulation_args.get("queue_start")
                        if queue_start is not None:
                            req_stats.queue_start = queue_start
                            # Update global clock if queue_start is provided
                            if sim_mode == MockSimulationMode.OFFLINE:
                                try:
                                    StateManager.set_global_clock(queue_start)
                                except:
                                    pass  # StateManager may not be initialized yet

                        # Session-aware: extract session info for BLOCKING mode
                        session_id = simulation_args.get("session_id")
                        if session_id is not None:
                            req_stats.session_id = session_id
                            req_stats.parent_session_id = simulation_args.get("parent_session_id")
                            req_stats.session_end = simulation_args.get("session_end", False)
                    else:
                        # Fallback for simulation tracking
                        req_stats.created_time = now
                        req_stats.last_event_time = now
                        req_stats.queue_start = now
                else:
                    # Fallback for backward compatibility
                    if tokenized_obj.sampling_params.custom_params is not None and "simulation" in tokenized_obj.sampling_params.custom_params:
                        tokenized_obj.sampling_params.custom_params["simulation"]["server_created_time"] = time.time()

                return original_send_one_request(self, tokenized_obj)
        else:  # SGLang 0.5.9 and earlier: (self, obj, tokenized_obj, created_time)
            def wrapped_send_one_request(self, obj, tokenized_obj, created_time):
                # Populate REQUEST_STATS with request information for HTTP requests
                # HTTP requests bypass recv_requests, so we need to populate REQUEST_STATS here
                if hasattr(tokenized_obj, 'rid') and hasattr(tokenized_obj, 'input_ids'):
                    req_stats = C_SchedulerHook.REQUEST_STATS[tokenized_obj.rid]
                    req_stats.rid = tokenized_obj.rid
                    req_stats.input_length = len(tokenized_obj.input_ids)
                    if hasattr(tokenized_obj.sampling_params, 'max_new_tokens'):
                        req_stats.output_length = tokenized_obj.sampling_params.max_new_tokens

                    # Set created_time and other fields for tracking
                    if (
                        tokenized_obj.sampling_params.custom_params is not None
                        and "simulation" in tokenized_obj.sampling_params.custom_params
                    ):
                        simulation_args = tokenized_obj.sampling_params.custom_params.get("simulation", {})
                        req_stats.created_time = simulation_args.get("created_time", created_time)
                        req_stats.last_event_time = req_stats.created_time
                        req_stats.queue_start = created_time

                        # Handle queue_start from simulation args if provided
                        queue_start = simulation_args.get("queue_start")
                        if queue_start is not None:
                            req_stats.queue_start = queue_start

                        # Session-aware: extract session info for BLOCKING mode (0.5.9)
                        session_id = simulation_args.get("session_id")
                        if session_id is not None:
                            req_stats.session_id = session_id
                            req_stats.parent_session_id = simulation_args.get("parent_session_id")
                            req_stats.session_end = simulation_args.get("session_end", False)
                    else:
                        # Fallback for basic tracking
                        req_stats.created_time = created_time
                        req_stats.last_event_time = created_time
                        req_stats.queue_start = created_time
                return original_send_one_request(self, obj, tokenized_obj, created_time)

        target._send_one_request = wrapped_send_one_request


class C_ModelRunnerHook(BaseHook):
    HOOK_CLASS_NAME = "ModelRunner"
    HOOK_MODULE_NAME = "sglang.srt.model_executor.model_runner"

    @classmethod
    def hook(cls, target):
        # SGLang 0.5.13+: Add dummy canary_manager attribute
        # This allows the scheduler to access canary_manager without error

        class DummyCanaryManager:
            """Dummy canary_manager for SGLang 0.5.13+ compatibility"""
            def __init__(self):
                pass
            def attach_radix_cache(self, tree_cache):
                pass
            def __getattr__(self, name):
                # Make it tolerant to any attribute access
                return None

        # Add canary_manager as an instance attribute
        def __init__wrapper(original_init):
            def __wrapped_init__(self, *args, **kwargs):
                # Call original __init__
                original_init(self, *args, **kwargs)
                # Ensure canary_manager exists on this instance
                if not hasattr(self, 'canary_manager'):
                    self.canary_manager = DummyCanaryManager()
                    logger.debug("Added dummy canary_manager to ModelRunner instance")
            return __wrapped_init__

        # Wrap __init__ method if it exists and can be wrapped
        if hasattr(target, '__init__') and not hasattr(target.__init__, '_has_canary_fix'):
            logger.info("Adding dummy canary_manager attribute support for SGLang 0.5.13+ compatibility")
            target.__init__ = __init__wrapper(target.__init__)
            target.__init__._has_canary_fix = True

        _version_dispatcher = VersionDispatcher()

        def override_initialize(self, *args, **kwargs):
            # Force CPU device for simulation — HiSim is CPU-based.
            # On GPU machines, server_args.device auto-detects to "cuda", causing
            # all mock pools (MockReqToTokenPool, MockTokenToKVPool, allocator)
            # to allocate on GPU → CUDA OOM at kv_indices.to(copy=True).
            # Forcing CPU here holistically ensures batch.device, prefix_lens
            # tensors, wrapped_sample allocations, and mock pools all stay on CPU.
            self.device = "cpu"
            self.server_args.device = "cpu"
            # x86 CPU defaults attention_backend to "intel_amx", which makes
            # get_last_loc (common.py:119) take the triton dispatch path
            # (uses_triton_dispatch = backend not in ("ascend","torch_native")).
            # Force "torch_native" so both write_cache_indices and get_last_loc
            # use pure-PyTorch non-triton paths — CPU compatible.
            self.server_args.attention_backend = "torch_native"
            logger.info(
                f"Forced CPU simulation mode: device={self.device}, "
                f"attention_backend={self.server_args.attention_backend}"
            )

            # First ensure model and hardware are registered in this subprocess
            # Because multiprocessing creates new processes without shared memory
            config_path = os.getenv("HISIM_CONFIG_PATH")
            logger.info(f"Subprocess override_initialize: HISIM_CONFIG_PATH={config_path}, exists={os.path.exists(config_path) if config_path else False}")
            if config_path and os.path.exists(config_path):
                with open(config_path) as f:
                    config = json.load(f)

                # Read configured context_length for model registration
                configured_context_len = config.get("scheduler", {}).get("context_length")

                predictor_config = config.get("predictor", {})
                logger.info(f"Subprocess override_initialize: predictor.name={predictor_config.get('name')}")
                if predictor_config.get("name") == "inference_predictor":
                    # Register model ALWAYS in subprocess to ensure availability
                    model_name = self.model_config.hf_config.__dict__.get("model_name", config.get("model", {}).get("name") if config else None)
                    logger.info(f"Subprocess: Attempting to register model {model_name}")
                    model_config_path = predictor_config.get("model_config")
                    task_test_root = predictor_config.get("task_test_root")
                    inference_predictor_root = predictor_config.get("inference_predictor_root")

                    if model_name and model_config_path and (task_test_root or inference_predictor_root):
                        if task_test_root and not os.path.isabs(model_config_path):
                            model_config_file = os.path.join(task_test_root, model_config_path)
                        else:
                            model_config_file = os.path.join(inference_predictor_root, model_config_path) if inference_predictor_root else model_config_path

                        if os.path.exists(model_config_file):
                            model_data = json.load(open(model_config_file))
                            logger.info(f"Subprocess: registering model {model_name} from {model_config_file}")

                            # Use configured context_length if available, otherwise use model config
                            model_context_len = model_data.get("original_seq_len", model_data.get("max_pos_len", 32768))
                            if configured_context_len:
                                model_context_len = max(model_context_len, configured_context_len)
                                logger.info(f"Subprocess: Using configured context_length={configured_context_len} instead of model config {model_context_len}")

                            ModelInfo.from_dict({
                                'name': model_name,
                                    'model_type': 'gpt_oss',
                                    'hidden_size': model_data.get("dim", model_data.get("hidden_size", 5120)),
                                    'num_attention_heads': model_data.get("n_heads", model_data.get("num_attention_heads", 32)),
                                    'num_hidden_layers': model_data.get("n_layers", model_data.get("num_hidden_layers", 32)),
                                    'vocab_size': model_data.get("vocab_size", 32000),
                                    'intermediate_size': model_data.get("moe_inter_dim", model_data.get("ffn_hidden_size", model_data.get("dim", 5120) * 4)),
                                    'num_key_value_heads': model_data.get("n_kv_heads", model_data.get("num_attention_heads", 32)),
                                    'max_position_embeddings': model_context_len,
                                    'max_seq_len': model_context_len,  # Also set max_seq_len
                                    'torch_dtype': 'float16',
                                    'layer_types': [],
                                }, save_to_registry=True)

                    # Register hardware if not already registered
                    hw_name = config.get("platform", {}).get("accelerator", {}).get("name")
                    logger.info(f"Subprocess override_initialize: hw_name={hw_name}, already_registered={AcceleratorInfo.find_by_hw_name(hw_name) is not None}")
                    # ALWAYS try to register in subprocess to ensure availability
                    hw_config_path = predictor_config.get("hardware_config")
                    task_test_root = predictor_config.get("task_test_root")
                    inference_predictor_root = predictor_config.get("inference_predictor_root")

                    if hw_name and hw_config_path and (task_test_root or inference_predictor_root):
                        if task_test_root and not os.path.isabs(hw_config_path):
                            hw_config_file = os.path.join(task_test_root, hw_config_path)
                        else:
                            hw_config_file = os.path.join(inference_predictor_root, hw_config_path) if inference_predictor_root else hw_config_path

                        if os.path.exists(hw_config_file):
                            hw_data = json.load(open(hw_config_file))
                            logger.info(f"Subprocess: registering hardware {hw_name} from {hw_config_file}")
                            AcceleratorInfo.from_dict({
                                'name': hw_name,
                                'vendor': 'NVIDIA',
                                'hbm_capacity_gb': hw_data.get("mem_size", 64),
                                'hbm_bandwidth_gb': hw_data.get("mem_bw", 1600),
                                'intra_node_bandwidth_gb': hw_data.get("intra_bw", 1600),
                                'inter_node_bandwidth_gb': hw_data.get("inter_bw", 1600),
                                'device_alias': [hw_name],
                            }, save_to_registry=True, force_registry=True)

            class MockModel:
                def forward(self):
                    pass

            self.model = MockModel()

            self.dtype = self.model_config.dtype
            self.kv_cache_dtype = (
                self.dtype
            )  # FIXME: get kv cache dtype from server args

            # SGLang 0.5.13+: Add required attributes for compatibility
            self.use_ngram_embedding = getattr(self.model_config, 'use_ngram_embedding', False)

            model = ConfigManager.get_model_info(self.model_config.hf_config.__dict__)
            hw = ConfigManager.get_accelerator_info()
            config = ConfigManager.get_scheduler_config(
                self.server_args.__dict__,
                "sglang",
                self.model_config.hf_config.__dict__,
            )

            assert model is not None and hw is not None and config is not None

            if self.server_args.max_total_tokens is not None:
                self.max_total_num_tokens = self.server_args.max_total_tokens
            else:
                # Get the configured context_length for accurate capacity estimation
                effective_context_len = getattr(self.server_args, 'context_length', None)
                if effective_context_len is None:
                    effective_context_len = model.max_seq_len
                else:
                    logger.info(f"Using server_args.context_length={effective_context_len} for kv cache capacity estimation instead of model.max_seq_len={model.max_seq_len}")

                # Temporarily set model context for accurate capacity estimation
                original_max_seq_len = model.max_seq_len
                model.max_seq_len = effective_context_len

                self.max_total_num_tokens = estimate_kv_cache_pool_capacity(
                    model, hw, config
                )

                # Restore original max_seq_len
                model.max_seq_len = original_max_seq_len
                logger.info(f"Calculated max_total_num_tokens={self.max_total_num_tokens} based on context_length={effective_context_len}")

            # Fix: Validate max_total_num_tokens before page_size alignment
            if self.max_total_num_tokens <= 0:
                raise ValueError(
                    f"Calculated max_total_num_tokens is non-positive: {self.max_total_num_tokens}. "
                    f"Context length: {effective_context_len}. "
                    f"This may be caused by insufficient memory configuration or invalid chunked_prefill_size. "
                    f"Try adjusting mem_fraction_static or chunked_prefill_size parameters."
                )

            if hasattr(self, "page_size") and self.page_size > 1:
                # Fix: Validate page_size alignment to prevent negative results
                if self.page_size <= 0:
                    raise ValueError(
                        f"page_size must be positive, got page_size={self.page_size}"
                    )
                if self.max_total_num_tokens < self.page_size:
                    logger.warning(
                        f"max_total_num_tokens ({self.max_total_num_tokens}) is smaller than page_size ({self.page_size}). "
                        f"This may cause issues with chunked prefill. Setting max_total_num_tokens to page_size."
                    )
                    self.max_total_num_tokens = self.page_size

                aligned_capacity = self.max_total_num_tokens // self.page_size * self.page_size
                if aligned_capacity <= 0:
                    raise ValueError(
                        f"Page size alignment resulted in non-positive capacity: {aligned_capacity}. "
                        f"Original capacity: {self.max_total_num_tokens}, page_size: {self.page_size}. "
                        f"Try using a smaller page_size or increasing memory allocation."
                    )
                self.max_total_num_tokens = aligned_capacity
                logger.info(f"Aligned max_total_num_tokens to {self.max_total_num_tokens} for page_size={self.page_size}")

            # Use configured context_length if available, otherwise fall back to model.max_seq_len
            effective_context_len = getattr(self.server_args, 'context_length', None)
            if effective_context_len is None:
                effective_context_len = model.max_seq_len
            else:
                logger.info(f"Using server_args.context_length={effective_context_len} for max_num_reqs calculation instead of model.max_seq_len={model.max_seq_len}")

            max_num_reqs = min(
                max(
                    int(self.max_total_num_tokens / effective_context_len * 512),
                    2048,
                ),
                4096,
            )
            logger.info(
                f"Model runner initialized with {self.max_total_num_tokens} tokens. Maximum number of requests: {max_num_reqs}"
            )

            model_has_mtp_layers = (
                self.model_config.num_nextn_predict_layers is not None
            )
            model_num_layers = (
                self.model_config.num_nextn_predict_layers
                if self.is_draft_worker and model_has_mtp_layers
                else max(
                    self.model_config.num_hidden_layers,
                    self.model_config.num_attention_layers,
                )
            )
            self.start_layer = getattr(self.model, "start_layer", 0)
            self.end_layer = getattr(self.model, "end_layer", model_num_layers)
            self.num_effective_layers = self.end_layer - self.start_layer

            # Use configured context_length if available, otherwise fall back to model.max_seq_len
            effective_context_len = getattr(self.server_args, 'context_length', None)
            if effective_context_len is None:
                effective_context_len = model.max_seq_len
            else:
                logger.info(f"Using server_args.context_length={effective_context_len} instead of model.max_seq_len={model.max_seq_len}")

            self.req_to_token_pool = MockReqToTokenPool(
                size=max_num_reqs,
                max_context_len=effective_context_len,
                device=self.device,
                enable_memory_saver=False,
            )

            self.token_to_kv_pool = MockTokenToKVPool(
                self.max_total_num_tokens,
                page_size=self.page_size,
                dtype=self.kv_cache_dtype,
                head_num=self.model_config.get_num_kv_heads(
                    1  # get_attention_tp_size()
                ),
                head_dim=self.model_config.head_dim,
                layer_num=self.num_effective_layers,
                device=self.device,
                enable_memory_saver=self.server_args.enable_memory_saver,
                start_layer=self.start_layer,
                end_layer=self.end_layer,
            )

            if self.page_size == 1:
                self.token_to_kv_pool_allocator = MockTokenToKVPoolAllocator(
                    size=self.max_total_num_tokens,
                    page_size=1,
                    dtype=self.kv_cache_dtype,
                    device=self.device,
                    kvcache=self.token_to_kv_pool,
                    need_sort=False,
                )
            else:
                self.token_to_kv_pool_allocator = MockPagedTokenToKVPoolAllocator(
                    size=self.max_total_num_tokens,
                    page_size=self.page_size,
                    dtype=self.kv_cache_dtype,
                    device=self.device,
                    kvcache=self.token_to_kv_pool,
                    need_sort=False,
                )

            # self.init_memory_pool(50)
            self.attn_backend = None
            self.graph_mem_usage = 0
            self.weight_load_mem_usage = 10

            # Override forward_stream with a CPU stream.
            # ModelRunner.__init__ (model_runner.py:548) creates forward_stream
            # using torch.get_device_module(self.device) — but at __init__ time
            # self.device is still "cuda" (auto-detected, before override_initialize
            # forces it to "cpu"). The resulting CUDA stream is later used by
            # Scheduler.init_overlap (scheduler.py:1193) inside
            # `with self.forward_stream_ctx:` on every run_batch call. Mixing a
            # CUDA stream with a CPU device_module (forced by override_initialize)
            # causes "illegal memory access". Replace it with a CPU stream.
            self.forward_stream = torch.get_device_module("cpu").Stream()

            self.max_running_requests = min(
                (
                    self.max_total_num_tokens // 2
                    if self.server_args.max_running_requests is None
                    else self.server_args.max_running_requests
                    // (
                        self.server_args.dp_size
                        if self.server_args.enable_dp_attention
                        else 1
                    )
                ),
                self.req_to_token_pool.size,
            )

            return

        def wrapped_forward_v1(self, *args, **kwargs):
            batch = args[0]
            from sglang.srt.layers.logits_processor import LogitsProcessorOutput

            output = LogitsProcessorOutput(
                next_token_logits=torch.empty(
                    size=(batch.batch_size, self.model_config.vocab_size),
                    device=self.device,
                )
            )

            return output, False

        _version_dispatcher.register_method(
            "forward", ["0.5.6", "0.5.6.post1", "0.5.6.post2"], wrapped_forward_v1
        )

        def wrapped_forward_v2(self, *args, **kwargs):
            from sglang.srt.model_executor.model_runner import ModelRunnerOutput

            output, _ = wrapped_forward_v1(self, *args, **kwargs)
            return ModelRunnerOutput(
                logits_output=output,
                can_run_graph=False,
                expert_distribution_metrics=None,
            )

        _version_dispatcher.register_method(
            "forward", ["0.5.7", "0.5.8", "0.5.8.post1", "0.5.9"], wrapped_forward_v2
        )

        # SGLang 0.5.13+: forward method signature changed - accepts ForwardBatch directly
        def wrapped_forward_v3(self, *args, **kwargs):
            from sglang.srt.model_executor.model_runner import ModelRunnerOutput
            from sglang.srt.layers.logits_processor import LogitsProcessorOutput

            # Get the batch - in 0.5.13 it's passed as the first argument
            batch = args[0]

            # Create logits output with required dimensions
            output = LogitsProcessorOutput(
                next_token_logits=torch.empty(
                    size=(batch.batch_size, self.model_config.vocab_size),
                    device=self.device,
                )
            )

            # Return ModelRunnerOutput object
            return ModelRunnerOutput(
                logits_output=output,
                can_run_graph=False,
                expert_distribution_metrics=None,
                routed_experts_output=None,
                indexer_topk_output=None,
            )

        _version_dispatcher.register_method(
            "forward", ["0.5.13", "0.5.13.post1", "0.5.13.post2"], wrapped_forward_v3
        )

        def wrapped_sample(self, *args, **kwargs):
            logits = args[0]
            ids = torch.ones(
                size=(logits.next_token_logits.shape[0],),
                device=self.device,
                dtype=torch.int64,
            )
            return ids

        def wrapped_compute_logprobs_only(*args, **kwargs):
            return None

        # Patch module-level is_cuda()/is_hip() branch selections that run on
        # the forward-prep path (before wrapped_forward can bypass real CUDA).
        # SGLang picks CUDA implementations at import time based on
        # torch.cuda.is_available(), NOT server_args.device. On a GPU machine
        # running CPU simulation, these pick CUDA kernels and crash with
        # "illegal memory access" when fed CPU tensors.
        try:
            from sglang.srt.model_executor import forward_batch_info as _fbi
            # clamp_position: used by ForwardBatch.init_new for decode positions.
            # Module-level `if is_cuda() or is_hip(): clamp_position = clamp_position_cuda`
            # at fbi:1453. CUDA jit_kernel crashes on CPU seq_lens tensor.
            if hasattr(_fbi, "_clamp_position_native") and hasattr(_fbi, "clamp_position"):
                _fbi.clamp_position = _fbi._clamp_position_native
                logger.info(
                    "Patched forward_batch_info.clamp_position -> _clamp_position_native "
                    "for CPU simulation"
                )
        except Exception as _e:
            logger.warning(f"Failed to patch forward_batch_info.clamp_position: {_e}")

        target.initialize = override_initialize
        target.forward = _version_dispatcher.get_compat_method("forward")
        target.sample = wrapped_sample
        target.compute_logprobs_only = wrapped_compute_logprobs_only
        return target


class C_HiCacheController(BaseHook):
    HOOK_CLASS_NAME = "HiCacheController"
    HOOK_MODULE_NAME = "sglang.srt.managers.cache_controller"

    KV_CACHE_BYTES: int = None
    DISK_READ_BANDWIDTH_BYTES: float = None
    DISK_WRITE_BANDWIDTH_BYTES: float = None

    @staticmethod
    def calc_prefetch_pages(
        required_pages: int, page_size_byte: int, max_dur: float, bandwidth: float
    ) -> tuple[int, float]:
        _prefetch_dur = required_pages * page_size_byte / bandwidth
        if _prefetch_dur > max_dur:
            _completed_pages = int(max_dur * bandwidth / page_size_byte)
            return max(_completed_pages, 1), max_dur
        else:
            return required_pages, _prefetch_dur

    @classmethod
    def hook(cls, target):
        def override_backup_thread_func(self, *args, **kwargs):
            # Async thread: perform no action
            # The action will be performed by `handle_backup_operation`
            pass

        def override_prefetch_thread_func(self, *args, **kwargs):
            # Async thread: perform no action
            # The action will be performed by `handle_prefetch_operation`
            pass

        def handle_backup_operation(self):
            if not self.enable_storage:
                return
            while True:
                try:
                    operation = self.backup_queue.get(block=False)
                    if operation is None:
                        return

                    if not self.backup_skip:
                        self._page_backup(operation)
                    # TODO: Track the backup operation according to the global clock
                    self.ack_backup_queue.put(operation)

                except Empty:
                    return

        def handle_prefetch_operation(self, hiradix_cache=None):
            """Handle prefetch operations for simulation.

            This function is assigned as an instance method on HiCacheController.
            `self` is the HiCacheController instance. `hiradix_cache` is the
            HiRadixCache instance passed from wrapped_check_hicache_events, which
            owns prefetch_loaded_tokens_by_reqid.

            Args:
                hiradix_cache: The HiRadixCache instance that owns
                    prefetch_loaded_tokens_by_reqid. When provided, completed
                    tokens are written there so that
                    scheduler.pop_prefetch_loaded_tokens() returns the correct value.
                    Without this, storage_hit_length=0 causes all L3 tokens to
                    be double-counted in both L2 and L3.
            """
            if not self.enable_storage:
                return

            if C_HiCacheController.KV_CACHE_BYTES is None:
                C_HiCacheController.KV_CACHE_BYTES = ConfigManager.get_kv_cache_bytes()
            if C_HiCacheController.DISK_READ_BANDWIDTH_BYTES is None:
                C_HiCacheController.DISK_READ_BANDWIDTH_BYTES = (
                    ConfigManager.get_platform_config().disk_read_bandwidth
                )

            # TODO: Overlap schedule
            remain_dur = StateManager.get_current_inference_dur()

            chunked_prefetch_operation = getattr(
                self, "chunked_prefetch_operation", None
            )
            if chunked_prefetch_operation is not None:
                operation = chunked_prefetch_operation["operation"]
                storage_hit_count = chunked_prefetch_operation["storage_hit_count"]
                completed_tokens, prefetch_dur = (
                    C_HiCacheController.calc_prefetch_pages(
                        (storage_hit_count - operation.completed_tokens),
                        C_HiCacheController.KV_CACHE_BYTES,
                        remain_dur,
                        C_HiCacheController.DISK_READ_BANDWIDTH_BYTES,
                    )
                )
                if completed_tokens < storage_hit_count - operation.completed_tokens:
                    operation.completed_tokens += int(completed_tokens)
                    remain_dur = 0
                else:
                    operation.completed_tokens = int(storage_hit_count)
                    operation.mark_terminate()
                    remain_dur -= prefetch_dur
                    setattr(self, "chunked_prefetch_operation", None)
                    # Release host memory after current operation is finished
                    self.append_host_mem_release(
                        operation.host_indices[storage_hit_count:]
                    )
                # update request states
                req_stats = C_SchedulerHook.REQUEST_STATS[operation.request_id]
                req_stats.prefetch_complete_tokens = operation.completed_tokens
                # Write to prefetch_loaded_tokens_by_reqid so that
                # scheduler.pop_prefetch_loaded_tokens() can return the correct
                # value, which in turn allows schedule_batch.py to correctly
                # subtract storage_portion from host_portion in the cache
                # breakdown calculation. Without this, storage_hit_length=0
                # causes all L3 tokens to be double-counted in both L2 and L3.
                if hiradix_cache is not None:
                    hiradix_cache.prefetch_loaded_tokens_by_reqid[operation.request_id] = operation.completed_tokens

            while remain_dur > 0:
                try:
                    operation = self.prefetch_queue.get(block=False)
                    if operation is None:
                        return

                    hash_value, storage_hit_count = self._storage_hit_query(operation)
                    # not to prefetch if not enough benefits
                    if (
                        self.prefetch_threshold is not None
                        and storage_hit_count < self.prefetch_threshold
                    ):
                        operation.mark_terminate()
                        self.append_host_mem_release(operation.host_indices)
                        continue

                    operation.hash_value = hash_value[
                        : (storage_hit_count // self.page_size)
                    ]
                    storage_hit_count = (
                        storage_hit_count // self.page_size * self.page_size
                    )

                    completed_tokens, prefetch_dur = (
                        C_HiCacheController.calc_prefetch_pages(
                            storage_hit_count,
                            C_HiCacheController.KV_CACHE_BYTES,
                            remain_dur,
                            C_HiCacheController.DISK_READ_BANDWIDTH_BYTES,
                        )
                    )
                    if completed_tokens < storage_hit_count:
                        # Continue to prefetch data next time.
                        operation.completed_tokens = int(completed_tokens)
                        setattr(
                            self,
                            "chunked_prefetch_operation",
                            {
                                "operation": operation,
                                "storage_hit_count": storage_hit_count,
                            },
                        )
                        remain_dur = 0
                    else:
                        operation.completed_tokens = int(
                            storage_hit_count // self.page_size * self.page_size
                        )
                        # TODO: Track the prefetch operation according to the global clock
                        operation.mark_terminate()
                        remain_dur -= prefetch_dur
                    # update request states
                    req_stats = C_SchedulerHook.REQUEST_STATS[operation.request_id]
                    req_stats.prefetch_complete_tokens = operation.completed_tokens
                    # Write to prefetch_loaded_tokens_by_reqid so that
                    # scheduler.pop_prefetch_loaded_tokens() can return the correct
                    # value, which in turn allows schedule_batch.py to correctly
                    # subtract storage_portion from host_portion in the cache
                    # breakdown calculation. Without this, storage_hit_length=0
                    # causes all L3 tokens to be double-counted in both L2 and L3.
                    if hiradix_cache is not None:
                        hiradix_cache.prefetch_loaded_tokens_by_reqid[operation.request_id] = operation.completed_tokens
                    # Release host memory after current operation is finished
                    self.append_host_mem_release(
                        operation.host_indices[storage_hit_count:]
                    )

                except Empty:
                    return

        def override_generic_page_set(
            self, hash_values, host_indices, extra_info=None
        ) -> bool:
            # Always pass extra_info to storage_backend.
            data = [
                self.mem_pool_host.get_data_page(host_indices[i * self.page_size])
                for i in range(len(hash_values))
            ]
            return self.storage_backend.batch_set(hash_values, data, extra_info)

        def sim_start_writing(self) -> None:
            """Simulation-safe version of start_writing.

            The original start_writing creates CUDA events and records them on
            a CUDA stream (cache_controller.py:732-748). On a machine with
            GPUs but no active CUDA context in the sim process,
            `start_event.record()` raises `torch.AcceleratorError: CUDA error:
            an illegal memory access was encountered`.

            In simulation, `sim_writing_check` (on HiRadixCache) drains
            `ack_write_queue` and calls `_finish_write_through_ack` without
            touching CUDA events — the event fields of HiCacheAck are ignored.
            So we skip CUDA event/stream entirely and append a HiCacheAck with
            None events.

            The mock `backup_from_device_all_layer` (sglang_mock_class.py) is
            numpy-only (computes timing, no real DMA), so it is safe to call.
            `move_indices` with io_backend="direct" only does `.cpu()` and
            `.sort()` on CPU tensors — also safe.
            """
            from sglang.srt.managers.cache_controller import CacheOperation, HiCacheAck

            if len(self.write_queue) == 0:
                return

            op = CacheOperation.merge_ops(self.write_queue)
            host_indices, device_indices = self.move_indices(
                op.host_indices, op.device_indices
            )
            self.write_queue.clear()

            # Mock host pool: numpy-only timing computation, no real DMA.
            self.mem_pool_host.backup_from_device_all_layer(
                self.mem_pool_device, host_indices, device_indices, self.io_backend
            )
            if self.has_draft:
                self.mem_pool_host_draft.backup_from_device_all_layer(
                    self.mem_pool_device_draft,
                    host_indices,
                    device_indices,
                    self.io_backend,
                )

            # Append ack with None events — sim_writing_check ignores events.
            self.ack_write_queue.append(
                HiCacheAck(start_event=None, finish_event=None, node_ids=op.node_ids)
            )

        def sim_start_loading(self) -> int:
            """Simulation-safe version of start_loading.

            The original start_loading uses layer_done_counter (CUDA events)
            and a CUDA stream (cache_controller.py:802-844). Same CUDA error
            risk as start_writing.

            In simulation, `sim_loading_check` drains `ack_load_queue` and
            calls `dec_lock_ref` without touching CUDA events. `sim_is_load_back_event_done`
            always returns True. So we skip CUDA event/stream/layer_done_counter
            entirely and append a HiCacheAck with None events.

            The mock `load_to_device_per_layer` is numpy-only (timing only),
            so calling it per-layer is safe.
            """
            from sglang.srt.managers.cache_controller import CacheOperation, HiCacheAck

            if len(self.load_queue) == 0:
                return -1

            # Skip layer_done_counter.update_producer() — it queries CUDA events.
            op = CacheOperation.merge_ops(self.load_queue)
            host_indices, device_indices = self.move_indices(
                op.host_indices, op.device_indices
            )
            self.load_queue.clear()

            # Mock host pool: numpy-only timing computation per layer.
            for i in range(self.layer_num):
                self.mem_pool_host.load_to_device_per_layer(
                    self.mem_pool_device,
                    host_indices,
                    device_indices,
                    i,
                    self.io_backend,
                )
                if self.has_draft and i < self.mem_pool_host_draft.layer_num:
                    self.mem_pool_host_draft.load_to_device_per_layer(
                        self.mem_pool_device_draft,
                        host_indices,
                        device_indices,
                        i,
                        self.io_backend,
                    )

            # Append ack with None events — sim_loading_check ignores events.
            self.ack_load_queue.append(
                HiCacheAck(start_event=None, finish_event=None, node_ids=op.node_ids)
            )
            # Return a dummy producer_id; sim_is_load_back_event_done ignores it.
            return 0

        target.prefetch_thread_func = override_prefetch_thread_func
        target.backup_thread_func = override_backup_thread_func
        target.handle_backup_operation = handle_backup_operation
        target.handle_prefetch_operation = handle_prefetch_operation
        target._generic_page_set = override_generic_page_set
        # Override CUDA-event/stream-dependent methods with simulation-safe versions.
        # Original start_writing/start_loading crash on machines with GPUs but no
        # active CUDA context in the sim process (illegal memory access at
        # start_event.record()). The sim_writing_check/sim_loading_check on
        # HiRadixCache drain the ack queues without touching CUDA events, so
        # None events are safe.
        target.start_writing = sim_start_writing
        target.start_loading = sim_start_loading
        return target


class C_HiRadixCacheHook(BaseHook):
    HOOK_CLASS_NAME = "HiRadixCache"
    HOOK_MODULE_NAME = "sglang.srt.mem_cache.hiradix_cache"

    @classmethod
    def hook(cls, target):
        original_check_hicache_events = target.check_hicache_events
        original_reset = target.reset
        original_evict = target.evict
        original_evict_host = target.evict_host
        original_match_prefix_helper = target._match_prefix_helper
        original_write_backup_storage = target.write_backup_storage

        def wrapped_reset(self):
            if hasattr(self, "cache_controller"):
                # Drain pending write-through acks first so that
                # write_backup_storage populates backup_queue before
                # handle_backup_operation runs.
                self.writing_check()
                self.cache_controller.handle_backup_operation()
            original_reset(self)

        def wrapped_evict(self, params):
            """Override eviction to ensure non-backed-up nodes are written to Host before eviction.

            In the original HiRadixCache evict(), when a node is not backuped:
            - write_back policy: calls write_backup() to write to Host first
            - write_through policy: calls _evict_regular() directly, removing
              the node from the tree entirely -> NO host data -> NO L2 hits

            This is problematic for write_through in simulation: nodes are first
            inserted into the radix tree (cache_finished_req), but _inc_hit_count
            only calls write_backup AFTER a cache hit (hit_count >= threshold).
            If a node is evicted BEFORE being hit again, it has no host_value
            and gets removed from the tree entirely via _evict_regular.

            Fix: for ALL write policies (write_through and write_back), when
            evicting a non-backuped node, first try write_backup() to write
            its data to Host. Only if write_backup fails (Host pool full),
            fall back to _evict_regular() to force-remove from tree.
            """

            def _prune_evicted_leaves(cache, node):
                """Recursively prune evicted leaf descendants from the tree.

                When write_backup fails (host pool full), we cannot preserve
                host data for a node's children. But the children may already
                be evicted (value=None) and still in the tree (kept by
                _evict_backuped). We need to remove these evicted leaf
                descendants so their parent becomes a leaf and can be
                _evict_regular'd.

                This function recursively removes evicted leaf children,
                then checks if their parents become leaves too.
                """
                pruned = 0
                changed = True
                while changed:
                    changed = False
                    to_prune = []
                    for key, child in list(node.children.items()):
                        if child.evicted and child.lock_ref == 0 and len(child.children) == 0:
                            to_prune.append((key, child))
                    for key, child in to_prune:
                        node.children.pop(key)
                        if child in cache.evictable_leaves:
                            cache.evictable_leaves.remove(child)
                        if child in cache.evictable_host_leaves:
                            cache.evictable_host_leaves.remove(child)
                        pruned += 1
                        changed = True

                # After pruning immediate children, some deeper paths may
                # now have evicted leaf grandchildren. We need to prune
                # recursively for any remaining evicted non-leaf children.
                for key, child in list(node.children.items()):
                    if child.evicted and child.lock_ref == 0:
                        sub_pruned = _prune_evicted_leaves(cache, child)
                        pruned += sub_pruned
                        # If child became a leaf after pruning its own children
                        if len(child.children) == 0:
                            node.children.pop(key)
                            if child in cache.evictable_leaves:
                                cache.evictable_leaves.remove(child)
                            if child in cache.evictable_host_leaves:
                                cache.evictable_host_leaves.remove(child)
                            pruned += 1

                return pruned
            from sglang.srt.mem_cache.base_prefix_cache import EvictResult
            import time as _evict_time

            start_time = _evict_time.perf_counter()
            leaves = list(self.evictable_leaves)
            eviction_heap = [
                (self.eviction_strategy.get_priority(node), node)
                for node in leaves
            ]
            heapq.heapify(eviction_heap)

            num_evicted = 0
            write_back_nodes = []
            # Diagnostic counters
            _wb_success = 0     # write_backup succeeded (node stays in tree for L2)
            _wb_fail = 0        # write_backup failed (host pool full)
            _evict_backuped_count = 0  # _evict_backuped (already backed up, stays in tree)
            _evict_regular_count = 0   # _evict_regular (removed from tree, no L2 possible)

            while num_evicted < params.num_tokens and len(eviction_heap):
                _priority, x = heapq.heappop(eviction_heap)

                if x.lock_ref > 0:
                    continue

                if x.evicted:
                    # Node was already evicted (value=None) — skip it.
                    # This can happen when _cleanup_session_kv sets value=None
                    # without calling _update_leaf_status, leaving the node
                    # in evictable_leaves with a stale state.
                    continue

                if not x.backuped:
                    # Try to write to Host first, regardless of write_policy.
                    # In write_through mode, nodes that haven't been hit enough
                    # times may not have host_value yet. We write them now so
                    # they can be discovered as L2 hits later.
                    _host_avail = self.cache_controller.mem_pool_host.available_size() if hasattr(self.cache_controller, 'mem_pool_host') else -1
                    _node_value_len = len(x.value) if x.value is not None else 0
                    written = self.write_backup(x, write_back=True)
                    if written > 0:
                        num_evicted += written
                        write_back_nodes.append(x)
                        _wb_success += 1
                    else:
                        _wb_fail += 1
                        # write_backup failed (host pool full).
                        # Try to prune evicted descendants and then _evict_regular
                        # to free HBM tokens even without host backup.
                        _pruned = _prune_evicted_leaves(self, x)
                        if len(x.children) == 0:
                            # Now a leaf — can use _evict_regular
                            num_evicted += self._evict_regular(x)
                            _evict_regular_count += 1
                        else:
                            logger.debug(
                                f"[Evict] write_backup FAILED: node_id={x.id} "
                                f"value_len={_node_value_len} host_avail={_host_avail}"
                            )
                else:
                    num_evicted += self._evict_backuped(x)
                    _evict_backuped_count += 1

                for child in x.parent.children.values():
                    if child in write_back_nodes:
                        continue
                    if not child.evicted:
                        break
                else:
                    # all children are evicted or no children
                    new_priority = self.eviction_strategy.get_priority(x.parent)
                    heapq.heappush(eviction_heap, (new_priority, x.parent))

            # Complete any pending write-through operations
            if write_back_nodes:
                self.writing_check(write_back=True)
                for node in write_back_nodes:
                    assert node.backuped
                    self._evict_backuped(node)

            # If still not enough tokens freed, do a second pass using
            # _evict_regular to force-remove nodes from the tree.
            num_still_need = params.num_tokens - num_evicted
            if num_still_need > 0:
                leaves = list(self.evictable_leaves)
                if leaves:
                    eviction_heap = [
                        (self.eviction_strategy.get_priority(node), node)
                        for node in leaves
                    ]
                    heapq.heapify(eviction_heap)

                    num_force_evicted = 0
                    while num_force_evicted < num_still_need and len(eviction_heap):
                        _priority, x = heapq.heappop(eviction_heap)

                        if x.lock_ref > 0:
                            continue
                        if x.evicted:
                            continue
                        if x.backuped:
                            # Already backed up to Host, use _evict_backuped
                            # which keeps the node in the tree (host_value intact)
                            num_force_evicted += self._evict_backuped(x)
                            _evict_backuped_count += 1
                        elif len(x.children) == 0:
                            # Not backed up AND is a leaf node.
                            # write_backup failed — Host pool likely full.
                            # Force-remove from tree to free HBM tokens.
                            num_force_evicted += self._evict_regular(x)
                            _evict_regular_count += 1
                        else:
                            # Not backed up AND has children — cannot _evict_regular
                            # directly (assertion: len(node.children) == 0).
                            # This happens when children were evicted via
                            # _evict_backuped (stay in tree with value=None)
                            # but parent still has HBM value.
                            #
                            # Fix: prune evicted leaf descendants from the tree
                            # first. Since write_backup failed, the host pool
                            # is full — these nodes have no host data to
                            # preserve. Removing them makes the parent a leaf,
                            # allowing _evict_regular.
                            _pruned = _prune_evicted_leaves(self, x)
                            if len(x.children) == 0:
                                num_force_evicted += self._evict_regular(x)
                                _evict_regular_count += 1
                            # else: still has non-evicted or non-leaf children — skip

                        if len(x.parent.children) == 0 and x.parent.lock_ref == 0:
                            new_priority = self.eviction_strategy.get_priority(x.parent)
                            heapq.heappush(eviction_heap, (new_priority, x.parent))

                    num_evicted += num_force_evicted

            # Diagnostic: log eviction breakdown
            host_available = self.cache_controller.mem_pool_host.available_size() if hasattr(self.cache_controller, 'mem_pool_host') else -1
            logger.info(
                f"[Evict] need={params.num_tokens} freed={num_evicted} "
                f"wb_success={_wb_success} wb_fail={_wb_fail} "
                f"evict_backuped={_evict_backuped_count} "
                f"evict_regular={_evict_regular_count} "
                f"host_pool_avail={host_available}"
            )

            self.update_eviction_metrics(num_evicted, start_time)
            return EvictResult(num_tokens_evicted=num_evicted)

        def wrapped_match_prefix_helper(self, node, key):
            """Modified _match_prefix_helper that walks past evicted nodes.

            In the original HiRadixCache, _match_prefix_helper continues
            walking past evicted nodes (node = child even when child.evicted).
            This means the prefix match extends through eviction boundaries.

            The user's requirement: when HBM tokens are reallocated (eviction),
            the corresponding hash must be removed from the HBM radix tree.
            Future requests should NOT match past the eviction boundary when
            counting device_indices (L1 hits). However, the traversal should
            still continue past evicted nodes so that:
            - Deeper non-evicted nodes are correctly included in device_indices (L1)
            - The walk-up in match_prefix correctly computes host_hit_length (L2)
              from the full chain of evicted ancestors back to the root.

            Implementation: continue walking past evicted nodes, but only
            collect non-evicted values into device_indices. Unlike the original
            code (which also skips evicted values), this version handles
            soft-evicted nodes (host_value=None) by continuing the walk.
            """
            import time as _time
            node.last_access_time = _time.monotonic()
            child_key = key.child_key(self.page_size)
            value = []

            while len(key) > 0 and child_key in node.children.keys():
                child = node.children[child_key]
                child.last_access_time = _time.monotonic()
                prefix_len = child.key.match(key, page_size=self.page_size)
                if prefix_len < len(child.key):
                    new_node = self._split_node(child.key, child, prefix_len)
                    if not new_node.evicted:
                        value.append(new_node.value)
                    node = new_node
                    break
                else:
                    if not child.evicted:
                        value.append(child.value)
                    # Always continue walking, regardless of eviction state.
                    # Only skip collecting value for evicted nodes.
                    node = child
                    key = key[prefix_len:]
                    if len(key):
                        child_key = key.child_key(self.page_size)

            return value, node

        def wrapped_match_prefix(self, params):
            """Override match_prefix to handle soft-evicted nodes (host_value=None).

            After evict_host soft-evicts a node (frees host memory but keeps
            the node in the tree), the node has evicted=True but host_value=None.
            The original match_prefix walks up from last_node and does:
                while last_node.evicted:
                    host_hit_length += len(last_node.host_value)  # CRASH if None!
            We fix this by skipping nodes with host_value=None in the walk-up.

            Note: wrapped_match_prefix_helper now walks past evicted nodes
            (does NOT break at the eviction boundary). This means last_node
            is deeper in the tree, and the walk-up accumulates host_hit_length
            from the full chain of evicted ancestors, properly accounting for
            L2 (Memory) hits.
            """
            if self.disable:
                return self._empty_match_result

            key = params.key
            key, _ = key.maybe_to_bigram_view(self.is_eagle)
            key = key.page_aligned(self.page_size)
            if len(key) == 0:
                return self._empty_match_result

            value, last_node = self._match_prefix_helper(self.root_node, key)
            if value:
                value = torch.cat(value)
            else:
                value = self._empty_match_result.device_indices

            host_hit_length = 0
            last_host_node = last_node
            # Walk up from last_node, computing host_hit_length.
            # Skip nodes with host_value=None (soft-evicted from Host).
            while last_node.evicted:
                if last_node.host_value is not None:
                    host_hit_length += len(last_node.host_value)
                else:
                    # Soft-evicted: host data freed, skip in host_hit_length.
                    # Still continue walking up to find ancestor nodes with host data.
                    pass
                last_node = last_node.parent

            # Find last_host_node: walk up until we find a backuped node.
            # For soft-evicted nodes (host_value=None), backuped=False, so they're skipped.
            while not last_host_node.backuped and last_host_node != self.root_node:
                last_host_node = last_host_node.parent

            from sglang.srt.mem_cache.base_prefix_cache import MatchResult
            return MatchResult(
                device_indices=value,
                last_device_node=last_node,
                last_host_node=last_host_node,
                best_match_node=last_host_node,
                host_hit_length=host_hit_length,
            )

        def override_init(self, params, server_args):
            if server_args.hicache_io_backend == "direct":
                # FIXME: move this logic into server_args parsing
                if server_args.hicache_mem_layout == "page_first":
                    server_args.hicache_mem_layout = "page_first_direct"
                    logger.warning(
                        "Page first layout is not supported with direct IO backend, switching to page first direct layout"
                    )

            self.page_size = params.page_size

            # Validate page_size for chunked prefill compatibility
            default_page_size = 16  # Recommended page size for effective chunked prefill
            max_reasonable_page_size = 2048  # Maximum page size limit for simulation

            if self.page_size > max_reasonable_page_size:
                logger.warning(
                    f"Page size {self.page_size} may cause chunked prefill issues with non-integer sequence lengths. "
                    f"Consider using smaller page_size (recommended: {default_page_size}) for better chunked prefill. "
                    f"Large page_size can cause memory allocation failures for sequences ending in partial pages."
                )
            elif self.page_size == 1:
                logger.info("Using page_size=1 (no paging, simplest mode)")
            else:
                logger.info(f"Using page_size={self.page_size} (verified for chunked prefill)")

            self.kv_cache = params.token_to_kv_pool_allocator.get_kvcache()
            # Replace the host pool
            self.token_to_kv_pool_host = MockTokenToKVPoolHost(
                self.kv_cache,
                server_args.hicache_ratio,
                server_args.hicache_size,
                self.page_size,
                server_args.hicache_mem_layout,
                pin_memory=False,
                device="cpu",
            )

            self.tp_group = params.tp_cache_group
            self.attn_cp_group = params.attn_cp_cache_group
            self.attn_tp_group = params.attn_tp_cache_group
            self.pp_group = params.pp_cache_group
            self.tp_world_size = torch.distributed.get_world_size(group=self.tp_group)
            self.pp_rank = params.pp_rank
            self.pp_size = params.pp_size
            self.enable_storage = server_args.hicache_storage_backend is not None
            self.enable_storage_metrics = self.enable_storage and params.enable_metrics
            self.extra_metric_labels = server_args.extra_metric_labels

            # Parse storage backend extra config with default values
            extra_config = {}
            prefetch_threshold = 0.8
            prefetch_timeout_base = 60.0
            prefetch_timeout_per_ki_token = 0.01
            prefetch_timeout_max = 600.0
            hicache_storage_pass_prefix_keys = True

            if hasattr(server_args, 'hicache_storage_backend_extra_config'):
                backend_extra_config = server_args.hicache_storage_backend_extra_config
                if backend_extra_config:
                    try:
                        import json
                        extra_config = json.loads(backend_extra_config)
                    except:
                        extra_config = {}

                    if isinstance(extra_config, dict):
                        prefetch_threshold = extra_config.get('prefetch_threshold', prefetch_threshold)
                        prefetch_timeout_base = extra_config.get('prefetch_timeout_base', prefetch_timeout_base)
                        prefetch_timeout_per_ki_token = extra_config.get('prefetch_timeout_per_ki_token', prefetch_timeout_per_ki_token)
                        prefetch_timeout_max = extra_config.get('prefetch_timeout_max', prefetch_timeout_max)
                        hicache_storage_pass_prefix_keys = extra_config.get('hicache_storage_pass_prefix_keys', True)

            # Create PrefetchTimeoutConfig
            from sglang.srt.mem_cache.hiradix_cache import PrefetchTimeoutConfig
            prefetch_timeout_config = PrefetchTimeoutConfig(
                base=float(prefetch_timeout_base),
                per_ki_token=float(prefetch_timeout_per_ki_token),
                max=float(prefetch_timeout_max),
            )

            self.prefetch_threshold = prefetch_threshold
            self.prefetch_timeout_base = prefetch_timeout_base
            self.prefetch_timeout_per_page = (
                self.page_size / 1024 * prefetch_timeout_per_ki_token
            )
            self.prefetch_timeout_config = prefetch_timeout_config
            self.hicache_storage_pass_prefix_keys = hicache_storage_pass_prefix_keys
            # TODO: support more timeout check functions
            self.is_prefetch_timeout = self._prefetch_timeout_check_linear_func
            self.prefetch_stop_policy = server_args.hicache_storage_prefetch_policy

            HiCacheController = getattr(
                importlib.import_module("sglang.srt.managers.cache_controller"),
                "HiCacheController",
            )

            # Try to import StorageMetricsCollector from different possible locations
            StorageMetricsCollector = None
            try:
                StorageMetricsCollector = getattr(
                    importlib.import_module("sglang.srt.observability.metrics_collector"),
                    "StorageMetricsCollector",
                )
            except (ImportError, AttributeError):
                try:
                    StorageMetricsCollector = getattr(
                        importlib.import_module("sglang.srt.metrics.collector"),
                        "StorageMetricsCollector",
                    )
                except (ImportError, AttributeError):
                    logger.warning("StorageMetricsCollector not found, metrics collection will be disabled")

            self.load_cache_event = threading.Event()
            self.cache_controller = HiCacheController(
                params.token_to_kv_pool_allocator,
                self.token_to_kv_pool_host,
                self.page_size,
                self.tp_group,
                load_cache_event=self.load_cache_event,
                write_policy=server_args.hicache_write_policy,
                io_backend=server_args.hicache_io_backend,
                storage_backend=server_args.hicache_storage_backend,
                prefetch_threshold=self.prefetch_threshold,
                model_name=server_args.served_model_name,
                storage_backend_extra_config=extra_config,
            )
            if self.enable_storage_metrics and StorageMetricsCollector is not None:
                # TODO: support pp
                labels = {
                    "storage_backend": server_args.hicache_storage_backend,
                    "tp_rank": self.cache_controller.tp_rank,
                    "dp_rank": self.cache_controller.dp_rank,
                }
                self.storage_metrics_collector = StorageMetricsCollector(labels=labels)
            elif self.enable_storage_metrics and StorageMetricsCollector is None:
                logger.warning("StorageMetricsCollector not available, storage metrics collection is disabled")

            # Record the nodes with ongoing write-through
            self.ongoing_write_through = {}
            # Record the node segments with ongoing load-back
            self.ongoing_load_back = {}
            # Record the ongoing prefetch requests
            self.ongoing_prefetch = {}
            self.ongoing_backup = {}
            # Track per-request tokens loaded from storage (L3 hits)
            self.prefetch_loaded_tokens_by_reqid: dict[str, int] = {}
            # List of async work - needed for HiRadixCache compatibility
            self.work_list: list = []
            # TODO: Dynamically adjust the threshold
            self.write_through_threshold = (
                1 if server_args.hicache_write_policy == "write_through" else 2
            )
            self.load_back_threshold = 0  # Allow load_back for any size (simulation)
            # Version: 0.5.9
            self.evictable_host_leaves = set()
            # super().__init__(params=params)
            target.__mro__[1].__init__(self, params=params)

        def wrapped_evict_host(self, num_tokens: int):
            """Override evict_host to SOFT-evict: free host memory but keep nodes in tree.

            Original evict_host removes nodes from parent.children (parent.children.pop),
            which makes them undiscoverable by match_prefix — no L2 hits possible.

            Fix: Instead of removing the node from the tree, we only free the host
            memory (set host_value=None) and keep the node in the tree. This preserves
            the tree structure so that:
            1. match_prefix can still discover the node (it's still in parent.children)
            2. If the node's value is also None (evicted from HBM), our modified
               _match_prefix_helper will treat it as a "permanently evected" boundary
               — it continues walking to deeper non-evected children.
            3. Deeper non-evected nodes remain discoverable for L1 hits.

            Nodes with host_value=None after this "soft eviction" are considered
            "host-evected" — their L2 data is gone, but the tree path is preserved.
            """
            _host_leaves_before = len(self.evictable_host_leaves)
            _host_avail_before = self.cache_controller.mem_pool_host.available_size() if hasattr(self.cache_controller, 'mem_pool_host') else -1

            leaves = list(self.evictable_host_leaves)
            eviction_heap = [
                (self.eviction_strategy.get_priority(node), node) for node in leaves
            ]
            heapq.heapify(eviction_heap)

            num_freed = 0
            _soft_evicted = 0
            while num_freed < num_tokens and len(eviction_heap):
                _priority, x = heapq.heappop(eviction_heap)
                if x == self.root_node:
                    break
                if not x.evicted:
                    continue
                if x.host_ref_counter > 0:
                    continue
                if x.host_value is None:
                    # Already soft-evicted, no host data to free.
                    continue

                # Soft eviction: free host memory but KEEP node in the tree.
                # Original code: parent.children.pop(key) → removes from tree
                # New code: just set host_value=None → stays in tree
                num_freed += self.cache_controller.evict_host(x.host_value)
                x.host_value = None  # Clear host data (was backed up, now freed)

                # Remove from evictable_host_leaves since host_value is gone
                if x in self.evictable_host_leaves:
                    self.evictable_host_leaves.remove(x)
                self._update_host_leaf_status(x.parent)

                # Check if parent should become a host leaf
                # (parent is evicted and now all children have no host_value)
                if x.parent.evicted and x.parent not in self.evictable_host_leaves:
                    all_children_host_freed = True
                    for child in x.parent.children.values():
                        if child.host_value is not None:
                            all_children_host_freed = False
                            break
                    if all_children_host_freed and x.parent.host_value is not None:
                        self.evictable_host_leaves.add(x.parent)

                _soft_evicted += 1

            if _soft_evicted > 0:
                _host_avail_after = self.cache_controller.mem_pool_host.available_size() if hasattr(self.cache_controller, 'mem_pool_host') else -1
                logger.info(
                    f"[EvictHost] soft-evicted {_soft_evicted} nodes (kept in tree) "
                    f"(host_pool: {_host_avail_before} -> {_host_avail_after})"
                )

        def wrapped_load_back(self, node, mem_quota=None):
            """Override load_back to handle soft-evicted nodes (host_value=None).

            Original load_back walks up from the given node, asserting node.backuped
            for each evicted ancestor, then loads host data back to device.
            This crashes when encountering soft-evicted nodes where host_value=None
            (backuped=False), because:
            1. assert node.backuped fails
            2. torch.cat([n.host_value for n in nodes_to_load]) crashes for None
            3. len(node.host_value) crashes for None

            Fix: Skip soft-evicted nodes in the walk-up. Only collect nodes with
            actual host data (host_value is not None) into nodes_to_load.
            """
            from sglang.srt.mem_cache.base_prefix_cache import EvictParams
            from sglang.srt.disaggregation.kv_events import StorageMedium
            import time as _lb_time
            start_time = _lb_time.perf_counter()
            last_hit_node = node
            nodes_to_load = []

            # Walk up from the evicted node, collecting nodes that have host data.
            # Skip soft-evicted nodes (host_value=None) — their host data is gone.
            while node.evicted:
                if node.host_value is not None:
                    # Node has host data — include in load_back
                    nodes_to_load.insert(0, node)
                # else: soft-evicted (host_value=None), skip — no host data to load
                node = node.parent

            ancester_node = node  # First non-evicted ancestor

            if not nodes_to_load:
                return None

            # Protect the ancestor nodes from eviction
            result = self.inc_lock_ref(ancester_node)
            delta = result.delta

            # Re-validate nodes_to_load: between the walk-up and now,
            # eviction may have soft-evicted some nodes (host_value→None).
            # Also protect nodes from host eviction by incrementing host_ref_counter.
            validated_nodes = []
            for n in nodes_to_load:
                if n.host_value is not None:
                    validated_nodes.append(n)
                    # Protect from host eviction during load_back
                    n.host_ref_counter += 1

            if not validated_nodes:
                self.dec_lock_ref(ancester_node)
                return None

            # Load it all or not at all
            host_indices = torch.cat([n.host_value for n in validated_nodes])
            if len(host_indices) < self.load_back_threshold or (
                len(host_indices) > mem_quota + delta if mem_quota is not None else False
            ):
                for n in validated_nodes:
                    n.host_ref_counter -= 1
                self.dec_lock_ref(ancester_node)
                return None

            device_indices = self.cache_controller.load(
                host_indices=host_indices,
                node_id=last_hit_node.id,
                **self._get_extra_pools(),
            )
            if device_indices is None:
                self.evict(EvictParams(num_tokens=len(host_indices)))
                device_indices = self.cache_controller.load(
                    host_indices=host_indices,
                    node_id=last_hit_node.id,
                    **self._get_extra_pools(),
                )
            self.dec_lock_ref(ancester_node)
            if device_indices is None:
                logger.warning(
                    "load_back: FAILED to load %d tokens for node %d "
                    "even after eviction (evictable_size=%d)",
                    len(host_indices),
                    last_hit_node.id,
                    self.evictable_size_,
                )
                for n in validated_nodes:
                    n.host_ref_counter -= 1
                return None

            self.ongoing_load_back[last_hit_node.id] = last_hit_node
            offset = 0
            for n in validated_nodes:
                if n.host_value is None:
                    # Host data was freed during eviction triggered by load_back.
                    # Skip this node — we can't restore what isn't there.
                    # Still need to release host eviction protection.
                    n.host_ref_counter -= 1
                    continue
                hv_len = len(n.host_value)
                n.value = device_indices[offset : offset + hv_len].clone()
                offset += hv_len
                # Release host eviction protection
                n.host_ref_counter -= 1
                self._record_store_event(n, medium=StorageMedium.GPU)
            self.evictable_size_ += len(device_indices)
            self.inc_lock_ref(last_hit_node)

            if self.metrics_collector is not None:
                self.metrics_collector.observe_load_back_duration(
                    _lb_time.perf_counter() - start_time
                )
                self.metrics_collector.increment_load_back_num_tokens(len(device_indices))

            return device_indices

        def wrapped_init_load_back(self, params):
            """Override init_load_back to handle soft-evicted best_match_node.

            Original init_load_back: if best_match_node is evicted, calls load_back.
            If load_back fails, walks up until finding a non-evicted node.

            With soft eviction, best_match_node may be evicted AND have no host data.
            load_back will return None (nothing to load). We then need to find the
            nearest ancestor with actual host data, or return empty if none exists.
            """
            last_node = params.best_match_node
            mem_quota = params.mem_quota
            if last_node.evicted:
                loading_values = self.load_back(last_node, mem_quota)
                if loading_values is not None:
                    logger.debug(
                        f"loading back {len(loading_values)} tokens for node {last_node.id}"
                    )
                    return loading_values, last_node

                # load_back returned None — could be soft-evicted (no host data)
                # or load_back failed (not enough GPU memory). Walk up.
                while last_node.evicted:
                    last_node = last_node.parent

            return (
                self._empty_match_result.device_indices,
                last_node,
            )

        def wrapped_check_hicache_events(self, *args, **kwargs):
            # Order matters in simulation:
            # 1. sim_writing_check drains ack_write_queue and calls
            #    _finish_write_through_ack -> write_backup_storage -> populates
            #    backup_queue. Without this, backup_queue is empty and
            #    handle_backup_operation() has nothing to process.
            # 2. handle_backup_operation drains backup_queue and calls
            #    _page_backup -> batch_set, putting data into storage.
            # 3. handle_prefetch_operation drains prefetch_queue and queries
            #    storage (batch_exists) to compute storage hits.
            # 4. original_check_hicache_events runs the rest (storage control
            #    queues, async work reaping). It internally calls
            #    self.writing_check() / self.loading_check(), which are now
            #    overridden to sim_* versions (no-op if queues already drained).
            self.writing_check()
            self.cache_controller.handle_backup_operation()
            self.cache_controller.handle_prefetch_operation(hiradix_cache=self)
            return original_check_hicache_events(self, *args, **kwargs)

        def sim_writing_check(self, write_back=False):
            """Simulation-safe version of writing_check.

            The original writing_check uses finish_event.synchronize() and
            _all_reduce() (torch.distributed) to confirm DMA completion.
            In CPU simulation there is no real GPU DMA, so CUDA events never
            report completion (or behave as no-ops), which causes the write
            queue to stall — `ongoing_write_through` is never drained, and
            `lock_ref` is never decremented via `_finish_write_through_ack`.
            Locked nodes cannot be evicted, and their host data is never
            published to storage via `write_backup_storage`.

            This simulation version treats ALL pending DMA operations as
            immediately completed. It drains `ack_write_queue` and calls
            `_finish_write_through_ack` for each ack without touching CUDA
            events or distributed sync.
            """
            if write_back:
                # Drain all pending write-through acks (blocking mode).
                # In simulation, every write_backup() call appends one ack to
                # ack_write_queue via start_writing(), so ongoing_write_through
                # and ack_write_queue should drain together. The guard prevents
                # an infinite loop if the invariant ever breaks.
                safety_iters = 0
                while len(self.ongoing_write_through) > 0 and safety_iters < 100:
                    safety_iters += 1
                    if len(self.cache_controller.ack_write_queue) == 0:
                        break
                    for _, _finish_event, ack_list in self.cache_controller.ack_write_queue:
                        for ack_id in ack_list:
                            self._finish_write_through_ack(ack_id, release_lock=False)
                    self.cache_controller.ack_write_queue.clear()
                return

            if len(self.ongoing_write_through) == 0:
                return

            # Non-blocking: drain all completed acks (treat all as completed)
            finish_count = len(self.cache_controller.ack_write_queue)
            if finish_count > 0:
                logger.debug(f"[sim_writing_check] Process {finish_count} write back ops")
            while finish_count > 0 and len(self.cache_controller.ack_write_queue) > 0:
                _, _finish_event, ack_list = self.cache_controller.ack_write_queue.pop(0)
                for ack_id in ack_list:
                    self._finish_write_through_ack(ack_id, release_lock=True)
                finish_count -= 1

        def sim_loading_check(self):
            """Simulation-safe version of loading_check.

            The original loading_check uses finish_event.query() and _all_reduce()
            to confirm load-back DMA completion. In CPU simulation, CUDA events
            never report completion, so `ack_load_queue` is never drained and
            `ongoing_load_back` entries never release their lock_ref.

            This version treats all pending load operations as immediately
            completed and drains `ack_load_queue` without touching CUDA events
            or distributed sync.
            """
            if len(self.cache_controller.ack_load_queue) == 0:
                return

            finish_count = len(self.cache_controller.ack_load_queue)
            if finish_count > 0:
                logger.debug(f"[sim_loading_check] Process {finish_count} load ops")
            while finish_count > 0 and len(self.cache_controller.ack_load_queue) > 0:
                _, _finish_event, ack_list = self.cache_controller.ack_load_queue.pop(0)
                for ack_id in ack_list:
                    end_node = self.ongoing_load_back.pop(ack_id, None)
                    if end_node is not None:
                        self.dec_lock_ref(end_node)
                finish_count -= 1

        def sim_is_load_back_event_done(self, consumer_index: int) -> bool:
            """Simulation-safe version of is_load_back_event_done.

            Always returns True in simulation (DMA treated as instantly
            complete). Also drains pending load acks via the overridden
            self.loading_check() (which is sim_loading_check).
            """
            self.loading_check()
            return True

        def wrapped_write_backup_storage(self, node, backup_len=None):
            """Guard write_backup_storage against node.host_value being None.

            In simulation, _cleanup_session_kv may evict a node's host_value
            (set to None) while a write-through operation for that node is still
            in ongoing_write_through. When _finish_write_through_ack later calls
            write_backup_storage, node.host_value is None, which would cause
            TypeError in _page_backup (host_indices=None).

            Skip the storage backup if host_value is None — the data has already
            been cleaned up by session-end eviction.
            """
            if getattr(node, 'host_value', None) is None:
                logger.debug(
                    f"[write_backup_storage] Skipping: node {node.id} "
                    f"host_value is None (likely evicted by session-end cleanup)"
                )
                return
            original_write_backup_storage(self, node, backup_len)

        target.__init__ = override_init
        target.check_hicache_events = wrapped_check_hicache_events
        # Override CUDA-event-dependent methods with simulation-safe versions.
        # These are assigned at the class level so all HiRadixCache instances
        # (including those created in subprocesses) use the simulation versions.
        target.writing_check = sim_writing_check
        target.loading_check = sim_loading_check
        target.is_load_back_event_done = sim_is_load_back_event_done
        target.reset = wrapped_reset
        target.evict = wrapped_evict
        target.evict_host = wrapped_evict_host
        target.load_back = wrapped_load_back
        target.init_load_back = wrapped_init_load_back
        target.match_prefix = wrapped_match_prefix
        target._match_prefix_helper = wrapped_match_prefix_helper
        target.write_backup_storage = wrapped_write_backup_storage
        return target


class C_StorageBackendFactory(BaseHook):
    HOOK_CLASS_NAME = "StorageBackendFactory"
    HOOK_MODULE_NAME = "sglang.srt.mem_cache.storage.backend_factory"

    @classmethod
    def hook(cls, target):
        def override_create_backend(cls, *args, **kwargs):
            logger.info("Creating hijacked cache storage backend.")
            return MockHiCacheStorage()

        target.create_backend = override_create_backend


# ====== Session TTL Management ======
# Tracks per-session TTL deadlines for session-aware eviction protection.
# Key = session_id, Value = virtual clock deadline (seconds) when TTL expires.
SESSION_TTL_TABLE: dict[str, float] = {}

# Tracks which sessions "own" which radix tree nodes.
# Used for eviction protection: if a node belongs to a TTL-protected session,
# it should not be evicted.
# Key = TreeNode.id, Value = set of session_ids that own this node.
NODE_OWNERS: dict[int, set[str]] = {}

# Maps TreeNode.id → TreeNode reference for efficient node lookup during cleanup.
NODE_MAP: dict[int, object] = {}


def _update_session_ttl(session_id: str, cache_control: dict):
    """Update/refresh session TTL. Last TTL wins (overwrites previous)."""
    if session_id is None or cache_control is None:
        return
    if cache_control.get("type") != "ephemeral":
        return
    ttl_minutes = cache_control.get("ttl", 5)
    ttl_minutes = max(0, min(ttl_minutes, 60))  # clamp to [0, 60]
    ttl_deadline = StateManager.get_global_clock() + ttl_minutes * 60
    SESSION_TTL_TABLE[session_id] = ttl_deadline
    logger.debug(
        f"Session TTL updated: session_id={session_id}, "
        f"ttl={ttl_minutes}min, deadline={ttl_deadline:.2f}s"
    )


def _is_session_protected(session_id: str) -> bool:
    """Check if a session is TTL-protected (not expired)."""
    if session_id is None or session_id not in SESSION_TTL_TABLE:
        return False
    if StateManager.get_global_clock() < SESSION_TTL_TABLE[session_id]:
        return True
    # TTL expired, clean up entry
    del SESSION_TTL_TABLE[session_id]
    return False


def _cleanup_expired_sessions():
    """Remove all expired session TTL entries."""
    now = StateManager.get_global_clock()
    expired = [sid for sid, deadline in SESSION_TTL_TABLE.items() if now >= deadline]
    for sid in expired:
        del SESSION_TTL_TABLE[sid]
    if expired:
        logger.debug(f"Cleaned up {len(expired)} expired sessions.")


def _tag_node_with_session(node, session_id: str):
    """Associate a radix tree node with a session for eviction tracking."""
    if node is None or session_id is None:
        return
    node_id = node.id
    NODE_MAP[node_id] = node  # Store reference for cleanup
    if node_id not in NODE_OWNERS:
        NODE_OWNERS[node_id] = set()
    NODE_OWNERS[node_id].add(session_id)


def _is_node_protected(node) -> bool:
    """Check if a node belongs to any TTL-protected session."""
    if node is None:
        return False
    node_id = node.id
    owner_sessions = NODE_OWNERS.get(node_id)
    if owner_sessions is None:
        return False
    for sid in owner_sessions:
        if _is_session_protected(sid):
            return True
    return False


def _remove_node_ownership(node):
    """Remove ownership tracking when a node is deleted."""
    if node is None:
        return
    node_id = node.id
    NODE_OWNERS.pop(node_id, None)
    NODE_MAP.pop(node_id, None)


def _clear_node_ownership():
    """Clear all node ownership tracking."""
    NODE_OWNERS.clear()
    NODE_MAP.clear()


def _cleanup_session_kv(session_id: str, tree_cache):
    """Clean up KV cache for a session that has ended.

    Only removes nodes EXCLUSIVELY owned by this session.
    Shared nodes only have this session's ownership removed.
    """
    if session_id is None:
        return

    # DEBUG: Log evictable_size_ before cleanup
    _evict_before = tree_cache.evictable_size() if hasattr(tree_cache, 'evictable_size') else 0
    _avail_before = 0
    try:
        _avail_before = tree_cache.token_to_kv_pool_allocator.available_size()
    except Exception:
        pass

    # 1. Find exclusively-owned nodes and clean ownership
    exclusive_node_ids = []
    for node_id in list(NODE_OWNERS.keys()):
        owners = NODE_OWNERS.get(node_id)
        if owners is None or session_id not in owners:
            continue
        owners.discard(session_id)
        if len(owners) == 0:
            exclusive_node_ids.append(node_id)
            NODE_OWNERS.pop(node_id, None)
        # else: shared node, ownership already removed

    # 2. Sort exclusive nodes by depth (deepest first)
    #    This ensures children are evicted before parents,
    #    so parents become leaves and can be deleted.
    exclusive_nodes = []
    for nid in exclusive_node_ids:
        node = NODE_MAP.get(nid)
        if node is not None and node is not tree_cache.root_node:
            exclusive_nodes.append(node)

    def _node_depth(n):
        d = 0
        while n.parent is not None and n is not tree_cache.root_node:
            d += 1
            n = n.parent
        return d

    exclusive_nodes.sort(key=_node_depth, reverse=True)

    # 3. Evict exclusive nodes bottom-up
    is_hiradix = hasattr(tree_cache, 'cache_controller')
    total_freed = 0
    for node in exclusive_nodes:
        # Skip if node already deleted from tree (parent.children was cleared)
        if node.parent is None or node not in node.parent.children.values():
            continue

        # Check if node has pending write-through operations.
        # Nodes with write_through_pending_id or lock_ref > 0 or host_ref_counter > 0
        # have ongoing DMA/backup operations that require host_value to remain
        # valid. Freeing host_value while a write_backup_storage operation is
        # pending would cause TypeError in _page_backup (host_indices=None).
        has_pending_write = (
            getattr(node, 'write_through_pending_id', None) is not None
            or getattr(node, 'lock_ref', 0) > 0
            or getattr(node, 'host_ref_counter', 0) > 0
        )

        # Skip L2 (host_value) cleanup if node has pending write operations.
        # We can still free L1 (node.value) since the data has already been
        # copied to host by write_backup. But we must NOT set host_value=None
        # or call evict_host(), and we must NOT remove the node from the tree
        # (via _delete_leaf) since the write-through ack needs to access the node.
        if has_pending_write:
            # Node has pending references (lock_ref > 0, pending DMA, or
            # load_back in progress). Per the token/hash model:
            #   - allocator tokens are only referenced/dereferenced, never freed
            #   - if ref_count > 0, the hash must NOT be evicted
            # Skip this node entirely — don't free, don't touch L1/L2, don't
            # remove from tree. When references are released (lock_ref → 0,
            # DMA completes), the node becomes evictable and the regular
            # evictor will handle it through the normal eviction path.
            continue

        # Skip if node has non-evicted children (not a leaf)
        if any(not c.evicted for c in node.children.values()):
            # Still has active children — just evict this node's value
            # but don't remove from tree
            if node.value is not None:
                val_len = len(node.value)
                # Save reference before setting value=None (free needs the tensor)
                node_value = node.value
                tree_cache.token_to_kv_pool_allocator.free(node_value)
                node.value = None
                # Update evictable_size_ and leaf status to keep tree invariants.
                # Without this, evictable_leaves may contain this evicted node,
                # causing write_backup(node) to crash (device_indices=None).
                tree_cache.evictable_size_ -= val_len
                tree_cache._update_leaf_status(node)
                tree_cache._update_leaf_status(node.parent)
                total_freed += val_len
            # Handle L2 (host_value) for HiRadixCache
            if is_hiradix and getattr(node, 'host_value', None) is not None:
                tree_cache.cache_controller.evict_host(node.host_value)
                node.host_value = None
                # Remove from evictable_host_leaves BEFORE _update_host_leaf_status,
                # because _update_host_leaf_status may re-add the node (it checks
                # evicted + no backuped children, but doesn't check host_value).
                # With host_value=None, the L2 evictor would crash on evict_host(None).
                if hasattr(tree_cache, 'evictable_host_leaves') and node in tree_cache.evictable_host_leaves:
                    tree_cache.evictable_host_leaves.remove(node)
                tree_cache._update_host_leaf_status(node.parent)
                # _update_host_leaf_status may add parent to evictable_host_leaves
                # even if parent.host_value is None (soft-evicted). Remove it to
                # prevent the L2 evictor from crashing on evict_host(None).
                if hasattr(tree_cache, 'evictable_host_leaves') and node.parent in tree_cache.evictable_host_leaves and getattr(node.parent, 'host_value', None) is None:
                    tree_cache.evictable_host_leaves.remove(node.parent)
            continue

        # Leaf node (or all children are evicted) — can fully delete
        # For leaf nodes with no children, we can call _delete_leaf first
        # (which removes from tree and updates evictable_size_),
        # then free the KV pool slots.
        # For nodes with only evicted children, soft-evict by setting value=None.
        if len(node.children) == 0:
            # True leaf: remove from tree, then free memory
            # _delete_leaf expects node.value to be non-None (not evicted)
            # so we must call it BEFORE freeing node.value
            if node.value is not None:
                total_freed += len(node.value)
                node_value = node.value  # save reference for freeing after _delete_leaf
                node_host_value = node.host_value  # save reference for L2 cleanup
                tree_cache._delete_leaf(node)
                # _delete_leaf only removes from evictable_leaves (L1), not
                # evictable_host_leaves (L2). Remove from L2 set to prevent
                # the L2 evictor from picking up a deleted node.
                if is_hiradix and hasattr(tree_cache, 'evictable_host_leaves') and node in tree_cache.evictable_host_leaves:
                    tree_cache.evictable_host_leaves.remove(node)
                # Free L1 device memory. The available_size() cap in the mock
                # allocator prevents inflation beyond max_total_num_tokens.
                tree_cache.token_to_kv_pool_allocator.free(node_value)
                # Free L2 (host_value) for HiRadixCache
                if is_hiradix and node_host_value is not None:
                    tree_cache.cache_controller.evict_host(node_host_value)
                    tree_cache._update_host_leaf_status(node.parent)
            else:
                # Already evicted, just remove from tree
                # Can't use _delete_leaf since it expects non-evicted nodes,
                # so remove manually
                if node in tree_cache.evictable_leaves:
                    tree_cache.evictable_leaves.remove(node)
                if hasattr(tree_cache, 'evictable_host_leaves') and node in tree_cache.evictable_host_leaves:
                    tree_cache.evictable_host_leaves.remove(node)
                key = node.key.child_key(tree_cache.page_size)
                node.parent.children.pop(key, None)
                tree_cache._update_leaf_status(node.parent)
                if hasattr(tree_cache, '_update_host_leaf_status'):
                    tree_cache._update_host_leaf_status(node.parent)
                # Free L2 (host_value) for HiRadixCache — evicted nodes may still have host data
                if is_hiradix and getattr(node, 'host_value', None) is not None:
                    tree_cache.cache_controller.evict_host(node.host_value)
                    node.host_value = None
            _remove_node_ownership(node)
            NODE_MAP.pop(node.id, None)
        else:
            # Has evicted children — soft evict (set value=None)
            if node.value is not None:
                val_len = len(node.value)
                # Save reference before setting value=None (free needs the tensor)
                node_value = node.value
                tree_cache.token_to_kv_pool_allocator.free(node_value)
                node.value = None
                # Update evictable_size_ and leaf status to keep tree invariants.
                tree_cache.evictable_size_ -= val_len
                tree_cache._update_leaf_status(node)
                tree_cache._update_leaf_status(node.parent)
                total_freed += val_len
            # Handle L2 (host_value) for HiRadixCache
            if is_hiradix and getattr(node, 'host_value', None) is not None:
                tree_cache.cache_controller.evict_host(node.host_value)
                node.host_value = None
                # Remove from evictable_host_leaves BEFORE _update_host_leaf_status,
                # because _update_host_leaf_status may re-add the node (same issue
                # as non-evicted children path: it doesn't check host_value).
                if hasattr(tree_cache, 'evictable_host_leaves') and node in tree_cache.evictable_host_leaves:
                    tree_cache.evictable_host_leaves.remove(node)
                tree_cache._update_host_leaf_status(node.parent)
                # Same guard as Path 2: parent may be added with host_value=None
                if hasattr(tree_cache, 'evictable_host_leaves') and node.parent in tree_cache.evictable_host_leaves and getattr(node.parent, 'host_value', None) is None:
                    tree_cache.evictable_host_leaves.remove(node.parent)

    # 4. Remove session TTL
    SESSION_TTL_TABLE.pop(session_id, None)

    # 5. Clean up NODE_MAP for removed nodes
    for nid in exclusive_node_ids:
        NODE_MAP.pop(nid, None)

    if total_freed > 0:
        # DEBUG: Log evictable_size_ after cleanup
        _evict_after = tree_cache.evictable_size() if hasattr(tree_cache, 'evictable_size') else 0
        _avail_after = 0
        try:
            _avail_after = tree_cache.token_to_kv_pool_allocator.available_size()
        except Exception:
            pass
        logger.info(
            f"[SessionEnd] session={session_id} freed {total_freed} tokens "
            f"({len(exclusive_nodes)} exclusive nodes cleaned) "
            f"evictable: {_evict_before} -> {_evict_after} (delta={_evict_after - _evict_before}), "
            f"available: {_avail_before} -> {_avail_after}"
        )
    else:
        _evict_after = tree_cache.evictable_size() if hasattr(tree_cache, 'evictable_size') else 0
        logger.info(
            f"[SessionEnd] session={session_id} freed 0 tokens "
            f"evictable: {_evict_before} -> {_evict_after} (delta={_evict_after - _evict_before})"
        )

    # SAFETY: Clamp evictable_size_ to 0 if negative.
    # Across rounds, double-decrement can occur when session-end cleanup
    # and regular eviction both process the same node's tokens.
    # A negative evictable_size_ deflates rem_total_tokens = available + evictable,
    # preventing the scheduler from admitting new prefills → simulation hang.
    if hasattr(tree_cache, 'evictable_size_') and tree_cache.evictable_size_ < 0:
        logger.warning(
            f"[SessionEnd] evictable_size_ went negative ({tree_cache.evictable_size_}), "
            f"clamping to 0"
        )
        tree_cache.evictable_size_ = 0


class C_SchedulerHook(BaseHook):
    HOOK_CLASS_NAME = "Scheduler"
    HOOK_MODULE_NAME = "sglang.srt.managers.scheduler"

    INFERENCE_PREDICTOR: InferTimePredictor = None

    REQUEST_STATS: dict[str, RequestStats] = defaultdict(RequestStats)
    ITERATION_STATS: list[dict] = []
    LAST_CPU_TS: float = 0
    LAST_FLUSH_TS: float = 0
    HISIM_BATCH: HisimScheduleBatch = None

    OVERLAP_SCHEDULE: bool = False

    SIM_MODE = MockSimulationMode(Envs.simulation_mode())
    OFFLINE_RECV_ALL_REQUEST: bool = False
    FUTURE_QUEUE: list[
        tuple[float, int, RequestStats]
    ] = []  # tuple(created time, salt, request)

    SCHEDULE_REQ_STATS = []

    @staticmethod
    def _write_cache_stats_to_file(stats_dict: dict):
        """将cache统计信息写入临时文件（用于跨进程共享）"""
        try:
            import os
            cache_stats_file = os.path.join(Envs.output_dir(), "cache_stats.json")
            os.makedirs(os.path.dirname(cache_stats_file), exist_ok=True)
            import json
            with open(cache_stats_file, "w") as f:
                json.dump(stats_dict, f)
        except Exception as e:
            logger.debug(f"Failed to write cache stats to file: {e}")

    @staticmethod
    def get_current_cache_stats() -> dict:
        """获取当前的cache统计信息（不依赖profiling）"""
        try:
            # 使用现有的REQUEST_STATS计算cache统计
            stats = list(C_SchedulerHook.REQUEST_STATS.values())

            if not stats:
                return {
                    'prefix_cache_reused_ratio': 0.0,
                    'memory_prefetch_ratio': 0.0,
                    'disk_prefetch_ratio': 0.0,
                    'total_input': 0,
                }

            # 过滤有效请求
            valid_stats = [s for s in stats if s.input_length > 0]

            if not valid_stats:
                return {
                    'prefix_cache_reused_ratio': 0.0,
                    'memory_prefetch_ratio': 0.0,
                    'disk_prefetch_ratio': 0.0,
                    'total_input': 0,
                }

            # 计算统计信息
            total_input = sum(s.input_length for s in valid_stats)
            total_reused_tokens = sum(s.final_reused_tokens for s in valid_stats)
            total_memory_hit_tokens = sum(getattr(s, 'memory_hit_tokens', 0) for s in valid_stats)
            total_disk_hit_tokens = sum(s.prefetch_complete_tokens for s in valid_stats)

            result = {
                'prefix_cache_reused_ratio': total_reused_tokens / total_input if total_input > 0 else 0.0,
                'memory_prefetch_ratio': total_memory_hit_tokens / total_input if total_input > 0 else 0.0,
                'disk_prefetch_ratio': total_disk_hit_tokens / total_input if total_input > 0 else 0.0,
                'total_input': total_input,
            }

            return result
        except Exception as e:
            logger.error(f"Error calculating cache stats: {e}")
            return {
                'prefix_cache_reused_ratio': 0.0,
                'memory_prefetch_ratio': 0.0,
                'disk_prefetch_ratio': 0.0,
                'total_input': 0,
            }

    @classmethod
    def hook(cls, target):
        original_init = target.__init__
        original_get_new_batch_prefill = target.get_new_batch_prefill
        original_run_batch = target.run_batch
        original_process_batch_result = target.process_batch_result
        original_event_loop_normal = target.event_loop_normal
        # SGLang 0.5.13+: recv_requests moved to SchedulerRequestReceiver
        # Try to capture if it exists on the target
        original_recv_requests = getattr(target, 'recv_requests', None)

        def override_event_loop_overlap(self, *args, **kwargs):
            # To reduce the complexity of the simulation, the overlapping schedule is not needed.
            return original_event_loop_normal(self, *args, **kwargs)

        def wrapped_init(self, *args, **kwargs):
            # SGLang 0.5.13 compatibility: Ensure ModelRunner has canary_manager attribute
            # This must be done before calling original_init because it accesses canary_manager
            try:
                from sglang.srt.model_executor.model_runner import ModelRunner
                if not hasattr(ModelRunner, 'canary_manager'):
                    logger.info("Adding canary_manager support to ModelRunner class for SGLang 0.5.13+ compatibility")

                    class DummyCanaryManager:
                        """Dummy canary_manager for SGLang 0.5.13+ compatibility"""
                        def __init__(self):
                            pass
                        def attach_radix_cache(self, tree_cache):
                            pass
                        def __getattr__(self, name):
                            # Make it tolerant to any attribute access
                            return None

                    # Add as a property to the class
                    def get_canary_manager(self):
                        if not hasattr(self, '_canary_manager'):
                            self._canary_manager = DummyCanaryManager()
                        return self._canary_manager

                    ModelRunner.canary_manager = property(get_canary_manager)
            except Exception as e:
                logger.debug(f"Could not add canary_manager support to ModelRunner: {e}")

            # Disable overlap schedule
            server_args = get_obj_from_args(
                "sglang.srt.server_args.ServerArgs", *args, **kwargs
            )
            # Disable overlap schedule in HiSim simulation mode to avoid timing calculation issues
            C_SchedulerHook.OVERLAP_SCHEDULE = False
            setattr(server_args, "disable_overlap_schedule", True)
            logger.debug(
                f"Overlap schedule disabled in HiSim simulation mode."
            )

            original_init(self, *args, **kwargs)

            try:
                # Ensure model and hardware are registered in the subprocess
                # Because multiprocessing creates new processes without shared memory
                config_path = os.getenv("HISIM_CONFIG_PATH")
                config_dict = json.load(open(config_path)) if config_path and os.path.exists(config_path) else {}
                model_name = self.model_config.hf_config.__dict__.get("model_name", config_dict.get("model", {}).get("name"))
                hw_name = self.server_args.model_path  # Use model_name as key for hardware lookup

                # Register model if not already registered in this process
                if not ModelInfo.find_by_model_name(model_name):
                    logger.info(f"Re-registering model in subprocess: {model_name}")

                    # Use configured context_length if available
                    configured_context_len = None
                    if config_path and os.path.exists(config_path):
                        config = json.load(open(config_path))
                        configured_context_len = config.get("scheduler", {}).get("context_length")

                    model_config = getattr(self.model_config, 'max_position_embeddings', 131072)
                    if configured_context_len:
                        model_config = max(model_config, configured_context_len)
                        logger.info(f"Using configured context_length={configured_context_len} for model registration")

                    ModelInfo.from_dict({
                        'name': model_name,
                        'model_type': 'gpt_oss',
                        'hidden_size': getattr(self.model_config, 'hidden_size', 5120),
                        'num_attention_heads': getattr(self.model_config, 'num_attention_heads', 32),
                        'num_hidden_layers': getattr(self.model_config, 'num_hidden_layers', 32),
                        'vocab_size': getattr(self.model_config, 'vocab_size', 32000),
                        'intermediate_size': getattr(self.model_config, 'intermediate_size', 13696),
                        'num_key_value_heads': getattr(self.model_config, 'num_key_value_heads', 8),
                        'max_position_embeddings': model_config,
                        'max_seq_len': model_config,  # Also update max_seq_len
                        'torch_dtype': str(getattr(self.model_config, 'dtype', 'float16')),
                        'layer_types': [],
                    }, save_to_registry=True)

                # Register hardware if not already registered in this process
                if hw_name and not AcceleratorInfo.find_by_hw_name(hw_name):
                    # Try to get hardware name from the config
                    config_path = os.getenv("HISIM_CONFIG_PATH")
                    if config_path:
                        with open(config_path) as f:
                            config = json.load(f)
                        hw_name = config.get("platform", {}).get("accelerator", {}).get("name", hw_name)

                    # Try to load and register hardware config if available
                    import json as json_lib
                    hw_config_path = os.getenv("HISIM_CONFIG_PATH")
                    if hw_config_path:
                        predictor_config = json.load(open(hw_config_path)).get("predictor", {})
                        hw_path = predictor_config.get("hardware_config")
                        task_test_root = predictor_config.get("task_test_root")

                        if hw_path and (task_test_root or predictor_config.get("inference_predictor_root")):
                            if task_test_root and not os.path.isabs(hw_path):
                                hw_file = os.path.join(task_test_root, hw_path)
                            else:
                                inf_root = predictor_config.get("inference_predictor_root")
                                hw_file = os.path.join(inf_root, hw_path) if inf_root else hw_path

                            if os.path.exists(hw_file):
                                hw_data = json_lib.load(open(hw_file))
                                logger.info(f"Re-registering hardware in subprocess: {hw_name} from {hw_file}")
                                AcceleratorInfo.from_dict({
                                    'name': hw_name,
                                    'vendor': 'NVIDIA',
                                    'hbm_capacity_gb': hw_data.get("mem_size", 64),
                                    'hbm_bandwidth_gb': hw_data.get("mem_bw", 1600),
                                    'intra_node_bandwidth_gb': hw_data.get("intra_bw", 1600),
                                    'inter_node_bandwidth_gb': hw_data.get("inter_bw", 1600),
                                    'device_alias': [hw_name],
                                }, save_to_registry=True)

                model = ConfigManager.get_model_info(
                    self.model_config.hf_config.__dict__
                )
                hw = ConfigManager.get_accelerator_info()
                sched_config = ConfigManager.get_scheduler_config(
                    self.server_args.__dict__,
                    "sglang",
                    self.model_config.hf_config.__dict__,
                )
                ConfigManager.set_scheduler_config(sched_config)
                ConfigManager.set_model_info(model)

                C_SchedulerHook.INFERENCE_PREDICTOR = (
                    ConfigManager.get_inference_time_predictor(model, hw, sched_config)
                )
            except Exception as e:
                logger.error(
                    f"Failed to initialize inference time predictor. Error: {e}"
                )
                raise e

            # Override on_idle: previously this wrapper swallowed
            # "pool memory leak" / "invariant" ValueErrors to keep simulation
            # running. That hid real accounting bugs. Now we let the original
            # on_idle run unmodified — if the invariant checker detects a
            # leak, the ValueError propagates and surfaces the real bug.
            if hasattr(self, 'on_idle') and hasattr(self, 'invariant_checker'):
                original_on_idle = self.on_idle

                def wrapped_on_idle(*args, **kwargs):
                    return original_on_idle(*args, **kwargs)

                self.on_idle = wrapped_on_idle

            # Fix: Sanitize mem_fraction_static to prevent invalid memory configuration
            # SGLang's automatic calculation can produce invalid values (e.g., -2.123)
            # when chunked_prefill_size is too large. When this happens, we should
            # use the value from HiSim config instead of overriding to 0.9.
            # This allows users to intentionally set a low mem_fraction_static
            # (e.g., 0.2) to create eviction pressure for cache hit rate testing.
            if hasattr(self, 'server_args') and hasattr(self.server_args, 'mem_fraction_static'):
                original_fraction = self.server_args.mem_fraction_static

                if original_fraction is None or original_fraction <= 0:
                    # SGLang's automatic calculation produced an invalid value.
                    # Use the value from HiSim config file instead of hardcoding 0.9.
                    from hisim.simulation.manager.env import Envs
                    try:
                        with open(Envs.config_path()) as f:
                            hisim_config = json.load(f)
                        config_fraction = hisim_config.get("scheduler", {}).get("mem_fraction_static")
                    except Exception:
                        config_fraction = None

                    if config_fraction is not None and config_fraction > 0:
                        corrected_fraction = max(0.01, min(0.95, config_fraction))
                        logger.info(
                            f"Detected mem_fraction_static invalid ({original_fraction}) from SGLang. "
                            f"Using config value {config_fraction} (clamped to {corrected_fraction})."
                        )
                    else:
                        corrected_fraction = 0.9
                        logger.warning(
                            f"Detected mem_fraction_static invalid ({original_fraction}) and "
                            f"no config override. Using default {corrected_fraction}."
                        )

                    self.server_args.mem_fraction_static = corrected_fraction

                    # Update our internal scheduler config to use the corrected value
                    sched_config = ConfigManager.get_scheduler_config(
                        self.server_args.__dict__,
                        "sglang",
                        self.model_config.hf_config.__dict__,
                    )
                    sched_config.mem_fraction_static = corrected_fraction
                    ConfigManager.set_scheduler_config(sched_config)
                elif original_fraction < 0.5:
                    # User intentionally set a low value — warn but don't override
                    logger.info(
                        f"mem_fraction_static={original_fraction} is low. "
                        f"HBM capacity will be limited, causing aggressive eviction. "
                        f"This is expected for cache hit rate testing."
                    )

            logger.info("=" * 60)
            logger.info("HiSim KVCache Hierarchy Configuration:")
            logger.info("=" * 60)
            logger.info(f"L1 (HBM/GPU):     {hw.hbm_capacity_gb}GB capacity, {hw.hbm_bandwidth_gb}GB/s bandwidth")
            platform_config = ConfigManager.get_platform_config()
            logger.info(f"L2 (Memory):      {platform_config.memory_read_bandwidth_gb or 'N/A'}GB/s read bandwidth (capacity: {platform_config.memory_capacity_gb or 'unlimited'}GB)")
            logger.info(f"L3 (Disk):        {platform_config.disk_read_bandwidth_gb or 'N/A'}GB/s read bandwidth (capacity: {platform_config.disk_capacity_gb or 'unlimited'}GB)")
            logger.info(f"HBM util:         {sched_config.mem_fraction_static:.1%} for KV cache")
            logger.info(f"Storage backend:  {'enabled' if sched_config.hicache_storage_backend else 'disabled'}")
            logger.info("=" * 60)

        def wrapped_recv_requests(self, *args, **kwargs) -> list:
            # Helper function to recv requests from the appropriate source
            def recv_from_source():
                if hasattr(self, 'request_receiver') and hasattr(self.request_receiver, 'recv_requests'):
                    # SGLang 0.5.13+: recv_requests moved to SchedulerRequestReceiver
                    return self.request_receiver.recv_requests(*args, **kwargs)
                elif original_recv_requests is not None and callable(original_recv_requests):
                    # SGLang 0.5.9 and earlier: recv_requests on Scheduler
                    return original_recv_requests(self, *args, **kwargs)
                return []

            recv_reqs = []

            if C_SchedulerHook.SIM_MODE == MockSimulationMode.BLOCKING:
                recv_reqs.extend(recv_from_source())
            elif C_SchedulerHook.SIM_MODE == MockSimulationMode.OFFLINE:
                # Initializing
                if not C_SchedulerHook.OFFLINE_RECV_ALL_REQUEST:
                    gen_requests = []
                    extra_requests = []
                    time.sleep(0.05)  # waiting requests

                    reqs = recv_from_source()

                    for req in reqs:
                        if req.__class__.__name__ == "TokenizedGenerateReqInput":
                            gen_requests.append(req)
                        else:
                            # Such as: /profile_start, /flush_cache, etc.
                            extra_requests.append(req)

                    # Add requests to future queue
                    for req in gen_requests:
                        sim_params = None
                        if req.sampling_params.custom_params is not None:
                            sim_params = req.sampling_params.custom_params.get(
                                "simulation"
                            )
                        if sim_params is None:
                            # There are some warm-up requests when starting the server without --skip-server-warmup.
                            extra_requests.append(req)
                            logger.warning(
                                "Failed to extract the simulation parameters required for simulation from the request. Ignore this warning if the request is a warm-up request."
                            )
                            continue
                        if sim_params.get("queue_start"):
                            logger.debug(
                                "Add request to waiting queue with custom queue start timestamp."
                            )

                        C_SchedulerHook.FUTURE_QUEUE.append(
                            (
                                sim_params.get("queue_start")
                                or sim_params["created_time"],
                                time.time_ns(),  # The request is not comparable, so add the salt to avoid comparison.
                                req,
                            )
                        )

                    if len(C_SchedulerHook.FUTURE_QUEUE) != 0:
                        _, _, gen_req = C_SchedulerHook.FUTURE_QUEUE[-1]
                        total_request = gen_req.sampling_params.custom_params[
                            "simulation"
                        ]["total_request"]

                        if len(C_SchedulerHook.FUTURE_QUEUE) == total_request:
                            C_SchedulerHook.OFFLINE_RECV_ALL_REQUEST = True
                            heapq.heapify(C_SchedulerHook.FUTURE_QUEUE)
                            logger.info(
                                "All requests received. Starting simulation now."
                            )
                        else:
                            logger.info(
                                f"Offline simulation mode enabled. {total_request} requests expected in total. Received {len(C_SchedulerHook.FUTURE_QUEUE)} requests so far."
                            )

                    if len(extra_requests) != 0:
                        # Schedule the extra requests immediately.
                        return extra_requests
                else:
                    # Extra requests include: flush request, abort request, etc.
                    recv_reqs.extend(recv_from_source())

                # Process the arrived requests only after all requests have been added to the future queue
                current_timestamp = StateManager.get_global_clock()
                while (
                    C_SchedulerHook.OFFLINE_RECV_ALL_REQUEST
                    and len(C_SchedulerHook.FUTURE_QUEUE) > 0
                ):
                    enqueue_time, _, req = C_SchedulerHook.FUTURE_QUEUE[0]
                    if enqueue_time > current_timestamp:
                        break
                    recv_reqs.append(req)
                    heapq.heappop(C_SchedulerHook.FUTURE_QUEUE)

            now = time.time()
            for req in recv_reqs:
                if req.__class__.__name__ in [
                    "BatchTokenizedGenerateReqInput",
                    "TokenizedGenerateReqInput",
                ]:
                    req_stats = C_SchedulerHook.REQUEST_STATS[req.rid]
                    req_stats.rid = req.rid
                    req_stats.input_length = len(req.input_ids)
                    req_stats.output_length = req.sampling_params.max_new_tokens
                    simulation_args = req.sampling_params.custom_params["simulation"]
                    if C_SchedulerHook.SIM_MODE == MockSimulationMode.BLOCKING:
                        if "server_created_time" not in simulation_args:
                            logger.warning(
                                "The request's creation time is missing, which may cause the TTFT to be inaccurate."
                            )
                        req_stats.created_time = simulation_args.get(
                            "server_created_time", now
                        )
                        req_stats.last_event_time = req_stats.created_time
                        req_stats.queue_start = now
                    elif C_SchedulerHook.SIM_MODE == MockSimulationMode.OFFLINE:
                        req_stats.created_time = simulation_args["created_time"]
                        # Critical fix: Store both absolute timestamp and simulation clock value
                        # This avoids negative token_latency caused by mixing time reference frames
                        req_stats.absolute_created_time = req_stats.created_time
                        req_stats.last_event_time = StateManager.get_global_clock()
                        req_stats.queue_start = StateManager.get_global_clock()

                    # Session-aware: extract session info from simulation_args
                    session_id = simulation_args.get("session_id")
                    if session_id is not None:
                        req_stats.session_id = session_id
                        req_stats.parent_session_id = simulation_args.get("parent_session_id")

            if recv_reqs and C_SchedulerHook.LAST_CPU_TS == 0:
                C_SchedulerHook.LAST_CPU_TS = time.time()
                # Don't reset global clock to 0 - it may have been properly initialized
                # by queue_start from previous requests, and resetting it here causes
                # the mismatch between last_event_time and request_response_time
                if StateManager.get_global_clock() == 0:
                    StateManager.set_global_clock(0)

            return recv_reqs

        def wrapped_get_new_batch_prefill(self, *args, **kwargs):
            # NOTE: The previous admission-control clamp on evictable_size_ has
            # been removed. It masked the underlying accounting imbalance
            # (negative evictable_size_ from split/restore length mismatch,
            # inflated available_size_ from duplicate free() calls) by forcing
            # rem_total_tokens = max_total - offset, which made the scheduler
            # over-admit prefills and led to decode batches whose token usage
            # exceeded max_total_num_tokens. Now the scheduler sees the true
            # (possibly negative or inflated) values: if rem_total_tokens <= 0
            # prefills are rejected (visible stall) and if available + evictable
            # > max the downstream invariant checker raises. Both surface the
            # real bug instead of hiding it.

            # DEBUG: Log admission state before calling original (only when abnormal)
            try:
                _avail_pre = self.token_to_kv_pool_allocator.available_size()
                _evict_pre = self.tree_cache.evictable_size()
                _max_pre = self.max_total_num_tokens
                _rem_pre = _avail_pre + _evict_pre
                _running = len(self.running_batch) if hasattr(self, 'running_batch') else -1
                _queue = len(self.waiting_queue) if hasattr(self, 'waiting_queue') else -1
                _batch_full = getattr(self.running_batch, 'batch_is_full', False) if hasattr(self, 'running_batch') else False
                if _evict_pre < 0 or _rem_pre < 0 or _batch_full:
                    logger.warning(
                        f"[AdmissionDebug] available={_avail_pre} evictable={_evict_pre} "
                        f"max={_max_pre} rem_total={_rem_pre} running={_running} queue={_queue} "
                        f"batch_is_full={_batch_full}"
                    )
            except Exception:
                pass

            new_batch = original_get_new_batch_prefill(self, *args, **kwargs)

            # DEBUG: Log if prefill was rejected (only log first 20 rejections)
            if new_batch is None:
                try:
                    if not hasattr(self, '_admission_reject_count'):
                        self._admission_reject_count = 0
                    self._admission_reject_count += 1
                    if self._admission_reject_count <= 20:
                        _avail_post = self.token_to_kv_pool_allocator.available_size()
                        _evict_post = self.tree_cache.evictable_size()
                        _max_post = self.max_total_num_tokens
                        _rem_post = _avail_post + _evict_post
                        _batch_full_post = getattr(self.running_batch, 'batch_is_full', False) if hasattr(self, 'running_batch') else False
                        logger.warning(
                            f"[AdmissionReject] available={_avail_post} evictable={_evict_post} "
                            f"max={_max_post} rem_total={_rem_post} "
                            f"batch_is_full={_batch_full_post} "
                            f"(count={self._admission_reject_count})"
                        )
                except Exception:
                    pass
            now = time.time()

            # Detailed debugging for large requests (>= 10000 tokens)
            if new_batch is not None:
                total_input_len = sum(req.extend_input_len if hasattr(req, 'extend_input_len') else req.fill_len for req in new_batch.reqs)
                # Removed debug logs for large requests

                for req in new_batch.reqs:
                    req_stats = C_SchedulerHook.REQUEST_STATS[req.rid]
                    # Use per-level cache breakdown if available (HiRadixCache).
                    # SGLang provides cached_tokens_device (L1/HBM),
                    # cached_tokens_host (L2/Memory), cached_tokens_storage (L3/Disk).
                    #
                    # Cache hit rate semantics (mutually exclusive, L1 bounded by HBM capacity):
                    #   L1 (HBM)   = device tokens only = tokens that were already in HBM
                    #   L2 (Memory) = host tokens only  = tokens loaded from host (needed load_back)
                    #   L3 (Disk)   = storage tokens   = tokens loaded from disk (needed prefetch)
                    #   Total       = L1 + L2 + L3    = total cache hit rate
                    #
                    # Key constraint: L1 cache size must not exceed max_total_num_tokens
                    # (the HBM KV cache pool size). When tokens are evicted from HBM
                    # (reallocated to new requests), the corresponding cache entries
                    # are no longer in L1 — they become L2 (host) or are removed.
                    # Compute per-level cache breakdown
                    input_len = req_stats.input_length
                    if hasattr(req, 'cached_tokens_device'):
                        device_portion = req.cached_tokens_device
                        host_portion = getattr(req, 'cached_tokens_host', 0)
                        storage_portion = getattr(req, 'cached_tokens_storage', 0)
                        # Fallback: if storage was not set by schedule_batch but
                        # prefetch completed, use the prefetch value
                        if storage_portion == 0 and req_stats.prefetch_complete_tokens > 0:
                            storage_portion = req_stats.prefetch_complete_tokens

                        # Raw SGLang values for diagnostics
                        raw_device = device_portion
                        raw_host = host_portion
                        raw_storage = storage_portion
                        raw_total = raw_device + raw_host + raw_storage

                        # Total reused tokens across all cache levels
                        total_cached = device_portion + host_portion + storage_portion

                        # Safety clamp: total_cached must not exceed input_length.
                        # In most cases device+host+storage <= input_length,
                        # but page alignment or timing issues can inflate values.
                        if total_cached > input_len and input_len > 0:
                            scale = input_len / total_cached
                            device_portion = int(device_portion * scale)
                            host_portion = int(host_portion * scale)
                            storage_portion = input_len - device_portion - host_portion
                            total_cached = input_len
                            logger.debug(
                                f"Clamping cache breakdown for req {req.rid}: "
                                f"raw_total={raw_total} -> {input_len} "
                                f"(raw: dev={raw_device} host={raw_host} storage={raw_storage})"
                            )

                        # Mutually exclusive breakdown:
                        # L1 = device only (tokens that were in HBM, no load_back needed)
                        # L2 = host only (tokens that needed load_back from host)
                        # L3 = storage only (tokens loaded from disk)
                        # Total = L1 + L2 + L3 = overall cache hit rate
                        req_stats.final_reused_tokens = device_portion
                        req_stats.memory_hit_tokens = host_portion
                        req_stats.prefetch_complete_tokens = storage_portion

                        # Detailed per-request prefill logging
                        total_hit = device_portion + host_portion + storage_portion
                        miss_len = max(0, input_len - total_hit)
                        prefix_len = len(req.prefix_indices) if hasattr(req, 'prefix_indices') else 0
                        host_hit = getattr(req, 'host_hit_length', 0)
                        storage_hit = getattr(req, 'storage_hit_length', 0)
                        hit_pct = f"({total_hit / input_len:.1%} hit)" if input_len > 0 else ""

                        # Cache space usage: L1 (HBM), L2 (Memory), L3 (Disk)
                        _l1_used = 0; _l1_cap = 0
                        _l2_used = 0; _l2_cap = 0
                        _l3_keys = -1
                        try:
                            _alloc = self.token_to_kv_pool_allocator
                            _l1_cap = getattr(_alloc, 'size', 0) or self.max_total_num_tokens
                            _l1_used = _l1_cap - _alloc.available_size()
                            if hasattr(self.tree_cache, 'cache_controller'):
                                _cc = self.tree_cache.cache_controller
                                if hasattr(_cc, 'mem_pool_host'):
                                    _hp = _cc.mem_pool_host
                                    _l2_cap = getattr(_hp, 'size', 0)
                                    _l2_used = _l2_cap - _hp.available_size() if _l2_cap > 0 else 0
                                if hasattr(_cc, 'hicache_storage'):
                                    _st = _cc.hicache_storage
                                    _l3_keys = len(getattr(_st, 'storage', set()))
                        except Exception:
                            pass

                        def _fmt_tok(used, cap):
                            """Format token count as usedK/capK (pct%)."""
                            if cap <= 0:
                                return ""
                            def _k(n):
                                return f"{n/1000:.0f}K" if n >= 1000 else f"{n}"
                            pct = f" {used/cap:.0%}" if cap > 0 else ""
                            return f"{_k(used)}/{_k(cap)}{pct}"

                        _l1_str = f"L1={_fmt_tok(_l1_used, _l1_cap)}" if _l1_cap > 0 else ""
                        _l2_str = f"L2={_fmt_tok(_l2_used, _l2_cap)}" if _l2_cap > 0 else ""
                        _l3_str = f"L3_keys={_l3_keys}" if _l3_keys >= 0 else ""
                        _space_parts = [s for s in [_l1_str, _l2_str, _l3_str] if s]
                        _space_str = " ".join(_space_parts)

                        logger.info(
                            f"[Prefill] req={req.rid} input={input_len} "
                            f"L1(HBM)={device_portion} L2(Mem)={host_portion} "
                            f"L3(Disk)={storage_portion} miss={miss_len} "
                            f"cached_total={total_hit} {hit_pct} "
                            f"[raw: prefix_idx={prefix_len} host_hit={host_hit} "
                            f"storage_hit={storage_hit} "
                            f"dev={raw_device} host={raw_host} disk={raw_storage}] "
                            f"[space: {_space_str}]"
                        )
                    else:
                        # No HiCache — all cached tokens are L1 (HBM only)
                        req_stats.final_reused_tokens = min(req.cached_tokens, input_len)
                        miss_len = max(0, input_len - req.cached_tokens)
                        # Cache space usage for non-HiCache path
                        _l1_used = 0; _l1_cap = 0
                        try:
                            _alloc = self.token_to_kv_pool_allocator
                            _l1_cap = getattr(_alloc, 'size', 0) or self.max_total_num_tokens
                            _l1_used = _l1_cap - _alloc.available_size()
                        except Exception:
                            pass
                        _l1_pct = f"{_l1_used/1000:.0f}K/{_l1_cap/1000:.0f}K {_l1_used/_l1_cap:.0%}" if _l1_cap > 0 else ""
                        logger.info(
                            f"[Prefill] req={req.rid} input={input_len} "
                            f"L1(HBM)={req.cached_tokens} miss={miss_len} "
                            f"cached_total={req.cached_tokens} "
                            f"[space: L1={_l1_pct}]"
                        )
                    if req_stats.queue_end == -1:
                        if C_SchedulerHook.SIM_MODE == MockSimulationMode.BLOCKING:
                            req_stats.queue_end = now
                        else:
                            req_stats.queue_end = StateManager.get_global_clock()

                    # Session-aware: update session TTL when request is scheduled
                    custom_params = getattr(req.sampling_params, 'custom_params', None)
                    sim_args = custom_params.get("simulation", {}) if custom_params else {}
                    if sim_args.get("session_id") and sim_args.get("cache_control"):
                        _update_session_ttl(
                            sim_args["session_id"], sim_args["cache_control"]
                        )

                    else:
                        # Chunked request - update state tracking
                        prefill_completed_len = getattr(req, 'prefill_completed_len', 0)
                        computed_indices_len = len(req.computed_indices) if hasattr(req, 'computed_indices') else 0
                        input_len = req.extend_input_len if hasattr(req, 'extend_input_len') else getattr(req, 'fill_len', 0)

                        # Removed debug logs for chunked request scheduling and stuck detection
                        # Keep stuck detection logic but remove logging
                        req_stats = C_SchedulerHook.REQUEST_STATS[req.rid]
                        last_computed_len = getattr(req_stats, 'last_computed_len', -1)
                        if computed_indices_len == last_computed_len:
                            req_stats.stuck_count = getattr(req_stats, 'stuck_count', 0) + 1
                        else:
                            req_stats.stuck_count = 0
                            req_stats.last_computed_len = computed_indices_len

            elif len(self.running_batch.reqs) == 0 and len(self.waiting_queue) > 0:
                # Removed debug logs for pending large requests
                large_pending = []
                current_time = time.time()
                # Removed debug logs for pending large requests

                # Prefetching
                StateManager.step_global_clock(0.005)
                StateManager.set_current_inference_dur(0.005)
            else:
                if C_SchedulerHook.SIM_MODE == MockSimulationMode.OFFLINE and (
                    len(C_SchedulerHook.FUTURE_QUEUE) != 0
                    and len(self.running_batch.reqs) == 0
                ):
                    next_created_time, _, req = C_SchedulerHook.FUTURE_QUEUE[0]
                    StateManager.set_global_clock(next_created_time + 1e-6)

            logger.debug(
                f"Get new batch prefill: global iteration={StateManager.get_iteration()}, "
                f"new batch={new_batch.batch_size() if new_batch is not None else 0}, "
                f"waiting queue={len(self.waiting_queue)}"
            )

            return new_batch

        def wrapped_run_batch(self, *args, **kwargs):
            batch = get_obj_from_args(
                "sglang.srt.managers.schedule_batch.ScheduleBatch", *args, **kwargs
            )

            # CRITICAL FIX: Simulate chunked prefill progress updates
            # This addresses the root cause where HiSim simulation doesn't update
            # prefill progress attributes (prefill_completed_len, computed_indices)
            # leading to stuck chunked prefill for large sequences
            if batch.forward_mode.is_extend():
                for req in batch.reqs:
                    # Get current progress values
                    prefill_completed_len = getattr(req, 'prefill_completed_len', 0)
                    extend_input_len = getattr(req, 'extend_input_len', 0)
                    input_len = getattr(req, 'fill_len', 0)

                    # Get chunk_size either from extend_input_len or default
                    chunk_size = extend_input_len if extend_input_len > 0 else 8192

                    # Simulate progress update based on the current chunk being processed
                    # In real prefill, computed_indices would be extended by the chunk size
                    if hasattr(req, 'computed_indices') and len(req.computed_indices) < input_len:
                        # Simulate extending computed_indices by this chunk's tokens
                        # We don't have actual token IDs, so we use a placeholder approach
                        current_token_count = len(req.computed_indices)
                        tokens_to_add = min(chunk_size, input_len - current_token_count)

                        # Extend computed_indices with placeholder indices
                        # This simulates that N more tokens have been processed
                        for i in range(tokens_to_add):
                            req.computed_indices.append(current_token_count + i)

                        # Removed debug logs for computed_indices update

                    # Update prefill_completed_len to reflect progress
                    new_prefill_completed_len = min(prefill_completed_len + chunk_size, input_len)
                    if new_prefill_completed_len > prefill_completed_len:
                        if not hasattr(req, 'prefill_completed_len'):
                            req.prefill_completed_len = 0
                        req.prefill_completed_len = new_prefill_completed_len

                        # Removed debug logs for prefill_completed_len update

            ret = original_run_batch(self, *args, **kwargs)

            if ret.__class__.__name__ == "GenerationBatchResult":
                hisim_batch = HisimScheduleBatch(reqs=[])
                if batch.forward_mode.is_extend():
                    for req in batch.reqs:
                        prefix_len = len(req.prefix_indices) if hasattr(req, 'prefix_indices') else 0
                        output_len = len(req.output_ids) if hasattr(req, 'output_ids') else 0
                        past_len = prefix_len + output_len
                        extend_input_len = getattr(req, 'extend_input_len', 0)

                        # Check for invalid past_kv_length
                        if past_len < 0:
                            logger.warning(
                                f"NEGATIVE past_len detected in extend: "
                                f"prefix_len={prefix_len}, output_len={output_len}, "
                                f"past_len={past_len}"
                            )

                        hisim_batch.reqs.append(
                            FakeRequest(
                                input_length=extend_input_len,
                                past_kv_length=past_len,
                            )
                        )
                elif batch.forward_mode.is_decode():
                    for req in batch.reqs:
                        prefix_len = len(req.prefix_indices) if hasattr(req, 'prefix_indices') else 0
                        output_len = len(req.output_ids) if hasattr(req, 'output_ids') else 0
                        past_len = prefix_len + output_len

                        # Check for invalid past_kv_length
                        if past_len < 0:
                            logger.warning(
                                f"NEGATIVE past_len detected in decode: "
                                f"prefix_len={prefix_len}, output_len={output_len}, "
                                f"past_len={past_len}, prefix_indices={prefix_len}, output_ids={output_len}"
                            )

                        hisim_batch.reqs.append(
                            FakeRequest(
                                input_length=1,
                                past_kv_length=past_len,
                            )
                        )

                if not hisim_batch.is_empty():
                    StateManager.inc_iteration()
                    predicted_latency = (
                        C_SchedulerHook.INFERENCE_PREDICTOR.predict_infer_time(
                            hisim_batch
                        )
                    )
                    predicted_latency = float(predicted_latency)

                    forward_latency = 0
                    if C_SchedulerHook.SIM_MODE == MockSimulationMode.BLOCKING:
                        now = time.time()
                        time.sleep(abs(predicted_latency))
                        now = time.time()
                        forward_latency = now - C_SchedulerHook.LAST_CPU_TS
                        C_SchedulerHook.LAST_CPU_TS = now
                    else:
                        now = time.time()
                        forward_latency = predicted_latency

                    StateManager.set_current_inference_dur(forward_latency)

                C_SchedulerHook.HISIM_BATCH = hisim_batch

            return ret

        def wrapped_process_batch_result(self, *args, **kwargs):
            batch = get_obj_from_args(
                "sglang.srt.managers.schedule_batch.ScheduleBatch", *args, **kwargs
            )

            # Debug: Log batch state before processing (using INFO level to ensure visibility)
            if batch is not None:
                if len(batch.reqs) > 0:
                    req_info = []
                    for req in batch.reqs:
                        is_chunked = getattr(req, 'is_chunked', 0) > 0
                        prefill_len = getattr(req, 'prefill_completed_len', 0)
                        input_len = req.extend_input_len if hasattr(req, 'extend_input_len') else getattr(req, 'fill_len', 0)
                        computed_len = len(req.computed_indices) if hasattr(req, 'computed_indices') else 0
                        mode = "EXTEND" if batch.forward_mode.is_extend() else "DECODE"

                        # Removed PROCESS_BATCH debug logs
                else:
                    # Removed empty batch debug logs
                    pass

            # Removed debug logs for batch processing state
            running_batch_size_before = len(self.running_batch.reqs)

            ret = original_process_batch_result(self, *args, **kwargs)

            # IMPORTANT FIX: Handle None return value for chunked prefill
            # When process_batch_result returns None, SGLang may have internally processed
            # the batch but didn't return a result structure. HiSim needs to manually
            # update running_batch to ensure proper state transition.
            if ret is None and batch is not None and batch.forward_mode.is_extend():
                # Removed debug logs for chunked prefill fix

                # DO NOT manually add requests to running_batch to avoid breaking SGLang's internal state
                # The key issue is that when process_batch_result returns None, SGLang has already
                # processed the batch internally. Our job is just to create a proper result object
                # to prevent downstream errors.
                # Removed debug logs for running_batch manipulation

                # Create a result object with required attributes
                # Import GenerationBatchResult from correct location for SGLang 0.5.13+
                try:
                    # Try to import from different locations for different SGLang versions
                    try:
                        from sglang.srt.managers.utils import GenerationBatchResult
                    except ImportError:
                        # Try older location
                        try:
                            from sglang.srt.server_args import GenerationBatchResult
                        except ImportError:
                            # Last resort: create a simple mock class
                            logger.warning("GenerationBatchResult not found, creating mock class")
                            class GenerationBatchResult:
                                def __init__(self):
                                    self.logits_output = None
                                    self.next_token_ids = None
                                    self.extend_input_len_per_req = None
                                    self.extend_logprob_start_len_per_req = None

                    # Create the result - use minimal required fields
                    result = GenerationBatchResult()

                    # Set extend_input_len_per_req if available
                    if hasattr(batch, 'extend_lens'):
                        try:
                            result.extend_input_len_per_req = batch.extend_lens
                        except (AttributeError, TypeError):
                            if hasattr(result, '__dict__'):
                                result.__dict__['extend_input_len_per_req'] = batch.extend_lens

                    # Set logprob start lengths if available
                    if hasattr(batch, 'extend_logprob_start_lens'):
                        try:
                            result.extend_logprob_start_len_per_req = batch.extend_logprob_start_lens
                        except (AttributeError, TypeError):
                            if hasattr(result, '__dict__'):
                                result.__dict__['extend_logprob_start_len_per_req'] = batch.extend_logprob_start_lens

                    # Add backward compatibility: attach batch to result for older HiSim logic
                    if not hasattr(result, 'batch') and hasattr(result, '__dict__'):
                        result.__dict__['batch'] = batch

                    # Replace None with our created result
                    ret = result
                    # Removed debug logs for result creation

                except Exception as e:
                    logger.error(f"Failed to create GenerationBatchResult object for None return: {e}")
                    import traceback
                    traceback.print_exc()
                    # Keep ret=None if creation fails

                # No manual running_batch update - let SGLang handle it
                running_batch_size_after = len(self.running_batch.reqs)

            # Removed debug logs for batch processing state

            # Removed debug logs for running batch state changes

            if batch is not None:
                if len(batch.reqs) == 0:
                    return ret

                hicache_l2_load_dur = StateManager.pop_hicache_l2_load_dur()
                hicache_l2_backup_dur = StateManager.pop_hicache_l2_backup_dur()
                clock_before = StateManager.get_global_clock()
                current_inference_dur = StateManager.get_current_inference_dur()

                if C_SchedulerHook.OVERLAP_SCHEDULE:
                    overlap_step = max(
                        hicache_l2_load_dur - StateManager.get_last_inference_dur(),
                        0,
                    )

                    StateManager.step_global_clock(overlap_step)
                    current_clock_after_overlap = StateManager.get_global_clock()
                    StateManager.step_global_clock(current_inference_dur)
                    current_clock_after_inference = StateManager.get_global_clock()
                    request_response_time = (
                        StateManager.get_global_clock() + hicache_l2_backup_dur
                    )
                    clock_after = StateManager.get_global_clock()
                    global_clock_final = StateManager.get_global_clock()
                else:
                    total_step = hicache_l2_load_dur + current_inference_dur + hicache_l2_backup_dur

                    # Check total_step for negative values
                    if total_step < 0:
                        logger.warning(
                            f"NEGATIVE total_step detected in Decode: "
                            f"hicache_l2_load_dur={hicache_l2_load_dur:.4f}, "
                            f"current_inference_dur={current_inference_dur:.4f}, "
                            f"hicache_l2_backup_dur={hicache_l2_backup_dur:.4f}, "
                            f"total_step={total_step:.4f}, "
                            f"#req_in_batch={len(batch.reqs)}"
                        )

                    StateManager.step_global_clock(total_step)
                    clock_after = StateManager.get_global_clock()
                    request_response_time = StateManager.get_global_clock()
                    global_clock_final = request_response_time
                for req in batch.reqs:
                    # SGLang 0.5.13+: is_chunked attribute may not exist
                    # Fixed logic: is_chunked should be True only when req.is_chunked > 0
                    is_chunked = getattr(req, 'is_chunked', 0) > 0

                    # Debug: Check for negative or invalid request statistics
                    req_stats = C_SchedulerHook.REQUEST_STATS.get(req.rid)
                    if req_stats and batch.forward_mode.is_decode():
                        # Check for latencies that could result in negative token usage
                        if req_stats.last_event_time > request_response_time:
                            logger.warning(
                                f"NEGATIVE latency detected in decode: "
                                f"rid={req.rid}, "
                                f"last_event_time={req_stats.last_event_time}, "
                                f"request_response_time={request_response_time}, "
                                f"difference={request_response_time - req_stats.last_event_time:.4f}s, "
                                f"input_length={req_stats.input_length}, "
                                f"output_length={req_stats.output_length}"
                            )

                    if not is_chunked:
                        if not req_stats:
                            logger.warning(f"Request stats not found for rid={req.rid}")
                        else:
                            token_latency = request_response_time - req_stats.last_event_time

                            # Debug: Check for negative token latency (likely cause of negative token usage)
                            if token_latency < 0:
                                logger.warning(
                                    f"NEGATIVE token_latency detected (Decode): "
                                    f"rid={req.rid}, "
                                    f"request_response_time={request_response_time:.4f}, "
                                    f"last_event_time={req_stats.last_event_time:.4f}, "
                                    f"token_latency={token_latency:.4f}, "
                                    f"input_length={req_stats.input_length}, "
                                    f"output_length={req_stats.output_length}"
                                )

                                # Also dump clock state
                                logger.warning(
                                    f"Clock state at negative token: "
                                    f"global_clock={StateManager.get_global_clock():.4f}, "
                                    f"current_inference_dur={StateManager.get_current_inference_dur():.4f}, "
                                    f"hicache_l2_load_dur={hicache_l2_load_dur:.4f}, "
                                    f"hicache_l2_backup_dur={hicache_l2_backup_dur:.4f}"
                                )

                            req_stats.gen_token_latencies.append(token_latency)
                        req_stats.last_event_time = request_response_time
                    else:
                        # Chunked request: track progress and handle completion
                        req_stats = C_SchedulerHook.REQUEST_STATS.get(req.rid)
                        if req_stats:
                            # Update chunked request statistics
                            if batch.forward_mode.is_extend():
                                # Prefill chunk - update progress
                                prefill_completed_len = getattr(req, 'prefill_completed_len', 0)
                                input_len = getattr(req, 'fill_len', 0) or len(getattr(req, 'computed_indices', []))
                                extend_input_len = getattr(req, 'extend_input_len', 0)

                                # Check if chunked prefill is completed with enhanced detection
                                computed_indices_len = len(req.computed_indices) if hasattr(req, 'computed_indices') else 0
                                chunk_size = getattr(req, 'prefill_chunk_size', 8192)  # Default chunk size from server args

                                # 迭代次数跟踪
                                chunk_iterations = getattr(req_stats, 'chunk_iterations', 0) + 1
                                req_stats.chunk_iterations = chunk_iterations

                                estimated_completion_after_chunks = (input_len + chunk_size - 1) // chunk_size

                                # 关键修复：基于迭代次数的强制完成检测
                                # 这解决了HiSim无法正确检测chunk完成的问题
                                force_complete = (
                                    chunk_iterations >= estimated_completion_after_chunks or
                                    chunk_iterations >= max(3, estimated_completion_after_chunks * 2)  # 超过预计2倍也强制完成
                                )

                                # Enhanced completion detection: use multiple indicators + forced completion
                                completed_indicators = [
                                    prefill_completed_len >= input_len,
                                    computed_indices_len >= input_len,
                                    # Additional detection: when completed length is very close to target
                                    abs(prefill_completed_len - input_len) < chunk_size,
                                    abs(computed_indices_len - input_len) < chunk_size,
                                    force_complete  # 强制完成条件
                                ]

                                if any(completed_indicators):
                                    completion_reason = "normal" if not force_complete else "forced"
                                    # Chunked prefill completed - mark for normal processing
                                    # Removed debug logs for chunked completion

                                    # Finalize this prefill step like a normal request
                                    req_stats.gen_token_latencies.append(
                                        request_response_time
                                        - req_stats.last_event_time  # queue duration
                                    )
                                    req_stats.last_event_time = request_response_time

                                    # 关键修复：清除chunked标志，强制状态转换
                                    if hasattr(req, 'is_chunked'):
                                        req.is_chunked = 0  # 清除chunked标志
                                    # Removed debug logs for is_chunked flag clearing
                                else:
                                    # Still in chunked prefill
                                    req_stats.last_event_time = request_response_time

                                    # Removed debug logs for chunked progress tracking
                            else:
                                # Decode chunk - should be rare for large requests
                                req_stats.gen_token_latencies.append(
                                    request_response_time
                                    - req_stats.last_event_time
                                )
                                req_stats.last_event_time = request_response_time
                # Iteration statistics
                C_SchedulerHook.ITERATION_STATS.append(
                    {
                        "requests": C_SchedulerHook.HISIM_BATCH.request_info(),
                        "forward_latency": current_inference_dur,
                        "l2_load_latency": hicache_l2_load_dur,
                        "l2_backup_latency": hicache_l2_backup_dur,
                    }
                )

                # Update cache_stats file regularly (every 10 iterations to avoid too much IO)
                if len(C_SchedulerHook.ITERATION_STATS) % 10 == 0:
                    try:
                        current_stats = C_SchedulerHook.get_current_cache_stats()
                        C_SchedulerHook._write_cache_stats_to_file(current_stats)
                    except Exception as e:
                        logger.debug(f"Failed to update cache stats file: {e}")

            C_SchedulerHook.LAST_CPU_TS = time.time()

            # Session-aware: cleanup expired session TTL entries
            _cleanup_expired_sessions()

            return ret

        def wrapped_profile(self, req, *args, **kwargs):
            stats: list[RequestStats] = []
            logger.info(f"DEBUG: REQUEST_STATS has {len(C_SchedulerHook.REQUEST_STATS)} items")
            for rid, item in C_SchedulerHook.REQUEST_STATS.items():
                logger.info(f"DEBUG: Request {rid}: rid={item.rid}, input_length={item.input_length}, created_time={item.created_time}")
                # Fix: 修复过滤条件，放弃过于严格的检查
                # 1. item.rid is not None 改为 item.rid，因为空字符串""也是有效的rid
                # 2. item.input_length > 0 改为 item.input_length > 0 是合理的，但添加更多日志
                if item.input_length > 0:
                    stats.append(item)

            stats = sorted(stats, key=lambda req: req.created_time)
            logger.info(f"DEBUG: After filtering, {len(stats)} requests have valid stats")

            output_dir = Envs.output_dir()
            os.makedirs(output_dir, exist_ok=True)

            if len(stats) > 0:
                # Remove warmup requests.
                if len(stats) > Envs.num_warmup():
                    metrics_stats = stats[Envs.num_warmup() :]
                else:
                    metrics_stats = stats

                min_created_time = metrics_stats[0].created_time
                # Align timestamps
                for item in stats:
                    item.created_time -= min_created_time
                    item.queue_start -= min_created_time
                    item.queue_end -= min_created_time
                    item.last_event_time -= min_created_time

                metrics = calc_metrics(metrics_stats)
                metrics["time_cost"] = time.time() - C_SchedulerHook.LAST_FLUSH_TS

                logger.info("=" * 60)
                logger.info("HiSim KVCache Hit Statistics:")
                logger.info("=" * 60)
                logger.info(f"Total input tokens:           {metrics.get('total_input', 0):,}")
                logger.info(f"L1 (HBM) hit tokens:          {int(metrics.get('total_input', 0) * metrics.get('prefix_cache_reused_ratio', 0)):,} ({metrics.get('prefix_cache_reused_ratio', 0):.2%})")
                logger.info(f"L2 (Memory) hit tokens:       {int(metrics.get('total_input', 0) * metrics.get('memory_prefetch_ratio', 0)):,} ({metrics.get('memory_prefetch_ratio', 0):.2%})")
                logger.info(f"L3 (Disk) hit tokens:          {int(metrics.get('total_input', 0) * metrics.get('disk_prefetch_ratio', 0)):,} ({metrics.get('disk_prefetch_ratio', 0):.2%})")
                logger.info(f"L1+L2+L3 combined hit rate:   {metrics.get('prefix_cache_reused_ratio', 0) + metrics.get('memory_prefetch_ratio', 0) + metrics.get('disk_prefetch_ratio', 0):.2%}")
                logger.info("=" * 60)

                try:
                    with open(f"{output_dir}/metrics.json", "w") as f:
                        f.write(json.dumps(metrics, cls=CustomJsonEncoder) + "\n")

                    with open(f"{output_dir}/iteration.jsonl", "w") as f:
                        for item in C_SchedulerHook.ITERATION_STATS:
                            f.write(json.dumps(item) + "\n")

                    with open(f"{output_dir}/request.jsonl", "w") as f:
                        for item in stats:
                            f.write(json.dumps(asdict(item)) + "\n")

                    logger.info(f"Simulation results saved to {output_dir}.")

                except Exception as e:
                    logger.error(f"Failed to dump results. Error: {e}")
            else:
                logger.warning(f"No request statistics available. Total REQUEST_STATS items: {len(C_SchedulerHook.REQUEST_STATS)}")
                # Log more details for debugging
                for rid, item in C_SchedulerHook.REQUEST_STATS.items():
                    logger.warning(f"  - Request {rid}: rid={item.rid}, input_length={item.input_length}, created_time={item.created_time}")

            StateManager.reset()
            SESSION_TTL_TABLE.clear()
            _clear_node_ownership()
            C_SchedulerHook.REQUEST_STATS.clear()
            C_SchedulerHook.ITERATION_STATS.clear()
            C_SchedulerHook.LAST_CPU_TS = 0
            C_SchedulerHook.LAST_FLUSH_TS = time.time()
            C_SchedulerHook.OFFLINE_RECV_ALL_REQUEST = False

            ProfileReqOutput = getattr(
                importlib.import_module("sglang.srt.managers.io_struct"),
                "ProfileReqOutput",
            )
            result = {
                "total_request": len(stats),
                "output_directory": output_dir,
            }

            return ProfileReqOutput(True, json.dumps(result))

        target.event_loop_overlap = override_event_loop_overlap
        target.__init__ = wrapped_init
        target.recv_requests = wrapped_recv_requests
        target.get_new_batch_prefill = wrapped_get_new_batch_prefill
        target.run_batch = wrapped_run_batch
        target.process_batch_result = wrapped_process_batch_result
        target.profile = wrapped_profile
        return target


class C_SchedulerRequestReceiverHook(BaseHook):
    """
    Hook for SchedulerRequestReceiver in SGLang 0.5.13+ where recv_requests moved.
    Populates C_SchedulerHook.REQUEST_STATS with request information.
    """
    HOOK_CLASS_NAME = "SchedulerRequestReceiver"
    HOOK_MODULE_NAME = "sglang.srt.managers.scheduler_components.request_receiver"

    @classmethod
    def hook(cls, target):
        original_recv_requests = target.recv_requests
        _logger = logger  # Local reference to logger

        def wrapped_recv_requests(self, *args, **kwargs):
            recv_reqs = original_recv_requests(self, *args, **kwargs)

            # Populate REQUEST_STATS with request information
            now = time.time()
            for req in recv_reqs:
                if req.__class__.__name__ in [
                    "BatchTokenizedGenerateReqInput",
                    "TokenizedGenerateReqInput",
                ]:
                    req_stats = C_SchedulerHook.REQUEST_STATS[req.rid]
                    req_stats.rid = req.rid
                    req_stats.input_length = len(req.input_ids)
                    req_stats.output_length = req.sampling_params.max_new_tokens

                    simulation_args = (req.sampling_params.custom_params or {}).get("simulation", {})
                    sim_mode = getattr(C_SchedulerHook, 'SIM_MODE', None)
                    if sim_mode == MockSimulationMode.BLOCKING:
                        if "server_created_time" not in simulation_args:
                            _logger.warning(
                                "The request's creation time is missing, which may cause the TTFT to be inaccurate."
                            )
                        req_stats.created_time = simulation_args.get(
                            "server_created_time", now
                        )
                        req_stats.last_event_time = req_stats.created_time
                        req_stats.queue_start = now
                    elif sim_mode == MockSimulationMode.OFFLINE:
                        req_stats.created_time = simulation_args.get("created_time", now)
                        # Fix: Use global clock for last_event_time in OFFLINE mode to avoid timing mismatch
                        req_stats.absolute_created_time = req_stats.created_time
                        try:
                            req_stats.last_event_time = StateManager.get_global_clock()
                        except:
                            req_stats.last_event_time = 0  # StateManager may not be initialized yet
                        queue_start = simulation_args.get("queue_start")
                        if queue_start is not None:
                            StateManager.set_global_clock(queue_start)
                        req_stats.queue_start = StateManager.get_global_clock()

                    # Session-aware: extract session info for SGLang 0.5.13+ path
                    session_id = simulation_args.get("session_id")
                    if session_id is not None:
                        req_stats.session_id = session_id
                        req_stats.parent_session_id = simulation_args.get("parent_session_id")
                        req_stats.session_end = simulation_args.get("session_end", False)

            if recv_reqs and getattr(C_SchedulerHook, 'LAST_CPU_TS', None) == 0:
                C_SchedulerHook.LAST_CPU_TS = time.time()
                StateManager.set_global_clock(0)

            return recv_reqs

        target.recv_requests = wrapped_recv_requests
        return target


class C_RadixCacheFixHook(BaseHook):
    """Hook to fix radix cache assertion errors during chunked operations
    and add session-aware TTL-protected eviction."""
    HOOK_CLASS_NAME = "RadixCache"
    HOOK_MODULE_NAME = "sglang.srt.mem_cache.radix_cache"

    @classmethod
    def hook(cls, target):
        original_cache_unfinished_req = target.cache_unfinished_req
        original_evict = target.evict

        # Import necessary classes for the wrapper
        from sglang.srt.mem_cache.radix_cache import RadixKey, MatchPrefixParams, InsertParams

        def wrapped_cache_unfinished_req(self, req, chunked=False):
            """Cache request when it is unfinished, with chunk length fix.

            IMPORTANT: This wrapper must call dec_lock_ref/inc_lock_ref after
            insert, matching the original RadixCache.cache_unfinished_req.
            Without these calls, newly inserted nodes keep lock_ref==0 and
            remain in evictable_leaves, allowing the evictor to prematurely
            evict KV cache of running requests that have finished prefill but
            not yet started decode.
            """
            if self.disable:
                return

            token_ids = req.get_fill_ids()
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, : len(token_ids)
            ]

            radix_key = RadixKey(
                token_ids, req.extra_key, is_bigram=self.is_eagle
            ).page_aligned(self.page_size)
            values = kv_indices[: len(radix_key)].to(dtype=torch.int64, copy=True)

            # Radix Cache takes one ref in memory pool
            result = self.insert(
                InsertParams(
                    key=radix_key,
                    value=values,
                    chunked=chunked,
                    priority=getattr(req, "priority", 0) or 0,
                )
            )
            new_prefix_len = result.prefix_len

            self.token_to_kv_pool_allocator.free(
                kv_indices[req.cache_protected_len : new_prefix_len]
            )

            # The prefix indices could be updated, reuse it
            match_result = self.match_prefix(MatchPrefixParams(key=radix_key))
            new_indices, new_last_node = (
                match_result.device_indices,
                match_result.last_device_node,
            )

            # Fix: Handle length mismatch that can occur during chunked operations
            # This can happen when the cache tree doesn't fully contain all expected prefixes
            # or request state during chunked operations gets out of sync
            if len(new_indices) != len(radix_key):
                logger.debug(
                    f"Cache length mismatch: len(new_indices)={len(new_indices)}, "
                    f"len(radix_key)={len(radix_key)}. "
                    f"Skipping cache update to maintain consistency."
                )
                # Even on length mismatch, we must update lock_ref so that
                # inserted nodes are not left with lock_ref==0 (which would
                # make them evictable while the request is still running).
                # new_last_node may be shorter than expected but is still a
                # valid tree node for locking the inserted prefix.
                self.dec_lock_ref(req.last_node)
                if new_last_node is not None:
                    self.inc_lock_ref(new_last_node)
                    req.last_node = new_last_node
                return

            self.req_to_token_pool.write(
                (req.req_pool_idx, slice(req.cache_protected_len, len(new_indices))),
                new_indices[req.cache_protected_len :],
            )

            req.cache_protected_len = len(new_indices)

            # Release the lock on the old last_node and acquire a lock on
            # the new last_node. This is critical: without inc_lock_ref,
            # the newly inserted nodes stay in evictable_leaves with
            # lock_ref==0, allowing the evictor to evict KV cache of
            # running requests that have finished prefill but not yet
            # started decode.
            self.dec_lock_ref(req.last_node)
            self.inc_lock_ref(new_last_node)

            # Update req.prefix_indices and req.last_node (matches original)
            if len(new_indices) < len(kv_indices):
                req.prefix_indices = torch.cat(
                    [new_indices, kv_indices[len(new_indices):]]
                )
            else:
                req.prefix_indices = new_indices
            req.last_node = new_last_node

            # Session-aware: tag nodes along the prefix path with session_id
            custom_params = getattr(req.sampling_params, 'custom_params', None) if hasattr(req, 'sampling_params') and req.sampling_params else None
            sim_args = custom_params.get("simulation", {}) if custom_params else {}
            session_id = sim_args.get("session_id")
            if session_id is not None:
                tag_node = new_last_node
                while tag_node is not None and tag_node is not self.root_node:
                    _tag_node_with_session(tag_node, session_id)
                    tag_node = tag_node.parent

        # Apply the wrapper only if original method exists
        if hasattr(target, 'cache_unfinished_req'):
            target.cache_unfinished_req = wrapped_cache_unfinished_req

        # --- Session-aware: TTL-protected eviction ---
        def wrapped_evict(self, params):
            """Eviction with session TTL protection: skip nodes owned by protected sessions."""
            if self.disable:
                from sglang.srt.mem_cache.base_prefix_cache import EvictResult
                return EvictResult()

            start_time = time.perf_counter()
            num_tokens = params.num_tokens
            leaves = list(self.evictable_leaves)
            eviction_heap = [
                (self.eviction_strategy.get_priority(node), node) for node in leaves
            ]
            heapq.heapify(eviction_heap)

            num_evicted = 0
            skipped_protected = 0

            while num_evicted < num_tokens and len(eviction_heap):
                _priority, x = heapq.heappop(eviction_heap)

                # Session TTL protection: skip nodes owned by TTL-protected sessions
                if _is_node_protected(x):
                    skipped_protected += 1
                    continue  # Skip without pushing back to heap

                self.token_to_kv_pool_allocator.free(x.value)
                num_evicted += len(x.value)
                self._delete_leaf(x)
                # Clean up ownership tracking for the deleted node
                _remove_node_ownership(x)

                if len(x.parent.children) == 0 and x.parent.lock_ref == 0:
                    new_priority = self.eviction_strategy.get_priority(x.parent)
                    heapq.heappush(eviction_heap, (new_priority, x.parent))

                self._record_remove_event(x)

            self.update_eviction_metrics(num_evicted, start_time)

            if skipped_protected > 0:
                logger.debug(
                    f"Eviction: evicted={num_evicted} tokens, "
                    f"skipped {skipped_protected} session-protected nodes"
                )

            from sglang.srt.mem_cache.base_prefix_cache import EvictResult
            return EvictResult(num_tokens_evicted=num_evicted)

        if hasattr(target, 'evict'):
            target.evict = wrapped_evict

        # --- Session-aware: node ownership tracking in cache_finished_req ---
        original_cache_finished_req = target.cache_finished_req

        def wrapped_cache_finished_req(self, req, is_insert=True):
            """Wrap cache_finished_req to track node→session ownership and handle session-end cleanup."""
            # Call original method first
            original_cache_finished_req(self, req, is_insert)

            # After caching, tag the affected nodes with the request's session_id
            custom_params = getattr(req.sampling_params, 'custom_params', None) if hasattr(req, 'sampling_params') and req.sampling_params else None
            sim_args = custom_params.get("simulation", {}) if custom_params else {}
            session_id = sim_args.get("session_id")
            if session_id is None:
                return

            # Walk the prefix path from root to last_node and tag all nodes
            # with the session_id for eviction tracking
            node = getattr(req, 'last_node', None)
            while node is not None and node is not self.root_node:
                _tag_node_with_session(node, session_id)
                node = node.parent

            # Session-end cleanup: evict exclusively-owned KV nodes
            if sim_args.get("session_end", False):
                _cleanup_session_kv(session_id, self)

        if hasattr(target, 'cache_finished_req'):
            target.cache_finished_req = wrapped_cache_finished_req

        return target


class C_PoolStatsObserverHook(BaseHook):
    """Hook to fix token accounting errors in simulation mode.

    Two issues can cause incorrect pool stats:
    1. available_size can exceed max_total_num_tokens due to invalid free()
       calls (duplicate frees, out-of-range indices) in the mock allocator.
    2. evictable_size can be double-counted when load_back() and
       dec_lock_ref() both increment evictable_size_ for the same nodes.

    Both issues lead to negative num_used = max_total_num_tokens - (available + evictable).
    More critically, the scheduler's rem_total_tokens = available + evictable can be
    inflated, allowing more requests than HBM can hold.

    The fix:
    - Clamp available + evictable to max_total_num_tokens (physical pool capacity)
    - Recalculate num_used and token_usage from the clamped values
    - This ensures the scheduler's admission control sees realistic token counts
    """
    HOOK_CLASS_NAME = "PoolStatsObserver"
    HOOK_MODULE_NAME = "sglang.srt.managers.scheduler_components.pool_stats_observer"

    @classmethod
    def hook(cls, target):
        original_get_pool_stats = target.get_pool_stats

        def wrapped_get_pool_stats(self, *args, **kwargs):
            """Fix token accounting errors in simulation mode."""
            pool_stats = original_get_pool_stats(self, *args, **kwargs)

            # Core fix: available + evictable must not exceed pool capacity.
            # This is a physical invariant: the pool has max_total_num_tokens
            # slots, and (available + evictable + used_by_running) = total.
            # If available + evictable > total, the scheduler would admit
            # too many requests (inflated rem_total_tokens).
            max_total = self.max_total_num_tokens
            available = pool_stats.full_available_size
            evictable = pool_stats.full_evictable_size
            total_free = available + evictable

            # FIX: Clamp negative evictable_size to 0.
            # Negative evictable_size can occur due to double-decrement across
            # rounds (session-end cleanup + regular eviction). This deflates
            # rem_total_tokens = available + evictable, preventing prefills.
            if evictable < 0:
                logger.warning(
                    f"[PoolStatsFix] NEGATIVE evictable={evictable}, clamping to 0 "
                    f"(available={available}, max_total={max_total})"
                )
                pool_stats.full_evictable_size = 0
                evictable = 0
                total_free = available + evictable

            if total_free > max_total:
                # Scale down proportionally to preserve the ratio
                # available : evictable, then clamp each individually
                scale = max_total / total_free
                clamped_available = int(available * scale)
                clamped_evictable = max_total - clamped_available

                logger.debug(
                    f"Clamping pool stats: available={available}->{clamped_available}, "
                    f"evictable={evictable}->{clamped_evictable} "
                    f"(total_free={total_free} > max_total={max_total})"
                )

                pool_stats.full_available_size = clamped_available
                pool_stats.full_evictable_size = clamped_evictable

            # Recalculate num_used and token_usage from the (possibly clamped) values
            num_used = max_total - (pool_stats.full_available_size + pool_stats.full_evictable_size)
            pool_stats.full_num_used = max(num_used, 0)
            pool_stats.full_token_usage = pool_stats.full_num_used / max_total if max_total > 0 else 0.0

            # Fix negative swa values
            if pool_stats.swa_num_used is not None and pool_stats.swa_num_used < 0:
                logger.debug(f"Fixing negative swa_num_used: {pool_stats.swa_num_used} -> 0")
                pool_stats.swa_num_used = 0
                pool_stats.swa_token_usage = 0.0

            # Fix negative mamba values
            if pool_stats.mamba_num_used is not None and pool_stats.mamba_num_used < 0:
                logger.debug(f"Fixing negative mamba_num_used: {pool_stats.mamba_num_used} -> 0")
                pool_stats.mamba_num_used = 0
                pool_stats.mamba_usage = 0.0

            # Fix negative hisparse values
            if pool_stats.hisparse_device_tokens is not None and pool_stats.hisparse_device_tokens < 0:
                logger.debug(f"Fixing negative hisparse_device_tokens: {pool_stats.hisparse_device_tokens} -> 0")
                pool_stats.hisparse_device_tokens = 0
                pool_stats.hisparse_device_token_usage = 0.0

            if pool_stats.hisparse_host_tokens is not None and pool_stats.hisparse_host_tokens < 0:
                logger.debug(f"Fixing negative hisparse_host_tokens: {pool_stats.hisparse_host_tokens} -> 0")
                pool_stats.hisparse_host_tokens = 0
                pool_stats.hisparse_host_token_usage = 0.0

            return pool_stats

        target.get_pool_stats = wrapped_get_pool_stats
        return target
