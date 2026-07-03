import json

from hisim.spec import ModelInfo, AcceleratorInfo, DataType
from hisim.simulation.types import PlatformConfig, SchedulerConfig
from hisim.simulation.manager import Envs
from hisim.simulation.utils import (
    calc_kv_cache_cell_elems,
    calc_kv_cache_per_layer_elems,
)
from hisim.time_predictor import (
    InferTimePredictor,
    AIConfiguratorTimePredictor,
)

# Import simple predictor as fallback
try:
    from hisim.time_predictor.simple_predictor import SimpleTimePredictor
    HAS_SIMPLE_PREDICTOR = True
except ImportError:
    HAS_SIMPLE_PREDICTOR = False

from hisim.utils import get_logger


logger = get_logger()


class ConfigManager:
    _model_info: ModelInfo = None
    _platform_config: PlatformConfig = None
    _scheduler_config: SchedulerConfig = None

    @classmethod
    def set_model_info(cls, model: ModelInfo):
        cls._model_info = model

    @classmethod
    def get_model_info(cls, hf_config: dict | None) -> ModelInfo:
        if hf_config is not None:
            model = ModelInfo.from_config(hf_config)
            if model is None:
                logger.error(
                    f"Failed to initialize model information with configuration: {hf_config}"
                )
        else:
            with open(Envs.config_path()) as f:
                config: dict = json.load(f)
            model = ModelInfo.find_by_model_name(config.get("model", {}).get("name"))

        # Fix: Ensure model_type is supported by aiconfigurator

        if model and model.model_type not in ["qwen", "qwen2", "qwen3", "llama", "chatglm", "deepseek_v3", "kimi_k2", "qwen3_moe"]:

            model.model_type = "llama"  # Use a supported type for estimation



        return model

    @classmethod
    def get_accelerator_info(cls) -> AcceleratorInfo:
        with open(Envs.config_path()) as f:
            config: dict = json.load(f)
        platform_config = config.get("platform", {})
        device_name = platform_config.get("accelerator", {}).get("name")
        hw = AcceleratorInfo.find_by_hw_name(device_name)
        if hw is None:
            logger.error(
                f"Failed to initialize device info with {device_name}. All available devices are: {AcceleratorInfo.list_all_hws().keys()}"
            )
            raise ValueError(f"Failed to initialize device info with {device_name}")
        else:
            logger
        return hw

    @classmethod
    def get_platform_config(cls) -> PlatformConfig:
        if cls._platform_config is None:
            hw = cls.get_accelerator_info()
            with open(Envs.config_path()) as f:
                config: dict = json.load(f)
            platform_config = config.get("platform", {})
            cls._platform_config = PlatformConfig(
                device=hw,
                disk_capacity_gb=platform_config.get("disk_capacity_gb"),
                disk_read_bandwidth_gb=platform_config.get("disk_read_bandwidth_gb"),
                disk_write_bandwidth_gb=platform_config.get("disk_write_bandwidth_gb"),
                memory_capacity_gb=platform_config.get("memory_capacity_gb"),
                memory_read_bandwidth_gb=platform_config.get(
                    "memory_read_bandwidth_gb"
                ),
                memory_write_bandwidth_gb=platform_config.get(
                    "memory_write_bandwidth_gb"
                ),
                num_device_per_node=platform_config.get("num_device_per_node"),
            )

            logger.info(
                f"Platform configuration initialized successfully. {cls._platform_config}"
            )

        return cls._platform_config

    @classmethod
    def set_scheduler_config(cls, config: SchedulerConfig):
        cls._scheduler_config = config

    @classmethod
    def get_kv_cache_bytes(cls) -> int:
        model = cls._model_info
        scheduler_config = cls._scheduler_config
        return (
            calc_kv_cache_cell_elems(
                model, scheduler_config.tp_size, scheduler_config.pp_size
            )
            * scheduler_config.data_type.bytes
        )

    @classmethod
    def get_kv_cache_bytes_per_layer(cls) -> int:
        model = cls._model_info
        scheduler_config = cls._scheduler_config
        return (
            calc_kv_cache_per_layer_elems(
                model, scheduler_config.tp_size, scheduler_config.pp_size
            )
            * scheduler_config.data_type.bytes
        )

    @classmethod
    def get_scheduler_config(
        cls, server_args: dict, backend: str, hf_config: dict | None = None
    ):
        model = ConfigManager.get_model_info(hf_config)

        internal_config = cls._parse_server_args(server_args, backend)

        with open(Envs.config_path()) as f:
            config: dict = json.load(f)
        scheduler_config = config.get("scheduler", {})

        tp_size = scheduler_config.get("tp_size")
        if tp_size is None:
            tp_size = internal_config.tp_size
        ep_size = scheduler_config.get("ep_size")
        if ep_size is None:
            ep_size = internal_config.ep_size
        dp_size = scheduler_config.get("dp_size")
        if dp_size is None:
            dp_size = internal_config.dp_size

        # Fix: Handle mem_fraction_static default value
        # In HiSim simulation, mem_fraction_static controls the simulated HBM pool
        # size for KV cache. Users may intentionally set a LOW value (e.g., 0.2)
        # to create eviction pressure for cache hit rate testing. We must NOT
        # override user-specified values with capacity-based constraints.
        #
        # The previous logic had a min_required_tokens=196000 check that forced
        # mem_fraction_static to 0.9 when estimated capacity was below 196K.
        # This was wrong for simulation — we don't need to actually fit all tokens
        # in HBM; the simulation correctly handles eviction when capacity is small.
        internal_mem_fraction = internal_config.mem_fraction_static
        config_mem_fraction = scheduler_config.get("mem_fraction_static")

        if config_mem_fraction is not None:
            # User explicitly set in config — use it directly, only clamp to [0.01, 0.95]
            mem_fraction_static = max(0.01, min(0.95, config_mem_fraction))
            logger.info(f"Using mem_fraction_static from config: {mem_fraction_static}")
        elif internal_mem_fraction is not None and internal_mem_fraction > 0:
            # SGLang server args provided a valid positive value
            mem_fraction_static = max(0.01, min(0.95, internal_mem_fraction))
            logger.info(f"Using mem_fraction_static from server args: {mem_fraction_static}")
        else:
            # Default to 0.9 (90% of HBM for KV cache)
            mem_fraction_static = 0.9
            logger.info(f"Using default mem_fraction_static: {mem_fraction_static}")

        # Log estimated capacity for diagnostics (but do NOT override)
        try:
            from hisim.simulation.utils import estimate_kv_cache_pool_capacity
            temp_config = SchedulerConfig(
                model=model,
                max_prefill_tokens=internal_config.max_prefill_tokens,
                chunked_prefill_size=internal_config.chunked_prefill_size,
                mem_fraction_static=mem_fraction_static,
                tp_size=tp_size,
                ep_size=ep_size,
                dp_size=dp_size,
                data_type=DataType.FP16,
                kv_cache_data_type=DataType.FP16,
                page_size=internal_config.page_size,
                backend_name="sglang",
            )
            temp_hw = ConfigManager.get_accelerator_info()
            estimated_capacity = estimate_kv_cache_pool_capacity(model, temp_hw, temp_config)
            logger.info(f"Estimated KV cache capacity with mem_fraction_static={mem_fraction_static}: {estimated_capacity} tokens")
        except Exception as e:
            logger.debug(f"Could not estimate capacity: {e}")

        if mem_fraction_static < 0.5:
            # User intentionally set a low value — warn but don't override.
            # This is useful for testing cache eviction pressure.
            logger.info(
                f"mem_fraction_static={mem_fraction_static} is low — HBM will have "
                f"limited capacity, causing aggressive eviction. "
                f"This is expected for cache hit rate testing."
            )
        dtype = scheduler_config.get("data_type")
        if dtype is not None:
            dtype = DataType(dtype.upper())
        else:
            dtype = DataType.from_torch_dtype(model.torch_dtype)

        kv_cache_dtype = scheduler_config.get("kv_cache_data_type")
        if kv_cache_dtype is not None:
            kv_cache_dtype = DataType(kv_cache_dtype)
        else:
            kv_cache_dtype = dtype

        logger.info(f"HISIM_DEBUG: model.model_type before fix = {model.model_type}, name = {model.name}")

        # Fix: Ensure model_type is supported
        if model.model_type not in ["qwen", "qwen2", "qwen3", "llama", "chatglm", "deepseek_v3", "kimi_k2", "qwen3_moe"]:
            model.model_type = "llama"

        logger.info(f"HISIM_DEBUG: model.model_type after fix = {model.model_type}")

        # Pass the validated mem_fraction_static to scheduler config
        logger.info(f"FINAL: mem_fraction_static = {mem_fraction_static}")

        # Read context_length from config and apply to model if specified
        context_length = scheduler_config.get("context_length")
        if context_length is not None:
            # Update model's max_seq_len and max_position_embeddings
            if hasattr(model, "max_seq_len"):
                model.max_seq_len = context_length
            if hasattr(model, "max_position_embeddings"):
                model.max_position_embeddings = context_length
            logger.info(f"Applied context_length from config: {context_length}")

        # Read hicache configuration from scheduler config
        hicache_storage_backend = scheduler_config.get("hicache_storage_backend")
        if hicache_storage_backend is not None:
            logger.info(f"HiCache storage backend enabled: {hicache_storage_backend}")
        else:
            logger.info("HiCache storage backend not configured (memory/disk prefix cache disabled)")

        hicache_storage_prefetch_policy = scheduler_config.get(
            "hicache_storage_prefetch_policy", "best_effort"
        )
        logger.info(f"HiCache prefetch policy: {hicache_storage_prefetch_policy}")

        enable_hierarchical_cache = scheduler_config.get("enable_hierarchical_cache", False)
        if enable_hierarchical_cache:
            logger.info("Hierarchical cache enabled (HiRadixCache will be used for L1)")
        else:
            logger.info("Hierarchical cache disabled (standard RadixCache will be used)")

        sched_config = SchedulerConfig(
            model=model,
            max_prefill_tokens=internal_config.max_prefill_tokens,
            chunked_prefill_size=internal_config.chunked_prefill_size,
            mem_fraction_static=mem_fraction_static,  # Fix: Use validated value instead of internal_config.mem_fraction_static
            tp_size=tp_size,
            ep_size=ep_size,
            dp_size=dp_size,
            # TODO: initialize with the runtime data type.
            data_type=dtype,
            kv_cache_data_type=kv_cache_dtype,
            page_size=internal_config.page_size,
            backend_name=backend,
            backend_version=scheduler_config.get("backend_version"),
            context_length=context_length,
            hicache_storage_backend=hicache_storage_backend,
            hicache_storage_prefetch_policy=hicache_storage_prefetch_policy,
        )
        return sched_config

    @classmethod
    def _parse_server_args(cls, server_args: dict, backend: str) -> SchedulerConfig:
        if backend == "sglang":
            return SchedulerConfig(
                model=None,
                tp_size=server_args.get("tp_size", 1),
                ep_size=server_args.get("ep_size", 1),
                dp_size=server_args.get("dp_size", 1),
                max_prefill_tokens=server_args.get("max_prefill_tokens"),
                chunked_prefill_size=server_args.get("chunked_prefill_size"),
                mem_fraction_static=server_args.get("mem_fraction_static"),
                page_size=server_args.get("page_size"),
                backend_name="sglang",
            )
        else:
            raise RuntimeError(f"Unsupported backend[{backend}] server args parser.")

    @classmethod
    def get_inference_time_predictor(
        cls, model: ModelInfo, hw: AcceleratorInfo, sched_config: SchedulerConfig
    ) -> InferTimePredictor:
        with open(Envs.config_path()) as f:
            config: dict = json.load(f)
        predictor_config = config.get("predictor", {})
        if predictor_config.get("name") == "aiconfigurator":
            device_name = predictor_config.get("device_name")
            hw.name = device_name
            database_mode = predictor_config.get("database_mode", "SILICON")
            prefill_scale_factor = predictor_config.get("prefill_scale_factor", 1)
            decode_scale_factor = predictor_config.get("decode_scale_factor", 1)
            xgb_model_path = predictor_config.get("xgb_model_path", None)

            try:
                return AIConfiguratorTimePredictor(
                    model,
                    hw=hw,
                    config=sched_config,
                    database_path=predictor_config.get("database_path"),
                    database_mode=database_mode,
                    prefill_scale_factor=prefill_scale_factor,
                    decode_scale_factor=decode_scale_factor,
                    xgb_model_path=xgb_model_path,
                )
            except Exception as aiconfig_error:
                logger.warning(f"AIConfigurator initialization failed: {aiconfig_error}")
                if HAS_SIMPLE_PREDICTOR:
                    logger.info("Falling back to SimpleTimePredictor for estimation")
                    return SimpleTimePredictor(model, hw, sched_config)
                raise ValueError(f"AIConfigurator initialization failed and SimpleTimePredictor not available: {aiconfig_error}")
        elif predictor_config.get("name") == "inference_predictor":
            inference_predictor_root = predictor_config.get("inference_predictor_root")
            if not inference_predictor_root:
                raise ValueError("inference_predictor_root must be specified")
            
            task_test_root = predictor_config.get("task_test_root")
            model_config_path = predictor_config.get("model_config")
            hardware_config_path = predictor_config.get("hardware_config")
            quant_config_path = predictor_config.get("quant_config")
            offload_config_path = predictor_config.get("offload_config")
            runtime_config = config.get("runtime", {})
            
            return InferencePredictorTimePredictor(
                model=model,
                hw=hw,
                config=sched_config,
                inference_predictor_root=inference_predictor_root,
                task_test_root=task_test_root,
                model_config_path=model_config_path,
                hardware_config_path=hardware_config_path,
                quant_config_path=quant_config_path,
                offload_config_path=offload_config_path,
                runtime_config=runtime_config,
            )
        elif predictor_config.get("name") == "simple":
            if HAS_SIMPLE_PREDICTOR:
                logger.info("Using SimpleTimePredictor as specified in config")
                return SimpleTimePredictor(model, hw, sched_config)
            raise ValueError("SimpleTimePredictor not available")
        else:
            raise ValueError(f"Unknown predictor name: {predictor_config.get('name')}")