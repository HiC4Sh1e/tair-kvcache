import os

# Bypass flashinfer version mismatch check for SGLang 0.5.13 compatibility
os.environ['FLASHINFER_DISABLE_VERSION_CHECK'] = '1'

import json
import json
import sys
import argparse
import torch
import hisim.hook as hisim_hook
from hisim.simulation.sglang import sgl_kernel_hook, sglang_hook
from hisim.simulation.sim_args import SimulationArgs
from hisim.utils import get_logger


# hook the sglang implementation
# Inference Predictor 在 CPU 运行，需要无 CUDA 环境下加载 sgl_kernel
# 因此无论 CUDA 是否可用都要安装 sgl_kernel_hook
hisim_hook.install_module_hooks([sgl_kernel_hook.M_SGLangKernelLoadUtilHook])
hisim_hook.install_class_hooks(
    [
        sglang_hook.C_SchedulerHook,
        sglang_hook.C_SchedulerRequestReceiverHook,
        sglang_hook.C_ModelRunnerHook,
        sglang_hook.C_TokenizerManagerHook,
        sglang_hook.C_StorageBackendFactory,
        sglang_hook.C_HiCacheController,
        sglang_hook.C_HiRadixCacheHook,
        sglang_hook.C_RadixCacheFixHook,
        sglang_hook.C_PoolStatsObserverHook,
    ]
)


logger = get_logger("hisim")


# Ref: https://github.com/sgl-project/sglang/blob/v0.5.6.post2/python/sglang/launch_server.py
if __name__ == "__main__":
    from sglang.srt.entrypoints.http_server import launch_server
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.utils import kill_process_tree
    from sglang.version import __version__ as sglang_version
    from hisim.simulation.sglang.version import COMPATIBLE_VERSIONS

    if sglang_version not in COMPATIBLE_VERSIONS:
        logger.warning(
            f"Current SGLang version {sglang_version} is not in the compatible versions "
            f"{COMPATIBLE_VERSIONS}, so errors may occur."
        )

    parser = argparse.ArgumentParser()

    g = parser.add_argument_group("sglang")
    ServerArgs.add_cli_args(g)

    g = parser.add_argument_group("simulation")
    SimulationArgs.add_cli_args(g)

    raw_args = parser.parse_args(sys.argv[1:])
    server_args = ServerArgs.from_cli_args(raw_args)
    simulation_args = SimulationArgs.from_cli_args(raw_args)

    config_path = os.getenv("HISIM_CONFIG_PATH")
    if config_path and os.path.exists(config_path):
        logger.info(f"Using config from {config_path}")
    elif simulation_args.config_path:
        os.environ["HISIM_CONFIG_PATH"] = simulation_args.config_path
        config_path = simulation_args.config_path
    else:
        config_path = "/tmp/hisim/config.json"
        logger.info(f"Export config to {config_path}")
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(simulation_args.to_dict(), f)
        os.environ["HISIM_CONFIG_PATH"] = config_path

    # Apply HiCache configuration from config file to server_args
    # This enables memory and disk prefix cache matching
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        scheduler_config = config.get("scheduler", {})

        # Set enable_hierarchical_cache first (required to use HiCache)
        enable_hierarchical_cache = scheduler_config.get("enable_hierarchical_cache", False)
        if enable_hierarchical_cache:
            setattr(server_args, "enable_hierarchical_cache", True)
            logger.info("Applied enable_hierarchical_cache from config: True")

        # Set hicache_storage_backend if configured
        hicache_storage_backend = scheduler_config.get("hicache_storage_backend")
        if hicache_storage_backend is not None:
            # Require additional HiCache parameters to be set for proper functioning
            hicache_ratio = scheduler_config.get("hicache_ratio")
            hicache_size = scheduler_config.get("hicache_size")
            hicache_io_backend = scheduler_config.get("hicache_io_backend", "direct")
            hicache_write_policy = scheduler_config.get("hicache_write_policy", "write_through")
            hicache_mem_layout = scheduler_config.get("hicache_mem_layout", "layer_first")

            if hicache_ratio is None or hicache_size is None:
                logger.warning(
                    "hicache_storage_backend is set but hicache_ratio or hicache_size is missing. "
                    "Disabling HiCache storage to avoid compatibility issues."
                )
                setattr(server_args, "hicache_storage_backend", None)
            else:
                setattr(server_args, "hicache_storage_backend", hicache_storage_backend)
                setattr(server_args, "hicache_ratio", hicache_ratio)
                setattr(server_args, "hicache_size", hicache_size)
                setattr(server_args, "hicache_io_backend", hicache_io_backend)
                setattr(server_args, "hicache_write_policy", hicache_write_policy)
                setattr(server_args, "hicache_mem_layout", hicache_mem_layout)
                logger.info(f"Applied HiCache configuration: backend={hicache_storage_backend}, ratio={hicache_ratio}, size={hicache_size}")

        # Set hicache_storage_prefetch_policy if configured
        hicache_storage_prefetch_policy = scheduler_config.get("hicache_storage_prefetch_policy")
        if hicache_storage_prefetch_policy is not None:
            setattr(server_args, "hicache_storage_prefetch_policy", hicache_storage_prefetch_policy)
            logger.info(f"Applied hicache_storage_prefetch_policy from config: {hicache_storage_prefetch_policy}")

    # Limit prefill batch size to 1 for per-request cache hit analysis
    setattr(server_args, "prefill_max_requests", 1)
    logger.info("Set prefill_max_requests=1 for per-request cache hit analysis")

    # Auto-register model and hardware from config if using inference_predictor
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        predictor_config = config.get("predictor", {})
        if predictor_config.get("name") == "inference_predictor":
            from hisim.spec import ModelInfo, AcceleratorInfo
            import json as json_lib

            # Register model if not already registered
            model_name = config.get("model", {}).get("name")
            if model_name and not ModelInfo.find_by_model_name(model_name):
                # Try to load model config if available
                model_config_path = predictor_config.get("model_config")
                task_test_root = predictor_config.get("task_test_root")

                # Determine actual config file location
                if model_config_path and (task_test_root or predictor_config.get("inference_predictor_root")):
                    # First try task_test_root (relative path)
                    if task_test_root and not os.path.isabs(model_config_path):
                        model_config_file = os.path.join(task_test_root, model_config_path)
                    else:
                        # Fallback to inference_predictor_root
                        inference_predictor_root = predictor_config.get("inference_predictor_root")
                        model_config_file = os.path.join(inference_predictor_root, model_config_path) if inference_predictor_root else model_config_path

                    if os.path.exists(model_config_file):
                        with open(model_config_file) as m_f:
                            model_data = json_lib.load(m_f)
                        logger.info(f"Auto-registering model: {model_name} from {model_config_file}")
                        ModelInfo.from_dict({
                            'name': model_name,
                            'model_type': 'gpt_oss',
                            'hidden_size': model_data.get("dim", model_data.get("hidden_size", 5120)),
                            'num_attention_heads': model_data.get("n_heads", model_data.get("num_attention_heads", 32)),
                            'num_hidden_layers': model_data.get("n_layers", model_data.get("num_hidden_layers", 32)),
                            'vocab_size': model_data.get("vocab_size", 32000),
                            'intermediate_size': model_data.get("moe_inter_dim", model_data.get("ffn_hidden_size", model_data.get("dim", 5120) * 4)),
                            'num_key_value_heads': model_data.get("n_kv_heads", model_data.get("num_attention_heads", 32)),
                            'max_position_embeddings': model_data.get("original_seq_len", model_data.get("max_pos_len", 32768)),
                            'torch_dtype': 'float16',
                            'layer_types': [],
                        }, save_to_registry=True)
                    else:
                        # Fallback to hardcoded values
                        logger.info(f"Auto-registering model: {model_name} (using hardcoded values)")
                        ModelInfo.from_dict({
                            'name': model_name, 'model_type': 'gpt_oss',
                            'hidden_size': 5120, 'num_attention_heads': 32,
                            'num_hidden_layers': 43, 'vocab_size': 32000,
                            'intermediate_size': 13696, 'num_key_value_heads': 8,
                            'max_position_embeddings': 131072,
                            'torch_dtype': 'float16', 'layer_types': [],
                        }, save_to_registry=True)

            # Register hardware if not already registered
            hw_name = config.get("platform", {}).get("accelerator", {}).get("name")
            if hw_name and not AcceleratorInfo.find_by_hw_name(hw_name):
                # Try to load hardware config if available
                hw_config_path = predictor_config.get("hardware_config")
                task_test_root = predictor_config.get("task_test_root")

                # Determine actual config file location
                if hw_config_path and (task_test_root or predictor_config.get("inference_predictor_root")):
                    # First try task_test_root (relative path)
                    if task_test_root and not os.path.isabs(hw_config_path):
                        hw_config_file = os.path.join(task_test_root, hw_config_path)
                    else:
                        # Fallback to inference_predictor_root
                        inference_predictor_root = predictor_config.get("inference_predictor_root")
                        hw_config_file = os.path.join(inference_predictor_root, hw_config_path) if inference_predictor_root else hw_config_path

                    if os.path.exists(hw_config_file):
                        with open(hw_config_file) as hw_f:
                            hw_data = json_lib.load(hw_f)
                        logger.info(f"Auto-registering hardware: {hw_name} from {hw_config_file}")
                        # Register using from_dict to ensure it's added to registry
                        AcceleratorInfo.from_dict({
                            'name': hw_name,
                            'vendor': 'NVIDIA',
                            'hbm_capacity_gb': hw_data.get("mem_size", 64),
                            'hbm_bandwidth_gb': hw_data.get("mem_bw", 1600),
                            'intra_node_bandwidth_gb': hw_data.get("intra_bw", 1600),
                            'inter_node_bandwidth_gb': hw_data.get("inter_bw", 1600),
                            'device_alias': [hw_name],
                        }, save_to_registry=True)

    # Apply context_length from config file to server_args
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        scheduler_config = config.get("scheduler", {})
        context_length = scheduler_config.get("context_length")
        if context_length is not None:
            server_args.context_length = context_length
            logger.info(f"Applied context_length from config: {context_length}")

    try:
        # 添加简单的HTTP server用于提供统计信息
        from http.server import HTTPServer, BaseHTTPRequestHandler
        import threading
        import json as json_lib

        class CacheStatsHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == '/get_cache_stats':
                    try:
                        # 从Scheduler获取cache统计
                        stats = sglang_hook.C_SchedulerHook.get_current_cache_stats()

                        self.send_response(200)
                        self.send_header('Content-Type', 'application/json')
                        self.end_headers()
                        self.wfile.write(json_lib.dumps(stats).encode())
                    except Exception as e:
                        logger.error(f"Error serving cache stats: {e}")
                        self.send_response(500)
                        self.end_headers()
                        self.wfile.write(json_lib.dumps({'error': str(e)}).encode())
                else:
                    self.send_response(404)
                    self.end_headers()

        # 启动简单的HTTP server用于cache统计查询
        # 复用服务绑定的端口-1（如果可用）或者使用固定端口
        stats_server_port = server_args.port + 10
        try:
            stats_server = HTTPServer(('127.0.0.1', stats_server_port), CacheStatsHandler)
            stats_server_thread = threading.Thread(target=stats_server.serve_forever, daemon=True)
            stats_server_thread.start()
            logger.info(f"Cache stats server started on port {stats_server_port}")
        except Exception as e:
            logger.warning(f"Failed to start cache stats server: {e}")

        # 启动主要的服务
        launch_server(server_args)

    finally:
        kill_process_tree(os.getpid(), include_parent=False)
