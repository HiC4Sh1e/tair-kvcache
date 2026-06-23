import os
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
        sglang_hook.C_InvariantCheckerHook,
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
        launch_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
