from dataclasses import asdict
from collections import defaultdict
import asyncio
import json
import os
from typing import Iterator, Optional
import numpy as np
import torch

from hisim.utils.logger import get_logger

import hisim.hook as hisim_hook
from hisim.simulation.sglang import sgl_kernel_hook
from hisim.simulation.sglang import sglang_hook
from hisim.simulation.base.runner import BaseBenchmarkRunner
from hisim.simulation.types import BenchmarkConfig
from hisim.dataset import (
    DatasetArgs,
    get_dataset,
    GenericRequest,
    BaseDataset,
)

# hook the sglang implementation
if not torch.cuda.is_available():
    # CPU Platform
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

HISIM_OUTPUT_DIR = "/tmp/hisim/simulation"
HISIM_METRICS_PATH = f"{HISIM_OUTPUT_DIR}/metrics.json"
os.environ["HISIM_OUTPUT_DIR"] = HISIM_OUTPUT_DIR

if os.getenv("HISIM_SIMULATION_MODE") is None:
    os.environ["HISIM_SIMULATION_MODE"] = "OFFLINE"

# The sglang must be imported after the hook installer
from sglang.srt.entrypoints.engine import Engine  # noqa
from sglang.srt.server_args import ServerArgs  # noqa
from transformers import AutoTokenizer  # noqa

from sglang.version import __version__ as sglang_version  # noqa
from hisim.simulation.sglang.version import COMPATIBLE_VERSIONS  # noqa


logger = get_logger("hisim")


if sglang_version not in COMPATIBLE_VERSIONS:
    logger.warning(
        f"Current SGLang version {sglang_version} is not in the compatible versions "
        f"{COMPATIBLE_VERSIONS}, so errors may occur."
    )


class SGLangBenchmarkRunner(BaseBenchmarkRunner):
    def __init__(self, server_args: ServerArgs):
        # disable some features which is not necessary for simulation.
        server_args.disable_cuda_graph = True
        # server_args.disable_overlap_schedule = True
        # server_args.__post_init__()  # recall `__post_init__` due to some args had been modified.
        self.server_args = server_args
        self.engine = Engine(**asdict(server_args))

        self._tokenizer: AutoTokenizer = None

    def flush_cache(self):
        self.engine.flush_cache()

    def reset_storage_cache(self):
        self.engine.clear_hicache_storage()

    def get_request(
        self,
        dataset: BaseDataset,
        ignore_timestamp: bool = False,
        with_queue_start: bool = False,
        request_rate: float = float("inf"),
    ) -> Iterator[tuple[GenericRequest, dict]]:
        yield_delay = 0
        for req in dataset:
            if ignore_timestamp:
                created_time = yield_delay
                yield_delay += np.random.exponential(1.0 / request_rate)
            else:
                created_time = req.custom_params.get("created_time", 0)

            simulation_params = {
                "total_request": len(dataset),  # include the warmup requests.
                "created_time": created_time,
                "session_id": req.session_id,
                "parent_session_id": req.parent_session_id,
                "cache_control": req.cache_control,
                "session_end": req.session_end,
            }
            if with_queue_start:
                simulation_params["queue_start"] = req.custom_params.get("queue_start")

            yield (req, simulation_params)

    async def async_benchmark(
        self,
        benchmark_config: BenchmarkConfig,
        dataset: Optional[BaseDataset] = None,
        dataset_args: Optional[DatasetArgs] = None,
    ):
        if (dataset is None) == (dataset_args is None):
            raise ValueError(
                "Exactly one of `dataset` or `dataset_args` must be provided."
            )

        if dataset is None and dataset_args is not None:
            if self._tokenizer is None:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.server_args.model_path
                )
            dataset = get_dataset(
                dataset_args,
                tokenizer=self._tokenizer,
            )

        await self.engine.tokenizer_manager.start_profile(profile_prefix="reset")

        if os.path.exists(HISIM_METRICS_PATH):
            with open(HISIM_METRICS_PATH, "w") as f:
                # clear data
                pass

        # Detect session-aware dataset
        has_sessions = len(dataset) > 0 and getattr(dataset[0], 'session_id', None) is not None

        if has_sessions and benchmark_config.max_concurrency is not None:
            await self._async_benchmark_session_aware(benchmark_config, dataset)
        else:
            await self._async_benchmark_default(benchmark_config, dataset)

        # dump result
        await self.engine.tokenizer_manager.start_profile()

        if os.path.exists(HISIM_METRICS_PATH):
            with open(HISIM_METRICS_PATH, "r") as f:
                metrics = json.loads(f.readline())
        else:
            logger.error(
                f"Failed to load metrics from serving backend. The metrics file should be loaded from {HISIM_METRICS_PATH}."
            )
            return None

        return metrics

    async def _async_benchmark_default(self, benchmark_config, dataset):
        """Original benchmark: all requests launched as concurrent tasks."""
        tasks = []
        logger.info(f"Created {len(dataset)} request tasks (default mode).")
        for req, simulation_params in self.get_request(
            dataset,
            ignore_timestamp=benchmark_config.ignore_request_timestamp,
            with_queue_start=benchmark_config.with_queue_start,
            request_rate=benchmark_config.request_rate,
        ):
            task = asyncio.create_task(
                self.engine.async_generate(
                    prompt=req.prompt,
                    input_ids=req.token_ids,
                    sampling_params={
                        "ignore_eos": True,
                        "max_new_tokens": req.output_length,
                        "custom_params": {
                            "simulation": simulation_params
                        },
                    },
                )
            )
            tasks.append(task)

        _ = await asyncio.gather(*tasks)

    async def _async_benchmark_session_aware(self, benchmark_config, dataset):
        """Session-aware concurrent benchmark.

        Each concurrent slot = one session.
        Within a session, requests are sent serially (await each one).
        After session completes, slot picks next available session.
        """
        # 1. Group requests by session_id, preserving order within session
        sessions: dict[str, list[tuple[GenericRequest, dict]]] = defaultdict(list)
        session_order = []  # preserve insertion order
        for req, simulation_params in self.get_request(
            dataset,
            ignore_timestamp=benchmark_config.ignore_request_timestamp,
            with_queue_start=benchmark_config.with_queue_start,
            request_rate=benchmark_config.request_rate,
        ):
            sid = req.session_id
            if sid not in sessions:
                session_order.append(sid)
            sessions[sid].append((req, simulation_params))

        logger.info(
            f"Session-aware mode: {len(dataset)} requests in "
            f"{len(session_order)} sessions, "
            f"max_concurrency={benchmark_config.max_concurrency}"
        )

        # 2. Create a shared session queue
        session_queue = asyncio.Queue()
        for sid in session_order:
            session_queue.put_nowait(sid)

        # 3. Worker coroutines (one per concurrent slot)
        async def session_worker():
            while True:
                try:
                    sid = session_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                # Process all requests in this session serially
                for req, sim_params in sessions[sid]:
                    await self.engine.async_generate(
                        prompt=req.prompt,
                        input_ids=req.token_ids,
                        sampling_params={
                            "ignore_eos": True,
                            "max_new_tokens": req.output_length,
                            "custom_params": {"simulation": sim_params},
                        },
                    )
                # Session complete — worker will pick next session in next iteration

        # 4. Launch workers
        max_concurrency = benchmark_config.max_concurrency
        workers = [asyncio.create_task(session_worker()) for _ in range(max_concurrency)]
        await asyncio.gather(*workers)

    def benchmark(
        self,
        benchmark_config: BenchmarkConfig,
        dataset: Optional[BaseDataset] = None,
        dataset_args: Optional[DatasetArgs] = None,
    ):
        return self.engine.loop.run_until_complete(
            self.async_benchmark(benchmark_config, dataset, dataset_args)
        )

    def get_iteration_stats(self) -> list[dict]:
        data = []
        file_path = f"{HISIM_OUTPUT_DIR}/iteration.jsonl"
        if os.path.exists(file_path):
            with open(file_path) as f:
                line = f.readline()
                while line:
                    data.append(json.loads(line))
                    line = f.readline()
        else:
            logger.error(f"The iteration statistics data({file_path}) does not exist.")
        return data

    def get_request_stats(self) -> list[dict]:
        data = []
        file_path = f"{HISIM_OUTPUT_DIR}/request.jsonl"
        if os.path.exists(file_path):
            with open(file_path) as f:
                line = f.readline()
                while line:
                    data.append(json.loads(line))
                    line = f.readline()
        else:
            logger.error(f"The request statistics data({file_path}) does not exist.")
        return data

    def shutdown(self):
        logger.info("Attempting to shut down the SGLang backend engine.")
        return self.engine.shutdown()
