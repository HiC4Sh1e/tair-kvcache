# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Tair KVCache is Alibaba Cloud's distributed KVCache system for LLM inference. The repository contains two major subsystems:

- **KVCache Manager (KVCM)**: A centralized metadata management service for KVCache, written in C++ with Python connectors for inference engines. Deployed as a standalone server providing HTTP/gRPC APIs.
- **HiSim**: A CPU-based LLM inference simulation system, written in Python. Predicts TTFT/TPOT/throughput by replaying traces without GPU resources.

## Build System

Bazel is the build system. Use `bazelisk` (preinstalled in dev images).

```bash
# Build and run the manager server
bazelisk run //kv_cache_manager:main

# Build with CUDA support
bazelisk build //kv_cache_manager/... --config=cuda12

# Build client with CUDA
bazelisk build //kv_cache_manager/client/... --config=client_with_cuda
```

### Bazel Config Flags

| Config | Purpose |
|--------|---------|
| `--config=cuda12` | CUDA 12.x support |
| `--config=mooncake` | Mooncake storage backend with CUDA |
| `--config=hf3fs` | HF3FS storage backend with CUDA |
| `--config=client` | C++ client build (enables Mooncake/HF3FS/TairMemPool) |
| `--config=client_with_cuda` | Client with CUDA + peer memory |
| `--config=debug` | Debug build (-O0, -g) |
| `--config=asan` | AddressSanitizer build |
| `--config=py3` / `py310` / `py311` / `py312` | Python version selection |

## Testing

```bash
# All unit tests
bazelisk test //kv_cache_manager/...

# Integration tests
bazelisk test //integration_test/...

# C++ client tests
bazelisk test //kv_cache_manager/client/... --config=client

# Single test target
bazelisk test //kv_cache_manager/manager/test:cache_manager_test

# Redis-dependent tests (requires local Redis/Valkey)
bazelisk test //kv_cache_manager/common/test:redis_client_real_service_test --test_tag_filters=redis

# With ASAN
bazelisk test //kv_cache_manager/... --config=debug --config=asan --test_env ASAN_OPTIONS=detect_odr_violation=0

# Force re-run (ignore cached results)
bazelisk test //kv_cache_manager/... --cache_test_results=no
```

HiSim uses standard Python testing:
```bash
cd hisim && pip install -e . && python -m pytest test/
```

## Architecture: KVCache Manager

### Control Plane Hierarchy
- **Storage**: A storage system instance (NFS, HF3FS, Mooncake, TairMemPool, Vineyard). Shared across Instance Groups.
- **Instance Group**: Quota boundary. All Instances in a group share storage quota. Configures which Storages are available.
- **Instance**: A single KVCache instance. **KVCache reuse only occurs within the same Instance — cross-Instance never matches.** One model config (dtype, block_size) bound to one Instance.

### Data Plane Concepts
- **Block**: Fixed-length token sequence with prefix dependency. Same tokens with different prefixes = different Block.
- **CacheLocation**: A storage position for one Block. State machine: `writing → serving → deleting`.
- **LocationSpec**: A sub-part of a CacheLocation, using URI format. Allows per-spec names for TP/PP/mixed-attention scenarios.

### Key Source Modules (`kv_cache_manager/`)

| Module | Responsibility |
|--------|---------------|
| `service/` | Access layer: HTTP + gRPC servers, service implementations (meta/admin/debug) |
| `manager/` | Core business logic: `CacheManager` (matching, two-phase write), `CacheReclaimer` (eviction), `MetaSearcher` (prefix/KV matching), `DataStorageSelector` |
| `meta/` | Metadata index: `MetaIndexer` (radix tree), storage backends (local, Redis, dummy), `MPSCWriteQueue` |
| `data_storage/` | Storage backend abstraction: `DataStorageBackend` interface with implementations (NFS, HF3FS, Mooncake, Vineyard, dummy) |
| `config/` | Configuration management: `RegistryManager`, instance/storage/group config, `LeaderElector` for HA |
| `metrics/` | Observability: `MetricsCollector`, `PrometheusExporter`, reporters (kmonitor, local, logging) |
| `event/` | Event publishing system for CacheLocation lifecycle events |
| `optimizer/` | Trace replay simulation: eviction policies (LRU, TTL, leaf-aware LRU), radix tree index, `OptimizerRunner` |
| `client/` | C++ SDK: `ManagerClient`, `MetaClient`, `TransferClient` (pybind). Storage SDKs (HF3FS, Mooncake, local file) |
| `py_connector/` | Python connectors for inference engines: SGLang, vLLM, TRT-LLM, RTP-LLM |
| `protocol/protobuf/` | Protobuf definitions: `meta_service.proto`, `admin_service.proto`, `debug_service.proto` |

### Request Flow
1. Inference engine calls Python connector (e.g., `py_connector/sglang/connector.py`)
2. Connector calls C++ client via pybind (`client/pybind/`) or HTTP/gRPC
3. Client communicates with Manager server (`service/`)
4. Server dispatches to `CacheManager` which uses `MetaSearcher` for matching and `DataStorageManager` for storage selection
5. Data transfer happens via storage SDKs (Mooncake/HF3FS/NFS)

## Architecture: HiSim

HiSim (`hisim/`) is a standalone Python package using dynamic interception to mock SGLang's inference framework:
- `src/hisim/simulation/sglang/` — SGLang framework hooks
- `src/hisim/simulation/manager/` — Simulation manager
- `src/hisim/time_predictor/` — Time prediction (AIConfigurator backend)
- `src/hisim/hook/` — Framework interception hooks
- `src/hisim/dataset/` — Trace data handling
- `src/hisim/spec/` — Hardware/model specifications

## Coding Conventions

- C++ code follows Google style with 4-space indent (see `.clang-format`, based on `BasedOnStyle: Google`)
- Pre-commit hooks auto-format: `clang-format` for C++/H, `buildifier` for BUILD/.bzl, `autopep8` for Python
- Commit messages must start with `[module_name]` and be at least 15 bytes (e.g., `[manager] fix prefix matching bug`)
- C++17 standard (`--cxxopt="-std=c++17"`)
- Build with `-Werror` — all warnings are errors

## Dev Image

Manager dev image: `ghcr.io/alibaba/tair-kvcache-kvcm-dev:latest` (Dockerfile at `open_source/docker/Dockerfile.dev`)

## Key Constraints from AGENTS.md

- **Instance isolation**: KVCache is only reused within the same `instance_id`. Cross-Instance never matches. Violating this leads to incorrect cache behavior.

## Debugging Integration Tests

Manager logs are in bazel runfiles under `integration_test/<test_method_name>/worker_0/logs/`. TransferClient logs at `<runfiles>/logs/kv_cache_manager_client.log`. Control log level via `--test_env=KVCM_LOG_LEVEL=DEBUG`.

## Proto Changes

When modifying `.proto` files in `kv_cache_manager/protocol/protobuf/`, run the proto generation script. See `docs/develop/proto_modification_guide.md` for details.