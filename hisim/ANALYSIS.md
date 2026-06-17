# HiSim 实现深度分析

## 一、整体架构

HiSim 的核心思想是：**不修改 SGLang 源码，通过 Python 的元编程机制（monkey-patching）在运行时拦截并替换 SGLang 的关键类和方法**，将真正的 GPU 推理替换为基于性能模型的延迟预测，从而在 CPU 上模拟完整的 LLM 推理调度流程。

整体数据流：

```
Trace/Dataset → SGLangBenchmarkRunner → Engine (被 Hook 的 SGLang)
                                          ↓
                                    TokenizerManager (Hook: 传递时间戳)
                                          ↓
                                    Scheduler (Hook: 虚拟时钟/请求调度)
                                          ↓
                                    ModelRunner (Hook: Mock模型/内存池)
                                          ↓
                                    HiRadixCache (Hook: Mock存储/带宽模拟)
                                          ↓
                                    HiCacheController (Hook: 同步式IO模拟)
                                          ↓
                                    StorageBackend (Hook: Mock存储后端)
                                          ↓
                               InferTimePredictor (延迟预测) → StateManager (虚拟时钟推进)
                                          ↓
                              metrics.json / request.jsonl / iteration.jsonl
```

---

## 二、依赖组件清单

### 2.1 pyproject.toml 中声明的依赖（需安装）

| 依赖 | 版本/来源 | 说明 | 是否需手动安装 |
|------|-----------|------|---------------|
| `numpy` | pip | 数值计算 | 正常 pip install |
| `scikit-learn` | pip | 机器学习工具 | 正常 pip install |
| `xgboost` | pip | XGBoost decode 延迟校正模型 | 正常 pip install |
| `aiconfigurator` | `git+https://github.com/ai-dynamo/aiconfigurator.git@h20e-higher-acc` | AIConfigurator 性能模型SDK，核心推理延迟预测器 | **需从特定 GitHub 分支安装** |

### 2.2 代码中 import 的外部依赖（隐式依赖，pyproject.toml 未声明）

| 依赖 | import 位置 | 说明 | 是否需手动安装 |
|------|-------------|------|---------------|
| `sglang` | `sglang_hook.py`, `sglang_bench.py`, `launch_server.py` | 被模拟的推理框架本体 | **必须安装，且版本需匹配 COMPATIBLE_VERSIONS** |
| `torch` | `sglang_mock_class.py`, `sglang_bench.py` | PyTorch（SGLang 自带） | 随 SGLang 安装 |
| `transformers` | `sglang_bench.py`, `base_dataset.py` | HuggingFace tokenizer | 随 SGLang 安装 |
| `psutil` | `sglang_mock_class.py` | 主机内存检查 | 需单独安装 |
| `packaging` | `version.py` | 版本号解析 | Python 自带 |
| `requests` | `model/base.py` | 从 HuggingFace/ModelScope 拉取模型配置 | 需单独安装 |

### 2.3 bench_serving.py 的额外依赖

| 依赖 | 说明 | 是否需手动安装 |
|------|------|---------------|
| `aiohttp` | 异步HTTP客户端 | 需单独安装 |
| `pybase64` | Base64编解码 | 需单独安装 |
| `datasets` | HuggingFace datasets | 需单独安装 |
| `PIL` (Pillow) | 图像处理 | 需单独安装 |
| `tqdm` | 进度条 | 需单独安装 |

### 2.4 可选依赖

| 依赖 | import 位置 | 说明 | 是否需手动安装 |
|------|-------------|------|---------------|
| `kvcm_py_optimizer` | `sglang_mock_class.py` | KVCM 优化器的 pybind 模块，用于 MockHiCacheStorage 的 KVCM 后端 | **需编译整个 C++ 项目获得**，`try/except` 保护，不可用则回退到简单 set 实现 |
| `termplotlib` + `gnuplot` | `bench_serving.py` | 终端绘图 | 可选 |

### 2.5 环境变量依赖

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `HISIM_CONFIG_PATH` | 无（必填） | 模拟配置 JSON 路径，含平台/模型/预测器配置 |
| `HISIM_OUTPUT_DIR` | `/tmp/hisim/output/` | 指标输出目录 |
| `HISIM_SIMULATION_MODE` | `OFFLINE` | 模拟模式：`BLOCKING`（实时 sleep）或 `OFFLINE`（虚拟时钟） |
| `HISIM_NUM_WARMUP` | `0` | 预热请求数 |
| `HISIM_RESET_HICACHE_STORAGE` | `0` | 是否重置 HiCache 存储 |
| `HISIM_BENCHMARK_OUT_DIR` | `cwd` | benchmark collection 模式的输出目录 |
| `HF_ENDPOINT` | `https://huggingface.co/` | HuggingFace 镜像端点 |

---

## 三、对 SGLang 的修改——Hook 机制详解

HiSim 通过**两种 Hook 注入机制**拦截 SGLang，均基于 Python 元编程：

### 3.1 Hook 框架

**类级 Hook**（`class_hook_entry.py`）：
- 替换 `builtins.__build_class__`，在 Python 解释器创建类对象时拦截
- 当类名匹配 `HOOK_CLASS_NAME` 且模块名匹配 `HOOK_MODULE_NAME` 时，调用 `hook.hook(target_class)` 修改类
- 类必须在 hook 安装**之后**定义才生效（SGLang 的类都是后 import 的，所以能拦截到）

**模块级 Hook**（`module_hook_entry.py`）：
- 注册自定义 `MetaPathFinder` 到 `sys.meta_path`
- 当 `import` 匹配 `HOOK_MODULE_NAME` 的模块时，先加载原始模块代码，再对其执行 hook
- 用于 hook 模块级函数（而非类方法）

### 3.2 具体拦截的 SGLang 模块和类

| Hook 类 | 目标 SGLang 类 | 目标模块 | 拦截方式 |
|---------|---------------|----------|---------|
| `C_EngineHook` | `Engine` | `sglang.srt.entrypoints.engine` | 类级 |
| `C_TokenizerManagerHook` | `TokenizerManager` | `sglang.srt.managers.tokenizer_manager` | 类级 |
| `C_ModelRunnerHook` | `ModelRunner` | `sglang.srt.model_executor.model_runner` | 类级 |
| `C_HiCacheController` | `HiCacheController` | `sglang.srt.managers.cache_controller` | 类级 |
| `C_HiRadixCacheHook` | `HiRadixCache` | `sglang.srt.mem_cache.hiradix_cache` | 类级 |
| `C_StorageBackendFactory` | `StorageBackendFactory` | `sglang.srt.mem_cache.storage.backend_factory` | 类级 |
| `C_SchedulerHook` | `Scheduler` | `sglang.srt.managers.scheduler` | 类级 |
| `M_SGLangKernelLoadUtilHook` | (模块级) | `sgl_kernel.load_utils` | 模块级 |

### 3.3 各 Hook 的修改逻辑与细节

#### (1) `M_SGLangKernelLoadUtilHook` — 绕过 CUDA kernel 加载

**目标模块**：`sgl_kernel.load_utils`
**条件**：仅当 `torch.cuda.is_available() == False`（CPU 平台）时安装
**修改**：将 `_load_architecture_specific_ops` 替换为空函数 `pass`
**原因**：CPU 环境无法加载 sgl_kernel 的 CUDA 共享库，会抛 `ImportError`，必须绕过

#### (2) `C_EngineHook` — 添加 HiCache 清理接口

**目标类**：`sglang.srt.entrypoints.engine.Engine`
**修改**：添加 `clear_hicache_storage()` 方法，通过 `loop.run_until_complete` 调用 `tokenizer_manager.clear_hicache_storage()`
**原因**：模拟运行时需要能在两次 benchmark 之间清除存储后端的缓存

#### (3) `C_TokenizerManagerHook` — 传递请求创建时间戳

**目标类**：`sglang.srt.managers.tokenizer_manager.TokenizerManager`
**修改**：包装 `_send_one_request`，在 `BLOCKING` 模式下将 `created_time` 写入 `sampling_params.custom_params["simulation"]["server_created_time"]`
**原因**：原始 SGLang 不将请求到达服务端的时间传递给 Scheduler。HiSim 需要精确的时间戳来计算排队延迟（TTFT 中的 queue 时间）

#### (4) `C_ModelRunnerHook` — 核心拦截：替换模型、内存池、forward

**目标类**：`sglang.srt.model_executor.model_runner.ModelRunner`
**这是最关键的 Hook**，替换了四个核心方法：

##### a) `initialize` — 替换模型初始化和内存池创建

- 将 `self.model` 替换为 `MockModel`（只有空 `forward()`）
- 不加载任何权重，不分配真实 KV cache
- 创建 `MockReqToTokenPool` 代替真实 `ReqToTokenPool`
  - `max_context_len` 强制设为 1（极大减少内存分配）
  - alloc/free 逻辑保留但简化
- 创建 `MockTokenToKVPool` 代替真实 `MHATokenToKVPool`
  - **head_num 和 head_dim 强制设为 1**（原始可能是 128/96），极大减少 GPU 内存
  - 保留完整的 buffer 结构和接口（`get_key_buffer`, `get_value_buffer`, `get_cpu_copy`, `load_cpu_copy` 等）
  - `set_kv_buffer` 直接返回 None（不写入 KV 数据）
- 创建 `MockTokenToKVPoolAllocator` 或 `MockPagedTokenToKVPoolAllocator`（取决于 page_size）
  - alloc/free 逻辑与 SGLang 一致，但不做排序优化
  - Paged 版本实现了 `alloc_extend` / `alloc_decode` 的 CPU 版本（替代 Triton kernel）
- 用 `estimate_kv_cache_pool_capacity()` 估算 `max_total_num_tokens`（基于 AIConfigurator 性能模型计算权重占用和剩余内存）
- `attn_backend` 设为 None
- `weight_load_mem_usage` 设为 10（原始值远大于此）
- `graph_mem_usage` 设为 0

##### b) `forward` — 替换模型前向计算

通过 `VersionDispatcher` 注册两个版本：
- **v0.5.6~0.5.6.post2**：返回 `LogitsProcessorOutput(next_token_logits=torch.empty(...))` + `False`
- **v0.5.7~0.5.9**：返回 `ModelRunnerOutput(logits_output=..., can_run_graph=False, ...)`

两者都只创建空 tensor，**不做任何计算**。版本分派是因为 0.5.7+ 返回类型从 tuple 变成了 `ModelRunnerOutput`。

##### c) `sample` — 替换采样逻辑

返回全 1 的 token ids（`torch.ones`），跳过真实采样。

##### d) `compute_logprobs_only` — 替换 logprobs 计算

直接返回 None。

#### (5) `C_HiCacheController` — 同步化异步 IO 操作

**目标类**：`sglang.srt.managers.cache_controller.HiCacheController`

原始 SGLang 的 `HiCacheController` 有两个异步线程：
- `backup_thread_func`：后台写入存储
- `prefetch_thread_func`：后台预取数据

**HiSim 的修改**：
- **`backup_thread_func` → 空函数**（异步线程不工作）
- **`prefetch_thread_func` → 空函数**（异步线程不工作）
- 新增 **`handle_backup_operation()`**：同步地从 `backup_queue` 取出操作，调用 `_page_backup`，然后放入 `ack_backup_queue`
- 新增 **`handle_prefetch_operation()`**：同步地处理预取，关键逻辑：
  1. 计算当前推理剩余时间 `remain_dur = StateManager.get_current_inference_dur()`
  2. 处理未完成的分块预取（`chunked_prefetch_operation`）
  3. 根据磁盘带宽计算预取完成量：`calc_prefetch_pages(required_pages, page_size_byte, max_dur, bandwidth)`
  4. 如果预取未在推理时间内完成，保存状态下次继续
  5. 更新请求级别的 `prefetch_complete_tokens`
- 替换 **`_generic_page_set`**：确保总是传递 `extra_info` 给 `storage_backend`

**核心逻辑**：将真实 GPU DMA 传输替换为基于带宽模型的虚拟时间计算。预取能否完成取决于当前推理迭代的剩余时间与所需传输时间的比较。

#### (6) `C_HiRadixCacheHook` — 替换主机内存池和事件处理

**目标类**：`sglang.srt.mem_cache.hiradix_cache.HiRadixCache`

##### a) `__init__` — 替换初始化

- 将 `token_to_kv_pool_host` 替换为 `MockTokenToKVPoolHost`
  - `pin_memory=False`（原始为 True，因为不需要真实 GPU pinned memory）
  - 保留完整的内存分配（仍然分配 CPU tensor），因为需要计算带宽模型所需的大小
  - `load_to_device_per_layer` 和 `backup_from_device_all_layer` 被替换为带宽模型计算
- 处理 `hicache_io_backend == "direct"` 时的 layout 修正（`page_first` → `page_first_direct`）
- 调用 `target.__mro__[1].__init__` 完成父类初始化

##### b) `check_hicache_events` — 同步化事件检查

- 在原始逻辑前插入 `handle_backup_operation()` 和 `handle_prefetch_operation()`
- 确保每次调度循环中同步执行 IO 操作

##### c) `reset` — 添加备份处理

- 在 reset 前调用 `cache_controller.handle_backup_operation()`

#### (7) `C_StorageBackendFactory` — 替换存储后端

**目标类**：`sglang.srt.mem_cache.storage.backend_factory.StorageBackendFactory`
**修改**：将 `create_backend` 替换为返回 `MockHiCacheStorage()`

**`MockHiCacheStorage` 的实现**：
- 两种后端模式：
  - **KVCM 模式**（当 `kvcm_py_optimizer` 可用时）：调用 KVCM 优化器的 `WriteCache` / `GetCacheLocation` / `ClearAllCaches`
  - **简单 set 模式**（默认）：用 Python `set` + 文件持久化模拟存储存在性
- `batch_exists()` 返回连续命中的 key 数量（前缀匹配语义）
- KVCM 模式下通过 `_pass_str_to_block_ids` 将 hash key 转为整数 block ID

#### (8) `C_SchedulerHook` — 核心拦截：虚拟时钟、延迟注入、指标采集

**目标类**：`sglang.srt.managers.scheduler.Scheduler`
**这是最复杂的 Hook**，替换了 7 个方法：

##### a) `__init__` — 禁用 overlap schedule + 初始化预测器

- 强制 `disable_overlap_schedule = True`（简化模拟）
- 从 `ConfigManager` 获取模型信息、硬件信息、调度器配置
- 初始化 `InferTimePredictor`（AIConfigurator）

##### b) `recv_requests` — 请求接收与时间戳注入

两种模式：
- **BLOCKING 模式**：直接透传，但记录 `server_created_time`、`queue_start` 等时间戳
- **OFFLINE 模式**：
  1. 首次收到请求时，将所有 `TokenizedGenerateReqInput` 加入 `FUTURE_QUEUE`（最小堆，按 created_time 排序）
  2. 等待全部请求到齐（通过 `total_request` 参数判断）
  3. 此后每轮调度只从 `FUTURE_QUEUE` 中弹出 `created_time <= global_clock` 的请求
  4. 实现了请求按时间到达的仿真

##### c) `get_new_batch_prefill` — 调度前缀填充批次

- 记录 `queue_end` 时间戳（请求被选中执行的时刻）
- 记录 `final_reused_tokens`（prefix cache 命中 token 数）
- 当无新批次且无运行请求时，步进全局时钟 5ms（模拟等待）
- OFFLINE 模式下，当无运行请求且 FUTURE_QUEUE 非空时，快进全局时钟到下一个请求时间

##### d) `run_batch` — 注入虚拟推理延迟

这是**延迟模拟的核心**：
1. 执行原始 `run_batch`（调度逻辑仍走 SGLang 原生代码）
2. 从 batch 中提取 `input_length` 和 `past_kv_length`，构造 `HisimScheduleBatch`
3. 调用 `InferTimePredictor.predict_infer_time()` 获取预测延迟
4. BLOCKING 模式：`time.sleep(abs(predicted_latency))`，用真实 sleep 模拟
5. OFFLINE 模式：直接将 `predicted_latency` 作为 `forward_latency`
6. 更新 `StateManager.set_current_inference_dur()`

##### e) `process_batch_result` — 时间推进与指标记录

1. 获取 HiCache L2 load/backup 延迟
2. 根据是否 overlap schedule 计算全局时钟推进：
   - 非 overlap：`global_clock += l2_load_dur + inference_dur + l2_backup_dur`
   - overlap：`global_clock += max(l2_load_dur - last_inference_dur, 0) + inference_dur`，backup 与下一次推理 overlap
3. 为每个请求记录 `gen_token_latencies`（每个 token 的延迟）
4. 收集迭代级统计

##### f) `profile` — 指标输出

- 采集所有请求的 `RequestStats`，去除 warmup
- 对齐时间戳（减去最小 created_time）
- 计算 TTFT/TPOT/ITL/e2e/吞吐等指标
- 输出到 `metrics.json`、`iteration.jsonl`、`request.jsonl`
- 重置所有状态

##### g) `event_loop_overlap` — 禁用 overlap 调度循环

直接降级为 `event_loop_normal`，简化模拟。

### 3.4 `MockTokenToKVPoolHost` — 主机端 KV 缓存模拟

**替换了真实的 GPU-CUDA pinned memory 传输**：

- `load_to_device_per_layer()`：
  - 分析 host/device indices 的连续段
  - 根据 KV cache 每层字节数 × 段长度计算传输字节数
  - 使用带宽模型：`bandwidth = x * bw / (t0 * bw + x)`（考虑了启动开销）
  - 将传输时间累加到 `StateManager._hicache_l2_load_dur`

- `backup_from_device_all_layer()`：
  - 类似逻辑，计算 D2H 传输时间
  - 累加到 `StateManager._hicache_l2_backup_dur`

### 3.5 版本兼容（`VersionDispatcher`）

支持的 SGLang 版本：`0.5.6`, `0.5.6.post1`, `0.5.6.post2`, `0.5.7`, `0.5.8`, `0.5.8.post1`, `0.5.9`

版本差异主要体现在：
- `ModelRunner.forward()` 返回类型不同（v0.5.6 返回 tuple，v0.5.7+ 返回 `ModelRunnerOutput`）
- `ReqToTokenPool.alloc/free` 签名不同（v0.5.9 支持chunked request的 `req_pool_idx`）
- `HiRadixCache.__init__` 中 v0.5.9 新增 `prefetch_loaded_tokens_by_reqid`

如果当前版本未精确匹配，会 fallback 到最新注册版本。

---

## 四、模拟时钟机制

HiSim 的核心是 **`StateManager` 维护的虚拟全局时钟**：

```
BLOCKING 模式：
  global_clock ≈ real_time (通过 time.sleep 对齐)
  用于与真实 GPU 对比验证

OFFLINE 模式：
  global_clock 完全虚拟，由以下事件推进：
  - 推理迭代完成：global_clock += predicted_latency
  - H2D 传输：global_clock += hicache_l2_load_dur
  - D2H 传输：global_clock += hicache_l2_backup_dur
  - 等待请求：global_clock = next_request_created_time
  - 空闲等待：global_clock += 0.005s
```

OFFLINE 模式的关键优势：**模拟速度不依赖真实推理时间**，可以在秒级完成数千请求的模拟。

---

## 五、Benchmark Collection Hook（额外）

`hisim/benchmark/collection/serving_hook/sglang_hook.py` 提供了另一套 Hook，用于从**真实 GPU 运行**中采集请求数据（非模拟用途）：

- `C_TokenizerManagerHook`：记录请求创建时间
- `C_SglangSchedulerInfoHook`：记录请求排队时间、prefix cache 命中长度、input/output ids
- `wrapped_profile`：将采集数据输出到 `.requests.jsonl`，作为 HiSim 模拟的 trace 输入

---

## 六、依赖安装总结

**必须安装**（pyproject.toml 已声明）：
- numpy, scikit-learn, xgboost
- aiconfigurator（从特定 GitHub 分支）

**必须安装**（隐式依赖）：
- **sglang**（版本需在 COMPATIBLE_VERSIONS 列表中：0.5.6 ~ 0.5.9）
- torch（随 sglang 安装）
- transformers（随 sglang 安装）
- psutil

**bench_serving.py 额外需要**：
- aiohttp, pybase64, datasets, Pillow, tqdm

**可选**：
- kvcm_py_optimizer（需编译 C++ 项目，MockHiCacheStorage 回退到 set 实现）
- termplotlib + gnuplot
