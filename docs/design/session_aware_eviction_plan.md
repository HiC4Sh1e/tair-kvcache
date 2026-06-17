# Session-Aware KV Cache Eviction — 修改方案与工作量分析

## 一、需求概述

在 HiSim 仿真请求中新增三个字段：

```python
"session_id": "sub-1",              # str/uuid，客户端传入或自动生成
"parent_session_id": "main-0",      # 父session的id，为空或与session_id相同表示没有父session
"cache_control": {
    "type": "ephemeral",             # 仅支持ephemeral一种类型
    "ttl": 5,                        # 缓存超时时间（分钟），默认5min，支持1h
}
```

**核心语义**：同一个 `session_id` 下的 KV cache blocks，在 TTL 超时之前不会被驱逐和重新分配；TTL 可被刷新，以最后一个 TTL 为准。

---

## 二、现有机制分析

### 2.1 当前请求流

```
用户请求 → Engine.generate(sampling_params=dict)
         → GenerateReqInput (io_struct.py:158)
         → TokenizerManager._create_tokenized_object()
         → TokenizedGenerateReqInput (io_struct.py:659)
         → Scheduler.handle_generate_request()
         → Req (schedule_batch.py:455)
         → RadixCache.match_prefix / cache_finished_req
```

**关键发现**：
- `GenerateReqInput` 已有 `extra_key` 字段（line 234），`TokenizedGenerateReqInput` 也有（line 713）
- `Req.__init__` 也接受 `extra_key`（line 486），但 **`handle_generate_request` 没有传递它**（line 1305-1332），这是一个现有 bug
- `SamplingParams.custom_params`（dict）可透传任意键值，HiSim 当前已利用此通道传递 `simulation` 参数

### 2.2 当前驱逐机制

```
内存不足 → evict_from_tree_cache() (common.py:227)
         → RadixCache.evict(num_tokens) (radix_cache.py:544)
         → _collect_leaves() → 仅收集 lock_ref==0 的叶子节点
         → 按 eviction_strategy.get_priority() 排序（小根堆）
         → 逐个 pop 最低优先级叶子，free() 其 KV cache tokens
         → 若父节点变为无子叶且 lock_ref==0，推入堆继续驱逐
```

**现有驱逐策略**（`evict_policy.py`）：

| 策略 | 优先级计算 |
|------|-----------|
| LRU | `node.last_access_time` |
| LFU | `(node.hit_count, node.last_access_time)` |
| FIFO | `node.creation_time` |
| PriorityStrategy | `(node.priority, node.last_access_time)` |

**关键发现**：
- `TreeNode` 已有 `priority` 字段（line 110），`PriorityStrategy` 已支持优先级驱逐
- `TreeNode` 已有 `lock_ref` 机制：`lock_ref > 0` 的节点不会被 `_collect_leaves()` 收集
- **无 TTL 机制**：没有任何基于时间的自动驱逐
- **无 session 隔离**：现有 `Session` 机制仅用于 continual prompting（请求链式拼接），不影响 KV cache 驱逐
- `RadixKey.extra_key` 可创建命名空间隔离（不同 extra_key 的缓存完全不相交）

### 2.3 HiSim 中的 Hook 点

当前 HiSim 拦截的 SGLang 类：

| Hook | 当前拦截方法 | 与本需求的关系 |
|------|-------------|---------------|
| `C_SchedulerHook` | `recv_requests`, `run_batch`, `process_batch_result`, `profile` | **需扩展**：传递 session_id/cache_control |
| `C_ModelRunnerHook` | `initialize`, `forward`, `sample` | 无关 |
| `C_HiRadixCacheHook` | `__init__`, `check_hicache_events`, `reset` | **需扩展**：TTL 感知的驱逐 |
| `C_HiCacheController` | `prefetch_thread_func`, `backup_thread_func` | 可能涉及：session 感知的 prefetch |
| `C_StorageBackendFactory` | `create_backend` | 可能涉及：session 感知的存储 |

---

## 三、修改方案

### 总体思路

采用 **两层修改** 策略：

1. **SGLang 层**（3rdparty/sglang）：扩展数据模型，新增 session_id/cache_control 字段到请求流；扩展 RadixCache 驱逐逻辑，支持 TTL 和 session 保护
2. **HiSim 层**（hisim/）：扩展 Hook 和 Mock 类，在仿真中注入虚拟时钟驱动的 TTL 驱逐逻辑

### 3.1 SGLang 侧修改

#### 3.1.1 数据模型扩展

**文件**: `3rdparty/sglang/python/sglang/srt/managers/io_struct.py`

```python
# 新增 CacheControl dataclass
@dataclass
class CacheControl:
    type: str = "ephemeral"       # 仅支持 ephemeral
    ttl: int = 5                   # 超时时间（分钟）

# GenerateReqInput 新增字段 (line ~234)
session_id: Optional[Union[List[str], str]] = None
parent_session_id: Optional[Union[List[str], str]] = None
cache_control: Optional[Union[List[Dict], Dict]] = None

# TokenizedGenerateReqInput 新增字段 (line ~713)
session_id: Optional[str] = None
parent_session_id: Optional[str] = None
cache_control: Optional[CacheControl] = None
```

**文件**: `3rdparty/sglang/python/sglang/srt/managers/tokenizer_manager.py`

在 `_create_tokenized_object()` (line ~780) 中传递新字段：
```python
session_id=obj.session_id,
parent_session_id=obj.parent_session_id,
cache_control=obj.cache_control,
```

**文件**: `3rdparty/sglang/python/sglang/srt/managers/scheduler.py`

在 `handle_generate_request()` (line ~1305) 中传递新字段：
```python
req = Req(
    ...
    session_id=recv_req.session_id,           # 新增
    parent_session_id=recv_req.parent_session_id,  # 新增
    cache_control=recv_req.cache_control,     # 新增
    extra_key=recv_req.extra_key,             # 顺便修复 bug
)
```

**文件**: `3rdparty/sglang/python/sglang/srt/managers/schedule_batch.py`

`Req.__init__` 新增参数：
```python
def __init__(self, ..., parent_session_id: Optional[str] = None, cache_control: Optional[CacheControl] = None):
    ...
    self.parent_session_id = parent_session_id
    self.cache_control = cache_control
```

#### 3.1.2 TreeNode 扩展

**文件**: `3rdparty/sglang/python/sglang/srt/mem_cache/radix_cache.py`

```python
class TreeNode:
    def __init__(self, id=None, priority=0):
        ...
        # 新增：session-aware eviction
        self.session_id: Optional[str] = None       # 所属 session
        self.ttl_deadline: Optional[float] = None    # TTL 到期时间戳（monotonic）
```

在 `_insert_helper()` 或 `cache_finished_req()` 中，从 `Req` 传递 session 信息到 TreeNode：

```python
def cache_finished_req(self, req: Req, is_insert=True):
    ...
    # 在插入节点时传播 session_id 和 TTL
    for node in inserted_nodes:
        node.session_id = req.session_id
        if req.cache_control and req.cache_control.type == "ephemeral":
            node.ttl_deadline = time.monotonic() + req.cache_control.ttl * 60
```

#### 3.1.3 驱逐策略扩展

**文件**: `3rdparty/sglang/python/sglang/srt/mem_cache/evict_policy.py`

新增 `SessionAwareStrategy`：

```python
class SessionAwareStrategy(EvictionStrategy):
    """Session-aware eviction with TTL protection.

    Priority order:
    1. TTL 未过期的 session 节点 → 最高优先级（不驱逐）
    2. 普通 LRU 节点 → 正常驱逐
    3. TTL 已过期的 session 节点 → 最低优先级（优先驱逐）
    """

    def get_priority(self, node: "TreeNode") -> Tuple[int, float]:
        now = time.monotonic()

        if node.ttl_deadline is not None:
            if now < node.ttl_deadline:
                # TTL 未过期 → 不可驱逐，返回极高优先级值
                return (2, node.last_access_time)
            else:
                # TTL 已过期 → 最优先驱逐
                return (0, node.last_access_time)
        else:
            # 无 session 保护 → 正常 LRU
            return (1, node.last_access_time)
```

**文件**: `3rdparty/sglang/python/sglang/srt/mem_cache/radix_cache.py`

修改 `evict()` 方法，增加 TTL 过滤：

```python
def evict(self, num_tokens: int):
    if self.disable:
        return

    start_time = time.perf_counter()
    leaves = self._collect_leaves()
    eviction_heap = [
        (self.eviction_strategy.get_priority(node), node) for node in leaves
    ]
    heapq.heapify(eviction_heap)

    num_evicted = 0
    while num_evicted < num_tokens and len(eviction_heap):
        priority, x = heapq.heappop(eviction_heap)

        # Session TTL 保护：跳过未过期的 session 节点
        if x.ttl_deadline is not None and time.monotonic() < x.ttl_deadline:
            continue  # 不驱逐，也不推回堆

        self.token_to_kv_pool_allocator.free(x.value)
        num_evicted += len(x.value)
        self._delete_leaf(x)

        if len(x.parent.children) == 0 and x.parent.lock_ref == 0:
            new_priority = self.eviction_strategy.get_priority(x.parent)
            heapq.heappush(eviction_heap, (new_priority, x.parent))

        self._record_remove_event(x)

    self.update_eviction_metrics(num_evicted, start_time)
```

#### 3.1.4 TTL 刷新机制

**文件**: `3rdparty/sglang/python/sglang/srt/mem_cache/radix_cache.py`

在 `match_prefix()` 中，当请求命中某 session 的缓存时，刷新该路径上所有节点的 TTL：

```python
def match_prefix(self, ...):
    ...
    # 匹配成功后，沿路径刷新 TTL
    if req.session_id is not None and req.cache_control is not None:
        node = last_node
        while node != self.root_node:
            if node.session_id == req.session_id and node.ttl_deadline is not None:
                node.ttl_deadline = time.monotonic() + req.cache_control.ttl * 60
            node = node.parent
```

#### 3.1.5 Entry Point 扩展

**文件**: `3rdparty/sglang/python/sglang/srt/entrypoints/engine.py`

在 `generate()` / `async_generate()` 中新增参数透传：

```python
async def async_generate(
    self,
    prompt,...,
    session_id: Optional[str] = None,
    parent_session_id: Optional[str] = None,
    cache_control: Optional[Dict] = None,
):
```

---

### 3.2 HiSim 侧修改

#### 3.2.1 仿真请求参数扩展

**文件**: `hisim/src/hisim/dataset/base_dataset.py`

`GenericRequest` 新增字段：

```python
@dataclass
class GenericRequest:
    prompt: Optional[str] = None
    token_ids: Optional[list[int]] = None
    input_length: int = -1
    output_length: int = -1
    custom_params: dict = field(default_factory=dict)
    # 新增
    session_id: Optional[str] = None
    parent_session_id: Optional[str] = None
    cache_control: Optional[dict] = None  # {"type": "ephemeral", "ttl": 5}
```

**文件**: `hisim/src/hisim/dataset/hisim_collection.py`

`HisimCollectionDataset._load_dataset()` 解析 trace 中的新字段：

```python
dataset.append(
    GenericRequest(
        ...
        session_id=req.get("session_id"),
        parent_session_id=req.get("parent_session_id"),
        cache_control=req.get("cache_control"),
    )
)
```

**文件**: `hisim/src/hisim/simulation/sglang/sglang_bench.py`

`SGLangBenchmarkRunner.get_request()` 和 `async_benchmark()` 传递新字段到 `sampling_params.custom_params`：

```python
simulation_params = {
    "total_request": len(dataset),
    "created_time": created_time,
    # 新增
    "session_id": req.session_id,
    "parent_session_id": req.parent_session_id,
    "cache_control": req.cache_control,
}
```

#### 3.2.2 SchedulerHook 扩展

**文件**: `hisim/src/hisim/simulation/sglang/sglang_hook.py`

`C_SchedulerHook.wrapped_recv_requests()` 中提取 session 信息并记录：

```python
# 在现有 simulation_args 解析后新增
simulation_args["session_id"] = req.sampling_params.custom_params.get("simulation", {}).get("session_id")
simulation_args["parent_session_id"] = req.sampling_params.custom_params.get("simulation", {}).get("parent_session_id")
simulation_args["cache_control"] = req.sampling_params.custom_params.get("simulation", {}).get("cache_control")

# 将 session 信息注入 req 对象
if hasattr(req, 'session_id'):
    req.session_id = simulation_args.get("session_id")
    req.parent_session_id = simulation_args.get("parent_session_id")
    req.cache_control = simulation_args.get("cache_control")
```

`C_SchedulerHook.wrapped_run_batch()` 中将 session 信息传播到 batch 级别：

```python
# 在构造 HisimScheduleBatch 时传递 session 信息
for req in batch.reqs:
    hisim_batch.reqs.append(
        FakeRequest(
            input_length=...,
            past_kv_length=...,
            session_id=getattr(req, 'session_id', None),
        )
    )
```

#### 3.2.3 虚拟时钟驱动的 TTL 驱逐

**关键区别**：在真实 SGLang 中 TTL 使用 `time.monotonic()`（挂钟时间），但在 HiSim OFFLINE 模式下需要使用虚拟全局时钟 `StateManager.get_global_clock()`。

**方案**：通过 Hook 修改 `RadixCache.evict()` 中的时间判断，用虚拟时钟替代挂钟时间。

**文件**: `hisim/src/hisim/simulation/sglang/sglang_hook.py`

新增 `C_RadixCacheHook`：

```python
class C_RadixCacheHook(BaseHook):
    HOOK_CLASS_NAME = "RadixCache"
    HOOK_MODULE_NAME = "sglang.srt.mem_cache.radix_cache"

    @classmethod
    def hook(cls, target):
        original_evict = target.evict

        def wrapped_evict(self, num_tokens: int):
            """用虚拟时钟驱动的 TTL 检查替代挂钟时间"""
            if self.disable:
                return

            start_time = time.perf_counter()
            leaves = self._collect_leaves()
            eviction_heap = [
                (self.eviction_strategy.get_priority(node), node) for node in leaves
            ]
            heapq.heapify(eviction_heap)

            virtual_now = StateManager.get_global_clock()

            num_evicted = 0
            while num_evicted < num_tokens and len(eviction_heap):
                priority, x = heapq.heappop(eviction_heap)

                # TTL 保护：虚拟时钟未过期则跳过
                if x.ttl_deadline is not None and virtual_now < x.ttl_deadline:
                    continue

                self.token_to_kv_pool_allocator.free(x.value)
                num_evicted += len(x.value)
                self._delete_leaf(x)

                if len(x.parent.children) == 0 and x.parent.lock_ref == 0:
                    new_priority = self.eviction_strategy.get_priority(x.parent)
                    heapq.heappush(eviction_heap, (new_priority, x.parent))

                self._record_remove_event(x)

            self.update_eviction_metrics(num_evicted, start_time)

        target.evict = wrapped_evict
```

**注意**：需要在 SGLang 侧 TreeNode 设置 `ttl_deadline` 时，也使用虚拟时钟。这可以通过在 `cache_finished_req` 的 Hook 中覆盖实现。

**文件**: `hisim/src/hisim/simulation/types.py`

`RequestStats` 新增字段：

```python
@dataclass
class RequestStats:
    ...
    session_id: Optional[str] = None
    parent_session_id: Optional[str] = None
    cache_control: Optional[dict] = None
```

#### 3.2.4 TTL 虚拟时钟适配

核心问题：SGLang 中 `time.monotonic()` 和 HiSim 中 `StateManager.get_global_clock()` 的时间域不同。

**解决方案**：在 HiSim OFFLINE 模式下，所有 TTL deadline 都以虚拟时钟为基准：

- `node.ttl_deadline = StateManager.get_global_clock() + ttl_minutes * 60`
- 驱逐检查时使用 `StateManager.get_global_clock()` 而非 `time.monotonic()`

需要新增一个时间源抽象，让 RadixCache 在不同模式下使用不同时钟：

```python
# 在 SGLang 侧新增
def _get_cache_clock() -> float:
    """默认使用 monotonic 时钟，HiSim 可 Hook 替换为虚拟时钟"""
    return time.monotonic()

# HiSim Hook 替换
def _get_cache_clock_virtual() -> float:
    return StateManager.get_global_clock()

# 在 C_RadixCacheHook 或新的 module hook 中
target._get_cache_clock = staticmethod(_get_cache_clock_virtual)
```

#### 3.2.5 launch_server 和 sglang_bench Hook 注册

**文件**: `hisim/src/hisim/simulation/sglang/launch_server.py`

在 `install_class_hooks` 列表中新增：
```python
hisim_hook.install_class_hooks([
    ...
    sglang_hook.C_RadixCacheHook,  # 新增
])
```

**文件**: `hisim/src/hisim/simulation/sglang/sglang_bench.py`

同样新增注册。

#### 3.2.6 指标采集扩展

**文件**: `hisim/src/hisim/simulation/utils.py`

`calc_metrics()` 新增 session 相关指标：
- `session_count`: 涉及的 session 数量
- `session_protected_evictions`: TTL 保护下被跳过驱逐的次数
- `session_ttl_expired_evictions`: TTL 过期后被驱逐的 token 数

---

### 3.3 Benchmark Collection 扩展

**文件**: `hisim/benchmark/collection/serving_hook/sglang_hook.py`

`RequestInfos` 新增字段，`C_SglangSchedulerInfoHook.wrapped_recv_requests()` 采集 session 信息。

---

## 四、修改文件清单

### SGLang 侧（3rdparty/sglang）

| # | 文件 | 修改内容 | 复杂度 |
|---|------|---------|--------|
| 1 | `srt/managers/io_struct.py` | 新增 `CacheControl` dataclass；`GenerateReqInput` 和 `TokenizedGenerateReqInput` 新增 3 个字段；更新 `__getitem__` 切片逻辑 | 中 |
| 2 | `srt/entrypoints/engine.py` | `generate()` / `async_generate()` 新增参数透传 | 低 |
| 3 | `srt/managers/tokenizer_manager.py` | `_create_tokenized_object()` 传递新字段 | 低 |
| 4 | `srt/managers/scheduler.py` | `handle_generate_request()` 传递新字段 + 修复 extra_key bug | 低 |
| 5 | `srt/managers/schedule_batch.py` | `Req.__init__` 新增 `parent_session_id`, `cache_control` 参数 | 低 |
| 6 | `srt/mem_cache/radix_cache.py` | `TreeNode` 新增 `session_id`, `ttl_deadline`；`evict()` 增加 TTL 过滤；`cache_finished_req()` 传播 session 信息；`match_prefix()` 刷新 TTL；新增 `_get_cache_clock()` 抽象 | 高 |
| 7 | `srt/mem_cache/evict_policy.py` | 新增 `SessionAwareStrategy` | 中 |
| 8 | `srt/mem_cache/hiradix_cache.py` | 可能需要适配 `reset()` 清理 session 元数据 | 低 |

### HiSim 侧（hisim/）

| # | 文件 | 修改内容 | 复杂度 |
|---|------|---------|--------|
| 9 | `src/hisim/dataset/base_dataset.py` | `GenericRequest` 新增 3 个字段 | 低 |
| 10 | `src/hisim/dataset/hisim_collection.py` | 解析 trace 中的新字段 | 低 |
| 11 | `src/hisim/dataset/random.py` | 支持生成带 session 的随机请求 | 低 |
| 12 | `src/hisim/simulation/types.py` | `RequestStats` 新增字段；`SchedulerConfig` 可能需调整 | 低 |
| 13 | `src/hisim/simulation/sglang/sglang_hook.py` | `C_SchedulerHook` 扩展 session 传递；新增 `C_RadixCacheHook`；Hook `_get_cache_clock` | 高 |
| 14 | `src/hisim/simulation/sglang/sglang_bench.py` | `get_request()` / `async_benchmark()` 传递 session 信息；Hook 注册 | 中 |
| 15 | `src/hisim/simulation/sglang/launch_server.py` | Hook 注册 | 低 |
| 16 | `src/hisim/simulation/utils.py` | `calc_metrics()` 新增 session 指标 | 低 |
| 17 | `src/hisim/time_predictor/base.py` | `FakeRequest` 新增 `session_id`（可选） | 低 |
| 18 | `benchmark/collection/serving_hook/sglang_hook.py` | 采集 session 信息 | 低 |

### 测试

| # | 文件 | 修改内容 | 复杂度 |
|---|------|---------|--------|
| 19 | `hisim/test/test_simulation_sglang_runner.py` | 新增 session 感知测试用例 | 中 |
| 20 | SGLang 单元测试 | 新增 `SessionAwareStrategy` 测试、TTL 刷新测试 | 中 |

---

## 五、工作量预估

### 按模块估算

| 模块 | 工作量 | 说明 |
|------|--------|------|
| **SGLang 数据流透传**（#1-5） | **2-3 天** | 主要是字段搬运，模式固定，但需注意 batch 请求的切片逻辑 |
| **SGLang 驱逐机制扩展**（#6-8） | **5-7 天** | 核心复杂度所在：TreeNode 元数据传播、evict 逻辑修改、TTL 刷新、虚拟时钟适配、边界条件（session 内节点部分过期） |
| **HiSim Hook 扩展**（#13-15） | **3-4 天** | 新增 C_RadixCacheHook、虚拟时钟适配、session 信息在 Hook 链中的传递 |
| **HiSim Dataset/Bench 扩展**（#9-12, 16-17） | **1-2 天** | 字段扩展，模式固定 |
| **测试**（#19-20） | **2-3 天** | 覆盖 session 隔离、TTL 过期驱逐、TTL 刷新、跨 session 驱逐优先级 |
| **集成验证 & Debug** | **2-3 天** | 端到端仿真测试，确保指标正确 |

### 总工作量：**15-22 人天**（约 3-4.5 周）

### 风险点

1. **TreeNode session 元数据传播的完整性**：`cache_finished_req()` 插入的节点需要正确传播 session_id，但已有节点（prefix match 共享的）可能属于不同 session。**设计决策**：共享节点（多个 session 命中同一前缀）应如何处理？建议采用"最先写入者拥有"策略，或允许节点同时属于多个 session（用 set 存储）。

2. **TTL 与虚拟时钟的精度**：OFFLINE 模式下虚拟时钟以推理延迟为步长推进，TTL 粒度为分钟级，精度足够。但 BLOCKING 模式下混用 `time.monotonic()` 和虚拟时钟需注意一致性。

3. **TTL 过期后的级联驱逐**：当 session 的 TTL 过期后，其下可能有大量节点变为可驱逐。如果一次性全部驱逐可能导致瞬时内存压力。建议在 `evict()` 中正常逐个驱逐即可（仅改变优先级，不改变驱逐数量）。

4. **parent_session_id 的语义**：当前方案未深入使用 `parent_session_id`。如果需要实现"父 session 过期时子 session 也过期"的级联失效，需要额外的 session 管理器来维护 session 树关系，工作量需额外增加 **3-5 天**。

5. **SGLang 版本兼容**：当前方案基于 0.5.9 版本分析，HiSim 需同步更新 `VersionDispatcher` 的兼容逻辑。
