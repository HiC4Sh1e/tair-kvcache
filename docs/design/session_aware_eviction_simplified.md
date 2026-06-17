# Session-Aware KV Cache Eviction — 简化方案详细任务拆解

## 总览

| 阶段 | 任务数 | 工作量 | 前置依赖 |
|------|--------|--------|---------|
| A. 数据模型与透传 | 5 | 1.5 天 | 无 |
| B. 核心驱逐逻辑 | 4 | 3 天 | A |
| C. 指标采集与输出 | 2 | 0.5 天 | B |
| D. 测试验证 | 3 | 1.5 天 | C |
| **合计** | **14** | **6.5 天** | — |

---

## 阶段 A：数据模型与透传（1.5 天）

### A1. GenericRequest 新增字段

**文件**: `hisim/src/hisim/dataset/base_dataset.py`
**位置**: `GenericRequest` dataclass（line 9-14）
**修改**:

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

**工作量**: 0.5h

---

### A2. HisimCollectionDataset 解析 session 字段

**文件**: `hisim/src/hisim/dataset/hisim_collection.py`
**位置**: `_load_dataset()`（line 28-62），`GenericRequest` 构造处（line 51-59）
**修改**:

```python
dataset.append(
    GenericRequest(
        token_ids=req["input_ids"],
        input_length=len(req["input_ids"]),
        output_length=req["output_length"],
        custom_params={
            "created_time": req[time_field_name] - min_created_ts,
        },
        # 新增
        session_id=req.get("session_id"),
        parent_session_id=req.get("parent_session_id"),
        cache_control=req.get("cache_control"),
    )
)
```

**工作量**: 0.5h

---

### A3. sglang_bench.py 透传 session 字段到 simulation_params

**文件**: `hisim/src/hisim/simulation/sglang/sglang_bench.py`

**修改点 1** — `get_request()`（line 81-103）：将 `GenericRequest` 的 session 字段写入 `simulation_params`

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

**修改点 2** — 无需修改 `async_benchmark()`，因为 `simulation_params` 已在上游构造，直接透传到 `custom_params["simulation"]`。

**工作量**: 0.5h

---

### A4. RequestStats 新增 session 字段

**文件**: `hisim/src/hisim/simulation/types.py`
**位置**: `RequestStats` dataclass（line 51-62）
**修改**:

```python
@dataclass
class RequestStats:
    rid: str = ""
    last_event_time: float = 1.0
    input_length: int = 1
    output_length: int = 1
    final_reused_tokens: int = 0
    prefetch_complete_tokens: int = 0
    queue_start: float = -1
    queue_end: float = -1
    created_time: float = -1
    gen_token_latencies: list[float] = field(default_factory=list)
    # 新增
    session_id: Optional[str] = None
    parent_session_id: Optional[str] = None
```

**工作量**: 0.5h

---

### A5. C_SchedulerHook 传递 session 信息到 Req 和 RequestStats

**文件**: `hisim/src/hisim/simulation/sglang/sglang_hook.py`

**修改点 1** — `wrapped_recv_requests()`（line 722-750 区域）：在现有的 `simulation_args` 解析后，提取 session 字段并写入 `req.extra_key`

```python
# 在 for req in recv_reqs 循环内，line 732 之后新增
simulation_args = req.sampling_params.custom_params["simulation"]

# --- 新增：session 透传 ---
session_id = simulation_args.get("session_id")
if session_id is not None:
    # 通过 extra_key 实现 session 缓存隔离
    # 同一 session_id 共享前缀缓存，不同 session_id 完全隔离
    req.extra_key = session_id

# 将 session 信息写入 RequestStats
req_stats.session_id = session_id
req_stats.parent_session_id = simulation_args.get("parent_session_id")
# --- 新增结束 ---
```

**修改点 2** — `wrapped_get_new_batch_prefill()`（line 758-790 区域）：在请求被调度执行时刷新 session TTL

```python
# 在 for req in new_batch.reqs 循环内（line 762 附近）新增
# 刷新 session TTL（命中缓存时延长时间）
sim_args = req.sampling_params.custom_params.get("simulation", {})
if sim_args.get("session_id") and sim_args.get("cache_control"):
    _update_session_ttl(sim_args["session_id"], sim_args["cache_control"])
```

**修改点 3** — `wrapped_profile()`（line 902-966 区域）：在 `reset` 时清理 `SESSION_TTL_TABLE`

```python
# 在 StateManager.reset() 后新增（line 950 附近）
SESSION_TTL_TABLE.clear()
```

**工作量**: 3h

---

## 阶段 B：核心驱逐逻辑（3 天）

### B1. Session TTL 元数据管理

**文件**: `hisim/src/hisim/simulation/sglang/sglang_hook.py`
**位置**: `C_SchedulerHook` 类定义之前（line 565 之前），新增模块级变量和函数

**新增代码**:

```python
# ====== Session TTL 管理器 ======
# key = session_id, value = TTL 过期时的虚拟时钟值
SESSION_TTL_TABLE: dict[str, float] = {}

def _update_session_ttl(session_id: str, cache_control: dict):
    """更新/刷新 session TTL。以最后一个 TTL 为准。"""
    if session_id is None or cache_control is None:
        return
    if cache_control.get("type") != "ephemeral":
        return
    ttl_minutes = cache_control.get("ttl", 5)
    ttl_minutes = max(0, min(ttl_minutes, 60))  # 限制范围 [0, 60]
    ttl_deadline = StateManager.get_global_clock() + ttl_minutes * 60
    SESSION_TTL_TABLE[session_id] = ttl_deadline
    logger.debug(
        f"Session TTL updated: session_id={session_id}, "
        f"ttl={ttl_minutes}min, deadline={ttl_deadline:.2f}s"
    )

def _is_session_protected(session_id: str) -> bool:
    """检查 session 是否受 TTL 保护（未过期）。"""
    if session_id is None or session_id not in SESSION_TTL_TABLE:
        return False
    if StateManager.get_global_clock() < SESSION_TTL_TABLE[session_id]:
        return True
    # TTL 过期，清理表项
    del SESSION_TTL_TABLE[session_id]
    return False

def _cleanup_expired_sessions():
    """清理所有已过期的 session TTL 表项。"""
    now = StateManager.get_global_clock()
    expired = [sid for sid, deadline in SESSION_TTL_TABLE.items() if now >= deadline]
    for sid in expired:
        del SESSION_TTL_TABLE[sid]
    if expired:
        logger.debug(f"Cleaned up {len(expired)} expired sessions.")
```

**设计说明**：
- `_update_session_ttl()`: 每次调用直接覆盖（语义：以最后一个 TTL 为准）
- `_is_session_protected()`: O(1) 字典查询，惰性删除过期项
- `_cleanup_expired_sessions()`: 批量清理，在每轮调度结束时调用

**工作量**: 2h

---

### B2. C_RadixCacheHook — TTL 保护驱逐

**文件**: `hisim/src/hisim/simulation/sglang/sglang_hook.py`
**位置**: `C_HiRadixCacheHook` 类之后（line 549 之后），新增类

**新增代码**:

```python
class C_RadixCacheHook(BaseHook):
    HOOK_CLASS_NAME = "RadixCache"
    HOOK_MODULE_NAME = "sglang.srt.mem_cache.radix_cache"

    @classmethod
    def hook(cls, target):
        original_evict = target.evict

        def wrapped_evict(self, num_tokens: int):
            """带 session TTL 保护的驱逐：跳过受保护 session 的节点"""
            if self.disable:
                return

            start_time = time.perf_counter()
            leaves = self._collect_leaves()
            eviction_heap = [
                (self.eviction_strategy.get_priority(node), node)
                for node in leaves
            ]
            heapq.heapify(eviction_heap)

            virtual_now = StateManager.get_global_clock()
            num_evicted = 0
            skipped_protected = 0

            while num_evicted < num_tokens and len(eviction_heap):
                priority, x = heapq.heappop(eviction_heap)

                # Session TTL 保护检查
                node_session_id = x.key.extra_key if x.key else None
                if node_session_id is not None and _is_session_protected(node_session_id):
                    skipped_protected += 1
                    continue  # 不驱逐，也不推回堆

                self.token_to_kv_pool_allocator.free(x.value)
                num_evicted += len(x.value)
                self._delete_leaf(x)

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

        target.evict = wrapped_evict
```

**关键设计**：
- `x.key.extra_key` 就是之前由 `C_SchedulerHook` 写入的 `session_id`
- 受保护节点被 `continue` 跳过，不推回堆 → 同一轮 evict 不会被重复检查
- 如果所有叶子节点都被保护，`eviction_heap` 会耗尽，驱逐提前终止 → 不会死循环

**工作量**: 4h

---

### B3. Hook 注册

**文件 1**: `hisim/src/hisim/simulation/sglang/sglang_bench.py`
**位置**: line 27-36，`install_class_hooks` 调用
**修改**:

```python
hisim_hook.install_class_hooks(
    [
        sglang_hook.C_SchedulerHook,
        sglang_hook.C_ModelRunnerHook,
        sglang_hook.C_TokenizerManagerHook,
        sglang_hook.C_StorageBackendFactory,
        sglang_hook.C_HiCacheController,
        sglang_hook.C_HiRadixCacheHook,
        sglang_hook.C_RadixCacheHook,  # 新增
    ]
)
```

**文件 2**: `hisim/src/hisim/simulation/sglang/launch_server.py`
**位置**: line 16-25，`install_class_hooks` 调用
**修改**:

```python
hisim_hook.install_class_hooks(
    [
        sglang_hook.C_SchedulerHook,
        sglang_hook.C_ModelRunnerHook,
        sglang_hook.C_TokenizerManagerHook,
        sglang_hook.C_StorageBackendFactory,
        sglang_hook.C_HiCacheController,
        sglang_hook.C_HiRadixCacheHook,
        sglang_hook.C_RadixCacheHook,  # 新增
    ]
)
```

**工作量**: 0.5h

---

### B4. TTL 过期清理调度

**文件**: `hisim/src/hisim/simulation/sglang/sglang_hook.py`
**位置**: `C_SchedulerHook.wrapped_process_batch_result()`（line 846-900 区域）

在 `process_batch_result` 结尾（return 之前）新增：

```python
# 清理过期 session
_cleanup_expired_sessions()
```

**设计**：每轮推理迭代结束时清理，避免 `SESSION_TTL_TABLE` 无限增长。清理频率与推理迭代频率一致，开销可忽略。

**工作量**: 0.5h

---

### B5. extra_key 传递的 bug 修复（必须先做）

**问题**: `Scheduler.handle_generate_request()` (scheduler.py:1305) 构造 `Req` 时未传递 `recv_req.extra_key`，导致 `extra_key` 机制对主请求流无效。

**解决方案**: 通过 Hook 在 `C_SchedulerHook.wrapped_recv_requests()` 中，在请求到达 Scheduler 后、进入 `handle_generate_request` 之前，将 session_id 写入 `TokenizedGenerateReqInput.extra_key`。

但实际发现：`recv_requests()` 收到的已经是 `TokenizedGenerateReqInput` 对象，此时 `extra_key` 为 None。我们需要在 `handle_generate_request` 被 SGLang 调用、`Req` 对象创建之后，再给 `Req` 设置 `extra_key`。

**实际方案**: 在 `wrapped_recv_requests()` 中，将 `session_id` 存入 `custom_params["simulation"]["__extra_key__"]`，然后在 `wrapped_get_new_batch_prefill()` 中（此时 `Req` 对象已创建），从 `custom_params` 读取并设置 `req.extra_key`：

```python
# wrapped_get_new_batch_prefill() 中，for req in new_batch.reqs 循环内
sim_args = req.sampling_params.custom_params.get("simulation", {})
extra_key_from_session = sim_args.get("__extra_key__")
if extra_key_from_session and hasattr(req, "extra_key"):
    req.extra_key = extra_key_from_session
```

**工作量**: 1.5h（需要验证 SGLang 的 `Req.extra_key` 在哪个时机被读取，确保在 `init_next_round_input` 之前设置生效）

---

## 阶段 C：指标采集与输出（0.5 天）

### C1. calc_metrics 新增 session 相关指标

**文件**: `hisim/src/hisim/simulation/utils.py`
**位置**: `calc_metrics()`（line 53-124）

在返回的 dict 中新增：

```python
# session 相关指标
"session_count": len(set(
    r.session_id for r in requests if r.session_id is not None
)),
"session_protected_requests": sum(
    1 for r in requests if r.session_id is not None
),
```

**工作量**: 0.5h

---

### C2. profile 输出中包含 session 信息

**文件**: `hisim/src/hisim/simulation/sglang/sglang_hook.py`
**位置**: `wrapped_profile()`（line 902-966）

`RequestStats` 已通过 A4 新增 `session_id` 和 `parent_session_id` 字段。`asdict(item)` 会自动序列化这些字段到 `request.jsonl`，无需额外修改。

**但需要**在 `wrapped_profile()` 的 `reset` 段新增 `SESSION_TTL_TABLE.clear()`（已在 A5 中覆盖）。

**工作量**: 0h（已包含在 A5 中）

---

## 阶段 D：测试验证（1.5 天）

### D1. 构造带 session 字段的测试数据集

**文件**: 新建 `hisim/test/assets/dataset/session_aware_requests.jsonl`

数据格式示例：
```json
{"input_ids": [1,2,3], "output_length": 10, "created_time": 0.0, "session_id": "sess-A", "cache_control": {"type": "ephemeral", "ttl": 5}}
{"input_ids": [4,5,6], "output_length": 10, "created_time": 0.1, "session_id": "sess-A", "cache_control": {"type": "ephemeral", "ttl": 5}}
{"input_ids": [1,2,3], "output_length": 10, "created_time": 0.2, "session_id": "sess-B", "cache_control": {"type": "ephemeral", "ttl": 5}}
```

设计要点：
- 同一 session 的两个请求（sess-A），第二个应命中前缀缓存
- 不同 session（sess-A vs sess-B）的相同 token_ids，应完全隔离不命中
- 无 session 的请求，不受 TTL 保护

**工作量**: 1h

---

### D2. 新增测试用例

**文件**: `hisim/test/test_simulation_sglang_runner.py`

```python
@pytest.mark.skipif(
    not check_framework("sglang", device="cpu"),
    reason="sglang is not installed.",
)
def test_session_aware_cache():
    """验证 session 隔离和 TTL 保护"""
    from sglang.srt.server_args import ServerArgs

    runner = SGLangBenchmarkRunner(
        server_args=ServerArgs(
            model_path=MODEL_PATH,
            load_format="dummy",
            device="cpu",
            enable_hierarchical_cache=True,
        )
    )

    # Test 1: session 隔离
    # 同 session 的相同 prefix 应命中缓存
    dataset_args = DatasetArgs(
        name="hisim_collection",
        filepath=str(Path(__file__).parent / "assets/dataset/session_aware_requests.jsonl"),
    )
    benchmark_config = BenchmarkConfig()
    metrics = runner.benchmark(benchmark_config, dataset_args=dataset_args)

    # 同 session 请求应有前缀缓存命中
    assert metrics["prefix_cache_reused_ratio"] > 0

    # Test 2: TTL 保护
    # session 的 KV cache 在 TTL 内不应被驱逐
    # （此测试需构造内存压力场景，待集成验证时补充）

    runner.shutdown()
```

**工作量**: 2h

---

### D3. 端到端集成验证

**场景 1 — Session 隔离验证**:
1. 发送 sess-A 的请求（input=[1,2,3,4,5]）
2. 发送 sess-B 的请求（input=[1,2,3,4,5]）——相同 token，不同 session
3. 验证 sess-B **未命中** sess-A 的前缀缓存（extra_key 不同 → radix tree 不同子树）

**场景 2 — TTL 保护验证**:
1. 发送 sess-A 的请求（ttl=5min）
2. 构造内存压力，触发 evict()
3. 验证 sess-A 的 KV cache **未被驱逐**（SESSION_TTL_TABLE 中受保护）
4. 推进虚拟时钟超过 5min
5. 再次触发 evict()
6. 验证 sess-A 的 KV cache **已被驱逐**（TTL 过期）

**场景 3 — TTL 刷新验证**:
1. 发送 sess-A 的请求（ttl=5min）
2. 推进虚拟时钟 3min
3. 发送 sess-A 的新请求（ttl=5min）→ 刷新 TTL
4. 推进虚拟时钟 4min（距首次请求 7min，但距刷新仅 4min）
5. 触发 evict()
6. 验证 sess-A 的 KV cache **未被驱逐**（TTL 被刷新）

**工作量**: 4h

---

## 任务依赖图

```
A1 ──┐
A2 ──┤
A3 ──┼──→ A5 ──→ B5 ──┐
A4 ──┘                  ├──→ B1 ──→ B2 ──→ B3 ──→ C1 ──→ D1 ──→ D2 ──→ D3
                        │         B4 ──┘
                        └────────┘
```

- A1-A4 互相独立，可并行
- A5 依赖 A3 和 A4（需知道 simulation_params 和 RequestStats 的结构）
- B5（extra_key bug 修复）依赖 A5（需确认 session_id 写入 extra_key 的时机）
- B1（TTL 管理器）依赖 A5（需知道 session_id 从哪取）
- B2（RadixCacheHook）依赖 B1（调用 _is_session_protected）
- B3（Hook 注册）依赖 B2
- B4（过期清理）依赖 B1
- C1 依赖 A4 和 B2
- D1-D3 依次依赖

---

## 详细工作量汇总

| 任务 | 描述 | 工作量 | 风险 |
|------|------|--------|------|
| A1 | GenericRequest 新增字段 | 0.5h | 低 |
| A2 | HisimCollectionDataset 解析 | 0.5h | 低 |
| A3 | sglang_bench.py 透传 | 0.5h | 低 |
| A4 | RequestStats 新增字段 | 0.5h | 低 |
| A5 | C_SchedulerHook session 透传 | 3h | 中 |
| B1 | Session TTL 管理器 | 2h | 低 |
| B2 | C_RadixCacheHook 驱逐拦截 | 4h | **高** |
| B3 | Hook 注册（2 个文件） | 0.5h | 低 |
| B4 | TTL 过期清理调度 | 0.5h | 低 |
| B5 | extra_key 传递 bug 修复 | 1.5h | **高** |
| C1 | calc_metrics 新增指标 | 0.5h | 低 |
| C2 | profile 输出 session 信息 | 0h | — |
| D1 | 构造测试数据集 | 1h | 低 |
| D2 | 新增测试用例 | 2h | 中 |
| D3 | 端到端集成验证 | 4h | **高** |
| **合计** | | **21h ≈ 2.6 天** | |

加上调试和意外处理的 buffer（×1.5）：**~6 天**

---

## 高风险任务详解

### B2 — C_RadixCacheHook（风险：高）

**风险点**：
1. `RadixCache.evict()` 被 `HiRadixCache` 继承，需确认 Hook 对两者都生效
2. `HiRadixCache` 已有 `C_HiRadixCacheHook`，需确认两个 Hook 不冲突
3. `_collect_leaves()` 返回的 `TreeNode.key.extra_key` 是否与写入的 `session_id` 一致——需要确认 `extra_key` 在 `cache_finished_req()` → `RadixKey(keys, req.extra_key)` 路径中被正确传递

**缓解**：先写一个最简版本只打 log（验证 `x.key.extra_key` 的值是否符合预期），再添加 TTL 保护逻辑。

### B5 — extra_key 传递 bug 修复（风险：高）

**风险点**：
1. `Req.extra_key` 的设置时机必须在 `init_next_round_input()` 调用 `tree_cache.match_prefix(RadixKey(..., extra_key=self.extra_key))` **之前**
2. `wrapped_get_new_batch_prefill()` 中 `Req` 对象刚创建，`extra_key` 是否仍可修改？

**缓解**：阅读 SGLang 源码确认 `init_next_round_input` 的调用链，找到最早可设置 `extra_key` 的 Hook 点。如果时机不对，可改为在 `C_SchedulerHook.wrapped_init` 中 Hook `handle_generate_request` 本身。

### D3 — 端到端集成验证（风险：高）

**风险点**：
1. TTL 保护场景需要构造内存压力（发送足够多的请求使 KV cache 满载），在 CPU 模拟环境下需精心设计请求数量和 `max_total_num_tokens`
2. 虚拟时钟推进的验证——需要确认 `StateManager.get_global_clock()` 在 OFFLINE 模式下的值是否与 TTL deadline 的计算基准一致

**缓解**：先在 BLOCKING 模式下验证（真实时间），再切换到 OFFLINE 模式。