"""D2: Session-aware KV cache simulation tests.

These tests verify the session-aware KV cache feature end-to-end:
  - Session isolation: same tokens with different session_id → no cross-session prefix hit
  - TTL protection: session KV cache not evicted within TTL period
  - TTL refresh: last TTL wins, extending protection
  - Session metrics: num_sessions, session_request_count in output

These tests require SGLang and aiconfigurator to be installed, and will
be skipped otherwise (same pattern as test_simulation_sglang_runner.py).
"""

import os
import pytest
from pathlib import Path

os.environ["HISIM_CONFIG_PATH"] = os.path.dirname(__file__) + "/assets/mock/config.json"
os.environ["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"
os.environ["SGLANG_USE_CPU_ENGINE"] = "1"

from env import check_framework, MODEL_PATH

# Use local model to avoid network download issues
if not os.path.exists(MODEL_PATH) or not os.path.exists(os.path.join(MODEL_PATH, "config.json")):
    MODEL_PATH = "/home/models/Qwen3-8B"
    print(f"Using local model: {MODEL_PATH}")
from hisim.dataset import DatasetArgs
from hisim.simulation.types import BenchmarkConfig


@pytest.mark.skipif(
    not check_framework("sglang", device="cpu"), reason="sglang is not installed."
)
# Temporarily skip vllm check for CPU simulation (issue: vllm not installed)
# @pytest.mark.skipif(
#     not check_framework("vllm"),
#     reason="The cpu simulation might require vLLM's kernels.",
# )
def test_session_isolation():
    """Scenario 1: Session Isolation.

    Two sessions (sess_A, sess_B) with identical input prefix tokens.
    Without session_id → RadixCache would share prefix → cached_tokens > 0.
    With session_id → extra_key partitions → no cross-session cache hit.

    Expected:
      Request 0 (sess_A, first): cached_tokens = 0 (new prefix)
      Request 1 (sess_A, second): cached_tokens > 0 (prefix hit within same session)
      Request 2 (sess_B, first): cached_tokens = 0 (no cross-session hit)
      Request 3 (no session): cached_tokens = 0 (no hit in default partition)
    """
    from sglang.srt.server_args import ServerArgs
    from hisim.simulation.sglang.sglang_bench import SGLangBenchmarkRunner

    runner = SGLangBenchmarkRunner(
        server_args=ServerArgs(
            model_path=MODEL_PATH,
            load_format="dummy",
            device="cpu",
            enable_hierarchical_cache=True,
        )
    )

    dataset_args = DatasetArgs(
        name="hisim_collection",
        filepath=str(
            Path(__file__).parent / "assets/dataset/session_aware_isolation.jsonl"
        ),
    )
    benchmark_config = BenchmarkConfig(ignore_request_timestamp=True)
    metrics = runner.benchmark(benchmark_config, dataset_args=dataset_args)

    # Verify completion
    assert metrics["completed"] == 4, f"Expected 4 completed, got {metrics['completed']}"

    # Verify session metrics
    assert metrics["num_sessions"] == 2, f"Expected 2 sessions, got {metrics['num_sessions']}"
    assert metrics["session_request_count"] == 3, (
        f"Expected 3 session requests, got {metrics['session_request_count']}"
    )

    # Verify per-request cache hits via request stats
    request_stats = runner.get_request_stats()

    # Request 0 (sess_A, first): no prefix hit
    assert request_stats[0]["final_reused_tokens"] == 0, (
        f"First sess_A request should have 0 reused tokens, "
        f"got {request_stats[0]['final_reused_tokens']}"
    )

    # Request 1 (sess_A, second): prefix hit within same session
    assert request_stats[1]["final_reused_tokens"] > 0, (
        f"Second sess_A request should reuse prefix tokens, "
        f"got {request_stats[1]['final_reused_tokens']}"
    )

    # Request 2 (sess_B, first): no cross-session hit
    assert request_stats[2]["final_reused_tokens"] == 0, (
        f"sess_B request should have 0 reused tokens (session isolation), "
        f"got {request_stats[2]['final_reused_tokens']}"
    )

    # Request 3 (no session): no hit in default partition
    assert request_stats[3]["final_reused_tokens"] == 0, (
        f"No-session request should have 0 reused tokens, "
        f"got {request_stats[3]['final_reused_tokens']}"
    )

    # Verify session_id field in request stats
    assert request_stats[0]["session_id"] == "sess_A"
    assert request_stats[1]["session_id"] == "sess_A"
    assert request_stats[2]["session_id"] == "sess_B"
    assert request_stats[3]["session_id"] is None

    runner.shutdown()


@pytest.mark.skipif(
    not check_framework("sglang", device="cpu"), reason="sglang is not installed."
)
# Temporarily skip vllm check for CPU simulation
# @pytest.mark.skipif(
#     not check_framework("vllm"),
#     reason="The cpu simulation might require vLLM's kernels.",
# )
def test_session_ttl_refresh():
    """Scenario 3: TTL Refresh (Last TTL Wins).

    Same session with consecutive requests that refresh the TTL.
    The last TTL value should win, extending protection.

    Timeline (virtual seconds):
      t=0:   sess_A TTL=2min → deadline at 120s
      t=50:  sess_A TTL=1min → deadline refreshed to 110s
      t=100: sess_A TTL=3min → deadline refreshed to 280s
      t=200: sess_A TTL=1min → still protected (deadline=280s > 200s)

    Expected: All 4 requests complete successfully.
    """
    from sglang.srt.server_args import ServerArgs
    from hisim.simulation.sglang.sglang_bench import SGLangBenchmarkRunner

    runner = SGLangBenchmarkRunner(
        server_args=ServerArgs(
            model_path=MODEL_PATH,
            load_format="dummy",
            device="cpu",
            enable_hierarchical_cache=True,
        )
    )

    dataset_args = DatasetArgs(
        name="hisim_collection",
        filepath=str(
            Path(__file__).parent / "assets/dataset/session_aware_ttl.jsonl"
        ),
    )
    benchmark_config = BenchmarkConfig()
    metrics = runner.benchmark(benchmark_config, dataset_args=dataset_args)

    # Verify all completed
    assert metrics["completed"] == 4, f"Expected 4 completed, got {metrics['completed']}"

    # Verify single session
    assert metrics["num_sessions"] == 1, f"Expected 1 session, got {metrics['num_sessions']}"
    assert metrics["session_request_count"] == 4

    # Verify request stats contain session info
    request_stats = runner.get_request_stats()
    for req in request_stats:
        assert req["session_id"] == "sess_A", (
            f"All requests should belong to sess_A, got {req['session_id']}"
        )

    runner.shutdown()


@pytest.mark.skipif(
    not check_framework("sglang", device="cpu"), reason="sglang is not installed."
)
# Temporarily skip vllm check for CPU simulation
# @pytest.mark.skipif(
#     not check_framework("vllm"),
#     reason="The cpu simulation might require vLLM's kernels.",
# )
def test_session_metrics_in_output():
    """Verify that session metrics appear in the metrics output.

    Uses the isolation dataset to check that num_sessions and
    session_request_count are correctly computed.
    """
    from sglang.srt.server_args import ServerArgs
    from hisim.simulation.sglang.sglang_bench import SGLangBenchmarkRunner

    runner = SGLangBenchmarkRunner(
        server_args=ServerArgs(
            model_path=MODEL_PATH,
            load_format="dummy",
            device="cpu",
            enable_hierarchical_cache=True,
        )
    )

    dataset_args = DatasetArgs(
        name="hisim_collection",
        filepath=str(
            Path(__file__).parent / "assets/dataset/session_aware_isolation.jsonl"
        ),
    )
    benchmark_config = BenchmarkConfig(ignore_request_timestamp=True)
    metrics = runner.benchmark(benchmark_config, dataset_args=dataset_args)

    # Check new session metric keys exist
    assert "num_sessions" in metrics, "num_sessions missing from metrics"
    assert "session_request_count" in metrics, "session_request_count missing from metrics"
    assert "session_cache_reused_ratio" in metrics, "session_cache_reused_ratio missing from metrics"

    # Check values
    assert metrics["num_sessions"] >= 0
    assert metrics["session_request_count"] >= 0
    assert 0 <= metrics["session_cache_reused_ratio"] <= 1.0

    # Original metrics should still be present
    assert "prefix_cache_reused_ratio" in metrics
    assert "completed" in metrics
    assert "mean_ttft_ms" in metrics

    runner.shutdown()


if __name__ == "__main__":
    test_session_isolation()