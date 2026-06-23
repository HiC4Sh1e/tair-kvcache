"""D3: End-to-end integration verification script.

Run this script to verify the session-aware KV cache feature works
correctly in a full HiSim simulation. Requires SGLang + aiconfigurator.

Usage:
  cd hisim
  python test_session_aware/e2e_run.py --model_path <your_model>

Or run individual scenarios:
  python test_session_aware/e2e_run.py --scenario isolation
  python test_session_aware/e2e_run.py --scenario ttl_refresh
  python test_session_aware/e2e_run.py --scenario metrics
"""

import argparse
import json
import os
import sys
import tempfile

# Add hisim src to path
hisim_src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if hisim_src not in sys.path:
    sys.path.insert(0, hisim_src)

# Add test directory for env.py
test_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test")
if test_dir not in sys.path:
    sys.path.insert(0, test_dir)


def check_prerequisites():
    """Check that SGLang is available."""
    try:
        from importlib.metadata import distributions
        pkgs = set()
        for d in distributions():
            if d.metadata and "Name" in d.metadata:
                pkgs.add(d.metadata["Name"])
        if "sglang" not in pkgs:
            print("ERROR: SGLang is not installed. Cannot run e2e tests.")
            return False
        return True
    except Exception as e:
        print(f"ERROR checking prerequisites: {e}")
        return False


def run_scenario_isolation(model_path):
    """Scenario 1: Session Isolation."""
    from sglang.srt.server_args import ServerArgs
    from hisim.simulation.sglang.sglang_bench import SGLangBenchmarkRunner
    from hisim.dataset import DatasetArgs
    from hisim.simulation.types import BenchmarkConfig

    dataset_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "test", "assets", "dataset", "session_aware_isolation.jsonl"
    )

    print("\n" + "=" * 60)
    print("Scenario 1: Session Isolation")
    print("=" * 60)

    runner = SGLangBenchmarkRunner(
        server_args=ServerArgs(
            model_path=model_path,
            load_format="dummy",
            device="cpu",
            enable_hierarchical_cache=False,  # Disable L2 cache to reduce memory
            max_total_tokens=200000,  # Limit to ~200K tokens for ~500GB memory
        )
    )

    dataset_args = DatasetArgs(
        name="hisim_collection",
        filepath=dataset_path,
    )
    benchmark_config = BenchmarkConfig(ignore_request_timestamp=True)
    metrics = runner.benchmark(benchmark_config, dataset_args=dataset_args)
    request_stats = runner.get_request_stats()

    print(f"\nMetrics:")
    print(f"  completed: {metrics['completed']}")
    print(f"  num_sessions: {metrics['num_sessions']}")
    print(f"  session_request_count: {metrics['session_request_count']}")
    print(f"  session_cache_reused_ratio: {metrics['session_cache_reused_ratio']:.4f}")
    print(f"  prefix_cache_reused_ratio: {metrics['prefix_cache_reused_ratio']:.4f}")

    print(f"\nPer-request cache hits:")
    for i, req in enumerate(request_stats):
        print(f"  Request {i}: session_id={req['session_id']}, "
              f"reused_tokens={req['final_reused_tokens']}, "
              f"input_length={req['input_length']}")

    # Validate expectations
    errors = []
    if metrics["completed"] != 4:
        errors.append(f"Expected 4 completed, got {metrics['completed']}")
    if metrics["num_sessions"] != 2:
        errors.append(f"Expected 2 sessions, got {metrics['num_sessions']}")
    if metrics["session_request_count"] != 3:
        errors.append(f"Expected 3 session requests, got {metrics['session_request_count']}")

    # Check per-request expectations
    if len(request_stats) >= 4:
        if request_stats[0]["final_reused_tokens"] != 0:
            errors.append(f"Req 0 (sess_A first): expected 0 reused, got {request_stats[0]['final_reused_tokens']}")
        if request_stats[1]["final_reused_tokens"] <= 0:
            errors.append(f"Req 1 (sess_A second): expected >0 reused, got {request_stats[1]['final_reused_tokens']}")
        if request_stats[2]["final_reused_tokens"] != 0:
            errors.append(f"Req 2 (sess_B first): expected 0 reused (isolation), got {request_stats[2]['final_reused_tokens']}")
        if request_stats[3]["final_reused_tokens"] != 0:
            errors.append(f"Req 3 (no session): expected 0 reused, got {request_stats[3]['final_reused_tokens']}")

    runner.shutdown()

    if errors:
        print(f"\nFAILURES:")
        for e in errors:
            print(f"  - {e}")
        return False
    else:
        print(f"\nScenario 1: PASSED")
        return True


def run_scenario_ttl_refresh(model_path):
    """Scenario 3: TTL Refresh (Last TTL Wins)."""
    from sglang.srt.server_args import ServerArgs
    from hisim.simulation.sglang.sglang_bench import SGLangBenchmarkRunner
    from hisim.dataset import DatasetArgs
    from hisim.simulation.types import BenchmarkConfig

    dataset_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "test", "assets", "dataset", "session_aware_ttl.jsonl"
    )

    print("\n" + "=" * 60)
    print("Scenario 3: TTL Refresh (Last TTL Wins)")
    print("=" * 60)

    runner = SGLangBenchmarkRunner(
        server_args=ServerArgs(
            model_path=model_path,
            load_format="dummy",
            device="cpu",
            enable_hierarchical_cache=False,  # Disable L2 cache to reduce memory
            max_total_tokens=200000,  # Limit to ~200K tokens for ~500GB memory
        )
    )

    dataset_args = DatasetArgs(
        name="hisim_collection",
        filepath=dataset_path,
    )
    benchmark_config = BenchmarkConfig()
    metrics = runner.benchmark(benchmark_config, dataset_args=dataset_args)
    request_stats = runner.get_request_stats()

    print(f"\nMetrics:")
    print(f"  completed: {metrics['completed']}")
    print(f"  num_sessions: {metrics['num_sessions']}")
    print(f"  session_request_count: {metrics['session_request_count']}")
    print(f"  session_cache_reused_ratio: {metrics['session_cache_reused_ratio']:.4f}")

    print(f"\nPer-request details:")
    for i, req in enumerate(request_stats):
        print(f"  Request {i}: session_id={req['session_id']}, "
              f"reused_tokens={req['final_reused_tokens']}, "
              f"created_time={req.get('created_time', 'N/A')}")

    errors = []
    if metrics["completed"] != 4:
        errors.append(f"Expected 4 completed, got {metrics['completed']}")
    if metrics["num_sessions"] != 1:
        errors.append(f"Expected 1 session, got {metrics['num_sessions']}")
    if metrics["session_request_count"] != 4:
        errors.append(f"Expected 4 session requests, got {metrics['session_request_count']}")

    # All requests should belong to sess_A
    for i, req in enumerate(request_stats):
        if req["session_id"] != "sess_A":
            errors.append(f"Req {i}: expected session_id=sess_A, got {req['session_id']}")

    runner.shutdown()

    if errors:
        print(f"\nFAILURES:")
        for e in errors:
            print(f"  - {e}")
        return False
    else:
        print(f"\nScenario 3: PASSED")
        return True


def run_scenario_metrics(model_path):
    """Verify session metrics in output."""
    from sglang.srt.server_args import ServerArgs
    from hisim.simulation.sglang.sglang_bench import SGLangBenchmarkRunner
    from hisim.dataset import DatasetArgs
    from hisim.simulation.types import BenchmarkConfig

    dataset_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "test", "assets", "dataset", "session_aware_isolation.jsonl"
    )

    print("\n" + "=" * 60)
    print("Scenario: Session Metrics in Output")
    print("=" * 60)

    runner = SGLangBenchmarkRunner(
        server_args=ServerArgs(
            model_path=model_path,
            load_format="dummy",
            device="cpu",
            enable_hierarchical_cache=False,  # Disable L2 cache to reduce memory
            max_total_tokens=200000,  # Limit to ~200K tokens for ~500GB memory
        )
    )

    dataset_args = DatasetArgs(
        name="hisim_collection",
        filepath=dataset_path,
    )
    benchmark_config = BenchmarkConfig(ignore_request_timestamp=True)
    metrics = runner.benchmark(benchmark_config, dataset_args=dataset_args)

    # Check all session metric keys exist
    session_keys = ["num_sessions", "session_request_count", "session_cache_reused_ratio"]
    missing = [k for k in session_keys if k not in metrics]
    if missing:
        print(f"FAIL: Missing session metric keys: {missing}")
        runner.shutdown()
        return False

    # Check original metrics still present
    original_keys = ["prefix_cache_reused_ratio", "completed", "mean_ttft_ms", "time_cost"]
    missing_orig = [k for k in original_keys if k not in metrics]
    if missing_orig:
        print(f"FAIL: Missing original metric keys: {missing_orig}")
        runner.shutdown()
        return False

    # Check session_id in request stats
    request_stats = runner.get_request_stats()
    session_ids_in_stats = [req.get("session_id") for req in request_stats]
    has_sessions = any(sid is not None for sid in session_ids_in_stats)
    if not has_sessions:
        print("FAIL: No session_id found in request stats")
        runner.shutdown()
        return False

    print(f"  num_sessions: {metrics['num_sessions']}")
    print(f"  session_request_count: {metrics['session_request_count']}")
    print(f"  session_cache_reused_ratio: {metrics['session_cache_reused_ratio']:.4f}")
    print(f"  session_ids in request stats: {session_ids_in_stats}")

    runner.shutdown()
    print(f"\nSession Metrics: PASSED")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Session-aware KV cache e2e verification")
    parser.add_argument("--model_path", type=str, default=None,
                       help="Model path for simulation")
    parser.add_argument("--scenario", type=str, default="all",
                       choices=["all", "isolation", "ttl_refresh", "metrics"],
                       help="Which scenario to run")
    args = parser.parse_args()

    if not check_prerequisites():
        sys.exit(1)

    # Set up environment
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(script_dir, "test", "assets", "mock", "config_low_memory.json")
    os.environ["HISIM_CONFIG_PATH"] = config_path
    os.environ["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"
    os.environ["SGLANG_USE_CPU_ENGINE"] = "1"
    print(f"Using config: {config_path}")

    model_path = args.model_path or os.getenv("BENCHMARK_TEST_MODEL_PATH", "Qwen/Qwen3-0.6B")

    results = {}
    if args.scenario in ("all", "isolation"):
        results["isolation"] = run_scenario_isolation(model_path)
    if args.scenario in ("all", "ttl_refresh"):
        results["ttl_refresh"] = run_scenario_ttl_refresh(model_path)
    if args.scenario in ("all", "metrics"):
        results["metrics"] = run_scenario_metrics(model_path)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"  {name}: {status}")

    all_passed = all(results.values())
    sys.exit(0 if all_passed else 1)