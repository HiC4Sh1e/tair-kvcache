"""Phase C: Standalone verification for session-aware metrics.

Tests that calc_metrics() correctly computes session-related metrics
from RequestStats data.

C1: num_sessions, session_request_count, session_cache_reused_ratio
C2: RequestStats serialization includes session_id/parent_session_id (already verified in A4)
"""

import sys
import os

hisim_src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if hisim_src not in sys.path:
    sys.path.insert(0, hisim_src)

from hisim.simulation.types import RequestStats
from hisim.simulation.utils import calc_metrics


def make_request_stats(rid, input_length=100, output_length=10,
                       final_reused_tokens=0, session_id=None,
                       parent_session_id=None):
    """Helper to create a RequestStats with sensible defaults."""
    return RequestStats(
        rid=rid,
        input_length=input_length,
        output_length=output_length,
        final_reused_tokens=final_reused_tokens,
        session_id=session_id,
        parent_session_id=parent_session_id,
        gen_token_latencies=[0.01] * output_length,
    )


def test_no_session_requests():
    """When no requests have session_id, session metrics should be zero."""
    requests = [
        make_request_stats("r1", input_length=100, final_reused_tokens=50),
        make_request_stats("r2", input_length=200, final_reused_tokens=100),
    ]
    metrics = calc_metrics(requests)
    assert metrics["num_sessions"] == 0, f"num_sessions should be 0, got {metrics['num_sessions']}"
    assert metrics["session_request_count"] == 0, f"session_request_count should be 0"
    assert metrics["session_cache_reused_ratio"] == 0, f"session_cache_reused_ratio should be 0"
    print("  No session requests — OK")


def test_single_session():
    """Single session with multiple requests."""
    requests = [
        make_request_stats("r1", input_length=100, final_reused_tokens=80,
                         session_id="sess_A"),
        make_request_stats("r2", input_length=100, final_reused_tokens=90,
                         session_id="sess_A"),
        make_request_stats("r3", input_length=50, final_reused_tokens=0),
    ]
    metrics = calc_metrics(requests)
    assert metrics["num_sessions"] == 1, f"num_sessions should be 1, got {metrics['num_sessions']}"
    assert metrics["session_request_count"] == 2, f"session_request_count should be 2, got {metrics['session_request_count']}"
    # session_input = 100 + 100 = 200, session_reused = 80 + 90 = 170
    expected_ratio = 170 / 200
    assert abs(metrics["session_cache_reused_ratio"] - expected_ratio) < 1e-6, \
        f"session_cache_reused_ratio should be {expected_ratio}, got {metrics['session_cache_reused_ratio']}"
    print("  Single session — OK")


def test_multiple_sessions():
    """Multiple sessions with different IDs."""
    requests = [
        make_request_stats("r1", input_length=100, final_reused_tokens=80,
                         session_id="sess_A"),
        make_request_stats("r2", input_length=200, final_reused_tokens=100,
                         session_id="sess_B"),
        make_request_stats("r3", input_length=50, final_reused_tokens=0,
                         session_id="sess_A"),
        make_request_stats("r4", input_length=50, final_reused_tokens=0),
    ]
    metrics = calc_metrics(requests)
    assert metrics["num_sessions"] == 2, f"num_sessions should be 2, got {metrics['num_sessions']}"
    assert metrics["session_request_count"] == 3, f"session_request_count should be 3"
    # session_input = 100 + 200 + 50 = 350, session_reused = 80 + 100 + 0 = 180
    expected_ratio = 180 / 350
    assert abs(metrics["session_cache_reused_ratio"] - expected_ratio) < 1e-6, \
        f"session_cache_reused_ratio should be {expected_ratio}, got {metrics['session_cache_reused_ratio']}"
    # Verify overall prefix_cache_reused_ratio is unchanged
    # total_input = 100 + 200 + 50 + 50 = 400, total_reused = 80 + 100 + 0 + 0 = 180
    expected_total_ratio = 180 / 400
    assert abs(metrics["prefix_cache_reused_ratio"] - expected_total_ratio) < 1e-6, \
        f"prefix_cache_reused_ratio should be {expected_total_ratio}"
    print("  Multiple sessions — OK")


def test_session_with_zero_reuse():
    """Session requests with zero reused tokens."""
    requests = [
        make_request_stats("r1", input_length=100, final_reused_tokens=0,
                         session_id="sess_A"),
    ]
    metrics = calc_metrics(requests)
    assert metrics["num_sessions"] == 1
    assert metrics["session_request_count"] == 1
    assert metrics["session_cache_reused_ratio"] == 0
    print("  Session with zero reuse — OK")


def test_existing_metrics_unchanged():
    """Verify that existing metrics are not affected by the new session fields."""
    requests = [
        make_request_stats("r1", input_length=100, output_length=5,
                         final_reused_tokens=50),
    ]
    metrics = calc_metrics(requests)
    # Check that all original metric keys are still present
    assert "num_requests" in metrics
    assert "completed" in metrics
    assert "total_input" in metrics
    assert "total_output" in metrics
    assert "duration" in metrics
    assert "request_throughput" in metrics
    assert "prefix_cache_reused_ratio" in metrics
    assert "mean_ttft_ms" in metrics
    assert "mean_tpot_ms" in metrics
    assert "time_cost" in metrics
    # New keys
    assert "num_sessions" in metrics
    assert "session_request_count" in metrics
    assert "session_cache_reused_ratio" in metrics
    print("  Existing metrics unchanged — OK")


def test_parent_session_id_not_counted():
    """parent_session_id should not affect session count (only session_id matters)."""
    requests = [
        make_request_stats("r1", input_length=100,
                         session_id="sess_A", parent_session_id="sess_P"),
        make_request_stats("r2", input_length=100,
                         session_id=None, parent_session_id="sess_P"),
    ]
    metrics = calc_metrics(requests)
    assert metrics["num_sessions"] == 1, "Only session_id=sess_A should count"
    assert metrics["session_request_count"] == 1, "Only r1 has session_id"
    print("  parent_session_id not counted — OK")


if __name__ == "__main__":
    print("=" * 60)
    print("Phase C: Session-Aware Metrics Verification")
    print("=" * 60)

    passed = 0
    failed = 0

    tests = [
        ("C1-1", test_no_session_requests),
        ("C1-2", test_single_session),
        ("C1-3", test_multiple_sessions),
        ("C1-4", test_session_with_zero_reuse),
        ("C1-5", test_existing_metrics_unchanged),
        ("C1-6", test_parent_session_id_not_counted),
    ]

    for name, test_fn in tests:
        try:
            test_fn()
            print(f"[PASS] {name}")
            passed += 1
        except Exception as e:
            print(f"[FAIL] {name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed (total {passed + failed})")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)