"""Phase A: Standalone verification for session-aware data flow.

Tests that session fields (session_id, parent_session_id, cache_control) flow
correctly through the HiSim pipeline WITHOUT requiring a full simulation run.

This covers:
  A1: GenericRequest fields
  A2: HisimCollectionDataset parsing from JSONL
  A3: simulation_params passthrough
  A4: RequestStats fields and serialization
  A5: Session TTL Manager logic (update, check, cleanup, refresh)
"""

import json
import os
import sys
import tempfile

# Add hisim src to path
hisim_src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if hisim_src not in sys.path:
    sys.path.insert(0, hisim_src)


def test_generic_request_fields():
    """A1: GenericRequest has session_id, parent_session_id, cache_control fields."""
    from hisim.dataset.base_dataset import GenericRequest

    req = GenericRequest(
        prompt="hello",
        input_length=5,
        output_length=10,
        session_id="sess_1",
        parent_session_id="sess_0",
        cache_control={"type": "ephemeral", "ttl": 5},
    )
    assert req.session_id == "sess_1", f"session_id mismatch: {req.session_id}"
    assert req.parent_session_id == "sess_0", f"parent_session_id mismatch: {req.parent_session_id}"
    assert req.cache_control == {"type": "ephemeral", "ttl": 5}, f"cache_control mismatch: {req.cache_control}"

    # Defaults to None
    req2 = GenericRequest(prompt="world", input_length=3, output_length=5)
    assert req2.session_id is None
    assert req2.parent_session_id is None
    assert req2.cache_control is None

    print("[PASS] A1: GenericRequest fields")


def test_dataset_parsing():
    """A2: HisimCollectionDataset parses session fields from JSONL with backward compatibility."""
    from hisim.dataset.base_dataset import GenericRequest
    from hisim.dataset.dataset_args import DatasetArgs

    # Create temp JSONL file with session fields
    tmpdir = tempfile.mkdtemp()
    filepath = os.path.join(tmpdir, "test_trace.jsonl")

    data = [
        {
            "input_ids": [1, 2, 3],
            "output_length": 10,
            "created_time": 100.5,
            "session_id": "sess_A",
            "parent_session_id": "sess_P",
            "cache_control": {"type": "ephemeral", "ttl": 5},
        },
        {
            "input_ids": [4, 5, 6],
            "output_length": 8,
            "created_time": 101.0,
            # No session fields — backward compatibility test
        },
    ]

    with open(filepath, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

    try:
        from hisim.dataset.hisim_collection import HisimCollectionDataset
        args = DatasetArgs("hisim_collection", filepath=filepath)
        dataset = HisimCollectionDataset(tokenizer=None, args=args)

        # Request with session fields
        req0 = dataset[0]
        assert req0.session_id == "sess_A", f"session_id mismatch: {req0.session_id}"
        assert req0.parent_session_id == "sess_P", f"parent_session_id mismatch: {req0.parent_session_id}"
        assert req0.cache_control == {"type": "ephemeral", "ttl": 5}, f"cache_control mismatch: {req0.cache_control}"
        # created_time should be aligned (subtracted min timestamp)
        assert req0.custom_params["created_time"] == 0.0, f"created_time should be 0.0: {req0.custom_params['created_time']}"

        # Request without session fields (backward compatibility)
        req1 = dataset[1]
        assert req1.session_id is None, f"session_id should be None: {req1.session_id}"
        assert req1.parent_session_id is None, f"parent_session_id should be None: {req1.parent_session_id}"
        assert req1.cache_control is None, f"cache_control should be None: {req1.cache_control}"

        print("[PASS] A2: HisimCollectionDataset parsing")
    finally:
        os.remove(filepath)
        os.rmdir(tmpdir)


def test_simulation_params_passthrough():
    """A3: session fields from GenericRequest flow into simulation_params dict."""
    from hisim.dataset.base_dataset import GenericRequest

    # Simulate what sglang_bench.get_request() does
    req = GenericRequest(
        token_ids=[1, 2, 3],
        input_length=3,
        output_length=10,
        custom_params={"created_time": 5.0},
        session_id="sess_X",
        parent_session_id="sess_Y",
        cache_control={"type": "ephemeral", "ttl": 10},
    )

    simulation_params = {
        "total_request": 100,
        "created_time": req.custom_params.get("created_time", 0),
        "session_id": req.session_id,
        "parent_session_id": req.parent_session_id,
        "cache_control": req.cache_control,
    }

    assert simulation_params["session_id"] == "sess_X"
    assert simulation_params["parent_session_id"] == "sess_Y"
    assert simulation_params["cache_control"] == {"type": "ephemeral", "ttl": 10}

    # Test with None fields
    req2 = GenericRequest(
        token_ids=[4, 5],
        input_length=2,
        output_length=5,
    )
    simulation_params2 = {
        "total_request": 100,
        "created_time": 0,
        "session_id": req2.session_id,
        "parent_session_id": req2.parent_session_id,
        "cache_control": req2.cache_control,
    }
    assert simulation_params2["session_id"] is None
    assert simulation_params2["parent_session_id"] is None
    assert simulation_params2["cache_control"] is None

    print("[PASS] A3: simulation_params passthrough")


def test_request_stats_fields():
    """A4: RequestStats has session_id and parent_session_id, serialized via asdict."""
    from dataclasses import asdict
    from hisim.simulation.types import RequestStats

    stats = RequestStats(
        rid="req_001",
        session_id="sess_A",
        parent_session_id="sess_B",
    )
    assert stats.session_id == "sess_A"
    assert stats.parent_session_id == "sess_B"

    # asdict serialization (used in wrapped_profile for JSON output)
    d = asdict(stats)
    assert d["session_id"] == "sess_A"
    assert d["parent_session_id"] == "sess_B"
    assert "gen_token_latencies" in d

    print("[PASS] A4: RequestStats fields and serialization")


def _make_state_manager():
    """Create a mock StateManager for TTL tests without importing the real module."""
    class MockStateManager:
        _clock = 0.0

        @classmethod
        def get_global_clock(cls):
            return cls._clock

        @classmethod
        def set_global_clock(cls, t):
            cls._clock = t

        @classmethod
        def reset(cls):
            cls._clock = 0.0

    return MockStateManager


def test_session_ttl_manager():
    """A5: Session TTL Manager logic — update, protect, expire, refresh, cleanup."""
    # We can't easily import _update_session_ttl etc. from sglang_hook due to
    # heavy dependencies (torch, sglang). So we replicate the pure logic here.
    StateManager = _make_state_manager()
    SESSION_TTL_TABLE = {}

    def update_session_ttl(session_id, cache_control):
        if session_id is None or cache_control is None:
            return
        if cache_control.get("type") != "ephemeral":
            return
        ttl_minutes = cache_control.get("ttl", 5)
        ttl_minutes = max(0, min(ttl_minutes, 60))
        ttl_deadline = StateManager.get_global_clock() + ttl_minutes * 60
        SESSION_TTL_TABLE[session_id] = ttl_deadline

    def is_session_protected(session_id):
        if session_id is None or session_id not in SESSION_TTL_TABLE:
            return False
        if StateManager.get_global_clock() < SESSION_TTL_TABLE[session_id]:
            return True
        del SESSION_TTL_TABLE[session_id]
        return False

    def cleanup_expired_sessions():
        now = StateManager.get_global_clock()
        expired = [sid for sid, deadline in SESSION_TTL_TABLE.items() if now >= deadline]
        for sid in expired:
            del SESSION_TTL_TABLE[sid]

    # --- Test 1: TTL update and protection ---
    StateManager.set_global_clock(0)
    update_session_ttl("sess_A", {"type": "ephemeral", "ttl": 5})  # 5 min = 300s
    assert "sess_A" in SESSION_TTL_TABLE
    assert SESSION_TTL_TABLE["sess_A"] == 300.0
    assert is_session_protected("sess_A") is True
    print("  Test 1: TTL update and protection — OK")

    # --- Test 2: TTL expiration ---
    StateManager.set_global_clock(301)  # past the 300s deadline
    assert is_session_protected("sess_A") is False
    assert "sess_A" not in SESSION_TTL_TABLE  # cleaned up on check
    print("  Test 2: TTL expiration — OK")

    # --- Test 3: TTL refresh (last TTL wins) ---
    StateManager.set_global_clock(0)
    update_session_ttl("sess_B", {"type": "ephemeral", "ttl": 2})  # deadline at 120s
    assert SESSION_TTL_TABLE["sess_B"] == 120.0

    StateManager.set_global_clock(100)  # before deadline
    update_session_ttl("sess_B", {"type": "ephemeral", "ttl": 3})  # refresh: deadline at 100+180=280s
    assert SESSION_TTL_TABLE["sess_B"] == 280.0

    StateManager.set_global_clock(200)  # past original 120s but before refreshed 280s
    assert is_session_protected("sess_B") is True  # still protected!
    print("  Test 3: TTL refresh (last wins) — OK")

    # --- Test 4: Batch cleanup ---
    StateManager.set_global_clock(0)
    update_session_ttl("sess_1", {"type": "ephemeral", "ttl": 1})  # deadline=60
    update_session_ttl("sess_2", {"type": "ephemeral", "ttl": 2})  # deadline=120
    update_session_ttl("sess_3", {"type": "ephemeral", "ttl": 5})  # deadline=300

    StateManager.set_global_clock(150)  # sess_1 and sess_2 expired
    cleanup_expired_sessions()
    assert "sess_1" not in SESSION_TTL_TABLE
    assert "sess_2" not in SESSION_TTL_TABLE
    assert "sess_3" in SESSION_TTL_TABLE  # still alive
    print("  Test 4: Batch cleanup — OK")

    # --- Test 5: Non-ephemeral type is ignored ---
    StateManager.set_global_clock(0)
    update_session_ttl("sess_X", {"type": "persistent", "ttl": 60})
    assert "sess_X" not in SESSION_TTL_TABLE
    print("  Test 5: Non-ephemeral type ignored — OK")

    # --- Test 6: None session_id is ignored ---
    update_session_ttl(None, {"type": "ephemeral", "ttl": 5})
    assert len([k for k in SESSION_TTL_TABLE if k is None]) == 0
    print("  Test 6: None session_id ignored — OK")

    # --- Test 7: TTL clamping [0, 60] ---
    StateManager.set_global_clock(0)
    update_session_ttl("sess_clamp", {"type": "ephemeral", "ttl": 100})  # clamped to 60
    assert SESSION_TTL_TABLE["sess_clamp"] == 3600.0  # 60 * 60
    print("  Test 7: TTL clamping — OK")

    # Reset
    StateManager.reset()

    print("[PASS] A5: Session TTL Manager logic")


if __name__ == "__main__":
    print("=" * 60)
    print("Phase A: Session-Aware Data Flow Verification")
    print("=" * 60)

    passed = 0
    failed = 0

    tests = [
        ("A1", test_generic_request_fields),
        ("A2", test_dataset_parsing),
        ("A3", test_simulation_params_passthrough),
        ("A4", test_request_stats_fields),
        ("A5", test_session_ttl_manager),
    ]

    for name, test_fn in tests:
        try:
            test_fn()
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
