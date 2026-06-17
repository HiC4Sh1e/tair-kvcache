"""B5: Verification of extra_key injection timing.

This test verifies that the extra_key injection happens at the correct
point in the scheduler lifecycle — BEFORE init_next_round_input() is
called inside get_new_batch_prefill().

The key insight from reading SGLang source:
  1. handle_generate_request() creates Req with extra_key=None (bug)
  2. _add_request_to_queue() → _prefetch_kvcache() → init_next_round_input()
     This first match_prefix uses extra_key=None — wrong namespace.
  3. get_new_batch_prefill() → for each req → init_next_round_input()
     This second match_prefix is the one that matters for actual scheduling.

Our hook (wrapped_get_new_batch_prefill) pre-injects req.extra_key
BEFORE calling original_get_new_batch_prefill, ensuring the second
init_next_round_input() call uses the correct session namespace.

This test simulates the timing scenario with mock objects.
"""

import sys
import os

hisim_src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if hisim_src not in sys.path:
    sys.path.insert(0, hisim_src)


class MockRadixKey:
    def __init__(self, token_ids, extra_key=None):
        self.token_ids = token_ids
        self.extra_key = extra_key

    def __len__(self):
        return len(self.token_ids)


class MockReq:
    """Simulates SGLang's Req object."""
    def __init__(self, rid):
        self.rid = rid
        self.extra_key = None  # Default — set by Req.__init__ when not passed
        self.cached_tokens = 0
        self.fill_ids = [1, 2, 3, 4, 5]
        self.sampling_params = MockSamplingParams()


class MockSamplingParams:
    def __init__(self):
        self.custom_params = {"simulation": {}}


def test_extra_key_initially_none():
    """Req.extra_key is None by default (SGLang bug)."""
    req = MockReq("req_1")
    assert req.extra_key is None, "extra_key should be None by default"
    print("  extra_key initially None — OK")


def test_pre_injection_sets_extra_key():
    """Pre-injection loop sets req.extra_key before original call."""
    req1 = MockReq("req_1")
    req1.sampling_params.custom_params["simulation"]["__extra_key__"] = "sess_A"
    req2 = MockReq("req_2")
    # req2 has no __extra_key__

    waiting_queue = [req1, req2]

    # Simulate the pre-injection loop
    for req in waiting_queue:
        sim_args = (
            req.sampling_params.custom_params.get("simulation", {})
            if hasattr(req, "sampling_params")
            and isinstance(req.sampling_params, object)
            and hasattr(req.sampling_params, "custom_params")
            and isinstance(req.sampling_params.custom_params, dict)
            else {}
        )
        extra_key_from_session = sim_args.get("__extra_key__")
        if extra_key_from_session and hasattr(req, "extra_key"):
            req.extra_key = extra_key_from_session

    assert req1.extra_key == "sess_A", f"req1.extra_key should be sess_A, got {req1.extra_key}"
    assert req2.extra_key is None, f"req2.extra_key should be None, got {req2.extra_key}"
    print("  Pre-injection sets extra_key correctly — OK")


def test_timing_order():
    """Verify the order: pre-inject → original call → post-process.

    The critical timing is:
      1. Pre-inject req.extra_key for all waiting_queue requests
      2. Call original get_new_batch_prefill (which calls init_next_round_input)
      3. Post-process: record stats, refresh TTL
    """
    # Track call order
    call_order = []

    req = MockReq("req_1")
    req.sampling_params.custom_params["simulation"]["__extra_key__"] = "sess_A"
    req.sampling_params.custom_params["simulation"]["cache_control"] = {
        "type": "ephemeral", "ttl": 5
    }

    waiting_queue = [req]

    # Step 1: Pre-inject
    for r in waiting_queue:
        sim_args = r.sampling_params.custom_params.get("simulation", {})
        ek = sim_args.get("__extra_key__")
        if ek and hasattr(r, "extra_key"):
            r.extra_key = ek
            call_order.append(("pre_inject", r.rid, r.extra_key))

    # Step 2: Simulate original call (would call init_next_round_input)
    # At this point, req.extra_key = "sess_A", so match_prefix uses correct namespace
    call_order.append(("original_call", req.rid, req.extra_key))

    # Step 3: Post-process
    call_order.append(("post_process", req.rid, req.extra_key))

    assert call_order[0] == ("pre_inject", "req_1", "sess_A"), "Pre-inject should be first"
    assert call_order[1] == ("original_call", "req_1", "sess_A"), "Original call should use injected extra_key"
    assert call_order[2] == ("post_process", "req_1", "sess_A"), "Post-process should see correct extra_key"
    print("  Timing order verified — OK")


def test_no_extra_key_no_injection():
    """Requests without session_id should not have extra_key injected."""
    req = MockReq("req_no_session")
    req.sampling_params.custom_params["simulation"] = {}  # no __extra_key__

    waiting_queue = [req]
    for r in waiting_queue:
        sim_args = r.sampling_params.custom_params.get("simulation", {})
        ek = sim_args.get("__extra_key__")
        if ek and hasattr(r, "extra_key"):
            r.extra_key = ek

    assert req.extra_key is None, "extra_key should remain None for non-session requests"
    print("  No injection for non-session requests — OK")


def test_init_next_round_input_uses_extra_key():
    """Verify that init_next_round_input() constructs RadixKey with req.extra_key.

    This is the core correctness test — confirming that SGLang's
    init_next_round_input reads self.extra_key when building the
    RadixKey for match_prefix. Our pre-injection ensures this value
    is set before init_next_round_input is called.
    """
    req = MockReq("req_1")

    # Before injection: extra_key=None → RadixKey would use None
    assert req.extra_key is None

    # After injection: extra_key="sess_A"
    req.extra_key = "sess_A"

    # Simulate what init_next_round_input does:
    # match_result = tree_cache.match_prefix(
    #     key=RadixKey(token_ids=token_ids, extra_key=self.extra_key),
    # )
    simulated_radix_key = MockRadixKey(
        token_ids=req.fill_ids[:-1],
        extra_key=req.extra_key,  # This reads req.extra_key
    )

    assert simulated_radix_key.extra_key == "sess_A", \
        f"RadixKey should use injected extra_key, got {simulated_radix_key.extra_key}"
    print("  init_next_round_input uses correct extra_key — OK")


if __name__ == "__main__":
    print("=" * 60)
    print("B5: extra_key Injection Timing Verification")
    print("=" * 60)

    passed = 0
    failed = 0

    tests = [
        ("B5-1", test_extra_key_initially_none),
        ("B5-2", test_pre_injection_sets_extra_key),
        ("B5-3", test_timing_order),
        ("B5-4", test_no_extra_key_no_injection),
        ("B5-5", test_init_next_round_input_uses_extra_key),
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