"""End-to-End Verification Guide for Session-Aware KV Cache Feature.

This script does NOT run the simulation — it provides instructions and
generates the test data needed for full end-to-end verification.

PREREQUISITES:
  - Phase A code changes (A1-A5) installed
  - Phase B code changes (B1-B5) implemented and installed
  - HiSim development environment set up (aiconfigurator, sglang, etc.)

USAGE:
  1. Run this script to generate test trace data:
       python e2e_verification_guide.py

  2. Run HiSim simulation with each trace file:
       cd hisim
       python -m hisim.simulation.sglang.sglang_bench \
         --model_path <your_model> \
         --dataset hisim_collection \
         --filepath test_session_aware/session_isolation.jsonl \
         [other simulation args]

  3. Check the output files in /tmp/hisim/simulation/:
     - request.jsonl: Should contain session_id, parent_session_id fields
     - metrics.json: Should contain session-related metrics (Phase C)

VERIFICATION SCENARIOS:
================================================================

Scenario 1: Session Isolation (session_isolation.jsonl)
----------------------------------------------------------------
Goal: Same input tokens, different session_id → no cross-session prefix cache hit.

Steps:
  1. Run simulation with session_isolation.jsonl
  2. Check request.jsonl output

Expected results:
  - Request 0 (sess_A, first): cached_tokens = 0 (new prefix)
  - Request 1 (sess_A, second): cached_tokens > 0 (prefix hit within same session)
  - Request 2 (sess_B, first): cached_tokens = 0 (no cross-session hit)
  - Request 3 (no session): cached_tokens = 0 (no hit in default partition)

How to check:
  python -c "
  import json
  with open('/tmp/hisim/simulation/request.jsonl') as f:
      for line in f:
          r = json.loads(line)
          print(f'rid={r.get(\"rid\")}, session_id={r.get(\"session_id\")}, '
                f'cached_tokens={r.get(\"final_reused_tokens\", 0)}')
  "


Scenario 2: TTL Protection (session_ttl_protection.jsonl)
----------------------------------------------------------------
Goal: Session KV cache is not evicted during TTL period, even under memory pressure.

NOTE: Requires Phase B (C_RadixCache eviction hook).

Steps:
  1. Run simulation with small total_num_tokens to create eviction pressure
  2. Check that sess_A's cache survives while unprotected requests' cache is evicted

Expected results:
  - sess_A's cached_tokens in later requests > 0 (protected by TTL)
  - Unprotected requests' cache evicted to make room


Scenario 3: TTL Refresh (session_ttl_refresh.jsonl)
----------------------------------------------------------------
Goal: Each request refreshes the session's TTL deadline. Last TTL wins.

Steps:
  1. Run simulation with session_ttl_refresh.jsonl
  2. Verify protection extends beyond original deadline

Expected results:
  - Request at t=200 should find sess_A still protected
  - TTL deadline should be 280s (100 + 180) not 120s (0 + 120)
  - SESSION_TTL_TABLE logs should show deadline progression: 120 → 110 → 280

How to check (with DEBUG logging enabled):
  - Look for "Session TTL updated" log messages
  - Verify deadlines match the expected progression
================================================================
"""

import json
import os

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Shared prefix tokens for isolation testing
SHARED_PREFIX = list(range(100, 200))


def make_request(input_ids, output_length, created_time, session_id=None,
                 parent_session_id=None, cache_control=None):
    req = {
        "input_ids": input_ids,
        "output_length": output_length,
        "created_time": created_time,
    }
    if session_id is not None:
        req["session_id"] = session_id
    if parent_session_id is not None:
        req["parent_session_id"] = parent_session_id
    if cache_control is not None:
        req["cache_control"] = cache_control
    return req


def generate_all_traces():
    """Generate all test trace files."""

    # Scenario 1: Session Isolation
    isolation_requests = [
        make_request(SHARED_PREFIX + [300, 301, 302], 10, 0.0,
                     session_id="sess_A", cache_control={"type": "ephemeral", "ttl": 30}),
        make_request(SHARED_PREFIX + [303, 304], 10, 1.0,
                     session_id="sess_A", cache_control={"type": "ephemeral", "ttl": 30}),
        make_request(SHARED_PREFIX + [305, 306], 10, 2.0,
                     session_id="sess_B", cache_control={"type": "ephemeral", "ttl": 30}),
        make_request(SHARED_PREFIX + [307, 308], 10, 3.0),
    ]
    filepath = os.path.join(OUTPUT_DIR, "session_isolation.jsonl")
    with open(filepath, "w") as f:
        for req in isolation_requests:
            f.write(json.dumps(req) + "\n")
    print(f"Generated: {filepath} ({len(isolation_requests)} requests)")

    # Scenario 2: TTL Protection
    ttl_requests = [
        make_request(list(range(100, 200)), 10, 0.0,
                     session_id="sess_A", cache_control={"type": "ephemeral", "ttl": 1}),
        make_request(list(range(500, 800)), 5, 30.0),
        make_request(list(range(800, 1100)), 5, 30.1),
        make_request(list(range(1100, 1400)), 5, 30.2),
        make_request(list(range(100, 200)), 10, 120.0,
                     session_id="sess_A", cache_control={"type": "ephemeral", "ttl": 1}),
    ]
    filepath = os.path.join(OUTPUT_DIR, "session_ttl_protection.jsonl")
    with open(filepath, "w") as f:
        for req in ttl_requests:
            f.write(json.dumps(req) + "\n")
    print(f"Generated: {filepath} ({len(ttl_requests)} requests)")

    # Scenario 3: TTL Refresh
    refresh_requests = [
        make_request(list(range(100, 200)), 10, 0.0,
                     session_id="sess_A", cache_control={"type": "ephemeral", "ttl": 2}),
        make_request(list(range(200, 300)), 10, 50.0,
                     session_id="sess_A", cache_control={"type": "ephemeral", "ttl": 1}),
        make_request(list(range(300, 400)), 10, 100.0,
                     session_id="sess_A", cache_control={"type": "ephemeral", "ttl": 3}),
        make_request(list(range(400, 500)), 10, 200.0,
                     session_id="sess_A", cache_control={"type": "ephemeral", "ttl": 1}),
    ]
    filepath = os.path.join(OUTPUT_DIR, "session_ttl_refresh.jsonl")
    with open(filepath, "w") as f:
        for req in refresh_requests:
            f.write(json.dumps(req) + "\n")
    print(f"Generated: {filepath} ({len(refresh_requests)} requests)")

    print("\nAll test trace files generated.")


if __name__ == "__main__":
    generate_all_traces()
