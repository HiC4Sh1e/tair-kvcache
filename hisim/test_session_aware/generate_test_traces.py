"""Generate test trace JSONL files for session-aware KV cache verification.

Produces 3 trace files for different verification scenarios:
  1. session_isolation.jsonl      - Verify cache isolation by session_id
  2. session_ttl_protection.jsonl - Verify TTL-protected cache not evicted
  3. session_ttl_refresh.jsonl    - Verify TTL refresh (last TTL wins)

Each line is a JSON object with fields:
  input_ids, output_length, created_time (or timestamp),
  session_id, parent_session_id, cache_control
"""

import json
import os
import sys

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Shared prefix tokens (100 tokens) — same across requests for prefix hit testing
SHARED_PREFIX = list(range(100, 200))  # 100 tokens


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


def generate_isolation_trace():
    """Scenario 1: Session Isolation

    Two sessions (sess_A, sess_B) with identical input tokens.
    Without session_id → RadixCache would share prefix → cached_tokens > 0.
    With session_id → extra_key partitions → no cross-session cache hit.

    Timeline:
      t=0:   sess_A request 1 (shared prefix + suffix_A) → prefill, cached=0
      t=1:   sess_A request 2 (shared prefix + suffix_A2) → should hit prefix in sess_A's partition
      t=2:   sess_B request 1 (shared prefix + suffix_B) → should NOT hit sess_A's prefix (different extra_key)
      t=3:   No-session request (shared prefix + suffix_C) → should NOT hit any session's prefix (no extra_key)
    """
    requests = [
        # sess_A: first request — builds prefix cache under extra_key="sess_A"
        make_request(
            input_ids=SHARED_PREFIX + [300, 301, 302],  # suffix_A
            output_length=10,
            created_time=0.0,
            session_id="sess_A",
            cache_control={"type": "ephemeral", "ttl": 30},
        ),
        # sess_A: second request — should reuse prefix within same session
        make_request(
            input_ids=SHARED_PREFIX + [303, 304],  # different suffix, same prefix
            output_length=10,
            created_time=1.0,
            session_id="sess_A",
            cache_control={"type": "ephemeral", "ttl": 30},
        ),
        # sess_B: first request — same tokens but different session → no prefix hit
        make_request(
            input_ids=SHARED_PREFIX + [305, 306],  # suffix_B
            output_length=10,
            created_time=2.0,
            session_id="sess_B",
            cache_control={"type": "ephemeral", "ttl": 30},
        ),
        # No session — same tokens but no session_id → default partition (extra_key=None)
        make_request(
            input_ids=SHARED_PREFIX + [307, 308],  # suffix_C
            output_length=10,
            created_time=3.0,
        ),
    ]

    filepath = os.path.join(OUTPUT_DIR, "session_isolation.jsonl")
    with open(filepath, "w") as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")
    print(f"Generated: {filepath} ({len(requests)} requests)")


def generate_ttl_protection_trace():
    """Scenario 2: TTL Protection

    Verify that session KV cache is protected from eviction during TTL period.

    Timeline (virtual seconds):
      t=0:     sess_A request with TTL=1min (60s) → deadline at t=60
      t=30:    Fill up cache to trigger eviction pressure
               (many large requests without session → should be evicted first)
      t=60+:   After TTL expires, sess_A cache should become evictable

    NOTE: This scenario requires Phase B (C_RadixCache eviction hook) to work.
    Without the hook, eviction happens in SGLang's original RadixCache.evict()
    which ignores session TTL protection.
    """
    requests = [
        # sess_A: protected session with TTL=1min
        make_request(
            input_ids=list(range(100, 200)),  # 100 tokens
            output_length=10,
            created_time=0.0,
            session_id="sess_A",
            cache_control={"type": "ephemeral", "ttl": 1},  # 1 minute TTL
        ),
        # Unprotected requests that fill up cache (large input)
        # These should be evicted first when under memory pressure
        make_request(
            input_ids=list(range(500, 800)),  # 300 tokens each
            output_length=5,
            created_time=30.0,
        ),
        make_request(
            input_ids=list(range(800, 1100)),  # 300 tokens
            output_length=5,
            created_time=30.1,
        ),
        make_request(
            input_ids=list(range(1100, 1400)),  # 300 tokens
            output_length=5,
            created_time=30.2,
        ),
        # After TTL expiry: new request to sess_A
        # Cache should be evictable now (assuming clock > 60s)
        make_request(
            input_ids=list(range(100, 200)),
            output_length=10,
            created_time=120.0,
            session_id="sess_A",
            cache_control={"type": "ephemeral", "ttl": 1},
        ),
    ]

    filepath = os.path.join(OUTPUT_DIR, "session_ttl_protection.jsonl")
    with open(filepath, "w") as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")
    print(f"Generated: {filepath} ({len(requests)} requests)")


def generate_ttl_refresh_trace():
    """Scenario 3: TTL Refresh (Last TTL Wins)

    Verify that each request to the same session refreshes the TTL deadline.

    Timeline (virtual seconds):
      t=0:    sess_A request with TTL=2min → deadline at t=120
      t=50:   sess_A request with TTL=1min → deadline refreshed to t=110
      t=100:  sess_A request with TTL=3min → deadline refreshed to t=280

    The last TTL should win, so protection extends beyond the original deadline.
    """
    requests = [
        # First request: TTL=2min → deadline at 0 + 120 = 120s
        make_request(
            input_ids=list(range(100, 200)),
            output_length=10,
            created_time=0.0,
            session_id="sess_A",
            cache_control={"type": "ephemeral", "ttl": 2},
        ),
        # Second request: TTL=1min → deadline refreshed to 50 + 60 = 110s
        make_request(
            input_ids=list(range(200, 300)),
            output_length=10,
            created_time=50.0,
            session_id="sess_A",
            cache_control={"type": "ephemeral", "ttl": 1},
        ),
        # Third request: TTL=3min → deadline refreshed to 100 + 180 = 280s
        make_request(
            input_ids=list(range(300, 400)),
            output_length=10,
            created_time=100.0,
            session_id="sess_A",
            cache_control={"type": "ephemeral", "ttl": 3},
        ),
        # Check at t=200: should still be protected (deadline=280 > 200)
        make_request(
            input_ids=list(range(400, 500)),
            output_length=10,
            created_time=200.0,
            session_id="sess_A",
            cache_control={"type": "ephemeral", "ttl": 1},
        ),
    ]

    filepath = os.path.join(OUTPUT_DIR, "session_ttl_refresh.jsonl")
    with open(filepath, "w") as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")
    print(f"Generated: {filepath} ({len(requests)} requests)")


if __name__ == "__main__":
    generate_isolation_trace()
    generate_ttl_protection_trace()
    generate_ttl_refresh_trace()
    print("\nAll test trace files generated successfully.")
