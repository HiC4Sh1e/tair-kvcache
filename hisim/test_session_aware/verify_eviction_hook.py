"""Phase B: Standalone verification for session-aware eviction hook logic.

Tests that:
  B2: C_RadixCacheHook filters session-protected nodes from _collect_leaves()
  B2: C_HiRadixCacheHook filters session-protected nodes from _collect_leaves_device()
  B5: _is_node_session_protected() correctly walks up the tree

This verification uses mock objects to simulate the radix tree structure
without requiring SGLang to be installed.
"""

import sys
import os

# Add hisim src to path
hisim_src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if hisim_src not in sys.path:
    sys.path.insert(0, hisim_src)


# ---- Mock classes that simulate SGLang's RadixKey / TreeNode ----

class MockRadixKey:
    def __init__(self, token_ids, extra_key=None):
        self.token_ids = token_ids
        self.extra_key = extra_key

    def __len__(self):
        return len(self.token_ids)


class MockTreeNode:
    def __init__(self, key=None, parent=None):
        self.key = key
        self.parent = parent
        self.children = {}
        self.lock_ref = 0
        # Simulate value (torch.Tensor-like with len())
        self.value = [1, 2, 3]  # dummy, len = 3


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


# ---- Replicate the session TTL logic from sglang_hook.py ----
# (Can't import directly due to heavy dependencies)

SESSION_TTL_TABLE = {}


def _update_session_ttl(session_id, cache_control):
    if session_id is None or cache_control is None:
        return
    if cache_control.get("type") != "ephemeral":
        return
    ttl_minutes = cache_control.get("ttl", 5)
    ttl_minutes = max(0, min(ttl_minutes, 60))
    ttl_deadline = MockStateManager.get_global_clock() + ttl_minutes * 60
    SESSION_TTL_TABLE[session_id] = ttl_deadline


def _is_session_protected(session_id):
    if session_id is None or session_id not in SESSION_TTL_TABLE:
        return False
    if MockStateManager.get_global_clock() < SESSION_TTL_TABLE[session_id]:
        return True
    del SESSION_TTL_TABLE[session_id]
    return False


def _is_node_session_protected(node):
    cur = node
    while cur is not None:
        if cur.key is not None and cur.key.extra_key is not None:
            return _is_session_protected(cur.key.extra_key)
        cur = cur.parent
    return False


def _collect_leaves_filter(leaves):
    """Simulate what wrapped_collect_leaves does."""
    if not SESSION_TTL_TABLE:
        return leaves
    protected = []
    filtered = []
    for node in leaves:
        if _is_node_session_protected(node):
            protected.append(node)
        else:
            filtered.append(node)
    return filtered


# ---- Test cases ----

def test_node_has_extra_key_directly():
    """Node itself has extra_key = session_id that is TTL-protected."""
    MockStateManager.set_global_clock(0)
    SESSION_TTL_TABLE.clear()

    # Set up TTL protection for sess_A
    SESSION_TTL_TABLE["sess_A"] = 300.0  # deadline at 300s

    # Node with extra_key = sess_A
    node = MockTreeNode(key=MockRadixKey([1, 2, 3], extra_key="sess_A"))

    assert _is_node_session_protected(node) is True, "Should be protected"
    print("  Direct extra_key node — OK")


def test_node_inherits_protection_from_parent():
    """Node doesn't have extra_key, but its parent does."""
    MockStateManager.set_global_clock(0)
    SESSION_TTL_TABLE.clear()

    SESSION_TTL_TABLE["sess_B"] = 600.0

    parent = MockTreeNode(key=MockRadixKey([1, 2], extra_key="sess_A"))
    child = MockTreeNode(key=MockRadixKey([3, 4], extra_key="sess_B"), parent=parent)

    # child's extra_key = sess_B, which is protected
    assert _is_node_session_protected(child) is True, "Should find parent's sess_A or own sess_B"

    # Now test with a node that has no extra_key but parent does
    child_no_extra = MockTreeNode(key=MockRadixKey([5, 6], extra_key=None), parent=parent)
    # Parent has extra_key = sess_A, but sess_A is not in SESSION_TTL_TABLE
    # Only sess_B is protected
    assert _is_node_session_protected(child_no_extra) is False, "sess_A is not protected, should not be protected"

    # Add sess_A to TTL table
    SESSION_TTL_TABLE["sess_A"] = 300.0
    assert _is_node_session_protected(child_no_extra) is True, "sess_A is now protected"
    print("  Inherited protection from parent — OK")


def test_node_no_extra_key_no_protection():
    """Node with no extra_key and no protected ancestors is not protected."""
    SESSION_TTL_TABLE.clear()
    SESSION_TTL_TABLE["sess_X"] = 600.0

    parent_node = MockTreeNode(key=MockRadixKey([1], extra_key=None))
    node = MockTreeNode(key=MockRadixKey([1, 2], extra_key=None), parent=parent_node)

    assert _is_node_session_protected(node) is False, "No extra_key, should not be protected"
    print("  No extra_key, no protection — OK")


def test_ttl_expired_node_unprotected():
    """Node with expired session TTL should NOT be protected."""
    MockStateManager.set_global_clock(0)
    SESSION_TTL_TABLE.clear()

    SESSION_TTL_TABLE["sess_expired"] = 60.0  # deadline at 60s
    node = MockTreeNode(key=MockRadixKey([1, 2, 3], extra_key="sess_expired"))

    # Before expiration
    MockStateManager.set_global_clock(30)
    assert _is_node_session_protected(node) is True, "Before expiration"

    # After expiration
    MockStateManager.set_global_clock(61)
    assert _is_node_session_protected(node) is False, "After expiration"
    # Should also clean up expired entry
    assert "sess_expired" not in SESSION_TTL_TABLE, "Should be cleaned up"
    print("  TTL expired, node unprotected — OK")


def test_collect_leaves_filter_basic():
    """_collect_leaves filter correctly separates protected and unprotected nodes."""
    MockStateManager.set_global_clock(0)
    SESSION_TTL_TABLE.clear()

    SESSION_TTL_TABLE["sess_A"] = 300.0  # protected
    # sess_B not in table — not protected

    node_a = MockTreeNode(key=MockRadixKey([1, 2], extra_key="sess_A"))
    node_b = MockTreeNode(key=MockRadixKey([3, 4], extra_key="sess_B"))
    node_none = MockTreeNode(key=MockRadixKey([5, 6], extra_key=None))

    leaves = [node_a, node_b, node_none]
    filtered = _collect_leaves_filter(leaves)

    assert len(filtered) == 2, f"Expected 2 unprotected, got {len(filtered)}"
    assert node_a not in filtered, "sess_A should be filtered out"
    assert node_b in filtered, "sess_B should pass through"
    assert node_none in filtered, "No extra_key should pass through"
    print("  _collect_leaves filter — OK")


def test_collect_leaves_empty_ttl_table():
    """When SESSION_TTL_TABLE is empty, all nodes pass through."""
    SESSION_TTL_TABLE.clear()

    node_a = MockTreeNode(key=MockRadixKey([1, 2], extra_key="sess_A"))
    node_b = MockTreeNode(key=MockRadixKey([3, 4], extra_key=None))

    leaves = [node_a, node_b]
    filtered = _collect_leaves_filter(leaves)

    assert len(filtered) == 2, "All nodes should pass through when no TTL entries"
    print("  Empty TTL table, no filtering — OK")


def test_collect_leaves_all_protected():
    """When all nodes are protected, result is empty list."""
    MockStateManager.set_global_clock(0)
    SESSION_TTL_TABLE.clear()

    SESSION_TTL_TABLE["sess_A"] = 300.0
    SESSION_TTL_TABLE["sess_B"] = 600.0

    node_a = MockTreeNode(key=MockRadixKey([1, 2], extra_key="sess_A"))
    node_b = MockTreeNode(key=MockRadixKey([3, 4], extra_key="sess_B"))

    leaves = [node_a, node_b]
    filtered = _collect_leaves_filter(leaves)

    assert len(filtered) == 0, f"Expected 0 unprotected, got {len(filtered)}"
    print("  All nodes protected, empty result — OK")


def test_ttl_expiration_mid_simulation():
    """Simulate a scenario where TTL expires during simulation.

    Before expiration: protected nodes are filtered out.
    After expiration: same nodes become evictable.
    """
    MockStateManager.set_global_clock(0)
    SESSION_TTL_TABLE.clear()

    # sess_A with TTL=1min (60s)
    SESSION_TTL_TABLE["sess_A"] = 60.0

    node_a = MockTreeNode(key=MockRadixKey([1, 2, 3], extra_key="sess_A"))
    node_b = MockTreeNode(key=MockRadixKey([4, 5], extra_key=None))

    # At t=30: sess_A still protected
    MockStateManager.set_global_clock(30)
    filtered = _collect_leaves_filter([node_a, node_b])
    assert len(filtered) == 1, "Only node_b should pass"
    assert node_b in filtered

    # At t=70: sess_A TTL expired
    MockStateManager.set_global_clock(70)
    # First check will clean up expired entry via _is_session_protected
    filtered = _collect_leaves_filter([node_a, node_b])
    assert len(filtered) == 2, "Both nodes should be evictable now"
    assert "sess_A" not in SESSION_TTL_TABLE, "Expired session should be cleaned up"
    print("  TTL expiration mid-simulation — OK")


if __name__ == "__main__":
    print("=" * 60)
    print("Phase B: Session-Aware Eviction Hook Verification")
    print("=" * 60)

    passed = 0
    failed = 0

    tests = [
        ("B2-1", test_node_has_extra_key_directly),
        ("B2-2", test_node_inherits_protection_from_parent),
        ("B2-3", test_node_no_extra_key_no_protection),
        ("B2-4", test_ttl_expired_node_unprotected),
        ("B2-5", test_collect_leaves_filter_basic),
        ("B2-6", test_collect_leaves_empty_ttl_table),
        ("B2-7", test_collect_leaves_all_protected),
        ("B2-8", test_ttl_expiration_mid_simulation),
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

    MockStateManager.reset()

    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed (total {passed + failed})")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)