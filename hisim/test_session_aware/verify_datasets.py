"""D1: Verify that test dataset files can be parsed by HisimCollectionDataset.

This test does NOT require SGLang or aiconfigurator. It only verifies
that the JSONL files in test/assets/dataset/ can be loaded and that
session fields are correctly parsed.
"""

import json
import os
import sys
import tempfile

hisim_src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if hisim_src not in sys.path:
    sys.path.insert(0, hisim_src)

from hisim.dataset.base_dataset import GenericRequest
from hisim.dataset.dataset_args import DatasetArgs


def test_isolation_dataset():
    """Verify session_aware_isolation.jsonl can be loaded with session fields."""
    from hisim.dataset.hisim_collection import HisimCollectionDataset

    filepath = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "test", "assets", "dataset", "session_aware_isolation.jsonl"
    )

    if not os.path.exists(filepath):
        # Check the test_session_aware directory too
        filepath = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "session_isolation.jsonl"
        )

    assert os.path.exists(filepath), f"Dataset file not found: {filepath}"

    args = DatasetArgs("hisim_collection", filepath=filepath)
    dataset = HisimCollectionDataset(tokenizer=None, args=args)
    assert len(dataset) == 4, f"Expected 4 requests, got {len(dataset)}"

    # Verify session fields
    req0 = dataset[0]
    assert req0.session_id == "sess_A", f"Request 0: session_id should be sess_A, got {req0.session_id}"
    assert req0.cache_control is not None, "Request 0: cache_control should not be None"
    assert req0.cache_control["type"] == "ephemeral" or req0.cache_control["type"] == "ephemeral", f"Request 0: cache_control type={req0.cache_control['type']}"

    req2 = dataset[2]
    assert req2.session_id == "sess_B", f"Request 2: session_id should be sess_B, got {req2.session_id}"

    req3 = dataset[3]
    assert req3.session_id is None, f"Request 3: session_id should be None, got {req3.session_id}"
    assert req3.cache_control is None, f"Request 3: cache_control should be None"

    print(f"  Isolation dataset: {len(dataset)} requests, session fields OK")


def test_ttl_dataset():
    """Verify session_aware_ttl.jsonl can be loaded with session fields."""
    from hisim.dataset.hisim_collection import HisimCollectionDataset

    filepath = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "test", "assets", "dataset", "session_aware_ttl.jsonl"
    )

    if not os.path.exists(filepath):
        filepath = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "session_ttl_refresh.jsonl"
        )

    assert os.path.exists(filepath), f"Dataset file not found: {filepath}"

    args = DatasetArgs("hisim_collection", filepath=filepath)
    dataset = HisimCollectionDataset(tokenizer=None, args=args)
    assert len(dataset) == 4, f"Expected 4 requests, got {len(dataset)}"

    # All requests should belong to sess_A
    for i, req in enumerate(dataset):
        assert req.session_id == "sess_A", f"Request {i}: session_id should be sess_A, got {req.session_id}"
        assert req.cache_control is not None, f"Request {i}: cache_control should not be None"

    # Verify TTL values differ
    ttls = [req.cache_control["ttl"] for req in dataset]
    assert ttls == [2, 1, 3, 1], f"Expected TTLs [2,1,3,1], got {ttls}"

    print(f"  TTL dataset: {len(dataset)} requests, all sess_A, TTLs={ttls}")


def test_raw_json_validation():
    """Validate raw JSON content of test dataset files."""
    datasets = {
        "isolation": os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "test", "assets", "dataset", "session_aware_isolation.jsonl"
        ),
        "ttl": os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "test", "assets", "dataset", "session_aware_ttl.jsonl"
        ),
    }

    for name, filepath in datasets.items():
        if not os.path.exists(filepath):
            print(f"  SKIP {name}: file not found at {filepath}")
            continue

        with open(filepath) as f:
            lines = f.readlines()

        assert len(lines) > 0, f"{name}: empty dataset file"
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            assert "input_ids" in data, f"{name} line {i}: missing input_ids"
            assert "output_length" in data, f"{name} line {i}: missing output_length"
            assert "created_time" in data, f"{name} line {i}: missing created_time"
            assert isinstance(data["input_ids"], list), f"{name} line {i}: input_ids not a list"
            assert data["output_length"] > 0, f"{name} line {i}: output_length must be > 0"

        print(f"  {name}: {len(lines)} lines, all valid JSON")


if __name__ == "__main__":
    print("=" * 60)
    print("D1: Test Dataset Verification")
    print("=" * 60)

    passed = 0
    failed = 0

    tests = [
        ("D1-1", test_isolation_dataset),
        ("D1-2", test_ttl_dataset),
        ("D1-3", test_raw_json_validation),
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