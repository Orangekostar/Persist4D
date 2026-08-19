from __future__ import annotations

import os
from pathlib import Path

import pytest

from scripts.p6b_artifacts import validate_p6b_artifact


@pytest.mark.skipif(
    os.environ.get("P6B_VERIFY_REAL_CACHE") != "1",
    reason="set P6B_VERIFY_REAL_CACHE=1 after publishing real P6-B evidence",
)
def test_real_p6b_artifact_is_source_bound_and_schema_valid() -> None:
    root = Path("artifacts/P6B/p6b_eval.json")
    assert root.is_file() and not root.is_symlink()
    import json

    payload = json.loads(root.read_text(encoding="utf-8"))
    validate_p6b_artifact(payload)
    assert payload["source_tree_contract"]["status"] == "pass"
    assert payload["decision"] in {"P6B_GO", "P6B_STOP"}
