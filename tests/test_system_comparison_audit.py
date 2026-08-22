from __future__ import annotations

from pathlib import Path

import pytest

from scripts.system_comparison_audit import (
    CodeAuditError,
    build_code_audit,
    render_code_audit_markdown,
    validate_code_audit,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = (
    REPO_ROOT / "artifacts/system_comparison/REScene_FULL_HISTORY_CODE_AUDIT.md"
)


def _checkpoint_payload() -> dict[str, object]:
    return {
        "epoch": 404,
        "global_step": 26730,
        "hyper_parameters": {
            "data": {
                "temporal_window": 2,
                "train_dataset": {
                    "datasets": [
                        {"dataset_name": "rio", "temporal_window": 2},
                        {"dataset_name": "scannet", "temporal_window": 1},
                    ]
                },
                "validation_dataset": {"temporal_window": 2},
                "test_dataset": {"temporal_window": 2},
            },
            "model": {
                "num_queries": 100,
                "non_parametric_queries": True,
                "random_queries": False,
                "random_query_both": False,
                "temporal_masking": False,
                "use_changes_loss": False,
            },
            "backbone": {
                "decoder_serializations": ["standard", "temporal_overlay"]
            },
            "general": {"topk_per_image": 100},
            "trainer": {"deterministic": False},
        },
    }


@pytest.fixture(scope="module")
def audit() -> dict[str, object]:
    return build_code_audit(
        repo_root=REPO_ROOT,
        checkpoint_payload=_checkpoint_payload(),
        checkpoint_sha256="8" * 64,
    )


def test_checkpoint_horizon_is_t2_and_longer_prefixes_are_zero_shot(
    audit: dict[str, object],
) -> None:
    checkpoint = audit["checkpoint"]
    assert checkpoint["data_temporal_window"] == 2
    assert checkpoint["rio_train_temporal_window"] == 2
    assert checkpoint["scannet_train_temporal_window"] == 1
    assert checkpoint["validation_temporal_window"] == 2
    assert checkpoint["test_temporal_window"] == 2
    assert checkpoint["epoch"] == 404
    assert checkpoint["global_step"] == 26730
    assert audit["conclusions"]["formal_method_name"] == (
        "ReScene4D Full-History (Frozen T2 Checkpoint)"
    )
    assert audit["conclusions"]["T3_T5_semantics"] == (
        "zero-shot temporal-horizon extension"
    )


def test_audit_records_t_greater_than_two_and_temporal_sharing_evidence(
    audit: dict[str, object],
) -> None:
    records = {
        (row["file_path"], row["function_or_class"]): row
        for row in audit["evidence"]
    }
    required = {
        ("datasets/semseg.py", "SemanticSegmentationDataset.load_scan_indices"),
        ("datasets/semseg.py", "SemanticSegmentationDataset._load_scan_sequence"),
        ("datasets/pointcept_utils.py", "voxelize"),
        ("models/pointcept.py", "PointceptBackbone.temporal_overlay"),
        ("models/rescene.py", "ReScene.initialize_queries"),
        ("models/rescene.py", "ReScene.forward"),
        ("models/rescene.py", "ReScene.mask_module"),
        ("trainer/trainer.py", "InstanceSegmentation._get_mask_and_scores"),
        (
            "scripts/evaluate_persist4d_p6a.py",
            "RealPredictionCacheProducer.__call__",
        ),
    }
    assert required <= set(records)
    assert all(row["line"] > 0 for row in records.values())
    assert all(row["relevant_behavior"] for row in records.values())
    assert all(row["scientific_implication"] for row in records.values())


def test_identity_evaluator_and_change_label_conclusions_are_explicit(
    audit: dict[str, object],
) -> None:
    conclusions = audit["conclusions"]
    assert conclusions["full_history_accepts_T_gt_2"] is True
    assert conclusions["issued_identity"] == "raw query index within each prefix forward"
    assert conclusions["cross_prefix_namespace"] == "not guaranteed stable"
    assert conclusions["determinism"] == "requires three-repeat empirical gate"
    assert conclusions["future_information"] == "forbidden by exact prefix loading"
    assert conclusions["change_labels"] == (
        "disabled with change_file=None; all-static placeholders are metric-only"
    )


def test_audit_has_all_eight_required_answers_and_renders_markdown(
    audit: dict[str, object],
) -> None:
    validated = validate_code_audit(audit)
    assert [row["id"] for row in validated["answers"]] == [
        "Q1",
        "Q2",
        "Q3",
        "Q4",
        "Q5",
        "Q6",
        "Q7",
        "Q8",
    ]
    markdown = render_code_audit_markdown(validated)
    for question in range(1, 9):
        assert f"## Q{question}." in markdown
    assert "file path" in markdown.lower()
    assert "scientific implication" in markdown.lower()
    assert "zero-shot temporal-horizon extension" in markdown


def test_audit_rejects_claim_without_evidence(audit: dict[str, object]) -> None:
    invalid = dict(audit)
    invalid["evidence"] = []
    with pytest.raises(CodeAuditError, match="evidence"):
        validate_code_audit(invalid)


def test_checked_in_code_audit_is_complete() -> None:
    markdown = AUDIT_PATH.read_text(encoding="utf-8")
    for question in range(1, 9):
        assert f"## Q{question}." in markdown
    assert "ReScene4D Full-History (Frozen T2 Checkpoint)" in markdown
    assert "zero-shot temporal-horizon extension" in markdown
