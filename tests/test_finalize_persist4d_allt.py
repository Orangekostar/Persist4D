from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path

import pytest

from scripts.finalize_persist4d_allt import (
    FinalizationError,
    build_delta_rows,
    build_verdict,
    validate_evaluation_artifacts,
)

METRICS = (
    "t_mAP",
    "t_mAP50",
    "t_mAP25",
    "t_REC",
    "prefix_overall_mAP",
)


def _rows(checkpoint: str, base: float):
    return [
        {
            "checkpoint_sha256": checkpoint,
            "T": horizon,
            **{metric: base + horizon / 100 for metric in METRICS},
        }
        for horizon in range(2, 6)
    ]


def test_all_t_verdict_requires_every_unrounded_metric_cell_to_be_positive() -> None:
    baseline = _rows("a" * 64, 0.2)
    candidate = _rows("b" * 64, 0.21)
    deltas = build_delta_rows(
        candidate_rows=candidate,
        baseline_rows=baseline,
        candidate="C2",
        baseline="FH-adapt",
        evidence_scope="protocol_b_43_masters_3_orders",
    )

    verdict = build_verdict(deltas)

    assert len(deltas) == 20
    assert verdict["tmap_all_t"] == "PASS"
    assert verdict["task_metrics_all_t"] == "PASS"
    assert verdict["failed_cells"] == []


def test_one_failed_horizon_cannot_be_compensated_by_average_gain() -> None:
    baseline = _rows("a" * 64, 0.2)
    candidate = _rows("b" * 64, 0.21)
    candidate[-1]["t_mAP"] = baseline[-1]["t_mAP"] - 0.001

    verdict = build_verdict(
        build_delta_rows(
            candidate_rows=candidate,
            baseline_rows=baseline,
            candidate="C2",
            baseline="FH-adapt",
            evidence_scope="protocol_b_43_masters_3_orders",
        )
    )

    assert verdict["tmap_all_t"] == "FAIL"
    assert verdict["task_metrics_all_t"] == "FAIL"
    assert verdict["failed_cells"] == [{"metric": "t_mAP", "T": 5}]


def test_delta_builder_rejects_horizon_specific_checkpoint_substitution() -> None:
    candidate = _rows("b" * 64, 0.21)
    candidate[-1]["checkpoint_sha256"] = "c" * 64

    with pytest.raises(FinalizationError, match="one checkpoint"):
        build_delta_rows(
            candidate_rows=candidate,
            baseline_rows=_rows("a" * 64, 0.2),
            candidate="C2",
            baseline="FH-adapt",
            evidence_scope="protocol_b_43_masters_3_orders",
        )


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _write_evaluation_artifacts(root: Path, *, source_commit: str) -> None:
    fieldnames = (
        "population_id",
        "model",
        "checkpoint_sha256",
        "training_seed",
        "evaluation_seed",
        "method",
        "reducer",
        "T",
        *METRICS,
        "local_current_AP",
        "num_master",
        "num_order_units",
        "num_reference_clusters",
    )
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in _rows("b" * 64, 0.21):
        writer.writerow(
            {
                "population_id": "protocol_b_43_masters_3_orders",
                "model": "C2",
                "training_seed": 45,
                "evaluation_seed": 45,
                "method": "B4",
                "reducer": "mean",
                "local_current_AP": 0.3,
                "num_master": 43,
                "num_order_units": 129,
                "num_reference_clusters": 6,
                **row,
            }
        )
    metrics = buffer.getvalue().encode("ascii")
    root.mkdir(parents=True, exist_ok=True)
    (root / "all_t_metrics.csv").write_bytes(metrics)
    (root / "run_summary.json").write_text(
        json.dumps(
            {
                "checkpoint_global_step": 200,
                "checkpoint_sha256": "b" * 64,
                "evaluation_seed": 45,
                "metric_row_count": 4,
                "metric_sha256": hashlib.sha256(metrics).hexdigest(),
                "model": "C2",
                "model_forward_count": 645,
                "new_sequence_count": 129,
                "population_id": "protocol_b_43_masters_3_orders",
                "reducers": ["mean"],
                "reused_sequence_count": 0,
                "sequence_count": 129,
                "source_commit": source_commit,
                "status": "pass",
                "training_seed": 45,
                "variant": "C2",
                "window_mode": "local_pair",
            }
        ),
        encoding="ascii",
    )
    manifest = {
        "cache_directory": "external:test",
        "checkpoint_sha256": "b" * 64,
        "entry_count": 129,
        "evaluation_seed": 45,
        "module_config": {},
        "module_config_sha256": "c" * 64,
        "population_id": "protocol_b_43_masters_3_orders",
        "population_sha256": "d" * 64,
        "records": [
            {
                "bytes": 1,
                "file_sha256": "e" * 64,
                "filename": f"{index:064x}.pt",
                "key_sha256": f"{index:064x}",
                "master_sequence_id": f"master-{index}",
                "order_id": "canonical",
                "reference_scene_id": "reference",
            }
            for index in range(129)
        ],
        "reused_entry_count": 0,
        "schema_version": 1,
        "score_reducers": ["mean"],
        "source_commit": source_commit,
        "status": "pass",
    }
    manifest["content_sha256"] = _canonical_sha256(manifest)
    (root / "cache_manifest.json").write_text(
        json.dumps(manifest), encoding="ascii"
    )


def test_evaluation_artifacts_bind_frozen_checkpoint_commit_and_coverage(
    tmp_path: Path,
) -> None:
    source_commit = "f" * 40
    _write_evaluation_artifacts(tmp_path, source_commit=source_commit)

    rows = validate_evaluation_artifacts(
        evaluation_root=tmp_path,
        expected_variant="C2",
        expected_checkpoint="b" * 64,
        expected_step=200,
        expected_source_commit=source_commit,
        expected_reducers=("mean",),
    )

    assert len(rows) == 4
    assert rows[0]["T"] == 2
    assert isinstance(rows[0]["t_mAP"], float)


def test_evaluation_artifacts_reject_wrong_source_commit(tmp_path: Path) -> None:
    _write_evaluation_artifacts(tmp_path, source_commit="f" * 40)

    with pytest.raises(FinalizationError, match="source commit"):
        validate_evaluation_artifacts(
            evaluation_root=tmp_path,
            expected_variant="C2",
            expected_checkpoint="b" * 64,
            expected_step=200,
            expected_source_commit="0" * 40,
            expected_reducers=("mean",),
        )
