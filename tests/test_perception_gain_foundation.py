from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch


def _foundation_module():
    try:
        return importlib.import_module("scripts.perception_gain_foundation")
    except ModuleNotFoundError as error:
        pytest.fail(f"perception foundation module is unavailable: {error}")


@dataclass(frozen=True)
class _Master:
    reference_id: str
    sequence_id: str


def test_r1_binding_uses_resolved_config_and_local_t2_provenance() -> None:
    foundation = _foundation_module()
    resolved = {"general": {"rootcause_objective_mode": "raw_sum"}}
    config_sha = foundation.canonical_json_sha256(resolved)
    selected = {
        "checkpoint": {
            "bytes": 754_813_672,
            "config_sha256": config_sha,
            "reference": "external:r1",
            "selected_epoch": 390,
            "selected_step": 25_740,
            "sha256": "a" * 64,
        }
    }
    variants = {
        "variants": {"R1": {"config_sha256": config_sha, "resolved_config": resolved}}
    }
    evaluation = {
        "content_sha256": "b" * 64,
        "run_sources": {"seed45.json": {"bytes": 12, "sha256": "c" * 64}},
    }

    binding = foundation.build_r1_binding(
        selected_checkpoint=selected,
        variant_manifest=variants,
        evaluation_provenance=evaluation,
        local_t2_references=("ref-b", "ref-a"),
    )

    assert binding["status"] == "PASS"
    assert binding["r1"]["resolved_config_sha256"] == config_sha
    assert binding["r1"]["objective_mode"] == "raw_sum"
    assert binding["local_t2"]["physical_reference_ids"] == ["ref-a", "ref-b"]
    assert binding["local_t2"]["validation_sequence_count"] == 154


def test_cal_panel_is_first_hash_ordered_sequence_per_reference() -> None:
    foundation = _foundation_module()
    masters = (
        _Master("b", "b-2"),
        _Master("a", "a-2"),
        _Master("b", "b-1"),
        _Master("a", "a-1"),
        _Master("outside", "x-1"),
    )

    selected = foundation.select_cal_panel_masters(masters, ("a", "b"))

    assert {master.reference_id for master in selected} == {"a", "b"}
    assert len(selected) == 2
    for master in selected:
        expected = min(
            (value for value in masters if value.reference_id == master.reference_id),
            key=lambda value: foundation.foundation_hash_order(
                value.reference_id, value.sequence_id
            ),
        )
        assert master == expected


def _supplement(score: float = 0.5):
    return {
        "stages": [
            {
                "observation": {
                    "features": torch.tensor([[1.0, 2.0]]),
                    "valid": torch.tensor([[True]]),
                },
                "prediction": {
                    "pred_masks": torch.tensor([[True]]),
                    "pred_scores": torch.tensor([score]),
                    "pred_classes": torch.tensor([4]),
                    "source_query_ids": torch.tensor([3]),
                    "source_class_ids": torch.tensor([4]),
                },
            }
        ]
    }


def test_live_cached_parity_is_tensor_exact_and_reports_mismatch_path() -> None:
    foundation = _foundation_module()

    passed = foundation.compare_live_cached_supplements(_supplement(), _supplement())
    failed = foundation.compare_live_cached_supplements(
        _supplement(), _supplement(score=0.5001)
    )

    assert passed == {
        "status": "PASS",
        "compared_tensor_count": 7,
        "mismatches": [],
    }
    assert failed["status"] == "FAIL"
    assert failed["mismatches"] == ["stages[0].prediction.pred_scores"]


def test_parity_rejects_different_stage_coverage() -> None:
    foundation = _foundation_module()
    with pytest.raises(foundation.FoundationError, match="stage coverage"):
        foundation.compare_live_cached_supplements(_supplement(), {"stages": []})


def test_best_candidate_iou_is_reported_for_every_decoder_stage_and_gt() -> None:
    foundation = _foundation_module()
    target = {
        "masks": torch.tensor([[True, True, False, False], [False, False, True, True]]),
        "point2segment": torch.arange(4),
        "ids": torch.tensor([11, 22]),
        "labels": torch.tensor([1, 2]),
    }
    first = {
        "pred_masks": [
            torch.tensor([[8.0, -8.0], [8.0, -8.0], [-8.0, 8.0], [-8.0, -8.0]])
        ],
        "pred_logits": torch.zeros(1, 2, 3),
    }
    final = {
        "pred_masks": [
            torch.tensor([[8.0, -8.0], [8.0, -8.0], [-8.0, 8.0], [-8.0, 8.0]])
        ],
        "pred_logits": torch.zeros(1, 2, 3),
        "aux_outputs": [first],
    }

    rows = foundation.best_candidate_iou_records(
        file_name="ref/sequence/T2", output=final, target=target
    )

    assert [
        (row["decoder_prediction_layer"], row["gt_instance_id"]) for row in rows
    ] == [
        (0, 11),
        (0, 22),
        (1, 11),
        (1, 22),
    ]
    assert [row["best_candidate_iou"] for row in rows] == pytest.approx(
        [1.0, 0.5, 1.0, 1.0]
    )
    assert [row["best_candidate_query_id"] for row in rows] == [0, 1, 0, 1]


def test_complete_asset_locator_does_not_probe_legacy_nfs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    foundation = _foundation_module()
    assets = {
        "r1_checkpoint": "r1",
        "concerto_pretrained": "concerto",
        "data_root": "data",
        "rio_metadata": "metadata",
        "metric_dataset_spec": "metric",
        "dev_base_cache_root": "base",
        "dev_supplement_root": "supplement",
    }
    path = tmp_path / "assets.local.json"
    path.write_text(json.dumps(assets), encoding="utf-8")

    class _UnavailableLegacy:
        def is_file(self) -> bool:
            raise AssertionError("legacy NFS must not be probed")

    monkeypatch.setattr(foundation, "LEGACY_CROSSWINDOW_ASSETS", _UnavailableLegacy())

    assert foundation._resolve_cache_assets(path) == assets
