from __future__ import annotations

import base64
import hashlib
import json
import zlib
from copy import deepcopy

import pytest
import torch

from scripts import p6a_metrics
from scripts.p6a_metrics import (
    IdentityAccumulator,
    OfficialMetricAccumulator,
    adapt_raw_local_pair,
    assert_shared_raw_predictions,
    build_official_metric_population_evidence,
    build_offline_reconstructed_prediction,
    build_online_endpoint_prediction,
    compute_endpoint_metrics,
    compute_official_raw_local_metrics,
    compute_official_temporal_metrics,
    compute_raw_local_metrics,
    global_hungarian_match,
    greedy_diagnostic_match,
    raw_observation_fingerprint,
    recompute_official_metric_evidence,
    recompute_official_metric_population_evidence,
    relative_retention,
)


def _raw_prediction() -> dict[str, torch.Tensor]:
    return {
        "pred_masks": torch.tensor([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=torch.bool),
        "pred_classes": torch.tensor([2, 3]),
        "pred_scores": torch.tensor([0.9, 0.8]),
    }


def _raw_target() -> dict[str, torch.Tensor]:
    return {
        "masks": torch.tensor([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=torch.bool),
        "labels": torch.tensor([2, 3]),
        "ids": torch.tensor([101, 7001]),
        "temporal_stages": torch.tensor([0, 0, 1, 1]),
    }


def _stage_prediction(stage: int, track_ids: list[int], *, flip: bool = False):
    masks = torch.tensor([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=torch.bool)
    if flip:
        masks = masks.flip(1)
    return {
        "stage": stage,
        "pred_masks": masks,
        "pred_classes": torch.tensor([2, 3]),
        "pred_scores": torch.tensor([0.9, 0.8]),
        "track_ids": torch.tensor(track_ids),
    }


def test_raw_observation_fingerprint_is_immutable_and_shared_across_methods():
    prediction = _raw_prediction()
    frozen = deepcopy(prediction)
    digest = raw_observation_fingerprint(prediction)
    prediction["pred_masks"][0, 0] = False

    assert raw_observation_fingerprint(frozen) == digest
    assert (
        assert_shared_raw_predictions(
            {"B0": frozen, "B1": deepcopy(frozen), "B2": frozen}
        )
        == digest
    )
    with pytest.raises(ValueError, match="fingerprint"):
        assert_shared_raw_predictions({"B0": frozen, "B1": prediction})


def test_raw_local_adapter_uses_newest_stage_only_and_preserves_values():
    prediction, target = adapt_raw_local_pair(_raw_prediction(), _raw_target(), stage=1)

    assert prediction["pred_masks"].shape == (2, 2)
    assert torch.equal(
        prediction["pred_masks"],
        torch.tensor([[0, 1], [0, 1]], dtype=torch.bool),
    )
    assert torch.equal(
        target["masks"], torch.tensor([[0, 0], [1, 1]], dtype=torch.bool)
    )
    assert torch.equal(prediction["pred_classes"], torch.tensor([2, 3]))
    assert "track_ids" not in prediction


def test_raw_local_metrics_are_exactly_shared_and_emit_ap_rec_keys():
    raw_prediction = _raw_prediction()
    raw_target = _raw_target()
    methods = {
        name: [(raw_prediction, raw_target)] for name in ("B0", "B1", "B2", "B3", "B4")
    }

    results = {
        name: compute_raw_local_metrics(predictions, targets)
        for name, pairs in methods.items()
        for predictions, targets in [tuple(zip(*pairs))]
    }

    assert all(result == results["B0"] for result in results.values())
    assert {
        "AP",
        "AP50",
        "AP25",
        "REC",
        "REC50",
        "REC25",
    } <= results["B0"].keys()
    assert all(0.0 <= results["B0"][key] <= 1.0 for key in results["B0"])


def test_raw_rec_is_mean_over_official_iou_thresholds_not_rec50_alias():
    prediction = {
        "pred_masks": torch.tensor([[1], [1], [1], [0], [0]], dtype=torch.bool),
        "pred_classes": torch.tensor([2]),
        "pred_scores": torch.tensor([0.9]),
    }
    target = {
        "masks": torch.ones((1, 5), dtype=torch.bool),
        "labels": torch.tensor([2]),
    }

    result = compute_raw_local_metrics([prediction], [target])

    assert result["REC50"] == 1.0
    assert result["REC"] == pytest.approx(3.0 / 9.0)


def test_dynamic_identity_accumulator_and_prefix_endpoint_are_causal():
    accumulator = IdentityAccumulator()
    accumulator.add_stage(_stage_prediction(0, [101, 7001]))
    before_future = build_online_endpoint_prediction(accumulator, endpoint=0)

    accumulator.add_stage(_stage_prediction(1, [7001, 90001], flip=True))
    after_future = build_online_endpoint_prediction(accumulator, endpoint=0)

    assert torch.equal(before_future["pred_masks"], after_future["pred_masks"])
    assert torch.equal(before_future["pred_classes"], after_future["pred_classes"])
    assert torch.equal(before_future["pred_scores"], after_future["pred_scores"])
    assert set(before_future["track_ids"].tolist()) == {101, 7001}
    assert set(
        build_online_endpoint_prediction(accumulator, endpoint=1)["track_ids"].tolist()
    ) == {
        101,
        7001,
        90001,
    }


def test_online_and_offline_reconstruction_are_explicitly_separate():
    accumulator = IdentityAccumulator()
    accumulator.add_stage(_stage_prediction(0, [101, 7001]))
    accumulator.add_stage(_stage_prediction(1, [101, 7001], flip=True))
    target = [_raw_target(), _raw_target()]

    metrics = compute_endpoint_metrics(accumulator, target, endpoint=0)
    assert {
        "online_t-mAP",
        "online_t-mAP50",
        "online_t-mAP25",
        "online_t-REC",
        "online_t-REC50",
        "online_t-REC25",
        "offline_reconstructed_t-mAP",
        "offline_reconstructed_t-mAP50",
        "offline_reconstructed_t-mAP25",
        "offline_reconstructed_t-REC",
        "offline_reconstructed_t-REC50",
        "offline_reconstructed_t-REC25",
    } <= metrics.keys()
    assert 0.0 <= metrics["online_t-mAP"] <= 1.0
    assert 0.0 <= metrics["offline_reconstructed_t-mAP"] <= 1.0
    assert build_offline_reconstructed_prediction(accumulator)["track_ids"].numel() == 2


def test_hungarian_is_cardinality_first_then_iou_and_beats_greedy_counterexample():
    ious = torch.tensor([[0.60, 0.55], [0.50, 0.10]])

    assert greedy_diagnostic_match(ious, threshold=0.5) == [(0, 0)]
    assert global_hungarian_match(ious, threshold=0.5) == [(0, 1), (1, 0)]


def test_hungarian_respects_class_threshold_ties_and_empty_inputs():
    ious = torch.tensor([[0.80, 0.80], [0.80, 0.80]])
    classes = torch.tensor([4, 5])
    assert global_hungarian_match(ious, gt_classes=classes, pred_classes=classes) == [
        (0, 0),
        (1, 1),
    ]
    assert global_hungarian_match(torch.tensor([[0.50]]), threshold=0.5) == [(0, 0)]
    assert global_hungarian_match(torch.tensor([[0.49]]), threshold=0.5) == []
    assert (
        global_hungarian_match(
            torch.tensor([[0.99]]),
            gt_classes=torch.tensor([1]),
            pred_classes=torch.tensor([2]),
        )
        == []
    )
    assert global_hungarian_match(torch.empty((0, 2))) == []
    assert global_hungarian_match([], []) == []


def test_hungarian_accepts_project_mask_shapes_as_a_direct_adapter():
    gt_masks = torch.tensor([[1, 0, 0], [0, 1, 1]], dtype=torch.bool)
    pred_masks = torch.tensor([[1, 0], [0, 1], [0, 1]], dtype=torch.bool)

    assert global_hungarian_match(gt_masks, pred_masks) == [(0, 0), (1, 1)]


def test_relative_retention_returns_none_for_zero_denominator():
    assert relative_retention(0.4, 0.0) is None
    assert relative_retention(0.4, 0.8) == pytest.approx(0.5)


def test_official_raw_metrics_use_stmetrics_class_macro_not_micro_diagnostic():
    point_count = 360
    first = torch.zeros(point_count, dtype=torch.bool)
    first[:120] = True
    second = torch.zeros(point_count, dtype=torch.bool)
    second[120:240] = True
    false_positive = torch.zeros(point_count, dtype=torch.bool)
    false_positive[240:] = True
    prediction = {
        "pred_masks": torch.stack((first, false_positive, second), dim=1),
        "pred_classes": torch.tensor([3, 3, 4]),
        "pred_scores": torch.tensor([0.9, 0.8, 0.7]),
    }
    target = {
        "masks": torch.stack((first, second)),
        "labels": torch.tensor([3, 4]),
        "ids": torch.tensor([1, 2]),
        "changes": torch.tensor([0, 1]),
        "temporal_stages": torch.zeros(point_count, dtype=torch.long),
    }

    official = compute_official_raw_local_metrics([prediction], [target])

    assert official == {
        "raw_local_AP": 1.0,
        "raw_local_AP50": 1.0,
        "raw_local_AP25": 1.0,
        "raw_local_REC": 1.0,
        "raw_local_REC50": 1.0,
        "raw_local_REC25": 1.0,
    }
    assert compute_raw_local_metrics(prediction, target)[
        "raw_local_AP"
    ] == pytest.approx(5.0 / 6.0)


def test_official_temporal_accumulator_exposes_fixed_stmetrics_keys():
    point_count = 120
    mask = torch.ones(point_count, dtype=torch.bool)
    prediction = {
        "pred_masks": mask[:, None],
        "pred_classes": torch.tensor([3]),
        "pred_scores": torch.tensor([0.9]),
    }
    target = {
        "masks": mask[None, :],
        "labels": torch.tensor([3]),
        "ids": torch.tensor([1]),
        "changes": torch.tensor([0]),
        "temporal_stages": torch.zeros(point_count, dtype=torch.long),
    }
    metric = OfficialMetricAccumulator(mode="strict_online")
    metric.update(prediction, target)

    assert metric.compute() == {
        "online_t-mAP": 1.0,
        "online_t-mAP50": 1.0,
        "online_t-mAP25": 1.0,
        "online_t-REC": 1.0,
        "online_t-REC50": 1.0,
        "online_t-REC25": 1.0,
    }
    assert compute_official_temporal_metrics([prediction], [target]) == metric.compute()


def test_official_metric_state_evidence_recomputes_exact_aggregate() -> None:
    point_count = 120
    mask = torch.ones(point_count, dtype=torch.bool)
    prediction = {
        "pred_masks": mask[:, None],
        "pred_classes": torch.tensor([3]),
        "pred_scores": torch.tensor([0.9]),
    }
    target = {
        "masks": mask[None, :],
        "labels": torch.tensor([3]),
        "ids": torch.tensor([1]),
        "changes": torch.tensor([0]),
        "temporal_stages": torch.zeros(point_count, dtype=torch.long),
    }
    metric = OfficialMetricAccumulator(mode="strict_online")
    metric.update(prediction, target)

    evidence = metric.export_evidence()

    assert recompute_official_metric_evidence(evidence) == metric.compute()
    tampered = deepcopy(evidence)
    tampered["states"][0]["data"] = tampered["states"][0]["data"] + "AA=="
    with pytest.raises(ValueError, match="SHA-256|evidence"):
        recompute_official_metric_evidence(tampered)


def test_official_metric_population_evidence_merges_identity_keyed_states() -> None:
    point_count = 120
    mask = torch.ones(point_count, dtype=torch.bool)
    prediction = {
        "pred_masks": mask[:, None],
        "pred_classes": torch.tensor([3]),
        "pred_scores": torch.tensor([0.9]),
    }
    target = {
        "masks": mask[None, :],
        "labels": torch.tensor([3]),
        "ids": torch.tensor([1]),
        "changes": torch.tensor([0]),
        "temporal_stages": torch.zeros(point_count, dtype=torch.long),
    }
    state = OfficialMetricAccumulator(mode="strict_online")
    state.update(prediction, target)
    records = [
        {
            "reference_scene_id": "ref-a",
            "master_sequence_id": "master-a",
            "order_id": "canonical",
            "prediction_digest": "a" * 64,
            "state": state.export_evidence(),
        },
        {
            "reference_scene_id": "ref-b",
            "master_sequence_id": "master-b",
            "order_id": "reverse",
            "prediction_digest": "b" * 64,
            "state": state.export_evidence(),
        },
    ]

    evidence = build_official_metric_population_evidence(records)
    computed, restored, per_sequence = (
        recompute_official_metric_population_evidence(evidence)
    )

    assert evidence["updates"] == 2
    assert computed == state.compute()
    assert per_sequence == (state.compute(), state.compute())
    assert [
        (row["reference_scene_id"], row["master_sequence_id"], row["order_id"])
        for row in restored
    ] == [("ref-a", "master-a", "canonical"), ("ref-b", "master-b", "reverse")]
    tampered = deepcopy(evidence)
    tampered["records_zlib_base64"] += "AA=="
    with pytest.raises(ValueError, match="SHA-256|population evidence"):
        recompute_official_metric_population_evidence(tampered)


@pytest.mark.parametrize(
    "limit_name",
    (
        "_MAX_OFFICIAL_METRIC_POPULATION_ENCODED_BYTES",
        "_MAX_OFFICIAL_METRIC_POPULATION_COMPRESSED_BYTES",
    ),
)
def test_official_metric_population_evidence_bounds_input_before_decode(
    monkeypatch: pytest.MonkeyPatch, limit_name: str
) -> None:
    evidence = _small_official_metric_population_evidence()
    monkeypatch.setattr(p6a_metrics, limit_name, 1, raising=False)

    with pytest.raises(ValueError, match="payload exceeds"):
        recompute_official_metric_population_evidence(evidence)


def test_official_metric_population_encoded_limit_precedes_core_serialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _small_official_metric_population_evidence()
    monkeypatch.setattr(
        p6a_metrics,
        "_MAX_OFFICIAL_METRIC_POPULATION_ENCODED_BYTES",
        1,
    )
    monkeypatch.setattr(
        p6a_metrics.json,
        "dumps",
        lambda *args, **kwargs: pytest.fail("oversized payload must not be serialized"),
    )

    with pytest.raises(ValueError, match="payload exceeds"):
        recompute_official_metric_population_evidence(evidence)


def test_official_metric_population_evidence_rejects_zip_bomb_before_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _small_official_metric_population_evidence()
    oversized = b" " * 1024
    evidence["records_zlib_base64"] = base64.b64encode(
        zlib.compress(oversized, level=9)
    ).decode("ascii")
    evidence["records_sha256"] = hashlib.sha256(oversized).hexdigest()
    _rebind_population_evidence(evidence)
    monkeypatch.setattr(
        p6a_metrics,
        "_MAX_OFFICIAL_METRIC_POPULATION_RAW_BYTES",
        128,
        raising=False,
    )

    with pytest.raises(ValueError, match="payload exceeds"):
        recompute_official_metric_population_evidence(evidence)


def _small_official_metric_population_evidence() -> dict[str, object]:
    point_count = 120
    mask = torch.ones(point_count, dtype=torch.bool)
    accumulator = OfficialMetricAccumulator(mode="strict_online")
    accumulator.update(
        {
            "pred_masks": mask[:, None],
            "pred_classes": torch.tensor([3]),
            "pred_scores": torch.tensor([0.9]),
        },
        {
            "masks": mask[None, :],
            "labels": torch.tensor([3]),
            "ids": torch.tensor([1]),
            "changes": torch.tensor([0]),
            "temporal_stages": torch.zeros(point_count, dtype=torch.long),
        },
    )
    return build_official_metric_population_evidence(
        [
            {
                "reference_scene_id": "ref-a",
                "master_sequence_id": "master-a",
                "order_id": "canonical",
                "prediction_digest": "a" * 64,
                "state": accumulator.export_evidence(),
            }
        ]
    )


def _rebind_population_evidence(evidence: dict[str, object]) -> None:
    core = {key: evidence[key] for key in evidence if key != "sha256"}
    evidence["sha256"] = hashlib.sha256(
        (json.dumps(core, sort_keys=True, indent=2) + "\n").encode("utf-8")
    ).hexdigest()


def test_offline_prefix_uses_full_state_without_future_point_masks():
    accumulator = IdentityAccumulator()
    first = _stage_prediction(0, [101, 7001])
    first["class_probs"] = torch.tensor([[0.9, 0.1], [0.9, 0.1]])
    second = _stage_prediction(1, [101, 7001], flip=True)
    second["class_probs"] = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    accumulator.add_stage(first)
    accumulator.add_stage(second)

    online = build_online_endpoint_prediction(accumulator, endpoint=0)
    offline = build_offline_reconstructed_prediction(accumulator, endpoint=0)

    assert online["pred_masks"].shape[0] == 4
    assert offline["pred_masks"].shape[0] == 4
    assert online["pred_classes"].tolist() == [0, 0]
    assert offline["pred_classes"].tolist() == [1, 1]
