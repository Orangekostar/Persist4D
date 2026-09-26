import io
import zipfile

import pytest
import torch

from scripts.perception_gain_v2_predictions import (
    PredictionExports,
    PredictionSink,
    decode_prediction,
    encode_prediction,
    prepare_prediction_assets,
)


def test_export_cache_separates_executed_source_and_input_identity(tmp_path):
    common = {
        "external_root": tmp_path,
        "methods": [{"method_id": "native", "inference_identity": "same-model"}],
        "role": "LOCAL-T2", "eval_seed": 45, "lock_sha256": "same-lock",
        "expected_units_by_horizon": {2: 154}, "source_methods": {"native": "LOCAL"},
    }
    old = PredictionExports(**common, execution_binding={"source": "old", "input": "a"})
    new = PredictionExports(**common, execution_binding={"source": "new", "input": "a"})
    changed_input = PredictionExports(**common, execution_binding={"source": "new", "input": "b"})
    roots = {exports.sinks["native"].root for exports in (old, new, changed_input)}
    assert len(roots) == 3
    assert new.sinks["native"].binding["execution_binding"] == {"source": "new", "input": "a"}


def test_prediction_package_roundtrips_exact_arrays_without_ground_truth():
    prediction = {
        "pred_masks": torch.tensor(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 0.0, 0.0],
            ]
        ),
        "pred_scores": torch.tensor([0.1, 0.7, 0.9]),
        "pred_classes": torch.tensor([3, 4, 5]),
        "ground_truth": torch.ones(5, 3),
        "target": {"masks": torch.ones(5, 3)},
    }
    identity = {
        "method_id": "m",
        "data_role": "PB",
        "logical_unit_id": "PB:00000",
        "T": 2,
        "eval_seed": 45,
        "scan_ids": ["a", "b"],
        "inference_identity": "b" * 64,
        "lock_sha256": "a" * 64,
    }
    encoded = encode_prediction(
        prediction, identity=identity, scan_vertex_offsets=[0, 2, 5]
    )
    restored, metadata, offsets = decode_prediction(encoded)
    assert metadata["identity"] == identity
    assert offsets.tolist() == [0, 2, 5]
    for name in ("pred_masks", "pred_scores", "pred_classes"):
        assert torch.equal(restored[name], prediction[name])
        assert restored[name].dtype == prediction[name].dtype
    with zipfile.ZipFile(io.BytesIO(encoded)) as archive:
        assert set(archive.namelist()) == {
            "metadata.npy",
            "masks_packed.npy",
            "scores.npy",
            "classes.npy",
            "scan_vertex_offsets.npy",
        }
    assert encoded == encode_prediction(
        prediction, identity=identity, scan_vertex_offsets=[0, 2, 5]
    )


def test_zero_candidates_preserve_the_observed_point_count():
    prediction = {
        "pred_masks": torch.zeros(9, 0, dtype=torch.bool),
        "pred_scores": torch.zeros(0),
        "pred_classes": torch.zeros(0, dtype=torch.int64),
    }
    identity = {
        "method_id": "m",
        "data_role": "CAL",
        "logical_unit_id": "CAL:00000",
        "T": 2,
        "eval_seed": 45,
        "scan_ids": ["a", "b"],
        "inference_identity": "b" * 64,
        "lock_sha256": None,
    }
    restored, _, _ = decode_prediction(
        encode_prediction(prediction, identity=identity, scan_vertex_offsets=[0, 4, 9])
    )
    assert restored["pred_masks"].shape == (9, 0)


def test_partial_prediction_inventory_preserves_denominators_and_rejects_changed_repeat(
    tmp_path,
):
    binding = {
        "method_id": "m",
        "data_role": "PB",
        "eval_seed": 45,
        "inference_identity": "b" * 64,
        "lock_sha256": "a" * 64,
    }
    sink = PredictionSink(
        tmp_path,
        binding=binding,
        expected_units_by_horizon={2: 129, 3: 129, 4: 129, 5: 129},
    )
    prediction = {
        "pred_masks": torch.tensor([[True], [False]]),
        "pred_scores": torch.tensor([0.9]),
        "pred_classes": torch.tensor([3]),
    }
    kwargs = {
        "logical_unit_id": "PB:00000",
        "horizon": 2,
        "scan_ids": ["a", "b"],
        "scan_vertex_offsets": [0, 1, 2],
    }
    sink.write(prediction, **kwargs)
    result = sink.finalize()
    assert result["status"] == "PARTIAL"
    assert result["expected_units_by_horizon"] == {
        "2": 129,
        "3": 129,
        "4": 129,
        "5": 129,
    }
    resumed = PredictionSink(
        tmp_path,
        binding=binding,
        expected_units_by_horizon={2: 129, 3: 129, 4: 129, 5: 129},
    )
    resumed.write(prediction, **kwargs)
    assert len(resumed.finalize()["records"]) == 1
    prediction["pred_scores"] = torch.tensor([0.8])
    with pytest.raises(ValueError, match="different output"):
        resumed.write(prediction, **kwargs)


def test_prediction_release_marks_missing_frozen_population_despite_nonempty_assets(
    tmp_path,
):
    from scripts.perception_gain_v2 import file_hash, write_json

    artifacts, external = tmp_path / "artifacts", tmp_path / "external"
    lock_path = artifacts / "selection/FINAL_LOCK.json"
    write_json(lock_path, {"confirmation_methods": {"PB": ["m"]}})
    sink = PredictionSink(
        external / "cache/predictions/m",
        binding={
            "method_id": "m",
            "data_role": "PB",
            "eval_seed": 45,
            "inference_identity": "b" * 64,
            "lock_sha256": file_hash(lock_path),
        },
        expected_units_by_horizon={2: 129, 3: 129, 4: 129, 5: 129},
    )
    sink.write(
        {
            "pred_masks": torch.zeros(2, 0, dtype=torch.bool),
            "pred_scores": torch.zeros(0),
            "pred_classes": torch.zeros(0, dtype=torch.long),
        },
        logical_unit_id="unit",
        horizon=2,
        scan_ids=["a", "b"],
        scan_vertex_offsets=[0, 1, 2],
    )
    manifest = prepare_prediction_assets(artifacts=artifacts, external_root=external)
    assert manifest["assets"]
    assert manifest["status"] == "PARTIAL"
    assert manifest["unfulfilled_requirements"] == [
        {"name": "predictions:PB:m:seed45", "status": "INCOMPLETE"}
    ]
    assert (
        prepare_prediction_assets(artifacts=artifacts, external_root=external)
        == manifest
    )
