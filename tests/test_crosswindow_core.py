from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

from datasets.task_memory_episode import StageMeta
from models.task_memory_routing import PredictionObservation
from scripts.crosswindow_cache import (
    align_mask,
    build_canonical_frame,
    build_data_roles,
    compare_vertex_order,
    iter_campaign_units,
    resolve_assets,
)
from scripts.crosswindow_campaign import (
    CampaignError,
    _verified_torch_load,
    atomic_write_run_state,
    load_resume_state,
    new_run_state,
)

PARENT = "96edca52d9d1cab7781bfa2a273baf9fe52b6a7d"
INSTRUCTION_SHA = "f7ee448782b0e8e4d2940a5b4e60748ce46d3dd07457eaf2460ab0f9006c01c2"


def _data_contract() -> dict[str, object]:
    inventory = [
        {"reference_id": reference, "role": "development"}
        for reference in ("alpha", "beta", "gamma", "delta", "epsilon")
    ]
    inventory.extend(
        [
            {"reference_id": "train-a", "role": "adaptation"},
            {"reference_id": "native-a", "role": "additional_native_refs"},
        ]
    )
    return {"native_reference_inventory": inventory}


def test_development_split_is_reference_stable_and_disjoint() -> None:
    expected = {
        "DEV-CAL": ["delta", "beta"],
        "DEV-SEL": ["gamma", "alpha", "epsilon"],
        "adaptation": ["train-a"],
        "additional_native_refs": ["native-a"],
    }

    forward = build_data_roles(
        _data_contract(),
        available_references=["alpha", "beta", "gamma", "delta", "epsilon"],
    )
    reverse = build_data_roles(
        _data_contract(),
        available_references=["epsilon", "delta", "gamma", "beta", "alpha"],
    )

    assert forward == expected
    assert reverse == expected
    assert set(forward["DEV-CAL"]).isdisjoint(forward["DEV-SEL"])


def test_asset_resolution_uses_cli_then_environment_then_fallback() -> None:
    resolved = resolve_assets(
        explicit={"data_root": "/cli/data"},
        environ={
            "PERSIST4D_DATA_ROOT": "/env/data",
            "PERSIST4D_R1_CHECKPOINT": "/env/r1.ckpt",
        },
        fallback={
            "external:data_root": "/fallback/data",
            "external:r1_checkpoint": "/fallback/r1.ckpt",
            "external:rio_metadata": "/fallback/3RScan.json",
        },
    )

    assert resolved.values["data_root"] == "/cli/data"
    assert resolved.sources["data_root"] == "cli"
    assert resolved.values["r1_checkpoint"] == "/env/r1.ckpt"
    assert resolved.sources["r1_checkpoint"] == "environment"
    assert resolved.values["rio_metadata"] == "/fallback/3RScan.json"
    assert resolved.sources["rio_metadata"] == "fallback"
    assert resolved.values["pb_base_cache_root"] is None
    assert resolved.sources["pb_base_cache_root"] == "unresolved"


def test_asset_resolution_derives_metric_spec_from_data_directory(
    tmp_path: Path,
) -> None:
    specification = tmp_path / "processed/rio/rio.yaml"
    specification.parent.mkdir(parents=True)
    specification.write_text("data: rio\n", encoding="utf-8")

    resolved = resolve_assets(
        explicit={"data_root": str(tmp_path)},
        environ={},
        fallback={},
    )

    assert resolved.values["metric_dataset_spec"] == str(specification)
    assert resolved.sources["metric_dataset_spec"] == "derived:data_root"


def test_resume_rejects_changed_identity(tmp_path: Path) -> None:
    path = tmp_path / "RUN_STATE.json"
    state = new_run_state(
        parent=PARENT,
        instruction_sha=INSTRUCTION_SHA,
        config_sha="a" * 64,
        data_sha="b" * 64,
    )
    atomic_write_run_state(path, state)

    with pytest.raises(CampaignError, match="identity"):
        load_resume_state(
            path,
            parent=PARENT,
            instruction_sha=INSTRUCTION_SHA,
            config_sha="c" * 64,
            data_sha="b" * 64,
        )


def test_resume_round_trip_preserves_completed_units(tmp_path: Path) -> None:
    path = tmp_path / "RUN_STATE.json"
    state = new_run_state(
        parent=PARENT,
        instruction_sha=INSTRUCTION_SHA,
        config_sha="a" * 64,
        data_sha="b" * 64,
    )
    state["completed_units"] = ["dev:alpha:order-0"]
    atomic_write_run_state(path, state)

    loaded = load_resume_state(
        path,
        parent=PARENT,
        instruction_sha=INSTRUCTION_SHA,
        config_sha="a" * 64,
        data_sha="b" * 64,
    )

    assert loaded == state


def test_verified_cache_bytes_include_digest_sidecar(tmp_path: Path) -> None:
    path = tmp_path / "cache.pt"
    torch.save({"payload": torch.tensor([1, 2, 3])}, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    sidecar = path.with_suffix(".pt.sha256")
    sidecar.write_text(f"{digest}  {path.name}\n", encoding="ascii")
    record = {
        "bytes": path.stat().st_size + sidecar.stat().st_size,
        "sha256": digest,
    }

    loaded = _verified_torch_load(path=path, record=record, hash_cache={})

    assert torch.equal(loaded["payload"], torch.tensor([1, 2, 3]))


def _single_scan_meta(vertex_ids: tuple[int, ...]) -> StageMeta:
    point_count = len(vertex_ids)
    point_indices = torch.arange(point_count, dtype=torch.long)
    return StageMeta(
        reference_id="reference-0",
        episode_id="episode-0",
        scan_ids_in_window=("scan-a",),
        absolute_stage_index=0,
        local_stage_ids=torch.zeros(point_count, dtype=torch.long),
        original_vertex_ids=(torch.tensor(vertex_ids, dtype=torch.long),),
        scan_vertex_offsets=torch.tensor([0, point_count], dtype=torch.long),
        point2segment=point_indices.clone(),
        segment_stage_ids=torch.zeros(point_count, dtype=torch.long),
        augmentation_transform_id="identity-v1",
        coordinate_frame_id="reference-0",
        voxel_inverse=point_indices.clone(),
        full_resolution_point2segment=point_indices.clone(),
    )


def _ledger_observation() -> PredictionObservation:
    return PredictionObservation(
        features=torch.tensor([[[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]]]),
        class_prob=torch.tensor(
            [[[0.8, 0.2], [0.3, 0.7], [0.5, 0.5]]], dtype=torch.float32
        ),
        confidence=torch.tensor([[0.9, 0.8, 0.7]]),
        valid=torch.tensor([[True, False, True]]),
        current_supported=torch.tensor([[True, False, True]]),
        previous_supported=torch.tensor([[False, True, False]]),
    )


def _ledger_prediction():
    from scripts.rescene_task_postprocess import OfficialTaskPrediction

    masks = torch.tensor(
        [
            [True, False, False],
            [False, True, True],
            [False, False, False],
        ],
        dtype=torch.bool,
    )
    return OfficialTaskPrediction(
        pred_masks=masks,
        pred_scores=torch.tensor([0.9, 0.8, 0.7]),
        pred_classes=torch.tensor([12, 13, 14]),
        source_query_ids=torch.tensor([0, 1, 1]),
        source_class_ids=torch.tensor([2, 3, 4]),
        temporal_stages=torch.zeros(3, dtype=torch.long),
        latest_stage_index=0,
        latest_stage_masks=masks.clone(),
    )


def test_align_mask_handles_non_identity_three_cycle_and_round_trip() -> None:
    mask = torch.tensor([False, False, True])

    canonical = align_mask(mask, from_ids=[20, 30, 10], to_ids=[10, 20, 30])
    restored = align_mask(canonical, from_ids=[10, 20, 30], to_ids=[20, 30, 10])

    assert canonical.tolist() == [True, False, False]
    assert torch.equal(restored, mask)


def test_point_order_comparison_distinguishes_permutation_from_set_mismatch() -> None:
    assert compare_vertex_order([10, 20, 30], [10, 20, 30]) == "IDENTICAL"
    assert compare_vertex_order([10, 20, 30], [20, 30, 10]) == "NON_IDENTITY"
    assert compare_vertex_order([10, 20, 30], [10, 20, 40]) == "SET_MISMATCH"


def test_candidate_ledger_preserves_multiclass_and_valid_query_union() -> None:
    frame = build_canonical_frame(
        producer_id="r1-b4",
        order_id="order-0",
        observation=_ledger_observation(),
        prediction=_ledger_prediction(),
        stage_meta=_single_scan_meta((30, 10, 20)),
    )

    assert [group.source_query_id for group in frame.groups] == [0, 1, 2]
    assert [group.candidate_indices for group in frame.groups] == [(0,), (1, 2), ()]
    assert [candidate.predicted_class_id for candidate in frame.candidates] == [
        12,
        13,
        14,
    ]
    assert [candidate.key.source_class_id for candidate in frame.candidates] == [
        2,
        3,
        4,
    ]
    assert [candidate.score for candidate in frame.candidates] == pytest.approx(
        [0.9, 0.8, 0.7]
    )
    assert frame.canonical_vertex_ids["scan-a"].tolist() == [10, 20, 30]
    candidate_slice = frame.candidates[0].slices[0]
    assert candidate_slice.original_vertex_ids.tolist() == [30, 10, 20]
    assert candidate_slice.canonical_vertex_ids.tolist() == [10, 20, 30]
    assert candidate_slice.original_score == pytest.approx(0.9)
    assert candidate_slice.mask.tolist() == [False, False, True]


def test_campaign_units_preserve_duplicate_logical_to_physical_bindings() -> None:
    base_record = {
        "reference_id": "reference-0",
        "sequence_id": "scan-a-scan-b",
        "filename": "base.pt",
        "sha256": "a" * 64,
        "bytes": 100,
    }
    supplement_record = {
        "reference_id": "reference-0",
        "sequence_id": "scan-a-scan-b",
        "filename": "supplement.pt",
        "sha256": "b" * 64,
        "bytes": 80,
        "base_cache_sha256": "a" * 64,
    }

    units = list(
        iter_campaign_units(
            role="protocol-b",
            role_references=["reference-0"],
            base_manifest={"records": [base_record, base_record]},
            supplement_manifest={"records": [supplement_record, supplement_record]},
        )
    )

    assert [unit.logical_index for unit in units] == [0, 1]
    assert [unit.logical_unit_id for unit in units] == [
        "protocol-b:00000",
        "protocol-b:00001",
    ]
    assert len({unit.physical_pair for unit in units}) == 1
