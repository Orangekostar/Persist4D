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


def _single_scan_meta(
    vertex_ids: tuple[int, ...],
    *,
    absolute_stage: int = 0,
    scan_id: str = "scan-a",
    episode_id: str = "episode-0",
) -> StageMeta:
    point_count = len(vertex_ids)
    point_indices = torch.arange(point_count, dtype=torch.long)
    return StageMeta(
        reference_id="reference-0",
        episode_id=episode_id,
        scan_ids_in_window=(scan_id,),
        absolute_stage_index=absolute_stage,
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


def test_fixed_u_joint_objective_equivalence_is_exhaustive() -> None:
    from scripts.evaluate_crosswindow_consensus import (
        exhaustive_fixed_u_equivalence,
    )

    result = exhaustive_fixed_u_equivalence()

    assert result["group_sizes"] == [2, 3]
    assert result["assignment_count"] > 0
    assert result["maximum_absolute_error"] <= 1e-12


def _single_query_frame(
    *,
    absolute_stage: int,
    feature: tuple[float, float],
    class_prob: tuple[float, float],
    scan_id: str,
    valid: bool = True,
    current_supported: bool = True,
):
    from scripts.rescene_task_postprocess import OfficialTaskPrediction

    observation = PredictionObservation(
        features=torch.tensor([[feature]], dtype=torch.float32),
        class_prob=torch.tensor([[class_prob]], dtype=torch.float32),
        confidence=torch.tensor([[0.9]], dtype=torch.float32),
        valid=torch.tensor([[valid]]),
        current_supported=torch.tensor([[current_supported]]),
        previous_supported=torch.tensor([[False]]),
    )
    masks = torch.tensor([[True], [False], [True]], dtype=torch.bool)
    prediction = OfficialTaskPrediction(
        pred_masks=masks,
        pred_scores=torch.tensor([0.8]),
        pred_classes=torch.tensor([1]),
        source_query_ids=torch.tensor([0]),
        source_class_ids=torch.tensor([1]),
        temporal_stages=torch.zeros(3, dtype=torch.long),
        latest_stage_index=0,
        latest_stage_masks=masks.clone(),
    )
    return build_canonical_frame(
        producer_id="r1-b4",
        order_id="order-0",
        observation=observation,
        prediction=prediction,
        stage_meta=_single_scan_meta(
            (30, 10, 20), absolute_stage=absolute_stage, scan_id=scan_id
        ),
    )


def _two_scan_query_frame(
    *,
    absolute_stage: int,
    feature: tuple[float, float],
    class_prob: tuple[float, float],
):
    from scripts.rescene_task_postprocess import OfficialTaskPrediction

    observation = PredictionObservation(
        features=torch.tensor([[feature]], dtype=torch.float32),
        class_prob=torch.tensor([[class_prob]], dtype=torch.float32),
        confidence=torch.tensor([[0.9]], dtype=torch.float32),
        valid=torch.tensor([[True]]),
        current_supported=torch.tensor([[True]]),
        previous_supported=torch.tensor([[True]]),
    )
    local_stages = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
    point_indices = torch.arange(6, dtype=torch.long)
    masks = torch.tensor(
        [[True], [False], [True], [False], [True], [True]], dtype=torch.bool
    )
    prediction = OfficialTaskPrediction(
        pred_masks=masks,
        pred_scores=torch.tensor([0.8]),
        pred_classes=torch.tensor([1]),
        source_query_ids=torch.tensor([0]),
        source_class_ids=torch.tensor([1]),
        temporal_stages=local_stages.clone(),
        latest_stage_index=1,
        latest_stage_masks=masks[3:].clone(),
    )
    stage_meta = StageMeta(
        reference_id="reference-0",
        episode_id="episode-0",
        scan_ids_in_window=("scan-a", "scan-b"),
        absolute_stage_index=absolute_stage,
        local_stage_ids=local_stages,
        original_vertex_ids=(
            torch.tensor([30, 10, 20]),
            torch.tensor([30, 10, 20]),
        ),
        scan_vertex_offsets=torch.tensor([0, 3, 6]),
        point2segment=point_indices.clone(),
        segment_stage_ids=local_stages.clone(),
        augmentation_transform_id="identity-v1",
        coordinate_frame_id="reference-0",
        voxel_inverse=point_indices.clone(),
        full_resolution_point2segment=point_indices.clone(),
    )
    return build_canonical_frame(
        producer_id="r1-b4",
        order_id="order-0",
        observation=observation,
        prediction=prediction,
        stage_meta=stage_meta,
    )


def test_private_nulls_keep_unmatched_groups_distinct() -> None:
    from models.crosswindow_state import CrossWindowState
    from models.overlap_entity_association import (
        A0_DEFAULT,
        associate,
        build_evidence,
    )

    frame = build_canonical_frame(
        producer_id="r1-b4",
        order_id="order-0",
        observation=_ledger_observation(),
        prediction=_ledger_prediction(),
        stage_meta=_single_scan_meta((30, 10, 20)),
    )
    state = CrossWindowState.empty(capacity=2, feature_dim=2, class_count=2)

    plan = associate(build_evidence(frame, state, None), A0_DEFAULT)

    assert len(set(plan.entity_for_group)) == len(frame.groups)
    assert plan.is_new_id == (True, True, True)


def test_full_capacity_rejects_residency_without_dropping_output() -> None:
    from models.crosswindow_state import CrossWindowState
    from models.overlap_entity_association import (
        A0_DEFAULT,
        associate,
        build_evidence,
        commit_observation,
    )

    state = CrossWindowState.empty(capacity=1, feature_dim=2, class_count=2)
    first = _single_query_frame(
        absolute_stage=0,
        feature=(1.0, 0.0),
        class_prob=(1.0, 0.0),
        scan_id="scan-a",
    )
    first_plan = associate(build_evidence(first, state, None), A0_DEFAULT)
    full_state, _ = commit_observation(first, state, first_plan)
    second = _single_query_frame(
        absolute_stage=1,
        feature=(-1.0, 0.0),
        class_prob=(0.0, 1.0),
        scan_id="scan-b",
    )
    second_plan = associate(build_evidence(second, full_state, None), A0_DEFAULT)

    next_state, committed = commit_observation(second, full_state, second_plan)

    assert committed.nonresident_groups == (0,)
    assert committed.public_id_for_group[0] >= 0
    assert committed.buffer is not None
    assert committed.buffer.groups[0].logical_id == committed.public_id_for_group[0]
    assert next_state.occupied_count == next_state.capacity == 1


def test_a2_missing_overlap_uses_base_score() -> None:
    from models.crosswindow_state import CrossWindowState
    from models.overlap_entity_association import (
        A0_DEFAULT,
        associate,
        build_evidence,
        commit_observation,
        score_a2,
    )

    state = CrossWindowState.empty(capacity=2, feature_dim=2, class_count=2)
    first = _single_query_frame(
        absolute_stage=0,
        feature=(1.0, 0.0),
        class_prob=(1.0, 0.0),
        scan_id="scan-a",
    )
    plan = associate(build_evidence(first, state, None), A0_DEFAULT)
    resident_state, _ = commit_observation(first, state, plan)
    second = _single_query_frame(
        absolute_stage=1,
        feature=(1.0, 0.0),
        class_prob=(1.0, 0.0),
        scan_id="scan-b",
    )
    bundle = build_evidence(second, resident_state, None)

    scores = score_a2(bundle, lambda_=0.25)

    assert not bundle.has_overlap.any().item()
    assert scores[0, 0].item() == pytest.approx(bundle.base[0, 0].item())


def test_resident_and_buffer_identity_merge_into_one_overlap_anchor() -> None:
    from models.crosswindow_state import CrossWindowState
    from models.overlap_entity_association import (
        A0_DEFAULT,
        associate,
        build_evidence,
        commit_observation,
    )

    state = CrossWindowState.empty(capacity=2, feature_dim=2, class_count=2)
    first = _single_query_frame(
        absolute_stage=0,
        feature=(1.0, 0.0),
        class_prob=(1.0, 0.0),
        scan_id="scan-a",
    )
    first_plan = associate(build_evidence(first, state, None), A0_DEFAULT)
    resident_state, committed = commit_observation(first, state, first_plan)
    second = _two_scan_query_frame(
        absolute_stage=1,
        feature=(1.0, 0.0),
        class_prob=(1.0, 0.0),
    )

    bundle = build_evidence(second, resident_state, committed.buffer)

    assert bundle.anchor_count == 1
    assert bundle.anchor_sources == ("resident",)
    assert bundle.has_overlap.tolist() == [[True]]
    assert bundle.overlap[0, 0].item() == pytest.approx(1.0)


def test_buffer_only_identity_promotes_without_renaming() -> None:
    from models.crosswindow_state import CrossWindowState
    from models.overlap_entity_association import (
        A0_DEFAULT,
        associate,
        build_evidence,
        commit_observation,
    )

    state = CrossWindowState.empty(capacity=1, feature_dim=2, class_count=2)
    exported_only = _single_query_frame(
        absolute_stage=0,
        feature=(1.0, 0.0),
        class_prob=(1.0, 0.0),
        scan_id="scan-a",
        valid=False,
        current_supported=False,
    )
    first_plan = associate(build_evidence(exported_only, state, None), A0_DEFAULT)
    empty_state, committed = commit_observation(exported_only, state, first_plan)
    second = _two_scan_query_frame(
        absolute_stage=1,
        feature=(1.0, 0.0),
        class_prob=(1.0, 0.0),
    )
    second_plan = associate(
        build_evidence(second, empty_state, committed.buffer), A0_DEFAULT
    )

    next_state, second_commit = commit_observation(second, empty_state, second_plan)

    assert second_plan.is_new_id == (False,)
    assert second_commit.public_id_for_group == committed.public_id_for_group
    assert second_plan.residency_reason == ("PROMOTED_BUFFER",)
    assert next_state.logical_ids[next_state.occupied].tolist() == [
        committed.public_id_for_group[0]
    ]


def test_strict_threshold_rejects_equal_score() -> None:
    from models.crosswindow_state import CrossWindowState
    from models.overlap_entity_association import (
        A0_DEFAULT,
        AssociationConfig,
        associate,
        build_evidence,
        commit_observation,
    )

    state = CrossWindowState.empty(capacity=2, feature_dim=2, class_count=2)
    first = _single_query_frame(
        absolute_stage=0,
        feature=(1.0, 0.0),
        class_prob=(1.0, 0.0),
        scan_id="scan-a",
    )
    plan = associate(build_evidence(first, state, None), A0_DEFAULT)
    resident_state, _ = commit_observation(first, state, plan)
    second = _single_query_frame(
        absolute_stage=1,
        feature=(1.0, 0.0),
        class_prob=(1.0, 0.0),
        scan_id="scan-b",
    )

    equal_threshold = associate(
        build_evidence(second, resident_state, None),
        AssociationConfig(family="A0-U", tau=1.0),
    )

    assert equal_threshold.is_new_id == (True,)


def test_class_probability_rows_are_validated_without_renormalizing() -> None:
    from models.crosswindow_state import CrossWindowState
    from models.overlap_entity_association import (
        CrossWindowAssociationError,
        build_evidence,
    )

    frame = _single_query_frame(
        absolute_stage=0,
        feature=(1.0, 0.0),
        class_prob=(0.8, 0.4),
        scan_id="scan-a",
    )
    state = CrossWindowState.empty(capacity=1, feature_dim=2, class_count=2)

    with pytest.raises(CrossWindowAssociationError, match="row sums"):
        build_evidence(frame, state, None)


def test_assignment_plan_cannot_commit_against_next_state() -> None:
    from models.crosswindow_state import CrossWindowState
    from models.overlap_entity_association import (
        A0_DEFAULT,
        CrossWindowAssociationError,
        associate,
        build_evidence,
        commit_observation,
    )

    state = CrossWindowState.empty(capacity=1, feature_dim=2, class_count=2)
    frame = _single_query_frame(
        absolute_stage=0,
        feature=(1.0, 0.0),
        class_prob=(1.0, 0.0),
        scan_id="scan-a",
    )
    plan = associate(build_evidence(frame, state, None), A0_DEFAULT)
    next_state, _ = commit_observation(frame, state, plan)

    with pytest.raises(CrossWindowAssociationError, match="another state"):
        commit_observation(frame, next_state, plan)


def _matrix_evidence(
    *, base: torch.Tensor, overlap: torch.Tensor, has_overlap: torch.Tensor
):
    from models.overlap_entity_association import EvidenceBundle

    group_count, anchor_count = base.shape
    return EvidenceBundle(
        source_state_sha256="a" * 64,
        frame_sha256="b" * 64,
        reference_id="reference-0",
        episode_id="episode-0",
        absolute_stage=1,
        group_keys=tuple(
            ("producer", "episode-0", "order-0", 1, index)
            for index in range(group_count)
        ),
        group_valid=(True,) * group_count,
        group_current_supported=(True,) * group_count,
        group_confidence=(0.9,) * group_count,
        anchor_logical_ids=tuple(range(10, 10 + anchor_count)),
        anchor_generations=(0,) * anchor_count,
        anchor_slots=tuple(range(anchor_count)),
        anchor_sources=("resident",) * anchor_count,
        free_slots=(),
        capacity=max(1, anchor_count),
        next_logical_id=10 + anchor_count,
        cosine=torch.zeros_like(base),
        class_compatibility=torch.zeros_like(base),
        base=base.float(),
        overlap=overlap.float(),
        has_overlap=has_overlap.bool(),
        zero_norm_query_count=0,
        zero_norm_anchor_count=0,
    )


def test_a1_locks_only_mutual_overlap_with_required_margins() -> None:
    from models.overlap_entity_association import A1_DEFAULT, associate

    bundle = _matrix_evidence(
        base=torch.tensor([[0.8, 0.8], [0.8, 0.8]]),
        overlap=torch.tensor([[0.9, 0.1], [0.1, 0.9]]),
        has_overlap=torch.ones((2, 2), dtype=torch.bool),
    )

    plan = associate(bundle, A1_DEFAULT)

    assert plan.anchor_for_group == (0, 1)
    assert plan.source_edge == ("OVERLAP_LOCK", "OVERLAP_LOCK")


def test_a1_single_mutual_candidate_uses_zero_second_best() -> None:
    from models.overlap_entity_association import A1_DEFAULT, associate

    bundle = _matrix_evidence(
        base=torch.tensor([[0.8]]),
        overlap=torch.tensor([[0.9]]),
        has_overlap=torch.ones((1, 1), dtype=torch.bool),
    )

    plan = associate(bundle, A1_DEFAULT)

    assert plan.anchor_for_group == (0,)
    assert plan.source_edge == ("OVERLAP_LOCK",)
    assert plan.decision_margin == pytest.approx((0.9,))


def test_private_dummies_keep_one_to_one_when_groups_compete() -> None:
    from models.overlap_entity_association import A0_DEFAULT, associate

    bundle = _matrix_evidence(
        base=torch.tensor([[0.9], [0.9]]),
        overlap=torch.zeros((2, 1)),
        has_overlap=torch.zeros((2, 1), dtype=torch.bool),
    )

    plan = associate(bundle, A0_DEFAULT)

    assert plan.entity_for_group.count(10) == 1
    assert len(set(plan.entity_for_group)) == 2
    assert plan.is_new_id.count(True) == 1


def test_preregistered_association_grid_is_exactly_three_three_six() -> None:
    from models.overlap_entity_association import preregistered_association_configs

    configs = preregistered_association_configs()

    assert [config.family for config in configs].count("A0-U") == 3
    assert [config.family for config in configs].count("A1") == 3
    assert [config.family for config in configs].count("A2") == 6
    assert len({config.config_id for config in configs}) == len(configs) == 12
