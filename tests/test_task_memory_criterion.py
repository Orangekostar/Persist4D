from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

from datasets.task_memory_episode import StageMeta
from models.criterion import SetCriterion
from models.task_memory_criterion import TaskMemoryCriterion
from models.task_memory_routing import (
    CommitResult,
    PredictionObservation,
    route_entities,
)
from models.task_memory_state import TaskMemoryConfig, TaskMemoryState
from models.task_memory_supervision import (
    TaskMemorySupervisionError,
    TrainingIdentityLedger,
    build_independent_assignment,
    build_tala_assignment,
    prediction_stage_ids,
)


class _ClassMatcher(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[int, int]] = []

    def forward(self, outputs, targets, _mask_type):
        batch_indices = []
        for logits, target in zip(outputs["pred_logits"], targets, strict=True):
            query_count = logits.shape[0]
            target_count = target["labels"].numel()
            self.calls.append((query_count, target_count))
            remaining = set(range(query_count))
            queries = []
            target_rows = []
            for target_row, label in enumerate(target["labels"].tolist()):
                if not remaining:
                    break
                query = max(
                    remaining, key=lambda index: (float(logits[index, label]), -index)
                )
                remaining.remove(query)
                queries.append(query)
                target_rows.append(target_row)
            batch_indices.append(
                (
                    torch.tensor(queries, dtype=torch.long),
                    torch.tensor(target_rows, dtype=torch.long),
                )
            )
        return batch_indices


def _meta(*, batch: int = 1, stage: int = 1) -> tuple[StageMeta, ...]:
    result = []
    for index in range(batch):
        result.append(
            StageMeta(
                reference_id=f"ref-{index}",
                episode_id=f"episode-{index}",
                scan_ids_in_window=("scan-0", "scan-1"),
                absolute_stage_index=stage,
                local_stage_ids=torch.tensor([0, 1]),
                original_vertex_ids=(torch.tensor([0]), torch.tensor([0])),
                scan_vertex_offsets=torch.tensor([0, 1, 2]),
                point2segment=torch.tensor([0, 1]),
                segment_stage_ids=torch.tensor([0, 1]),
                augmentation_transform_id="identity-v1",
                coordinate_frame_id=f"ref-{index}",
                voxel_inverse=torch.tensor([0, 1]),
                full_resolution_point2segment=torch.tensor([0, 1]),
            )
        )
    return tuple(result)


def _state() -> TaskMemoryState:
    state = TaskMemoryState.empty(
        batch_size=1,
        capacity=2,
        feature_dim=2,
        class_count=3,
        device="cpu",
        dtype=torch.float32,
        config=TaskMemoryConfig(association_threshold=0.5),
    )
    return replace(
        state,
        embedding=torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
        class_prob=torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]),
        confidence=torch.tensor([[0.9, 0.8]]),
        occupied=torch.tensor([[True, True]]),
        active=torch.tensor([[True, True]]),
        age=torch.tensor([[0, 0]]),
        last_seen=torch.tensor([[0, 0]]),
        stage_watermark=torch.tensor([0]),
        logical_ids=torch.tensor([[10, 11]]),
        generations=torch.tensor([[0, 0]]),
        next_logical_id=torch.tensor([12]),
    )


def _route():
    features = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]])
    class_prob = torch.tensor(
        [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]]
    )
    valid = torch.ones(1, 4, dtype=torch.bool)
    observation = PredictionObservation(
        features=features,
        class_prob=class_prob,
        confidence=torch.tensor([[0.9, 0.8, 0.7, 0.6]]),
        valid=valid,
        current_supported=valid.clone(),
        previous_supported=torch.zeros_like(valid),
    )
    route = route_entities(observation, _state(), _meta())
    assert route.query_to_slot.tolist() == [[0, 1, -1, -1]]
    return route


def _outputs(*, batch: int = 1, requires_grad: bool = False):
    logits = torch.tensor(
        [
            [
                [-2.0, 8.0, -2.0],
                [1.0, 0.0, -1.0],
                [-2.0, 9.0, -2.0],
                [8.0, -2.0, -2.0],
            ]
            for _ in range(batch)
        ],
        requires_grad=requires_grad,
    )
    masks = [
        torch.tensor(
            [[2.0, -1.0, 0.5, -0.5], [3.0, 1.0, -0.5, 0.5]],
            requires_grad=requires_grad,
        )
        for _ in range(batch)
    ]
    return {"pred_logits": logits, "pred_masks": masks, "pred_changes": None}


def _target(
    ids: tuple[int, ...] = (101, 202),
    *,
    previous_only_first: bool = False,
):
    labels = torch.tensor([0 if identity == 101 else 1 for identity in ids])
    masks = []
    for identity in ids:
        if identity == 101:
            masks.append([1, 0 if previous_only_first else 1])
        else:
            masks.append([0, 1])
    return {
        "ids": torch.tensor(ids, dtype=torch.long),
        "labels": labels,
        "segment_mask": torch.tensor(masks, dtype=torch.bool).reshape(len(ids), 2),
    }


def _ledger(*, duplicate: bool = False) -> TrainingIdentityLedger:
    ledger = TrainingIdentityLedger()
    ledger.bind(10, 0, ("ref-0", 101))
    if duplicate:
        ledger.bind(11, 0, ("ref-0", 101))
    return ledger


def _base_criterion(matcher: nn.Module) -> SetCriterion:
    return SetCriterion(
        num_classes=3,
        matcher=matcher,
        weight_dict={"loss_ce": 1.0, "loss_mask": 1.0, "loss_dice": 1.0},
        eos_coef=0.1,
        losses=["labels", "masks"],
        num_points=-1,
        oversample_ratio=3.0,
        importance_sample_ratio=0.75,
        class_weights=-1,
        num_changes=6,
        change_weights=[1.0] * 6,
    )


def test_ledger_binds_only_real_matched_unambiguous_births() -> None:
    state = _state()
    commit = CommitResult(
        state=state,
        query_to_slot=torch.tensor([[0, 1, -1, -1]]),
        query_to_logical_id=torch.tensor([[10, 11, -1, -1]]),
        query_to_generation=torch.tensor([[0, 0, -1, -1]]),
        births=torch.tensor([[True, True, False, False]]),
        rejected_births=torch.zeros(1, 4, dtype=torch.bool),
        active=state.active,
        route_commitment="a" * 64,
    )
    ledger = TrainingIdentityLedger()

    diagnostics = ledger.bind_births(
        commit,
        batch_index=0,
        assigned_indices=(torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2])),
        identity_keys=(("ref-0", 101), ("ref-0", 202), ("ref-0", 303)),
        target_labels=torch.tensor([0, 1, 0]),
        ambiguity_metadata=[[202, 999]],
    )

    assert ledger.resolve(10, 0) == ("ref-0", 101)
    assert ledger.resolve(11, 0) is None
    assert diagnostics == {
        "bound_births": 1,
        "ambiguous_births_excluded": 1,
        "ignored_births_excluded": 0,
        "unmatched_births": 0,
    }
    assert TrainingIdentityLedger.from_state_dict(ledger.state_dict()).bindings == (
        ledger.bindings
    )


def test_prediction_stage_ids_support_segment_and_point_masks() -> None:
    meta = _meta()[0]
    point_meta = replace(
        meta,
        point2segment=torch.tensor([0, 0, 1, 1]),
        voxel_inverse=torch.tensor([0, 2]),
        full_resolution_point2segment=torch.tensor([0, 1]),
    )

    assert prediction_stage_ids(meta, 2).tolist() == [0, 1]
    assert prediction_stage_ids(point_meta, 4).tolist() == [0, 0, 1, 1]


def test_assignments_fail_closed_on_invalid_matcher_indices() -> None:
    def invalid_matcher(_outputs, _targets, _mask_type):
        return [(torch.tensor([4]), torch.tensor([0]))]

    with pytest.raises(TaskMemorySupervisionError, match="out of range"):
        build_independent_assignment(
            outputs=_outputs(),
            targets=[_target()],
            matcher=invalid_matcher,
            mask_type="segment_mask",
        )


def test_tala_rejects_route_from_different_stage_metadata() -> None:
    mismatched_meta = (
        replace(_meta()[0], augmentation_transform_id="different-transform"),
    )

    with pytest.raises(TaskMemorySupervisionError, match="metadata commitment"):
        build_tala_assignment(
            outputs=_outputs(),
            targets=[_target()],
            matcher=_ClassMatcher(),
            mask_type="segment_mask",
            route=_route(),
            ledgers=[_ledger()],
            identity_keys=((("ref-0", 101), ("ref-0", 202)),),
            ambiguity_metadata=([],),
            stage_meta=mismatched_meta,
        )


def test_tala_keeps_wrong_inherited_route_and_matches_only_newborn_residual() -> None:
    matcher = _ClassMatcher()
    assignment = build_tala_assignment(
        outputs=_outputs(),
        targets=[_target()],
        matcher=matcher,
        mask_type="segment_mask",
        route=_route(),
        ledgers=[_ledger()],
        identity_keys=((("ref-0", 101), ("ref-0", 202)),),
        ambiguity_metadata=([],),
        stage_meta=_meta(),
    )

    source, target = assignment.indices[0]
    assert list(zip(source.tolist(), target.tolist(), strict=True)) == [(0, 0), (2, 1)]
    assert assignment.fixed_indices[0][0].tolist() == [0]
    assert assignment.residual_indices[0][0].tolist() == [2]
    assert matcher.calls == [(3, 1)]
    assert assignment.diagnostics["wrong_route_repairs"] == 0


def test_duplicate_inherited_gt_selects_stable_primary_and_reserves_loser() -> None:
    assignment = build_tala_assignment(
        outputs=_outputs(),
        targets=[_target()],
        matcher=_ClassMatcher(),
        mask_type="segment_mask",
        route=_route(),
        ledgers=[_ledger(duplicate=True)],
        identity_keys=((("ref-0", 101), ("ref-0", 202)),),
        ambiguity_metadata=([],),
        stage_meta=_meta(),
    )

    source, target = assignment.indices[0]
    assert list(zip(source.tolist(), target.tolist(), strict=True)) == [(0, 0), (2, 1)]
    assert assignment.duplicate_queries[0].tolist() == [1]
    assert assignment.reserved_queries[0].tolist() == [0, 1]
    assert assignment.diagnostics["duplicate_inherited_queries"] == 1


def test_ambiguity_exclusion_preserves_independent_spatial_matching() -> None:
    matcher = _ClassMatcher()
    outputs = _outputs()
    outputs["pred_logits"][0, 0] = torch.tensor([10.0, -2.0, -2.0])
    assignment = build_tala_assignment(
        outputs=outputs,
        targets=[_target()],
        matcher=matcher,
        mask_type="segment_mask",
        route=_route(),
        ledgers=[_ledger()],
        identity_keys=((("ref-0", 101), ("ref-0", 202)),),
        ambiguity_metadata=([[101, 999]],),
        stage_meta=_meta(),
    )

    assert assignment.fixed_indices[0][0].numel() == 0
    assert assignment.reserved_queries[0].numel() == 0
    assert assignment.diagnostics["ambiguous_inheritance_excluded"] == 1
    assert matcher.calls == [(4, 2)]
    assert 0 in assignment.indices[0][0].tolist()


def test_previous_only_target_stays_positive_and_absent_target_is_reserved() -> None:
    previous_only = build_tala_assignment(
        outputs=_outputs(),
        targets=[_target(previous_only_first=True)],
        matcher=_ClassMatcher(),
        mask_type="segment_mask",
        route=_route(),
        ledgers=[_ledger()],
        identity_keys=((("ref-0", 101), ("ref-0", 202)),),
        ambiguity_metadata=([],),
        stage_meta=_meta(),
    )
    assert previous_only.fixed_indices[0][0].tolist() == [0]
    assert previous_only.fixed_indices[0][1].tolist() == [0]
    assert previous_only.diagnostics["previous_only_positive_queries"] == 1

    criterion = TaskMemoryCriterion(
        _base_criterion(_ClassMatcher()),
        matcher_mode="tala",
        post_conditioning_aux_start=0,
    )
    previous_result = criterion.compute_with_assignments(
        _outputs(requires_grad=True),
        [_target(previous_only_first=True)],
        mask_type="segment_mask",
        route=_route(),
        ledgers=[_ledger()],
        identity_keys=((("ref-0", 101), ("ref-0", 202)),),
        ambiguity_metadata=([],),
        stage_meta=_meta(),
    )
    assert previous_result.losses["loss_tala_empty_current"].item() > 0

    absent = build_tala_assignment(
        outputs=_outputs(),
        targets=[_target((202,))],
        matcher=_ClassMatcher(),
        mask_type="segment_mask",
        route=_route(),
        ledgers=[_ledger()],
        identity_keys=((("ref-0", 202),),),
        ambiguity_metadata=([],),
        stage_meta=_meta(),
    )
    assert absent.absent_inherited_queries[0].tolist() == [0]
    assert absent.reserved_queries[0].tolist() == [0]
    assert 0 not in absent.indices[0][0].tolist()
    assert absent.diagnostics["absent_inherited_queries"] == 1


def test_tala_absent_entity_has_no_object_and_differentiable_empty_current_loss() -> (
    None
):
    matcher = _ClassMatcher()
    criterion = TaskMemoryCriterion(
        _base_criterion(matcher), matcher_mode="tala", post_conditioning_aux_start=0
    )
    outputs = _outputs(requires_grad=True)
    result = criterion.compute_with_assignments(
        outputs,
        [_target((202,))],
        mask_type="segment_mask",
        route=_route(),
        ledgers=[_ledger()],
        identity_keys=((("ref-0", 202),),),
        ambiguity_metadata=([],),
        stage_meta=_meta(),
    )

    assert result.losses["loss_tala_empty_current"].item() > 0
    result.losses["loss_ce"].backward(retain_graph=True)
    assert outputs["pred_logits"].grad[0, 0, -1] < 0
    result.losses["loss_tala_empty_current"].backward()
    assert outputs["pred_masks"][0].grad[1, 0] > 0


def test_independent_mode_handles_empty_target_beside_nonempty_sample() -> None:
    matcher = _ClassMatcher()
    criterion = TaskMemoryCriterion(
        _base_criterion(matcher),
        matcher_mode="independent",
        post_conditioning_aux_start=0,
    )
    outputs = _outputs(batch=2, requires_grad=True)
    targets = [
        {
            "ids": torch.empty(0, dtype=torch.long),
            "labels": torch.empty(0, dtype=torch.long),
            "segment_mask": torch.empty(0, 2, dtype=torch.bool),
        },
        _target((202,)),
    ]
    result = criterion.compute_with_assignments(
        outputs,
        targets,
        mask_type="segment_mask",
    )
    total = sum(result.losses.values())

    assert all(torch.isfinite(value) for value in result.losses.values())
    assert result.final_assignment.indices[0][0].numel() == 0
    assert result.final_assignment.indices[1][0].numel() == 1
    total.backward()
    assert outputs["pred_logits"].grad is not None
    assert outputs["pred_masks"][0].grad is not None
    assert torch.count_nonzero(outputs["pred_masks"][0].grad) == 0


def test_independent_mode_preserves_original_set_criterion_components() -> None:
    outputs = _outputs()
    targets = [_target()]
    direct = _base_criterion(_ClassMatcher())
    wrapped = TaskMemoryCriterion(
        _base_criterion(_ClassMatcher()),
        matcher_mode="independent",
        post_conditioning_aux_start=0,
    )

    expected = direct(outputs, targets, mask_type="segment_mask")
    actual = wrapped.compute_with_assignments(
        outputs, targets, mask_type="segment_mask"
    ).losses

    for name, value in expected.items():
        assert torch.equal(actual[name], value)
    assert actual["loss_tala_empty_current"].item() == 0.0


def test_pre_conditioning_aux_is_independent_and_post_layers_keep_tala() -> None:
    matcher = _ClassMatcher()
    criterion = TaskMemoryCriterion(
        _base_criterion(matcher), matcher_mode="tala", post_conditioning_aux_start=1
    )
    outputs = _outputs()
    outputs["aux_outputs"] = [_outputs(), _outputs(), _outputs()]
    result = criterion.compute_with_assignments(
        outputs,
        [_target()],
        mask_type="segment_mask",
        route=_route(),
        ledgers=[_ledger()],
        identity_keys=((("ref-0", 101), ("ref-0", 202)),),
        ambiguity_metadata=([],),
        stage_meta=_meta(),
    )

    assert [assignment.matcher_mode for assignment in result.aux_assignments] == [
        "independent",
        "tala",
        "tala",
    ]
    assert result.final_assignment.matcher_mode == "tala"
    assert result.aux_assignments[0].fixed_indices[0][0].numel() == 0
    for assignment in (*result.aux_assignments[1:], result.final_assignment):
        assert assignment.fixed_indices[0][0].tolist() == [0]
        assert assignment.fixed_indices[0][1].tolist() == [0]


def test_sequence_loss_example_separates_temporal_supervision_components() -> None:
    artifact = (
        Path(__file__).parents[1]
        / "artifacts/task_memory_retention_v2/implementation/sequence_loss_example.csv"
    )
    with artifact.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 16
    assert {row["loss_component"] for row in rows} == {
        "loss_ce",
        "loss_dice",
        "loss_mask",
        "loss_tala_empty_current",
    }
    assert {
        (row["layer"], row["matcher_mode"])
        for row in rows
        if row["sequence_case"] == "previous_only"
    } == {("aux_0", "independent"), ("aux_1", "tala"), ("final", "tala")}
    assert any(
        row["sequence_case"] == "ambiguous"
        and row["ambiguity_exclusions"] == "1"
        and row["fixed_pairs"] == ""
        for row in rows
    )
    assert any(
        row["sequence_case"] == "previous_only"
        and row["loss_component"] == "loss_tala_empty_current"
        and float(row["loss_value"]) > 0
        for row in rows
    )

    criterion = TaskMemoryCriterion(
        _base_criterion(_ClassMatcher()),
        matcher_mode="tala",
        post_conditioning_aux_start=1,
    )
    outputs = _outputs()
    outputs["aux_outputs"] = [_outputs(), _outputs()]
    previous_result = criterion.compute_with_assignments(
        outputs,
        [_target(previous_only_first=True)],
        mask_type="segment_mask",
        route=_route(),
        ledgers=[_ledger()],
        identity_keys=((("ref-0", 101), ("ref-0", 202)),),
        ambiguity_metadata=([],),
        stage_meta=_meta(),
    )
    ambiguous_result = criterion.compute_with_assignments(
        _outputs(),
        [_target()],
        mask_type="segment_mask",
        route=_route(),
        ledgers=[_ledger()],
        identity_keys=((("ref-0", 101), ("ref-0", 202)),),
        ambiguity_metadata=([[101, 999]],),
        stage_meta=_meta(),
    )
    for row in rows:
        suffix = {"aux_0": "_0", "aux_1": "_1", "final": ""}[row["layer"]]
        result = (
            previous_result
            if row["sequence_case"] == "previous_only"
            else ambiguous_result
        )
        actual = result.losses[f"{row['loss_component']}{suffix}"]
        assert float(row["loss_value"]) == pytest.approx(float(actual), abs=1e-9)
