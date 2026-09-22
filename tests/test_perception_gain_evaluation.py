from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def _evaluation_module():
    try:
        return importlib.import_module("scripts.perception_gain_evaluation")
    except ModuleNotFoundError as error:
        pytest.fail(f"perception evaluation module is unavailable: {error}")


def _native_evaluation_module():
    try:
        return importlib.import_module("scripts.perception_gain_native_evaluation")
    except ModuleNotFoundError as error:
        pytest.fail(f"native perception evaluation module is unavailable: {error}")


def _local_evaluation_module():
    try:
        return importlib.import_module("scripts.perception_gain_local_evaluation")
    except ModuleNotFoundError as error:
        pytest.fail(f"LOCAL-T2 perception evaluation module is unavailable: {error}")


def _metrics(t2: float, t3: float, t4: float, t5: float):
    return {2: t2, 3: t3, 4: t4, 5: t5}


def test_evaluation_roles_bind_pb_to_validation_without_changing_development() -> None:
    evaluation = _evaluation_module()

    assert evaluation.evaluation_population_source("PB") == "validation"
    assert evaluation.evaluation_population_source("CAL") == "training"
    assert evaluation.evaluation_population_source("SEL") == "training"
    assert (
        evaluation.evaluation_population_source("ADDITIONAL")
        == "native_validation_slices"
    )
    with pytest.raises(evaluation.PerceptionEvaluationError, match="role"):
        evaluation.evaluation_population_source("TRAIN")


def _candidate(
    method: str,
    update: int,
    values,
    *,
    parameters: int = 0,
    coverage: str = "COMPLETE",
):
    return {
        "method_id": method,
        "optimizer_update": update,
        "new_parameter_count": parameters,
        "metrics": values,
        "coverage_status": coverage,
        "checkpoint": f"{method}-{update}.ckpt",
    }


def test_scores_and_ranking_follow_mean_long_min_then_cost_update_id() -> None:
    evaluation = _evaluation_module()
    baseline = _metrics(0.20, 0.20, 0.20, 0.20)
    candidates = [
        _candidate("late", 750, _metrics(0.21, 0.21, 0.20, 0.20), parameters=10),
        _candidate("cheap", 750, _metrics(0.21, 0.21, 0.20, 0.20), parameters=5),
        _candidate("mean", 250, _metrics(0.21, 0.21, 0.21, 0.21), parameters=99),
    ]

    ranked = evaluation.rank_candidates(candidates, baseline_metrics=baseline)

    assert [row["method_id"] for row in ranked] == ["mean", "cheap", "late"]
    assert ranked[0]["S_mean"] == pytest.approx(0.01)
    assert ranked[0]["S_long"] == pytest.approx(0.01)
    assert ranked[0]["S_min"] == pytest.approx(0.01)


def test_pilot_selects_at_most_two_new_arms_and_always_continues_c0() -> None:
    evaluation = _evaluation_module()
    baseline = _metrics(0.20, 0.20, 0.20, 0.20)
    rows = [
        _candidate("C0", 250, _metrics(0.20, 0.20, 0.20, 0.20)),
        _candidate("C0", 750, _metrics(0.205, 0.205, 0.205, 0.205)),
        _candidate("S-BAL", 250, _metrics(0.19, 0.20, 0.20, 0.20)),
        _candidate("S-BAL", 750, _metrics(0.201, 0.201, 0.201, 0.201)),
        _candidate("S-WORST", 250, _metrics(0.199, 0.199, 0.199, 0.199)),
        _candidate("S-WORST", 750, _metrics(0.202, 0.202, 0.202, 0.202)),
        _candidate("Q-SEM", 250, _metrics(0.18, 0.18, 0.18, 0.18)),
        _candidate("Q-SEM", 750, _metrics(0.203, 0.203, 0.203, 0.203)),
        _candidate("A-OPEN", 250, _metrics(0.10, 0.10, 0.10, 0.10)),
        _candidate("A-OPEN", 750, _metrics(0.11, 0.11, 0.11, 0.11)),
    ]

    decision = evaluation.select_pilot_promotions(
        rows,
        baseline_metrics=baseline,
    )

    assert decision["full_training_arms"] == ["C0", "Q-SEM", "S-WORST"]
    assert decision["promotion_status"] == "TWO_ARM_GATE"
    assert decision["selected_pilot_points"]["C0"]["optimizer_update"] == 750
    assert decision["resume_updates"] == {
        "C0": 750,
        "Q-SEM": 750,
        "S-WORST": 750,
    }


def test_cal_checkpoint_selection_and_sel_parent_gates_are_deterministic() -> None:
    evaluation = _evaluation_module()
    d0 = _metrics(0.20, 0.20, 0.20, 0.20)
    cal = [
        _candidate("C0", 0, d0),
        _candidate("C0", 3000, _metrics(0.206, 0.206, 0.206, 0.206)),
        _candidate("S-BAL", 0, d0, parameters=0),
        _candidate("S-BAL", 3000, _metrics(0.209, 0.209, 0.209, 0.209), parameters=0),
    ]
    selected = evaluation.select_cal_checkpoints(
        cal,
        baseline_metrics=d0,
        completed_full_arms=("C0", "S-BAL"),
    )
    assert selected["C0"]["optimizer_update"] == 3000
    assert selected["S-BAL"]["optimizer_update"] == 3000

    sel_rows = [
        _candidate("C0", 3000, _metrics(0.206, 0.206, 0.206, 0.206)),
        _candidate("S-BAL", 3000, _metrics(0.209, 0.209, 0.209, 0.209)),
    ]
    parent = evaluation.select_perception_parent(
        sel_rows,
        d0_metrics=d0,
        c0_method_id="C0",
    )
    assert parent["status"] == "NEW_PERCEPTION_SELECTED"
    assert parent["selected_method_id"] == "S-BAL"

    fallback = evaluation.select_perception_parent(
        [
            _candidate("C0", 3000, _metrics(0.206, 0.206, 0.206, 0.206)),
            _candidate("S-BAL", 3000, _metrics(0.207, 0.207, 0.207, 0.207)),
        ],
        d0_metrics=d0,
        c0_method_id="C0",
    )
    assert fallback["status"] == "CONTINUATION_ONLY"
    assert fallback["selected_method_id"] == "C0"

    keep = evaluation.select_perception_parent(
        [_candidate("C0", 3000, _metrics(0.201, 0.201, 0.201, 0.201))],
        d0_metrics=d0,
        c0_method_id="C0",
    )
    assert keep == {
        "status": "KEEP_R1",
        "selected_method_id": "R1",
        "selected": None,
    }


def test_refiner_gate_requires_mean_long_and_per_t_floor() -> None:
    evaluation = _evaluation_module()
    parent = _metrics(0.20, 0.20, 0.20, 0.20)

    passed = evaluation.select_refiner(
        _candidate("R-REFINE", 1000, _metrics(0.203, 0.203, 0.203, 0.203)),
        parent_metrics=parent,
    )
    failed = evaluation.select_refiner(
        _candidate("R-REFINE", 1000, _metrics(0.199, 0.205, 0.205, 0.205)),
        parent_metrics=parent,
    )

    assert passed["enabled"] is True
    assert passed["status"] == "REFINER_SELECTED"
    assert failed["enabled"] is False
    assert failed["status"] == "KEEP_PARENT"


def test_incomplete_coverage_is_null_and_locks_are_immutable(tmp_path: Path) -> None:
    evaluation = _evaluation_module()
    baseline = _metrics(0.20, 0.20, 0.20, 0.20)
    incomplete = _candidate(
        "S-BAL",
        750,
        _metrics(0.30, 0.30, 0.30, 0.30),
        coverage="INCOMPLETE",
    )

    comparison = evaluation.compare_candidate(incomplete, baseline_metrics=baseline)

    assert comparison["comparison_status"] == "INCOMPLETE_COVERAGE"
    assert comparison["S_mean"] is None
    lock = tmp_path / "PERCEPTION_LOCK.json"
    payload = {"schema_version": "perception-lock-v1", "selected": "R1"}
    first = evaluation.write_immutable_lock(lock, payload)
    second = evaluation.write_immutable_lock(lock, payload)
    assert first == second
    assert json.loads(lock.read_text(encoding="utf-8")) == first
    with pytest.raises(evaluation.PerceptionEvaluationError, match="immutable"):
        evaluation.write_immutable_lock(lock, {**payload, "selected": "C0"})


def test_extract_pooled_d0_metrics_requires_complete_official_rows() -> None:
    evaluation = _evaluation_module()
    rows = [
        {
            "method": "D0",
            "reference": "all",
            "T": horizon,
            "t_mAP": 0.1 * horizon,
            "reference_count": 4,
            "logical_unit_count": 23,
        }
        for horizon in (2, 3, 4, 5)
    ]
    rows.append(
        {
            "method": "D-LAST",
            "reference": "all",
            "T": 2,
            "t_mAP": 0.99,
            "reference_count": 4,
            "logical_unit_count": 23,
        }
    )

    result = evaluation.extract_pooled_d0_metrics(
        rows,
        expected_reference_count=4,
        expected_logical_unit_count=23,
    )

    assert result == {2: 0.2, 3: pytest.approx(0.3), 4: 0.4, 5: 0.5}

    with pytest.raises(evaluation.PerceptionEvaluationError, match="coverage"):
        evaluation.extract_pooled_d0_metrics(
            rows[:-2],
            expected_reference_count=4,
            expected_logical_unit_count=23,
        )


def test_build_checkpoint_candidate_preserves_coverage_and_identity() -> None:
    evaluation = _evaluation_module()
    candidate = evaluation.build_checkpoint_candidate(
        variant="Q-SEM",
        optimizer_update=250,
        checkpoint_reference="external:training/Q-SEM/update=0250.ckpt",
        checkpoint_sha256="a" * 64,
        metrics={2: 0.2, 3: 0.3, 4: 0.4, 5: 0.5},
        expected_logical_units=23,
        completed_logical_units=23,
        new_parameter_count=8321,
    )

    assert candidate["coverage_status"] == "COMPLETE"
    assert candidate["method_id"] == "Q-SEM"
    assert candidate["optimizer_update"] == 250
    assert candidate["new_parameter_count"] == 8321

    incomplete = evaluation.build_checkpoint_candidate(
        variant="Q-SEM",
        optimizer_update=750,
        checkpoint_reference="external:training/Q-SEM/update=0750.ckpt",
        checkpoint_sha256="b" * 64,
        metrics={2: 0.2, 3: 0.3, 4: 0.4, 5: 0.5},
        expected_logical_units=23,
        completed_logical_units=22,
        new_parameter_count=8321,
    )
    assert incomplete["coverage_status"] == "INCOMPLETE"


def test_live_cache_key_uses_registered_canonical_order() -> None:
    evaluation = _evaluation_module()

    key = evaluation.build_live_cache_key(
        master_sequence_id="scan-a-scan-b-scan-c-scan-d-scan-e",
        reference_scene_id="reference",
        stage_index=2,
        scan_ids=("scan-a", "scan-b", "scan-c", "scan-d", "scan-e"),
        local_window_scan_ids=("scan-b", "scan-c"),
    )

    assert key == {
        "master_sequence_id": "scan-a-scan-b-scan-c-scan-d-scan-e",
        "reference_scene_id": "reference",
        "order_id": "canonical",
        "stage_index": 2,
        "history_scan_ids": ["scan-a", "scan-b", "scan-c"],
        "local_window_scan_ids": ["scan-b", "scan-c"],
    }


def test_live_provenance_binds_dataset_instead_of_legacy_base_cache() -> None:
    evaluation = _evaluation_module()

    provenance = evaluation.build_live_provenance(
        source_commit="a" * 40,
        checkpoint_sha256="b" * 64,
        config_sha256="c" * 64,
        episodes=(
            {
                "reference_id": "ref-a",
                "sequence_id": "seq-a",
                "scan_ids": ["a", "b", "c", "d", "e"],
            },
        ),
    )

    assert provenance == {
        "source_commit": "a" * 40,
        "checkpoint_sha256": "b" * 64,
        "config_sha256": "c" * 64,
        "dataset_sha256": (
            "88ce99d2dd328f885a9d702b5bc7d83bfc2b66f37232c000a886a7671be14f3f"
        ),
    }


def test_live_replay_payload_uses_exact_five_stage_identity() -> None:
    evaluation = _evaluation_module()
    stages = tuple(
        {
            "observation": {"stage": stage},
            "prediction": {"stage": stage},
            "stage_meta": {"stage": stage},
            "target": {"stage": stage},
        }
        for stage in range(5)
    )

    base, supplement = evaluation.build_live_replay_payload(
        reference_id="reference",
        sequence_id="sequence",
        episode_id="episode",
        scan_ids=("s0", "s1", "s2", "s3", "s4"),
        stages=stages,
    )

    assert base["episode"]["scan_ids"] == ["s0", "s1", "s2", "s3", "s4"]
    assert [row["stage_meta"]["stage"] for row in base["stages"]] == list(range(5))
    assert [row["observation"]["stage"] for row in supplement["stages"]] == list(
        range(5)
    )
    assert all(row["base_overlap"] is None for row in supplement["stages"])
    with pytest.raises(evaluation.PerceptionEvaluationError, match="five stages"):
        evaluation.build_live_replay_payload(
            reference_id="reference",
            sequence_id="sequence",
            episode_id="episode",
            scan_ids=("s0", "s1", "s2", "s3", "s4"),
            stages=stages[:4],
        )


def test_trained_checkpoint_load_does_not_reopen_external_r1(tmp_path: Path) -> None:
    evaluation = _evaluation_module()
    system = torch.nn.Linear(2, 1)
    checkpoint = tmp_path / "update=0250.ckpt"
    torch.save({"state_dict": system.state_dict()}, checkpoint)

    audit, digest, reference, parameter_count, sources = (
        evaluation._load_evaluation_weights(
            system=system,
            variant="C0",
            optimizer_update=250,
            r1_checkpoint=tmp_path / "unavailable-r1.ckpt",
            checkpoint=checkpoint,
            scorer_checkpoint=None,
        )
    )

    assert audit["missing_keys"] == []
    assert len(digest) == 64
    assert reference == str(checkpoint)
    assert parameter_count == 0
    assert [source["role"] for source in sources] == [
        "r1_initialization",
        "trained_checkpoint",
    ]


def test_native_prefix_key_is_an_exact_causal_prefix() -> None:
    evaluation = _native_evaluation_module()
    key = evaluation.build_native_prefix_key(
        reference_id="ref-a",
        sequence_id="scan0001_00-scan0001_01-scan0001_02-scan0001_03-scan0001_04",
        context_index=7,
        scan_ids=(
            "scan0001_00",
            "scan0001_01",
            "scan0001_02",
            "scan0001_03",
            "scan0001_04",
        ),
        scan_indices=(10, 11, 12, 13, 14),
        horizon=3,
    )

    assert key == {
        "master_sequence_id": "scan0001_00-scan0001_01-scan0001_02-scan0001_03-scan0001_04",
        "reference_scene_id": "ref-a",
        "order_id": "canonical",
        "context_index": 7,
        "context_scan_indices": [10, 11, 12, 13, 14],
        "horizon": 3,
        "history_scan_ids": ["scan0001_00", "scan0001_01", "scan0001_02"],
        "scan_indices": [10, 11, 12],
        "task_quality": True,
    }
    with pytest.raises(evaluation.PerceptionEvaluationError, match="native prefix"):
        evaluation.build_native_prefix_key(
            reference_id="ref-a",
            sequence_id="bad",
            context_index=7,
            scan_ids=("a", "b"),
            scan_indices=(10, 11),
            horizon=3,
        )
    additional = evaluation.build_native_prefix_key(
        reference_id="ref-b",
        sequence_id="scan0002_00-scan0002_01-scan0002_02",
        context_index=8,
        scan_ids=("scan0002_00", "scan0002_01", "scan0002_02"),
        scan_indices=(20, 21, 22),
        horizon=3,
    )
    assert additional["history_scan_ids"] == [
        "scan0002_00",
        "scan0002_01",
        "scan0002_02",
    ]


def test_live_population_units_preserve_frozen_order_and_exact_coverage() -> None:
    evaluation = _evaluation_module()
    specs = (
        SimpleNamespace(reference_id="r1", source_sequence_id="s1"),
        SimpleNamespace(reference_id="r0", source_sequence_id="s0"),
        SimpleNamespace(reference_id="r1", source_sequence_id="s2"),
    )

    units = evaluation.select_live_population_units(
        role="PB",
        role_references=("r0", "r1"),
        episode_specs=specs,
        expected_logical_units=3,
    )

    assert [unit.logical_unit_id for unit in units] == [
        "PB:00000",
        "PB:00001",
        "PB:00002",
    ]
    assert [unit.sequence_id for unit in units] == ["s1", "s0", "s2"]
    with pytest.raises(evaluation.PerceptionEvaluationError, match="coverage"):
        evaluation.select_live_population_units(
            role="PB",
            role_references=("r0", "missing"),
            episode_specs=specs,
            expected_logical_units=2,
        )


def test_additional_population_uses_native_terminal_horizon_slices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluation = _evaluation_module()
    episode_module = importlib.import_module("datasets.task_memory_episode")
    task_evaluation = importlib.import_module("scripts.evaluate_task_memory")
    preflight = importlib.import_module("scripts.preflight_task_memory_episode")
    expected = {2: (111, 40), 3: (77, 23), 4: (32, 8)}

    class FakeSpec:
        @classmethod
        def from_master(cls, master, **kwargs):
            return SimpleNamespace(
                reference_id=master.reference_id,
                source_sequence_id=master.sequence_id,
                scan_ids=master.scan_ids,
                scan_indices=tuple(range(len(master.scan_ids))),
                context_index=master.context_index,
                **kwargs,
            )

    def fake_base(config, *, data_root, horizon, population_id):
        assert population_id == "additional_native_refs"
        return SimpleNamespace(horizon=horizon)

    def fake_masters(base, **kwargs):
        unit_count, reference_count = expected[base.horizon]
        return tuple(
            SimpleNamespace(
                reference_id=f"ref-{index % reference_count:02d}",
                sequence_id=f"sequence-{base.horizon}-{index:03d}",
                scan_ids=tuple(
                    f"scan{index:04d}_{stage:02d}" for stage in range(base.horizon)
                ),
                role="additional_native_refs",
                context_index=index,
            )
            for index in range(unit_count)
        )

    monkeypatch.setattr(episode_module, "TaskMemoryEpisodeSpec", FakeSpec)
    monkeypatch.setattr(episode_module, "build_native_episode_masters", fake_masters)
    monkeypatch.setattr(task_evaluation, "_rio_population_base", fake_base)
    monkeypatch.setattr(preflight, "load_reference_by_scene", lambda path: {})
    monkeypatch.setattr(preflight, "_role_by_reference", lambda contract: {})

    slices = evaluation.build_additional_population_slices(
        config=object(),
        data_root=Path("/data"),
        metadata_path=Path("/metadata.json"),
        data_contract={},
        role_references=tuple(f"ref-{index:02d}" for index in range(40)),
    )

    assert [(item.horizon, len(item.units)) for item in slices] == [
        (2, 111),
        (3, 77),
        (4, 32),
    ]
    logical_ids = [unit.logical_unit_id for item in slices for unit in item.units]
    assert len(set(logical_ids)) == 220
    assert logical_ids[0].startswith("ADDITIONAL-T2:")
    assert logical_ids[-1].startswith("ADDITIONAL-T4:")
    identity = evaluation.summarize_additional_population_slices(slices)
    assert identity["expected_logical_unit_count"] == 220
    assert identity["expected_prefix_count"] == 220
    assert identity["population_by_horizon"] == {
        "2": {"reference_count": 40, "logical_unit_count": 111},
        "3": {"reference_count": 23, "logical_unit_count": 77},
        "4": {"reference_count": 8, "logical_unit_count": 32},
    }
    assert len(identity["population_manifest_sha256"]) == 64


def test_native_and_refiner_clis_accept_additional_population() -> None:
    native = _native_evaluation_module()
    refiner = importlib.import_module("scripts.perception_refiner_evaluation")

    native_arguments = native._parser().parse_args(
        [
            "--variant",
            "C0",
            "--update",
            "0",
            "--role",
            "ADDITIONAL",
            "--horizons",
            "2",
            "3",
            "4",
        ]
    )
    refiner_arguments = refiner._parser().parse_args(
        [
            "--parent-variant",
            "C0",
            "--parent-update",
            "0",
            "--refiner-update",
            "0",
            "--refiner-checkpoint",
            "/tmp/refiner.ckpt",
            "--role",
            "ADDITIONAL",
        ]
    )

    assert native_arguments.role == "ADDITIONAL"
    assert native_arguments.horizons == [2, 3, 4]
    assert refiner_arguments.role == "ADDITIONAL"


def test_local_t2_summary_preserves_official_metrics_and_identity() -> None:
    evaluation = _local_evaluation_module()
    result = evaluation.build_local_t2_summary(
        variant="C0",
        optimizer_update=3000,
        checkpoint_reference="external:training/C0/update=3000.ckpt",
        checkpoint_sha256="a" * 64,
        config_sha256="b" * 64,
        population_sha256="c" * 64,
        metrics={
            "t_mAP": 0.42,
            "t_mAP50": 0.44,
            "t_mAP25": 0.46,
            "overall_mAP": 0.4,
            "stage1_mAP": 0.5,
            "stage2_mAP": 0.3,
        },
        elapsed_seconds=36.0,
        gpu_name="NVIDIA A40",
    )

    assert result["status"] == "PASS"
    assert result["population_id"] == "official_like_rio_validation_t2_154"
    assert result["validation_sequence_count"] == 154
    assert result["population_manifest_sha256"] == "c" * 64
    assert result["metrics"]["t_mAP"] == pytest.approx(0.42)
    assert result["metrics"]["overall_mAP"] == pytest.approx(0.4)
    assert result["SpatialStageMean"] == pytest.approx(0.4)
    assert result["gpu_hours"] == pytest.approx(0.01)


def test_confirmation_d0_rows_keep_pooled_and_per_reference_metrics() -> None:
    evaluation = _evaluation_module()
    rows = [
        {"method": "D0", "reference": "all", "T": 2, "t_mAP": 0.2},
        {"method": "D0", "reference": "ref-a", "T": 2, "t_mAP": 0.3},
        {"method": "D-LAST", "reference": "all", "T": 2, "t_mAP": 0.4},
    ]

    selected = evaluation.select_d0_metric_rows(rows, data_role="PB")

    assert [row["reference"] for row in selected] == ["all", "ref-a"]
    assert all(row["data_role"] == "PB" for row in selected)
