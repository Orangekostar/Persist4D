from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from datasets.task_memory_episode import NativeEpisodeMaster
from scripts.train_task_memory import (
    FORMAL_OPTIMIZER_UPDATES,
    SMOKE_REFERENCE_ID,
    SMOKE_SEQUENCE_ID,
    VARIANTS,
    _tensor_state_sha256,
    build_real_gradient_smoke_payload,
    build_real_smoke_draw_plan,
    checkpoint_interval,
    classify_smoke_identity_events,
    compose_variant_config,
    materialize_training_contracts,
    remap_r1_training_state,
    resolved_variant_diff,
    validate_run_budget,
    validate_run_directory,
)


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_each_m2_variant_resolves_frozen_training_budget(variant: str) -> None:
    config = compose_variant_config(
        variant,
        pretrained=Path("/tmp/concerto_base.pth"),
        run_dir=Path("/tmp/task-memory-run"),
    )

    assert config.task_memory_training.optimizer_updates == 3000
    assert config.task_memory_training.scheduler_total_updates == 3000
    assert config.task_memory_training.devices == 2
    assert config.task_memory_training.gradient_accumulation == 4
    assert config.task_memory_training.effective_episode_batch == 8
    assert dict(config.task_memory_training.episode_buckets) == {
        "single_scan": 0.2,
        "T2": 0.2,
        "T3": 0.2,
        "T4": 0.2,
        "T5": 0.2,
    }


def test_wbase_to_qindep_diff_is_model_routing_only() -> None:
    baseline = compose_variant_config(
        "W-BASE", pretrained=Path("/tmp/base.pth"), run_dir=Path("/tmp/w")
    )
    query = compose_variant_config(
        "Q-INDEP", pretrained=Path("/tmp/base.pth"), run_dir=Path("/tmp/q")
    )

    diff = resolved_variant_diff(
        baseline,
        query,
        ignore_paths={
            "callbacks",
            "general.experiment_name",
            "general.save_dir",
            "logging",
        },
    )

    assert set(diff) == {
        "model.task_memory_enabled",
        "task_memory_training.state_enabled",
        "task_memory_training.variant",
    }


def test_qindep_to_qtala_diff_is_supervision_only() -> None:
    independent = compose_variant_config(
        "Q-INDEP", pretrained=Path("/tmp/base.pth"), run_dir=Path("/tmp/q1")
    )
    tala = compose_variant_config(
        "Q-TALA", pretrained=Path("/tmp/base.pth"), run_dir=Path("/tmp/q2")
    )

    diff = resolved_variant_diff(
        independent,
        tala,
        ignore_paths={
            "callbacks",
            "general.experiment_name",
            "general.save_dir",
            "logging",
            "task_memory_training.variant",
        },
    )

    assert set(diff) == {"task_memory_training.matcher_mode"}


def test_full_history_control_changes_only_window_and_variant_identity() -> None:
    baseline = compose_variant_config(
        "W-BASE", pretrained=Path("/tmp/base.pth"), run_dir=Path("/tmp/w")
    )
    full_history = compose_variant_config(
        "FH-MATCH", pretrained=Path("/tmp/base.pth"), run_dir=Path("/tmp/fh")
    )

    diff = resolved_variant_diff(
        baseline,
        full_history,
        ignore_paths={
            "callbacks",
            "general.experiment_name",
            "general.save_dir",
            "logging",
            "task_memory_training.variant",
        },
    )

    assert set(diff) == {"task_memory_training.window_mode"}
    assert full_history.task_memory_training.window_mode == "full_history"


def test_formal_and_pilot_share_the_same_3000_update_scheduler() -> None:
    assert FORMAL_OPTIMIZER_UPDATES == 3000
    assert checkpoint_interval(smoke=False) == 750
    assert checkpoint_interval(smoke=True) == 1
    validate_run_budget(
        stop_after_updates=300, devices=2, gradient_accumulation=4, smoke=False
    )
    validate_run_budget(
        stop_after_updates=3000, devices=2, gradient_accumulation=4, smoke=False
    )

    with pytest.raises(ValueError, match="frozen budget"):
        validate_run_budget(
            stop_after_updates=3000,
            devices=1,
            gradient_accumulation=4,
            smoke=False,
        )


def test_run_directory_guard_distinguishes_parent_worker_and_resume(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "formal" / "W-BASE"
    run_dir.mkdir(parents=True)
    (run_dir / "run_plan.json").write_text("{}\n")

    with pytest.raises(RuntimeError, match="not empty"):
        validate_run_directory(
            run_dir, resume=None, is_distributed_worker=False
        )

    validate_run_directory(run_dir, resume=None, is_distributed_worker=True)
    validate_run_directory(
        run_dir,
        resume=run_dir / "last.ckpt",
        is_distributed_worker=False,
    )


def test_real_smoke_draw_plan_repeats_fixed_h5_panel_for_two_updates() -> None:
    scan_ids = tuple(SMOKE_SEQUENCE_ID.split("-"))
    master = NativeEpisodeMaster(
        reference_id=SMOKE_REFERENCE_ID,
        sequence_id=SMOKE_SEQUENCE_ID,
        scan_ids=scan_ids,
        scan_indices=(10, 11, 12, 13, 14),
        role="development",
        context_index=7,
    )

    plan = build_real_smoke_draw_plan(
        (master,), episode_count=4, seed=45
    )

    assert len(plan) == 4
    assert [spec.draw_index for spec in plan] == [0, 1, 2, 3]
    assert all(spec.horizon == 5 and spec.scan_ids == scan_ids for spec in plan)
    assert len({spec.augmentation_seed for spec in plan}) == 4


def test_real_smoke_event_classifier_requires_inheritance_gap_and_newborn() -> None:
    events = classify_smoke_identity_events(
        ({1, 2}, {2, 3}, {1, 2, 3, 4}, {2, 4}, {1, 5})
    )

    assert events["inherited_entity_id"] == 1
    assert events["absent_entity_id"] == 1
    assert events["reappearing_entity_id"] == 1
    assert events["newborn_entity_id"] == 3
    assert events["stage_identity_counts"] == [2, 2, 4, 2, 2]

    with pytest.raises(ValueError, match="reappearance"):
        classify_smoke_identity_events(({1}, {1, 2}, {1, 2}, {1, 2}))


def test_real_gradient_smoke_payload_requires_two_step_learning_path() -> None:
    common = {
        "births": {"bound_births": 1},
        "chunk_boundary_state_detached": [True, True, True],
        "commit_calls": 5,
        "horizon": 5,
        "optimizer_step": True,
        "route_calls": 5,
        "stage_lineage": [
            {"birth_count": 1, "matched_query_count": 1, "stage": stage}
            for stage in range(1, 6)
        ],
        "state_parity_max_abs": 0.0,
        "task_read_parameter_change": {"total_change_norm": 0.5},
    }
    first = {
        **common,
        "chunk_state_gradients": [{"gradient_norm": 0.0}],
        "task_read_gradient": {
            "missing_gradient_names": [],
            "nonfinite_gradient_names": [],
            "output_projection_gradient_norm": 1.0,
            "upstream_gradient_norm": 0.0,
        },
    }
    second = {
        **common,
        "chunk_state_gradients": [{"gradient_norm": 0.25}],
        "task_read_gradient": {
            "missing_gradient_names": [],
            "nonfinite_gradient_names": [],
            "output_projection_gradient_norm": 1.0,
            "upstream_gradient_norm": 0.75,
        },
    }

    payload = build_real_gradient_smoke_payload(
        code_commit="a" * 40,
        common_initialization_sha256="b" * 64,
        initial_task_read_sha256="b" * 64,
        final_task_read_sha256="c" * 64,
        initialization_audit={"output_projection_zero": True},
        load_audit={"r1_subtree_exact": True},
        plan_summary={
            "smoke_events": {
                "inherited_entity_id": 1,
                "absent_entity_id": 1,
                "reappearing_entity_id": 1,
                "newborn_entity_id": 2,
            }
        },
        progress={"completed_global_episodes": 2, "next_draw_index": 2},
        training_audit=(first, second),
        elapsed_seconds=1.5,
        gpu_name="test-gpu",
    )

    assert payload["status"] == "PASS"
    assert all(payload["gates"].values())

    broken_second = {
        **second,
        "task_read_gradient": {
            **second["task_read_gradient"],
            "upstream_gradient_norm": 0.0,
        },
    }
    with pytest.raises(RuntimeError, match="failed gates"):
        build_real_gradient_smoke_payload(
            code_commit="a" * 40,
            common_initialization_sha256="b" * 64,
            initial_task_read_sha256="b" * 64,
            final_task_read_sha256="c" * 64,
            initialization_audit={"output_projection_zero": True},
            load_audit={"r1_subtree_exact": True},
            plan_summary=payload["plan"],
            progress=payload["progress"],
            training_audit=(first, broken_second),
            elapsed_seconds=1.5,
            gpu_name="test-gpu",
        )


def test_r1_criterion_buffers_use_only_explicit_wrapper_aliases() -> None:
    incoming = {
        "model.weight": torch.tensor([1.0]),
        "criterion.empty_weight": torch.tensor([2.0]),
    }
    mapped, aliases = remap_r1_training_state(
        incoming,
        target_keys={
            "model.weight",
            "model.task_read.weight",
            "criterion.base_criterion.empty_weight",
        },
    )

    assert set(mapped) == {
        "model.weight",
        "criterion.base_criterion.empty_weight",
    }
    assert aliases == {
        "criterion.empty_weight": "criterion.base_criterion.empty_weight"
    }


def test_task_read_state_hash_supports_scalar_parameters() -> None:
    first = _tensor_state_sha256(
        {"attention_scale": torch.tensor(128.0).sqrt()}
    )
    second = _tensor_state_sha256(
        {"attention_scale": torch.tensor(128.0).sqrt()}
    )

    assert first == second
    assert len(first) == 64


def test_materialized_training_contracts_freeze_four_controlled_arms(
    tmp_path: Path,
) -> None:
    source_commit = "a" * 40
    common_initialization = "b" * 64

    materialize_training_contracts(
        output_root=tmp_path,
        source_commit=source_commit,
        common_initialization_sha256=common_initialization,
    )

    manifest = json.loads((tmp_path / "variants.json").read_text())
    assert manifest["status"] == "FROZEN_READY_FOR_M2_PREFIX"
    assert manifest["source_commit"] == source_commit
    assert manifest["common_task_read_initialization"]["sha256"] == (
        common_initialization
    )
    assert set(manifest["variants"]) == set(VARIANTS)
    assert all(
        record["state"] == "FROZEN_NOT_RUN"
        for record in manifest["variants"].values()
    )
    assert all(
        "$PERSIST4D_RUN_ROOT" in record["commands"]["pilot_300"]
        for record in manifest["variants"].values()
    )

    for variant in VARIANTS:
        path = tmp_path / "resolved_configs" / f"{variant}.yaml"
        config = OmegaConf.load(path)
        assert config.backbone.name == "external:concerto_pretrained"
        assert config.general.save_dir == (
            f"external:run_root/training/formal/{variant}"
        )
        assert config.task_memory_training.optimizer_updates == 3000
        assert manifest["variants"][variant]["resolved_config_sha256"] == (
            hashlib.sha256(path.read_bytes()).hexdigest()
        )

    comparisons = {
        "W-BASE_to_Q-INDEP.json": {
            "model.task_memory_enabled",
            "task_memory_training.state_enabled",
        },
        "Q-INDEP_to_Q-TALA.json": {"task_memory_training.matcher_mode"},
        "W-BASE_to_FH-MATCH.json": {"task_memory_training.window_mode"},
    }
    for filename, expected in comparisons.items():
        payload = json.loads((tmp_path / "config_diffs" / filename).read_text())
        assert set(payload["controlled_differences"]) == expected


def test_materialized_training_exposure_and_public_paths_are_exact(
    tmp_path: Path,
) -> None:
    materialize_training_contracts(
        output_root=tmp_path,
        source_commit="a" * 40,
        common_initialization_sha256="b" * 64,
    )

    with (tmp_path / "costs_and_exposure.csv").open(newline="") as handle:
        rows = {row["variant"]: row for row in csv.DictReader(handle)}

    assert set(rows) == set(VARIANTS)
    for variant, row in rows.items():
        assert row["status"] == "NOT_RUN"
        assert row["optimizer_updates"] == "3000"
        assert row["global_episode_draws"] == "24000"
        assert row["single_scan_episodes"] == "4800"
        assert row["T2_episodes"] == "4800"
        assert row["T3_episodes"] == "4800"
        assert row["T4_episodes"] == "4800"
        assert row["T5_episodes"] == "4800"
        assert row["loss_evaluated_stages"] == "72000"
        assert row["actual_gpu_hours"] == ""
        expected_inputs = "168000" if variant == "FH-MATCH" else "120000"
        assert row["encoder_scan_inputs"] == expected_inputs

    public_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(tmp_path.rglob("*"))
        if path.is_file()
    )
    for forbidden in ("/home/", "/mnt/", "192.168.", "node107", "ww@"):
        assert forbidden not in public_text
