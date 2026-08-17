from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import fields, replace
from itertools import product
from pathlib import Path

import numpy as np
import pytest
import torch

import scripts.evaluate_persist4d as evaluator
from scripts.evaluate_persist4d import (
    METHOD_NAME,
    SequenceAccumulator,
    _accumulate_shared_stage,
    _begin_source_tree_contract,
    _compose_runtime_config,
    _derive_conclusion,
    _finalize_source_tree_contract,
    _legacy_parity_result,
    _render_markdown,
    _summarize_method_metrics,
    identity_diagnostics,
    local_identity_ids,
    main,
)


def test_runtime_config_explicitly_enables_query_feature_export() -> None:
    config, _ = _compose_runtime_config()

    assert config.model.return_query_features is True


def test_identity_diagnostics_count_switches_and_reactivation() -> None:
    result = identity_diagnostics(
        gt_ids_by_stage=[[7], [7], [], [7]],
        predicted_ids_by_stage=[[3], [3], [], [3]],
    )

    assert result == {
        "matched_identity_observations": 3,
        "identity_switches": 0,
        "reactivation_events": 1,
        "correct_reactivations": 1,
        "reactivation_accuracy": 1.0,
    }


def test_identity_diagnostics_count_changed_slot_as_incorrect_reactivation() -> None:
    result = identity_diagnostics(
        gt_ids_by_stage=[[7], [7], [], [7]],
        predicted_ids_by_stage=[[3], [3], [], [4]],
    )

    assert result == {
        "matched_identity_observations": 3,
        "identity_switches": 1,
        "reactivation_events": 1,
        "correct_reactivations": 0,
        "reactivation_accuracy": 0.0,
    }


def test_identity_diagnostics_returns_none_without_reactivation() -> None:
    result = identity_diagnostics([[1], [1]], [[2], [2]])

    assert result["reactivation_events"] == 0
    assert result["reactivation_accuracy"] is None


@pytest.mark.parametrize(
    ("gt_ids", "predicted_ids", "message"),
    [
        ([[1]], [[1], [1]], "stage"),
        ([[1], [2]], [[1], []], "align"),
        ([[1, 1]], [[2, 3]], "duplicate"),
        ([[-1]], [[2]], "non-negative"),
        ([[1]], [[-1]], "non-negative"),
        ([[True]], [[2]], "integral"),
        ([[1]], [[False]], "integral"),
        ([[1.0]], [[2]], "integral"),
    ],
)
def test_identity_diagnostics_rejects_malformed_ids(
    gt_ids: object,
    predicted_ids: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        identity_diagnostics(gt_ids, predicted_ids)


def test_identity_diagnostics_rejects_duplicate_predicted_ids() -> None:
    with pytest.raises(ValueError, match="predicted.*duplicate"):
        identity_diagnostics([[1, 2]], [[7, 7]])


def test_sequence_accumulator_stores_cpu_masks_and_class_means() -> None:
    accumulator = SequenceAccumulator.empty(capacity=3, class_count=2)
    first_masks = torch.tensor(
        [[1.0, 0.0, 2.0], [0.0, 1.0, 0.0]],
        requires_grad=True,
    )
    first_prob = torch.tensor(
        [[0.8, 0.2], [0.1, 0.9]],
        requires_grad=True,
    )

    accumulator.add_stage(
        first_masks,
        first_prob,
        torch.tensor([2, 0]),
    )
    accumulator.add_stage(
        [torch.tensor([0.0, 1.0]), torch.tensor([1.0, 1.0])],
        torch.tensor([[0.6, 0.4], [0.4, 0.6]]),
        torch.tensor([2, -1]),
    )

    assert [set(stage) for stage in accumulator.stage_masks] == [{0, 2}, {2}]
    assert torch.equal(
        accumulator.stage_masks[0][2],
        torch.tensor([True, False, True]),
    )
    assert accumulator.stage_masks[0][2].device.type == "cpu"
    assert not accumulator.stage_masks[0][2].requires_grad
    assert torch.equal(accumulator.class_prob_count, torch.tensor([1, 0, 2]))
    torch.testing.assert_close(
        accumulator.class_prob_mean(),
        torch.tensor([[0.1, 0.9], [0.0, 0.0], [0.7, 0.3]]),
    )


def test_sequence_accumulator_clones_stored_masks() -> None:
    accumulator = SequenceAccumulator.empty(capacity=2, class_count=2)
    source_masks = torch.tensor([[True, False]])

    accumulator.add_stage(
        source_masks,
        torch.tensor([[0.25, 0.75]]),
        torch.tensor([1]),
    )
    source_masks.fill_(False)

    assert torch.equal(
        accumulator.stage_masks[0][1],
        torch.tensor([True, False]),
    )


def test_local_identity_baseline_uses_query_index_for_valid_observations() -> None:
    valid = torch.tensor([True, False, True, True])

    identities = local_identity_ids(valid, capacity=4)

    assert torch.equal(identities, torch.tensor([0, -1, 2, 3]))
    assert identities.device == valid.device


@pytest.mark.parametrize(
    ("valid", "capacity", "message"),
    [
        (torch.ones(2, 1, dtype=torch.bool), 2, "shape"),
        (torch.ones(2), 2, "bool"),
        (torch.ones(2, dtype=torch.bool), 3, "capacity"),
        (torch.ones(2, dtype=torch.bool), True, "capacity"),
    ],
)
def test_local_identity_baseline_rejects_invalid_contract(
    valid: object,
    capacity: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        local_identity_ids(valid, capacity=capacity)


def test_stage_comparison_reuses_masks_and_classes_with_local_baseline_ids() -> None:
    persistent = SequenceAccumulator.empty(capacity=4, class_count=2)
    baseline = SequenceAccumulator.empty(capacity=4, class_count=2)
    masks = torch.tensor(
        [
            [True, False, False],
            [False, True, False],
            [False, False, True],
            [True, True, False],
        ]
    )
    class_prob = torch.tensor(
        [[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.4, 0.6]]
    )

    baseline_ids = _accumulate_shared_stage(
        persistent=persistent,
        baseline=baseline,
        masks=masks,
        class_prob=class_prob,
        persistent_slot_ids=torch.tensor([2, -1, 0, 1]),
        valid_observations=torch.tensor([True, False, True, True]),
    )

    assert torch.equal(baseline_ids, torch.tensor([0, -1, 2, 3]))
    assert set(persistent.stage_masks[0]) == {0, 1, 2}
    assert set(baseline.stage_masks[0]) == {0, 2, 3}
    torch.testing.assert_close(persistent.class_prob_sum[2], class_prob[0])
    torch.testing.assert_close(baseline.class_prob_sum[0], class_prob[0])
    torch.testing.assert_close(persistent.class_prob_sum[0], class_prob[2])
    torch.testing.assert_close(baseline.class_prob_sum[2], class_prob[2])


def test_method_metric_summary_is_shared_by_persistent_and_baseline() -> None:
    class Metric:
        def compute(self):
            return {
                "val_mean_t-AP": torch.tensor(0.25, dtype=torch.float64),
                "val_mean_t-REC": torch.tensor(0.5, dtype=torch.float64),
                "val_mean_stage1-AP": torch.tensor(0.2, dtype=torch.float64),
                "val_mean_stage2-AP": torch.tensor(0.3, dtype=torch.float64),
            }

    identity = {
        "matched_identity_observations": 4,
        "identity_switches": 1,
        "reactivation_events": 2,
        "correct_reactivations": 1,
    }

    persistent = _summarize_method_metrics(
        Metric(),
        horizon=2,
        identity_totals=identity,
        rejected_births=3,
    )
    baseline = _summarize_method_metrics(
        Metric(),
        horizon=2,
        identity_totals=identity,
    )

    expected = {
        "t_mAP": 0.25,
        "t_REC": 0.5,
        "per_stage_AP": {"1": 0.2, "2": 0.3},
        **identity,
        "reactivation_accuracy": 0.5,
    }
    assert persistent == {**expected, "rejected_births": 3}
    assert baseline == expected


def _git(repo: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *arguments],
        cwd=repo,
        text=True,
        capture_output=True,
        check=True,
    )


def _temporary_git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "tests@example.invalid")
    _git(repo, "config", "user.name", "Persist4D Tests")
    (repo / ".gitignore").write_text("data/\nthird_party/\n", encoding="utf-8")
    (repo / "source.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(repo, "add", ".gitignore", "source.py")
    _git(repo, "commit", "-qm", "initial")
    return repo


@pytest.mark.parametrize("staged", [False, True], ids=["unstaged", "staged"])
def test_source_tree_contract_rejects_tracked_source_changes(
    tmp_path: Path,
    staged: bool,
) -> None:
    repo = _temporary_git_repo(tmp_path)
    (repo / "source.py").write_text("VALUE = 2\n", encoding="utf-8")
    if staged:
        _git(repo, "add", "source.py")

    with pytest.raises(RuntimeError, match="tracked source"):
        _begin_source_tree_contract(
            repo_root=repo,
            output_paths=(repo / "result.json", repo / "result.md"),
        )


def test_source_tree_contract_rejects_untracked_non_output(
    tmp_path: Path,
) -> None:
    repo = _temporary_git_repo(tmp_path)
    (repo / "notes.txt").write_text("not an artifact\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="untracked"):
        _begin_source_tree_contract(
            repo_root=repo,
            output_paths=(repo / "result.json", repo / "result.md"),
        )


def test_source_tree_contract_rejects_hidden_index_flags(tmp_path: Path) -> None:
    repo = _temporary_git_repo(tmp_path)
    _git(repo, "update-index", "--skip-worktree", "source.py")

    with pytest.raises(RuntimeError, match="index flags"):
        _begin_source_tree_contract(
            repo_root=repo,
            output_paths=(repo / "result.json", repo / "result.md"),
        )


def test_source_tree_contract_allows_only_exact_outputs_and_ignored_inputs(
    tmp_path: Path,
) -> None:
    repo = _temporary_git_repo(tmp_path)
    output = repo / "artifacts" / "P5" / "result.json"
    markdown = repo / "artifacts" / "P5" / "result.md"
    output.parent.mkdir(parents=True)
    output.write_text("{}\n", encoding="utf-8")
    markdown.write_text("report\n", encoding="utf-8")
    (repo / "data").mkdir()
    (repo / "data" / "dataset.bin").write_bytes(b"ignored")
    (repo / "third_party").mkdir()
    (repo / "third_party" / "dependency.py").write_text(
        "ignored = True\n",
        encoding="utf-8",
    )

    guard = _begin_source_tree_contract(
        repo_root=repo,
        output_paths=(output, markdown),
    )
    contract = _finalize_source_tree_contract(guard)

    commit = _git(repo, "rev-parse", "HEAD").stdout.strip()
    assert contract == {
        "schema_version": 1,
        "status": "pass",
        "source_commit": commit,
        "tracked_tree_clean": True,
        "index_clean": True,
        "allowed_untracked_outputs": [
            "repo:artifacts/P5/result.json",
            "repo:artifacts/P5/result.md",
        ],
        "only_declared_outputs_untracked": True,
        "generation_head_unchanged": True,
    }


def test_source_tree_contract_rejects_head_change_during_generation(
    tmp_path: Path,
) -> None:
    repo = _temporary_git_repo(tmp_path)
    guard = _begin_source_tree_contract(
        repo_root=repo,
        output_paths=(repo / "result.json", repo / "result.md"),
    )
    (repo / "source.py").write_text("VALUE = 3\n", encoding="utf-8")
    _git(repo, "add", "source.py")
    _git(repo, "commit", "-qm", "change head")

    with pytest.raises(RuntimeError, match="HEAD changed"):
        _finalize_source_tree_contract(guard)


def test_source_tree_contract_rechecks_head_after_all_status_queries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _temporary_git_repo(tmp_path)
    guard = _begin_source_tree_contract(
        repo_root=repo,
        output_paths=(repo / "result.json", repo / "result.md"),
    )
    original_git_paths = evaluator._git_paths

    def git_paths_with_head_change(
        repo_root: Path,
        *arguments: str,
    ) -> tuple[str, ...]:
        paths = original_git_paths(repo_root, *arguments)
        if arguments[:2] == ("ls-files", "--others"):
            _git(repo, "commit", "--allow-empty", "-qm", "change during status")
        return paths

    monkeypatch.setattr(evaluator, "_git_paths", git_paths_with_head_change)

    with pytest.raises(RuntimeError, match="HEAD changed"):
        _finalize_source_tree_contract(guard)


def test_real_cli_path_publishes_only_declared_outputs_from_clean_head(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _temporary_git_repo(tmp_path)
    output = repo / "artifacts" / "P5" / "result.json"
    markdown = repo / "artifacts" / "P5" / "result.md"
    commit = _git(repo, "rev-parse", "HEAD").stdout.strip()

    def runner(_args):
        artifact = _complete_artifact()
        artifact["source_commit"] = commit
        artifact["source_tree_contract"]["source_commit"] = commit
        return artifact

    monkeypatch.setattr(evaluator, "PROJECT_ROOT", repo)
    monkeypatch.setattr(evaluator, "_validate_options", lambda _args: None)
    monkeypatch.setattr(evaluator, "run_real_evaluation", runner)

    return_code = main(
        [
            "--output",
            str(output),
            "--markdown",
            str(markdown),
        ]
    )

    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert return_code == 0
    assert artifact["source_tree_contract"]["allowed_untracked_outputs"] == [
        "repo:artifacts/P5/result.json",
        "repo:artifacts/P5/result.md",
    ]
    assert markdown.read_text(encoding="utf-8") == _render_markdown(artifact)


def test_sequence_accumulator_has_only_fixed_bookkeeping_fields() -> None:
    accumulator = SequenceAccumulator.empty(capacity=4, class_count=3)

    assert [field.name for field in fields(accumulator)] == [
        "capacity",
        "stage_masks",
        "class_prob_sum",
        "class_prob_count",
    ]
    assert accumulator.class_prob_sum.shape == (4, 3)
    assert accumulator.class_prob_count.shape == (4,)
    assert not hasattr(accumulator, "query_features")
    with pytest.raises(TypeError, match="query_features"):
        accumulator.add_stage(
            torch.ones(1, 1),
            torch.ones(1, 3),
            torch.zeros(1, dtype=torch.long),
            query_features=torch.ones(1, 8),
        )


@pytest.mark.parametrize(
    ("masks", "class_prob", "slot_ids", "message"),
    [
        (torch.ones(2, 3, 1), torch.ones(2, 2), torch.tensor([0, 1]), "masks"),
        (torch.ones(2, 3), torch.ones(2, 2, 1), torch.tensor([0, 1]), "class_prob"),
        (torch.ones(2, 3), torch.ones(2, 2), torch.tensor([[0, 1]]), "slot_ids"),
        (torch.ones(1, 3), torch.ones(2, 2), torch.tensor([0, 1]), "Q"),
        (torch.ones(2, 3), torch.ones(2, 3), torch.tensor([0, 1]), "class"),
        (torch.ones(2, 3), torch.ones(2, 2), torch.tensor([0, 0]), "more than once"),
        (torch.ones(2, 3), torch.ones(2, 2), torch.tensor([-2, 0]), "range"),
        (torch.ones(2, 3), torch.ones(2, 2), torch.tensor([0, 3]), "range"),
        (torch.ones(2, 3), torch.ones(2, 2), torch.tensor([0.0, 1.0]), "integer"),
        (
            torch.tensor([[float("nan")]]),
            torch.ones(1, 2),
            torch.tensor([0]),
            "finite",
        ),
        (
            torch.ones(1, 1),
            torch.tensor([[float("inf"), 0.0]]),
            torch.tensor([0]),
            "finite",
        ),
        ([torch.ones(2), torch.ones(3)], torch.ones(2, 2), torch.tensor([0, 1]), "shape"),
    ],
)
def test_sequence_accumulator_rejects_malformed_stage_inputs(
    masks: object,
    class_prob: object,
    slot_ids: object,
    message: str,
) -> None:
    accumulator = SequenceAccumulator.empty(capacity=3, class_count=2)

    with pytest.raises(ValueError, match=message):
        accumulator.add_stage(masks, class_prob, slot_ids)

    assert accumulator.stage_masks == []
    assert torch.count_nonzero(accumulator.class_prob_sum).item() == 0
    assert torch.count_nonzero(accumulator.class_prob_count).item() == 0


@pytest.mark.parametrize(
    ("capacity", "class_count", "message"),
    [
        (0, 2, "capacity"),
        (True, 2, "capacity"),
        (2, 0, "class_count"),
        (2, 1.0, "class_count"),
    ],
)
def test_sequence_accumulator_empty_rejects_invalid_dimensions(
    capacity: object,
    class_count: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        SequenceAccumulator.empty(capacity=capacity, class_count=class_count)


def _metric_block(
    horizon: int,
    *,
    t_map: float,
    t_rec: float,
    per_stage_ap: float,
    observations: int | None = None,
    switches: int = 0,
    reactivations: int = 0,
    correct_reactivations: int = 0,
) -> dict[str, object]:
    accuracy = (
        correct_reactivations / reactivations if reactivations else None
    )
    return {
        "t_mAP": t_map,
        "t_REC": t_rec,
        "per_stage_AP": {
            str(stage): per_stage_ap for stage in range(1, horizon + 1)
        },
        "matched_identity_observations": (
            horizon if observations is None else observations
        ),
        "identity_switches": switches,
        "reactivation_events": reactivations,
        "correct_reactivations": correct_reactivations,
        "reactivation_accuracy": accuracy,
    }


def _metric_delta(
    persistent: dict[str, object],
    baseline: dict[str, object],
) -> dict[str, object]:
    return {
        "t_mAP": persistent["t_mAP"] - baseline["t_mAP"],
        "t_REC": persistent["t_REC"] - baseline["t_REC"],
        "per_stage_AP": {
            stage: persistent["per_stage_AP"][stage]
            - baseline["per_stage_AP"][stage]
            for stage in persistent["per_stage_AP"]
        },
        "matched_identity_observations": (
            persistent["matched_identity_observations"]
            - baseline["matched_identity_observations"]
        ),
        "identity_switches": (
            persistent["identity_switches"] - baseline["identity_switches"]
        ),
        "reactivation_events": (
            persistent["reactivation_events"]
            - baseline["reactivation_events"]
        ),
        "correct_reactivations": (
            persistent["correct_reactivations"]
            - baseline["correct_reactivations"]
        ),
        "reactivation_accuracy": (
            persistent["reactivation_accuracy"]
            - baseline["reactivation_accuracy"]
            if persistent["reactivation_accuracy"] is not None
            and baseline["reactivation_accuracy"] is not None
            else None
        ),
    }


def _horizon_result(horizon: int) -> dict[str, object]:
    persistent = _metric_block(
        horizon,
        t_map=0.25,
        t_rec=0.5,
        per_stage_ap=0.25,
    )
    persistent["rejected_births"] = 0
    baseline = _metric_block(
        horizon,
        t_map=0.2,
        t_rec=0.5,
        per_stage_ap=0.2,
    )
    return {
        "T": horizon,
        "loaded_sequences": {2: 154, 3: 120, 4: 75, 5: 43}[horizon],
        "persistent": persistent,
        "internal_baseline": baseline,
        "delta": _metric_delta(persistent, baseline),
        "resources": {
            "peak_allocated_cuda_bytes": 1024,
            "mean_latency_ms": 2.0,
            "throughput_sequences_per_second": 500.0,
            "serialized_state_bytes": 256,
        },
    }


def _complete_artifact() -> dict[str, object]:
    return {
        "schema_version": 2,
        "status": "pass",
        "method": METHOD_NAME,
        "source_commit": "a" * 40,
        "source_tree_contract": {
            "schema_version": 1,
            "status": "pass",
            "source_commit": "a" * 40,
            "tracked_tree_clean": True,
            "index_clean": True,
            "allowed_untracked_outputs": [],
            "only_declared_outputs_untracked": True,
            "generation_head_unchanged": True,
        },
        "checkpoint": {
            "ref": "repo:checkpoints/rescene4d_concerto_t2_repro.ckpt",
            "sha256": "b" * 64,
        },
        "settings": {
            "capacity": 100,
            "local_window": 2,
            "internal_baseline_identity": "local_query_index",
            "shared_rescene_outputs": True,
        },
        "legacy_parity": {
            "verified_by": "in_evaluator_fixed_t2_sample_toggle",
            "sample_count": 1,
            "legacy_predictions_unchanged": True,
            "query_feature_shape": [1, 100, 128],
        },
        "horizons": [_horizon_result(value) for value in (2, 3, 4, 5)],
        "bounded_state": {
            "constant_shape": True,
            "maximum_state_bytes": 256,
        },
        "conclusion": {
            "label": "P5_ASSOCIATION_DIAGNOSIS",
            "reason": "bounded_execution_without_t4_t5_identity_improvement",
            "identity_improvements": [],
        },
        "errors": [],
    }


def _set_artifact_path(
    artifact: dict[str, object],
    path: tuple[str | int, ...],
    value: object,
) -> None:
    cursor: object = artifact
    for component in path[:-1]:
        if isinstance(component, int):
            assert isinstance(cursor, list)
        else:
            assert isinstance(cursor, dict)
        cursor = cursor[component]
    final = path[-1]
    if isinstance(final, int):
        assert isinstance(cursor, list)
    else:
        assert isinstance(cursor, dict)
    cursor[final] = value


def _run_mock_artifact(
    tmp_path: Path,
    artifact: dict[str, object],
) -> tuple[int, dict[str, object]]:
    output = tmp_path / "result.json"
    return_code = main(
        ["--output", str(output)],
        runner=lambda _args: artifact,
    )
    return return_code, json.loads(output.read_text(encoding="utf-8"))


def test_legacy_parity_result_requires_exact_legacy_predictions() -> None:
    disabled = {
        "pred_logits": torch.tensor([[1.0, 2.0]]),
        "nested": [torch.tensor([3]), {"value": "same"}],
    }
    enabled = {
        "pred_logits": disabled["pred_logits"].clone(),
        "nested": [torch.tensor([3]), {"value": "same"}],
        "query_features": torch.ones(1, 4, 8),
        "persistent_slot_ids": torch.arange(4).unsqueeze(0),
        "persistent_association_scores": torch.ones(1, 4),
        "persistent_rejected_births": torch.zeros(1, 4, dtype=torch.bool),
    }

    result = _legacy_parity_result(disabled, enabled, capacity=4)

    assert result == {
        "verified_by": "in_evaluator_fixed_t2_sample_toggle",
        "sample_count": 1,
        "legacy_predictions_unchanged": True,
        "query_feature_shape": [1, 4, 8],
    }


def test_legacy_parity_result_rejects_changed_legacy_prediction() -> None:
    disabled = {"pred_logits": torch.tensor([[1.0, 2.0]])}
    enabled = {
        "pred_logits": torch.tensor([[1.0, 3.0]]),
        "query_features": torch.ones(1, 2, 4),
        "persistent_slot_ids": torch.arange(2).unsqueeze(0),
        "persistent_association_scores": torch.ones(1, 2),
        "persistent_rejected_births": torch.zeros(1, 2, dtype=torch.bool),
    }

    with pytest.raises(RuntimeError, match="legacy prediction"):
        _legacy_parity_result(disabled, enabled, capacity=2)


def test_legacy_parity_handles_numpy_outputs() -> None:
    disabled = {
        "sampled_coords": np.array([[1.0, 2.0]], dtype=np.float32),
    }
    enabled = {
        "sampled_coords": disabled["sampled_coords"].copy(),
        "query_features": torch.ones(1, 2, 4),
        "persistent_slot_ids": torch.arange(2).unsqueeze(0),
        "persistent_association_scores": torch.ones(1, 2),
        "persistent_rejected_births": torch.zeros(1, 2, dtype=torch.bool),
    }

    result = _legacy_parity_result(disabled, enabled, capacity=2)

    assert result["legacy_predictions_unchanged"] is True


@pytest.mark.parametrize("same_type", [True, False], ids=["same", "different"])
def test_legacy_parity_rejects_unknown_runtime_outputs(same_type: bool) -> None:
    class FirstUnknown:
        pass

    class SecondUnknown:
        pass

    disabled = {"runtime_output": FirstUnknown()}
    enabled = {
        "runtime_output": FirstUnknown() if same_type else SecondUnknown(),
        "query_features": torch.ones(1, 2, 4),
        "persistent_slot_ids": torch.arange(2).unsqueeze(0),
        "persistent_association_scores": torch.ones(1, 2),
        "persistent_rejected_births": torch.zeros(1, 2, dtype=torch.bool),
    }

    with pytest.raises(ValueError, match="unsupported legacy parity value"):
        _legacy_parity_result(disabled, enabled, capacity=2)


def test_legacy_parity_rejects_numpy_dtype_change() -> None:
    disabled = {"sampled_coords": np.array([[1.0, 2.0]], dtype=np.float32)}
    enabled = {
        "sampled_coords": np.array([[1.0, 2.0]], dtype=np.float64),
        "query_features": torch.ones(1, 2, 4),
        "persistent_slot_ids": torch.arange(2).unsqueeze(0),
        "persistent_association_scores": torch.ones(1, 2),
        "persistent_rejected_births": torch.zeros(1, 2, dtype=torch.bool),
    }

    with pytest.raises(RuntimeError, match="legacy prediction"):
        _legacy_parity_result(disabled, enabled, capacity=2)


def test_legacy_snapshot_records_tensor_metadata_and_clones_content() -> None:
    source = torch.tensor([[1.0, 2.0]], dtype=torch.float32)

    snapshot = evaluator._legacy_value_snapshot(source, path="pred_logits")
    source.fill_(9.0)

    assert snapshot.device == torch.device("cpu")
    assert snapshot.dtype == torch.float32
    assert snapshot.shape == (1, 2)
    assert torch.equal(snapshot.content, torch.tensor([[1.0, 2.0]]))

    changed_device = replace(snapshot, device=torch.device("meta"))
    with pytest.raises(RuntimeError, match="legacy prediction"):
        evaluator._require_legacy_value_equal(
            changed_device,
            snapshot,
            path="pred_logits",
        )


def test_legacy_snapshot_records_ndarray_metadata_and_copies_content() -> None:
    source = np.array([[1.0, 2.0]], dtype=np.float32)

    snapshot = evaluator._legacy_value_snapshot(source, path="sampled_coords")
    source.fill(9.0)

    assert snapshot.dtype == np.dtype(np.float32)
    assert snapshot.shape == (1, 2)
    assert np.array_equal(
        snapshot.content,
        np.array([[1.0, 2.0]], dtype=np.float32),
    )


@pytest.mark.parametrize(
    ("metric", "expected_evidence"),
    [
        ("t_REC", "T4:t_REC"),
        ("identity_switches", "T4:identity_switches"),
        ("reactivation_accuracy", "T4:reactivation_accuracy"),
    ],
)
def test_conclusion_pass_requires_derived_t4_or_t5_identity_improvement(
    metric: str,
    expected_evidence: str,
) -> None:
    artifact = _complete_artifact()
    horizon = artifact["horizons"][2]
    persistent = horizon["persistent"]
    baseline = horizon["internal_baseline"]
    if metric == "t_REC":
        persistent["t_REC"] = 0.6
    elif metric == "identity_switches":
        baseline["identity_switches"] = 1
    else:
        persistent.update(
            {
                "identity_switches": 1,
                "reactivation_events": 1,
                "correct_reactivations": 1,
                "reactivation_accuracy": 1.0,
            }
        )
        baseline.update(
            {
                "identity_switches": 1,
                "reactivation_events": 1,
                "correct_reactivations": 0,
                "reactivation_accuracy": 0.0,
            }
        )
    horizon["delta"] = _metric_delta(persistent, baseline)

    conclusion = _derive_conclusion(
        artifact["horizons"],
        artifact["bounded_state"],
    )

    assert conclusion == {
        "label": "P5_MVP_PASS",
        "reason": "bounded_execution_with_t4_t5_identity_improvement",
        "identity_improvements": [expected_evidence],
    }


def test_conclusion_does_not_compare_switches_with_different_observations() -> None:
    artifact = _complete_artifact()
    horizon = artifact["horizons"][2]
    horizon["persistent"]["matched_identity_observations"] = 3
    horizon["internal_baseline"]["matched_identity_observations"] = 4
    horizon["internal_baseline"]["identity_switches"] = 1
    horizon["delta"] = _metric_delta(
        horizon["persistent"],
        horizon["internal_baseline"],
    )

    conclusion = _derive_conclusion(
        artifact["horizons"],
        artifact["bounded_state"],
    )

    assert conclusion["label"] == "P5_ASSOCIATION_DIAGNOSIS"
    assert conclusion["identity_improvements"] == []


def test_markdown_renderer_covers_evidence_contract() -> None:
    report = _render_markdown(_complete_artifact())

    assert "Checkpoint SHA-256: `" + "b" * 64 + "`" in report
    assert "Legacy predictions unchanged: `true`" in report
    assert "Internal baseline identity: `local_query_index`" in report
    assert "| 2 | 154 | persistent |" in report
    assert "| 2 | 154 | internal_baseline |" in report
    assert "| 2 | delta (persistent - baseline) |" in report
    assert "Peak CUDA bytes" in report
    assert "Conclusion: `P5_ASSOCIATION_DIAGNOSIS`" in report
    assert (
        "Reason: `bounded_execution_without_t4_t5_identity_improvement`"
        in report
    )


def test_importing_evaluator_does_not_import_real_runtime_dependencies() -> None:
    code = """
import sys
import scripts.evaluate_persist4d  # noqa: F401
for name in ('hydra', 'pytorch_lightning', 'trainer.trainer', 'models.rescene'):
    assert name not in sys.modules, name
"""

    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_cli_requires_output_path() -> None:
    assert main([], runner=lambda _args: _complete_artifact()) == 2


def test_cli_writes_complete_mock_result_atomically(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "result.json"
    markdown = tmp_path / "nested" / "result.md"
    seen = []

    def runner(args):
        seen.append(args)
        assert not output.exists()
        return _complete_artifact()

    return_code = main(
        ["--output", str(output), "--markdown", str(markdown)],
        runner=runner,
    )

    assert return_code == 0
    assert len(seen) == 1
    assert json.loads(output.read_text(encoding="utf-8")) == _complete_artifact()
    assert markdown.read_text(encoding="utf-8") == _render_markdown(
        _complete_artifact()
    )
    assert set(output.parent.iterdir()) == {output, markdown}


def test_cli_rejects_existing_output_without_running_or_overwriting(
    tmp_path: Path,
) -> None:
    output = tmp_path / "result.json"
    output.write_text("keep-me\n", encoding="utf-8")
    called = False

    def runner(_args):
        nonlocal called
        called = True
        return _complete_artifact()

    return_code = main(["--output", str(output)], runner=runner)

    assert return_code != 0
    assert not called
    assert output.read_text(encoding="utf-8") == "keep-me\n"


def test_cli_rejects_external_checkpoint_before_running(
    tmp_path: Path,
) -> None:
    output = tmp_path / "result.json"
    checkpoint = tmp_path / "external.ckpt"
    checkpoint.write_bytes(b"not-a-formal-checkpoint")
    call_count = 0

    def runner(_args):
        nonlocal call_count
        call_count += 1
        return _complete_artifact()

    return_code = main(
        ["--output", str(output), "--checkpoint", str(checkpoint)],
        runner=runner,
    )

    report_text = output.read_text(encoding="utf-8")
    assert return_code != 0
    assert call_count == 0
    assert json.loads(report_text) == {
        "schema_version": 2,
        "status": "failed",
        "method": METHOD_NAME,
        "conclusion": {
            "label": "P5_STREAMING_BLOCKED",
            "reason": "evaluation_failed",
            "identity_improvements": [],
        },
        "errors": [{"type": "ValueError", "code": "invalid_input"}],
    }
    assert str(checkpoint) not in report_text


def test_cli_rejects_checkpoint_symlink_before_running(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    checkpoint = tmp_path / "checkpoint.ckpt"
    checkpoint.symlink_to(
        Path(__file__).resolve().parents[1]
        / "checkpoints"
        / "rescene4d_concerto_t2_repro.ckpt"
    )
    call_count = 0

    def runner(_args):
        nonlocal call_count
        call_count += 1
        return _complete_artifact()

    return_code = main(
        ["--output", str(output), "--checkpoint", str(checkpoint)],
        runner=runner,
    )

    assert return_code != 0
    assert call_count == 0
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "failed"


def test_cli_writes_failed_artifact_when_runner_raises(tmp_path: Path) -> None:
    output = tmp_path / "failure" / "result.json"

    def runner(_args):
        raise RuntimeError("synthetic evaluator failure")

    return_code = main(["--output", str(output)], runner=runner)

    assert return_code != 0
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "schema_version": 2,
        "status": "failed",
        "method": METHOD_NAME,
        "conclusion": {
            "label": "P5_STREAMING_BLOCKED",
            "reason": "evaluation_failed",
            "identity_improvements": [],
        },
        "errors": [
            {
                "type": "RuntimeError",
                "code": "runtime_error",
            }
        ],
    }


def test_cli_failure_artifact_redacts_exception_paths(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "result.json"
    private_path = "/home/private/checkpoints/missing.ckpt"

    def runner(_args):
        raise FileNotFoundError(private_path)

    return_code = main(["--output", str(output)], runner=runner)

    report_text = output.read_text(encoding="utf-8")
    report = json.loads(report_text)
    assert return_code != 0
    assert report["errors"] == [
        {"type": "FileNotFoundError", "code": "missing_file"}
    ]
    assert private_path not in report_text
    assert "/home/" not in report_text
    assert private_path in capsys.readouterr().err


def test_cli_turns_incomplete_mock_result_into_failed_artifact(
    tmp_path: Path,
) -> None:
    output = tmp_path / "result.json"
    malformed = _complete_artifact()
    malformed["horizons"] = [{"T": 2}]

    return_code = main(
        ["--output", str(output)],
        runner=lambda _args: malformed,
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert return_code != 0
    assert report["status"] == "failed"
    assert report["method"] == METHOD_NAME
    assert report["errors"]


def test_cli_writes_failed_artifact_for_malformed_horizons(
    tmp_path: Path,
) -> None:
    output = tmp_path / "result.json"

    return_code = main(
        ["--output", str(output), "--horizons", "2", "4"],
        runner=lambda _args: _complete_artifact(),
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert return_code != 0
    assert report["status"] == "failed"
    assert report["errors"][0]["type"] == "ValueError"


def test_cli_requires_q100_capacity_before_running(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    call_count = 0

    def runner(_args):
        nonlocal call_count
        call_count += 1
        return _complete_artifact()

    return_code = main(
        ["--output", str(output), "--capacity", "99"],
        runner=runner,
    )

    assert return_code != 0
    assert call_count == 0
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "failed"


@pytest.mark.parametrize(
    "reference",
    [
        "repo:",
        "repo:/home/private/model.ckpt",
        "repo:./checkpoints/rescene4d_concerto_t2_repro.ckpt",
        "repo:checkpoints/../rescene4d_concerto_t2_repro.ckpt",
        "repo:checkpoints//rescene4d_concerto_t2_repro.ckpt",
        "repo:checkpoints/",
        r"repo:checkpoints\rescene4d_concerto_t2_repro.ckpt",
        "repo:home/private/model.ckpt",
        "repo:mnt/shared/model.ckpt",
        "repo:checkpoints/other.ckpt",
        "external:stable-model-token",
        "external:/home/private/model.ckpt",
    ],
)
def test_cli_rejects_nonformal_checkpoint_references(
    tmp_path: Path,
    reference: str,
) -> None:
    artifact = _complete_artifact()
    artifact["checkpoint"]["ref"] = reference

    return_code, report = _run_mock_artifact(tmp_path, artifact)

    assert return_code != 0
    assert report["status"] == "failed"


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("t_mAP", -0.01),
        ("t_mAP", 1.01),
        ("t_mAP", float("nan")),
        ("t_REC", -0.01),
        ("t_REC", 1.01),
        ("t_REC", float("inf")),
    ],
)
@pytest.mark.parametrize("method", ["persistent", "internal_baseline"])
def test_cli_rejects_invalid_bounded_horizon_metrics(
    tmp_path: Path,
    field: str,
    invalid_value: float,
    method: str,
) -> None:
    artifact = _complete_artifact()
    artifact["horizons"][0][method][field] = invalid_value

    return_code, report = _run_mock_artifact(tmp_path, artifact)

    assert return_code != 0
    assert report["status"] == "failed"
    assert report["errors"] == [{"type": "ValueError", "code": "invalid_input"}]


@pytest.mark.parametrize("invalid_value", [-0.01, 1.01, float("nan")])
@pytest.mark.parametrize("method", ["persistent", "internal_baseline"])
def test_cli_rejects_invalid_per_stage_ap(
    tmp_path: Path,
    invalid_value: float,
    method: str,
) -> None:
    artifact = _complete_artifact()
    artifact["horizons"][0][method]["per_stage_AP"]["1"] = invalid_value

    return_code, report = _run_mock_artifact(tmp_path, artifact)

    assert return_code != 0
    assert report["status"] == "failed"


@pytest.mark.parametrize(
    ("section", "field"),
    [
        (method, field)
        for method in ("persistent", "internal_baseline")
        for field in (
            "matched_identity_observations",
            "identity_switches",
            "reactivation_events",
            "correct_reactivations",
        )
    ]
    + [
        ("persistent", "rejected_births"),
        ("resources", "peak_allocated_cuda_bytes"),
        ("resources", "serialized_state_bytes"),
    ],
)
def test_cli_rejects_negative_horizon_counts(
    tmp_path: Path,
    section: str,
    field: str,
) -> None:
    artifact = _complete_artifact()
    artifact["horizons"][0][section][field] = -1

    return_code, report = _run_mock_artifact(tmp_path, artifact)

    assert return_code != 0
    assert report["status"] == "failed"


@pytest.mark.parametrize("invalid_value", [1.5, True])
@pytest.mark.parametrize("method", ["persistent", "internal_baseline"])
def test_cli_rejects_nonintegral_horizon_counts(
    tmp_path: Path,
    invalid_value: object,
    method: str,
) -> None:
    artifact = _complete_artifact()
    artifact["horizons"][0][method]["identity_switches"] = invalid_value

    return_code, report = _run_mock_artifact(tmp_path, artifact)

    assert return_code != 0
    assert report["status"] == "failed"


@pytest.mark.parametrize(
    "path",
    [
        ("schema_version",),
        ("source_tree_contract", "schema_version"),
        ("settings", "capacity"),
        ("settings", "local_window"),
        ("legacy_parity", "sample_count"),
        ("legacy_parity", "query_feature_shape", 0),
        ("horizons", 0, "T"),
        ("horizons", 0, "loaded_sequences"),
        ("horizons", 0, "persistent", "matched_identity_observations"),
        ("horizons", 0, "persistent", "identity_switches"),
        ("horizons", 0, "persistent", "reactivation_events"),
        ("horizons", 0, "persistent", "correct_reactivations"),
        ("horizons", 0, "persistent", "rejected_births"),
        ("horizons", 0, "internal_baseline", "matched_identity_observations"),
        ("horizons", 0, "internal_baseline", "identity_switches"),
        ("horizons", 0, "internal_baseline", "reactivation_events"),
        ("horizons", 0, "internal_baseline", "correct_reactivations"),
        ("horizons", 0, "delta", "identity_switches"),
        ("horizons", 0, "resources", "peak_allocated_cuda_bytes"),
        ("horizons", 0, "resources", "serialized_state_bytes"),
        ("bounded_state", "maximum_state_bytes"),
    ],
)
def test_cli_rejects_boolean_values_for_integer_schema_fields(
    tmp_path: Path,
    path: tuple[str | int, ...],
) -> None:
    artifact = _complete_artifact()
    _set_artifact_path(artifact, path, True)

    return_code, report = _run_mock_artifact(tmp_path, artifact)

    assert return_code != 0
    assert report["status"] == "failed"


@pytest.mark.parametrize(
    ("horizon_index", "expected_count"),
    [(0, 154), (1, 120), (2, 75), (3, 43)],
)
def test_cli_requires_official_filtered_sequence_counts(
    tmp_path: Path,
    horizon_index: int,
    expected_count: int,
) -> None:
    artifact = _complete_artifact()
    artifact["horizons"][horizon_index]["loaded_sequences"] = expected_count + 1

    return_code, report = _run_mock_artifact(tmp_path, artifact)

    assert return_code != 0
    assert report["status"] == "failed"


@pytest.mark.parametrize("method", ["persistent", "internal_baseline"])
def test_cli_rejects_matched_observations_above_query_event_bound(
    tmp_path: Path,
    method: str,
) -> None:
    artifact = _complete_artifact()
    horizon = artifact["horizons"][1]
    maximum_query_events = horizon["loaded_sequences"] * horizon["T"] * 100
    horizon[method]["matched_identity_observations"] = maximum_query_events + 1
    horizon["delta"] = _metric_delta(
        horizon["persistent"],
        horizon["internal_baseline"],
    )

    return_code, report = _run_mock_artifact(tmp_path, artifact)

    assert return_code != 0
    assert report["status"] == "failed"


def test_cli_rejects_birth_rejections_above_query_event_bound(
    tmp_path: Path,
) -> None:
    artifact = _complete_artifact()
    horizon = artifact["horizons"][1]
    maximum_query_events = horizon["loaded_sequences"] * horizon["T"] * 100
    horizon["persistent"]["rejected_births"] = maximum_query_events + 1

    return_code, report = _run_mock_artifact(tmp_path, artifact)

    assert return_code != 0
    assert report["status"] == "failed"


@pytest.mark.parametrize("method", ["persistent", "internal_baseline"])
def test_cli_rejects_t2_reactivation_events(
    tmp_path: Path,
    method: str,
) -> None:
    artifact = _complete_artifact()
    horizon = artifact["horizons"][0]
    horizon[method].update(
        {
            "reactivation_events": 1,
            "correct_reactivations": 1,
            "reactivation_accuracy": 1.0,
        }
    )
    horizon["delta"] = _metric_delta(
        horizon["persistent"],
        horizon["internal_baseline"],
    )

    return_code, report = _run_mock_artifact(tmp_path, artifact)

    assert return_code != 0
    assert report["status"] == "failed"


def _set_identity_statistics(
    artifact: dict[str, object],
    *,
    horizon_index: int,
    method: str,
    observations: int,
    switches: int,
    reactivations: int = 0,
    correct_reactivations: int = 0,
) -> None:
    horizon = artifact["horizons"][horizon_index]
    horizon[method].update(
        {
            "matched_identity_observations": observations,
            "identity_switches": switches,
            "reactivation_events": reactivations,
            "correct_reactivations": correct_reactivations,
            "reactivation_accuracy": (
                correct_reactivations / reactivations
                if reactivations
                else None
            ),
        }
    )
    horizon["delta"] = _metric_delta(
        horizon["persistent"],
        horizon["internal_baseline"],
    )


def _enumerated_stage_capacity_reactivation_maxima(
    horizon: int,
    stage_capacity: int,
) -> list[int]:
    patterns: list[tuple[tuple[bool, ...], int]] = []
    for trajectory in product((False, True), repeat=horizon):
        if not any(trajectory):
            continue
        seen_observation = False
        previous_observation = False
        reactivations = 0
        for observed in trajectory:
            if observed and seen_observation and not previous_observation:
                reactivations += 1
            seen_observation = seen_observation or observed
            previous_observation = observed
        patterns.append((trajectory, reactivations))

    states = {(0,) * horizon: 0}
    for trajectory, reactivations in patterns:
        next_states: dict[tuple[int, ...], int] = {}
        for occupancy, accumulated_reactivations in states.items():
            maximum_count = min(
                stage_capacity - occupancy[stage]
                for stage, observed in enumerate(trajectory)
                if observed
            )
            for count in range(maximum_count + 1):
                next_occupancy = tuple(
                    occupancy[stage] + count * observed
                    for stage, observed in enumerate(trajectory)
                )
                next_states[next_occupancy] = max(
                    next_states.get(next_occupancy, -1),
                    accumulated_reactivations + count * reactivations,
                )
        states = next_states

    maxima = [-1] * (horizon * stage_capacity + 1)
    for occupancy, reactivations in states.items():
        observations = sum(occupancy)
        maxima[observations] = max(maxima[observations], reactivations)
    return maxima


def test_reactivation_bound_matches_stage_capacity_pattern_dp() -> None:
    for horizon in range(2, 6):
        for stage_capacity in range(1, 4):
            expected_maxima = _enumerated_stage_capacity_reactivation_maxima(
                horizon,
                stage_capacity,
            )
            for observations, expected in enumerate(expected_maxima):
                assert evaluator._maximum_aggregate_reactivations(
                    observations,
                    horizon=horizon,
                    stage_capacity=stage_capacity,
                ) == expected, (horizon, stage_capacity, observations)


def test_t5_stage_capacity_allows_interleaved_gt_reactivations() -> None:
    diagnostics = identity_diagnostics(
        gt_ids_by_stage=[[1], [2], [1], [2], [1]],
        predicted_ids_by_stage=[[10], [20], [10], [20], [10]],
    )

    assert diagnostics["matched_identity_observations"] == 5
    assert diagnostics["reactivation_events"] == 3
    assert evaluator._maximum_aggregate_reactivations(
        5,
        horizon=5,
        stage_capacity=1,
    ) == 3


@pytest.mark.parametrize("method", ["persistent", "internal_baseline"])
@pytest.mark.parametrize(
    ("horizon_index", "observations", "switches"),
    [(1, 4, 2), (3, 6, 4)],
    ids=["T3", "T5"],
)
def test_cli_accepts_identity_switches_at_aggregate_track_bound(
    tmp_path: Path,
    method: str,
    horizon_index: int,
    observations: int,
    switches: int,
) -> None:
    artifact = _complete_artifact()
    _set_identity_statistics(
        artifact,
        horizon_index=horizon_index,
        method=method,
        observations=observations,
        switches=switches,
    )

    return_code, report = _run_mock_artifact(tmp_path, artifact)

    assert return_code == 0
    assert report == artifact


@pytest.mark.parametrize("method", ["persistent", "internal_baseline"])
@pytest.mark.parametrize(
    ("horizon_index", "observations", "switches"),
    [(1, 4, 3), (3, 6, 5)],
    ids=["T3", "T5"],
)
def test_cli_rejects_identity_switches_above_aggregate_track_bound(
    tmp_path: Path,
    method: str,
    horizon_index: int,
    observations: int,
    switches: int,
) -> None:
    artifact = _complete_artifact()
    _set_identity_statistics(
        artifact,
        horizon_index=horizon_index,
        method=method,
        observations=observations,
        switches=switches,
    )

    return_code, report = _run_mock_artifact(tmp_path, artifact)

    assert return_code != 0
    assert report["status"] == "failed"


@pytest.mark.parametrize("method", ["persistent", "internal_baseline"])
def test_cli_rejects_counted_transitions_above_aggregate_track_bound(
    tmp_path: Path,
    method: str,
) -> None:
    artifact = _complete_artifact()
    _set_identity_statistics(
        artifact,
        horizon_index=1,
        method=method,
        observations=4,
        switches=2,
        reactivations=1,
        correct_reactivations=1,
    )

    return_code, report = _run_mock_artifact(tmp_path, artifact)

    assert return_code != 0
    assert report["status"] == "failed"


@pytest.mark.parametrize("method", ["persistent", "internal_baseline"])
@pytest.mark.parametrize(
    ("horizon_index", "observations", "reactivations"),
    [(1, 3, 1), (3, 3, 2)],
    ids=["T3", "T5"],
)
def test_cli_accepts_reactivations_at_aggregate_horizon_bound(
    tmp_path: Path,
    method: str,
    horizon_index: int,
    observations: int,
    reactivations: int,
) -> None:
    artifact = _complete_artifact()
    _set_identity_statistics(
        artifact,
        horizon_index=horizon_index,
        method=method,
        observations=observations,
        switches=0,
        reactivations=reactivations,
        correct_reactivations=reactivations,
    )

    return_code, report = _run_mock_artifact(tmp_path, artifact)

    assert return_code == 0
    assert report == artifact


@pytest.mark.parametrize("method", ["persistent", "internal_baseline"])
@pytest.mark.parametrize(
    ("horizon_index", "observations", "reactivations"),
    [(1, 3, 2), (3, 4, 3)],
    ids=["T3", "T5"],
)
def test_cli_rejects_reactivations_above_aggregate_horizon_bound(
    tmp_path: Path,
    method: str,
    horizon_index: int,
    observations: int,
    reactivations: int,
) -> None:
    artifact = _complete_artifact()
    _set_identity_statistics(
        artifact,
        horizon_index=horizon_index,
        method=method,
        observations=observations,
        switches=0,
        reactivations=reactivations,
        correct_reactivations=reactivations,
    )

    return_code, report = _run_mock_artifact(tmp_path, artifact)

    assert return_code != 0
    assert report["status"] == "failed"


_EXACT_REACTIVATION_BOUNDARIES = [
    pytest.param(0, 30_800, 0, id="T2_2M"),
    pytest.param(1, 23_999, 11_999, id="T3_2M_minus_1"),
    pytest.param(1, 24_000, 12_000, id="T3_2M"),
    pytest.param(1, 24_001, 12_000, id="T3_2M_plus_1"),
    pytest.param(1, 36_000, 12_000, id="T3_3M"),
    pytest.param(2, 29_999, 14_999, id="T4_4M_minus_1"),
    pytest.param(2, 30_000, 15_000, id="T4_4M"),
    pytest.param(3, 12_900, 8_600, id="T5_3M"),
    pytest.param(3, 12_901, 8_600, id="T5_3M_plus_1"),
    pytest.param(3, 12_902, 8_601, id="T5_3M_plus_2"),
    pytest.param(
        3,
        17_200,
        10_750,
        id="T5_official_4M",
    ),
    pytest.param(3, 21_500, 12_900, id="T5_5M"),
]


@pytest.mark.parametrize("method", ["persistent", "internal_baseline"])
@pytest.mark.parametrize(
    ("horizon_index", "observations", "maximum_reactivations"),
    _EXACT_REACTIVATION_BOUNDARIES,
)
def test_cli_accepts_exact_aggregate_reactivation_bound(
    tmp_path: Path,
    method: str,
    horizon_index: int,
    observations: int,
    maximum_reactivations: int,
) -> None:
    artifact = _complete_artifact()
    _set_identity_statistics(
        artifact,
        horizon_index=horizon_index,
        method=method,
        observations=observations,
        switches=0,
        reactivations=maximum_reactivations,
        correct_reactivations=maximum_reactivations,
    )

    return_code, report = _run_mock_artifact(tmp_path, artifact)

    assert return_code == 0
    assert report == artifact


@pytest.mark.parametrize("method", ["persistent", "internal_baseline"])
@pytest.mark.parametrize(
    ("horizon_index", "observations", "maximum_reactivations"),
    _EXACT_REACTIVATION_BOUNDARIES,
)
def test_cli_rejects_reactivations_above_exact_aggregate_bound(
    tmp_path: Path,
    method: str,
    horizon_index: int,
    observations: int,
    maximum_reactivations: int,
) -> None:
    artifact = _complete_artifact()
    _set_identity_statistics(
        artifact,
        horizon_index=horizon_index,
        method=method,
        observations=observations,
        switches=0,
        reactivations=maximum_reactivations + 1,
        correct_reactivations=maximum_reactivations + 1,
    )

    return_code, report = _run_mock_artifact(tmp_path, artifact)

    assert return_code != 0
    assert report["status"] == "failed"


@pytest.mark.parametrize(
    (
        "observations",
        "switches",
        "reactivations",
        "correct_reactivations",
        "accuracy",
    ),
    [
        (1, 2, 0, 0, None),
        (1, 1, 0, 0, None),
        (1, 0, 1, 0, 0.0),
        (3, 0, 1, 0, 0.0),
        (3, 0, 1, 2, 1.0),
        (3, 0, 0, 0, 0.0),
        (3, 0, 2, 1, None),
        (3, 0, 2, 1, 0.4),
    ],
)
@pytest.mark.parametrize("method", ["persistent", "internal_baseline"])
def test_cli_rejects_impossible_identity_statistics(
    tmp_path: Path,
    observations: int,
    switches: int,
    reactivations: int,
    correct_reactivations: int,
    accuracy: float | None,
    method: str,
) -> None:
    artifact = _complete_artifact()
    metrics = artifact["horizons"][0][method]
    metrics.update(
        {
            "matched_identity_observations": observations,
            "identity_switches": switches,
            "reactivation_events": reactivations,
            "correct_reactivations": correct_reactivations,
            "reactivation_accuracy": accuracy,
        }
    )

    return_code, report = _run_mock_artifact(tmp_path, artifact)

    assert return_code != 0
    assert report["status"] == "failed"


@pytest.mark.parametrize("invalid_accuracy", [-0.01, 1.01, float("nan")])
@pytest.mark.parametrize("method", ["persistent", "internal_baseline"])
def test_cli_rejects_invalid_reactivation_accuracy(
    tmp_path: Path,
    invalid_accuracy: float,
    method: str,
) -> None:
    artifact = _complete_artifact()
    artifact["horizons"][3][method].update(
        {
            "matched_identity_observations": 3,
            "identity_switches": 1,
            "reactivation_events": 2,
            "correct_reactivations": 1,
            "reactivation_accuracy": invalid_accuracy,
        }
    )

    return_code, report = _run_mock_artifact(tmp_path, artifact)

    assert return_code != 0
    assert report["status"] == "failed"


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("loaded_sequences", 0),
        ("mean_latency_ms", 0.0),
        ("mean_latency_ms", float("nan")),
        ("throughput_sequences_per_second", 0.0),
        ("throughput_sequences_per_second", float("inf")),
        ("peak_allocated_cuda_bytes", 0),
        ("serialized_state_bytes", 0),
    ],
)
def test_cli_rejects_invalid_horizon_runtime_measurements(
    tmp_path: Path,
    field: str,
    invalid_value: float,
) -> None:
    artifact = _complete_artifact()
    if field == "loaded_sequences":
        artifact["horizons"][0][field] = invalid_value
    else:
        artifact["horizons"][0]["resources"][field] = invalid_value

    return_code, report = _run_mock_artifact(tmp_path, artifact)

    assert return_code != 0
    assert report["status"] == "failed"


def test_cli_rejects_inconsistent_latency_and_throughput(
    tmp_path: Path,
) -> None:
    artifact = _complete_artifact()
    artifact["horizons"][0]["resources"].update(
        {
            "mean_latency_ms": 2.0,
            "throughput_sequences_per_second": 499.9,
        }
    )

    return_code, report = _run_mock_artifact(tmp_path, artifact)

    assert return_code != 0
    assert report["status"] == "failed"


def test_cli_accepts_latency_and_throughput_within_rounding_tolerance(
    tmp_path: Path,
) -> None:
    artifact = _complete_artifact()
    artifact["horizons"][0]["resources"].update(
        {
            "mean_latency_ms": 2.0,
            "throughput_sequences_per_second": 500.04,
        }
    )

    return_code, report = _run_mock_artifact(tmp_path, artifact)

    assert return_code == 0
    assert report == artifact


def test_cli_accepts_consistent_nonzero_reactivation_statistics(
    tmp_path: Path,
) -> None:
    artifact = _complete_artifact()
    horizon = artifact["horizons"][3]
    horizon["persistent"].update(
        {
            "matched_identity_observations": 3,
            "identity_switches": 1,
            "reactivation_events": 2,
            "correct_reactivations": 1,
            "reactivation_accuracy": 0.5,
        }
    )
    horizon["delta"] = _metric_delta(
        horizon["persistent"],
        horizon["internal_baseline"],
    )

    return_code, report = _run_mock_artifact(tmp_path, artifact)

    assert return_code == 0
    assert report == artifact


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("tracked_tree_clean", False),
        ("index_clean", False),
        ("only_declared_outputs_untracked", False),
        ("generation_head_unchanged", False),
        ("allowed_untracked_outputs", ["repo:artifacts/other.json"]),
        ("source_commit", "c" * 40),
    ],
)
def test_cli_rejects_invalid_source_tree_contract(
    tmp_path: Path,
    field: str,
    invalid_value: object,
) -> None:
    artifact = _complete_artifact()
    artifact["source_tree_contract"][field] = invalid_value

    return_code, report = _run_mock_artifact(tmp_path, artifact)

    assert return_code != 0
    assert report["status"] == "failed"


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("verified_by", "external_test_claim"),
        ("sample_count", 0),
        ("legacy_predictions_unchanged", False),
        ("query_feature_shape", [1, 99, 128]),
    ],
)
def test_cli_rejects_invalid_legacy_parity(
    tmp_path: Path,
    field: str,
    invalid_value: object,
) -> None:
    artifact = _complete_artifact()
    artifact["legacy_parity"][field] = invalid_value

    return_code, report = _run_mock_artifact(tmp_path, artifact)

    assert return_code != 0
    assert report["status"] == "failed"


def test_cli_rejects_baseline_that_does_not_share_rescene_outputs(
    tmp_path: Path,
) -> None:
    artifact = _complete_artifact()
    artifact["settings"]["shared_rescene_outputs"] = False

    return_code, report = _run_mock_artifact(tmp_path, artifact)

    assert return_code != 0
    assert report["status"] == "failed"


@pytest.mark.parametrize(
    "field",
    [
        "t_mAP",
        "t_REC",
        "per_stage_AP",
        "matched_identity_observations",
        "identity_switches",
        "reactivation_events",
        "correct_reactivations",
        "reactivation_accuracy",
    ],
)
def test_cli_rejects_delta_not_derived_from_both_methods(
    tmp_path: Path,
    field: str,
) -> None:
    artifact = _complete_artifact()
    delta = artifact["horizons"][0]["delta"]
    if field == "per_stage_AP":
        delta[field]["1"] += 0.01
    elif field == "reactivation_accuracy":
        delta[field] = 0.0
    else:
        delta[field] += 1

    return_code, report = _run_mock_artifact(tmp_path, artifact)

    assert return_code != 0
    assert report["status"] == "failed"


def test_cli_rejects_handwritten_conclusion_after_metric_change(
    tmp_path: Path,
) -> None:
    artifact = _complete_artifact()
    horizon = artifact["horizons"][2]
    horizon["persistent"]["t_REC"] = 0.6
    horizon["delta"] = _metric_delta(
        horizon["persistent"],
        horizon["internal_baseline"],
    )

    return_code, report = _run_mock_artifact(tmp_path, artifact)

    assert return_code != 0
    assert report["status"] == "failed"


def test_cli_accepts_automatically_derived_t4_improvement(
    tmp_path: Path,
) -> None:
    artifact = _complete_artifact()
    horizon = artifact["horizons"][2]
    horizon["persistent"]["t_REC"] = 0.6
    horizon["delta"] = _metric_delta(
        horizon["persistent"],
        horizon["internal_baseline"],
    )
    artifact["conclusion"] = _derive_conclusion(
        artifact["horizons"],
        artifact["bounded_state"],
    )

    return_code, report = _run_mock_artifact(tmp_path, artifact)

    assert return_code == 0
    assert report["conclusion"] == {
        "label": "P5_MVP_PASS",
        "reason": "bounded_execution_with_t4_t5_identity_improvement",
        "identity_improvements": ["T4:t_REC"],
    }
