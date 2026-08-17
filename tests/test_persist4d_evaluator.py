from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import fields
from pathlib import Path

import pytest
import torch

from scripts.evaluate_persist4d import (
    METHOD_NAME,
    SequenceAccumulator,
    identity_diagnostics,
    main,
)


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


def _horizon_result(horizon: int) -> dict[str, object]:
    return {
        "T": horizon,
        "loaded_sequences": 1,
        "t_mAP": 0.25,
        "t_REC": 0.5,
        "per_stage_AP": {
            str(stage): 0.25 for stage in range(1, horizon + 1)
        },
        "matched_identity_observations": horizon,
        "identity_switches": 0,
        "reactivation_events": 0,
        "correct_reactivations": 0,
        "reactivation_accuracy": None,
        "rejected_births": 0,
        "peak_allocated_cuda_bytes": 1024,
        "mean_latency_ms": 2.0,
        "throughput_sequences_per_second": 500.0,
        "serialized_state_bytes": 256,
    }


def _complete_artifact() -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "pass",
        "method": METHOD_NAME,
        "source_commit": "a" * 40,
        "checkpoint": {
            "ref": "repo:checkpoints/rescene4d_concerto_t2_repro.ckpt",
            "sha256": "b" * 64,
        },
        "settings": {"capacity": 100, "local_window": 2},
        "horizons": [_horizon_result(value) for value in (2, 3, 4, 5)],
        "bounded_state": {
            "constant_shape": True,
            "maximum_state_bytes": 256,
        },
        "errors": [],
    }


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
    seen = []

    def runner(args):
        seen.append(args)
        assert not output.exists()
        return _complete_artifact()

    return_code = main(["--output", str(output)], runner=runner)

    assert return_code == 0
    assert len(seen) == 1
    assert json.loads(output.read_text(encoding="utf-8")) == _complete_artifact()
    assert list(output.parent.iterdir()) == [output]


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


def test_cli_writes_failed_artifact_when_runner_raises(tmp_path: Path) -> None:
    output = tmp_path / "failure" / "result.json"

    def runner(_args):
        raise RuntimeError("synthetic evaluator failure")

    return_code = main(["--output", str(output)], runner=runner)

    assert return_code != 0
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "schema_version": 1,
        "status": "failed",
        "method": METHOD_NAME,
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
def test_cli_rejects_invalid_bounded_horizon_metrics(
    tmp_path: Path,
    field: str,
    invalid_value: float,
) -> None:
    artifact = _complete_artifact()
    artifact["horizons"][0][field] = invalid_value

    return_code, report = _run_mock_artifact(tmp_path, artifact)

    assert return_code != 0
    assert report["status"] == "failed"
    assert report["errors"] == [{"type": "ValueError", "code": "invalid_input"}]


@pytest.mark.parametrize("invalid_value", [-0.01, 1.01, float("nan")])
def test_cli_rejects_invalid_per_stage_ap(
    tmp_path: Path,
    invalid_value: float,
) -> None:
    artifact = _complete_artifact()
    artifact["horizons"][0]["per_stage_AP"]["1"] = invalid_value

    return_code, report = _run_mock_artifact(tmp_path, artifact)

    assert return_code != 0
    assert report["status"] == "failed"


@pytest.mark.parametrize(
    "field",
    [
        "matched_identity_observations",
        "identity_switches",
        "reactivation_events",
        "correct_reactivations",
        "rejected_births",
        "peak_allocated_cuda_bytes",
        "serialized_state_bytes",
    ],
)
def test_cli_rejects_negative_horizon_counts(
    tmp_path: Path,
    field: str,
) -> None:
    artifact = _complete_artifact()
    artifact["horizons"][0][field] = -1

    return_code, report = _run_mock_artifact(tmp_path, artifact)

    assert return_code != 0
    assert report["status"] == "failed"


@pytest.mark.parametrize("invalid_value", [1.5, True])
def test_cli_rejects_nonintegral_horizon_counts(
    tmp_path: Path,
    invalid_value: object,
) -> None:
    artifact = _complete_artifact()
    artifact["horizons"][0]["identity_switches"] = invalid_value

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
def test_cli_rejects_impossible_identity_statistics(
    tmp_path: Path,
    observations: int,
    switches: int,
    reactivations: int,
    correct_reactivations: int,
    accuracy: float | None,
) -> None:
    artifact = _complete_artifact()
    horizon = artifact["horizons"][0]
    horizon.update(
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
def test_cli_rejects_invalid_reactivation_accuracy(
    tmp_path: Path,
    invalid_accuracy: float,
) -> None:
    artifact = _complete_artifact()
    artifact["horizons"][0].update(
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
        ("serialized_state_bytes", 0),
    ],
)
def test_cli_rejects_invalid_horizon_runtime_measurements(
    tmp_path: Path,
    field: str,
    invalid_value: float,
) -> None:
    artifact = _complete_artifact()
    artifact["horizons"][0][field] = invalid_value

    return_code, report = _run_mock_artifact(tmp_path, artifact)

    assert return_code != 0
    assert report["status"] == "failed"


def test_cli_rejects_inconsistent_latency_and_throughput(
    tmp_path: Path,
) -> None:
    artifact = _complete_artifact()
    artifact["horizons"][0].update(
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
    artifact["horizons"][0].update(
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
    artifact["horizons"][0].update(
        {
            "matched_identity_observations": 3,
            "identity_switches": 1,
            "reactivation_events": 2,
            "correct_reactivations": 1,
            "reactivation_accuracy": 0.5,
        }
    )

    return_code, report = _run_mock_artifact(tmp_path, artifact)

    assert return_code == 0
    assert report == artifact
