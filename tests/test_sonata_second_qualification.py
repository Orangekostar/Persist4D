from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch


def test_evaluation_contract_is_matched_and_preregistered() -> None:
    from scripts import evaluate_sonata_second_checkpoint as qualification

    assert qualification.EVALUATION_SEEDS == (45, 46, 47)
    assert qualification.METRIC_KEYS == {
        "t_mAP": "val_mean_t-AP",
        "t_mAP50": "val_mean_t-AP_50",
        "t_mAP25": "val_mean_t-AP_25",
        "overall_mAP": "val_mean_AP",
        "stage1_mAP": "val_mean_stage1-AP",
        "stage2_mAP": "val_mean_stage2-AP",
    }
    assert qualification.RUNTIME_CONTRACT == {
        "accelerator": "gpu",
        "devices": 1,
        "batch_size": 1,
        "num_workers": 4,
        "precision": "32-true",
    }


def test_composed_configs_share_runtime_but_preserve_model_identity() -> None:
    from scripts import evaluate_sonata_second_checkpoint as qualification

    sonata = qualification.compose_evaluation_config("sonata")
    concerto = qualification.compose_evaluation_config("concerto")

    for config in (sonata, concerto):
        assert config.general.gpus == 1
        assert config.general.train_mode is False
        assert config.data.batch_size == 1
        assert config.data.test_batch_size == 1
        assert config.data.num_workers == 4
        assert config.trainer.precision == "32-true"
    assert sonata.backbone.model_lib == "sonata"
    assert concerto.backbone.model_lib == "concerto"
    assert sonata.model.temporal_masking is True
    assert concerto.model.temporal_masking is False


def test_strict_load_rejects_incomplete_state_dict(tmp_path: Path) -> None:
    from scripts import evaluate_sonata_second_checkpoint as qualification

    system = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
    state = system.state_dict()
    state.pop(next(iter(state)))
    path = tmp_path / "incomplete.ckpt"
    torch.save({"state_dict": state}, path)

    with pytest.raises(RuntimeError, match="strict task checkpoint load failed"):
        qualification.strict_load_task_checkpoint(system, path)


def test_strict_load_accepts_only_complete_lightning_mapping(tmp_path: Path) -> None:
    from scripts import evaluate_sonata_second_checkpoint as qualification

    system = torch.nn.Linear(2, 1)
    path = tmp_path / "complete.ckpt"
    torch.save({"state_dict": system.state_dict(), "epoch": 4}, path)

    assert qualification.strict_load_task_checkpoint(system, path) == {
        "state_dict_entry_count": 2,
        "strict": True,
    }


def test_normalize_metrics_requires_all_finite_unit_interval_values() -> None:
    from scripts import evaluate_sonata_second_checkpoint as qualification

    source = {
        source_name: torch.tensor(0.25)
        for source_name in qualification.METRIC_KEYS.values()
    }
    assert qualification.normalize_metrics(source) == {
        output_name: pytest.approx(0.25)
        for output_name in qualification.METRIC_KEYS
    }

    source.pop("val_mean_AP")
    with pytest.raises(ValueError, match="missing evaluation metric"):
        qualification.normalize_metrics(source)


@pytest.mark.parametrize("value", [float("nan"), -0.01, 1.01])
def test_normalize_metrics_rejects_invalid_values(value: float) -> None:
    from scripts import evaluate_sonata_second_checkpoint as qualification

    source = {source_name: 0.25 for source_name in qualification.METRIC_KEYS.values()}
    source["val_mean_t-AP"] = value

    with pytest.raises(ValueError, match="invalid evaluation metric"):
        qualification.normalize_metrics(source)


def _training_manifest() -> dict[str, object]:
    return {
        "status": "pass",
        "budget": {"completed_epochs": 450, "optimizer_steps": 29700},
        "bindings": {
            "source_commit": "1" * 40,
            "config_sha256": "2" * 64,
            "weight_sha256": "3" * 64,
        },
        "checkpoint_selection": {
            "monitor": "val_mean_t-AP",
            "mode": "max",
            "selected_epoch": 449,
            "selection_metric_exact": 0.243,
        },
        "checkpoints": [
            {
                "role": "best_validation",
                "epoch": 449,
                "global_step": 29700,
                "sha256": "4" * 64,
                "byte_size": 100,
                "state_dict_entry_count": 798,
            }
        ],
    }


def test_checkpoint_manifest_requires_preregistered_top1_and_full_budget() -> None:
    from scripts import evaluate_sonata_second_checkpoint as qualification

    manifest = qualification.build_checkpoint_manifest(
        training_manifest=_training_manifest(),
        sonata_file={"sha256": "4" * 64, "byte_size": 100, "mode": "0444"},
        concerto_file={"sha256": "5" * 64, "byte_size": 200, "mode": "0664"},
        evidence_source_commit="6" * 40,
    )

    assert manifest["sonata"]["epoch"] == 449
    assert manifest["sonata"]["global_step"] == 29700
    assert manifest["sonata"]["selection"] == "highest val_mean_t-AP"
    assert manifest["sonata"]["mode"] == "0444"
    assert manifest["training"]["completed_epochs"] == 450
    assert manifest["selection_used_protocol_b"] is False


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (("budget", "completed_epochs", 449), "450 completed epochs"),
        (("checkpoint_selection", "monitor", "val_mean_AP"), "selection contract"),
        (("checkpoint_selection", "mode", "min"), "selection contract"),
    ],
)
def test_checkpoint_manifest_rejects_invalid_training_contract(
    mutation: tuple[str, str, object], message: str
) -> None:
    from scripts import evaluate_sonata_second_checkpoint as qualification

    training = deepcopy(_training_manifest())
    parent, key, value = mutation
    training[parent][key] = value

    with pytest.raises(ValueError, match=message):
        qualification.build_checkpoint_manifest(
            training_manifest=training,
            sonata_file={"sha256": "4" * 64, "byte_size": 100, "mode": "0444"},
            concerto_file={"sha256": "5" * 64, "byte_size": 200, "mode": "0664"},
            evidence_source_commit="6" * 40,
        )


def test_qualification_gate_green_requires_threshold_and_spatial_parity() -> None:
    from scripts import evaluate_sonata_second_checkpoint as qualification

    result = qualification.qualification_gate(
        sonata={"t_mAP": 0.297, "overall_mAP": 0.40},
        concerto={"t_mAP": 0.25, "overall_mAP": 0.40},
        provenance_complete=True,
        completed_epochs=450,
    )

    assert result["label"] == "SQ-GREEN"
    assert result["authorizes_ss6"] is True


def test_qualification_gate_yellow_below_external_threshold() -> None:
    from scripts import evaluate_sonata_second_checkpoint as qualification

    result = qualification.qualification_gate(
        sonata={"t_mAP": 0.296, "overall_mAP": 0.40},
        concerto={"t_mAP": 0.25, "overall_mAP": 0.40},
        provenance_complete=True,
        completed_epochs=450,
    )

    assert result["label"] == "SQ-YELLOW"
    assert result["authorizes_ss6"] is False


def test_qualification_gate_yellow_on_spatial_regression() -> None:
    from scripts import evaluate_sonata_second_checkpoint as qualification

    result = qualification.qualification_gate(
        sonata={"t_mAP": 0.30, "overall_mAP": 0.39},
        concerto={"t_mAP": 0.25, "overall_mAP": 0.40},
        provenance_complete=True,
        completed_epochs=450,
    )

    assert result["label"] == "SQ-YELLOW"


def test_qualification_gate_red_on_dual_collapse_or_invalid_provenance() -> None:
    from scripts import evaluate_sonata_second_checkpoint as qualification

    dual_collapse = qualification.qualification_gate(
        sonata={"t_mAP": 0.20, "overall_mAP": 0.30},
        concerto={"t_mAP": 0.25, "overall_mAP": 0.40},
        provenance_complete=True,
        completed_epochs=450,
    )
    invalid = qualification.qualification_gate(
        sonata={"t_mAP": 0.30, "overall_mAP": 0.40},
        concerto={"t_mAP": 0.25, "overall_mAP": 0.40},
        provenance_complete=False,
        completed_epochs=450,
    )

    assert dual_collapse["label"] == "SQ-RED"
    assert invalid["label"] == "SQ-RED"


def _evaluation_run(model: str, seed: int) -> dict[str, object]:
    from scripts import evaluate_sonata_second_checkpoint as qualification

    spec = qualification.MODEL_SPECS[model]
    value = 0.30 if model == "sonata" else 0.25
    return {
        "status": "pass",
        "scope": "official_like_t2",
        "model": model,
        "seed": seed,
        "source_commit": "7" * 40,
        "evaluation_contract_sha256": qualification.evaluation_contract()["sha256"],
        "config_name": spec.config_name,
        "portable_config_sha256": ("8" if model == "sonata" else "9") * 64,
        "data_manifest_sha256": qualification.DATA_MANIFEST_SHA256,
        "validation_sequence_count": 154,
        "runtime": {
            **qualification.RUNTIME_CONTRACT,
            "device_index": 0,
            "gpu_name": "NVIDIA A40",
            "seed_workers": True,
        },
        "task_checkpoint": {"sha256": spec.task_checkpoint_sha256},
        "strict_load": {"state_dict_entry_count": 798, "strict": True},
        "metrics": {key: value for key in qualification.METRIC_KEYS},
    }


def test_validate_run_matrix_requires_exact_matched_cross_product() -> None:
    from scripts import evaluate_sonata_second_checkpoint as qualification

    runs = [
        _evaluation_run(model, seed)
        for model in qualification.MODEL_SPECS
        for seed in qualification.EVALUATION_SEEDS
    ]

    validated = qualification.validate_run_matrix(runs)

    assert set(validated) == {"sonata", "concerto"}
    assert [run["seed"] for run in validated["sonata"]] == [45, 46, 47]


def test_validate_run_matrix_rejects_missing_or_runtime_drift() -> None:
    from scripts import evaluate_sonata_second_checkpoint as qualification

    runs = [
        _evaluation_run(model, seed)
        for model in qualification.MODEL_SPECS
        for seed in qualification.EVALUATION_SEEDS
    ]
    with pytest.raises(ValueError, match="exact model/seed cross-product"):
        qualification.validate_run_matrix(runs[:-1])

    drifted = deepcopy(runs)
    drifted[-1]["runtime"]["num_workers"] = 0
    with pytest.raises(ValueError, match="runtime contract"):
        qualification.validate_run_matrix(drifted)


def test_summarize_runs_reports_matched_three_seed_statistics() -> None:
    from scripts import evaluate_sonata_second_checkpoint as qualification

    runs = [_evaluation_run("sonata", seed) for seed in qualification.EVALUATION_SEEDS]
    for index, run in enumerate(runs):
        run["metrics"] = {key: 0.2 + index * 0.1 for key in qualification.METRIC_KEYS}

    summary = qualification.summarize_runs(runs)

    assert summary["seed_count"] == 3
    assert summary["t_mAP_mean"] == pytest.approx(0.3)
    assert summary["t_mAP_std"] == pytest.approx(0.1)
    assert summary["t_mAP_min"] == pytest.approx(0.2)
    assert summary["t_mAP_max"] == pytest.approx(0.4)
