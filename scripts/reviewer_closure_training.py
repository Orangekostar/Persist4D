"""Frozen audit and runtime contracts for T2-to-T3 ReScene adaptation."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

import torch
import yaml
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "conf"
RECIPE_PATH = PROJECT_ROOT / "configs/reviewer_closure/rescene_t3_adapted.yaml"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints/rescene4d_concerto_t2_repro.ckpt"
OUTPUT_ROOT = PROJECT_ROOT / "artifacts/reviewer_closure"

ALLOWED_ADAPTATION_CONFIG_PATHS = (
    "callbacks.0.dirpath",
    "callbacks.1.dirpath",
    "callbacks.2.every_n_epochs",
    "callbacks.2.filename",
    "data.batch_size",
    "data.temporal_window",
    "data.test_dataset.temporal_window",
    "data.train_dataloader.batch_size",
    "data.train_dataset.datasets.0.temporal_window",
    "data.validation_dataset.temporal_window",
    "general.checkpoint",
    "general.experiment_name",
    "general.p2_fail_closed_runtime",
    "general.p2_weighted_objective",
    "general.project_name",
    "general.reviewer_closure_fail_closed_runtime",
    "general.reviewer_closure_weighted_objective",
    "general.save_dir",
    "logging.0.save_dir",
    "model.config.temporal_window",
    "optimizer.lr",
    "p2_preflight.artifact_path",
    "p2_preflight.target",
    "scheduler.scheduler.epochs",
    "scheduler.scheduler.max_lr",
    "trainer.accumulate_grad_batches",
    "trainer.max_epochs",
)


class ReviewerClosureTrainingError(ValueError):
    """Raised when the frozen adaptation recipe or source evidence differs."""


def sha256_file(path: str | Path) -> str:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ReviewerClosureTrainingError(f"required source is unavailable: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(value: Mapping[str, object]) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _content_sha256(value: Mapping[str, object]) -> str:
    payload = dict(value)
    payload.pop("content_sha256", None)
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _mapping(value: object, *, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ReviewerClosureTrainingError(f"{name} must be a mapping")
    return dict(value)


def _exact_keys(value: Mapping[str, object], fields: set[str], *, name: str) -> None:
    if set(value) != fields:
        raise ReviewerClosureTrainingError(f"{name} fields differ")


def load_t3_adaptation_recipe(path: str | Path = RECIPE_PATH) -> dict[str, object]:
    source_path = Path(path)
    if source_path.is_symlink() or not source_path.is_file():
        raise ReviewerClosureTrainingError(
            "T3 adaptation recipe must be a regular file"
        )
    try:
        value = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ReviewerClosureTrainingError(
            "cannot decode T3 adaptation recipe"
        ) from error
    recipe = _mapping(value, name="T3 adaptation recipe")
    _exact_keys(
        recipe,
        {
            "schema_version",
            "stage_name",
            "comparison_level",
            "paper_name",
            "source",
            "data",
            "model",
            "optimization",
            "evidence",
        },
        name="T3 adaptation recipe",
    )
    if (
        recipe["schema_version"] != 1
        or recipe["stage_name"] != "persist4d-reviewer-closure-t3-horizon-adaptation"
        or recipe["comparison_level"] != 2
        or recipe["paper_name"] != "ReScene4D T2-to-T3 Horizon-Adapted"
    ):
        raise ReviewerClosureTrainingError("T3 adaptation identity differs")

    source = _mapping(recipe["source"], name="source")
    if source != {
        "checkpoint_reference": ("repo:checkpoints/rescene4d_concerto_t2_repro.ckpt"),
        "checkpoint_sha256": (
            "85ed1aba60320cd19798536b71b91dbc156b7ea60f838832bc0bbbdba131546e"
        ),
        "checkpoint_epoch": 404,
        "checkpoint_global_step": 26730,
        "initialization": "weights_only_strict",
        "p2_config_reference": "repo:conf/config_p2_rescene4d_concerto_t2.yaml",
        "p2_config_sha256": (
            "3077e89b9bca60049e1551654d5b3df78178d5647d84f0037bcb7ab91807c443"
        ),
    }:
        raise ReviewerClosureTrainingError("T3 source contract differs")

    data = _mapping(recipe["data"], name="data")
    if data != {
        "split_policy": "same_as_p2_reproduction",
        "rio_temporal_window": 3,
        "rio_sequence_database_reference": (
            "repo:data/processed/rio/sequence_database_sliding_3.yaml"
        ),
        "rio_sequence_database_sha256": (
            "20184ca27316bd668c084c39af72144d018e556b9562826b511a1413f4986893"
        ),
        "rio_raw_sequence_counts": {
            "train": 858,
            "validation": 123,
            "test": 113,
        },
        "scannet_temporal_window": 1,
        "mix_weights": [1.0, 0.8],
        "epoch_sample_multiple": 32,
        "mixed_epoch_samples": 1536,
    }:
        raise ReviewerClosureTrainingError("T3 data contract differs")

    model = _mapping(recipe["model"], name="model")
    if model != {
        "backbone": "Concerto",
        "freeze": "backbone_encoder",
        "num_queries": 100,
        "query_initialization": "fps_non_parametric",
        "temporal_window": 3,
        "label_space": "nyu40_18_instance_classes",
        "loss_definition": "p2_weighted_set_criterion_with_contrastive",
    }:
        raise ReviewerClosureTrainingError("T3 model contract differs")

    optimization = _mapping(recipe["optimization"], name="optimization")
    if optimization != {
        "optimizer": "AdamW",
        "max_lr": 5.0e-5,
        "betas": [0.9, 0.999],
        "eps": 1.0e-8,
        "weight_decay": 0.01,
        "amsgrad": False,
        "scheduler": "OneCycleLR",
        "scheduler_reset": True,
        "epochs": 45,
        "devices": 2,
        "per_device_batch_size": 1,
        "gradient_accumulation": 16,
        "effective_batch_size": 32,
        "train_batches_per_rank": 768,
        "optimizer_updates_per_epoch": 48,
        "total_optimizer_updates": 2160,
        "precision": "32-true",
        "seed": 45,
        "validation_every_n_epochs": 15,
    }:
        raise ReviewerClosureTrainingError("T3 optimization contract differs")

    evidence = _mapping(recipe["evidence"], name="evidence")
    if evidence != {
        "actual_scan_exposures_required": True,
        "actual_optimizer_updates_required": True,
        "wall_clock_and_gpu_hours_required": True,
        "nonfinite_loss_policy": "fail_closed",
        "checkpoint_reload_required": True,
        "recipe_changes_after_smoke": "forbidden",
    }:
        raise ReviewerClosureTrainingError("T3 evidence contract differs")
    return copy.deepcopy(recipe)


def _compose_p2_config() -> DictConfig:
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.2"):
        return compose(config_name="config_p2_rescene4d_concerto_t2")


def compose_t3_adaptation_config(
    recipe: Mapping[str, object],
) -> tuple[DictConfig, DictConfig]:
    validated = load_t3_adaptation_recipe(RECIPE_PATH)
    if dict(recipe) != validated:
        raise ReviewerClosureTrainingError("in-memory T3 recipe differs")
    t2 = _compose_p2_config()
    adapted = OmegaConf.create(
        OmegaConf.to_container(t2, resolve=False, throw_on_missing=True)
    )
    OmegaConf.set_struct(adapted, False)
    del adapted["p2_preflight"]
    del adapted.general["p2_weighted_objective"]
    del adapted.general["p2_fail_closed_runtime"]
    adapted.general.reviewer_closure_weighted_objective = True
    adapted.general.reviewer_closure_fail_closed_runtime = True
    adapted.general.project_name = "rescene4d_t2_to_t3_horizon_adapted"
    adapted.general.experiment_name = "rescene4d_t2_to_t3_horizon_adapted"
    adapted.general.save_dir = "checkpoints/reviewer_closure_rescene_t3_adapted"
    adapted.general.checkpoint = str(CHECKPOINT_PATH)
    adapted.model.config.temporal_window = 3
    adapted.data.train_dataset.datasets[0].temporal_window = 3
    adapted.data.validation_dataset.temporal_window = 3
    adapted.data.test_dataset.temporal_window = 3
    adapted.data.batch_size = 1
    adapted.optimizer.lr = 5.0e-5
    adapted.trainer.max_epochs = 45
    adapted.trainer.accumulate_grad_batches = 16
    adapted.callbacks[2].filename = "rescene4d_t2_to_t3_horizon_adapted"
    adapted.callbacks[2].every_n_epochs = 45
    return t2, adapted


def _flatten(value: object, *, prefix: str = "") -> dict[str, object]:
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            result.update(_flatten(item, prefix=path))
        return result
    if isinstance(value, list):
        result = {}
        for index, item in enumerate(value):
            path = f"{prefix}.{index}" if prefix else str(index)
            result.update(_flatten(item, prefix=path))
        return result
    return {prefix: value}


def adaptation_config_differences(
    t2: DictConfig, adapted: DictConfig
) -> dict[str, dict[str, object]]:
    before = _flatten(OmegaConf.to_container(t2, resolve=True, throw_on_missing=True))
    after = _flatten(
        OmegaConf.to_container(adapted, resolve=True, throw_on_missing=True)
    )
    differences = {}
    for path in sorted(set(before) | set(after)):
        if before.get(path) != after.get(path):
            differences[path] = {"t2": before.get(path), "adapted": after.get(path)}
    return differences


def inspect_t3_source_checkpoint(
    path: str | Path,
    *,
    expected_sha256: str,
) -> dict[str, object]:
    source = Path(path)
    digest = sha256_file(source)
    if digest != expected_sha256:
        raise ReviewerClosureTrainingError("T3 source checkpoint SHA256 differs")
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ReviewerClosureTrainingError("T3 source checkpoint must be a mapping")
    state_dict = checkpoint.get("state_dict")
    optimizers = checkpoint.get("optimizer_states")
    schedulers = checkpoint.get("lr_schedulers")
    hparams = checkpoint.get("hyper_parameters")
    if (
        not isinstance(state_dict, Mapping)
        or not isinstance(optimizers, list)
        or not isinstance(schedulers, list)
        or not isinstance(hparams, Mapping)
    ):
        raise ReviewerClosureTrainingError("T3 source checkpoint state differs")
    try:
        source_window = int(hparams["model"]["config"]["temporal_window"])
        scheduler_last_epoch = int(schedulers[0]["last_epoch"])
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise ReviewerClosureTrainingError(
            "T3 source checkpoint metadata differs"
        ) from error
    result = {
        "reference": "repo:checkpoints/rescene4d_concerto_t2_repro.ckpt",
        "sha256": digest,
        "byte_size": source.stat().st_size,
        "epoch": checkpoint.get("epoch"),
        "global_step": checkpoint.get("global_step"),
        "pytorch_lightning_version": checkpoint.get("pytorch-lightning_version"),
        "optimizer_state_count": len(optimizers),
        "scheduler_state_count": len(schedulers),
        "scheduler_last_epoch": scheduler_last_epoch,
        "source_temporal_window": source_window,
        "state_dict_entry_count": len(state_dict),
    }
    if (
        result["epoch"] != 404
        or result["global_step"] != 26730
        or result["optimizer_state_count"] != 1
        or result["scheduler_state_count"] != 1
        or result["scheduler_last_epoch"] != 26730
        or result["source_temporal_window"] != 2
        or result["state_dict_entry_count"] != 798
    ):
        raise ReviewerClosureTrainingError("T3 source checkpoint contract differs")
    return result


def _audit_rows() -> list[dict[str, str]]:
    return [
        {
            "field": "local_source_checkpoint",
            "classification": "known",
            "value": "epoch 404, global step 26730, exact SHA256",
            "evidence": "frozen full Lightning checkpoint",
        },
        {
            "field": "t3_sequence_database",
            "classification": "known",
            "value": (
                "RIO sliding T3 raw train/validation/test 858/123/113; "
                "active train 855 after 3 known-empty exclusions"
            ),
            "evidence": "content-bound local YAML and fail-closed loader",
        },
        {
            "field": "model_loss_taxonomy",
            "classification": "known",
            "value": "Concerto, 100 queries, NYU40-18, weighted criterion",
            "evidence": "P2 config and executable code",
        },
        {
            "field": "official_concerto_weight_identity",
            "classification": "unknown",
            "value": "not reported",
            "evidence": "P2 provenance audit",
        },
        {
            "field": "official_optimizer_precision_details",
            "classification": "unknown",
            "value": "betas/eps/weight decay/precision not fully reported",
            "evidence": "P2 provenance audit",
        },
        {
            "field": "official_augmentation_exactness",
            "classification": "unknown",
            "value": "exact transform list and versions not reported",
            "evidence": "P2 provenance audit",
        },
        {
            "field": "local_p2_recipe",
            "classification": "reconstructed",
            "value": "local paper-aligned reproduction with safety fixes",
            "evidence": "P2 config audit and reproduction target",
        },
        {
            "field": "checkpoint_selection",
            "classification": "reconstructed",
            "value": "best validation checkpoint at epoch 404",
            "evidence": "checkpoint callback state and metadata",
        },
        {
            "field": "adaptation_duration",
            "classification": "assumed",
            "value": "45 epochs / 2160 optimizer updates",
            "evidence": "single frozen reviewer-closure choice",
        },
        {
            "field": "adaptation_learning_rate",
            "classification": "assumed",
            "value": "fresh OneCycleLR with max LR 5e-5",
            "evidence": "10x lower than P2 max LR; no sweep",
        },
        {
            "field": "adaptation_batch_topology",
            "classification": "assumed",
            "value": "2 A40, batch 1/GPU, accumulation 16",
            "evidence": "effective batch 32 preserved pending smoke",
        },
    ]


def _publish_exact(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file():
            raise FileExistsError(f"output is not a regular file: {path}")
        if path.read_bytes() == payload:
            return
        raise FileExistsError(f"output already contains different content: {path}")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def build_t3_training_audit(
    *,
    recipe_path: str | Path = RECIPE_PATH,
    checkpoint_path: str | Path = CHECKPOINT_PATH,
    output_root: str | Path = OUTPUT_ROOT,
) -> dict[str, object]:
    recipe = load_t3_adaptation_recipe(recipe_path)
    source = recipe["source"]
    if (
        sha256_file(PROJECT_ROOT / "conf/config_p2_rescene4d_concerto_t2.yaml")
        != source["p2_config_sha256"]
    ):
        raise ReviewerClosureTrainingError("P2 source config SHA256 differs")
    data = recipe["data"]
    if (
        sha256_file(
            PROJECT_ROOT / "data/processed/rio/sequence_database_sliding_3.yaml"
        )
        != data["rio_sequence_database_sha256"]
    ):
        raise ReviewerClosureTrainingError("T3 sequence database SHA256 differs")
    checkpoint = inspect_t3_source_checkpoint(
        checkpoint_path,
        expected_sha256=str(source["checkpoint_sha256"]),
    )
    t2, adapted = compose_t3_adaptation_config(recipe)
    differences = adaptation_config_differences(t2, adapted)
    if set(differences) != set(ALLOWED_ADAPTATION_CONFIG_PATHS):
        raise ReviewerClosureTrainingError("T3 adaptation config differences differ")
    checkpoint_difference = differences.get("general.checkpoint")
    if not isinstance(checkpoint_difference, Mapping) or checkpoint_difference.get(
        "adapted"
    ) != str(CHECKPOINT_PATH):
        raise ReviewerClosureTrainingError("T3 checkpoint config difference differs")
    published_differences = {path: dict(values) for path, values in differences.items()}
    published_differences["general.checkpoint"]["adapted"] = (
        "repo:checkpoints/rescene4d_concerto_t2_repro.ckpt"
    )
    rows = _audit_rows()
    counts = dict(sorted(Counter(row["classification"] for row in rows).items()))
    artifact: dict[str, object] = {
        "schema_version": 1,
        "status": "pass",
        "selected_level": 2,
        "paper_name": recipe["paper_name"],
        "level1_claimed": False,
        "level1_rejection_reason": (
            "official recipe provenance is incomplete and the local T2 source is a "
            "documented reproduction with explicit choices"
        ),
        "recipe_sha256": sha256_file(recipe_path),
        "checkpoint": checkpoint,
        "classification_counts": counts,
        "audit_rows": rows,
        "allowed_config_differences": published_differences,
    }
    artifact["content_sha256"] = _content_sha256(artifact)
    report = [
        "# ReScene Horizon Training Audit",
        "",
        "Selected comparison: **Level 2**, `ReScene4D T2-to-T3 Horizon-Adapted`.",
        "",
        "Level 1 is not claimed: official recipe provenance is incomplete and the local T2 source is a documented reproduction with explicit choices.",
        "",
        f"The source is the exact epoch 404 checkpoint at global step {checkpoint['global_step']}; optimizer and scheduler state are audited but not resumed.",
        "",
        "| Field | Classification | Value | Evidence |",
        "|---|---|---|---|",
    ]
    report.extend(
        f"| {row['field']} | {row['classification']} | {row['value']} | {row['evidence']} |"
        for row in rows
    )
    report.extend(
        (
            "",
            "## Frozen Adaptation",
            "",
            "RIO changes from T2 to T3; ScanNet remains T1. Backbone, query count, label space, weighted loss definition, and effective batch size remain fixed. A fresh AdamW/OneCycle schedule runs once for 45 epochs (2160 updates) at max LR 5e-5. Actual scan exposures, wall time, GPU-hours, and checkpoint reload are mandatory runtime evidence.",
            "",
        )
    )
    output = Path(output_root)
    _publish_exact(
        output / "t3_training_recipe_audit.json",
        (json.dumps(artifact, allow_nan=False, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        ),
    )
    _publish_exact(
        output / "REScene_HORIZON_TRAINING_AUDIT.md",
        "\n".join(report).encode("utf-8"),
    )
    return {
        "status": artifact["status"],
        "selected_level": artifact["selected_level"],
        "paper_name": artifact["paper_name"],
        "classification_counts": artifact["classification_counts"],
        "content_sha256": artifact["content_sha256"],
    }


__all__ = [
    "ALLOWED_ADAPTATION_CONFIG_PATHS",
    "ReviewerClosureTrainingError",
    "adaptation_config_differences",
    "build_t3_training_audit",
    "compose_t3_adaptation_config",
    "inspect_t3_source_checkpoint",
    "load_t3_adaptation_recipe",
]
