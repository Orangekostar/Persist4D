#!/usr/bin/env python3
"""Freeze and qualify the Sonata second-perception task checkpoint."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import stat
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
EVALUATION_SEEDS = (45, 46, 47)
RUNTIME_CONTRACT = {
    "accelerator": "gpu",
    "devices": 1,
    "batch_size": 1,
    "num_workers": 4,
    "precision": "32-true",
}
METRIC_KEYS = {
    "t_mAP": "val_mean_t-AP",
    "t_mAP50": "val_mean_t-AP_50",
    "t_mAP25": "val_mean_t-AP_25",
    "overall_mAP": "val_mean_AP",
    "stage1_mAP": "val_mean_stage1-AP",
    "stage2_mAP": "val_mean_stage2-AP",
}


@dataclass(frozen=True)
class ModelSpec:
    config_name: str
    pretrained_environment: str
    pretrained_checkpoint: Path
    pretrained_reference: str
    pretrained_sha256: str
    task_checkpoint: Path
    task_checkpoint_reference: str
    task_checkpoint_sha256: str


MODEL_SPECS = {
    "sonata": ModelSpec(
        config_name="config_rescene4d_sonata_second",
        pretrained_environment="SONATA_CHECKPOINT",
        pretrained_checkpoint=(
            Path.home()
            / "persist4d-sonata-second-perception-v1/training/.verified_inputs/"
            "sonata-c5ced5acdae30d1c469713398073a866e25e6e414e23feed5dc025373657ac50.pth"
        ),
        pretrained_reference="external:verified_pretrained/sonata.pth",
        pretrained_sha256=(
            "c5ced5acdae30d1c469713398073a866e25e6e414e23feed5dc025373657ac50"
        ),
        task_checkpoint=(
            PROJECT_ROOT / "checkpoints/rescene4d_sonata_t2_second.ckpt"
        ),
        task_checkpoint_reference=(
            "repo-ignored:checkpoints/rescene4d_sonata_t2_second.ckpt"
        ),
        task_checkpoint_sha256=(
            "3d6432711dd9639d9e9203134d846d9a1a29f09b7fb3fbb85375e2127945a199"
        ),
    ),
    "concerto": ModelSpec(
        config_name="config_p2_rescene4d_concerto_t2",
        pretrained_environment="CONCERTO_CHECKPOINT",
        pretrained_checkpoint=(
            Path.home() / ".cache/persist4d/concerto/concerto_base.pth"
        ),
        pretrained_reference="external:verified_pretrained/concerto_base.pth",
        pretrained_sha256=(
            "845ec7dec97a5fabff8fadb5d9858ac6734347b612d1a4b574213419c139de07"
        ),
        task_checkpoint=(
            PROJECT_ROOT / "checkpoints/rescene4d_concerto_t2_repro.ckpt"
        ),
        task_checkpoint_reference=(
            "repo-ignored:checkpoints/rescene4d_concerto_t2_repro.ckpt"
        ),
        task_checkpoint_sha256=(
            "85ed1aba60320cd19798536b71b91dbc156b7ea60f838832bc0bbbdba131546e"
        ),
    ),
}
OUTPUT_ROOT = PROJECT_ROOT / "artifacts/sonata_second_perception_v1/checkpoint"
TRAINING_MANIFEST_PATH = (
    PROJECT_ROOT
    / "artifacts/sonata_second_perception_v1/training/TRAINING_MANIFEST.json"
)
DATA_MANIFEST_SHA256 = (
    "694d4394b481b02bf49573dc121276a452526fe274657ca7d1915ba0afba5c4e"
)


def compose_evaluation_config(model_name: str) -> Any:
    """Compose one identity config under the shared SS5 runtime contract."""

    from hydra import compose, initialize_config_dir

    try:
        spec = MODEL_SPECS[model_name]
    except KeyError as error:
        raise ValueError(f"unsupported model identity: {model_name}") from error
    os.environ[spec.pretrained_environment] = str(spec.pretrained_checkpoint)
    with initialize_config_dir(
        version_base=None,
        config_dir=str(PROJECT_ROOT / "conf"),
    ):
        return compose(
            config_name=spec.config_name,
            overrides=[
                "general.gpus=1",
                "general.train_mode=false",
                "data.batch_size=1",
                "data.test_batch_size=1",
                "data.num_workers=4",
                "trainer.precision=32-true",
            ],
        )


def strict_load_task_checkpoint(system: Any, checkpoint_path: Path) -> dict[str, object]:
    """Strictly load a complete Lightning task state into a fresh system."""

    import torch

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or not isinstance(
        payload.get("state_dict"), Mapping
    ):
        raise TypeError("task checkpoint must contain a full Lightning state_dict")
    state_dict = payload["state_dict"]
    try:
        incompatible = system.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise RuntimeError("strict task checkpoint load failed") from error
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("strict task checkpoint load failed")
    return {
        "state_dict_entry_count": len(state_dict),
        "strict": True,
    }


def normalize_metrics(values: Mapping[str, object]) -> dict[str, float]:
    """Extract the six preregistered metrics as finite unit-interval scalars."""

    normalized = {}
    for output_name, source_name in METRIC_KEYS.items():
        if source_name not in values:
            raise ValueError(f"missing evaluation metric: {source_name}")
        value = values[source_name]
        if hasattr(value, "detach"):
            value = value.detach().cpu().item()
        try:
            number = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid evaluation metric: {source_name}") from error
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            raise ValueError(f"invalid evaluation metric: {source_name}")
        normalized[output_name] = number
    return normalized


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _require_input(path: Path, expected_sha256: str, *, name: str) -> dict[str, object]:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise FileNotFoundError(f"missing {name}") from error
    if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{name} must be a regular non-symlink file")
    digest = _sha256_file(path)
    if digest != expected_sha256:
        raise ValueError(f"{name} SHA256 mismatch")
    return {
        "byte_size": metadata.st_size,
        "sha256": digest,
    }


def _portable_config_sha256(config: Any, spec: ModelSpec) -> str:
    from omegaconf import OmegaConf

    resolved = OmegaConf.to_container(config, resolve=True)

    def replace(value: object) -> object:
        if isinstance(value, dict):
            return {key: replace(item) for key, item in value.items()}
        if isinstance(value, list):
            return [replace(item) for item in value]
        if value == str(spec.pretrained_checkpoint):
            return spec.pretrained_reference
        return value

    return _canonical_sha256(replace(resolved))


def evaluation_contract() -> dict[str, object]:
    contract = {
        "schema_version": 1,
        "seeds": list(EVALUATION_SEEDS),
        "runtime": RUNTIME_CONTRACT,
        "validation_split": "rio/validation/T2",
        "validation_entrypoint": "pytorch_lightning.Trainer.validate",
        "stochastic_grid_sample": "seeded_and_reported",
        "metrics": METRIC_KEYS,
        "data_manifest_sha256": DATA_MANIFEST_SHA256,
    }
    return {**contract, "sha256": _canonical_sha256(contract)}


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def evaluate_model(
    *,
    model_name: str,
    seed: int,
    device_index: int,
    limit_val_batches: int | None = None,
) -> dict[str, object]:
    """Evaluate one model/seed through the shared official-like T2 harness."""

    import hydra
    import torch
    from pytorch_lightning import Trainer, seed_everything

    from trainer.trainer import InstanceSegmentation

    if seed not in EVALUATION_SEEDS:
        raise ValueError("seed is outside the preregistered SS5 set")
    if not torch.cuda.is_available() or not 0 <= device_index < torch.cuda.device_count():
        raise ValueError("device must identify one available CUDA GPU")
    spec = MODEL_SPECS[model_name]
    pretrained = _require_input(
        spec.pretrained_checkpoint,
        spec.pretrained_sha256,
        name=f"{model_name} pretrained checkpoint",
    )
    task = _require_input(
        spec.task_checkpoint,
        spec.task_checkpoint_sha256,
        name=f"{model_name} task checkpoint",
    )
    seed_everything(seed, workers=True)
    config = compose_evaluation_config(model_name)
    system = InstanceSegmentation(config)
    strict_load = strict_load_task_checkpoint(system, spec.task_checkpoint)
    validation_dataset = hydra.utils.instantiate(config.data.validation_dataset)
    system.validation_dataset = validation_dataset
    system.labels_info = validation_dataset.label_info
    for parameter in system.parameters():
        parameter.requires_grad_(False)
    system.eval()
    validation_loader = system.val_dataloader()
    trainer = Trainer(
        accelerator="gpu",
        devices=[device_index],
        logger=False,
        callbacks=[],
        enable_checkpointing=False,
        enable_model_summary=False,
        default_root_dir=OUTPUT_ROOT,
        deterministic=bool(config.trainer.deterministic),
        precision="32-true",
        limit_val_batches=(
            limit_val_batches if limit_val_batches is not None else 1.0
        ),
    )
    started = time.perf_counter()
    results = trainer.validate(system, dataloaders=validation_loader, verbose=False)
    elapsed_seconds = time.perf_counter() - started
    if not isinstance(results, list) or len(results) != 1:
        raise RuntimeError("Lightning validation returned an invalid result set")
    metrics = normalize_metrics(results[0])
    contract = evaluation_contract()
    return {
        "schema_version": 1,
        "status": "pass",
        "scope": "smoke" if limit_val_batches is not None else "official_like_t2",
        "model": model_name,
        "seed": seed,
        "source_commit": _git_commit(),
        "evaluation_contract_sha256": contract["sha256"],
        "config_name": spec.config_name,
        "portable_config_sha256": _portable_config_sha256(config, spec),
        "data_manifest_sha256": DATA_MANIFEST_SHA256,
        "validation_sequence_count": (
            min(len(validation_dataset), limit_val_batches)
            if limit_val_batches is not None
            else len(validation_dataset)
        ),
        "runtime": {
            **RUNTIME_CONTRACT,
            "device_index": device_index,
            "gpu_name": torch.cuda.get_device_name(device_index),
            "seed_workers": True,
        },
        "pretrained_checkpoint": {
            "reference": spec.pretrained_reference,
            **pretrained,
        },
        "task_checkpoint": {
            "reference": spec.task_checkpoint_reference,
            **task,
        },
        "strict_load": strict_load,
        "metrics": metrics,
        "elapsed_seconds": elapsed_seconds,
    }


def _publish(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_file() and not path.is_symlink() and path.read_bytes() == payload:
            return
        raise FileExistsError(f"refusing to overwrite SS5 output: {path.name}")
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


def _json_payload(value: object) -> bytes:
    return (
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def build_checkpoint_manifest(
    *,
    training_manifest: Mapping[str, object],
    sonata_file: Mapping[str, object],
    concerto_file: Mapping[str, object],
    evidence_source_commit: str,
    pretrained_files: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, object]:
    """Build the SS5 checkpoint contract from validated SS4 evidence."""

    if training_manifest.get("status") != "pass":
        raise ValueError("training evidence is not complete")
    budget = training_manifest.get("budget")
    if not isinstance(budget, Mapping) or budget.get("completed_epochs") != 450:
        raise ValueError("checkpoint qualification requires 450 completed epochs")
    selection = training_manifest.get("checkpoint_selection")
    if (
        not isinstance(selection, Mapping)
        or selection.get("monitor") != "val_mean_t-AP"
        or selection.get("mode") != "max"
    ):
        raise ValueError("invalid preregistered checkpoint selection contract")
    checkpoints = training_manifest.get("checkpoints")
    if not isinstance(checkpoints, list):
        raise TypeError("training checkpoint inventory is invalid")
    selected = [
        value
        for value in checkpoints
        if isinstance(value, Mapping) and value.get("role") == "best_validation"
    ]
    if len(selected) != 1:
        raise ValueError("training checkpoint inventory lacks one best checkpoint")
    selected_checkpoint = selected[0]
    if (
        selected_checkpoint.get("epoch") != selection.get("selected_epoch")
        or selected_checkpoint.get("sha256") != sonata_file.get("sha256")
        or selected_checkpoint.get("byte_size") != sonata_file.get("byte_size")
    ):
        raise ValueError("frozen Sonata checkpoint differs from SS4 selection")
    mode = sonata_file.get("mode")
    if not isinstance(mode, str) or int(mode, 8) & 0o222:
        raise ValueError("frozen Sonata checkpoint must be read-only")
    bindings = training_manifest.get("bindings")
    if not isinstance(bindings, Mapping):
        raise TypeError("training provenance bindings are invalid")
    manifest = {
        "schema_version": 1,
        "status": "pass",
        "stage": "SS5",
        "selection_used_protocol_b": False,
        "selection_contract": {
            "monitor": "val_mean_t-AP",
            "mode": "max",
            "validation_event_count": selection.get("validation_event_count"),
        },
        "sonata": {
            "reference": MODEL_SPECS["sonata"].task_checkpoint_reference,
            "selection": "highest val_mean_t-AP",
            "epoch": selected_checkpoint.get("epoch"),
            "global_step": selected_checkpoint.get("global_step"),
            "selection_metric_exact": selection.get("selection_metric_exact"),
            "state_dict_entry_count": selected_checkpoint.get(
                "state_dict_entry_count"
            ),
            **dict(sonata_file),
        },
        "concerto": {
            "reference": MODEL_SPECS["concerto"].task_checkpoint_reference,
            **dict(concerto_file),
        },
        "training": {
            "completed_epochs": budget.get("completed_epochs"),
            "optimizer_steps": budget.get("optimizer_steps"),
            "source_commit": bindings.get("source_commit"),
            "config_sha256": bindings.get("config_sha256"),
            "pretrained_sonata_sha256": bindings.get("weight_sha256"),
        },
        "evidence_source_commit": evidence_source_commit,
        "evaluation_contract": evaluation_contract(),
    }
    if pretrained_files is not None:
        manifest["pretrained_inputs"] = {
            name: dict(value) for name, value in pretrained_files.items()
        }
    manifest["content_sha256"] = _canonical_sha256(manifest)
    return manifest


def qualification_gate(
    *,
    sonata: Mapping[str, object],
    concerto: Mapping[str, object],
    provenance_complete: bool,
    completed_epochs: int,
) -> dict[str, object]:
    """Apply the preregistered SS5 qualification gate without tolerance."""

    values = {}
    for identity, metrics in (("sonata", sonata), ("concerto", concerto)):
        for metric in ("t_mAP", "overall_mAP"):
            try:
                value = float(metrics[metric])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"invalid {identity} qualification metric") from error
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"invalid {identity} qualification metric")
            values[(identity, metric)] = value
    if not provenance_complete or completed_epochs != 450:
        label = "SQ-RED"
        reason = "provenance or completed 450-epoch training contract is invalid"
    elif (
        values[("sonata", "t_mAP")] >= 0.297
        and values[("sonata", "overall_mAP")]
        >= values[("concerto", "overall_mAP")]
    ):
        label = "SQ-GREEN"
        reason = "temporal threshold and matched spatial parity both pass"
    elif (
        values[("sonata", "t_mAP")] < values[("concerto", "t_mAP")]
        and values[("sonata", "overall_mAP")]
        < values[("concerto", "overall_mAP")]
    ):
        label = "SQ-RED"
        reason = "Sonata is weaker than Concerto on both temporal and spatial metrics"
    else:
        label = "SQ-YELLOW"
        reason = "functional evidence does not satisfy every automatic progression gate"
    return {
        "label": label,
        "reason": reason,
        "authorizes_ss6": label == "SQ-GREEN",
        "threshold_t_mAP": 0.297,
        "spatial_tolerance": None,
    }


def _validated_output_metrics(values: object) -> dict[str, float]:
    if not isinstance(values, Mapping) or set(values) != set(METRIC_KEYS):
        raise ValueError("official-like run metric schema differs")
    normalized = {}
    for name in METRIC_KEYS:
        try:
            value = float(values[name])
        except (TypeError, ValueError) as error:
            raise ValueError("official-like run metric is invalid") from error
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("official-like run metric is invalid")
        normalized[name] = value
    return normalized


def validate_run_matrix(
    runs: Sequence[Mapping[str, object]],
) -> dict[str, list[Mapping[str, object]]]:
    """Validate the exact matched model-by-seed official-like run matrix."""

    expected_pairs = {
        (model, seed) for model in MODEL_SPECS for seed in EVALUATION_SEEDS
    }
    observed_pairs = []
    for run in runs:
        observed_pairs.append((run.get("model"), run.get("seed")))
    if len(runs) != len(expected_pairs) or set(observed_pairs) != expected_pairs:
        raise ValueError("official-like runs must cover the exact model/seed cross-product")
    if len(set(observed_pairs)) != len(observed_pairs):
        raise ValueError("official-like runs must cover the exact model/seed cross-product")

    expected_contract = evaluation_contract()["sha256"]
    shared_bindings = set()
    config_hashes: dict[str, set[object]] = {model: set() for model in MODEL_SPECS}
    validated: dict[str, list[Mapping[str, object]]] = {
        model: [] for model in MODEL_SPECS
    }
    for run in runs:
        model = str(run["model"])
        spec = MODEL_SPECS[model]
        runtime = run.get("runtime")
        if not isinstance(runtime, Mapping) or any(
            runtime.get(key) != value for key, value in RUNTIME_CONTRACT.items()
        ):
            raise ValueError("official-like run runtime contract differs")
        if runtime.get("seed_workers") is not True:
            raise ValueError("official-like run runtime contract differs")
        task_checkpoint = run.get("task_checkpoint")
        strict_load = run.get("strict_load")
        if (
            run.get("status") != "pass"
            or run.get("scope") != "official_like_t2"
            or run.get("evaluation_contract_sha256") != expected_contract
            or run.get("data_manifest_sha256") != DATA_MANIFEST_SHA256
            or run.get("config_name") != spec.config_name
            or not isinstance(task_checkpoint, Mapping)
            or task_checkpoint.get("sha256") != spec.task_checkpoint_sha256
            or not isinstance(strict_load, Mapping)
            or strict_load.get("strict") is not True
        ):
            raise ValueError("official-like run provenance binding differs")
        sequence_count = run.get("validation_sequence_count")
        if not isinstance(sequence_count, int) or sequence_count <= 0:
            raise ValueError("official-like validation sequence count is invalid")
        _validated_output_metrics(run.get("metrics"))
        shared_bindings.add(
            (
                run.get("source_commit"),
                run.get("evaluation_contract_sha256"),
                run.get("data_manifest_sha256"),
                sequence_count,
                runtime.get("device_index"),
                runtime.get("gpu_name"),
            )
        )
        config_hashes[model].add(run.get("portable_config_sha256"))
        validated[model].append(run)
    if len(shared_bindings) != 1:
        raise ValueError("official-like shared evaluation binding differs")
    if any(len(hashes) != 1 or None in hashes for hashes in config_hashes.values()):
        raise ValueError("official-like per-model config binding differs")
    for model_runs in validated.values():
        model_runs.sort(key=lambda run: int(run["seed"]))
    return validated


def summarize_runs(runs: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Compute preregistered three-seed descriptive statistics."""

    if len(runs) != len(EVALUATION_SEEDS) or {
        run.get("seed") for run in runs
    } != set(EVALUATION_SEEDS):
        raise ValueError("summary requires the three preregistered seeds")
    summary: dict[str, object] = {"seed_count": len(runs)}
    normalized = [_validated_output_metrics(run.get("metrics")) for run in runs]
    for metric in METRIC_KEYS:
        values = [row[metric] for row in normalized]
        summary[f"{metric}_mean"] = statistics.mean(values)
        summary[f"{metric}_std"] = statistics.stdev(values)
        summary[f"{metric}_min"] = min(values)
        summary[f"{metric}_max"] = max(values)
    return summary


def _csv_payload(rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _read_json(path: Path) -> Mapping[str, object]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"JSON artifact must contain a mapping: {path.name}")
    return value


def _format_percent(value: object) -> str:
    return f"{100.0 * float(value):.3f}"


def finalize_qualification(*, output_root: Path = OUTPUT_ROOT) -> dict[str, object]:
    checkpoint_manifest_path = output_root / "CHECKPOINT_MANIFEST.json"
    checkpoint_manifest = _read_json(checkpoint_manifest_path)
    runs = [
        _read_json(output_root / "runs" / f"{model}_seed{seed}.json")
        for model in MODEL_SPECS
        for seed in EVALUATION_SEEDS
    ]
    validated = validate_run_matrix(runs)
    source_commits = {run["source_commit"] for run in runs}
    if source_commits != {checkpoint_manifest.get("evidence_source_commit")}:
        raise ValueError("checkpoint and evaluation source commits differ")
    summaries = {
        model: summarize_runs(model_runs)
        for model, model_runs in validated.items()
    }
    training = checkpoint_manifest.get("training")
    provenance_complete = (
        checkpoint_manifest.get("status") == "pass"
        and checkpoint_manifest.get("selection_used_protocol_b") is False
        and isinstance(training, Mapping)
        and training.get("completed_epochs") == 450
    )
    completed_epochs = int(training.get("completed_epochs", 0)) if isinstance(
        training, Mapping
    ) else 0
    gate = qualification_gate(
        sonata={
            "t_mAP": summaries["sonata"]["t_mAP_mean"],
            "overall_mAP": summaries["sonata"]["overall_mAP_mean"],
        },
        concerto={
            "t_mAP": summaries["concerto"]["t_mAP_mean"],
            "overall_mAP": summaries["concerto"]["overall_mAP_mean"],
        },
        provenance_complete=provenance_complete,
        completed_epochs=completed_epochs,
    )

    per_seed_fields = (
        "model",
        "seed",
        *METRIC_KEYS.keys(),
        "validation_sequence_count",
        "checkpoint_sha256",
        "portable_config_sha256",
        "elapsed_seconds",
    )
    per_seed_rows = []
    for model, model_runs in validated.items():
        for run in model_runs:
            per_seed_rows.append(
                {
                    "model": model,
                    "seed": run["seed"],
                    **dict(run["metrics"]),
                    "validation_sequence_count": run["validation_sequence_count"],
                    "checkpoint_sha256": run["task_checkpoint"]["sha256"],
                    "portable_config_sha256": run["portable_config_sha256"],
                    "elapsed_seconds": run["elapsed_seconds"],
                }
            )
    per_seed_payload = _csv_payload(per_seed_rows, per_seed_fields)

    summary_fields = (
        "model",
        "evidence",
        "seed_count",
        *(
            field
            for metric in METRIC_KEYS
            for field in (f"{metric}_mean", f"{metric}_std")
        ),
    )
    summary_rows = []
    external_rows = {
        "rescene4d_s_paper_reported": {"t_mAP_mean": 0.332, "overall_mAP_mean": 0.409},
        "rescene4d_c_paper_reported": {"t_mAP_mean": 0.348, "overall_mAP_mean": 0.433},
    }
    summary_rows.append(
        {
            "model": "rescene4d_s_paper_reported",
            "evidence": "external_paper_reported",
            "seed_count": "",
            **{
                field: external_rows["rescene4d_s_paper_reported"].get(field, "")
                for field in summary_fields[3:]
            },
        }
    )
    for model in ("sonata", "concerto"):
        summary_rows.append(
            {
                "model": f"our_{model}_reimplementation",
                "evidence": "measured_same_local_harness",
                **{
                    field: summaries[model].get(field, "")
                    for field in summary_fields
                    if field not in {"model", "evidence"}
                },
            }
        )
    summary_rows.append(
        {
            "model": "rescene4d_c_paper_reported",
            "evidence": "external_paper_reported",
            "seed_count": "",
            **{
                field: external_rows["rescene4d_c_paper_reported"].get(field, "")
                for field in summary_fields[3:]
            },
        }
    )
    summary_payload = _csv_payload(summary_rows, summary_fields)

    sonata = summaries["sonata"]
    concerto = summaries["concerto"]
    metric_headers = " | ".join(METRIC_KEYS)
    report_lines = [
        "# Sonata Qualification Report",
        "",
        f"- Gate: `{gate['label']}`",
        f"- Automatic SS6 authorization: `{str(gate['authorizes_ss6']).lower()}`",
        f"- Decision reason: {gate['reason']}",
        "- Evaluation: same local official-like T2 validation harness, seeds 45/46/47",
        "- Runtime: one NVIDIA A40, batch size 1, 4 workers, precision 32-true",
        "- Validation stochasticity: train-style GridSample retained and reported by seed",
        "- Scope: internal reviewer-closure gate; not a publication standard",
        "",
        f"| Model | Evidence | {metric_headers} |",
        f"|---|---|{'---:|' * len(METRIC_KEYS)}",
        "| ReScene4D-S paper reported | external | 33.200 |  |  | 40.900 |  |  |",
        "| Our Sonata reimplementation | measured mean | "
        + " | ".join(
            _format_percent(sonata[f"{metric}_mean"]) for metric in METRIC_KEYS
        )
        + " |",
        "| Our Concerto reimplementation | measured mean | "
        + " | ".join(
            _format_percent(concerto[f"{metric}_mean"]) for metric in METRIC_KEYS
        )
        + " |",
        "| ReScene4D-C paper reported | external | 34.800 |  |  | 43.300 |  |  |",
        "",
        "The paper-reported rows are external references and were not substituted",
        "for local measurements. The Sonata row is this task's reimplementation,",
        "not an official ReScene4D-S checkpoint.",
        "",
    ]
    if not gate["authorizes_ss6"]:
        report_lines.extend(
            [
                "Automatic progression stops at SS5. SS6 and SS7 were not run.",
                "",
            ]
        )
    report_payload = "\n".join(report_lines).encode("utf-8")
    output_payloads = {
        "official_like_per_seed.csv": per_seed_payload,
        "official_like_summary.csv": summary_payload,
        "SONATA_QUALIFICATION_REPORT.md": report_payload,
    }
    qualification_manifest = {
        "schema_version": 1,
        "status": "pass",
        "stage": "SS5",
        "source_commit": next(iter(source_commits)),
        "gate": gate,
        "run_sha256": {
            f"{run['model']}_seed{run['seed']}.json": _sha256_file(
                output_root / "runs" / f"{run['model']}_seed{run['seed']}.json"
            )
            for run in runs
        },
        "output_sha256": {
            name: hashlib.sha256(payload).hexdigest()
            for name, payload in output_payloads.items()
        },
    }
    qualification_manifest["content_sha256"] = _canonical_sha256(
        qualification_manifest
    )
    output_payloads["QUALIFICATION_MANIFEST.json"] = _json_payload(
        qualification_manifest
    )
    for name, payload in output_payloads.items():
        _publish(output_root / name, payload)
    return {
        "status": "pass",
        "gate": gate,
        "sonata_t_mAP_mean": sonata["t_mAP_mean"],
        "concerto_t_mAP_mean": concerto["t_mAP_mean"],
    }


def _checkpoint_payload_facts(
    path: Path, *, expected_epoch: int, expected_global_step: int
) -> dict[str, int]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = payload.get("state_dict") if isinstance(payload, Mapping) else None
    if (
        not isinstance(payload, Mapping)
        or payload.get("epoch") != expected_epoch
        or payload.get("global_step") != expected_global_step
        or not isinstance(state_dict, Mapping)
    ):
        raise ValueError("frozen Sonata checkpoint payload differs from SS4 evidence")
    return {"state_dict_entry_count": len(state_dict)}


def _file_contract(spec: ModelSpec, *, pretrained: bool) -> dict[str, object]:
    path = spec.pretrained_checkpoint if pretrained else spec.task_checkpoint
    digest = spec.pretrained_sha256 if pretrained else spec.task_checkpoint_sha256
    name = "pretrained checkpoint" if pretrained else "task checkpoint"
    contract = _require_input(path, digest, name=name)
    contract["mode"] = f"{stat.S_IMODE(path.stat().st_mode):04o}"
    contract["reference"] = (
        spec.pretrained_reference if pretrained else spec.task_checkpoint_reference
    )
    return contract


def prepare_checkpoint_artifacts(*, output_root: Path = OUTPUT_ROOT) -> dict[str, object]:
    training_payload = TRAINING_MANIFEST_PATH.read_bytes()
    training_manifest = json.loads(training_payload)
    if not isinstance(training_manifest, Mapping):
        raise TypeError("SS4 training manifest must be a mapping")
    sonata_task = _file_contract(MODEL_SPECS["sonata"], pretrained=False)
    concerto_task = _file_contract(MODEL_SPECS["concerto"], pretrained=False)
    selection = training_manifest.get("checkpoint_selection")
    budget = training_manifest.get("budget")
    if not isinstance(selection, Mapping) or not isinstance(budget, Mapping):
        raise TypeError("SS4 training manifest is incomplete")
    payload_facts = _checkpoint_payload_facts(
        MODEL_SPECS["sonata"].task_checkpoint,
        expected_epoch=int(selection["selected_epoch"]),
        expected_global_step=int(budget["optimizer_steps"]),
    )
    if payload_facts["state_dict_entry_count"] != 798:
        raise ValueError("frozen Sonata state_dict entry count differs")
    pretrained = {
        name: _file_contract(spec, pretrained=True)
        for name, spec in MODEL_SPECS.items()
    }
    manifest = build_checkpoint_manifest(
        training_manifest=training_manifest,
        sonata_file=sonata_task,
        concerto_file=concerto_task,
        evidence_source_commit=_git_commit(),
        pretrained_files=pretrained,
    )
    manifest["training_manifest_sha256"] = hashlib.sha256(
        training_payload
    ).hexdigest()
    manifest.pop("content_sha256")
    manifest["content_sha256"] = _canonical_sha256(manifest)
    markdown = "\n".join(
        [
            "# Sonata Second Checkpoint Selection",
            "",
            "- Status: `pass`",
            "- Selection rule: highest `val_mean_t-AP` (`mode=max`)",
            "- Protocol-B/B2/B3/B4 information used for selection: `false`",
            "- Selected epoch: `449`",
            "- Global optimizer step: `29700`",
            f"- Exact selection metric: `{manifest['sonata']['selection_metric_exact']}`",
            f"- Sonata SHA256: `{manifest['sonata']['sha256']}`",
            f"- Sonata bytes: `{manifest['sonata']['byte_size']}`",
            f"- Sonata file mode: `{manifest['sonata']['mode']}`",
            f"- Training source commit: `{manifest['training']['source_commit']}`",
            f"- Training config SHA256: `{manifest['training']['config_sha256']}`",
            "",
            "The checkpoint is the preregistered Top-1 validation checkpoint from",
            "the completed 450-epoch local Sonata reimplementation. It is not an",
            "official ReScene4D-S checkpoint.",
            "",
        ]
    ).encode("utf-8")
    _publish(output_root / "CHECKPOINT_MANIFEST.json", _json_payload(manifest))
    _publish(output_root / "CHECKPOINT_SELECTION.md", markdown)
    return {
        "status": "pass",
        "sonata_checkpoint_sha256": manifest["sonata"]["sha256"],
        "concerto_checkpoint_sha256": manifest["concerto"]["sha256"],
    }


def run_and_publish_evaluation(
    *, model_name: str, seed: int, device_index: int, output_root: Path
) -> dict[str, object]:
    result = evaluate_model(
        model_name=model_name,
        seed=seed,
        device_index=device_index,
    )
    _publish(
        output_root / "runs" / f"{model_name}_seed{seed}.json",
        _json_payload(result),
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)
    subparsers.add_parser("prepare")
    subparsers.add_parser("finalize")
    for stage in ("smoke", "evaluate"):
        command = subparsers.add_parser(stage)
        command.add_argument("--model", choices=tuple(MODEL_SPECS), required=True)
        command.add_argument("--device", type=int, default=0)
        if stage == "evaluate":
            command.add_argument("--seed", type=int, choices=EVALUATION_SEEDS, required=True)
            command.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    arguments = parser.parse_args(argv)
    if arguments.stage == "prepare":
        result = prepare_checkpoint_artifacts()
    elif arguments.stage == "finalize":
        result = finalize_qualification()
    elif arguments.stage == "smoke":
        result = evaluate_model(
            model_name=arguments.model,
            seed=EVALUATION_SEEDS[0],
            device_index=arguments.device,
            limit_val_batches=1,
        )
    else:
        result = run_and_publish_evaluation(
            model_name=arguments.model,
            seed=arguments.seed,
            device_index=arguments.device,
            output_root=arguments.output_root,
        )
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
