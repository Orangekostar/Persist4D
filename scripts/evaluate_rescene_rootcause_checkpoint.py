#!/usr/bin/env python3
"""Prepare, evaluate, and summarize formal ReScene root-cause checkpoints."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import os
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_sonata_second_checkpoint import (
    normalize_metrics,
    strict_load_task_checkpoint,
)
from scripts.rescene_rootcause_preflight import portable_variant_config
from utils.rescene_rootcause_evaluation import (
    EVALUATION_SEEDS,
    METRIC_NAMES,
    RootCauseEvaluationError,
    build_checkpoint_manifest,
    summarize_epoch_runs,
    validate_checkpoint_manifest_binding,
    validate_checkpoint_payload,
)
from utils.rescene_rootcause_preflight import (
    canonical_sha256,
    validate_portable_payload,
)

CONFIG_NAME = "config_rescene4d_concerto_rootcause"
DEFAULT_AUTHORIZATION = (
    PROJECT_ROOT
    / "artifacts/rescene_task_learning_root_cause_v1/short_curves/variant_manifest.json"
)
RUNTIME_CONTRACT = {
    "accelerator": "gpu",
    "devices": 1,
    "batch_size": 1,
    "num_workers": 4,
    "precision": "32-true",
    "seed_workers": True,
}


def _load_json(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RootCauseEvaluationError(f"{name} is not readable JSON") from error
    if not isinstance(value, dict):
        raise RootCauseEvaluationError(f"{name} must contain an object")
    return value


def _stable_file_identity(path: Path) -> dict[str, object]:
    try:
        before = path.lstat()
        if path.is_symlink() or not stat.S_ISREG(before.st_mode):
            raise OSError
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                size += len(block)
                digest.update(block)
        after = path.lstat()
    except OSError as error:
        raise RootCauseEvaluationError("formal input file is unavailable") from error
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ) or size != after.st_size:
        raise RootCauseEvaluationError("formal input changed while hashing")
    return {"bytes": size, "sha256": digest.hexdigest()}


def _validate_content_hash(payload: Mapping[str, Any], *, name: str) -> None:
    expected = payload.get("content_sha256")
    unsigned = dict(payload)
    unsigned.pop("content_sha256", None)
    if not isinstance(expected, str) or canonical_sha256(unsigned) != expected:
        raise RootCauseEvaluationError(f"{name} content hash differs")


def _validate_authorization(payload: Mapping[str, Any]) -> None:
    expected = payload.get("authorization_sha256")
    unsigned = dict(payload)
    unsigned.pop("authorization_sha256", None)
    if not isinstance(expected, str) or canonical_sha256(unsigned) != expected:
        raise RootCauseEvaluationError("variant authorization hash differs")
    validate_portable_payload(payload)


@contextmanager
def _evaluation_environment(pretrained: Path) -> Iterator[None]:
    values = {
        "CONCERTO_CHECKPOINT": str(pretrained),
        "RESCENE_ROOTCAUSE_VARIANT": "EVAL",
        "RESCENE_ROOTCAUSE_OUTPUT_DIR": "checkpoints/rootcause_evaluation",
        "RESCENE_ROOTCAUSE_OBJECTIVE_MODE": "weighted",
    }
    names = (*values, "RESCENE_ROOTCAUSE_COMMON_STATE", "RESCENE_ROOTCAUSE_COMMON_SHA256")
    previous = {name: os.environ.get(name) for name in names}
    os.environ.update(values)
    os.environ.pop("RESCENE_ROOTCAUSE_COMMON_STATE", None)
    os.environ.pop("RESCENE_ROOTCAUSE_COMMON_SHA256", None)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def compose_evaluation_config(pretrained: Path) -> DictConfig:
    """Compose one shared evaluation config for every root-cause variant."""

    with _evaluation_environment(pretrained), initialize_config_dir(
        version_base="1.2", config_dir=str(PROJECT_ROOT / "conf")
    ):
        config = compose(
            config_name=CONFIG_NAME,
            overrides=[
                "general.gpus=1",
                "general.train_mode=false",
                "general.rootcause_fail_closed_runtime=false",
                "data.batch_size=1",
                "data.test_batch_size=1",
                "data.num_workers=4",
                "trainer.precision=32-true",
            ],
        )
        OmegaConf.resolve(config)
    return config


def portable_evaluation_config(
    config: DictConfig | Mapping[str, Any],
    *,
    pretrained: Path,
    pretrained_reference: str,
) -> dict[str, Any]:
    resolved = (
        OmegaConf.to_container(config, resolve=True)
        if isinstance(config, DictConfig)
        else copy.deepcopy(dict(config))
    )
    if not isinstance(resolved, dict):
        raise RootCauseEvaluationError("evaluation config is invalid")
    actual = str(pretrained)

    def replace(value: object) -> object:
        if isinstance(value, dict):
            return {key: replace(item) for key, item in value.items()}
        if isinstance(value, list):
            return [replace(item) for item in value]
        return pretrained_reference if value == actual else value

    portable = replace(resolved)
    validate_portable_payload(portable)
    return portable


def evaluation_contract() -> dict[str, object]:
    contract: dict[str, object] = {
        "schema_version": 1,
        "seeds": list(EVALUATION_SEEDS),
        "runtime": RUNTIME_CONTRACT,
        "validation_split": "rio/validation/T2",
        "validation_sequence_count": 154,
        "validation_entrypoint": "pytorch_lightning.Trainer.validate",
        "metrics": list(METRIC_NAMES),
        "primary_metric": "SpatialStageMean",
    }
    contract["sha256"] = canonical_sha256(contract)
    return contract


def _checkpoint_training_config(
    payload: Mapping[str, Any],
    *,
    variant: str,
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    config = payload.get("hyper_parameters")
    if isinstance(config, DictConfig):
        config = OmegaConf.to_container(config, resolve=True)
    if not isinstance(config, Mapping):
        raise RootCauseEvaluationError("checkpoint config state is invalid")
    initialization = authorization["initialization"]
    portable = portable_variant_config(
        config,
        variant=variant,
        pretrained_reference=initialization["pretrained"]["reference"],
        common_reference=initialization["common_state"]["reference"],
    )
    expected = authorization["variants"][variant]
    if portable != expected["resolved_config"] or canonical_sha256(
        portable
    ) != expected["config_sha256"]:
        raise RootCauseEvaluationError("checkpoint training config differs")
    return portable


def prepare_checkpoint(
    *,
    variant: str,
    completed_epoch: int,
    checkpoint_path: Path,
    candidate_path: Path,
    authorization_path: Path,
) -> dict[str, object]:
    import torch

    authorization = _load_json(authorization_path, name="variant authorization")
    _validate_authorization(authorization)
    candidate = _load_json(candidate_path, name="candidate record")
    file_identity = _stable_file_identity(checkpoint_path)
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise RootCauseEvaluationError("checkpoint is unreadable") from error
    facts = validate_checkpoint_payload(payload, completed_epoch=completed_epoch)
    if _stable_file_identity(checkpoint_path) != file_identity:
        raise RootCauseEvaluationError("checkpoint changed while validating")
    portable_config = _checkpoint_training_config(
        payload, variant=variant, authorization=authorization
    )
    facts["training_config_sha256"] = canonical_sha256(portable_config)
    return build_checkpoint_manifest(
        variant=variant,
        completed_epoch=completed_epoch,
        authorization=authorization,
        candidate=candidate,
        file_identity=file_identity,
        checkpoint_facts=facts,
    )


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def evaluate_checkpoint(
    *,
    checkpoint_path: Path,
    checkpoint_manifest_path: Path,
    authorization_path: Path,
    pretrained_path: Path,
    seed: int,
    device_index: int,
    limit_val_batches: int | None = None,
) -> dict[str, object]:
    import hydra
    import torch
    from pytorch_lightning import Trainer, seed_everything

    from trainer.trainer import InstanceSegmentation

    if seed not in EVALUATION_SEEDS:
        raise RootCauseEvaluationError("evaluation seed is not preregistered")
    if limit_val_batches is not None and limit_val_batches <= 0:
        raise RootCauseEvaluationError("smoke batch limit must be positive")
    if not torch.cuda.is_available() or not 0 <= device_index < torch.cuda.device_count():
        raise RootCauseEvaluationError("evaluation device is unavailable")
    manifest = _load_json(checkpoint_manifest_path, name="checkpoint manifest")
    _validate_content_hash(manifest, name="checkpoint manifest")
    authorization = _load_json(authorization_path, name="variant authorization")
    _validate_authorization(authorization)
    variant = validate_checkpoint_manifest_binding(
        manifest, authorization=authorization
    )
    identity = _stable_file_identity(checkpoint_path)
    for field in ("bytes", "sha256"):
        if identity[field] != manifest["checkpoint"][field]:
            raise RootCauseEvaluationError("checkpoint file differs from manifest")
    expected_pretrained = authorization["initialization"]["pretrained"]
    pretrained_identity = _stable_file_identity(pretrained_path)
    for field in ("bytes", "sha256"):
        if pretrained_identity[field] != expected_pretrained[field]:
            raise RootCauseEvaluationError("pretrained encoder differs")

    seed_everything(seed, workers=True)
    config = compose_evaluation_config(pretrained_path)
    portable_config = portable_evaluation_config(
        config,
        pretrained=pretrained_path,
        pretrained_reference=expected_pretrained["reference"],
    )
    system = InstanceSegmentation(config)
    strict_load = strict_load_task_checkpoint(system, checkpoint_path)
    if _stable_file_identity(checkpoint_path) != identity:
        raise RootCauseEvaluationError("checkpoint changed while loading")
    validation_dataset = hydra.utils.instantiate(config.data.validation_dataset)
    if len(validation_dataset) != 154:
        raise RootCauseEvaluationError("validation sequence count differs")
    if limit_val_batches is not None and limit_val_batches > len(validation_dataset):
        raise RootCauseEvaluationError("smoke batch limit exceeds validation split")
    system.validation_dataset = validation_dataset
    system.labels_info = validation_dataset.label_info
    for parameter in system.parameters():
        parameter.requires_grad_(False)
    system.eval()
    trainer = Trainer(
        accelerator="gpu",
        devices=[device_index],
        logger=False,
        callbacks=[],
        enable_checkpointing=False,
        enable_model_summary=False,
        default_root_dir="checkpoints/rootcause_evaluation",
        deterministic=bool(config.trainer.deterministic),
        precision="32-true",
        limit_val_batches=(
            limit_val_batches if limit_val_batches is not None else 1.0
        ),
    )
    started = time.perf_counter()
    results = trainer.validate(
        system, dataloaders=system.val_dataloader(), verbose=False
    )
    elapsed = time.perf_counter() - started
    if not isinstance(results, list) or len(results) != 1:
        raise RootCauseEvaluationError("Lightning validation result is invalid")
    metrics = normalize_metrics(results[0])
    contract = evaluation_contract()
    return {
        "schema_version": 1,
        "status": "pass",
        "scope": "smoke" if limit_val_batches is not None else "official_like_t2",
        "variant": variant,
        "completed_epoch": manifest["checkpoint"]["selected_epoch"],
        "seed": seed,
        "source_commit": _git_head(),
        "contract_sha256": contract["sha256"],
        "variant_authorization_sha256": authorization["authorization_sha256"],
        "checkpoint_manifest_sha256": manifest["content_sha256"],
        "checkpoint_sha256": identity["sha256"],
        "evaluation_config_sha256": canonical_sha256(portable_config),
        "validation_sequence_count": (
            min(len(validation_dataset), limit_val_batches)
            if limit_val_batches is not None
            else len(validation_dataset)
        ),
        "runtime": {
            **RUNTIME_CONTRACT,
            "device_index": device_index,
            "gpu_name": torch.cuda.get_device_name(device_index),
        },
        "strict_load": strict_load,
        "metrics": metrics,
        "SpatialStageMean": (
            metrics["stage1_mAP"] + metrics["stage2_mAP"]
        )
        / 2.0,
        "elapsed_seconds": elapsed,
    }


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True)
        + "\n"
    ).encode("ascii")


def _csv_bytes(rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("ascii")


def _publish(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_file() and not path.is_symlink() and path.read_bytes() == payload:
            return
        raise RootCauseEvaluationError("refusing to overwrite evaluation output")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def summarize_outputs(
    *,
    run_paths: Sequence[Path],
    variants: Sequence[str],
    completed_epoch: int,
) -> dict[str, bytes]:
    runs = [_load_json(path, name="official-like run") for path in run_paths]
    summary = summarize_epoch_runs(
        runs, variants=variants, completed_epoch=completed_epoch
    )
    per_seed_fields = (
        "variant",
        "completed_epoch",
        "seed",
        *METRIC_NAMES,
        "SpatialStageMean",
        "validation_sequence_count",
        "checkpoint_sha256",
        "elapsed_seconds",
    )
    per_seed_rows = [
        {
            "variant": run["variant"],
            "completed_epoch": run["completed_epoch"],
            "seed": run["seed"],
            **run["metrics"],
            "SpatialStageMean": (
                float(run["metrics"]["stage1_mAP"])
                + float(run["metrics"]["stage2_mAP"])
            )
            / 2.0,
            "validation_sequence_count": run["validation_sequence_count"],
            "checkpoint_sha256": run["checkpoint_sha256"],
            "elapsed_seconds": run["elapsed_seconds"],
        }
        for run in sorted(runs, key=lambda row: (row["variant"], row["seed"]))
    ]
    summary_fields = (
        "variant",
        "seed_count",
        *(f"{name}_{statistic}" for name in (*METRIC_NAMES, "SpatialStageMean") for statistic in ("mean", "std")),
        "paired_spatial_delta_mean",
        "paired_spatial_positive_seed_count",
    )
    summary_rows = [
        {
            field: (
                variant if field == "variant" else summary[variant].get(field, "")
            )
            for field in summary_fields
        }
        for variant in variants
    ]
    return {
        "per_seed.csv": _csv_bytes(per_seed_rows, per_seed_fields),
        "summary.csv": _csv_bytes(summary_rows, summary_fields),
        "summary.json": _json_bytes(summary),
    }


def _parse_variants(value: str) -> tuple[str, ...]:
    variants = tuple(value.split(","))
    if not variants or len(variants) != len(set(variants)):
        raise argparse.ArgumentTypeError("variants must be a unique comma-separated list")
    return variants


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--variant", required=True)
    prepare.add_argument("--epoch", type=int, required=True)
    prepare.add_argument("--checkpoint", type=Path, required=True)
    prepare.add_argument("--candidate", type=Path, required=True)
    prepare.add_argument("--authorization", type=Path, default=DEFAULT_AUTHORIZATION)
    prepare.add_argument("--output", type=Path, required=True)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--checkpoint-manifest", type=Path, required=True)
    evaluate.add_argument("--authorization", type=Path, default=DEFAULT_AUTHORIZATION)
    evaluate.add_argument("--pretrained", type=Path, required=True)
    evaluate.add_argument("--seed", type=int, choices=EVALUATION_SEEDS, required=True)
    evaluate.add_argument("--device", type=int, required=True)
    evaluate.add_argument("--limit-val-batches", type=int)
    evaluate.add_argument("--output", type=Path, required=True)

    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--runs", type=Path, nargs="+", required=True)
    summarize.add_argument("--variants", type=_parse_variants, required=True)
    summarize.add_argument("--epoch", type=int, required=True)
    summarize.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "prepare":
        result = prepare_checkpoint(
            variant=arguments.variant,
            completed_epoch=arguments.epoch,
            checkpoint_path=arguments.checkpoint,
            candidate_path=arguments.candidate,
            authorization_path=arguments.authorization,
        )
        _publish(arguments.output, _json_bytes(result))
    elif arguments.command == "evaluate":
        result = evaluate_checkpoint(
            checkpoint_path=arguments.checkpoint,
            checkpoint_manifest_path=arguments.checkpoint_manifest,
            authorization_path=arguments.authorization,
            pretrained_path=arguments.pretrained,
            seed=arguments.seed,
            device_index=arguments.device,
            limit_val_batches=arguments.limit_val_batches,
        )
        _publish(arguments.output, _json_bytes(result))
    else:
        outputs = summarize_outputs(
            run_paths=arguments.runs,
            variants=arguments.variants,
            completed_epoch=arguments.epoch,
        )
        for name, payload in outputs.items():
            _publish(arguments.output_dir / name, payload)
        result = {"status": "pass", "outputs": sorted(outputs)}
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
