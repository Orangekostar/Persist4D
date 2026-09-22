#!/usr/bin/env python3
"""Evaluate Perception Gain checkpoints on the frozen official-like RIO T2 split."""

from __future__ import annotations

import argparse
import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

from scripts.perception_gain_evaluation import (
    CAL_UPDATES,
    DEFAULT_ASSETS,
    DEFAULT_EXTERNAL_ROOT,
    VARIANTS,
    PerceptionEvaluationError,
    _atomic_json,
    _external_reference,
    _load_evaluation_weights,
    _sha256,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts/perception_gain_v1"
LOCAL_T2_SEQUENCE_COUNT = 154


def build_local_t2_summary(
    *,
    variant: str,
    optimizer_update: int,
    checkpoint_reference: str,
    checkpoint_sha256: str,
    config_sha256: str,
    population_sha256: str,
    metrics: Mapping[str, object],
    elapsed_seconds: float,
    gpu_name: str,
) -> dict[str, object]:
    required = (
        "t_mAP",
        "t_mAP50",
        "t_mAP25",
        "overall_mAP",
        "stage1_mAP",
        "stage2_mAP",
    )
    normalised = {}
    for name, value in metrics.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        number = float(value)
        if math.isfinite(number) and 0.0 <= number <= 1.0:
            normalised[str(name)] = number
    if (
        variant not in VARIANTS
        or optimizer_update not in CAL_UPDATES
        or not isinstance(checkpoint_reference, str)
        or not checkpoint_reference
        or len(checkpoint_sha256) != 64
        or len(config_sha256) != 64
        or len(population_sha256) != 64
        or any(name not in normalised for name in required)
        or not math.isfinite(float(elapsed_seconds))
        or elapsed_seconds < 0.0
        or not isinstance(gpu_name, str)
        or not gpu_name
    ):
        raise PerceptionEvaluationError("LOCAL-T2 result contract differs")
    return {
        "schema_version": "perception-gain-local-t2-v1",
        "status": "PASS",
        "population_id": "official_like_rio_validation_t2_154",
        "variant": variant,
        "optimizer_update": optimizer_update,
        "checkpoint": checkpoint_reference,
        "checkpoint_sha256": checkpoint_sha256,
        "resolved_config_sha256": config_sha256,
        "population_manifest_sha256": population_sha256,
        "validation_sequence_count": LOCAL_T2_SEQUENCE_COUNT,
        "metrics": normalised,
        "SpatialStageMean": (normalised["stage1_mAP"] + normalised["stage2_mAP"]) / 2.0,
        "elapsed_seconds": float(elapsed_seconds),
        "gpu_hours": float(elapsed_seconds) / 3600.0,
        "gpu_name": gpu_name,
        "eval_seed": 45,
        "output_policy": "native",
    }


def run_local_t2_evaluation(
    *,
    variant: str,
    optimizer_update: int,
    checkpoint: Path | None,
    scorer_checkpoint: Path | None,
    assets_path: Path = DEFAULT_ASSETS,
    external_root: Path = DEFAULT_EXTERNAL_ROOT,
    output_path: Path | None = None,
    device_name: str = "cuda:0",
) -> dict[str, object]:
    import torch
    from omegaconf import OmegaConf, open_dict
    from pytorch_lightning import Trainer, seed_everything

    from scripts.evaluate_persist4d import _validate_cuda_device
    from scripts.evaluate_sonata_second_checkpoint import normalize_metrics
    from scripts.evaluate_task_memory import NATIVE_POPULATION_ID, _rio_population_base
    from scripts.perception_gain_foundation import _resolve_cache_assets
    from scripts.task_memory_contracts import canonical_json_sha256
    from scripts.train_perception_gain import compose_variant_config
    from trainer.perception_gain_trainer import PerceptionGainTrainer

    if variant not in VARIANTS or optimizer_update not in CAL_UPDATES:
        raise PerceptionEvaluationError("LOCAL-T2 checkpoint request is invalid")
    assets = _resolve_cache_assets(assets_path)
    device = _validate_cuda_device(device_name)
    run_dir = (
        external_root
        / "evaluation/local-t2"
        / variant
        / f"update={optimizer_update:04d}"
    )
    config = compose_variant_config(
        variant,
        pretrained=Path(assets["concerto_pretrained"]),
        run_dir=run_dir,
    )
    with open_dict(config):
        config.general.train_mode = False
        config.general.gpus = 1
        config.data.batch_size = 1
        config.data.test_batch_size = 1
        config.data.num_workers = 4
        config.trainer.precision = "32-true"
        config.model.return_query_features = False
        if variant == "Q-SEM" and scorer_checkpoint is not None:
            config.perception_training.scorer_checkpoint = str(scorer_checkpoint)
    system = PerceptionGainTrainer(config)
    load_audit, checkpoint_sha, checkpoint_reference, _, weight_sources = (
        _load_evaluation_weights(
            system=system,
            variant=variant,
            optimizer_update=optimizer_update,
            r1_checkpoint=Path(assets["r1_checkpoint"]),
            checkpoint=checkpoint,
            scorer_checkpoint=scorer_checkpoint,
        )
    )
    if checkpoint is not None:
        checkpoint_reference = _external_reference(
            checkpoint, external_root=external_root
        )
    validation_dataset = _rio_population_base(
        config,
        data_root=Path(assets["data_root"]),
        horizon=2,
        population_id=NATIVE_POPULATION_ID,
    )
    if len(validation_dataset) != LOCAL_T2_SEQUENCE_COUNT:
        raise PerceptionEvaluationError("LOCAL-T2 validation sequence count differs")
    sequence_names = tuple(str(value) for value in validation_dataset.sequence_names)
    if len(sequence_names) != LOCAL_T2_SEQUENCE_COUNT:
        raise PerceptionEvaluationError("LOCAL-T2 population identity differs")
    system.validation_dataset = validation_dataset
    system.labels_info = validation_dataset.label_info
    system.requires_grad_(False).eval()
    seed_everything(45, workers=True)
    trainer = Trainer(
        accelerator="gpu",
        devices=[device.index],
        logger=False,
        callbacks=[],
        enable_checkpointing=False,
        enable_model_summary=False,
        default_root_dir=run_dir,
        deterministic=bool(config.trainer.deterministic),
        precision="32-true",
    )
    started = time.perf_counter()
    results = trainer.validate(
        system,
        dataloaders=system.val_dataloader(),
        verbose=False,
    )
    elapsed = time.perf_counter() - started
    if not isinstance(results, list) or len(results) != 1:
        raise PerceptionEvaluationError("LOCAL-T2 evaluator returned invalid results")
    result = build_local_t2_summary(
        variant=variant,
        optimizer_update=optimizer_update,
        checkpoint_reference=checkpoint_reference,
        checkpoint_sha256=checkpoint_sha,
        config_sha256=canonical_json_sha256(
            OmegaConf.to_container(config, resolve=True)
        ),
        population_sha256=canonical_json_sha256(
            {
                "population_id": "official_like_rio_validation_t2_154",
                "sequence_names": sequence_names,
            }
        ),
        metrics=normalize_metrics(results[0]),
        elapsed_seconds=elapsed,
        gpu_name=torch.cuda.get_device_name(device),
    )
    result["load_audit"] = load_audit
    result["weight_sources"] = weight_sources
    result["source_sha256"] = _sha256(Path(__file__))
    if output_path is None:
        output_path = (
            ARTIFACT_ROOT
            / "confirmation/local-t2"
            / variant
            / f"update={optimizer_update:04d}.json"
        )
    _atomic_json(output_path, result)
    del system
    torch.cuda.empty_cache()
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--update", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--scorer-checkpoint", type=Path)
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--external-root", type=Path, default=DEFAULT_EXTERNAL_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    result = run_local_t2_evaluation(
        variant=arguments.variant,
        optimizer_update=arguments.update,
        checkpoint=arguments.checkpoint,
        scorer_checkpoint=arguments.scorer_checkpoint,
        assets_path=arguments.assets,
        external_root=arguments.external_root,
        output_path=arguments.output,
        device_name=arguments.device,
    )
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_local_t2_summary", "main", "run_local_t2_evaluation"]
