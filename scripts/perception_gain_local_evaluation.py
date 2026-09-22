#!/usr/bin/env python3
"""Evaluate Perception Gain checkpoints on the frozen official-like RIO T2 split."""

from __future__ import annotations

import argparse
import math
import time
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from pathlib import Path

from scripts.perception_gain_evaluation import (
    CAL_UPDATES,
    DEFAULT_ASSETS,
    DEFAULT_EXTERNAL_ROOT,
    VARIANTS,
    PerceptionEvaluationError,
    _atomic_json,
    _external_reference,
    _load_json,
    _load_evaluation_weights,
    _sha256,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts/perception_gain_v1"
LOCAL_T2_SEQUENCE_COUNT = 154
LOCAL_EVALUATION_PROVENANCE = (
    PROJECT_ROOT
    / "artifacts/rescene_task_learning_root_cause_v1/full_candidate/FULL_EVALUATION_PROVENANCE.json"
)


def local_evaluation_seeds() -> tuple[int, ...]:
    sources = _load_json(LOCAL_EVALUATION_PROVENANCE)["run_sources"]
    return tuple(
        sorted(int(name.removeprefix("seed").removesuffix(".json")) for name in sources)
    )


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
    eval_seed: int = 45,
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
        or eval_seed not in local_evaluation_seeds()
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
        "eval_seed": eval_seed,
        "output_policy": "native",
    }


def audit_local_population(*, artifacts: Path, external_root: Path) -> dict:
    """Bind original native T2 sequences independently of legacy role metadata."""
    from datasets.task_memory_episode import _scan_scene
    from scripts.evaluate_task_memory import NATIVE_POPULATION_ID, _rio_population_base
    from scripts.perception_gain_v2 import read_json, write_json
    from scripts.preflight_task_memory_episode import load_reference_by_scene
    from scripts.task_memory_contracts import canonical_json_sha256
    from scripts.train_perception_gain import compose_variant_config

    assets = read_json(external_root / "assets.local.json")
    config = compose_variant_config(
        "C0", pretrained=Path(assets["concerto_pretrained"]),
        run_dir=external_root / "evaluation/local-population-audit",
        recipe_config=read_json(artifacts / "training/C0-L-s45/recipe.json"),
    )
    dataset = _rio_population_base(config, data_root=Path(assets["data_root"]),
                                   horizon=2, population_id=NATIVE_POPULATION_ID)
    names = tuple(str(value) for value in dataset.sequence_names)
    reference_by_scene = load_reference_by_scene(Path(assets["rio_metadata"]))
    references = {reference_by_scene[_scan_scene(scan)] for sequence in names for scan in sequence.split("-")}
    roles = read_json(artifacts / "DATA_ROLES.json")["roles"]
    if (len(names) != len(set(names)) or len(names) != LOCAL_T2_SEQUENCE_COUNT
        or references & set(roles["TRAIN"])
        or not references.issubset(set(roles["PB"]) | set(roles["ADDITIONAL"]))):
        raise PerceptionEvaluationError("Original LOCAL-T2 population differs or overlaps TRAIN")
    declared = set(roles["LOCAL-T2"])
    result = {
        "status": "PASS", "scope": "Metadata only; no model prediction or scoring",
        "validation_sequence_count": len(names), "validation_reference_count": len(references),
        "validation_reference_ids": sorted(references), "sequence_names": names,
        "population_manifest_sha256": canonical_json_sha256({
            "population_id": "official_like_rio_validation_t2_154", "sequence_names": names,
        }),
        "declared_reference_count": len(declared), "declared_reference_ids": sorted(declared),
        "extra_references_vs_legacy": sorted(references - declared),
        "absent_references_vs_legacy": sorted(declared - references),
        "overlap_counts": {role: len(references & set(values)) for role, values in roles.items()},
        "metadata_correction": references != declared,
        "decision": "Preserve every original native T2 sequence; report actual references. Legacy V1 bootstrap passed ADDITIONAL references as LOCAL-T2 without reading the native T2 population.",
        "legacy_binding_source": "scripts/perception_gain_campaign.py::bootstrap/build_data_roles(local_t2_reference_ids=additional)",
        "legacy_binding_source_sha256": _sha256(PROJECT_ROOT / "scripts/perception_gain_campaign.py"),
        "rio_metadata_sha256": _sha256(Path(assets["rio_metadata"])),
        "data_roles_sha256": _sha256(artifacts / "DATA_ROLES.json"),
        "input_manifest_sha256": _sha256(artifacts / "data/STAGING_MANIFEST.json"),
        "original_evaluation_manifest_sha256": _sha256(LOCAL_EVALUATION_PROVENANCE),
    }
    path = artifacts / "foundation/LOCAL_POPULATION_AUDIT.json"
    if path.exists() and canonical_json_sha256(read_json(path)) != canonical_json_sha256(result):
        raise PerceptionEvaluationError("Audited original LOCAL population changed")
    write_json(path, result)
    return result


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
    recipe_config: Mapping[str, object] | None = None,
    artifact_root: Path | None = None,
    eval_seed: int = 45,
    prediction_callback=None,
) -> dict[str, object]:
    import torch
    from omegaconf import OmegaConf, open_dict
    from pytorch_lightning import Trainer, seed_everything

    from scripts.evaluate_persist4d import _validate_cuda_device
    from scripts.evaluate_sonata_second_checkpoint import normalize_metrics
    from scripts.evaluate_task_memory import NATIVE_POPULATION_ID, _rio_population_base
    from scripts.perception_gain_foundation import (
        _resolve_cache_assets,
        resolve_live_assets,
    )
    from scripts.task_memory_contracts import canonical_json_sha256
    from scripts.train_perception_gain import compose_variant_config
    from trainer.perception_gain_trainer import PerceptionGainTrainer

    code = None
    if recipe_config is not None:
        from scripts.perception_gain_v2_config import live_execution_provenance

        code = live_execution_provenance(recipe_config)
    if (
        variant not in VARIANTS
        or optimizer_update not in CAL_UPDATES
        or eval_seed not in local_evaluation_seeds()
    ):
        raise PerceptionEvaluationError("LOCAL-T2 checkpoint request is invalid")
    assets = (
        resolve_live_assets(assets_path)
        if recipe_config is not None
        else _resolve_cache_assets(assets_path)
    )
    path_variant = (
        str(recipe_config["recipe_id"]) if recipe_config is not None else variant
    )
    device = _validate_cuda_device(device_name)
    run_dir = (
        external_root
        / "evaluation/local-t2"
        / path_variant
        / f"update={optimizer_update:04d}"
    )
    if recipe_config is not None or eval_seed != 45:
        run_dir = run_dir / f"seed={eval_seed}"
    config = compose_variant_config(
        variant,
        pretrained=Path(assets["concerto_pretrained"]),
        run_dir=run_dir,
        recipe_config=recipe_config,
    )
    with open_dict(config):
        config.general.train_mode = False
        config.general.seed = eval_seed
        config.general.gpus = 1
        config.data.batch_size = 1
        config.data.test_batch_size = 1
        config.data.num_workers = 2 if recipe_config is not None else 4
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
    audit_population = recipe_config is not None or prediction_callback is not None
    references = set()
    if audit_population:
        from datasets.task_memory_episode import _scan_scene
        from scripts.preflight_task_memory_episode import load_reference_by_scene

        reference_by_scene = load_reference_by_scene(Path(assets["rio_metadata"]))
        references = {
            reference_by_scene[_scan_scene(scan)]
            for sequence in sequence_names
            for scan in sequence.split("-")
        }
    observed_sequences = set()
    process_predictions = system._process_predictions

    def export_processed_predictions(**kwargs):
        predictions = process_predictions(**kwargs)
        for prediction, sequence, target in zip(
            predictions, kwargs["file_names"], kwargs["target_full_res"], strict=True
        ):
            sequence = str(sequence)
            if sequence not in sequence_names or sequence in observed_sequences:
                raise PerceptionEvaluationError("LOCAL-T2 emitted an unexpected or repeated sequence")
            observed_sequences.add(sequence)
            if prediction_callback is not None:
                from scripts.system_comparison_inference import (
                    normalize_temporal_stages,
                )

                stages = normalize_temporal_stages(
                    target["temporal_stages"], name="LOCAL-T2 temporal stages"
                )
                counts = [int((stages == stage).sum()) for stage in (0, 1)]
                if not torch.equal(stages, torch.repeat_interleave(torch.arange(2), torch.tensor(counts))):
                    raise PerceptionEvaluationError("LOCAL-T2 prediction point order differs")
                prediction_callback(
                    method="LOCAL", logical_unit_id=sequence, horizon=2,
                    scan_ids=sequence.split("-"), prediction=prediction,
                    scan_vertex_offsets=[0, counts[0], sum(counts)],
                )
        return predictions

    if audit_population:
        system._process_predictions = export_processed_predictions
    system.validation_dataset = validation_dataset
    system.labels_info = validation_dataset.label_info
    system.requires_grad_(False).eval()
    seed_everything(eval_seed, workers=True)
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
    from scripts.system_comparison_inference import deterministic_inference_runtime

    with (deterministic_inference_runtime(eval_seed, device) if recipe_config is not None else nullcontext()):
        results = trainer.validate(
            system,
            dataloaders=system.val_dataloader(),
            verbose=False,
        )
    elapsed = time.perf_counter() - started
    if not isinstance(results, list) or len(results) != 1:
        raise PerceptionEvaluationError("LOCAL-T2 evaluator returned invalid results")
    if audit_population and observed_sequences != set(sequence_names):
        raise PerceptionEvaluationError("LOCAL-T2 did not score its complete sequence population")
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
        eval_seed=eval_seed,
    )
    result["load_audit"] = load_audit
    if audit_population:
        result["validation_reference_count"] = len(references)
        result["completed_sequences"] = sorted(observed_sequences)
        result["validation_reference_ids"] = sorted(references)
    result["weight_sources"] = weight_sources
    result["source_sha256"] = _sha256(Path(__file__))
    if recipe_config is not None:
        result["recipe"] = dict(recipe_config)
        result["execution_provenance"] = code
        result["original_evaluation_manifest_sha256"] = _sha256(
            LOCAL_EVALUATION_PROVENANCE
        )
    if output_path is None:
        output_root = artifact_root if artifact_root is not None else ARTIFACT_ROOT
        output_path = (
            output_root
            / "confirmation/local-t2"
            / path_variant
            / f"update={optimizer_update:04d}.json"
        )
        if recipe_config is not None or eval_seed != 45:
            output_path = output_path.parent / f"seed={eval_seed}" / output_path.name
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
    parser.add_argument("--recipe-config", type=Path)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--eval-seed", type=int, choices=local_evaluation_seeds(), default=45
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    recipe = (
        _load_json(arguments.recipe_config)
        if arguments.recipe_config is not None
        else None
    )
    result = run_local_t2_evaluation(
        variant=arguments.variant,
        optimizer_update=arguments.update,
        checkpoint=arguments.checkpoint,
        scorer_checkpoint=arguments.scorer_checkpoint,
        assets_path=arguments.assets,
        external_root=arguments.external_root,
        output_path=arguments.output,
        device_name=arguments.device,
        recipe_config=recipe,
        artifact_root=arguments.artifact_root,
        eval_seed=arguments.eval_seed,
    )
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_local_t2_summary", "main", "run_local_t2_evaluation"]
