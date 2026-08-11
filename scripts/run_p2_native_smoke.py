#!/usr/bin/env python3
"""Run P2 single-GPU native training checks on one real 3RScan T=2 window."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "P2"
DEFAULT_CHECKPOINT = (
    Path.home() / ".cache" / "persist4d" / "concerto" / "concerto_base.pth"
)
CONCERTO_CHECKPOINT_SHA256 = (
    "845ec7dec97a5fabff8fadb5d9858ac6734347b612d1a4b574213419c139de07"
)
CONCERTO_CHECKPOINT_REFERENCE = "local_cache:persist4d/concerto/concerto_base.pth"
SOURCE_TREE_ARTIFACT_PREFIX = "artifacts/P2/"
SEED = 45
TINY_SAMPLE_NAME = "scene0112_00-scene0112_01"
TINY_OVERFIT_STEPS = 128
REQUIRED_TMAP_KEYS = (
    "val_mean_t-AP",
    "val_mean_t-AP_50",
    "val_mean_t-AP_25",
    "val_mean_AP",
    "val_mean_stage1-AP",
    "val_mean_stage2-AP",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_provenance(path: Path, *, sha256: str | None = None) -> dict[str, str]:
    checksum = sha256 if sha256 is not None else sha256_file(path)
    if checksum != CONCERTO_CHECKPOINT_SHA256:
        raise ValueError(
            f"Concerto checkpoint SHA256 mismatch: expected "
            f"{CONCERTO_CHECKPOINT_SHA256}, got {checksum}"
        )
    return {
        "reference": CONCERTO_CHECKPOINT_REFERENCE,
        "sha256": checksum,
    }


def _project_file_provenance(path: str | Path) -> dict[str, str]:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    resolved = candidate.resolve()
    try:
        relative = resolved.relative_to(PROJECT_ROOT)
    except ValueError as error:
        raise ValueError(f"input file is outside the repository: {path}") from error
    if not resolved.is_file():
        raise FileNotFoundError(f"required input file does not exist: {relative}")
    return {
        "reference": f"repo:{relative.as_posix()}",
        "sha256": sha256_file(resolved),
    }


def _portable_config_value(value: Any, checkpoint: Path) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _portable_config_value(item, checkpoint)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_portable_config_value(item, checkpoint) for item in value]
    if not isinstance(value, str) or not Path(value).is_absolute():
        return value

    resolved = Path(value).resolve()
    if resolved == checkpoint.resolve():
        return CONCERTO_CHECKPOINT_REFERENCE
    try:
        relative = resolved.relative_to(PROJECT_ROOT)
    except ValueError as error:
        raise ValueError(
            f"resolved config contains a non-portable absolute path: {value}"
        ) from error
    return f"repo:{relative.as_posix()}"


def _resolved_config_provenance(config: Any, checkpoint: Path) -> dict[str, Any]:
    from omegaconf import OmegaConf

    resolved = OmegaConf.to_container(config, resolve=True)
    portable = _portable_config_value(resolved, checkpoint)
    serialized = json.dumps(
        portable,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return {
        "format": "canonical-json-sort-keys-v1",
        "portable_references": True,
        "serialized_bytes": len(serialized),
        "sha256": hashlib.sha256(serialized).hexdigest(),
    }


def _input_provenance(
    config: Any,
    dataset: Any,
    dataset_index: int,
    checkpoint: Path,
) -> dict[str, Any]:
    if dataset.dataset_name != "rio" or len(dataset.data_dir) != 1:
        raise AssertionError("P2 preflight input provenance requires RIO-only data")
    if dataset.sequence_names[dataset_index] != TINY_SAMPLE_NAME:
        raise AssertionError("input provenance resolved an unexpected temporal sample")

    scan_indices = [int(index) for index in dataset.sequence_indices[dataset_index]]
    records = [dataset.data[index] for index in scan_indices]
    data_dir = Path(dataset.data_dir[0])
    train_config = config.data.train_dataset
    change_ground_truth = dataset.change_files[dataset_index]
    if change_ground_truth is None:
        raise AssertionError("T=2 training sample has no change ground truth")

    return {
        "dataset": "3RScan",
        "sample_name": dataset.sequence_names[dataset_index],
        "temporal_window": int(dataset.temporal_window),
        "processed_point_clouds": [
            _project_file_provenance(record["filepath"]) for record in records
        ],
        "instance_ground_truth": [
            _project_file_provenance(record["instance_gt_filepath"])
            for record in records
        ],
        "change_ground_truth": _project_file_provenance(change_ground_truth),
        "sequence_database": _project_file_provenance(
            data_dir / f"sequence_database_sliding_{dataset.temporal_window}.yaml"
        ),
        "split_database": _project_file_provenance(
            data_dir / f"{dataset.mode}_database.yaml"
        ),
        "semantic_label_database": _project_file_provenance(
            train_config.label_db_filepath
        ),
        "change_label_database": _project_file_provenance(
            train_config.change_label_db_filepath
        ),
        "color_statistics": _project_file_provenance(train_config.color_mean_std),
        "train_augmentations": {
            "image": _project_file_provenance(train_config.image_augmentations_path),
            "volume": _project_file_provenance(train_config.volume_augmentations_path),
        },
        "resolved_composed_config": _resolved_config_provenance(config, checkpoint),
    }


def seed_everything(seed: int = SEED) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _is_contrastive_diagnostic(name: str) -> bool:
    return name.startswith("loss_") and "_contrastive_layer" in name


def objective_breakdown(losses: Mapping[str, Any], weight_dict: Mapping[str, float]):
    from trainer.trainer import aggregate_objective_loss

    objective = aggregate_objective_loss(losses, weight_dict)
    final_keys = ("loss_ce", "loss_mask", "loss_dice")
    final_segmentation = sum(
        losses[key] * weight_dict[key] for key in final_keys if key in losses
    )
    segmentation_terms = [
        value * weight_dict[key] for key, value in losses.items() if key in weight_dict
    ]
    all_segmentation = sum(segmentation_terms)
    aggregate_contrastive = sum(
        losses[key]
        for key in ("loss_segment_contrastive", "loss_aux_contrastive")
        if key in losses
    )
    return {
        "objective": objective,
        "final_head_segmentation": final_segmentation,
        "all_segmentation": all_segmentation,
        "aggregate_contrastive": aggregate_contrastive,
        "diagnostic_keys": sorted(
            key for key in losses if _is_contrastive_diagnostic(key)
        ),
    }


def classify_parameters(
    named_parameters: Iterable[tuple[str, Any]],
) -> dict[str, list[str]]:
    groups = {
        "frozen_encoder": [],
        "trainable_concerto_decoder": [],
        "trainable_rescene_decoder": [],
        "trainable_rescene_heads": [],
        "trainable_objective": [],
    }
    head_prefixes = (
        "model.mask_features_head.",
        "model.query_projection.",
        "model.np_feature_projection.",
        "model.mask_embed_head.",
        "model.class_embed_head.",
        "model.change_embed_head.",
    )
    for name, parameter in named_parameters:
        is_encoder = (
            ".backbone.model.enc." in name or ".backbone.model.embedding." in name
        )
        is_decoder = ".backbone.model.dec." in name
        if is_encoder:
            groups["frozen_encoder"].append(name)
        elif is_decoder and parameter.requires_grad:
            groups["trainable_concerto_decoder"].append(name)
        elif parameter.requires_grad and name.startswith(head_prefixes):
            groups["trainable_rescene_heads"].append(name)
        elif parameter.requires_grad and name.startswith("model."):
            groups["trainable_rescene_decoder"].append(name)
        elif parameter.requires_grad:
            groups["trainable_objective"].append(name)
    return groups


def validate_tmap_schema(keys: Iterable[str]) -> list[str]:
    key_set = set(keys)
    missing = sorted(set(REQUIRED_TMAP_KEYS) - key_set)
    if missing:
        raise ValueError("missing required t-mAP schema keys: " + ", ".join(missing))
    return sorted(set(REQUIRED_TMAP_KEYS))


def evaluate_tiny_overfit_gates(
    history: Sequence[Mapping[str, float]],
) -> dict[str, Any]:
    if len(history) != TINY_OVERFIT_STEPS:
        raise ValueError(
            f"tiny overfit requires {TINY_OVERFIT_STEPS} optimizer steps, "
            f"got {len(history)}"
        )
    first_seg = statistics.median(
        float(row["final_head_segmentation"]) for row in history[:10]
    )
    last_seg = statistics.median(
        float(row["final_head_segmentation"]) for row in history[-10:]
    )
    first_contrastive = float(history[0]["aggregate_contrastive"])
    final_contrastive = float(history[-1]["aggregate_contrastive"])
    all_contrastive = [float(row["aggregate_contrastive"]) for row in history]
    segmentation_ratio = last_seg / first_seg if first_seg > 0 else math.inf
    contrastive_ratio = (
        final_contrastive / first_contrastive if first_contrastive > 0 else math.inf
    )
    gates = {
        "final_segmentation_median_ratio_le_0.25": segmentation_ratio <= 0.25,
        "final_contrastive_ratio_le_0.50": contrastive_ratio <= 0.50,
        "contrastive_positive_and_finite": all(
            value > 0 and math.isfinite(value) for value in all_contrastive
        ),
        "classification_accuracy_ge_0.75": (
            float(history[-1]["classification_accuracy"]) >= 0.75
        ),
        "mean_dice_ge_0.90": float(history[-1]["mean_dice"]) >= 0.90,
    }
    return {
        "steps": len(history),
        "initial_10_final_segmentation_median": first_seg,
        "final_10_final_segmentation_median": last_seg,
        "final_segmentation_median_ratio": segmentation_ratio,
        "initial_aggregate_contrastive": first_contrastive,
        "final_aggregate_contrastive": final_contrastive,
        "final_contrastive_ratio": contrastive_ratio,
        "final_classification_accuracy": float(history[-1]["classification_accuracy"]),
        "final_mean_dice": float(history[-1]["mean_dice"]),
        "gates": gates,
        "passed": all(gates.values()),
    }


def render_tiny_overfit_markdown(
    result: Mapping[str, Any],
    *,
    sample_name: str,
    elapsed_seconds: float,
    peak_vram_mib: float,
) -> str:
    status = "PASS" if result["passed"] else "FAIL"
    gate_lines = [
        f"| `{name}` | {'PASS' if passed else 'FAIL'} |"
        for name, passed in result["gates"].items()
    ]
    return "\n".join(
        [
            "# P2 Preflight-only Tiny Overfit",
            "",
            "This is not an official mixed-data reproduction and is not G2 evidence.",
            "",
            f"- Status: **{status}**",
            f"- Sample: `{sample_name}`",
            f"- Optimizer steps: {result['steps']}",
            f"- Elapsed: {elapsed_seconds:.3f} s",
            f"- Peak allocated VRAM: {peak_vram_mib:.1f} MiB",
            (
                f"- Final-head segmentation median ratio: "
                f"{result['final_segmentation_median_ratio']:.6f}"
            ),
            (
                f"- Aggregate contrastive final/initial ratio: "
                f"{result['final_contrastive_ratio']:.6f}"
            ),
            (
                f"- Final matcher classification accuracy: "
                f"{result['final_classification_accuracy']:.6f}"
            ),
            f"- Final mean soft Dice: {result['final_mean_dice']:.6f}",
            "",
            "| Gate | Result |",
            "|---|---|",
            *gate_lines,
            "",
        ]
    )


def _write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    _assert_artifact_privacy(serialized)
    path.write_text(serialized, encoding="utf-8")


def _write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _assert_artifact_privacy(text)
    path.write_text(text, encoding="utf-8")


def _assert_artifact_privacy(text: str) -> None:
    forbidden = (
        "/" + "home" + "/",
        "/" + "Users" + "/",
        "GPU" + "-",
        "ssh" + "://",
    )
    found = [token for token in forbidden if token in text]
    if found:
        raise ValueError(
            "artifact contains private runtime identifiers: " + ", ".join(found)
        )


def _tensor_digest(named_parameters: Iterable[tuple[str, Any]]) -> str:
    import torch

    digest = hashlib.sha256()
    for name, parameter in sorted(named_parameters, key=lambda item: item[0]):
        tensor = parameter.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _named_subset(system: Any, names: Sequence[str]) -> list[tuple[str, Any]]:
    parameters = dict(system.named_parameters())
    return [(name, parameters[name]) for name in names]


def _gradient_summary(system: Any, names: Sequence[str]) -> dict[str, Any]:
    parameters = dict(system.named_parameters())
    norms = []
    finite = True
    missing = 0
    for name in names:
        gradient = parameters[name].grad
        if gradient is None:
            missing += 1
            continue
        finite = finite and bool(gradient.isfinite().all().item())
        norms.append(float(gradient.detach().float().norm().cpu()))
    return {
        "parameter_tensors": len(names),
        "missing_grad_tensors": missing,
        "finite": finite,
        "nonzero_grad_tensors": sum(value > 0 for value in norms),
        "max_grad_norm": max(norms, default=0.0),
    }


def _recursive_equal(left: Any, right: Any) -> bool:
    import torch

    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return left.shape == right.shape and torch.equal(left.cpu(), right.cpu())
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            _recursive_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            _recursive_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def _dependency_provenance() -> dict[str, str]:
    import concerto
    import detectron2
    import flash_attn
    import sonata
    import stmetrics

    modules = {
        "Concerto": concerto,
        "Detectron2": detectron2,
        "flash-attn": flash_attn,
        "Sonata": sonata,
        "stmetrics": stmetrics,
    }
    result = {}
    for label, module in modules.items():
        module_file = Path(module.__file__).resolve()
        if "tests" in module_file.parts:
            raise RuntimeError(f"{label} resolved to a test double")
        result[label] = str(getattr(module, "__version__", "source-install"))
    return result


def _compose_config(checkpoint: Path):
    from hydra import compose, initialize_config_dir
    from omegaconf import open_dict

    with initialize_config_dir(
        version_base="1.2", config_dir=str(PROJECT_ROOT / "conf")
    ):
        config = compose(
            config_name="config_p2_rescene4d_concerto_t2",
            overrides=[
                "data/datasets=rio",
                "~data.train_dataset.epoch_sample_multiple",
                "~data.train_dataset.sampler_seed",
                "++data.train_dataset.known_empty_scan_policy=official_substitute",
            ],
        )
    processed_dir = PROJECT_ROOT / "data" / "processed" / "rio"
    with open_dict(config):
        config.general.gpus = 1
        config.general.compile_model = False
        config.general.p2_fail_closed_runtime = False
        config.data.batch_size = 1
        config.data.num_workers = 0
        config.backbone.name = str(checkpoint.resolve())
        for split in ("train_dataset", "validation_dataset", "test_dataset"):
            dataset_config = config.data[split]
            dataset_config.data_dir = str(processed_dir)
            dataset_config.label_db_filepath = str(
                processed_dir / "label_database.yaml"
            )
            dataset_config.change_label_db_filepath = str(
                processed_dir / "change_label_database.yaml"
            )
            dataset_config.color_mean_std = str(processed_dir / "color_mean_std.yaml")
            dataset_config.temporal_window = 2
            dataset_config.known_empty_scan_policy = "official_substitute"
    if not config.loss.contrastive_loss:
        raise RuntimeError("P2 runtime composed with contrastive loss disabled")
    return config


def _configure_lightning_weighted_sampler(config: Any) -> Any:
    from omegaconf import OmegaConf, open_dict

    # Resolve collations while they still point at the original train dataset.
    OmegaConf.resolve(config)
    rio_dataset = OmegaConf.to_container(
        config.data.train_dataset,
        resolve=False,
    )
    rio_dataset["target"] = rio_dataset.pop("_target_")
    with open_dict(config):
        config.data.train_dataset = OmegaConf.create(
            {
                "_target_": "datasets.multi_dataset.MultiDataset.from_config",
                "datasets": [rio_dataset],
                "weights": [1.0],
                "epoch_sample_multiple": 1,
                "sampler_seed": SEED,
                "fail_closed": True,
            }
        )
        config.general.p2_fail_closed_runtime = True
    return config


def _compose_runtime(
    checkpoint: Path,
    device: Any,
    *,
    save_dir: Path | None = None,
    scheduler_total_steps: int | None = None,
    lightning_weighted_sampler: bool = False,
):
    from omegaconf import open_dict

    seed_everything(SEED)
    config = _compose_config(checkpoint)
    with open_dict(config):
        if save_dir is not None:
            config.general.save_dir = str(save_dir.resolve())
        if scheduler_total_steps is not None:
            config.scheduler.scheduler.total_steps = scheduler_total_steps
    if lightning_weighted_sampler:
        _configure_lightning_weighted_sampler(config)

    from trainer.trainer import InstanceSegmentation

    system = InstanceSegmentation(config).to(device)
    if not system.model.backbone.model.__class__.__module__.startswith("concerto"):
        raise RuntimeError(
            "runtime backbone is not the installed Concerto implementation"
        )
    return config, system


def _materialize_named_train_batch(config: Any, sample_name: str, device: Any):
    import hydra
    import numpy as np

    from scripts.profile_temporal_scaling import (
        move_data_to_device,
        move_targets_to_device,
    )

    dataset = hydra.utils.instantiate(config.data.train_dataset)
    try:
        dataset_index = dataset.sequence_names.index(sample_name)
    except ValueError as error:
        raise ValueError(
            f"required T=2 train sample is absent: {sample_name}"
        ) from error
    provenance = _input_provenance(
        config,
        dataset,
        dataset_index,
        Path(config.backbone.name),
    )
    collate = hydra.utils.instantiate(config.data.train_collation)
    seed_everything(SEED)
    sample = dataset[dataset_index]
    stages = sorted(int(value) for value in np.unique(sample[0][:, 3]).tolist())
    if stages != [0, 1]:
        raise ValueError(f"sample {sample_name} has temporal stages {stages}")
    raw_points = int(sample[0].shape[0])
    seed_everything(SEED)
    data, targets, names = collate([sample])
    if list(names) != [sample_name] or not targets:
        raise ValueError(
            "native collator did not preserve the requested supervised sample"
        )
    voxels = int(data.features.shape[0])
    instances = int(targets[0]["labels"].numel())
    data = move_data_to_device(data, device)
    targets = move_targets_to_device(targets, device)
    return (
        data,
        targets,
        {
            "name": sample_name,
            "raw_points": raw_points,
            "voxels": voxels,
            "supervised_instances": instances,
            "temporal_stages": stages,
        },
        provenance,
    )


def _materialize_smallest_validation_batch(config: Any, device: Any):
    import hydra

    from scripts.profile_temporal_scaling import (
        move_data_to_device,
        move_targets_to_device,
    )

    dataset = hydra.utils.instantiate(config.data.validation_dataset)
    candidates = []
    for index, scan_indices in enumerate(dataset.sequence_indices):
        raw_points = sum(int(dataset.data[int(i)]["file_len"]) for i in scan_indices)
        candidates.append((raw_points, dataset.sequence_names[index], index))
    raw_points, sample_name, index = min(candidates)
    collate = hydra.utils.instantiate(config.data.validation_collation)
    seed_everything(SEED)
    sample = dataset[index]
    seed_everything(SEED)
    data, targets, names = collate([sample])
    if not targets or not data.target_full:
        raise ValueError("validation collator returned no supervised targets")
    voxels = int(data.features.shape[0])
    data = move_data_to_device(data, device)
    targets = move_targets_to_device(targets, device)
    return (
        dataset,
        data,
        targets,
        list(names),
        {
            "name": sample_name,
            "raw_points": int(raw_points),
            "voxels": voxels,
        },
    )


def _make_optimizer_scheduler(
    config: Any, system: Any, *, total_steps: int
) -> tuple[Any, Any]:
    import hydra
    from omegaconf import OmegaConf

    optimizer = hydra.utils.instantiate(config.optimizer, params=system.parameters())
    scheduler_config = OmegaConf.create(
        OmegaConf.to_container(config.scheduler.scheduler, resolve=True)
    )
    scheduler_config.total_steps = total_steps
    scheduler = hydra.utils.instantiate(scheduler_config, optimizer=optimizer)
    return optimizer, scheduler


def _forward_losses(system: Any, data: Any, targets: Sequence[Mapping[str, Any]]):
    output = system.forward(
        data,
        point2segment=[target["point2segment"] for target in targets],
        raw_coordinates=system._process_raw_coordinates(data),
        is_eval=False,
        targets=targets,
    )
    losses = system.criterion(output, targets, mask_type=system.mask_type)
    breakdown = objective_breakdown(losses, system.criterion.weight_dict)
    return output, losses, breakdown


def _assert_loss_contract(
    losses: Mapping[str, Any], breakdown: Mapping[str, Any]
) -> None:
    import torch

    required = {
        "loss_ce",
        "loss_mask",
        "loss_dice",
        "loss_segment_contrastive",
        "loss_aux_contrastive",
    }
    missing = sorted(required - set(losses))
    if missing:
        raise AssertionError("missing native loss terms: " + ", ".join(missing))
    segmentation = [
        value
        for key, value in losses.items()
        if key.startswith(("loss_ce", "loss_mask", "loss_dice"))
    ]
    aggregate = [
        losses["loss_segment_contrastive"],
        losses["loss_aux_contrastive"],
    ]
    if not all(
        bool(torch.isfinite(value).all().item()) for value in segmentation + aggregate
    ):
        raise AssertionError("native segmentation/contrastive losses must be finite")
    if float(losses["loss_segment_contrastive"].detach().cpu()) <= 1e-8:
        raise AssertionError("native segment contrastive loss must be positive")
    if not bool(torch.isfinite(breakdown["objective"]).all().item()):
        raise AssertionError("native weighted objective must be finite")


def _matching_quality(
    system: Any, output: Mapping[str, Any], targets: Sequence[Mapping[str, Any]]
):

    outputs_without_aux = {
        key: value for key, value in output.items() if key != "aux_outputs"
    }
    indices = system.criterion.matcher(
        outputs_without_aux, targets, mask_type=system.mask_type
    )
    correct = 0
    total = 0
    dice_values = []
    for batch_id, (source, target_index) in enumerate(indices):
        if len(source) == 0:
            continue
        source = source.to(output["pred_logits"].device)
        target_index = target_index.to(targets[batch_id]["labels"].device)
        predicted = output["pred_logits"][batch_id, source, :-1].argmax(dim=-1)
        expected = targets[batch_id]["labels"][target_index]
        correct += int((predicted == expected).sum().item())
        total += int(source.numel())
        prediction_masks = output["pred_masks"][batch_id][:, source].T.sigmoid()
        target_masks = targets[batch_id][system.mask_type][target_index].float()
        numerator = 2.0 * (prediction_masks * target_masks).sum(dim=1)
        denominator = prediction_masks.sum(dim=1) + target_masks.sum(dim=1)
        dice_values.extend(
            ((numerator + 1e-6) / (denominator + 1e-6)).detach().cpu().tolist()
        )
    return {
        "classification_accuracy": correct / total if total else 0.0,
        "mean_dice": statistics.mean(dice_values) if dice_values else 0.0,
        "matched_instances": total,
    }


def _one_optimizer_step(
    system: Any, data: Any, targets: Any, optimizer: Any, scheduler: Any
):
    system.train()
    optimizer.zero_grad(set_to_none=True)
    output, losses, breakdown = _forward_losses(system, data, targets)
    _assert_loss_contract(losses, breakdown)
    quality = _matching_quality(system, output, targets)
    breakdown["objective"].backward()
    optimizer.step()
    scheduler.step()
    return output, losses, breakdown, quality


def _run_validation_evaluator(system: Any, config: Any, device: Any) -> dict[str, Any]:
    import torch

    dataset, data, targets, names, sample = _materialize_smallest_validation_batch(
        config, device
    )
    system.validation_dataset = dataset
    system.instance_metric.reset()
    system.eval()
    with torch.inference_mode():
        output = system.forward(
            data,
            point2segment=[target["point2segment"] for target in targets],
            raw_coordinates=system._process_raw_coordinates(data),
            is_eval=True,
            targets=targets,
        )
        predictions = system._process_predictions(
            output=output,
            target_low_res=targets,
            target_full_res=data.target_full,
            inverse_maps=data.inverse_maps,
            file_names=names,
            full_res_coords=data.original_coordinates,
            original_colors=data.original_colors,
            original_normals=data.original_normals,
            raw_coords=None,
            idx=data.idx,
        )
        system.instance_metric.update(predictions, data.target_full)
        result = system.instance_metric.compute()
    schema = validate_tmap_schema(result.keys())
    required_values = [result[key] for key in schema]
    finite_count = sum(
        bool(torch.isfinite(value).all().item()) for value in required_values
    )
    system.instance_metric.reset()
    return {
        "pipeline_executed": True,
        "input": "real_3rscan_validation_window_with_current_model_predictions",
        "model_state": (
            "pretrained_concerto_encoder_with_seeded_randomly_initialized_decoder_"
            "and_heads_after_two_native_smoke_steps"
        ),
        "sample": sample,
        "schema_keys": schema,
        "required_value_count": len(required_values),
        "finite_required_value_count": finite_count,
        "model_metric_values_published": False,
        "g2_metric_evidence": False,
    }


def _cpu_snapshot(value: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _cpu_snapshot(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_snapshot(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_snapshot(item) for item in value)
    return copy.deepcopy(value)


def _persistent_checkpoint_callback_states(
    states: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        state_key: {
            field: _cpu_snapshot(value)
            for field, value in state.items()
            if field != "current_score"
        }
        for state_key, state in states.items()
    }


def _instantiate_formal_checkpoint_callbacks(config: Any) -> list[Any]:
    import hydra
    from pytorch_lightning.callbacks import ModelCheckpoint

    target = "pytorch_lightning.callbacks.ModelCheckpoint"
    callback_configs = [
        item for item in config.callbacks if getattr(item, "_target_", None) == target
    ]
    if len(callback_configs) != 3:
        raise AssertionError(
            "P2 Lightning resume smoke requires exactly three formal "
            "ModelCheckpoint callbacks"
        )

    callbacks = [hydra.utils.instantiate(item) for item in callback_configs]
    if not all(type(callback) is ModelCheckpoint for callback in callbacks):
        raise AssertionError("formal checkpoint callback instantiated wrong type")
    if any(callback.save_weights_only for callback in callbacks):
        raise AssertionError("formal ModelCheckpoint callback is weights-only")
    if sum(bool(callback.save_last) for callback in callbacks) != 1:
        raise AssertionError(
            "formal ModelCheckpoint callbacks require exactly one save-last owner"
        )
    state_keys = [callback.state_key for callback in callbacks]
    if len(set(state_keys)) != len(state_keys):
        raise AssertionError("formal checkpoint callback state keys are not unique")

    expected_state_fields = {
        "monitor",
        "best_model_score",
        "best_model_path",
        "current_score",
        "dirpath",
        "best_k_models",
        "kth_best_model_path",
        "kth_value",
        "last_model_path",
    }
    for callback in callbacks:
        if set(callback.state_dict()) != expected_state_fields:
            raise AssertionError(
                "formal ModelCheckpoint state schema changed from the validated "
                "Lightning contract"
            )
    return callbacks


def _run_lightning_checkpoint_resume(
    checkpoint: Path,
    device: Any,
    data: Any,
    targets: Any,
    sample_name: str,
) -> dict[str, Any]:
    import gc
    from types import MethodType

    import torch
    from pytorch_lightning import Callback, Trainer
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

    from main_instance_segmentation import (
        _P2_FORMAL_MODEL_STATE_SCHEMA_SHA256,
        _P2_FORMAL_ONECYCLE_TOTAL_STEPS,
        _P2_FORMAL_OPTIMIZER_STEPS_PER_EPOCH,
        _P2_FORMAL_TRAIN_BATCHES_PER_EPOCH,
        _P2_FORMAL_TRAINABLE_PARAMETER_SCHEMA_SHA256,
        find_resume_checkpoint,
        require_p2_resume_checkpoint,
    )
    from utils.p2_preflight import p2_training_semantic_sha256

    if sample_name != TINY_SAMPLE_NAME:
        raise AssertionError("Lightning resume smoke requires the pinned 3RScan sample")
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()

    class ResumeStateProbe(Callback):
        def __init__(self, events: list[str], dataset: Any) -> None:
            self.snapshot = None
            self.events = events
            self.dataset = dataset

        def on_train_batch_start(
            self,
            trainer: Any,
            pl_module: Any,
            batch: Any,
            batch_idx: int,
        ) -> None:
            if self.snapshot is not None:
                return
            if len(trainer.optimizers) != 1 or len(trainer.lr_scheduler_configs) != 1:
                raise AssertionError(
                    "Lightning resume smoke requires one optimizer and one scheduler"
                )
            scheduler = trainer.lr_scheduler_configs[0].scheduler
            sampler_generator = pl_module._train_sampler_generator()
            if sampler_generator is None:
                raise AssertionError(
                    "Lightning resume probe found no train sampler generator"
                )
            if (
                getattr(pl_module, "train_dataset", None) is not self.dataset
                or sampler_generator is not self.dataset.sampler.generator
            ):
                raise AssertionError(
                    "Lightning resume probe observed the wrong sampler generator"
                )
            self.events.append("on_train_batch_start_state_observed")
            self.snapshot = {
                "global_step": int(trainer.global_step),
                "model_digest": _tensor_digest(pl_module.state_dict().items()),
                "optimizer": _cpu_snapshot(trainer.optimizers[0].state_dict()),
                "scheduler": _cpu_snapshot(scheduler.state_dict()),
                "checkpoint_callbacks": {
                    callback.state_key: _cpu_snapshot(callback.state_dict())
                    for callback in trainer.checkpoint_callbacks
                },
            }

    class PinnedBatchDataset(Dataset):
        def __init__(self, num_samples: int) -> None:
            self.batch = (data, targets, [sample_name])
            self.sampled_indices = []
            generator = torch.Generator()
            generator.manual_seed(SEED)
            self.initial_generator_state = generator.get_state().detach().cpu().clone()
            self.sampler = WeightedRandomSampler(
                weights=torch.ones(4, dtype=torch.double),
                num_samples=num_samples,
                replacement=True,
                generator=generator,
            )

        def __len__(self) -> int:
            return 4

        def __getitem__(self, index: int) -> Any:
            if index < 0 or index >= len(self):
                raise IndexError(index)
            self.sampled_indices.append(index)
            return self.batch

    def bind_pinned_dataset_on_setup(
        system: Any,
        dataset: PinnedBatchDataset,
        events: list[str],
        validation_dataset: Any | None = None,
    ) -> dict[str, Any]:
        original_setup = system.setup
        original_on_load_checkpoint = system.on_load_checkpoint
        binding = {
            "configured_sampler_present": False,
            "restored_to_actual_loader_generator": False,
        }

        def setup_with_pinned_dataset(module: Any, stage: Any = None) -> None:
            original_setup(stage)
            if stage not in ("fit", None):
                return
            configured_generator = module._train_sampler_generator()
            binding["configured_sampler_present"] = configured_generator is not None
            if not binding["configured_sampler_present"]:
                raise AssertionError(
                    "formal-like single-RIO config has no sampler generator"
                )
            events.append("setup_configured_sampler_verified")
            module.train_dataset = dataset
            if module._train_sampler_generator() is not dataset.sampler.generator:
                raise AssertionError("setup bound the wrong actual loader generator")
            events.append("setup_actual_loader_sampler_bound")
            if validation_dataset is not None:
                module.validation_dataset = validation_dataset

        def on_load_checkpoint_with_identity_check(
            module: Any,
            checkpoint_payload: Mapping[str, Any],
        ) -> None:
            if (
                getattr(module, "train_dataset", None) is not dataset
                or module._train_sampler_generator() is not dataset.sampler.generator
            ):
                raise AssertionError(
                    "checkpoint restore target is not the actual loader generator"
                )
            events.append("on_load_checkpoint_actual_loader_sampler_verified")
            original_on_load_checkpoint(checkpoint_payload)
            sampler_payload = checkpoint_payload.get(
                module._TRAIN_SAMPLER_CHECKPOINT_KEY
            )
            expected_state = (
                sampler_payload.get("generator_state")
                if isinstance(sampler_payload, Mapping)
                else None
            )
            actual_state = module._train_sampler_generator().get_state()
            if not isinstance(expected_state, torch.Tensor) or not torch.equal(
                actual_state.detach().cpu(),
                expected_state.detach().cpu(),
            ):
                raise AssertionError(
                    "checkpoint state was not restored to the actual loader generator"
                )
            binding["restored_to_actual_loader_generator"] = True
            events.append("on_load_checkpoint_actual_loader_sampler_restored")

        system.setup = MethodType(setup_with_pinned_dataset, system)
        system.on_load_checkpoint = MethodType(
            on_load_checkpoint_with_identity_check,
            system,
        )
        return binding

    def pinned_batch_loader(
        num_samples: int,
    ) -> tuple[PinnedBatchDataset, Any]:
        dataset = PinnedBatchDataset(num_samples)
        return dataset, DataLoader(
            dataset,
            batch_size=None,
            sampler=dataset.sampler,
            num_workers=0,
            collate_fn=lambda item: item,
        )

    def pinned_validation_loader(batch: Any) -> Any:
        return DataLoader(
            [None],
            batch_size=None,
            num_workers=0,
            collate_fn=lambda _: batch,
        )

    def make_trainer(
        *,
        max_epochs: int,
        max_steps: int,
        root_dir: Path,
        callbacks: Sequence[Any],
        limit_val_batches: int,
    ) -> Any:
        return Trainer(
            accelerator="gpu",
            devices=[device.index],
            max_epochs=max_epochs,
            max_steps=max_steps,
            accumulate_grad_batches=4,
            limit_train_batches=_P2_FORMAL_TRAIN_BATCHES_PER_EPOCH,
            limit_val_batches=limit_val_batches,
            num_sanity_val_steps=0,
            logger=False,
            callbacks=list(callbacks),
            default_root_dir=root_dir,
            enable_progress_bar=False,
            enable_model_summary=False,
            deterministic=False,
            use_distributed_sampler=False,
        )

    temporary_checkpoint = None
    verified_resume_checkpoint = None
    details = None
    with tempfile.TemporaryDirectory(prefix="p2-lightning-checkpoint-") as directory:
        checkpoint_dir = Path(directory)
        temporary_checkpoint = checkpoint_dir / "last.ckpt"

        source_config, source_system = _compose_runtime(
            checkpoint,
            device,
            save_dir=checkpoint_dir,
            scheduler_total_steps=_P2_FORMAL_ONECYCLE_TOTAL_STEPS,
            lightning_weighted_sampler=True,
        )
        source_checkpoint_callbacks = _instantiate_formal_checkpoint_callbacks(
            source_config
        )
        source_checkpoint_state_keys = {
            callback.state_key for callback in source_checkpoint_callbacks
        }
        monitored_checkpoint_callbacks = [
            callback
            for callback in source_checkpoint_callbacks
            if callback.monitor == "val_mean_t-AP" and callback.save_last
        ]
        if len(monitored_checkpoint_callbacks) != 1:
            raise AssertionError(
                "formal checkpoint config has no unique monitored save-last callback"
            )
        monitored_checkpoint_callback = monitored_checkpoint_callbacks[0]
        model_class = f"{type(source_system).__module__}.{type(source_system).__name__}"
        model_name = type(source_system.model).__name__
        backbone_type = type(source_system.model.backbone.model)
        backbone_class = f"{backbone_type.__module__}.{backbone_type.__name__}"
        configured_datasets = source_config.data.train_dataset.datasets
        if len(configured_datasets) != 1:
            raise AssertionError("Lightning resume smoke requires one RIO dataset")
        configured_rio = configured_datasets[0]
        configured_data_dirs = configured_rio.data_dir
        if isinstance(configured_data_dirs, str):
            configured_data_dirs = [configured_data_dirs]
        data_dir_names = {Path(path).name for path in configured_data_dirs}
        if (
            model_class != "trainer.trainer.InstanceSegmentation"
            or model_name != "ReScene"
            or not backbone_class.startswith("concerto.")
            or configured_rio.dataset_name != "rio"
            or data_dir_names != {"rio"}
            or not source_config.general.p2_fail_closed_runtime
        ):
            raise AssertionError(
                "Lightning resume smoke did not construct ReScene+Concerto on RIO"
            )
        parameter_names = tuple(name for name, _ in source_system.named_parameters())
        source_trainable_parameter_names = {
            name
            for name, parameter in source_system.named_parameters()
            if parameter.requires_grad
        }
        source_dataset, source_loader = pinned_batch_loader(
            _P2_FORMAL_TRAIN_BATCHES_PER_EPOCH
        )
        (
            source_validation_dataset,
            source_validation_data,
            source_validation_targets,
            source_validation_names,
            source_validation_sample,
        ) = _materialize_smallest_validation_batch(source_config, device)
        source_validation_loader = pinned_validation_loader(
            (
                source_validation_data,
                source_validation_targets,
                source_validation_names,
            )
        )
        source_events: list[str] = []
        source_binding = bind_pinned_dataset_on_setup(
            source_system,
            source_dataset,
            source_events,
            validation_dataset=source_validation_dataset,
        )
        source_trainer = make_trainer(
            max_epochs=1,
            max_steps=-1,
            root_dir=checkpoint_dir,
            callbacks=source_checkpoint_callbacks,
            limit_val_batches=1,
        )
        source_trainer.fit(
            source_system,
            train_dataloaders=source_loader,
            val_dataloaders=source_validation_loader,
        )
        if source_trainer.global_step != _P2_FORMAL_OPTIMIZER_STEPS_PER_EPOCH:
            raise AssertionError(
                f"first Lightning fit stopped at global_step={source_trainer.global_step}"
            )
        source_train_dataset = getattr(source_system, "train_dataset", None)
        if (
            source_train_dataset is not source_dataset
            or getattr(source_train_dataset, "sampler", None)
            is not source_loader.sampler
            or source_loader.sampler.generator is None
            or not source_binding["configured_sampler_present"]
            or source_events
            != [
                "setup_configured_sampler_verified",
                "setup_actual_loader_sampler_bound",
            ]
            or len(source_dataset.sampled_indices) != _P2_FORMAL_TRAIN_BATCHES_PER_EPOCH
        ):
            raise AssertionError(
                "single-RIO preflight did not train through its weighted sampler"
            )
        source_sampler_state_after_step = (
            source_loader.sampler.generator.get_state().detach().cpu().clone()
        )
        sampler_source_state_advanced = not torch.equal(
            source_sampler_state_after_step,
            source_dataset.initial_generator_state,
        )
        if not sampler_source_state_advanced:
            raise AssertionError("source weighted sampler generator did not advance")
        monitored_callback_state = _cpu_snapshot(
            monitored_checkpoint_callback.state_dict()
        )
        observed_val_batches = source_trainer.num_val_batches
        source_validation_batch_count = (
            sum(int(value) for value in observed_val_batches)
            if isinstance(observed_val_batches, (list, tuple))
            else int(observed_val_batches)
        )
        source_validation_dataset_name = source_validation_dataset.dataset_name
        source_real_validation_path_executed = (
            source_validation_batch_count == 1
            and source_validation_dataset_name == "rio"
            and bool(source_validation_sample["name"])
        )
        if not source_real_validation_path_executed:
            raise AssertionError(
                "Lightning source did not execute one real RIO validation batch"
            )
        monitored_topk_checkpoint = Path(monitored_callback_state["best_model_path"])
        monitored_callback_history_populated = (
            monitored_callback_state["monitor"] == "val_mean_t-AP"
            and monitored_callback_state["best_model_score"] is not None
            and monitored_callback_state["current_score"] is not None
            and bool(monitored_callback_state["best_k_models"])
            and monitored_topk_checkpoint.is_file()
            and monitored_topk_checkpoint != temporary_checkpoint
            and monitored_callback_state["last_model_path"] == str(temporary_checkpoint)
        )
        if not monitored_callback_history_populated:
            raise AssertionError(
                "real validation did not populate monitored checkpoint history"
            )
        monitored_topk_checkpoint_bytes = monitored_topk_checkpoint.stat().st_size
        checkpoint_bytes = temporary_checkpoint.stat().st_size

        selected_checkpoint = find_resume_checkpoint(
            checkpoint_dir,
            formal_p2=True,
            cfg=source_config,
        )
        selected_by_main = selected_checkpoint == str(temporary_checkpoint)
        if not selected_by_main:
            raise AssertionError(
                "main formal P2 resume selection missed Lightning checkpoint"
            )
        verified_resume_checkpoint = require_p2_resume_checkpoint(
            source_config,
            selected_checkpoint,
        )
        verified_resume_checkpoint = Path(verified_resume_checkpoint)
        verified_snapshot_created = (
            verified_resume_checkpoint.is_file()
            and verified_resume_checkpoint != temporary_checkpoint
            and verified_resume_checkpoint.parent == checkpoint_dir / ".verified_inputs"
            and verified_resume_checkpoint.stat().st_size == checkpoint_bytes
        )
        if not verified_snapshot_created:
            raise AssertionError(
                "main formal P2 resume validation did not create a verified snapshot"
            )
        monitored_topk_checkpoint.unlink()
        monitored_topk_checkpoint_removed = not monitored_topk_checkpoint.exists()
        if not monitored_topk_checkpoint_removed:
            raise AssertionError("temporary monitored top-k checkpoint was not removed")
        source_config_semantic_sha256 = p2_training_semantic_sha256(source_config)

        # This full-state pickle is trusted because it was created above in this temp dir.
        saved = torch.load(
            verified_resume_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        saved_checkpoint_callbacks = saved.pop("callbacks", None)
        checkpoint_callbacks_nonempty = bool(saved_checkpoint_callbacks)
        if not checkpoint_callbacks_nonempty:
            raise AssertionError("Lightning full checkpoint has no callback state")
        if (
            not isinstance(saved_checkpoint_callbacks, Mapping)
            or set(saved_checkpoint_callbacks) != source_checkpoint_state_keys
        ):
            raise AssertionError(
                "Lightning full checkpoint does not contain all three formal "
                "ModelCheckpoint states"
            )
        source_callback_states = {
            callback.state_key: _cpu_snapshot(callback.state_dict())
            for callback in source_checkpoint_callbacks
        }
        if not _recursive_equal(
            saved_checkpoint_callbacks,
            source_callback_states,
        ):
            raise AssertionError(
                "saved formal ModelCheckpoint states differ from source Trainer"
            )
        sampler_checkpoint_key = source_system._TRAIN_SAMPLER_CHECKPOINT_KEY
        sampler_payload = saved.pop(sampler_checkpoint_key, None)
        if not isinstance(sampler_payload, Mapping):
            raise TypeError(
                "Lightning full checkpoint has no sampler generator payload"
            )
        sampler_generator_checkpointed = isinstance(
            sampler_payload.get("generator_state"),
            torch.Tensor,
        )
        sampler_resume_scope = sampler_payload.get("resume_scope")
        if (
            not sampler_generator_checkpointed
            or sampler_resume_scope != source_system._TRAIN_SAMPLER_RESUME_SCOPE
        ):
            raise AssertionError("Lightning sampler checkpoint payload is invalid")
        saved_sampler_state = sampler_payload["generator_state"].detach().cpu().clone()
        if not torch.equal(saved_sampler_state, source_sampler_state_after_step):
            raise AssertionError(
                "checkpoint sampler state differs from the actual source loader"
            )
        expected_stream_generator = torch.Generator()
        expected_stream_generator.set_state(saved_sampler_state)
        expected_stream = iter(
            WeightedRandomSampler(
                weights=torch.ones(4, dtype=torch.double),
                num_samples=_P2_FORMAL_TRAIN_BATCHES_PER_EPOCH,
                replacement=True,
                generator=expected_stream_generator,
            )
        )
        expected_next_indices = [next(expected_stream) for _ in range(4)]
        sampler_class = (
            f"{type(source_loader.sampler).__module__}."
            f"{type(source_loader.sampler).__name__}"
        )
        saved_state_dict = saved.pop("state_dict")
        saved_model_digest = _tensor_digest(saved_state_dict.items())
        optimizer_parameter_contract = saved.pop(
            source_system._OPTIMIZER_PARAMETER_CONTRACT_KEY,
            None,
        )
        optimizer_states = saved.pop("optimizer_states")
        scheduler_states = saved.pop("lr_schedulers")
        if len(optimizer_states) != 1 or len(scheduler_states) != 1:
            raise AssertionError(
                "Lightning full checkpoint requires one optimizer and one scheduler"
            )
        saved_optimizer = optimizer_states[0]
        saved_scheduler = scheduler_states[0]
        optimizer_parameter_ids = {
            parameter_id
            for group in saved_optimizer["param_groups"]
            for parameter_id in group["params"]
        }
        optimizer_state_slot_ids = set(saved_optimizer["state"])
        optimizer_state_slot_step_values = sorted(
            {
                float(
                    slot["step"].item()
                    if isinstance(slot["step"], torch.Tensor)
                    else slot["step"]
                )
                for slot in saved_optimizer["state"].values()
            }
        )
        optimizer_contract_parameters = (
            optimizer_parameter_contract.get("parameters")
            if isinstance(optimizer_parameter_contract, Mapping)
            else None
        )
        optimizer_contract_parameter_ids = (
            set(optimizer_contract_parameters)
            if isinstance(optimizer_contract_parameters, Mapping)
            else set()
        )
        optimizer_contract_parameter_names = (
            {
                metadata.get("name")
                for metadata in optimizer_contract_parameters.values()
                if isinstance(metadata, Mapping)
            }
            if isinstance(optimizer_contract_parameters, Mapping)
            else set()
        )
        optimizer_contract_parameter_groups = (
            optimizer_parameter_contract.get("param_groups")
            if isinstance(optimizer_parameter_contract, Mapping)
            else None
        )
        optimizer_parameter_group_order_matches_contract = (
            optimizer_contract_parameter_groups
            == [list(group["params"]) for group in saved_optimizer["param_groups"]]
        )
        trainable_parameter_schema = (
            optimizer_parameter_contract.get("trainable_parameters")
            if isinstance(optimizer_parameter_contract, Mapping)
            else None
        )
        trainable_parameter_schema_sha256 = (
            optimizer_parameter_contract.get("trainable_parameter_schema_sha256")
            if isinstance(optimizer_parameter_contract, Mapping)
            else None
        )
        expected_trainable_parameter_schema = [
            [
                metadata.get("name"),
                metadata.get("shape"),
                metadata.get("dtype"),
            ]
            for group in saved_optimizer["param_groups"]
            for parameter_id in group["params"]
            for metadata in [
                optimizer_contract_parameters.get(parameter_id, {})
                if isinstance(optimizer_contract_parameters, Mapping)
                else {}
            ]
        ]
        trainable_parameter_schema_payload = json.dumps(
            trainable_parameter_schema
            if isinstance(trainable_parameter_schema, list)
            else [],
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("ascii")
        observed_trainable_parameter_schema_sha256 = hashlib.sha256(
            trainable_parameter_schema_payload
        ).hexdigest()
        trainable_parameter_schema_order_matches_contract = (
            isinstance(trainable_parameter_schema, list)
            and trainable_parameter_schema == expected_trainable_parameter_schema
        )
        trainable_parameter_schema_matches_formal_contract = (
            trainable_parameter_schema_order_matches_contract
            and trainable_parameter_schema_sha256
            == observed_trainable_parameter_schema_sha256
            and trainable_parameter_schema_sha256
            == _P2_FORMAL_TRAINABLE_PARAMETER_SCHEMA_SHA256
        )
        if not trainable_parameter_schema_matches_formal_contract:
            raise AssertionError(
                "formal Lightning checkpoint trainable parameter schema does not "
                "match the main checkpoint contract"
            )
        trainable_parameter_schema_entry_count = len(trainable_parameter_schema)
        trainable_parameter_schema_serialized_bytes = len(
            trainable_parameter_schema_payload
        )
        optimizer_contract_state_dict = (
            optimizer_parameter_contract.get("state_dict")
            if isinstance(optimizer_parameter_contract, Mapping)
            else None
        )
        model_state_schema_sha256 = (
            optimizer_parameter_contract.get("state_dict_schema_sha256")
            if isinstance(optimizer_parameter_contract, Mapping)
            else None
        )
        model_state_schema_entry_count = (
            len(optimizer_contract_state_dict)
            if isinstance(optimizer_contract_state_dict, Mapping)
            else 0
        )
        model_state_schema_serialized_bytes = (
            len(
                json.dumps(
                    [
                        [name, metadata["shape"], metadata["dtype"]]
                        for name, metadata in sorted(
                            optimizer_contract_state_dict.items()
                        )
                    ],
                    ensure_ascii=True,
                    separators=(",", ":"),
                ).encode("ascii")
            )
            if isinstance(optimizer_contract_state_dict, Mapping)
            else 0
        )
        model_state_schema_matches_formal_contract = (
            model_state_schema_entry_count == len(saved_state_dict)
            and set(optimizer_contract_state_dict) == set(saved_state_dict)
            and model_state_schema_sha256 == _P2_FORMAL_MODEL_STATE_SCHEMA_SHA256
        )
        optimizer_state_slots_cover_all_trainable_parameters = (
            bool(optimizer_parameter_ids)
            and optimizer_parameter_ids == optimizer_state_slot_ids
            and optimizer_parameter_ids == optimizer_contract_parameter_ids
            and optimizer_parameter_group_order_matches_contract
            and optimizer_contract_parameter_names == source_trainable_parameter_names
            and trainable_parameter_schema_matches_formal_contract
            and model_state_schema_matches_formal_contract
        )
        if not optimizer_state_slots_cover_all_trainable_parameters:
            raise AssertionError(
                "formal Lightning checkpoint optimizer state does not cover "
                "every trainable parameter"
            )
        trainable_parameter_tensor_count = len(source_trainable_parameter_names)
        optimizer_param_group_parameter_count = len(optimizer_parameter_ids)
        optimizer_parameter_contract_count = len(optimizer_contract_parameter_ids)
        optimizer_state_slot_count = len(optimizer_state_slot_ids)
        optimizer_state_slot_step_min = min(optimizer_state_slot_step_values)
        optimizer_state_slot_step_max = max(optimizer_state_slot_step_values)
        optimizer_state_slots_all_at_global_step = optimizer_state_slot_step_values == [
            float(_P2_FORMAL_OPTIMIZER_STEPS_PER_EPOCH)
        ]
        del optimizer_states, scheduler_states
        saved_global_step = int(saved["global_step"])
        saved_epoch = int(saved["epoch"])
        saved_scheduler_last_epoch = int(saved_scheduler["last_epoch"])
        fit_loop = saved["loops"]["fit_loop"]
        fit_loop_epoch_total = dict(fit_loop["epoch_progress"]["total"])
        fit_loop_batches_that_stepped = int(
            fit_loop["epoch_loop.state_dict"]["_batches_that_stepped"]
        )
        fit_loop_optimizer_steps_completed = int(
            fit_loop["epoch_loop.automatic_optimization.optim_progress"]["optimizer"][
                "step"
            ]["total"]["completed"]
        )
        fit_loop_scheduler_steps_completed = int(
            fit_loop["epoch_loop.scheduler_progress"]["total"]["completed"]
        )
        fit_loop_train_batches_completed = int(
            fit_loop["epoch_loop.batch_progress"]["total"]["completed"]
        )
        fit_loop_validation_batches_completed = int(
            fit_loop["epoch_loop.val_loop.batch_progress"]["total"]["completed"]
        )
        source_checkpoint_generated_at_real_validation_epoch_end = (
            fit_loop_epoch_total
            == {
                "ready": 1,
                "completed": 0,
                "started": 1,
                "processed": 0,
            }
            and fit_loop_optimizer_steps_completed
            == _P2_FORMAL_OPTIMIZER_STEPS_PER_EPOCH
            and fit_loop_scheduler_steps_completed
            == _P2_FORMAL_OPTIMIZER_STEPS_PER_EPOCH
            and fit_loop_train_batches_completed == _P2_FORMAL_TRAIN_BATCHES_PER_EPOCH
            and fit_loop_batches_that_stepped
            == _P2_FORMAL_OPTIMIZER_STEPS_PER_EPOCH - 1
            and fit_loop_validation_batches_completed == 1
            and monitored_callback_history_populated
        )
        if (
            saved_epoch != 0
            or saved_global_step != _P2_FORMAL_OPTIMIZER_STEPS_PER_EPOCH
            or saved_scheduler_last_epoch != _P2_FORMAL_OPTIMIZER_STEPS_PER_EPOCH
            or not source_checkpoint_generated_at_real_validation_epoch_end
        ):
            raise AssertionError(
                "formal Lightning checkpoint is not a real epoch0 validation-end "
                "checkpoint"
            )
        del saved, saved_state_dict
        del source_sampler_state_after_step, expected_stream_generator
        del sampler_payload, source_binding, monitored_callback_state
        del monitored_checkpoint_callback, monitored_checkpoint_callbacks
        del source_dataset, source_loader
        del source_validation_data, source_validation_dataset
        del source_validation_loader, source_validation_names
        del source_validation_targets
        del source_events, source_train_dataset
        del source_callback_states, source_checkpoint_callbacks
        del source_config, source_trainer, source_system
        gc.collect()
        torch.cuda.empty_cache()

        resume_config, resumed_system = _compose_runtime(
            checkpoint,
            device,
            save_dir=checkpoint_dir,
            scheduler_total_steps=_P2_FORMAL_ONECYCLE_TOTAL_STEPS,
            lightning_weighted_sampler=True,
        )
        resume_config_matches_checkpoint = (
            p2_training_semantic_sha256(resume_config) == source_config_semantic_sha256
        )
        if not resume_config_matches_checkpoint:
            raise AssertionError("resumed runtime config changed from saved config")
        if (
            tuple(name for name, _ in resumed_system.named_parameters())
            != parameter_names
        ):
            raise AssertionError("resumed model parameter structure changed")
        resumed_dataset, resumed_loader = pinned_batch_loader(
            _P2_FORMAL_TRAIN_BATCHES_PER_EPOCH
        )
        resume_events: list[str] = []
        resumed_binding = bind_pinned_dataset_on_setup(
            resumed_system,
            resumed_dataset,
            resume_events,
        )
        probe = ResumeStateProbe(resume_events, resumed_dataset)
        resumed_checkpoint_callbacks = _instantiate_formal_checkpoint_callbacks(
            resume_config
        )
        resumed_trainer = make_trainer(
            max_epochs=2,
            max_steps=_P2_FORMAL_OPTIMIZER_STEPS_PER_EPOCH + 1,
            root_dir=checkpoint_dir,
            callbacks=[*resumed_checkpoint_callbacks, probe],
            limit_val_batches=0,
        )
        restore_fit_completed = False
        resumed_trainer.fit(
            resumed_system,
            train_dataloaders=resumed_loader,
            ckpt_path=verified_resume_checkpoint,
            weights_only=False,
        )
        restore_fit_completed = True
        if probe.snapshot is None:
            raise AssertionError(
                "Lightning resume probe did not observe restored state"
            )

        model_restored = probe.snapshot["model_digest"] == saved_model_digest
        optimizer_restored = _recursive_equal(
            probe.snapshot["optimizer"], saved_optimizer
        )
        scheduler_restored = _recursive_equal(
            probe.snapshot["scheduler"], saved_scheduler
        )
        checkpoint_callback_persistent_states_restored = _recursive_equal(
            _persistent_checkpoint_callback_states(
                probe.snapshot["checkpoint_callbacks"]
            ),
            _persistent_checkpoint_callback_states(saved_checkpoint_callbacks),
        )
        saved_monitored_current_score = saved_checkpoint_callbacks[
            next(
                state_key
                for state_key, state in saved_checkpoint_callbacks.items()
                if state.get("monitor") == "val_mean_t-AP"
            )
        ]["current_score"]
        resumed_monitored_current_score = probe.snapshot["checkpoint_callbacks"][
            next(
                state_key
                for state_key, state in probe.snapshot["checkpoint_callbacks"].items()
                if state.get("monitor") == "val_mean_t-AP"
            )
        ]["current_score"]
        checkpoint_callback_transient_score_not_restored = (
            saved_monitored_current_score is not None
            and resumed_monitored_current_score is None
        )
        sampler_state_restored = resumed_binding["restored_to_actual_loader_generator"]
        del saved_checkpoint_callbacks, saved_optimizer, saved_scheduler

        resumed_scheduler = resumed_trainer.lr_scheduler_configs[0].scheduler
        final_model_digest = _tensor_digest(resumed_system.state_dict().items())
        final_optimizer = _cpu_snapshot(resumed_trainer.optimizers[0].state_dict())
        final_scheduler = _cpu_snapshot(resumed_scheduler.state_dict())
        model_advanced = final_model_digest != probe.snapshot["model_digest"]
        optimizer_advanced = not _recursive_equal(
            final_optimizer, probe.snapshot["optimizer"]
        )
        scheduler_advanced = not _recursive_equal(
            final_scheduler, probe.snapshot["scheduler"]
        )
        final_sampler_state = (
            resumed_dataset.sampler.generator.get_state().detach().cpu().clone()
        )
        sampler_state_advanced = not torch.equal(
            final_sampler_state,
            saved_sampler_state,
        )
        sampler_stream_continuous = (
            resumed_dataset.sampled_indices == expected_next_indices
        )
        sampler_state_restored_to_actual_loader_generator = resumed_binding[
            "restored_to_actual_loader_generator"
        ]
        expected_restore_event_order = [
            "setup_configured_sampler_verified",
            "setup_actual_loader_sampler_bound",
            "on_load_checkpoint_actual_loader_sampler_verified",
            "on_load_checkpoint_actual_loader_sampler_restored",
            "on_train_batch_start_state_observed",
        ]
        if resume_events != expected_restore_event_order:
            raise AssertionError(
                "unexpected Lightning sampler restore order: " + repr(resume_events)
            )
        restored_scheduler_last_epoch = int(probe.snapshot["scheduler"]["last_epoch"])
        advanced_scheduler_last_epoch = int(final_scheduler["last_epoch"])
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        peak_vram_mib = torch.cuda.max_memory_allocated(device) / 1024**2
        details = {
            "kind": "pytorch_lightning_full_checkpoint_resume",
            "lightning_full_resume_validation": True,
            "model_class": model_class,
            "sample_name": sample_name,
            "architecture": {
                "model": model_name,
                "backbone": "Concerto",
                "backbone_class": backbone_class,
            },
            "data_scope": "single_real_3rscan_t2_window",
            "scannet_used": False,
            "mixed_sampler_resume_validation": False,
            "sampler_scope": "single_real_3rscan_preflight_weighted_sampler",
            "sampler_class": sampler_class,
            "sampler_generator_checkpointed": sampler_generator_checkpointed,
            "sampler_resume_scope": sampler_resume_scope,
            "sampler_source_state_advanced": sampler_source_state_advanced,
            "sampler_state_restored": sampler_state_restored,
            "sampler_state_advanced": sampler_state_advanced,
            "sampler_state_restored_to_actual_loader_generator": (
                sampler_state_restored_to_actual_loader_generator
            ),
            "sampler_stream_continuous": sampler_stream_continuous,
            "sampler_restore_event_order": list(resume_events),
            "main_formal_resume_checkpoint_selected": selected_by_main,
            "main_verified_resume_snapshot_used": (
                verified_snapshot_created and restore_fit_completed
            ),
            "resume_config_matches_checkpoint": resume_config_matches_checkpoint,
            "checkpoint_callbacks_nonempty": checkpoint_callbacks_nonempty,
            "formal_model_checkpoint_callback_count": len(resumed_checkpoint_callbacks),
            "formal_model_checkpoint_state_keys_unique": (
                len(source_checkpoint_state_keys) == 3
            ),
            "formal_model_checkpoint_persistent_states_restored": (
                checkpoint_callback_persistent_states_restored
            ),
            "formal_model_checkpoint_transient_current_score_policy": (
                "not_restored_by_pytorch_lightning_2_6_5"
            ),
            "formal_model_checkpoint_transient_current_score_not_restored": (
                checkpoint_callback_transient_score_not_restored
            ),
            "source_real_validation_path_executed": (
                source_real_validation_path_executed
            ),
            "source_validation_batch_count": source_validation_batch_count,
            "source_validation_dataset": source_validation_dataset_name,
            "source_validation_sample": source_validation_sample,
            "monitored_callback_history_populated": (
                monitored_callback_history_populated
            ),
            "monitored_topk_checkpoint_bytes": (monitored_topk_checkpoint_bytes),
            "monitored_topk_checkpoint_removed": (monitored_topk_checkpoint_removed),
            "source_checkpoint_generated_at_real_validation_epoch_end": (
                source_checkpoint_generated_at_real_validation_epoch_end
            ),
            "source_fit_loop_progress": {
                "epoch_total": fit_loop_epoch_total,
                "batches_that_stepped": fit_loop_batches_that_stepped,
                "optimizer_steps_completed": (fit_loop_optimizer_steps_completed),
                "scheduler_steps_completed": (fit_loop_scheduler_steps_completed),
                "train_batches_completed": fit_loop_train_batches_completed,
                "validation_batches_completed": (fit_loop_validation_batches_completed),
            },
            "lightning_ckpt_path_restore": restore_fit_completed,
            "lightning_fit_weights_only": False,
            "checkpoint_bytes": checkpoint_bytes,
            "elapsed_seconds": elapsed,
            "peak_allocated_vram_mib": peak_vram_mib,
            "formal_optimizer_steps_per_epoch": (_P2_FORMAL_OPTIMIZER_STEPS_PER_EPOCH),
            "formal_train_batches_per_epoch": (_P2_FORMAL_TRAIN_BATCHES_PER_EPOCH),
            "gradient_accumulation_steps": 4,
            "formal_completed_epoch_boundary": (
                saved_epoch == 0
                and saved_global_step == _P2_FORMAL_OPTIMIZER_STEPS_PER_EPOCH
            ),
            "trainable_parameter_tensor_count": (trainable_parameter_tensor_count),
            "optimizer_param_group_parameter_count": (
                optimizer_param_group_parameter_count
            ),
            "optimizer_parameter_contract_count": (optimizer_parameter_contract_count),
            "optimizer_parameter_group_order_matches_contract": (
                optimizer_parameter_group_order_matches_contract
            ),
            "trainable_parameter_schema_entry_count": (
                trainable_parameter_schema_entry_count
            ),
            "trainable_parameter_schema_serialized_bytes": (
                trainable_parameter_schema_serialized_bytes
            ),
            "trainable_parameter_schema_sha256": (
                trainable_parameter_schema_sha256
            ),
            "trainable_parameter_schema_order_matches_contract": (
                trainable_parameter_schema_order_matches_contract
            ),
            "trainable_parameter_schema_matches_formal_contract": (
                trainable_parameter_schema_matches_formal_contract
            ),
            "optimizer_state_slot_count": optimizer_state_slot_count,
            "optimizer_state_slot_step_unique_values": (
                optimizer_state_slot_step_values
            ),
            "optimizer_state_slot_step_min": optimizer_state_slot_step_min,
            "optimizer_state_slot_step_max": optimizer_state_slot_step_max,
            "optimizer_state_slots_all_at_global_step": (
                optimizer_state_slots_all_at_global_step
            ),
            "optimizer_state_slots_cover_all_trainable_parameters": (
                optimizer_state_slots_cover_all_trainable_parameters
            ),
            "model_state_schema_entry_count": model_state_schema_entry_count,
            "model_state_schema_serialized_bytes": (
                model_state_schema_serialized_bytes
            ),
            "model_state_schema_sha256": model_state_schema_sha256,
            "model_state_schema_matches_formal_contract": (
                model_state_schema_matches_formal_contract
            ),
            "saved_epoch": saved_epoch,
            "saved_global_step": saved_global_step,
            "restored_global_step": probe.snapshot["global_step"],
            "advanced_global_step": int(resumed_trainer.global_step),
            "model_state_scope": "lightning_module_state_dict",
            "model_state_restored": model_restored,
            "optimizer_state_restored": optimizer_restored,
            "scheduler_state_restored": scheduler_restored,
            "model_state_advanced": model_advanced,
            "optimizer_state_advanced": optimizer_advanced,
            "scheduler_state_advanced": scheduler_advanced,
            "saved_scheduler_last_epoch": saved_scheduler_last_epoch,
            "restored_scheduler_last_epoch": restored_scheduler_last_epoch,
            "advanced_scheduler_last_epoch": advanced_scheduler_last_epoch,
        }

        del saved_sampler_state
        del final_optimizer, final_scheduler, final_sampler_state, probe
        del resumed_binding, resumed_checkpoint_callbacks
        del resumed_dataset, resumed_loader, resume_events
        del resumed_trainer, resumed_system

    if (
        details is None
        or temporary_checkpoint is None
        or verified_resume_checkpoint is None
    ):
        raise AssertionError("Lightning checkpoint resume smoke produced no result")
    details["temporary_checkpoint_removed"] = not temporary_checkpoint.exists()
    details["verified_resume_snapshot_removed"] = (
        not verified_resume_checkpoint.exists()
    )
    checks = {
        "main_formal_resume_checkpoint_selected": details[
            "main_formal_resume_checkpoint_selected"
        ],
        "main_verified_resume_snapshot_used": details[
            "main_verified_resume_snapshot_used"
        ],
        "resume_config_matches_checkpoint": details["resume_config_matches_checkpoint"],
        "sampler_generator_checkpointed": details["sampler_generator_checkpointed"],
        "sampler_source_state_advanced": details["sampler_source_state_advanced"],
        "sampler_state_restored": details["sampler_state_restored"],
        "sampler_state_advanced": details["sampler_state_advanced"],
        "sampler_state_restored_to_actual_loader_generator": details[
            "sampler_state_restored_to_actual_loader_generator"
        ],
        "sampler_stream_continuous": details["sampler_stream_continuous"],
        "checkpoint_callbacks_nonempty": details["checkpoint_callbacks_nonempty"],
        "formal_model_checkpoint_callback_count_is_3": (
            details["formal_model_checkpoint_callback_count"] == 3
        ),
        "formal_model_checkpoint_state_keys_unique": details[
            "formal_model_checkpoint_state_keys_unique"
        ],
        "formal_model_checkpoint_persistent_states_restored": details[
            "formal_model_checkpoint_persistent_states_restored"
        ],
        "formal_model_checkpoint_transient_current_score_not_restored": details[
            "formal_model_checkpoint_transient_current_score_not_restored"
        ],
        "source_real_validation_path_executed": details[
            "source_real_validation_path_executed"
        ],
        "monitored_callback_history_populated": details[
            "monitored_callback_history_populated"
        ],
        "monitored_topk_checkpoint_removed": details[
            "monitored_topk_checkpoint_removed"
        ],
        "source_checkpoint_generated_at_real_validation_epoch_end": details[
            "source_checkpoint_generated_at_real_validation_epoch_end"
        ],
        "lightning_ckpt_path_restore": details["lightning_ckpt_path_restore"],
        "formal_completed_epoch_boundary": details["formal_completed_epoch_boundary"],
        "optimizer_state_slots_cover_all_trainable_parameters": details[
            "optimizer_state_slots_cover_all_trainable_parameters"
        ],
        "optimizer_parameter_group_order_matches_contract": details[
            "optimizer_parameter_group_order_matches_contract"
        ],
        "model_state_schema_matches_formal_contract": details[
            "model_state_schema_matches_formal_contract"
        ],
        "saved_global_step_is_66": details["saved_global_step"] == 66,
        "restored_global_step_is_66": details["restored_global_step"] == 66,
        "advanced_global_step_is_67": details["advanced_global_step"] == 67,
        "model_state_restored": details["model_state_restored"],
        "optimizer_state_restored": details["optimizer_state_restored"],
        "scheduler_state_restored": details["scheduler_state_restored"],
        "model_state_advanced": details["model_state_advanced"],
        "optimizer_state_advanced": details["optimizer_state_advanced"],
        "scheduler_state_advanced": details["scheduler_state_advanced"],
        "saved_scheduler_last_epoch_is_66": (
            details["saved_scheduler_last_epoch"] == 66
        ),
        "restored_scheduler_last_epoch_is_66": (
            details["restored_scheduler_last_epoch"] == 66
        ),
        "advanced_scheduler_last_epoch_is_67": (
            details["advanced_scheduler_last_epoch"] == 67
        ),
        "temporary_checkpoint_removed": details["temporary_checkpoint_removed"],
        "verified_resume_snapshot_removed": details["verified_resume_snapshot_removed"],
    }
    details["passed"] = all(checks.values())
    if not details["passed"]:
        failed = [key for key, passed in checks.items() if not passed]
        raise AssertionError(
            "Lightning full checkpoint resume validation failed: " + ", ".join(failed)
        )
    gc.collect()
    torch.cuda.empty_cache()
    return details


def _hardware(device: Any) -> dict[str, Any]:
    import torch

    properties = torch.cuda.get_device_properties(device)
    if "A40" not in properties.name:
        raise RuntimeError(
            f"P2 native check requires one NVIDIA A40, got {properties.name}"
        )
    return {
        "device_alias": "device-0",
        "name": properties.name,
        "total_memory_mib": round(properties.total_memory / 1024**2),
        "gpu_count_used": 1,
    }


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def _build_source_tree_contract(
    source_commit: str | None = None,
) -> dict[str, Any]:
    from utils.p2_preflight import build_p2_source_tree_contract

    return build_p2_source_tree_contract(
        source_commit=source_commit,
        repo_root=PROJECT_ROOT,
    )


def _build_runtime_source_contract() -> dict[str, Any]:
    from utils.p2_preflight import build_p2_runtime_source_contract

    return build_p2_runtime_source_contract(repo_root=PROJECT_ROOT)


def _build_runtime_environment_contract() -> dict[str, Any]:
    from utils.p2_preflight import build_p2_runtime_environment_contract

    return build_p2_runtime_environment_contract()


def _tracked_tree_record_is_clean(record: Mapping[str, Any]) -> bool:
    expected = record.get("expected_tracked_tree_sha256")
    observed = record.get("observed_tracked_tree_sha256")
    return (
        record.get("index_flag_paths") == []
        and isinstance(expected, str)
        and len(expected) == 64
        and all(character in "0123456789abcdef" for character in expected)
        and observed == expected
    )


def _require_passing_source_tree_contract(contract: Mapping[str, Any]) -> None:
    committed_paths = contract.get("committed_paths_since_source")
    dirty_paths = contract.get("dirty_paths")
    if (
        contract.get("schema_version") == 1
        and contract.get("status") == "pass"
        and contract.get("allowed_dirty_prefixes") == [SOURCE_TREE_ARTIFACT_PREFIX]
        and isinstance(committed_paths, list)
        and all(
            path.startswith(SOURCE_TREE_ARTIFACT_PREFIX) for path in committed_paths
        )
        and isinstance(dirty_paths, list)
        and all(path.startswith(SOURCE_TREE_ARTIFACT_PREFIX) for path in dirty_paths)
        and _tracked_tree_record_is_clean(contract)
        and not contract.get("disallowed_committed_paths")
        and not contract.get("disallowed_dirty_paths")
        and not contract.get("errors")
    ):
        return
    problems = [
        *list(contract.get("disallowed_committed_paths") or []),
        *list(contract.get("disallowed_dirty_paths") or []),
        *list(contract.get("errors") or []),
    ]
    if not problems:
        problems.append("invalid_source_tree_contract")
    raise RuntimeError(
        "P2 GPU artifact source tree contract is not pass: "
        + ", ".join(str(problem) for problem in problems)
    )


def _require_passing_runtime_source_contract(
    contract: Mapping[str, Any],
) -> None:
    repositories = contract.get("repositories")
    if (
        contract.get("schema_version") == 1
        and contract.get("status") == "pass"
        and isinstance(repositories, Mapping)
        and bool(repositories)
        and all(
            isinstance(record, Mapping)
            and record.get("status") == "pass"
            and record.get("errors") == []
            and _tracked_tree_record_is_clean(record)
            for record in repositories.values()
        )
        and contract.get("errors") == []
    ):
        return
    problems = list(contract.get("errors") or [])
    if not problems:
        problems.append("invalid_runtime_source_contract")
    raise RuntimeError(
        "P2 GPU artifact runtime source contract is not pass: "
        + ", ".join(str(problem) for problem in problems)
    )


def _require_passing_runtime_environment_contract(
    contract: Mapping[str, Any],
) -> None:
    from utils.p2_preflight import _validate_runtime_environment_contract

    problems: list[str] = []
    _validate_runtime_environment_contract(
        {"runtime_environment_contract": contract},
        problems,
    )
    if not problems:
        return
    raise RuntimeError(
        "P2 GPU artifact runtime environment contract is not pass: "
        + ", ".join(str(problem) for problem in problems)
    )


def _begin_source_tree_contract() -> dict[str, Any]:
    contract = _build_source_tree_contract()
    _require_passing_source_tree_contract(contract)
    runtime_contract = _build_runtime_source_contract()
    _require_passing_runtime_source_contract(runtime_contract)
    runtime_environment_contract = _build_runtime_environment_contract()
    _require_passing_runtime_environment_contract(runtime_environment_contract)
    source_commit = contract.get("observed_head")
    if (
        not isinstance(source_commit, str)
        or len(source_commit) != 40
        or any(character not in "0123456789abcdef" for character in source_commit)
    ):
        raise RuntimeError("P2 GPU artifact generation requires a full Git HEAD")
    if contract.get("source_commit") != source_commit:
        raise RuntimeError("P2 GPU artifact source commit differs from Git HEAD")
    if _git_commit() != source_commit:
        raise RuntimeError("Git HEAD changed during generation preflight")
    return {
        "source_commit": source_commit,
        "source_tree_contract_before": contract,
        "runtime_source_contract_before": copy.deepcopy(runtime_contract),
        "runtime_environment_contract_before": copy.deepcopy(
            runtime_environment_contract
        ),
    }


def _finalize_source_tree_contract(
    guard: Mapping[str, Any],
) -> dict[str, Any]:
    source_commit = str(guard["source_commit"])
    if _git_commit() != source_commit:
        raise RuntimeError("Git HEAD changed during generation")
    contract = _build_source_tree_contract(source_commit)
    if contract.get("observed_head") != source_commit:
        raise RuntimeError("Git HEAD changed during generation")
    _require_passing_source_tree_contract(contract)
    runtime_contract = _build_runtime_source_contract()
    _require_passing_runtime_source_contract(runtime_contract)
    if runtime_contract != guard.get("runtime_source_contract_before"):
        raise RuntimeError("P2 runtime source changed during generation")
    runtime_environment_contract = _build_runtime_environment_contract()
    _require_passing_runtime_environment_contract(runtime_environment_contract)
    if runtime_environment_contract != guard.get("runtime_environment_contract_before"):
        raise RuntimeError("P2 runtime environment changed during generation")
    result = dict(contract)
    result["generation_head_unchanged"] = True
    result["dirty_paths_before_generation"] = list(
        guard["source_tree_contract_before"]["dirty_paths"]
    )
    return result


def run_native_smoke(
    checkpoint: Path,
    device: Any,
    *,
    source_tree_guard: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    import gc

    import torch

    source_tree_guard = source_tree_guard or _begin_source_tree_contract()
    seed_everything(SEED)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    config, system = _compose_runtime(checkpoint, device)
    data, targets, sample, input_provenance = _materialize_named_train_batch(
        config, TINY_SAMPLE_NAME, device
    )
    groups = classify_parameters(system.named_parameters())
    required_groups = (
        "frozen_encoder",
        "trainable_concerto_decoder",
        "trainable_rescene_decoder",
        "trainable_rescene_heads",
    )
    if not all(groups[name] for name in required_groups):
        raise AssertionError("native model parameter groups are incomplete")
    if any(
        dict(system.named_parameters())[name].requires_grad
        for name in groups["frozen_encoder"]
    ):
        raise AssertionError("Concerto encoder/embedding parameters are not frozen")
    frozen_before = _tensor_digest(_named_subset(system, groups["frozen_encoder"]))
    trainable_names = [
        name
        for group_name, names in groups.items()
        if group_name != "frozen_encoder"
        for name in names
    ]
    trainable_before = _tensor_digest(_named_subset(system, trainable_names))
    all_before_first_step = _tensor_digest(system.named_parameters())

    optimizer, scheduler = _make_optimizer_scheduler(config, system, total_steps=4)
    initial_lr = float(optimizer.param_groups[0]["lr"])
    _output, losses, breakdown, quality = _one_optimizer_step(
        system, data, targets, optimizer, scheduler
    )
    lr_after_first_step = float(optimizer.param_groups[0]["lr"])
    frozen_grad = _gradient_summary(system, groups["frozen_encoder"])
    concerto_decoder_grad = _gradient_summary(
        system, groups["trainable_concerto_decoder"]
    )
    rescene_decoder_grad = _gradient_summary(
        system, groups["trainable_rescene_decoder"]
    )
    head_grad = _gradient_summary(system, groups["trainable_rescene_heads"])
    objective_grad = _gradient_summary(system, groups["trainable_objective"])
    if frozen_grad["nonzero_grad_tensors"] != 0:
        raise AssertionError("frozen Concerto encoder received nonzero gradients")
    if (
        not concerto_decoder_grad["finite"]
        or concerto_decoder_grad["nonzero_grad_tensors"] == 0
    ):
        raise AssertionError(
            "Concerto decoder did not receive finite nonzero gradients"
        )
    if (
        not rescene_decoder_grad["finite"]
        or rescene_decoder_grad["nonzero_grad_tensors"] == 0
    ):
        raise AssertionError("ReScene decoder did not receive finite nonzero gradients")
    if not head_grad["finite"] or head_grad["nonzero_grad_tensors"] == 0:
        raise AssertionError("ReScene heads did not receive finite nonzero gradients")
    if initial_lr == lr_after_first_step:
        raise AssertionError("OneCycle learning rate did not advance")
    global_step = 1
    trainable_after_first = _tensor_digest(_named_subset(system, trainable_names))
    all_after_first_step = _tensor_digest(system.named_parameters())
    if trainable_after_first == trainable_before:
        raise AssertionError("native decoder/head parameters did not update")
    raw_losses = {
        key: float(value.detach().cpu())
        for key, value in losses.items()
        if not _is_contrastive_diagnostic(key)
    }
    first_step_weighted_objective = float(breakdown["objective"].detach().cpu())
    first_step_final_head_segmentation = float(
        breakdown["final_head_segmentation"].detach().cpu()
    )
    first_step_aggregate_contrastive = float(
        breakdown["aggregate_contrastive"].detach().cpu()
    )
    del _output, losses, breakdown

    with tempfile.TemporaryDirectory(prefix="p2-native-checkpoint-") as directory:
        checkpoint_path = Path(directory) / "roundtrip.ckpt"
        torch.save(
            {
                "model": system.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "global_step": global_step,
            },
            checkpoint_path,
        )
        saved_bytes = checkpoint_path.stat().st_size
        saved_model_digest = _tensor_digest(system.named_parameters())
        saved_optimizer = copy.deepcopy(optimizer.state_dict())
        saved_scheduler = copy.deepcopy(scheduler.state_dict())

        _one_optimizer_step(system, data, targets, optimizer, scheduler)
        loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        system.load_state_dict(loaded["model"])
        optimizer.load_state_dict(loaded["optimizer"])
        scheduler.load_state_dict(loaded["scheduler"])
        global_step = int(loaded["global_step"])
        restored = (
            _tensor_digest(system.named_parameters()) == saved_model_digest
            and _recursive_equal(optimizer.state_dict(), saved_optimizer)
            and _recursive_equal(scheduler.state_dict(), saved_scheduler)
            and global_step == 1
        )
        if not restored:
            raise AssertionError(
                "model/optimizer/scheduler/global-step roundtrip failed"
            )
        restored_model_digest = _tensor_digest(system.named_parameters())
        restored_optimizer = copy.deepcopy(optimizer.state_dict())
        _one_optimizer_step(system, data, targets, optimizer, scheduler)
        global_step += 1
        advanced_model_changed = (
            _tensor_digest(system.named_parameters()) != restored_model_digest
        )
        advanced_optimizer_changed = not _recursive_equal(
            optimizer.state_dict(), restored_optimizer
        )
        advanced_after_restore = (
            global_step == 2
            and scheduler.last_epoch == 2
            and advanced_model_changed
            and advanced_optimizer_changed
        )
        if not advanced_after_restore:
            raise AssertionError(
                "restored checkpoint did not advance one optimizer step"
            )
        del loaded, saved_optimizer, saved_scheduler, restored_optimizer
    native_checkpoint_removed = not checkpoint_path.exists()
    if not native_checkpoint_removed:
        raise AssertionError("native roundtrip checkpoint was not removed")

    frozen_after = _tensor_digest(_named_subset(system, groups["frozen_encoder"]))
    trainable_after = _tensor_digest(_named_subset(system, trainable_names))
    evaluator = _run_validation_evaluator(system, config, device)
    torch.cuda.synchronize(device)
    native_elapsed = time.perf_counter() - started
    native_peak_vram_mib = torch.cuda.max_memory_allocated(device) / 1024**2

    scheduler_last_epoch = int(scheduler.last_epoch)
    del optimizer, scheduler, system
    gc.collect()
    torch.cuda.empty_cache()

    lightning_resume = _run_lightning_checkpoint_resume(
        checkpoint,
        device,
        data,
        targets,
        sample["name"],
    )

    report = {
        "scope": "preflight-only",
        "verification_mode": "artifact_contract_not_reexecution",
        "official_mixed_data_reproduction": False,
        "g2_evidence": False,
        "seed": SEED,
        "source_commit": source_tree_guard["source_commit"],
        "checkpoint": checkpoint_provenance(checkpoint),
        "hardware": _hardware(device),
        "dependencies": _dependency_provenance(),
        "sample": sample,
        "input_provenance": input_provenance,
        "objective_contract": {
            "segmentation_weights": {"class": 2.0, "mask": 5.0, "dice": 2.0},
            "aggregate_contrastive_counted_once": True,
            "per_layer_contrastive_diagnostics_excluded": True,
        },
        "smoke": {
            "passed": True,
            "elapsed_seconds": native_elapsed,
            "peak_allocated_vram_mib": native_peak_vram_mib,
            "optimizer_global_step": global_step,
            "scheduler_last_epoch": scheduler_last_epoch,
            "initial_lr": initial_lr,
            "lr_after_first_step": lr_after_first_step,
            "first_step_model_bitwise_changed": (
                all_after_first_step != all_before_first_step
            ),
            "encoder_bitwise_unchanged": frozen_before == frozen_after,
            "decoder_head_changed": trainable_before != trainable_after,
            "segment_contrastive_positive": raw_losses["loss_segment_contrastive"]
            > 1e-8,
            "raw_objective_losses": raw_losses,
            "first_step_weighted_objective": first_step_weighted_objective,
            "first_step_final_head_segmentation": (first_step_final_head_segmentation),
            "first_step_aggregate_contrastive": first_step_aggregate_contrastive,
            "first_step_matching": quality,
            "gradients": {
                "frozen_encoder": frozen_grad,
                "trainable_concerto_decoder": concerto_decoder_grad,
                "trainable_rescene_decoder": rescene_decoder_grad,
                "trainable_rescene_heads": head_grad,
                "trainable_objective": objective_grad,
            },
            "parameter_tensors": {key: len(value) for key, value in groups.items()},
        },
        "checkpoint_roundtrip": {
            "passed": restored and advanced_after_restore,
            "kind": "native_model_optimizer_scheduler_state_roundtrip",
            "lightning_full_resume_validation": False,
            "temporary_checkpoint_removed": native_checkpoint_removed,
            "checkpoint_bytes": saved_bytes,
            "restored_global_step": 1,
            "advanced_global_step": global_step,
            "advanced_model_bitwise_changed": advanced_model_changed,
            "advanced_optimizer_state_changed": advanced_optimizer_changed,
        },
        "lightning_checkpoint_resume": lightning_resume,
        "validation_evaluator": evaluator,
    }
    if not report["smoke"]["encoder_bitwise_unchanged"]:
        raise AssertionError("frozen Concerto encoder changed during native smoke")
    report_path = ARTIFACT_DIR / "native_smoke_report.json"
    report["runtime_source_contract"] = copy.deepcopy(
        source_tree_guard["runtime_source_contract_before"]
    )
    report["runtime_environment_contract"] = copy.deepcopy(
        source_tree_guard["runtime_environment_contract_before"]
    )
    report["source_tree_contract"] = _finalize_source_tree_contract(source_tree_guard)
    _write_json(report, report_path)
    report["source_tree_contract"] = _finalize_source_tree_contract(source_tree_guard)
    _write_json(report, report_path)
    if (
        _finalize_source_tree_contract(source_tree_guard)
        != report["source_tree_contract"]
    ):
        raise RuntimeError("source tree changed while finalizing native artifact")
    return report


def run_tiny_overfit(
    checkpoint: Path,
    device: Any,
    *,
    source_tree_guard: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    import torch

    source_tree_guard = source_tree_guard or _begin_source_tree_contract()
    seed_everything(SEED)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    config, system = _compose_runtime(checkpoint, device)
    data, targets, sample, input_provenance = _materialize_named_train_batch(
        config, TINY_SAMPLE_NAME, device
    )
    groups = classify_parameters(system.named_parameters())
    frozen_before = _tensor_digest(_named_subset(system, groups["frozen_encoder"]))
    trainable_names = [
        name
        for group_name, names in groups.items()
        if group_name != "frozen_encoder"
        for name in names
    ]
    trainable_before = _tensor_digest(_named_subset(system, trainable_names))
    optimizer, scheduler = _make_optimizer_scheduler(
        config, system, total_steps=TINY_OVERFIT_STEPS
    )
    history = []
    for step in range(1, TINY_OVERFIT_STEPS + 1):
        output, losses, breakdown, quality = _one_optimizer_step(
            system, data, targets, optimizer, scheduler
        )
        history.append(
            {
                "step": step,
                "objective": float(breakdown["objective"].detach().cpu()),
                "final_head_segmentation": float(
                    breakdown["final_head_segmentation"].detach().cpu()
                ),
                "all_segmentation": float(breakdown["all_segmentation"].detach().cpu()),
                "aggregate_contrastive": float(
                    breakdown["aggregate_contrastive"].detach().cpu()
                ),
                "classification_accuracy": quality["classification_accuracy"],
                "mean_dice": quality["mean_dice"],
                "matched_instances": quality["matched_instances"],
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
        )
        del output, losses, breakdown
        if step == 1 or step % 16 == 0:
            print(
                f"tiny-overfit step={step:03d}/{TINY_OVERFIT_STEPS} "
                f"seg={history[-1]['final_head_segmentation']:.6f} "
                f"contrastive={history[-1]['aggregate_contrastive']:.6f} "
                f"acc={history[-1]['classification_accuracy']:.3f} "
                f"dice={history[-1]['mean_dice']:.3f}",
                flush=True,
            )

    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    gate_result = evaluate_tiny_overfit_gates(history)
    frozen_after = _tensor_digest(_named_subset(system, groups["frozen_encoder"]))
    trainable_after = _tensor_digest(_named_subset(system, trainable_names))
    encoder_unchanged = frozen_before == frozen_after
    decoder_head_changed = trainable_before != trainable_after
    passed = gate_result["passed"] and encoder_unchanged and decoder_head_changed
    report = {
        "scope": "preflight-only",
        "verification_mode": "artifact_contract_not_reexecution",
        "official_mixed_data_reproduction": False,
        "g2_evidence": False,
        "seed": SEED,
        "source_commit": source_tree_guard["source_commit"],
        "checkpoint": checkpoint_provenance(checkpoint),
        "hardware": _hardware(device),
        "sample_name": TINY_SAMPLE_NAME,
        "sample": sample,
        "input_provenance": input_provenance,
        "optimizer": "AdamW",
        "scheduler": "OneCycleLR",
        "max_lr": 5e-4,
        "steps": TINY_OVERFIT_STEPS,
        "elapsed_seconds": elapsed,
        "peak_allocated_vram_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
        **{key: value for key, value in gate_result.items() if key != "passed"},
        "encoder_bitwise_unchanged": encoder_unchanged,
        "decoder_head_changed": decoder_head_changed,
        "passed": passed,
        "history": history,
    }
    report["runtime_source_contract"] = copy.deepcopy(
        source_tree_guard["runtime_source_contract_before"]
    )
    report["runtime_environment_contract"] = copy.deepcopy(
        source_tree_guard["runtime_environment_contract_before"]
    )
    report["source_tree_contract"] = _finalize_source_tree_contract(source_tree_guard)
    markdown = render_tiny_overfit_markdown(
        report,
        sample_name=TINY_SAMPLE_NAME,
        elapsed_seconds=elapsed,
        peak_vram_mib=report["peak_allocated_vram_mib"],
    )
    report_path = ARTIFACT_DIR / "tiny_overfit_report.json"
    markdown_path = ARTIFACT_DIR / "tiny_overfit_report.md"
    _write_json(report, report_path)
    _write_text(markdown, markdown_path)
    report["source_tree_contract"] = _finalize_source_tree_contract(source_tree_guard)
    _write_json(report, report_path)
    if (
        _finalize_source_tree_contract(source_tree_guard)
        != report["source_tree_contract"]
    ):
        raise RuntimeError("source tree changed while finalizing tiny artifact")
    if not passed:
        failed = [name for name, value in report["gates"].items() if not value]
        if not encoder_unchanged:
            failed.append("encoder_bitwise_unchanged")
        if not decoder_head_changed:
            failed.append("decoder_head_changed")
        raise RuntimeError("tiny overfit gates failed: " + ", ".join(failed))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("all", "smoke", "tiny"), default="all")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.chdir(PROJECT_ROOT)
    source_tree_guard = _begin_source_tree_contract()
    try:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the P2 native training checks")
        checkpoint_provenance(args.checkpoint)
        device = torch.device("cuda", args.device)
        torch.cuda.set_device(device)
        torch.set_float32_matmul_precision("high")
        _hardware(device)
        if args.mode in {"all", "smoke"}:
            run_native_smoke(
                args.checkpoint,
                device,
                source_tree_guard=source_tree_guard,
            )
        if args.mode in {"all", "tiny"}:
            run_tiny_overfit(
                args.checkpoint,
                device,
                source_tree_guard=source_tree_guard,
            )
    finally:
        _finalize_source_tree_contract(source_tree_guard)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
