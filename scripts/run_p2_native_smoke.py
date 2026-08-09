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


def _compose_runtime(checkpoint: Path, device: Any):
    from hydra import compose, initialize_config_dir
    from omegaconf import open_dict

    seed_everything(SEED)
    with initialize_config_dir(
        version_base="1.2", config_dir=str(PROJECT_ROOT / "conf")
    ):
        config = compose(
            config_name="config_p2_rescene4d_concerto_t2",
            overrides=["data/datasets=rio"],
        )
    processed_dir = PROJECT_ROOT / "data" / "processed" / "rio"
    with open_dict(config):
        config.general.gpus = 1
        config.general.compile_model = False
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
    if not config.loss.contrastive_loss:
        raise RuntimeError("P2 runtime composed with contrastive loss disabled")

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


def run_native_smoke(checkpoint: Path, device: Any) -> dict[str, Any]:
    import torch

    seed_everything(SEED)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    config, system = _compose_runtime(checkpoint, device)
    data, targets, sample = _materialize_named_train_batch(
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

    torch.cuda.synchronize(device)
    frozen_after = _tensor_digest(_named_subset(system, groups["frozen_encoder"]))
    trainable_after = _tensor_digest(_named_subset(system, trainable_names))
    evaluator = _run_validation_evaluator(system, config, device)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    raw_losses = {
        key: float(value.detach().cpu())
        for key, value in losses.items()
        if not _is_contrastive_diagnostic(key)
    }
    report = {
        "scope": "preflight-only",
        "official_mixed_data_reproduction": False,
        "g2_evidence": False,
        "seed": SEED,
        "source_commit": _git_commit(),
        "checkpoint": checkpoint_provenance(checkpoint),
        "hardware": _hardware(device),
        "dependencies": _dependency_provenance(),
        "sample": sample,
        "objective_contract": {
            "segmentation_weights": {"class": 2.0, "mask": 5.0, "dice": 2.0},
            "aggregate_contrastive_counted_once": True,
            "per_layer_contrastive_diagnostics_excluded": True,
        },
        "smoke": {
            "passed": True,
            "elapsed_seconds": elapsed,
            "peak_allocated_vram_mib": torch.cuda.max_memory_allocated(device)
            / 1024**2,
            "optimizer_global_step": global_step,
            "scheduler_last_epoch": scheduler.last_epoch,
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
            "first_step_weighted_objective": float(
                breakdown["objective"].detach().cpu()
            ),
            "first_step_final_head_segmentation": float(
                breakdown["final_head_segmentation"].detach().cpu()
            ),
            "first_step_aggregate_contrastive": float(
                breakdown["aggregate_contrastive"].detach().cpu()
            ),
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
            "temporary_checkpoint_removed": True,
            "checkpoint_bytes": saved_bytes,
            "restored_global_step": 1,
            "advanced_global_step": global_step,
            "advanced_model_bitwise_changed": advanced_model_changed,
            "advanced_optimizer_state_changed": advanced_optimizer_changed,
        },
        "validation_evaluator": evaluator,
    }
    if not report["smoke"]["encoder_bitwise_unchanged"]:
        raise AssertionError("frozen Concerto encoder changed during native smoke")
    _write_json(report, ARTIFACT_DIR / "native_smoke_report.json")
    return report


def run_tiny_overfit(checkpoint: Path, device: Any) -> dict[str, Any]:
    import torch

    seed_everything(SEED)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    config, system = _compose_runtime(checkpoint, device)
    data, targets, sample = _materialize_named_train_batch(
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
        "official_mixed_data_reproduction": False,
        "g2_evidence": False,
        "seed": SEED,
        "source_commit": _git_commit(),
        "checkpoint": checkpoint_provenance(checkpoint),
        "hardware": _hardware(device),
        "sample_name": TINY_SAMPLE_NAME,
        "sample": sample,
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
    _write_json(report, ARTIFACT_DIR / "tiny_overfit_report.json")
    markdown = render_tiny_overfit_markdown(
        report,
        sample_name=TINY_SAMPLE_NAME,
        elapsed_seconds=elapsed,
        peak_vram_mib=report["peak_allocated_vram_mib"],
    )
    _write_text(markdown, ARTIFACT_DIR / "tiny_overfit_report.md")
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
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the P2 native training checks")
    checkpoint_provenance(args.checkpoint)
    device = torch.device("cuda", args.device)
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")
    _hardware(device)
    if args.mode in {"all", "smoke"}:
        run_native_smoke(args.checkpoint, device)
    if args.mode in {"all", "tiny"}:
        run_tiny_overfit(args.checkpoint, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
