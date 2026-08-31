"""Manifest-bound runtime for ReScene decoder diagnostics."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import subprocess
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from scripts.evaluate_rescene_rootcause_checkpoint import (
    _load_json,
    _stable_file_identity,
    _validate_authorization,
    _validate_content_hash,
    compose_evaluation_config,
)
from scripts.evaluate_sonata_second_checkpoint import strict_load_task_checkpoint
from utils.rescene_rootcause_diagnostics import (
    attention_mask_records,
    query_conflict_records,
    query_initialization_records,
    superpoint_feature_records,
)
from utils.rescene_rootcause_evaluation import (
    RootCauseEvaluationError,
    validate_checkpoint_manifest_binding,
)
from utils.rescene_rootcause_preflight import canonical_sha256

DIAGNOSTIC_MODES = (
    "query_initialization",
    "query_conflicts",
    "attention_mask_recall",
    "superpoint_features",
)

QUERY_INITIALIZATION_FIELDS = (
    "record_type",
    "file_name",
    "num_queries",
    "foreground_query_fraction",
    "background_query_fraction",
    "gt_instance_count",
    "gt_instance_coverage",
    "query_content_norm_mean",
    "query_content_norm_max",
    "query_content_zero_fraction",
    "gt_instance_id",
    "gt_label",
    "size_points",
    "size_bin",
    "query_count",
    "covered_by_fps_query",
)
QUERY_CONFLICT_FIELDS = (
    "file_name",
    "decoder_prediction_layer",
    "feeds_next_attention",
    "query_count",
    "active_query_count",
    "gt_instance_count",
    "gt_coverage_iou25",
    "gt_coverage_iou50",
    "mean_queries_per_gt_iou25",
    "mean_queries_per_gt_iou50",
    "competed_active_query_fraction",
    "competing_query_pairwise_iou_mean",
    "distinct_gt_covered_iou25",
    "query_utilization_iou25",
    "distinct_gt_per_utilized_query",
)
ATTENTION_MASK_FIELDS = (
    "file_name",
    "decoder_prediction_layer",
    "query_id",
    "gt_instance_id",
    "match_iou",
    "gt_point_count",
    "allowed_gt_fraction",
    "masked_gt_fraction",
    "post_sample_all_masked_reset_count",
    "post_sample_query_count",
    "post_sample_reset_fraction",
)
SUPERPOINT_FEATURE_FIELDS = (
    "file_name",
    "gt_instance_id",
    "gt_label",
    "size_points",
    "size_bin",
    "segments_per_gt",
    "within_instance_feature_variance",
    "nearest_instance_cosine_margin",
    "mean_segment_purity",
    "mean_gt_instances_per_segment",
)
MODE_FIELDS = {
    "query_initialization": QUERY_INITIALIZATION_FIELDS,
    "query_conflicts": QUERY_CONFLICT_FIELDS,
    "attention_mask_recall": ATTENTION_MASK_FIELDS,
    "superpoint_features": SUPERPOINT_FEATURE_FIELDS,
}


class RootCauseDiagnosticCollector:
    """Capture one diagnostic without modifying the model's numerical path."""

    def __init__(self, mode: str) -> None:
        if mode not in DIAGNOSTIC_MODES:
            raise RootCauseEvaluationError("decoder diagnostic mode is invalid")
        self.mode = mode
        self.rows: list[dict[str, object]] = []
        self.sequence_count = 0
        self._file_names: Sequence[str] = ()
        self._initial_queries: list[dict[str, torch.Tensor]] = []
        self._reset_layers: list[list[dict[str, int]]] = []

    def install(self, system: Any) -> None:
        model = system.model
        original_validation_step = system.validation_step

        def validation_step(batch: Any, batch_idx: int) -> Any:
            if (
                not isinstance(batch, (tuple, list))
                or len(batch) != 3
                or not isinstance(batch[2], (tuple, list))
            ):
                raise RootCauseEvaluationError("diagnostic validation batch differs")
            self._file_names = tuple(str(name) for name in batch[2])
            self._initial_queries = []
            self._reset_layers = []
            return original_validation_step(batch, batch_idx)

        system.validation_step = validation_step

        if self.mode == "query_initialization":
            original_initialize_queries = model.initialize_queries

            def initialize_queries(*args: Any, **kwargs: Any) -> Any:
                result = original_initialize_queries(*args, **kwargs)
                pcd_features = kwargs.get("pcd_features")
                coords = kwargs.get("coords")
                if pcd_features is None and args:
                    pcd_features = args[0]
                if coords is None and len(args) > 1:
                    coords = args[1]
                queries, _, sampled_coords = result
                if (
                    sampled_coords is None
                    or not isinstance(coords, list)
                    or not isinstance(queries, torch.Tensor)
                ):
                    raise RootCauseEvaluationError("FPS query diagnostic capture differs")
                captured = []
                for batch_index, full_coordinates in enumerate(coords[-1]):
                    sampled = sampled_coords[batch_index]
                    matches = (
                        full_coordinates[:, None, :] == sampled[None, :, :]
                    ).all(dim=2)
                    if not torch.all(matches.sum(dim=0) == 1).item():
                        raise RootCauseEvaluationError("FPS query coordinate binding differs")
                    captured.append(
                        {
                            "sampled_indices": matches.long().argmax(dim=0).detach().cpu(),
                            "query_content_norms": queries[batch_index]
                            .norm(dim=-1)
                            .detach()
                            .cpu(),
                        }
                    )
                self._initial_queries = captured
                return result

            model.initialize_queries = initialize_queries

        if self.mode == "attention_mask_recall":
            original_sample_and_batch = model.sample_and_batch_features

            def sample_and_batch_features(*args: Any, **kwargs: Any) -> Any:
                result = original_sample_and_batch(*args, **kwargs)
                if kwargs.get("extra") is not None:
                    batched_attention = result[1]
                    if not isinstance(batched_attention, torch.Tensor):
                        raise RootCauseEvaluationError(
                            "attention reset diagnostic capture differs"
                        )
                    all_masked = batched_attention.sum(dim=1) == batched_attention.shape[1]
                    self._reset_layers.append(
                        [
                            {
                                "reset_count": int(all_masked[index].sum().item()),
                                "query_count": int(all_masked.shape[1]),
                            }
                            for index in range(all_masked.shape[0])
                        ]
                    )
                return result

            model.sample_and_batch_features = sample_and_batch_features

        original_forward = system.forward

        def forward(*args: Any, **kwargs: Any) -> Any:
            output = original_forward(*args, **kwargs)
            self._forward_hook(system, args, kwargs, output)
            return output

        system.forward = forward

    def _forward_hook(
        self,
        module: Any,
        args: Sequence[Any],
        kwargs: Mapping[str, Any],
        output: Mapping[str, Any],
    ) -> None:
        targets = kwargs.get("targets")
        if (
            not isinstance(targets, list)
            or len(targets) != 1
            or len(self._file_names) != 1
            or not isinstance(output, Mapping)
        ):
            raise RootCauseEvaluationError("diagnostic forward batch differs")
        target = targets[0]
        if not isinstance(target, Mapping):
            raise RootCauseEvaluationError("diagnostic target differs")
        file_name = self._file_names[0]
        if self.mode == "query_initialization":
            if len(self._initial_queries) != 1:
                raise RootCauseEvaluationError("FPS query diagnostic capture differs")
            capture = self._initial_queries[0]
            records = query_initialization_records(
                file_name=file_name,
                sampled_indices=capture["sampled_indices"],
                query_content_norms=capture["query_content_norms"],
                target=target,
            )
        elif self.mode == "query_conflicts":
            records = query_conflict_records(
                file_name=file_name, output=output, target=target
            )
        elif self.mode == "attention_mask_recall":
            if any(len(layer) != 1 for layer in self._reset_layers):
                raise RootCauseEvaluationError("attention reset diagnostic capture differs")
            records = attention_mask_records(
                file_name=file_name,
                output=output,
                target=target,
                reset_counts=[layer[0] for layer in self._reset_layers],
            )
        else:
            segment_features = output.get("segment_features")
            if (
                not isinstance(segment_features, list)
                or len(segment_features) != 1
                or not isinstance(segment_features[0], list)
                or len(segment_features[0]) != 1
            ):
                raise RootCauseEvaluationError("superpoint diagnostic capture differs")
            records = superpoint_feature_records(
                file_name=file_name,
                segment_features=segment_features[0][0],
                target=target,
            )
        self.rows.extend(records)
        self.sequence_count += 1


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def diagnostic_contract(mode: str) -> dict[str, object]:
    contracts = {
        "query_initialization": {
            "fps_binding": "exact full-resolution decoder-coordinate index",
            "foreground": "union of evaluated low-resolution GT instance masks",
            "size_bins_points": ["<100", "100-999", ">=1000"],
        },
        "query_conflicts": {
            "mask_threshold": 0.5,
            "active_query": "nonempty thresholded point mask",
            "competition_assignment": "best full-point IoU at IoU>=0.25",
            "higher_score": "maximum foreground class probability",
        },
        "attention_mask_recall": {
            "matching": "maximum-total full-point IoU one-to-one",
            "allowed_threshold": 0.5,
            "recall_scope": "pre-pooling source mask",
            "reset_scope": "actual post-sampling all-memory-masked condition",
        },
        "superpoint_features": {
            "feature_normalization": "L2",
            "within_variance": "mean squared L2 distance to GT segment centroid",
            "nearest_margin": "one minus nearest GT centroid cosine similarity",
            "purity": "maximum evaluated-GT point fraction per segment",
        },
    }
    payload: dict[str, object] = {
        "schema_version": 1,
        "mode": mode,
        "scope": "project_diagnostic_not_external_method_metric",
        "formula": contracts[mode],
    }
    payload["sha256"] = canonical_sha256(payload)
    return payload


def run_decoder_diagnostic(
    *,
    mode: str,
    checkpoint_path: Path,
    checkpoint_manifest_path: Path,
    authorization_path: Path,
    pretrained_path: Path,
    device_index: int,
    limit_val_batches: int | None = None,
) -> dict[str, object]:
    """Run one manifest-bound diagnostic over the official-like T2 split."""

    import hydra
    from pytorch_lightning import Trainer, seed_everything

    from trainer.trainer import InstanceSegmentation

    if mode not in DIAGNOSTIC_MODES:
        raise RootCauseEvaluationError("decoder diagnostic mode is invalid")
    if limit_val_batches is not None and limit_val_batches <= 0:
        raise RootCauseEvaluationError("diagnostic batch limit must be positive")
    if not torch.cuda.is_available() or not 0 <= device_index < torch.cuda.device_count():
        raise RootCauseEvaluationError("diagnostic device is unavailable")
    manifest = _load_json(checkpoint_manifest_path, name="checkpoint manifest")
    _validate_content_hash(manifest, name="checkpoint manifest")
    authorization = _load_json(authorization_path, name="variant authorization")
    _validate_authorization(authorization)
    variant = validate_checkpoint_manifest_binding(
        manifest, authorization=authorization
    )
    checkpoint_identity = _stable_file_identity(checkpoint_path)
    if any(
        checkpoint_identity[field] != manifest["checkpoint"][field]
        for field in ("bytes", "sha256")
    ):
        raise RootCauseEvaluationError("diagnostic checkpoint differs from manifest")
    pretrained_identity = _stable_file_identity(pretrained_path)
    expected_pretrained = authorization["initialization"]["pretrained"]
    if any(
        pretrained_identity[field] != expected_pretrained[field]
        for field in ("bytes", "sha256")
    ):
        raise RootCauseEvaluationError("diagnostic pretrained encoder differs")

    seed_everything(45, workers=True)
    config = compose_evaluation_config(pretrained_path)
    system = InstanceSegmentation(config)
    strict_load = strict_load_task_checkpoint(system, checkpoint_path)
    if _stable_file_identity(checkpoint_path) != checkpoint_identity:
        raise RootCauseEvaluationError("diagnostic checkpoint changed while loading")
    validation_dataset = hydra.utils.instantiate(config.data.validation_dataset)
    if len(validation_dataset) != 154:
        raise RootCauseEvaluationError("diagnostic validation split differs")
    if limit_val_batches is not None and limit_val_batches > len(validation_dataset):
        raise RootCauseEvaluationError("diagnostic batch limit exceeds validation split")
    system.validation_dataset = validation_dataset
    system.labels_info = validation_dataset.label_info
    collector = RootCauseDiagnosticCollector(mode)
    collector.install(system)
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
        default_root_dir="checkpoints/rootcause_diagnostics",
        deterministic=bool(config.trainer.deterministic),
        precision="32-true",
        limit_val_batches=(
            limit_val_batches if limit_val_batches is not None else 1.0
        ),
    )
    started = time.perf_counter()
    trainer.validate(system, dataloaders=system.val_dataloader(), verbose=False)
    elapsed_seconds = time.perf_counter() - started
    expected_sequences = limit_val_batches or 154
    if collector.sequence_count != expected_sequences or not collector.rows:
        raise RootCauseEvaluationError("diagnostic sequence coverage differs")
    contract = diagnostic_contract(mode)
    return {
        "schema_version": 1,
        "status": "pass",
        "scope": "smoke" if limit_val_batches is not None else "official_like_t2",
        "mode": mode,
        "variant": variant,
        "completed_epoch": manifest["checkpoint"]["selected_epoch"],
        "seed": 45,
        "source_commit": _git_head(),
        "contract": contract,
        "checkpoint_sha256": checkpoint_identity["sha256"],
        "checkpoint_manifest_sha256": manifest["content_sha256"],
        "variant_authorization_sha256": authorization["authorization_sha256"],
        "validation_sequence_count": collector.sequence_count,
        "row_count": len(collector.rows),
        "elapsed_seconds": elapsed_seconds,
        "strict_load": strict_load,
        "rows": collector.rows,
    }


def _csv_bytes(rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("ascii")


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True)
        + "\n"
    ).encode("ascii")


def _publish(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_file() and not path.is_symlink() and path.read_bytes() == payload:
            return
        raise RootCauseEvaluationError("refusing to overwrite diagnostic output")
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def publish_diagnostic(
    *, result: Mapping[str, Any], csv_path: Path, manifest_path: Path
) -> dict[str, object]:
    mode = result.get("mode")
    rows = result.get("rows")
    if mode not in MODE_FIELDS or not isinstance(rows, list):
        raise RootCauseEvaluationError("diagnostic output schema differs")
    csv_payload = _csv_bytes(rows, MODE_FIELDS[mode])
    manifest = dict(result)
    manifest.pop("rows")
    manifest["csv_sha256"] = hashlib.sha256(csv_payload).hexdigest()
    manifest["content_sha256"] = canonical_sha256(manifest)
    _publish(csv_path, csv_payload)
    _publish(manifest_path, _json_bytes(manifest))
    return manifest
