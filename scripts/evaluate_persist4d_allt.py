#!/usr/bin/env python3
"""Evaluate one Persist4D checkpoint continuously at T2-T5."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_METADATA = Path("/home/ww/3RScan.json")
DEFAULT_PRETRAINED = Path(
    "/mnt/shared/ww/persist4d-node1-7-migration/checkpoints/concerto_base.pth"
)
DEFAULT_SPLIT_MANIFEST = (
    PROJECT_ROOT / "artifacts/allt_task_superiority_v1/split_manifest.json"
)
DEFAULT_CACHE_ROOT = Path(
    "/mnt/shared/ww/persist4d-allt-task-superiority-v1/evaluation_cache"
)

REPORT_HORIZONS = (2, 3, 4, 5)
ORDER_IDS = ("canonical", "reverse", "sha256_seed45")
WINDOW_MODES = ("local_pair", "full_history")
RESULT_FIELDS = (
    "population_id",
    "model",
    "checkpoint_sha256",
    "training_seed",
    "evaluation_seed",
    "method",
    "reducer",
    "T",
    "t_mAP",
    "t_mAP50",
    "t_mAP25",
    "t_REC",
    "prefix_overall_mAP",
    "local_current_AP",
    "num_master",
    "num_order_units",
    "num_reference_clusters",
)
_CACHE_KEY_FIELDS = {
    "schema_version",
    "checkpoint_sha256",
    "module_config_sha256",
    "population_id",
    "evaluation_seed",
    "postprocess_version",
    "master_sequence_id",
    "reference_scene_id",
    "order_id",
    "window_mode",
    "stage_requests",
}
_STAGE_REQUEST_FIELDS = {
    "master_sequence_id",
    "reference_scene_id",
    "order_id",
    "context_index",
    "context_scan_indices",
    "stage_index",
    "history_scan_ids",
    "history_scan_indices",
    "inference_scan_ids",
    "inference_scan_indices",
    "window_mode",
}


class AllTEvaluationError(ValueError):
    """Raised when evaluation could mix checkpoints, prefixes, or populations."""


def _nonempty(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise AllTEvaluationError(f"{name} must be a non-empty string")
    return value


def _sha256(value: object, *, name: str) -> str:
    text = _nonempty(value, name=name)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise AllTEvaluationError(f"{name} must be a lowercase SHA-256")
    return text


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise AllTEvaluationError(f"{name} must be an integer >= {minimum}")
    return value


def _resolve_metric_reducers(
    *, window_mode: str, requested: Sequence[str] | None
) -> tuple[str, ...]:
    available = (
        ("official",)
        if window_mode == "full_history"
        else ("mean", "latest", "max")
    )
    if requested is None:
        return available
    if isinstance(requested, (str, bytes)) or not requested:
        raise AllTEvaluationError("metric reducers must be a non-empty sequence")
    reducers = tuple(_nonempty(value, name="metric reducer") for value in requested)
    if len(set(reducers)) != len(reducers):
        raise AllTEvaluationError("metric reducers must be unique")
    if any(reducer not in available for reducer in reducers):
        raise AllTEvaluationError(
            f"requested metric reducer is not available for {window_mode}"
        )
    return reducers


def _checkpoint_identity(
    payload: Mapping[str, object], *, variant: str
) -> tuple[int, int]:
    expected_variant = _nonempty(variant, name="variant")
    step = _integer(payload.get("global_step", 0), name="checkpoint global_step")
    metadata = payload.get("allt_metadata")
    hparams = payload.get("hyper_parameters")
    training_seed: object | None = None
    checkpoint_variant: object | None = None
    if isinstance(metadata, Mapping):
        training_seed = metadata.get("training_seed")
        checkpoint_variant = metadata.get("variant")
    if isinstance(hparams, Mapping):
        general = hparams.get("general")
        allt_training = hparams.get("allt_training")
        if training_seed is None and isinstance(general, Mapping):
            training_seed = general.get("seed")
        if checkpoint_variant is None and isinstance(allt_training, Mapping):
            checkpoint_variant = allt_training.get("variant")
    if checkpoint_variant is not None and checkpoint_variant != expected_variant:
        raise AllTEvaluationError("checkpoint variant metadata differs")
    if training_seed is None:
        raise AllTEvaluationError("checkpoint lacks bound training seed metadata")
    return step, _integer(training_seed, name="training seed")


def _compute_metric_values(
    accumulators: Mapping[tuple[str, int], object],
    *,
    keys: Sequence[tuple[str, int]],
    workers: int,
) -> dict[tuple[str, int], Mapping[str, float]]:
    if (
        isinstance(workers, bool)
        or not isinstance(workers, int)
        or not 1 <= workers <= 32
    ):
        raise AllTEvaluationError("metric workers must be within 1-32")
    ordered_keys = tuple(keys)
    if not ordered_keys or len(set(ordered_keys)) != len(ordered_keys):
        raise AllTEvaluationError("metric compute keys must be non-empty and unique")
    if any(key not in accumulators for key in ordered_keys):
        raise AllTEvaluationError("metric compute key lacks an accumulator")

    def compute(key: tuple[str, int]) -> tuple[tuple[str, int], Mapping[str, float]]:
        values = accumulators[key].compute()
        if not isinstance(values, Mapping):
            raise AllTEvaluationError("metric accumulator result must be a mapping")
        return key, values

    with ThreadPoolExecutor(max_workers=min(workers, len(ordered_keys))) as executor:
        return dict(executor.map(compute, ordered_keys))


def _compact_metric_bundle(
    value: Mapping[str, object], *, reducers: Sequence[str]
) -> dict[str, object]:
    key = value.get("key")
    pairs = value.get("pairs")
    selected = tuple(reducers)
    if not isinstance(key, Mapping) or not isinstance(pairs, Mapping):
        raise AllTEvaluationError("metric bundle lacks its key or pairs")
    if not selected or len(set(selected)) != len(selected):
        raise AllTEvaluationError("metric bundle reducers must be non-empty and unique")
    if any(reducer not in pairs for reducer in selected):
        raise AllTEvaluationError("metric bundle lacks a requested reducer")
    return {
        "key": key,
        "pairs": {reducer: pairs[reducer] for reducer in selected},
    }


def _string_list(value: object, *, name: str) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise AllTEvaluationError(f"{name} must be a non-empty sequence")
    result = [_nonempty(item, name=f"{name} item") for item in value]
    if len(set(result)) != len(result):
        raise AllTEvaluationError(f"{name} must contain unique values")
    return result


def _index_list(value: object, *, name: str) -> list[int]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise AllTEvaluationError(f"{name} must be a non-empty sequence")
    result = [_integer(item, name=f"{name} item") for item in value]
    if len(set(result)) != len(result):
        raise AllTEvaluationError(f"{name} must contain unique values")
    return result


def build_stage_requests(
    *,
    master_sequence_id: str,
    reference_scene_id: str,
    order_id: str,
    scan_ids: Sequence[str],
    scan_indices: Sequence[int],
    context_index: int,
    window_mode: str,
    context_scan_indices: Sequence[int] | None = None,
) -> tuple[dict[str, object], ...]:
    """Build exact T1-T5 requests without resetting the episode state."""

    master = _nonempty(master_sequence_id, name="master_sequence_id")
    reference = _nonempty(reference_scene_id, name="reference_scene_id")
    if order_id not in ORDER_IDS:
        raise AllTEvaluationError("order_id is not preregistered")
    if window_mode not in WINDOW_MODES:
        raise AllTEvaluationError("window_mode is invalid")
    ids = _string_list(scan_ids, name="scan_ids")
    indices = _index_list(scan_indices, name="scan_indices")
    if len(ids) != 5 or len(indices) != 5:
        raise AllTEvaluationError("evaluation requires one real five-scan master")
    context_indices = _index_list(
        indices if context_scan_indices is None else context_scan_indices,
        name="context_scan_indices",
    )
    if len(context_indices) != 5 or set(context_indices) != set(indices):
        raise AllTEvaluationError("context indices must contain the ordered scan indices")
    context = _integer(context_index, name="context_index")
    result = []
    for stage_index in range(5):
        history_ids = ids[: stage_index + 1]
        history_indices = indices[: stage_index + 1]
        window_start = 0 if window_mode == "full_history" else max(0, stage_index - 1)
        result.append(
            {
                "master_sequence_id": master,
                "reference_scene_id": reference,
                "order_id": order_id,
                "context_index": context,
                "context_scan_indices": list(context_indices),
                "stage_index": stage_index,
                "history_scan_ids": list(history_ids),
                "history_scan_indices": list(history_indices),
                "inference_scan_ids": list(history_ids[window_start:]),
                "inference_scan_indices": list(history_indices[window_start:]),
                "window_mode": window_mode,
            }
        )
    return tuple(result)


def _validate_stage_requests(value: object) -> tuple[dict[str, object], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 5:
        raise AllTEvaluationError("stage_requests must cover exact T1-T5")
    normalized = []
    for expected_stage, raw in enumerate(value):
        if not isinstance(raw, Mapping) or set(raw) != _STAGE_REQUEST_FIELDS:
            raise AllTEvaluationError("stage request fields differ")
        request = dict(raw)
        master = _nonempty(request["master_sequence_id"], name="master_sequence_id")
        reference = _nonempty(
            request["reference_scene_id"], name="reference_scene_id"
        )
        order = _nonempty(request["order_id"], name="order_id")
        if order not in ORDER_IDS:
            raise AllTEvaluationError("stage request order is not preregistered")
        window_mode = _nonempty(request["window_mode"], name="window_mode")
        if window_mode not in WINDOW_MODES:
            raise AllTEvaluationError("stage request window mode is invalid")
        stage = _integer(request["stage_index"], name="stage_index")
        if stage != expected_stage:
            raise AllTEvaluationError("stage requests are not consecutive T1-T5")
        context = _integer(request["context_index"], name="context_index")
        context_indices = _index_list(
            request["context_scan_indices"], name="context_scan_indices"
        )
        if len(context_indices) != 5:
            raise AllTEvaluationError("context_scan_indices must contain five scans")
        history_ids = _string_list(request["history_scan_ids"], name="history_scan_ids")
        history_indices = _index_list(
            request["history_scan_indices"], name="history_scan_indices"
        )
        inference_ids = _string_list(
            request["inference_scan_ids"], name="inference_scan_ids"
        )
        inference_indices = _index_list(
            request["inference_scan_indices"], name="inference_scan_indices"
        )
        if len(history_ids) != stage + 1 or len(history_indices) != stage + 1:
            raise AllTEvaluationError("history does not end at its stage")
        expected_start = 0 if window_mode == "full_history" else max(0, stage - 1)
        if (
            inference_ids != history_ids[expected_start:]
            or inference_indices != history_indices[expected_start:]
        ):
            raise AllTEvaluationError("inference window differs from the causal contract")
        normalized.append(
            {
                "master_sequence_id": master,
                "reference_scene_id": reference,
                "order_id": order,
                "context_index": context,
                "context_scan_indices": context_indices,
                "stage_index": stage,
                "history_scan_ids": history_ids,
                "history_scan_indices": history_indices,
                "inference_scan_ids": inference_ids,
                "inference_scan_indices": inference_indices,
                "window_mode": window_mode,
            }
        )
    identity_fields = (
        "master_sequence_id",
        "reference_scene_id",
        "order_id",
        "context_index",
        "context_scan_indices",
        "window_mode",
    )
    if any(
        request[field] != normalized[0][field]
        for request in normalized[1:]
        for field in identity_fields
    ):
        raise AllTEvaluationError("stage requests do not belong to one episode")
    for stage, request in enumerate(normalized):
        final = normalized[-1]
        if (
            request["history_scan_ids"] != final["history_scan_ids"][: stage + 1]
            or request["history_scan_indices"]
            != final["history_scan_indices"][: stage + 1]
        ):
            raise AllTEvaluationError("stage histories are not nested prefixes")
    if set(normalized[-1]["history_scan_indices"]) != set(
        normalized[-1]["context_scan_indices"]
    ):
        raise AllTEvaluationError("ordered scans differ from the dataset context")
    return tuple(normalized)


def build_sequence_cache_key(
    *,
    checkpoint_sha256: str,
    module_config_sha256: str,
    population_id: str,
    evaluation_seed: int,
    postprocess_version: str,
    stage_requests: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    stages = _validate_stage_requests(stage_requests)
    first = stages[0]
    return {
        "schema_version": 1,
        "checkpoint_sha256": _sha256(
            checkpoint_sha256, name="checkpoint_sha256"
        ),
        "module_config_sha256": _sha256(
            module_config_sha256, name="module_config_sha256"
        ),
        "population_id": _nonempty(population_id, name="population_id"),
        "evaluation_seed": _integer(evaluation_seed, name="evaluation_seed"),
        "postprocess_version": _nonempty(
            postprocess_version, name="postprocess_version"
        ),
        "master_sequence_id": first["master_sequence_id"],
        "reference_scene_id": first["reference_scene_id"],
        "order_id": first["order_id"],
        "window_mode": first["window_mode"],
        "stage_requests": [dict(request) for request in stages],
    }


def _validate_sequence_cache_key(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != _CACHE_KEY_FIELDS:
        raise AllTEvaluationError("sequence cache key fields differ")
    key = dict(value)
    if key["schema_version"] != 1:
        raise AllTEvaluationError("sequence cache key version differs")
    stages = _validate_stage_requests(key["stage_requests"])
    first = stages[0]
    _sha256(key["checkpoint_sha256"], name="checkpoint_sha256")
    _sha256(key["module_config_sha256"], name="module_config_sha256")
    _nonempty(key["population_id"], name="population_id")
    _integer(key["evaluation_seed"], name="evaluation_seed")
    _nonempty(key["postprocess_version"], name="postprocess_version")
    for field in ("master_sequence_id", "reference_scene_id", "order_id", "window_mode"):
        if key[field] != first[field]:
            raise AllTEvaluationError("sequence cache identity differs from stage requests")
    key["stage_requests"] = [dict(request) for request in stages]
    return key


def sequence_cache_key_sha256(value: Mapping[str, object]) -> str:
    key = _validate_sequence_cache_key(value)
    encoded = json.dumps(
        key,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _metric_rows_by_checkpoint(
    rows: Sequence[Mapping[str, object]], *, name: str
) -> dict[str, dict[int, float]]:
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence) or not rows:
        raise AllTEvaluationError(f"{name} rows must be non-empty")
    result: dict[str, dict[int, float]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise AllTEvaluationError(f"{name} row must be a mapping")
        checkpoint = _sha256(row.get("checkpoint_sha256"), name="checkpoint_sha256")
        horizon = _integer(row.get("T"), name="T")
        raw_value = row.get("t_mAP")
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise AllTEvaluationError("t_mAP must be numeric")
        value = float(raw_value)
        if not math.isfinite(value):
            raise AllTEvaluationError("t_mAP must be finite")
        by_horizon = result.setdefault(checkpoint, {})
        if horizon in by_horizon:
            raise AllTEvaluationError("checkpoint contains a duplicate horizon")
        by_horizon[horizon] = value
    if any(tuple(sorted(values)) != REPORT_HORIZONS for values in result.values()):
        raise AllTEvaluationError("each checkpoint must cover exactly T2-T5")
    return result


def rank_development_checkpoints(
    candidate_rows: Sequence[Mapping[str, object]],
    baseline_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    candidates = _metric_rows_by_checkpoint(candidate_rows, name="candidate")
    baselines = _metric_rows_by_checkpoint(baseline_rows, name="baseline")
    if len(baselines) != 1:
        raise AllTEvaluationError("development selection requires one fixed baseline checkpoint")
    baseline = next(iter(baselines.values()))
    ranked = []
    for checkpoint, values in candidates.items():
        deltas = {horizon: values[horizon] - baseline[horizon] for horizon in REPORT_HORIZONS}
        ranked.append(
            {
                "checkpoint_sha256": checkpoint,
                "minimum_t_map_delta": min(deltas.values()),
                "mean_t_map": sum(values.values()) / len(REPORT_HORIZONS),
                "all_t_positive": all(value > 0.0 for value in deltas.values()),
                "t_map_delta_by_horizon": {
                    str(horizon): deltas[horizon] for horizon in REPORT_HORIZONS
                },
            }
        )
    ranked.sort(
        key=lambda row: (
            -float(row["minimum_t_map_delta"]),
            -float(row["mean_t_map"]),
            str(row["checkpoint_sha256"]),
        )
    )
    return ranked


def validate_result_rows(rows: Sequence[Mapping[str, object]]) -> None:
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence) or not rows:
        raise AllTEvaluationError("result rows must be non-empty")
    groups: dict[tuple[object, ...], set[int]] = {}
    metric_fields = (
        "t_mAP",
        "t_mAP50",
        "t_mAP25",
        "t_REC",
        "prefix_overall_mAP",
        "local_current_AP",
    )
    for raw in rows:
        if not isinstance(raw, Mapping) or set(raw) != set(RESULT_FIELDS):
            raise AllTEvaluationError("result row fields differ")
        checkpoint = _sha256(raw["checkpoint_sha256"], name="checkpoint_sha256")
        horizon = _integer(raw["T"], name="T")
        for field in metric_fields:
            value = raw[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise AllTEvaluationError(f"{field} must be finite numeric evidence")
        for field in ("num_master", "num_order_units", "num_reference_clusters"):
            _integer(raw[field], name=field, minimum=1)
        identity = (
            _nonempty(raw["population_id"], name="population_id"),
            _nonempty(raw["model"], name="model"),
            checkpoint,
            _integer(raw["training_seed"], name="training_seed"),
            _integer(raw["evaluation_seed"], name="evaluation_seed"),
            _nonempty(raw["method"], name="method"),
            _nonempty(raw["reducer"], name="reducer"),
        )
        horizons = groups.setdefault(identity, set())
        if horizon in horizons:
            raise AllTEvaluationError("result group contains a duplicate horizon")
        horizons.add(horizon)
    if any(tuple(sorted(horizons)) != REPORT_HORIZONS for horizons in groups.values()):
        raise AllTEvaluationError("each result group must cover exactly T2-T5")


@dataclass(frozen=True)
class EvaluationSequence:
    stage_requests: tuple[dict[str, object], ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "stage_requests", _validate_stage_requests(self.stage_requests)
        )


def _canonical_json_sha256(value: object) -> str:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise AllTEvaluationError("value is not canonical portable JSON") from error
    return hashlib.sha256(encoded).hexdigest()


def _new_model_forward_count(
    *, window_mode: str, sequence_count: int, reused_count: int
) -> int:
    if window_mode not in WINDOW_MODES:
        raise AllTEvaluationError("forward count window mode is invalid")
    if (
        isinstance(sequence_count, bool)
        or not isinstance(sequence_count, int)
        or isinstance(reused_count, bool)
        or not isinstance(reused_count, int)
        or not 0 <= reused_count <= sequence_count
    ):
        raise AllTEvaluationError("forward count inputs are invalid")
    per_sequence = 5 if window_mode == "local_pair" else sum(range(1, 6))
    return (sequence_count - reused_count) * per_sequence


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if len(result) != 40:
        raise AllTEvaluationError("Git HEAD is invalid")
    return result


def _load_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AllTEvaluationError(f"cannot load JSON: {path}") from error
    if not isinstance(value, dict):
        raise AllTEvaluationError(f"JSON root must be an object: {path}")
    return value


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise AllTEvaluationError(f"output cannot be a symlink: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    _atomic_write(
        path,
        (
            json.dumps(
                payload,
                allow_nan=False,
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("ascii"),
    )


def _csv_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    validate_result_rows(rows)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=RESULT_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _resolved_data_path(value: object, data_root: Path) -> object:
    if not isinstance(value, str) or not value.startswith("data/"):
        return value
    return str(data_root / Path(value).relative_to("data"))


def _rio_evaluation_dataset(config: Any, *, data_root: Path, split: str):
    import hydra
    from omegaconf import OmegaConf

    if split == "train":
        from datasets.semseg import SemanticSegmentationDataset

        training = OmegaConf.to_container(config.data.train_dataset, resolve=True)
        if not isinstance(training, dict) or not isinstance(
            training.get("datasets"), list
        ):
            raise AllTEvaluationError("training data config is unavailable")
        excluded = {
            "_target_",
            "datasets",
            "epoch_sample_multiple",
            "sampler_seed",
            "weights",
        }
        common = {key: value for key, value in training.items() if key not in excluded}
        candidates = [
            item
            for item in training["datasets"]
            if isinstance(item, dict) and item.get("dataset_name") == "rio"
        ]
        if len(candidates) != 1:
            raise AllTEvaluationError("training config does not bind one RIO source")
        parameters = {**common, **candidates[0]}
        parameters.pop("target", None)
        parameters.pop("_target_", None)
        parameters.update(
            {
                "image_augmentations_path": None,
                "mode": "train",
                "temporal_window": 5,
                "volume_augmentations_path": None,
            }
        )
        for key in (
            "data_dir",
            "label_db_filepath",
            "change_label_db_filepath",
            "color_mean_std",
        ):
            if key in parameters:
                parameters[key] = _resolved_data_path(parameters[key], data_root)
        dataset = SemanticSegmentationDataset(**parameters)
    elif split == "validation":
        dataset_config = OmegaConf.create(
            OmegaConf.to_container(config.data.validation_dataset, resolve=True)
        )
        dataset_config.temporal_window = 5
        for key in (
            "data_dir",
            "label_db_filepath",
            "change_label_db_filepath",
            "color_mean_std",
        ):
            if key in dataset_config:
                dataset_config[key] = _resolved_data_path(dataset_config[key], data_root)
        dataset = hydra.utils.instantiate(dataset_config)
    else:
        raise AllTEvaluationError("evaluation split must be train or validation")
    collate = hydra.utils.instantiate(config.data.validation_collation)
    return dataset, collate


def _validate_dataset_context(dataset: object, sequence: EvaluationSequence) -> None:
    request = sequence.stage_requests[0]
    context_index = int(request["context_index"])
    names = getattr(dataset, "sequence_names", None)
    indices = getattr(dataset, "sequence_indices", None)
    if (
        not isinstance(names, (list, tuple))
        or indices is None
        or context_index >= len(names)
        or names[context_index] != request["master_sequence_id"]
    ):
        raise AllTEvaluationError("dataset context identity differs from evaluation")
    try:
        observed = tuple(int(value) for value in indices[context_index])
    except (IndexError, TypeError, ValueError) as error:
        raise AllTEvaluationError("dataset context indices are unavailable") from error
    if observed != tuple(request["context_scan_indices"]):
        raise AllTEvaluationError("dataset context indices differ from evaluation")


def _development_sequences(
    *, data_root: Path, metadata: Path, split_manifest: Path
) -> tuple[EvaluationSequence, ...]:
    from scripts.p6a_protocol import load_t5_masters

    split = _load_json(split_manifest)
    assignments = split.get("master_assignments")
    if not isinstance(assignments, list):
        raise AllTEvaluationError("split manifest lacks master assignments")
    development_ids = {
        item.get("sequence_id")
        for item in assignments
        if isinstance(item, Mapping) and item.get("role") == "development"
    }
    masters = load_t5_masters(
        data_root / "processed/rio/sequence_database_sliding_5.yaml",
        data_root / "processed/rio/train_database.yaml",
        metadata_path=metadata,
        expected_split="train",
        expected_master_count=262,
        expected_cluster_count=44,
        require_supervised=True,
        substitution_policy="reject",
    )
    selected = [master for master in masters if master.sequence_id in development_ids]
    if (
        len(selected) != 47
        or len({master.reference_scene_id for master in selected}) != 8
        or {master.sequence_id for master in selected} != development_ids
    ):
        raise AllTEvaluationError("development population differs from the frozen split")
    return tuple(
        EvaluationSequence(
            build_stage_requests(
                master_sequence_id=master.sequence_id,
                reference_scene_id=master.reference_scene_id,
                order_id="canonical",
                scan_ids=master.scan_ids,
                scan_indices=master.scan_indices,
                context_index=master.validation_index,
                context_scan_indices=master.scan_indices,
                window_mode="local_pair",
            )
        )
        for master in sorted(selected, key=lambda item: item.sequence_id)
    )


def _protocol_b_sequences(
    *, data_root: Path, metadata: Path, window_mode: str
) -> tuple[EvaluationSequence, ...]:
    from scripts.p6a_protocol import build_protocol_b

    protocol = build_protocol_b(
        data_root / "processed/rio/sequence_database_sliding_5.yaml",
        data_root / "processed/rio/validation_database.yaml",
        metadata_path=metadata,
        expected_split="validation",
        expected_master_count=43,
        expected_cluster_count=6,
        horizons=REPORT_HORIZONS,
        seed=45,
        require_supervised=True,
        substitution_policy="reject",
    )
    sequences = []
    for master in protocol.masters:
        for order_id in protocol.order_variants:
            variant = protocol.variants[master.sequence_id][order_id]
            sequences.append(
                EvaluationSequence(
                    build_stage_requests(
                        master_sequence_id=master.sequence_id,
                        reference_scene_id=master.reference_scene_id,
                        order_id=order_id,
                        scan_ids=variant.scan_ids,
                        scan_indices=variant.scan_indices,
                        context_index=master.validation_index,
                        context_scan_indices=master.scan_indices,
                        window_mode=window_mode,
                    )
                )
            )
    if len(sequences) != 129:
        raise AllTEvaluationError("Protocol-B population must contain 129 order units")
    return tuple(sequences)


def _population(
    *,
    config: Any,
    population: str,
    window_mode: str,
    data_root: Path,
    metadata: Path,
    split_manifest: Path,
) -> tuple[object, object, tuple[EvaluationSequence, ...], str, str]:
    if population == "development":
        if window_mode == "full_history":
            local_sequences = _development_sequences(
                data_root=data_root,
                metadata=metadata,
                split_manifest=split_manifest,
            )
            sequences = tuple(
                EvaluationSequence(
                    build_stage_requests(
                        master_sequence_id=item.stage_requests[0][
                            "master_sequence_id"
                        ],
                        reference_scene_id=item.stage_requests[0][
                            "reference_scene_id"
                        ],
                        order_id="canonical",
                        scan_ids=item.stage_requests[-1]["history_scan_ids"],
                        scan_indices=item.stage_requests[-1]["history_scan_indices"],
                        context_index=item.stage_requests[0]["context_index"],
                        context_scan_indices=item.stage_requests[0][
                            "context_scan_indices"
                        ],
                        window_mode="full_history",
                    )
                )
                for item in local_sequences
            )
        else:
            sequences = _development_sequences(
                data_root=data_root,
                metadata=metadata,
                split_manifest=split_manifest,
            )
        split = "train"
        population_id = "development_train_holdout_47_masters_canonical"
    elif population == "protocol_b":
        sequences = _protocol_b_sequences(
            data_root=data_root, metadata=metadata, window_mode=window_mode
        )
        split = "validation"
        population_id = "protocol_b_43_masters_3_orders"
    else:
        raise AllTEvaluationError("population must be development or protocol_b")
    dataset, collate = _rio_evaluation_dataset(
        config, data_root=data_root, split=split
    )
    for sequence in sequences:
        _validate_dataset_context(dataset, sequence)
    population_sha256 = _canonical_json_sha256(
        [sequence.stage_requests for sequence in sequences]
    )
    return dataset, collate, sequences, population_id, population_sha256


def _module_config(variant: str) -> tuple[dict[str, object], str]:
    from scripts.train_persist4d_allt import VARIANTS

    if variant not in VARIANTS:
        raise AllTEvaluationError("variant is not an All-T model")
    specification = dict(VARIANTS[variant])
    document = {
        "adapter_insertion": "after_first_complete_decoder_pass",
        "local_enhancement": (
            {
                "kind": "qcl_inspired_current_prediction",
                "overlap_threshold": 0.25,
                "support_budget": 4096,
            }
            if specification["local_enhancement_enabled"]
            else None
        ),
        "memory": {
            "association_threshold": 0.5,
            "capacity": 100,
            "class_weight": 0.25,
            "confidence_threshold": 0.5,
            "mask_threshold": 0.5,
            "maximum_update_rate": 0.2,
            "minimum_mask_support": 1,
            "update_rate": 0.2,
        },
        "postprocess": "official_rescene_candidate_lineage_v1",
        "score_reducers": (
            ["official"]
            if specification["window_mode"] == "full_history"
            else ["mean", "latest", "max"]
        ),
        "variant": variant,
        **specification,
    }
    return document, _canonical_json_sha256(document)


def _load_system(
    *, variant: str, checkpoint: Path, pretrained: Path, device: object, scratch: Path
) -> tuple[object, Any, int, int]:
    import torch

    from scripts.train_persist4d_allt import _compose_config
    from trainer.persist4d_allt_trainer import Persist4DAllTTrainer

    config = _compose_config(
        variant=variant,
        pretrained=pretrained,
        run_dir=scratch,
        optimizer_updates=400,
        devices=1,
        gradient_accumulation=4,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or not isinstance(
        payload.get("state_dict"), Mapping
    ):
        raise AllTEvaluationError("checkpoint lacks a Lightning state_dict")
    checkpoint_step, training_seed = _checkpoint_identity(payload, variant=variant)
    system = Persist4DAllTTrainer(config)
    incompatible = system.load_state_dict(payload["state_dict"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise AllTEvaluationError("checkpoint state differs from the variant model")
    system.to(device)
    system.eval()
    return system, config, checkpoint_step, training_seed


def _pack_pair(pair: object) -> dict[str, object]:
    from scripts.system_comparison_metrics import CausalPrefixPair

    if not isinstance(pair, CausalPrefixPair):
        raise AllTEvaluationError("cache pair must be a causal prefix pair")
    return {
        "prediction": dict(pair.prediction),
        "target": dict(pair.target),
        "horizon": pair.horizon,
        "observed_scan_ids": list(pair.observed_scan_ids),
    }


def _unpack_pair(value: object):
    from scripts.system_comparison_metrics import validate_causal_prefix_pair

    if not isinstance(value, Mapping) or set(value) != {
        "prediction",
        "target",
        "horizon",
        "observed_scan_ids",
    }:
        raise AllTEvaluationError("cached causal pair fields differ")
    return validate_causal_prefix_pair(
        prediction=value["prediction"],
        target=value["target"],
        horizon=value["horizon"],
        observed_scan_ids=value["observed_scan_ids"],
    )


def _validate_sequence_bundle(
    value: object, *, expected_key: Mapping[str, object]
) -> dict[str, object]:
    from scripts.p6a_cache import validate_cache_payload
    from scripts.system_comparison_inference import validate_full_history_payload
    from scripts.system_comparison_v2_cache import validate_task_sidecar

    if not isinstance(value, Mapping) or set(value) != {
        "schema_version",
        "key",
        "pairs",
        "raw_payloads",
        "sidecars",
        "full_history_payloads",
    }:
        raise AllTEvaluationError("sequence cache bundle fields differ")
    bundle = dict(value)
    if bundle["schema_version"] != 1 or bundle["key"] != dict(expected_key):
        raise AllTEvaluationError("sequence cache key differs")
    key = _validate_sequence_cache_key(bundle["key"])
    pairs = bundle["pairs"]
    if not isinstance(pairs, Mapping):
        raise AllTEvaluationError("sequence cache pairs must be a mapping")
    expected_reducers = (
        {"official"} if key["window_mode"] == "full_history" else {"mean", "latest", "max"}
    )
    if set(pairs) != expected_reducers:
        raise AllTEvaluationError("sequence cache reducers differ")
    for by_horizon in pairs.values():
        if not isinstance(by_horizon, Mapping) or set(by_horizon) != {
            str(horizon) for horizon in REPORT_HORIZONS
        }:
            raise AllTEvaluationError("sequence cache horizons differ")
        for pair in by_horizon.values():
            _unpack_pair(pair)
    raw_payloads = bundle["raw_payloads"]
    sidecars = bundle["sidecars"]
    full_payloads = bundle["full_history_payloads"]
    if key["window_mode"] == "local_pair":
        if (
            not isinstance(raw_payloads, list)
            or len(raw_payloads) != 5
            or not isinstance(sidecars, list)
            or len(sidecars) != 5
            or full_payloads != []
        ):
            raise AllTEvaluationError("local sequence cache coverage differs")
        for raw, sidecar in zip(raw_payloads, sidecars, strict=True):
            validate_cache_payload(raw)
            validate_task_sidecar(sidecar)
    else:
        if raw_payloads != [] or sidecars != [] or not isinstance(
            full_payloads, list
        ) or len(full_payloads) != 5:
            raise AllTEvaluationError("full-history cache coverage differs")
        for payload in full_payloads:
            validate_full_history_payload(payload)
    return bundle


def _write_sequence_bundle(
    *, cache_root: Path, key: Mapping[str, object], bundle: Mapping[str, object]
) -> tuple[dict[str, object], bool]:
    import torch

    key_sha256 = sequence_cache_key_sha256(key)
    cache_root.mkdir(parents=True, exist_ok=True)
    path = cache_root / f"{key_sha256}.pt"
    reused = False
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file():
            raise AllTEvaluationError("sequence cache path is not a regular file")
        try:
            loaded = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as error:
            raise AllTEvaluationError("sequence cache cannot be loaded safely") from error
        _validate_sequence_bundle(loaded, expected_key=key)
        reused = True
    else:
        _validate_sequence_bundle(bundle, expected_key=key)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{key_sha256}.", suffix=".tmp", dir=cache_root
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            torch.save(dict(bundle), temporary)
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            temporary.replace(path)
            try:
                persisted = torch.load(path, map_location="cpu", weights_only=True)
            except Exception as error:
                raise AllTEvaluationError(
                    "new sequence cache cannot be loaded safely"
                ) from error
            _validate_sequence_bundle(persisted, expected_key=key)
        finally:
            temporary.unlink(missing_ok=True)
    return (
        {
            "bytes": path.stat().st_size,
            "file_sha256": _file_sha256(path),
            "filename": path.name,
            "key_sha256": key_sha256,
            "master_sequence_id": key["master_sequence_id"],
            "order_id": key["order_id"],
            "reference_scene_id": key["reference_scene_id"],
        },
        reused,
    )


def _load_sequence_bundle(
    *, cache_root: Path, key: Mapping[str, object]
) -> dict[str, object] | None:
    import torch

    path = cache_root / f"{sequence_cache_key_sha256(key)}.pt"
    if not path.exists() and not path.is_symlink():
        return None
    if path.is_symlink() or not path.is_file():
        raise AllTEvaluationError("sequence cache path is not a regular file")
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise AllTEvaluationError("sequence cache cannot be loaded safely") from error
    return _validate_sequence_bundle(value, expected_key=key)


def _produce_local_sequence(
    *,
    system: object,
    dataset: object,
    collate: object,
    sequence: EvaluationSequence,
    provenance: Mapping[str, object],
    population_sha256: str,
    class_mapper: object,
    evaluation_seed: int,
    device: object,
) -> dict[str, object]:
    import torch

    from models.persistent_memory import build_local_observation
    from scripts.evaluate_persist4d import (
        _latest_full_resolution_masks,
        _move_data_to_device,
        _move_targets_to_device,
        _segment_stages,
    )
    from scripts.evaluate_persist4d_p6a import (
        _frozen_inference_seed,
        cache_payload_from_inference,
    )
    from scripts.rescene_task_postprocess import extract_official_task_prediction
    from scripts.system_comparison_v2_analysis import build_v2_causal_pair
    from scripts.system_comparison_v2_cache import build_task_sidecar
    from scripts.system_comparison_v2_inference import (
        OfficialCandidateTrajectoryAccumulator,
    )
    from trainer.persist4d_allt_trainer import build_detached_memory_read_state

    trajectories = {
        reducer: OfficialCandidateTrajectoryAccumulator(score_reducer=reducer)
        for reducer in ("mean", "latest", "max")
    }
    raw_payloads = []
    sidecars = []
    pairs = {reducer: {} for reducer in trajectories}
    state = None
    read_state = None
    with _frozen_inference_seed(evaluation_seed, device):
        for request in sequence.stage_requests:
            sample = dataset.load_scan_indices(
                int(request["context_index"]),
                tuple(request["inference_scan_indices"]),
                change_file=None,
            )
            data, targets, names = collate([sample])
            if list(names) != [request["master_sequence_id"]] or len(targets) != 1:
                raise AllTEvaluationError("collator changed the requested local sequence")
            target_full_values = getattr(data, "target_full", None)
            if not isinstance(target_full_values, Sequence) or len(target_full_values) != 1:
                raise AllTEvaluationError("collated local data lacks one full target")
            target_full = target_full_values[0]
            data = _move_data_to_device(data, device)
            targets = _move_targets_to_device(targets, device)
            target = targets[0]
            stages = _segment_stages(target)
            latest_local_stage = int(stages.max().item())
            raw_coordinates = system._process_raw_coordinates(data)
            with torch.inference_mode():
                output = system(
                    data,
                    point2segment=[target["point2segment"]],
                    raw_coordinates=raw_coordinates,
                    is_eval=True,
                    memory_read_state=(
                        read_state
                        if getattr(system.model, "memory_read_enabled", False)
                        else None
                    ),
                )
            observation = build_local_observation(
                output,
                [stages],
                latest_stage=latest_local_stage,
                **system.observation_settings,
            )
            if state is None:
                state = system.persistent_memory.empty_state(observation)
            step = system.persistent_memory.step(
                observation,
                state,
                stage_index=int(request["stage_index"]),
            )
            state = step.state.detach()
            read_state = build_detached_memory_read_state(state)
            track_ids = [
                None if value < 0 else int(value)
                for value in step.slot_ids[0].detach().cpu().tolist()
            ]
            full_masks = _latest_full_resolution_masks(
                system,
                output,
                target,
                data,
                latest_local_stage=latest_local_stage,
            )
            raw_key = {
                "master_sequence_id": request["master_sequence_id"],
                "reference_scene_id": request["reference_scene_id"],
                "order_id": request["order_id"],
                "stage_index": request["stage_index"],
                "history_scan_ids": request["history_scan_ids"],
                "local_window_scan_ids": request["inference_scan_ids"],
            }
            raw = cache_payload_from_inference(
                key=raw_key,
                provenance=provenance,
                observation=observation,
                full_masks=full_masks,
                full_target=target_full,
                latest_local_stage=latest_local_stage,
            )
            official = extract_official_task_prediction(
                system=system,
                output=output,
                target_low_resolution=target,
                target_full_resolution=target_full,
                data=data,
                class_mapper=class_mapper,
                latest_stage_index=latest_local_stage,
            )
            sidecar = build_task_sidecar(
                raw_cache_payload=raw,
                official_prediction=official,
                protocol_manifest_sha256=population_sha256,
            )
            raw_payloads.append(raw)
            sidecars.append(sidecar)
            track_step = {
                "stage_id": int(request["stage_index"]),
                "track_ids": track_ids,
            }
            horizon = int(request["stage_index"]) + 1
            for reducer, trajectory in trajectories.items():
                trajectory.add_stage(sidecar, track_step)
                if horizon in REPORT_HORIZONS:
                    pairs[reducer][str(horizon)] = _pack_pair(
                        build_v2_causal_pair(
                            snapshot=trajectory.snapshot(),
                            raw_payloads=raw_payloads,
                            class_mapper=class_mapper,
                        )
                    )
    return {
        "schema_version": 1,
        "pairs": pairs,
        "raw_payloads": raw_payloads,
        "sidecars": sidecars,
        "full_history_payloads": [],
    }


def _produce_full_history_sequence(
    *,
    system: object,
    dataset: object,
    collate: object,
    sequence: EvaluationSequence,
    provenance: Mapping[str, object],
    class_mapper: object,
    evaluation_seed: int,
    device: object,
) -> dict[str, object]:
    from scripts.evaluate_persist4d import (
        _move_data_to_device,
        _move_targets_to_device,
    )
    from scripts.system_comparison_inference import FullHistoryPredictionProducer
    from scripts.system_comparison_metrics import causal_prefix_pair_from_payload

    producer = FullHistoryPredictionProducer(
        dataset=dataset,
        collate=collate,
        system=system,
        device=device,
        provenance=provenance,
        class_mapper=class_mapper,
        move_data=_move_data_to_device,
        move_targets=_move_targets_to_device,
        background_class=int(system.observation_settings["background_class"]),
        confidence_threshold=float(
            system.observation_settings["confidence_threshold"]
        ),
        mask_threshold=float(system.observation_settings["mask_threshold"]),
        minimum_mask_support=int(
            system.observation_settings["minimum_mask_support"]
        ),
        seed=evaluation_seed,
    )
    payloads = []
    pairs = {"official": {}}
    for request in sequence.stage_requests:
        horizon = int(request["stage_index"]) + 1
        key = {
            "master_sequence_id": request["master_sequence_id"],
            "reference_scene_id": request["reference_scene_id"],
            "order_id": request["order_id"],
            "context_index": request["context_index"],
            "context_scan_indices": request["context_scan_indices"],
            "horizon": horizon,
            "history_scan_ids": request["history_scan_ids"],
            "scan_indices": request["history_scan_indices"],
            "task_quality": horizon >= 2,
        }
        payload = producer.produce_bundle(key).payload
        payloads.append(payload)
        if horizon in REPORT_HORIZONS:
            pairs["official"][str(horizon)] = _pack_pair(
                causal_prefix_pair_from_payload(payload)
            )
    return {
        "schema_version": 1,
        "pairs": pairs,
        "raw_payloads": [],
        "sidecars": [],
        "full_history_payloads": payloads,
    }


def _metric_rows(
    *,
    bundles: Sequence[Mapping[str, object]],
    population_id: str,
    model: str,
    checkpoint_sha256: str,
    training_seed: int,
    evaluation_seed: int,
    window_mode: str,
    requested_reducers: Sequence[str] | None = None,
    metric_workers: int = 4,
) -> list[dict[str, object]]:
    from scripts.analyze_persist4d_allt import AllTBaselineAccumulator
    from scripts.analyze_r1_downstream_validation import resolve_metric_dataset_spec

    reducers = _resolve_metric_reducers(
        window_mode=window_mode,
        requested=requested_reducers,
    )
    accumulators = {
        (reducer, horizon): AllTBaselineAccumulator(
            dataset_spec=resolve_metric_dataset_spec(PROJECT_ROOT), include_task=True
        )
        for reducer in reducers
        for horizon in REPORT_HORIZONS
    }
    references = set()
    masters = set()
    for bundle in bundles:
        key = bundle["key"]
        references.add(key["reference_scene_id"])
        masters.add(key["master_sequence_id"])
        for reducer in reducers:
            for horizon in REPORT_HORIZONS:
                accumulators[(reducer, horizon)].update(
                    _unpack_pair(bundle["pairs"][reducer][str(horizon)])
                )
    metric_keys = tuple(
        (reducer, horizon)
        for reducer in reducers
        for horizon in REPORT_HORIZONS
    )
    computed = _compute_metric_values(
        accumulators,
        keys=metric_keys,
        workers=metric_workers,
    )
    rows = []
    method = "FullHistory" if window_mode == "full_history" else "B4"
    for reducer, horizon in metric_keys:
        values = computed[(reducer, horizon)]
        rows.append(
            {
                "population_id": population_id,
                "model": model,
                "checkpoint_sha256": checkpoint_sha256,
                "training_seed": training_seed,
                "evaluation_seed": evaluation_seed,
                "method": method,
                "reducer": reducer,
                "T": horizon,
                "t_mAP": values["t_mAP"],
                "t_mAP50": values["t_mAP50"],
                "t_mAP25": values["t_mAP25"],
                "t_REC": values["t_REC"],
                "prefix_overall_mAP": values["prefix_overall_mAP"],
                "local_current_AP": values["local_current_AP"],
                "num_master": len(masters),
                "num_order_units": len(bundles),
                "num_reference_clusters": len(references),
            }
        )
    validate_result_rows(rows)
    return rows


def run_evaluation(
    *,
    variant: str,
    checkpoint: Path,
    population: str,
    evaluation_seed: int,
    device_name: str,
    pretrained: Path,
    metadata: Path,
    data_root: Path,
    split_manifest: Path,
    cache_root: Path,
    output_root: Path,
    maximum_sequences: int | None = None,
    model_name: str | None = None,
    reducers: Sequence[str] | None = None,
    metric_workers: int = 4,
) -> dict[str, object]:
    import torch

    from scripts.evaluate_persist4d_p6a import build_rio_class_mapper
    from scripts.system_comparison_inference import deterministic_inference_runtime
    from scripts.train_persist4d_allt import VARIANTS

    started = time.time()
    source_commit = _git_head()
    checkpoint = checkpoint.expanduser().resolve()
    pretrained = pretrained.expanduser().resolve()
    metadata = metadata.expanduser().resolve()
    data_root = data_root.expanduser().resolve()
    split_manifest = split_manifest.expanduser().resolve()
    cache_root = cache_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    for path, name in (
        (checkpoint, "checkpoint"),
        (pretrained, "pretrained checkpoint"),
        (metadata, "3RScan metadata"),
        (split_manifest, "split manifest"),
    ):
        if path.is_symlink() or not path.is_file():
            raise AllTEvaluationError(f"{name} must be a regular non-symlink file")
    try:
        cache_root.relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        pass
    else:
        raise AllTEvaluationError("large evaluation cache must remain outside Git")
    if not torch.cuda.is_available():
        raise AllTEvaluationError("CUDA is required for real checkpoint evaluation")
    device = torch.device(device_name)
    if device.type != "cuda" or device.index is None or device.index >= torch.cuda.device_count():
        raise AllTEvaluationError("evaluation device is unavailable")
    checkpoint_sha256 = _file_sha256(checkpoint)
    module_document, module_sha256 = _module_config(variant)
    window_mode = str(VARIANTS[variant]["window_mode"])
    selected_reducers = _resolve_metric_reducers(
        window_mode=window_mode, requested=reducers
    )
    if (
        isinstance(metric_workers, bool)
        or not isinstance(metric_workers, int)
        or not 1 <= metric_workers <= 32
    ):
        raise AllTEvaluationError("metric workers must be within 1-32")
    result_model = variant if model_name is None else _nonempty(model_name, name="model")
    system, config, checkpoint_step, training_seed = _load_system(
        variant=variant,
        checkpoint=checkpoint,
        pretrained=pretrained,
        device=device,
        scratch=cache_root / "runtime",
    )
    dataset, collate, sequences, population_id, population_sha256 = _population(
        config=config,
        population=population,
        window_mode=window_mode,
        data_root=data_root,
        metadata=metadata,
        split_manifest=split_manifest,
    )
    if maximum_sequences is not None:
        if maximum_sequences <= 0 or maximum_sequences >= len(sequences):
            raise AllTEvaluationError("maximum_sequences must define a strict smoke subset")
        sequences = sequences[:maximum_sequences]
        population_id = f"smoke_{population_id}_{maximum_sequences}_sequences"
        population_sha256 = _canonical_json_sha256(
            [sequence.stage_requests for sequence in sequences]
        )
    class_mapper = build_rio_class_mapper(dataset)
    postprocess_version = (
        "official_rescene_candidate_lineage_v1+"
        "persistent_memory_slot_identity_v1"
    )
    raw_provenance = {
        "source_commit": source_commit,
        "checkpoint_sha256": checkpoint_sha256,
        "config_sha256": module_sha256,
        "dataset_sha256": population_sha256,
    }
    full_provenance = {
        "source_commit": source_commit,
        "checkpoint_sha256": checkpoint_sha256,
        "config_sha256": module_sha256,
        "protocol_sha256": population_sha256,
    }
    cache_directory = cache_root / population_id / variant / checkpoint_sha256
    bundles = []
    cache_records = []
    reused_count = 0
    with deterministic_inference_runtime(evaluation_seed, device):
        for position, sequence in enumerate(sequences, start=1):
            key = build_sequence_cache_key(
                checkpoint_sha256=checkpoint_sha256,
                module_config_sha256=module_sha256,
                population_id=population_id,
                evaluation_seed=evaluation_seed,
                postprocess_version=postprocess_version,
                stage_requests=sequence.stage_requests,
            )
            bundle = _load_sequence_bundle(cache_root=cache_directory, key=key)
            reused = bundle is not None
            if bundle is None:
                produced = (
                    _produce_full_history_sequence(
                        system=system,
                        dataset=dataset,
                        collate=collate,
                        sequence=sequence,
                        provenance=full_provenance,
                        class_mapper=class_mapper,
                        evaluation_seed=evaluation_seed,
                        device=device,
                    )
                    if window_mode == "full_history"
                    else _produce_local_sequence(
                        system=system,
                        dataset=dataset,
                        collate=collate,
                        sequence=sequence,
                        provenance=raw_provenance,
                        population_sha256=population_sha256,
                        class_mapper=class_mapper,
                        evaluation_seed=evaluation_seed,
                        device=device,
                    )
                )
                bundle = {**produced, "key": key}
            record, cache_reused = _write_sequence_bundle(
                cache_root=cache_directory, key=key, bundle=bundle
            )
            if reused != cache_reused:
                raise AllTEvaluationError("cache reuse accounting differs")
            reused_count += int(reused)
            cache_records.append(record)
            bundles.append(
                _compact_metric_bundle(bundle, reducers=selected_reducers)
            )
            if position % 5 == 0 or position == len(sequences):
                print(
                    f"[allt-eval] {variant}/{population_id} "
                    f"{position}/{len(sequences)} sequences",
                    file=sys.stderr,
                    flush=True,
                )
    del system
    torch.cuda.empty_cache()
    rows = _metric_rows(
        bundles=bundles,
        population_id=population_id,
        model=result_model,
        checkpoint_sha256=checkpoint_sha256,
        training_seed=training_seed,
        evaluation_seed=evaluation_seed,
        window_mode=window_mode,
        requested_reducers=selected_reducers,
        metric_workers=metric_workers,
    )
    csv_content = _csv_bytes(rows)
    _atomic_write(output_root / "all_t_metrics.csv", csv_content)
    cache_manifest = {
        "cache_directory": (
            f"external:allt_task_superiority_v1/evaluation_cache/{population_id}/"
            f"{variant}/{checkpoint_sha256}"
        ),
        "checkpoint_sha256": checkpoint_sha256,
        "entry_count": len(cache_records),
        "evaluation_seed": evaluation_seed,
        "module_config": module_document,
        "module_config_sha256": module_sha256,
        "population_id": population_id,
        "population_sha256": population_sha256,
        "records": cache_records,
        "reused_entry_count": reused_count,
        "schema_version": 1,
        "source_commit": source_commit,
        "status": "pass",
    }
    cache_manifest["content_sha256"] = _canonical_json_sha256(cache_manifest)
    _atomic_json(output_root / "cache_manifest.json", cache_manifest)
    summary = {
        "checkpoint_global_step": checkpoint_step,
        "checkpoint_sha256": checkpoint_sha256,
        "elapsed_seconds": time.time() - started,
        "evaluation_seed": evaluation_seed,
        "metric_row_count": len(rows),
        "metric_sha256": hashlib.sha256(csv_content).hexdigest(),
        "metric_worker_count": min(metric_workers, len(rows)),
        "model_forward_count": _new_model_forward_count(
            window_mode=window_mode,
            sequence_count=len(sequences),
            reused_count=reused_count,
        ),
        "new_sequence_count": len(sequences) - reused_count,
        "model": result_model,
        "population_id": population_id,
        "reducers": list(selected_reducers),
        "reused_sequence_count": reused_count,
        "sequence_count": len(sequences),
        "source_commit": source_commit,
        "status": "pass",
        "training_seed": training_seed,
        "variant": variant,
        "window_mode": window_mode,
    }
    _atomic_json(output_root / "run_summary.json", summary)
    if _git_head() != source_commit:
        raise AllTEvaluationError("Git HEAD changed during evaluation")
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--population", choices=("development", "protocol_b"), required=True
    )
    parser.add_argument("--evaluation-seed", type=int, default=45)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--pretrained", type=Path, default=DEFAULT_PRETRAINED)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--maximum-sequences", type=int)
    parser.add_argument("--model-name")
    parser.add_argument(
        "--reducers",
        nargs="+",
        choices=("mean", "latest", "max", "official"),
    )
    parser.add_argument("--metric-workers", type=int, default=4)
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = run_evaluation(
        variant=args.variant,
        checkpoint=args.checkpoint,
        population=args.population,
        evaluation_seed=args.evaluation_seed,
        device_name=args.device,
        pretrained=args.pretrained,
        metadata=args.metadata,
        data_root=args.data_root,
        split_manifest=args.split_manifest,
        cache_root=args.cache_root,
        output_root=args.output_root,
        maximum_sequences=args.maximum_sequences,
        model_name=args.model_name,
        reducers=args.reducers,
        metric_workers=args.metric_workers,
    )
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0


__all__ = [
    "RESULT_FIELDS",
    "AllTEvaluationError",
    "EvaluationSequence",
    "_checkpoint_identity",
    "_compact_metric_bundle",
    "_compute_metric_values",
    "_resolve_metric_reducers",
    "build_sequence_cache_key",
    "build_stage_requests",
    "rank_development_checkpoints",
    "run_evaluation",
    "sequence_cache_key_sha256",
    "validate_result_rows",
]


if __name__ == "__main__":
    raise SystemExit(main())
