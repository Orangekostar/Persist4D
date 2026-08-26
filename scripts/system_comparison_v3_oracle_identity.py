#!/usr/bin/env python3
"""Measure post-prediction Oracle-ID headroom on official local candidates."""

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
from collections import defaultdict
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import Tensor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_persist4d_p6a import (
    build_rio_class_mapper,
    build_tracker_factories,
    cache_payload_to_frozen_observation,
)
from scripts.p6a_metrics import match_instances_hungarian
from scripts.run_system_comparison import REPRODUCIBILITY_BINDING, _build_frozen_setup
from scripts.system_comparison_metrics import (
    CausalTaskAccumulator,
    compute_causal_task_metrics,
)
from scripts.system_comparison_v2_analysis import (
    build_v2_causal_pair,
    load_v2_sequences,
)
from scripts.system_comparison_v2_cache import validate_task_sidecar
from scripts.system_comparison_v2_inference import (
    OfficialCandidateTrajectoryAccumulator,
)

OUTPUT_ROOT = PROJECT_ROOT / "artifacts/reviewer_closure_v3/oracle_identity"
CACHE_ROOT = Path(
    "/mnt/shared/ww/persist4d-tmap-root-cause-v2/system_comparison_v2_full"
)
CACHE_MANIFEST = PROJECT_ROOT / "artifacts/system_comparison_v2/cache_manifest.json"
V2_ROOT = PROJECT_ROOT / "artifacts/system_comparison_v2"
SCORE_ROOT = PROJECT_ROOT / "artifacts/reviewer_closure_v3/score_sensitivity"
METADATA = Path("/home/ww/3RScan.json")
METHODS = ("FullHistory", "B2", "B4", "Oracle-ID")
LINKED_METHODS = ("B2", "B4", "Oracle-ID")
TRACKERS = ("B2", "B4")
ORDERS = ("canonical", "reverse", "sha256_seed45")
ORDER_SCOPES = (*ORDERS, "all")
HORIZONS = (2, 3, 4, 5)
TASK_FIELDS = (
    "causal_prefix_t_mAP",
    "causal_prefix_t_mAP50",
    "causal_prefix_t_mAP25",
    "causal_prefix_t_REC",
    "causal_prefix_t_REC50",
    "causal_prefix_t_REC25",
    "current_stage_AP",
    "current_stage_AP50",
    "current_stage_AP25",
    "current_stage_REC",
)
PER_SEQUENCE_FIELDS = (
    "method",
    "reference_scene_id",
    "master_sequence_id",
    "order_id",
    "horizon",
    *TASK_FIELDS,
)
AGGREGATE_FIELDS = (
    "method",
    "order_id",
    "horizon",
    "sequence_count",
    *TASK_FIELDS,
)


class OracleIdentityError(ValueError):
    """Raised when post-prediction Oracle-ID isolation is violated."""


def _tensor(value: object, *, name: str, ndim: int) -> Tensor:
    if not isinstance(value, Tensor) or value.ndim != ndim:
        raise OracleIdentityError(f"{name} must be a rank-{ndim} tensor")
    result = value.detach().cpu().contiguous().clone()
    if result.is_floating_point() and not torch.isfinite(result).all().item():
        raise OracleIdentityError(f"{name} must contain finite values")
    return result


def _tensor_digest(value: Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii") + b"\0")
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
    digest.update(b"\0" + tensor.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class OfficialCandidateStage:
    """Official candidate tensors frozen before Oracle GT becomes available."""

    stage_index: int
    predicted_masks: Tensor
    predicted_scores: Tensor
    predicted_classes: Tensor
    source_query_ids: Tensor
    content_sha256: str


@dataclass(frozen=True)
class OracleStageTarget:
    """GT fields permitted only in the post-prediction linkage function."""

    gt_ids: Tensor
    gt_masks: Tensor

    def __post_init__(self) -> None:
        ids = _tensor(self.gt_ids, name="Oracle GT IDs", ndim=1).long()
        masks = _tensor(self.gt_masks, name="Oracle GT masks", ndim=2).bool()
        if masks.shape[0] != ids.numel():
            raise OracleIdentityError("Oracle target fields must share the GT count")
        values = ids.tolist()
        if any(value < 0 for value in values) or len(set(values)) != len(values):
            raise OracleIdentityError(
                "Oracle GT IDs must be unique non-negative integers"
            )
        object.__setattr__(self, "gt_ids", ids)
        object.__setattr__(self, "gt_masks", masks)


@dataclass(frozen=True)
class OracleTrajectoryKey:
    kind: str
    predicted_class_id: int
    oracle_gt_id: Hashable | None = None
    stage_index: int | None = None
    candidate_index: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"persistent", "ephemeral"}:
            raise OracleIdentityError("Oracle trajectory key kind is invalid")
        if isinstance(self.predicted_class_id, bool) or not isinstance(
            self.predicted_class_id, int
        ):
            raise OracleIdentityError("predicted class ID must be an integer")
        if self.kind == "persistent":
            if self.oracle_gt_id is None:
                raise OracleIdentityError("persistent Oracle key requires a GT ID")
            if self.stage_index is not None or self.candidate_index is not None:
                raise OracleIdentityError("persistent Oracle key cannot be stage-local")
        elif (
            self.oracle_gt_id is not None
            or isinstance(self.stage_index, bool)
            or not isinstance(self.stage_index, int)
            or self.stage_index < 0
            or isinstance(self.candidate_index, bool)
            or not isinstance(self.candidate_index, int)
            or self.candidate_index < 0
        ):
            raise OracleIdentityError("ephemeral Oracle key fields are invalid")


@dataclass(frozen=True)
class OracleTrajectorySnapshot:
    prediction: dict[str, Tensor]
    keys: tuple[OracleTrajectoryKey, ...]
    stage_count: int
    score_reducer: str


@dataclass(frozen=True)
class _Occurrence:
    stage_index: int
    mask: Tensor
    score: float


def _candidate_digest(
    *, masks: Tensor, scores: Tensor, classes: Tensor, query_ids: Tensor
) -> str:
    digest = hashlib.sha256()
    for name, value in (
        ("pred_masks", masks),
        ("pred_scores", scores),
        ("pred_classes", classes),
        ("source_query_ids", query_ids),
    ):
        digest.update(name.encode("ascii") + b"\0")
        digest.update(_tensor_digest(value).encode("ascii") + b"\0")
    return digest.hexdigest()


def freeze_official_candidate_stage(
    sidecar: Mapping[str, object],
) -> OfficialCandidateStage:
    """Freeze candidate masks/classes/scores before any GT access."""

    validate_task_sidecar(sidecar)
    key = sidecar.get("key")
    task = sidecar.get("task_prediction")
    if not isinstance(key, Mapping) or not isinstance(task, Mapping):
        raise OracleIdentityError("official sidecar fields must be mappings")
    stage_index = key.get("stage_index")
    if isinstance(stage_index, bool) or not isinstance(stage_index, int):
        raise OracleIdentityError("official sidecar stage must be an integer")
    masks = _tensor(task.get("pred_masks"), name="candidate masks", ndim=2).bool()
    scores = _tensor(task.get("pred_scores"), name="candidate scores", ndim=1).float()
    classes = _tensor(
        task.get("pred_classes"), name="candidate classes", ndim=1
    ).long()
    query_ids = _tensor(
        task.get("source_query_ids"), name="candidate source query IDs", ndim=1
    ).long()
    count = scores.numel()
    if masks.shape[1] != count or classes.numel() != count or query_ids.numel() != count:
        raise OracleIdentityError("official candidate tensors do not align")
    return OfficialCandidateStage(
        stage_index=stage_index,
        predicted_masks=masks,
        predicted_scores=scores,
        predicted_classes=classes,
        source_query_ids=query_ids,
        content_sha256=_candidate_digest(
            masks=masks, scores=scores, classes=classes, query_ids=query_ids
        ),
    )


def link_oracle_identities(
    stage: OfficialCandidateStage,
    target: OracleStageTarget | None,
    *,
    iou_threshold: float = 0.5,
) -> tuple[int | None, ...]:
    """Assign GT IDs after candidate prediction using mask IoU only."""

    if not isinstance(stage, OfficialCandidateStage):
        raise OracleIdentityError("official candidate stage is required")
    if target is None:
        raise OracleIdentityError("GT target is required after candidate prediction")
    if not isinstance(target, OracleStageTarget):
        raise OracleIdentityError("GT target must be an OracleStageTarget")
    if not math.isfinite(float(iou_threshold)) or not 0 <= float(iou_threshold) <= 1:
        raise OracleIdentityError("IoU threshold must be finite within [0, 1]")
    if target.gt_masks.shape[1] != stage.predicted_masks.shape[0]:
        raise OracleIdentityError("Oracle GT and candidates must share point count")
    pairs = match_instances_hungarian(
        target.gt_masks,
        stage.predicted_masks.transpose(0, 1),
        threshold=float(iou_threshold),
    )
    linked: list[int | None] = [None] * stage.predicted_scores.numel()
    for gt_index, candidate_index in pairs:
        linked[candidate_index] = int(target.gt_ids[gt_index].item())
    return tuple(linked)


class OracleCandidateTrajectoryAccumulator:
    """Link frozen official candidates under `(GT ID, predicted class)` keys."""

    def __init__(self, *, score_reducer: str = "mean") -> None:
        if score_reducer != "mean":
            raise OracleIdentityError("Oracle-ID primary score reducer must be mean")
        self.score_reducer = score_reducer
        self._stage_point_counts: list[int] = []
        self._keys: list[OracleTrajectoryKey] = []
        self._occurrences: dict[OracleTrajectoryKey, list[_Occurrence]] = {}
        self._candidate_digests: list[str] = []
        self._persistent_occurrences = 0
        self._ephemeral_occurrences = 0

    @property
    def stage_count(self) -> int:
        return len(self._stage_point_counts)

    @property
    def candidate_digests(self) -> tuple[str, ...]:
        return tuple(self._candidate_digests)

    @property
    def linkage_counts(self) -> dict[str, int]:
        return {
            "persistent_occurrences": self._persistent_occurrences,
            "ephemeral_occurrences": self._ephemeral_occurrences,
        }

    def add_stage(
        self,
        stage: OfficialCandidateStage,
        oracle_ids: Sequence[int | None],
    ) -> None:
        if not isinstance(stage, OfficialCandidateStage):
            raise OracleIdentityError("official candidate stage is required")
        if stage.stage_index != self.stage_count:
            raise OracleIdentityError("Oracle stages must be committed in order")
        values = tuple(oracle_ids)
        count = int(stage.predicted_scores.numel())
        if len(values) != count:
            raise OracleIdentityError("Oracle requires one linkage per candidate")
        if any(
            value is not None
            and (isinstance(value, bool) or not isinstance(value, int) or value < 0)
            for value in values
        ):
            raise OracleIdentityError("Oracle link IDs must be non-negative integers")
        current_digest = _candidate_digest(
            masks=stage.predicted_masks,
            scores=stage.predicted_scores,
            classes=stage.predicted_classes,
            query_ids=stage.source_query_ids,
        )
        if current_digest != stage.content_sha256:
            raise OracleIdentityError("official candidate fields changed before linkage")

        self._stage_point_counts.append(int(stage.predicted_masks.shape[0]))
        stage_keys: set[OracleTrajectoryKey] = set()
        for candidate_index, oracle_gt_id in enumerate(values):
            predicted_class_id = int(stage.predicted_classes[candidate_index].item())
            if oracle_gt_id is None:
                key = OracleTrajectoryKey(
                    kind="ephemeral",
                    predicted_class_id=predicted_class_id,
                    stage_index=stage.stage_index,
                    candidate_index=candidate_index,
                )
                self._ephemeral_occurrences += 1
            else:
                key = OracleTrajectoryKey(
                    kind="persistent",
                    predicted_class_id=predicted_class_id,
                    oracle_gt_id=oracle_gt_id,
                )
                self._persistent_occurrences += 1
            if key in stage_keys:
                raise OracleIdentityError("one stage contains a duplicate Oracle key")
            stage_keys.add(key)
            if key not in self._occurrences:
                self._keys.append(key)
                self._occurrences[key] = []
            score = float(stage.predicted_scores[candidate_index].item())
            self._occurrences[key].append(
                _Occurrence(
                    stage_index=stage.stage_index,
                    mask=stage.predicted_masks[:, candidate_index].clone(),
                    score=score,
                )
            )
        self._candidate_digests.append(current_digest)

    def snapshot(self) -> OracleTrajectorySnapshot:
        total_points = sum(self._stage_point_counts)
        output_masks = torch.zeros((total_points, len(self._keys)), dtype=torch.bool)
        output_scores = torch.empty(len(self._keys), dtype=torch.float32)
        output_classes = torch.empty(len(self._keys), dtype=torch.long)
        offsets = [0]
        for point_count in self._stage_point_counts:
            offsets.append(offsets[-1] + point_count)
        for column, key in enumerate(self._keys):
            occurrences = self._occurrences[key]
            for occurrence in occurrences:
                start = offsets[occurrence.stage_index]
                stop = offsets[occurrence.stage_index + 1]
                if occurrence.mask.numel() != stop - start:
                    raise OracleIdentityError("Oracle occurrence point count changed")
                output_masks[start:stop, column] = occurrence.mask
            output_scores[column] = sum(item.score for item in occurrences) / len(
                occurrences
            )
            output_classes[column] = key.predicted_class_id
        return OracleTrajectorySnapshot(
            prediction={
                "pred_masks": output_masks,
                "pred_scores": output_scores,
                "pred_classes": output_classes,
            },
            keys=tuple(self._keys),
            stage_count=self.stage_count,
            score_reducer=self.score_reducer,
        )


def _oracle_target(raw_payload: Mapping[str, object]) -> OracleStageTarget:
    target = raw_payload.get("target")
    if not isinstance(target, Mapping):
        raise OracleIdentityError("raw cache payload requires an Oracle target")
    if target.get("gt_class_semantics") != "rescene_model_index_0_based":
        raise OracleIdentityError("raw cache target class semantics differ")
    if target.get("change_labels_valid") is not False:
        raise OracleIdentityError("raw cache change-label validity differs")
    return OracleStageTarget(
        gt_ids=target.get("gt_ids"),
        gt_masks=target.get("gt_masks"),
    )


def _read_json(path: Path) -> Mapping[str, object]:
    if not path.is_file() or path.is_symlink():
        raise OracleIdentityError(f"required JSON is unavailable: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise OracleIdentityError(f"required JSON must contain a mapping: {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.is_symlink():
        raise OracleIdentityError(f"required CSV is unavailable: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise OracleIdentityError(f"required CSV is empty: {path}")
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _csv_bytes(
    rows: Sequence[Mapping[str, object]], fields: Sequence[str]
) -> bytes:
    if not rows:
        raise OracleIdentityError("Oracle output rows must not be empty")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        if set(row) != set(fields):
            raise OracleIdentityError("Oracle CSV fields differ from contract")
        writer.writerow(row)
    return stream.getvalue().encode("utf-8")


def _write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _task_metrics(pair: object) -> dict[str, float]:
    values = compute_causal_task_metrics([pair])
    return {field: float(values[field]) for field in TASK_FIELDS}


def _task_accumulator(
    values: dict[tuple[str, str, int], CausalTaskAccumulator],
    key: tuple[str, str, int],
) -> CausalTaskAccumulator:
    if key not in values:
        values[key] = CausalTaskAccumulator()
    return values[key]


def _frozen_v2_maps() -> tuple[
    dict[tuple[str, str, str, int], Mapping[str, str]],
    dict[tuple[str, str, int], Mapping[str, str]],
]:
    sequence_rows = _read_csv(V2_ROOT / "per_sequence_results.csv")
    aggregate_rows = _read_csv(V2_ROOT / "aggregate_results.csv")
    aggregate_rows.extend(_read_csv(V2_ROOT / "per_order_results.csv"))
    sequence_map = {
        (
            row["method"],
            row["master_sequence_id"],
            row["order_id"],
            int(row["horizon"]),
        ): row
        for row in sequence_rows
    }
    aggregate_map = {
        (row["method"], row["order_id"], int(row["horizon"])): row
        for row in aggregate_rows
    }
    if len(sequence_map) != 2 * 129 * len(HORIZONS) or len(aggregate_map) != 32:
        raise OracleIdentityError("frozen V2 coverage differs")
    return sequence_map, aggregate_map


def _score_maps() -> tuple[
    dict[tuple[str, str, str, int], Mapping[str, str]],
    dict[tuple[str, str, int], Mapping[str, str]],
]:
    sequence = [
        row
        for row in _read_csv(SCORE_ROOT / "per_sequence.csv")
        if row["tracker"] in TRACKERS and row["score_reducer"] == "mean"
    ]
    aggregate = [
        row
        for row in _read_csv(SCORE_ROOT / "aggregate.csv")
        if row["tracker"] in TRACKERS and row["score_reducer"] == "mean"
    ]
    sequence_map = {
        (
            row["tracker"],
            row["master_sequence_id"],
            row["order_id"],
            int(row["horizon"]),
        ): row
        for row in sequence
    }
    aggregate_map = {
        (row["tracker"], row["order_id"], int(row["horizon"])): row
        for row in aggregate
    }
    if len(sequence_map) != len(TRACKERS) * 129 * len(HORIZONS) or len(
        aggregate_map
    ) != len(TRACKERS) * len(ORDER_SCOPES) * len(HORIZONS):
        raise OracleIdentityError("score-sensitivity mean coverage differs")
    return sequence_map, aggregate_map


def _regression_differences(
    row: Mapping[str, object], reference: Mapping[str, str]
) -> list[float]:
    mapping = {
        **{field: field for field in TASK_FIELDS[:6]},
        "current_stage_AP": "trajectory_current_slice_AP",
        "current_stage_AP50": "trajectory_current_slice_AP50",
        "current_stage_AP25": "trajectory_current_slice_AP25",
        "current_stage_REC": "trajectory_current_slice_REC",
    }
    return [
        abs(float(row[current]) - float(reference[frozen]))
        for current, frozen in mapping.items()
    ]


def _frozen_row(
    source: Mapping[str, str],
    *,
    method: str,
    aggregate: bool,
) -> dict[str, object]:
    prefix = {
        "method": method,
        "order_id": source["order_id"],
        "horizon": int(source["horizon"]),
    }
    if aggregate:
        prefix["sequence_count"] = 129 if source["order_id"] == "all" else 43
    else:
        prefix = {
            "method": method,
            "reference_scene_id": source["reference_scene_id"],
            "master_sequence_id": source["master_sequence_id"],
            **prefix,
        }
    return {**prefix, **{field: float(source[field]) for field in TASK_FIELDS}}


def _report(aggregate_rows: Sequence[Mapping[str, object]], *, regression: float) -> str:
    index = {
        (str(row["method"]), int(row["horizon"])): row
        for row in aggregate_rows
        if row["order_id"] == "all"
    }
    lines = [
        "# Oracle-ID Headroom",
        "",
        "## OR0 Status",
        "",
        "**PASS.** GT identity is introduced only after official local candidate",
        "masks, predicted classes, and scores are frozen. It is used only as the",
        "trajectory linkage key under mask-IoU Hungarian matching at 0.5.",
        "",
        "## All-Order FullHistory t-mAP",
        "",
        "| Horizon | FullHistory | B2 official | B4 official | Oracle-ID | B4 - B2 | Oracle - B4 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for horizon in HORIZONS:
        full = float(index[("FullHistory", horizon)]["causal_prefix_t_mAP"])
        b2 = float(index[("B2", horizon)]["causal_prefix_t_mAP"])
        b4 = float(index[("B4", horizon)]["causal_prefix_t_mAP"])
        oracle = float(index[("Oracle-ID", horizon)]["causal_prefix_t_mAP"])
        lines.append(
            f"| T{horizon} | {100 * full:.3f} | {100 * b2:.3f} | "
            f"{100 * b4:.3f} | {100 * oracle:.3f} | {100 * (b4 - b2):+.3f} | "
            f"{100 * (oracle - b4):+.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "`B4 - B2` is recovered identity value under identical official local",
            "candidates. `Oracle - B4` is diagnostic linkage headroom; Oracle-ID is",
            "not a method or baseline and cannot improve missing/wrong local candidates.",
            "Predicted class remains unchanged and is part of every persistent key,",
            "so GT class is never substituted for model semantics.",
            "",
            "## Invariants",
            "",
            "- Candidate mask/class/score source: frozen V2 official task sidecars.",
            "- Matching: one-to-one Hungarian on candidate/GT mask IoU only, threshold 0.5.",
            "- Persistent key: `(oracle_gt_id, predicted_class_id)`.",
            "- Unmatched candidate key: `(stage_index, candidate_index, predicted_class_id)`.",
            "- Score reducer: mean; candidate masks, classes, and scores are unmodified.",
            f"- Fresh B2/B4 regression maximum absolute difference: `{regression}`.",
            "- FullHistory values are frozen V2 evidence under the same protocol.",
            "",
        ]
    )
    return "\n".join(lines)


def run_oracle_identity(
    *,
    cache_root: Path = CACHE_ROOT,
    cache_manifest_path: Path = CACHE_MANIFEST,
    metadata_path: Path = METADATA,
    output_root: Path = OUTPUT_ROOT,
) -> dict[str, object]:
    binding = _read_json(REPRODUCIBILITY_BINDING)
    setup = _build_frozen_setup(
        binding=binding,
        metadata_path=metadata_path,
        device_name=None,
    )
    cache_manifest = _read_json(cache_manifest_path)
    sequences = load_v2_sequences(
        cache_manifest=cache_manifest,
        cache_root=cache_root,
    )
    class_mapper = build_rio_class_mapper(setup.dataset)
    factories = build_tracker_factories(setup.p6a_config)
    if not set(TRACKERS) <= set(factories):
        raise OracleIdentityError("registered tracker factories are incomplete")
    frozen_sequences, frozen_aggregate = _frozen_v2_maps()
    score_sequences, score_aggregate = _score_maps()

    accumulators: dict[tuple[str, str, int], CausalTaskAccumulator] = {}
    counts: dict[tuple[str, int], int] = defaultdict(int)
    per_sequence_rows: list[dict[str, object]] = []
    regression_differences: list[float] = []
    candidate_audit_count = 0
    persistent_occurrences = 0
    ephemeral_occurrences = 0

    for sequence_index, sequence in enumerate(sequences, start=1):
        trackers = {
            name: factories[name](f"{sequence.master_sequence_id}:{sequence.order_id}")
            for name in TRACKERS
        }
        trajectories = {
            name: OfficialCandidateTrajectoryAccumulator(score_reducer="mean")
            for name in TRACKERS
        }
        oracle = OracleCandidateTrajectoryAccumulator(score_reducer="mean")
        for stage_index, (raw, sidecar) in enumerate(
            zip(sequence.raw_payloads, sequence.sidecars, strict=True)
        ):
            candidate_stage = freeze_official_candidate_stage(sidecar)
            target = _oracle_target(raw)
            links = link_oracle_identities(candidate_stage, target, iou_threshold=0.5)
            before = candidate_stage.content_sha256
            oracle.add_stage(candidate_stage, links)
            after = _candidate_digest(
                masks=candidate_stage.predicted_masks,
                scores=candidate_stage.predicted_scores,
                classes=candidate_stage.predicted_classes,
                query_ids=candidate_stage.source_query_ids,
            )
            if before != after:
                raise OracleIdentityError("Oracle linkage changed official candidates")
            candidate_audit_count += 1

            observation = cache_payload_to_frozen_observation(raw)
            for name in TRACKERS:
                step = trackers[name].step(observation, stage_id=stage_index)
                trajectories[name].add_stage(sidecar, step)
            horizon = stage_index + 1
            if horizon not in HORIZONS:
                continue
            snapshots = {
                "B2": trajectories["B2"].snapshot(),
                "B4": trajectories["B4"].snapshot(),
                "Oracle-ID": oracle.snapshot(),
            }
            for method in LINKED_METHODS:
                pair = build_v2_causal_pair(
                    snapshot=snapshots[method],
                    raw_payloads=sequence.raw_payloads[:horizon],
                    class_mapper=class_mapper,
                )
                row = {
                    "method": method,
                    "reference_scene_id": sequence.reference_scene_id,
                    "master_sequence_id": sequence.master_sequence_id,
                    "order_id": sequence.order_id,
                    "horizon": horizon,
                    **_task_metrics(pair),
                }
                per_sequence_rows.append(row)
                for order_id in (sequence.order_id, "all"):
                    _task_accumulator(
                        accumulators, (method, order_id, horizon)
                    ).update(pair)
                if method in TRACKERS:
                    reference = score_sequences[
                        (
                            method,
                            sequence.master_sequence_id,
                            sequence.order_id,
                            horizon,
                        )
                    ]
                    regression_differences.extend(
                        _regression_differences(row, reference)
                    )
            full = frozen_sequences[
                (
                    "FullHistory",
                    sequence.master_sequence_id,
                    sequence.order_id,
                    horizon,
                )
            ]
            per_sequence_rows.append(
                _frozen_row(full, method="FullHistory", aggregate=False)
            )
            for order_id in (sequence.order_id, "all"):
                counts[(order_id, horizon)] += 1
        linkage_counts = oracle.linkage_counts
        persistent_occurrences += linkage_counts["persistent_occurrences"]
        ephemeral_occurrences += linkage_counts["ephemeral_occurrences"]
        if sequence_index % 10 == 0 or sequence_index == len(sequences):
            print(
                f"[oracle] completed {sequence_index}/{len(sequences)} sequences",
                flush=True,
            )

    method_order = {name: index for index, name in enumerate(METHODS)}
    order_order = {name: index for index, name in enumerate(ORDER_SCOPES)}
    per_sequence_rows.sort(
        key=lambda row: (
            str(row["reference_scene_id"]),
            str(row["master_sequence_id"]),
            str(row["order_id"]),
            int(row["horizon"]),
            method_order[str(row["method"])],
        )
    )
    if len(per_sequence_rows) != len(METHODS) * 129 * len(HORIZONS):
        raise OracleIdentityError("Oracle per-sequence coverage differs")

    aggregate_rows: list[dict[str, object]] = []
    for order_id in ORDER_SCOPES:
        for horizon in HORIZONS:
            full = frozen_aggregate[("FullHistory", order_id, horizon)]
            aggregate_rows.append(
                _frozen_row(full, method="FullHistory", aggregate=True)
            )
            for method in LINKED_METHODS:
                row = {
                    "method": method,
                    "order_id": order_id,
                    "horizon": horizon,
                    "sequence_count": counts[(order_id, horizon)],
                    **accumulators[(method, order_id, horizon)].compute(),
                }
                aggregate_rows.append(row)
                if method in TRACKERS:
                    regression_differences.extend(
                        _regression_differences(
                            row, score_aggregate[(method, order_id, horizon)]
                        )
                    )
    aggregate_rows.sort(
        key=lambda row: (
            order_order[str(row["order_id"])],
            int(row["horizon"]),
            method_order[str(row["method"])],
        )
    )
    if (
        len(aggregate_rows) != len(METHODS) * len(ORDER_SCOPES) * len(HORIZONS)
        or any(counts[("all", horizon)] != 129 for horizon in HORIZONS)
        or any(
            counts[(order, horizon)] != 43
            for order in ORDERS
            for horizon in HORIZONS
        )
    ):
        raise OracleIdentityError("Oracle aggregate coverage differs")
    regression = max(regression_differences, default=0.0)
    if regression > 1e-12:
        raise OracleIdentityError("fresh B2/B4 differ from V3 mean evidence")
    if candidate_audit_count != 645 or ephemeral_occurrences <= 0:
        raise OracleIdentityError("Oracle candidate/ephemeral audit coverage differs")

    outputs = {
        "oracle_per_sequence.csv": _csv_bytes(
            per_sequence_rows, PER_SEQUENCE_FIELDS
        ),
        "oracle_aggregate.csv": _csv_bytes(aggregate_rows, AGGREGATE_FIELDS),
        "ORACLE_IDENTITY_HEADROOM.md": _report(
            aggregate_rows, regression=regression
        ).encode("utf-8"),
    }
    for name, content in outputs.items():
        _write(output_root / name, content)

    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    manifest = {
        "schema_version": 1,
        "status": "pass",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source_commit": source_commit,
        "gate_or0": {
            "status": "PASS",
            "gt_access": "post_prediction_linkage_only",
            "iou_threshold": 0.5,
            "candidate_fields_unchanged": True,
            "predicted_class_retained": True,
            "unmatched_candidates_ephemeral": True,
        },
        "coverage": {
            "sequence_count": 129,
            "methods": list(METHODS),
            "orders": list(ORDERS),
            "horizons": list(HORIZONS),
            "per_sequence_row_count": len(per_sequence_rows),
            "aggregate_row_count": len(aggregate_rows),
        },
        "inputs": {
            "reproducibility_binding": {
                "reference": "repo:artifacts/system_comparison/reproducibility_binding.json",
                "sha256": _sha256(REPRODUCIBILITY_BINDING),
            },
            "cache_manifest": {
                "reference": "repo:artifacts/system_comparison_v2/cache_manifest.json",
                "sha256": _sha256(cache_manifest_path),
                "records_sha256": cache_manifest["records_sha256"],
                "external_reference": cache_manifest["external_cache_reference"],
            },
            "frozen_v2_manifest": {
                "reference": "repo:artifacts/system_comparison_v2/manifest.json",
                "sha256": _sha256(V2_ROOT / "manifest.json"),
            },
            "score_sensitivity_manifest": {
                "reference": "repo:artifacts/reviewer_closure_v3/score_sensitivity/manifest.json",
                "sha256": _sha256(SCORE_ROOT / "manifest.json"),
            },
            "checkpoint_sha256": cache_manifest["checkpoint_sha256"],
            "protocol_manifest_sha256": cache_manifest[
                "protocol_manifest_sha256"
            ],
            "config_sha256": binding["config_sha256"],
        },
        "execution": {
            "mode": "frozen_official_candidates_post_prediction_oracle_linkage",
            "gpu_inference_performed": False,
            "gt_available_to_cache_generation": False,
            "gt_available_to_registered_trackers": False,
            "gt_available_to_oracle_linkage": True,
            "score_reducer": "mean",
            "metric_backend": "stmetrics via CausalTaskAccumulator",
        },
        "audit": {
            "candidate_stage_count": candidate_audit_count,
            "candidate_field_digest_status": "pass_exact",
            "b2_b4_v3_mean_regression_max_abs_diff": regression,
            "persistent_occurrence_count": persistent_occurrences,
            "ephemeral_occurrence_count": ephemeral_occurrences,
            "matching_uses_gt_class": False,
        },
        "outputs": {
            name: {
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
            for name, content in outputs.items()
        },
        "scripts": {
            "oracle_identity_sha256": _sha256(Path(__file__)),
            "test_sha256": _sha256(
                PROJECT_ROOT / "tests/test_system_comparison_v3_oracle_identity.py"
            ),
            "official_trajectory_sha256": _sha256(
                PROJECT_ROOT / "scripts/system_comparison_v2_inference.py"
            ),
        },
    }
    _write(
        output_root / "manifest.json",
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(),
    )
    return {
        "status": "pass",
        "gate": "OR0",
        "per_sequence_rows": len(per_sequence_rows),
        "aggregate_rows": len(aggregate_rows),
        "b2_b4_regression_max_abs_diff": regression,
        "persistent_occurrences": persistent_occurrences,
        "ephemeral_occurrences": ephemeral_occurrences,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--cache-manifest", type=Path, default=CACHE_MANIFEST)
    parser.add_argument("--metadata", type=Path, default=METADATA)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    result = run_oracle_identity(
        cache_root=arguments.cache_root,
        cache_manifest_path=arguments.cache_manifest,
        metadata_path=arguments.metadata,
        output_root=arguments.output_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "OfficialCandidateStage",
    "OracleCandidateTrajectoryAccumulator",
    "OracleIdentityError",
    "OracleStageTarget",
    "OracleTrajectoryKey",
    "OracleTrajectorySnapshot",
    "freeze_official_candidate_stage",
    "link_oracle_identities",
    "run_oracle_identity",
]
