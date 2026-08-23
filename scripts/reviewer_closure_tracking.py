"""Full-History cross-prefix tracking with frozen P6-A association classes."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor

from scripts.p6a_association import FrozenObservation, TrackStep, freeze_observation
from scripts.reviewer_closure_sidecar import sidecar_key_for_source_prediction
from scripts.system_comparison_metrics import (
    IdentityAssignmentUpdate,
    compute_deployment_identity_metrics,
    match_identity_update,
)
from scripts.system_comparison_inference import unpack_bool_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REVIEWER_MANIFEST_PATH = (
    PROJECT_ROOT / "artifacts/reviewer_closure/reviewer_closure_manifest.json"
)
REPLAY_MANIFEST_PATH = (
    PROJECT_ROOT / "artifacts/reviewer_closure/full_history_replay_v2/manifest.json"
)
REPLAY_ENTRY_ROOT = (
    PROJECT_ROOT / "artifacts/reviewer_closure/full_history_replay_v2/entries"
)
SIDECAR_MANIFEST_PATH = (
    PROJECT_ROOT
    / "artifacts/reviewer_closure/full_history_observations_v2/manifest.json"
)
SIDECAR_ENTRY_ROOT = (
    PROJECT_ROOT / "artifacts/reviewer_closure/full_history_observations_v2/entries"
)
TRACKING_ARTIFACT_PATH = (
    PROJECT_ROOT / "artifacts/reviewer_closure/full_history_tracker_raw.json"
)
_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")

PAPER_METHOD_NAMES = {
    "FullHistoryNative": "ReScene4D Full-History",
    "B1": "Pairwise Feature Association",
    "B2": "Pairwise Feature-Class Association",
    "B3": "EMA Temporal Association",
    "B4": "Full-History + Persistent-State Diagnostic",
}


def _tensor(value: object, *, name: str, ndim: int) -> Tensor:
    if not isinstance(value, Tensor):
        raise ValueError(f"{name} must be a tensor")
    result = value.detach().cpu().contiguous().clone()
    if result.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}")
    if result.is_floating_point() and not torch.isfinite(result).all().item():
        raise ValueError(f"{name} must contain finite values")
    return result


def _integer_tensor(value: object, *, name: str) -> Tensor:
    result = _tensor(value, name=name, ndim=1)
    if result.dtype == torch.bool or result.is_floating_point() or result.is_complex():
        raise ValueError(f"{name} must use an integer dtype")
    return result.long()


@dataclass(frozen=True)
class FullHistoryTrackerStage:
    key: Mapping[str, object]
    observation: FrozenObservation
    local_query_ids: Tensor
    gt_ids: Tensor
    gt_classes: Tensor
    gt_masks: Tensor
    native_issued_ids: Tensor
    pred_classes: Tensor
    pred_masks: Tensor

    def __post_init__(self) -> None:
        key = sidecar_key_for_source_prediction(self.key)
        observation = freeze_observation(self.observation)
        if observation.features.ndim != 2:
            raise ValueError("tracker stage observation must be unbatched")
        local_ids = _integer_tensor(self.local_query_ids, name="local query IDs")
        gt_ids = _integer_tensor(self.gt_ids, name="GT IDs")
        gt_classes = _integer_tensor(self.gt_classes, name="GT classes")
        gt_masks = _tensor(self.gt_masks, name="GT masks", ndim=2)
        native_ids = _integer_tensor(self.native_issued_ids, name="native issued IDs")
        pred_classes = _integer_tensor(self.pred_classes, name="predicted classes")
        pred_masks = _tensor(self.pred_masks, name="predicted masks", ndim=2)
        if gt_masks.dtype != torch.bool or pred_masks.dtype != torch.bool:
            raise ValueError("tracker stage masks must use bool dtype")
        query_count = observation.query_count
        if (
            local_ids.numel() != query_count
            or len(set(local_ids.tolist())) != query_count
            or torch.any(local_ids < 0).item()
        ):
            raise ValueError("local query IDs must be unique and query-aligned")
        if (
            gt_ids.shape != gt_classes.shape
            or gt_masks.shape[0] != gt_ids.numel()
            or len(set(gt_ids.tolist())) != gt_ids.numel()
            or native_ids.shape != pred_classes.shape
            or pred_masks.shape[1] != native_ids.numel()
            or len(set(native_ids.tolist())) != native_ids.numel()
        ):
            raise ValueError("tracker stage identity tensors do not align")
        if len(observation.latest_mask) != 1:
            raise ValueError("tracker stage requires exactly one current-stage mask")
        observation_masks = observation.latest_mask[0]
        point_count = observation_masks.shape[1]
        if gt_masks.shape[1] != point_count or pred_masks.shape[0] != point_count:
            raise ValueError("tracker stage point axes do not align")
        positions = {
            int(value): index for index, value in enumerate(local_ids.tolist())
        }
        if not set(native_ids.tolist()) <= set(positions):
            raise ValueError("native issued IDs are outside the local query namespace")
        selected_positions = [positions[int(value)] for value in native_ids.tolist()]
        valid_positions = observation.valid.nonzero(as_tuple=True)[0].tolist()
        if set(selected_positions) != set(valid_positions):
            raise ValueError("native issued IDs differ from valid observations")
        selected_masks = observation_masks[selected_positions].bool().transpose(0, 1)
        if not torch.equal(selected_masks.contiguous(), pred_masks):
            raise ValueError("native prediction masks differ from sidecar observations")
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "observation", observation)
        object.__setattr__(self, "local_query_ids", local_ids)
        object.__setattr__(self, "gt_ids", gt_ids)
        object.__setattr__(self, "gt_classes", gt_classes)
        object.__setattr__(self, "gt_masks", gt_masks)
        object.__setattr__(self, "native_issued_ids", native_ids)
        object.__setattr__(self, "pred_classes", pred_classes)
        object.__setattr__(self, "pred_masks", pred_masks)

    @property
    def horizon(self) -> int:
        return int(self.key["horizon"])


@dataclass(frozen=True)
class FullHistoryTrackerSequence:
    stages: tuple[FullHistoryTrackerStage, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.stages, tuple) or len(self.stages) != 4:
            raise ValueError("tracker sequence must contain exact O2-O5 prefixes")
        if any(not isinstance(stage, FullHistoryTrackerStage) for stage in self.stages):
            raise ValueError("tracker sequence contains an invalid stage")
        if [stage.horizon for stage in self.stages] != [2, 3, 4, 5]:
            raise ValueError("tracker sequence must contain exact O2-O5 prefixes")
        final = self.stages[-1].key
        for stage in self.stages:
            key = stage.key
            if any(
                key[field] != final[field]
                for field in (
                    "reference_scene_id",
                    "master_sequence_id",
                    "order_id",
                )
            ):
                raise ValueError("tracker sequence scope differs")
            horizon = stage.horizon
            if (
                key["history_scan_ids"] != final["history_scan_ids"][:horizon]
                or key["scan_indices"] != final["scan_indices"][:horizon]
            ):
                raise ValueError("tracker sequence prefixes are not nested causally")

    @property
    def reference_scene_id(self) -> str:
        return str(self.stages[0].key["reference_scene_id"])

    @property
    def master_sequence_id(self) -> str:
        return str(self.stages[0].key["master_sequence_id"])

    @property
    def order_id(self) -> str:
        return str(self.stages[0].key["order_id"])

    @property
    def sequence_id(self) -> str:
        return f"{self.master_sequence_id}:{self.order_id}"


@dataclass(frozen=True)
class FullHistoryTrackingResult:
    updates: dict[tuple[str, str, str], tuple[IdentityAssignmentUpdate, ...]]
    per_sequence_metrics: dict[tuple[str, str, str, int], dict[str, int | float | None]]
    references: dict[tuple[str, str], str]


def run_full_history_tracker(
    sequence: FullHistoryTrackerSequence,
    factory: Callable[[str], object],
    *,
    method_id: str,
) -> tuple[TrackStep, ...]:
    if not isinstance(sequence, FullHistoryTrackerSequence):
        raise ValueError("tracker input must be a validated sequence")
    if not callable(factory) or not isinstance(method_id, str) or not method_id:
        raise ValueError("tracker factory and method ID are required")
    tracker = factory(sequence.sequence_id)
    steps = []
    for stage in sequence.stages:
        step = tracker.step(stage.observation, stage_id=stage.horizon)
        if not isinstance(step, TrackStep) or step.method != method_id:
            raise ValueError("tracker step method differs from its factory")
        steps.append(step)
    return tuple(steps)


def build_full_history_tracker_factories(
    *,
    config_path: str | Path = PROJECT_ROOT / "artifacts/P6A/configs/p6a_default.yaml",
) -> dict[str, Callable[[str], object]]:
    from scripts.evaluate_persist4d_p6a import build_tracker_factories

    path = Path(config_path)
    system_config_path = (
        PROJECT_ROOT / "configs/system_comparison/persist4d_incumbent.yaml"
    )
    if path.is_symlink() or not path.is_file():
        raise ValueError("frozen P6-A config is unavailable")
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        system_config = yaml.safe_load(system_config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ValueError("frozen tracker config cannot be decoded") from error
    if not isinstance(config, Mapping) or not isinstance(system_config, Mapping):
        raise ValueError("frozen tracker config must be a mapping")
    try:
        expected_digest = system_config["sources"]["p6a_config"]["sha256"]
    except (KeyError, TypeError) as error:
        raise ValueError(
            "system comparison lacks the frozen P6-A config hash"
        ) from error
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_digest:
        raise ValueError("frozen P6-A config hash differs")
    factories = build_tracker_factories(config)
    return {method: factories[method] for method in ("B1", "B2", "B3", "B4")}


def _native_update(stage: FullHistoryTrackerStage) -> IdentityAssignmentUpdate:
    return match_identity_update(
        horizon=stage.horizon,
        gt_ids=stage.gt_ids,
        gt_classes=stage.gt_classes,
        gt_masks=stage.gt_masks,
        issued_ids=stage.native_issued_ids,
        pred_classes=stage.pred_classes,
        pred_masks=stage.pred_masks,
        minimum_iou=0.5,
    )


def _tracked_update(
    stage: FullHistoryTrackerStage,
    step: TrackStep,
) -> IdentityAssignmentUpdate:
    positions = {
        int(value): index for index, value in enumerate(stage.local_query_ids.tolist())
    }
    issued = []
    for local_id in stage.native_issued_ids.tolist():
        track_id = step.track_ids[positions[int(local_id)]]
        if isinstance(track_id, bool) or not isinstance(track_id, int) or track_id < 0:
            raise ValueError("tracker issued IDs must be non-negative integers")
        issued.append(track_id)
    return match_identity_update(
        horizon=stage.horizon,
        gt_ids=stage.gt_ids,
        gt_classes=stage.gt_classes,
        gt_masks=stage.gt_masks,
        issued_ids=torch.tensor(issued, dtype=torch.long),
        pred_classes=stage.pred_classes,
        pred_masks=stage.pred_masks,
        minimum_iou=0.5,
    )


def evaluate_full_history_tracker_sequences(
    sequences: Iterable[FullHistoryTrackerSequence],
    *,
    tracker_factories: Mapping[str, Callable[[str], object]],
) -> FullHistoryTrackingResult:
    if isinstance(sequences, (str, bytes)) or not isinstance(sequences, Iterable):
        raise ValueError("tracker evaluation requires sequences")
    if not isinstance(tracker_factories, Mapping) or not tracker_factories:
        raise ValueError("tracker evaluation requires factories")
    if not set(tracker_factories) <= {"B1", "B2", "B3", "B4"}:
        raise ValueError("tracker evaluation received an unregistered method")
    updates: dict[tuple[str, str, str], tuple[IdentityAssignmentUpdate, ...]] = {}
    metrics: dict[tuple[str, str, str, int], dict[str, int | float | None]] = {}
    references: dict[tuple[str, str], str] = {}
    seen_scopes: set[tuple[str, str]] = set()
    for sequence in sequences:
        if not isinstance(sequence, FullHistoryTrackerSequence):
            raise ValueError("tracker evaluation sequence is invalid")
        scope = (sequence.master_sequence_id, sequence.order_id)
        if scope in seen_scopes:
            raise ValueError("tracker evaluation contains a duplicate sequence scope")
        seen_scopes.add(scope)
        references[scope] = sequence.reference_scene_id
        method_updates: dict[str, tuple[IdentityAssignmentUpdate, ...]] = {
            "FullHistoryNative": tuple(
                _native_update(stage) for stage in sequence.stages
            )
        }
        for method, factory in tracker_factories.items():
            steps = run_full_history_tracker(sequence, factory, method_id=method)
            method_updates[method] = tuple(
                _tracked_update(stage, step)
                for stage, step in zip(sequence.stages, steps, strict=True)
            )
        for method, values in method_updates.items():
            update_key = (*scope, method)
            updates[update_key] = values
            for horizon in range(2, 6):
                metrics[(*scope, method, horizon)] = (
                    compute_deployment_identity_metrics(values[: horizon - 1])
                )
    if not seen_scopes:
        raise ValueError("tracker evaluation requires sequences")
    return FullHistoryTrackingResult(
        updates=updates,
        per_sequence_metrics=metrics,
        references=references,
    )


def full_history_tracker_identity_rows(
    result: FullHistoryTrackingResult,
) -> list[dict[str, object]]:
    if not isinstance(result, FullHistoryTrackingResult):
        raise ValueError("tracker result is invalid")
    rows = []
    for (
        master,
        order,
        method,
        horizon,
    ), metrics in result.per_sequence_metrics.items():
        rows.append(
            {
                "method_id": method,
                "method": PAPER_METHOD_NAMES[method],
                "reference_scene_id": result.references[(master, order)],
                "master_sequence_id": master,
                "order_id": order,
                "horizon": horizon,
                "tracker_initialization_horizon": 2,
                **metrics,
            }
        )
    rows.sort(
        key=lambda row: (
            str(row["reference_scene_id"]),
            str(row["master_sequence_id"]),
            str(row["order_id"]),
            int(row["horizon"]),
            str(row["method_id"]),
        )
    )
    return rows


def tracking_content_sha256(value: Mapping[str, object]) -> str:
    payload = dict(value)
    payload.pop("content_sha256", None)
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_full_history_tracking_artifact(
    result: FullHistoryTrackingResult,
    *,
    reviewer_manifest_content_sha256: str,
    replay_manifest_content_sha256: str,
    sidecar_manifest_content_sha256: str,
    tracker_config_sha256: str,
    source_commit: str,
    expected_sequence_count: int = 129,
) -> dict[str, object]:
    if (
        isinstance(expected_sequence_count, bool)
        or not isinstance(expected_sequence_count, int)
        or expected_sequence_count <= 0
    ):
        raise ValueError("expected sequence count must be positive")
    digests = (
        reviewer_manifest_content_sha256,
        replay_manifest_content_sha256,
        sidecar_manifest_content_sha256,
        tracker_config_sha256,
    )
    if any(
        not isinstance(value, str) or _HEX64.fullmatch(value) is None
        for value in digests
    ):
        raise ValueError("tracking artifact SHA256 is invalid")
    if not isinstance(source_commit, str) or _HEX40.fullmatch(source_commit) is None:
        raise ValueError("tracking artifact source commit is invalid")
    rows = full_history_tracker_identity_rows(result)
    scopes = {(str(row["master_sequence_id"]), str(row["order_id"])) for row in rows}
    if len(scopes) != expected_sequence_count:
        raise ValueError("tracking artifact sequence coverage differs")
    expected_cells = {
        (master, order, method, horizon)
        for master, order in scopes
        for method in PAPER_METHOD_NAMES
        for horizon in range(2, 6)
    }
    actual_cells = {
        (
            str(row["master_sequence_id"]),
            str(row["order_id"]),
            str(row["method_id"]),
            int(row["horizon"]),
        )
        for row in rows
    }
    if actual_cells != expected_cells or len(rows) != len(expected_cells):
        raise ValueError("tracking artifact cell coverage differs")
    artifact: dict[str, object] = {
        "schema_version": 1,
        "status": "pass",
        "source_commit": source_commit,
        "tracker_initialization_horizon": 2,
        "identity_match_minimum_iou": 0.5,
        "sequence_count": len(scopes),
        "row_count": len(rows),
        "provenance": {
            "reviewer_manifest_content_sha256": reviewer_manifest_content_sha256,
            "replay_manifest_content_sha256": replay_manifest_content_sha256,
            "sidecar_manifest_content_sha256": sidecar_manifest_content_sha256,
            "tracker_config_sha256": tracker_config_sha256,
        },
        "rows": rows,
    }
    artifact["content_sha256"] = tracking_content_sha256(artifact)
    return artifact


def run_full_history_tracking_evaluation(
    *,
    output_path: str | Path = TRACKING_ARTIFACT_PATH,
) -> dict[str, object]:
    from scripts.run_reviewer_closure import (
        _git_head,
        _require_source_tree_clean,
        publish_exact_json,
    )

    _require_source_tree_clean()
    reviewer = _load_json(REVIEWER_MANIFEST_PATH, name="reviewer manifest")
    replay = _load_json(REPLAY_MANIFEST_PATH, name="replay manifest")
    sidecar = _load_json(SIDECAR_MANIFEST_PATH, name="sidecar manifest")
    config_path = PROJECT_ROOT / "artifacts/P6A/configs/p6a_default.yaml"
    result = evaluate_full_history_tracker_sequences(
        iter_full_history_tracker_sequences(),
        tracker_factories=build_full_history_tracker_factories(config_path=config_path),
    )
    artifact = build_full_history_tracking_artifact(
        result,
        reviewer_manifest_content_sha256=reviewer["content_sha256"],
        replay_manifest_content_sha256=replay["content_sha256"],
        sidecar_manifest_content_sha256=sidecar["content_sha256"],
        tracker_config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
        source_commit=_git_head(),
    )
    publish_exact_json(output_path, artifact)
    return artifact


__all__ = [
    "FullHistoryTrackerSequence",
    "FullHistoryTrackerStage",
    "FullHistoryTrackingResult",
    "PAPER_METHOD_NAMES",
    "build_full_history_tracking_artifact",
    "build_full_history_tracker_factories",
    "evaluate_full_history_tracker_sequences",
    "full_history_tracker_identity_rows",
    "iter_full_history_tracker_sequences",
    "run_full_history_tracking_evaluation",
    "run_full_history_tracker",
    "tracking_content_sha256",
    "validate_full_history_tracking_manifests",
]


def _load_json(path: str | Path, *, name: str) -> dict[str, Any]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"{name} is not a regular file")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} cannot be decoded") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return dict(value)


def _key_identity(key: Mapping[str, object]) -> str:
    return json.dumps(
        sidecar_key_for_source_prediction(key),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def validate_full_history_tracking_manifests(
    *,
    reviewer_manifest: Mapping[str, object],
    replay_manifest: Mapping[str, object],
    sidecar_manifest: Mapping[str, object],
) -> dict[str, object]:
    from scripts.reviewer_closure_protocol import (
        full_history_observation_keys,
        validate_reviewer_closure_manifest,
    )
    from scripts.reviewer_closure_sidecar import (
        build_full_history_observation_sidecar_manifest,
        source_prediction_entry_for_key,
        validate_source_prediction_manifest,
    )
    from scripts.system_comparison_inference import build_full_history_cache_manifest

    config_path = PROJECT_ROOT / "configs/reviewer_closure/protocol.yaml"
    system_manifest_path = (
        PROJECT_ROOT / "artifacts/system_comparison/system_comparison_manifest.json"
    )
    source_manifest_path = (
        PROJECT_ROOT
        / "artifacts/system_comparison/full_history_predictions/manifest.json"
    )
    reviewer = validate_reviewer_closure_manifest(
        reviewer_manifest,
        config_path=config_path,
        system_manifest_path=system_manifest_path,
    )
    system = _load_json(system_manifest_path, name="system manifest")
    source = validate_source_prediction_manifest(
        _load_json(source_manifest_path, name="source prediction manifest"),
        system_manifest=system,
    )
    expected_keys = full_history_observation_keys(reviewer)
    replay_entries = replay_manifest.get("entries")
    if isinstance(replay_entries, (str, bytes)) or not isinstance(
        replay_entries, Sequence
    ):
        raise ValueError("replay prediction manifest entries are invalid")
    replay = build_full_history_cache_manifest(
        replay_entries,
        expected_keys=[
            source_prediction_entry_for_key(source, key)["key"] for key in expected_keys
        ],
        expected_provenance=source["provenance"],
    )
    if replay != replay_manifest:
        raise ValueError("replay prediction manifest content differs")
    sidecar_entries = sidecar_manifest.get("entries")
    sidecar_commit = sidecar_manifest.get("sidecar_code_commit")
    if isinstance(sidecar_entries, (str, bytes)) or not isinstance(
        sidecar_entries, Sequence
    ):
        raise ValueError("sidecar manifest entries are invalid")
    sidecar = build_full_history_observation_sidecar_manifest(
        sidecar_entries,
        expected_keys=expected_keys,
        source_prediction_manifest=source,
        replay_prediction_manifest=replay,
        system_manifest=system,
        reviewer_manifest=reviewer,
        sidecar_code_commit=sidecar_commit,
    )
    if sidecar != sidecar_manifest:
        raise ValueError("sidecar manifest content differs")
    return {
        "reviewer_manifest": reviewer,
        "source_prediction_manifest": source,
        "replay_manifest": replay,
        "sidecar_manifest": sidecar,
        "expected_keys": expected_keys,
    }


def _require_exact_cache_files(
    directory: str | Path,
    entries: Sequence[Mapping[str, object]],
    *,
    name: str,
) -> Path:
    root = Path(directory)
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"{name} cache directory is unavailable")
    expected = {str(entry["filename"]) for entry in entries}
    actual = {path.name for path in root.iterdir()}
    if actual != expected:
        raise ValueError(f"{name} cache directory coverage differs")
    return root


def _tracker_stage_from_payloads(
    sidecar: Mapping[str, object],
    replay: Mapping[str, object],
) -> FullHistoryTrackerStage:
    key = sidecar_key_for_source_prediction(sidecar["key"])
    replay_key = sidecar_key_for_source_prediction(replay["key"])
    provenance = sidecar["provenance"]
    if (
        key != replay_key
        or provenance["source_prediction_content_sha256"] != replay["content_sha256"]
    ):
        raise ValueError("sidecar and replay prediction binding differs")
    observation = sidecar["observation"]
    current_masks = unpack_bool_matrix(observation["current_stage_masks"])
    target = replay["target"]
    stages = _integer_tensor(target["temporal_stages"], name="target stages")
    selector = stages == int(key["horizon"]) - 1
    target_masks = unpack_bool_matrix(target["masks"])[:, selector]
    identity = replay["identity_prediction"]
    return FullHistoryTrackerStage(
        key=key,
        observation=freeze_observation(
            {
                "features": observation["features"],
                "class_prob": observation["class_prob"],
                "confidence": observation["confidence"],
                "valid": observation["valid"],
                "latest_mask": current_masks.float(),
                "mask_support": observation["mask_support"],
            }
        ),
        local_query_ids=observation["local_query_ids"],
        gt_ids=target["ids"],
        gt_classes=target["labels"],
        gt_masks=target_masks,
        native_issued_ids=identity["issued_ids"],
        pred_classes=identity["pred_classes"],
        pred_masks=unpack_bool_matrix(identity["pred_masks"]),
    )


def iter_full_history_tracker_sequences(
    *,
    reviewer_manifest_path: str | Path = REVIEWER_MANIFEST_PATH,
    replay_manifest_path: str | Path = REPLAY_MANIFEST_PATH,
    replay_entry_root: str | Path = REPLAY_ENTRY_ROOT,
    sidecar_manifest_path: str | Path = SIDECAR_MANIFEST_PATH,
    sidecar_entry_root: str | Path = SIDECAR_ENTRY_ROOT,
):
    from scripts.reviewer_closure_sidecar import (
        load_full_history_observation_sidecar_entry,
    )
    from scripts.system_comparison_inference import load_full_history_cache_entry

    validated = validate_full_history_tracking_manifests(
        reviewer_manifest=_load_json(reviewer_manifest_path, name="reviewer manifest"),
        replay_manifest=_load_json(replay_manifest_path, name="replay manifest"),
        sidecar_manifest=_load_json(sidecar_manifest_path, name="sidecar manifest"),
    )
    replay_manifest = validated["replay_manifest"]
    sidecar_manifest = validated["sidecar_manifest"]
    replay_root = _require_exact_cache_files(
        replay_entry_root,
        replay_manifest["entries"],
        name="replay prediction",
    )
    sidecar_root = _require_exact_cache_files(
        sidecar_entry_root,
        sidecar_manifest["entries"],
        name="sidecar",
    )
    replay_index = {
        _key_identity(entry["key"]): entry for entry in replay_manifest["entries"]
    }
    sidecar_index = {
        _key_identity(entry["key"]): entry for entry in sidecar_manifest["entries"]
    }
    expected_keys = validated["expected_keys"]
    scopes: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    for key in expected_keys:
        scope = (str(key["master_sequence_id"]), str(key["order_id"]))
        scopes.setdefault(scope, []).append(key)
    for scope in sorted(scopes):
        keys = sorted(scopes[scope], key=lambda key: int(key["horizon"]))
        stages = []
        for key in keys:
            identity = _key_identity(key)
            replay = load_full_history_cache_entry(
                replay_root,
                replay_index[identity],
                expected_provenance=replay_manifest["provenance"],
            )
            sidecar = load_full_history_observation_sidecar_entry(
                sidecar_root,
                sidecar_index[identity],
            )
            stages.append(_tracker_stage_from_payloads(sidecar, replay))
        yield FullHistoryTrackerSequence(stages=tuple(stages))
