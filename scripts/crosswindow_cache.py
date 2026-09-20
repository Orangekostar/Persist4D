"""Asset and data-role contracts for the CrossWindow evidence campaign."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from datasets.task_memory_episode import StageMeta
from models.task_memory_routing import PredictionObservation
from scripts.rescene_task_postprocess import OfficialTaskPrediction


class CrossWindowCacheError(ValueError):
    """Raised when cache provenance or data roles violate the campaign contract."""


ASSET_KEYS = (
    "data_root",
    "rio_metadata",
    "r1_checkpoint",
    "concerto_pretrained",
    "metric_dataset_spec",
    "dev_base_cache_root",
    "dev_supplement_root",
    "pb_base_cache_root",
    "pb_supplement_root",
    "fh_native_cache_root",
    "train_observation_cache_root",
    "external_run_root",
)

ASSET_ENVIRONMENT = {
    "data_root": "PERSIST4D_DATA_ROOT",
    "rio_metadata": "PERSIST4D_RIO_METADATA",
    "r1_checkpoint": "PERSIST4D_R1_CHECKPOINT",
    "concerto_pretrained": "PERSIST4D_CONCERTO_PRETRAINED",
    "metric_dataset_spec": "PERSIST4D_METRIC_DATASET_SPEC",
    "dev_base_cache_root": "PERSIST4D_DEV_BASE_CACHE_ROOT",
    "dev_supplement_root": "PERSIST4D_DEV_SUPPLEMENT_ROOT",
    "pb_base_cache_root": "PERSIST4D_PB_BASE_CACHE_ROOT",
    "pb_supplement_root": "PERSIST4D_PB_SUPPLEMENT_ROOT",
    "fh_native_cache_root": "PERSIST4D_FH_NATIVE_CACHE_ROOT",
    "train_observation_cache_root": "PERSIST4D_TRAIN_OBSERVATION_CACHE_ROOT",
    "external_run_root": "PERSIST4D_RUN_ROOT",
}

_FALLBACK_ALIASES = {
    "data_root": ("data_root", "external:data_root"),
    "rio_metadata": ("rio_metadata", "external:rio_metadata"),
    "r1_checkpoint": ("r1_checkpoint", "external:r1_checkpoint"),
    "concerto_pretrained": (
        "concerto_pretrained",
        "external:concerto_pretrained",
    ),
    "metric_dataset_spec": ("metric_dataset_spec",),
    "dev_base_cache_root": ("dev_base_cache_root",),
    "dev_supplement_root": ("dev_supplement_root",),
    "pb_base_cache_root": ("pb_base_cache_root",),
    "pb_supplement_root": ("pb_supplement_root",),
    "fh_native_cache_root": ("fh_native_cache_root",),
    "train_observation_cache_root": ("train_observation_cache_root",),
    "external_run_root": ("external_run_root", "external:run_root"),
}


@dataclass(frozen=True)
class AssetResolution:
    values: dict[str, str | None]
    sources: dict[str, str]


@dataclass(frozen=True)
class CampaignUnit:
    logical_index: int
    role: str
    reference_id: str
    sequence_id: str
    base_filename: str
    base_sha256: str
    base_bytes: int
    supplement_filename: str
    supplement_sha256: str
    supplement_bytes: int

    @property
    def logical_unit_id(self) -> str:
        return f"{self.role}:{self.logical_index:05d}"

    @property
    def physical_pair(self) -> tuple[str, str]:
        return self.base_sha256, self.supplement_sha256


@dataclass(frozen=True, order=True)
class CandidateKey:
    producer_id: str
    episode_id: str
    order_id: str
    absolute_stage: int
    source_query_id: int
    source_class_id: int
    candidate_index: int


@dataclass(frozen=True)
class CandidateSlice:
    scan_id: str
    original_vertex_ids: Tensor
    canonical_vertex_ids: Tensor
    mask: Tensor
    source_window: tuple[str, ...]
    original_score: float


@dataclass(frozen=True)
class CandidateRecord:
    key: CandidateKey
    predicted_class_id: int
    score: float
    slices: tuple[CandidateSlice, ...]


@dataclass(frozen=True)
class QueryGroup:
    source_query_id: int
    feature: Tensor
    class_prob: Tensor
    confidence: float
    valid: bool
    current_supported: bool
    previous_supported: bool
    candidate_indices: tuple[int, ...]


@dataclass(frozen=True)
class CanonicalFrame:
    producer_id: str
    reference_id: str
    episode_id: str
    order_id: str
    absolute_stage: int
    source_window: tuple[str, ...]
    canonical_vertex_ids: dict[str, Tensor]
    groups: tuple[QueryGroup, ...]
    candidates: tuple[CandidateRecord, ...]


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise CrossWindowCacheError("asset values must be non-empty strings or null")
    return value


def resolve_assets(
    *,
    explicit: Mapping[str, Any],
    environ: Mapping[str, str],
    fallback: Mapping[str, Any],
) -> AssetResolution:
    """Resolve asset values without probing paths or leaking them into Git artifacts."""
    unknown = set(explicit) - set(ASSET_KEYS)
    if unknown:
        raise CrossWindowCacheError(f"unknown explicit asset keys: {sorted(unknown)}")
    values: dict[str, str | None] = {}
    sources: dict[str, str] = {}
    for key in ASSET_KEYS:
        explicit_value = _optional_text(explicit.get(key))
        environment_value = _optional_text(environ.get(ASSET_ENVIRONMENT[key]))
        fallback_value = None
        for alias in _FALLBACK_ALIASES[key]:
            if alias in fallback and fallback[alias] is not None:
                fallback_value = _optional_text(fallback[alias])
                break
        if explicit_value is not None:
            values[key], sources[key] = explicit_value, "cli"
        elif environment_value is not None:
            values[key], sources[key] = environment_value, "environment"
        elif fallback_value is not None:
            values[key], sources[key] = fallback_value, "fallback"
        else:
            values[key], sources[key] = None, "unresolved"
    if values["metric_dataset_spec"] is None and values["data_root"] is not None:
        data_root = Path(values["data_root"])
        candidates = (
            data_root / "processed/rio/rio.yaml",
            data_root / "data/processed/rio/rio.yaml",
        )
        specification = next((path for path in candidates if path.is_file()), None)
        if specification is not None:
            values["metric_dataset_spec"] = str(specification)
            sources["metric_dataset_spec"] = "derived:data_root"
    return AssetResolution(values=values, sources=sources)


def _id_tensor(values: Sequence[int] | Tensor, *, name: str) -> Tensor:
    if isinstance(values, Tensor):
        tensor = values.detach().cpu().contiguous()
        if tensor.ndim != 1 or tensor.dtype == torch.bool or tensor.is_floating_point():
            raise CrossWindowCacheError(f"{name} must be a rank-1 integer sequence")
        return tensor.long()
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise CrossWindowCacheError(f"{name} must be a rank-1 integer sequence")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise CrossWindowCacheError(f"{name} must contain integers")
    return torch.tensor(list(values), dtype=torch.long)


def align_mask(
    mask: Tensor,
    *,
    from_ids: Sequence[int] | Tensor,
    to_ids: Sequence[int] | Tensor,
) -> Tensor:
    """Align the first mask dimension from one explicit vertex order to another."""
    if not isinstance(mask, Tensor) or mask.ndim < 1:
        raise CrossWindowCacheError("mask must be a tensor with a point dimension")
    source = _id_tensor(from_ids, name="from_ids")
    destination = _id_tensor(to_ids, name="to_ids")
    if source.numel() != mask.shape[0]:
        raise CrossWindowCacheError("mask and from_ids point counts differ")
    if (
        source.unique().numel() != source.numel()
        or destination.unique().numel() != destination.numel()
    ):
        raise CrossWindowCacheError("vertex IDs must be unique")
    if source.numel() != destination.numel() or set(source.tolist()) != set(
        destination.tolist()
    ):
        raise CrossWindowCacheError("vertex ID sets differ")
    source_positions = {value: index for index, value in enumerate(source.tolist())}
    reorder = torch.tensor(
        [source_positions[value] for value in destination.tolist()],
        dtype=torch.long,
        device=mask.device,
    )
    return mask.index_select(0, reorder).clone()


def compare_vertex_order(
    previous_ids: Sequence[int] | Tensor,
    current_ids: Sequence[int] | Tensor,
) -> str:
    previous = _id_tensor(previous_ids, name="previous_ids")
    current = _id_tensor(current_ids, name="current_ids")
    if (
        previous.unique().numel() != previous.numel()
        or current.unique().numel() != current.numel()
    ):
        raise CrossWindowCacheError("vertex IDs must be unique")
    if torch.equal(previous, current):
        return "IDENTICAL"
    if previous.numel() == current.numel() and set(previous.tolist()) == set(
        current.tolist()
    ):
        return "NON_IDENTITY"
    return "SET_MISMATCH"


def _validate_frame_lineage(
    observation: PredictionObservation,
    prediction: OfficialTaskPrediction,
    stage_meta: StageMeta,
) -> None:
    observation.validate()
    prediction.validate()
    if observation.batch_size != 1:
        raise CrossWindowCacheError(
            "canonical frame requires observation batch size one"
        )
    if not isinstance(stage_meta, StageMeta):
        raise CrossWindowCacheError("stage metadata must be StageMeta")
    point_count = prediction.pred_masks.shape[0]
    if stage_meta.local_stage_ids.numel() != point_count or not torch.equal(
        stage_meta.local_stage_ids.detach().cpu().long(),
        prediction.temporal_stages.detach().cpu().long(),
    ):
        raise CrossWindowCacheError("prediction and stage point lineage differ")
    scan_count = len(stage_meta.scan_ids_in_window)
    offsets = stage_meta.scan_vertex_offsets.detach().cpu().long()
    if (
        offsets.shape != (scan_count + 1,)
        or offsets[0].item() != 0
        or offsets[-1].item() != point_count
        or torch.any(offsets[1:] <= offsets[:-1]).item()
        or len(stage_meta.original_vertex_ids) != scan_count
    ):
        raise CrossWindowCacheError("stage scan offsets differ from prediction points")
    query_ids = prediction.source_query_ids.detach().cpu().long()
    if torch.any((query_ids < 0) | (query_ids >= observation.query_count)).item():
        raise CrossWindowCacheError("candidate source query is outside observation")


def build_canonical_frame(
    *,
    producer_id: str,
    order_id: str,
    observation: PredictionObservation,
    prediction: OfficialTaskPrediction,
    stage_meta: StageMeta,
) -> CanonicalFrame:
    """Preserve the complete prediction ledger in a per-scan canonical point order."""
    if not isinstance(producer_id, str) or not producer_id:
        raise CrossWindowCacheError("producer_id must be a non-empty string")
    if not isinstance(order_id, str) or not order_id:
        raise CrossWindowCacheError("order_id must be a non-empty string")
    _validate_frame_lineage(observation, prediction, stage_meta)

    source_window = tuple(stage_meta.scan_ids_in_window)
    if len(set(source_window)) != len(source_window):
        raise CrossWindowCacheError("source window scan IDs must be unique")
    offsets = stage_meta.scan_vertex_offsets.detach().cpu().long()
    canonical_vertex_ids: dict[str, Tensor] = {}
    canonical_masks: dict[str, Tensor] = {}
    for local_stage, scan_id in enumerate(source_window):
        start = int(offsets[local_stage].item())
        stop = int(offsets[local_stage + 1].item())
        vertex_ids = stage_meta.original_vertex_ids[local_stage].detach().cpu().long()
        if vertex_ids.shape != (stop - start,):
            raise CrossWindowCacheError("scan vertex IDs differ from scan point count")
        canonical_ids = vertex_ids.sort().values
        canonical_vertex_ids[scan_id] = canonical_ids.clone()
        canonical_masks[scan_id] = align_mask(
            prediction.pred_masks[start:stop].detach().cpu(),
            from_ids=vertex_ids,
            to_ids=canonical_ids,
        ).bool()

    source_query_ids = prediction.source_query_ids.detach().cpu().long().tolist()
    source_class_ids = prediction.source_class_ids.detach().cpu().long().tolist()
    predicted_classes = prediction.pred_classes.detach().cpu().long().tolist()
    scores = prediction.pred_scores.detach().cpu().float().tolist()
    candidates = []
    for candidate_index, (
        query_id,
        source_class_id,
        predicted_class,
        score,
    ) in enumerate(
        zip(
            source_query_ids,
            source_class_ids,
            predicted_classes,
            scores,
            strict=True,
        )
    ):
        key = CandidateKey(
            producer_id=producer_id,
            episode_id=stage_meta.episode_id,
            order_id=order_id,
            absolute_stage=stage_meta.absolute_stage_index,
            source_query_id=int(query_id),
            source_class_id=int(source_class_id),
            candidate_index=candidate_index,
        )
        slices = tuple(
            CandidateSlice(
                scan_id=scan_id,
                original_vertex_ids=stage_meta.original_vertex_ids[local_stage]
                .detach()
                .cpu()
                .long()
                .clone(),
                canonical_vertex_ids=canonical_vertex_ids[scan_id].clone(),
                mask=canonical_masks[scan_id][:, candidate_index].clone(),
                source_window=source_window,
                original_score=float(score),
            )
            for local_stage, scan_id in enumerate(source_window)
        )
        candidates.append(
            CandidateRecord(
                key=key,
                predicted_class_id=int(predicted_class),
                score=float(score),
                slices=slices,
            )
        )

    candidates_by_query: dict[int, list[int]] = {}
    for index, candidate in enumerate(candidates):
        candidates_by_query.setdefault(candidate.key.source_query_id, []).append(index)
    valid_queries = set(
        observation.valid[0].detach().cpu().nonzero(as_tuple=True)[0].tolist()
    )
    group_queries = sorted(valid_queries | set(candidates_by_query))
    groups = tuple(
        QueryGroup(
            source_query_id=query_id,
            feature=observation.features[0, query_id].detach().cpu().clone(),
            class_prob=observation.class_prob[0, query_id].detach().cpu().clone(),
            confidence=float(observation.confidence[0, query_id].detach().cpu().item()),
            valid=bool(observation.valid[0, query_id].item()),
            current_supported=bool(observation.current_supported[0, query_id].item()),
            previous_supported=bool(observation.previous_supported[0, query_id].item()),
            candidate_indices=tuple(candidates_by_query.get(query_id, ())),
        )
        for query_id in group_queries
    )
    return CanonicalFrame(
        producer_id=producer_id,
        reference_id=stage_meta.reference_id,
        episode_id=stage_meta.episode_id,
        order_id=order_id,
        absolute_stage=stage_meta.absolute_stage_index,
        source_window=source_window,
        canonical_vertex_ids=canonical_vertex_ids,
        groups=groups,
        candidates=tuple(candidates),
    )


def _manifest_record(
    value: object,
    *,
    name: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise CrossWindowCacheError(f"{name} manifest record must be a mapping")
    record = dict(value)
    for field in ("reference_id", "sequence_id", "filename", "sha256"):
        item = record.get(field)
        if not isinstance(item, str) or not item:
            raise CrossWindowCacheError(f"{name} manifest record lacks {field}")
    digest = str(record["sha256"])
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise CrossWindowCacheError(f"{name} manifest record has invalid sha256")
    size = record.get("bytes")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise CrossWindowCacheError(f"{name} manifest record has invalid bytes")
    return record


def iter_campaign_units(
    *,
    role: str,
    role_references: Sequence[str],
    base_manifest: Mapping[str, object],
    supplement_manifest: Mapping[str, object],
) -> Iterator[CampaignUnit]:
    """Bind logical units to physical cache pairs without deduplicating records."""
    if not isinstance(role, str) or not role:
        raise CrossWindowCacheError("campaign role must be a non-empty string")
    if (
        isinstance(role_references, (str, bytes))
        or len(set(role_references)) != len(role_references)
        or any(
            not isinstance(reference, str) or not reference
            for reference in role_references
        )
    ):
        raise CrossWindowCacheError("role references must be unique non-empty strings")
    base_values = base_manifest.get("records")
    supplement_values = supplement_manifest.get("records")
    if not isinstance(base_values, list) or not isinstance(supplement_values, list):
        raise CrossWindowCacheError("cache manifests must contain record lists")
    if len(base_values) != len(supplement_values):
        raise CrossWindowCacheError("base and supplement logical-unit counts differ")

    selected = set(role_references)
    seen_references: set[str] = set()
    units = []
    for logical_index, (base_value, supplement_value) in enumerate(
        zip(base_values, supplement_values, strict=True)
    ):
        base = _manifest_record(base_value, name="base")
        supplement = _manifest_record(supplement_value, name="supplement")
        identity = (base["reference_id"], base["sequence_id"])
        if identity != (supplement["reference_id"], supplement["sequence_id"]):
            raise CrossWindowCacheError(
                "base and supplement logical-unit identities differ"
            )
        if supplement.get("base_cache_sha256") != base["sha256"]:
            raise CrossWindowCacheError("supplement does not bind to the base cache")
        reference_id = str(base["reference_id"])
        if reference_id not in selected:
            continue
        seen_references.add(reference_id)
        units.append(
            CampaignUnit(
                logical_index=logical_index,
                role=role,
                reference_id=reference_id,
                sequence_id=str(base["sequence_id"]),
                base_filename=str(base["filename"]),
                base_sha256=str(base["sha256"]),
                base_bytes=int(base["bytes"]),
                supplement_filename=str(supplement["filename"]),
                supplement_sha256=str(supplement["sha256"]),
                supplement_bytes=int(supplement["bytes"]),
            )
        )
    if missing := selected - seen_references:
        raise CrossWindowCacheError(
            f"role references have no logical units: {', '.join(sorted(missing))}"
        )
    return iter(units)


def build_data_roles(
    data_contract: Mapping[str, Any],
    *,
    available_references: Sequence[str],
) -> dict[str, list[str]]:
    """Build the preregistered reference split without splitting a reference."""
    inventory = data_contract.get("native_reference_inventory")
    if isinstance(inventory, (str, bytes)) or not isinstance(inventory, Sequence):
        raise CrossWindowCacheError("data contract lacks native_reference_inventory")
    available = set(available_references)
    if len(available) != len(available_references) or any(
        not isinstance(value, str) or not value for value in available_references
    ):
        raise CrossWindowCacheError(
            "available references must be unique non-empty strings"
        )
    by_role: dict[str, list[str]] = {
        "development": [],
        "adaptation": [],
        "additional_native_refs": [],
    }
    seen: set[str] = set()
    for record in inventory:
        if not isinstance(record, Mapping):
            raise CrossWindowCacheError("reference inventory entries must be mappings")
        reference = record.get("reference_id")
        role = record.get("role")
        if not isinstance(reference, str) or not reference:
            raise CrossWindowCacheError("reference inventory contains an invalid ID")
        if reference in seen:
            raise CrossWindowCacheError("reference inventory contains duplicate IDs")
        seen.add(reference)
        if role in by_role:
            by_role[str(role)].append(reference)
    development = [value for value in by_role["development"] if value in available]
    development.sort(
        key=lambda value: hashlib.sha256(f"crosswindow-v1:{value}".encode()).hexdigest()
    )
    boundary = len(development) // 2
    return {
        "DEV-CAL": development[:boundary],
        "DEV-SEL": development[boundary:],
        "adaptation": sorted(by_role["adaptation"]),
        "additional_native_refs": sorted(by_role["additional_native_refs"]),
    }


__all__ = [
    "ASSET_ENVIRONMENT",
    "ASSET_KEYS",
    "AssetResolution",
    "CampaignUnit",
    "CandidateKey",
    "CandidateRecord",
    "CandidateSlice",
    "CanonicalFrame",
    "CrossWindowCacheError",
    "QueryGroup",
    "align_mask",
    "build_canonical_frame",
    "build_data_roles",
    "compare_vertex_order",
    "iter_campaign_units",
    "resolve_assets",
]
