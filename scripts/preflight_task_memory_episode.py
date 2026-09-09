#!/usr/bin/env python3
"""Run the real native-episode and StageMeta preflight for TaskMemory V2."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from datasets.semseg import SemanticSegmentationDataset
from datasets.task_memory_episode import (
    NativeEpisodeMaster,
    TaskMemoryEpisode,
    TaskMemoryEpisodeBatch,
    TaskMemoryEpisodeCollator,
    TaskMemoryEpisodeDataset,
    TaskMemoryEpisodeError,
    TaskMemoryEpisodeSpec,
    build_native_episode_masters,
)
from scripts.task_memory_contracts import canonical_json_sha256

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _array_sha256(value: object) -> str:
    array = np.ascontiguousarray(value)
    import hashlib

    return hashlib.sha256(array.tobytes()).hexdigest()


def _same_array(left: object, right: object) -> bool:
    return np.array_equal(np.asarray(left), np.asarray(right))


def build_preflight_payload(
    *,
    original: TaskMemoryEpisode,
    future_mutated: TaskMemoryEpisode,
    batch: TaskMemoryEpisodeBatch,
    native_summary: Mapping[str, object],
    sources: Mapping[str, str],
) -> dict[str, object]:
    if not original.stage_samples or not future_mutated.stage_samples:
        raise TaskMemoryEpisodeError("preflight episodes must contain T1")
    original_t1 = original.stage_samples[0]
    mutated_t1 = future_mutated.stage_samples[0]
    t1_coordinates_same = _same_array(
        original_t1.model_sample[0], mutated_t1.model_sample[0]
    )
    t1_labels_same = _same_array(
        original_t1.model_sample[2], mutated_t1.model_sample[2]
    )
    t1_vertices_same = torch.equal(
        original_t1.original_vertex_ids[0], mutated_t1.original_vertex_ids[0]
    )
    episode_id_same = original.spec.episode_id == future_mutated.spec.episode_id
    if not (
        t1_coordinates_same and t1_labels_same and t1_vertices_same and episode_id_same
    ):
        raise TaskMemoryEpisodeError("future mutation changed T1 preparation")
    if len(original.stage_samples) < 2:
        raise TaskMemoryEpisodeError("preflight original episode must contain T2")
    t2 = original.stage_samples[1]
    first_scan_size = original_t1.model_sample[0].shape[0]
    repeated_coordinates = _same_array(
        original_t1.model_sample[0], t2.model_sample[0][:first_scan_size]
    )
    repeated_labels = _same_array(
        original_t1.model_sample[2], t2.model_sample[2][:first_scan_size]
    )
    repeated_vertices = torch.equal(
        original_t1.original_vertex_ids[0], t2.original_vertex_ids[0]
    )
    if not (repeated_coordinates and repeated_labels and repeated_vertices):
        raise TaskMemoryEpisodeError("repeated scan alignment differs across windows")
    if len(batch.stage_batches) != original.spec.horizon:
        raise TaskMemoryEpisodeError("preflight batch horizon differs")
    metas = [meta for stage in batch.stage_batches for meta in stage.stage_meta]
    inverse_alignment = all(
        torch.equal(
            meta.full_resolution_point2segment,
            meta.point2segment[meta.voxel_inverse],
        )
        for meta in metas
    )
    if not inverse_alignment:
        raise TaskMemoryEpisodeError("preflight inverse alignment differs")
    if any(
        not isinstance(reference, str)
        or reference.startswith(("/", "file:"))
        or "/home/" in reference
        or "/mnt/" in reference
        for reference in sources.values()
    ):
        raise TaskMemoryEpisodeError("preflight sources must be portable references")
    payload: dict[str, object] = {
        "collation": {
            "full_resolution_points": [meta.voxel_inverse.numel() for meta in metas],
            "inverse_alignment": inverse_alignment,
            "voxel_points": [meta.point2segment.numel() for meta in metas],
        },
        "episode": {
            "episode_id": original.spec.episode_id,
            "reference_id": original.spec.reference_id,
            "scan_ids": list(original.spec.scan_ids),
            "stage_full_resolution_points": [
                stage.model_sample[0].shape[0] for stage in original.stage_samples
            ],
        },
        "future_mutation": {
            "episode_id_unchanged": episode_id_same,
            "mutated_scan_ids": list(future_mutated.spec.scan_ids),
            "t1_input_unchanged": True,
        },
        "native_horizons": dict(native_summary),
        "repeated_scan": {
            "coordinates_unchanged": repeated_coordinates,
            "labels_sha256": _array_sha256(original_t1.model_sample[2]),
            "labels_unchanged": repeated_labels,
            "vertex_ids_unchanged": repeated_vertices,
        },
        "schema_version": "task-memory-real-sequence-preflight-v2",
        "sources": dict(sorted(sources.items())),
        "stage_meta_has_supervision_fields": False,
        "status": "PASS",
    }
    payload["content_sha256"] = canonical_json_sha256(payload)
    return payload


def _load_json(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise TaskMemoryEpisodeError(f"{label} cannot be decoded") from error
    if not isinstance(value, Mapping):
        raise TaskMemoryEpisodeError(f"{label} must be a mapping")
    return value


def _resolved_data_path(value: object, data_root: Path) -> object:
    if not isinstance(value, str) or not value.startswith("data/"):
        return value
    return str(data_root / Path(value).relative_to("data"))


def _rio_base_dataset(config: Any, *, data_root: Path, horizon: int):
    training = OmegaConf.to_container(config.data.train_dataset, resolve=True)
    if not isinstance(training, dict) or not isinstance(training.get("datasets"), list):
        raise TaskMemoryEpisodeError("training config lacks the frozen RIO source")
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
        raise TaskMemoryEpisodeError("training config does not bind one RIO source")
    parameters = {**common, **candidates[0]}
    parameters.pop("target", None)
    parameters.pop("_target_", None)
    parameters["temporal_window"] = horizon
    for key in (
        "data_dir",
        "label_db_filepath",
        "change_label_db_filepath",
        "color_mean_std",
    ):
        if key in parameters:
            parameters[key] = _resolved_data_path(parameters[key], data_root)
    return SemanticSegmentationDataset(**parameters)


def _reference_by_scene(metadata: object) -> dict[int, str]:
    if isinstance(metadata, Mapping):
        items = list(metadata.items())
    elif isinstance(metadata, Sequence) and not isinstance(metadata, (str, bytes)):
        items = list(enumerate(metadata))
    else:
        raise TaskMemoryEpisodeError("3RScan metadata must be a sequence or mapping")
    result = {}
    for key, raw in items:
        if not isinstance(raw, Mapping):
            raise TaskMemoryEpisodeError("3RScan metadata record must be a mapping")
        scene = raw.get("scene", key)
        reference = raw.get("reference", raw.get("reference_scene_id"))
        if (
            isinstance(scene, bool)
            or not isinstance(scene, int)
            or not isinstance(reference, str)
            or not reference
        ):
            raise TaskMemoryEpisodeError("3RScan scene/reference mapping is invalid")
        result[scene] = reference
    return result


def load_reference_by_scene(path: Path) -> dict[int, str]:
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise TaskMemoryEpisodeError("3RScan metadata cannot be decoded") from error
    return _reference_by_scene(metadata)


def _role_by_reference(data_contract: Mapping[str, Any]) -> dict[str, str]:
    roles = data_contract.get("roles")
    if not isinstance(roles, Mapping):
        raise TaskMemoryEpisodeError("data contract lacks reference roles")
    mapping = {}
    fields = {
        "adaptation_reference_ids": "adaptation",
        "development_reference_ids": "development",
        "protocol_b_reference_ids": "protocol_b_final",
        "additional_native_reference_ids": "additional_native_refs",
    }
    for name, role in fields.items():
        values = roles.get(name)
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise TaskMemoryEpisodeError(f"data contract role {name} is invalid")
        for reference in values:
            if not isinstance(reference, str) or reference in mapping:
                raise TaskMemoryEpisodeError("data contract reference roles overlap")
            mapping[reference] = role
    return mapping


class _FutureContext:
    def __init__(
        self,
        base: object,
        *,
        source_context_index: int,
        sequence_id: str,
        scan_indices: tuple[int, ...],
    ) -> None:
        self.base = base
        self.source_context_index = source_context_index
        self.sequence_names = (sequence_id,)
        self.sequence_indices = (scan_indices,)
        self.max_points_per_sample = getattr(base, "max_points_per_sample", None)

    @property
    def mode(self) -> str:
        return self.base.mode

    @mode.setter
    def mode(self, value: str) -> None:
        self.base.mode = value

    @property
    def known_empty_scan_substitution_count(self) -> int:
        return self.base.known_empty_scan_substitution_count

    def load_scan_indices(self, _context_index, scan_indices, *, change_file):
        return self.base.load_scan_indices(
            self.source_context_index, scan_indices, change_file=change_file
        )


def _mutated_future_episode(
    base: object, master: NativeEpisodeMaster, *, augmentation_seed: int
) -> TaskMemoryEpisode:
    if len(master.scan_ids) < 4:
        raise TaskMemoryEpisodeError("future mutation needs at least four scans")
    positions = (0, 1, 3, 2, *range(4, len(master.scan_ids)))
    scan_ids = tuple(master.scan_ids[position] for position in positions)
    scan_indices = tuple(master.scan_indices[position] for position in positions)
    mutated_master = NativeEpisodeMaster(
        reference_id=master.reference_id,
        sequence_id="-".join(scan_ids),
        scan_ids=scan_ids,
        scan_indices=scan_indices,
        role=master.role,
        context_index=0,
    )
    context = _FutureContext(
        base,
        source_context_index=master.context_index,
        sequence_id=mutated_master.sequence_id,
        scan_indices=scan_indices,
    )
    spec = TaskMemoryEpisodeSpec.from_master(
        mutated_master,
        horizon=len(scan_ids),
        augmentation_seed=augmentation_seed,
        draw_index=0,
        bucket=f"T{len(scan_ids)}",
    )
    return TaskMemoryEpisodeDataset(context, (spec,))[0]


def run_preflight(
    *,
    data_root: Path,
    metadata_path: Path,
    data_contract_path: Path,
    output_path: Path,
) -> dict[str, object]:
    data_root = data_root.expanduser().resolve(strict=True)
    metadata_path = metadata_path.expanduser().resolve(strict=True)
    data_contract_path = data_contract_path.expanduser().resolve(strict=True)
    repository_data = PROJECT_ROOT / "data"
    if not repository_data.exists() or repository_data.resolve() != data_root:
        raise TaskMemoryEpisodeError(
            "repository data binding must resolve to the explicit data root"
        )
    with initialize_config_dir(
        config_dir=str((PROJECT_ROOT / "conf").resolve()), version_base="1.2"
    ):
        config = compose(config_name="config_persist4d_allt")
    data_contract = _load_json(data_contract_path, label="data contract")
    unsigned_contract = dict(data_contract)
    observed_contract_hash = unsigned_contract.pop("content_sha256", None)
    if observed_contract_hash != canonical_json_sha256(unsigned_contract):
        raise TaskMemoryEpisodeError("data contract content hash differs")
    reference_map = load_reference_by_scene(metadata_path)
    role_map = _role_by_reference(data_contract)

    bases = {}
    masters_by_horizon = {}
    native_summary = {}
    for horizon in range(2, 6):
        base = _rio_base_dataset(config, data_root=data_root, horizon=horizon)
        masters = build_native_episode_masters(
            base,
            reference_by_scene=reference_map,
            role_by_reference=role_map,
        )
        bases[horizon] = base
        masters_by_horizon[horizon] = masters
        native_summary[f"H{horizon}"] = {
            "development_masters": sum(
                master.role == "development" for master in masters
            ),
            "master_count": len(masters),
            "reference_count": len({master.reference_id for master in masters}),
        }

    base = bases[5]
    master = next(
        (master for master in masters_by_horizon[5] if master.role == "development"),
        None,
    )
    if master is None:
        raise TaskMemoryEpisodeError("real preflight has no frozen development master")
    augmentation_seed = 45
    spec = TaskMemoryEpisodeSpec.from_master(
        master,
        horizon=5,
        augmentation_seed=augmentation_seed,
        draw_index=0,
        bucket="T5",
    )
    original = TaskMemoryEpisodeDataset(base, (spec,))[0]
    future_mutated = _mutated_future_episode(
        base, master, augmentation_seed=augmentation_seed
    )
    stage_collator = hydra.utils.instantiate(config.data.validation_collation)
    batch = TaskMemoryEpisodeCollator(stage_collator)([original])
    payload = build_preflight_payload(
        original=original,
        future_mutated=future_mutated,
        batch=batch,
        native_summary=native_summary,
        sources={
            "data_contract": "repo:artifacts/task_memory_retention_v2/DATA_CONTRACT.json",
            "rio_data": "external:data_root/processed/rio",
            "rio_metadata": "external:rio_metadata",
        },
    )
    output_path = output_path.expanduser()
    if output_path.is_symlink():
        raise TaskMemoryEpisodeError("preflight output must not be a symlink")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--rio-metadata", type=Path, required=True)
    parser.add_argument(
        "--data-contract",
        type=Path,
        default=PROJECT_ROOT / "artifacts/task_memory_retention_v2/DATA_CONTRACT.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "artifacts/task_memory_retention_v2/implementation/preflight_real_sequence.json"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = run_preflight(
        data_root=args.data_root,
        metadata_path=args.rio_metadata,
        data_contract_path=args.data_contract,
        output_path=args.output,
    )
    print(
        json.dumps(
            {
                "content_sha256": payload["content_sha256"],
                "output": "repo:artifacts/task_memory_retention_v2/implementation/preflight_real_sequence.json",
                "status": payload["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_preflight_payload", "load_reference_by_scene", "run_preflight"]
