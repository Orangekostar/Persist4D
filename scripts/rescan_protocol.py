"""Frozen causal protocol for independent ReScan validation."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_BASELINES = (
    "Pairwise Feature Association",
    "Pairwise Feature-Class Association",
    "EMA Temporal Association",
    "Persist4D",
)


class RescanProtocolError(ValueError):
    """Raised when an external protocol differs from its frozen contract."""


def _mapping(value: object, *, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise RescanProtocolError(f"{name} must be a mapping")
    return dict(value)


def _sequence(value: object, *, name: str) -> list[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise RescanProtocolError(f"{name} must be a sequence")
    return list(value)


def _canonical_json_bytes(value: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _content_sha256(value: Mapping[str, object]) -> str:
    payload = copy.deepcopy(dict(value))
    payload.pop("content_sha256", None)
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _require_digest(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise RescanProtocolError(f"{name} must be a lowercase SHA256")
    return value


def build_rescan_protocol(
    dataset_manifest: Mapping[str, object],
    label_map: Mapping[str, object],
    *,
    checkpoint_sha256: str,
    config_sha256: str,
) -> dict[str, object]:
    dataset = _mapping(dataset_manifest, name="dataset manifest")
    labels = _mapping(label_map, name="label map")
    if dataset.get("status") != "pass":
        raise RescanProtocolError("dataset manifest must pass")
    chronology = _mapping(dataset.get("chronology"), name="chronology")
    if chronology.get("status") != "official_index_order":
        raise RescanProtocolError("official chronology is required")
    dataset_digest = _require_digest(
        dataset.get("dataset_content_sha256"), name="dataset content digest"
    )
    checkpoint_digest = _require_digest(checkpoint_sha256, name="checkpoint_sha256")
    configuration_digest = _require_digest(config_sha256, name="config_sha256")

    mappings = _sequence(labels.get("mappings"), name="label mappings")
    encountered = set(
        _sequence(
            _mapping(dataset.get("summary"), name="dataset summary").get(
                "encountered_class_ids"
            ),
            name="encountered class ids",
        )
    )
    exact = {
        int(entry["source_class_id"])
        for raw_entry in mappings
        for entry in [_mapping(raw_entry, name="label mapping")]
        if entry.get("status") == "exact"
    }
    eligible_classes = sorted(encountered.intersection(exact))
    excluded_classes = sorted(encountered - exact)

    protocol_scenes = []
    global_offset = 0
    transition_count = 0
    stable_identity_count = 0
    gap_count = 0
    for scene_index, raw_scene in enumerate(
        _sequence(dataset.get("scenes"), name="dataset scenes")
    ):
        scene = _mapping(raw_scene, name=f"dataset scene {scene_index}")
        scene_id = scene.get("scene_id")
        if not isinstance(scene_id, str) or not scene_id:
            raise RescanProtocolError("scene_id must be non-empty")
        capture_ids = [
            str(value)
            for value in _sequence(scene.get("capture_ids"), name="capture ids")
        ]
        if len(capture_ids) < 2 or len(set(capture_ids)) != len(capture_ids):
            raise RescanProtocolError("each scene requires unique ordered captures")
        stable_ids = [
            int(value)
            for value in _sequence(
                scene.get("stable_object_identity_ids"),
                name="stable object identity ids",
            )
        ]
        gaps = _sequence(scene.get("gap_opportunities"), name="gap opportunities")
        stages = []
        for stage_index, capture_id in enumerate(capture_ids):
            if stage_index == 0:
                local_ids = [capture_id]
                local_indices = [global_offset]
            else:
                local_ids = capture_ids[stage_index - 1 : stage_index + 1]
                local_indices = [
                    global_offset + stage_index - 1,
                    global_offset + stage_index,
                ]
            stages.append(
                {
                    "stage_index": stage_index,
                    "target_capture_id": capture_id,
                    "local_input_capture_ids": local_ids,
                    "global_capture_indices": local_indices,
                }
            )
        protocol_scenes.append(
            {
                "scene_id": scene_id,
                "scene_cluster_index": scene_index,
                "capture_ids": capture_ids,
                "global_capture_indices": list(
                    range(global_offset, global_offset + len(capture_ids))
                ),
                "level_b_identity_ids": stable_ids,
                "level_a_semantic_inconsistent_identity_ids": copy.deepcopy(
                    scene.get("semantic_inconsistent_identity_ids", [])
                ),
                "gap_opportunities": copy.deepcopy(gaps),
                "stages": stages,
            }
        )
        global_offset += len(capture_ids)
        transition_count += len(capture_ids) - 1
        stable_identity_count += len(stable_ids)
        gap_count += len(gaps)

    label_map_digest = hashlib.sha256(_canonical_json_bytes(labels)).hexdigest()
    protocol: dict[str, object] = {
        "schema_version": 1,
        "status": "pass",
        "protocol_id": "rescan-official-order-frozen-local-pair-v1",
        "order_policy": {
            "chronology": "official_index_order",
            "artificial_permutations": False,
        },
        "frozen_inference": {
            "checkpoint_sha256": checkpoint_digest,
            "config_sha256": configuration_digest,
            "local_perception_window": 2,
            "first_capture_initialization_window": 1,
            "persistent_memory": {
                "capacity": 100,
                "association_threshold": 0.5,
                "class_weight": 0.25,
                "update_rate": 0.2,
                "max_update_rate": 0.2,
            },
        },
        "ground_truth_boundary": {
            "inference_fields": [
                "xyz",
                "normals",
                "rgb",
                "geometric_segment_ids",
            ],
            "post_inference_only_fields": [
                "class_idx",
                "instance_idx",
                "ambiguity_alternatives",
            ],
            "object_ground_truth_used_for_association": False,
        },
        "level_a": {
            "status": "enabled" if eligible_classes else "disabled",
            "eligible_source_class_ids": eligible_classes,
            "excluded_source_class_ids": excluded_classes,
            "metrics": [
                "t_mAP",
                "t_mAP50",
                "t_mAP25",
                "t_REC",
                "normalized_id_switch_rate",
                "gap_recovery_accuracy",
                "gap_recovery_recall",
                "fragmentation_count",
                "merge_count",
            ],
        },
        "level_b": {
            "status": "enabled",
            "eligible_rule": (
                "official stable instance_idx in [0,255], excluding wall/floor/ceiling"
            ),
            "stable_identity_count": stable_identity_count,
            "metrics": [
                "observation_coverage",
                "normalized_id_switch_rate",
                "fragmentation_count",
                "merge_count",
                "gap_recovery_accuracy",
                "gap_recovery_recall",
            ],
        },
        "baselines": list(_BASELINES),
        "statistics": {
            "cluster_unit": "independent physical scene",
            "bootstrap_replicates": 10000,
            "seed": 45,
            "paired": True,
        },
        "external_gate_thresholds": {
            "minimum_gap_opportunities": 10,
            "minimum_gap_scene_clusters": 3,
            "minimum_observation_coverage": 0.1,
            "contradiction_requires_negative_upper_ci": True,
        },
        "population": {
            "independent_scene_cluster_count": len(protocol_scenes),
            "temporal_sequence_count": len(protocol_scenes),
            "capture_count": global_offset,
            "transition_count": transition_count,
            "gap_opportunity_count": gap_count,
        },
        "provenance": {
            "dataset_content_sha256": dataset_digest,
            "label_map_sha256": label_map_digest,
            "official_code_commit": ("f45283be31119e9bd955d40bc159b1774dfed092"),
        },
        "scenes": protocol_scenes,
    }
    protocol["content_sha256"] = _content_sha256(protocol)
    validate_rescan_protocol(protocol, dataset_manifest=dataset)
    return protocol


def validate_rescan_protocol(
    value: Mapping[str, object],
    *,
    dataset_manifest: Mapping[str, object],
) -> dict[str, object]:
    protocol = _mapping(value, name="ReScan protocol")
    dataset = _mapping(dataset_manifest, name="dataset manifest")
    if protocol.get("status") != "pass" or protocol.get("schema_version") != 1:
        raise RescanProtocolError("protocol status or schema differs")
    order = _mapping(protocol.get("order_policy"), name="order policy")
    if order != {
        "chronology": "official_index_order",
        "artificial_permutations": False,
    }:
        raise RescanProtocolError("protocol chronology or permutation policy differs")
    provenance = _mapping(protocol.get("provenance"), name="provenance")
    if provenance.get("dataset_content_sha256") != dataset.get(
        "dataset_content_sha256"
    ):
        raise RescanProtocolError("protocol dataset digest differs")
    expected_scenes = _sequence(dataset.get("scenes"), name="dataset scenes")
    observed_scenes = _sequence(protocol.get("scenes"), name="protocol scenes")
    if len(observed_scenes) != len(expected_scenes):
        raise RescanProtocolError("protocol scene coverage differs")
    global_offset = 0
    transition_count = 0
    stable_count = 0
    gap_count = 0
    for index, (raw_observed, raw_expected) in enumerate(
        zip(observed_scenes, expected_scenes, strict=True)
    ):
        observed = _mapping(raw_observed, name=f"protocol scene {index}")
        expected = _mapping(raw_expected, name=f"dataset scene {index}")
        capture_ids = [
            str(item)
            for item in _sequence(expected.get("capture_ids"), name="capture ids")
        ]
        if (
            observed.get("scene_id") != expected.get("scene_id")
            or observed.get("capture_ids") != capture_ids
        ):
            raise RescanProtocolError("protocol scene order differs")
        stages = _sequence(observed.get("stages"), name="protocol stages")
        if len(stages) != len(capture_ids):
            raise RescanProtocolError("protocol stage coverage differs")
        for stage_index, raw_stage in enumerate(stages):
            stage = _mapping(raw_stage, name="protocol stage")
            expected_local = (
                [capture_ids[0]]
                if stage_index == 0
                else capture_ids[stage_index - 1 : stage_index + 1]
            )
            expected_indices = (
                [global_offset]
                if stage_index == 0
                else [global_offset + stage_index - 1, global_offset + stage_index]
            )
            if (
                stage.get("stage_index") != stage_index
                or stage.get("target_capture_id") != capture_ids[stage_index]
                or stage.get("local_input_capture_ids") != expected_local
                or stage.get("global_capture_indices") != expected_indices
            ):
                raise RescanProtocolError("protocol local input is not causal pairwise")
        global_offset += len(capture_ids)
        transition_count += len(capture_ids) - 1
        stable_count += len(
            _sequence(
                expected.get("stable_object_identity_ids"),
                name="stable object identities",
            )
        )
        gap_count += len(
            _sequence(expected.get("gap_opportunities"), name="gap opportunities")
        )
    population = _mapping(protocol.get("population"), name="population")
    expected_population = {
        "independent_scene_cluster_count": len(expected_scenes),
        "temporal_sequence_count": len(expected_scenes),
        "capture_count": global_offset,
        "transition_count": transition_count,
        "gap_opportunity_count": gap_count,
    }
    if population != expected_population:
        raise RescanProtocolError("protocol population differs")
    level_b = _mapping(protocol.get("level_b"), name="level B")
    if level_b.get("stable_identity_count") != stable_count:
        raise RescanProtocolError("Level B stable identity coverage differs")
    if protocol.get("baselines") != list(_BASELINES):
        raise RescanProtocolError("external baselines differ")
    if protocol.get("content_sha256") != _content_sha256(protocol):
        raise RescanProtocolError("protocol content digest differs")
    return protocol


def write_rescan_protocol(path: str | Path, payload: Mapping[str, object]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    content = _canonical_json_bytes(payload)
    if output.exists() or output.is_symlink():
        if output.is_symlink() or not output.is_file():
            raise FileExistsError("protocol path is not a regular file")
        if output.read_bytes() == content:
            return
        raise FileExistsError("protocol already contains different content")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.link(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_manifest", type=Path)
    parser.add_argument("label_map", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--config-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    dataset = json.loads(arguments.dataset_manifest.read_text(encoding="utf-8"))
    labels = json.loads(arguments.label_map.read_text(encoding="utf-8"))
    protocol = build_rescan_protocol(
        dataset,
        labels,
        checkpoint_sha256=arguments.checkpoint_sha256,
        config_sha256=arguments.config_sha256,
    )
    write_rescan_protocol(arguments.output, protocol)
    print(json.dumps(protocol["population"], allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
