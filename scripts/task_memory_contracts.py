#!/usr/bin/env python3
"""Fail-closed M0 contracts for TaskMemory Retention V2."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import shutil
import subprocess
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_JSON_NAMES = (
    "START_STATE.json",
    "DATA_CONTRACT.json",
    "BASELINE_CONTRACT.json",
    "BUDGET_CONTRACT.json",
    "EVALUATION_CONTRACT.json",
    "PROFILE_CONTRACT.json",
)
PUBLIC_TEXT_NAMES = (
    "EVIDENCE_MAP.md",
    "OUTPUT_CONTRACT.md",
    "COMMANDS.md",
)
LOCAL_RESOLVER_NAME = "external_assets.local.json"


class TaskMemoryContractError(RuntimeError):
    """Raised when an M0 fact cannot be proven from the bound inputs."""


@dataclass(frozen=True)
class AssetBindings:
    data_root: Path
    rio_metadata: Path
    r1_checkpoint: Path
    concerto_pretrained: Path
    run_root: Path


@dataclass(frozen=True)
class ContractBundle:
    output_root: Path
    public_documents: tuple[str, ...]
    reference_count: int
    scan_count: int


def canonical_json_sha256(value: object) -> str:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise TaskMemoryContractError(
            "contract values must be finite portable JSON"
        ) from error
    return hashlib.sha256(encoded).hexdigest()


def _self_hashed(value: Mapping[str, object]) -> dict[str, object]:
    result = dict(value)
    if "content_sha256" in result:
        raise TaskMemoryContractError("unsigned contract already has content_sha256")
    result["content_sha256"] = canonical_json_sha256(result)
    return result


def _load_yaml(path: Path, *, label: str) -> object:
    if path.is_symlink() or not path.is_file():
        raise TaskMemoryContractError(f"{label} must be a regular non-symlink file")
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise TaskMemoryContractError(f"{label} cannot be decoded") from error


def _load_json(path: Path, *, label: str) -> object:
    if path.is_symlink() or not path.is_file():
        raise TaskMemoryContractError(f"{label} must be a regular non-symlink file")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise TaskMemoryContractError(f"{label} cannot be decoded") from error


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(
    path: Path,
    *,
    logical_reference: str,
    expected_bytes: int,
    expected_sha256: str,
    label: str,
) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise TaskMemoryContractError(f"{label} must be a regular non-symlink file")
    observed_bytes = path.stat().st_size
    if observed_bytes != expected_bytes:
        raise TaskMemoryContractError(f"{label} byte size differs")
    observed_sha256 = _sha256_file(path)
    if observed_sha256 != expected_sha256:
        raise TaskMemoryContractError(f"{label} SHA256 differs")
    return {
        "bytes": observed_bytes,
        "logical_reference": logical_reference,
        "sha256": observed_sha256,
        "status": "VERIFIED",
    }


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TaskMemoryContractError(f"{label} must be a mapping")
    return value


def _string_list(value: object, *, label: str) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TaskMemoryContractError(f"{label} must be a sequence")
    result = list(value)
    if any(not isinstance(item, str) or not item for item in result):
        raise TaskMemoryContractError(f"{label} must contain non-empty strings")
    if len(result) != len(set(result)):
        raise TaskMemoryContractError(f"{label} contains duplicates")
    return result


def _portable_suffix(reference: str, prefix: str) -> PurePosixPath:
    suffix = reference[len(prefix) :]
    relative = PurePosixPath(suffix)
    if not suffix or relative.is_absolute() or ".." in relative.parts:
        raise TaskMemoryContractError(f"invalid logical reference {reference!r}")
    return relative


def _resolve_reference(
    reference: object,
    *,
    config_path: Path,
    assets: AssetBindings,
) -> Path:
    if not isinstance(reference, str) or not reference:
        raise TaskMemoryContractError("logical reference must be a non-empty string")
    exact = {
        "external:data_root": assets.data_root,
        "external:rio_metadata": assets.rio_metadata,
        "external:r1_checkpoint": assets.r1_checkpoint,
        "external:concerto_pretrained": assets.concerto_pretrained,
        "external:run_root": assets.run_root,
    }
    if reference in exact:
        return Path(exact[reference]).expanduser().resolve()
    if reference.startswith("external:data_root/"):
        suffix = _portable_suffix(reference, "external:data_root/")
        return (assets.data_root / Path(*suffix.parts)).expanduser().resolve()
    if reference.startswith("repo:"):
        suffix = _portable_suffix(reference, "repo:")
        return (PROJECT_ROOT / Path(*suffix.parts)).resolve()
    if reference.startswith("config:"):
        suffix = _portable_suffix(reference, "config:")
        return (config_path.parent / Path(*suffix.parts)).resolve()
    raise TaskMemoryContractError(f"cannot resolve logical reference {reference!r}")


def _resolve_source(
    descriptor: object,
    *,
    label: str,
    config_path: Path,
    assets: AssetBindings,
) -> tuple[str, Path, str]:
    source = _mapping(descriptor, label=f"source {label}")
    reference = source.get("reference")
    expected_sha256 = source.get("sha256")
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise TaskMemoryContractError(f"source {label} has invalid SHA256")
    path = _resolve_reference(reference, config_path=config_path, assets=assets)
    if path.is_symlink() or not path.is_file():
        raise TaskMemoryContractError(
            f"cannot resolve source {label} from logical reference {reference!r}"
        )
    observed_sha256 = _sha256_file(path)
    if observed_sha256 != expected_sha256:
        raise TaskMemoryContractError(f"source {label} SHA256 differs")
    return str(reference), path, observed_sha256


def _git_output(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise TaskMemoryContractError(
            f"git {' '.join(args)} failed: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def _repository_contract(config: Mapping[str, Any]) -> dict[str, object]:
    repository = _mapping(config.get("repository"), label="repository")
    branch = repository.get("branch")
    parent = repository.get("reviewed_parent")
    if not isinstance(branch, str) or not branch:
        raise TaskMemoryContractError("repository branch is invalid")
    if (
        not isinstance(parent, str)
        or len(parent) != 40
        or any(character not in "0123456789abcdef" for character in parent)
    ):
        raise TaskMemoryContractError("reviewed parent is invalid")
    observed_branch = _git_output("branch", "--show-current")
    if observed_branch != branch:
        raise TaskMemoryContractError(
            f"current branch {observed_branch!r} differs from frozen branch {branch!r}"
        )
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", parent, "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if ancestor.returncode != 0:
        raise TaskMemoryContractError("reviewed parent is not an ancestor of HEAD")
    return {
        "branch": branch,
        "reviewed_parent": parent,
        "reviewed_parent_is_ancestor": True,
        "start_head": _git_output("rev-parse", "HEAD"),
    }


def _metadata_records(value: object) -> dict[int, Mapping[str, Any]]:
    if isinstance(value, Mapping):
        items = list(value.items())
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        items = list(enumerate(value))
    else:
        raise TaskMemoryContractError("3RScan metadata must be a sequence or mapping")
    records: dict[int, Mapping[str, Any]] = {}
    for position, (key, raw) in enumerate(items):
        record = _mapping(raw, label=f"3RScan metadata record {position}")
        scene = record.get("scene", key)
        reference = record.get("reference", record.get("reference_scene_id"))
        if isinstance(scene, bool) or not isinstance(scene, int) or scene < 0:
            raise TaskMemoryContractError("3RScan metadata scene is invalid")
        if not isinstance(reference, str) or not reference:
            raise TaskMemoryContractError("3RScan metadata reference UUID is missing")
        if scene in records:
            raise TaskMemoryContractError("3RScan metadata has duplicate scene indices")
        records[scene] = record
    return records


def _raw_scan_uuid(record: Mapping[str, Any], sub_scene: int) -> str | None:
    if sub_scene == 0:
        value = record.get("reference", record.get("reference_scene_id"))
        return value if isinstance(value, str) and value else None
    scans = record.get("scans")
    if isinstance(scans, Sequence) and not isinstance(scans, (str, bytes)):
        position = sub_scene - 1
        if 0 <= position < len(scans) and isinstance(scans[position], Mapping):
            value = scans[position].get("reference", scans[position].get("scan_id"))
            return value if isinstance(value, str) and value else None
    return None


def _dataset_relative_path(value: object) -> Path | None:
    if not isinstance(value, str) or value in {"", "None"}:
        return None
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise TaskMemoryContractError("processed dataset path must be portable")
    parts = path.parts[1:] if path.parts and path.parts[0] == "data" else path.parts
    if not parts:
        raise TaskMemoryContractError("processed dataset path is empty")
    return Path(*parts)


def _logical_dataset_path(relative: Path | None) -> str:
    if relative is None:
        return "NOT_AVAILABLE"
    return f"external:data_root/{relative.as_posix()}"


def _scan_rows(
    *,
    data_root: Path,
    metadata: Mapping[int, Mapping[str, Any]],
    train_database: object,
    validation_database: object,
    fixed_development: set[str],
    protocol_references: set[str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen_scan_ids: set[str] = set()
    for split, raw_records in (
        ("train", train_database),
        ("validation", validation_database),
    ):
        if isinstance(raw_records, (str, bytes)) or not isinstance(
            raw_records, Sequence
        ):
            raise TaskMemoryContractError(f"{split} database must be a sequence")
        for position, raw in enumerate(raw_records):
            record = _mapping(raw, label=f"{split} scan record {position}")
            scene = record.get("scene")
            sub_scene = record.get("sub_scene")
            if (
                isinstance(scene, bool)
                or not isinstance(scene, int)
                or isinstance(sub_scene, bool)
                or not isinstance(sub_scene, int)
                or scene < 0
                or sub_scene < 0
            ):
                raise TaskMemoryContractError(f"{split} scan identity is invalid")
            if scene not in metadata:
                raise TaskMemoryContractError(
                    f"{split} scene {scene} is missing from 3RScan metadata"
                )
            metadata_record = metadata[scene]
            reference = metadata_record.get(
                "reference", metadata_record.get("reference_scene_id")
            )
            if not isinstance(reference, str) or not reference:
                raise TaskMemoryContractError("reference UUID is invalid")
            scan_id = f"scene{scene:04d}_{sub_scene:02d}"
            if scan_id in seen_scan_ids:
                raise TaskMemoryContractError(f"duplicate processed scan {scan_id}")
            seen_scan_ids.add(scan_id)
            processed_relative = _dataset_relative_path(record.get("filepath"))
            instance_relative = _dataset_relative_path(
                record.get("instance_gt_filepath")
            )
            processed_readable = bool(
                processed_relative is not None
                and (data_root / processed_relative).is_file()
            )
            instance_readable = bool(
                instance_relative is not None
                and (data_root / instance_relative).is_file()
            )
            if reference in fixed_development:
                role = "development"
            elif reference in protocol_references:
                role = "protocol_b_final"
            elif split == "train":
                role = "adaptation"
            else:
                role = "additional_native_refs"
            rows.append(
                {
                    "canonical_instance_mapping_available": instance_readable,
                    "instance_gt_readable": instance_readable,
                    "instance_gt_reference": _logical_dataset_path(instance_relative),
                    "processed_readable": processed_readable,
                    "processed_reference": _logical_dataset_path(processed_relative),
                    "raw_scan_uuid": _raw_scan_uuid(metadata_record, sub_scene)
                    or "NOT_AVAILABLE",
                    "reference_id": reference,
                    "role": role,
                    "scan_id": scan_id,
                    "source_split": split,
                }
            )
    return sorted(rows, key=lambda row: (str(row["reference_id"]), str(row["scan_id"])))


def _native_inventory(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["reference_id"])].append(row)
    result = []
    for reference in sorted(grouped):
        reference_rows = grouped[reference]
        roles = {str(row["role"]) for row in reference_rows}
        splits = {str(row["source_split"]) for row in reference_rows}
        if len(roles) != 1 or len(splits) != 1:
            raise TaskMemoryContractError(
                f"reference {reference!r} crosses role or split boundaries"
            )
        readable = sum(
            row["processed_readable"] is True and row["instance_gt_readable"] is True
            for row in reference_rows
        )
        result.append(
            {
                "max_native_horizon": min(5, readable),
                "metadata_scan_count": len(reference_rows),
                "readable_supervised_scan_count": readable,
                "reference_id": reference,
                "role": next(iter(roles)),
                "source_split": next(iter(splits)),
            }
        )
    return result


def _protocol_references(
    protocol: Mapping[str, Any], data_config: Mapping[str, Any]
) -> tuple[set[str], int]:
    masters = protocol.get("masters")
    if not isinstance(masters, list):
        raise TaskMemoryContractError("Protocol-B masters are unavailable")
    references = {
        master.get("reference_scene_id")
        for master in masters
        if isinstance(master, Mapping)
        and isinstance(master.get("reference_scene_id"), str)
    }
    expected_masters = data_config.get("expected_protocol_masters")
    expected_references = data_config.get("expected_protocol_references")
    expected_units = data_config.get("expected_protocol_order_units")
    protocol_config = _mapping(protocol.get("protocol"), label="Protocol-B protocol")
    order_variants = _string_list(
        protocol_config.get("order_variants"), label="Protocol-B order variants"
    )
    if len(masters) != expected_masters:
        raise TaskMemoryContractError("Protocol-B master count differs")
    if len(references) != expected_references:
        raise TaskMemoryContractError("Protocol-B reference count differs")
    order_units = len(masters) * len(order_variants)
    if order_units != expected_units:
        raise TaskMemoryContractError("Protocol-B order-unit count differs")
    return {str(reference) for reference in references}, order_units


def _fixed_roles(
    split: Mapping[str, Any], protocol_references: set[str]
) -> tuple[set[str], set[str]]:
    reference_split = _mapping(
        split.get("reference_split"), label="frozen reference split"
    )
    adaptation = set(
        _string_list(
            reference_split.get("adaptation_reference_ids"),
            label="adaptation references",
        )
    )
    development = set(
        _string_list(
            reference_split.get("development_reference_ids"),
            label="development references",
        )
    )
    frozen_protocol = set(
        _string_list(
            split.get("protocol_b_reference_ids"), label="frozen Protocol-B references"
        )
    )
    if frozen_protocol != protocol_references:
        raise TaskMemoryContractError("frozen and live Protocol-B references differ")
    if adaptation & development:
        raise TaskMemoryContractError("adaptation/development role overlap")
    if (adaptation | development) & protocol_references:
        raise TaskMemoryContractError("training/development and Protocol-B role overlap")
    return adaptation, development


def _training_contract(config: Mapping[str, Any]) -> tuple[dict[str, object], list[dict[str, object]]]:
    training = _mapping(config.get("training"), label="training")
    required_ints = (
        "seed",
        "devices",
        "physical_episode_batch_per_gpu",
        "gradient_accumulation",
        "optimizer_updates",
        "pilot_updates",
    )
    normalized: dict[str, object] = {}
    for name in required_ints:
        value = training.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise TaskMemoryContractError(f"training {name} must be a positive integer")
        normalized[name] = value
    evaluations = training.get("evaluation_updates")
    if evaluations != [0, 750, 1500, 2250, 3000]:
        raise TaskMemoryContractError("M2 evaluation updates differ")
    if normalized["optimizer_updates"] != 3000 or normalized["pilot_updates"] != 300:
        raise TaskMemoryContractError("M2 update schedule differs")
    precision = training.get("precision")
    if precision != "32-true":
        raise TaskMemoryContractError("M2 precision must be 32-true")
    effective_batch = (
        int(normalized["devices"])
        * int(normalized["physical_episode_batch_per_gpu"])
        * int(normalized["gradient_accumulation"])
    )
    if effective_batch != 8:
        raise TaskMemoryContractError("effective episode batch must equal 8")
    normalized["effective_episode_batch"] = effective_batch
    normalized["evaluation_updates"] = list(evaluations)
    normalized["precision"] = precision
    ordered_names = ("single_scan", "T2", "T3", "T4", "T5")
    raw_buckets = _mapping(training.get("episode_buckets"), label="episode buckets")
    if set(raw_buckets) != set(ordered_names):
        raise TaskMemoryContractError("episode bucket names differ")
    buckets = []
    for name in ordered_names:
        fraction = raw_buckets[name]
        if not isinstance(fraction, (int, float)) or isinstance(fraction, bool):
            raise TaskMemoryContractError("episode bucket fractions must be numeric")
        buckets.append({"fraction": float(fraction), "name": name})
    if any(item["fraction"] != 0.2 for item in buckets):
        raise TaskMemoryContractError("episode buckets must each be exactly 20 percent")
    return dict(sorted(normalized.items())), buckets


def _budget_contract(
    config: Mapping[str, Any], training: Mapping[str, object], buckets: list[dict[str, object]]
) -> dict[str, object]:
    raw_caps = _mapping(config.get("budgets"), label="budgets")
    expected = {
        "training_gpu_hours": 120,
        "evaluation_gpu_hours": 30,
        "evaluation_cache_bytes": 40 * 1024**3,
        "permanent_state_bytes": 2 * 1024**2,
    }
    caps = {name: raw_caps.get(name) for name in expected}
    if caps != expected:
        raise TaskMemoryContractError("initial campaign caps differ")
    return _self_hashed(
        {
            "amendment_count": 0,
            "amendment_policy": "once_before_formal_run_from_measured_throughput_only",
            "caps": caps,
            "episode_buckets": buckets,
            "pilot_is_exact_schedule_prefix": True,
            "schema_version": "task-memory-budget-v2",
            "status": "FROZEN_INITIAL",
            "training": dict(training),
        }
    )


def _gpu_inventory() -> list[dict[str, object]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        return []
    inventory = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",", maxsplit=4)]
        if len(fields) != 5:
            continue
        try:
            inventory.append(
                {
                    "index": int(fields[0]),
                    "memory_free_mib": int(fields[3]),
                    "memory_total_mib": int(fields[2]),
                    "name": fields[1],
                    "utilization_percent": int(fields[4]),
                }
            )
        except ValueError:
            continue
    return inventory


def _output_contract() -> str:
    return """# Output Contract

`commit0` is the compulsory control: stage t commits only the current scan and never revises older masks.

`lag1` is the primary policy: the single W=2 forward at stage t replaces exactly the previous scan's provisional output, freezes all older scans, and commits the current scan provisionally. The last scan remains provisional; no nonexistent future scan is used to flush it.

Route identity has priority over class-compatible same-scan IoU matching. The fallback threshold is 0.5 and exact score ties use source query index. One `(logical_track_id, generation, class_id)` candidate is selected per stage and class without GT-guided old/new selection.

The model may read only working state. The lag-one buffer is bounded to the previous scan; older outputs enter an append-only archive whose materialization and bytes are reported separately. Prediction route is read-only and state commit happens exactly once per stage.

FH-native and FH-lag1 are distinct comparator rows. No checkpoint, output policy, or reducer may be selected separately by horizon.
"""


def _commands() -> str:
    return """# Commands

```bash
export PERSIST4D_DATA_ROOT=/actual/data/root
export PERSIST4D_RIO_METADATA=/actual/3RScan.json
export PERSIST4D_R1_CHECKPOINT=/actual/R1.ckpt
export PERSIST4D_CONCERTO_PRETRAINED=/actual/concerto_base.pth
export PERSIST4D_RUN_ROOT=/actual/new-run-root

python -m scripts.prepare_task_memory_v2 \\
  --config conf/task_memory_v2/experiment.yaml \\
  --data-root "$PERSIST4D_DATA_ROOT" \\
  --rio-metadata "$PERSIST4D_RIO_METADATA" \\
  --r1-checkpoint "$PERSIST4D_R1_CHECKPOINT" \\
  --concerto-pretrained "$PERSIST4D_CONCERTO_PRETRAINED" \\
  --run-root "$PERSIST4D_RUN_ROOT" \\
  --output artifacts/task_memory_retention_v2
```
"""


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    encoded = (
        json.dumps(value, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True)
        + "\n"
    )
    if path.is_symlink():
        raise TaskMemoryContractError(f"refusing to replace symlink {path.name}")
    path.write_text(encoded, encoding="utf-8")


def _write_inventory(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fieldnames = (
        "reference_id",
        "scan_id",
        "raw_scan_uuid",
        "source_split",
        "role",
        "processed_reference",
        "processed_readable",
        "instance_gt_reference",
        "instance_gt_readable",
        "canonical_instance_mapping_available",
    )
    if path.is_symlink():
        raise TaskMemoryContractError("refusing to replace inventory symlink")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def freeze_task_memory_contracts(
    *, config_path: Path, assets: AssetBindings, output_root: Path
) -> ContractBundle:
    config_path = Path(config_path).expanduser().resolve()
    config = _mapping(_load_yaml(config_path, label="experiment config"), label="config")
    if config.get("schema_version") != "task-memory-retention-v2-config-v1":
        raise TaskMemoryContractError("experiment config schema differs")
    if config.get("experiment") != "task_memory_retention_v2":
        raise TaskMemoryContractError("experiment name differs")

    normalized_assets = AssetBindings(
        data_root=Path(assets.data_root).expanduser().resolve(),
        rio_metadata=Path(assets.rio_metadata).expanduser().resolve(),
        r1_checkpoint=Path(assets.r1_checkpoint).expanduser().resolve(),
        concerto_pretrained=Path(assets.concerto_pretrained).expanduser().resolve(),
        run_root=Path(assets.run_root).expanduser().resolve(),
    )
    if not normalized_assets.data_root.is_dir():
        raise TaskMemoryContractError("data root is unavailable")
    if not normalized_assets.run_root.is_dir():
        raise TaskMemoryContractError("run root is unavailable")

    repository = _repository_contract(config)
    sources_config = _mapping(config.get("sources"), label="sources")
    required_sources = {
        "evidence_map",
        "task_prompt",
        "old_handoff",
        "frozen_split",
        "protocol_b",
    }
    if set(sources_config) != required_sources:
        raise TaskMemoryContractError("source set differs")
    resolved_sources = {
        name: _resolve_source(
            sources_config[name],
            label=name,
            config_path=config_path,
            assets=normalized_assets,
        )
        for name in sorted(required_sources)
    }

    assets_config = _mapping(config.get("assets"), label="assets")
    if set(assets_config) != {"r1_checkpoint", "concerto_pretrained"}:
        raise TaskMemoryContractError("weight asset set differs")
    weight_identities = {}
    for name, path, logical in (
        ("r1_checkpoint", normalized_assets.r1_checkpoint, "external:r1_checkpoint"),
        (
            "concerto_pretrained",
            normalized_assets.concerto_pretrained,
            "external:concerto_pretrained",
        ),
    ):
        descriptor = _mapping(assets_config[name], label=name)
        expected_bytes = descriptor.get("bytes")
        expected_sha256 = descriptor.get("sha256")
        if isinstance(expected_bytes, bool) or not isinstance(expected_bytes, int):
            raise TaskMemoryContractError(f"{name} expected bytes are invalid")
        if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
            raise TaskMemoryContractError(f"{name} expected SHA256 is invalid")
        weight_identities[name] = _identity(
            path,
            logical_reference=logical,
            expected_bytes=expected_bytes,
            expected_sha256=expected_sha256,
            label=name,
        )

    data_config = _mapping(config.get("data"), label="data")
    metadata_path = _resolve_reference(
        data_config.get("metadata_reference"),
        config_path=config_path,
        assets=normalized_assets,
    )
    train_database_path = _resolve_reference(
        data_config.get("train_database_reference"),
        config_path=config_path,
        assets=normalized_assets,
    )
    validation_database_path = _resolve_reference(
        data_config.get("validation_database_reference"),
        config_path=config_path,
        assets=normalized_assets,
    )
    metadata_value = _load_json(metadata_path, label="3RScan metadata")
    metadata = _metadata_records(metadata_value)
    train_database = _load_yaml(train_database_path, label="RIO train database")
    validation_database = _load_yaml(
        validation_database_path, label="RIO validation database"
    )
    split = _mapping(
        _load_json(resolved_sources["frozen_split"][1], label="frozen split"),
        label="frozen split",
    )
    protocol = _mapping(
        _load_json(resolved_sources["protocol_b"][1], label="Protocol-B"),
        label="Protocol-B",
    )
    protocol_references, protocol_units = _protocol_references(protocol, data_config)
    frozen_adaptation, frozen_development = _fixed_roles(split, protocol_references)
    rows = _scan_rows(
        data_root=normalized_assets.data_root,
        metadata=metadata,
        train_database=train_database,
        validation_database=validation_database,
        fixed_development=frozen_development,
        protocol_references=protocol_references,
    )
    inventory = _native_inventory(rows)
    observed_references = {str(item["reference_id"]) for item in inventory}
    missing_frozen = (frozen_adaptation | frozen_development | protocol_references) - observed_references
    if missing_frozen:
        raise TaskMemoryContractError(
            f"frozen references are absent from processed databases: {sorted(missing_frozen)!r}"
        )
    roles = {
        "adaptation_reference_ids": sorted(
            item["reference_id"] for item in inventory if item["role"] == "adaptation"
        ),
        "development_reference_ids": sorted(frozen_development),
        "protocol_b_reference_ids": sorted(protocol_references),
        "additional_native_reference_ids": sorted(
            item["reference_id"]
            for item in inventory
            if item["role"] == "additional_native_refs"
            and int(item["max_native_horizon"]) >= 2
        ),
    }
    role_sets = [set(value) for value in roles.values()]
    if any(
        role_sets[left] & role_sets[right]
        for left in range(len(role_sets))
        for right in range(left + 1, len(role_sets))
    ):
        raise TaskMemoryContractError("reference role overlap")

    training, buckets = _training_contract(config)
    output_config = _mapping(config.get("output"), label="output")
    if output_config != {"primary_policy": "lag1", "control_policy": "commit0"}:
        raise TaskMemoryContractError("output policy contract differs")
    source_descriptors = {
        name: {
            "logical_reference": resolved_sources[name][0],
            "sha256": resolved_sources[name][2],
        }
        for name in sorted(resolved_sources)
    }

    availability = {}
    adaptation_inventory = [
        item for item in inventory if item["role"] == "adaptation"
    ]
    for horizon in range(1, 6):
        availability["single_scan" if horizon == 1 else f"T{horizon}"] = sum(
            int(item["max_native_horizon"]) >= horizon
            for item in adaptation_inventory
        )
    data_contract = _self_hashed(
        {
            "episode_bucket_availability_reference_counts": availability,
            "independent_generalization": (
                "AVAILABLE_SEPARATE_POPULATION"
                if roles["additional_native_reference_ids"]
                else "NOT_ESTABLISHED"
            ),
            "native_reference_inventory": inventory,
            "populations": {
                "additional_native_refs": {
                    "reference_count": len(roles["additional_native_reference_ids"]),
                    "retention_curve_must_remain_separate": True,
                },
                "protocol_b_common_129": {
                    "master_count": len(protocol["masters"]),
                    "order_unit_count": protocol_units,
                    "reference_count": len(protocol_references),
                },
            },
            "role_policy": {
                "development": "frozen_old_8_base_exposed_adaptation_holdout",
                "new_train_references": "adaptation",
                "protocol_b": "final_only",
                "validation_non_protocol": "additional_native_refs_final_only",
            },
            "roles": roles,
            "schema_version": "task-memory-data-v2",
            "sources": source_descriptors,
        }
    )
    budget_contract = _budget_contract(config, training, buckets)
    baseline_contract = _self_hashed(
        {
            "comparators": [
                {"name": "B2", "policies": ["commit0", "lag1"]},
                {"name": "B4", "policies": ["commit0", "lag1"]},
                {"name": "D-LAST", "policies": ["commit0", "lag1"]},
                {"name": "D-EMA", "policies": ["commit0", "lag1"]},
                {"name": "FH-R1-native", "policies": ["FH-native"]},
                {"name": "FH-R1-lag1", "policies": ["lag1"]},
            ],
            "frozen_observed_t_map_percent": {
                "B4": {"T2": 21.964, "T3": 13.550, "T4": 8.016, "T5": 5.854},
                "FH-R1": {"T2": 22.878, "T3": 14.831, "T4": 10.275, "T5": 8.157},
                "old-C2": {"T2": 22.503, "T3": 14.106, "T4": 7.600, "T5": 5.132},
            },
            "primary_policy": "lag1",
            "primary_reducer": "mean",
            "schema_version": "task-memory-baseline-v2",
        }
    )
    evaluation_contract = _self_hashed(
        {
            "all_t_epsilon": 1e-6,
            "checkpoint_policy_reducer_fixed_across_horizons": True,
            "cluster_unit": "reference_id",
            "horizons": [2, 3, 4, 5],
            "identity_zero_denominator": "N/A",
            "metrics": ["t_mAP", "t_mAP50", "t_mAP25", "t_REC", "prefix_overall_mAP"],
            "primary_policy": "lag1",
            "primary_reducer": "mean",
            "protocol_b_order_units_are_correlated": True,
            "schema_version": "task-memory-evaluation-v2",
        }
    )
    profile_contract = _self_hashed(
        {
            "device_count": 1,
            "device_type": "A40",
            "reference_units": 6,
            "repeats": 10,
            "restore_cloned_state_each_repeat": True,
            "schema_version": "task-memory-profile-v2",
            "time_scopes": ["model_update", "end_to_end_update", "output_materialization", "cumulative_T1_to_T"],
            "warmups": 5,
        }
    )
    metadata_identity = {
        "bytes": metadata_path.stat().st_size,
        "logical_reference": str(data_config["metadata_reference"]),
        "sha256": _sha256_file(metadata_path),
        "status": "VERIFIED",
    }
    start_state = _self_hashed(
        {
            "assets": {**weight_identities, "rio_metadata": metadata_identity},
            "environment": {
                "gpu_inventory": _gpu_inventory(),
                "python": platform.python_version(),
                "storage": {
                    "data_root": {
                        "free_bytes": shutil.disk_usage(
                            normalized_assets.data_root
                        ).free,
                        "logical_reference": "external:data_root",
                    },
                    "run_root": {
                        "free_bytes": shutil.disk_usage(
                            normalized_assets.run_root
                        ).free,
                        "logical_reference": "external:run_root",
                    },
                },
            },
            "experiment": "task_memory_retention_v2",
            "legacy_inputs": {
                "allt_cache_root": "external:allt_task_superiority_v1/evaluation_cache",
                "allt_handoff": "repo:artifacts/allt_task_superiority_v1/HANDOFF.md",
                "allt_handoff_sha256": resolved_sources["old_handoff"][2],
                "policy": "read_only",
            },
            "repository": repository,
            "schema_version": "task-memory-start-v2",
            "source_documents": source_descriptors,
        }
    )

    output_root = Path(output_root).expanduser()
    if output_root.is_symlink():
        raise TaskMemoryContractError("output root must not be a symlink")
    output_root.mkdir(parents=True, exist_ok=True)
    documents = {
        "START_STATE.json": start_state,
        "DATA_CONTRACT.json": data_contract,
        "BASELINE_CONTRACT.json": baseline_contract,
        "BUDGET_CONTRACT.json": budget_contract,
        "EVALUATION_CONTRACT.json": evaluation_contract,
        "PROFILE_CONTRACT.json": profile_contract,
    }
    for name, document in documents.items():
        _write_json(output_root / name, document)
    evidence_bytes = resolved_sources["evidence_map"][1].read_bytes()
    (output_root / "EVIDENCE_MAP.md").write_bytes(evidence_bytes)
    (output_root / "OUTPUT_CONTRACT.md").write_text(
        _output_contract(), encoding="utf-8"
    )
    (output_root / "COMMANDS.md").write_text(_commands(), encoding="utf-8")
    _write_inventory(output_root / "references_inventory.csv", rows)
    local_resolver = {
        "external:concerto_pretrained": str(normalized_assets.concerto_pretrained),
        "external:data_root": str(normalized_assets.data_root),
        "external:r1_checkpoint": str(normalized_assets.r1_checkpoint),
        "external:rio_metadata": str(normalized_assets.rio_metadata),
        "external:run_root": str(normalized_assets.run_root),
    }
    _write_json(output_root / LOCAL_RESOLVER_NAME, local_resolver)

    return ContractBundle(
        output_root=output_root,
        public_documents=(
            *PUBLIC_JSON_NAMES,
            *PUBLIC_TEXT_NAMES,
            "references_inventory.csv",
        ),
        reference_count=len(inventory),
        scan_count=len(rows),
    )


__all__ = [
    "AssetBindings",
    "ContractBundle",
    "TaskMemoryContractError",
    "canonical_json_sha256",
    "freeze_task_memory_contracts",
]
