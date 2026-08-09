#!/usr/bin/env python3
"""Audit temporal horizons represented by the native 3RScan metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


AUDITED_SPLITS = ("train", "val")
HORIZON_THRESHOLDS = range(2, 7)
VALIDATION_ALIAS = "validation"
SOURCE_REFERENCE = "external:3RScan/3RScan.json"


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_metadata(path: str | Path) -> list[dict[str, Any]]:
    """Load a 3RScan metadata JSON array."""
    metadata_path = Path(path)
    try:
        with metadata_path.open(encoding="utf-8") as metadata_file:
            metadata = json.load(metadata_file)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in metadata file {metadata_path}: {error}") from error

    if not isinstance(metadata, list):
        raise ValueError("metadata must be a JSON array")
    return metadata


def _require_nonempty_string(
    record: dict[str, Any], field: str, context: str
) -> str:
    if field not in record:
        raise ValueError(f"{context} is missing required field '{field}'")
    value = record[field]
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} field '{field}' must be a non-empty string")
    return value


def _distribution(lengths: list[int]) -> dict[str, int]:
    counts = Counter(lengths)
    distribution = {
        f"T={length}": counts[length]
        for length in range(1, max(lengths, default=0) + 1)
    }
    distribution.update(
        {
            f"T>={threshold}": sum(length >= threshold for length in lengths)
            for threshold in HORIZON_THRESHOLDS
        }
    )
    return distribution


def build_audit(
    metadata: object,
    source_path: str | Path,
    source_sha256: str | None = None,
) -> dict[str, Any]:
    """Build deterministic source-wide and train/validation temporal statistics."""
    if not isinstance(metadata, list):
        raise ValueError("metadata must be a JSON array")

    source_lengths: list[int] = []
    split_lengths: dict[str, list[int]] = {split: [] for split in AUDITED_SPLITS}
    excluded_splits: Counter[str] = Counter()
    scenes: list[dict[str, Any]] = []

    for group_index, group in enumerate(metadata):
        context = f"metadata group at index {group_index}"
        if not isinstance(group, dict):
            raise ValueError(f"{context} must be an object")

        reference_id = _require_nonempty_string(group, "reference", context)
        raw_split = _require_nonempty_string(group, "type", context)
        if "scans" not in group:
            raise ValueError(f"{context} is missing required field 'scans'")
        scans = group["scans"]
        if not isinstance(scans, list):
            raise ValueError(f"{context} field 'scans' must be an array")

        scan_ids = [reference_id]
        for scan_index, scan in enumerate(scans):
            scan_context = f"{context} scan at index {scan_index}"
            if not isinstance(scan, dict):
                raise ValueError(f"{scan_context} must be an object")
            scan_ids.append(_require_nonempty_string(scan, "reference", scan_context))

        temporal_length = len(scan_ids)
        source_lengths.append(temporal_length)
        split = "val" if raw_split == VALIDATION_ALIAS else raw_split
        if split not in AUDITED_SPLITS:
            excluded_splits[split] += 1
            continue

        split_lengths[split].append(temporal_length)
        scenes.append(
            {
                "reference_id": reference_id,
                "split": split,
                "T": temporal_length,
                "scan_ids": scan_ids,
            }
        )

    audited_lengths = [
        temporal_length
        for split in AUDITED_SPLITS
        for temporal_length in split_lengths[split]
    ]
    change_reason = (
        "The metadata has no complete static object universe, no timestamps, and no "
        "reliable chronological added/removed trajectory representation; static, rigid, "
        "non-rigid, added, and removed trajectory counts cannot be inferred safely."
    )
    return {
        "schema_version": 1,
        "source": {"path": str(source_path), "sha256": source_sha256},
        "scope": {
            "audited_splits": list(AUDITED_SPLITS),
            "split_normalization": {VALIDATION_ALIAS: "val"},
            "global_distribution_scope": "all_source_metadata_groups",
            "scene_records_scope": "train_and_validation_only",
        },
        "scan_order_semantics": "metadata_order_only_no_timestamps",
        "source_scene_count": len(metadata),
        "audited_scene_count": len(scenes),
        "excluded_splits": dict(sorted(excluded_splits.items())),
        "global_distribution": _distribution(source_lengths),
        "audited_distribution": _distribution(audited_lengths),
        "split_distribution": {
            split: _distribution(split_lengths[split]) for split in AUDITED_SPLITS
        },
        "change_trajectory_statistics": {
            "status": "not_computed",
            "reason": change_reason,
        },
        "scenes": scenes,
    }


def _distribution_rows(distribution: dict[str, int]) -> list[str]:
    exact_keys = sorted(
        (key for key in distribution if key.startswith("T=")),
        key=lambda key: int(key.removeprefix("T=")),
    )
    threshold_keys = [f"T>={threshold}" for threshold in HORIZON_THRESHOLDS]
    return [f"| {key} | {distribution[key]} |" for key in exact_keys + threshold_keys]


def render_markdown(audit: dict[str, Any]) -> str:
    """Render an audit dictionary as deterministic Markdown."""
    source = audit["source"]
    lines = [
        "# 3RScan Temporal Distribution Audit",
        "",
        f"- Source: `{source['path']}`",
        f"- SHA-256: `{source['sha256']}`",
        f"- Source scene groups: {audit['source_scene_count']}",
        f"- Audited train/val scene groups: {audit['audited_scene_count']}",
        "- Scan IDs preserve metadata order only; the source JSON provides no timestamps.",
        "",
        "## Excluded Splits",
        "",
        "| Split | Scene groups |",
        "| --- | ---: |",
    ]
    lines.extend(
        f"| {split} | {count} |"
        for split, count in audit["excluded_splits"].items()
    )

    distribution_sections = [
        (
            "Global Source Distribution",
            audit["global_distribution"],
            "Includes excluded splits only to verify the complete source metadata.",
        ),
        (
            "Audited Train/Val Distribution",
            audit["audited_distribution"],
            "Excludes test and other non-audited splits.",
        ),
        ("Train Distribution", audit["split_distribution"]["train"], None),
        ("Validation Distribution", audit["split_distribution"]["val"], None),
    ]
    for title, distribution, qualifier in distribution_sections:
        lines.extend(["", f"## {title}", ""])
        if qualifier:
            lines.extend([qualifier, ""])
        lines.extend(["| Horizon | Scene groups |", "| --- | ---: |"])
        lines.extend(_distribution_rows(distribution))

    change_statistics = audit["change_trajectory_statistics"]
    lines.extend(
        [
            "",
            "## Change Trajectory Statistics",
            "",
            f"Status: {change_statistics['status'].replace('_', ' ')}.",
            "",
            change_statistics["reason"],
            "",
            "## Audited Scenes",
            "",
            "| Reference ID | Split | T | Scan IDs (metadata order) |",
            "| --- | --- | ---: | --- |",
        ]
    )
    lines.extend(
        "| {reference_id} | {split} | {T} | {scan_ids} |".format(
            reference_id=scene["reference_id"],
            split=scene["split"],
            T=scene["T"],
            scan_ids=", ".join(scene["scan_ids"]),
        )
        for scene in audit["scenes"]
    )
    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit native temporal horizons in 3RScan metadata."
    )
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--json-output", required=True, type=Path)
    parser.add_argument("--markdown-output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    metadata_path = args.metadata.resolve()
    metadata = load_metadata(metadata_path)
    audit = build_audit(
        metadata,
        source_path=SOURCE_REFERENCE,
        source_sha256=sha256_file(metadata_path),
    )

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.markdown_output.write_text(render_markdown(audit), encoding="utf-8")


if __name__ == "__main__":
    main()
