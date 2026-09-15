#!/usr/bin/env python3
"""Materialize canonical TaskMemory training and checkpoint summaries."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

from scripts.task_memory_contracts import canonical_json_sha256

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT_ROOT / "artifacts/task_memory_retention_v2"
HORIZONS = (2, 3, 4, 5)
M2_VARIANTS = ("W-BASE", "Q-INDEP", "Q-TALA", "FH-MATCH")
M2_UPDATES = (0, 750, 1500, 2250, 3000)
TERMINAL_UPDATES = {
    "W-BASE": 3000,
    "Q-INDEP": 3000,
    "Q-TALA": 3000,
    "FH-MATCH": 3000,
    "M3-BASE-CONT": 1500,
    "M3-V-LAST": 1500,
    "M3-V-CORE": 1500,
    "FH-CONT": 1500,
}
SELECTED_UPDATES = {
    "W-BASE": (3000, "m2_arm_best_mean", False),
    "Q-INDEP": (1500, "m2_gate_same_update_control", False),
    "Q-TALA": (1500, "m3_parent_selected", False),
    "FH-MATCH": (750, "fh_cont_parent_selected", False),
    "M3-BASE-CONT": (1500, "m3_same_update_control", False),
    "M3-V-CORE": (1500, "final_candidate_selected", True),
    "FH-CONT": (1500, "matched_continuation_selected", True),
}
CURVE_FIELDS = (
    "phase",
    "variant",
    "update",
    "T",
    "t_mAP",
    "checkpoint_sha256",
    "policy",
    "reducer",
    "status",
)
SNAPSHOT_FIELDS = (
    "selection_role",
    "phase",
    "variant",
    "update",
    "checkpoint_sha256",
    "T2_t_mAP",
    "T3_t_mAP",
    "T4_t_mAP",
    "T5_t_mAP",
    "mean_t_mAP",
    "policy",
    "reducer",
    "selected_for_final",
)


class TrainingSummaryError(RuntimeError):
    """Raised when completed training evidence is incomplete or inconsistent."""


def _load_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise TrainingSummaryError(f"cannot read JSON: {path}") from error
    if not isinstance(value, dict):
        raise TrainingSummaryError(f"JSON root must be an object: {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, UnicodeError, csv.Error) as error:
        raise TrainingSummaryError(f"cannot read CSV: {path}") from error
    if not rows:
        raise TrainingSummaryError(f"CSV is empty: {path}")
    return rows


def _csv_bytes(rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> bytes:
    if not rows or any(set(row) != set(fields) for row in rows):
        raise TrainingSummaryError("summary CSV schema differs")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("ascii")


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise TrainingSummaryError(f"output cannot be a symlink: {path}")
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_identity(
    summary: Mapping[str, object], *, selected_update: int
) -> dict[str, object]:
    if (
        summary.get("status") != "COMPLETE"
        or not isinstance(summary.get("variant"), str)
        or summary.get("completed_global_step") not in {1500, 3000}
    ):
        raise TrainingSummaryError("training run summary is incomplete")
    records = summary.get("checkpoints")
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise TrainingSummaryError("checkpoint records differ")
    by_name = {
        record.get("name"): record
        for record in records
        if isinstance(record, Mapping) and isinstance(record.get("name"), str)
    }
    selected = by_name.get(f"update={selected_update:04d}.ckpt")
    last = by_name.get("last.ckpt")
    if not isinstance(selected, Mapping):
        raise TrainingSummaryError("selected checkpoint is unavailable")
    if not isinstance(last, Mapping):
        raise TrainingSummaryError("last checkpoint is unavailable")

    def normalize(record: Mapping[str, object]) -> dict[str, object]:
        sha256 = record.get("sha256")
        size = record.get("bytes")
        if (
            not isinstance(sha256, str)
            or len(sha256) != 64
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
        ):
            raise TrainingSummaryError("checkpoint identity differs")
        return {"bytes": size, "sha256": sha256}

    return {
        "variant": summary["variant"],
        "selected": {"update": selected_update, **normalize(selected)},
        "last": {
            "update": summary["completed_global_step"],
            **normalize(last),
            "optimizer_state_retained": True,
        },
    }


def curve_snapshot(
    rows: Sequence[Mapping[str, object]], *, variant: str, update: int
) -> dict[str, object]:
    selected = [
        row
        for row in rows
        if row.get("variant") == variant and int(row.get("update", -1)) == update
    ]
    indexed = {int(row["T"]): row for row in selected}
    if len(selected) != 4 or set(indexed) != set(HORIZONS):
        raise TrainingSummaryError("curve horizon coverage differs")
    checkpoints = {str(row["checkpoint_sha256"]) for row in selected}
    phases = {str(row["phase"]) for row in selected}
    if (
        len(checkpoints) != 1
        or len(next(iter(checkpoints))) != 64
        or len(phases) != 1
        or any(row.get("policy") != "lag1" for row in selected)
        or any(row.get("reducer") != "mean" for row in selected)
        or any(row.get("status") != "PASS" for row in selected)
    ):
        raise TrainingSummaryError("curve checkpoint identity differs")
    values = {}
    for horizon in HORIZONS:
        value = float(indexed[horizon]["t_mAP"])
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise TrainingSummaryError("curve t_mAP differs")
        values[horizon] = value
    return {
        "phase": next(iter(phases)),
        "variant": variant,
        "update": update,
        "checkpoint_sha256": next(iter(checkpoints)),
        **{f"T{horizon}_t_mAP": values[horizon] for horizon in HORIZONS},
        "mean_t_mAP": sum(values.values()) / len(values),
        "policy": "lag1",
        "reducer": "mean",
    }


def _normalize_curve_row(
    row: Mapping[str, object], *, phase: str, variant: str, update: int
) -> dict[str, object]:
    if (
        row.get("variant") != variant
        or row.get("policy") != "lag1"
        or row.get("reducer") != "mean"
        or row.get("status", "PASS") != "PASS"
    ):
        raise TrainingSummaryError("learning-curve row differs")
    horizon = int(row["T"])
    value = float(row["t_mAP"])
    checkpoint = str(row["checkpoint_sha256"])
    if horizon not in HORIZONS or not 0 <= value <= 1 or len(checkpoint) != 64:
        raise TrainingSummaryError("learning-curve value differs")
    return {
        "phase": phase,
        "variant": variant,
        "update": update,
        "T": horizon,
        "t_mAP": value,
        "checkpoint_sha256": checkpoint,
        "policy": "lag1",
        "reducer": "mean",
        "status": "PASS",
    }


def _load_curves(root: Path) -> tuple[list[dict[str, object]], list[Path]]:
    rows = []
    sources = []
    for variant in M2_VARIANTS:
        for update in M2_UPDATES:
            path = root / f"evaluation/M2/{variant}/update={update:04d}/metrics.csv"
            sources.append(path)
            selected = [
                row
                for row in _read_csv(path)
                if row.get("population_id") == "development_common_h5_canonical"
                and row.get("policy") == "lag1"
                and row.get("reducer") == "mean"
            ]
            if len(selected) != 4:
                raise TrainingSummaryError("M2 learning-curve coverage differs")
            rows.extend(
                _normalize_curve_row(
                    row, phase="M2", variant=variant, update=update
                )
                for row in selected
            )
    for filename, phase in (
        ("M3_learning_curves.csv", "M3"),
        ("FH_CONT_learning_curves.csv", "M5_MATCHED_CONTROL"),
    ):
        path = root / "training" / filename
        sources.append(path)
        for row in _read_csv(path):
            rows.append(
                _normalize_curve_row(
                    row,
                    phase=phase,
                    variant=str(row["variant"]),
                    update=int(row["update"]),
                )
            )
    if len(rows) != 160:
        raise TrainingSummaryError("canonical learning-curve coverage differs")
    return sorted(
        rows,
        key=lambda row: (str(row["phase"]), str(row["variant"]), int(row["update"]), int(row["T"])),
    ), sources


def materialize_training_summary(root: Path = DEFAULT_ROOT) -> dict[str, object]:
    root = root.expanduser().resolve()
    curves, sources = _load_curves(root)
    terminal = []
    for variant, update in TERMINAL_UPDATES.items():
        terminal.append(
            {
                "selection_role": "fixed_terminal_update",
                **curve_snapshot(curves, variant=variant, update=update),
                "selected_for_final": variant in {"M3-V-CORE", "FH-CONT"},
            }
        )
    selected = []
    checkpoint_records = []
    for variant, (update, role, selected_for_final) in SELECTED_UPDATES.items():
        snapshot = curve_snapshot(curves, variant=variant, update=update)
        selected.append(
            {
                "selection_role": role,
                **snapshot,
                "selected_for_final": selected_for_final,
            }
        )
        summary_path = root / f"training/formal/{variant}/run_summary.json"
        sources.append(summary_path)
        identity = checkpoint_identity(_load_json(summary_path), selected_update=update)
        for kind in ("selected", "last"):
            identity[kind]["logical_reference"] = (
                f"external:run_root/training/formal/{variant}/"
                + (
                    f"update={update:04d}.ckpt"
                    if kind == "selected"
                    else "last.ckpt"
                )
            )
        identity["selection_role"] = role
        identity["selected_for_final"] = selected_for_final
        checkpoint_records.append(identity)
    output_root = root / "training"
    outputs = {
        "learning_curves.csv": _csv_bytes(curves, CURVE_FIELDS),
        "terminal_update_comparison.csv": _csv_bytes(terminal, SNAPSHOT_FIELDS),
        "selected_checkpoint_comparison.csv": _csv_bytes(selected, SNAPSHOT_FIELDS),
    }
    for filename, content in outputs.items():
        _atomic_write(output_root / filename, content)
    manifest: dict[str, object] = {
        "schema_version": "task-memory-selected-checkpoints-v1",
        "selection_population": "development_common_h5_canonical",
        "policy": "lag1",
        "reducer": "mean",
        "records": checkpoint_records,
        "sources": [
            {
                "logical_reference": f"repo:{path.relative_to(PROJECT_ROOT)}",
                "sha256": _sha256(path),
            }
            for path in sorted(set(sources))
        ],
        "status": "PASS",
    }
    manifest["content_sha256"] = canonical_json_sha256(manifest)
    _atomic_write(
        output_root / "selected_checkpoints.json",
        (json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n").encode(
            "ascii"
        ),
    )
    return {
        "curve_rows": len(curves),
        "terminal_rows": len(terminal),
        "selected_rows": len(selected),
        "content_sha256": manifest["content_sha256"],
        "status": "PASS",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    print(json.dumps(materialize_training_summary(args.artifact_root), sort_keys=True))
    return 0


__all__ = [
    "TrainingSummaryError",
    "checkpoint_identity",
    "curve_snapshot",
    "materialize_training_summary",
]


if __name__ == "__main__":
    raise SystemExit(main())
