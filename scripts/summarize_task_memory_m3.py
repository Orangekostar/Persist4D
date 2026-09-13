#!/usr/bin/env python3
"""Build the registered M3 comparison and visual-diagnostic artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts/task_memory_retention_v2"
M3_VARIANTS = ("M3-BASE-CONT", "M3-V-LAST", "M3-V-CORE")
M3_UPDATES = (0, 375, 750, 1125, 1500)
HORIZONS = (2, 3, 4, 5)


class M3SummaryError(ValueError):
    """Raised when M3 evidence is incomplete or internally inconsistent."""


def _clean_float(value: float) -> float:
    if not math.isfinite(value):
        raise M3SummaryError("M3 summary values must be finite")
    return round(value, 15)


def _group_curves(
    curve_rows: Sequence[Mapping[str, object]],
) -> dict[tuple[str, int], dict[int, float]]:
    grouped: dict[tuple[str, int], dict[int, float]] = {}
    identities: dict[tuple[str, int], str] = {}
    for row in curve_rows:
        variant = str(row["variant"])
        update = int(row["update"])
        horizon = int(row["T"])
        tmap = _clean_float(float(row["t_mAP"]))
        identity = str(row["checkpoint_sha256"])
        key = (variant, update)
        if horizon in grouped.setdefault(key, {}):
            raise M3SummaryError("duplicate M3 curve row")
        grouped[key][horizon] = tmap
        previous = identities.setdefault(key, identity)
        if previous != identity:
            raise M3SummaryError("checkpoint identity differs within an M3 curve")
    for key, values in grouped.items():
        if set(values) != set(HORIZONS):
            raise M3SummaryError(f"M3 curve is incomplete: {key}")
    return grouped


def build_m3_comparison_rows(
    curve_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    grouped = _group_curves(curve_rows)
    identities = {
        (str(row["variant"]), int(row["update"])): str(
            row["checkpoint_sha256"]
        )
        for row in curve_rows
    }
    rows = []
    for variant in ("M3-V-LAST", "M3-V-CORE"):
        for update in sorted(
            update
            for candidate, update in grouped
            if candidate == variant
        ):
            key = (variant, update)
            base_key = ("M3-BASE-CONT", update)
            latest_key = ("M3-V-LAST", update)
            if base_key not in grouped or latest_key not in grouped:
                raise M3SummaryError("M3 comparison lacks a same-update control")
            values = grouped[key]
            base = grouped[base_key]
            latest = grouped[latest_key]
            base_deltas = [
                _clean_float(values[horizon] - base[horizon])
                for horizon in HORIZONS
            ]
            latest_deltas = [
                _clean_float(values[horizon] - latest[horizon])
                for horizon in HORIZONS
            ]
            mean_tmap = _clean_float(
                sum(values[horizon] for horizon in HORIZONS) / len(HORIZONS)
            )
            base_mean = _clean_float(
                sum(base[horizon] for horizon in HORIZONS) / len(HORIZONS)
            )
            latest_mean = _clean_float(
                sum(latest[horizon] for horizon in HORIZONS) / len(HORIZONS)
            )
            rows.append(
                {
                    "variant": variant,
                    "update": update,
                    "checkpoint_sha256": identities[key],
                    **{
                        f"T{horizon}_t_mAP": values[horizon]
                        for horizon in HORIZONS
                    },
                    "mean_t_mAP": mean_tmap,
                    "delta_mean_vs_base_cont": _clean_float(
                        mean_tmap - base_mean
                    ),
                    "delta_mean_vs_v_last": _clean_float(
                        mean_tmap - latest_mean
                    ),
                    "s_min_vs_base_cont": min(base_deltas),
                    "s_min_vs_v_last": min(latest_deltas),
                    "s_min_over_controls": min(base_deltas + latest_deltas),
                    "selection_rank": 0,
                    "selected_by_prespecified_rule": False,
                }
            )
    ranked = sorted(
        range(len(rows)),
        key=lambda index: (
            -float(rows[index]["s_min_over_controls"]),
            -float(rows[index]["delta_mean_vs_base_cont"]),
            int(rows[index]["update"]),
            str(rows[index]["variant"]),
        ),
    )
    for rank, index in enumerate(ranked, start=1):
        rows[index]["selection_rank"] = rank
        rows[index]["selected_by_prespecified_rule"] = rank == 1
    return rows


def build_content_ablation_rows(
    *,
    native: Mapping[int, float],
    repeated_mean: Mapping[int, float],
    checkpoint_sha256: str,
    update: int,
) -> list[dict[str, object]]:
    if set(native) != set(HORIZONS) or set(repeated_mean) != set(HORIZONS):
        raise M3SummaryError("content ablation must cover T2-T5")
    native_mean = _clean_float(sum(native.values()) / len(HORIZONS))
    repeated_mean_average = _clean_float(
        sum(repeated_mean.values()) / len(HORIZONS)
    )
    return [
        {
            "variant": "M3-V-CORE",
            "update": update,
            "checkpoint_sha256": checkpoint_sha256,
            "T": horizon,
            "native_t_mAP": _clean_float(native[horizon]),
            "repeated_mean_t_mAP": _clean_float(repeated_mean[horizon]),
            "delta_vs_native_t_mAP": _clean_float(
                repeated_mean[horizon] - native[horizon]
            ),
            "native_four_t_mean": native_mean,
            "repeated_mean_four_t_mean": repeated_mean_average,
            "delta_four_t_mean_vs_native": _clean_float(
                repeated_mean_average - native_mean
            ),
        }
        for horizon in HORIZONS
    ]


def build_memory_rows(schema: Mapping[str, object]) -> list[dict[str, object]]:
    task_bytes = int(schema["task_state_bytes"])
    visual_bytes = int(schema["visual_state_bytes"])
    combined_bytes = int(schema["combined_state_bytes"])
    budget_bytes = int(schema["budget_bytes"])
    representatives = int(schema["representatives_per_entity"])
    if task_bytes + visual_bytes != combined_bytes:
        raise M3SummaryError("visual state byte accounting differs")
    rows = []
    for variant in M3_VARIANTS:
        enabled = variant != "M3-BASE-CONT"
        total = combined_bytes if enabled else task_bytes
        rows.append(
            {
                "variant": variant,
                "representatives_per_entity": representatives if enabled else 0,
                "task_state_bytes": task_bytes,
                "visual_state_bytes": visual_bytes if enabled else 0,
                "combined_state_bytes": total,
                "budget_bytes": budget_bytes,
                "within_budget": total <= budget_bytes,
            }
        )
    return rows


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise M3SummaryError(f"cannot read JSON: {path}") from error
    if not isinstance(value, dict):
        raise M3SummaryError(f"JSON root must be an object: {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, UnicodeError, csv.Error) as error:
        raise M3SummaryError(f"cannot read CSV: {path}") from error
    if not rows:
        raise M3SummaryError(f"CSV is empty: {path}")
    return rows


def _metric_curve(run_dir: Path, *, variant: str, update: int) -> list[dict[str, object]]:
    manifest = _read_json(run_dir / "manifest.json")
    if manifest.get("status") != "PASS" or manifest.get("variant") != variant:
        raise M3SummaryError(f"evaluation manifest failed: {run_dir}")
    checkpoint_sha256 = str(manifest.get("checkpoint_sha256"))
    rows = [
        row
        for row in _read_csv(run_dir / "metrics.csv")
        if row["policy"] == "lag1" and row["reducer"] == "mean"
    ]
    if {int(row["T"]) for row in rows} != set(HORIZONS):
        raise M3SummaryError(f"evaluation lacks lag1 mean T2-T5: {run_dir}")
    return [
        {
            "variant": variant,
            "update": update,
            "T": int(row["T"]),
            "t_mAP": float(row["t_mAP"]),
            "checkpoint_sha256": checkpoint_sha256,
            "policy": "lag1",
            "reducer": "mean",
            "status": "PASS",
        }
        for row in rows
    ]


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise M3SummaryError(f"cannot write empty CSV: {path}")
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise M3SummaryError("CSV rows have inconsistent fields")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        Path(temporary_name).replace(path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _tmaps(run_dir: Path, *, expected_control: str) -> tuple[str, dict[int, float]]:
    manifest = _read_json(run_dir / "manifest.json")
    if (
        manifest.get("status") != "PASS"
        or manifest.get("variant") != "M3-V-CORE"
        or manifest.get("visual_content_control") != expected_control
    ):
        raise M3SummaryError(f"content-control manifest differs: {run_dir}")
    rows = [
        row
        for row in _read_csv(run_dir / "metrics.csv")
        if row["policy"] == "lag1" and row["reducer"] == "mean"
    ]
    values = {int(row["T"]): float(row["t_mAP"]) for row in rows}
    if set(values) != set(HORIZONS):
        raise M3SummaryError("content control lacks T2-T5")
    return str(manifest["checkpoint_sha256"]), values


def build_artifacts(
    *,
    artifact_root: Path,
    native_control_dir: Path,
    repeated_mean_control_dir: Path,
    v_last_diagnostic_dir: Path,
) -> dict[str, int]:
    curves = [
        row
        for variant in M3_VARIANTS
        for update in M3_UPDATES
        for row in _metric_curve(
            artifact_root
            / "evaluation/M3"
            / variant
            / f"update={update:04d}",
            variant=variant,
            update=update,
        )
    ]
    comparisons = build_m3_comparison_rows(curves)
    schema = _read_json(artifact_root / "visual/state_schema.json")
    memory_rows = build_memory_rows(schema)
    native_sha, native = _tmaps(native_control_dir, expected_control="native")
    repeated_sha, repeated = _tmaps(
        repeated_mean_control_dir, expected_control="repeated_mean"
    )
    if native_sha != repeated_sha:
        raise M3SummaryError("content controls use different checkpoints")
    content_rows = build_content_ablation_rows(
        native=native,
        repeated_mean=repeated,
        checkpoint_sha256=native_sha,
        update=1500,
    )
    diagnostic_rows = []
    for run_dir in (
        v_last_diagnostic_dir,
        native_control_dir,
        repeated_mean_control_dir,
    ):
        manifest = _read_json(run_dir / "manifest.json")
        if manifest.get("status") != "PASS":
            raise M3SummaryError(f"visual diagnostic run failed: {run_dir}")
        diagnostic_rows.extend(_read_csv(run_dir / "visual_diagnostics.csv"))
    outputs = {
        artifact_root / "training/M3_learning_curves.csv": curves,
        artifact_root / "training/M3_comparison.csv": comparisons,
        artifact_root / "visual/memory_bytes.csv": memory_rows,
        artifact_root / "visual/paired_content_ablation.csv": content_rows,
        artifact_root / "visual/selection_diagnostics.csv": diagnostic_rows,
    }
    for path, rows in outputs.items():
        _atomic_csv(path, rows)
    return {str(path.relative_to(PROJECT_ROOT)): len(rows) for path, rows in outputs.items()}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--native-control-dir", type=Path, required=True)
    parser.add_argument("--repeated-mean-control-dir", type=Path, required=True)
    parser.add_argument("--v-last-diagnostic-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    counts = build_artifacts(
        artifact_root=args.artifact_root.resolve(),
        native_control_dir=args.native_control_dir.resolve(),
        repeated_mean_control_dir=args.repeated_mean_control_dir.resolve(),
        v_last_diagnostic_dir=args.v_last_diagnostic_dir.resolve(),
    )
    print(json.dumps({"outputs": counts, "status": "PASS"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
