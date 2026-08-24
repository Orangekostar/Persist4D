"""Build paper-facing comparison tables from frozen evidence CSVs."""

from __future__ import annotations

import argparse
import csv
from collections.abc import Iterable, Mapping
from pathlib import Path

PUBLISHED_ROWS = (
    ("Mask4D", 1.3, 2.9, 8.7, 2.1, 5.5, 21.2),
    ("Mask4Former", 17.0, 38.9, 59.1, 21.7, 45.6, 66.3),
    ("Mask3D + Semantic Matching", 20.1, 32.9, 38.6, 25.9, 42.3, 73.9),
    ("Mask3D + Geometric Matching", 20.7, 43.1, 62.4, 29.7, 54.1, 70.9),
    ("ReScene4D (C)", 34.8, 52.5, 66.8, 43.3, 64.3, 81.9),
)
PUBLISHED_SOURCE = "https://arxiv.org/abs/2601.11508"
HORIZONS = (4, 5)


class PaperEvidenceError(ValueError):
    """Raised when frozen inputs do not provide an exact requested cell."""


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _one(rows: Iterable[Mapping[str, str]], **conditions: object) -> Mapping[str, str]:
    matches = [
        row
        for row in rows
        if all(str(row.get(key)) == str(value) for key, value in conditions.items())
    ]
    if len(matches) != 1:
        raise PaperEvidenceError(
            f"expected one row for {conditions}, found {len(matches)}"
        )
    return matches[0]


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise PaperEvidenceError(f"refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _published_table() -> list[dict[str, object]]:
    columns = ("t_mAP", "t_mAP50", "t_mAP25", "mAP", "mAP50", "mAP25")
    return [
        {
            "method": method,
            **dict(zip(columns, values, strict=True)),
            "unit": "percent",
            "result_status": "reported_result_not_rerun",
            "protocol": "ReScene4D standard published protocol",
            "source": PUBLISHED_SOURCE,
        }
        for method, *values in PUBLISHED_ROWS
    ]


def _common_prefix_table(root: Path) -> list[dict[str, object]]:
    closure = root / "artifacts/reviewer_closure"
    tracker_rows = _read_csv(closure / "full_history_tracker_aggregate.csv")
    adapted_rows = _read_csv(closure / "rescene_horizon_adaptation_results.csv")
    compute_rows = _read_csv(closure / "rescene_horizon_compute.csv")

    method_specs = (
        (
            "FullHistoryNative",
            "ReScene4D Full-History",
            "FullHistoryFrozen",
            tracker_rows,
        ),
        ("B1", "Full-History + Pairwise Feature", "FullHistoryFrozen", tracker_rows),
        ("B2", "Full-History + Feature-Class", "FullHistoryFrozen", tracker_rows),
        ("B3", "Full-History + EMA", "FullHistoryFrozen", tracker_rows),
        (
            "FullHistoryAdaptedB2",
            "T2-to-T3 Horizon-Adapted Full-History + Feature-Class",
            "FullHistoryAdapted",
            adapted_rows,
        ),
        ("Persist4D", "Persist4D", "Persist4D", adapted_rows),
    )
    output = []
    for method_id, paper_name, compute_id, result_rows in method_specs:
        for horizon in HORIZONS:
            result_key = "method_id"
            result = _one(result_rows, **{result_key: method_id, "horizon": horizon})
            compute = _one(compute_rows, method_id=compute_id, horizon=horizon)
            is_persist4d = method_id == "Persist4D"
            is_native = method_id == "FullHistoryNative"
            output.append(
                {
                    "method": paper_name,
                    "history_strategy": (
                        "bounded local window + persistent entity state"
                        if is_persist4d
                        else "expanding full-history perception"
                    ),
                    "horizon": horizon,
                    "t_mAP": result["causal_prefix_t_mAP"],
                    "t_REC": result["causal_prefix_t_REC"],
                    "normalized_id_switch_rate": result["normalized_id_switch_rate"],
                    "gap_recovery_recall": result["gap_recovery_recall"],
                    "update_latency_ms": compute["median_latency_ms"],
                    "peak_allocated_vram_mib": compute["peak_allocated_mib"],
                    "historical_state_bytes": compute["historical_state_bytes"],
                    "explicit_history_input_bytes": compute[
                        "explicit_history_input_bytes"
                    ],
                    "runtime_scope": (
                        "complete Persist4D update"
                        if is_persist4d
                        else (
                            "full-history perception"
                            if is_native
                            else "full-history perception; tracker overhead excluded"
                        )
                    ),
                    "task_metric_source": result["task_metric_source"],
                    "sequence_count": result["sequence_count"],
                }
            )
    return output


def build(root: Path, output_dir: Path) -> None:
    table_2 = _common_prefix_table(root)
    table_3_methods = {
        "Full-History + Feature-Class",
        "Full-History + EMA",
        "T2-to-T3 Horizon-Adapted Full-History + Feature-Class",
        "Persist4D",
    }
    _write_csv(output_dir / "table_1_published_4dsis.csv", _published_table())
    _write_csv(output_dir / "table_2_common_prefix.csv", table_2)
    _write_csv(
        output_dir / "table_3_strong_alternatives.csv",
        [row for row in table_2 if row["method"] in table_3_methods],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/final_evidence/tables"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    build(root, output_dir)


if __name__ == "__main__":
    main()
