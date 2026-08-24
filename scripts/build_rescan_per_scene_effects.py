"""Build explicit per-scene Persist4D minus B2 effect rows."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

OUTPUT_FIELDS = (
    "scene_id",
    "level",
    "metric",
    "higher_is_better",
    "comparator",
    "comparator_value",
    "method",
    "method_value",
    "effect",
    "absolute_effect",
    "relative_effect",
)

_METRICS = (
    ("level_a", "online_t_mAP", True),
    ("level_a", "online_t_REC", True),
    ("level_b", "observation_coverage", True),
    ("level_b", "gap_recovery_accuracy", True),
    ("level_b", "gap_recovery_recall", True),
    ("level_b", "normalized_id_switch_rate", False),
    ("level_b", "fragmentation_count", False),
    ("level_b", "merge_count", False),
)


def _parse_metric_value(row: Mapping[str, str], column: str) -> float:
    try:
        value = float(row[column])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid metric value for {column}") from error
    if not math.isfinite(value):
        raise ValueError(f"nonfinite metric value for {column}")
    return value


def build_per_scene_effect_rows(
    rows: Sequence[Mapping[str, str]],
) -> list[dict[str, object]]:
    """Build deterministic B4-minus-B2 rows for each scene and fixed metric."""
    by_scene: dict[str, dict[str, Mapping[str, str]]] = defaultdict(dict)
    for row in rows:
        scene_id = row.get("scene_id")
        method_code = row.get("method_code")
        if not scene_id:
            raise ValueError("missing scene_id")
        methods = by_scene[scene_id]
        if method_code in ("B2", "B4"):
            if method_code in methods:
                raise ValueError(f"duplicate {method_code} for {scene_id}")
            methods[method_code] = row

    for scene_id, methods in by_scene.items():
        missing = [method for method in ("B2", "B4") if method not in methods]
        if missing:
            raise ValueError(f"missing {missing[0]} for {scene_id}")

    output: list[dict[str, object]] = []
    for scene_id in sorted(by_scene):
        comparator_row = by_scene[scene_id]["B2"]
        method_row = by_scene[scene_id]["B4"]
        for level, metric, higher_is_better in _METRICS:
            column = f"{level}_{metric}"
            comparator_value = _parse_metric_value(comparator_row, column)
            method_value = _parse_metric_value(method_row, column)
            effect = method_value - comparator_value
            absolute_effect = abs(effect)
            relative_effect: float | str = (
                effect / abs(comparator_value) if comparator_value != 0.0 else ""
            )
            output.append(
                {
                    "scene_id": scene_id,
                    "level": level,
                    "metric": metric,
                    "higher_is_better": higher_is_better,
                    "comparator": "B2",
                    "comparator_value": comparator_value,
                    "method": "B4",
                    "method_value": method_value,
                    "effect": effect,
                    "absolute_effect": absolute_effect,
                    "relative_effect": relative_effect,
                }
            )
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", dest="input_path", type=Path, required=True)
    parser.add_argument("--output", dest="output_path", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    with arguments.input_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    effect_rows = build_per_scene_effect_rows(rows)
    arguments.output_path.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=OUTPUT_FIELDS,
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(effect_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
