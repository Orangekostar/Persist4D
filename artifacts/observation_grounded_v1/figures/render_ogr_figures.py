#!/usr/bin/env python3
"""Aggregate fixed OGR diagnostics and render the contract figures."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ARM_ORDER = ("R-TARGET", "R-ID", "R-QFB", "R-OBS")
COLORS = {"blue": "#0072B2", "orange": "#D55E00", "gray": "#666666"}
TARGET_REQUIRED_DENOMINATORS = {
    "all_rows": 407,
    "reliable_current_rows": 170,
    "positive_rows": 170,
    "canonical_rows": 165,
    "canonical_reliable_current_rows": 165,
    "anchor_complete_rows": 67,
}


def finite(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite: {value!r}")
    return result


def finite_or_none(value: str, label: str) -> float | None:
    if value in ("NA", ""):
        return None
    return finite(value, label)


def exact_bool(value: str, label: str) -> bool:
    if value not in ("True", "False"):
        raise ValueError(f"{label} must be the string True/False, got {value!r}")
    return value == "True"


def integer(value: str, label: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not an integer: {value!r}") from exc
    if str(result) != value and not (value.startswith("+") and str(result) == value[1:]):
        raise ValueError(f"{label} is not an integer-formatted value: {value!r}")
    return result


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def file_record(root: Path, path: Path) -> dict[str, Any]:
    relative = path.relative_to(root).as_posix()
    return {
        "path": relative,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_target(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = root / "TARGET_FLOW.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != TARGET_REQUIRED_DENOMINATORS["all_rows"]:
        raise ValueError(f"TARGET_FLOW row count changed: {len(rows)}")

    for index, row in enumerate(rows, 1):
        exact_bool(row["current_target_evaluable"], f"TARGET_FLOW row {index} current_target_evaluable")
        exact_bool(row["canonical_entity_event"], f"TARGET_FLOW row {index} canonical_entity_event")
        integer(row["positive_segment_count"], f"TARGET_FLOW row {index} positive_segment_count")

    reliable_current = [
        row
        for row in rows
        if row["history_binding"] == "RELIABLE"
        and exact_bool(row["current_target_evaluable"], "current_target_evaluable")
    ]
    positive = [row for row in rows if integer(row["positive_segment_count"], "positive_segment_count") > 0]
    canonical = [row for row in rows if exact_bool(row["canonical_entity_event"], "canonical_entity_event")]
    canonical_reliable_current = [
        row
        for row in rows
        if exact_bool(row["canonical_entity_event"], "canonical_entity_event")
        and row["history_binding"] == "RELIABLE"
        and exact_bool(row["current_target_evaluable"], "current_target_evaluable")
    ]
    counts = {
        "all_rows": len(rows),
        "reliable_current_rows": len(reliable_current),
        "positive_rows": len(positive),
        "canonical_rows": len(canonical),
        "canonical_reliable_current_rows": len(canonical_reliable_current),
    }
    for name, expected in TARGET_REQUIRED_DENOMINATORS.items():
        if name == "anchor_complete_rows":
            continue
        if counts[name] != expected:
            raise ValueError(f"TARGET_FLOW denominator {name}={counts[name]} expected {expected}")

    target_fields = ("raw_target_IoU", "raw_sameclass_best_IoU", "published_target_IoU")
    complete = [
        row
        for row in canonical_reliable_current
        if all(finite_or_none(row[field], f"TARGET_FLOW {field}") is not None for field in target_fields)
    ]
    if len(complete) != TARGET_REQUIRED_DENOMINATORS["canonical_reliable_current_rows"]:
        raise ValueError(f"TARGET_FLOW common finite denominator changed: {len(complete)}")
    target_means = {
        field: sum(finite(row[field], f"TARGET_FLOW {field}") for row in complete) / len(complete)
        for field in target_fields
    }
    summary_rows = [
        {
            "panel": "source_counts",
            "metric": "all_rows",
            "value": counts["all_rows"],
            "denominator": counts["all_rows"],
            "source": "TARGET_FLOW.csv",
            "filter": "all rows",
        },
        {
            "panel": "source_counts",
            "metric": "reliable_current_rows",
            "value": counts["reliable_current_rows"],
            "denominator": counts["all_rows"],
            "source": "TARGET_FLOW.csv",
            "filter": "history_binding=RELIABLE and current_target_evaluable=True",
        },
        {
            "panel": "source_counts",
            "metric": "positive_rows",
            "value": counts["positive_rows"],
            "denominator": counts["all_rows"],
            "source": "TARGET_FLOW.csv",
            "filter": "positive_segment_count>0",
        },
        {
            "panel": "source_counts",
            "metric": "canonical_rows",
            "value": counts["canonical_rows"],
            "denominator": counts["all_rows"],
            "source": "TARGET_FLOW.csv",
            "filter": "canonical_entity_event=True",
        },
        {
            "panel": "source_counts",
            "metric": "canonical_reliable_current_rows",
            "value": counts["canonical_reliable_current_rows"],
            "denominator": counts["all_rows"],
            "source": "TARGET_FLOW.csv",
            "filter": "canonical_entity_event=True and history_binding=RELIABLE and current_target_evaluable=True",
        },
    ]
    names = {
        "raw_target_IoU": "raw_target_IoU",
        "raw_sameclass_best_IoU": "raw_sameclass_best_IoU",
        "published_target_IoU": "published_target_IoU",
    }
    for field in target_fields:
        summary_rows.append(
            {
                "panel": "target_evidence",
                "metric": names[field],
                "value": target_means[field],
                "denominator": len(complete),
                "source": "TARGET_FLOW.csv",
                "filter": "canonical_entity_event=True and history_binding=RELIABLE and current_target_evaluable=True; common finite rows",
            }
        )
    return summary_rows, {
        "counts": counts,
        "common_denominator": len(complete),
        "means": target_means,
    }


def read_anchor(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = root / "ANCHOR_REFERENCE.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    complete = [row for row in rows if row["status"] == "COMPLETE"]
    if len(complete) != TARGET_REQUIRED_DENOMINATORS["anchor_complete_rows"]:
        raise ValueError(f"ANCHOR_REFERENCE COMPLETE denominator changed: {len(complete)}")
    fields = (
        "paired_raw_target_IoU",
        "raw_target_IoU",
        "paired_published_target_IoU",
        "published_target_IoU",
    )
    common = [
        row
        for row in complete
        if all(finite_or_none(row[field], f"ANCHOR_REFERENCE {field}") is not None for field in fields)
    ]
    if len(common) != len(complete):
        raise ValueError(f"ANCHOR_REFERENCE common finite denominator changed: {len(common)}")
    means = {
        field: sum(finite(row[field], f"ANCHOR_REFERENCE {field}") for row in common) / len(common)
        for field in fields
    }
    summary_rows = [
        {
            "panel": "anchor_evidence",
            "metric": field,
            "value": means[field],
            "denominator": len(common),
            "source": "ANCHOR_REFERENCE.csv",
            "filter": "status=COMPLETE; common finite paired and baseline rows",
        }
        for field in fields
    ]
    return summary_rows, {"complete_rows": len(complete), "common_denominator": len(common), "means": means}


def read_timing(root: Path, run_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    plan_path = root / "RUN_PLAN.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    measured = plan["evaluation_forecast"]["measured_costs"]
    rows: list[dict[str, Any]] = []
    probes: dict[str, float] = {}
    for variant in ARM_ORDER:
        status_path = run_root / "probe" / f"{variant}-s246" / "STATUS.json"
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("status") != "COMPLETE":
            raise ValueError(f"{variant} probe status is not COMPLETE")
        if status.get("identity", {}).get("variant") != variant or status.get("identity", {}).get("adapter_seed") != 246:
            raise ValueError(f"{variant} probe identity mismatch")
        if status.get("planned_updates") != 20 or len(status.get("seconds_per_update", [])) != 20:
            raise ValueError(f"{variant} probe does not contain 20 updates")
        value = finite(status["timed_p90_seconds"], f"{variant} timed_p90_seconds")
        probes[variant] = value
        rows.append(
            {
                "panel": "probe_timing",
                "variant": variant,
                "role": "PROBE",
                "metric": "timed_p90_seconds",
                "value": value,
                "denominator": 16,
                "source": f"probe/{variant}-s246/STATUS.json",
                "filter": "STATUS.json timed_p90_seconds; final 16 of 20 updates",
            }
        )

    structure: dict[str, float] = {}
    for key in ("R-OBS:CAL", "R-OBS:PB", "FH-COMMIT0:CAL", "FH-COMMIT0:PB"):
        if key not in measured:
            raise ValueError(f"RUN_PLAN missing measured cost {key}")
        value = finite(measured[key]["p90_stage_seconds"], f"RUN_PLAN {key} p90_stage_seconds")
        structure[key] = value
        method, role = key.split(":")
        binding = next(row for row in plan["evaluation_forecast"]["timing_evidence"]
                       if row["method"] == method and row["role"] == role)
        evidence_path = run_root / binding["path"]
        if sha256_file(evidence_path) != binding["sha256"]:
            raise ValueError(f"Timing evidence identity changed: {key}")
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        stage_count = sum(len(episode["timings"]) for episode in evidence["episodes"])
        rows.append(
            {
                "panel": "structure_timing",
                "variant": method,
                "role": role,
                "metric": "p90_stage_seconds",
                "value": value,
                "denominator": stage_count,
                "source": "RUN_PLAN.json",
                "filter": f"evaluation_forecast.measured_costs[{key}]; unscaled p90_stage_seconds",
            }
        )
    return rows, {"probes": probes, "structure": structure}


def format_value(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return format(value, ".17g")
    return str(value)


def annotate_bars(axis: Any, bars: Any, values: list[float], labels: list[str], ylim: tuple[float, float]) -> None:
    upper = ylim[1]
    for bar, value, label in zip(bars, values, labels):
        if value > upper * 0.84:
            y, va = value - upper * 0.025, "top"
        else:
            y, va = value + upper * 0.025, "bottom"
        axis.text(bar.get_x() + bar.get_width() / 2, y, label, ha="center", va=va, fontsize=9)


def configure_axis(axis: Any, ylabel: str, ylim: tuple[float, float]) -> None:
    axis.set_ylabel(ylabel)
    axis.set_ylim(*ylim)
    axis.grid(axis="y", color="#dddddd", linewidth=0.7)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def render_target(root: Path, figures: Path, target: dict[str, Any], anchor: dict[str, Any]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    figure.suptitle(
        "Target evidence | M4-200/shared-M0 | GT-assisted diagnostic\n"
        "Diagnostic only; not an OGR formal score or theoretical upper bound",
        fontsize=12,
    )

    target_labels = ["raw target", "same-class best", "published target"]
    target_values = [
        target["means"]["raw_target_IoU"],
        target["means"]["raw_sameclass_best_IoU"],
        target["means"]["published_target_IoU"],
    ]
    target_bars = axes[0].bar(
        target_labels,
        target_values,
        color=[COLORS["blue"], COLORS["gray"], COLORS["orange"]],
        edgecolor="#333333",
        linewidth=0.7,
        hatch=["", "..", "//"],
    )
    axes[0].set_title("Shared M0 history: designated target evidence")
    configure_axis(axes[0], "IoU", (0, 1))
    axes[0].tick_params(axis="x", labelrotation=12)
    annotate_bars(axes[0], target_bars, target_values, [f"{v:.3f}\n(n={target['common_denominator']})" for v in target_values], (0, 1))

    paired_labels = ["raw IoU", "published IoU"]
    x = [0, 1]
    width = 0.34
    baseline = [anchor["means"]["paired_raw_target_IoU"], anchor["means"]["paired_published_target_IoU"]]
    intervened = [anchor["means"]["raw_target_IoU"], anchor["means"]["published_target_IoU"]]
    bars_baseline = axes[1].bar(
        [v - width / 2 for v in x],
        baseline,
        width,
        label="baseline",
        color=[COLORS["blue"], COLORS["orange"]],
        edgecolor="#333333",
        linewidth=0.7,
    )
    bars_intervened = axes[1].bar(
        [v + width / 2 for v in x],
        intervened,
        width,
        label="GT-assisted anchor",
        color=[COLORS["blue"], COLORS["orange"]],
        edgecolor="#333333",
        linewidth=0.7,
        hatch="//",
    )
    axes[1].set_xticks(x, paired_labels)
    axes[1].set_title("Fixed GT-assisted anchor reference")
    configure_axis(axes[1], "IoU", (0, 1))
    axes[1].legend(frameon=False, fontsize=9, loc="upper right")
    annotate_bars(axes[1], bars_baseline, baseline, [f"{v:.3f}\n(n={anchor['common_denominator']})" for v in baseline], (0, 1))
    annotate_bars(axes[1], bars_intervened, intervened, [f"{v:.3f}\n(n={anchor['common_denominator']})" for v in intervened], (0, 1))
    save_figure(figure, figures / "target_evidence")


def render_resource(figures: Path, timing: dict[str, Any]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    figure.suptitle(
        "Resource evidence | nonformal probes/structural timing\n"
        "Partial resource evidence; not a same-card P6 profile",
        fontsize=12,
    )

    probe_values = [timing["probes"][variant] for variant in ARM_ORDER]
    probe_bars = axes[0].bar(
        list(ARM_ORDER),
        probe_values,
        color=[COLORS["blue"], COLORS["orange"], COLORS["gray"], COLORS["blue"]],
        edgecolor="#333333",
        linewidth=0.7,
        hatch=["", "", "", "//"],
    )
    axes[0].set_title("Probe timed p90: final 16 of 20 updates")
    probe_upper = max(probe_values) * 1.28
    configure_axis(axes[0], "seconds/update", (0, probe_upper))
    axes[0].tick_params(axis="x", labelrotation=15)
    annotate_bars(axes[0], probe_bars, probe_values, [f"{v:.3f}\n(n=16 timed)" for v in probe_values], (0, probe_upper))

    roles = ["CAL", "PB"]
    x = [0, 1]
    width = 0.34
    obs = [timing["structure"][f"R-OBS:{role}"] for role in roles]
    fh = [timing["structure"][f"FH-COMMIT0:{role}"] for role in roles]
    obs_bars = axes[1].bar(
        [v - width / 2 for v in x],
        obs,
        width,
        label="R-OBS",
        color=COLORS["blue"],
        edgecolor="#333333",
        linewidth=0.7,
    )
    fh_bars = axes[1].bar(
        [v + width / 2 for v in x],
        fh,
        width,
        label="FH-COMMIT0",
        color=COLORS["orange"],
        edgecolor="#333333",
        linewidth=0.7,
        hatch="//",
    )
    axes[1].set_xticks(x, roles)
    axes[1].set_title("Unscaled stage p90: R-OBS vs FH-COMMIT0")
    structure_upper = max(obs + fh) * 1.38
    configure_axis(axes[1], "seconds/stage", (0, structure_upper))
    axes[1].legend(frameon=False, fontsize=9, loc="upper right")
    annotate_bars(axes[1], obs_bars, obs, [f"{v:.3f}\n(p90)" for v in obs], (0, structure_upper))
    annotate_bars(axes[1], fh_bars, fh, [f"{v:.3f}\n(p90)" for v in fh], (0, structure_upper))
    save_figure(figure, figures / "resource_probe")


def save_figure(figure: Any, stem: Path) -> None:
    svg_path = stem.with_suffix(".svg")
    figure.savefig(svg_path, format="svg", dpi=160, facecolor="white")
    svg_text = svg_path.read_text(encoding="utf-8")
    svg_path.write_text("\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n", encoding="utf-8")
    figure.savefig(stem.with_suffix(".pdf"), format="pdf", dpi=160, facecolor="white")
    figure.savefig(stem.with_suffix(".png"), format="png", dpi=160, facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True, help="Private run root containing probe/<variant>-s246/STATUS.json")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    run_root = args.run_root.expanduser().resolve()

    diagnostic_rows, target = read_target(root)
    anchor_rows, anchor = read_anchor(root)
    timing_rows, timing = read_timing(root, run_root)

    summary_fields = ["panel", "metric", "value", "denominator", "source", "filter"]
    write_csv(root / "DIAGNOSTIC_SUMMARY.csv", summary_fields, diagnostic_rows)
    write_csv(root / "ANCHOR_SUMMARY.csv", summary_fields, anchor_rows)
    timing_fields = ["panel", "variant", "role", "metric", "value", "denominator", "source", "filter"]
    write_csv(root / "TIMING_SUMMARY.csv", timing_fields, timing_rows)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    render_target(root, figures, target, anchor)
    render_resource(figures, timing)

    source_paths = [
        root / "VISUAL_CONTRACT.md",
        root / "TARGET_FLOW.csv",
        root / "ANCHOR_REFERENCE.csv",
        root / "RUN_PLAN.json",
    ] + [run_root / "probe" / f"{variant}-s246" / "STATUS.json" for variant in ARM_ORDER]
    source_records = []
    for path in source_paths:
        if path.is_relative_to(root):
            source_records.append(file_record(root, path))
        else:
            source_records.append(
                {
                    "path": f"probe/{path.parent.name}/STATUS.json",
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )

    aggregation_paths = [root / name for name in ("DIAGNOSTIC_SUMMARY.csv", "ANCHOR_SUMMARY.csv", "TIMING_SUMMARY.csv")]
    figure_paths = [
        figures / "target_evidence.svg",
        figures / "target_evidence.pdf",
        figures / "target_evidence.png",
        figures / "resource_probe.svg",
        figures / "resource_probe.pdf",
        figures / "resource_probe.png",
    ]
    manifest = {
        "schema": 1,
        "status": "GENERATED",
        "contract": "VISUAL_CONTRACT.md",
        "execution": "read-only aggregation and CPU plotting; no GPU, network, training, or scientific inference",
        "source_files": source_records,
        "aggregations": [file_record(root, path) for path in aggregation_paths],
        "figures": [file_record(root, path) for path in figure_paths],
        "families": {
            "quality": {
                "status": "NOT_GENERATED_BUDGET_BLOCKED",
                "reason": "Formal AP endpoints are unavailable because the fixed evaluation phase is budget blocked.",
            },
            "target": {
                "status": "GENERATED_SHARED_HISTORY_DIAGNOSTIC",
                "source": "TARGET_FLOW.csv and ANCHOR_REFERENCE.csv",
                "denominators": {
                    "target_all": target["counts"]["all_rows"],
                    "target_reliable_current": target["counts"]["reliable_current_rows"],
                    "target_positive": target["counts"]["positive_rows"],
                    "target_canonical": target["counts"]["canonical_rows"],
                    "target_panel_common_finite": target["common_denominator"],
                    "anchor_complete_common_finite": anchor["common_denominator"],
                },
                "scope": "M4-200/shared-M0 and GT-assisted diagnostic; not an OGR formal score or theoretical upper bound.",
            },
            "recovery_resources": {
                "status": "PARTIAL_PROBE_AND_STRUCTURE_TIMING_ONLY",
                "reason": "Recovery/loss evidence and the seven-configuration P6 profile require unavailable formal endpoints; only nonformal probes and measured structural timing are present.",
            },
        },
    }
    (root / "FIGURE_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"target_denominator": target["common_denominator"], "anchor_denominator": anchor["common_denominator"], "probe_variants": list(timing["probes"]), "figures": len(figure_paths)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
