"""Evidence-validated publication figures for the reviewer-closure study."""

from __future__ import annotations

import argparse
import csv
import html
import io
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt

plt.switch_backend("Agg")


METHODS = ("FullHistory", "Persist4D")
HORIZONS = (2, 3, 4, 5)
DECOMPOSITION_HORIZONS = (4, 5)
IOU_THRESHOLDS = tuple(value / 100 for value in range(25, 91, 5))
COVERAGE_THRESHOLDS = (0.25, 0.50, 0.75)
COVERAGE_CATEGORIES = (
    "no_candidate_observation",
    "wrong_class",
    "insufficient_iou",
    "associable",
)
FAILURE_CATEGORIES = (
    "local_observation_miss",
    "class_failure",
    "high_iou_mask_failure",
    "identity_fragmentation",
    "identity_merge",
    "wrong_gap_recovery",
    "capacity_failure",
    "unknown_unresolved",
)
TRACKER_METHODS = ("FullHistoryNative", "B2", "Persist4D")
ADAPTATION_TASK_METHODS = (
    "FullHistoryFrozenB2",
    "FullHistoryAdaptedB2",
    "Persist4D",
)

FULL_COLOR = "#0072B2"
PERSIST_COLOR = "#D55E00"
TRACKER_COLOR = "#009E73"
ORACLE_COLOR = "#666666"
COVERAGE_COLORS = {
    "no_candidate_observation": "#A6A6A6",
    "wrong_class": "#E69F00",
    "insufficient_iou": "#56B4E9",
    "associable": "#009E73",
}
FAILURE_COLORS = {
    "local_observation_miss": "#A6A6A6",
    "class_failure": "#E69F00",
    "high_iou_mask_failure": "#56B4E9",
    "identity_fragmentation": "#0072B2",
    "identity_merge": "#CC79A7",
    "wrong_gap_recovery": "#009E73",
    "capacity_failure": "#D55E00",
    "unknown_unresolved": "#222222",
}
DISPLAY = {
    "FullHistory": "Full-History",
    "Persist4D": "Persist4D",
    "Oracle": "P6-A GT-ID diagnostic",
    "FullHistoryNative": "Full-History native",
    "B2": "B2 feature + class",
    "FullHistoryFrozenB2": "Frozen ReScene + B2",
    "FullHistoryAdaptedB2": "Horizon-adapted ReScene + B2",
    "no_candidate_observation": "No candidate",
    "wrong_class": "Wrong class",
    "insufficient_iou": "Insufficient IoU",
    "associable": "Associable",
    "local_observation_miss": "Observation miss",
    "class_failure": "Class failure",
    "high_iou_mask_failure": "Mask < 0.50 IoU",
    "identity_fragmentation": "Fragmentation",
    "identity_merge": "Merge",
    "wrong_gap_recovery": "Wrong recovery",
    "capacity_failure": "Capacity",
    "unknown_unresolved": "Unknown / unresolved",
}


class FigureEvidenceError(ValueError):
    """Raised when a figure source is incomplete or internally inconsistent."""


def _number(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise FigureEvidenceError(f"{name} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise FigureEvidenceError(f"{name} must be finite") from error
    if not math.isfinite(result):
        raise FigureEvidenceError(f"{name} must be finite")
    return result


def _integer(value: object, *, name: str) -> int:
    result = _number(value, name=name)
    if result < 0 or result != int(result):
        raise FigureEvidenceError(f"{name} must be a non-negative integer")
    return int(result)


def _optional_number(value: object, *, name: str) -> float | None:
    if value is None or value == "":
        return None
    return _number(value, name=name)


def _unique_cells(
    rows: Sequence[Mapping[str, object]],
    *,
    keys: Sequence[str],
    label: str,
) -> dict[tuple[object, ...], Mapping[str, object]]:
    result: dict[tuple[object, ...], Mapping[str, object]] = {}
    for row in rows:
        cell = tuple(row.get(key) for key in keys)
        if cell in result:
            raise FigureEvidenceError(f"{label} contains duplicate cell {cell}")
        result[cell] = row
    return result


def _normalize_iou_rows(
    rows: Sequence[Mapping[str, object]],
) -> dict[tuple[str, int, float], float]:
    normalized = []
    for row in rows:
        normalized.append(
            {
                **row,
                "method": str(row.get("method")),
                "horizon": _integer(row.get("horizon"), name="IoU horizon"),
                "iou_threshold": round(
                    _number(row.get("iou_threshold"), name="IoU threshold"), 2
                ),
            }
        )
    cells = _unique_cells(
        normalized,
        keys=("method", "horizon", "iou_threshold"),
        label="IoU sweep",
    )
    expected = {
        (method, horizon, threshold)
        for method in METHODS
        for horizon in DECOMPOSITION_HORIZONS
        for threshold in IOU_THRESHOLDS
    }
    if set(cells) != expected:
        raise FigureEvidenceError("IoU sweep lacks exact T4/T5 threshold coverage")
    result = {}
    for cell, row in cells.items():
        if row.get("aggregation") != "pooled class-macro official stmetrics":
            raise FigureEvidenceError("IoU sweep aggregation is not the frozen metric")
        if _integer(row.get("sequence_count"), name="IoU sequence count") != 129:
            raise FigureEvidenceError("IoU sweep must contain 129 sequences")
        value = _number(row.get("temporal_ap"), name="temporal AP")
        if not 0 <= value <= 1:
            raise FigureEvidenceError("temporal AP must lie in [0, 1]")
        result[cell] = value
    return result


def _normalize_coverage_rows(
    rows: Sequence[Mapping[str, object]],
) -> dict[tuple[str, int, float, str], float]:
    normalized = []
    for row in rows:
        normalized.append(
            {
                **row,
                "method": str(row.get("method")),
                "horizon": _integer(row.get("horizon"), name="coverage horizon"),
                "iou_threshold": round(
                    _number(row.get("iou_threshold"), name="coverage threshold"), 2
                ),
                "category": str(row.get("category")),
            }
        )
    cells = _unique_cells(
        normalized,
        keys=("method", "horizon", "iou_threshold", "category"),
        label="Observation coverage",
    )
    expected = {
        (method, horizon, threshold, category)
        for method in METHODS
        for horizon in HORIZONS
        for threshold in COVERAGE_THRESHOLDS
        for category in COVERAGE_CATEGORIES
    }
    if set(cells) != expected:
        raise FigureEvidenceError("Observation coverage lacks exact frozen coverage")
    result = {}
    for group in (
        (method, horizon, threshold)
        for method in METHODS
        for horizon in HORIZONS
        for threshold in COVERAGE_THRESHOLDS
    ):
        fractions = []
        counts = []
        totals = set()
        for category in COVERAGE_CATEGORIES:
            row = cells[(*group, category)]
            fraction = _number(row.get("fraction"), name="coverage fraction")
            count = _integer(row.get("count"), name="coverage count")
            total = _integer(
                row.get("total_gt_entity_stages"), name="coverage total"
            )
            if not 0 <= fraction <= 1:
                raise FigureEvidenceError("coverage fraction must lie in [0, 1]")
            fractions.append(fraction)
            counts.append(count)
            totals.add(total)
            result[(*group, category)] = fraction
        if len(totals) != 1 or sum(counts) != next(iter(totals)):
            raise FigureEvidenceError("coverage counts do not match the group total")
        if not math.isclose(sum(fractions), 1.0, abs_tol=1e-9):
            raise FigureEvidenceError("coverage fractions must sum to one")
    return result


def _normalize_failure_rows(
    rows: Sequence[Mapping[str, object]],
) -> dict[tuple[int, str], float]:
    normalized = []
    for row in rows:
        normalized.append(
            {
                **row,
                "method": str(row.get("method")),
                "horizon": _integer(row.get("horizon"), name="failure horizon"),
                "category": str(row.get("category")),
            }
        )
    cells = _unique_cells(
        normalized,
        keys=("method", "horizon", "category"),
        label="Failure decomposition",
    )
    expected = {
        ("Persist4D", horizon, category)
        for horizon in DECOMPOSITION_HORIZONS
        for category in FAILURE_CATEGORIES
    }
    if set(cells) != expected:
        raise FigureEvidenceError("Failure decomposition lacks exact T4/T5 taxonomy")
    result = {}
    for horizon in DECOMPOSITION_HORIZONS:
        fractions = []
        counts = []
        totals = set()
        for category in FAILURE_CATEGORIES:
            row = cells[("Persist4D", horizon, category)]
            if not str(row.get("operational_definition", "")).strip():
                raise FigureEvidenceError("Failure category lacks operational definition")
            fraction = _number(row.get("fraction"), name="failure fraction")
            count = _integer(row.get("count"), name="failure count")
            total = _integer(row.get("total_failure_events"), name="failure total")
            fractions.append(fraction)
            counts.append(count)
            totals.add(total)
            result[(horizon, category)] = fraction
        if len(totals) != 1 or sum(counts) != next(iter(totals)):
            raise FigureEvidenceError("failure counts do not match the horizon total")
        if not math.isclose(sum(fractions), 1.0, abs_tol=1e-9):
            raise FigureEvidenceError("failure fractions must sum to one")
    return result


def _normalize_oracle_rows(
    rows: Sequence[Mapping[str, object]],
) -> dict[tuple[str, int], float]:
    normalized = []
    for row in rows:
        normalized.append(
            {
                **row,
                "method": str(row.get("method")),
                "horizon": _integer(row.get("horizon"), name="oracle horizon"),
            }
        )
    cells = _unique_cells(
        normalized,
        keys=("method", "horizon"),
        label="Oracle diagnostic",
    )
    expected = {
        (method, horizon)
        for method in ("FullHistory", "Oracle", "Persist4D")
        for horizon in HORIZONS
    }
    if set(cells) != expected:
        raise FigureEvidenceError("Oracle diagnostic lacks exact method/horizon coverage")
    result = {}
    for cell, row in cells.items():
        value = _number(row.get("t_mAP"), name="oracle t-mAP")
        if not 0 <= value <= 1:
            raise FigureEvidenceError("oracle t-mAP must lie in [0, 1]")
        semantics = str(row.get("diagnostic_semantics", ""))
        if cell[0] == "Oracle" and "unmatched candidates retained" not in semantics:
            raise FigureEvidenceError("Oracle diagnostic semantics are incomplete")
        result[cell] = value
    return result


def _normalize_tracker_rows(
    rows: Sequence[Mapping[str, object]],
) -> dict[tuple[str, int], float | None]:
    normalized = []
    for row in rows:
        normalized.append(
            {
                **row,
                "method_id": str(row.get("method_id")),
                "horizon": _integer(row.get("horizon"), name="tracker horizon"),
            }
        )
    cells = _unique_cells(
        normalized,
        keys=("method_id", "horizon"),
        label="Strong-baseline identity scaling",
    )
    expected = {
        (method, horizon) for method in TRACKER_METHODS for horizon in HORIZONS
    }
    if set(cells) != expected:
        raise FigureEvidenceError("Strong-baseline identity scaling lacks exact coverage")
    result = {}
    for cell, row in cells.items():
        if _integer(row.get("sequence_count"), name="tracker sequence count") != 129:
            raise FigureEvidenceError("tracker evidence must contain 129 sequences")
        value = _optional_number(
            row.get("normalized_id_switch_rate"), name="ID-switch rate"
        )
        if cell[1] == 2 and cell[0] != "Persist4D" and value is not None:
            raise FigureEvidenceError(
                "native Full-History and B2 T2 ID-switch rates must be not applicable"
            )
        if (cell[1] > 2 or cell[0] == "Persist4D") and (
            value is None or not 0 <= value <= 1
        ):
            raise FigureEvidenceError("T3-T5 ID-switch rate must lie in [0, 1]")
        result[cell] = value
    return result


def _normalize_adaptation_task_rows(
    rows: Sequence[Mapping[str, object]],
) -> dict[tuple[str, int], tuple[float, float]]:
    selected = [
        {
            **row,
            "method_id": str(row.get("method_id")),
            "horizon": _integer(row.get("horizon"), name="adaptation horizon"),
        }
        for row in rows
        if row.get("method_id") in ADAPTATION_TASK_METHODS
        and row.get("order_id") == "all"
    ]
    cells = _unique_cells(
        selected,
        keys=("method_id", "horizon"),
        label="Horizon-adaptation task scaling",
    )
    expected = {
        (method, horizon)
        for method in ADAPTATION_TASK_METHODS
        for horizon in HORIZONS
    }
    if set(cells) != expected:
        raise FigureEvidenceError(
            "Horizon-adaptation task scaling lacks exact coverage"
        )
    result = {}
    for cell, row in cells.items():
        if _integer(row.get("sequence_count"), name="adaptation sequence count") != 129:
            raise FigureEvidenceError(
                "Horizon-adaptation evidence must contain 129 sequences"
            )
        task_values = (
            _number(row.get("causal_prefix_t_mAP"), name="causal-prefix t-mAP"),
            _number(row.get("causal_prefix_t_REC"), name="causal-prefix t-REC"),
        )
        if any(not 0 <= value <= 1 for value in task_values):
            raise FigureEvidenceError(
                "Horizon-adaptation task metrics must lie in [0, 1]"
            )
        result[cell] = task_values
    return result


def _style() -> dict[str, object]:
    return {
        "font.family": "DejaVu Sans",
        "font.size": 9.0,
        "axes.titlesize": 10.0,
        "axes.labelsize": 9.0,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 8.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#666666",
        "axes.linewidth": 0.8,
        "axes.grid": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "svg.fonttype": "none",
        "svg.hashsalt": "persist4d-reviewer-closure",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }


def _format_axis(axis: plt.Axes) -> None:
    axis.tick_params(length=3, width=0.8, color="#666666")
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.6, zorder=0)


def _draw_iou(axis: plt.Axes, data: Mapping[tuple[str, int, float], float], horizon: int) -> None:
    for method, color, marker, linestyle in (
        ("FullHistory", FULL_COLOR, "o", "-"),
        ("Persist4D", PERSIST_COLOR, "s", "--"),
    ):
        values = [data[(method, horizon, threshold)] for threshold in IOU_THRESHOLDS]
        axis.plot(
            IOU_THRESHOLDS,
            values,
            color=color,
            marker=marker,
            markersize=3.8,
            linewidth=1.8,
            linestyle=linestyle,
            label=DISPLAY[method],
            zorder=3,
        )
    axis.set_title(f"T{horizon}")
    axis.set_xlabel("IoU threshold")
    axis.set_ylabel("Temporal AP")
    axis.set_xlim(0.23, 0.92)
    axis.set_ylim(bottom=0)
    axis.set_xticks((0.25, 0.50, 0.75, 0.90))
    _format_axis(axis)


def _draw_coverage(
    axis: plt.Axes,
    data: Mapping[tuple[str, int, float, str], float],
    horizon: int,
) -> None:
    positions = []
    tick_labels = []
    for threshold_index, threshold in enumerate(COVERAGE_THRESHOLDS):
        for method_index, method in enumerate(METHODS):
            positions.append(threshold_index * 2.4 + method_index * 0.86)
            tick_labels.append("FH" if method == "FullHistory" else "P4D")
    bottoms = [0.0] * len(positions)
    for category in COVERAGE_CATEGORIES:
        values = []
        for threshold in COVERAGE_THRESHOLDS:
            for method in METHODS:
                values.append(data[(method, horizon, threshold, category)])
        bars = axis.bar(
            positions,
            values,
            width=0.72,
            bottom=bottoms,
            color=COVERAGE_COLORS[category],
            edgecolor="#FFFFFF",
            linewidth=0.5,
            label=DISPLAY[category],
            zorder=2,
        )
        for index, bar in enumerate(bars):
            if index % 2 == 1:
                bar.set_hatch("//")
                bar.set_edgecolor("#FFFFFF")
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]
    axis.set_title(f"T{horizon}")
    axis.set_ylabel("GT entity-stage fraction")
    axis.set_ylim(0, 1)
    axis.set_yticks((0, 0.25, 0.50, 0.75, 1.0))
    axis.set_xticks(positions, tick_labels)
    for threshold_index, threshold in enumerate(COVERAGE_THRESHOLDS):
        center = threshold_index * 2.4 + 0.43
        axis.text(
            center,
            -0.16,
            f"IoU {threshold:.2f}",
            ha="center",
            va="top",
            transform=axis.get_xaxis_transform(),
            fontsize=8,
        )
    _format_axis(axis)


def _draw_associable_summary(
    axis: plt.Axes,
    data: Mapping[tuple[str, int, float, str], float],
) -> None:
    for method, color, marker in (
        ("FullHistory", FULL_COLOR, "o"),
        ("Persist4D", PERSIST_COLOR, "s"),
    ):
        for horizon, linestyle in ((4, "-"), (5, "--")):
            values = [
                data[(method, horizon, threshold, "associable")]
                for threshold in COVERAGE_THRESHOLDS
            ]
            axis.plot(
                COVERAGE_THRESHOLDS,
                values,
                color=color,
                marker=marker,
                markersize=4,
                linewidth=1.8,
                linestyle=linestyle,
                label=f"{'FH' if method == 'FullHistory' else 'P4D'} T{horizon}",
                zorder=3,
            )
    axis.set_title("Associable observation coverage")
    axis.set_xlabel("IoU threshold")
    axis.set_ylabel("GT entity-stage fraction")
    axis.set_xlim(0.20, 0.80)
    axis.set_ylim(0, 0.75)
    axis.set_xticks(COVERAGE_THRESHOLDS)
    axis.legend(frameon=False, ncol=2, loc="lower left")
    _format_axis(axis)


def _draw_failures(axis: plt.Axes, data: Mapping[tuple[int, str], float]) -> None:
    for horizon_index, horizon in enumerate(DECOMPOSITION_HORIZONS):
        left = 0.0
        for category in FAILURE_CATEGORIES:
            value = data[(horizon, category)]
            axis.barh(
                horizon_index,
                value,
                left=left,
                height=0.56,
                color=FAILURE_COLORS[category],
                edgecolor="#FFFFFF",
                linewidth=0.5,
                hatch="xx" if category == "unknown_unresolved" else None,
                label=DISPLAY[category] if horizon_index == 0 else None,
                zorder=2,
            )
            if value >= 0.075:
                axis.text(
                    left + value / 2,
                    horizon_index,
                    f"{100 * value:.0f}%",
                    ha="center",
                    va="center",
                    color="white" if category == "unknown_unresolved" else "#222222",
                    fontsize=7.5,
                    fontweight="bold",
                    zorder=3,
                )
            left += value
    axis.set_yticks((0, 1), ("T4", "T5"))
    axis.invert_yaxis()
    axis.set_xlim(0, 1)
    axis.set_xticks((0, 0.25, 0.50, 0.75, 1.0), ("0", "25", "50", "75", "100"))
    axis.set_xlabel("Failure events (%)")
    _format_axis(axis)


def _draw_oracle(axis: plt.Axes, data: Mapping[tuple[str, int], float]) -> None:
    x_positions = list(range(len(HORIZONS)))
    width = 0.24
    for offset, (method, color, hatch) in enumerate(
        (
            ("FullHistory", FULL_COLOR, None),
            ("Persist4D", PERSIST_COLOR, "//"),
            ("Oracle", ORACLE_COLOR, "xx"),
        )
    ):
        axis.bar(
            [value + (offset - 1) * width for value in x_positions],
            [data[(method, horizon)] for horizon in HORIZONS],
            width=width,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            hatch=hatch,
            label=DISPLAY[method],
            zorder=2,
        )
    axis.set_xticks(x_positions, [f"T{horizon}" for horizon in HORIZONS])
    axis.set_ylabel("Temporal AP")
    axis.set_ylim(bottom=0)
    _format_axis(axis)


def _source_description(*names: str) -> str:
    return "Source: " + ", ".join(names) + ". Measured reviewer-closure artifacts."


def _publish_exact(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file():
            raise FileExistsError(f"figure output is not a regular file: {path}")
        if path.read_bytes() == payload:
            return path
        raise FileExistsError(f"figure output contains different content: {path}")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return path


def _svg_metadata(payload: bytes, *, title: str, description: str) -> bytes:
    raw_svg = payload.decode("utf-8")
    svg = "\n".join(line.rstrip() for line in raw_svg.splitlines())
    if raw_svg.endswith("\n"):
        svg += "\n"
    start = svg.find("<svg ")
    end = svg.find(">", start)
    if start < 0 or end < 0:
        raise FigureEvidenceError("Matplotlib did not produce a valid SVG root")
    metadata = (
        f"\n <title>{html.escape(title)}</title>"
        f"\n <desc>{html.escape(description)}</desc>"
    )
    return (svg[: end + 1] + metadata + svg[end + 1 :]).encode("utf-8")


def _figure_payload(
    figure: plt.Figure,
    *,
    suffix: str,
    title: str,
    description: str,
) -> bytes:
    buffer = io.BytesIO()
    common = {"bbox_inches": "tight", "pad_inches": 0.08}
    if suffix == ".svg":
        figure.savefig(buffer, format="svg", metadata={"Date": None}, **common)
        return _svg_metadata(buffer.getvalue(), title=title, description=description)
    if suffix == ".pdf":
        figure.savefig(
            buffer,
            format="pdf",
            metadata={
                "Title": title,
                "Subject": description,
                "Creator": "Persist4D reviewer-closure figure renderer",
                "CreationDate": None,
                "ModDate": None,
            },
            **common,
        )
        return buffer.getvalue()
    if suffix == ".png":
        figure.savefig(
            buffer,
            format="png",
            dpi=300,
            metadata={"Title": title, "Description": description},
            **common,
        )
        return buffer.getvalue()
    raise AssertionError(f"unsupported figure suffix: {suffix}")


def _publish_figure(
    figure: plt.Figure,
    output_directory: Path,
    stem: str,
    *,
    title: str,
    description: str,
) -> tuple[Path, ...]:
    paths = []
    for suffix in (".svg", ".pdf", ".png"):
        payload = _figure_payload(
            figure,
            suffix=suffix,
            title=title,
            description=description,
        )
        paths.append(_publish_exact(output_directory / f"{stem}{suffix}", payload))
    plt.close(figure)
    return tuple(paths)


def _panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.13,
        1.07,
        label,
        transform=axis.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
    )


def render_phase_iii_figures(
    iou_rows: Sequence[Mapping[str, object]],
    coverage_rows: Sequence[Mapping[str, object]],
    failure_rows: Sequence[Mapping[str, object]],
    oracle_rows: Sequence[Mapping[str, object]],
    output_directory: Path,
) -> tuple[Path, ...]:
    """Render the four decomposition panels and their combined paper figure."""

    iou = _normalize_iou_rows(iou_rows)
    coverage = _normalize_coverage_rows(coverage_rows)
    failures = _normalize_failure_rows(failure_rows)
    oracle = _normalize_oracle_rows(oracle_rows)
    output_directory = Path(output_directory)
    results: list[Path] = []

    with plt.rc_context(_style()):
        figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.25), sharey=True, constrained_layout=True)
        for axis, horizon in zip(axes, DECOMPOSITION_HORIZONS):
            _draw_iou(axis, iou, horizon)
        axes[0].legend(frameon=False, loc="upper right")
        results.extend(
            _publish_figure(
                figure,
                output_directory,
                "iou_threshold_curve",
                title="Temporal AP across IoU thresholds",
                description=_source_description("tmap_iou_sweep.csv"),
            )
        )

        figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.55), sharey=True, constrained_layout=True)
        for axis, horizon in zip(axes, DECOMPOSITION_HORIZONS):
            _draw_coverage(axis, coverage, horizon)
        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(handles, labels, frameon=False, ncol=4, loc="outside upper center")
        results.extend(
            _publish_figure(
                figure,
                output_directory,
                "observation_coverage",
                title="Observation coverage at three IoU thresholds",
                description=_source_description("observation_coverage.csv"),
            )
        )

        figure, axis = plt.subplots(figsize=(7.2, 3.35), constrained_layout=True)
        _draw_failures(axis, failures)
        handles, labels = axis.get_legend_handles_labels()
        figure.legend(handles, labels, frameon=False, ncol=4, loc="outside upper center")
        results.extend(
            _publish_figure(
                figure,
                output_directory,
                "failure_decomposition",
                title="Persist4D failure composition at T4 and T5",
                description=_source_description("failure_decomposition.csv"),
            )
        )

        figure, axis = plt.subplots(figsize=(7.2, 3.35), constrained_layout=True)
        _draw_oracle(axis, oracle)
        axis.legend(frameon=False, loc="upper right")
        axis.text(
            0.01,
            0.98,
            "P6-A relabels matched predictions only; unmatched candidates remain.",
            transform=axis.transAxes,
            va="top",
            fontsize=8,
            color="#666666",
        )
        results.extend(
            _publish_figure(
                figure,
                output_directory,
                "oracle_association_gain",
                title="P6-A GT-ID-only association diagnostic",
                description=_source_description("oracle_association_results.csv"),
            )
        )

        figure = plt.figure(figsize=(11.4, 7.2), constrained_layout=True)
        grid = figure.add_gridspec(2, 2, height_ratios=(1.05, 0.95))
        iou_grid = grid[0, 0].subgridspec(1, 2, wspace=0.12)
        iou_axes = [figure.add_subplot(iou_grid[0, index]) for index in range(2)]
        for axis, horizon in zip(iou_axes, DECOMPOSITION_HORIZONS):
            _draw_iou(axis, iou, horizon)
        iou_axes[1].set_ylabel("")
        iou_axes[0].legend(frameon=False, loc="upper right")
        _panel_label(iou_axes[0], "(a)")

        coverage_axis = figure.add_subplot(grid[0, 1])
        _draw_associable_summary(coverage_axis, coverage)
        _panel_label(coverage_axis, "(b)")

        failure_axis = figure.add_subplot(grid[1, 0])
        _draw_failures(failure_axis, failures)
        handles, labels = failure_axis.get_legend_handles_labels()
        failure_axis.legend(
            handles,
            labels,
            frameon=False,
            ncol=2,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.32),
        )
        _panel_label(failure_axis, "(c)")

        oracle_axis = figure.add_subplot(grid[1, 1])
        _draw_oracle(oracle_axis, oracle)
        oracle_axis.legend(frameon=False, loc="upper right")
        oracle_axis.text(
            0.01,
            0.98,
            "GT-ID-only; unmatched candidates retained",
            transform=oracle_axis.transAxes,
            va="top",
            fontsize=8,
            color="#666666",
        )
        _panel_label(oracle_axis, "(d)")
        results.extend(
            _publish_figure(
                figure,
                output_directory,
                "performance_decomposition",
                title="Why similar temporal AP emerges from different failure modes",
                description=_source_description(
                    "tmap_iou_sweep.csv",
                    "observation_coverage.csv",
                    "failure_decomposition.csv",
                    "oracle_association_results.csv",
                ),
            )
        )
    return tuple(results)


def render_strong_baseline_identity_scaling(
    rows: Sequence[Mapping[str, object]], output_directory: Path
) -> tuple[Path, ...]:
    """Render native, B2, and persistent identity scaling over T2-T5."""

    data = _normalize_tracker_rows(rows)
    colors = {
        "FullHistoryNative": FULL_COLOR,
        "B2": TRACKER_COLOR,
        "Persist4D": PERSIST_COLOR,
    }
    markers = {"FullHistoryNative": "o", "B2": "^", "Persist4D": "s"}
    styles = {"FullHistoryNative": ":", "B2": "-.", "Persist4D": "-"}
    with plt.rc_context(_style()):
        figure, axis = plt.subplots(figsize=(7.2, 3.65), constrained_layout=True)
        for method in TRACKER_METHODS:
            horizons = tuple(
                horizon for horizon in HORIZONS if data[(method, horizon)] is not None
            )
            values = [data[(method, horizon)] for horizon in horizons]
            axis.plot(
                horizons,
                values,
                color=colors[method],
                marker=markers[method],
                linestyle=styles[method],
                linewidth=1.9,
                markersize=5,
                label=DISPLAY[method],
                zorder=3,
            )
        axis.set_xticks(HORIZONS, [f"T{horizon}" for horizon in HORIZONS])
        axis.set_xlim(1.85, 5.18)
        axis.set_ylim(0, 1.05)
        axis.set_ylabel("Normalized ID-switch rate")
        axis.set_xlabel("Temporal horizon")
        axis.text(
            0.02,
            0.035,
            "T2 initialization: no ID-transition rate for Full-History/B2",
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            color="#666666",
            fontsize=8,
        )
        axis.legend(frameon=False, loc="center right")
        _format_axis(axis)
        return _publish_figure(
            figure,
            Path(output_directory),
            "strong_baseline_identity_scaling",
            title="Strong-baseline deployment identity scaling",
            description=_source_description("full_history_tracker_aggregate.csv"),
        )


def render_horizon_adaptation_task_scaling(
    rows: Sequence[Mapping[str, object]], output_directory: Path
) -> tuple[Path, ...]:
    """Render pooled task scaling for the adapted strong alternative."""

    data = _normalize_adaptation_task_rows(rows)
    colors = {
        "FullHistoryFrozenB2": FULL_COLOR,
        "FullHistoryAdaptedB2": TRACKER_COLOR,
        "Persist4D": PERSIST_COLOR,
    }
    markers = {
        "FullHistoryFrozenB2": "o",
        "FullHistoryAdaptedB2": "^",
        "Persist4D": "s",
    }
    styles = {
        "FullHistoryFrozenB2": ":",
        "FullHistoryAdaptedB2": "-.",
        "Persist4D": "-",
    }
    with plt.rc_context(_style()):
        figure, axes = plt.subplots(
            1,
            2,
            figsize=(7.2, 3.45),
            constrained_layout=True,
        )
        for metric_index, (axis, label) in enumerate(
            zip(axes, ("Causal-prefix t-mAP", "Causal-prefix t-REC"))
        ):
            for method in ADAPTATION_TASK_METHODS:
                axis.plot(
                    HORIZONS,
                    [data[(method, horizon)][metric_index] for horizon in HORIZONS],
                    color=colors[method],
                    marker=markers[method],
                    linestyle=styles[method],
                    linewidth=1.9,
                    markersize=5,
                    label=DISPLAY[method],
                    zorder=3,
                )
            axis.set_xticks(HORIZONS, [f"T{horizon}" for horizon in HORIZONS])
            axis.set_xlim(1.85, 5.15)
            axis.set_ylim(bottom=0)
            axis.set_xlabel("Temporal horizon")
            axis.set_ylabel(label)
            _format_axis(axis)
        axes[0].legend(frameon=False, loc="upper right")
        return _publish_figure(
            figure,
            Path(output_directory),
            "horizon_adaptation_task_scaling",
            title="Task scaling under T2-to-T3 horizon adaptation",
            description=_source_description(
                "rescene_horizon_adaptation_results.csv"
            ),
        )


def _read_csv(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts/reviewer_closure"),
    )
    arguments = parser.parse_args(argv)
    root = arguments.artifact_root
    output = root / "figures"
    render_phase_iii_figures(
        _read_csv(root / "tmap_iou_sweep.csv"),
        _read_csv(root / "observation_coverage.csv"),
        _read_csv(root / "failure_decomposition.csv"),
        _read_csv(root / "oracle_association_results.csv"),
        output,
    )
    tracker_rows = [
        row
        for row in _read_csv(root / "full_history_tracker_aggregate.csv")
        if row.get("method_id") in TRACKER_METHODS
    ]
    render_strong_baseline_identity_scaling(tracker_rows, output)
    render_horizon_adaptation_task_scaling(
        _read_csv(root / "rescene_horizon_adaptation_results.csv"),
        output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FigureEvidenceError",
    "render_horizon_adaptation_task_scaling",
    "render_phase_iii_figures",
    "render_strong_baseline_identity_scaling",
]
