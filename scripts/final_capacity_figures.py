"""Publication figures for the frozen capacity-sensitivity evidence."""

from __future__ import annotations

import argparse
import csv
import io
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CAPACITIES = (64, 100, 128, 160, 200)
HORIZONS = (2, 3, 4, 5)
HORIZON_COLORS = {
    2: "#0072B2",
    3: "#E69F00",
    4: "#009E73",
    5: "#CC79A7",
}
HORIZON_STYLES = {2: "-", 3: "--", 4: "-.", 5: ":"}
MAIN_COLOR = "#0072B2"
MAX_COLOR = "#D55E00"
NEUTRAL = "#666666"


class CapacityFigureError(ValueError):
    """Raised when capacity figure inputs lack frozen result coverage."""


def _number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or value is None or value == "":
        raise CapacityFigureError(f"{name} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise CapacityFigureError(f"{name} must be finite") from error
    if not math.isfinite(result):
        raise CapacityFigureError(f"{name} must be finite")
    return result


def _optional_number(value: object, *, name: str) -> float | None:
    if value is None or value == "":
        return None
    return _number(value, name=name)


def _normalized_cells(
    rows: Sequence[Mapping[str, object]], *, expected_sequence_count: int
) -> dict[tuple[int, int], dict[str, object]]:
    if not rows:
        raise CapacityFigureError("capacity figure rows must not be empty")
    cells = {}
    for raw in rows:
        capacity = int(_number(raw.get("capacity"), name="capacity"))
        horizon = int(_number(raw.get("horizon"), name="horizon"))
        sequence_count = int(_number(raw.get("sequence_count"), name="sequence count"))
        if sequence_count != expected_sequence_count:
            raise CapacityFigureError("capacity figure sequence coverage differs")
        cell = (capacity, horizon)
        if cell in cells:
            raise CapacityFigureError("capacity figure contains duplicate cells")
        cells[cell] = dict(raw)
    expected = {(capacity, horizon) for capacity in CAPACITIES for horizon in HORIZONS}
    if set(cells) != expected:
        raise CapacityFigureError("capacity figure lacks exact K by T coverage")
    return cells


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9.5,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
            "lines.linewidth": 1.6,
            "lines.markersize": 4.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "svg.fonttype": "none",
            "svg.hashsalt": "persist4d-final-capacity",
            "pdf.fonttype": 42,
        }
    )


def _source_note(figure: plt.Figure, *, sequence_count: int) -> None:
    figure.text(
        0.01,
        0.008,
        (
            "Source: capacity_aggregate.csv; frozen common-prefix replay, "
            f"n={sequence_count} sequences / 6 scene clusters."
        ),
        ha="left",
        va="bottom",
        fontsize=6.6,
        color=NEUTRAL,
    )


def _figure_c1(
    cells: Mapping[tuple[int, int], Mapping[str, object]], *, sequence_count: int
) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(6.6, 3.5))
    horizons = list(HORIZONS)
    main_rows = [cells[(100, horizon)] for horizon in horizons]
    q25 = [
        _number(row["peak_occupied_slots_q25"], name="occupancy q25")
        for row in main_rows
    ]
    median = [
        _number(row["peak_occupied_slots_median"], name="occupancy median")
        for row in main_rows
    ]
    q75 = [
        _number(row["peak_occupied_slots_q75"], name="occupancy q75")
        for row in main_rows
    ]
    maxima = [
        _number(row["peak_occupied_slots_max"], name="occupancy maximum")
        for row in main_rows
    ]
    axis.fill_between(
        horizons,
        q25,
        q75,
        color=MAIN_COLOR,
        alpha=0.20,
        label="IQR",
        zorder=2,
    )
    axis.plot(
        horizons,
        median,
        color=MAIN_COLOR,
        marker="o",
        label="Median",
        zorder=3,
    )
    axis.plot(
        horizons,
        maxima,
        color=MAX_COLOR,
        marker="s",
        linestyle="--",
        label="Maximum",
        zorder=3,
    )
    for index, capacity in enumerate(CAPACITIES):
        axis.axhline(
            capacity,
            color="#A6A6A6" if capacity != 100 else "#222222",
            linewidth=0.75 if capacity != 100 else 1.0,
            linestyle=(0, (2 + index, 2)),
            zorder=1,
        )
        axis.text(
            5.08,
            capacity,
            f"K={capacity}",
            va="center",
            fontsize=6.8,
            color="#666666" if capacity != 100 else "#222222",
        )
    axis.annotate(
        f"Observed maximum = {max(maxima):.0f}",
        xy=(horizons[maxima.index(max(maxima))], max(maxima)),
        xytext=(3.45, 47),
        arrowprops={"arrowstyle": "-", "color": MAX_COLOR, "linewidth": 0.8},
        color=MAX_COLOR,
        fontsize=7.5,
    )
    axis.set_xlim(1.8, 5.55)
    axis.set_ylim(0, max(CAPACITIES) * 1.07)
    axis.set_xticks(horizons, [f"T{horizon}" for horizon in horizons])
    axis.set_xlabel("Temporal horizon")
    axis.set_ylabel("Peak occupied slots")
    axis.set_title("Capacity headroom under frozen local observations", loc="left")
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.5, alpha=0.8)
    axis.legend(frameon=False, loc="upper left", ncol=3)
    _source_note(figure, sequence_count=sequence_count)
    figure.tight_layout(rect=(0, 0.035, 1, 1))
    return figure


def _metric_limits(values: Sequence[float]) -> tuple[float, float]:
    lower = min(values)
    upper = max(values)
    padding = max(0.015, (upper - lower) * 0.16)
    return max(0.0, lower - padding), min(1.0, upper + padding)


def _figure_c2(
    cells: Mapping[tuple[int, int], Mapping[str, object]], *, sequence_count: int
) -> plt.Figure:
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), sharex=True)
    panels = (
        ("causal_prefix_t_mAP", "(a) Causal-prefix t-mAP"),
        ("causal_prefix_t_REC", "(b) Causal-prefix t-REC"),
        ("normalized_id_switch_rate", "(c) Normalized ID-switch rate"),
        ("gap_recovery_recall", "(d) Gap-recovery recall"),
    )
    for axis, (metric, title) in zip(axes.flat, panels, strict=True):
        plotted_values = []
        for horizon in HORIZONS:
            values = [
                _optional_number(cells[(capacity, horizon)].get(metric), name=metric)
                for capacity in CAPACITIES
            ]
            if all(value is None for value in values):
                continue
            if any(value is None for value in values):
                raise CapacityFigureError(
                    f"{metric} is partially missing at T{horizon}"
                )
            numeric = [float(value) for value in values if value is not None]
            plotted_values.extend(numeric)
            axis.plot(
                CAPACITIES,
                numeric,
                color=HORIZON_COLORS[horizon],
                linestyle=HORIZON_STYLES[horizon],
                marker="o" if horizon in (2, 4) else "s",
                label=f"T{horizon}",
            )
        axis.axvline(100, color="#666666", linewidth=0.8, linestyle=(0, (2, 2)))
        axis.set_title(title, loc="left")
        axis.set_ylabel("Rate")
        axis.set_ylim(*_metric_limits(plotted_values))
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.5)
        axis.set_xticks(CAPACITIES)
        if metric == "gap_recovery_recall":
            axis.text(
                0.02,
                0.04,
                "T2 unavailable: no gap opportunity",
                transform=axis.transAxes,
                fontsize=6.8,
                color=NEUTRAL,
            )
    axes[1, 0].set_xlabel("Persistent-state capacity K")
    axes[1, 1].set_xlabel("Persistent-state capacity K")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        frameon=False,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
    )
    _source_note(figure, sequence_count=sequence_count)
    figure.tight_layout(rect=(0, 0.045, 1, 0.94))
    return figure


def _figure_c3(
    cells: Mapping[tuple[int, int], Mapping[str, object]], *, sequence_count: int
) -> plt.Figure:
    state_bytes = []
    for capacity in CAPACITIES:
        values = {
            int(_number(cells[(capacity, horizon)]["state_bytes"], name="state bytes"))
            for horizon in HORIZONS
        }
        if len(values) != 1:
            raise CapacityFigureError("fixed-K state bytes differ across horizons")
        state_bytes.append(next(iter(values)))
    kib = [value / 1024 for value in state_bytes]
    figure, axis = plt.subplots(figsize=(5.6, 3.4))
    axis.plot(CAPACITIES, kib, color=MAIN_COLOR, marker="o")
    axis.scatter([100], [kib[1]], color=MAX_COLOR, marker="s", zorder=4)
    for capacity, value in zip(CAPACITIES, kib, strict=True):
        axis.annotate(
            f"{value:.1f}",
            (capacity, value),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=7,
        )
    axis.set_xticks(CAPACITIES)
    axis.set_xlabel("Persistent-state capacity K")
    axis.set_ylabel("Allocated state tensors (KiB)")
    axis.set_title("Persistent-state storage scales linearly with K", loc="left")
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.5)
    axis.text(
        0.01,
        0.98,
        "Tensor state only; excludes model weights, masks, allocator overhead, and VRAM.",
        transform=axis.transAxes,
        va="top",
        fontsize=7,
        color=NEUTRAL,
    )
    _source_note(figure, sequence_count=sequence_count)
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    return figure


def _export(figure: plt.Figure, *, stem: str) -> dict[str, bytes]:
    outputs = {}
    for suffix in ("svg", "pdf", "png"):
        buffer = io.BytesIO()
        kwargs: dict[str, object] = {
            "format": suffix,
            "bbox_inches": "tight",
            "pad_inches": 0.04,
        }
        if suffix == "svg":
            kwargs["metadata"] = {"Date": None}
        elif suffix == "pdf":
            kwargs["metadata"] = {"CreationDate": None, "ModDate": None}
        else:
            kwargs["dpi"] = 300
            kwargs["metadata"] = {"Software": "Persist4D final capacity renderer"}
        figure.savefig(buffer, **kwargs)
        payload = buffer.getvalue()
        if suffix == "svg":
            payload = b"\n".join(line.rstrip() for line in payload.splitlines()) + b"\n"
        outputs[f"{stem}.{suffix}"] = payload
    plt.close(figure)
    return outputs


def render_capacity_figures(
    aggregate_rows: Sequence[Mapping[str, object]],
    *,
    expected_sequence_count: int = 129,
) -> dict[str, bytes]:
    _style()
    cells = _normalized_cells(
        aggregate_rows, expected_sequence_count=expected_sequence_count
    )
    figures = (
        (
            "figure_c1_occupancy_vs_horizon",
            _figure_c1(cells, sequence_count=expected_sequence_count),
        ),
        (
            "figure_c2_performance_vs_capacity",
            _figure_c2(cells, sequence_count=expected_sequence_count),
        ),
        (
            "figure_c3_state_bytes_vs_capacity",
            _figure_c3(cells, sequence_count=expected_sequence_count),
        ),
    )
    outputs = {}
    for stem, figure in figures:
        outputs.update(_export(figure, stem=stem))
    return outputs


def _publish_exact(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_file() and not path.is_symlink() and path.read_bytes() == payload:
            return
        raise FileExistsError(f"figure exists with different content: {path}")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
            temporary = Path(stream.name)
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> int:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aggregate",
        default=repository / "artifacts/final_evidence/capacity/capacity_aggregate.csv",
        type=Path,
    )
    parser.add_argument(
        "--output-directory",
        default=repository / "artifacts/final_evidence/figures",
        type=Path,
    )
    args = parser.parse_args()
    with args.aggregate.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    outputs = render_capacity_figures(rows)
    for filename, payload in outputs.items():
        _publish_exact(args.output_directory / filename, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
