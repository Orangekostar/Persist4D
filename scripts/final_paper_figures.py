"""Generate the two paper-facing figures for the frozen evidence package."""

from __future__ import annotations

import argparse
import csv
import io
from collections.abc import Iterable, Mapping
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
GRAY = "#666666"
LIGHT_GRAY = "#E6E6E6"
HORIZONS = (2, 3, 4, 5)


class FinalFigureError(ValueError):
    """Raised when figure inputs lack a frozen method/horizon cell."""


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
        raise FinalFigureError(
            f"expected one row for {conditions}, found {len(matches)}"
        )
    return matches[0]


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
            "legend.fontsize": 7.2,
            "lines.linewidth": 1.7,
            "lines.markersize": 4.8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "svg.fonttype": "none",
            "svg.hashsalt": "persist4d-final-paper",
            "pdf.fonttype": 42,
        }
    )


def _timeline_box(axis: plt.Axes, x: float, y: float, label: str, color: str) -> None:
    axis.add_patch(
        Rectangle(
            (x - 0.06, y - 0.075),
            0.12,
            0.15,
            facecolor=color,
            edgecolor="#333333",
            linewidth=0.7,
        )
    )
    axis.text(x, y, label, ha="center", va="center", color="white", fontsize=7.5)


def _figure_1(compute_rows: list[dict[str, str]]) -> plt.Figure:
    figure = plt.figure(figsize=(7.2, 4.0))
    diagram = figure.add_axes((0.05, 0.14, 0.58, 0.78))
    scaling = figure.add_axes((0.69, 0.20, 0.28, 0.64))
    diagram.set_axis_off()
    diagram.set_xlim(0, 1)
    diagram.set_ylim(0, 1)
    diagram.text(
        0,
        0.98,
        "Finite Temporal Context Is Not Persistent Entity State",
        ha="left",
        va="top",
        fontsize=10.5,
        fontweight="bold",
    )
    visit_x = (0.10, 0.29, 0.48, 0.67, 0.86)
    labels = ("V1", "V2", "gap", "gap", "V5")
    for x, label in zip(visit_x, labels, strict=True):
        color = BLUE if label != "gap" else "#BDBDBD"
        _timeline_box(diagram, x, 0.76, label, color)
    diagram.add_patch(
        FancyArrowPatch(
            (0.04, 0.76),
            (0.94, 0.76),
            arrowstyle="->",
            mutation_scale=10,
            linewidth=0.8,
            color="#333333",
            zorder=0,
        )
    )

    diagram.text(0.01, 0.54, "Expanding context", ha="left", fontweight="bold")
    diagram.text(0.01, 0.47, "reprocess [S1, ..., St]", ha="left", color=GRAY)
    for x in (0.42, 0.52, 0.62, 0.72, 0.82):
        diagram.add_patch(
            Rectangle((x, 0.43), 0.075, 0.09, facecolor=LIGHT_GRAY, edgecolor=GRAY)
        )
    diagram.text(0.86, 0.475, "T grows", ha="center", va="center", fontsize=7.3)

    diagram.text(0.01, 0.27, "Persist4D", ha="left", fontweight="bold")
    diagram.text(0.01, 0.20, "bounded [S(t-1), St]", ha="left", color=GRAY)
    diagram.add_patch(
        Rectangle((0.42, 0.16), 0.15, 0.09, facecolor="#D9F0E5", edgecolor=GREEN)
    )
    diagram.text(0.495, 0.205, "local", ha="center", va="center", color=GREEN)
    diagram.add_patch(
        FancyArrowPatch(
            (0.60, 0.205),
            (0.89, 0.205),
            arrowstyle="->",
            mutation_scale=10,
            linewidth=2.0,
            color=ORANGE,
        )
    )
    diagram.text(0.745, 0.27, "dormant entity ID persists", ha="center", color=ORANGE)

    method_specs = (
        ("FullHistoryAdapted", "Adapted full history", BLUE, "o"),
        ("Persist4D", "Persist4D", ORANGE, "s"),
    )
    for method_id, label, color, marker in method_specs:
        values = [
            float(
                _one(compute_rows, method_id=method_id, horizon=t)["median_latency_ms"]
            )
            for t in HORIZONS
        ]
        scaling.plot(HORIZONS, values, marker=marker, color=color, label=label)
    scaling.set_title("Measured update cost", loc="left")
    scaling.set_xticks(HORIZONS, [f"T{t}" for t in HORIZONS])
    scaling.set_xlabel("Temporal horizon")
    scaling.set_ylabel("Median latency (ms)")
    scaling.grid(axis="y", color=LIGHT_GRAY, linewidth=0.6)
    scaling.legend(frameon=False, loc="best")
    figure.text(
        0.05,
        0.03,
        "Conceptual distinction; latency from rescan_horizon_compute.csv (6 profile clusters).",
        ha="left",
        va="bottom",
        fontsize=6.6,
        color=GRAY,
    )
    return figure


def _figure_2(
    tracker_rows: list[dict[str, str]],
    adapted_rows: list[dict[str, str]],
    compute_rows: list[dict[str, str]],
) -> plt.Figure:
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), sharex=True)
    method_specs = (
        (
            "B2",
            "Full-history + feature-class",
            BLUE,
            "o",
            tracker_rows,
            "FullHistoryFrozen",
        ),
        (
            "FullHistoryAdaptedB2",
            "Adapted full-history + feature-class",
            GREEN,
            "^",
            adapted_rows,
            "FullHistoryAdapted",
        ),
        ("Persist4D", "Persist4D", ORANGE, "s", adapted_rows, "Persist4D"),
    )
    panels = (
        ("causal_prefix_t_mAP", "(a) Causal-prefix t-mAP", False),
        ("gap_recovery_recall", "(b) Gap-recovery recall", False),
        ("median_latency_ms", "(c) Median update latency (ms)", True),
        ("peak_allocated_mib", "(d) Peak allocated VRAM (MiB)", True),
    )
    for axis, (metric, title, is_compute) in zip(axes.flat, panels, strict=True):
        for method_id, label, color, marker, result_rows, compute_id in method_specs:
            source = compute_rows if is_compute else result_rows
            source_id = compute_id if is_compute else method_id
            values = []
            xs = []
            for horizon in HORIZONS:
                row = _one(source, method_id=source_id, horizon=horizon)
                value = row.get(metric, "")
                if value == "":
                    continue
                xs.append(horizon)
                values.append(float(value))
            axis.plot(xs, values, marker=marker, color=color, label=label)
        axis.set_title(title, loc="left")
        axis.set_xticks(HORIZONS, [f"T{t}" for t in HORIZONS])
        axis.grid(axis="y", color=LIGHT_GRAY, linewidth=0.6)
        if not is_compute:
            axis.set_ylim(bottom=0)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    figure.text(
        0.01,
        0.008,
        (
            "Task/identity: 129 common-prefix sequence/order scopes, 6 scene clusters. "
            "Full-history compute excludes lightweight tracker overhead."
        ),
        ha="left",
        va="bottom",
        fontsize=6.5,
        color=GRAY,
    )
    figure.tight_layout(rect=(0, 0.035, 1, 0.91))
    return figure


def _export(figure: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension, kwargs in (
        ("svg", {"format": "svg", "metadata": {"Date": None}}),
        ("pdf", {"format": "pdf", "metadata": {"CreationDate": None}}),
        ("png", {"format": "png", "metadata": {"Software": "Persist4D"}, "dpi": 220}),
    ):
        buffer = io.BytesIO()
        figure.savefig(buffer, bbox_inches="tight", **kwargs)
        payload = buffer.getvalue()
        if extension == "svg":
            payload = b"\n".join(line.rstrip() for line in payload.splitlines()) + b"\n"
        (output_dir / f"{stem}.{extension}").write_bytes(payload)
    plt.close(figure)


def build(root: Path, output_dir: Path) -> None:
    closure = root / "artifacts/reviewer_closure"
    tracker_rows = _read_csv(closure / "full_history_tracker_aggregate.csv")
    adapted_rows = _read_csv(closure / "rescene_horizon_adaptation_results.csv")
    compute_rows = _read_csv(closure / "rescene_horizon_compute.csv")
    _style()
    _export(
        _figure_1(compute_rows),
        output_dir,
        "main_figure_1_finite_context_vs_persistent_state",
    )
    _export(
        _figure_2(tracker_rows, adapted_rows, compute_rows),
        output_dir,
        "main_figure_2_accuracy_identity_compute_scaling",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/final_evidence/figures"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    build(root, output_dir)


if __name__ == "__main__":
    main()
