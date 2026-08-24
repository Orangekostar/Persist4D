"""Audit T2 population and protocol shifts without changing metric semantics."""

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
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ORDERS = ("canonical", "reverse", "sha256_seed45")


class ProtocolShiftError(ValueError):
    """Raised when frozen protocol-shift inputs are incomplete or inconsistent."""


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProtocolShiftError(f"{name} must be a mapping")
    return value


def _sequence(value: object, *, name: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ProtocolShiftError(f"{name} must be a sequence")
    return value


def _string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ProtocolShiftError(f"{name} must be a nonempty string")
    return value


def _float(value: object, *, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ProtocolShiftError(f"{name} must be numeric") from error
    if not math.isfinite(number):
        raise ProtocolShiftError(f"{name} must be finite")
    return number


def _integer(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise ProtocolShiftError(f"{name} must be an integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as error:
        raise ProtocolShiftError(f"{name} must be an integer") from error
    return number


def build_population_rows(
    protocol_manifest: Mapping[str, object],
    t2_database: Mapping[str, object],
    *,
    expected_master_count: int = 43,
) -> list[dict[str, object]]:
    root = _mapping(protocol_manifest, name="protocol manifest")
    masters = _sequence(root.get("masters"), name="protocol masters")
    if len(masters) != expected_master_count:
        raise ProtocolShiftError(
            f"protocol master count differs from {expected_master_count}"
        )
    database = _mapping(t2_database, name="T2 sequence database")
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for index, raw_master in enumerate(masters):
        master = _mapping(raw_master, name=f"master[{index}]")
        master_id = _string(
            master.get("master_sequence_id"), name=f"master[{index}].sequence_id"
        )
        reference = _string(
            master.get("reference_scene_id"), name=f"master[{index}].reference"
        )
        orders = _mapping(master.get("orders"), name=f"master[{index}].orders")
        canonical = _mapping(
            orders.get("canonical"), name=f"master[{index}].canonical"
        )
        prefixes = _mapping(
            canonical.get("prefixes"), name=f"master[{index}].canonical.prefixes"
        )
        prefix = _mapping(prefixes.get("2"), name=f"master[{index}].canonical.T2")
        sequence_id = _string(
            prefix.get("sequence_id"), name=f"master[{index}].canonical.T2.sequence"
        )
        scan_ids = tuple(
            _string(value, name=f"master[{index}].canonical.T2.scan_id")
            for value in _sequence(
                prefix.get("scan_ids"), name=f"master[{index}].canonical.T2.scan_ids"
            )
        )
        if len(scan_ids) != 2 or sequence_id != "-".join(scan_ids):
            raise ProtocolShiftError("canonical T2 sequence and ordered scan IDs differ")
        if sequence_id in seen:
            raise ProtocolShiftError("canonical T2 pairs are not unique")
        seen.add(sequence_id)

        present = sequence_id in database
        record = (
            _mapping(database[sequence_id], name=f"T2 database[{sequence_id}]")
            if present
            else {}
        )
        split = record.get("type", "") if present else ""
        filepath = record.get("filepath", "") if present else ""
        supervised = bool(
            present
            and split == "validation"
            and isinstance(filepath, str)
            and filepath not in {"", "None"}
        )
        sub_scenes = record.get("sub_scenes", ()) if present else ()
        if isinstance(sub_scenes, Sequence) and not isinstance(
            sub_scenes, (str, bytes)
        ):
            serialized_sub_scenes = ";".join(str(value) for value in sub_scenes)
        else:
            serialized_sub_scenes = ""
        rows.append(
            {
                "master_sequence_id": master_id,
                "reference_scene_id": reference,
                "sequence_id": sequence_id,
                "scan_id_1": scan_ids[0],
                "scan_id_2": scan_ids[1],
                "exact_ordered_pair_present": present,
                "official_like_split": split,
                "official_like_supervised": supervised,
                "scene": record.get("scene", "") if present else "",
                "sub_scenes": serialized_sub_scenes,
                "official_like_filepath": filepath,
            }
        )
    return rows


def summarize_population(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if not rows:
        raise ProtocolShiftError("population rows must not be empty")
    exact_count = sum(bool(row.get("exact_ordered_pair_present")) for row in rows)
    requested = len(rows)
    return {
        "requested_pair_count": requested,
        "exact_pair_count": exact_count,
        "missing_pair_count": requested - exact_count,
        "full_exact_subset_identifiable": exact_count == requested
        and all(bool(row.get("official_like_supervised")) for row in rows),
    }


def _select_t2_rows(
    rows: Sequence[Mapping[str, object]], *, expected_orders: set[str]
) -> dict[str, Mapping[str, object]]:
    selected: dict[str, Mapping[str, object]] = {}
    for row in rows:
        if row.get("method") != "FullHistory" or str(row.get("horizon")) != "2":
            continue
        order = str(row.get("order_id"))
        if order in selected:
            raise ProtocolShiftError("T2 FullHistory rows contain duplicate orders")
        selected[order] = row
    if set(selected) != expected_orders:
        raise ProtocolShiftError("T2 FullHistory order coverage differs")
    return selected


def _protocol_metric_row(
    *, record_id: str, row: Mapping[str, object], order_scope: str
) -> dict[str, object]:
    return {
        "record_id": record_id,
        "source": "System Comparison V1 frozen CSV",
        "population": "Protocol B common-T5 masters",
        "order_scope": order_scope,
        "sequence_count": _integer(row.get("sequence_count"), name="sequence_count"),
        "t_mAP": _float(row.get("causal_prefix_t_mAP"), name="t_mAP"),
        "t_mAP50": _float(row.get("causal_prefix_t_mAP50"), name="t_mAP50"),
        "t_mAP25": _float(row.get("causal_prefix_t_mAP25"), name="t_mAP25"),
        "current_stage_AP": _float(
            row.get("current_stage_AP"), name="current_stage_AP"
        ),
        "direct_comparison_group": "protocol_b_t2",
        "notes": "official stmetrics; Protocol-B causal-prefix construction",
    }


def build_metric_rows(
    per_order_rows: Sequence[Mapping[str, object]],
    aggregate_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    per_order = _select_t2_rows(per_order_rows, expected_orders=set(ORDERS))
    pooled = _select_t2_rows(aggregate_rows, expected_orders={"all"})["all"]
    rows: list[dict[str, object]] = [
        {
            "record_id": "R0",
            "source": "ReScene4D paper reported",
            "population": "official T2 benchmark",
            "order_scope": "paper_reported",
            "sequence_count": "",
            "t_mAP": 0.348,
            "t_mAP50": 0.525,
            "t_mAP25": 0.668,
            "current_stage_AP": "",
            "direct_comparison_group": "official_like_t2",
            "notes": "reference only; paper-reported result",
        },
        {
            "record_id": "R1",
            "source": "P2_G2_REPRODUCTION_REPORT.md",
            "population": "official-like supervised T2 validation",
            "order_scope": "P2 sliding-T2",
            "sequence_count": 154,
            "t_mAP": 0.27939,
            "t_mAP50": 0.46565,
            "t_mAP25": 0.60945,
            "current_stage_AP": "",
            "direct_comparison_group": "official_like_t2",
            "notes": "authoritative single-GPU P2 reproduction",
        },
    ]
    for record_id, order in zip(("R2", "R3", "R4"), ORDERS, strict=True):
        rows.append(
            _protocol_metric_row(
                record_id=record_id,
                row=per_order[order],
                order_scope=order,
            )
        )
    rows.append(
        _protocol_metric_row(
            record_id="R5", row=pooled, order_scope="three_order_pooled"
        )
    )
    return rows


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_reference(path: Path) -> str:
    try:
        relative = path.resolve().relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        return f"external:{path.name}"
    return f"repo:{relative.as_posix()}"


def _csv_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    if not rows:
        raise ProtocolShiftError("CSV rows must not be empty")
    fields = list(rows[0])
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        if list(row) != fields:
            raise ProtocolShiftError("CSV row fields differ")
        writer.writerow(
            {
                key: (
                    "true"
                    if value is True
                    else "false"
                    if value is False
                    else value
                )
                for key, value in row.items()
            }
        )
    return stream.getvalue().encode("utf-8")


def _publish(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def render_report(
    population_rows: Sequence[Mapping[str, object]],
    metric_rows: Sequence[Mapping[str, object]],
    *,
    source_hashes: Mapping[str, str],
) -> str:
    summary = summarize_population(population_rows)
    by_id = {str(row["record_id"]): row for row in metric_rows}
    canonical = float(by_id["R2"]["t_mAP"])
    per_order = [float(by_id[key]["t_mAP"]) for key in ("R2", "R3", "R4")]
    pooled = float(by_id["R5"]["t_mAP"])
    lines = [
        "# Protocol Shift Audit",
        "",
        "## Frozen inputs",
        "",
    ]
    lines.extend(f"- `{path}`: `{digest}`" for path, digest in source_hashes.items())
    lines.extend(
        [
            "",
            "## R0-R5 observations",
            "",
            "| ID | Population | Order scope | N | t-mAP |",
            "| --- | --- | --- | ---: | ---: |",
        ]
    )
    for row in metric_rows:
        count = row["sequence_count"] if row["sequence_count"] != "" else "not reported"
        lines.append(
            f"| {row['record_id']} | {row['population']} | {row['order_scope']} | "
            f"{count} | {100.0 * float(row['t_mAP']):.6f}% |"
        )
    lines.extend(
        [
            "",
            "R0 and R1 target the official/official-like T2 benchmark and are the only "
            "intended benchmark-level comparison. R2-R4 are directly comparable order "
            "diagnostics inside Protocol B. R5 pools the same three Protocol-B orders; "
            "it is not an independent benchmark population.",
            "",
            "34.8 and 19.10 are not directly comparable: R0 is paper-reported official "
            "T2, while R5 is a pooled causal-prefix result over 43 common-T5 masters "
            "and three metadata-derived orders.",
            "",
            "## Exact matched-subset audit",
            "",
            f"- Requested canonical T2 pairs: {summary['requested_pair_count']}",
            f"- Exact ordered pairs present in the P2 sliding-T2 DB: {summary['exact_pair_count']}",
            f"- Missing exact ordered pairs: {summary['missing_pair_count']}",
            "- Exact full 43-pair control: NOT IDENTIFIABLE FROM CURRENT ARTIFACTS",
            "",
            "Only exact ordered sequence IDs count as matches. Reverse pairs and pairs "
            "from the same scene are not substituted. Because 29 of 43 canonical pairs "
            "are absent, an official-like 43-pair evaluation cannot be constructed. "
            "The 14 available pairs are retained as inventory evidence only and are not "
            "presented as the requested matched control.",
            "",
            "## Order effect",
            "",
            f"- Canonical R2 t-mAP: {100.0 * canonical:.6f}%",
            f"- Pooled R5 t-mAP: {100.0 * pooled:.6f}%",
            f"- Pooled minus canonical: {100.0 * (pooled - canonical):.6f} percentage points",
            f"- Per-order max-minus-min spread: {100.0 * (max(per_order) - min(per_order)):.6f} percentage points",
            "",
            "The canonical-to-pooled difference measures sensitivity to the registered "
            "Protocol-B orders, not a model degradation from the paper score. The common-T5 "
            "subset effect cannot be isolated exactly with current artifacts. These deltas "
            "must not be added into a causal decomposition of 34.8 to 19.10.",
            "",
            "## Gate E1",
            "",
            "`E1 = PASS`: population, order, evaluator, and comparability boundaries are "
            "explicit. The exact 43-pair subset effect remains non-identifiable and is "
            "reported as such rather than approximated.",
            "",
        ]
    )
    return "\n".join(lines)


def run_audit(
    *,
    protocol_manifest_path: Path,
    t2_database_path: Path,
    per_order_results_path: Path,
    aggregate_results_path: Path,
    output_root: Path,
) -> dict[str, object]:
    protocol = json.loads(protocol_manifest_path.read_text(encoding="utf-8"))
    t2_database = yaml.safe_load(t2_database_path.read_text(encoding="utf-8"))
    with per_order_results_path.open(newline="", encoding="utf-8") as handle:
        per_order = list(csv.DictReader(handle))
    with aggregate_results_path.open(newline="", encoding="utf-8") as handle:
        aggregate = list(csv.DictReader(handle))
    population_rows = build_population_rows(protocol, t2_database)
    metric_rows = build_metric_rows(per_order, aggregate)
    source_paths = (
        protocol_manifest_path,
        t2_database_path,
        per_order_results_path,
        aggregate_results_path,
    )
    source_hashes = {
        _source_reference(path): _sha256_file(path) for path in source_paths
    }
    _publish(output_root / "protocol_shift_population.csv", _csv_bytes(population_rows))
    _publish(output_root / "protocol_shift_metrics.csv", _csv_bytes(metric_rows))
    _publish(
        output_root / "PROTOCOL_SHIFT_AUDIT.md",
        render_report(
            population_rows, metric_rows, source_hashes=source_hashes
        ).encode("utf-8"),
    )
    return summarize_population(population_rows)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol-manifest",
        type=Path,
        default=PROJECT_ROOT / "artifacts/P6A/protocol_b_manifest.json",
    )
    parser.add_argument(
        "--t2-database",
        type=Path,
        default=PROJECT_ROOT / "data/processed/rio/sequence_database_sliding_2.yaml",
    )
    parser.add_argument(
        "--per-order-results",
        type=Path,
        default=PROJECT_ROOT / "artifacts/system_comparison/per_order_results.csv",
    )
    parser.add_argument(
        "--aggregate-results",
        type=Path,
        default=PROJECT_ROOT / "artifacts/system_comparison/aggregate_results.csv",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts/tmap_root_cause_v2",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary = run_audit(
        protocol_manifest_path=args.protocol_manifest,
        t2_database_path=args.t2_database,
        per_order_results_path=args.per_order_results,
        aggregate_results_path=args.aggregate_results,
        output_root=args.output_root,
    )
    print(json.dumps(summary, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
