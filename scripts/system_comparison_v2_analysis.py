"""CPU-only System Comparison V2 analysis from matched cache pairs."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from scripts.evaluate_persist4d_p6a import (
    build_rio_class_mapper,
    build_temporal_target,
    build_tracker_factories,
    cache_payload_to_frozen_observation,
)
from scripts.p6a_cache import validate_cache_entry
from scripts.run_system_comparison import (
    FULL_CACHE_MANIFEST,
    FULL_ENTRY_CACHE,
    REPRODUCIBILITY_BINDING,
    SYSTEM_ROOT,
    _build_frozen_setup,
)
from scripts.system_comparison_inference import load_full_history_cache_entry
from scripts.system_comparison_metrics import (
    CausalTaskAccumulator,
    causal_prefix_pair_from_payload,
    compute_causal_task_metrics,
    validate_causal_prefix_pair,
)
from scripts.system_comparison_v2_cache import (
    load_task_sidecar,
    observation_fingerprint,
)
from scripts.system_comparison_v2_inference import (
    OfficialCandidateTrajectoryAccumulator,
    V2TrajectorySnapshot,
)

METHODS = ("FullHistory", "Persist4D-V2")
ORDERS = ("canonical", "reverse", "sha256_seed45")
HORIZONS = (2, 3, 4, 5)
TASK_FIELDS = (
    "causal_prefix_t_mAP",
    "causal_prefix_t_mAP50",
    "causal_prefix_t_mAP25",
    "causal_prefix_t_REC",
    "causal_prefix_t_REC50",
    "causal_prefix_t_REC25",
    "current_stage_AP",
    "current_stage_AP50",
    "current_stage_AP25",
    "current_stage_REC",
)
IDENTITY_COUNT_FIELDS = (
    "deployment_id_switches",
    "identity_transition_opportunities",
    "fragmentation_count",
    "fragmentation_opportunities",
    "merge_count",
    "merge_opportunities",
    "gap_opportunities",
    "recovery_attempts",
    "correct_recoveries",
)
IDENTITY_RATE_FIELDS = (
    "normalized_id_switch_rate",
    "fragmentation_rate",
    "merge_rate",
    "gap_recovery_accuracy",
    "gap_recovery_recall",
)
IDENTITY_FIELDS = (*IDENTITY_COUNT_FIELDS, *IDENTITY_RATE_FIELDS)
AGGREGATE_FIELDS = (
    "method",
    "order_id",
    "horizon",
    "sequence_count",
    *TASK_FIELDS,
    *IDENTITY_FIELDS,
)
PER_SEQUENCE_FIELDS = (
    "method",
    "reference_scene_id",
    "master_sequence_id",
    "order_id",
    "horizon",
    *TASK_FIELDS,
    *IDENTITY_FIELDS,
    "update_scan_count",
    "update_point_count",
    "cumulative_scan_count",
)


class V2AnalysisError(RuntimeError):
    """Raised when V2 analysis evidence is incomplete or inconsistent."""


@dataclass(frozen=True)
class CachedV2Sequence:
    reference_scene_id: str
    master_sequence_id: str
    order_id: str
    raw_payloads: tuple[Mapping[str, object], ...]
    sidecars: tuple[Mapping[str, object], ...]


def _read_json(path: Path) -> Mapping[str, object]:
    if path.is_symlink() or not path.is_file():
        raise V2AnalysisError(f"required JSON is unavailable: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise V2AnalysisError(f"required JSON must contain a mapping: {path}")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    if path.is_symlink() or not path.is_file():
        raise V2AnalysisError(f"required CSV is unavailable: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise V2AnalysisError(f"required CSV is empty: {path}")
    return rows


def _optional_number(value: str) -> int | float | None:
    if value == "":
        return None
    try:
        integer = int(value)
    except ValueError:
        number = float(value)
        if not math.isfinite(number):
            raise V2AnalysisError("CSV metric must be finite")
        return number
    return integer


def _identity_from_old(row: Mapping[str, str]) -> dict[str, object]:
    return {field: _optional_number(row[field]) for field in IDENTITY_FIELDS}


def _csv_bytes(
    rows: Sequence[Mapping[str, object]], fields: Sequence[str]
) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        if set(row) != set(fields):
            raise V2AnalysisError("CSV row fields differ from output contract")
        writer.writerow(
            {field: "" if row[field] is None else row[field] for field in fields}
        )
    return stream.getvalue().encode("utf-8")


def _publish(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file():
            raise FileExistsError(f"output is not a regular file: {path}")
        if path.read_bytes() == payload:
            return
        raise FileExistsError(f"output already contains different content: {path}")
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
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _map_classes(values: Tensor, mapper: Callable[[int], int]) -> Tensor:
    return torch.tensor(
        [mapper(int(value)) for value in values.detach().cpu().long().tolist()],
        dtype=torch.long,
    )


def build_v2_causal_pair(
    *,
    snapshot: V2TrajectorySnapshot,
    raw_payloads: Sequence[Mapping[str, object]],
    class_mapper: Callable[[int], int],
) -> object:
    horizon = snapshot.stage_count
    if horizon != len(raw_payloads) or not 1 <= horizon <= 5:
        raise V2AnalysisError("V2 snapshot and raw prefix horizons differ")
    target = build_temporal_target(raw_payloads)
    key = raw_payloads[-1]["key"]
    return validate_causal_prefix_pair(
        prediction=snapshot.prediction,
        target={
            "masks": target["masks"],
            "labels": _map_classes(target["labels"], class_mapper),
            "ids": target["ids"],
            "changes": target["changes"],
            "temporal_stages": target["temporal_stages"],
        },
        horizon=horizon,
        observed_scan_ids=key["history_scan_ids"],
    )


def load_v2_sequences(
    *, cache_manifest: Mapping[str, object], cache_root: Path
) -> tuple[CachedV2Sequence, ...]:
    if cache_manifest.get("status") != "pass" or cache_manifest.get(
        "entry_count"
    ) != 645:
        raise V2AnalysisError("V2 cache manifest must pass with 645 entries")
    records = cache_manifest.get("records")
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise V2AnalysisError("V2 cache records must be a sequence")
    grouped: dict[
        tuple[str, str, str], dict[int, tuple[Mapping[str, object], Mapping[str, object]]]
    ] = {}
    raw_directory = cache_root / "raw_predictions/entries"
    sidecar_directory = cache_root / "task_sidecars/entries"
    for record in records:
        if not isinstance(record, Mapping) or not isinstance(record.get("key"), Mapping):
            raise V2AnalysisError("V2 cache record is invalid")
        key = record["key"]
        identity = (
            str(key["reference_scene_id"]),
            str(key["master_sequence_id"]),
            str(key["order_id"]),
        )
        stage = int(key["stage_index"])
        stages = grouped.setdefault(identity, {})
        if stage in stages:
            raise V2AnalysisError("V2 cache contains duplicate sequence stages")
        raw_entry = record["raw_entry"]
        sidecar_entry = record["sidecar_entry"]
        if not isinstance(raw_entry, Mapping) or not isinstance(
            sidecar_entry, Mapping
        ):
            raise V2AnalysisError("V2 cache entries must be mappings")
        raw = validate_cache_entry(
            raw_directory / str(raw_entry["filename"]), raw_entry
        )
        sidecar = load_task_sidecar(
            sidecar_directory / str(sidecar_entry["filename"])
        )
        fingerprint = observation_fingerprint(raw)
        if (
            raw["key"] != key
            or sidecar["key"] != key
            or fingerprint != record["raw_observation_fingerprint"]
            or fingerprint
            != sidecar["provenance"]["source_raw_observation_fingerprint"]
        ):
            raise V2AnalysisError("V2 raw/sidecar pair binding differs")
        stages[stage] = (raw, sidecar)
    if len(grouped) != 129:
        raise V2AnalysisError("V2 cache must contain 129 sequences")
    sequences = []
    for identity, stages in grouped.items():
        if set(stages) != set(range(5)):
            raise V2AnalysisError("V2 sequence stage coverage is incomplete")
        pairs = [stages[stage] for stage in range(5)]
        sequences.append(
            CachedV2Sequence(
                reference_scene_id=identity[0],
                master_sequence_id=identity[1],
                order_id=identity[2],
                raw_payloads=tuple(pair[0] for pair in pairs),
                sidecars=tuple(pair[1] for pair in pairs),
            )
        )
    order_index = {name: index for index, name in enumerate(ORDERS)}
    return tuple(
        sorted(
            sequences,
            key=lambda value: (
                value.master_sequence_id,
                order_index[value.order_id],
            ),
        )
    )


def _task_accumulators() -> dict[tuple[str, str, int], CausalTaskAccumulator]:
    return {
        (method, order, horizon): CausalTaskAccumulator()
        for method in METHODS
        for order in (*ORDERS, "all")
        for horizon in HORIZONS
    }


def _old_row_maps() -> tuple[dict[tuple[str, str, str, int], dict[str, str]], dict[tuple[str, str, int], dict[str, str]]]:
    per_sequence = _read_csv(SYSTEM_ROOT / "per_sequence_results.csv")
    aggregate = _read_csv(SYSTEM_ROOT / "aggregate_results.csv")
    aggregate.extend(_read_csv(SYSTEM_ROOT / "per_order_results.csv"))
    sequence_map = {
        (
            row["method"],
            row["master_sequence_id"],
            row["order_id"],
            int(row["horizon"]),
        ): row
        for row in per_sequence
    }
    aggregate_map = {
        (row["method"], row["order_id"], int(row["horizon"])): row
        for row in aggregate
    }
    expected_sequence = len(METHODS) * 129 * len(HORIZONS)
    if len(sequence_map) != expected_sequence:
        raise V2AnalysisError("frozen per-sequence coverage differs")
    expected_aggregate = len(METHODS) * (len(ORDERS) + 1) * len(HORIZONS)
    if len(aggregate_map) != expected_aggregate:
        raise V2AnalysisError("frozen aggregate coverage differs")
    return sequence_map, aggregate_map


def _full_entries() -> tuple[Mapping[str, object], dict[tuple[str, str, int], Mapping[str, object]]]:
    manifest = _read_json(FULL_CACHE_MANIFEST)
    if manifest.get("status") != "pass" or manifest.get("entry_count") != 645:
        raise V2AnalysisError("FullHistory cache manifest must pass")
    entries = manifest.get("entries")
    if isinstance(entries, (str, bytes)) or not isinstance(entries, Sequence):
        raise V2AnalysisError("FullHistory entries must be a sequence")
    result = {}
    for entry in entries:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("key"), Mapping):
            raise V2AnalysisError("FullHistory entry is invalid")
        key = entry["key"]
        identity = (
            str(key["master_sequence_id"]),
            str(key["order_id"]),
            int(key["horizon"]),
        )
        if identity in result:
            raise V2AnalysisError("FullHistory cache contains duplicate cells")
        result[identity] = entry
    return manifest, result


def _aggregate_rows(
    *,
    accumulators: Mapping[tuple[str, str, int], CausalTaskAccumulator],
    old_aggregate: Mapping[tuple[str, str, int], Mapping[str, str]],
    order_id: str,
) -> list[dict[str, object]]:
    rows = []
    for method in METHODS:
        old_method = "FullHistory" if method == "FullHistory" else "Persist4D"
        for horizon in HORIZONS:
            metrics = accumulators[(method, order_id, horizon)].compute()
            old = old_aggregate[(old_method, order_id, horizon)]
            rows.append(
                {
                    "method": method,
                    "order_id": order_id,
                    "horizon": horizon,
                    "sequence_count": 129 if order_id == "all" else 43,
                    **metrics,
                    **_identity_from_old(old),
                }
            )
    return rows


def _attribution_report(
    aggregate_rows: Sequence[Mapping[str, object]],
    old_aggregate: Mapping[tuple[str, str, int], Mapping[str, str]],
) -> bytes:
    current = {
        (str(row["method"]), int(row["horizon"])): row
        for row in aggregate_rows
        if row["order_id"] == "all"
    }
    lines = [
        "# Old vs System Comparison V2 Attribution",
        "",
        "- Status: `pass`",
        "- V2 task candidates: official local ReScene candidates",
        "- Identity linkage: unchanged B4 raw-query tracker",
        "- Persistent trajectory key: `(track_id, official_class_id)`",
        "- Unmatched candidates: retained with stage-local ephemeral keys",
        "- Primary score reducer: mean official per-stage candidate score",
        "",
        "| Horizon | FullHistory t-mAP | Legacy Persist4D t-mAP | V2 Persist4D t-mAP | V2 - legacy | Legacy current AP | V2 current AP |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for horizon in HORIZONS:
        full = current[("FullHistory", horizon)]
        v2 = current[("Persist4D-V2", horizon)]
        legacy = old_aggregate[("Persist4D", "all", horizon)]
        legacy_tmap = float(legacy["causal_prefix_t_mAP"])
        legacy_ap = float(legacy["current_stage_AP"])
        lines.append(
            f"| T{horizon} | {float(full['causal_prefix_t_mAP']):.6f} | "
            f"{legacy_tmap:.6f} | {float(v2['causal_prefix_t_mAP']):.6f} | "
            f"{float(v2['causal_prefix_t_mAP']) - legacy_tmap:+.6f} | "
            f"{legacy_ap:.6f} | {float(v2['current_stage_AP']):.6f} |"
        )
    lines.extend(
        [
            "",
            "The V2-minus-legacy differences isolate a task-prediction semantics change",
            "while keeping the registered B4 tracker algorithm and identity reporting",
            "path unchanged. They are not an additive causal decomposition of the total",
            "FullHistory gap. Frozen V1 identity fields are copied only after exact",
            "keyed regression and therefore remain byte-for-byte numerically unchanged.",
            "",
        ]
    )
    return "\n".join(lines).encode("utf-8")


def run_v2_analysis(
    *,
    metadata_path: Path,
    cache_root: Path,
    cache_manifest_path: Path,
    output_root: Path,
) -> Mapping[str, object]:
    analysis_source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    binding = _read_json(REPRODUCIBILITY_BINDING)
    setup = _build_frozen_setup(
        binding=binding,
        metadata_path=metadata_path,
        device_name=None,
    )
    cache_manifest = _read_json(cache_manifest_path)
    sequences = load_v2_sequences(
        cache_manifest=cache_manifest, cache_root=cache_root
    )
    class_mapper = build_rio_class_mapper(setup.dataset)
    tracker_factory = build_tracker_factories(setup.p6a_config)["B4"]
    full_manifest, full_entries = _full_entries()
    old_sequences, old_aggregate = _old_row_maps()
    accumulators = _task_accumulators()
    per_sequence_rows: list[dict[str, object]] = []

    for sequence in sequences:
        tracker = tracker_factory(
            f"{sequence.master_sequence_id}:{sequence.order_id}"
        )
        trajectory = OfficialCandidateTrajectoryAccumulator(
            score_reducer="mean"
        )
        for stage, (raw, sidecar) in enumerate(
            zip(sequence.raw_payloads, sequence.sidecars, strict=True)
        ):
            step = tracker.step(
                cache_payload_to_frozen_observation(raw), stage_id=stage
            )
            trajectory.add_stage(sidecar, step)
            horizon = stage + 1
            if horizon not in HORIZONS:
                continue
            v2_pair = build_v2_causal_pair(
                snapshot=trajectory.snapshot(),
                raw_payloads=sequence.raw_payloads[:horizon],
                class_mapper=class_mapper,
            )
            full_entry = full_entries[
                (sequence.master_sequence_id, sequence.order_id, horizon)
            ]
            full_payload = load_full_history_cache_entry(
                FULL_ENTRY_CACHE,
                full_entry,
                expected_provenance=full_manifest["provenance"],
            )
            full_pair = causal_prefix_pair_from_payload(full_payload)
            for method, pair, old_method in (
                ("FullHistory", full_pair, "FullHistory"),
                ("Persist4D-V2", v2_pair, "Persist4D"),
            ):
                metrics = compute_causal_task_metrics([pair])
                accumulators[(method, sequence.order_id, horizon)].update(pair)
                accumulators[(method, "all", horizon)].update(pair)
                old = old_sequences[
                    (
                        old_method,
                        sequence.master_sequence_id,
                        sequence.order_id,
                        horizon,
                    )
                ]
                per_sequence_rows.append(
                    {
                        "method": method,
                        "reference_scene_id": sequence.reference_scene_id,
                        "master_sequence_id": sequence.master_sequence_id,
                        "order_id": sequence.order_id,
                        "horizon": horizon,
                        **metrics,
                        **_identity_from_old(old),
                        "update_scan_count": int(old["update_scan_count"]),
                        "update_point_count": int(old["update_point_count"]),
                        "cumulative_scan_count": int(old["cumulative_scan_count"]),
                    }
                )

    per_order_rows = _aggregate_rows(
        accumulators=accumulators,
        old_aggregate=old_aggregate,
        order_id="canonical",
    )
    for order in ORDERS[1:]:
        per_order_rows.extend(
            _aggregate_rows(
                accumulators=accumulators,
                old_aggregate=old_aggregate,
                order_id=order,
            )
        )
    aggregate_rows = _aggregate_rows(
        accumulators=accumulators,
        old_aggregate=old_aggregate,
        order_id="all",
    )
    if len(per_sequence_rows) != 2 * 129 * 4:
        raise V2AnalysisError("V2 per-sequence coverage differs")

    full_regression = []
    for row in aggregate_rows + per_order_rows:
        if row["method"] != "FullHistory":
            continue
        old = old_aggregate[("FullHistory", str(row["order_id"]), int(row["horizon"]))]
        full_regression.extend(
            abs(float(row[field]) - float(old[field])) for field in TASK_FIELDS
        )
    if max(full_regression, default=0.0) > 1e-12:
        raise V2AnalysisError("FullHistory task metric regression failed")

    outputs = {
        "aggregate_results.csv": _csv_bytes(aggregate_rows, AGGREGATE_FIELDS),
        "per_order_results.csv": _csv_bytes(per_order_rows, AGGREGATE_FIELDS),
        "per_sequence_results.csv": _csv_bytes(
            per_sequence_rows, PER_SEQUENCE_FIELDS
        ),
        "OLD_VS_V2_ATTRIBUTION.md": _attribution_report(
            aggregate_rows, old_aggregate
        ),
    }
    for filename, payload in outputs.items():
        _publish(output_root / filename, payload)
    manifest = {
        "schema_version": 2,
        "status": "pass",
        "cache_source_commit": cache_manifest["source_commit"],
        "analysis_source_commit": analysis_source_commit,
        "checkpoint_sha256": cache_manifest["checkpoint_sha256"],
        "protocol_manifest_sha256": cache_manifest[
            "protocol_manifest_sha256"
        ],
        "cache_manifest_sha256": _file_sha256(cache_manifest_path),
        "score_reducer": "mean",
        "task_candidate_semantics": "official_local_candidates",
        "persistent_key_semantics": "track_id_plus_official_class_id",
        "unmatched_candidate_semantics": "stage_local_ephemeral_retained",
        "causal_commitment": "latest_stage_only_no_future_rewrite",
        "identity_metric_source": "frozen_v1_raw_query_b4_path",
        "identity_regression_status": "pass_exact",
        "full_history_task_regression_max_abs_diff": max(
            full_regression, default=0.0
        ),
        "coverage": {
            "sequence_count": 129,
            "horizons": list(HORIZONS),
            "methods": list(METHODS),
            "per_sequence_row_count": len(per_sequence_rows),
            "per_order_row_count": len(per_order_rows),
            "aggregate_row_count": len(aggregate_rows),
        },
        "outputs": {
            filename: {
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
            }
            for filename, payload in outputs.items()
        },
        "frozen_v1_inputs": {
            "aggregate_results_sha256": _file_sha256(
                SYSTEM_ROOT / "aggregate_results.csv"
            ),
            "per_order_results_sha256": _file_sha256(
                SYSTEM_ROOT / "per_order_results.csv"
            ),
            "per_sequence_results_sha256": _file_sha256(
                SYSTEM_ROOT / "per_sequence_results.csv"
            ),
        },
    }
    manifest_bytes = (
        json.dumps(
            manifest,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    _publish(output_root / "manifest.json", manifest_bytes)
    return {
        "status": "pass",
        "sequence_count": 129,
        "aggregate_row_count": len(aggregate_rows),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
    }


__all__ = [
    "CachedV2Sequence",
    "V2AnalysisError",
    "build_v2_causal_pair",
    "load_v2_sequences",
    "run_v2_analysis",
]
