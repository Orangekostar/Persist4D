"""Five-path task post-processing attribution for System Comparison V2."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
from torch import Tensor

from scripts.evaluate_persist4d_p6a import (
    build_rio_class_mapper,
    build_temporal_target,
    build_tracker_factories,
    cache_payload_to_frozen_observation,
    stage_prediction_from_track_step,
)
from scripts.p6a_association import TrackStep
from scripts.p6a_metrics import IdentityAccumulator, build_online_endpoint_prediction
from scripts.run_system_comparison import (
    FULL_CACHE_MANIFEST,
    FULL_ENTRY_CACHE,
    REPRODUCIBILITY_BINDING,
    _build_frozen_setup,
)
from scripts.system_comparison_inference import load_full_history_cache_entry
from scripts.system_comparison_metrics import (
    CausalTaskAccumulator,
    causal_prefix_pair_from_payload,
    validate_causal_prefix_pair,
)
from scripts.system_comparison_v2_analysis import (
    HORIZONS,
    ORDERS,
    TASK_FIELDS,
    build_v2_causal_pair,
    load_v2_sequences,
)
from scripts.system_comparison_v2_inference import (
    OfficialCandidateTrajectoryAccumulator,
)

METHODS = ("F0", "L0", "L1", "P0", "P1")
METHOD_DESCRIPTIONS = {
    "F0": "FullHistory official task prediction",
    "L0": "Local official candidates without persistent identity",
    "L1": "Legacy raw-observation conversion without persistent identity",
    "P0": "Legacy raw-observation conversion with B4 identity",
    "P1": "V2 official candidates with B4 identity",
}
OUTPUT_FIELDS = (
    "method",
    "method_description",
    "order_id",
    "horizon",
    "sequence_count",
    "trajectory_candidate_count_total",
    "trajectory_candidate_count_mean",
    "empty_trajectory_mask_count",
    "empty_trajectory_mask_rate",
    "current_stage_candidate_count_total",
    "current_stage_candidate_count_mean",
    "score_count",
    "score_min",
    "score_q25",
    "score_median",
    "score_q75",
    "score_max",
    "score_mean",
    "score_std",
    *TASK_FIELDS,
)


class AttributionError(RuntimeError):
    """Raised when attribution paths are incomplete or incomparable."""


def _read_json(path: Path) -> Mapping[str, object]:
    if path.is_symlink() or not path.is_file():
        raise AttributionError(f"required JSON is unavailable: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise AttributionError(f"required JSON must contain a mapping: {path}")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _csv_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=OUTPUT_FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        if set(row) != set(OUTPUT_FIELDS):
            raise AttributionError("attribution row fields differ")
        writer.writerow(row)
    return stream.getvalue().encode("utf-8")


def _map_classes(values: Tensor, mapper) -> Tensor:
    return torch.tensor(
        [mapper(int(value)) for value in values.detach().cpu().long().tolist()],
        dtype=torch.long,
    )


def _legacy_pair(
    *,
    prediction: Mapping[str, object],
    raw_payloads: Sequence[Mapping[str, object]],
    class_mapper,
) -> object:
    horizon = len(raw_payloads)
    target = build_temporal_target(raw_payloads)
    key = raw_payloads[-1]["key"]
    return validate_causal_prefix_pair(
        prediction={
            "pred_masks": prediction["pred_masks"],
            "pred_scores": prediction["pred_scores"],
            "pred_classes": _map_classes(prediction["pred_classes"], class_mapper),
        },
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


def _none_step(*, stage: int, query_count: int, sequence_id: str) -> TrackStep:
    return TrackStep(
        method="L0",
        sequence_id=sequence_id,
        stage_id=stage,
        track_ids=(None,) * query_count,
        matched_previous=(-1,) * query_count,
        scores=(None,) * query_count,
        births=(False,) * query_count,
        valid=(True,) * query_count,
    )


def _full_entry_map(
    manifest: Mapping[str, object],
) -> dict[tuple[str, str, int], Mapping[str, object]]:
    entries = manifest.get("entries")
    if isinstance(entries, (str, bytes)) or not isinstance(entries, Sequence):
        raise AttributionError("FullHistory entries must be a sequence")
    result = {}
    for entry in entries:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("key"), Mapping):
            raise AttributionError("FullHistory entry is invalid")
        key = entry["key"]
        identity = (
            str(key["master_sequence_id"]),
            str(key["order_id"]),
            int(key["horizon"]),
        )
        if identity in result:
            raise AttributionError("FullHistory cache contains duplicate cells")
        result[identity] = entry
    return result


def _accumulators() -> dict[tuple[str, str, int], CausalTaskAccumulator]:
    return {
        (method, order, horizon): CausalTaskAccumulator()
        for method in METHODS
        for order in (*ORDERS, "all")
        for horizon in HORIZONS
    }


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise AttributionError("score distribution is empty")
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def _summary_row(
    *,
    method: str,
    order: str,
    horizon: int,
    accumulator: CausalTaskAccumulator,
    stats: Mapping[str, object],
) -> dict[str, object]:
    scores = sorted(float(value) for value in stats["scores"])
    count = int(stats["sequence_count"])
    candidate_count = int(stats["candidate_count"])
    empty_count = int(stats["empty_count"])
    current_count = int(stats["current_count"])
    mean = sum(scores) / len(scores)
    variance = sum((value - mean) ** 2 for value in scores) / len(scores)
    return {
        "method": method,
        "method_description": METHOD_DESCRIPTIONS[method],
        "order_id": order,
        "horizon": horizon,
        "sequence_count": count,
        "trajectory_candidate_count_total": candidate_count,
        "trajectory_candidate_count_mean": candidate_count / count,
        "empty_trajectory_mask_count": empty_count,
        "empty_trajectory_mask_rate": (
            empty_count / candidate_count if candidate_count else 0.0
        ),
        "current_stage_candidate_count_total": current_count,
        "current_stage_candidate_count_mean": current_count / count,
        "score_count": len(scores),
        "score_min": scores[0],
        "score_q25": _quantile(scores, 0.25),
        "score_median": _quantile(scores, 0.5),
        "score_q75": _quantile(scores, 0.75),
        "score_max": scores[-1],
        "score_mean": mean,
        "score_std": math.sqrt(variance),
        **accumulator.compute(),
    }


def _update_stats(
    stats: dict[tuple[str, str, int], dict[str, object]],
    *,
    method: str,
    order: str,
    horizon: int,
    pair,
) -> None:
    selector = pair.target["temporal_stages"] == horizon - 1
    masks = pair.prediction["pred_masks"]
    scores = pair.prediction["pred_scores"]
    candidate_count = int(scores.numel())
    empty_count = int((~masks.any(dim=0)).sum().item())
    current_count = int(masks[selector].any(dim=0).sum().item())
    for scope in (order, "all"):
        key = (method, scope, horizon)
        entry = stats.setdefault(
            key,
            {
                "sequence_count": 0,
                "candidate_count": 0,
                "empty_count": 0,
                "current_count": 0,
                "scores": [],
            },
        )
        entry["sequence_count"] += 1
        entry["candidate_count"] += candidate_count
        entry["empty_count"] += empty_count
        entry["current_count"] += current_count
        entry["scores"].extend(scores.detach().cpu().tolist())


def _report(rows: Sequence[Mapping[str, object]]) -> bytes:
    pooled = {
        (str(row["method"]), int(row["horizon"])): row
        for row in rows
        if row["order_id"] == "all"
    }
    lines = [
        "# Post-processing Attribution",
        "",
        "- Status: `pass`",
        "- Population: `43 masters x 3 orders`",
        "- Horizons: `T2-T5`",
        "- Primary P1 score reducer: `mean`",
        "",
        "| Path | Horizon | Candidates / sequence | Current-stage candidates / sequence | Current AP | Causal t-mAP | t-mREC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        for horizon in HORIZONS:
            row = pooled[(method, horizon)]
            lines.append(
                f"| {method} | T{horizon} | "
                f"{float(row['trajectory_candidate_count_mean']):.3f} | "
                f"{float(row['current_stage_candidate_count_mean']):.3f} | "
                f"{float(row['current_stage_AP']):.6f} | "
                f"{float(row['causal_prefix_t_mAP']):.6f} | "
                f"{float(row['causal_prefix_t_REC']):.6f} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "L0/L1 and P0/P1 are controlled post-processing contrasts on the same",
            "fresh local raw/sidecar cache. F0 uses the frozen FullHistory official",
            "cache. Pairwise differences are diagnostics and are not assumed to add",
            "up to the complete FullHistory-versus-Persist4D gap.",
            "",
        ]
    )
    return "\n".join(lines).encode("utf-8")


def run_postprocessing_attribution(
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
    sequences = load_v2_sequences(cache_manifest=cache_manifest, cache_root=cache_root)
    full_manifest = _read_json(FULL_CACHE_MANIFEST)
    full_entries = _full_entry_map(full_manifest)
    class_mapper = build_rio_class_mapper(setup.dataset)
    factories = build_tracker_factories(setup.p6a_config)
    accumulators = _accumulators()
    stats: dict[tuple[str, str, int], dict[str, object]] = {}

    for sequence in sequences:
        sequence_id = f"{sequence.master_sequence_id}:{sequence.order_id}"
        b0_tracker = factories["B0"](sequence_id)
        b4_tracker = factories["B4"](sequence_id)
        l0_trajectory = OfficialCandidateTrajectoryAccumulator(score_reducer="mean")
        p1_trajectory = OfficialCandidateTrajectoryAccumulator(score_reducer="mean")
        l1_accumulator = IdentityAccumulator()
        p0_accumulator = IdentityAccumulator()
        for stage, (raw, sidecar) in enumerate(
            zip(sequence.raw_payloads, sequence.sidecars, strict=True)
        ):
            observation = cache_payload_to_frozen_observation(raw)
            b0_step = b0_tracker.step(observation, stage_id=stage)
            b4_step = b4_tracker.step(observation, stage_id=stage)
            query_count = int(raw["observation"]["features"].shape[0])
            l0_trajectory.add_stage(
                sidecar,
                _none_step(
                    stage=stage,
                    query_count=query_count,
                    sequence_id=sequence_id,
                ),
            )
            p1_trajectory.add_stage(sidecar, b4_step)
            l1_accumulator.add_stage(stage_prediction_from_track_step(raw, b0_step))
            p0_accumulator.add_stage(stage_prediction_from_track_step(raw, b4_step))
            horizon = stage + 1
            if horizon not in HORIZONS:
                continue
            raw_prefix = sequence.raw_payloads[:horizon]
            pairs = {
                "L0": build_v2_causal_pair(
                    snapshot=l0_trajectory.snapshot(),
                    raw_payloads=raw_prefix,
                    class_mapper=class_mapper,
                ),
                "L1": _legacy_pair(
                    prediction=build_online_endpoint_prediction(
                        l1_accumulator, endpoint=stage
                    ),
                    raw_payloads=raw_prefix,
                    class_mapper=class_mapper,
                ),
                "P0": _legacy_pair(
                    prediction=build_online_endpoint_prediction(
                        p0_accumulator, endpoint=stage
                    ),
                    raw_payloads=raw_prefix,
                    class_mapper=class_mapper,
                ),
                "P1": build_v2_causal_pair(
                    snapshot=p1_trajectory.snapshot(),
                    raw_payloads=raw_prefix,
                    class_mapper=class_mapper,
                ),
            }
            full_payload = load_full_history_cache_entry(
                FULL_ENTRY_CACHE,
                full_entries[(sequence.master_sequence_id, sequence.order_id, horizon)],
                expected_provenance=full_manifest["provenance"],
            )
            pairs["F0"] = causal_prefix_pair_from_payload(full_payload)
            for method, pair in pairs.items():
                for scope in (sequence.order_id, "all"):
                    accumulators[(method, scope, horizon)].update(pair)
                _update_stats(
                    stats,
                    method=method,
                    order=sequence.order_id,
                    horizon=horizon,
                    pair=pair,
                )

    rows = []
    for order in (*ORDERS, "all"):
        for method in METHODS:
            for horizon in HORIZONS:
                rows.append(
                    _summary_row(
                        method=method,
                        order=order,
                        horizon=horizon,
                        accumulator=accumulators[(method, order, horizon)],
                        stats=stats[(method, order, horizon)],
                    )
                )
    if len(rows) != 80:
        raise AttributionError("attribution coverage differs")
    csv_payload = _csv_bytes(rows)
    report_payload = _report(rows)
    _publish(output_root / "postprocessing_attribution.csv", csv_payload)
    _publish(output_root / "POSTPROCESSING_ATTRIBUTION.md", report_payload)
    manifest = {
        "schema_version": 1,
        "status": "pass",
        "analysis_source_commit": analysis_source_commit,
        "cache_source_commit": cache_manifest["source_commit"],
        "cache_manifest_sha256": _file_sha256(cache_manifest_path),
        "full_history_manifest_sha256": _file_sha256(FULL_CACHE_MANIFEST),
        "methods": METHOD_DESCRIPTIONS,
        "population": {
            "masters": 43,
            "orders": list(ORDERS),
            "horizons": list(HORIZONS),
        },
        "outputs": {
            "postprocessing_attribution.csv": hashlib.sha256(csv_payload).hexdigest(),
            "POSTPROCESSING_ATTRIBUTION.md": hashlib.sha256(report_payload).hexdigest(),
        },
        "interpretation": "pairwise_diagnostics_not_additive_causal_decomposition",
    }
    manifest_payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    _publish(output_root / "postprocessing_attribution_manifest.json", manifest_payload)
    return {
        "status": "pass",
        "row_count": len(rows),
        "manifest_sha256": hashlib.sha256(manifest_payload).hexdigest(),
    }


__all__ = ["AttributionError", "run_postprocessing_attribution"]
