"""Run the frozen reviewer-closure performance decomposition on CPU."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import subprocess
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch

from scripts.reviewer_closure_decomposition import (
    COVERAGE_CATEGORIES,
    FAILURE_CATEGORIES,
    OfficialTemporalCurveAccumulator,
    build_oracle_accumulator,
    classify_ceiling,
    classify_decomposition_failure,
    classify_observation_coverage,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "artifacts/reviewer_closure"
TRACKED_LOCAL_MANIFEST = (
    PROJECT_ROOT / "artifacts/system_comparison/persistent_predictions/manifest.json"
)
TRACKED_FULL_MANIFEST = (
    PROJECT_ROOT / "artifacts/system_comparison/full_history_predictions/manifest.json"
)
SIDECAR_ROOT = OUTPUT_ROOT / "full_history_observations_v2"
SYSTEM_AGGREGATE = PROJECT_ROOT / "artifacts/system_comparison/aggregate_results.csv"
REPRODUCIBILITY_BINDING = (
    PROJECT_ROOT / "artifacts/system_comparison/reproducibility_binding.json"
)
METADATA = Path("/home/ww/3RScan.json")
HORIZONS = (2, 3, 4, 5)
CURVE_HORIZONS = (4, 5)
IOU_THRESHOLDS = tuple(round(0.25 + 0.05 * index, 2) for index in range(14))
COVERAGE_THRESHOLDS = (0.25, 0.50, 0.75)
_SCIENTIFIC_PATHS = (
    "models",
    "datasets",
    "configs/system_comparison/persist4d_incumbent.yaml",
    "scripts/evaluate_persist4d.py",
    "scripts/evaluate_persist4d_p6a.py",
    "scripts/p6a_analysis.py",
    "scripts/p6a_association.py",
    "scripts/p6a_cache.py",
    "scripts/p6a_metrics.py",
    "scripts/p6a_protocol.py",
    "scripts/system_comparison_inference.py",
    "scripts/system_comparison_metrics.py",
    "scripts/system_comparison_protocol.py",
)


class DecompositionRunError(RuntimeError):
    """Raised when frozen evidence or output publication fails closed."""


def _read_json(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise DecompositionRunError(f"JSON input is unavailable: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DecompositionRunError(f"JSON input cannot be decoded: {path}") from error
    if not isinstance(value, dict):
        raise DecompositionRunError(f"JSON input must contain a mapping: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_bytes(value: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _publish_exact(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise FileExistsError(f"refusing to replace differing artifact: {path}")
        return
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


def _csv_bytes(rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _key_identity(value: Mapping[str, object]) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def _git(*arguments: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", *arguments],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
    )


def _validate_cache_lineage(
    *,
    frozen_cache_root: Path,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    local_path = frozen_cache_root / "persistent_predictions/manifest.json"
    full_path = frozen_cache_root / "full_history_predictions/manifest.json"
    local = _read_json(local_path)
    full = _read_json(full_path)
    tracked_local = _read_json(TRACKED_LOCAL_MANIFEST)
    tracked_full = _read_json(TRACKED_FULL_MANIFEST)
    for name, backup, tracked in (
        ("persistent", local, tracked_local),
        ("full-history", full, tracked_full),
    ):
        backup_entries = backup.get("entries")
        tracked_entries = tracked.get("entries")
        if not isinstance(backup_entries, list) or not isinstance(tracked_entries, list):
            raise DecompositionRunError(f"{name} manifest entries are invalid")
        backup_keys = {_key_identity(entry["key"]) for entry in backup_entries}
        tracked_keys = {_key_identity(entry["key"]) for entry in tracked_entries}
        if len(backup_keys) != 645 or backup_keys != tracked_keys:
            raise DecompositionRunError(f"{name} backup keys differ from frozen keys")
    local_provenance = local.get("provenance")
    tracked_local_provenance = tracked_local.get("provenance")
    full_provenance = full.get("provenance")
    tracked_full_provenance = tracked_full.get("provenance")
    if not all(
        isinstance(value, Mapping)
        for value in (
            local_provenance,
            tracked_local_provenance,
            full_provenance,
            tracked_full_provenance,
        )
    ):
        raise DecompositionRunError("cache provenance is invalid")
    for field in ("checkpoint_sha256", "config_sha256", "dataset_sha256"):
        if local_provenance.get(field) != tracked_local_provenance.get(field):
            raise DecompositionRunError(f"persistent {field} differs")
    for field in ("checkpoint_sha256", "config_sha256", "protocol_sha256"):
        if full_provenance.get(field) != tracked_full_provenance.get(field):
            raise DecompositionRunError(f"full-history {field} differs")
    backup_commit = str(local_provenance["source_commit"])
    tracked_commit = str(tracked_local_provenance["source_commit"])
    if backup_commit != str(full_provenance["source_commit"]):
        raise DecompositionRunError("backup cache source commits differ")
    if tracked_commit != str(tracked_full_provenance["source_commit"]):
        raise DecompositionRunError("tracked cache source commits differ")
    if _git("merge-base", "--is-ancestor", backup_commit, tracked_commit).returncode:
        raise DecompositionRunError("backup source is not an ancestor of frozen source")
    if _git("diff", "--quiet", f"{backup_commit}..{tracked_commit}", "--", *_SCIENTIFIC_PATHS).returncode:
        raise DecompositionRunError("scientific cache code changed across source lineage")
    backup_results = frozen_cache_root / "per_sequence_results.csv"
    tracked_results = PROJECT_ROOT / "artifacts/system_comparison/per_sequence_results.csv"
    if _sha256_file(backup_results) != _sha256_file(tracked_results):
        raise DecompositionRunError("backup and frozen per-sequence results differ")
    binding = {
        "backup_source_commit": backup_commit,
        "frozen_source_commit": tracked_commit,
        "persistent_manifest_file_sha256": _sha256_file(local_path),
        "full_history_manifest_file_sha256": _sha256_file(full_path),
        "frozen_per_sequence_results_sha256": _sha256_file(tracked_results),
    }
    return local, full, binding


def _mapped_classes(
    class_prob: torch.Tensor,
    *,
    background_class: int,
    class_mapper: object,
) -> torch.Tensor:
    if class_prob.ndim != 2 or not callable(class_mapper):
        raise DecompositionRunError("class probabilities or mapper are invalid")
    foreground = class_prob.detach().cpu().clone()
    foreground[:, background_class] = -float("inf")
    return torch.tensor(
        [class_mapper(int(value)) for value in foreground.argmax(dim=1).tolist()],
        dtype=torch.long,
    )


def _mapped_target_classes(values: torch.Tensor, class_mapper: object) -> torch.Tensor:
    if values.ndim != 1 or not callable(class_mapper):
        raise DecompositionRunError("target classes or mapper are invalid")
    return torch.tensor(
        [class_mapper(int(value)) for value in values.detach().cpu().tolist()],
        dtype=torch.long,
    )


def _aggregate_reference_rows() -> list[dict[str, object]]:
    with SYSTEM_AGGREGATE.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected = []
    for row in rows:
        if row["order_id"] != "all" or int(row["horizon"]) not in HORIZONS:
            continue
        selected.append(
            {
                "method": row["method"],
                "horizon": int(row["horizon"]),
                "t_mAP": float(row["causal_prefix_t_mAP"]),
                "t_mAP50": float(row["causal_prefix_t_mAP50"]),
                "t_mAP25": float(row["causal_prefix_t_mAP25"]),
                "t_REC": float(row["causal_prefix_t_REC"]),
                "t_REC50": float(row["causal_prefix_t_REC50"]),
                "t_REC25": float(row["causal_prefix_t_REC25"]),
                "diagnostic_semantics": "frozen primary system metric",
            }
        )
    if len(selected) != 8:
        raise DecompositionRunError("frozen system aggregate coverage differs")
    return selected


def _failure_definition(category: str) -> str:
    return {
        "local_observation_miss": "no valid raw candidate has positive spatial overlap",
        "class_failure": "spatial candidate exists but no class-compatible candidate overlaps",
        "high_iou_mask_failure": "class-compatible candidate overlaps but remains below IoU 0.50",
        "identity_fragmentation": "one GT entity is assigned multiple issued IDs",
        "identity_merge": "one issued ID is assigned to multiple GT entities",
        "wrong_gap_recovery": "a reactivation after absence recovers the wrong GT identity",
        "capacity_failure": "a valid birth is rejected by bounded state capacity",
        "unknown_unresolved": "available evidence does not uniquely identify one registered cause",
    }[category]


def run_decomposition(
    *,
    frozen_cache_root: Path,
    metadata_path: Path,
    output_root: Path = OUTPUT_ROOT,
) -> dict[str, object]:
    from scripts.evaluate_persist4d_p6a import (
        build_association_events,
        build_rio_class_mapper,
        build_tracker_factories,
        load_cached_protocol_sequences,
        prefix_causality_coordinator,
    )
    from scripts.p6a_metrics import (
        OfficialMetricAccumulator,
        build_offline_reconstructed_prediction,
    )
    from scripts.reviewer_closure_sidecar import (
        load_full_history_observation_sidecar_entry,
    )
    from scripts.run_system_comparison import _build_frozen_setup
    from scripts.system_comparison_analysis import _persistent_task_pair
    from scripts.system_comparison_inference import (
        load_full_history_cache_entry,
        unpack_bool_matrix,
    )
    from scripts.system_comparison_metrics import (
        causal_prefix_pair_from_payload,
        current_stage_pair,
    )

    frozen_cache_root = frozen_cache_root.resolve()
    _local_manifest, full_manifest, cache_binding = _validate_cache_lineage(
        frozen_cache_root=frozen_cache_root
    )
    reproducibility = _read_json(REPRODUCIBILITY_BINDING)
    setup = _build_frozen_setup(
        binding=reproducibility,
        metadata_path=metadata_path.resolve(),
        device_name=None,
    )
    sequences = load_cached_protocol_sequences(
        protocol=setup.protocol,
        cache_directory=frozen_cache_root / "persistent_predictions/entries",
        manifest_path=frozen_cache_root / "persistent_predictions/manifest.json",
    )
    if len(sequences) != 129:
        raise DecompositionRunError("persistent cache must contain 129 sequences")
    full_entries = {
        (
            str(entry["key"]["master_sequence_id"]),
            str(entry["key"]["order_id"]),
            int(entry["key"]["horizon"]),
        ): entry
        for entry in full_manifest["entries"]
    }
    sidecar_manifest = _read_json(SIDECAR_ROOT / "manifest.json")
    sidecar_entries = {
        (
            str(entry["key"]["master_sequence_id"]),
            str(entry["key"]["order_id"]),
            int(entry["key"]["horizon"]),
        ): entry
        for entry in sidecar_manifest["entries"]
    }
    if len(sidecar_entries) != 516:
        raise DecompositionRunError("Full-History sidecar must contain 516 entries")

    class_mapper = build_rio_class_mapper(setup.dataset)
    tracker_factory = build_tracker_factories(setup.p6a_config)["B4"]
    background_class = int(
        setup.p6a_config["baselines"]["b4"]["background_class"]
    )
    curve_metrics = {
        (method, horizon): OfficialTemporalCurveAccumulator(IOU_THRESHOLDS)
        for method in ("FullHistory", "Persist4D")
        for horizon in CURVE_HORIZONS
    }
    oracle_metrics = {
        horizon: OfficialMetricAccumulator(mode="strict_online") for horizon in HORIZONS
    }
    coverage_counts: Counter[tuple[str, int, float, str]] = Counter()
    coverage_totals: Counter[tuple[str, int, float]] = Counter()
    coverage_lookup: dict[tuple[str, str, int, int], str] = {}
    association_events = []

    for sequence in sequences:
        scope = (sequence.master_sequence_id, sequence.order_id)
        payloads = sequence.payloads
        coordinated = prefix_causality_coordinator(
            payloads,
            {"B4": tracker_factory},
            endpoints=(1, 2, 3, 4),
            sequence_id=f"{scope[0]}:{scope[1]}",
            background_class=background_class,
        )
        oracle = build_oracle_accumulator(
            payloads,
            sequence_id=f"{scope[0]}:{scope[1]}:oracle",
            background_class=background_class,
        )
        full_payloads = {
            horizon: load_full_history_cache_entry(
                frozen_cache_root / "full_history_predictions/entries",
                full_entries[(*scope, horizon)],
                expected_provenance=full_manifest["provenance"],
            )
            for horizon in HORIZONS
        }

        for stage, payload in enumerate(payloads):
            observation = payload["observation"]
            target = payload["target"]
            prediction_classes = _mapped_classes(
                observation["class_prob"],
                background_class=background_class,
                class_mapper=class_mapper,
            )
            target_classes = _mapped_target_classes(target["gt_classes"], class_mapper)
            horizon = stage + 1
            for threshold in COVERAGE_THRESHOLDS:
                outcomes = classify_observation_coverage(
                    prediction_masks=observation["masks"],
                    prediction_classes=prediction_classes,
                    valid=observation["valid"],
                    target_masks=target["gt_masks"],
                    target_classes=target_classes,
                    threshold=threshold,
                )
                if horizon in HORIZONS:
                    for category in outcomes:
                        coverage_counts[("Persist4D", horizon, threshold, category)] += 1
                    coverage_totals[("Persist4D", horizon, threshold)] += len(outcomes)
                if math.isclose(threshold, 0.50):
                    for gt_id, category in zip(target["gt_ids"].tolist(), outcomes, strict=True):
                        coverage_lookup[(scope[0], scope[1], stage, int(gt_id))] = category

        for horizon in HORIZONS:
            persistent_pair = _persistent_task_pair(
                payloads=payloads,
                prediction=coordinated.online_predictions["B4"][horizon - 1],
                horizon=horizon,
                class_mapper=class_mapper,
            )
            full_pair = causal_prefix_pair_from_payload(full_payloads[horizon])
            oracle_pair = _persistent_task_pair(
                payloads=payloads,
                prediction=build_offline_reconstructed_prediction(
                    oracle, endpoint=horizon - 1
                ),
                horizon=horizon,
                class_mapper=class_mapper,
            )
            oracle_metrics[horizon].update(oracle_pair.prediction, oracle_pair.target)
            if horizon in CURVE_HORIZONS:
                curve_metrics[("Persist4D", horizon)].update(
                    persistent_pair.prediction, persistent_pair.target
                )
                curve_metrics[("FullHistory", horizon)].update(
                    full_pair.prediction, full_pair.target
                )
            association_events.extend(
                build_association_events(
                    payloads[:horizon],
                    coordinated.online_steps["B4"][horizon - 1],
                    method="B4",
                    reference_scene_id=sequence.reference_scene_id,
                    master_sequence_id=scope[0],
                    order_id=scope[1],
                    prefix=horizon,
                    cache_digest=coordinated.content_digest,
                    background_class=background_class,
                )
            )

            sidecar = load_full_history_observation_sidecar_entry(
                SIDECAR_ROOT / "entries", sidecar_entries[(*scope, horizon)]
            )
            raw = sidecar["observation"]
            current = current_stage_pair(full_pair)
            full_prediction_classes = _mapped_classes(
                raw["class_prob"],
                background_class=background_class,
                class_mapper=class_mapper,
            )
            full_masks = unpack_bool_matrix(raw["current_stage_masks"])
            for threshold in COVERAGE_THRESHOLDS:
                outcomes = classify_observation_coverage(
                    prediction_masks=full_masks,
                    prediction_classes=full_prediction_classes,
                    valid=raw["valid"],
                    target_masks=current.target["masks"],
                    target_classes=current.target["labels"],
                    threshold=threshold,
                )
                for category in outcomes:
                    coverage_counts[("FullHistory", horizon, threshold, category)] += 1
                coverage_totals[("FullHistory", horizon, threshold)] += len(outcomes)

    sweep_rows = []
    for method in ("FullHistory", "Persist4D"):
        for horizon in CURVE_HORIZONS:
            values = curve_metrics[(method, horizon)].compute()
            sweep_rows.extend(
                {
                    "method": method,
                    "horizon": horizon,
                    "iou_threshold": threshold,
                    "temporal_ap": values[threshold],
                    "aggregation": "pooled class-macro official stmetrics",
                    "sequence_count": 129,
                }
                for threshold in IOU_THRESHOLDS
            )

    coverage_rows = []
    for method in ("FullHistory", "Persist4D"):
        for horizon in HORIZONS:
            for threshold in COVERAGE_THRESHOLDS:
                total = coverage_totals[(method, horizon, threshold)]
                if total <= 0:
                    raise DecompositionRunError("coverage population is empty")
                for category in COVERAGE_CATEGORIES:
                    count = coverage_counts[(method, horizon, threshold, category)]
                    coverage_rows.append(
                        {
                            "method": method,
                            "horizon": horizon,
                            "iou_threshold": threshold,
                            "category": category,
                            "count": count,
                            "total_gt_entity_stages": total,
                            "fraction": count / total,
                        }
                    )

    oracle_rows = _aggregate_reference_rows()
    for horizon in HORIZONS:
        values = oracle_metrics[horizon].compute()
        oracle_rows.append(
            {
                "method": "Oracle",
                "horizon": horizon,
                "t_mAP": values["online_t-mAP"],
                "t_mAP50": values["online_t-mAP50"],
                "t_mAP25": values["online_t-mAP25"],
                "t_REC": values["online_t-REC"],
                "t_REC50": values["online_t-REC50"],
                "t_REC25": values["online_t-REC25"],
                "diagnostic_semantics": (
                    "P6-A offline GT-ID readout; unmatched candidates retained; "
                    "masks/classes unchanged"
                ),
            }
        )
    oracle_rows.sort(key=lambda row: (int(row["horizon"]), str(row["method"])))

    failure_counts: Counter[tuple[int, str]] = Counter()
    failure_totals: Counter[int] = Counter()
    for event in association_events:
        if event.prefix not in CURVE_HORIZONS or event.is_failure is not True:
            continue
        coverage = None
        if event.gt_entity_id is not None:
            coverage = coverage_lookup.get(
                (
                    event.master_sequence_id,
                    event.order_id,
                    event.stage_id,
                    int(event.gt_entity_id),
                )
            )
        category = classify_decomposition_failure(
            event, coverage_category=coverage
        )
        failure_counts[(event.prefix, category)] += 1
        failure_totals[event.prefix] += 1
    failure_rows = []
    for horizon in CURVE_HORIZONS:
        total = failure_totals[horizon]
        if total <= 0:
            raise DecompositionRunError("failure population is empty")
        for category in FAILURE_CATEGORIES:
            count = failure_counts[(horizon, category)]
            failure_rows.append(
                {
                    "method": "Persist4D",
                    "horizon": horizon,
                    "category": category,
                    "count": count,
                    "total_failure_events": total,
                    "fraction": count / total,
                    "operational_definition": _failure_definition(category),
                }
            )

    metric_index = {
        (str(row["method"]), int(row["horizon"])): float(row["t_mAP"])
        for row in oracle_rows
    }
    ceiling = classify_ceiling(
        persistent={horizon: metric_index[("Persist4D", horizon)] for horizon in CURVE_HORIZONS},
        full_history={horizon: metric_index[("FullHistory", horizon)] for horizon in CURVE_HORIZONS},
        oracle={horizon: metric_index[("Oracle", horizon)] for horizon in CURVE_HORIZONS},
    )

    _publish_exact(
        output_root / "tmap_iou_sweep.csv",
        _csv_bytes(
            sweep_rows,
            (
                "method",
                "horizon",
                "iou_threshold",
                "temporal_ap",
                "aggregation",
                "sequence_count",
            ),
        ),
    )
    _publish_exact(
        output_root / "observation_coverage.csv",
        _csv_bytes(
            coverage_rows,
            (
                "method",
                "horizon",
                "iou_threshold",
                "category",
                "count",
                "total_gt_entity_stages",
                "fraction",
            ),
        ),
    )
    _publish_exact(
        output_root / "oracle_association_results.csv",
        _csv_bytes(
            oracle_rows,
            (
                "method",
                "horizon",
                "t_mAP",
                "t_mAP50",
                "t_mAP25",
                "t_REC",
                "t_REC50",
                "t_REC25",
                "diagnostic_semantics",
            ),
        ),
    )
    _publish_exact(
        output_root / "failure_decomposition.csv",
        _csv_bytes(
            failure_rows,
            (
                "method",
                "horizon",
                "category",
                "count",
                "total_failure_events",
                "fraction",
                "operational_definition",
            ),
        ),
    )
    aggregation_note = f"""# Metric Aggregation Note

- `tmap_iou_sweep.csv` pools all 129 sequence/order scopes inside one official `stmetrics` accumulator per method and horizon, then reports class-macro temporal AP. It is not a mean of per-sequence AP values.
- `observation_coverage.csv` is a micro-average over GT entity/stage instances. Categories are mutually exclusive at each threshold.
- `failure_decomposition.csv` uses prefix-specific P6-A failure events. Each event receives exactly one primary category; insufficient evidence remains `unknown_unresolved`.
- GT is used only after frozen inference for Oracle identity assignment. Masks, classes, scores, features, and model forward outputs are unchanged.
- Oracle follows the existing P6-A offline diagnostic: unmatched valid candidates remain stage-unique births rather than being removed with GT.
- Oracle ceiling gate: minimum absolute t-mAP gain `0.05` and at least `50%` closure of a positive Full-History gap at T4 or T5.
- Phase III classification: `{ceiling}`.
"""
    _publish_exact(
        output_root / "METRIC_AGGREGATION_NOTE.md", aggregation_note.encode("utf-8")
    )
    manifest: dict[str, object] = {
        "schema_version": 1,
        "status": "pass",
        "source_commit": _git("rev-parse", "HEAD").stdout.decode().strip(),
        "cache_binding": cache_binding,
        "sidecar_manifest_content_sha256": sidecar_manifest["content_sha256"],
        "iou_thresholds": list(IOU_THRESHOLDS),
        "coverage_thresholds": list(COVERAGE_THRESHOLDS),
        "sequence_count": len(sequences),
        "sweep_row_count": len(sweep_rows),
        "coverage_row_count": len(coverage_rows),
        "oracle_row_count": len(oracle_rows),
        "failure_row_count": len(failure_rows),
        "oracle_policy": "p6a_offline_gt_ids_unmatched_candidates_retained",
        "ceiling_classification": ceiling,
        "living_scenes_triggered": ceiling == "ASSOCIATION_CEILING",
    }
    manifest["content_sha256"] = hashlib.sha256(_canonical_json_bytes(manifest)).hexdigest()
    _publish_exact(output_root / "phase_iii_manifest.json", _canonical_json_bytes(manifest))
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-cache-root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=METADATA)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_decomposition(
        frozen_cache_root=args.frozen_cache_root,
        metadata_path=args.metadata,
        output_root=args.output_root,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
