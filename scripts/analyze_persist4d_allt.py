#!/usr/bin/env python3
"""Analyze frozen R1 caches at every T without model reinference."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import sys
import tempfile
from collections import defaultdict
from collections.abc import Hashable, Mapping, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.p6a_metrics import OfficialMetricAccumulator
from scripts.persist4d_allt_contract import canonical_json_sha256
from scripts.system_comparison_metrics import CausalPrefixPair, CausalTaskAccumulator

R1_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts/r1_downstream_validation_v1"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts/allt_task_superiority_v1/baseline"
DEFAULT_CACHE_ROOT = Path("/mnt/shared/ww/persist4d-r1-downstream-validation-v1/cache")
DEFAULT_CHECKPOINT = Path(
    "/mnt/shared/ww/persist4d-rescene-finalization/checkpoints/"
    "629ff7624dcac15e6022906e808e2e05b3ec61c60a1116ab0e278f0cfd2368dd.ckpt"
)
DEFAULT_PRETRAINED = Path(
    "/mnt/shared/ww/persist4d-node1-7-migration/checkpoints/concerto_base.pth"
)
DEFAULT_METADATA = Path("/home/ww/3RScan.json")
REPORT_HORIZONS = (2, 3, 4, 5)
REDUCERS = ("mean", "latest", "max")
OUTPUT_FIELDS = (
    "population_id",
    "model",
    "checkpoint_sha256",
    "training_seed",
    "evaluation_seed",
    "method",
    "reducer",
    "T",
    "t_mAP",
    "t_mAP50",
    "t_mAP25",
    "t_REC",
    "prefix_overall_mAP",
    "local_current_AP",
    "num_master",
    "num_order_units",
    "num_reference_clusters",
)


class AllTBaselineError(RuntimeError):
    """Raised when the frozen all-T baseline cannot be proven."""


class AllTBaselineAccumulator:
    """Accumulate temporal, current-stage, and full-prefix legacy metrics."""

    def __init__(
        self, *, dataset_spec: str | Path, min_region_size: int = 100
    ) -> None:
        from stmetrics import InstanceMetrics, LegacyAPEvaluator

        specification = Path(dataset_spec)
        if not specification.is_file():
            raise AllTBaselineError("dataset specification is unavailable")
        if (
            isinstance(min_region_size, bool)
            or not isinstance(min_region_size, int)
            or min_region_size <= 0
        ):
            raise AllTBaselineError("min_region_size must be a positive integer")
        self._task = CausalTaskAccumulator(
            metric_factory=lambda mode: OfficialMetricAccumulator(
                mode=mode,
                dataset_spec=specification,
                min_region_size=min_region_size,
            )
        )
        self._legacy = InstanceMetrics(
            dataset=str(specification),
            heads=[LegacyAPEvaluator(recall=True, aux="changes")],
            log_prefix="val",
            min_region_size=min_region_size,
            timestep_key="temporal_stages",
        )
        self._count = 0

    def update(self, pair: CausalPrefixPair) -> None:
        if not isinstance(pair, CausalPrefixPair):
            raise AllTBaselineError("all-T metric input must be a causal prefix pair")
        self._task.update(pair)
        self._legacy.update([pair.prediction], [pair.target])
        self._count += 1

    def compute(self) -> dict[str, float]:
        if not self._count:
            raise AllTBaselineError("all-T metric accumulator is empty")
        task = self._task.compute()
        legacy = self._legacy.compute()
        values = {
            "local_current_AP": task["current_stage_AP"],
            "prefix_overall_mAP": legacy["val_mean_AP"],
            "t_REC": task["causal_prefix_t_REC"],
            "t_mAP": task["causal_prefix_t_mAP"],
            "t_mAP25": task["causal_prefix_t_mAP25"],
            "t_mAP50": task["causal_prefix_t_mAP50"],
        }
        result = {
            key: float(value.detach().cpu().item() if hasattr(value, "detach") else value)
            for key, value in values.items()
        }
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in result.values()):
            raise AllTBaselineError("all-T metric values must be finite rates")
        return result


def validate_legacy_regression(
    new_rows: Sequence[Mapping[str, Any]],
    old_rows: Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    metric_mapping = {
        "local_current_AP": "current_stage_AP",
        "t_REC": "causal_prefix_t_REC",
        "t_mAP": "causal_prefix_t_mAP",
        "t_mAP25": "causal_prefix_t_mAP25",
        "t_mAP50": "causal_prefix_t_mAP50",
    }
    new_index: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    for row in new_rows:
        key = (str(row.get("method")), str(row.get("reducer")), int(row.get("T")))
        if key in new_index:
            raise AllTBaselineError("new all-T rows contain duplicate cells")
        new_index[key] = row

    checked_cells = 0
    checked_values = 0
    for old in old_rows:
        if old.get("order_id") != "all" or int(old.get("horizon")) not in {2, 4, 5}:
            continue
        key = (
            str(old.get("method")),
            str(old.get("score_reducer")),
            int(old.get("horizon")),
        )
        current = new_index.get(key)
        if current is None:
            raise AllTBaselineError(f"legacy regression cell is missing: {key}")
        for new_field, old_field in metric_mapping.items():
            if float(current[new_field]) != float(old[old_field]):
                raise AllTBaselineError(
                    f"legacy metric regression at {key} field {new_field}"
                )
            checked_values += 1
        checked_cells += 1
    if not checked_cells:
        raise AllTBaselineError("legacy regression has no comparable cells")
    return {
        "checked_cells": checked_cells,
        "checked_values": checked_values,
        "status": "pass",
    }


def plan_missing_cache_keys(
    expected: Sequence[Hashable], observed: Sequence[Hashable]
) -> list[Hashable]:
    if len(set(expected)) != len(expected) or len(set(observed)) != len(observed):
        raise AllTBaselineError("cache keys must be unique")
    expected_set = set(expected)
    unexpected = set(observed) - expected_set
    if unexpected:
        raise AllTBaselineError("cache contains unexpected keys")
    observed_set = set(observed)
    return [key for key in expected if key not in observed_set]


def _git_head() -> str:
    import subprocess

    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if len(result) != 40:
        raise AllTBaselineError("Git HEAD is invalid")
    return result


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    if path.is_symlink() or not path.is_file():
        raise AllTBaselineError(f"required CSV is unavailable: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _csv_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    if not rows:
        raise AllTBaselineError("CSV output cannot be empty")
    if any(tuple(row) != OUTPUT_FIELDS for row in rows):
        raise AllTBaselineError("all-T CSV schema differs")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=OUTPUT_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise AllTBaselineError(f"output must not be a symlink: {path}")
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
    temporary.replace(path)


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    encoded = (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")
    _atomic_write(path, encoded)


def _cache_key(record: Mapping[str, object], *, local: bool) -> tuple[str, str, int]:
    key = record.get("key")
    if not isinstance(key, Mapping):
        raise AllTBaselineError("cache record lacks a key")
    horizon_field = "stage_index" if local else "horizon"
    horizon = int(key[horizon_field]) + (1 if local else 0)
    return str(key["master_sequence_id"]), str(key["order_id"]), horizon


def run_baseline_analysis(
    *,
    cache_root: Path,
    output_root: Path,
    checkpoint: Path,
    pretrained: Path,
    metadata: Path,
) -> dict[str, object]:
    from scripts.analyze_r1_downstream_validation import (
        _validate_new_cache_binding,
        resolve_metric_dataset_spec,
    )
    from scripts.evaluate_persist4d_p6a import (
        build_rio_class_mapper,
        build_tracker_factories,
        cache_payload_to_frozen_observation,
    )
    from scripts.r1_downstream_context import build_r1_setup
    from scripts.system_comparison_inference import load_full_history_cache_entry
    from scripts.system_comparison_metrics import causal_prefix_pair_from_payload
    from scripts.system_comparison_v2_analysis import (
        build_v2_causal_pair,
        load_v2_sequences,
    )
    from scripts.system_comparison_v2_inference import (
        OfficialCandidateTrajectoryAccumulator,
    )
    from scripts.system_comparison_v3_identity import run_fresh_tracker_steps
    from scripts.system_comparison_v3_score_sensitivity import (
        assert_score_only_snapshots,
    )

    compact, local_progress, full_progress = _validate_new_cache_binding(
        cache_root=cache_root, artifact_root=R1_ARTIFACT_ROOT
    )
    setup = build_r1_setup(
        contract_path=R1_ARTIFACT_ROOT / "EXPERIMENT_CONTRACT.md",
        protocol_path=PROJECT_ROOT / "artifacts/P6A/protocol_b_manifest.json",
        checkpoint_path=checkpoint,
        pretrained_path=pretrained,
        metadata_path=metadata,
        data_root=PROJECT_ROOT,
        source_commit=_git_head(),
        device_name=None,
    )
    dataset_spec = resolve_metric_dataset_spec(PROJECT_ROOT)
    sequences = load_v2_sequences(
        cache_manifest={**local_progress, "entry_count": 645},
        cache_root=cache_root,
    )
    full_entries = {
        _cache_key(record, local=False): record for record in full_progress["records"]
    }
    local_keys = [_cache_key(record, local=True) for record in local_progress["records"]]
    full_keys = list(full_entries)
    expected_keys = [
        (sequence.master_sequence_id, sequence.order_id, horizon)
        for sequence in sequences
        for horizon in range(1, 6)
    ]
    missing_local = plan_missing_cache_keys(expected_keys, local_keys)
    missing_full = plan_missing_cache_keys(expected_keys, full_keys)
    if missing_local or missing_full:
        raise AllTBaselineError("frozen cache lacks required keys")

    factories = build_tracker_factories(setup.p6a_config)
    class_mapper = build_rio_class_mapper(setup.dataset)
    accumulators: dict[tuple[str, str, int], AllTBaselineAccumulator] = {}
    counts: dict[tuple[str, str, int], int] = defaultdict(int)

    def update(method: str, reducer: str, horizon: int, pair: CausalPrefixPair) -> None:
        key = (method, reducer, horizon)
        if key not in accumulators:
            accumulators[key] = AllTBaselineAccumulator(dataset_spec=dataset_spec)
        accumulators[key].update(pair)
        counts[key] += 1

    for sequence_index, sequence in enumerate(sequences, start=1):
        observations = tuple(
            cache_payload_to_frozen_observation(raw) for raw in sequence.raw_payloads
        )
        steps = {
            method: run_fresh_tracker_steps(
                factory=factories[method],
                observations=observations,
                sequence_id=f"{sequence.master_sequence_id}:{sequence.order_id}",
            )
            for method in ("B2", "B4")
        }
        trajectories = {
            method: {
                reducer: OfficialCandidateTrajectoryAccumulator(score_reducer=reducer)
                for reducer in REDUCERS
            }
            for method in ("B2", "B4")
        }
        for stage, sidecar in enumerate(sequence.sidecars):
            horizon = stage + 1
            snapshots = {}
            for method in ("B2", "B4"):
                method_snapshots = {}
                for reducer in REDUCERS:
                    trajectory = trajectories[method][reducer]
                    trajectory.add_stage(sidecar, steps[method][stage])
                    method_snapshots[reducer] = trajectory.snapshot()
                assert_score_only_snapshots(method_snapshots)
                snapshots[method] = method_snapshots
            if horizon not in REPORT_HORIZONS:
                continue

            full_record = full_entries[
                (sequence.master_sequence_id, sequence.order_id, horizon)
            ]
            full_payload = load_full_history_cache_entry(
                cache_root / "full_history/entries",
                full_record,
                expected_provenance=full_progress["provenance"],
            )
            update(
                "FullHistory",
                "official",
                horizon,
                causal_prefix_pair_from_payload(full_payload),
            )
            for method in ("B2", "B4"):
                for reducer in REDUCERS:
                    update(
                        method,
                        reducer,
                        horizon,
                        build_v2_causal_pair(
                            snapshot=snapshots[method][reducer],
                            raw_payloads=sequence.raw_payloads[:horizon],
                            class_mapper=class_mapper,
                        ),
                    )
        if sequence_index % 10 == 0 or sequence_index == len(sequences):
            print(
                f"[allt-baseline] completed {sequence_index}/{len(sequences)} sequences",
                file=sys.stderr,
                flush=True,
            )

    expected_cells = {
        ("FullHistory", "official", horizon) for horizon in REPORT_HORIZONS
    } | {
        (method, reducer, horizon)
        for method in ("B2", "B4")
        for reducer in REDUCERS
        for horizon in REPORT_HORIZONS
    }
    if set(accumulators) != expected_cells or any(
        counts[key] != 129 for key in expected_cells
    ):
        raise AllTBaselineError("all-T aggregate coverage differs")

    rows = []
    for method, reducer, horizon in sorted(expected_cells):
        values = accumulators[(method, reducer, horizon)].compute()
        rows.append(
            {
                "population_id": "protocol_b_129_units",
                "model": "R1_epoch390",
                "checkpoint_sha256": compact["checkpoint_sha256"],
                "training_seed": 45,
                "evaluation_seed": 45,
                "method": method,
                "reducer": reducer,
                "T": horizon,
                **values,
                "num_master": 43,
                "num_order_units": counts[(method, reducer, horizon)],
                "num_reference_clusters": 6,
            }
        )

    old_task_path = R1_ARTIFACT_ROOT / "metrics/task_aggregate.csv"
    regression = validate_legacy_regression(rows, _read_csv(old_task_path))
    metrics_bytes = _csv_bytes(rows)
    _atomic_write(output_root / "all_t_metrics.csv", metrics_bytes)
    replay = {
        "cache_reads": {"full_history": 645, "local": 645},
        "checkpoint_sha256": compact["checkpoint_sha256"],
        "derived_metric_rows": len(rows),
        "legacy_metric_source_sha256": _file_sha256(old_task_path),
        "legacy_regression": regression,
        "missing_cache_keys": {"full_history": [], "local": []},
        "model_forward_count": 0,
        "new_prediction_count": 0,
        "protocol_sha256": compact["protocol_sha256"],
        "schema_version": 1,
        "source_commit": _git_head(),
        "status": "pass",
    }
    replay["content_sha256"] = canonical_json_sha256(replay)
    _atomic_json(output_root / "replay_status.json", replay)
    return {
        "all_t_metric_rows": len(rows),
        "legacy_regression": regression,
        "output_sha256": hashlib.sha256(metrics_bytes).hexdigest(),
        "status": "pass",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--pretrained", type=Path, default=DEFAULT_PRETRAINED)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_baseline_analysis(
        cache_root=args.cache_root,
        output_root=args.output_root,
        checkpoint=args.checkpoint,
        pretrained=args.pretrained,
        metadata=args.metadata,
    )
    print(json.dumps(result, allow_nan=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
