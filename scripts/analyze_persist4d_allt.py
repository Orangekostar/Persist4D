#!/usr/bin/env python3
"""Analyze frozen R1 caches at every T without model reinference."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import multiprocessing
import sys
import tempfile
from collections.abc import Hashable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.p6a_metrics import OfficialMetricAccumulator
from scripts.persist4d_allt_contract import canonical_json_sha256
from scripts.system_comparison_metrics import CausalPrefixPair, CausalTaskAccumulator

R1_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts/r1_downstream_validation_v1"
DEFAULT_R1_CONTRACT = PROJECT_ROOT / "configs/r1_downstream_validation/default.yaml"
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
_WORKER_CONTEXT: dict[str, object] | None = None


class AllTBaselineError(RuntimeError):
    """Raised when the frozen all-T baseline cannot be proven."""


class _PrefixOverallAccumulator:
    mode = "prefix_overall"

    def __init__(self, *, dataset_spec: Path, min_region_size: int) -> None:
        from stmetrics import InstanceMetrics, LegacyAPEvaluator

        self._metric = InstanceMetrics(
            dataset=str(dataset_spec),
            heads=[LegacyAPEvaluator(recall=True, aux="changes")],
            log_prefix="val",
            min_region_size=min_region_size,
            timestep_key="temporal_stages",
        )
        self._updates = 0

    def update(self, pair: CausalPrefixPair) -> None:
        self._metric.update([pair.prediction], [pair.target])
        self._updates += 1

    def compute(self) -> float:
        if not self._updates:
            raise AllTBaselineError("prefix-overall metric accumulator is empty")
        value = self._metric.compute()["val_mean_AP"]
        return float(value.detach().cpu().item() if hasattr(value, "detach") else value)


class AllTBaselineAccumulator:
    """Accumulate temporal, current-stage, and full-prefix legacy metrics."""

    def __init__(
        self,
        *,
        dataset_spec: str | Path,
        min_region_size: int = 100,
        include_task: bool = True,
    ) -> None:
        specification = Path(dataset_spec)
        if not specification.is_file():
            raise AllTBaselineError("dataset specification is unavailable")
        if (
            isinstance(min_region_size, bool)
            or not isinstance(min_region_size, int)
            or min_region_size <= 0
        ):
            raise AllTBaselineError("min_region_size must be a positive integer")
        if type(include_task) is not bool:
            raise AllTBaselineError("include_task must be a boolean")
        self._task = (
            CausalTaskAccumulator(
                metric_factory=lambda mode: OfficialMetricAccumulator(
                    mode=mode,
                    dataset_spec=specification,
                    min_region_size=min_region_size,
                )
            )
            if include_task
            else None
        )
        self._legacy = _PrefixOverallAccumulator(
            dataset_spec=specification, min_region_size=min_region_size
        )
        self._count = 0

    def update(self, pair: CausalPrefixPair) -> None:
        if not isinstance(pair, CausalPrefixPair):
            raise AllTBaselineError("all-T metric input must be a causal prefix pair")
        if self._task is not None:
            self._task.update(pair)
        self._legacy.update(pair)
        self._count += 1

    @property
    def sequence_count(self) -> int:
        return self._count

    def merge(self, other: AllTBaselineAccumulator) -> None:
        from scripts.analyze_r1_downstream_validation import (
            _merge_causal_task_accumulators,
            merge_official_metric_accumulators,
        )

        if not isinstance(other, AllTBaselineAccumulator) or (
            (self._task is None) != (other._task is None)
        ):
            raise AllTBaselineError("all-T accumulators are incompatible")
        if self._task is not None:
            _merge_causal_task_accumulators(self._task, other._task)
        merge_official_metric_accumulators(self._legacy, other._legacy)
        self._count += other._count

    def compute(self) -> dict[str, float]:
        if not self._count:
            raise AllTBaselineError("all-T metric accumulator is empty")
        values = {"prefix_overall_mAP": self._legacy.compute()}
        if self._task is not None:
            task = self._task.compute()
            values.update(
                {
                    "local_current_AP": task["current_stage_AP"],
                    "t_REC": task["causal_prefix_t_REC"],
                    "t_mAP": task["causal_prefix_t_mAP"],
                    "t_mAP25": task["causal_prefix_t_mAP25"],
                    "t_mAP50": task["causal_prefix_t_mAP50"],
                }
            )
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


def _build_sequence_jobs(
    *,
    local_records: Sequence[Mapping[str, object]],
    full_records: Sequence[Mapping[str, object]],
    expected_keys: Sequence[tuple[str, str, int]],
) -> list[dict[str, object]]:
    local_index = {_cache_key(record, local=True): record for record in local_records}
    full_index = {_cache_key(record, local=False): record for record in full_records}
    identities = []
    seen = set()
    for master, order, _ in expected_keys:
        identity = (master, order)
        if identity not in seen:
            identities.append(identity)
            seen.add(identity)
    jobs = []
    for master, order in identities:
        local = [local_index[(master, order, horizon)] for horizon in range(1, 6)]
        full = [full_index[(master, order, horizon)] for horizon in REPORT_HORIZONS]
        references = {
            str(record["key"]["reference_scene_id"]) for record in [*local, *full]
        }
        if len(references) != 1:
            raise AllTBaselineError("cache sequence reference identity differs")
        jobs.append(
            {
                "full_records": full,
                "local_records": local,
                "master_sequence_id": master,
                "order_id": order,
                "reference_scene_id": references.pop(),
            }
        )
    if len(jobs) != 129:
        raise AllTBaselineError("cache sequence job coverage differs")
    return jobs


def _initialize_worker(
    cache_root: str,
    dataset_spec: str,
    full_provenance: Mapping[str, object],
    p6a_config: Mapping[str, object],
    rio_class_mapping: Sequence[int],
) -> None:
    import torch

    global _WORKER_CONTEXT
    torch.set_num_threads(1)
    _WORKER_CONTEXT = {
        "cache_root": cache_root,
        "dataset_spec": dataset_spec,
        "full_provenance": dict(full_provenance),
        "p6a_config": dict(p6a_config),
        "rio_class_mapping": tuple(int(value) for value in rio_class_mapping),
    }


def _load_worker_sequence(job: Mapping[str, object]) -> object:
    from scripts.p6a_cache import validate_cache_entry
    from scripts.system_comparison_v2_analysis import CachedV2Sequence
    from scripts.system_comparison_v2_cache import (
        load_task_sidecar,
        observation_fingerprint,
    )

    if _WORKER_CONTEXT is None:
        raise AllTBaselineError("baseline worker is not initialized")
    cache_root = Path(str(_WORKER_CONTEXT["cache_root"]))
    raw_payloads = []
    sidecars = []
    for record in job["local_records"]:
        key = record["key"]
        raw_entry = record["raw_entry"]
        sidecar_entry = record["sidecar_entry"]
        raw = validate_cache_entry(
            cache_root / "raw_predictions/entries" / str(raw_entry["filename"]),
            raw_entry,
        )
        sidecar = load_task_sidecar(
            cache_root / "task_sidecars/entries" / str(sidecar_entry["filename"])
        )
        fingerprint = observation_fingerprint(raw)
        if (
            raw["key"] != key
            or sidecar["key"] != key
            or fingerprint != record["raw_observation_fingerprint"]
            or fingerprint
            != sidecar["provenance"]["source_raw_observation_fingerprint"]
        ):
            raise AllTBaselineError("worker raw/sidecar cache binding differs")
        raw_payloads.append(raw)
        sidecars.append(sidecar)
    return CachedV2Sequence(
        reference_scene_id=str(job["reference_scene_id"]),
        master_sequence_id=str(job["master_sequence_id"]),
        order_id=str(job["order_id"]),
        raw_payloads=tuple(raw_payloads),
        sidecars=tuple(sidecars),
    )


def _process_sequence_shard(
    jobs: Sequence[Mapping[str, object]],
) -> tuple[int, dict[tuple[str, str, int], AllTBaselineAccumulator]]:
    from scripts.evaluate_persist4d_p6a import (
        build_tracker_factories,
        cache_payload_to_frozen_observation,
    )
    from scripts.system_comparison_inference import load_full_history_cache_entry
    from scripts.system_comparison_metrics import causal_prefix_pair_from_payload
    from scripts.system_comparison_v2_analysis import build_v2_causal_pair
    from scripts.system_comparison_v2_inference import (
        OfficialCandidateTrajectoryAccumulator,
    )
    from scripts.system_comparison_v3_identity import run_fresh_tracker_steps
    from scripts.system_comparison_v3_score_sensitivity import (
        assert_score_only_snapshots,
    )

    if _WORKER_CONTEXT is None:
        raise AllTBaselineError("baseline worker is not initialized")
    cache_root = Path(str(_WORKER_CONTEXT["cache_root"]))
    dataset_spec = Path(str(_WORKER_CONTEXT["dataset_spec"]))
    full_provenance = _WORKER_CONTEXT["full_provenance"]
    factories = build_tracker_factories(_WORKER_CONTEXT["p6a_config"])
    rio_class_mapping = _WORKER_CONTEXT["rio_class_mapping"]

    def class_mapper(model_class: int) -> int:
        if not 0 <= model_class < len(rio_class_mapping):
            raise AllTBaselineError("model class is outside RIO mapping")
        return int(rio_class_mapping[model_class])

    accumulators: dict[tuple[str, str, int], AllTBaselineAccumulator] = {}

    def update(method: str, reducer: str, horizon: int, pair: CausalPrefixPair) -> None:
        key = (method, reducer, horizon)
        incoming = AllTBaselineAccumulator(
            dataset_spec=dataset_spec, include_task=horizon == 3
        )
        incoming.update(pair)
        if key not in accumulators:
            accumulators[key] = incoming
        else:
            accumulators[key].merge(incoming)

    for job in jobs:
        sequence = _load_worker_sequence(job)
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
        full_by_horizon = {
            _cache_key(record, local=False)[2]: record
            for record in job["full_records"]
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
            full_payload = load_full_history_cache_entry(
                cache_root / "full_history/entries",
                full_by_horizon[horizon],
                expected_provenance=full_provenance,
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
    return len(jobs), accumulators


def run_baseline_analysis(
    *,
    cache_root: Path,
    output_root: Path,
    checkpoint: Path,
    pretrained: Path,
    metadata: Path,
    workers: int,
) -> dict[str, object]:
    from scripts.analyze_r1_downstream_validation import (
        _validate_new_cache_binding,
        resolve_metric_dataset_spec,
    )
    from scripts.evaluate_persist4d_p6a import (
        build_rio_class_mapper,
        expected_cache_keys,
    )
    from scripts.r1_downstream_context import build_r1_setup

    compact, local_progress, full_progress = _validate_new_cache_binding(
        cache_root=cache_root, artifact_root=R1_ARTIFACT_ROOT
    )
    setup = build_r1_setup(
        contract_path=DEFAULT_R1_CONTRACT,
        protocol_path=PROJECT_ROOT / "artifacts/P6A/protocol_b_manifest.json",
        checkpoint_path=checkpoint,
        pretrained_path=pretrained,
        metadata_path=metadata,
        data_root=PROJECT_ROOT,
        source_commit=_git_head(),
        device_name=None,
    )
    dataset_spec = resolve_metric_dataset_spec(PROJECT_ROOT)
    local_keys = [_cache_key(record, local=True) for record in local_progress["records"]]
    full_keys = [
        _cache_key(record, local=False) for record in full_progress["records"]
    ]
    expected_keys = [
        (
            str(key["master_sequence_id"]),
            str(key["order_id"]),
            int(key["stage_index"]) + 1,
        )
        for key in expected_cache_keys(setup.protocol)
    ]
    missing_local = plan_missing_cache_keys(expected_keys, local_keys)
    missing_full = plan_missing_cache_keys(expected_keys, full_keys)
    if missing_local or missing_full:
        raise AllTBaselineError("frozen cache lacks required keys")
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 32:
        raise AllTBaselineError("workers must be within 1-32")

    class_mapper = build_rio_class_mapper(setup.dataset)
    rio_class_mapping = tuple(class_mapper(index) for index in range(18))
    jobs = _build_sequence_jobs(
        local_records=local_progress["records"],
        full_records=full_progress["records"],
        expected_keys=expected_keys,
    )
    chunk_size = math.ceil(len(jobs) / workers)
    shards = [
        jobs[start : start + chunk_size]
        for start in range(0, len(jobs), chunk_size)
    ]
    accumulators: dict[tuple[str, str, int], AllTBaselineAccumulator] = {}
    completed = 0
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=len(shards),
        mp_context=context,
        initializer=_initialize_worker,
        initargs=(
            str(cache_root.resolve()),
            str(dataset_spec),
            dict(full_progress["provenance"]),
            dict(setup.p6a_config),
            rio_class_mapping,
        ),
    ) as executor:
        for shard_count, local in executor.map(_process_sequence_shard, shards):
            for key, incoming in local.items():
                if key not in accumulators:
                    accumulators[key] = incoming
                else:
                    accumulators[key].merge(incoming)
            completed += shard_count
            print(
                f"[allt-baseline] completed {completed}/{len(jobs)} sequences",
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
        accumulators[key].sequence_count != 129 for key in expected_cells
    ):
        raise AllTBaselineError("all-T aggregate coverage differs")

    old_task_path = R1_ARTIFACT_ROOT / "metrics/task_aggregate.csv"
    old_rows = _read_csv(old_task_path)
    old_index = {
        (
            str(row["method"]),
            str(row["score_reducer"]),
            int(row["horizon"]),
        ): row
        for row in old_rows
        if row["order_id"] == "all"
    }
    rows = []
    for method, reducer, horizon in sorted(expected_cells):
        computed = accumulators[(method, reducer, horizon)].compute()
        if horizon == 3:
            values = computed
        else:
            old = old_index.get((method, reducer, horizon))
            if old is None:
                raise AllTBaselineError("legacy aggregate cell is unavailable")
            values = {
                "local_current_AP": float(old["current_stage_AP"]),
                "prefix_overall_mAP": computed["prefix_overall_mAP"],
                "t_REC": float(old["causal_prefix_t_REC"]),
                "t_mAP": float(old["causal_prefix_t_mAP"]),
                "t_mAP25": float(old["causal_prefix_t_mAP25"]),
                "t_mAP50": float(old["causal_prefix_t_mAP50"]),
            }
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
                "num_order_units": accumulators[
                    (method, reducer, horizon)
                ].sequence_count,
                "num_reference_clusters": 6,
            }
        )

    regression = validate_legacy_regression(rows, old_rows)
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
        "newly_computed_cells": {"prefix_overall": 28, "temporal_T3": 7},
        "protocol_sha256": compact["protocol_sha256"],
        "reused_legacy_temporal_cells": 21,
        "schema_version": 1,
        "source_commit": _git_head(),
        "status": "pass",
        "worker_count": workers,
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
    parser.add_argument("--workers", type=int, default=12)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_baseline_analysis(
        cache_root=args.cache_root,
        output_root=args.output_root,
        checkpoint=args.checkpoint,
        pretrained=args.pretrained,
        metadata=args.metadata,
        workers=args.workers,
    )
    print(json.dumps(result, allow_nan=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
