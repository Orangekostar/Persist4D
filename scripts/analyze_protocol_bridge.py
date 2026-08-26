#!/usr/bin/env python3
"""Analyze PB1 population, order, and horizon factors without conflating them."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_protocol_bridge import (
    BRIDGE_DATABASE,
    BRIDGE_INVENTORY,
    DEFAULT_CACHE_ROOT,
    EVALUATION_SEEDS,
    METHOD,
    OUTPUT_ROOT,
    PROTOCOL_MANIFEST,
    RuntimeBinding,
    validate_group_cache,
)

V2_ROOT = PROJECT_ROOT / "artifacts/system_comparison_v2"
START_STATE = PROJECT_ROOT / "artifacts/reviewer_closure_v3/START_STATE.json"
BOOTSTRAP_SEED = 45047
BOOTSTRAP_REPLICATES = 10_000


class ProtocolBridgeAnalysisError(ValueError):
    """Raised when PB1 analysis inputs or paired units differ from registration."""


def _number(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise ProtocolBridgeAnalysisError(f"{name} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ProtocolBridgeAnalysisError(f"{name} must be numeric") from error
    if not math.isfinite(result):
        raise ProtocolBridgeAnalysisError(f"{name} must be finite")
    return result


def _read_csv(path: Path) -> list[dict[str, str]]:
    if path.is_symlink() or not path.is_file():
        raise ProtocolBridgeAnalysisError(f"required CSV is unavailable: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ProtocolBridgeAnalysisError(f"required CSV is empty: {path}")
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hardware_contract() -> dict[str, object]:
    payload = json.loads(START_STATE.read_text(encoding="utf-8"))
    environment = payload.get("environment")
    if not isinstance(environment, Mapping):
        raise ProtocolBridgeAnalysisError("START_STATE environment is unavailable")
    gpus = environment.get("gpus")
    if not isinstance(gpus, list) or not gpus:
        raise ProtocolBridgeAnalysisError("START_STATE GPU inventory is unavailable")
    if not all(
        isinstance(gpu, Mapping)
        and isinstance(gpu.get("model"), str)
        and isinstance(gpu.get("memory_mib"), int)
        and isinstance(gpu.get("driver"), str)
        for gpu in gpus
    ):
        raise ProtocolBridgeAnalysisError("START_STATE GPU inventory is incomplete")
    cuda_runtime = environment.get("cuda_runtime")
    if not isinstance(cuda_runtime, str):
        raise ProtocolBridgeAnalysisError("START_STATE CUDA runtime is unavailable")
    models = {str(gpu["model"]) for gpu in gpus}
    memory = {int(gpu["memory_mib"]) for gpu in gpus}
    drivers = {str(gpu["driver"]) for gpu in gpus}
    if len(models) != 1 or len(memory) != 1 or len(drivers) != 1:
        raise ProtocolBridgeAnalysisError("START_STATE GPU inventory is heterogeneous")
    return {
        "gpu_inference_performed": True,
        "device_alias": "not_recorded_in_frozen_cache",
        "gpu_model": models.pop(),
        "memory_mib": memory.pop(),
        "driver": drivers.pop(),
        "cuda_runtime": cuda_runtime,
        "available_gpu_count": len(gpus),
        "limitation": (
            "The exact device alias was not recorded in the frozen PB1 cache; "
            "all audited visible GPUs had identical hardware properties."
        ),
        "source": {
            "reference": "repo:artifacts/reviewer_closure_v3/START_STATE.json",
            "sha256": _sha256(START_STATE),
        },
    }


def _write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ProtocolBridgeAnalysisError("refusing symbolic-link output")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _csv_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    if not rows:
        raise ProtocolBridgeAnalysisError("analysis CSV rows must not be empty")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    import io

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def build_population_effect_rows(
    aggregate_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    pooled = [row for row in aggregate_rows if row.get("scope") == "pooled"]
    indexed = {
        (int(str(row["seed"])), str(row["population"])): row for row in pooled
    }
    effects: list[dict[str, object]] = []
    deltas = []
    for seed in EVALUATION_SEEDS:
        try:
            full = indexed[(seed, "full154_t2")]
            bridge = indexed[(seed, "bridge43_canonical_t2")]
        except KeyError as error:
            raise ProtocolBridgeAnalysisError(
                "population comparison lacks a registered seed/population"
            ) from error
        if int(str(full["sequence_count"])) != 154 or int(
            str(bridge["sequence_count"])
        ) != 43:
            raise ProtocolBridgeAnalysisError("population coverage differs")
        full_value = _number(full["t_mAP"], name="full t_mAP")
        bridge_value = _number(bridge["t_mAP"], name="bridge t_mAP")
        delta = bridge_value - full_value
        deltas.append(delta)
        effects.append(
            {
                "record_type": "seed",
                "seed": str(seed),
                "method": METHOD,
                "full154_t_mAP": full_value,
                "bridge43_canonical_t_mAP": bridge_value,
                "delta_population_t_mAP": delta,
                "full_sequence_count": 154,
                "bridge_sequence_count": 43,
                "same_runtime_binding": "true",
                "interpretation": "population-bridge diagnostic",
            }
        )
    effects.append(
        {
            "record_type": "summary",
            "seed": "mean/range",
            "method": METHOD,
            "delta_mean": float(np.mean(deltas)),
            "delta_min": float(np.min(deltas)),
            "delta_max": float(np.max(deltas)),
            "full_sequence_count": 154,
            "bridge_sequence_count": 43,
            "same_runtime_binding": "true",
            "interpretation": "descriptive mean/range across preregistered seeds",
        }
    )
    return effects


def _bootstrap_interval(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (6,):
        raise ProtocolBridgeAnalysisError("cluster bootstrap requires six effects")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, 6, size=(BOOTSTRAP_REPLICATES, 6))
    samples = array[indices].mean(axis=1)
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return float(lower), float(upper)


def build_order_effect_rows(
    per_sequence_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    horizon_two = [row for row in per_sequence_rows if int(str(row["horizon"])) == 2]
    if not horizon_two:
        raise ProtocolBridgeAnalysisError("order analysis lacks T2 rows")
    index: dict[tuple[str, str, str, str], float] = {}
    cluster_masters: dict[tuple[str, str], set[str]] = defaultdict(set)
    methods: set[str] = set()
    clusters: set[str] = set()
    for row in horizon_two:
        method = str(row["method"])
        cluster = str(row["reference_scene_id"])
        master = str(row["master_sequence_id"])
        order = str(row["order_id"])
        if order not in {"canonical", "reverse", "sha256_seed45"}:
            raise ProtocolBridgeAnalysisError("order row is not preregistered")
        key = (method, cluster, master, order)
        if key in index:
            raise ProtocolBridgeAnalysisError("duplicate method/cluster/master/order row")
        index[key] = _number(
            row["causal_prefix_t_mAP"], name="causal_prefix_t_mAP"
        )
        cluster_masters[(method, cluster)].add(master)
        methods.add(method)
        clusters.add(cluster)
    if len(clusters) != 6:
        raise ProtocolBridgeAnalysisError("order analysis must retain six clusters")
    comparisons = (
        ("reverse", "reverse-minus-canonical"),
        ("sha256_seed45", "sha256_seed45-minus-canonical"),
    )
    output: list[dict[str, object]] = []
    for method in sorted(methods):
        for order, comparison in comparisons:
            comparison_effects = []
            for cluster in sorted(clusters):
                masters = sorted(cluster_masters[(method, cluster)])
                deltas = []
                for master in masters:
                    try:
                        canonical = index[(method, cluster, master, "canonical")]
                        alternative = index[(method, cluster, master, order)]
                    except KeyError as error:
                        raise ProtocolBridgeAnalysisError(
                            "order comparison is not paired within master"
                        ) from error
                    deltas.append(alternative - canonical)
                effect = float(np.mean(deltas))
                comparison_effects.append(effect)
                output.append(
                    {
                        "record_type": "cluster_effect",
                        "method": method,
                        "comparison": comparison,
                        "reference_scene_id": cluster,
                        "master_count": len(masters),
                        "paired_master_mean_delta_t_mAP": effect,
                        "independence_unit": "reference_scene_id",
                        "order_unit_count_is_independent": "false",
                    }
                )
            lower, upper = _bootstrap_interval(comparison_effects)
            output.append(
                {
                    "record_type": "cluster_summary",
                    "method": method,
                    "comparison": comparison,
                    "reference_scene_id": "all-six",
                    "master_count": sum(
                        len(cluster_masters[(method, cluster)])
                        for cluster in clusters
                    ),
                    "paired_master_mean_delta_t_mAP": float(
                        np.mean(comparison_effects)
                    ),
                    "cluster_effect_min": float(np.min(comparison_effects)),
                    "cluster_effect_max": float(np.max(comparison_effects)),
                    "cluster_bootstrap_95_low": lower,
                    "cluster_bootstrap_95_high": upper,
                    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                    "bootstrap_seed": BOOTSTRAP_SEED,
                    "independence_unit": "reference_scene_id",
                    "order_unit_count_is_independent": "false",
                    "interval_interpretation": "descriptive robustness, N=6 clusters",
                }
            )
    return output


def build_horizon_retention_rows(
    aggregate_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], dict[int, Mapping[str, object]]] = defaultdict(dict)
    for row in aggregate_rows:
        horizon = int(str(row["horizon"]))
        if horizon not in {2, 3, 4, 5}:
            continue
        key = (str(row["method"]), str(row["order_id"]))
        if horizon in groups[key]:
            raise ProtocolBridgeAnalysisError("duplicate method/order/horizon row")
        groups[key][horizon] = row
    output = []
    for (method, order), rows in sorted(groups.items()):
        if set(rows) != {2, 3, 4, 5}:
            raise ProtocolBridgeAnalysisError("horizon group lacks exact T2-T5")
        denominator = _number(rows[2]["causal_prefix_t_mAP"], name="T2 t_mAP")
        if denominator <= 0:
            raise ProtocolBridgeAnalysisError("T2 retention denominator must be positive")
        for horizon in (2, 3, 4, 5):
            row = rows[horizon]
            value = _number(row["causal_prefix_t_mAP"], name="horizon t_mAP")
            output.append(
                {
                    "method": method,
                    "order_id": order,
                    "horizon": horizon,
                    "sequence_count": int(str(row["sequence_count"])),
                    "t_mAP": value,
                    "t_REC": _number(
                        row["causal_prefix_t_REC"], name="horizon t_REC"
                    ),
                    "current_local_AP_calibration": _number(
                        row["current_stage_AP"], name="current-stage AP"
                    ),
                    "relative_retention": value / denominator,
                    "paired_within_master_order": "true",
                    "cross_horizon_independence_claimed": "false",
                    "source_channel": "frozen Protocol-B V2 exact-prefix evaluation",
                }
            )
    return output


def _pooled_order_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    output = []
    for row in rows:
        if int(str(row["horizon"])) != 2:
            continue
        output.append(
            {
                "record_type": "pooled_descriptive",
                "method": row["method"],
                "comparison": str(row["order_id"]),
                "reference_scene_id": "pooled-43",
                "master_count": int(str(row["sequence_count"])),
                "pooled_t_mAP": _number(
                    row["causal_prefix_t_mAP"], name="pooled order t_mAP"
                ),
                "pooled_t_REC": _number(
                    row["causal_prefix_t_REC"], name="pooled order t_REC"
                ),
                "current_stage_AP": _number(
                    row["current_stage_AP"], name="pooled current AP"
                ),
                "independence_unit": "reference_scene_id",
                "order_unit_count_is_independent": "false",
            }
        )
    return output


def _cache_contract(cache_root: Path) -> dict[str, object]:
    paths = sorted(cache_root.glob("seed*_*.pt"))
    if len(paths) != 6:
        raise ProtocolBridgeAnalysisError("PB1 cache must contain six groups")
    first = torch.load(paths[0], map_location="cpu", weights_only=False)
    binding_raw = first.get("runtime_binding") if isinstance(first, Mapping) else None
    if not isinstance(binding_raw, Mapping):
        raise ProtocolBridgeAnalysisError("PB1 cache lacks runtime binding")
    binding = RuntimeBinding(**dict(binding_raw))
    expected_evidence = {
        "protocol_sha256": _sha256(PROTOCOL_MANIFEST),
        "bridge_inventory_sha256": _sha256(BRIDGE_INVENTORY),
        "bridge_database_sha256": _sha256(BRIDGE_DATABASE),
    }
    groups = []
    cache_files = []
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        validate_group_cache(payload, binding=binding)
        if payload.get("evidence_binding") != expected_evidence:
            raise ProtocolBridgeAnalysisError(
                "PB1 cache evidence binding differs from Protocol B / PB0"
            )
        groups.append(dict(payload["group"]))
        cache_files.append(
            {
                "reference": f"external:protocol_bridge/{path.name}",
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    expected = {
        (seed, population)
        for seed in EVALUATION_SEEDS
        for population in ("full154_t2", "bridge43_canonical_t2")
    }
    observed = {(int(row["seed"]), str(row["population"])) for row in groups}
    if observed != expected:
        raise ProtocolBridgeAnalysisError("PB1 cache group coverage differs")
    return {
        "runtime_binding": binding.as_dict(),
        "groups": groups,
        "cache_files": cache_files,
    }


def _report(
    population: Sequence[Mapping[str, object]],
    order: Sequence[Mapping[str, object]],
    horizon: Sequence[Mapping[str, object]],
) -> str:
    summary = next(row for row in population if row["record_type"] == "summary")
    seed_rows = [row for row in population if row["record_type"] == "seed"]
    lines = [
        "# Protocol Bridge Evaluation",
        "",
        "## PB1 Status",
        "",
        "**PASS.** Population, order, and horizon are reported as separate factors.",
        "No additive decomposition from the paper-reported 34.8% result is made.",
        "",
        "## Population Effect",
        "",
        "The full-154 and exact-43 canonical T2 populations use the same frozen",
        "checkpoint, batch-1 FP32 validation runtime, official ReScene post-processing, class",
        "map, metric specification, min-region size, and evaluation seed.",
        "",
        "| Seed | Full-154 t-mAP | Exact-43 t-mAP | Delta (43 - 154) |",
        "|---:|---:|---:|---:|",
    ]
    for row in seed_rows:
        lines.append(
            f"| {row['seed']} | {100 * float(row['full154_t_mAP']):.3f} | "
            f"{100 * float(row['bridge43_canonical_t_mAP']):.3f} | "
            f"{100 * float(row['delta_population_t_mAP']):+.3f} |"
        )
    lines.extend(
        [
            "",
            (
                "The mean population delta is "
                f"`{100 * float(summary['delta_mean']):+.3f}` percentage points; "
                "the registered-seed range is "
                f"`[{100 * float(summary['delta_min']):+.3f}, "
                f"{100 * float(summary['delta_max']):+.3f}]`."
            ),
            "This is a population-bridge diagnostic, not an official benchmark score.",
            "",
            "## Order Effect",
            "",
            "Order uses the frozen Protocol-B V2 prediction/evaluator path on the same",
            "43 masters. Pooled values are descriptive. Pairing is within master, and",
            "all six `reference_scene_id` effects are retained. The 129 order-units are",
            "not treated as independent; cluster bootstrap intervals are descriptive",
            "robustness evidence only.",
            "",
        ]
    )
    summaries = [row for row in order if row["record_type"] == "cluster_summary"]
    lines.extend(
        [
            "| Method | Comparison | Mean cluster effect | Six-cluster range | Bootstrap 95% |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in summaries:
        lines.append(
            f"| {row['method']} | {row['comparison']} | "
            f"{100 * float(row['paired_master_mean_delta_t_mAP']):+.3f} | "
            f"[{100 * float(row['cluster_effect_min']):+.3f}, "
            f"{100 * float(row['cluster_effect_max']):+.3f}] | "
            f"[{100 * float(row['cluster_bootstrap_95_low']):+.3f}, "
            f"{100 * float(row['cluster_bootstrap_95_high']):+.3f}] |"
        )
    lines.extend(
        [
            "",
            "## Horizon Effect",
            "",
            "Horizon uses the frozen exact-prefix Protocol-B T2-T5 results. Retention",
            "is computed within each method/order as `t-mAP(T) / t-mAP(T2)`; absolute",
            "cross-horizon pooled values are not treated as independent samples.",
            "The `current_local_AP_calibration` column is the frozen V2 current-stage",
            "calibration channel; V3 score closure separately audits direct latest-stage",
            "official sidecars.",
            "",
            f"Horizon table rows: `{len(horizon)}` (2 methods x 3 orders x 4 horizons).",
            "",
            "## Runtime Boundary",
            "",
            "The local 27.939% P2 result remains a frozen external-to-this-stage local",
            "reference. PB1 reruns its own controlled seed groups only to compare the two",
            "populations under one runtime. The paper-reported 34.8% result is not mixed",
            "into these differences.",
        ]
    )
    return "\n".join(lines) + "\n"


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def run_analysis(
    *,
    output_root: Path = OUTPUT_ROOT,
    cache_root: Path = DEFAULT_CACHE_ROOT,
) -> dict[str, object]:
    aggregate = _read_csv(output_root / "bridge_aggregate.csv")
    population = build_population_effect_rows(aggregate)
    v2_per_sequence = _read_csv(V2_ROOT / "per_sequence_results.csv")
    v2_per_order = _read_csv(V2_ROOT / "per_order_results.csv")
    order = _pooled_order_rows(v2_per_order) + build_order_effect_rows(
        v2_per_sequence
    )
    horizon = build_horizon_retention_rows(v2_per_order)
    if len([row for row in order if row["record_type"] == "cluster_effect"]) != 24:
        raise ProtocolBridgeAnalysisError("order analysis lacks all method/cluster effects")
    if len(horizon) != 24:
        raise ProtocolBridgeAnalysisError("horizon analysis coverage differs")
    cache_contract = _cache_contract(cache_root)

    outputs = {
        "population_effect.csv": _csv_bytes(population),
        "order_effect.csv": _csv_bytes(order),
        "horizon_retention.csv": _csv_bytes(horizon),
    }
    for name, content in outputs.items():
        _write(output_root / name, content)
    report = _report(population, order, horizon).encode("utf-8")
    _write(output_root / "PROTOCOL_BRIDGE_EVALUATION.md", report)

    output_descriptors = {}
    for name in (
        "bridge_per_sequence.csv",
        "bridge_aggregate.csv",
        "population_effect.csv",
        "order_effect.csv",
        "horizon_retention.csv",
        "PROTOCOL_BRIDGE_EVALUATION.md",
    ):
        path = output_root / name
        output_descriptors[name] = {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    manifest = {
        "schema_version": 1,
        "status": "pass",
        "generated_at": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "source_commit": _git_head(),
        "hardware": _hardware_contract(),
        "gate_pb1": {
            "status": "PASS",
            "population_seed_count": 3,
            "population_comparisons_are_runtime_paired": True,
            "order_master_count": 43,
            "order_count": 3,
            "order_unit_count": 129,
            "order_units_treated_as_independent": False,
            "reference_cluster_count": 6,
            "all_cluster_effects_reported": True,
            "horizons": [2, 3, 4, 5],
            "factors_reported_separately": ["population", "order", "horizon"],
            "additive_34_8_to_19_1_decomposition": False,
        },
        "runtime": cache_contract,
        "inputs": {
            "protocol_b_manifest": {
                "reference": "repo:artifacts/P6A/protocol_b_manifest.json",
                "sha256": _sha256(PROJECT_ROOT / "artifacts/P6A/protocol_b_manifest.json"),
            },
            "pb0_bridge_manifest": {
                "reference": "repo:artifacts/reviewer_closure_v3/protocol_bridge/bridge_manifest.json",
                "sha256": _sha256(output_root / "bridge_manifest.json"),
            },
            "v2_per_sequence": {
                "reference": "repo:artifacts/system_comparison_v2/per_sequence_results.csv",
                "sha256": _sha256(V2_ROOT / "per_sequence_results.csv"),
            },
            "v2_per_order": {
                "reference": "repo:artifacts/system_comparison_v2/per_order_results.csv",
                "sha256": _sha256(V2_ROOT / "per_order_results.csv"),
            },
        },
        "outputs": output_descriptors,
        "scripts": {
            "evaluator_sha256": _sha256(
                PROJECT_ROOT / "scripts/evaluate_protocol_bridge.py"
            ),
            "analyzer_sha256": _sha256(
                PROJECT_ROOT / "scripts/analyze_protocol_bridge.py"
            ),
            "test_sha256": _sha256(
                PROJECT_ROOT / "tests/test_protocol_bridge_evaluation.py"
            ),
        },
        "statistics": {
            "evaluation_seeds": list(EVALUATION_SEEDS),
            "cluster_unit": "reference_scene_id",
            "cluster_count": 6,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_interpretation": "descriptive robustness evidence",
        },
    }
    _write(
        output_root / "protocol_bridge_manifest.json",
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return {
        "status": "pass",
        "gate": "PB1",
        "population_rows": len(population),
        "order_rows": len(order),
        "horizon_rows": len(horizon),
        "manifest": str(output_root / "protocol_bridge_manifest.json"),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    print(
        json.dumps(
            run_analysis(
                output_root=arguments.output_root,
                cache_root=arguments.cache_root,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
