import copy
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.analyze_protocol_bridge import (
    build_horizon_retention_rows,
    build_order_effect_rows,
    build_population_effect_rows,
)
from scripts.evaluate_protocol_bridge import (
    EVALUATION_SEEDS,
    RuntimeBinding,
    build_evaluation_plan,
    validate_group_cache,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PB1_ROOT = PROJECT_ROOT / "artifacts/reviewer_closure_v3/protocol_bridge"


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _binding() -> RuntimeBinding:
    digest = "a" * 64
    return RuntimeBinding(
        checkpoint_sha256=digest,
        runtime_config_sha256="b" * 64,
        postprocess_sha256="c" * 64,
        metric_adapter_sha256="d" * 64,
        metric_spec_sha256="e" * 64,
        min_region_size=100,
        precision="32-true",
        batch_size=1,
        num_workers=4,
    )


def test_evaluation_plan_changes_only_population_selection():
    full = [f"official-{index:03d}" for index in range(154)]
    bridge = [
        {
            "master_sequence_id": f"master-{index:02d}",
            "reference_scene_id": f"cluster-{index % 6}",
            "sequence_id": f"bridge-{index:02d}",
            "exact_ordered_pair": "true",
            "validation_supervised": "true",
            "pair_substituted": "false",
            "reverse_pair_substituted": "false",
            "future_stage_leakage": "false",
        }
        for index in range(43)
    ]

    plan = build_evaluation_plan(full, bridge)

    assert EVALUATION_SEEDS == (45, 46, 47)
    assert len(plan) == 6
    assert {(item.seed, item.population) for item in plan} == {
        (seed, population)
        for seed in EVALUATION_SEEDS
        for population in ("full154_t2", "bridge43_canonical_t2")
    }
    assert {item.order_id for item in plan} == {"official", "canonical"}
    assert {item.horizon for item in plan} == {2}
    assert {
        item.sequence_count for item in plan if item.population == "full154_t2"
    } == {154}
    assert {
        item.sequence_count
        for item in plan
        if item.population == "bridge43_canonical_t2"
    } == {43}


def test_group_cache_rejects_any_runtime_binding_drift():
    binding = _binding()
    payload = {
        "schema_version": 1,
        "group": {
            "seed": 45,
            "population": "full154_t2",
            "order_id": "official",
            "horizon": 2,
            "sequence_count": 154,
        },
        "runtime_binding": binding.as_dict(),
        "records": [{} for _ in range(154)],
    }
    validate_group_cache(payload, binding=binding)

    changed = copy.deepcopy(payload)
    changed["runtime_binding"]["min_region_size"] = 50
    with pytest.raises(ValueError, match="runtime binding"):
        validate_group_cache(changed, binding=binding)


def test_population_effect_is_seed_paired_and_reports_mean_and_range():
    rows = []
    for seed, full, bridge in ((45, 0.20, 0.25), (46, 0.30, 0.28), (47, 0.25, 0.29)):
        for population, value, count in (
            ("full154_t2", full, 154),
            ("bridge43_canonical_t2", bridge, 43),
        ):
            rows.append(
                {
                    "scope": "pooled",
                    "seed": str(seed),
                    "population": population,
                    "order_id": "official" if count == 154 else "canonical",
                    "horizon": "2",
                    "sequence_count": str(count),
                    "reference_cluster_count": "6" if count == 43 else "",
                    "t_mAP": str(value),
                    "t_REC": "0.4",
                    "local_current_AP": "0.5",
                }
            )

    effects = build_population_effect_rows(rows)

    seed_rows = [row for row in effects if row["record_type"] == "seed"]
    summary = next(row for row in effects if row["record_type"] == "summary")
    assert [row["seed"] for row in seed_rows] == ["45", "46", "47"]
    assert [float(row["delta_population_t_mAP"]) for row in seed_rows] == pytest.approx(
        [0.05, -0.02, 0.04]
    )
    assert float(summary["delta_mean"]) == pytest.approx(0.07 / 3)
    assert float(summary["delta_min"]) == pytest.approx(-0.02)
    assert float(summary["delta_max"]) == pytest.approx(0.05)


def test_order_effect_keeps_six_reference_clusters_as_paired_units():
    rows = []
    for cluster in range(6):
        for master in range(2):
            for order, offset in (
                ("canonical", 0.0),
                ("reverse", 0.01),
                ("sha256_seed45", -0.02),
            ):
                rows.append(
                    {
                        "method": "Persist4D-V2",
                        "reference_scene_id": f"cluster-{cluster}",
                        "master_sequence_id": f"master-{cluster}-{master}",
                        "order_id": order,
                        "horizon": "2",
                        "causal_prefix_t_mAP": str(0.2 + cluster / 100 + offset),
                    }
                )

    effects = build_order_effect_rows(rows)

    cluster_rows = [row for row in effects if row["record_type"] == "cluster_effect"]
    assert len(cluster_rows) == 12
    assert {row["reference_scene_id"] for row in cluster_rows} == {
        f"cluster-{index}" for index in range(6)
    }
    assert {row["comparison"] for row in cluster_rows} == {
        "reverse-minus-canonical",
        "sha256_seed45-minus-canonical",
    }
    assert {row["independence_unit"] for row in cluster_rows} == {
        "reference_scene_id"
    }
    assert all(row["order_unit_count_is_independent"] == "false" for row in effects)


def test_horizon_retention_uses_each_method_order_t2_denominator():
    rows = [
        {
            "method": method,
            "order_id": order,
            "horizon": str(horizon),
            "sequence_count": "43",
            "causal_prefix_t_mAP": str(base / horizon),
            "causal_prefix_t_REC": str(0.5 / horizon),
            "current_stage_AP": "0.4",
        }
        for method, base in (("FullHistory", 0.4), ("Persist4D-V2", 0.5))
        for order in ("canonical", "reverse", "sha256_seed45")
        for horizon in (2, 3, 4, 5)
    ]

    retention = build_horizon_retention_rows(rows)

    assert len(retention) == 24
    assert all(
        float(row["relative_retention"]) == pytest.approx(2 / int(row["horizon"]))
        for row in retention
    )
    assert all(row["paired_within_master_order"] == "true" for row in retention)


@pytest.mark.parametrize(
    "script",
    ("evaluate_protocol_bridge.py", "analyze_protocol_bridge.py"),
)
def test_cli_entrypoints_work_when_invoked_by_file_path(script):
    completed = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / script), "--help"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_real_pb1_artifact_contract_is_complete_and_hash_bound():
    manifest_path = PB1_ROOT / "protocol_bridge_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    gate = manifest["gate_pb1"]
    assert manifest["status"] == "pass"
    assert gate["status"] == "PASS"
    assert gate["population_seed_count"] == 3
    assert gate["order_unit_count"] == 129
    assert gate["order_units_treated_as_independent"] is False
    assert gate["reference_cluster_count"] == 6
    assert gate["additive_34_8_to_19_1_decomposition"] is False
    for name, descriptor in manifest["outputs"].items():
        path = PB1_ROOT / name
        assert path.stat().st_size == descriptor["bytes"]
        assert _sha256(path) == descriptor["sha256"]

    with (PB1_ROOT / "population_effect.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        population = list(csv.DictReader(handle))
    assert [row["seed"] for row in population[:3]] == ["45", "46", "47"]
    assert population[-1]["record_type"] == "summary"

    with (PB1_ROOT / "order_effect.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        order = list(csv.DictReader(handle))
    cluster_rows = [row for row in order if row["record_type"] == "cluster_effect"]
    assert len(cluster_rows) == 24
    assert len({row["reference_scene_id"] for row in cluster_rows}) == 6
    assert all(row["order_unit_count_is_independent"] == "false" for row in order)

    with (PB1_ROOT / "horizon_retention.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        horizon = list(csv.DictReader(handle))
    assert len(horizon) == 24
    assert {int(row["horizon"]) for row in horizon} == {2, 3, 4, 5}
