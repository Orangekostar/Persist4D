"""Locked populations, shared live parents and coverage-aware V2 confirmation."""

from __future__ import annotations

import datetime as dt
import json
import math
import statistics
import time
from pathlib import Path

from scripts.perception_gain_v2 import (
    PROJECT_ROOT,
    file_hash,
    read_json,
    utc_now,
    write_json,
)
from scripts.perception_gain_v2_config import content_hash, live_execution_provenance
from scripts.perception_gain_v2_lock import (
    effective_parent_weight_identity,
    external_path,
    inference_identity,
)
from scripts.perception_gain_v2_selection import compare, metrics

POPULATIONS = {
    "PB": {str(t): {"logical_units": 129, "references": 6} for t in (2, 3, 4, 5)},
    "ADDITIONAL": {
        "2": {"logical_units": 111, "references": 40},
        "3": {"logical_units": 77, "references": 23},
        "4": {"logical_units": 32, "references": 8},
    },
}


def allocate_confirmation_balance(
    *, remaining: float, required: float, default: float
) -> dict:
    if any(
        not math.isfinite(value) or value < 0
        for value in (remaining, required, default)
    ):
        raise ValueError("Confirmation allocation requires finite nonnegative budgets")
    reserve = max(required, default)
    funded = min(remaining, reserve)
    return {
        "remaining_gpu_hours": remaining,
        "confirmation": funded,
        "replication_and_recovery": remaining - funded,
        "transfer_into_confirmation": max(0.0, funded - default),
        "unfunded_confirmation": max(0.0, required - funded),
        "perception_exploration": 0.0,
        "development_search": 0.0,
        "refinement_search": 0.0,
        "scope": "Unspent allocation after final lock; includes profile within confirmation",
    }


def confirmation_budget(
    config: dict, *, artifacts: Path, lock: dict
) -> tuple[dict, dict]:
    try:
        forecast = forecast_confirmation(config, artifacts=artifacts, lock=lock)
    except (OSError, ValueError, KeyError) as error:
        forecast = {
            "status": "MEASURED_FORECAST_UNAVAILABLE",
            "reason": str(error),
            "required_confirmation_gpu_hours": config["budget"]["confirmation_reserve"],
            "limitation": "Retain the original planned reserve; this is not an empirical runtime estimate.",
        }
    events = {
        row["event_id"]: row
        for line in (artifacts / "budget/LEDGER.jsonl").read_text().splitlines()
        if line.strip()
        for row in (json.loads(line),)
    }
    settled = sum(row["gpu_hours"] for row in events.values())
    state = read_json(artifacts / "RUN_STATE.json")
    live = sum(
        max(0.0, time.time() - dt.datetime.fromisoformat(row["start_utc"]).timestamp())
        * len(row["gpus"])
        / 3600
        for row in state["tasks"].values()
        if row["status"] == "RUNNING"
    )
    allocation = allocate_confirmation_balance(
        remaining=max(0.0, config["cumulative_gpu_hour_cap"] - settled - live),
        required=forecast["required_confirmation_gpu_hours"],
        default=config["budget"]["confirmation_reserve"],
    )
    return forecast, allocation


def confirmation_groups(lock: dict, role: str) -> list[dict]:
    groups = {}
    for name in lock["confirmation_methods"][role]:
        method = lock["methods"][name]
        native = method["kind"] == "NATIVE"
        identity = inference_identity(
            {
                **method,
                "kind": "NATIVE" if native else "D0",
                "refiner": None,
                "association_config": None,
            }
        )
        group = groups.setdefault(
            identity,
            {
                "identity": identity,
                "native": native,
                "parent": method,
                "method_ids": [],
                "heads": [],
                "associations": [],
            },
        )
        group["method_ids"].append(name)
        if method["kind"] == "REFINER":
            group["heads"].append(name)
        if method["kind"] == "ASSOCIATION":
            group["associations"].append(name)
        if group["heads"] and group["associations"]:
            raise ValueError("Confirmation cannot combine association and repair")
    return list(groups.values())


def normalize_evidence(
    result: dict, *, method_id: str, source_method: str, role: str
) -> dict:
    expected = POPULATIONS[role]
    rows = [
        row
        for row in result.get("metric_rows", [])
        if row.get("method") == source_method and str(row.get("T")) in expected
    ]
    pooled = {str(row["T"]): row for row in rows if row.get("reference") == "all"}
    coverage, values = {}, {}
    for horizon, population in expected.items():
        row = pooled.get(horizon, {})
        observed = result.get("population_by_horizon", {}).get(horizon, {})
        completed = result.get("completed_units_by_horizon", {}).get(horizon)
        units = (
            len(completed)
            if completed is not None
            else observed.get("logical_unit_count", 0)
        )
        references = row.get("reference_count", 0)
        value = row.get("t_mAP")
        finite = (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and 0 <= value <= 1
        )
        complete = (
            units == population["logical_units"]
            and row.get("logical_unit_count") == population["logical_units"]
            and references == population["references"]
            and finite
        )
        coverage[horizon] = {
            "expected_logical_units": population["logical_units"],
            "actual_logical_units": units,
            "expected_references": population["references"],
            "actual_references": references,
            "status": "COMPLETE" if complete else "INCOMPLETE",
        }
        values[horizon] = float(value) if finite else None
    full = (
        result.get("status") == "PASS"
        and not result.get("incomplete_units")
        and all(row["status"] == "COMPLETE" for row in coverage.values())
    )
    return {
        "method_id": method_id,
        "data_role": role,
        "coverage_status": "COMPLETE" if full else "INCOMPLETE",
        "metrics": values,
        "coverage_by_horizon": coverage,
        "metric_rows": [
            {**row, "source_method": source_method, "method": method_id} for row in rows
        ],
        "incomplete_units": result.get("incomplete_units", []),
        "source_status": result.get("status"),
    }


def deployment_decision(final: dict, native: dict, d0: dict) -> dict:
    def strict_all(reference):
        left, right = metrics(final), metrics(reference)
        return (
            None
            if left is None or right is None
            else all(left[t] > right[t] + 1e-6 for t in (2, 3, 4, 5))
        )

    native_all, d0_all = strict_all(native), strict_all(d0)
    paired = compare(final, d0)
    deploy = (
        native_all is True
        and paired["comparison_status"] == "COMPLETE"
        and paired["S_mean"] > 0
        and paired["S_min"] >= -0.001
    )
    return {
        "tmap_all_T_vs_native_FH": native_all,
        "tmap_all_T_vs_D0": d0_all,
        "default_method": "FINAL" if deploy else "R1-D0",
        "final_vs_D0": paired,
        "scope": "PB only; all required prefixes and references must be covered",
    }


def load_lock(config: dict, artifacts: Path) -> tuple[dict, str]:
    path = artifacts / "selection/FINAL_LOCK.json"
    lock = read_json(path)
    if lock.get("status") != "LOCKED" or lock.get("content_sha256") != content_hash(
        {key: value for key, value in lock.items() if key != "content_sha256"}
    ):
        raise ValueError("Final lock is missing or its immutable content changed")
    for key, relative in (
        ("data_roles_sha256", "DATA_ROLES.json"),
        ("input_manifest_sha256", "data/STAGING_MANIFEST.json"),
        ("run_config_sha256", "RUN_CONFIG.json"),
        ("local_population_audit_sha256", "foundation/LOCAL_POPULATION_AUDIT.json"),
    ):
        if lock[key] != file_hash(artifacts / relative):
            raise ValueError(f"Locked confirmation input changed: {relative}")
    for method in lock["methods"].values():
        if method["inference_identity"] != inference_identity(method):
            raise ValueError("Locked method inference identity changed")
    return lock, file_hash(path)


def method_inputs(method: dict, *, external_root: Path, artifacts: Path) -> dict:
    assets = read_json(external_root / "assets.local.json")
    recipe = read_json(artifacts / f"training/{method['recipe_id']}/recipe.json")
    if recipe != method["recipe"]:
        raise ValueError("Locked recipe changed")
    checkpoint = (
        external_path(method["parent_checkpoint"], external_root)
        if method["parent_checkpoint"]
        else None
    )
    checkpoint_hash = file_hash(checkpoint or Path(assets["r1_checkpoint"]))
    if checkpoint_hash != method["parent_checkpoint_file_sha256"]:
        raise ValueError("Locked parent weights changed")
    scorer = Path(assets["scorer_checkpoint"]) if method["scorer_checkpoint"] else None
    if scorer is not None and file_hash(scorer) != method["scorer_sha256"]:
        raise ValueError("Locked scorer weights changed")
    if (
        effective_parent_weight_identity(
            checkpoint_hash,
            architecture=method["architecture_variant"],
            update=method["parent_optimizer_update"],
            scorer_sha256=method["scorer_sha256"],
        )
        != method["parent_weight_sha256"]
    ):
        raise ValueError("Locked effective parent identity changed")
    return {
        "variant": method["architecture_variant"],
        "optimizer_update": method["parent_optimizer_update"],
        "checkpoint": checkpoint,
        "scorer_checkpoint": scorer,
        "recipe_config": recipe,
        "assets_path": external_root / "assets.local.json",
        "external_root": external_root,
        "artifact_root": artifacts,
    }


def forecast_confirmation(config: dict, *, artifacts: Path, lock: dict) -> dict:
    """Forecast from completed live CAL time, including its official CPU metrics."""
    d0 = read_json(artifacts / "foundation/CAL/D0.json")
    native = read_json(artifacts / "foundation/CAL/FH-native.json")
    if d0["status"] != "PASS" or native["status"] != "PASS":
        raise ValueError(
            "Confirmation forecast requires complete measured CAL baselines"
        )
    d0_unit = d0["elapsed_seconds"] / 23
    native_scan = native["elapsed_seconds"] / (23 * 14)
    repair_rates = []
    for path in (artifacts / "refiner").glob("*/CAL.json"):
        result = read_json(path)
        if result.get("status") == "PASS" and result.get("candidates"):
            repair_rates.append(
                max(0.0, result["elapsed_seconds"] / 23 - d0_unit)
                / len(result["candidates"])
            )
    entries = []
    for role in ("PB", "ADDITIONAL"):
        for group in confirmation_groups(lock, role):
            if group["native"]:
                scans = 129 * 14 if role == "PB" else 111 * 2 + 77 * 3 + 32 * 4
                seconds = scans * native_scan
            else:
                # Each ADDITIONAL sequence has <= five scans and one metric prefix;
                # a complete five-scan CAL unit is a conservative per-unit estimate.
                units = 129 if role == "PB" else 111 + 77 + 32
                if group["heads"] and not repair_rates:
                    raise ValueError(
                        "Selected repair has no complete measured CAL cost"
                    )
                seconds = units * (
                    d0_unit + len(group["heads"]) * max(repair_rates, default=0)
                )
                if group["associations"]:
                    from models.overlap_entity_association import AssociationConfig

                    association = AssociationConfig(
                        **lock["methods"][group["associations"][0]][
                            "association_config"
                        ]
                    ).config_id
                    replay = read_json(
                        artifacts / f"association/CAL/{association}.json"
                    )
                    seconds += units * replay["elapsed_seconds"] / 23
            entries.append(
                {
                    "role": role,
                    "method_ids": group["method_ids"],
                    "predicted_gpu_hours": seconds / 3600,
                }
            )
    for name in lock["confirmation_methods"]["LOCAL-T2"]:
        entries.append(
            {
                "role": "LOCAL-T2",
                "method_ids": [name],
                "predicted_gpu_hours": 154
                * 2
                * len(lock["confirmation_populations"]["LOCAL-T2"]["eval_seeds"])
                * native_scan
                / 3600,
            }
        )
    measured_prediction = sum(row["predicted_gpu_hours"] for row in entries)
    reserve = measured_prediction * 1.25 + config["budget"]["profile_maximum"]
    return {
        "utc": utc_now(),
        "status": "FORECAST_FROM_MEASURED_CAL",
        "entries": entries,
        "evaluation_prediction_gpu_hours": measured_prediction,
        "safety_factor": 1.25,
        "profile_reserved_gpu_hours": config["budget"]["profile_maximum"],
        "required_confirmation_gpu_hours": reserve,
        "limitations": "Linear population scaling; LOCAL uses measured native CAL scan cost. Actual occupancy remains charged independently; no claim that a forecast is a hard runtime bound.",
    }


def evaluate_group(
    config: dict,
    *,
    group: dict,
    lock: dict,
    lock_sha256: str,
    role: str,
    artifacts: Path,
    external_root: Path,
) -> tuple[dict, Path]:
    inputs = method_inputs(
        group["parent"], external_root=external_root, artifacts=artifacts
    )
    binding = {
        "lock_sha256": lock_sha256,
        "group_identity": group["identity"],
        "methods": group["method_ids"],
        "role": role,
        "roles_sha256": lock["data_roles_sha256"],
        "input_manifest_sha256": lock["input_manifest_sha256"],
        "eval_seed": 45,
        "relevant_source_digest": live_execution_provenance(inputs["recipe_config"])[
            "relevant_source_digest"
        ],
    }
    path = artifacts / f"confirmation/{role}/{content_hash(binding)}.json"
    if path.exists():
        result = read_json(path)
        if result.get("confirmation_binding") != binding:
            raise ValueError("Confirmation cache binding differs")
        return result, path
    from scripts.perception_gain_v2_predictions import PredictionExports

    exports = PredictionExports(
        external_root=external_root,
        methods=[lock["methods"][name] for name in group["method_ids"]],
        role=role,
        eval_seed=45,
        lock_sha256=lock_sha256,
        expected_units_by_horizon={
            t: value["logical_units"] for t, value in POPULATIONS[role].items()
        },
        source_methods={
            name: (
                f"FH-{lock['methods'][name]['architecture_variant']}-native"
                if group["native"]
                else name
                if lock["methods"][name]["kind"] in {"ASSOCIATION", "REFINER"}
                else "PARENT"
            )
            for name in group["method_ids"]
        },
    )
    inputs["prediction_callback"] = exports
    if group["native"]:
        from scripts.perception_gain_native_evaluation import (
            run_native_checkpoint_evaluation,
        )

        result = run_native_checkpoint_evaluation(
            **inputs,
            role=role,
            horizons=tuple(map(int, POPULATIONS[role])),
            roles_path=artifacts / "DATA_ROLES.json",
            output_path=path,
        )
    elif group["associations"]:
        from scripts.perception_gain_v2_live_association import (
            run_live_association_evaluation,
        )

        result = run_live_association_evaluation(
            config,
            external_root=external_root,
            parent_method=group["parent"],
            association_method=lock["methods"][group["associations"][0]],
            role=role,
            output_path=path,
            prediction_callback=exports,
        )
    else:
        from scripts.perception_refiner_evaluation import run_refiner_evaluation

        heads = []
        for name in group["heads"]:
            head = lock["methods"][name]["refiner"]
            checkpoint = external_path(head["checkpoint"], external_root)
            if file_hash(checkpoint) != head["checkpoint_sha256"]:
                raise ValueError("Locked refiner weights changed")
            heads.append(
                {
                    "method_id": name,
                    "checkpoint": checkpoint,
                    "optimizer_update": head["optimizer_update"],
                    "binding": head["binding"],
                }
            )
        variant, update, checkpoint = (
            inputs.pop("variant"),
            inputs.pop("optimizer_update"),
            inputs.pop("checkpoint"),
        )
        result = run_refiner_evaluation(
            **inputs,
            parent_variant=variant,
            parent_update=update,
            parent_checkpoint=checkpoint,
            refiner_update=0,
            refiner_checkpoint=None,
            refiner_checkpoints=heads,
            role=role,
            roles_path=artifacts / "DATA_ROLES.json",
            output_path=path,
        )
    result["prediction_exports"] = exports.finalize()
    result["confirmation_binding"] = binding
    write_json(path, result)
    return result, path


def run_confirmation(config: dict, *, external_root: Path) -> dict:
    artifacts = PROJECT_ROOT / config["artifact_root"]
    lock, lock_sha = load_lock(config, artifacts)
    forecast, allocation = confirmation_budget(config, artifacts=artifacts, lock=lock)
    write_json(artifacts / "budget/CONFIRMATION_FORECAST.json", forecast)
    write_json(artifacts / "budget/CONFIRMATION_CURRENT_ALLOCATION.json", allocation)
    results, failures = {}, {}
    for role in ("PB", "ADDITIONAL"):
        rows = {}
        for group in confirmation_groups(lock, role):
            try:
                result, path = evaluate_group(
                    config,
                    group=group,
                    lock=lock,
                    lock_sha256=lock_sha,
                    role=role,
                    artifacts=artifacts,
                    external_root=external_root,
                )
                for name in group["method_ids"]:
                    method = lock["methods"][name]
                    source = (
                        f"FH-{method['architecture_variant']}-native"
                        if group["native"]
                        else (
                            name
                            if method["kind"] in {"REFINER", "ASSOCIATION"}
                            else "PARENT"
                        )
                    )
                    rows[name] = {
                        **normalize_evidence(
                            result, method_id=name, source_method=source, role=role
                        ),
                        "artifact": str(path.relative_to(PROJECT_ROOT)),
                        "artifact_sha256": file_hash(path),
                    }
            except (OSError, RuntimeError, ValueError, KeyError, ImportError) as error:
                failures[f"{role}/{group['identity']}"] = str(error)
                for name in group["method_ids"]:
                    rows[name] = normalize_evidence(
                        {"status": "BLOCKED"},
                        method_id=name,
                        source_method="PARENT",
                        role=role,
                    )
                    rows[name]["reason"] = str(error)
            write_json(artifacts / f"confirmation/{role}/METHODS.json", rows)
        results[role] = rows
    local = {}
    from scripts.perception_gain_local_evaluation import run_local_t2_evaluation

    for name in lock["confirmation_methods"]["LOCAL-T2"]:
        seeds = {}
        method = lock["methods"][name]
        for seed in lock["confirmation_populations"]["LOCAL-T2"]["eval_seeds"]:
            try:
                inputs = method_inputs(
                    method, external_root=external_root, artifacts=artifacts
                )
                binding = {
                    "lock_sha256": lock_sha,
                    "method_identity": method["inference_identity"],
                    "eval_seed": seed,
                    "relevant_source_digest": live_execution_provenance(
                        inputs["recipe_config"]
                    )["relevant_source_digest"],
                }
                path = artifacts / f"confirmation/LOCAL-T2/{content_hash(binding)}.json"
                if path.exists():
                    row = read_json(path)
                    if row.get("confirmation_binding") != binding:
                        raise ValueError("LOCAL confirmation cache binding differs")
                else:
                    from scripts.perception_gain_v2_predictions import PredictionExports

                    exports = PredictionExports(
                        external_root=external_root,
                        methods=[method],
                        role="LOCAL-T2",
                        eval_seed=seed,
                        lock_sha256=lock_sha,
                        expected_units_by_horizon={2: 154},
                        source_methods={name: "LOCAL"},
                    )
                    try:
                        row = run_local_t2_evaluation(
                            **inputs,
                            eval_seed=seed,
                            output_path=path,
                            prediction_callback=exports,
                        )
                    finally:
                        exported = exports.finalize()
                    row["prediction_exports"] = exported
                    row["confirmation_binding"] = binding
                    write_json(path, row)
                seeds[str(seed)] = row
            except (OSError, RuntimeError, ValueError, KeyError) as error:
                seeds[str(seed)] = {
                    "status": "BLOCKED",
                    "reason": str(error),
                    "eval_seed": seed,
                }
        full = all(
            row.get("status") == "PASS"
            and row.get("validation_sequence_count") == 154
            and row.get("validation_reference_count")
            == lock["confirmation_populations"]["LOCAL-T2"]["references"]
            and row.get("population_manifest_sha256")
            == lock["confirmation_populations"]["LOCAL-T2"][
                "population_manifest_sha256"
            ]
            for row in seeds.values()
        )
        local[name] = {
            "coverage_status": "COMPLETE" if full else "INCOMPLETE",
            "expected_references": lock["confirmation_populations"]["LOCAL-T2"][
                "references"
            ],
            "expected_units_per_seed": 154,
            "seeds": seeds,
            "mean_metrics": (
                {
                    key: statistics.mean(row["metrics"][key] for row in seeds.values())
                    for key in next(iter(seeds.values()))["metrics"]
                }
                if full
                else None
            ),
        }
        write_json(artifacts / "confirmation/LOCAL-T2/METHODS.json", local)
    aliases = lock["aliases"]
    pb = results["PB"]
    decision = deployment_decision(
        pb[aliases["FINAL"]], pb[aliases["FH-R1-native"]], pb[aliases["R1-D0"]]
    )
    report = {
        "status": (
            "COMPLETE"
            if all(
                row["coverage_status"] == "COMPLETE"
                for role in results.values()
                for row in role.values()
            )
            and all(row["coverage_status"] == "COMPLETE" for row in local.values())
            else "BLOCKED"
        ),
        "lock_sha256": lock_sha,
        "populations": results,
        "local_t2": local,
        "deployment": decision,
        "failures": failures,
        "forecast": forecast,
    }
    write_json(artifacts / "confirmation/CONFIRMATION.json", report)
    return {"status": report["status"], "deployment": decision, "failures": failures}
