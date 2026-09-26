"""Execute the fixed H/L pilots and a single mechanism with its matched control."""

from __future__ import annotations

import datetime as dt
import json
import shutil
import time
from pathlib import Path

from scripts.perception_gain_v2 import (
    PROJECT_ROOT,
    append_event,
    file_hash,
    read_json,
    stage_input_file,
    train_recipe,
    utc_now,
    write_json,
)
from scripts.perception_gain_v2_config import R1_SHA256, live_execution_provenance
from scripts.perception_gain_v2_selection import (
    compare,
    rank,
    select_full_promotion,
    select_learning_rate,
    select_perception,
)


def resolve_checkpoint(
    recipe_id: str, update: int, *, external_root: Path
) -> Path | None:
    if update == 0:
        return None
    local = external_root / f"training/{recipe_id}/update={update:04d}.ckpt"
    if local.is_file():
        return local
    assets = read_json(external_root / "assets.local.json")
    source = None
    if recipe_id == "C0-H-s45" and update in {250, 750}:
        source = Path(assets[f"C0_H_{update:04d}"])
    if recipe_id == "S-BAL-H-s45" and update == 250:
        path = Path(assets["S_BAL_H_resume"]).parent / "update=0250.ckpt"
        if path.is_file():
            source = path
    if source is not None:
        destination = (
            external_root / f"inputs/imported/{recipe_id}/update={update:04d}.ckpt"
        )
        stage_input_file(source, destination, copy_file=False)
        if file_hash(source) != file_hash(destination):
            raise ValueError("Imported checkpoint differs from the original V1 asset")
        source_config = source.parent / "resolved_config.yaml"
        destination_config = destination.parent / source_config.name
        if destination_config.exists() and file_hash(destination_config) != file_hash(
            source_config
        ):
            raise ValueError("Imported legacy checkpoint config changed")
        shutil.copyfile(source_config, destination_config)
        return destination
    raise FileNotFoundError(f"Missing actual checkpoint: {recipe_id}@{update}")


def baseline_candidate(artifacts: Path, role: str) -> dict:
    path = artifacts / f"foundation/{role}/D0.json"
    result = (
        read_json(path)
        if path.exists()
        else {
            "candidate": {
                "coverage_status": "INCOMPLETE",
                "metrics": {},
                "optimizer_update": 0,
                "new_parameter_count": 0,
                "checkpoint_sha256": R1_SHA256,
                "reason": "R1 live baseline has not completed",
            }
        }
    )
    return {
        **result["candidate"],
        "method_id": "R1-D0",
        "recipe_id": "C0-L-s45",
        "architecture_variant": "C0",
        "component_gate_passed": True,
    }


def _step_cost(
    artifacts: Path,
    *,
    phase: str,
    recipe_id: str,
    start: float,
    gpu_count: int,
    updates: int = 0,
) -> None:
    elapsed = time.monotonic() - start
    append_event(
        artifacts / "budget/STEP_COSTS.jsonl",
        {
            "phase": phase,
            "recipe_id": recipe_id,
            "end_utc": utc_now(),
            "elapsed_seconds": elapsed,
            "gpu_count": gpu_count,
            "gpu_hours": elapsed * gpu_count / 3600,
            "optimizer_updates": updates,
            "accounting": "ATTRIBUTION_ONLY; total charged once by parent task reservation",
        },
    )


def budget_decision(
    config: dict,
    *,
    external_root: Path,
    label: str,
    predicted_gpu_hours: float,
    category: str = "perception",
) -> dict:
    artifacts = PROJECT_ROOT / config["artifact_root"]
    state = read_json(artifacts / "RUN_STATE.json")
    events = [
        json.loads(line)
        for line in (artifacts / "budget/LEDGER.jsonl").read_text().splitlines()
        if line.strip()
    ]
    unique = {row["event_id"]: row for row in events if row.get("scope") == "V2"}
    settled = sum(row["gpu_hours"] for row in unique.values())
    live = sum(
        max(0.0, time.time() - dt.datetime.fromisoformat(row["start_utc"]).timestamp())
        * len(row["gpus"])
        / 3600
        for row in state["tasks"].values()
        if row["status"] == "RUNNING"
    )
    remaining = (
        config["cumulative_gpu_hour_cap"] - state["prior_gpu_hours"] - settled - live
    )
    costs_path = artifacts / "budget/STEP_COSTS.jsonl"
    costs = (
        [
            json.loads(line)
            for line in costs_path.read_text().splitlines()
            if line.strip()
        ]
        if costs_path.exists()
        else []
    )
    spent = sum(row["gpu_hours"] for row in costs if row["phase"] == category)
    # Completed continuation preceded substep attribution; charge its actual reservation.
    if category == "perception":
        spent += sum(
            row["gpu_hours"]
            for row in unique.values()
            if row.get("task") in {"HIGH_CONT", "LOW_PAIR"}
        )
    if category == "refinement":
        spent += sum(
            row["gpu_hours"]
            for row in unique.values()
            if row.get("task") in {"REPAIR_R1", "REPAIR_P"}
        )
    reserve = state["confirmation_reserve_gpu_hours"]
    core_refinement = max(
        0.0,
        config["budget"]["refinement"]
        - sum(
            row["gpu_hours"]
            for row in unique.values()
            if row.get("task") in {"REPAIR_R1", "REPAIR_P"}
        ),
    )
    if state["tasks"]["REPAIR_R1"]["status"] == "COMPLETE":
        core_refinement = 0.0
    needed = predicted_gpu_hours * 1.25
    result = {
        "label": label,
        "utc": utc_now(),
        "category": category,
        "predicted_gpu_hours": predicted_gpu_hours,
        "safety_factor": 1.25,
        "category_spent": spent,
        "category_limit": config["budget"][category],
        "remaining_gpu_hours": remaining,
        "protected_confirmation_gpu_hours": reserve,
        "protected_refinement_gpu_hours": core_refinement,
        "affordable": needed <= config["budget"][category] - spent
        and needed <= remaining - reserve - core_refinement,
    }
    append_event(artifacts / "budget/DECISIONS.jsonl", result)
    return result


def measured_update_cost(external_root: Path) -> float:
    estimates = []
    for path in (external_root / "training").glob("*-s45/run_summary.json"):
        summary = read_json(path)
        plan_path = path.parent / "run_plan.json"
        if plan_path.exists() and summary.get("status") == "COMPLETE":
            plan = read_json(plan_path)
            start = (
                plan["plan"]["next_global_draw_index"]
                // plan["contract"]["effective_batch_size"]
            )
            updates = summary["completed_global_step"] - start
            if updates >= 50:
                estimates.append(summary["gpu_hours"] / updates)
    if not estimates:
        raise ValueError(
            "No measured 50-update throughput is available for budget prediction"
        )
    return max(estimates)


def remaining_full_updates(
    methods: list[str], artifacts: Path, external_root: Path
) -> int:
    """Do not reserve another full run for an already verified endpoint."""
    remaining = 0
    for name in methods:
        run = external_root / f"training/{name}"
        path = run / "run_summary.json"
        summary = read_json(path) if path.exists() else {}
        recipe = read_json(artifacts / f"training/{name}/recipe.json")
        if (
            summary.get("status") == "COMPLETE"
            and summary.get("completed_global_step") == 3000
            and summary.get("recipe") == recipe
        ):
            checkpoint = run / "update=3000.ckpt"
            rows = [
                row
                for row in summary.get("checkpoints", [])
                if row.get("name") == checkpoint.name
            ]
            if (
                len(rows) != 1
                or not checkpoint.is_file()
                or rows[0].get("sha256") != file_hash(checkpoint)
            ):
                raise ValueError(f"Completed full checkpoint identity changed: {name}")
        else:
            remaining += 2250
    return remaining


def evaluate(
    config: dict, *, external_root: Path, recipe_id: str, update: int, role: str
) -> dict:
    from scripts.perception_gain_evaluation import run_checkpoint_evaluation
    from scripts.perception_gain_v2_lock import effective_parent_weight_identity

    artifacts = PROJECT_ROOT / config["artifact_root"]
    recipe = read_json(artifacts / f"training/{recipe_id}/recipe.json")
    path = artifacts / f"evaluation/{role}/{recipe_id}/update={update:04d}.json"
    checkpoint = resolve_checkpoint(recipe_id, update, external_root=external_root)
    assets = read_json(external_root / "assets.local.json")
    weight = effective_parent_weight_identity(
        file_hash(checkpoint) if checkpoint else R1_SHA256,
        architecture=recipe["architecture_variant"],
        update=update,
        scorer_sha256=(
            file_hash(Path(assets["scorer_checkpoint"]))
            if recipe["architecture_variant"] == "Q-SEM"
            else None
        ),
    )
    code = live_execution_provenance(recipe)
    expected = {
        "weight_hash": weight,
        "inference_recipe_hash": recipe["inference_recipe_hash"],
        "input_manifest_hash": file_hash(artifacts / "data/STAGING_MANIFEST.json"),
        "roles_sha256": file_hash(artifacts / "DATA_ROLES.json"),
        "role": role,
        "point_order_transform": "canonical_vertices/identity_geometry",
        "eval_seed": 45,
        "publisher": "D0/LAST/lag1/mean",
        "relevant_source_digest": code["relevant_source_digest"],
    }
    if path.exists():
        saved = read_json(path)
        if (
            saved.get("status") != "PASS"
            or saved.get("recipe") != recipe
            or any(
                saved.get("cache_binding", {}).get(key) != value
                for key, value in expected.items()
            )
        ):
            raise ValueError(
                f"Evaluation evidence changed or incomplete; explicit recovery required: {path}"
            )
        return {
            **saved["candidate"],
            "recipe_id": recipe_id,
            "architecture_variant": recipe["architecture_variant"],
        }
    # These zero-update variants have identical inference to R1. Reuse the proven live baseline.
    if update == 0 and recipe["architecture_variant"] in {"C0", "S-BAL", "S-WORST"}:
        return {
            **baseline_candidate(artifacts, role),
            "method_id": recipe_id,
            "recipe_id": recipe_id,
            "architecture_variant": recipe["architecture_variant"],
            "optimizer_update": 0,
            "zero_step_alias": "R1-D0",
        }
    started = time.monotonic()
    try:
        result = run_checkpoint_evaluation(
            variant=recipe["architecture_variant"],
            optimizer_update=update,
            checkpoint=checkpoint,
            scorer_checkpoint=(
                Path(assets["scorer_checkpoint"])
                if recipe["architecture_variant"] == "Q-SEM"
                else None
            ),
            role=role,
            assets_path=external_root / "assets.local.json",
            roles_path=artifacts / "DATA_ROLES.json",
            external_root=external_root,
            output_path=path,
            recipe_config=recipe,
            artifact_root=artifacts,
        )
    finally:
        _step_cost(
            artifacts,
            phase="development",
            recipe_id=recipe_id,
            start=started,
            gpu_count=1,
        )
    return {
        **result["candidate"],
        "recipe_id": recipe_id,
        "architecture_variant": recipe["architecture_variant"],
    }


def _safe_evaluate(config: dict, **kwargs) -> dict:
    try:
        return evaluate(config, **kwargs)
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        return {
            "method_id": kwargs["recipe_id"],
            "recipe_id": kwargs["recipe_id"],
            "optimizer_update": kwargs["update"],
            "coverage_status": "INCOMPLETE",
            "metrics": {},
            "status": "BLOCKED",
            "reason": str(error),
        }


def comparable_cohort(external_root: Path) -> dict:
    assets = read_json(external_root / "assets.local.json")
    paths = {
        name: external_root / f"training/{name}/run_plan.json"
        for name in ("S-BAL-H-s45", "C0-L-s45", "S-BAL-L-s45")
    }
    paths["C0-H-s45"] = Path(assets["C0_H_0750"]).parent / "run_plan.json"
    identities = {}
    for name, path in paths.items():
        if not path.exists():
            return {"comparable": False, "reason": f"Missing actual run plan: {name}"}
        plan = read_json(path)
        identities[name] = {
            "contract": plan["contract"],
            "plan": {
                key: value
                for key, value in plan["plan"].items()
                if key != "next_global_draw_index"
            },
            "seed": plan["seed"],
            "r1_checkpoint_sha256": plan["r1_checkpoint_sha256"],
        }
    values = list(identities.values())
    return {
        "comparable": all(value == values[0] for value in values),
        "runtime_and_draw_identities": identities,
        "legacy_numerics_check": "validated by live checkpoint loader before H metrics enter selection",
    }


def run_aux(config: dict, *, external_root: Path) -> dict:
    artifacts = PROJECT_ROOT / config["artifact_root"]
    baseline = baseline_candidate(artifacts, "CAL")
    if baseline["coverage_status"] != "COMPLETE":
        result = {
            "status": "BLOCKED",
            "reason": "Complete R1 CAL comparison is unavailable",
            "probes": {"H": [], "L": []},
            "arms": {},
        }
        write_json(artifacts / "selection/AUX.json", result)
        return result
    probes = {label: [] for label in ("H", "L")}
    for label, probe_rows in probes.items():
        for variant in ("C0", "S-BAL"):
            for update in (250, 750):
                probe_rows.append(
                    _safe_evaluate(
                        config,
                        external_root=external_root,
                        recipe_id=f"{variant}-{label}-s45",
                        update=update,
                        role="CAL",
                    )
                )
    cohort = comparable_cohort(external_root)
    high_complete = all(row["coverage_status"] == "COMPLETE" for row in probes["H"])
    choice = select_learning_rate(
        probes, baseline, comparable_high=cohort["comparable"] and high_complete
    )
    choice["cohort_audit"] = cohort
    choice["all_probes"] = probes
    write_json(artifacts / "selection/LEARNING_RATE.json", choice)
    label = choice["learning_rate_label"]
    rows = {}
    cost = measured_update_cost(external_root)
    # Reserve the whole matched extension before optional pilots. Thus S-WORST is pruned first.
    for variant in ("A-OPEN", "Q-SEM", "S-WORST"):
        name = f"{variant}-{label}-s45"
        decision = budget_decision(
            config,
            external_root=external_root,
            label=f"pilot:{name}+paired-full",
            predicted_gpu_hours=cost * (750 + 2 * 2250),
        )
        if not decision["affordable"]:
            rows[name] = {"status": "SKIPPED_BUDGET", "budget": decision}
            continue
        points = []
        if variant in {"A-OPEN", "Q-SEM"}:
            zero = _safe_evaluate(
                config,
                external_root=external_root,
                recipe_id=name,
                update=0,
                role="CAL",
            )
            points.append(zero)
            delta = compare(zero, baseline)
            if delta["comparison_status"] != "COMPLETE":
                rows[name] = {
                    "status": "BLOCKED",
                    "reason": "Zero-step CAL incomplete",
                    "cal": points,
                }
                continue
            if delta["S_mean"] < -0.03 and delta["S_min"] < -0.05:
                rows[name] = {"status": "SKIPPED_SEVERE_ZERO_STEP", "cal": points}
                continue
        started = time.monotonic()
        try:
            summary = train_recipe(
                config, recipe_id=name, endpoint=750, external_root=external_root
            )
        except (OSError, RuntimeError, ValueError) as error:
            rows[name] = {"status": "BLOCKED", "reason": str(error), "cal": points}
            continue
        finally:
            _step_cost(
                artifacts,
                phase="perception",
                recipe_id=name,
                start=started,
                gpu_count=config["runtime"]["perception_devices"],
                updates=750,
            )
        points.extend(
            _safe_evaluate(
                config,
                external_root=external_root,
                recipe_id=name,
                update=update,
                role="CAL",
            )
            for update in (250, 750)
        )
        rows[name] = {
            "status": summary["status"],
            "cal": points,
            "completed_global_step": summary["completed_global_step"],
        }
        write_json(
            artifacts / "selection/AUX.json",
            {"status": "IN_PROGRESS", "learning_rate": label, "arms": rows},
        )
    result = {
        "status": "COMPLETE",
        "learning_rate": label,
        "arms": rows,
        "probes": probes,
    }
    write_json(artifacts / "selection/AUX.json", result)
    return result


def run_perception(config: dict, *, external_root: Path) -> dict:
    artifacts = PROJECT_ROOT / config["artifact_root"]
    baseline = baseline_candidate(artifacts, "CAL")
    aux_path = artifacts / "selection/AUX.json"
    if not aux_path.exists():
        result = {
            "status": "BLOCKED",
            "reason": "Pilot selection unavailable",
            "selected": baseline_candidate(artifacts, "SEL"),
            "full_training_recipes": [],
        }
        write_json(artifacts / "selection/PERCEPTION.json", result)
        return result
    aux = read_json(aux_path)
    points = [row for values in aux["probes"].values() for row in values]
    points += [
        row
        for value in aux["arms"].values()
        if value.get("completed_global_step") == 750
        for row in value["cal"]
        if row["optimizer_update"] in {250, 750}
    ]
    completed = set()
    for row in points:
        name = row["method_id"]
        path = external_root / f"training/{name}/run_summary.json"
        if name == "C0-H-s45" or (
            path.exists() and read_json(path).get("completed_global_step", 0) >= 750
        ):
            completed.add(name)
    points = [row for row in points if row["method_id"] in completed]
    controls = {
        name: "C0-" + name.rsplit("-", 2)[-2] + "-s45"
        for name in completed
        if not name.startswith("C0-")
    }
    controls.update({"C0-fallback-H": "C0-H-s45", "C0-fallback-L": "C0-L-s45"})
    promotion = select_full_promotion(points, baseline, control_by_method=controls)
    write_json(artifacts / "selection/FULL_PROMOTION.json", promotion)
    methods = promotion["full_training_recipes"]
    if methods:
        decision = budget_decision(
            config,
            external_root=external_root,
            label="full:" + "+".join(methods),
            predicted_gpu_hours=measured_update_cost(external_root)
            * remaining_full_updates(methods, artifacts, external_root),
        )
        if not decision["affordable"]:
            result = {
                "status": "SKIPPED_BUDGET",
                "budget": decision,
                "selected": baseline_candidate(artifacts, "SEL"),
                "full_training_recipes": [],
                "promotion": promotion,
            }
            write_json(artifacts / "selection/PERCEPTION.json", result)
            return result
    completed_full, failures = [], {}
    for name in methods:
        started = time.monotonic()
        try:
            imported = resolve_checkpoint(name, 750, external_root=external_root)
            legacy = (
                imported.parent / "resolved_config.yaml"
                if name == "C0-H-s45"
                and imported.parent != external_root / f"training/{name}"
                else None
            )
            summary = train_recipe(
                config,
                recipe_id=name,
                endpoint=3000,
                external_root=external_root,
                import_resume=imported,
                legacy_config=legacy,
            )
            if (
                summary["status"] == "COMPLETE"
                and summary["completed_global_step"] == 3000
            ):
                completed_full.append(name)
        except (OSError, RuntimeError, ValueError) as error:
            failures[name] = str(error)
        finally:
            _step_cost(
                artifacts,
                phase="perception",
                recipe_id=name,
                start=started,
                gpu_count=config["runtime"]["perception_devices"],
                updates=2250,
            )
    selected_cal = {}
    for name in completed_full:
        rows = [
            _safe_evaluate(
                config,
                external_root=external_root,
                recipe_id=name,
                update=update,
                role="CAL",
            )
            for update in (0, 250, 750, 1500, 2250, 3000)
        ]
        ranking = rank(rows, baseline)
        # A full CAL schedule is required; unavailable points are not silently omitted.
        if len(ranking) == 6:
            selected_cal[name] = ranking[0]
        else:
            failures[name] = "Full CAL schedule incomplete"
    write_json(artifacts / "selection/PERCEPTION_CAL_LOCK.json", selected_cal)
    sel = [
        _safe_evaluate(
            config,
            external_root=external_root,
            recipe_id=name,
            update=row["optimizer_update"],
            role="SEL",
        )
        for name, row in selected_cal.items()
    ]
    c0_id = methods[0] if methods else "C0-NOT-RUN"
    choice = select_perception(sel, baseline_candidate(artifacts, "SEL"), c0_id=c0_id)
    chosen = choice["selected"]
    chosen["component_gate_passed"] = True
    same_update = {}
    for name, row in selected_cal.items():
        if name != c0_id and c0_id in selected_cal:
            control = _safe_evaluate(
                config,
                external_root=external_root,
                recipe_id=c0_id,
                update=row["optimizer_update"],
                role="CAL",
            )
            same_update[name] = compare(row, control)
    result = {
        "status": "COMPLETE" if not failures else "BLOCKED",
        **choice,
        "selected": chosen,
        "full_training_recipes": methods,
        "completed_full": completed_full,
        "cal_selected": selected_cal,
        "same_update_cal": same_update,
        "failures": failures,
        "promotion": promotion,
    }
    write_json(artifacts / "selection/PERCEPTION.json", result)
    return result
