"""Fixed-step seed-46 replication of the deployed V2 components, without selection."""

from __future__ import annotations

import time
from pathlib import Path

from scripts.perception_gain_v2 import (
    PROJECT_ROOT,
    file_hash,
    read_json,
    train_recipe,
    utc_now,
    write_json,
)
from scripts.perception_gain_v2_config import live_execution_provenance
from scripts.perception_gain_v2_perception import (
    baseline_candidate,
    evaluate,
    measured_update_cost,
    resolve_checkpoint,
)
from scripts.perception_gain_v2_selection import compare


def replication_plan(lock: dict) -> dict:
    final = lock["final_method"]
    parent = lock["methods"][lock["aliases"]["FINAL-parent-D0"]]
    selected = lock["methods"][lock["aliases"]["P-D0"]]
    perception, endpoints, required = [], {}, []
    for original in (selected, parent):
        update = original["parent_optimizer_update"]
        if update == 0:
            continue
        architecture = original["architecture_variant"]
        rate = original["recipe"]["learning_rate_label"]
        recipe_id = f"{architecture}-{rate}-s46"
        component_id = f"perception:{recipe_id}@{update:04d}"
        if original is parent and component_id not in required:
            required.append(component_id)
        if any(row["component_id"] == component_id for row in perception):
            continue
        control = None if architecture == "C0" else f"C0-{rate}-s46"
        perception.append(
            {
                "component_id": component_id,
                "original_recipe_id": original["recipe_id"],
                "recipe_id": recipe_id,
                "optimizer_update": update,
                "control_recipe_id": control,
            }
        )
        for name in (recipe_id, control):
            if name is not None:
                endpoints[name] = max(endpoints.get(name, 0), update)
    repair = None
    if final.get("refiner") is not None:
        # The deployed head binding identifies the original parent. Do not replace
        # an adapted P cache with the available R1 cache merely to finish cheaply.
        parent_name = next(
            (
                name
                for name, selection in lock["repair_selections"].items()
                if (selection.get("adopted_candidate") or {}).get("method_id")
                == final["method_id"]
            ),
            None,
        )
        if parent_name is None:
            candidates = [
                name
                for name, selection in lock["repair_selections"].items()
                if selection.get("cal_selected")
            ]
            if len(candidates) != 1:
                raise ValueError(
                    "Deployed refiner has no unambiguous frozen parent selection"
                )
            parent_name = candidates[0]
        selection = lock["repair_selections"][parent_name]["cal_selected"]
        fixed = {
            name: row["optimizer_update"]
            for name, row in selection.items()
            if row.get("optimizer_update") is not None
        }
        selected_step = final["refiner"]["optimizer_update"]
        updates = sorted(set([*fixed.values(), selected_step]))
        if (
            any(step not in {0, 500, 1000, 1500} for step in updates)
            or max(updates) == 0
        ):
            raise ValueError(
                "Deployed refiner must have a trained, fixed V2 checkpoint"
            )
        new_parent = parent["parent_optimizer_update"] > 0
        parent_recipe_id = (
            f"{parent['architecture_variant']}-{parent['recipe']['learning_rate_label']}-s46"
            if new_parent
            else parent["recipe_id"]
        )
        repair = {
            "component_id": f"repair:{parent_name}",
            "parent_id": parent_name,
            "parent_recipe_id": parent_recipe_id,
            "parent_optimizer_update": parent["parent_optimizer_update"],
            "fresh_parent_cache": new_parent,
            "stop_after_updates": max(updates),
            "fixed_updates": fixed,
            "evaluation_updates": updates,
            "deployed_mode": final["refiner"]["input_mode"],
            "deployed_update": selected_step,
        }
        required.append(repair["component_id"])
    return {
        "training_seed": 46,
        "evaluation_seed": 45,
        "perception": perception,
        "training_endpoints": endpoints,
        "repair": repair,
        "required_deployment_components": required,
        "association_replication": (
            "NOT_APPLICABLE" if final.get("association_config") else None
        ),
        "zero_step_perception": (
            "NOT_APPLICABLE" if selected["parent_optimizer_update"] == 0 else None
        ),
        "scorer_condition": "The original frozen scorer500 is reused; it is not retrained for seed46.",
    }


def component_result(candidate: dict, baselines: dict[str, dict]) -> dict:
    comparisons = {name: compare(candidate, value) for name, value in baselines.items()}
    complete = bool(comparisons) and all(
        row["comparison_status"] == "COMPLETE" for row in comparisons.values()
    )
    supported = complete and all(
        row["S_mean"] > 0.0 and row["S_min"] >= -0.002 for row in comparisons.values()
    )
    return {
        "replication": (
            "SUPPORTED" if supported else "MIXED" if complete else "NOT_COMPLETED"
        ),
        "comparisons": comparisons,
        "candidate": candidate,
    }


def pipeline_result(required: list[str], components: dict) -> str:
    if not required:
        return "NOT_APPLICABLE"
    rows = [components.get(name, {}).get("replication") for name in required]
    if any(status not in {"SUPPORTED", "MIXED"} for status in rows):
        return "NOT_COMPLETED"
    return "SUPPORTED" if all(status == "SUPPORTED" for status in rows) else "MIXED"


def commit_replication_scope(
    lock: dict, *, artifacts: Path, external_root: Path
) -> dict:
    """Make the spend/scope decision before any PB metrics can be observed."""
    plan = replication_plan(lock)
    available = lock["confirmation_allocation"]["replication_and_recovery"]
    approved, costs, endpoints = [], {}, {}
    if plan["perception"]:
        update_cost = measured_update_cost(external_root)
        sel_seconds = read_json(artifacts / "foundation/SEL/D0.json")["elapsed_seconds"]
    for item in plan["perception"]:
        needed = {
            name: item["optimizer_update"]
            for name in (item["recipe_id"], item["control_recipe_id"])
            if name is not None
        }
        # The scheduler reserves both training GPUs even during one-GPU evaluation.
        estimated = (
            sum(max(0, step - endpoints.get(name, 0)) for name, step in needed.items())
            * update_cost
            + 2 * sel_seconds * len(needed) / 3600
        ) * 1.25
        allowed = estimated <= available
        costs[item["component_id"]] = {
            "predicted_reserved_gpu_hours": estimated,
            "approved": allowed,
        }
        if allowed:
            available -= estimated
            approved.append(item["component_id"])
            for name, step in needed.items():
                endpoints[name] = max(endpoints.get(name, 0), step)
    if repair := plan["repair"]:
        source = artifacts / f"refiner/{repair['parent_id']}"
        # A prior full CAL run includes eight heads; using its whole runtime for
        # the fixed-point SEL run is conservative and includes official metrics.
        evaluation_seconds = read_json(source / "CAL.json")["elapsed_seconds"]
        train_seconds = sum(
            read_json(source / f"{label}/TRAINING.json")["elapsed_seconds"]
            for label in ("NEW", "PAIR")
        )
        cache_seconds = (
            read_json(source / "DATA_MANIFEST.json")["elapsed_seconds"]
            if repair["fresh_parent_cache"]
            else 0.0
        )
        estimated = (
            2 * (evaluation_seconds + train_seconds + cache_seconds) / 3600 * 1.25
        )
        parent_ready = not repair["fresh_parent_cache"] or (
            endpoints.get(repair["parent_recipe_id"], 0)
            >= repair["parent_optimizer_update"]
        )
        allowed = parent_ready and estimated <= available
        costs[repair["component_id"]] = {
            "predicted_reserved_gpu_hours": estimated,
            "approved": allowed,
        }
        if allowed:
            available -= estimated
            approved.append(repair["component_id"])
    return {
        "plan": plan,
        "approved_components": approved,
        "approved_training_endpoints": endpoints,
        "forecast": costs,
        "safety_factor": 1.25,
        "reserved_gpu_count": 2,
        "uncommitted_recovery_gpu_hours": available,
        "decision_timing": "FINAL_LOCK_BEFORE_ANY_PB_METRIC",
        "selection_on_seed46": False,
    }


def _repair_replication(
    config: dict, *, external_root: Path, lock: dict, plan: dict
) -> dict:
    from scripts.perception_gain_v2_confirmation import method_inputs
    from scripts.perception_gain_v2_evidence import validate_refiner_cache
    from scripts.perception_gain_v2_lock import effective_parent_weight_identity
    from scripts.prepare_perception_refiner import run as prepare
    from scripts.train_perception_refiner import refiner_v2_binding, train_mask_refiner
    from scripts.perception_refiner_evaluation import run_refiner_evaluation

    artifacts = PROJECT_ROOT / config["artifact_root"]
    spec = plan["repair"]
    original = lock["methods"][lock["aliases"]["FINAL-parent-D0"]]
    inputs = method_inputs(original, external_root=external_root, artifacts=artifacts)
    recipe = read_json(artifacts / f"training/{spec['parent_recipe_id']}/recipe.json")
    checkpoint = resolve_checkpoint(
        spec["parent_recipe_id"],
        spec["parent_optimizer_update"],
        external_root=external_root,
    )
    weight = effective_parent_weight_identity(
        (
            file_hash(checkpoint)
            if checkpoint
            else original["parent_checkpoint_file_sha256"]
        ),
        architecture=recipe["architecture_variant"],
        update=spec["parent_optimizer_update"],
        scorer_sha256=original.get("scorer_sha256"),
    )
    public = artifacts / f"replication/refiner/{spec['parent_id']}"
    original_manifest = read_json(
        artifacts / f"refiner/{spec['parent_id']}/DATA_MANIFEST.json"
    )
    if spec["fresh_parent_cache"]:
        cache_root = external_root / f"cache/refiner/{spec['parent_id']}-s46"
        manifest_path = public / "DATA_MANIFEST.json"
        if not manifest_path.exists():
            prepare(
                variant=recipe["architecture_variant"],
                optimizer_update=spec["parent_optimizer_update"],
                checkpoint=checkpoint,
                scorer_checkpoint=inputs["scorer_checkpoint"],
                assets_path=inputs["assets_path"],
                roles_path=artifacts / "DATA_ROLES.json",
                external_root=external_root,
                device_name="cuda:0",
                output_root=cache_root,
                manifest_output=manifest_path,
                recipe_config=recipe,
                required_inventory=original_manifest["inventory"],
            )
        manifest = read_json(manifest_path)
    else:
        cache_root = external_root / f"cache/refiner/{spec['parent_id']}"
        manifest = original_manifest
    audit = validate_refiner_cache(
        manifest,
        recipe=recipe,
        parent_weight=weight,
        artifacts=artifacts,
        cache_root=cache_root,
        required_inventory=original_manifest["inventory"],
        external_root=external_root,
    )
    write_json(public / "CACHE_REUSE_AUDIT.json", audit)
    shards = tuple(cache_root / row["path"] for row in manifest["shards"])
    summaries, heads = {}, []
    for label, mode in (("NEW", "NEW_ONLY"), ("PAIR", "OLD_NEW")):
        binding = refiner_v2_binding(
            input_mode=mode,
            parent_recipe_hash=recipe["inference_recipe_hash"],
            parent_weight_hash=weight,
            cache_paths=shards,
            seed=46,
        )
        output = external_root / f"training/refiner/{spec['parent_id']}/{label}-s46"
        summary_path = output / "run_summary.json"
        summary = read_json(summary_path) if summary_path.exists() else None
        if summary is not None and summary.get("binding") != binding:
            raise ValueError("Seed46 head belongs to a different parent/cache/seed")
        if (
            summary is None
            or summary["completed_updates"] != spec["stop_after_updates"]
        ):
            resume = output / "last.ckpt"
            summary = train_mask_refiner(
                cache_paths=shards,
                output_dir=output,
                binding=binding,
                seed=46,
                stop_after_updates=spec["stop_after_updates"],
                device="cuda:0",
                resume=resume if resume.exists() else None,
            )
        summaries[label] = summary
        write_json(public / f"{label}/TRAINING.json", summary)
        for update in spec["evaluation_updates"]:
            heads.append(
                {
                    "method_id": f"R-{label}-{spec['parent_id']}-s46@{update:04d}",
                    "checkpoint": output / f"update={update:04d}.ckpt",
                    "optimizer_update": update,
                    "binding": binding,
                }
            )
    if (
        summaries["NEW"]["initial_state_sha256"]
        != summaries["PAIR"]["initial_state_sha256"]
    ):
        raise ValueError("Seed46 paired heads did not share the same initialization")
    cache_key = {
        "final_lock_sha256": lock["content_sha256"],
        "parent_weight_sha256": weight,
        "inference_recipe_hash": recipe["inference_recipe_hash"],
        "source": live_execution_provenance(recipe)["relevant_source_digest"],
        "heads": {head["method_id"]: file_hash(head["checkpoint"]) for head in heads},
        "role": "SEL",
        "eval_seed": 45,
        "training_seed": 46,
    }
    output_path = public / "SEL.json"
    key_path = public / "SEL_BINDING.json"
    if output_path.exists():
        if not key_path.exists() or read_json(key_path) != cache_key:
            raise ValueError(
                "Seed46 evaluation binding changed; explicit recovery required"
            )
        evaluation = read_json(output_path)
    else:
        write_json(key_path, cache_key)
        evaluation = run_refiner_evaluation(
            parent_variant=recipe["architecture_variant"],
            parent_update=spec["parent_optimizer_update"],
            parent_checkpoint=checkpoint,
            scorer_checkpoint=inputs["scorer_checkpoint"],
            refiner_update=0,
            refiner_checkpoint=None,
            refiner_checkpoints=heads,
            role="SEL",
            assets_path=inputs["assets_path"],
            roles_path=artifacts / "DATA_ROLES.json",
            external_root=external_root,
            recipe_config=recipe,
            artifact_root=artifacts,
            output_path=output_path,
        )
    candidates = evaluation["candidates"]

    def candidate(label, update):
        return candidates.get(f"R-{label}-{spec['parent_id']}-s46@{update:04d}", {})

    label = "PAIR" if spec["deployed_mode"] == "OLD_NEW" else "NEW"
    baselines = {"parent": evaluation["parent_candidate"]}
    if label == "PAIR":
        baselines["NEW-at-fixed-s45-step"] = candidate(
            "NEW", spec["fixed_updates"].get("NEW", -1)
        )
    result = component_result(candidate(label, spec["deployed_update"]), baselines)
    result.update(
        {
            "training": summaries,
            "parent_recipe_id": spec["parent_recipe_id"],
            "fresh_parent_cache": spec["fresh_parent_cache"],
            "fixed_point_candidates": {
                mode: candidate(mode, update)
                for mode, update in spec["fixed_updates"].items()
            },
            "same_step_PAIR_minus_NEW": {
                str(update): compare(
                    candidate("PAIR", update), candidate("NEW", update)
                )
                for update in spec["evaluation_updates"]
            },
            "evaluation_status": evaluation["status"],
        }
    )
    write_json(public / "RESULT.json", result)
    return result


def run_replication(config: dict, *, external_root: Path) -> dict:
    from scripts.perception_gain_v2_confirmation import load_lock
    from scripts.perception_gain_v2_perception import _step_cost

    artifacts = PROJECT_ROOT / config["artifact_root"]
    lock, lock_sha = load_lock(config, artifacts)
    commitment = lock["replication_commitment"]
    plan = replication_plan(lock)
    if plan != commitment["plan"]:
        raise ValueError("Seed46 scope differs from the pre-PB lock")
    public = artifacts / "replication"
    destination = public / "REPLICATION.json"
    if destination.exists():
        previous = read_json(destination)
        if previous["final_lock_sha256"] != lock_sha:
            raise ValueError("Replication belongs to another final lock")
        # No implicit retry of a completed negative result or the same failure.
        return previous
    started = time.monotonic()
    components, training, failures = {}, {}, {}
    approved = set(commitment["approved_components"])
    for recipe_id, endpoint in commitment["approved_training_endpoints"].items():
        try:
            recipe = read_json(artifacts / f"training/{recipe_id}/recipe.json")
            original = read_json(
                artifacts / f"training/{recipe_id.removesuffix('46')}45/recipe.json"
            )
            if recipe != {
                **original,
                "recipe_id": recipe_id,
                "train_seed": 46,
                "training_recipe_hash": recipe["training_recipe_hash"],
            }:
                raise ValueError("Seed46 changes more than the training seed")
            training[recipe_id] = train_recipe(
                config,
                recipe_id=recipe_id,
                endpoint=endpoint,
                external_root=external_root,
            )
        except (OSError, RuntimeError, ValueError, KeyError) as error:
            failures[recipe_id] = str(error)
    baseline = baseline_candidate(artifacts, "SEL")
    for item in plan["perception"]:
        name = item["component_id"]
        if name not in approved:
            continue
        try:
            if any(
                recipe_id in failures
                for recipe_id in (item["recipe_id"], item["control_recipe_id"])
            ):
                raise ValueError(
                    "Required seed46 parent/control training did not complete"
                )
            candidate = evaluate(
                config,
                external_root=external_root,
                recipe_id=item["recipe_id"],
                update=item["optimizer_update"],
                role="SEL",
            )
            baselines = {"R1": baseline}
            if item["control_recipe_id"]:
                baselines["same-LR-same-step-C0-46"] = evaluate(
                    config,
                    external_root=external_root,
                    recipe_id=item["control_recipe_id"],
                    update=item["optimizer_update"],
                    role="SEL",
                )
            components[name] = component_result(candidate, baselines)
        except (OSError, RuntimeError, ValueError, KeyError) as error:
            failures[name] = str(error)
    if plan["repair"] and plan["repair"]["component_id"] in approved:
        name = plan["repair"]["component_id"]
        try:
            if plan["repair"]["parent_recipe_id"] in failures:
                raise ValueError("Deployed seed46 parent training failed")
            components[name] = _repair_replication(
                config, external_root=external_root, lock=lock, plan=plan
            )
        except (OSError, RuntimeError, ValueError, KeyError) as error:
            failures[name] = str(error)
    skipped = [name for name in commitment["forecast"] if name not in approved]
    incomplete = any(
        row["replication"] == "NOT_COMPLETED" for row in components.values()
    )
    pipeline = pipeline_result(plan["required_deployment_components"], components)
    report = {
        "status": (
            "BLOCKED"
            if failures or incomplete
            else "SKIPPED_BUDGET" if skipped else "COMPLETE"
        ),
        "final_lock_sha256": lock_sha,
        "created_utc": utc_now(),
        "replication_status": (
            "BLOCKED"
            if failures or incomplete
            else "NOT_RUN_BUDGET" if pipeline == "NOT_COMPLETED" else pipeline
        ),
        "pipeline_replication": pipeline,
        "training_seed": 46,
        "evaluation_seed": 45,
        "plan": plan,
        "components": components,
        "training": training,
        "skipped_budget": skipped,
        "failures": failures,
        "deployed_seed45_system_changed": False,
        "executed_source_sha256": file_hash(Path(__file__)),
    }
    _step_cost(
        artifacts,
        phase="replication",
        recipe_id="fixed-s46",
        start=started,
        gpu_count=2,
    )
    write_json(destination, report)
    return report
