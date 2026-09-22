"""Immutable V2 deployment identities and the finite confirmation inventory."""

from __future__ import annotations

import hashlib
from pathlib import Path

from scripts.perception_gain_v2 import (
    PROJECT_ROOT,
    append_event,
    file_hash,
    read_json,
    utc_now,
    write_json,
)
from scripts.perception_gain_v2_config import R1_SHA256, content_hash, executed_identity
from scripts.perception_gain_v2_perception import baseline_candidate, resolve_checkpoint
from scripts.perception_gain_v2_selection import compare, gate, select_final


def effective_parent_weight_identity(
    checkpoint_sha256: str,
    *,
    architecture: str,
    update: int,
    scorer_sha256: str | None,
) -> str:
    if architecture == "Q-SEM" and update == 0:
        if scorer_sha256 is None:
            raise ValueError("Zero-step Q requires its separate frozen scorer")
        return hashlib.sha256(
            f"{checkpoint_sha256}:Q-SEM:{scorer_sha256}".encode("ascii")
        ).hexdigest()
    return checkpoint_sha256


def inference_identity(method: dict) -> str:
    architecture = method["architecture_variant"]
    if method["parent_weight_sha256"] == R1_SHA256 and architecture in {
        "C0",
        "S-BAL",
        "S-WORST",
    }:
        architecture = "C0"
    head = method.get("refiner")
    return content_hash(
        {
            "kind": method["kind"],
            "architecture": architecture,
            "parent_weight_sha256": method["parent_weight_sha256"],
            "scorer_sha256": method.get("scorer_sha256"),
            "head": (
                {key: head[key] for key in ("checkpoint_sha256", "input_mode")}
                if head
                else None
            ),
            "association_config": method.get("association_config"),
            "eval_seed": 45,
            "state_capacity": 100,
            "publisher": "native" if method["kind"] == "NATIVE" else "lag1/mean",
        }
    )


def external_path(reference: str, external_root: Path) -> Path:
    if not isinstance(reference, str) or not reference.startswith("external:"):
        raise ValueError("Method weight requires an external artifact reference")
    path = external_root / reference.removeprefix("external:")
    if not path.resolve().is_relative_to(external_root.resolve()):
        raise ValueError("Method weight escapes the bound external root")
    return path


def describe(
    candidate: dict,
    *,
    parent: dict | None,
    artifacts: Path,
    external_root: Path,
    kind: str | None = None,
) -> dict:
    parent = candidate if parent is None else parent
    recipe_id = parent["recipe_id"]
    recipe = read_json(artifacts / f"training/{recipe_id}/recipe.json")
    assets = read_json(external_root / "assets.local.json")
    update = parent["optimizer_update"]
    checkpoint = resolve_checkpoint(recipe_id, update, external_root=external_root)
    checkpoint_file_sha256 = file_hash(
        checkpoint if checkpoint is not None else Path(assets["r1_checkpoint"])
    )
    scorer_sha256 = (
        file_hash(Path(assets["scorer_checkpoint"]))
        if recipe["architecture_variant"] == "Q-SEM"
        else None
    )
    weight = effective_parent_weight_identity(
        checkpoint_file_sha256,
        architecture=recipe["architecture_variant"],
        update=update,
        scorer_sha256=scorer_sha256,
    )
    if parent.get("checkpoint_sha256") != weight:
        raise ValueError("Selected parent weight differs from evaluated identity")
    association = candidate.get("association_config")
    head = None
    if "binding" in candidate:
        binding = candidate["binding"]
        if (
            binding["parent_weight_hash"] != weight
            or binding["parent_recipe_hash"] != recipe["inference_recipe_hash"]
        ):
            raise ValueError("Selected repair is bound to another parent recipe/weight")
        path = external_path(candidate["checkpoint"], external_root)
        if file_hash(path) != candidate["checkpoint_sha256"]:
            raise ValueError("Selected repair checkpoint changed")
        head = {
            key: candidate[key]
            for key in (
                "checkpoint",
                "checkpoint_sha256",
                "optimizer_update",
                "binding",
            )
        }
        head["input_mode"] = binding["input_mode"]
    if association is not None and head is not None:
        raise ValueError("V2 does not combine association with refinement")
    kind = kind or ("REFINER" if head else "ASSOCIATION" if association else "D0")
    method = {
        "method_id": candidate["method_id"],
        "kind": kind,
        "recipe_id": recipe_id,
        "architecture_variant": recipe["architecture_variant"],
        "recipe": recipe,
        "parent_optimizer_update": update,
        "parent_checkpoint": (
            "external:" + str(checkpoint.relative_to(external_root))
            if checkpoint
            else None
        ),
        "parent_weight_sha256": weight,
        "parent_checkpoint_file_sha256": checkpoint_file_sha256,
        "refiner": head,
        "association_config": association,
        "scorer_checkpoint": (
            "asset:scorer_checkpoint"
            if recipe["architecture_variant"] == "Q-SEM"
            else None
        ),
        "scorer_sha256": scorer_sha256,
        "eval_seed": 45,
        "training_seed": recipe["train_seed"],
        "state_capacity": 100,
        "publisher": "native" if kind == "NATIVE" else "lag1/mean",
        "window_size": None if kind == "NATIVE" else 2,
    }
    method["inference_identity"] = inference_identity(method)
    return method


def run_lock(config: dict, *, external_root: Path) -> dict:
    from scripts.perception_gain_evaluation import write_immutable_lock
    from scripts.perception_gain_local_evaluation import local_evaluation_seeds
    from scripts.p6a_metrics import official_temporal_iou_thresholds
    from scripts.perception_gain_v2_confirmation import confirmation_budget
    from scripts.perception_gain_v2_replication import commit_replication_scope

    artifacts = PROJECT_ROOT / config["artifact_root"]
    destination = artifacts / "selection/FINAL_LOCK.json"
    if destination.exists():
        locked = read_json(destination)
        if locked["content_sha256"] != content_hash(
            {key: value for key, value in locked.items() if key != "content_sha256"}
        ):
            raise ValueError("Immutable final lock identity differs")
        for field, path in (
            ("data_roles_sha256", artifacts / "DATA_ROLES.json"),
            ("input_manifest_sha256", artifacts / "data/STAGING_MANIFEST.json"),
            ("run_config_sha256", artifacts / "RUN_CONFIG.json"),
        ):
            if locked[field] != file_hash(path):
                raise ValueError(
                    "Locked confirmation dependency changed; explicit invalidation required"
                )
        return {
            "status": "COMPLETE",
            "lock_sha256": file_hash(destination),
            "selected_method_id": locked["final_method_id"],
            "confirmation_reserve_gpu_hours": max(
                config["budget"]["confirmation_reserve"],
                locked.get("confirmation_forecast", {}).get(
                    "required_confirmation_gpu_hours", 0.0
                ),
            ),
        }

    def optional(path):
        return read_json(path) if path.exists() else {"status": "NOT_RUN"}

    r1 = baseline_candidate(artifacts, "SEL")
    perception = optional(artifacts / "selection/PERCEPTION.json")
    parent = perception.get("selected", r1)
    repairs = {
        name: optional(artifacts / f"refiner/{name}/SELECTION.json")
        for name in ("R1", "P")
    }
    association = optional(artifacts / "association/SELECTION.json")
    candidates, descriptors, aliases, omitted = [], {}, {}, []

    def register(label, candidate, *, parent_candidate=None, kind=None, eligible=False):
        method = describe(
            candidate,
            parent=parent_candidate,
            artifacts=artifacts,
            external_root=external_root,
            kind=kind,
        )
        identity = method["inference_identity"]
        if identity not in descriptors:
            descriptors[identity] = method
        else:
            method = descriptors[identity]
        aliases[label] = method["method_id"]
        if eligible and not any(
            row["method_id"] == method["method_id"] for row in candidates
        ):
            candidates.append({**candidate, "method_id": method["method_id"]})
        return method

    register("R1-D0", r1, eligible=True)
    register("P-D0", parent, eligible=True)
    for name, result in repairs.items():
        if result.get("adopted_candidate") is not None:
            candidate = result["adopted_candidate"]
            try:
                register(
                    f"{name}-repair",
                    candidate,
                    parent_candidate=r1 if name == "R1" else parent,
                    eligible=True,
                )
            except (OSError, ValueError, KeyError) as error:
                omitted.append({"method": f"{name}-repair", "reason": str(error)})
        else:
            omitted.append(
                {
                    "method": f"{name}-repair",
                    "status": result.get("status"),
                    "selection": result.get("selection"),
                }
            )
    if association.get("adopted_candidate") is not None:
        register(
            "R1-association",
            association["adopted_candidate"],
            parent_candidate=r1,
            eligible=True,
        )
    else:
        omitted.append(
            {"method": "R1-association", "status": association.get("status")}
        )
    c0 = None
    for name in perception.get("completed_full", []):
        if name.startswith("C0-") and name in perception.get("cal_selected", {}):
            c0 = perception["cal_selected"][name]
            selected = next(
                (
                    row
                    for row in perception.get("evaluated", [])
                    if row["method_id"] == name
                ),
                None,
            )
            if selected is not None and gate(
                compare(selected, r1), mean=0.005, minimum=-0.001
            ):
                register(
                    "C0-best-D0",
                    {**selected, "component_gate_passed": True},
                    eligible=True,
                )
            else:
                register("C0-best-D0", c0)
    decision = select_final(candidates, r1)
    selected_id = decision["selected"]["method_id"]
    final = next(
        value for value in descriptors.values() if value["method_id"] == selected_id
    )
    aliases["FINAL"] = selected_id

    r1_native = register(
        "FH-R1-native", {**r1, "method_id": "FH-R1-native"}, kind="NATIVE"
    )
    p_native = register(
        "FH-P-native", {**parent, "method_id": "FH-P-native"}, kind="NATIVE"
    )
    c0_native = (
        register("C0-best-native", {**c0, "method_id": "C0-best-native"}, kind="NATIVE")
        if c0
        else None
    )
    pb = [
        aliases[label]
        for label in ("FH-R1-native", "R1-D0", "P-D0", "FH-P-native", "FINAL")
    ]
    if c0:
        pb.append(aliases["C0-best-D0"])
    if final["kind"] == "REFINER":
        parent_name = (
            "R1"
            if final["recipe_id"] == r1["recipe_id"]
            and final["parent_weight_sha256"] == R1_SHA256
            else "P"
        )
        selected_parent = r1 if parent_name == "R1" else parent
        parent_method = register("FINAL-parent-D0", selected_parent)
        pb.append(parent_method["method_id"])
        if final["refiner"]["input_mode"] == "OLD_NEW":
            control = repairs[parent_name]["cal_selected"]["NEW"]
            new = register(
                "FINAL-NEW-control", control, parent_candidate=selected_parent
            )
            if new["refiner"]["input_mode"] != "NEW_ONLY":
                raise ValueError("PAIR confirmation requires its matched NEW control")
            pb.append(new["method_id"])
    else:
        aliases["FINAL-parent-D0"] = (
            aliases["R1-D0"] if final["kind"] == "ASSOCIATION" else selected_id
        )
    confirmation = {
        "PB": list(dict.fromkeys(pb)),
        "LOCAL-T2": list(
            dict.fromkeys(
                [
                    r1_native["method_id"],
                    p_native["method_id"],
                    *([c0_native["method_id"]] if c0_native else []),
                ]
            )
        ),
        "ADDITIONAL": list(
            dict.fromkeys(
                [r1_native["method_id"], aliases["FINAL-parent-D0"], selected_id]
            )
        ),
    }
    assets = read_json(external_root / "assets.local.json")
    payload = {
        "schema_version": "perception-gain-final-lock-v2",
        "status": "LOCKED",
        "locked_utc": utc_now(),
        "final_method_id": selected_id,
        "final_method": final,
        "aliases": aliases,
        "methods": {value["method_id"]: value for value in descriptors.values()},
        "selection": decision,
        "omitted_candidates": omitted,
        "perception_selection": perception,
        "repair_selections": repairs,
        "association_selection": association,
        "confirmation_methods": confirmation,
        "confirmation_populations": {
            "PB": {"references": 6, "logical_units": 129, "prefixes": 516},
            "LOCAL-T2": {
                "references": 41,
                "logical_units_per_seed": 154,
                "eval_seeds": local_evaluation_seeds(),
            },
            "ADDITIONAL": {
                "2": {"references": 40, "logical_units": 111},
                "3": {"references": 23, "logical_units": 77},
                "4": {"references": 8, "logical_units": 32},
            },
        },
        "data_roles_sha256": file_hash(artifacts / "DATA_ROLES.json"),
        "input_manifest_sha256": file_hash(artifacts / "data/STAGING_MANIFEST.json"),
        "run_config_sha256": file_hash(artifacts / "RUN_CONFIG.json"),
        "selection_thresholds": config["selection"],
        "official_temporal_iou_thresholds": official_temporal_iou_thresholds(
            dataset_spec=assets["metric_dataset_spec"]
        ),
        "code": executed_identity(
            PROJECT_ROOT,
            [Path(__file__), PROJECT_ROOT / "scripts/perception_gain_v2_selection.py"],
        ),
    }
    forecast, allocation = confirmation_budget(
        config, artifacts=artifacts, lock=payload
    )
    payload["confirmation_forecast"] = forecast
    payload["confirmation_allocation"] = allocation
    payload["replication_commitment"] = commit_replication_scope(
        payload, artifacts=artifacts, external_root=external_root
    )
    write_json(artifacts / "budget/CONFIRMATION_FORECAST.json", forecast)
    append_event(
        artifacts / "budget/ALLOCATION_EVENTS.jsonl",
        {
            "utc": utc_now(),
            "event": "FINAL_LOCK_BEFORE_ANY_PB_METRIC",
            "forecast": forecast,
            "allocation": allocation,
            "replication_commitment": payload["replication_commitment"],
        },
    )
    payload["content_sha256"] = content_hash(payload)
    write_immutable_lock(destination, payload)
    return {
        "status": "COMPLETE",
        "lock_sha256": file_hash(destination),
        "selected_method_id": selected_id,
        "confirmation_reserve_gpu_hours": max(
            config["budget"]["confirmation_reserve"],
            forecast["required_confirmation_gpu_hours"],
        ),
    }
