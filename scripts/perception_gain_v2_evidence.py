"""V2 live baselines and paired repair on the existing producer/metric paths."""

from __future__ import annotations

from pathlib import Path

from scripts.perception_gain_v2 import PROJECT_ROOT, file_hash, read_json, write_json
from scripts.perception_gain_v2_config import R1_SHA256


def validate_refiner_cache(
    manifest: dict,
    *,
    recipe: dict,
    parent_weight: str,
    artifacts: Path,
    cache_root: Path,
    required_inventory: dict | None,
    external_root: Path | None = None,
) -> dict:
    from scripts.perception_gain_v2_config import (
        content_hash,
        live_execution_provenance,
    )

    binding = manifest.get("cache_binding", {})
    expected = {
        "parent_weight_hash": parent_weight,
        "inference_recipe_hash": recipe["inference_recipe_hash"],
        "input_manifest_hash": file_hash(artifacts / "data/STAGING_MANIFEST.json"),
        "roles_sha256": file_hash(artifacts / "DATA_ROLES.json"),
        "inventory_sha256": content_hash(manifest.get("inventory")),
        "point_order_transform": "canonical_vertices/identity_geometry",
        "role": "TRAIN",
        "eval_seed": 45,
        "publisher": "D0/lag1/mean",
    }
    if external_root is not None:
        from omegaconf import OmegaConf

        from scripts.train_perception_gain import compose_variant_config

        assets = read_json(external_root / "assets.local.json")
        resolved = compose_variant_config(
            recipe["architecture_variant"],
            pretrained=Path(assets["concerto_pretrained"]),
            run_dir=cache_root.parent / "feature_producer",
            recipe_config=recipe,
        )
        resolved.model.return_query_features = True
        if recipe["architecture_variant"] == "Q-SEM":
            resolved.perception_training.scorer_checkpoint = assets["scorer_checkpoint"]
        expected["resolved_config_sha256"] = content_hash(
            OmegaConf.to_container(resolved, resolve=True)
        )
        if manifest["p6a_config_sha256"] != file_hash(
            PROJECT_ROOT / "conf/p6a/default.yaml"
        ):
            raise ValueError("Refiner producer postprocessing configuration changed")
    if (
        manifest.get("status") != "PASS"
        or manifest.get("recipe") != recipe
        or manifest.get("checkpoint_sha256") != parent_weight
        or any(binding.get(key) != value for key, value in expected.items())
        or (
            required_inventory is not None
            and manifest.get("inventory") != required_inventory
        )
    ):
        raise ValueError(
            "Existing refiner cache input/parent/role/inventory binding differs"
        )
    previous = manifest["execution_provenance"]
    if content_hash(previous["source_files"]) != binding.get(
        "relevant_source_digest"
    ) or previous["executed_code_commit"] != binding.get("executed_code_commit"):
        raise ValueError("Refiner cache source provenance is inconsistent")
    current = live_execution_provenance(recipe)
    review_path = artifacts / "refiner/CACHE_COMPATIBILITY.json"
    review = read_json(review_path) if review_path.exists() else {"entries": []}
    accepted = []
    if set(previous["source_files"]) != set(current["source_files"]):
        raise ValueError("Refiner cache source dependency inventory changed")
    for name, digest in previous["source_files"].items():
        now = current["source_files"][name]
        if digest == now:
            continue
        match = next(
            (
                row
                for row in review["entries"]
                if row["source"] == name
                and row["producer_sha256"] == digest
                and row["consumer_sha256"] == now
                and row["scope"] == "TRAIN_CACHE_ONLY"
            ),
            None,
        )
        if match is None:
            raise ValueError(
                f"Refiner cache dependency changed without exact compatibility review: {name}"
            )
        accepted.append(match)
    for shard in manifest["shards"]:
        path = cache_root / shard["path"]
        if (
            path.parent.resolve() != cache_root.resolve()
            or file_hash(path) != shard["sha256"]
        ):
            raise ValueError("Refiner source shard differs from producer manifest")
    return {
        "status": "VERIFIED",
        "producer_commit": previous["executed_code_commit"],
        "consumer_commit": current["executed_code_commit"],
        "source_compatibility_reviews": accepted,
        "dependency_bindings": expected,
    }


def run_baselines(config: dict, *, external_root: Path) -> dict:
    import torch

    from models.perception_gain import CausalMaskRefiner
    from scripts.perception_gain_evaluation import (
        extract_pooled_d0_metrics,
        run_checkpoint_evaluation,
    )
    from scripts.perception_gain_foundation import _metric_class_mapping
    from scripts.perception_gain_native_evaluation import (
        run_native_checkpoint_evaluation,
    )
    from scripts.perception_refiner_evaluation import run_refiner_evaluation
    from scripts.replay_crosswindow_association import E0ReplayAccumulator
    from scripts.train_perception_refiner import (
        _atomic_torch_save,
        _frozen_payload,
        refiner_v2_binding,
    )

    artifacts = PROJECT_ROOT / config["artifact_root"]
    recipe = read_json(artifacts / "training/C0-L-s45/recipe.json")
    common = {
        "assets_path": external_root / "assets.local.json",
        "roles_path": artifacts / "DATA_ROLES.json",
        "external_root": external_root,
        "recipe_config": recipe,
        "artifact_root": artifacts,
    }
    smoke_heads = []
    for mode in ("NEW_ONLY", "OLD_NEW"):
        binding = refiner_v2_binding(
            input_mode=mode,
            parent_recipe_hash=recipe["inference_recipe_hash"],
            parent_weight_hash=R1_SHA256,
            cache_paths=(),
        )
        torch.manual_seed(45)
        zero = CausalMaskRefiner(input_mode=mode)
        path = external_root / f"foundation/zero-{mode}.ckpt"
        _atomic_torch_save(
            path, _frozen_payload(zero, cache_paths=(), updates=0, binding=binding)
        )
        smoke_heads.append(
            {
                "method_id": f"ZERO-{mode}",
                "checkpoint": path,
                "optimizer_update": 0,
                "binding": binding,
            }
        )
    first = run_refiner_evaluation(
        **common,
        parent_variant="C0",
        parent_update=0,
        parent_checkpoint=None,
        scorer_checkpoint=None,
        refiner_update=0,
        refiner_checkpoint=None,
        refiner_checkpoints=smoke_heads,
        role="CAL",
        maximum_units=1,
        export_replay_root=external_root / "cache/smoke-r1",
        output_path=artifacts / "foundation/smoke/zero-pair.json",
    )
    repeated = run_refiner_evaluation(
        **common,
        parent_variant="C0",
        parent_update=0,
        parent_checkpoint=None,
        scorer_checkpoint=None,
        refiner_update=0,
        refiner_checkpoint=None,
        refiner_checkpoints=(),
        role="CAL",
        maximum_units=1,
        output_path=artifacts / "foundation/smoke/repeated-parent.json",
    )
    assets = read_json(external_root / "assets.local.json")
    original_d0 = E0ReplayAccumulator(
        dataset_spec=assets["metric_dataset_spec"],
        class_mapping=_metric_class_mapping(Path(assets["metric_dataset_spec"])),
        checkpoint_sha256=R1_SHA256,
        source_commit=first["execution_provenance"]["executed_code_commit"],
        index_trigger_count=0,
    )
    replay_payload = torch.load(
        external_root / "cache/smoke-r1/unit-0001.pt",
        map_location="cpu",
        weights_only=False,
    )
    original_d0.update(
        logical_unit_id=replay_payload["logical_unit_id"],
        base=replay_payload["base"],
        supplement=replay_payload["supplement"],
    )
    original_metrics = extract_pooled_d0_metrics(
        original_d0.finalize()["metric_rows"],
        expected_reference_count=1,
        expected_logical_unit_count=1,
    )
    smoke_pass = (
        first["status"] == repeated["status"] == "SMOKE_PASS"
        and first["method_output_sha256"]["PARENT"]
        == repeated["method_output_sha256"]["PARENT"]
        and all(
            value == first["method_output_sha256"]["PARENT"]
            for value in first["method_output_sha256"].values()
        )
        and first["parent_metrics"] == repeated["parent_metrics"]
        and first["d0_replay_bridge_prefixes"]
        == repeated["d0_replay_bridge_prefixes"]
        == 5
        and all(
            abs(first["parent_metrics"][t] - original_metrics[t]) <= 1e-12
            for t in (2, 3, 4, 5)
        )
    )
    smoke = {
        "status": "SMOKE_PASS" if smoke_pass else "SMOKE_FAILED",
        "first": first["method_output_sha256"],
        "repeated": repeated["method_output_sha256"],
        "network_forward_count": first["network_forward_count"]
        + repeated["network_forward_count"],
        "original_d0_metrics": original_metrics,
    }
    write_json(artifacts / "foundation/SMOKE.json", smoke)
    if not smoke_pass:
        return {
            "status": "BLOCKED",
            "reason": "R1 repeated/zero/D0 bridge live smoke differs",
            "smoke": smoke,
        }
    baselines = {}
    for role in ("CAL", "SEL"):
        for method in ("D0", "FH-native"):
            try:
                path = artifacts / f"foundation/{role}/{method}.json"
                if method == "D0":
                    result = run_checkpoint_evaluation(
                        **common,
                        variant="C0",
                        optimizer_update=0,
                        checkpoint=None,
                        scorer_checkpoint=None,
                        role=role,
                        output_path=path,
                        export_replay_root=external_root / f"cache/r1-live/{role}",
                    )
                else:
                    result = run_native_checkpoint_evaluation(
                        **common,
                        variant="C0",
                        optimizer_update=0,
                        checkpoint=None,
                        scorer_checkpoint=None,
                        role=role,
                        output_path=path,
                    )
                baselines[f"{role}/{method}"] = {
                    "status": result["status"],
                    "artifact": str(path.relative_to(PROJECT_ROOT)),
                    "sha256": file_hash(path),
                }
            except (OSError, RuntimeError, ValueError, KeyError) as error:
                baselines[f"{role}/{method}"] = {
                    "status": "BLOCKED",
                    "reason": str(error),
                }
    complete = all(value["status"] == "PASS" for value in baselines.values())
    result = {
        "status": "COMPLETE" if complete else "BLOCKED",
        "smoke": smoke,
        "baselines": baselines,
        "baseline_status": "BASELINES_COMPLETE" if complete else "BASELINES_PARTIAL",
    }
    write_json(artifacts / "foundation/BASELINE_STATUS.json", result)
    return result


def run_repair(
    config: dict,
    *,
    external_root: Path,
    parent_id: str,
    recipe: dict,
    parent_update: int,
    parent_checkpoint: Path | None,
    required_inventory: dict | None = None,
) -> dict:
    """Generate one real cache, train matched heads, then evaluate CAL once."""
    from scripts.perception_gain_v2_lock import effective_parent_weight_identity
    from scripts.perception_refiner_evaluation import run_refiner_evaluation
    from scripts.prepare_perception_refiner import run as prepare
    from scripts.train_perception_refiner import refiner_v2_binding, train_mask_refiner

    artifacts = PROJECT_ROOT / config["artifact_root"]
    public = artifacts / f"refiner/{parent_id}"
    cache_root = external_root / f"cache/refiner/{parent_id}"
    manifest_path = public / "DATA_MANIFEST.json"
    assets = read_json(external_root / "assets.local.json")
    parent_weight = effective_parent_weight_identity(
        R1_SHA256 if parent_checkpoint is None else file_hash(parent_checkpoint),
        architecture=recipe["architecture_variant"],
        update=parent_update,
        scorer_sha256=(
            file_hash(Path(assets["scorer_checkpoint"]))
            if recipe["architecture_variant"] == "Q-SEM"
            else None
        ),
    )
    manifest = read_json(manifest_path) if manifest_path.exists() else None
    if manifest is not None:
        audit = validate_refiner_cache(
            manifest,
            recipe=recipe,
            parent_weight=parent_weight,
            artifacts=artifacts,
            cache_root=cache_root,
            required_inventory=required_inventory,
            external_root=external_root,
        )
        write_json(public / "CACHE_REUSE_AUDIT.json", audit)
    else:
        manifest = prepare(
            variant=recipe["architecture_variant"],
            optimizer_update=parent_update,
            checkpoint=parent_checkpoint,
            scorer_checkpoint=(
                Path(assets["scorer_checkpoint"])
                if recipe["architecture_variant"] == "Q-SEM"
                else None
            ),
            assets_path=external_root / "assets.local.json",
            roles_path=artifacts / "DATA_ROLES.json",
            external_root=external_root,
            device_name="cuda:0",
            output_root=cache_root,
            manifest_output=manifest_path,
            recipe_config=recipe,
            required_inventory=required_inventory,
        )
        if manifest["status"] != "PASS":
            return {
                "status": "BLOCKED",
                "reason": "Refiner TRAIN cache has incomplete original population",
                "manifest": str(manifest_path.relative_to(PROJECT_ROOT)),
            }
    shards = tuple(cache_root / row["path"] for row in manifest["shards"])
    summaries, heads = {}, []
    for label, mode in (("NEW", "NEW_ONLY"), ("PAIR", "OLD_NEW")):
        binding = refiner_v2_binding(
            input_mode=mode,
            parent_recipe_hash=recipe["inference_recipe_hash"],
            parent_weight_hash=parent_weight,
            cache_paths=shards,
        )
        output = external_root / f"training/refiner/{parent_id}/{label}-s45"
        summary_path = output / "run_summary.json"
        existing = read_json(summary_path) if summary_path.exists() else None
        if (
            existing is not None
            and existing.get("binding") == binding
            and existing.get("completed_updates") == 1500
        ):
            summary = existing
        else:
            resume = output / "last.ckpt"
            try:
                summary = train_mask_refiner(
                    cache_paths=shards,
                    output_dir=output,
                    binding=binding,
                    resume=resume if resume.exists() else None,
                    device="cuda:0",
                )
            except (OSError, RuntimeError, ValueError) as error:
                summary = {
                    "status": (
                        "EXCLUDED_NUMERICAL"
                        if "non-finite" in str(error)
                        else "BLOCKED"
                    ),
                    "reason": str(error),
                    "completed_updates": None,
                    "binding": binding,
                }
        summaries[label] = summary
        write_json(public / f"{label}/TRAINING.json", summary)
        if summary.get("completed_updates") != 1500:
            continue
        for update in (0, 500, 1000, 1500):
            heads.append(
                {
                    "method_id": f"R-{label}-{parent_id}@{update:04d}",
                    "checkpoint": output / f"update={update:04d}.ckpt",
                    "optimizer_update": update,
                    "binding": binding,
                }
            )
    full_pair = all(
        summary.get("completed_updates") == 1500 for summary in summaries.values()
    )
    if (
        full_pair
        and summaries["NEW"]["initial_state_sha256"]
        != summaries["PAIR"]["initial_state_sha256"]
    ):
        raise ValueError("Paired repair initialization differs")
    if not heads:
        return {
            "status": "BLOCKED",
            "parent_id": parent_id,
            "training": summaries,
            "reason": "Neither refiner completed the fixed trajectory",
        }
    evaluation = run_refiner_evaluation(
        parent_variant=recipe["architecture_variant"],
        parent_update=parent_update,
        parent_checkpoint=parent_checkpoint,
        scorer_checkpoint=(
            Path(assets["scorer_checkpoint"])
            if recipe["architecture_variant"] == "Q-SEM"
            else None
        ),
        refiner_update=0,
        refiner_checkpoint=None,
        refiner_checkpoints=heads,
        role="CAL",
        assets_path=external_root / "assets.local.json",
        roles_path=artifacts / "DATA_ROLES.json",
        external_root=external_root,
        recipe_config=recipe,
        artifact_root=artifacts,
        output_path=public / "CAL.json",
    )
    if evaluation["status"] != "PASS":
        return {
            "status": "BLOCKED",
            "parent_id": parent_id,
            "reason": "Refiner CAL coverage incomplete",
        }
    from scripts.perception_gain_v2_selection import compare, rank, select_repair

    selected, selected_heads = {}, []
    for label in ("NEW", "PAIR"):
        options = [
            row
            for name, row in evaluation["candidates"].items()
            if name.startswith(f"R-{label}-{parent_id}@")
        ]
        ranking = rank(options, evaluation["parent_candidate"])
        selected[label] = (
            ranking[0]
            if ranking
            else {
                "method_id": f"R-{label}-{parent_id}",
                "coverage_status": "INCOMPLETE",
                "metrics": {},
                "optimizer_update": None,
            }
        )
        if not ranking:
            continue
        selected_heads.append(
            next(
                head
                for head in heads
                if head["method_id"] == selected[label]["method_id"]
            )
        )
    write_json(public / "CAL_SELECTION.json", selected)
    same_step = {}
    for update in (0, 500, 1000, 1500):
        same_step[str(update)] = compare(
            evaluation["candidates"].get(
                f"R-PAIR-{parent_id}@{update:04d}", selected["PAIR"]
            ),
            evaluation["candidates"].get(
                f"R-NEW-{parent_id}@{update:04d}", selected["NEW"]
            ),
        )
    write_json(public / "SAME_STEP_CAL.json", same_step)
    sel = run_refiner_evaluation(
        parent_variant=recipe["architecture_variant"],
        parent_update=parent_update,
        parent_checkpoint=parent_checkpoint,
        scorer_checkpoint=(
            Path(assets["scorer_checkpoint"])
            if recipe["architecture_variant"] == "Q-SEM"
            else None
        ),
        refiner_update=0,
        refiner_checkpoint=None,
        refiner_checkpoints=selected_heads,
        role="SEL",
        assets_path=external_root / "assets.local.json",
        roles_path=artifacts / "DATA_ROLES.json",
        external_root=external_root,
        recipe_config=recipe,
        artifact_root=artifacts,
        output_path=public / "SEL.json",
    )
    choice = select_repair(
        parent=sel["parent_candidate"],
        new=sel["candidates"].get(selected["NEW"]["method_id"], selected["NEW"]),
        pair=sel["candidates"].get(selected["PAIR"]["method_id"], selected["PAIR"]),
    )
    adopted = None
    if choice["selected_mode"] != "KEEP_PARENT":
        adopted = {
            **sel["candidates"][selected[choice["selected_mode"]]["method_id"]],
            "component_gate_passed": True,
            "parent_id": parent_id,
            "parent_update": parent_update,
            "tie_update": parent_update
            + selected[choice["selected_mode"]]["optimizer_update"],
        }
        adopted["new_parameter_count"] += sel["parent_candidate"]["new_parameter_count"]
    result = {
        "status": "COMPLETE" if full_pair and sel["status"] == "PASS" else "BLOCKED",
        "parent_id": parent_id,
        "cal_selected": selected,
        "selection": choice,
        "adopted_candidate": adopted,
        "parent_candidate": sel["parent_candidate"],
        "training_completed_updates": {
            key: value["completed_updates"] for key, value in summaries.items()
        },
        "data_manifest_sha256": file_hash(manifest_path),
    }
    write_json(public / "SELECTION.json", result)
    return result


def run_optional_parent_repair(config: dict, *, external_root: Path) -> dict:
    from scripts.perception_gain_v2_perception import (
        budget_decision,
        resolve_checkpoint,
    )

    artifacts = PROJECT_ROOT / config["artifact_root"]
    perception_path = artifacts / "selection/PERCEPTION.json"
    r1_path = artifacts / "refiner/R1/SELECTION.json"
    if not perception_path.exists() or not r1_path.exists():
        result = {
            "status": "BLOCKED",
            "reason": "Perception or paired R1 selection is unavailable",
        }
    else:
        perception, r1 = read_json(perception_path), read_json(r1_path)
        parent = perception["selected"]
        if parent["method_id"] == "R1-D0":
            result = {
                "status": "NOT_APPLICABLE",
                "reason": "P is R1; reuse its existing paired repair",
            }
        elif r1.get("status") != "COMPLETE" or not r1["selection"]["base_positive"]:
            result = {
                "status": "NOT_APPLICABLE",
                "reason": "Complete R1 paired control did not establish a positive repair base gate",
            }
        else:
            import json

            prior = [
                json.loads(line)
                for line in (artifacts / "budget/LEDGER.jsonl").read_text().splitlines()
                if line.strip()
            ]
            # Include cache generation, head training, CAL/SEL and failed reservation cost.
            measured = sum(
                row["gpu_hours"]
                for row in {
                    row["event_id"]: row
                    for row in prior
                    if row.get("scope") == "V2" and row.get("task") == "REPAIR_R1"
                }.values()
            )
            decision = budget_decision(
                config,
                external_root=external_root,
                label="P paired repair",
                predicted_gpu_hours=measured,
                category="refinement",
            )
            if not decision["affordable"]:
                result = {"status": "SKIPPED_BUDGET", "budget": decision}
            else:
                recipe = read_json(
                    artifacts / f"training/{parent['recipe_id']}/recipe.json"
                )
                result = run_repair(
                    config,
                    external_root=external_root,
                    parent_id="P",
                    recipe=recipe,
                    parent_update=parent["optimizer_update"],
                    parent_checkpoint=resolve_checkpoint(
                        parent["recipe_id"],
                        parent["optimizer_update"],
                        external_root=external_root,
                    ),
                    required_inventory=read_json(
                        artifacts / "refiner/R1/DATA_MANIFEST.json"
                    )["inventory"],
                )
    write_json(artifacts / "refiner/P/SELECTION.json", result)
    return result
