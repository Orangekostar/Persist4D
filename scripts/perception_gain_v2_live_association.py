"""Evaluate a locked association by replaying each freshly produced unit once."""

from __future__ import annotations

from pathlib import Path

from scripts.perception_gain_v2 import PROJECT_ROOT, read_json, write_json
from scripts.perception_gain_v2_config import R1_SHA256, executed_identity


def merge_association_results(parent: dict, replays: list[dict], *, method_id: str):
    """Require the replayed D0 to reproduce every live pooled parent metric."""
    result = dict(parent)
    rows = [row for replay in replays for row in replay["metric_rows"]]
    live = {
        (row["reference"], row["T"]): row
        for row in parent["metric_rows"]
        if row["method"] == "PARENT"
    }
    bridge = {
        (row["reference"], row["T"]): row for row in rows if row["method"] == "D0"
    }
    matches = (
        bool(live)
        and set(live) == set(bridge)
        and all(
            row["logical_unit_count"] == bridge[key]["logical_unit_count"]
            and row["reference_count"] == bridge[key]["reference_count"]
            and all(
                abs(row[field] - bridge[key][field]) <= 1e-12
                for field in ("t_mAP", "t_mAP50", "t_mAP25", "t_REC")
            )
            for key, row in live.items()
        )
    )
    result["association_d0_bridge"] = {"status": "PASS" if matches else "FAIL"}
    result["association_diagnostics"] = [
        {key: value for key, value in replay.items() if key != "metric_rows"}
        for replay in replays
    ]
    if matches:
        result["metric_rows"] = [
            *parent["metric_rows"],
            *(row for row in rows if row["method"] == method_id),
        ]
    else:
        result["status"] = "BLOCKED"
        result["reason"] = "Association D0 replay differs from the live parent"
    return result


def run_live_association_evaluation(
    config: dict,
    *,
    external_root: Path,
    parent_method: dict,
    association_method: dict,
    role: str,
    output_path: Path,
    prediction_callback=None,
) -> dict:
    from models.overlap_entity_association import AssociationConfig
    from scripts.perception_gain_foundation import _metric_class_mapping
    from scripts.perception_gain_v2_confirmation import method_inputs
    from scripts.perception_refiner_evaluation import run_refiner_evaluation
    from scripts.replay_crosswindow_association import E2ReplayAccumulator

    if (
        role not in {"PB", "ADDITIONAL"}
        or parent_method["parent_checkpoint_file_sha256"] != R1_SHA256
        or association_method["parent_checkpoint_file_sha256"] != R1_SHA256
        or parent_method.get("refiner")
        or association_method.get("refiner")
    ):
        raise ValueError("Locked association must independently use the frozen R1")
    artifacts = PROJECT_ROOT / config["artifact_root"]
    assets = read_json(external_root / "assets.local.json")
    method_id = association_method["method_id"]
    association = AssociationConfig(**association_method["association_config"])
    accumulators, failed_horizons = {}, set()
    source = executed_identity(PROJECT_ROOT, [Path(__file__)])

    def export_association(**entry):
        if prediction_callback is not None and entry["method"] == method_id:
            prediction_callback(**entry)

    def consume(*, logical_unit_id, base, supplement):
        horizon = len(base["episode"]["scan_ids"])
        if horizon not in accumulators:
            accumulators[horizon] = E2ReplayAccumulator(
                dataset_spec=assets["metric_dataset_spec"],
                class_mapping=_metric_class_mapping(
                    Path(assets["metric_dataset_spec"])
                ),
                checkpoint_sha256=R1_SHA256,
                source_commit=source["executed_code_commit"],
                data_role="PROTOCOL-B" if role == "PB" else "ADDITIONAL",
                association_configs={method_id: association},
                episode_horizon=horizon,
                horizons=(2, 3, 4, 5) if role == "PB" else (horizon,),
                prediction_callback=export_association,
            )
        try:
            accumulators[horizon].update(
                logical_unit_id=logical_unit_id, base=base, supplement=supplement
            )
        except Exception:
            # A failed replay may already have updated a metric accumulator.
            failed_horizons.add(horizon)
            raise

    inputs = method_inputs(
        parent_method, external_root=external_root, artifacts=artifacts
    )
    variant, update, checkpoint = (
        inputs.pop("variant"),
        inputs.pop("optimizer_update"),
        inputs.pop("checkpoint"),
    )
    parent = run_refiner_evaluation(
        **inputs,
        parent_variant=variant,
        parent_update=update,
        parent_checkpoint=checkpoint,
        refiner_update=0,
        refiner_checkpoint=None,
        refiner_checkpoints=[],
        role=role,
        roles_path=artifacts / "DATA_ROLES.json",
        output_path=output_path,
        replay_callback=consume,
        prediction_callback=prediction_callback,
    )
    if failed_horizons:
        result = {
            **parent,
            "status": "BLOCKED",
            "reason": "Live association replay failed",
            "association_failed_horizons": sorted(failed_horizons),
        }
    else:
        result = merge_association_results(
            parent,
            [value.finalize() for value in accumulators.values()],
            method_id=method_id,
        )
    write_json(output_path, result)
    return result
