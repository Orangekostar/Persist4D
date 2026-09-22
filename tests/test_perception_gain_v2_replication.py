from copy import deepcopy

from scripts.perception_gain_v2_replication import (
    replication_plan,
    component_result,
    pipeline_result,
    commit_replication_scope,
)


def _parent(name, update):
    return {
        "method_id": name,
        "kind": "D0",
        "recipe_id": f"{name}-L-s45",
        "architecture_variant": name,
        "parent_optimizer_update": update,
        "recipe": {"learning_rate_label": "L"},
        "parent_weight_sha256": name,
        "refiner": None,
        "association_config": None,
    }


def _lock(parent, final=None, parent_name="R1"):
    final = final or parent
    methods = {"P": parent, "FINAL": final, "FINAL-PARENT": parent}
    return {
        "methods": methods,
        "final_method": final,
        "aliases": {"P-D0": "P", "FINAL-parent-D0": "FINAL-PARENT"},
        "repair_selections": {
            parent_name: {
                "cal_selected": {
                    "NEW": {"optimizer_update": 1000},
                    "PAIR": {"optimizer_update": 500},
                }
            }
        },
    }


def test_new_perception_replicates_the_locked_step_with_same_lr_c0():
    plan = replication_plan(_lock(_parent("S-BAL", 750)))
    assert plan["training_endpoints"] == {"S-BAL-L-s46": 750, "C0-L-s46": 750}
    assert plan["perception"][0]["optimizer_update"] == 750
    assert plan["perception"][0]["control_recipe_id"] == "C0-L-s46"


def test_refiner_replicates_both_fixed_points_and_both_same_step_controls():
    parent = _parent("C0", 0)
    final = deepcopy(parent)
    final.update(
        kind="REFINER", refiner={"input_mode": "OLD_NEW", "optimizer_update": 500}
    )
    plan = replication_plan(_lock(parent, final))
    assert plan["training_endpoints"] == {}
    assert plan["repair"]["stop_after_updates"] == 1000
    assert plan["repair"]["evaluation_updates"] == [500, 1000]
    assert plan["repair"]["fresh_parent_cache"] is False
    assert plan["required_deployment_components"] == ["repair:R1"]


def test_full_combination_requires_the_replicated_parent_cache():
    parent = _parent("S-BAL", 1500)
    final = deepcopy(parent)
    final.update(
        kind="REFINER", refiner={"input_mode": "OLD_NEW", "optimizer_update": 500}
    )
    plan = replication_plan(_lock(parent, final, parent_name="P"))
    assert plan["repair"]["fresh_parent_cache"] is True
    assert plan["repair"]["parent_recipe_id"] == "S-BAL-L-s46"
    assert len(plan["required_deployment_components"]) == 2


def test_pipeline_support_requires_every_deployed_component_and_positive_gain():
    baseline = {
        "coverage_status": "COMPLETE",
        "metrics": {str(t): 0.3 for t in (2, 3, 4, 5)},
    }
    tied = component_result(baseline, {"parent": baseline})
    assert tied["replication"] == "MIXED"
    supported = component_result(
        {
            "coverage_status": "COMPLETE",
            "metrics": {str(t): 0.31 for t in (2, 3, 4, 5)},
        },
        {"parent": baseline},
    )
    assert (
        pipeline_result(
            ["perception", "repair"], {"perception": supported, "repair": tied}
        )
        == "MIXED"
    )
    assert (
        pipeline_result(["perception", "repair"], {"perception": supported})
        == "NOT_COMPLETED"
    )
    assert pipeline_result([], {}) == "NOT_APPLICABLE"


def test_undeployed_perception_does_not_stand_in_for_pipeline_support():
    parent = _parent("C0", 0)
    final = deepcopy(parent)
    final.update(kind="ASSOCIATION", association_config={"tau": 0.6})
    lock = _lock(parent, final)
    lock["methods"]["P"] = _parent("S-BAL", 750)
    plan = replication_plan(lock)
    assert len(plan["perception"]) == 1
    assert plan["required_deployment_components"] == []
    assert plan["association_replication"] == "NOT_APPLICABLE"


def test_nonpositive_mean_and_large_single_horizon_regression_are_not_supported():
    baseline = {
        "coverage_status": "COMPLETE",
        "metrics": {str(t): 0.3 for t in (2, 3, 4, 5)},
    }
    candidate = {
        "coverage_status": "COMPLETE",
        "metrics": {"2": 0.2979, "3": 0.4, "4": 0.4, "5": 0.4},
    }
    assert component_result(candidate, {"parent": baseline})["replication"] == "MIXED"
    candidate["metrics"].pop("5")
    assert (
        component_result(candidate, {"parent": baseline})["replication"]
        == "NOT_COMPLETED"
    )


def test_budget_scope_is_committed_before_pb_and_counts_both_reserved_gpus(
    tmp_path, monkeypatch
):
    import json
    from scripts import perception_gain_v2_replication as runner

    path = tmp_path / "foundation/SEL/D0.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"elapsed_seconds": 3600}))
    monkeypatch.setattr(runner, "measured_update_cost", lambda root: 0.01)
    lock = _lock(_parent("S-BAL", 750))
    lock["confirmation_allocation"] = {"replication_and_recovery": 20}
    result = commit_replication_scope(lock, artifacts=tmp_path, external_root=tmp_path)
    item = next(iter(result["forecast"].values()))
    assert item["predicted_reserved_gpu_hours"] == (15 + 4) * 1.25
    assert result["approved_components"] == []
    assert result["uncommitted_recovery_gpu_hours"] == 20
    assert result["decision_timing"] == "FINAL_LOCK_BEFORE_ANY_PB_METRIC"
