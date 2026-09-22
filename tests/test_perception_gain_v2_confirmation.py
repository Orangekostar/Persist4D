from copy import deepcopy
import hashlib
import json

from scripts.perception_gain_v2_lock import effective_parent_weight_identity

from scripts.perception_gain_v2_confirmation import (
    allocate_confirmation_balance,
    confirmation_groups,
    deployment_decision,
    normalize_evidence,
)


def test_confirmation_reserve_transfer_cannot_exceed_actual_balance():
    allocation = allocate_confirmation_balance(remaining=60, required=50, default=40)
    assert allocation["confirmation"] == 50
    assert allocation["replication_and_recovery"] == 10
    assert allocation["transfer_into_confirmation"] == 10
    partial = allocate_confirmation_balance(remaining=30, required=50, default=40)
    assert partial["confirmation"] == 30
    assert partial["unfunded_confirmation"] == 20
    assert partial["replication_and_recovery"] == 0


def _result(value):
    return {
        "status": "PASS",
        "population_by_horizon": {
            str(t): {"logical_unit_count": 129, "reference_count": 6}
            for t in (2, 3, 4, 5)
        },
        "metric_rows": [
            {
                "method": "PARENT",
                "T": t,
                "reference": "all",
                "t_mAP": value,
                "logical_unit_count": 129,
                "reference_count": 6,
            }
            for t in (2, 3, 4, 5)
        ],
        "incomplete_units": [],
    }


def test_zero_step_q_identity_includes_separate_frozen_scorer():
    raw, scorer = "a" * 64, "b" * 64
    assert (
        effective_parent_weight_identity(
            raw, architecture="Q-SEM", update=0, scorer_sha256=scorer
        )
        == hashlib.sha256(f"{raw}:Q-SEM:{scorer}".encode("ascii")).hexdigest()
    )
    assert (
        effective_parent_weight_identity(
            raw, architecture="Q-SEM", update=750, scorer_sha256=scorer
        )
        == raw
    )


def test_cached_zero_step_q_uses_effective_weight_identity(tmp_path, monkeypatch):
    from scripts import perception_gain_v2_perception as runner
    from scripts.perception_gain_v2 import file_hash
    from scripts.perception_gain_v2_config import R1_SHA256

    artifacts = tmp_path / "artifacts"
    external = tmp_path / "external"
    external.mkdir()
    scorer = external / "scorer.pt"
    scorer.write_bytes(b"frozen-scorer-test-fixture")
    (external / "assets.local.json").write_text(
        json.dumps({"scorer_checkpoint": str(scorer)})
    )
    recipe = {
        "recipe_id": "Q-SEM-L-s45",
        "architecture_variant": "Q-SEM",
        "inference_recipe_hash": "q-inference",
    }
    for relative, value in (
        ("training/Q-SEM-L-s45/recipe.json", recipe),
        ("DATA_ROLES.json", {}),
        ("data/STAGING_MANIFEST.json", {}),
    ):
        path = artifacts / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    binding = {
        "weight_hash": effective_parent_weight_identity(
            R1_SHA256, architecture="Q-SEM", update=0, scorer_sha256=file_hash(scorer)
        ),
        "inference_recipe_hash": "q-inference",
        "input_manifest_hash": file_hash(artifacts / "data/STAGING_MANIFEST.json"),
        "roles_sha256": file_hash(artifacts / "DATA_ROLES.json"),
        "role": "CAL",
        "point_order_transform": "canonical_vertices/identity_geometry",
        "eval_seed": 45,
        "publisher": "D0/LAST/lag1/mean",
        "relevant_source_digest": "source-test-fixture",
    }
    saved = artifacts / "evaluation/CAL/Q-SEM-L-s45/update=0000.json"
    saved.parent.mkdir(parents=True)
    saved.write_text(
        json.dumps(
            {
                "status": "PASS",
                "recipe": recipe,
                "cache_binding": binding,
                "candidate": {"checkpoint_sha256": binding["weight_hash"]},
            }
        )
    )
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        runner,
        "live_execution_provenance",
        lambda recipe: {"relevant_source_digest": "source-test-fixture"},
    )
    result = runner.evaluate(
        {"artifact_root": "artifacts"},
        external_root=external,
        recipe_id=recipe["recipe_id"],
        update=0,
        role="CAL",
    )
    assert result["checkpoint_sha256"] == binding["weight_hash"]


def test_incomplete_native_coverage_never_claims_full_population_win():
    final = normalize_evidence(
        _result(0.4), method_id="final", source_method="PARENT", role="PB"
    )
    partial = _result(0.3)
    partial["population_by_horizon"]["5"]["logical_unit_count"] = 128
    native = normalize_evidence(
        partial, method_id="native", source_method="PARENT", role="PB"
    )
    baseline = normalize_evidence(
        _result(0.35), method_id="d0", source_method="PARENT", role="PB"
    )
    decision = deployment_decision(final, native, baseline)
    assert decision["tmap_all_T_vs_native_FH"] is None
    assert decision["tmap_all_T_vs_D0"] is True
    assert decision["default_method"] == "R1-D0"


def test_deployment_tolerance_does_not_change_strict_d0_all_t_field():
    raw = _result(0.31)
    raw["metric_rows"][0]["t_mAP"] = 0.2995
    final = normalize_evidence(
        raw, method_id="final", source_method="PARENT", role="PB"
    )
    native = normalize_evidence(
        _result(0.29), method_id="native", source_method="PARENT", role="PB"
    )
    d0 = normalize_evidence(
        _result(0.3), method_id="d0", source_method="PARENT", role="PB"
    )
    decision = deployment_decision(final, native, d0)
    assert decision["tmap_all_T_vs_native_FH"] is True
    assert decision["tmap_all_T_vs_D0"] is False
    assert decision["default_method"] == "FINAL"


def test_additional_population_is_fixed_and_has_no_t5():
    raw = _result(0.4)
    normalized = normalize_evidence(
        raw, method_id="final", source_method="PARENT", role="ADDITIONAL"
    )
    assert set(normalized["metrics"]) == {"2", "3", "4"}
    assert normalized["coverage_status"] == "INCOMPLETE"
    assert normalized["coverage_by_horizon"]["2"]["expected_logical_units"] == 111


def test_confirmation_groups_share_only_identical_live_parents():
    parent = {
        "method_id": "parent",
        "kind": "D0",
        "parent_weight_sha256": "a" * 64,
        "recipe": {"architecture_variant": "C0"},
        "architecture_variant": "C0",
        "scorer_sha256": None,
        "refiner": None,
        "association_config": None,
    }
    head = deepcopy(parent)
    head.update(
        method_id="pair",
        kind="REFINER",
        refiner={"input_mode": "OLD_NEW", "checkpoint_sha256": "b" * 64},
    )
    other = deepcopy(parent)
    other.update(method_id="other", parent_weight_sha256="c" * 64)
    lock = {
        "methods": {x["method_id"]: x for x in (parent, head, other)},
        "confirmation_methods": {"PB": ["parent", "pair", "other"]},
    }
    groups = confirmation_groups(lock, "PB")
    assert sorted(len(group["method_ids"]) for group in groups) == [1, 2]
    combined = next(group for group in groups if "pair" in group["method_ids"])
    assert combined["method_ids"] == ["parent", "pair"]
    assert combined["heads"] == ["pair"]
