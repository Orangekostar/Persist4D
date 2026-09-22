from copy import deepcopy
import hashlib

from scripts.perception_gain_v2_lock import effective_parent_weight_identity

from scripts.perception_gain_v2_confirmation import (
    confirmation_groups,
    deployment_decision,
    normalize_evidence,
)


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
    assert (
        effective_parent_weight_identity(
            raw, architecture="C0", update=0, scorer_sha256=None
        )
        == raw
    )


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
