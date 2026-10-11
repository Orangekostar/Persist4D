import copy

import pytest


def test_sha256_works_in_the_pinned_python310_environment(tmp_path):
    from scripts.native_long_postselection import sha256

    path = tmp_path / "evidence.json"
    path.write_bytes(b"abc")
    assert sha256(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def comparison(arm, long_gain=.01, t1=0., step=2970):
    return {"arm": arm, "control": "E1", "step": step, "control_step": step,
            "added_parameters": 0 if arm == "E2" else 198528,
            "delta": {"T1": t1, "T2": 0., "T3": long_gain, "T4": long_gain,
                      "T5": long_gain, "S_long": long_gain}}


def test_long_gain_does_not_hide_short_horizon_regression():
    from scripts.native_long_postselection import module_shortlist

    result = module_shortlist([comparison("E2", .02, -.003), comparison("E3")])
    assert result["selected_module"] == "E3"
    rows = {r["arm"]: r for r in result["modules"]}
    assert rows["E2"]["status"] == "LONG_TRADEOFF"
    assert rows["E2"]["regressed_horizons"] == ["T1"]
    assert rows["E3"]["status"] == "PROVISIONAL_LONG_GAIN"


def test_no_positive_module_is_an_explicit_empty_shortlist():
    from scripts.native_long_postselection import module_shortlist

    assert module_shortlist([comparison("E2", -.01), comparison("E3", 0.)])[
        "selected_module"] is None


def test_module_attribution_requires_same_step_control():
    from scripts.native_long_postselection import module_shortlist

    row = comparison("E3")
    row["control_step"] = 23760
    with pytest.raises(ValueError, match="same-step"):
        module_shortlist([row])


def test_replication_requires_full_pair_and_reserve_within_remaining_budget():
    from scripts.native_long_postselection import replication_budget

    result = replication_budget({"E1": 428., "E3": 399.}, "E3", remaining=92.6, reserve=8.)
    assert result["status"] == "NOT_REPLICATED_BUDGET"
    assert result["forecast_pair_gpu_hours"] == pytest.approx(1033.75)
    assert result["updates_per_arm"] == 29700
    assert result["authorized_to_start"] is False


def reference(horizon, ref, ap, n=3):
    return {"horizon": horizon, "reference_id": ref, "AP": ap, "completed": n, "expected": n}


def test_reference_diagnostics_keep_missing_horizons_missing():
    from scripts.native_long_postselection import reference_deltas

    candidate = {"by_reference": [reference(3, "A", .4), reference(2, "B", .2)]}
    control = {"by_reference": [reference(3, "A", .3), reference(2, "B", .3)]}
    result = reference_deltas(candidate, control)
    assert len(result["horizon_rows"]) == 2
    assert result["long_reference_rows"] == [
        {"reference_id": "A", "available_horizons": [3], "mean_available_delta": pytest.approx(.1)}]
    assert result["long_reference_signs"] == {"positive": 1, "zero": 0, "negative": 0}
    assert "pooled_AP" not in result


def test_reference_pair_rejects_changed_population_or_denominator():
    from scripts.native_long_postselection import reference_deltas

    row = {"by_reference": [reference(3, "A", .3)]}
    other = copy.deepcopy(row)
    other["by_reference"][0]["expected"] = 4
    with pytest.raises(ValueError, match="reference"):
        reference_deltas(row, other)


def test_incomplete_or_nonfinite_result_cannot_enter_tables():
    from scripts.native_long_postselection import validate_result

    counts = {f"T{h}": 3 for h in range(1, 6)}
    row = {"status": "COMPLETE", "expected": counts, "completed": dict(counts),
           "metrics": {**{f"T{h}": .3 for h in range(1, 6)}, "S_long": .3}}
    validate_result(row, counts)
    row["completed"]["T5"] = 2
    with pytest.raises(ValueError, match="complete"):
        validate_result(row, counts)
    row["completed"]["T5"] = 3
    row["metrics"]["T2"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        validate_result(row, counts)
