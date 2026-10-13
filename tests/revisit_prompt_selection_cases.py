import math
import pytest

from scripts.select_revisit_prompt import select_cal_rows


CHECKPOINTS = [100, 200, 300, 400]
HORIZONS = [1, 2, 3, 4, 5]
REPEATS = [0, 1, 2]
EXPECTED_INPUTS = {"1": 31, "2": 19, "3": 14, "4": 14, "5": 6}
PARAMETERS = {
    "M0": 0,
    "M1": 84224,
    "M2": 99073,
    "M3": 51200,
    "M4": 84224,
    "M5": 84224,
    "M6": 101761,
}


def _run_lock(replication_variants=None):
    lock = {
        "N": 400,
        "checkpoint_steps": CHECKPOINTS,
        "expected_inputs": {"CAL": EXPECTED_INPUTS},
        "trainable_parameters": PARAMETERS,
    }
    if replication_variants is not None:
        lock["replication_variants"] = list(replication_variants)
    return lock


def _point_rows(
    variant,
    step,
    ap,
    current=None,
    *,
    adapter_seed=146,
    repeat_delta=0.0001,
):
    current = ap if current is None else current
    rows = []
    for repeat in REPEATS:
        for horizon in HORIZONS:
            rows.append(
                {
                    "variant": variant,
                    "adapter_seed": adapter_seed,
                    "step": step,
                    "role": "CAL",
                    "repeat_index": repeat,
                    "horizon": horizon,
                    "AP": ap[horizon] + repeat * repeat_delta,
                    "current_AP": current[horizon] + repeat * repeat_delta,
                    "expected_inputs": EXPECTED_INPUTS[str(horizon)],
                    "completed_inputs": EXPECTED_INPUTS[str(horizon)],
                    "status": "COMPLETE",
                    "gap": 0,
                }
            )
    return rows


def _matrix(
    *,
    baseline=None,
    trained=None,
    current=None,
    adapter_seed=146,
    repeat_delta=0.0001,
):
    baseline = baseline or {
        1: 0.70,
        2: 0.60,
        3: 0.50,
        4: 0.50,
        5: 0.50,
    }
    trained = trained or {
        variant: {
            horizon: baseline[horizon] + 0.01
            for horizon in HORIZONS
        }
        for variant in ("M1", "M2", "M3", "M4", "M5", "M6")
    }
    rows = _point_rows(
        "M0",
        0,
        baseline,
        adapter_seed=adapter_seed,
        repeat_delta=repeat_delta,
    )
    for variant, values in trained.items():
        for step in CHECKPOINTS:
            rows.extend(
                _point_rows(
                    variant,
                    step,
                    values[step] if isinstance(values, dict) and step in values else values,
                    current=(current or {}).get(variant),
                    adapter_seed=adapter_seed,
                    repeat_delta=repeat_delta,
                )
            )
    return rows


def _rows_at(rows, variant, step):
    return [row for row in rows if row["variant"] == variant and row["step"] == step]


def test_selects_earlier_step_when_metrics_are_exactly_tied():
    rows = _matrix()
    for step in (100, 200):
        for row in _rows_at(rows, "M1", step):
            row["AP"] = 0.75 if row["horizon"] <= 2 else 0.70
            row["current_AP"] = row["AP"]
    result = select_cal_rows(rows, _run_lock())
    assert result["selected_by_variant"]["M1"]["step"] == 100


def test_metric_tolerance_uses_next_metric_before_step():
    rows = _matrix()
    for row in _rows_at(rows, "M1", 100):
        row["AP"] = 0.70
        row["current_AP"] = 0.60
    for row in _rows_at(rows, "M1", 200):
        row["AP"] = 0.7000005
        row["current_AP"] = 0.70
    result = select_cal_rows(rows, _run_lock())
    assert result["selected_by_variant"]["M1"]["step"] == 200


def test_metric_difference_at_tolerance_is_still_a_tie():
    rows = _matrix(repeat_delta=0.0)
    for row in _rows_at(rows, "M1", 100):
        row["AP"] = 0.70
        row["current_AP"] = 0.60
    for row in _rows_at(rows, "M1", 200):
        row["AP"] = 0.700001
        row["current_AP"] = 0.60
    result = select_cal_rows(rows, _run_lock())
    assert result["selected_by_variant"]["M1"]["step"] == 100


def test_cross_variant_tie_uses_fewer_parameters():
    rows = _matrix()
    for variant in ("M1", "M2", "M3", "M4", "M5", "M6"):
        for row in rows:
            if row["variant"] == variant:
                row["AP"] = 0.72 if row["horizon"] <= 2 else 0.70
                row["current_AP"] = row["AP"]
    result = select_cal_rows(rows, _run_lock())
    assert result["best_overall"]["variant"] == "M3"


def test_missing_repeat_removes_variant_from_selection():
    rows = _matrix()
    rows.remove(next(row for row in rows if row["variant"] == "M5" and row["step"] == 300 and row["repeat_index"] == 2 and row["horizon"] == 4))
    result = select_cal_rows(rows, _run_lock())
    assert result["variant_status"]["M5"] == "INCOMPLETE"
    assert result["selected_by_variant"]["M5"] is None
    assert result["status"] == "EXPERIMENT_INCOMPLETE"


def test_nan_metric_makes_variant_incomplete():
    rows = _matrix()
    row = next(row for row in rows if row["variant"] == "M6" and row["step"] == 100 and row["repeat_index"] == 1 and row["horizon"] == 3)
    row["AP"] = math.nan
    result = select_cal_rows(rows, _run_lock())
    assert result["variant_status"]["M6"] == "INCOMPLETE"
    assert result["selected_by_variant"]["M6"] is None


def test_negative_candidate_has_no_guard_winner_but_keeps_descriptive_best():
    baseline = {1: 0.70, 2: 0.60, 3: 0.50, 4: 0.50, 5: 0.50}
    trained = {
        variant: {h: (0.69 if h == 1 else 0.59 if h == 2 else 0.65) for h in HORIZONS}
        for variant in ("M1", "M2", "M3", "M4", "M5", "M6")
    }
    result = select_cal_rows(_matrix(baseline=baseline, trained=trained), _run_lock())
    assert result["candidate_star"] is None
    assert result["candidate_descriptive_star"]["variant"] == "M4"
    assert result["candidate_descriptive_star"]["guard_pass"] is False


def test_incomplete_m0_does_not_fabricate_control():
    rows = _matrix()
    rows = [row for row in rows if not (row["variant"] == "M0" and row["horizon"] == 5)]
    result = select_cal_rows(rows, _run_lock())
    assert result["variant_status"]["M0"] == "INCOMPLETE"
    assert result["control_star"] is None
    assert result["best_overall"]["variant"] != "M0"


def test_step_zero_for_trained_variant_is_ignored():
    rows = _matrix()
    rows.extend(_point_rows("M1", 0, {h: 0.99 for h in HORIZONS}))
    result = select_cal_rows(rows, _run_lock())
    assert result["selected_by_variant"]["M1"]["step"] in CHECKPOINTS
    assert result["selected_by_variant"]["M1"]["step"] != 0


def test_irrelevant_zero_gap_field_does_not_affect_completeness():
    rows = _matrix()
    for row in rows:
        row["gap"] = 0
    result = select_cal_rows(rows, _run_lock())
    assert all(status == "COMPLETE" for status in result["variant_status"].values())


def test_replication_positive_gate_returns_candidate_and_control():
    baseline = {1: 0.70, 2: 0.60, 3: 0.50, 4: 0.50, 5: 0.50}
    trained = {
        variant: {h: baseline[h] + (0.01 if variant in {"M1", "M2", "M3"} else 0.10) for h in HORIZONS}
        for variant in ("M1", "M2", "M3", "M4", "M5", "M6")
    }
    result = select_cal_rows(
        _matrix(baseline=baseline, trained=trained, repeat_delta=0.0002),
        _run_lock(),
    )
    replication = result["replication"]
    assert replication["eligible"] is True
    assert replication["variants_to_train"] == ["M4", "M3"]
    assert replication["seed"] == 147
    assert replication["N"] == 400


def test_replication_negative_current_gate_is_reported():
    baseline = {1: 0.70, 2: 0.60, 3: 0.50, 4: 0.50, 5: 0.50}
    trained = {
        variant: {
            h: baseline[h] + (0.01 if variant in {"M1", "M2", "M3"} else 0.10)
            for h in HORIZONS
        }
        for variant in ("M1", "M2", "M3", "M4", "M5", "M6")
    }
    current = {
        variant: {h: baseline[h] + (0.01 if variant in {"M1", "M2", "M3"} else -0.10) for h in HORIZONS}
        for variant in ("M1", "M2", "M3", "M4", "M5", "M6")
    }
    result = select_cal_rows(
        _matrix(baseline=baseline, trained=trained, current=current),
        _run_lock(),
    )
    assert result["replication"]["eligible"] is False
    assert any("current" in reason.lower() for reason in result["replication"]["reasons"])


def test_replication_requires_strictly_more_than_twice_range():
    baseline = {1: 0.70, 2: 0.60, 3: 0.50, 4: 0.50, 5: 0.50}
    trained = {
        variant: {h: baseline[h] + (0.01 if variant in {"M1", "M2", "M3"} else 0.02) for h in HORIZONS}
        for variant in ("M1", "M2", "M3", "M4", "M5", "M6")
    }
    rows = _matrix(baseline=baseline, trained=trained, repeat_delta=0.0025)
    result = select_cal_rows(rows, _run_lock())
    assert result["replication"]["eligible"] is False
    assert "DELTA_S_LONG_NOT_GREATER_THAN_TWICE_RANGE" in result["replication"]["reasons"]


def test_run_lock_rejects_unregistered_n_and_nonquarter_checkpoints():
    with pytest.raises(ValueError, match="N"):
        select_cal_rows(_matrix(), {**_run_lock(), "N": 200})
    with pytest.raises(ValueError, match="checkpoint"):
        select_cal_rows(_matrix(), {**_run_lock(), "checkpoint_steps": [50, 100, 200, 400]})


def test_seed147_cannot_replace_locked_control_family_with_m0_after_a_negative_result():
    rows = _matrix()
    values = {"M1": {h: .1 for h in HORIZONS}, "M4": {h: .2 for h in HORIZONS}}
    rows.extend(_matrix(trained=values, adapter_seed=147))
    result = select_cal_rows(rows, _run_lock(["M4", "M1"]), seed=147)
    assert result["control_star"]["variant"] == "M1"
    assert result["candidate_star"]["variant"] == "M4"
    assert result["control_star"]["guard_pass"] is False
    assert result["family_selection_frozen"] is True
    with pytest.raises(ValueError, match="famil"):
        select_cal_rows(rows, _run_lock(), seed=147)


def test_seed147_cli_uses_main_family_lock_without_overwriting_it(tmp_path, monkeypatch):
    import csv
    import json
    from scripts.select_revisit_prompt import main

    monkeypatch.chdir(tmp_path)
    folder = tmp_path/"artifacts/revisit_prompt_v1"
    folder.mkdir(parents=True)
    run_path = folder/"RUN_LOCK.json"
    run_path.write_text(json.dumps(_run_lock()))
    main_path = folder/"CAL_SELECTION_LOCK.json"
    main_path.write_text(json.dumps(dict(status="COMPLETE",replication=dict(eligible=True,
        variants_to_train=["M4","M1"]))))
    run_before, main_before = run_path.read_bytes(), main_path.read_bytes()
    rows = _matrix()+_matrix(adapter_seed=147)
    csv_path = folder/"CAL_ALL.csv"
    with csv_path.open("w",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    assert main(["--seed","147"]) == 0
    result = json.loads((folder/"REPLICATION_LOCK.json").read_text())
    assert result["family_selection_frozen"] and result["control_star"]["variant"] == "M1"
    assert result["candidate_star"]["variant"] == "M4"
    assert run_path.read_bytes() == run_before and main_path.read_bytes() == main_before
    assert (folder/result["input_identities"]["results_snapshot"]).read_bytes() == csv_path.read_bytes()

