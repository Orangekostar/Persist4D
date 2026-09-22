import pytest

from scripts.perception_gain_v2_report import confirmation_rows, ledger_totals


def test_missing_confirmation_keeps_the_original_denominators_and_unknown_scores():
    rows, references = confirmation_rows({}, {})
    pb = [row for row in rows if row["population"] == "PB"]
    assert sum(row["expected_units"] for row in pb if row["method"] == "FINAL") == 516
    assert all(row["actual_units"] == 0 and row["t_mAP"] is None for row in rows)
    assert all(
        row["expected_units"] == 129 and row["expected_references"] == 6 for row in pb
    )
    additional = [row for row in rows if row["population"] == "ADDITIONAL"]
    assert {
        (row["T"], row["expected_units"], row["expected_references"])
        for row in additional
    } == {(2, 111, 40), (3, 77, 23), (4, 32, 8)}
    local = [row for row in rows if row["population"] == "LOCAL-T2"]
    assert {row["eval_seed"] for row in local} == {45, 46, 47}
    assert all(
        row["expected_units"] == 154 and row["expected_references"] == 41
        for row in local
    )
    assert references == []


def test_local_report_uses_audited_population_and_measured_references():
    lock = {
        "confirmation_methods": {"LOCAL-T2": ["native"]},
        "confirmation_populations": {"LOCAL-T2": {"references": 46}},
    }
    report = {
        "local_t2": {
            "native": {
                "seeds": {
                    "45": {
                        "validation_sequence_count": 154,
                        "validation_reference_count": 46,
                        "status": "PASS",
                        "metrics": {"t_mAP": 0.4},
                    }
                }
            }
        }
    }
    rows, _ = confirmation_rows(lock, report)
    measured = next(
        row
        for row in rows
        if row["population"] == "LOCAL-T2" and row["eval_seed"] == 45
    )
    assert measured["actual_references"] == measured["expected_references"] == 46
    assert measured["expected_units"] == measured["actual_units"] == 154


def test_cost_report_deduplicates_settled_events_without_losing_prior_cost():
    events = [
        {"event_id": "prior", "scope": "PRIOR", "gpu_hours": 12.1},
        {"event_id": "run", "scope": "V2", "gpu_hours": 2.3},
    ]
    result = ledger_totals([*events, events[1]])
    assert result["prior_gpu_hours"] == 12.1
    assert result["v2_gpu_hours"] == 2.3
    assert result["settled_gpu_hours"] == pytest.approx(14.4)
