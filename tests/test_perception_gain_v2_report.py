import pytest

from scripts.perception_gain_v2_report import (
    auxiliary_complete,
    confirmation_rows,
    ledger_totals,
)


def completed_auxiliary():
    return {
        "status": "COMPLETE",
        "learning_rate": "L",
        "probes": {
            rate: [{"coverage_status": "COMPLETE"} for _ in range(4)]
            for rate in ("H", "L")
        },
        "arms": {
            f"{variant}-L-s45": {
                "status": "COMPLETE",
                "completed_global_step": 750,
                "cal": [
                    {"optimizer_update": step, "coverage_status": "COMPLETE"}
                    for step in ((250, 750) if variant == "S-WORST" else (0, 250, 750))
                ],
            }
            for variant in ("A-OPEN", "Q-SEM", "S-WORST")
        },
    }


def test_completed_auxiliary_requires_real_training_and_cal_coverage():
    result = completed_auxiliary()
    assert auxiliary_complete(result)
    result["arms"]["A-OPEN-L-s45"]["cal"][-1]["coverage_status"] = "INCOMPLETE"
    assert not auxiliary_complete(result)


def test_auxiliary_wrapper_completion_does_not_hide_a_blocked_or_missing_arm():
    result = completed_auxiliary()
    result["arms"]["A-OPEN-L-s45"] = {"status": "BLOCKED", "reason": "Loader failed"}
    assert not auxiliary_complete(result)
    del result["arms"]["A-OPEN-L-s45"]
    assert not auxiliary_complete(result)
    assert not auxiliary_complete({})


def test_auxiliary_allows_the_prescribed_skips_but_not_a_missing_probe():
    result = completed_auxiliary()
    result["arms"]["S-WORST-L-s45"] = {"status": "SKIPPED_BUDGET"}
    result["arms"]["Q-SEM-L-s45"] = {
        "status": "SKIPPED_SEVERE_ZERO_STEP",
        "cal": [{"optimizer_update": 0, "coverage_status": "COMPLETE"}],
    }
    assert auxiliary_complete(result)
    result["probes"]["H"][0]["coverage_status"] = "INCOMPLETE"
    assert not auxiliary_complete(result)


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


@pytest.mark.parametrize(
    "result,expected",
    [
        (
            {
                "status": "BLOCKED",
                "reason": "Pilot selection unavailable",
                "selected": {"method_id": "R1-D0", "architecture_variant": "C0"},
                "full_training_recipes": [],
            },
            "KEEP_R1",
        ),
        (
            {
                "status": "COMPLETE",
                "selected": {"method_id": "C0-L-s45", "architecture_variant": "C0"},
            },
            "CONTINUATION_ONLY",
        ),
        (
            {
                "status": "COMPLETE",
                "selected": {
                    "method_id": "A-OPEN-L-s45",
                    "architecture_variant": "A-OPEN",
                },
            },
            "NEW_PERCEPTION",
        ),
    ],
)
def test_perception_label_distinguishes_retained_r1_from_trained_c0(result, expected):
    from scripts.perception_gain_v2_report import perception_selection_label

    assert perception_selection_label(result) == expected
