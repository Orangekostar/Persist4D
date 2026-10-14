"""A user-authorized pool extension retains actual cost and lifetime limits."""

import json

import pytest

from scripts.revisit_prompt_campaign import available_budget, forecast_cost_plan


def test_authorized_extension_admits_complete_training_and_charges_existing_events(tmp_path):
    budget = dict(prior_gpu_hours=1714.8140614156707, lifetime_cap_gpu_hours=1823.,
                  new_cap_gpu_hours=100., reserve_gpu_hours=8.,
                  authorized_pool_cap_gpu_hours=100.)
    pilots = {f"M{i}": dict(update_seconds=20., startup_seconds=20.) for i in range(1, 7)}
    evaluation = dict(seconds_per_stage=1.3070398115989224,
                      seconds_per_input=4.547738039261839, startup_seconds=17.99315675199614)
    manifest = dict(counts={"CAL": {"1": 31, "2": 19, "3": 14, "4": 14, "5": 6},
                           "SEL": {"1": 39, "2": 21, "3": 21, "4": 15, "5": 15},
                           "PB": {"2": 129, "3": 129, "4": 129, "5": 129}})
    cost = forecast_cost_plan(budget, pilots, evaluation, manifest)
    assert cost["N"] == 400
    assert cost["checkpoint_steps"] == [100, 200, 300, 400]
    plan = {**cost, "budget": budget}
    event = dict(event_id="previous-failed-attempt", phase="P2", gpu_hours=1.25)
    (tmp_path / "COST_LEDGER.jsonl").write_text(json.dumps(event) + "\n")
    assert available_budget(tmp_path, plan, phase="P2") == pytest.approx(21.98705937232974)
    with pytest.raises(ValueError):
        available_budget(tmp_path, {**plan, "budget": {**budget, "lifetime_cap_gpu_hours": 1800.}}, phase="P2")
    with pytest.raises(ValueError):
        forecast_cost_plan({k: v for k, v in budget.items() if k != "authorized_pool_cap_gpu_hours"},
                           pilots, evaluation, manifest)


@pytest.mark.parametrize("cap", [float("nan"), float("inf"), -1., 0.])
def test_pool_authorization_must_be_positive_finite(cap, tmp_path):
    budget = dict(prior_gpu_hours=0., lifetime_cap_gpu_hours=200., new_cap_gpu_hours=100.,
                  reserve_gpu_hours=8., authorized_pool_cap_gpu_hours=cap)
    plan = dict(budget=budget, phase_caps_gpu_hours={f"P{i}": (100. if i == 2 else 0.) for i in range(6)})
    with pytest.raises(ValueError):
        available_budget(tmp_path, plan, phase="P2")
