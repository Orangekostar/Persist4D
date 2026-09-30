def test_budget_modes_keep_full_schedule_and_reservation():
    from scripts.native_long_campaign import choose_budget_mode

    costs = {"E0": 10., "E1": 12., "E2": 15., "E3": 20.}
    assert choose_budget_mode(costs, remaining=65., reserve=8.)["mode"] == "FULL_FOUR"
    assert choose_budget_mode(costs, remaining=56., reserve=8.)["mode"] == "SCREEN_ONE"
    assert choose_budget_mode(costs, remaining=51., reserve=8.)["mode"] == "PRIORITY_F"
    mode = choose_budget_mode(costs, remaining=30., reserve=8.)
    assert mode["mode"] == "BASELINE_RECOVERY_ONLY"
    assert mode["full_arms"] == ["E0"]
    assert choose_budget_mode(costs, remaining=17., reserve=8.)["full_arms"] == []


def test_resume_preserves_budget_lock_bytes(tmp_path, monkeypatch):
    import json

    from scripts import native_long_campaign as campaign
    from scripts.short_module_screen import write_json

    monkeypatch.setattr(campaign, "ARTIFACTS", tmp_path / "artifacts")
    forecast = {"costs_gpu_hours": {"E0": 380., "E1": 400., "E2": 410., "E3": 420.},
                "integrated_horizon_mass": {1: 4 / 9, 2: 5 / 9}}
    monkeypatch.setattr(campaign, "forecast_costs", lambda *args: forecast)
    write_json(tmp_path / "resources/COST_MEASUREMENTS.json",
               {"per_rank_batch": 1, "world_size_planned": 2, "accumulation": 16})
    write_json(tmp_path / "resources/CHECKPOINT_MEMORY_PROBE.json", {"status": "OOM"})
    write_json(tmp_path / "resources/IO_MEASUREMENTS.json", {})
    write_json(tmp_path / "POPULATION.json", {})
    write_json(tmp_path / "RESOURCE_PLAN.json", {"prior": {"gpu_hours": 80.}})
    write_json(tmp_path / "RUN_STATE.json",
               {"arms": {arm: {"seed45_updates": 0} for arm in ("E0", "E1", "E2", "E3")}})
    (tmp_path / "COST_LEDGER.jsonl").write_text(json.dumps({"gpu_hours": .1}) + "\n")
    campaign.freeze_cost_plan(tmp_path)
    lock = tmp_path / "selection/BUDGET_LOCK.json"
    original = lock.read_bytes()
    campaign.freeze_cost_plan(tmp_path)
    assert lock.read_bytes() == original
    assert "pre_result_forecast_correction" not in json.loads(lock.read_text())
