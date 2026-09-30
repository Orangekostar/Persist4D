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
