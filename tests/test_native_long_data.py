import random

import numpy as np
import torch


def test_holdouts_are_stratified_disjoint_and_preserve_train_t5():
    from datasets.native_long_dataset import assign_development_roles

    records = [{"reference_id": f"r{h}_{i}", "Tmax": h} for h in (2, 3, 4, 5)
               for i in range(12)]
    selected = assign_development_roles(records)
    roles = {role: [r for r in selected if r["role"] == role] for role in ("TRAIN", "CAL", "SEL")}
    assert len(roles["CAL"]) == len(roles["SEL"]) == 8
    assert all(sum(r["Tmax"] >= 5 for r in roles[role]) >= 2 for role in roles)
    assert selected == assign_development_roles(list(reversed(records)))


def test_draw_plan_pairs_domains_orders_and_long_inputs():
    from datasets.native_long_dataset import NativeDrawPlan

    refs = [{"reference_id": f"r{i}", "role": "TRAIN", "Tmax": 5,
             "scan_ids": [f"s{i}_{t}" for t in range(5)],
             "scan_indices": list(range(i * 5, (i + 1) * 5))} for i in range(3)]
    plan = NativeDrawPlan(refs, scannet_count=4, seed=45)
    for draw in (0, 100, 32 * 2969, 32 * 2970, 32 * 11880, 32 * 29699):
        short = plan.draw(draw, arm="E0")
        long = plan.draw(draw, arm="E1")
        assert short["domain"] == long["domain"]
        assert short["reference_id"] == long["reference_id"]
        assert short["augmentation_seed"] == long["augmentation_seed"]
        assert short["scan_ids"] == long["scan_ids"][:len(short["scan_ids"])]
        assert long == plan.draw(draw, arm="E2") == plan.draw(draw, arm="E3")
        assert len(set(long["scan_ids"])) == long["horizon"]
    start = 32 * 990
    assert [plan.draw(i, arm="E1") for i in range(start, start + 32)] == [
        NativeDrawPlan(refs, scannet_count=4, seed=45).draw(i, arm="E1")
        for i in range(start, start + 32)]


def test_per_input_rng_isolation_and_curriculum_boundaries():
    from datasets.native_long_dataset import horizon_weights, isolated_rng

    assert horizon_weights(0) == {2: 1., 3: 0., 4: 0., 5: 0.}
    assert horizon_weights(7425) == {2: .25, 3: .25, 4: .25, 5: .25}
    before = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    with isolated_rng(9001):
        values = (random.random(), np.random.rand(), torch.rand(5))
    with isolated_rng(9001):
        assert values[0] == random.random()
        assert values[1] == np.random.rand()
        assert torch.equal(values[2], torch.rand(5))
    assert before[0] == random.getstate()
    assert np.array_equal(before[1][1], np.random.get_state()[1])
    assert torch.equal(before[2], torch.get_rng_state())
