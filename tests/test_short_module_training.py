"""The training draw plan is independent of head losses and GT validity."""

from scripts.short_module_training import build_sample_plan


def test_sampling_plan_has_fixed_horizon_mix_and_small_candidate_replacement():
    records = [{"role": "TRAIN", "reference_id": ref, "horizon": h, "input_id": f"{ref}:{h}"}
               for ref in ("a", "b") for h in (1, 2)]
    counts = {r["input_id"]: 2 if r["reference_id"] == "a" else 8 for r in records}
    first = build_sample_plan(records, counts, seed=45, updates=20)
    assert first == build_sample_plan(list(reversed(records)), counts, seed=45, updates=20)
    assert first != build_sample_plan(records, counts, seed=46, updates=20)
    for batch in first:
        assert [draw["horizon"] for draw in batch] == [2, 2, 2, 1]
        assert sum(len(draw["candidates"]) for draw in batch) == 16
        for draw in batch:
            if draw["input_id"].startswith("b"):
                assert len(set(draw["candidates"])) == 4
            else:
                assert max(draw["candidates"]) < 2
    counts["a:1"] = 0
    no_empty = build_sample_plan(records, counts, seed=45, updates=20)
    assert all(draw["input_id"] != "a:1" for batch in no_empty for draw in batch)
