import copy

import pytest

from scripts.native_long_execution import select_cal_points


def point(arm, step, t1, t2, long):
    return {"arm": arm, "step": step, "seed": 45, "status": "COMPLETE",
            "role": "CAL", "completed": {f"T{h}": 5 for h in range(1, 6)},
            "expected": {f"T{h}": 5 for h in range(1, 6)},
            "metrics": {"T1": t1, "T2": t2, "T3": long, "T4": long, "T5": long, "S_long": long}}


def test_native_selection_uses_arm_specific_keys_then_earlier_tolerance():
    rows = [point("E0", 2970, .5, .4, .3), point("E0", 5940, .6, .4, .2),
            point("E1", 2970, .5, .4, .3), point("E1", 5940, .7, .6, .29),
            point("E1", 8910, .5, .4, .3000005)]
    assert select_cal_points(rows, ["E0", "E1"])["selected_updates"] == {"E0": 5940, "E1": 2970}
    broken = copy.deepcopy(rows)
    broken[0]["completed"]["T2"] = 3
    with pytest.raises(ValueError, match="complete"):
        select_cal_points(broken, ["E0", "E1"])
    missing = copy.deepcopy(rows)
    del missing[0]["expected"]["T5"]
    with pytest.raises(ValueError, match="complete"):
        select_cal_points(missing, ["E0", "E1"])
