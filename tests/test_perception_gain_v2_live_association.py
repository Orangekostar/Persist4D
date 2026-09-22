from copy import deepcopy

from scripts.perception_gain_v2_live_association import merge_association_results


def test_live_association_requires_identical_full_parent_bridge():
    row = {
        "reference": "all",
        "T": 4,
        "method": "PARENT",
        "logical_unit_count": 32,
        "reference_count": 8,
        "t_mAP": 0.25,
        "t_mAP50": 0.3,
        "t_mAP25": 0.5,
        "t_REC": 0.6,
    }
    parent = {"status": "PASS", "metric_rows": [row]}
    replay = {
        "status": "PASS",
        "metric_rows": [
            {**row, "method": "D0"},
            {**row, "method": "A*", "t_mAP": 0.26},
        ],
    }
    result = merge_association_results(parent, [replay], method_id="A*")
    assert result["status"] == "PASS"
    assert [item["method"] for item in result["metric_rows"]] == ["PARENT", "A*"]
    for field, wrong in (
        ("t_mAP", 0.249),
        ("logical_unit_count", 31),
        ("reference_count", 7),
    ):
        changed = deepcopy(replay)
        changed["metric_rows"][0][field] = wrong
        rejected = merge_association_results(parent, [changed], method_id="A*")
        assert rejected["status"] == "BLOCKED"
        assert all(item["method"] != "A*" for item in rejected["metric_rows"])
