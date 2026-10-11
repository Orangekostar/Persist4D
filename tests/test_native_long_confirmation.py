import pytest


def test_historical_filter_keeps_a_sequence_with_one_empty_scan():
    import numpy as np

    from scripts.native_long_confirmation import has_supervised_instance

    background = np.zeros((2, 12))
    valid = background.copy()
    valid[0, 10:12] = [3, 7]
    invalid_instance = valid.copy()
    invalid_instance[0, 11] = -1
    assert not has_supervised_instance(background, [3])
    assert has_supervised_instance(valid, [3])
    assert not has_supervised_instance(invalid_instance, [3])
    assert any(has_supervised_instance(scan, [3]) for scan in (background, valid))


def aliases():
    return {name: {"index": i, "reference_id": "room", "physical_scan_id": f"scan{i}"}
            for i, name in enumerate(("scene0001_00", "scene0001_01", "scene0001_02"))}


def test_formal_record_rejects_any_training_reference():
    from scripts.native_long_confirmation import bind_record

    with pytest.raises(ValueError, match="TRAIN"):
        bind_record("PB", "unit", ["scene0001_00", "scene0001_01"], aliases(), {"room"})


def test_formal_record_keeps_distinct_protocol_units_and_order():
    from scripts.native_long_confirmation import bind_record

    scans = ["scene0001_00", "scene0001_01"]
    left = bind_record("PB", "master1:canonical", scans, aliases(), set(), ambiguities=[[1, 2]])
    right = bind_record("PB", "master2:canonical", scans, aliases(), set(), ambiguities=[[1, 2]])
    reverse = bind_record("PB", "master1:canonical", list(reversed(scans)), aliases(), set())
    assert len({left["input_id"], right["input_id"], reverse["input_id"]}) == 3
    assert left["scan_indices"] == [0, 1]
    assert reverse["scan_indices"] == [1, 0]
    assert left["ambiguities"] == [[1, 2]]
    assert left["physical_scan_ids"] == ["scan0", "scan1"]


def test_repeated_scan_or_cross_reference_prefix_is_rejected():
    from scripts.native_long_confirmation import bind_record

    with pytest.raises(ValueError, match="distinct"):
        bind_record("PB", "u", ["scene0001_00", "scene0001_00"], aliases(), set())
    rows = aliases()
    rows["scene0001_01"]["reference_id"] = "other"
    with pytest.raises(ValueError, match="reference"):
        bind_record("PB", "u", ["scene0001_00", "scene0001_01"], rows, set())
