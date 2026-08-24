from __future__ import annotations

from datasets.multiscan_adapter import MultiScanObjectAnnotation, detect_natural_gaps


def _object(object_id: int, label: str) -> MultiScanObjectAnnotation:
    return MultiScanObjectAnnotation(
        object_id=object_id,
        label=label,
        class_name=label.rsplit(".", 1)[0],
        mobility_type="movable",
        eligible=True,
    )


def _visibility_objects(
    pattern: str, *, label: str = "chair.1"
) -> tuple[tuple[MultiScanObjectAnnotation, ...], ...]:
    return tuple((_object(17, label),) if value == "1" else () for value in pattern)


def test_maximal_absence_interval_is_one_gap_of_length_two() -> None:
    scan_ids = tuple(f"scene_00069_{index:02d}" for index in range(5))

    gaps = detect_natural_gaps(
        scene_id="scene_00069",
        scan_ids=scan_ids,
        objects_by_scan=_visibility_objects("11001"),
    )

    assert gaps == [
        {
            "scene_id": "scene_00069",
            "object_id": 17,
            "class": "chair",
            "last_visible_before_gap": "scene_00069_01",
            "first_visible_after_gap": "scene_00069_04",
            "gap_length": 2,
        }
    ]


def test_two_maximal_absence_intervals_are_two_gap_episodes() -> None:
    scan_ids = tuple(f"scene_00069_{index:02d}" for index in range(5))

    gaps = detect_natural_gaps(
        scene_id="scene_00069",
        scan_ids=scan_ids,
        objects_by_scan=_visibility_objects("10101"),
    )

    assert [gap["gap_length"] for gap in gaps] == [1, 1]
    assert [gap["last_visible_before_gap"] for gap in gaps] == [
        "scene_00069_00",
        "scene_00069_02",
    ]
    assert [gap["first_visible_after_gap"] for gap in gaps] == [
        "scene_00069_02",
        "scene_00069_04",
    ]


def test_structural_and_removed_objects_are_not_gap_eligible() -> None:
    scan_ids = tuple(f"scene_00069_{index:02d}" for index in range(3))
    for label in ("wall.1", "floor.1", "ceiling.1", "remove.1"):
        objects = _visibility_objects("101", label=label)

        assert (
            detect_natural_gaps(
                scene_id="scene_00069",
                scan_ids=scan_ids,
                objects_by_scan=objects,
            )
            == []
        )
