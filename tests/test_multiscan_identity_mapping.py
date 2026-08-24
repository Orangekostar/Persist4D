from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from datasets.multiscan_adapter import (
    MultiScanAdapterError,
    build_multiscan_identity_records,
    inspect_multiscan_instance_payload,
    read_multiscan_annotation,
)


def _write_annotation(
    path: Path,
    *,
    scan_id: str,
    objects: list[dict[str, object]],
) -> Path:
    path.write_text(
        json.dumps(
            {
                "version": "multiscan@0.0.1",
                "scanId": scan_id,
                "objects": objects,
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_payload(
    path: Path,
    *,
    inst2obj_id: dict[int, int],
    inst2obj: dict[int, str],
) -> Path:
    torch.save(
        {
            "xyz": torch.zeros((2, 3)),
            "instance_ids": torch.tensor(list(inst2obj_id)),
            "inst2obj_id": inst2obj_id,
            "inst2obj": inst2obj,
        },
        path,
    )
    return path


def test_release_payload_local_instances_resolve_to_scene_scoped_identity(
    tmp_path: Path,
) -> None:
    scan_a = "scene_00069_00"
    scan_b = "scene_00069_01"
    annotation_a = read_multiscan_annotation(
        _write_annotation(
            tmp_path / f"{scan_a}.annotations.json",
            scan_id=scan_a,
            objects=[{"objectId": 17, "label": "chair.1", "mobilityType": "movable"}],
        ),
        expected_scan_id=scan_a,
    )
    annotation_b = read_multiscan_annotation(
        _write_annotation(
            tmp_path / f"{scan_b}.annotations.json",
            scan_id=scan_b,
            objects=[{"objectId": 17, "label": "chair.1", "mobilityType": "movable"}],
        ),
        expected_scan_id=scan_b,
    )
    payload_a = inspect_multiscan_instance_payload(
        _write_payload(
            tmp_path / f"{scan_a}.pth",
            inst2obj_id={2: 17},
            inst2obj={2: "chair.1"},
        )
    )
    payload_b = inspect_multiscan_instance_payload(
        _write_payload(
            tmp_path / f"{scan_b}.pth",
            inst2obj_id={8: 17},
            inst2obj={8: "chair.1"},
        )
    )

    records = build_multiscan_identity_records(
        scene_id="scene_00069",
        annotations=(annotation_a, annotation_b),
        payloads=(payload_a, payload_b),
    )

    assert [(record.scan_id, record.local_instance_id) for record in records] == [
        (scan_a, 2),
        (scan_b, 8),
    ]
    assert records[0].identity_key == ("scene_00069", 17)
    assert records[1].identity_key == records[0].identity_key
    assert records[0].object_label == records[1].object_label == "chair.1"


def test_payload_inspection_requires_real_inst2obj_id_key(tmp_path: Path) -> None:
    path = tmp_path / "missing.pth"
    torch.save({"inst2obj": {2: "chair.1"}}, path)

    with pytest.raises(MultiScanAdapterError, match="inst2obj_id"):
        inspect_multiscan_instance_payload(path)


def test_payload_and_annotation_object_ids_must_agree(tmp_path: Path) -> None:
    scan_id = "scene_00069_00"
    annotation = read_multiscan_annotation(
        _write_annotation(
            tmp_path / f"{scan_id}.annotations.json",
            scan_id=scan_id,
            objects=[{"objectId": 18, "label": "chair.1"}],
        ),
        expected_scan_id=scan_id,
    )
    payload = inspect_multiscan_instance_payload(
        _write_payload(
            tmp_path / f"{scan_id}.pth",
            inst2obj_id={2: 17},
            inst2obj={2: "chair.1"},
        )
    )

    with pytest.raises(MultiScanAdapterError, match="annotation objectId"):
        build_multiscan_identity_records(
            scene_id="scene_00069",
            annotations=(annotation,),
            payloads=(payload,),
        )


def test_cross_scan_object_id_label_conflict_fails_closed(tmp_path: Path) -> None:
    annotations = []
    payloads = []
    for suffix, local_instance, label in (
        ("00", 2, "chair.1"),
        ("01", 8, "table.1"),
    ):
        scan_id = f"scene_00069_{suffix}"
        annotations.append(
            read_multiscan_annotation(
                _write_annotation(
                    tmp_path / f"{scan_id}.annotations.json",
                    scan_id=scan_id,
                    objects=[{"objectId": 17, "label": label}],
                ),
                expected_scan_id=scan_id,
            )
        )
        payloads.append(
            inspect_multiscan_instance_payload(
                _write_payload(
                    tmp_path / f"{scan_id}.pth",
                    inst2obj_id={local_instance: 17},
                    inst2obj={local_instance: label},
                )
            )
        )

    with pytest.raises(MultiScanAdapterError, match="cross-scan objectId"):
        build_multiscan_identity_records(
            scene_id="scene_00069",
            annotations=tuple(annotations),
            payloads=tuple(payloads),
        )


def test_annotation_requires_explicit_object_id_without_index_fallback(
    tmp_path: Path,
) -> None:
    scan_id = "scene_00069_00"
    path = _write_annotation(
        tmp_path / f"{scan_id}.annotations.json",
        scan_id=scan_id,
        objects=[{"label": "chair.1", "type": "movable"}],
    )

    with pytest.raises(MultiScanAdapterError, match="objectId"):
        read_multiscan_annotation(path, expected_scan_id=scan_id)
