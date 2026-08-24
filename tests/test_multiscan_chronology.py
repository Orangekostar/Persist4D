from __future__ import annotations

from scripts.audit_multiscan_dataset import build_chronology_audit


def test_chronology_is_dataset_order_only_without_cross_capture_clock() -> None:
    inventory = {
        "status": "pass",
        "selected_scene_list_sha256": "a" * 64,
        "scenes": [
            {
                "scene_id": "scene_00069",
                "scan_ids": [
                    "scene_00069_00",
                    "scene_00069_01",
                    "scene_00069_02",
                ],
                "number_of_scans": 3,
                "official_split": "train",
            }
        ],
    }

    audit = build_chronology_audit(inventory)

    assert audit["status"] == "DATASET_ORDER_ONLY"
    assert audit["physical_chronology_proven"] is False
    assert audit["ordered_revisit_protocol_allowed"] is True
    assert audit["ordering_rule"] == "numeric scan suffix within each physical scene"
    assert audit["selected_scene_list_sha256"] == "a" * 64
    assert audit["scene_orders"] == [
        {
            "scene_id": "scene_00069",
            "ordered_scan_ids": [
                "scene_00069_00",
                "scene_00069_01",
                "scene_00069_02",
            ],
        }
    ]
    assert "not proven physical chronology" in audit["claim_boundary"]
