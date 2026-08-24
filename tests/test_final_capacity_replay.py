import torch

from scripts.evaluate_persist4d_p6a import CachedProtocolSequence
from scripts.final_evidence_capacity import evaluate_capacity_sequences


def _payload(stage: int) -> dict[str, object]:
    point_count = 120
    mask = torch.ones((1, point_count), dtype=torch.bool)
    return {
        "schema_version": 3,
        "key": {
            "master_sequence_id": "master-fixture",
            "reference_scene_id": "reference-fixture",
            "order_id": "canonical",
            "stage_index": stage,
            "history_scan_ids": [f"scan-{index}" for index in range(stage + 1)],
            "local_window_scan_ids": [
                f"scan-{index}" for index in range(max(0, stage - 1), stage + 1)
            ],
        },
        "provenance": {
            "source_commit": "1" * 40,
            "checkpoint_sha256": "2" * 64,
            "config_sha256": "3" * 64,
            "dataset_sha256": "4" * 64,
        },
        "observation": {
            "features": torch.tensor([[1.0, 0.0]], dtype=torch.float32),
            "class_prob": torch.tensor([[0.99, 0.0, 0.01]]),
            "confidence": torch.tensor([0.99]),
            "valid": torch.tensor([True]),
            "masks": mask,
            "mask_support": torch.tensor([point_count]),
            "local_query_ids": torch.tensor([0]),
        },
        "target": {
            "gt_ids": torch.tensor([10]),
            "gt_classes": torch.tensor([0]),
            "gt_masks": mask.clone(),
            "changes": torch.tensor([0]),
            "change_labels_valid": False,
            "change_label_semantics": (
                "unavailable_for_protocol_b_order_stress_test_all_static_placeholder"
            ),
            "gt_class_semantics": "rescene_model_index_0_based",
        },
    }


class _Metric:
    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.count = 0

    def update(self, prediction: object, target: object) -> None:
        self.count += 1

    def compute(self) -> dict[str, float]:
        assert self.count > 0
        return {
            "online_t-mAP": 0.5,
            "online_t-mAP50": 0.6,
            "online_t-mAP25": 0.7,
            "online_t-REC": 0.8,
            "online_t-REC50": 0.9,
            "online_t-REC25": 1.0,
            "raw_local_AP": 0.4,
            "raw_local_AP50": 0.5,
            "raw_local_AP25": 0.6,
            "raw_local_REC": 0.7,
        }


def test_capacity_evaluation_covers_every_capacity_and_horizon() -> None:
    sequence = CachedProtocolSequence(
        reference_scene_id="reference-fixture",
        master_sequence_id="master-fixture",
        order_id="canonical",
        payloads=tuple(_payload(stage) for stage in range(5)),
    )

    result = evaluate_capacity_sequences(
        (sequence,),
        capacities=(1, 2),
        class_mapper=lambda value: value,
        background_class=2,
        metric_factory=_Metric,
        expected_sequence_count=1,
    )

    assert len(result.per_sequence_rows) == 2 * 4
    assert len(result.aggregate_rows) == 2 * 4
    assert {(row["capacity"], row["horizon"]) for row in result.aggregate_rows} == {
        (capacity, horizon) for capacity in (1, 2) for horizon in (2, 3, 4, 5)
    }
    assert all(row["sequence_count"] == 1 for row in result.aggregate_rows)
    assert all(row["causal_prefix_t_mAP"] == 0.5 for row in result.aggregate_rows)
    assert all(row["causal_prefix_t_REC"] == 0.8 for row in result.aggregate_rows)
    assert all(row["peak_occupied_slots"] == 1 for row in result.per_sequence_rows)
    assert all(row["accepted_births"] == 1 for row in result.per_sequence_rows)
    assert all(row["rejected_births"] == 0 for row in result.per_sequence_rows)
    assert len({row["observation_sha256"] for row in result.per_sequence_rows}) == 1
