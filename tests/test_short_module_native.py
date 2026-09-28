"""Native cache omits redundant floats, preserves retained class/query lineage."""

import torch

from scripts.rescene_task_postprocess import (
    OfficialTaskPrediction,
    OfficialTaskSoftEvidence,
)
from scripts.short_module_native import pack_prediction, unpack_prediction


def test_compact_roundtrip_preserves_soft_evidence_without_point_float_copies():
    z = torch.tensor([[1., 1.], [-1., -1.]])
    full_z = z[torch.tensor([0, 0, 1])]
    soft = OfficialTaskSoftEvidence(
        "FP32_SEGMENT_LOGITS", z, z, full_z, full_z.sigmoid(),
        torch.tensor([[.2, .3, .5], [.2, .3, .5]]), torch.ones(2, 3), torch.ones(2, 4),
        torch.tensor([7, 7]), torch.tensor([0, 1]), torch.tensor([0, 1]),
        torch.tensor([0, 0, 1]), torch.tensor([0, 0, 1]))
    stages = torch.tensor([0, 0, 1])
    parent = OfficialTaskPrediction(full_z > 0, torch.tensor([.8, .2]), torch.tensor([3, 4]),
                                    soft.source_query_ids, soft.source_class_ids, stages, 1,
                                    (full_z > 0)[2:], soft)
    packed = pack_prediction(parent)
    assert "full_resolution_logits" not in packed["soft"]
    assert "full_resolution_probabilities" not in packed["soft"]
    restored = unpack_prediction(packed)
    restored.validate()
    for key, value in parent.prediction().items():
        assert torch.equal(restored.prediction()[key], value)
    assert torch.equal(restored.soft_evidence.full_resolution_logits, full_z)
