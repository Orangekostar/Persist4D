import torch

from scripts.short_module_diagnostics import pairwise_geometry


def test_joint_and_separate_geometry_distinguish_complementary_candidates(tmp_path):
    spec = tmp_path / "dataset.yaml"
    spec.write_text("name: rio\nclass_labels: [chair]\nvalid_class_ids: [3]\n"
                    "aux: changes\naux_labels: [static]\nvalid_aux_ids: [0]\n")
    prediction = {"pred_masks": torch.tensor([[1, 1, 1, 0], [1, 0, 1, 1]], dtype=torch.bool).T,
                  "pred_scores": torch.tensor([.1, .9]), "pred_classes": torch.tensor([3, 3])}
    target = {"masks": torch.ones(1, 4, dtype=torch.bool), "labels": torch.tensor([3]),
              "ids": torch.tensor([7]), "changes": torch.tensor([0]),
              "temporal_stages": torch.tensor([0, 0, 1, 1])}
    result = pairwise_geometry(prediction, target, dataset_spec=spec, min_region_size=1)
    assert result[7]["U_joint"] == .5
    assert result[7]["U_separate"] == 1
    assert result[7]["candidate_worst"] == {0: .5, 1: .5}
