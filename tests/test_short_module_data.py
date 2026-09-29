"""Hand-computed supervision tests using the installed official matcher."""

import pytest
import torch

from scripts.short_module_data import (
    geometry_assignment,
    geometry_loss,
    low_segment_targets,
    quality_labels,
    select_population,
    threshold_quality_loss,
)


@pytest.fixture
def spec(tmp_path):
    path = tmp_path / "dataset.yaml"
    path.write_text("name: rio\nclass_labels: [chair, table]\nvalid_class_ids: [3, 4]\n"
                    "aux: changes\naux_labels: [static]\nvalid_aux_ids: [0]\n")
    return path


def fixture_pair():
    # GT1 disappears in stage2. GT2 exists in both stages.
    target = {"masks": torch.tensor([[1, 1, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 1, 1, 0, 0, 1, 1]], dtype=torch.bool),
              "labels": torch.tensor([3, 3]), "ids": torch.tensor([11, 22]),
              "changes": torch.zeros(2, dtype=torch.long),
              "temporal_stages": torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])}
    prediction = {"pred_masks": torch.tensor([[1, 1, 0, 0, 0, 0, 0, 0],
                                               [0, 0, 1, 1, 0, 0, 1, 0],
                                               [0, 0, 0, 0, 0, 0, 0, 0],
                                               [0, 0, 1, 1, 0, 0, 1, 1]], dtype=torch.bool).T,
                  "pred_scores": torch.tensor([.1, .9, .5, .4]),
                  "pred_classes": torch.tensor([3, 3, 3, 4])}
    return prediction, target


@pytest.mark.parametrize("reverse_stages", [False, True])
def test_official_labels_disappearance_strict_threshold_and_score_invariance(spec, reverse_stages):
    prediction, target = fixture_pair()
    if reverse_stages:
        target["temporal_stages"] = 1 - target["temporal_stages"]  # Appearance instead of disappearance.
    labels = quality_labels(prediction, target, dataset_spec=spec, min_region_size=1)
    assert torch.equal(labels["concat"], torch.tensor([1., .75, 0., 0.]))
    assert torch.equal(labels["temporal"], torch.tensor([1., .5, 0., 0.]))
    assert labels["gt_ids"] == [11, 22, None, None]
    assert labels["status"] == ["MATCHED", "MATCHED", "EMPTY", "VALID_NEGATIVE"]
    assert torch.equal(labels["valid"], torch.tensor([1, 1, 0, 1], dtype=torch.bool))
    assert not labels["events"][1].any()  # Strict > .5, not >= .5.
    order = torch.tensor([3, 1, 0, 2])
    reordered = {"pred_masks": prediction["pred_masks"][:, order],
                 "pred_scores": torch.tensor([.99, .01, .8, .6]),
                 "pred_classes": prediction["pred_classes"][order]}
    other = quality_labels(reordered, target, dataset_spec=spec, min_region_size=1)
    assert torch.equal(other["temporal"], labels["temporal"][order])
    assert torch.equal(other["valid"], labels["valid"][order])
    assert labels["ambiguity_metadata"] == "AMBIGUITY_METADATA_UNAVAILABLE"


def test_unknown_void_and_ambiguous_candidates_are_not_fake_negatives(spec):
    prediction, target = fixture_pair()
    prediction["pred_masks"][:, 0] = torch.tensor([0, 0, 0, 0, 1, 1, 0, 0])
    labels = quality_labels(prediction, target, dataset_spec=spec, min_region_size=1)
    assert labels["status"][0] == "IGNORE"
    target["ambiguities"] = [[11, 22]]
    labels = quality_labels(prediction, target, dataset_spec=spec, min_region_size=1)
    assert labels["status"][1] == "AMBIGUOUS"
    assert not labels["valid"][1]
    prediction["pred_classes"][0] = 999
    labels = quality_labels(prediction, target, dataset_spec=spec, min_region_size=1)
    assert labels["status"][0] == "UNKNOWN_CLASS"
    assert not labels["threshold_valid"][0].any()


def test_partial_void_has_threshold_validity_and_frozen_geometry_eligibility(spec):
    target = {"masks": torch.tensor([[1] * 100 + [0] * 150], dtype=torch.bool),
              "labels": torch.tensor([3]), "ids": torch.tensor([11]),
              "changes": torch.zeros(1, dtype=torch.long),
              "temporal_stages": torch.zeros(250, dtype=torch.long)}
    prediction = {"pred_masks": torch.stack((target["masks"][0],
                    torch.ones(250, dtype=torch.bool), ~target["masks"][0]), dim=1),
                  "pred_scores": torch.tensor([.9, .1, .8]),
                  "pred_classes": torch.tensor([3, 3, 3])}
    labels = quality_labels(prediction, target, dataset_spec=spec, min_region_size=1)
    assert labels["temporal"].tolist() == pytest.approx([1., .4, 0.])
    assert labels["valid"].tolist() == [True, True, False]
    assert labels["geometry_valid"].tolist() == [True, False, False]
    assert labels["all_thresholds_unscored"].tolist() == [False, False, True]
    thresholds = torch.tensor(labels["thresholds"])
    assert torch.equal(labels["threshold_valid"][1], thresholds >= labels["ignore_proportion"][1])
    assert labels["threshold_valid"][0].all()
    assert not labels["threshold_valid"][2].any()
    # Q eligibility can change without admitting this column into M assignment.
    assignment = geometry_assignment(prediction, target, labels,
                                    candidate_keys=[("input", i) for i in range(3)])
    assert assignment.tolist() == [11, -1, -1]
    order = torch.tensor([2, 0, 1])
    other = quality_labels({"pred_masks": prediction["pred_masks"][:, order],
                            "pred_scores": torch.tensor([.01, .2, .99]),
                            "pred_classes": prediction["pred_classes"][order]},
                           target, dataset_spec=spec, min_region_size=1)
    for key in ("temporal", "concat", "threshold_valid", "geometry_valid"):
        assert torch.equal(other[key], labels[key][order])


def test_threshold_loss_excludes_only_invalid_entries_and_preserves_gradients():
    probabilities = torch.tensor([[.2, .8], [.7, .3]], requires_grad=True)
    targets = torch.tensor([[0., 1.], [0., 1.]])
    valid = torch.tensor([[False, True], [True, False]])
    loss = threshold_quality_loss(probabilities, targets, valid)
    assert loss.item() == pytest.approx((-torch.log(torch.tensor(.8))
                                         - torch.log(torch.tensor(.3))).item() / 2)
    loss.backward()
    assert torch.equal(probabilities.grad == 0, ~valid)
    assert torch.isfinite(probabilities.grad).all()
    probabilities.grad.zero_()
    zero = threshold_quality_loss(probabilities, targets, torch.zeros_like(valid))
    zero.backward()
    assert zero.item() == 0
    assert not probabilities.grad.any()


def test_low_segment_targets_include_known_background_and_exclude_unknown():
    # Full-eval partition intentionally absent: only voxel inverse -> low segment.
    values, weights = low_segment_targets(
        low_point2segment=torch.tensor([1, 0, 1]),
        voxel_inverse=torch.tensor([0, 1, 2, 0, 1]),
        gt_mask=torch.tensor([1, 1, 0, 0, 1], dtype=torch.bool),
        semantic_labels=torch.tensor([3, 3, 0, 255, 255]), segment_count=2)
    assert torch.equal(values, torch.tensor([1., .5]))
    assert torch.equal(weights, torch.tensor([1., 2.]))


def test_geometry_preserve_uses_all_candidate_slots_and_shared_weights():
    logits = torch.zeros(2, 2, requires_grad=True)
    delta = torch.zeros_like(logits)
    targets = torch.tensor([[1., 0.], [0., 0.]])
    weights = torch.ones(2)
    arguments = {"logits": logits, "delta": delta, "targets": targets, "weights": weights,
                     "segment_stages": torch.zeros(2, dtype=torch.long),
                     "matched": torch.tensor([True, False]), "usable": torch.tensor([True, True]),
                     "sample_indices": [torch.tensor([0, 1])]}
    preserve = geometry_loss("M1", **arguments)
    # Matched BCE=ln2 and Dice=1-(2*.5+1)/(1+1+1)=1/3; divide by 2 slots.
    assert preserve.item() == pytest.approx((0.69314718056 + 1 / 3) / 2)
    erase = geometry_loss("M0", **arguments)
    assert erase > preserve
    preserve.backward()
    assert torch.equal(logits.grad[:, 1], torch.zeros(2))
    assert logits.grad[:, 0].abs().sum() > 0


def test_geometry_assignment_is_one_entity_across_stages_and_one_to_one(spec):
    prediction, target = fixture_pair()
    prediction["pred_classes"][3] = 3
    labels = quality_labels(prediction, target, dataset_spec=spec, min_region_size=1)
    assigned = geometry_assignment(prediction, target, labels,
                                   candidate_keys=[("input", i, 3, i) for i in range(4)])
    assert assigned.tolist() == [11, -1, -1, 22]


def test_population_preserves_logical_pairs_but_t1_is_unique_and_disjoint():
    def record(ref, a, b, logical):
        return {"reference_id": ref, "scan_ids": [a, b], "scan_indices": [0, 1],
                    "context_index": 0, "dataset_horizon": 2, "logical_unit_id": logical}
    train = [record(f"ref{i}", f"{i}a", f"{i}b", f"train{i}") for i in range(8)]
    cal = [record("cal", "ca", "cb", "cal0"), record("cal", "ca", "cb", "cal1")]
    sel = [record("sel", "sa", "sb", "sel0")]
    result = select_population(train, cal, sel, excluded_references={"cal", "sel", "pb"})
    assert result["counts"]["CAL"] == {"references": 1, "T2": 2, "unique_pairs": 1, "T1": 2}
    assert result["counts"]["TRAIN"]["references"] == 8
    assert all(r["horizon"] == len(r["scan_ids"]) for r in result["records"])
    assert result == select_population(list(reversed(train)), cal, sel,
                                       excluded_references={"cal", "sel", "pb"})
    with pytest.raises(ValueError, match="overlap"):
        select_population(train + [record("cal", "ca", "cc", "bad")], cal, sel,
                          excluded_references={"cal", "sel", "pb"})


def test_geometry_unknown_supervision_still_receives_gt_free_preservation_regularizer():
    delta = torch.ones(2, 2, requires_grad=True)
    loss = geometry_loss("M1", logits=delta, delta=delta, targets=torch.zeros(2, 2),
                         weights=torch.ones(2), segment_stages=torch.zeros(2, dtype=torch.long),
                         matched=torch.tensor([False, False]), usable=torch.tensor([False, False]),
                         sample_indices=[torch.tensor([0, 1])])
    assert loss.item() == pytest.approx(.01)
    loss.backward()
    assert torch.all(delta.grad > 0)
