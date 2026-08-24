from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import torch

from scripts.rescene_task_postprocess import extract_official_task_prediction
from trainer.trainer import InstanceSegmentation


def _system(*, filter_out_instances: bool = False) -> SimpleNamespace:
    system = SimpleNamespace(
        config=SimpleNamespace(
            general=SimpleNamespace(
                topk_per_image=4,
                filter_out_instances=filter_out_instances,
                scores_threshold=0.0,
                iou_threshold=0.8,
            )
        ),
        device=torch.device("cpu"),
        decoder_id=0,
        model=SimpleNamespace(train_on_segments=True),
    )
    system._get_predictions = lambda output: [
        {
            "pred_logits": output["pred_logits"].softmax(dim=-1)[..., :-1],
            "pred_masks": output["pred_masks"],
        }
    ]
    system._get_batch_masks = lambda prediction, bid, targets: prediction[0][
        "pred_masks"
    ][bid][targets[bid]["point2segment"]]
    system._get_mask_and_scores = lambda *args, **kwargs: (
        InstanceSegmentation._get_mask_and_scores(system, *args, **kwargs)
    )
    system._get_full_res_mask = (
        lambda mask, inverse_map, point2segment_full, is_heatmap=False: mask[
            inverse_map
        ]
    )
    system._filter_and_sort_predictions = lambda *args, **kwargs: (
        InstanceSegmentation._filter_and_sort_predictions(system, *args, **kwargs)
    )
    return system


def _fixture() -> tuple[dict[str, object], dict[str, object], dict[str, object], object]:
    output = {
        "pred_logits": torch.tensor(
            [[[4.0, 1.0, -2.0], [0.5, 3.5, -2.0]]], dtype=torch.float32
        ),
        "pred_masks": [
            torch.tensor(
                [[3.0, -3.0], [2.0, -2.0], [-2.0, 2.0]], dtype=torch.float32
            )
        ],
        "query_features": torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
        "aux_outputs": [],
    }
    target_low = {
        "point2segment": torch.tensor([0, 1, 2]),
        "temporal_stages": torch.tensor([0, 1, 1]),
    }
    target_full = {
        "point2segment": torch.tensor([0, 1, 2, 3]),
        "temporal_stages": torch.tensor([0, 1, 1, 1]),
    }
    data = SimpleNamespace(inverse_maps=[torch.tensor([0, 1, 1, 2])])
    return output, target_low, target_full, data


def _legacy_task_prediction(
    system: SimpleNamespace,
    output: dict[str, object],
    target_low: dict[str, object],
    target_full: dict[str, object],
    data: object,
) -> dict[str, torch.Tensor]:
    prediction = system._get_predictions(output)
    selected = prediction[system.decoder_id]
    low_masks = system._get_batch_masks(prediction, 0, [target_low])
    scores, low_masks, classes, heatmap = system._get_mask_and_scores(
        selected["pred_logits"][0].detach().cpu(),
        low_masks,
        selected["pred_logits"][0].shape[0],
        output["pred_logits"].shape[2] - 1,
    )
    masks = system._get_full_res_mask(
        low_masks, data.inverse_maps[0], target_full["point2segment"]
    )
    heatmap = system._get_full_res_mask(
        heatmap,
        data.inverse_maps[0],
        target_full["point2segment"],
        is_heatmap=True,
    )
    classes, masks, scores, _ = system._filter_and_sort_predictions(
        np.asarray(masks), scores, classes, np.asarray(heatmap)
    )
    return {
        "pred_masks": torch.as_tensor(masks).bool(),
        "pred_scores": torch.as_tensor(scores).float(),
        "pred_classes": torch.as_tensor(classes).long() + 10,
    }


def test_mask_and_scores_optional_lineage_preserves_legacy_outputs() -> None:
    system = _system()
    mask_cls = torch.tensor([[0.8, 0.2], [0.1, 0.9]])
    mask_pred = torch.tensor([[2.0, -2.0], [1.0, 3.0], [-1.0, 2.0]])

    legacy = system._get_mask_and_scores(mask_cls, mask_pred, 2, 2)
    with_lineage = system._get_mask_and_scores(
        mask_cls, mask_pred, 2, 2, return_lineage=True
    )

    assert len(legacy) == 4
    assert len(with_lineage) == 6
    for before, after in zip(legacy, with_lineage[:4], strict=True):
        assert torch.equal(before, after)
    assert with_lineage[4].tolist() == [1, 0, 0, 1]
    assert with_lineage[5].tolist() == [1, 0, 1, 0]


def test_official_task_postprocess_matches_existing_fullhistory_path() -> None:
    system = _system()
    output, target_low, target_full, data = _fixture()
    frozen_output = deepcopy(output)

    legacy = _legacy_task_prediction(system, output, target_low, target_full, data)
    processed = extract_official_task_prediction(
        system=system,
        output=output,
        target_low_resolution=target_low,
        target_full_resolution=target_full,
        data=data,
        class_mapper=lambda value: value + 10,
        latest_stage_index=1,
    )

    assert torch.equal(processed.pred_masks, legacy["pred_masks"])
    assert torch.equal(processed.pred_classes, legacy["pred_classes"])
    assert torch.allclose(processed.pred_scores, legacy["pred_scores"], atol=0, rtol=0)
    assert processed.source_query_ids.shape == processed.pred_scores.shape
    assert processed.source_class_ids.shape == processed.pred_scores.shape
    assert torch.equal(processed.pred_classes, processed.source_class_ids + 10)
    assert processed.latest_stage_masks.shape[0] == 3
    assert processed.latest_stage_masks.shape[1] == processed.pred_scores.shape[0]
    assert torch.equal(output["pred_logits"], frozen_output["pred_logits"])
    assert torch.equal(output["pred_masks"][0], frozen_output["pred_masks"][0])
