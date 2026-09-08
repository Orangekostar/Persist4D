import pytest
import torch

from datasets.persist4d_sequence_dataset import EpisodeSpec, Persist4DEpisodeBatch
from models.persistent_memory import PersistentMemory
from models.persistent_memory_read import DetachedMemoryReadState
from trainer.persist4d_allt_trainer import (
    Persist4DAllTTrainer,
    adapter_gradient_snapshot,
    bounded_warmup_steps,
    build_detached_memory_read_state,
    prefix_balanced_stage_coefficients,
    update_prediction_memory,
)


@pytest.mark.parametrize(
    ("total_steps", "fraction", "expected"),
    [(2, 0.05, 1), (4000, 0.05, 200)],
)
def test_bounded_warmup_steps(total_steps, fraction, expected) -> None:
    assert bounded_warmup_steps(total_steps, fraction) == expected


def test_bounded_warmup_steps_rejects_single_update() -> None:
    with pytest.raises(ValueError, match="at least two"):
        bounded_warmup_steps(1, 0.05)


def test_episode_batch_transfer_rebuilds_frozen_container() -> None:
    trainer = Persist4DAllTTrainer.__new__(Persist4DAllTTrainer)
    spec = EpisodeSpec(
        reference_id="reference",
        sequence_id="scan",
        scan_indices=(0,),
        role="adaptation",
        context_index=0,
        horizon=1,
        augmentation_seed=45,
        draw_index=0,
    )
    original_tensor = torch.ones(2)
    batch = Persist4DEpisodeBatch(
        specs=(spec,),
        stage_batches=(
            ({"features": original_tensor}, [{"labels": original_tensor}], ["scan"]),
        ),
        ambiguity_metadata=({"kept_on_cpu": True},),
    )

    moved = trainer.transfer_batch_to_device(batch, torch.device("meta"), 0)

    assert moved is not batch
    assert moved.specs is batch.specs
    assert moved.ambiguity_metadata is batch.ambiguity_metadata
    assert moved.stage_batches[0][0]["features"].device.type == "meta"
    assert moved.stage_batches[0][1][0]["labels"].device.type == "meta"
    assert original_tensor.device.type == "cpu"


def test_adapter_gradient_snapshot_separates_real_gradients_from_frozen_state() -> None:
    output = torch.nn.Parameter(torch.ones(2))
    inner = torch.nn.Parameter(torch.ones(2))
    frozen = torch.nn.Parameter(torch.ones(2), requires_grad=False)
    output.grad = torch.tensor([2.0, 0.0])
    inner.grad = torch.zeros(2)

    snapshot = adapter_gradient_snapshot(
        [
            ("model.memory_read.output_projection.weight", output),
            ("model.memory_read.query_projection.weight", inner),
            ("model.backbone.encoder.weight", frozen),
        ]
    )

    assert snapshot["nonzero_adapter_gradients"] == [
        "model.memory_read.output_projection.weight"
    ]
    assert snapshot["zero_adapter_gradients"] == [
        "model.memory_read.query_projection.weight"
    ]
    assert snapshot["frozen_gradient_names"] == []
    assert snapshot["nonfinite_gradient_names"] == []


@pytest.mark.parametrize(
    ("horizon", "expected"),
    [
        (1, (1.0,)),
        (2, (0.5, 0.5)),
        (3, (5 / 12, 5 / 12, 1 / 6)),
        (4, (13 / 36, 13 / 36, 7 / 36, 1 / 12)),
        (5, (77 / 240, 77 / 240, 47 / 240, 9 / 80, 1 / 20)),
    ],
)
def test_prefix_balanced_stage_coefficients(horizon, expected) -> None:
    observed = prefix_balanced_stage_coefficients(horizon)

    assert observed == pytest.approx(expected)
    assert sum(observed) == pytest.approx(1.0)


def _output(feature: torch.Tensor) -> dict[str, object]:
    logits = torch.full((1, 100, 19), -8.0, device=feature.device)
    logits[:, 0, 3] = 8.0
    masks = torch.full((2, 100), -8.0, device=feature.device)
    masks[:, 0] = 8.0
    return {
        "query_features": feature,
        "pred_logits": logits,
        "pred_masks": [masks],
    }


def _target() -> list[dict[str, torch.Tensor]]:
    return [
        {
            "labels": torch.tensor([3]),
            "masks": torch.ones(1, 2),
            "point2segment": torch.tensor([0, 1]),
            "temporal_stages": torch.tensor([1, 1]),
        }
    ]


def test_prediction_memory_update_is_detached_and_has_no_gt_fields() -> None:
    features = torch.randn(1, 100, 128, requires_grad=True)
    memory = PersistentMemory(capacity=100)

    state, read_state = update_prediction_memory(
        output=_output(features),
        targets=_target(),
        state=None,
        memory=memory,
        stage_index=0,
        observation_settings={
            "background_class": 18,
            "confidence_threshold": 0.5,
            "mask_threshold": 0.5,
            "minimum_mask_support": 1,
        },
    )

    assert isinstance(read_state, DetachedMemoryReadState)
    assert read_state.embeddings.grad_fn is None
    assert not read_state.embeddings.requires_grad
    assert state.embedding.grad_fn is None
    assert vars(read_state).keys() == {
        "embeddings",
        "occupied_mask",
        "active_mask",
        "confidence",
        "last_seen",
    }
    assert read_state.occupied_mask.sum().item() == 1


def test_detached_read_state_cannot_alias_mutable_memory() -> None:
    features = torch.randn(1, 100, 128)
    memory = PersistentMemory(capacity=100)
    state, _ = update_prediction_memory(
        output=_output(features),
        targets=_target(),
        state=None,
        memory=memory,
        stage_index=0,
        observation_settings={
            "background_class": 18,
            "confidence_threshold": 0.5,
            "mask_threshold": 0.5,
            "minimum_mask_support": 1,
        },
    )
    read_state = build_detached_memory_read_state(state)
    before = read_state.embeddings.clone()

    state.embedding.add_(4.0)

    assert torch.equal(read_state.embeddings, before)
