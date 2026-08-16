import pytest
import torch
import torch.nn.functional as F

from models.criterion import (
    ContrastiveLoss,
    _streaming_info_nce_chunk,
    infoNCE_chunked_loss,
)


def _dense_info_nce_reference(
    features: torch.Tensor,
    instance_masks: torch.Tensor,
    temporal_stages: torch.Tensor,
    logit_scale: torch.Tensor,
    bias: torch.Tensor,
    *,
    include_self: bool,
    norm_type: str,
    stage_weight_same: float,
    stage_weight_cross: float,
) -> torch.Tensor:
    normalized = F.normalize(
        torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0),
        dim=-1,
        eps=1e-6,
    )
    similarities = normalized @ normalized.T
    if norm_type == "log_odds":
        similarities = torch.clamp(
            similarities,
            *ContrastiveLoss.CLAMP_LIMITS[features.dtype],
        )
        similarities = 2 * torch.atanh(similarities)
    elif norm_type == "clamp":
        similarities = ((1 + similarities) / 2).clamp(min=0, max=1)
    logits = similarities * logit_scale + bias
    if not include_self:
        logits = logits.masked_fill(
            torch.eye(len(features), dtype=torch.bool, device=features.device),
            -torch.inf,
        )

    positive_mask = instance_masks.T.float() @ instance_masks.float() > 0
    if not include_self:
        positive_mask.fill_diagonal_(False)
    valid_rows = positive_mask.any(dim=1)
    pair_weights = torch.where(
        temporal_stages[:, None] == temporal_stages[None, :],
        torch.as_tensor(
            stage_weight_same,
            dtype=features.dtype,
            device=features.device,
        ),
        torch.as_tensor(
            stage_weight_cross,
            dtype=features.dtype,
            device=features.device,
        ),
    )
    positive_logits = logits.masked_fill(~positive_mask, -torch.inf)
    positive_logits = positive_logits + torch.where(
        positive_mask,
        torch.log(pair_weights),
        torch.zeros_like(pair_weights),
    )
    return (
        torch.logsumexp(logits[valid_rows], dim=1)
        - torch.logsumexp(positive_logits[valid_rows], dim=1)
    ).mean()


def _instance_masks(num_segments: int, num_instances: int) -> torch.Tensor:
    instance_ids = torch.arange(num_segments) % num_instances
    return F.one_hot(instance_ids, num_classes=num_instances).T.bool()


def test_streaming_logsumexp_accumulates_float16_blocks_in_float32() -> None:
    num_candidates = 70_000
    anchors = torch.zeros(1, 4, dtype=torch.float16, requires_grad=True)
    candidates = torch.zeros(
        num_candidates,
        4,
        dtype=torch.float16,
        requires_grad=True,
    )

    denominator, numerator = _streaming_info_nce_chunk(
        anchors,
        candidates,
        torch.tensor([0]),
        torch.tensor([1]),
        torch.ones(1, dtype=torch.float16),
        torch.ones((), dtype=torch.float16),
        torch.zeros((), dtype=torch.float16),
        anchor_start=0,
        candidate_chunk_size=num_candidates,
        include_self=False,
        norm_type="temperature",
        clamp_limits=None,
    )
    loss = (denominator - numerator).mean()
    loss.backward()

    assert denominator.dtype == torch.float32
    assert torch.isfinite(loss)
    assert torch.isfinite(anchors.grad).all()
    assert torch.isfinite(candidates.grad).all()
    torch.testing.assert_close(
        loss,
        torch.tensor(float(num_candidates - 1)).log(),
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.parametrize("norm_type", ["log_odds", "temperature", "clamp"])
@pytest.mark.parametrize("include_self", [False, True])
@pytest.mark.parametrize("candidate_chunk_size", [1, 7])
def test_streaming_infonce_matches_dense_value_and_feature_gradient(
    candidate_chunk_size: int,
    include_self: bool,
    norm_type: str,
) -> None:
    generator = torch.Generator().manual_seed(17)
    features = torch.randn(37, 11, dtype=torch.float64, generator=generator)
    masks = _instance_masks(num_segments=37, num_instances=5)
    stages = torch.arange(37) % 2

    dense_features = features.clone().requires_grad_(True)
    dense_scale = torch.tensor(1.3, dtype=torch.float64, requires_grad=True)
    dense_bias = torch.tensor(-0.2, dtype=torch.float64, requires_grad=True)
    dense_loss = _dense_info_nce_reference(
        dense_features,
        masks,
        stages,
        dense_scale,
        dense_bias,
        include_self=include_self,
        norm_type=norm_type,
        stage_weight_same=0.8,
        stage_weight_cross=1.7,
    )
    dense_gradients = torch.autograd.grad(
        dense_loss,
        (dense_features, dense_scale, dense_bias),
    )

    streaming_features = features.clone().requires_grad_(True)
    streaming_scale = torch.tensor(1.3, dtype=torch.float64, requires_grad=True)
    streaming_bias = torch.tensor(-0.2, dtype=torch.float64, requires_grad=True)
    streaming_loss = infoNCE_chunked_loss(
        features=streaming_features,
        instance_masks=masks,
        chunk_size=13,
        candidate_chunk_size=candidate_chunk_size,
        logit_scale=streaming_scale,
        bias=streaming_bias,
        include_self=include_self,
        temporal_stages=stages,
        stage_weight_same=0.8,
        stage_weight_cross=1.7,
        norm_type=norm_type,
        clamp_limits=ContrastiveLoss.CLAMP_LIMITS[torch.float64],
    )
    streaming_gradients = torch.autograd.grad(
        streaming_loss,
        (streaming_features, streaming_scale, streaming_bias),
    )

    torch.testing.assert_close(streaming_loss, dense_loss, rtol=1e-11, atol=1e-12)
    for streaming_gradient, dense_gradient in zip(
        streaming_gradients,
        dense_gradients,
    ):
        torch.testing.assert_close(
            streaming_gradient,
            dense_gradient,
            rtol=1e-9,
            atol=1e-10,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an A40-class GPU")
def test_scene0186_dual_sample_streaming_infonce_has_bounded_peak_memory() -> None:
    device = torch.device("cuda", 0)
    properties = torch.cuda.get_device_properties(device)
    if properties.total_memory < 40 * 1024**3:
        pytest.skip("requires at least 40 GiB of device memory")

    num_segments = 16_131
    feature_dim = 128
    masks = _instance_masks(num_segments, num_instances=51).to(device)
    stages = (torch.arange(num_segments, device=device) % 2).long()
    generator = torch.Generator(device=device).manual_seed(186)
    first = torch.randn(
        num_segments,
        feature_dim,
        device=device,
        generator=generator,
        requires_grad=True,
    )
    second = torch.randn(
        num_segments,
        feature_dim,
        device=device,
        generator=generator,
        requires_grad=True,
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    losses = [
        infoNCE_chunked_loss(
            features=sample,
            instance_masks=masks,
            chunk_size=2048,
            temporal_stages=stages,
            stage_weight_same=1.0,
            stage_weight_cross=1.0,
            norm_type="log_odds",
            clamp_limits=ContrastiveLoss.CLAMP_LIMITS[torch.float32],
        )
        for sample in (first, second)
    ]
    loss = torch.stack(losses).mean()
    loss.backward()
    torch.cuda.synchronize(device)

    peak_bytes = torch.cuda.max_memory_allocated(device)
    assert torch.isfinite(loss)
    assert torch.isfinite(first.grad).all()
    assert torch.isfinite(second.grad).all()
    assert peak_bytes < 3 * 1024**3, (
        f"streaming InfoNCE retained {peak_bytes / 1024**3:.2f} GiB for "
        "two scene0186-scale samples"
    )
