from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import yaml

from models.rescene import ReScene


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_OUTPUT_KEYS = {
    "pred_logits",
    "pred_changes",
    "pred_masks",
    "aux_outputs",
    "sampled_coords",
    "backbone_features",
    "segment_features",
}


class _SparseBatch:
    def __init__(self, features):
        self.decomposed_features = features
        self.decomposed_coordinates = [
            torch.zeros(len(batch), 3, device=batch.device) for batch in features
        ]
        self.F = torch.cat(features)


class _StubBackbone(nn.Module):
    PLANES = [4, 4, 4, 4, 4]

    def __init__(self):
        super().__init__()
        self.features = [
            torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]),
            torch.tensor(
                [
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                    [1.0, 1.0, 0.0, 0.0],
                ]
            ),
        ]

    def forward(self, _x):
        pcd_features = _SparseBatch(self.features)
        return pcd_features, [pcd_features], [[], []]

    def sparse_from_sample(self, features, reference):
        sizes = [len(batch) for batch in reference.decomposed_features]
        return _SparseBatch(list(features.split(sizes)))


class _PassthroughAttention(nn.Module):
    def forward(self, queries, *_args, **_kwargs):
        return queries


class _DeterministicDecoderUpdate(nn.Module):
    def __init__(self):
        super().__init__()
        self.offset = nn.Parameter(torch.tensor([0.5, -0.25, 0.75, -1.0]))

    def forward(self, queries):
        return queries + self.offset


class _LightweightDecoderReScene(ReScene):
    def get_pos_encs(self, _coords):
        return [
            [
                [
                    torch.zeros(len(batch), self.mask_dim, device=batch.device)
                    for batch in self.backbone.features
                ]
            ]
        ]

    def attn_mask(self, *_args, **_kwargs):
        return [
            torch.zeros(
                len(batch), self.num_queries, dtype=torch.bool, device=batch.device
            )
            for batch in self.backbone.features
        ]


def _build_model(**overrides):
    kwargs = {
        "config": SimpleNamespace(backbone=_StubBackbone()),
        "hidden_dim": 4,
        "num_queries": 3,
        "num_heads": 1,
        "dim_feedforward": 8,
        "sample_sizes": [None],
        "shared_decoder": True,
        "num_classes": 2,
        "num_decoders": 1,
        "dropout": 0.0,
        "pre_norm": False,
        "positional_encoding_type": "fourier",
        "non_parametric_queries": False,
        "train_on_segments": False,
        "normalize_pos_enc": True,
        "use_level_embed": False,
        "scatter_type": "mean",
        "hlevels": [0],
        "use_np_features": False,
        "voxel_size": 0.02,
        "max_sample_size": False,
        "random_queries": False,
        "gauss_scale": 1.0,
        "random_query_both": False,
        "random_normal": False,
        "D": 3,
        "num_changes": 2,
        "temporal_masking": False,
        "use_changes_loss": False,
        "save_segment_info": False,
    }
    kwargs.update(overrides)
    model = _LightweightDecoderReScene(**kwargs)
    model.cross_attention[0][0] = _PassthroughAttention()
    model.self_attention[0][0] = _PassthroughAttention()
    model.ffn_attention[0][0] = _DeterministicDecoderUpdate()
    with torch.no_grad():
        model.query_feat.weight.copy_(
            torch.tensor(
                [
                    [1.0, 2.0, 4.0, 8.0],
                    [2.0, 3.0, 5.0, 7.0],
                    [-1.0, 0.0, 1.0, 3.0],
                ]
            )
        )
    return model


def _initial_queries(model):
    return model.query_feat.weight.unsqueeze(0).repeat(2, 1, 1)


def _expected_query_features(model):
    decoder_update = model.ffn_attention[0][0]
    return model.decoder_norm(_initial_queries(model) + decoder_update.offset)


def _forward(model):
    return model(SimpleNamespace())


def test_rescene_config_disables_query_features_by_default():
    config = yaml.safe_load(
        (REPO_ROOT / "conf" / "model" / "rescene.yaml").read_text(encoding="utf-8")
    )

    assert config["return_query_features"] is False


def test_default_output_keys_are_unchanged():
    model = _build_model()

    output = _forward(model)

    assert model.return_query_features is False
    assert set(output) == BASE_OUTPUT_KEYS


def test_enabled_query_features_are_final_decoder_normalized_queries():
    model = _build_model(return_query_features=True)
    expected = _expected_query_features(model)

    output = _forward(model)

    assert model.return_query_features is True
    assert set(output) == BASE_OUTPUT_KEYS | {"query_features"}
    assert output["query_features"].shape == (2, 3, 4)
    torch.testing.assert_close(output["query_features"], expected)


def test_query_features_follow_a_non_identity_decoder_update():
    model = _build_model(return_query_features=True)
    initial_query_features = model.decoder_norm(_initial_queries(model))

    query_features = _forward(model)["query_features"]

    assert not torch.allclose(query_features, initial_query_features)


def test_query_features_preserve_gradient_flow():
    model = _build_model(return_query_features=True)

    query_features = _forward(model)["query_features"]
    query_features[0, 0, 0].backward()

    assert query_features.requires_grad
    assert model.query_feat.weight.grad is not None
    assert torch.count_nonzero(model.query_feat.weight.grad) > 0
    decoder_update = model.ffn_attention[0][0]
    assert decoder_update.offset.grad is not None
    assert torch.count_nonzero(decoder_update.offset.grad) > 0
