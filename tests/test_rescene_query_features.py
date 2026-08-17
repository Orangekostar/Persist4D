import os
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
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
_VERIFY_REAL_GPU = os.environ.get("P5_VERIFY_GPU_ARTIFACTS") == "1"


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


def _seed_real_inference(seed: int = 45) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _assert_nested_exact(actual, expected, *, path: str) -> None:
    assert type(actual) is type(expected), path
    if isinstance(expected, torch.Tensor):
        assert actual.shape == expected.shape, path
        assert actual.dtype == expected.dtype, path
        assert actual.device == expected.device, path
        assert torch.equal(actual, expected), path
        return
    if isinstance(expected, np.ndarray):
        assert np.array_equal(actual, expected), path
        return
    if isinstance(expected, Mapping):
        assert tuple(actual) == tuple(expected), path
        for key in expected:
            _assert_nested_exact(
                actual[key],
                expected[key],
                path=f"{path}.{key}",
            )
        return
    if isinstance(expected, Sequence) and not isinstance(
        expected,
        (str, bytes, bytearray),
    ):
        assert len(actual) == len(expected), path
        for index, (actual_item, expected_item) in enumerate(
            zip(actual, expected, strict=True)
        ):
            _assert_nested_exact(
                actual_item,
                expected_item,
                path=f"{path}[{index}]",
            )
        return
    if expected is None or isinstance(expected, (str, bytes, int, float, bool)):
        assert actual == expected, path


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


@pytest.mark.skipif(
    not _VERIFY_REAL_GPU,
    reason="set P5_VERIFY_GPU_ARTIFACTS=1 for the canonical GPU parity gate",
)
def test_real_t2_checkpoint_query_export_preserves_every_legacy_tensor():
    import hydra
    from omegaconf import OmegaConf

    from scripts.evaluate_persist4d import (
        _compose_runtime_config,
        _move_data_to_device,
        _move_targets_to_device,
    )
    from trainer.trainer import InstanceSegmentation

    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0"
    assert torch.cuda.is_available()
    checkpoint = REPO_ROOT / "checkpoints" / "rescene4d_concerto_t2_repro.ckpt"
    assert checkpoint.is_file()
    assert not checkpoint.is_symlink()

    config, _ = _compose_runtime_config()
    config.model.return_query_features = False
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert isinstance(payload, Mapping)
    assert isinstance(payload.get("state_dict"), Mapping)
    system = InstanceSegmentation(config)
    incompatible = system.load_state_dict(payload["state_dict"], strict=True)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    del payload

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    system.to(device)
    system.eval()

    dataset_config = OmegaConf.create(
        OmegaConf.to_container(config.data.validation_dataset, resolve=True)
    )
    dataset_config.temporal_window = 2
    dataset = hydra.utils.instantiate(dataset_config)
    collate = hydra.utils.instantiate(config.data.validation_collation)
    assert len(dataset.sequence_indices) > 0
    scan_indices = tuple(int(index) for index in dataset.sequence_indices[0])
    sequence_name = dataset.sequence_names[0]

    def materialize_fixed_sample():
        _seed_real_inference()
        sample = dataset.load_scan_indices(
            0,
            scan_indices,
            change_file=None,
        )
        data, targets, names = collate([sample])
        assert list(names) == [sequence_name]
        data = _move_data_to_device(data, device)
        targets = _move_targets_to_device(targets, device)
        raw_coordinates = system._process_raw_coordinates(data)
        return data, targets, raw_coordinates

    deterministic_enabled = torch.are_deterministic_algorithms_enabled()
    deterministic_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    cudnn_benchmark = torch.backends.cudnn.benchmark
    cudnn_deterministic = torch.backends.cudnn.deterministic
    cuda_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_allow_tf32 = torch.backends.cudnn.allow_tf32
    matmul_precision = torch.get_float32_matmul_precision()
    try:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        torch.use_deterministic_algorithms(True)

        disabled_data, disabled_targets, disabled_raw = materialize_fixed_sample()
        enabled_data, enabled_targets, enabled_raw = materialize_fixed_sample()

        system.model.return_query_features = False
        _seed_real_inference()
        with torch.inference_mode():
            disabled = system(
                disabled_data,
                point2segment=[disabled_targets[0]["point2segment"]],
                raw_coordinates=disabled_raw,
                is_eval=True,
                targets=disabled_targets,
            )

        system.model.return_query_features = True
        _seed_real_inference()
        with torch.inference_mode():
            enabled = system(
                enabled_data,
                point2segment=[enabled_targets[0]["point2segment"]],
                raw_coordinates=enabled_raw,
                is_eval=True,
                targets=enabled_targets,
            )

        assert set(disabled) == BASE_OUTPUT_KEYS
        assert set(enabled) == BASE_OUTPUT_KEYS | {"query_features"}
        for key in sorted(BASE_OUTPUT_KEYS):
            _assert_nested_exact(enabled[key], disabled[key], path=key)
        query_features = enabled["query_features"]
        assert query_features.shape == (1, 100, 128)
        assert torch.isfinite(query_features).all()
    finally:
        torch.use_deterministic_algorithms(
            deterministic_enabled,
            warn_only=deterministic_warn_only,
        )
        torch.backends.cudnn.benchmark = cudnn_benchmark
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.set_float32_matmul_precision(matmul_precision)
        torch.backends.cuda.matmul.allow_tf32 = cuda_allow_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_allow_tf32
