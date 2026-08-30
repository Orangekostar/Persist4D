from __future__ import annotations

import copy

import pytest
import torch
from omegaconf import OmegaConf

import main_instance_segmentation as entrypoint
from utils.rescene_rootcause_preflight import (
    RootCauseContractError,
    build_tensor_state_manifest,
    validate_common_tensor_state,
)


def _state() -> dict[str, torch.Tensor]:
    return {
        "model.backbone.encoder.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "model.class_embed_head.bias": torch.tensor([0.25, -0.5]),
        "criterion.empty_weight": torch.tensor([1.0, 0.2]),
    }


def test_tensor_state_manifest_binds_schema_content_and_trainability() -> None:
    state = _state()
    manifest = build_tensor_state_manifest(
        state,
        trainable_names={"model.class_embed_head.bias"},
    )

    assert manifest["tensor_count"] == 3
    assert manifest["total_elements"] == 10
    assert len(manifest["schema_sha256"]) == 64
    assert len(manifest["content_sha256"]) == 64
    assert len(manifest["trainable_schema_sha256"]) == 64
    assert [entry["name"] for entry in manifest["tensors"]] == sorted(state)
    assert next(
        entry
        for entry in manifest["tensors"]
        if entry["name"] == "model.class_embed_head.bias"
    )["trainable"] is True


def test_content_change_preserves_schema_but_changes_content_hash() -> None:
    left = _state()
    right = copy.deepcopy(left)
    right["model.class_embed_head.bias"][0] += 1

    left_manifest = build_tensor_state_manifest(left)
    right_manifest = build_tensor_state_manifest(right)

    assert left_manifest["schema_sha256"] == right_manifest["schema_sha256"]
    assert left_manifest["content_sha256"] != right_manifest["content_sha256"]


def test_common_state_requires_exact_shared_tensor_bytes() -> None:
    expected = _state()
    observed = copy.deepcopy(expected)

    result = validate_common_tensor_state(observed, expected)
    assert result["shared_tensor_count"] == len(expected)
    observed["model.class_embed_head.bias"][1] += 1
    with pytest.raises(RootCauseContractError, match="tensor content"):
        validate_common_tensor_state(observed, expected)


def test_strong_variant_allows_only_registered_new_parameters() -> None:
    expected = _state()
    observed = copy.deepcopy(expected)
    observed["model.np_feature_projection.0.weight"] = torch.ones(2, 2)

    result = validate_common_tensor_state(
        observed,
        expected,
        allowed_new_prefixes=("model.np_feature_projection.",),
    )
    assert result["new_tensor_names"] == ["model.np_feature_projection.0.weight"]
    with pytest.raises(RootCauseContractError, match="unexpected"):
        validate_common_tensor_state(observed, expected)


def test_missing_common_tensor_is_never_allowed() -> None:
    expected = _state()
    observed = copy.deepcopy(expected)
    observed.pop("model.class_embed_head.bias")

    with pytest.raises(RootCauseContractError, match="missing"):
        validate_common_tensor_state(
            observed,
            expected,
            allowed_new_prefixes=("model.np_feature_projection.",),
        )


def test_entrypoint_loads_common_state_after_pretrained_backbone(monkeypatch) -> None:
    events: list[str] = []

    class _System:
        pass

    system = _System()
    config = OmegaConf.create(
        {
            "general": {
                "seed": 45,
                "gpus": 1,
                "save_dir": "saved/test-rootcause-common",
                "backbone_checkpoint": "pretrained.ckpt",
                "checkpoint": None,
                "rootcause_common_initialization": "common.pt",
                "rootcause_common_initialization_sha256": "a" * 64,
            }
        }
    )

    def instantiate(cfg):
        events.append("instantiate")
        return system

    def load_pretrained(cfg, model):
        assert model is system
        events.append("pretrained")
        return cfg, model

    def load_common(model, path, *, expected_sha256, allowed_new_prefixes):
        assert model is system
        assert path == "common.pt"
        assert expected_sha256 == "a" * 64
        assert allowed_new_prefixes == ()
        events.append("common")
        return {"status": "pass"}

    monkeypatch.setattr(entrypoint, "InstanceSegmentation", instantiate)
    monkeypatch.setattr(
        entrypoint,
        "load_backbone_checkpoint_with_missing_or_exsessive_keys",
        load_pretrained,
    )
    monkeypatch.setattr(entrypoint, "load_common_initialization", load_common)
    monkeypatch.setattr(entrypoint.rank_zero_only, "rank", 1)

    _, observed = entrypoint.get_parameters(config)

    assert observed is system
    assert events == ["instantiate", "pretrained", "common"]


@pytest.mark.parametrize(
    ("common_state", "common_sha256"),
    [(None, None), ("common.pt", None), (None, "a" * 64)],
)
def test_rootcause_entrypoint_requires_complete_common_initialization_binding(
    monkeypatch,
    common_state: str | None,
    common_sha256: str | None,
) -> None:
    config = OmegaConf.create(
        {
            "general": {
                "seed": 45,
                "gpus": 1,
                "save_dir": "saved/test-rootcause-common",
                "backbone_checkpoint": None,
                "checkpoint": None,
                "rootcause_fail_closed_runtime": True,
                "rootcause_common_initialization": common_state,
                "rootcause_common_initialization_sha256": common_sha256,
            }
        }
    )
    monkeypatch.setattr(entrypoint.rank_zero_only, "rank", 1)
    monkeypatch.setattr(
        entrypoint,
        "InstanceSegmentation",
        lambda _cfg: pytest.fail("model must not instantiate without common state"),
    )

    with pytest.raises(RuntimeError, match="common initialization"):
        entrypoint.get_parameters(config)


@pytest.mark.parametrize(
    ("model_config", "expected_prefixes"),
    [
        (
            {"use_np_features": True, "scatter_type": "mean"},
            ("model.np_feature_projection.",),
        ),
        (
            {"use_np_features": False, "scatter_type": "adaptive"},
            ("model.scatter_fn.",),
        ),
        (
            {"use_np_features": True, "scatter_type": "adaptive"},
            ("model.np_feature_projection.", "model.scatter_fn."),
        ),
        ({"use_np_features": False, "scatter_type": "mean"}, ()),
    ],
)
def test_entrypoint_allows_only_configured_strong_variant_namespaces(
    monkeypatch,
    model_config: dict[str, object],
    expected_prefixes: tuple[str, ...],
) -> None:
    class _System:
        pass

    system = _System()
    config = OmegaConf.create(
        {
            "general": {
                "seed": 45,
                "gpus": 1,
                "save_dir": "saved/test-strong-local-common",
                "backbone_checkpoint": None,
                "checkpoint": None,
                "rootcause_fail_closed_runtime": True,
                "rootcause_common_initialization": "common.pt",
                "rootcause_common_initialization_sha256": "a" * 64,
            },
            "model": model_config,
        }
    )
    observed_prefixes: list[tuple[str, ...]] = []

    def load_common(
        model,
        path,
        *,
        expected_sha256,
        allowed_new_prefixes,
    ):
        assert model is system
        assert path == "common.pt"
        assert expected_sha256 == "a" * 64
        observed_prefixes.append(allowed_new_prefixes)
        return {"status": "pass"}

    monkeypatch.setattr(entrypoint, "InstanceSegmentation", lambda _cfg: system)
    monkeypatch.setattr(entrypoint, "load_common_initialization", load_common)
    monkeypatch.setattr(entrypoint.rank_zero_only, "rank", 1)

    entrypoint.get_parameters(config)

    assert observed_prefixes == [expected_prefixes]
