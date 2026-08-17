from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from datasets.streaming_sequence import causal_windows
from models.persistent_memory import (
    PersistentMemory,
    PersistentMemoryState,
)
from models.streaming_rescene import StreamingReScene

_PERSISTENT_KEYS = {
    "persistent_slot_ids",
    "persistent_association_scores",
    "persistent_rejected_births",
}
_REPO_ROOT = Path(__file__).resolve().parents[1]
_P5_ARTIFACT = _REPO_ROOT / "artifacts" / "P5" / "persist4d_mvp_eval.json"
_P5_REPORT = _REPO_ROOT / "artifacts" / "P5" / "persist4d_mvp_eval.md"
_VERIFY_REAL_GPU = os.environ.get("P5_VERIFY_GPU_ARTIFACTS") == "1"
_P5_ROOT_KEYS = {
    "schema_version",
    "status",
    "method",
    "source_commit",
    "source_tree_contract",
    "checkpoint",
    "settings",
    "legacy_parity",
    "horizons",
    "bounded_state",
    "conclusion",
    "errors",
}
_P5_HORIZON_KEYS = {
    "T",
    "loaded_sequences",
    "persistent",
    "internal_baseline",
    "delta",
    "resources",
}
_P5_METRIC_KEYS = {
    "t_mAP",
    "t_REC",
    "per_stage_AP",
    "matched_identity_observations",
    "identity_switches",
    "reactivation_events",
    "correct_reactivations",
    "reactivation_accuracy",
}
_P5_PERSISTENT_METRIC_KEYS = _P5_METRIC_KEYS | {
    "rejected_births",
}
_P5_RESOURCE_KEYS = {
    "peak_allocated_cuda_bytes",
    "mean_latency_ms",
    "throughput_sequences_per_second",
    "serialized_state_bytes",
}
_UUID_PATTERN = re.compile(
    r"\b[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\b"
)


def _load_p5_artifact() -> dict[str, object]:
    assert _P5_ARTIFACT.is_file(), (
        "missing real Persist4D artifact; run scripts/evaluate_persist4d.py "
        "on GPU0"
    )
    payload = json.loads(_P5_ARTIFACT.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_causal_windows_preserve_order_and_emit_bootstrap_window() -> None:
    assert causal_windows([4, 2, 9, 7]) == (
        (4,),
        (4, 2),
        (2, 9),
        (9, 7),
    )


@pytest.mark.parametrize("scan_indices", [[], [1]])
def test_causal_windows_require_at_least_two_indices(scan_indices) -> None:
    with pytest.raises(ValueError, match="scan_indices"):
        causal_windows(scan_indices)


@pytest.mark.parametrize(
    "scan_indices",
    [[1.0, 2], ["1", 2], [True, 2], [np.bool_(False), 2]],
)
def test_causal_windows_reject_non_integral_and_boolean_indices(
    scan_indices,
) -> None:
    with pytest.raises(ValueError, match="scan_indices"):
        causal_windows(scan_indices)


def test_causal_windows_reject_duplicate_indices() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        causal_windows([3, 1, 3])


def _settings() -> dict[str, object]:
    return {
        "background_class": 2,
        "confidence_threshold": 0.5,
        "mask_threshold": 0.5,
        "minimum_mask_support": 1,
    }


def _base_output(batch_size: int = 1) -> dict[str, object]:
    query_features = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0]]]
    ).repeat(batch_size, 1, 1)
    pred_logits = torch.tensor(
        [[[4.0, 0.0, -4.0], [0.0, 4.0, -4.0]]]
    ).repeat(batch_size, 1, 1)
    return {
        "pred_logits": pred_logits,
        "pred_changes": object(),
        "pred_masks": [
            torch.tensor([[10.0, 10.0], [10.0, 10.0]])
            for _ in range(batch_size)
        ],
        "aux_outputs": [object()],
        "sampled_coords": object(),
        "backbone_features": object(),
        "segment_features": [object()],
        "query_features": query_features,
    }


class _FakeReScene(nn.Module):
    def __init__(self, output: Mapping[str, object] | None = None) -> None:
        super().__init__()
        self.return_query_features = True
        self.output = dict(_base_output() if output is None else output)
        self.call_count = 0
        self.calls: list[tuple[object, object, object, bool]] = []

    def forward(
        self,
        x: object,
        point2segment: object = None,
        raw_coordinates: object = None,
        is_eval: bool = False,
    ) -> dict[str, object]:
        self.call_count += 1
        self.calls.append((x, point2segment, raw_coordinates, is_eval))
        return self.output


def _empty_state(
    *,
    batch_size: int = 1,
    capacity: int = 3,
    stage_watermark: int = -1,
) -> PersistentMemoryState:
    state = PersistentMemoryState.empty(
        batch_size=batch_size,
        capacity=capacity,
        feature_dim=2,
        class_count=3,
        device="cpu",
        dtype=torch.float32,
    )
    return replace(
        state,
        stage_watermark=torch.full(
            (batch_size,),
            stage_watermark,
            dtype=torch.long,
        ),
    )


class _IndexableStage:
    def __index__(self) -> int:
        return 6


def _forward_step(
    wrapper: StreamingReScene,
    *,
    state: PersistentMemoryState | None = None,
    stage_index: object = 1,
    segment_stages: list[torch.Tensor] | object | None = None,
    point2segment: object | None = None,
) -> tuple[dict[str, object], PersistentMemoryState]:
    if segment_stages is None:
        segment_stages = [torch.tensor([0, 1])]
    if point2segment is None:
        batch_size = (
            len(segment_stages)
            if isinstance(segment_stages, list) and segment_stages
            else 1
        )
        point2segment = [torch.tensor([0, 1]) for _ in range(batch_size)]
    return wrapper.forward_step(
        x=object(),
        point2segment=point2segment,
        raw_coordinates=None,
        segment_stages=segment_stages,
        state=state,
        stage_index=stage_index,
        is_eval=True,
    )


def test_init_registers_modules_and_requires_query_export() -> None:
    base_model = _FakeReScene()
    memory = PersistentMemory(capacity=3)

    wrapper = StreamingReScene(base_model, memory, _settings())

    assert wrapper.base_model is base_model
    assert wrapper.memory is memory
    assert dict(wrapper.named_children()) == {
        "base_model": base_model,
        "memory": memory,
    }

    base_model.return_query_features = False
    with pytest.raises(ValueError, match="return_query_features"):
        StreamingReScene(base_model, memory, _settings())


def test_init_rejects_invalid_module_types() -> None:
    class _NotAModule:
        return_query_features = True

    with pytest.raises(ValueError, match="nn.Module"):
        StreamingReScene(
            _NotAModule(),
            PersistentMemory(),
            _settings(),
        )

    with pytest.raises(ValueError, match="PersistentMemory"):
        StreamingReScene(_FakeReScene(), nn.Identity(), _settings())


@pytest.mark.parametrize("missing_key", sorted(_settings()))
def test_init_rejects_missing_observation_setting(missing_key: str) -> None:
    settings = _settings()
    del settings[missing_key]
    base_model = _FakeReScene()

    with pytest.raises(ValueError, match="exactly"):
        StreamingReScene(base_model, PersistentMemory(), settings)

    assert base_model.call_count == 0


def test_init_rejects_unknown_observation_setting() -> None:
    settings = _settings()
    settings["unknown"] = 1
    base_model = _FakeReScene()

    with pytest.raises(ValueError, match="exactly"):
        StreamingReScene(base_model, PersistentMemory(), settings)

    assert base_model.call_count == 0


@pytest.mark.parametrize(
    ("setting_name", "invalid_value"),
    [
        ("background_class", -1),
        ("background_class", 1.0),
        ("background_class", True),
        ("confidence_threshold", -0.01),
        ("confidence_threshold", 1.01),
        ("confidence_threshold", float("nan")),
        ("confidence_threshold", float("inf")),
        ("confidence_threshold", True),
        ("mask_threshold", -0.01),
        ("mask_threshold", 1.01),
        ("mask_threshold", float("nan")),
        ("mask_threshold", float("inf")),
        ("mask_threshold", False),
        ("minimum_mask_support", 0),
        ("minimum_mask_support", -1),
        ("minimum_mask_support", 1.0),
        ("minimum_mask_support", True),
    ],
)
def test_init_rejects_invalid_observation_setting_values(
    setting_name: str,
    invalid_value: object,
) -> None:
    settings = _settings()
    settings[setting_name] = invalid_value
    base_model = _FakeReScene()

    with pytest.raises(ValueError, match=setting_name):
        StreamingReScene(base_model, PersistentMemory(), settings)

    assert base_model.call_count == 0


def test_forward_step_checks_background_class_upper_bound_after_inference() -> None:
    settings = _settings()
    settings["background_class"] = 3
    base_model = _FakeReScene()
    wrapper = StreamingReScene(base_model, PersistentMemory(), settings)

    with pytest.raises(ValueError, match="background_class"):
        _forward_step(wrapper)

    assert base_model.call_count == 1


def test_init_defensively_copies_observation_settings() -> None:
    settings = _settings()
    expected = settings.copy()
    wrapper = StreamingReScene(
        _FakeReScene(),
        PersistentMemory(capacity=3),
        settings,
    )

    settings.clear()
    result, _ = _forward_step(wrapper)

    assert wrapper.observation_settings == expected
    assert wrapper.observation_settings is not settings
    assert _PERSISTENT_KEYS <= result.keys()


def test_forward_step_preserves_predictions_and_passes_base_arguments() -> None:
    base_model = _FakeReScene()
    original_keys = tuple(base_model.output)
    original_value_ids = {
        key: id(value) for key, value in base_model.output.items()
    }
    original_pred_logits = base_model.output["pred_logits"].clone()
    original_query_features = base_model.output["query_features"].clone()
    original_pred_masks = [
        mask.clone() for mask in base_model.output["pred_masks"]
    ]
    wrapper = StreamingReScene(
        base_model,
        PersistentMemory(capacity=3),
        _settings(),
    )
    x = object()
    point2segment = [torch.tensor([0, 1])]
    raw_coordinates = object()

    result, state = wrapper.forward_step(
        x=x,
        point2segment=point2segment,
        raw_coordinates=raw_coordinates,
        segment_stages=[torch.tensor([0, 1])],
        state=None,
        stage_index=7,
        is_eval=False,
    )

    assert base_model.calls == [(x, point2segment, raw_coordinates, False)]
    assert result is not base_model.output
    assert set(result) == set(original_keys) | _PERSISTENT_KEYS
    for key, value in base_model.output.items():
        assert result[key] is value
        assert id(value) == original_value_ids[key]
    torch.testing.assert_close(
        base_model.output["pred_logits"],
        original_pred_logits,
    )
    torch.testing.assert_close(
        base_model.output["query_features"],
        original_query_features,
    )
    for actual, expected in zip(
        base_model.output["pred_masks"],
        original_pred_masks,
        strict=True,
    ):
        torch.testing.assert_close(actual, expected)
    assert tuple(base_model.output) == original_keys
    assert _PERSISTENT_KEYS.isdisjoint(base_model.output)
    assert torch.equal(result["persistent_slot_ids"], torch.tensor([[0, 1]]))
    assert torch.isneginf(result["persistent_association_scores"]).all()
    assert not torch.any(result["persistent_rejected_births"])
    assert torch.equal(
        state.occupied,
        torch.tensor([[True, True, False]]),
    )
    assert torch.equal(state.stage_watermark, torch.tensor([7]))
    assert not hasattr(wrapper, "state")
    assert not hasattr(wrapper, "outputs")
    assert not hasattr(wrapper, "observation")


def test_forward_step_exposes_rejected_births() -> None:
    wrapper = StreamingReScene(
        _FakeReScene(),
        PersistentMemory(capacity=1),
        _settings(),
    )

    result, _ = _forward_step(wrapper)

    assert torch.equal(
        result["persistent_rejected_births"],
        torch.tensor([[False, True]]),
    )


def test_forward_step_passes_state_explicitly_across_steps() -> None:
    wrapper = StreamingReScene(
        _FakeReScene(),
        PersistentMemory(capacity=3),
        _settings(),
    )
    first_result, first_state = _forward_step(wrapper, stage_index=4)

    second_result, second_state = _forward_step(
        wrapper,
        state=first_state,
        stage_index=5,
        segment_stages=[torch.tensor([1, 2])],
    )

    assert torch.equal(
        second_result["persistent_slot_ids"],
        first_result["persistent_slot_ids"],
    )
    assert second_state is not first_state
    assert torch.equal(first_state.stage_watermark, torch.tensor([4]))
    assert torch.equal(second_state.stage_watermark, torch.tensor([5]))
    assert torch.equal(first_state.age, torch.zeros_like(first_state.age))
    assert torch.equal(
        second_state.age,
        torch.tensor([[1, 1, 0]]),
    )


def test_forward_step_state_none_resets_memory() -> None:
    wrapper = StreamingReScene(
        _FakeReScene(),
        PersistentMemory(capacity=3),
        _settings(),
    )

    first_result, first_state = _forward_step(wrapper, stage_index=3)
    reset_result, reset_state = _forward_step(wrapper, stage_index=3)

    assert torch.equal(
        reset_result["persistent_slot_ids"],
        first_result["persistent_slot_ids"],
    )
    assert reset_state is not first_state
    assert torch.equal(reset_state.age, torch.zeros_like(reset_state.age))
    assert not hasattr(wrapper, "state")


def test_forward_step_rejects_state_batch_mismatch() -> None:
    base_model = _FakeReScene(_base_output(batch_size=2))
    wrapper = StreamingReScene(
        base_model,
        PersistentMemory(capacity=3),
        _settings(),
    )
    one_item_state = _empty_state(batch_size=1)

    with pytest.raises(ValueError, match="batch"):
        _forward_step(
            wrapper,
            state=one_item_state,
            stage_index=2,
            segment_stages=[torch.tensor([1, 2]), torch.tensor([0, 2])],
        )

    assert base_model.call_count == 0


@pytest.mark.parametrize("stage_index", [3, 4])
def test_forward_step_rejects_non_increasing_global_stage(
    stage_index: int,
) -> None:
    base_model = _FakeReScene()
    wrapper = StreamingReScene(
        base_model,
        PersistentMemory(capacity=3),
        _settings(),
    )
    state = _empty_state(stage_watermark=4)

    with pytest.raises(ValueError, match="later than"):
        _forward_step(wrapper, state=state, stage_index=stage_index)

    assert base_model.call_count == 0


@pytest.mark.parametrize(
    "stage_index",
    [True, -1, 1.0, None, torch.iinfo(torch.long).max + 1],
)
def test_forward_step_rejects_invalid_global_stage_before_inference(
    stage_index: object,
) -> None:
    base_model = _FakeReScene()
    wrapper = StreamingReScene(
        base_model,
        PersistentMemory(capacity=3),
        _settings(),
    )

    with pytest.raises(ValueError, match="stage_index"):
        _forward_step(wrapper, stage_index=stage_index)

    assert base_model.call_count == 0


def test_forward_step_accepts_an_indexable_global_stage() -> None:
    base_model = _FakeReScene()
    wrapper = StreamingReScene(
        base_model,
        PersistentMemory(capacity=3),
        _settings(),
    )

    _, state = _forward_step(wrapper, stage_index=_IndexableStage())

    assert base_model.call_count == 1
    assert torch.equal(state.stage_watermark, torch.tensor([6]))


def test_forward_step_validates_state_before_inference() -> None:
    base_model = _FakeReScene()
    wrapper = StreamingReScene(
        base_model,
        PersistentMemory(capacity=3),
        _settings(),
    )
    invalid_state = replace(
        _empty_state(),
        stage_watermark=torch.tensor([-2]),
    )

    with pytest.raises(ValueError, match="stage_watermark"):
        _forward_step(wrapper, state=invalid_state)

    assert base_model.call_count == 0


def test_forward_step_rejects_invalid_state_type_before_inference() -> None:
    base_model = _FakeReScene()
    wrapper = StreamingReScene(
        base_model,
        PersistentMemory(capacity=3),
        _settings(),
    )

    with pytest.raises(ValueError, match="PersistentMemoryState"):
        _forward_step(wrapper, state=object())

    assert base_model.call_count == 0


def test_forward_step_rejects_state_capacity_mismatch_before_inference() -> None:
    base_model = _FakeReScene()
    wrapper = StreamingReScene(
        base_model,
        PersistentMemory(capacity=3),
        _settings(),
    )

    with pytest.raises(ValueError, match="capacity"):
        _forward_step(wrapper, state=_empty_state(capacity=2))

    assert base_model.call_count == 0


@pytest.mark.parametrize(
    "point2segment",
    [[], [torch.tensor([0]), torch.tensor([0])], object()],
)
def test_forward_step_rejects_point_batch_mismatch_before_inference(
    point2segment: object,
) -> None:
    base_model = _FakeReScene()
    wrapper = StreamingReScene(
        base_model,
        PersistentMemory(capacity=3),
        _settings(),
    )

    with pytest.raises(ValueError, match="point2segment"):
        _forward_step(wrapper, point2segment=point2segment)

    assert base_model.call_count == 0


def test_forward_step_rejects_missing_query_features() -> None:
    output = _base_output()
    del output["query_features"]
    wrapper = StreamingReScene(
        _FakeReScene(output),
        PersistentMemory(capacity=3),
        _settings(),
    )

    with pytest.raises(ValueError, match="query_features"):
        _forward_step(wrapper)


@pytest.mark.parametrize("output_key", ["query_features", "pred_logits", "pred_masks"])
def test_forward_step_rejects_non_finite_base_outputs(output_key: str) -> None:
    output = _base_output()
    if output_key == "pred_masks":
        output[output_key][0][0, 0] = float("inf")
    else:
        output[output_key][0, 0, 0] = float("nan")
    wrapper = StreamingReScene(
        _FakeReScene(output),
        PersistentMemory(capacity=3),
        _settings(),
    )

    with pytest.raises(ValueError, match="finite"):
        _forward_step(wrapper)


@pytest.mark.parametrize(
    "segment_stages",
    [
        (),
        [],
        [object()],
        [torch.tensor([])],
        [torch.tensor([[0, 1]])],
        [torch.tensor([0.0, 1.0])],
        [torch.tensor([False, True])],
        [torch.tensor([-1, 0])],
        [torch.tensor([0.0, float("inf")])],
    ],
)
def test_forward_step_rejects_invalid_local_stage_collections(
    segment_stages: object,
) -> None:
    base_model = _FakeReScene()
    wrapper = StreamingReScene(
        base_model,
        PersistentMemory(capacity=3),
        _settings(),
    )

    with pytest.raises(ValueError, match="segment_stages"):
        _forward_step(wrapper, segment_stages=segment_stages)

    assert base_model.call_count == 0


def test_forward_step_rejects_mixed_latest_local_stages() -> None:
    base_model = _FakeReScene(_base_output(batch_size=2))
    wrapper = StreamingReScene(
        base_model,
        PersistentMemory(capacity=3),
        _settings(),
    )

    with pytest.raises(ValueError, match="latest local stage"):
        _forward_step(
            wrapper,
            segment_stages=[torch.tensor([0, 1]), torch.tensor([0, 2])],
        )

    assert base_model.call_count == 0


def test_forward_step_uses_maximum_as_latest_local_stage() -> None:
    output = _base_output()
    output["pred_masks"] = [
        torch.tensor(
            [
                [-10.0, -10.0],
                [10.0, 10.0],
                [-10.0, -10.0],
            ]
        )
    ]
    wrapper = StreamingReScene(
        _FakeReScene(output),
        PersistentMemory(capacity=3),
        _settings(),
    )

    result, _ = _forward_step(
        wrapper,
        segment_stages=[torch.tensor([0, 2, 1])],
    )

    assert torch.equal(result["persistent_slot_ids"], torch.tensor([[0, 1]]))


@pytest.mark.skipif(
    not _VERIFY_REAL_GPU,
    reason="set P5_VERIFY_GPU_ARTIFACTS=1 after the real GPU0 evaluation",
)
def test_real_persist4d_artifact_passes_bounded_gpu_gate() -> None:
    from scripts.evaluate_persist4d import (
        _build_parser,
        _derive_conclusion,
        _validate_complete_artifact,
    )

    artifact = _load_p5_artifact()
    assert set(artifact) == _P5_ROOT_KEYS
    assert artifact["schema_version"] == 2
    assert artifact["status"] == "pass"
    assert artifact["method"] == "persist4d_p5_single_memory"
    assert artifact["errors"] == []
    assert artifact["settings"] == {
        "capacity": 100,
        "local_window": 2,
        "internal_baseline_identity": "local_query_index",
        "shared_rescene_outputs": True,
    }

    source_contract = artifact["source_tree_contract"]
    assert source_contract == {
        "schema_version": 1,
        "status": "pass",
        "source_commit": artifact["source_commit"],
        "tracked_tree_clean": True,
        "index_clean": True,
        "allowed_untracked_outputs": [
            "repo:artifacts/P5/persist4d_mvp_eval.json",
            "repo:artifacts/P5/persist4d_mvp_eval.md",
        ],
        "only_declared_outputs_untracked": True,
        "generation_head_unchanged": True,
    }

    parity = artifact["legacy_parity"]
    assert parity["verified_by"] == "in_evaluator_fixed_t2_sample_toggle"
    assert parity["sample_count"] == 1
    assert parity["legacy_predictions_unchanged"] is True
    assert parity["query_feature_shape"] == [1, 100, 128]

    checkpoint = _REPO_ROOT / "checkpoints" / "rescene4d_concerto_t2_repro.ckpt"
    assert checkpoint.is_file()
    assert not checkpoint.is_symlink()
    assert artifact["checkpoint"] == {
        "ref": "repo:checkpoints/rescene4d_concerto_t2_repro.ckpt",
        "sha256": _sha256(checkpoint),
    }

    horizons = artifact["horizons"]
    assert [horizon["T"] for horizon in horizons] == [2, 3, 4, 5]
    state_sizes = []
    for horizon in horizons:
        assert set(horizon) == _P5_HORIZON_KEYS
        assert horizon["loaded_sequences"] > 0
        assert set(horizon["persistent"]) == _P5_PERSISTENT_METRIC_KEYS
        assert set(horizon["internal_baseline"]) == _P5_METRIC_KEYS
        assert set(horizon["delta"]) == _P5_METRIC_KEYS
        for method in ("persistent", "internal_baseline"):
            metrics = horizon[method]
            assert set(metrics["per_stage_AP"]) == {
                str(stage) for stage in range(1, horizon["T"] + 1)
            }
            assert math.isfinite(metrics["t_mAP"])
            assert math.isfinite(metrics["t_REC"])
        resources = horizon["resources"]
        assert set(resources) == _P5_RESOURCE_KEYS
        assert math.isfinite(resources["mean_latency_ms"])
        assert math.isfinite(resources["throughput_sequences_per_second"])
        assert resources["mean_latency_ms"] > 0.0
        assert resources["throughput_sequences_per_second"] > 0.0
        assert resources["peak_allocated_cuda_bytes"] > 0
        assert resources["serialized_state_bytes"] > 0
        state_sizes.append(resources["serialized_state_bytes"])

    bounded = artifact["bounded_state"]
    assert bounded["constant_shape"] is True
    assert bounded["maximum_state_bytes"] == max(state_sizes)
    assert artifact["conclusion"] == _derive_conclusion(horizons, bounded)
    assert artifact["conclusion"]["label"] in {
        "P5_MVP_PASS",
        "P5_ASSOCIATION_DIAGNOSIS",
    }

    args = _build_parser().parse_args(
        [
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(_P5_ARTIFACT),
            "--markdown",
            str(_P5_REPORT),
            "--horizons",
            "2",
            "3",
            "4",
            "5",
            "--device",
            "cuda:0",
        ]
    )
    _validate_complete_artifact(artifact, args=args)


@pytest.mark.skipif(
    not _VERIFY_REAL_GPU,
    reason="set P5_VERIFY_GPU_ARTIFACTS=1 after the real GPU0 evaluation",
)
def test_real_persist4d_markdown_matches_json_measurements() -> None:
    from scripts.evaluate_persist4d import _render_markdown

    artifact = _load_p5_artifact()
    assert _P5_REPORT.is_file()
    report = _P5_REPORT.read_text(encoding="utf-8")
    assert report == _render_markdown(artifact)
    assert "Persist4D MVP" in report
    assert "not an official AP target" in report
    assert "Legacy predictions unchanged: `true`" in report
    assert "fixed-capacity state" in report
    assert "internal no-memory baseline" in report
    assert (
        f"Conclusion: `{artifact['conclusion']['label']}`"
        in report
    )
    assert f"Reason: `{artifact['conclusion']['reason']}`" in report
    assert (
        f"Maximum serialized state bytes: "
        f"`{artifact['bounded_state']['maximum_state_bytes']}`"
    ) in report


@pytest.mark.skipif(
    not _VERIFY_REAL_GPU,
    reason="set P5_VERIFY_GPU_ARTIFACTS=1 after the real GPU0 evaluation",
)
def test_real_persist4d_artifacts_are_portable_and_private() -> None:
    artifact = _load_p5_artifact()
    assert _P5_REPORT.is_file()
    serialized = _P5_ARTIFACT.read_text(encoding="utf-8") + _P5_REPORT.read_text(
        encoding="utf-8"
    )
    serialized.encode("ascii")

    assert artifact["checkpoint"]["ref"].startswith("repo:")
    assert not _UUID_PATTERN.search(serialized)
    assert not re.search(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", serialized)
    for private_marker in (
        "/home/",
        "/Users/",
        "/mnt/",
        "\\Users\\",
        "CUDA_VISIBLE_DEVICES",
        "CONCERTO_CHECKPOINT",
    ):
        assert private_marker not in serialized
