import json
import os
import signal
import time
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import ClassVar

import pytest
import pytorch_lightning as pl
import torch
from pytorch_lightning.strategies import DDPStrategy
from torch.multiprocessing.spawn import ProcessRaisedException
from torch.utils.data import DataLoader

import models.criterion as criterion_module
import trainer.trainer as trainer_module
from models.criterion import ContrastiveLoss, SetCriterion
from trainer.trainer import (
    InstanceSegmentation,
    _batch_collective_device,
    aggregate_objective_loss,
)

SINGLE_POINT_ERROR = "only a single point gives nans in cross-attention"


@pytest.fixture(scope="module", autouse=True)
def _force_cpu_only_lightning():
    previous = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = previous


def _batch(*, empty_target: bool = False, empty_points: bool = False):
    data = SimpleNamespace(
        features=torch.zeros(1, 1),
        coordinates=(
            torch.empty(0, 1) if empty_points else torch.ones(1, 1)
        ),
        batch_size=1,
        inverse_maps=[],
        target_full=[],
        original_colors=[],
        idx=[],
        original_normals=[],
        original_coordinates=[],
    )
    target = (
        []
        if empty_target
        else [
            {
                "labels": torch.tensor([0]),
                "point2segment": torch.tensor([0]),
            }
        ]
    )
    return data, target, ["scene-001"]


def _collate_valid_batch(items):
    return _batch()


def _model_output(objective):
    def prediction_layer():
        return {
            "pred_logits": torch.ones(1, 1, 2, device=objective.device),
            "pred_masks": [torch.ones(1, 1, device=objective.device)],
            "pred_changes": None,
        }

    return {
        **prediction_layer(),
        "aux_outputs": [prediction_layer()],
        "segment_features": [
            [torch.ones(1, 1, device=objective.device)]
        ],
        "objective": objective,
    }


class _SyntheticCriterion:
    weight_dict: ClassVar[dict] = {}
    losses: ClassVar[tuple[str, ...]] = ("labels", "masks")

    def __init__(self, owner=None):
        self.owner = owner
        self.use_contrastive_loss = (
            owner is not None
            and owner.ddp_failure
            in {"missing_segment_features", "segment_feature_layer_drift"}
        )

    def __call__(self, output, target, mask_type):
        objective = output["objective"]
        if (
            self.owner is not None
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            num_masks = objective.detach().new_ones(1)
            torch.distributed.all_reduce(num_masks)
        if (
            self.owner is not None
            and self.owner.ddp_failure == "criterion_error"
            and self.owner.global_rank == 0
            and self.owner._current_batch_idx == 1
        ):
            raise ValueError("rank-local criterion failure")
        if (
            self.owner is not None
            and self.owner.ddp_failure == "nonfinite_loss"
            and self.owner.global_rank == 0
            and self.owner._current_batch_idx == 1
        ):
            return {"loss_mock": objective * torch.tensor(float("nan"))}
        if (
            self.owner is not None
            and self.owner.global_rank == 0
            and self.owner._current_batch_idx == 1
            and self.owner.ddp_failure == "empty_objective"
        ):
            return {"loss_mock": objective.new_empty(0)}
        if (
            self.owner is not None
            and self.owner.global_rank == 0
            and self.owner._current_batch_idx == 1
            and self.owner.ddp_failure == "vector_objective"
        ):
            return {"loss_mock": objective.repeat(2)}
        return {"loss_mock": objective}


def _step_owner(
    *,
    forward,
    p2_fail_closed_runtime=True,
    p2_weighted_objective=True,
):
    owner = SimpleNamespace(
        model=SimpleNamespace(num_levels=1, num_decoders=1),
        config=SimpleNamespace(
            general=SimpleNamespace(
                max_batch_size=10,
                use_dbscan=False,
                p2_fail_closed_runtime=p2_fail_closed_runtime,
                p2_weighted_objective=p2_weighted_objective,
            )
        ),
        forward=forward,
        _process_raw_coordinates=lambda data: None,
    )
    owner._eval_step = MethodType(InstanceSegmentation._eval_step, owner)
    return owner


def _objective_owner(*, criterion):
    owner = _step_owner(
        forward=lambda *args, **kwargs: _model_output(torch.tensor(1.0))
    )
    owner.mask_type = "segment_mask"
    owner.criterion = criterion
    owner._get_mean_loss = lambda losses, prefix: {}
    owner.log_dict = lambda *args, **kwargs: None
    owner._process_predictions = lambda **kwargs: []
    owner.instance_metric = SimpleNamespace(update=lambda *args, **kwargs: None)
    owner.aux_metric = None
    return owner


def _assert_batch_context(error, *, stage: str, batch_idx: int, reason: str):
    message = str(error.value)
    assert "Batch contract violation" in message
    assert f"stage={stage}" in message
    assert f"batch_idx={batch_idx}" in message
    assert "file_names=['scene-001']" in message
    assert f"reason={reason}" in message


def test_batch_collective_uses_module_device_before_cpu_collator_tensors():
    module = SimpleNamespace(device=torch.device("cuda:1"))
    data, _, _ = _batch()

    assert _batch_collective_device(module, data) == torch.device("cuda:1")


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_objective_rejects_nonfinite_raw_terms(bad_value):
    with pytest.raises(ValueError, match="non-finite raw objective term 'loss_mock'"):
        aggregate_objective_loss(
            {"loss_mock": torch.tensor(bad_value)},
            {},
        )


def test_objective_rejects_nonfinite_aggregate_from_finite_raw_terms():
    largest = torch.tensor(torch.finfo(torch.float32).max)

    with pytest.raises(ValueError, match="non-finite aggregate objective"):
        aggregate_objective_loss(
            {"loss_a": largest, "loss_b": largest},
            {},
        )


@pytest.mark.parametrize(
    ("value", "reason"),
    [
        (torch.empty(0), "empty raw objective term 'loss_mock'"),
        (torch.ones(1), "non-scalar raw objective term 'loss_mock'"),
        (torch.ones(2), "non-scalar raw objective term 'loss_mock'"),
    ],
)
def test_objective_rejects_empty_and_vector_raw_terms(value, reason):
    with pytest.raises(ValueError, match=reason):
        aggregate_objective_loss({"loss_mock": value}, {})


def test_objective_rejects_non_scalar_aggregate():
    with pytest.raises(ValueError, match="non-scalar aggregate objective"):
        aggregate_objective_loss(
            {"loss_mock": torch.tensor(1.0)},
            {"loss_mock": torch.ones(2)},
        )


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_contrastive_loss_rejects_nonfinite_output(monkeypatch, bad_value):
    loss_module = ContrastiveLoss(loss_type="infonce", use_chunked_loss=True)
    loss_module.p2_fail_closed_runtime = True
    monkeypatch.setattr(
        criterion_module,
        "infoNCE_chunked_loss",
        lambda **kwargs: torch.tensor(bad_value),
    )

    with pytest.raises(ValueError, match="non-finite contrastive loss"):
        loss_module(torch.ones(2, 2), torch.eye(2))


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_contrastive_loss_rejects_nonfinite_features_before_sanitizing(
    monkeypatch,
    bad_value,
):
    loss_module = ContrastiveLoss(loss_type="infonce", use_chunked_loss=True)
    loss_module.p2_fail_closed_runtime = True
    loss_called = False

    def finite_loss(**kwargs):
        nonlocal loss_called
        loss_called = True
        return torch.tensor(0.0)

    monkeypatch.setattr(criterion_module, "infoNCE_chunked_loss", finite_loss)
    features = torch.tensor([[bad_value, 1.0], [1.0, 0.0]])

    with pytest.raises(ValueError, match="non-finite contrastive features"):
        loss_module(features, torch.eye(2))

    assert loss_called is False


def test_contrastive_runtime_safety_off_preserves_feature_sanitizing(monkeypatch):
    loss_module = ContrastiveLoss(loss_type="infonce", use_chunked_loss=True)
    loss_module.p2_fail_closed_runtime = False
    monkeypatch.setattr(
        criterion_module,
        "infoNCE_chunked_loss",
        lambda **kwargs: torch.tensor(0.0),
    )

    loss = loss_module(
        torch.tensor([[float("nan"), 1.0], [1.0, 0.0]]),
        torch.eye(2),
    )

    assert loss.item() == 0.0


def test_criterion_reduces_num_masks_before_first_matcher(monkeypatch):
    events = []

    class MatcherFailure(RuntimeError):
        pass

    def fail_matcher(outputs, targets, mask_type):
        events.append("matcher")
        raise MatcherFailure("rank-local matcher failure")

    criterion = SimpleNamespace(
        matcher=fail_matcher,
        losses=[],
        use_contrastive_loss=False,
        p2_fail_closed_runtime=True,
    )
    monkeypatch.setattr(
        criterion_module,
        "is_dist_avail_and_initialized",
        lambda: True,
    )
    monkeypatch.setattr(criterion_module, "get_world_size", lambda: 2)
    monkeypatch.setattr(
        torch.distributed,
        "all_reduce",
        lambda tensor: events.append("num_masks_all_reduce"),
    )

    with pytest.raises(MatcherFailure, match="rank-local matcher failure"):
        SetCriterion.forward(
            criterion,
            {"pred_logits": torch.zeros(1, 1, 1)},
            [{"labels": torch.tensor([0])}],
            "segment_mask",
        )

    assert events == ["num_masks_all_reduce", "matcher"]


def test_criterion_uses_pred_logits_device_for_first_p2_collective(monkeypatch):
    events = []

    class MatcherFailure(RuntimeError):
        pass

    def fail_matcher(outputs, targets, mask_type):
        events.append("matcher")
        raise MatcherFailure("matcher reached")

    criterion = SimpleNamespace(
        matcher=fail_matcher,
        losses=[],
        use_contrastive_loss=False,
        p2_fail_closed_runtime=True,
    )
    monkeypatch.setattr(
        criterion_module,
        "is_dist_avail_and_initialized",
        lambda: True,
    )
    monkeypatch.setattr(criterion_module, "get_world_size", lambda: 2)
    monkeypatch.setattr(
        torch.distributed,
        "all_reduce",
        lambda tensor: events.append(
            ("num_masks_all_reduce", tensor.device.type)
        ),
    )

    with pytest.raises(MatcherFailure, match="matcher reached"):
        SetCriterion.forward(
            criterion,
            {
                "metadata": [],
                "pred_logits": torch.zeros(1, 1, 1),
            },
            [{"labels": torch.tensor([0])}],
            "segment_mask",
        )

    assert events == [("num_masks_all_reduce", "cpu"), "matcher"]


def test_runtime_safety_defaults_off_and_preserves_upstream_empty_batch_skip(
    monkeypatch,
):
    reductions = []
    owner = SimpleNamespace(
        config=SimpleNamespace(
            general=SimpleNamespace(max_batch_size=10, use_dbscan=False)
        ),
        forward=lambda *args, **kwargs: pytest.fail("forward called"),
        _process_raw_coordinates=lambda data: None,
    )
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    monkeypatch.setattr(
        torch.distributed,
        "all_reduce",
        lambda tensor, op: reductions.append((tensor, op)),
    )
    monkeypatch.setattr(
        torch.distributed,
        "all_gather_object",
        lambda output, value: output.__setitem__(0, value),
    )

    result = InstanceSegmentation.training_step(
        owner,
        _batch(empty_target=True),
        0,
    )

    assert result is None
    assert reductions == []


@pytest.mark.parametrize(
    "step_method",
    [InstanceSegmentation.training_step, InstanceSegmentation.validation_step],
)
def test_p2_flags_off_preserve_upstream_unweighted_objective(step_method):
    def criterion(*args, **kwargs):
        return {
            "loss_ce": torch.tensor(1.0),
            "loss_segment_contrastive_layer0": torch.tensor(100.0),
        }

    criterion.weight_dict = {"loss_ce": 2.0}
    owner = _objective_owner(criterion=criterion)
    owner.config.general.p2_fail_closed_runtime = False
    owner.config.general.p2_weighted_objective = False

    objective = step_method(owner, _batch(), 0)

    assert objective.item() == 101.0


def test_weighted_objective_does_not_enable_runtime_nonfinite_checks():
    def criterion(*args, **kwargs):
        return {"loss_mock": torch.tensor(float("nan"))}

    criterion.weight_dict = {}
    owner = _objective_owner(criterion=criterion)
    owner.config.general.p2_fail_closed_runtime = False
    owner.config.general.p2_weighted_objective = True

    objective = InstanceSegmentation.training_step(owner, _batch(), 0)

    assert torch.isnan(objective)


@pytest.mark.parametrize(
    "step_method",
    [InstanceSegmentation.training_step, InstanceSegmentation.validation_step],
)
def test_runtime_safety_off_preserves_original_forward_exception(step_method):
    original_error = RuntimeError("rank-local generic forward failure")

    def fail_forward(*args, **kwargs):
        raise original_error

    owner = _step_owner(
        forward=fail_forward,
        p2_fail_closed_runtime=False,
        p2_weighted_objective=False,
    )

    with pytest.raises(RuntimeError) as error:
        step_method(owner, _batch(), 0)

    assert error.value is original_error


def test_eval_runtime_safety_defaults_off_and_preserves_empty_batch_skip(
    monkeypatch,
):
    reductions = []
    owner = SimpleNamespace(
        config=SimpleNamespace(
            general=SimpleNamespace(max_batch_size=10, use_dbscan=False)
        ),
        forward=lambda *args, **kwargs: pytest.fail("forward called"),
        _process_raw_coordinates=lambda data: None,
    )
    owner._eval_step = MethodType(InstanceSegmentation._eval_step, owner)
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(
        torch.distributed,
        "all_reduce",
        lambda tensor, op: reductions.append((tensor, op)),
    )

    result = InstanceSegmentation.validation_step(
        owner,
        _batch(empty_points=True),
        0,
    )

    assert result == 0.0
    assert reductions == []


def test_criterion_runtime_safety_off_preserves_matcher_first(monkeypatch):
    events = []

    def fail_matcher(outputs, targets, mask_type):
        events.append("matcher")
        raise RuntimeError("matcher failure")

    criterion = SimpleNamespace(
        matcher=fail_matcher,
        losses=[],
        use_contrastive_loss=False,
        p2_fail_closed_runtime=False,
    )
    monkeypatch.setattr(
        criterion_module,
        "is_dist_avail_and_initialized",
        lambda: True,
    )
    monkeypatch.setattr(
        torch.distributed,
        "all_reduce",
        lambda tensor: events.append("num_masks_all_reduce"),
    )

    with pytest.raises(RuntimeError, match="matcher failure"):
        SetCriterion.forward(
            criterion,
            {"pred_logits": torch.zeros(1, 1, 1)},
            [{"labels": torch.tensor([0])}],
            "segment_mask",
        )

    assert events == ["matcher"]


def test_contrastive_runtime_safety_off_preserves_nan_fallback(monkeypatch):
    loss_module = ContrastiveLoss(loss_type="infonce", use_chunked_loss=True)
    loss_module.p2_fail_closed_runtime = False
    monkeypatch.setattr(
        criterion_module,
        "infoNCE_chunked_loss",
        lambda **kwargs: torch.tensor(float("nan")),
    )

    loss = loss_module(torch.ones(2, 2), torch.eye(2))

    assert loss.item() == 0.0


@pytest.mark.parametrize("configured_value", [True, False, None])
def test_matcher_setup_injects_reconstructible_runtime_flag(
    monkeypatch,
    configured_value,
):
    matcher_spec = object()
    loss_spec = object()
    matcher = SimpleNamespace(cost_class=1.0, cost_mask=1.0, cost_dice=1.0)
    contrastive_loss = SimpleNamespace()
    criterion = SimpleNamespace(contrastive_loss=contrastive_loss)
    general = SimpleNamespace(ignore_mask_idx=[])
    if configured_value is not None:
        general.p2_fail_closed_runtime = configured_value
    config = SimpleNamespace(
        matcher=matcher_spec,
        loss=loss_spec,
        general=general,
    )
    owner = SimpleNamespace(model=SimpleNamespace(num_levels=1, num_decoders=1))

    def instantiate(spec, **kwargs):
        if spec is matcher_spec:
            return matcher
        assert spec is loss_spec
        assert kwargs["matcher"] is matcher
        return criterion

    monkeypatch.setattr(trainer_module.hydra.utils, "instantiate", instantiate)

    result = InstanceSegmentation._setup_matcher_and_loss(owner, config)

    expected = bool(configured_value)
    assert result.p2_fail_closed_runtime is expected
    assert result.contrastive_loss.p2_fail_closed_runtime is expected


@pytest.mark.parametrize(
    ("step_method", "expected_reductions"),
    [
        (InstanceSegmentation.training_step, 3),
        (InstanceSegmentation.validation_step, 4),
        (InstanceSegmentation.test_step, 3),
    ],
)
def test_normal_p2_step_uses_expected_scalar_consensus_all_reduces(
    monkeypatch,
    step_method,
    expected_reductions,
):
    reductions = []
    owner = _objective_owner(criterion=_SyntheticCriterion())

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(
        torch.distributed,
        "all_reduce",
        lambda tensor, op: reductions.append((tensor.clone(), op)),
    )

    step_method(owner, _batch(), 0)

    assert len(reductions) == expected_reductions
    assert all(tensor.numel() == 1 for tensor, _ in reductions)
    assert all(tensor.dtype == torch.int32 for tensor, _ in reductions)
    assert all(op is torch.distributed.ReduceOp.MAX for _, op in reductions)


def test_training_step_fails_fast_for_empty_targets_with_batch_context():
    owner = _step_owner(forward=lambda *args, **kwargs: pytest.fail("forward called"))

    with pytest.raises(RuntimeError) as error:
        InstanceSegmentation.training_step(owner, _batch(empty_target=True), 7)

    _assert_batch_context(
        error,
        stage="train",
        batch_idx=7,
        reason="empty target list",
    )


@pytest.mark.parametrize("field", ["labels", "point2segment"])
def test_training_preflight_rejects_missing_required_target_fields(field):
    data, target, file_names = _batch()
    target[0].pop(field)
    owner = _objective_owner(criterion=_SyntheticCriterion())

    with pytest.raises(RuntimeError) as error:
        InstanceSegmentation.training_step(
            owner,
            (data, target, file_names),
            14,
        )

    _assert_batch_context(
        error,
        stage="train",
        batch_idx=14,
        reason=f"target[0] missing required field '{field}'",
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("labels", 0),
        ("labels", torch.tensor(0)),
        ("point2segment", []),
    ],
)
def test_training_preflight_rejects_invalid_required_target_tensors(
    field,
    value,
):
    data, target, file_names = _batch()
    target[0][field] = value
    owner = _objective_owner(criterion=_SyntheticCriterion())

    with pytest.raises(RuntimeError) as error:
        InstanceSegmentation.training_step(
            owner,
            (data, target, file_names),
            15,
        )

    _assert_batch_context(
        error,
        stage="train",
        batch_idx=15,
        reason=f"target[0] field '{field}' must be a non-scalar tensor",
    )


def test_test_step_accepts_prediction_only_target_without_labels():
    data, target, file_names = _batch()
    target[0].pop("labels")
    owner = _objective_owner(criterion=_SyntheticCriterion())

    result = InstanceSegmentation.test_step(
        owner,
        (data, target, file_names),
        16,
    )

    assert result == 0.0


class _ExplodingTargetList(list):
    def __iter__(self):
        raise RuntimeError("rank-local preflight failure")


def test_training_synchronizes_input_preflight_exceptions():
    data, target, file_names = _batch()
    owner = _objective_owner(criterion=_SyntheticCriterion())

    with pytest.raises(RuntimeError) as error:
        InstanceSegmentation.training_step(
            owner,
            (data, _ExplodingTargetList(target), file_names),
            16,
        )

    _assert_batch_context(
        error,
        stage="train",
        batch_idx=16,
        reason="input RuntimeError: rank-local preflight failure",
    )


@pytest.mark.parametrize(
    ("output", "reason"),
    [
        (
            {"pred_masks": [torch.ones(1, 1)], "aux_outputs": []},
            "forward output missing required field 'pred_logits'",
        ),
        (
            {
                "pred_logits": torch.empty(0, 1, 2),
                "pred_masks": [torch.ones(1, 1)],
                "aux_outputs": [],
            },
            "forward output field 'pred_logits' must be a non-empty tensor",
        ),
        (
            {
                "pred_logits": torch.ones(1, 1, 2),
                "pred_masks": [],
                "aux_outputs": [],
            },
            "forward output field 'pred_masks' must contain non-empty tensors",
        ),
        (
            {
                "pred_logits": torch.full((1, 1, 2), float("nan")),
                "pred_masks": [torch.ones(1, 1)],
                "aux_outputs": [],
            },
            "forward output field 'pred_logits' contains non-finite values",
        ),
        (
            {
                "pred_logits": torch.ones(1, 1, 2),
                "pred_masks": [torch.full((1, 1), float("inf"))],
                "aux_outputs": [],
            },
            "forward output field 'pred_masks[0]' contains non-finite values",
        ),
        (
            {
                "pred_logits": torch.ones(1, 1, 2, device="meta"),
                "pred_masks": [torch.ones(1, 1)],
                "aux_outputs": [],
            },
            "forward output field 'pred_logits' must be on device cpu (got meta)",
        ),
    ],
)
def test_training_rejects_malformed_forward_output_before_criterion(
    output,
    reason,
):
    owner = _step_owner(forward=lambda *args, **kwargs: output)
    owner.mask_type = "segment_mask"
    owner.criterion = _SyntheticCriterion()

    with pytest.raises(RuntimeError) as error:
        InstanceSegmentation.training_step(owner, _batch(), 16)

    _assert_batch_context(
        error,
        stage="train",
        batch_idx=16,
        reason=reason,
    )


@pytest.mark.parametrize(
    ("scenario", "reason"),
    [
        (
            "missing_aux_outputs",
            "forward output missing required field 'aux_outputs'",
        ),
        (
            "empty_aux_outputs",
            "forward output field 'aux_outputs' must be a non-empty sequence",
        ),
        (
            "nonfinite_aux_logits",
            "forward output field 'aux_outputs[0].pred_logits' contains "
            "non-finite values",
        ),
        (
            "wrong_device_aux_masks",
            "forward output field 'aux_outputs[0].pred_masks[0]' must be on "
            "device cpu (got meta)",
        ),
    ],
)
def test_training_rejects_malformed_aux_output_before_criterion(
    scenario,
    reason,
):
    output = _model_output(torch.tensor(1.0))
    if scenario == "missing_aux_outputs":
        output.pop("aux_outputs")
    elif scenario == "empty_aux_outputs":
        output["aux_outputs"] = []
    elif scenario == "nonfinite_aux_logits":
        output["aux_outputs"][0]["pred_logits"].fill_(float("nan"))
    elif scenario == "wrong_device_aux_masks":
        output["aux_outputs"][0]["pred_masks"][0] = torch.ones(
            1,
            1,
            device="meta",
        )
    owner = _objective_owner(criterion=_SyntheticCriterion())
    owner.forward = lambda *args, **kwargs: output

    with pytest.raises(RuntimeError) as error:
        InstanceSegmentation.training_step(owner, _batch(), 17)

    _assert_batch_context(
        error,
        stage="train",
        batch_idx=17,
        reason=reason,
    )


def test_training_rejects_rank_local_aux_output_count_drift():
    output = _model_output(torch.tensor(1.0))
    owner = _objective_owner(criterion=_SyntheticCriterion())
    owner.forward = lambda *args, **kwargs: output
    owner.model = SimpleNamespace(num_levels=2, num_decoders=1)

    with pytest.raises(RuntimeError) as error:
        InstanceSegmentation.training_step(owner, _batch(), 18)

    _assert_batch_context(
        error,
        stage="train",
        batch_idx=18,
        reason="forward output field 'aux_outputs' must contain 2 entries (got 1)",
    )


@pytest.mark.parametrize(
    ("scenario", "reason"),
    [
        (
            "missing",
            "forward output missing required field 'segment_features'",
        ),
        (
            "empty",
            "forward output field 'segment_features' must be a non-empty "
            "sequence",
        ),
        (
            "nonfinite",
            "forward output field 'segment_features[0][0]' contains "
            "non-finite values",
        ),
        (
            "wrong_device",
            "forward output field 'segment_features[0][0]' must be on device "
            "cpu (got meta)",
        ),
    ],
)
def test_training_rejects_malformed_contrastive_segment_features(
    scenario,
    reason,
):
    output = _model_output(torch.tensor(1.0))
    if scenario == "missing":
        output.pop("segment_features")
    elif scenario == "empty":
        output["segment_features"] = []
    elif scenario == "nonfinite":
        output["segment_features"][0][0].fill_(float("inf"))
    elif scenario == "wrong_device":
        output["segment_features"][0][0] = torch.ones(1, 1, device="meta")
    criterion = _SyntheticCriterion()
    criterion.use_contrastive_loss = True
    owner = _objective_owner(criterion=criterion)
    owner.forward = lambda *args, **kwargs: output

    with pytest.raises(RuntimeError) as error:
        InstanceSegmentation.training_step(owner, _batch(), 19)

    _assert_batch_context(
        error,
        stage="train",
        batch_idx=19,
        reason=reason,
    )


@pytest.mark.parametrize("component", ["features", "coordinates"])
@pytest.mark.parametrize("container_kind", ["list", "tensor"])
def test_training_step_fails_fast_for_collator_empty_point_clouds(
    component, container_kind
):
    data, target, file_names = _batch()
    empty_value = [] if container_kind == "list" else torch.empty(0, 1)
    setattr(data, component, empty_value)
    owner = _step_owner(forward=lambda *args, **kwargs: pytest.fail("forward called"))

    with pytest.raises(RuntimeError) as error:
        InstanceSegmentation.training_step(owner, (data, target, file_names), 8)

    _assert_batch_context(
        error,
        stage="train",
        batch_idx=8,
        reason="empty point cloud",
    )


@pytest.mark.parametrize("component", ["features", "coordinates"])
def test_training_step_fails_fast_for_missing_point_cloud_components(component):
    data, target, file_names = _batch()
    delattr(data, component)
    owner = _step_owner(forward=lambda *args, **kwargs: pytest.fail("forward called"))

    with pytest.raises(RuntimeError) as error:
        InstanceSegmentation.training_step(owner, (data, target, file_names), 8)

    _assert_batch_context(
        error,
        stage="train",
        batch_idx=8,
        reason="empty point cloud",
    )


def test_training_step_wraps_single_point_failure_with_batch_context():
    original_error = RuntimeError(SINGLE_POINT_ERROR)

    def fail_forward(*args, **kwargs):
        raise original_error

    owner = _step_owner(forward=fail_forward)

    with pytest.raises(RuntimeError) as error:
        InstanceSegmentation.training_step(owner, _batch(), 8)

    _assert_batch_context(
        error,
        stage="train",
        batch_idx=8,
        reason=SINGLE_POINT_ERROR,
    )
    assert error.value.__cause__ is original_error


@pytest.mark.parametrize(
    "original_error",
    [
        RuntimeError("rank-local generic forward failure"),
        LookupError("rank-local non-runtime forward failure"),
    ],
)
def test_training_step_synchronizes_all_ordinary_forward_exceptions(original_error):
    def fail_forward(*args, **kwargs):
        raise original_error

    owner = _step_owner(forward=fail_forward)

    with pytest.raises(RuntimeError) as error:
        InstanceSegmentation.training_step(owner, _batch(), 8)

    _assert_batch_context(
        error,
        stage="train",
        batch_idx=8,
        reason=f"forward {type(original_error).__name__}: {original_error}",
    )
    assert error.value.__cause__ is original_error


@pytest.mark.parametrize(
    ("step_method", "stage"),
    [
        (InstanceSegmentation.validation_step, "val"),
        (InstanceSegmentation.test_step, "test"),
    ],
)
def test_eval_steps_wrap_single_point_failure_with_batch_context(step_method, stage):
    original_error = RuntimeError(SINGLE_POINT_ERROR)

    def fail_forward(*args, **kwargs):
        raise original_error

    owner = _step_owner(forward=fail_forward)

    with pytest.raises(RuntimeError) as error:
        step_method(owner, _batch(), 9)

    _assert_batch_context(
        error,
        stage=stage,
        batch_idx=9,
        reason=SINGLE_POINT_ERROR,
    )
    assert error.value.__cause__ is original_error


@pytest.mark.parametrize(
    ("step_method", "stage"),
    [
        (InstanceSegmentation.validation_step, "val"),
        (InstanceSegmentation.test_step, "test"),
    ],
)
def test_eval_steps_synchronize_generic_forward_exceptions(step_method, stage):
    original_error = RuntimeError("rank-local generic eval forward failure")

    def fail_forward(*args, **kwargs):
        raise original_error

    owner = _step_owner(forward=fail_forward)

    with pytest.raises(RuntimeError) as error:
        step_method(owner, _batch(), 9)

    _assert_batch_context(
        error,
        stage=stage,
        batch_idx=9,
        reason=f"forward RuntimeError: {original_error}",
    )
    assert error.value.__cause__ is original_error


@pytest.mark.parametrize(
    ("step_method", "stage"),
    [
        (InstanceSegmentation.validation_step, "val"),
        (InstanceSegmentation.test_step, "test"),
    ],
)
def test_eval_preflight_synchronizes_missing_metadata(step_method, stage):
    data, target, file_names = _batch()
    del data.target_full
    owner = _objective_owner(criterion=_SyntheticCriterion())

    with pytest.raises(RuntimeError) as error:
        step_method(owner, (data, target, file_names), 17)

    _assert_batch_context(
        error,
        stage=stage,
        batch_idx=17,
        reason="missing eval data attribute 'target_full'",
    )


@pytest.mark.parametrize(
    ("step_method", "stage"),
    [
        (InstanceSegmentation.validation_step, "val"),
        (InstanceSegmentation.test_step, "test"),
    ],
)
def test_eval_synchronizes_prediction_and_metric_exceptions(step_method, stage):
    original_error = RuntimeError("rank-local metric update failure")
    owner = _objective_owner(criterion=_SyntheticCriterion())

    def fail_metric(*args, **kwargs):
        raise original_error

    owner.instance_metric = SimpleNamespace(update=fail_metric)

    with pytest.raises(RuntimeError) as error:
        step_method(owner, _batch(), 18)

    _assert_batch_context(
        error,
        stage=stage,
        batch_idx=18,
        reason=f"evaluation RuntimeError: {original_error}",
    )
    assert error.value.__cause__ is original_error


@pytest.mark.parametrize(
    ("step_method", "stage"),
    [
        (InstanceSegmentation.training_step, "train"),
        (InstanceSegmentation.validation_step, "val"),
    ],
)
def test_train_and_eval_synchronize_criterion_exceptions(step_method, stage):
    original_error = ValueError("rank-local criterion failure")

    def criterion(*args, **kwargs):
        raise original_error

    criterion.weight_dict = {}
    owner = _objective_owner(criterion=criterion)

    with pytest.raises(RuntimeError) as error:
        step_method(owner, _batch(), 12)

    _assert_batch_context(
        error,
        stage=stage,
        batch_idx=12,
        reason=f"criterion ValueError: {original_error}",
    )
    assert error.value.__cause__ is original_error


@pytest.mark.parametrize(
    ("step_method", "stage"),
    [
        (InstanceSegmentation.training_step, "train"),
        (InstanceSegmentation.validation_step, "val"),
    ],
)
def test_train_and_eval_synchronize_nonfinite_objectives(step_method, stage):
    def criterion(*args, **kwargs):
        return {"loss_mock": torch.tensor(float("nan"))}

    criterion.weight_dict = {}
    owner = _objective_owner(criterion=criterion)

    with pytest.raises(RuntimeError) as error:
        step_method(owner, _batch(), 13)

    _assert_batch_context(
        error,
        stage=stage,
        batch_idx=13,
        reason=(
            "objective ValueError: non-finite raw objective term 'loss_mock'"
        ),
    )


@pytest.mark.parametrize(
    ("step_method", "stage"),
    [
        (InstanceSegmentation.validation_step, "val"),
        (InstanceSegmentation.test_step, "test"),
    ],
)
def test_eval_steps_fail_fast_for_empty_point_clouds(step_method, stage):
    owner = _step_owner(forward=lambda *args, **kwargs: pytest.fail("forward called"))

    with pytest.raises(RuntimeError) as error:
        step_method(owner, _batch(empty_points=True), 10)

    _assert_batch_context(
        error,
        stage=stage,
        batch_idx=10,
        reason="empty point cloud",
    )


@pytest.mark.parametrize(
    ("step_method", "stage"),
    [
        (InstanceSegmentation.validation_step, "val"),
        (InstanceSegmentation.test_step, "test"),
    ],
)
@pytest.mark.parametrize(
    ("target", "reason"),
    [
        ([], "empty target list"),
        (torch.tensor([0]), "invalid target list"),
    ],
)
def test_eval_steps_fail_fast_for_invalid_targets(
    step_method, stage, target, reason
):
    data, _, file_names = _batch()
    owner = _step_owner(forward=lambda *args, **kwargs: pytest.fail("forward called"))

    with pytest.raises(RuntimeError) as error:
        step_method(owner, (data, target, file_names), 11)

    _assert_batch_context(
        error,
        stage=stage,
        batch_idx=11,
        reason=reason,
    )


class _AutomaticOptimizationHarness(pl.LightningModule):
    def __init__(self, failure: str):
        super().__init__()
        self.failure = failure
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.config = SimpleNamespace(
            general=SimpleNamespace(
                max_batch_size=10,
                use_dbscan=False,
                p2_fail_closed_runtime=True,
                p2_weighted_objective=True,
            )
        )
        self.mask_type = "segment_mask"
        self.criterion = _SyntheticCriterion()
        self.optimizer_ref = None
        self.scheduler_ref = None

    def forward(self, *args, **kwargs):
        if self.failure == "single_point":
            raise RuntimeError(SINGLE_POINT_ERROR)
        if self.failure == "nonfinite_loss":
            return _model_output(
                self.weight * torch.tensor(float("nan"))
            )
        if self.failure == "nonfinite_gradient":
            return _model_output(self.weight.square())
        raise AssertionError("empty-target batches must fail before forward")

    def _process_raw_coordinates(self, data):
        return None

    def _get_mean_loss(self, losses, prefix):
        return {}

    def training_step(self, batch, batch_idx):
        return InstanceSegmentation.training_step(self, batch, batch_idx)

    def on_before_optimizer_step(self, optimizer):
        if self.failure == "nonfinite_gradient":
            self.weight.grad.fill_(float("nan"))
        return InstanceSegmentation.on_before_optimizer_step(self, optimizer)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=0.1, weight_decay=0.01
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=0.1, total_steps=2
        )
        self.optimizer_ref = optimizer
        self.scheduler_ref = scheduler
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


@pytest.mark.parametrize(
    "failure",
    [
        "empty_target",
        "single_point",
        "nonfinite_loss",
        "nonfinite_gradient",
    ],
)
@pytest.mark.filterwarnings("ignore:GPU available but not used.*")
@pytest.mark.filterwarnings(
    "ignore:The 'train_dataloader' does not have many workers.*"
)
def test_fail_fast_does_not_advance_optimizer_global_step_or_scheduler(failure):
    model = _AutomaticOptimizationHarness(failure)
    initial_weight = model.weight.detach().clone()
    initial_scheduler_epoch = 0
    batch = _batch(empty_target=failure == "empty_target")
    dataloader = DataLoader([0], batch_size=1, collate_fn=lambda _: batch)
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        limit_train_batches=1,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
    )

    with pytest.raises(RuntimeError, match="Batch contract violation"):
        trainer.fit(model, train_dataloaders=dataloader)

    assert trainer.global_step == 0
    assert torch.equal(model.weight.detach(), initial_weight)
    assert model.optimizer_ref.state == {}
    assert model.scheduler_ref.last_epoch == initial_scheduler_epoch


def test_gradient_gate_synchronizes_nonfinite_gradients_before_optimizer(
    monkeypatch,
):
    model = _AutomaticOptimizationHarness(failure="none")
    model.weight.grad = torch.tensor(float("nan"))
    model._p2_optimizer_context = {
        "batch_idx": 21,
        "file_names": ["scene-001"],
        "device": torch.device("cpu"),
    }
    model.log_dict = lambda *args, **kwargs: None
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    with pytest.raises(RuntimeError) as error:
        InstanceSegmentation.on_before_optimizer_step(model, optimizer)

    _assert_batch_context(
        error,
        stage="train",
        batch_idx=21,
        reason=(
            "non-finite gradient at optimizer param_group=0, parameter=0"
        ),
    )


def test_normal_gradient_gate_uses_one_scalar_consensus_all_reduce(monkeypatch):
    reductions = []
    model = _AutomaticOptimizationHarness(failure="none")
    model.weight.grad = torch.tensor(1.0)
    model.log_dict = lambda *args, **kwargs: None
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(
        torch.distributed,
        "all_reduce",
        lambda tensor, op: reductions.append((tensor.clone(), op)),
    )

    InstanceSegmentation.on_before_optimizer_step(model, optimizer)

    assert len(reductions) == 1
    assert reductions[0][0].dtype == torch.int32
    assert reductions[0][0].numel() == 1
    assert reductions[0][1] is torch.distributed.ReduceOp.MAX


def test_gradient_gate_defaults_off_and_preserves_upstream_hook(
    monkeypatch,
):
    reductions = []
    model = _AutomaticOptimizationHarness(failure="none")
    model.config.general.p2_fail_closed_runtime = False
    model.weight.grad = torch.tensor(float("nan"))
    model.log_dict = lambda *args, **kwargs: None
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    monkeypatch.setattr(
        trainer_module.pl.utilities,
        "grad_norm",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(
        torch.distributed,
        "all_reduce",
        lambda tensor, op: reductions.append((tensor, op)),
    )

    InstanceSegmentation.on_before_optimizer_step(model, optimizer)

    assert reductions == []


class _AsymmetricDDPHarness(_AutomaticOptimizationHarness):
    def __init__(self, state_dir: Path, failure: str):
        super().__init__(failure="none")
        self.state_dir = str(state_dir)
        self.ddp_failure = failure
        self.model = SimpleNamespace(num_levels=1, num_decoders=1)
        self.mask_type = "segment_mask"
        self.criterion = _SyntheticCriterion(self)

    def _write_state(self, event: str, error: RuntimeError | None = None):
        payload = {
            "batch_idx": int(self._current_batch_idx),
            "global_step": int(self.global_step),
            "optimizer_state_entries": len(self.optimizer_ref.state),
            "scheduler_last_epoch": int(self.scheduler_ref.last_epoch),
            "weight": float(self.weight.detach()),
        }
        if error is not None:
            payload["error"] = str(error)
        path = Path(self.state_dir) / (
            f"rank-{self.global_rank}-batch-{self._current_batch_idx}-{event}.json"
        )
        path.write_text(json.dumps(payload), encoding="utf-8")

    def forward(self, *args, **kwargs):
        if (
            self.ddp_failure == "single_point"
            and self.global_rank == 0
            and self._current_batch_idx == 1
        ):
            raise RuntimeError(SINGLE_POINT_ERROR)
        if (
            self.ddp_failure == "generic_forward"
            and self.global_rank == 0
            and self._current_batch_idx == 1
        ):
            raise RuntimeError("rank-local generic forward failure")
        if (
            self.ddp_failure == "malformed_output"
            and self.global_rank == 0
            and self._current_batch_idx == 1
        ):
            return {
                "pred_logits": torch.empty(0, 1, 2),
                "pred_masks": [],
                "aux_outputs": [],
            }
        if (
            self.ddp_failure == "nonfinite_output"
            and self.global_rank == 0
            and self._current_batch_idx == 1
        ):
            output = _model_output(self.weight.square())
            output["pred_masks"][0].fill_(float("nan"))
            return output
        if (
            self.ddp_failure == "wrong_device_output"
            and self.global_rank == 0
            and self._current_batch_idx == 1
        ):
            output = _model_output(self.weight.square())
            output["pred_logits"] = torch.ones(1, 1, 2, device="meta")
            return output
        if (
            self.ddp_failure == "nonfinite_aux_output"
            and self.global_rank == 0
            and self._current_batch_idx == 1
        ):
            output = _model_output(self.weight.square())
            output["aux_outputs"][0]["pred_masks"][0].fill_(float("nan"))
            return output
        if (
            self.ddp_failure == "missing_aux_outputs"
            and self.global_rank == 0
            and self._current_batch_idx == 1
        ):
            output = _model_output(self.weight.square())
            output["aux_outputs"] = []
            return output
        if (
            self.ddp_failure == "aux_output_count_drift"
            and self.global_rank == 0
            and self._current_batch_idx == 1
        ):
            output = _model_output(self.weight.square())
            output["aux_outputs"].append(output["aux_outputs"][0].copy())
            return output
        if (
            self.ddp_failure == "missing_segment_features"
            and self.global_rank == 0
            and self._current_batch_idx == 1
        ):
            output = _model_output(self.weight.square())
            output.pop("segment_features")
            return output
        if (
            self.ddp_failure == "segment_feature_layer_drift"
            and self.global_rank == 0
            and self._current_batch_idx == 1
        ):
            output = _model_output(self.weight.square())
            output["segment_features"].append(output["segment_features"][0])
            return output
        if (
            self.ddp_failure == "unexpected_pred_changes"
            and self.global_rank == 0
            and self._current_batch_idx == 1
        ):
            output = _model_output(self.weight.square())
            output["pred_changes"] = torch.ones(1, 1, 2)
            return output
        return _model_output(self.weight.square())

    def _get_mean_loss(self, losses, prefix):
        return {}

    def training_step(self, batch, batch_idx):
        data, target, _ = batch
        self._current_batch_idx = batch_idx
        if self.global_rank == 0 and batch_idx == 1:
            if self.ddp_failure == "empty_target":
                target = []
            elif self.ddp_failure == "empty_coordinates":
                data.coordinates = torch.empty(0, data.coordinates.shape[1])
            elif self.ddp_failure == "missing_labels":
                target[0].pop("labels")
            elif self.ddp_failure == "preflight_error":
                target = _ExplodingTargetList(target)
        if (
            self.global_rank == 0
            and batch_idx == 1
            and self.ddp_failure == "malformed_batch"
        ):
            rank_batch = (data, target)
        else:
            rank_batch = (
                data,
                target,
                [f"rank-{self.global_rank}-scene"],
            )
        self._write_state("entered")
        try:
            result = InstanceSegmentation.training_step(self, rank_batch, batch_idx)
        except Exception as error:
            self._write_state("failure", error)
            raise
        self._write_state("returned")
        return result

    def optimizer_step(self, *args, **kwargs):
        result = super().optimizer_step(*args, **kwargs)
        self._write_state("optimizer-step")
        return result

    def lr_scheduler_step(self, scheduler, metric):
        super().lr_scheduler_step(scheduler, metric)
        self._write_state("scheduler-step")

    def on_before_optimizer_step(self, optimizer):
        if self.ddp_failure == "nonfinite_gradient" and self.global_rank == 0:
            self.weight.grad.fill_(float("nan"))
        try:
            return super().on_before_optimizer_step(optimizer)
        except Exception as error:
            if self.ddp_failure == "nonfinite_gradient":
                self._write_state("gradient-failure", error)
            raise


@pytest.mark.filterwarnings("ignore:GPU available but not used.*")
@pytest.mark.filterwarnings(
    "ignore:The 'train_dataloader' does not have many workers.*"
)
@pytest.mark.parametrize(
    ("failure", "reason"),
    [
        ("empty_target", "empty target list"),
        ("empty_coordinates", "empty point cloud"),
        ("single_point", SINGLE_POINT_ERROR),
        (
            "generic_forward",
            "forward RuntimeError: rank-local generic forward failure",
        ),
        (
            "criterion_error",
            "criterion ValueError: rank-local criterion failure",
        ),
        (
            "nonfinite_loss",
            "objective ValueError: non-finite raw objective term 'loss_mock'",
        ),
        (
            "missing_labels",
            "target[0] missing required field 'labels'",
        ),
        (
            "malformed_output",
            "forward output field 'pred_logits' must be a non-empty tensor",
        ),
        (
            "malformed_batch",
            "input ValueError: not enough values to unpack (expected 3, got 2)",
        ),
        (
            "preflight_error",
            "input RuntimeError: rank-local preflight failure",
        ),
        (
            "nonfinite_output",
            "forward output field 'pred_masks[0]' contains non-finite values",
        ),
        (
            "wrong_device_output",
            "forward output field 'pred_logits' must be on device cpu (got meta)",
        ),
        (
            "nonfinite_aux_output",
            "forward output field 'aux_outputs[0].pred_masks[0]' contains "
            "non-finite values",
        ),
        (
            "missing_aux_outputs",
            "forward output field 'aux_outputs' must be a non-empty sequence",
        ),
        (
            "aux_output_count_drift",
            "forward output field 'aux_outputs' must contain 1 entries (got 2)",
        ),
        (
            "missing_segment_features",
            "forward output missing required field 'segment_features'",
        ),
        (
            "segment_feature_layer_drift",
            "forward output field 'segment_features' must contain 1 layer (got 2)",
        ),
        (
            "unexpected_pred_changes",
            "forward output field 'pred_changes' must be None when change "
            "objective is disabled",
        ),
        (
            "empty_objective",
            "objective ValueError: empty raw objective term 'loss_mock'",
        ),
        (
            "vector_objective",
            "objective ValueError: non-scalar raw objective term 'loss_mock'",
        ),
    ],
)
def test_asymmetric_ddp_batch_failure_reaches_consensus_without_pseudo_steps(
    tmp_path, failure, reason
):
    model = _AsymmetricDDPHarness(tmp_path, failure)
    dataloader = DataLoader(
        list(range(8)),
        batch_size=1,
        collate_fn=_collate_valid_batch,
        num_workers=0,
    )
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=2,
        strategy=DDPStrategy(start_method="spawn", find_unused_parameters=True),
        max_epochs=1,
        accumulate_grad_batches=4,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
    )
    deadline_seconds = 30

    def fail_on_timeout(signum, frame):
        raise TimeoutError(f"DDP fail-fast exceeded {deadline_seconds}s")

    previous_handler = signal.signal(signal.SIGALRM, fail_on_timeout)
    signal.alarm(deadline_seconds)
    started_at = time.monotonic()
    try:
        with pytest.raises(ProcessRaisedException) as error:
            trainer.fit(model, train_dataloaders=dataloader)
    except TimeoutError:
        for process in getattr(trainer.strategy.launcher, "procs", []):
            if process.is_alive():
                process.kill()
            process.join()
        pytest.fail(f"DDP fail-fast exceeded {deadline_seconds}s")
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)

    assert time.monotonic() - started_at < deadline_seconds
    assert "Batch contract violation" in str(error.value)
    assert "stage=train" in str(error.value)
    assert f"reason={reason}" in str(error.value)

    for rank in (0, 1):
        for event in ("entered", "returned"):
            valid_state = json.loads(
                (
                    tmp_path / f"rank-{rank}-batch-0-{event}.json"
                ).read_text(encoding="utf-8")
            )
            assert valid_state["global_step"] == 0
            assert valid_state["optimizer_state_entries"] == 0
            assert valid_state["scheduler_last_epoch"] == 0

        failed_state = json.loads(
            (
                tmp_path / f"rank-{rank}-batch-1-failure.json"
            ).read_text(encoding="utf-8")
        )
        assert failed_state["global_step"] == 0
        assert failed_state["optimizer_state_entries"] == 0
        assert failed_state["scheduler_last_epoch"] == 0
        assert failed_state["weight"] == 1.0
        assert f"reason={reason}" in failed_state["error"]
        assert not (tmp_path / f"rank-{rank}-batch-1-returned.json").exists()

    rank_zero_error = json.loads(
        (tmp_path / "rank-0-batch-1-failure.json").read_text(encoding="utf-8")
    )["error"]
    rank_one_error = json.loads(
        (tmp_path / "rank-1-batch-1-failure.json").read_text(encoding="utf-8")
    )["error"]
    assert rank_zero_error == rank_one_error
    assert "rank=0" in rank_zero_error
    assert not list(tmp_path.glob("rank-*-optimizer-step.json"))
    assert not list(tmp_path.glob("rank-*-scheduler-step.json"))


@pytest.mark.filterwarnings("ignore:GPU available but not used.*")
@pytest.mark.filterwarnings(
    "ignore:The 'train_dataloader' does not have many workers.*"
)
def test_asymmetric_ddp_nonfinite_gradient_fails_before_optimizer_step(tmp_path):
    reason = "non-finite gradient at optimizer param_group=0, parameter=0"
    model = _AsymmetricDDPHarness(tmp_path, "nonfinite_gradient")
    dataloader = DataLoader(
        list(range(8)),
        batch_size=1,
        collate_fn=_collate_valid_batch,
        num_workers=0,
    )
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=2,
        strategy=DDPStrategy(start_method="spawn", find_unused_parameters=True),
        max_epochs=1,
        accumulate_grad_batches=4,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
    )
    deadline_seconds = 30

    def fail_on_timeout(signum, frame):
        raise TimeoutError(f"DDP gradient fail-fast exceeded {deadline_seconds}s")

    previous_handler = signal.signal(signal.SIGALRM, fail_on_timeout)
    signal.alarm(deadline_seconds)
    started_at = time.monotonic()
    try:
        with pytest.raises(ProcessRaisedException) as error:
            trainer.fit(model, train_dataloaders=dataloader)
    except TimeoutError:
        for process in getattr(trainer.strategy.launcher, "procs", []):
            if process.is_alive():
                process.kill()
            process.join()
        pytest.fail(f"DDP gradient fail-fast exceeded {deadline_seconds}s")
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)

    assert time.monotonic() - started_at < deadline_seconds
    assert "Batch contract violation" in str(error.value)
    assert f"reason={reason}" in str(error.value)

    errors = []
    for rank in (0, 1):
        for batch_idx in range(4):
            for event in ("entered", "returned"):
                state = json.loads(
                    (
                        tmp_path
                        / f"rank-{rank}-batch-{batch_idx}-{event}.json"
                    ).read_text(encoding="utf-8")
                )
                assert state["global_step"] == 0
                assert state["optimizer_state_entries"] == 0
                assert state["scheduler_last_epoch"] == 0
                assert state["weight"] == 1.0

        failed_state = json.loads(
            (
                tmp_path / f"rank-{rank}-batch-3-gradient-failure.json"
            ).read_text(encoding="utf-8")
        )
        assert failed_state["global_step"] == 0
        assert failed_state["optimizer_state_entries"] == 0
        assert failed_state["scheduler_last_epoch"] == 0
        assert failed_state["weight"] == 1.0
        assert f"reason={reason}" in failed_state["error"]
        errors.append(failed_state["error"])

    assert errors[0] == errors[1]
    assert "rank=0" in errors[0]
    assert not list(tmp_path.glob("rank-*-optimizer-step.json"))
    assert not list(tmp_path.glob("rank-*-scheduler-step.json"))


class _RankLocalMetric:
    def __init__(self, owner):
        self.owner = owner

    def update(self, predictions, targets):
        if self.owner.ddp_failure == "metric_error" and self.owner.global_rank == 0:
            raise RuntimeError("rank-local metric update failure")


class _AsymmetricDDPEvalHarness(_AutomaticOptimizationHarness):
    def __init__(self, state_dir: Path, failure: str):
        super().__init__(failure="none")
        self.state_dir = str(state_dir)
        self.ddp_failure = failure
        self.mask_type = "segment_mask"
        self.criterion = _SyntheticCriterion(self)
        self.instance_metric = _RankLocalMetric(self)
        self.aux_metric = None

    def _write_eval_state(self, event: str, error: Exception | None = None):
        payload = {
            "global_step": int(self.global_step),
            "weight": float(self.weight.detach()),
        }
        if error is not None:
            payload["error"] = str(error)
        path = Path(self.state_dir) / f"eval-rank-{self.global_rank}-{event}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")

    def forward(self, *args, **kwargs):
        return _model_output(self.weight.square())

    def _process_predictions(self, **kwargs):
        return []

    def _get_mean_loss(self, losses, prefix):
        return {}

    def _eval_step(self, batch, stage, batch_idx=None):
        return InstanceSegmentation._eval_step(
            self,
            batch,
            stage,
            batch_idx=batch_idx,
        )

    def validation_step(self, batch, batch_idx):
        data, target, _ = batch
        self._current_batch_idx = batch_idx
        if self.ddp_failure == "missing_eval_attr" and self.global_rank == 0:
            del data.target_full
        rank_batch = (
            data,
            target,
            [f"rank-{self.global_rank}-scene"],
        )
        self._write_eval_state("entered")
        try:
            result = InstanceSegmentation.validation_step(
                self,
                rank_batch,
                batch_idx,
            )
        except Exception as error:
            self._write_eval_state("failure", error)
            raise
        self._write_eval_state("returned")
        return result


@pytest.mark.filterwarnings("ignore:GPU available but not used.*")
@pytest.mark.filterwarnings(
    "ignore:The 'val_dataloader' does not have many workers.*"
)
@pytest.mark.parametrize(
    ("failure", "reason"),
    [
        (
            "missing_eval_attr",
            "missing eval data attribute 'target_full'",
        ),
        (
            "metric_error",
            "evaluation RuntimeError: rank-local metric update failure",
        ),
    ],
)
def test_asymmetric_ddp_eval_failure_reaches_consensus(
    tmp_path,
    failure,
    reason,
):
    model = _AsymmetricDDPEvalHarness(tmp_path, failure)
    dataloader = DataLoader(
        [0],
        batch_size=1,
        collate_fn=_collate_valid_batch,
        num_workers=0,
    )
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=2,
        strategy=DDPStrategy(start_method="spawn", find_unused_parameters=True),
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
    )
    deadline_seconds = 30

    def fail_on_timeout(signum, frame):
        raise TimeoutError(f"DDP eval fail-fast exceeded {deadline_seconds}s")

    previous_handler = signal.signal(signal.SIGALRM, fail_on_timeout)
    signal.alarm(deadline_seconds)
    started_at = time.monotonic()
    try:
        with pytest.raises(ProcessRaisedException) as error:
            trainer.validate(model, dataloaders=dataloader)
    except TimeoutError:
        for process in getattr(trainer.strategy.launcher, "procs", []):
            if process.is_alive():
                process.kill()
            process.join()
        pytest.fail(f"DDP eval fail-fast exceeded {deadline_seconds}s")
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)

    assert time.monotonic() - started_at < deadline_seconds
    assert "Batch contract violation" in str(error.value)
    assert "stage=val" in str(error.value)
    assert f"reason={reason}" in str(error.value)

    errors = []
    for rank in (0, 1):
        entered_state = json.loads(
            (tmp_path / f"eval-rank-{rank}-entered.json").read_text(
                encoding="utf-8"
            )
        )
        failed_state = json.loads(
            (tmp_path / f"eval-rank-{rank}-failure.json").read_text(
                encoding="utf-8"
            )
        )
        assert entered_state["global_step"] == 0
        assert failed_state["global_step"] == 0
        assert failed_state["weight"] == 1.0
        assert f"reason={reason}" in failed_state["error"]
        assert not (tmp_path / f"eval-rank-{rank}-returned.json").exists()
        errors.append(failed_state["error"])

    assert errors[0] == errors[1]
    assert "rank=0" in errors[0]
