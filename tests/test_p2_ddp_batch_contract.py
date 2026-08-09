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

from trainer.trainer import InstanceSegmentation, _batch_collective_device

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
    target = [] if empty_target else [{"point2segment": torch.tensor([0])}]
    return data, target, ["scene-001"]


def _collate_valid_batch(items):
    return _batch()


class _SyntheticCriterion:
    weight_dict: ClassVar[dict] = {}

    def __call__(self, output, target, mask_type):
        return {"loss_mock": output}


def _step_owner(*, forward):
    owner = SimpleNamespace(
        config=SimpleNamespace(
            general=SimpleNamespace(max_batch_size=10, use_dbscan=False)
        ),
        forward=forward,
        _process_raw_coordinates=lambda data: None,
    )
    owner._eval_step = MethodType(InstanceSegmentation._eval_step, owner)
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
            general=SimpleNamespace(max_batch_size=10, use_dbscan=False)
        )
        self.optimizer_ref = None
        self.scheduler_ref = None

    def forward(self, *args, **kwargs):
        if self.failure == "single_point":
            raise RuntimeError(SINGLE_POINT_ERROR)
        raise AssertionError("empty-target batches must fail before forward")

    def _process_raw_coordinates(self, data):
        return None

    def training_step(self, batch, batch_idx):
        return InstanceSegmentation.training_step(self, batch, batch_idx)

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


@pytest.mark.parametrize("failure", ["empty_target", "single_point"])
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


class _AsymmetricDDPHarness(_AutomaticOptimizationHarness):
    def __init__(self, state_dir: Path, failure: str):
        super().__init__(failure="none")
        self.state_dir = str(state_dir)
        self.ddp_failure = failure
        self.mask_type = "segment_mask"
        self.criterion = _SyntheticCriterion()

    def _write_state(self, event: str, error: RuntimeError | None = None):
        payload = {
            "batch_idx": int(self._current_batch_idx),
            "global_step": int(self.global_step),
            "optimizer_state_entries": len(self.optimizer_ref.state),
            "scheduler_last_epoch": int(self.scheduler_ref.last_epoch),
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
        return self.weight.square()

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
        rank_batch = (
            data,
            target,
            [f"rank-{self.global_rank}-scene"],
        )
        self._write_state("entered")
        try:
            result = InstanceSegmentation.training_step(self, rank_batch, batch_idx)
        except RuntimeError as error:
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
