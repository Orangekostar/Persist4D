from __future__ import annotations

from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from utils.rescene_rootcause_callbacks import (
    EpochSetCheckpointCallback,
    RootCauseHorizonCallback,
)
from utils.rescene_rootcause_preflight import (
    FULL_EPOCHS,
    OPTIMIZER_STEPS_PER_EPOCH,
    SHORT_HORIZON_EPOCHS,
    TOTAL_OPTIMIZER_STEPS,
    RootCauseContractError,
    onecycle_lr_trace,
    validate_full_schedule,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _compose_rootcause():
    with initialize_config_dir(
        version_base="1.2", config_dir=str(REPO_ROOT / "conf")
    ):
        return compose(config_name="config_rescene4d_concerto_rootcause")


def _config(*, epochs: int = 450, total_steps: int = 29_700) -> dict:
    return {
        "optimizer": {"lr": 5.0e-4},
        "scheduler": {
            "scheduler": {
                "_target_": "torch.optim.lr_scheduler.OneCycleLR",
                "max_lr": 5.0e-4,
                "total_steps": total_steps,
                "pct_start": 0.3,
                "anneal_strategy": "cos",
                "cycle_momentum": True,
                "base_momentum": 0.85,
                "max_momentum": 0.95,
                "div_factor": 25.0,
                "final_div_factor": 10_000.0,
                "three_phase": False,
                "last_epoch": -1,
            }
        },
        "trainer": {"max_epochs": epochs, "accumulate_grad_batches": 8},
    }


def test_full_schedule_constants_match_registered_trajectory() -> None:
    assert FULL_EPOCHS == 450
    assert SHORT_HORIZON_EPOCHS == 90
    assert OPTIMIZER_STEPS_PER_EPOCH == 66
    assert TOTAL_OPTIMIZER_STEPS == 29_700
    assert SHORT_HORIZON_EPOCHS * OPTIMIZER_STEPS_PER_EPOCH == 5_940


def test_schedule_contract_rejects_short_onecycle_or_implicit_steps() -> None:
    assert validate_full_schedule(_config())["total_steps"] == 29_700
    with pytest.raises(RootCauseContractError, match="450"):
        validate_full_schedule(_config(epochs=90))
    with pytest.raises(RootCauseContractError, match="29,700"):
        validate_full_schedule(_config(total_steps=5_940))
    implicit = _config()
    implicit["scheduler"]["scheduler"]["total_steps"] = -1
    with pytest.raises(RootCauseContractError, match="29,700"):
        validate_full_schedule(implicit)


def test_truncated_trace_is_exact_prefix_of_full_onecycle() -> None:
    selected = (0, 1, 65, 66, 989, 1979, 2969, 3959, 4949, 5939)
    full = onecycle_lr_trace(_config(), selected_steps=selected)
    truncated = onecycle_lr_trace(
        _config(),
        selected_steps=selected,
        execution_limit_steps=SHORT_HORIZON_EPOCHS * OPTIMIZER_STEPS_PER_EPOCH,
    )

    assert truncated == full
    assert set(full) == set(selected)
    assert all(torch.isfinite(torch.tensor(value)) for value in full.values())


def test_trace_rejects_step_at_or_beyond_execution_limit() -> None:
    with pytest.raises(RootCauseContractError, match="execution limit"):
        onecycle_lr_trace(
            _config(), selected_steps=(5_940,), execution_limit_steps=5_940
        )


def test_rootcause_config_retains_full_schedule_and_bounded_checkpoints() -> None:
    config = _compose_rootcause()

    assert config.rootcause_preflight.target == "rescene_task_learning_root_cause_v1"
    assert config.trainer.max_epochs == 450
    assert config.trainer.accumulate_grad_batches == 8
    assert config.trainer.check_val_every_n_epoch == 15
    assert config.scheduler.scheduler.total_steps == 29_700
    assert config.general.rootcause_objective_mode == "weighted"
    assert config.general.rootcause_fail_closed_runtime is True
    callbacks = OmegaConf.to_container(config.callbacks, resolve=True)
    assert [callback["_target_"] for callback in callbacks] == [
        "pytorch_lightning.callbacks.ModelCheckpoint",
        "utils.rescene_rootcause_callbacks.EpochSetCheckpointCallback",
        "utils.rescene_rootcause_callbacks.RootCauseHorizonCallback",
        "pytorch_lightning.callbacks.LearningRateMonitor",
    ]
    assert callbacks[0]["save_top_k"] == 1
    assert callbacks[0]["save_last"] is True
    assert callbacks[1]["completed_epochs"] == [60, 90, 450]
    assert callbacks[2]["completed_epoch"] == 90


class _Trainer:
    def __init__(self, current_epoch: int, output_dir: Path | None = None) -> None:
        self.current_epoch = current_epoch
        self.should_stop = False
        self.saved: list[tuple[str, bool]] = []
        self.default_root_dir = str(output_dir) if output_dir is not None else "unused"

    def save_checkpoint(self, path: str, *, weights_only: bool = False) -> None:
        self.saved.append((path, weights_only))


def test_horizon_callback_stops_only_at_first_completed_epoch_90() -> None:
    callback = RootCauseHorizonCallback(completed_epoch=90)
    before = _Trainer(88)
    boundary = _Trainer(89)
    resumed = _Trainer(90)

    callback.on_train_epoch_end(before, None)
    callback.on_train_epoch_end(boundary, None)
    callback.on_train_epoch_end(resumed, None)

    assert before.should_stop is False
    assert boundary.should_stop is True
    assert resumed.should_stop is False


@pytest.mark.parametrize(
    ("current_epoch", "expected_name"),
    [(58, None), (59, "epoch=060.ckpt"), (89, "epoch=090.ckpt"), (449, "epoch=450.ckpt")],
)
def test_epoch_set_checkpoint_uses_completed_epoch_labels(
    tmp_path: Path, current_epoch: int, expected_name: str | None
) -> None:
    callback = EpochSetCheckpointCallback(
        output_dir=str(tmp_path), completed_epochs=(60, 90, 450)
    )
    trainer = _Trainer(current_epoch, tmp_path)

    callback.on_train_epoch_end(trainer, None)

    assert [Path(path).name for path, _ in trainer.saved] == (
        [] if expected_name is None else [expected_name]
    )
    assert all(weights_only is False for _, weights_only in trainer.saved)
