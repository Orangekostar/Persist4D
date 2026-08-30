"""Trajectory-preserving callbacks for ReScene root-cause training."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pytorch_lightning import Callback

from utils.rescene_rootcause_preflight import (
    FULL_EPOCHS,
    SHORT_HORIZON_EPOCHS,
    RootCauseContractError,
)


class RootCauseHorizonCallback(Callback):
    """Stop only at the first 90-completed-epoch boundary."""

    def __init__(self, completed_epoch: int = SHORT_HORIZON_EPOCHS) -> None:
        super().__init__()
        if completed_epoch != SHORT_HORIZON_EPOCHS:
            raise RootCauseContractError("short-curve horizon must be 90 epochs")
        self.completed_epoch = completed_epoch

    def on_train_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        if int(trainer.current_epoch) + 1 == self.completed_epoch:
            trainer.should_stop = True


class EpochSetCheckpointCallback(Callback):
    """Save full-state checkpoints only at registered completed epochs."""

    def __init__(
        self,
        output_dir: str,
        completed_epochs: Sequence[int] = (60, 90, FULL_EPOCHS),
    ) -> None:
        super().__init__()
        normalized = tuple(int(value) for value in completed_epochs)
        if normalized != (60, 90, FULL_EPOCHS):
            raise RootCauseContractError(
                "exact checkpoint epochs must be 60, 90, and 450"
            )
        self.output_dir = output_dir
        self.completed_epochs = normalized

    def on_train_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        completed_epoch = int(trainer.current_epoch) + 1
        if completed_epoch not in self.completed_epochs:
            return
        checkpoint = Path(self.output_dir) / f"epoch={completed_epoch:03d}.ckpt"
        trainer.save_checkpoint(str(checkpoint), weights_only=False)
