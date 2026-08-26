"""Runtime evidence callback for the formal Sonata training candidate."""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from pytorch_lightning import Callback

RUNTIME_EVENTS_NAME = ".sonata_runtime_events.jsonl"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def append_runtime_event(output_dir: str | Path, event: Mapping[str, Any]) -> None:
    """Append one portable, durable JSON event."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    payload = dict(event)
    payload.setdefault("timestamp_utc", _utc_now())
    encoded = (
        json.dumps(payload, allow_nan=False, ensure_ascii=True, sort_keys=True)
        + "\n"
    ).encode("ascii")
    descriptor = os.open(
        output_path / RUNTIME_EVENTS_NAME,
        os.O_APPEND | os.O_CREAT | os.O_WRONLY,
        0o644,
    )
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _finite_metrics(values: Mapping[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for name, value in values.items():
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                continue
            value = value.detach().cpu().item()
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        number = float(value)
        if math.isfinite(number):
            metrics[str(name)] = number
    return metrics


def _batch_evidence_counts(batch: object) -> tuple[int, int]:
    if not isinstance(batch, (tuple, list)) or len(batch) < 2:
        raise RuntimeError("Sonata evidence callback received an invalid train batch")
    targets = batch[1]
    if not isinstance(targets, (tuple, list)) or not targets:
        raise RuntimeError("Sonata evidence callback requires non-empty targets")
    stage_observations = 0
    for target in targets:
        if not isinstance(target, Mapping):
            raise TypeError("Sonata evidence callback requires target mappings")
        stages = target.get("temporal_stages")
        if not isinstance(stages, torch.Tensor) or stages.numel() == 0:
            raise RuntimeError("Sonata evidence callback requires temporal stages")
        stage_observations += int(torch.unique(stages.detach()).numel())
    return len(targets), stage_observations


def _distributed_sum(value: int, device: torch.device) -> int:
    tensor = torch.tensor(value, dtype=torch.long, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return int(tensor.detach().cpu().item())


def _distributed_max(value: float, device: torch.device) -> float:
    tensor = torch.tensor(value, dtype=torch.float64, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.detach().cpu().item())


class SonataTrainingEvidenceCallback(Callback):
    """Record epoch-boundary evidence without changing training semantics."""

    def __init__(self, output_dir: str) -> None:
        super().__init__()
        self.output_dir = output_dir
        self._started_at = 0.0
        self._epoch_samples = 0
        self._epoch_stages = 0
        self._total_samples = 0
        self._total_stages = 0
        self._last_completed_epoch = -1

    def _emit(self, trainer: Any, event: Mapping[str, Any]) -> None:
        if bool(getattr(trainer, "is_global_zero", False)):
            append_runtime_event(self.output_dir, event)

    def _restore_completed_totals(self, global_step: int) -> None:
        path = Path(self.output_dir) / RUNTIME_EVENTS_NAME
        if not path.is_file():
            return
        latest_step = -1
        for line in path.read_text(encoding="ascii").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            step = event.get("optimizer_steps")
            if (
                event.get("event") == "epoch_completed"
                and isinstance(step, int)
                and latest_step < step <= global_step
            ):
                samples = event.get("samples_seen_total")
                stages = event.get("stage_observations_total")
                if isinstance(samples, int) and isinstance(stages, int):
                    latest_step = step
                    self._total_samples = samples
                    self._total_stages = stages
                    completed_epoch = event.get("completed_epoch")
                    if isinstance(completed_epoch, int):
                        self._last_completed_epoch = completed_epoch

    def on_fit_start(self, trainer: Any, pl_module: Any) -> None:
        self._started_at = time.monotonic()
        self._restore_completed_totals(int(trainer.global_step))
        device = getattr(pl_module, "device", torch.device("cpu"))
        if isinstance(device, torch.device) and device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        self._emit(
            trainer,
            {
                "schema_version": 1,
                "event": "fit_started",
                "optimizer_steps": int(trainer.global_step),
                "resumed": bool(trainer.global_step),
            },
        )

    def on_train_epoch_start(self, trainer: Any, pl_module: Any) -> None:
        self._epoch_samples = 0
        self._epoch_stages = 0

    def on_train_batch_end(
        self,
        trainer: Any,
        pl_module: Any,
        outputs: Any,
        batch: object,
        batch_idx: int,
    ) -> None:
        samples, stages = _batch_evidence_counts(batch)
        self._epoch_samples += samples
        self._epoch_stages += stages

    def on_train_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        device = getattr(pl_module, "device", torch.device("cpu"))
        if not isinstance(device, torch.device):
            device = torch.device(device)
        samples = _distributed_sum(self._epoch_samples, device)
        stages = _distributed_sum(self._epoch_stages, device)
        self._total_samples += samples
        self._total_stages += stages
        self._last_completed_epoch = int(trainer.current_epoch)
        peak_allocated = 0.0
        peak_reserved = 0.0
        if device.type == "cuda":
            peak_allocated = torch.cuda.max_memory_allocated(device) / 1024**2
            peak_reserved = torch.cuda.max_memory_reserved(device) / 1024**2
        peak_allocated = _distributed_max(peak_allocated, device)
        peak_reserved = _distributed_max(peak_reserved, device)
        learning_rate = None
        if trainer.optimizers and trainer.optimizers[0].param_groups:
            learning_rate = float(trainer.optimizers[0].param_groups[0]["lr"])
        self._emit(
            trainer,
            {
                "schema_version": 1,
                "event": "epoch_completed",
                "completed_epoch": self._last_completed_epoch,
                "optimizer_steps": int(trainer.global_step),
                "samples_seen_epoch": samples,
                "samples_seen_total": self._total_samples,
                "stage_observations_epoch": stages,
                "stage_observations_total": self._total_stages,
                "learning_rate": learning_rate,
                "metrics": _finite_metrics(trainer.callback_metrics),
                "peak_allocated_vram_mib": peak_allocated,
                "peak_reserved_vram_mib": peak_reserved,
                "process_wall_clock_seconds": time.monotonic() - self._started_at,
            },
        )

    def on_exception(
        self, trainer: Any, pl_module: Any, exception: BaseException
    ) -> None:
        self._emit(
            trainer,
            {
                "schema_version": 1,
                "event": "fit_interrupted",
                "completed_epoch": self._last_completed_epoch,
                "optimizer_steps": int(trainer.global_step),
                "exception_type": type(exception).__name__,
                "process_wall_clock_seconds": (
                    time.monotonic() - self._started_at if self._started_at else 0.0
                ),
            },
        )

    def on_fit_end(self, trainer: Any, pl_module: Any) -> None:
        self._emit(
            trainer,
            {
                "schema_version": 1,
                "event": "fit_completed",
                "completed_epoch": self._last_completed_epoch,
                "optimizer_steps": int(trainer.global_step),
                "process_wall_clock_seconds": time.monotonic() - self._started_at,
            },
        )
