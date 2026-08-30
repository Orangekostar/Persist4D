#!/usr/bin/env python3
"""Measure repeated frozen-Concerto features under two stochastic policies."""

from __future__ import annotations

import argparse
import copy
import csv
import io
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_p2_native_smoke import (
    DEFAULT_CHECKPOINT,
    TINY_SAMPLE_NAME,
    _compose_runtime,
    _materialize_named_train_batch,
    seed_everything,
)
from utils.rescene_runtime_audit import (
    disable_drop_path,
    encoder_stochasticity_gate,
    feature_repetition_statistics,
    stochastic_module_inventory,
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "artifacts/rescene_task_learning_root_cause_v1/audit"
)


def _publish(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _json_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True)
        + "\n"
    ).encode("ascii")


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _flatten_level(level: Any) -> torch.Tensor:
    values = getattr(level, "decomposed_features", None)
    if values is None and isinstance(level, torch.Tensor):
        values = [level]
    if not isinstance(values, (list, tuple)) or not values:
        raise RuntimeError("Concerto output level has no decomposed features")
    return torch.cat(
        [value.detach().float().reshape(-1) for value in values]
    ).cpu()


def _repeated_backbone_features(
    backbone: torch.nn.Module, data: Any, *, passes: int = 8
) -> dict[str, list[torch.Tensor]]:
    levels: dict[str, list[torch.Tensor]] = {}
    with torch.no_grad():
        for _ in range(passes):
            point, auxiliary, _ = backbone(copy.deepcopy(data))
            for index, value in enumerate([point, *auxiliary]):
                levels.setdefault(f"decoder_level_{index}", []).append(
                    _flatten_level(value)
                )
    return levels


def run_audit(*, device_index: int) -> dict[str, object]:
    if not torch.cuda.is_available() or not 0 <= device_index < torch.cuda.device_count():
        raise ValueError("encoder audit device is unavailable")
    device = torch.device(f"cuda:{device_index}")
    seed_everything(45)
    config, system = _compose_runtime(DEFAULT_CHECKPOINT, device)
    data, _, sample, provenance = _materialize_named_train_batch(
        config, TINY_SAMPLE_NAME, device
    )
    system.train()
    backbone = system.model.backbone
    inventory = stochastic_module_inventory(backbone)
    seed_everything(45)
    current_passes = _repeated_backbone_features(backbone, data)
    current = {
        level: feature_repetition_statistics(values)
        for level, values in current_passes.items()
    }
    seed_everything(45)
    with disable_drop_path(backbone) as changed_modules:
        disabled_passes = _repeated_backbone_features(backbone, data)
    disabled = {
        level: feature_repetition_statistics(values)
        for level, values in disabled_passes.items()
    }
    return {
        "schema_version": 1,
        "status": "pass",
        "source_commit": _git_head(),
        "sample": sample,
        "input_provenance": provenance,
        "pass_count": 8,
        "model_training": bool(system.training),
        "backbone_training": bool(backbone.training),
        "frozen_parameter_count": sum(
            int(not parameter.requires_grad) for parameter in backbone.parameters()
        ),
        "stochastic_modules": inventory,
        "drop_path_disabled_modules": changed_modules,
        "policies": {"current": current, "drop_path_disabled": disabled},
        "gate": encoder_stochasticity_gate(current),
    }


def _csv_bytes(result: dict[str, object]) -> bytes:
    fields = [
        "policy",
        "level",
        "pass_count",
        "element_count",
        "mean_cosine",
        "minimum_cosine",
        "relative_rms_deviation",
        "mean_feature_variance",
    ]
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for policy, levels in result["policies"].items():
        for level, statistics in levels.items():
            writer.writerow({"policy": policy, "level": level, **statistics})
    return output.getvalue().encode("ascii")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    result = run_audit(device_index=args.device)
    _publish(args.output_dir / "encoder_stochasticity.csv", _csv_bytes(result))
    _publish(
        args.output_dir / "encoder_stochasticity_summary.json", _json_bytes(result)
    )
    print(json.dumps({"status": "pass", "gate": result["gate"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
