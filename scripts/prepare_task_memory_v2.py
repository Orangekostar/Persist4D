#!/usr/bin/env python3
"""Freeze TaskMemory Retention V2 inputs and M0 contracts."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path

from scripts.task_memory_contracts import AssetBindings, freeze_task_memory_contracts

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _environment_path(name: str, legacy_name: str | None = None) -> Path | None:
    value = os.environ.get(name)
    if value is None and legacy_name is not None:
        value = os.environ.get(legacy_name)
    return Path(value) if value else None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "conf/task_memory_v2/experiment.yaml",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=_environment_path("PERSIST4D_DATA_ROOT"),
    )
    parser.add_argument(
        "--rio-metadata",
        type=Path,
        default=_environment_path("PERSIST4D_RIO_METADATA", "RIO_METADATA"),
    )
    parser.add_argument(
        "--r1-checkpoint",
        type=Path,
        default=_environment_path("PERSIST4D_R1_CHECKPOINT", "R1_CHECKPOINT"),
    )
    parser.add_argument(
        "--concerto-pretrained",
        type=Path,
        default=_environment_path(
            "PERSIST4D_CONCERTO_PRETRAINED", "CONCERTO_PRETRAINED"
        ),
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=_environment_path("PERSIST4D_RUN_ROOT"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts/task_memory_retention_v2",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    missing = [
        option
        for option, value in (
            ("--data-root", args.data_root),
            ("--rio-metadata", args.rio_metadata),
            ("--r1-checkpoint", args.r1_checkpoint),
            ("--concerto-pretrained", args.concerto_pretrained),
            ("--run-root", args.run_root),
        )
        if value is None
    ]
    if missing:
        parser.error(f"missing required runtime bindings: {', '.join(missing)}")
    bundle = freeze_task_memory_contracts(
        config_path=args.config,
        assets=AssetBindings(
            data_root=args.data_root,
            rio_metadata=args.rio_metadata,
            r1_checkpoint=args.r1_checkpoint,
            concerto_pretrained=args.concerto_pretrained,
            run_root=args.run_root,
        ),
        output_root=args.output,
    )
    print(
        json.dumps(
            {
                "output": str(bundle.output_root),
                "public_document_count": len(bundle.public_documents),
                "reference_count": bundle.reference_count,
                "scan_count": bundle.scan_count,
                "status": "COMPLETE",
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
