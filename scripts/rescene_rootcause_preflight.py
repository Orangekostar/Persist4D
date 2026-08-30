#!/usr/bin/env python3
"""Emit the portable core contract for ReScene root-cause experiments."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from utils.rescene_rootcause_preflight import (
    EXPERIMENT_ID,
    FULL_EPOCHS,
    MANDATORY_VARIANTS,
    MAX_SHORT_CURVE_VARIANTS,
    OPTIMIZER_STEPS_PER_EPOCH,
    ROOTCAUSE_VARIANTS,
    SHORT_HORIZON_EPOCHS,
    TOTAL_OPTIMIZER_STEPS,
    canonical_sha256,
    validate_portable_payload,
)


def build_core_contract() -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "status": "pass",
        "experiment_id": EXPERIMENT_ID,
        "schedule": {
            "full_epochs": FULL_EPOCHS,
            "short_horizon_epochs": SHORT_HORIZON_EPOCHS,
            "optimizer_steps_per_epoch": OPTIMIZER_STEPS_PER_EPOCH,
            "total_optimizer_steps": TOTAL_OPTIMIZER_STEPS,
        },
        "variants": {
            name: {
                "description": contract.description,
                "allowed_paths": list(contract.allowed_paths),
                "gate": contract.gate,
            }
            for name, contract in ROOTCAUSE_VARIANTS.items()
        },
        "mandatory_variants": list(MANDATORY_VARIANTS),
        "maximum_short_curve_variants": MAX_SHORT_CURVE_VARIANTS,
    }
    payload["contract_sha256"] = canonical_sha256(payload)
    validate_portable_payload(payload)
    return payload


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    encoded = (
        json.dumps(
            build_core_contract(),
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")
    if args.output is None:
        print(encoded.decode("ascii"), end="")
    else:
        _publish(args.output, encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
