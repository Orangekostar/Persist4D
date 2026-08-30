"""Shared CLI for formal ReScene root-cause decoder diagnostics."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from utils.rescene_rootcause_diagnostic_runtime import (
    publish_diagnostic,
    run_decoder_diagnostic,
)


def main_for_mode(mode: str, argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--limit-val-batches", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    result = run_decoder_diagnostic(
        mode=mode,
        checkpoint_path=arguments.checkpoint,
        checkpoint_manifest_path=arguments.checkpoint_manifest,
        authorization_path=arguments.authorization,
        pretrained_path=arguments.pretrained,
        device_index=arguments.device,
        limit_val_batches=arguments.limit_val_batches,
    )
    manifest = publish_diagnostic(
        result=result,
        csv_path=arguments.output,
        manifest_path=arguments.manifest_output,
    )
    print(json.dumps(manifest, allow_nan=False, sort_keys=True))
    return 0
