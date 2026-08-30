#!/usr/bin/env python3
"""Finalize a full ReScene-Strong evaluation against frozen Concerto."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.finalize_rescene_rootcause_full_evaluation import (
    _publish,
    build_full_evaluation_outputs,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--full-training-manifest", type=Path, required=True)
    parser.add_argument("--baseline-per-seed", type=Path, required=True)
    parser.add_argument("--start-state", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args(argv)
    root_names = build_full_evaluation_outputs(
        run_paths=arguments.runs,
        authorization_path=arguments.authorization,
        checkpoint_manifest_path=arguments.checkpoint_manifest,
        full_training_manifest_path=arguments.full_training_manifest,
        baseline_path=arguments.baseline_per_seed,
        start_state_path=arguments.start_state,
        study_kind="strong_local",
    )
    outputs = {
        "official_like_per_seed.csv": root_names["official_like_per_seed.csv"],
        "official_like_summary.csv": root_names["official_like_summary.csv"],
        "STRONG_LOCAL_FULL_VERDICT.json": root_names["ROOT_CAUSE_FULL_VERDICT.json"],
        "STRONG_LOCAL_FULL_VERDICT.md": root_names["ROOT_CAUSE_FULL_VERDICT.md"],
        "STRONG_LOCAL_FULL_PROVENANCE.json": root_names[
            "FULL_EVALUATION_PROVENANCE.json"
        ],
    }
    for name, payload in outputs.items():
        _publish(arguments.output_dir / name, payload)
    verdict = json.loads(outputs["STRONG_LOCAL_FULL_VERDICT.json"])
    print(
        json.dumps(
            {
                "content_sha256": verdict["content_sha256"],
                "status": "pass",
                "verdict": verdict["verdict"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
