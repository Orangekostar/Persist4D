#!/usr/bin/env python3
"""Create the one-time common root-cause initialization and portable manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_p2_native_smoke import DEFAULT_CHECKPOINT
from utils.rescene_rootcause_preflight import (
    build_external_file_manifest,
    build_tensor_state_manifest,
    canonical_sha256,
    portable_reference,
    validate_portable_payload,
)

DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "artifacts/rescene_task_learning_root_cause_v1/initialization/COMMON_INITIALIZATION.json"
)


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _file_sha256(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(block)
            digest.update(block)
    return size, digest.hexdigest()


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _compose_config() -> Any:
    with initialize_config_dir(
        version_base="1.2", config_dir=str(PROJECT_ROOT / "conf")
    ):
        config = compose(config_name="config_rescene4d_concerto_rootcause")
    with open_dict(config):
        config.backbone.name = str(DEFAULT_CHECKPOINT.resolve())
        config.general.save_dir = "external:checkpoint/rootcause_runs"
        config.general.rootcause_common_initialization = None
        config.general.rootcause_common_initialization_sha256 = None
    return config


def _portable_config(config: Any, pretrained_sha256: str) -> dict[str, object]:
    payload = OmegaConf.to_container(config, resolve=True)
    payload["backbone"]["name"] = portable_reference(
        "checkpoint/concerto_pretrained", pretrained_sha256
    )
    validate_portable_payload(payload)
    return payload


def _compact_tensor_manifest(manifest: dict[str, object]) -> dict[str, object]:
    return {
        key: manifest[key]
        for key in (
            "schema_version",
            "tensor_count",
            "total_elements",
            "schema_sha256",
            "trainable_schema_sha256",
            "content_sha256",
        )
    }


def _atomic_torch_save(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True)
        + "\n"
    ).encode("ascii")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_initialization(*, state_path: Path, manifest_path: Path) -> dict[str, object]:
    source_commit = _git_head()
    pretrained_bytes, pretrained_sha256 = _file_sha256(DEFAULT_CHECKPOINT)
    _seed_all(45)
    config = _compose_config()
    portable_config = _portable_config(config, pretrained_sha256)
    config_sha256 = canonical_sha256(portable_config)
    from trainer.trainer import InstanceSegmentation

    system = InstanceSegmentation(config)
    state = {name: tensor.detach().cpu() for name, tensor in system.state_dict().items()}
    trainable_names = {
        name for name, parameter in system.named_parameters() if parameter.requires_grad
    }
    encoder_names = {
        name
        for name in state
        if name.startswith(
            ("model.backbone.model.embedding.", "model.backbone.model.enc.")
        )
    }
    trainable_state = {name: state[name] for name in sorted(trainable_names)}
    encoder_state = {name: state[name] for name in sorted(encoder_names)}
    full_manifest = build_tensor_state_manifest(
        state, trainable_names=trainable_names
    )
    trainable_manifest = build_tensor_state_manifest(
        trainable_state, trainable_names=set(trainable_state)
    )
    encoder_manifest = build_tensor_state_manifest(encoder_state)
    _atomic_torch_save(
        {
            "schema_version": 1,
            "state_dict": state,
            "seed": 45,
            "source_commit": source_commit,
            "config_sha256": config_sha256,
        },
        state_path,
    )
    file_manifest = build_external_file_manifest(
        state_path,
        logical_name="rootcause_common_initial_state",
        reference="external:checkpoint/rootcause_common/" + "0" * 64,
        creating_commit=source_commit,
        config_sha256=config_sha256,
        upstream_checkpoint_sha256=pretrained_sha256,
        selected_epoch=0,
        selected_step=0,
    )
    file_manifest["reference"] = portable_reference(
        "checkpoint/rootcause_common", str(file_manifest["sha256"])
    )
    manifest = {
        "schema_version": 1,
        "status": "pass",
        "experiment": "rescene_task_learning_root_cause_v1",
        "seed": 45,
        "source_commit": source_commit,
        "config_sha256": config_sha256,
        "common_state": file_manifest,
        "tensor_state": _compact_tensor_manifest(full_manifest),
        "trainable_state": _compact_tensor_manifest(trainable_manifest),
        "frozen_encoder_state": _compact_tensor_manifest(encoder_manifest),
        "concerto_pretrained": {
            "reference": portable_reference(
                "checkpoint/concerto_pretrained", pretrained_sha256
            ),
            "sha256": pretrained_sha256,
            "bytes": pretrained_bytes,
        },
        "rng_contract": {
            "python_seed": 45,
            "numpy_seed": 45,
            "torch_seed": 45,
            "sampler_seed": 45,
        },
    }
    validate_portable_payload(manifest)
    _atomic_json(manifest, manifest_path)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-path", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    result = build_initialization(
        state_path=args.state_path.resolve(), manifest_path=args.manifest
    )
    print(
        json.dumps(
            {
                "status": "pass",
                "sha256": result["common_state"]["sha256"],
                "bytes": result["common_state"]["bytes"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
