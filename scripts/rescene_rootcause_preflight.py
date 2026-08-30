#!/usr/bin/env python3
"""Emit the portable core contract for ReScene root-cause experiments."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import hydra
import pytorch_lightning
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.rescene_rootcause_preflight import (
    EXPERIMENT_ID,
    FULL_EPOCHS,
    MANDATORY_VARIANTS,
    MAX_SHORT_CURVE_VARIANTS,
    OPTIMIZER_STEPS_PER_EPOCH,
    ROOTCAUSE_VARIANTS,
    SHORT_HORIZON_EPOCHS,
    TOTAL_OPTIMIZER_STEPS,
    RootCauseContractError,
    authorize_short_curve_variants,
    canonical_sha256,
    validate_full_schedule,
    validate_portable_payload,
    validate_variant_isolation,
)

SELECTED_SHORT_VARIANTS = ("R0", "R1", "R2", "R4")
ROOTCAUSE_CONFIG_NAME = "config_rescene4d_concerto_rootcause"
AUDIT_ROOT = PROJECT_ROOT / "artifacts/rescene_task_learning_root_cause_v1/audit"
INITIALIZATION_MANIFEST = (
    PROJECT_ROOT
    / "artifacts/rescene_task_learning_root_cause_v1/initialization/COMMON_INITIALIZATION.json"
)
START_STATE = PROJECT_ROOT / "artifacts/rescene_task_learning_root_cause_v1/START_STATE.json"

_VARIANT_OVERRIDES = {
    "R0": (),
    "R1": (),
    "R2": ("data.batch_size=4", "trainer.accumulate_grad_batches=4"),
    "R4": ("data.train_dataset.filter_out_classes=[0,1]",),
}


def _load_json(path: Path, *, name: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RootCauseContractError(f"{name} is not readable JSON") from error
    if not isinstance(payload, dict):
        raise RootCauseContractError(f"{name} must contain an object")
    return payload


def _stable_file_identity(path: Path) -> dict[str, object]:
    path = Path(path)
    try:
        before = path.stat()
        if path.is_symlink() or not path.is_file():
            raise OSError
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                size += len(block)
                digest.update(block)
        after = path.stat()
    except OSError as error:
        raise RootCauseContractError("formal external file is unavailable") from error
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ) or size != after.st_size:
        raise RootCauseContractError("formal external file changed while hashing")
    return {"bytes": size, "sha256": digest.hexdigest()}


def _git(*arguments: str, cwd: Path = PROJECT_ROOT) -> str:
    try:
        return subprocess.run(
            ["git", *arguments],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except subprocess.CalledProcessError as error:
        raise RootCauseContractError("formal Git identity is unavailable") from error


@contextmanager
def _temporary_environment(values: Mapping[str, str]) -> Iterator[None]:
    previous = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def variant_overrides(variant: str) -> tuple[str, ...]:
    try:
        return _VARIANT_OVERRIDES[variant]
    except KeyError as error:
        raise RootCauseContractError("short-curve variant is not selected") from error


def compose_variant_config(
    variant: str,
    *,
    pretrained: str | Path,
    common_state: str | Path,
    common_sha256: str,
    output: str | Path,
) -> dict[str, Any]:
    """Compose the exact root-cause config used by one short curve."""

    if variant not in SELECTED_SHORT_VARIANTS:
        raise RootCauseContractError("short-curve variant is not selected")
    objective = "raw_sum" if variant == "R1" else "weighted"
    environment = {
        "CONCERTO_CHECKPOINT": os.fspath(pretrained),
        "RESCENE_ROOTCAUSE_VARIANT": variant,
        "RESCENE_ROOTCAUSE_OUTPUT_DIR": os.fspath(output),
        "RESCENE_ROOTCAUSE_COMMON_STATE": os.fspath(common_state),
        "RESCENE_ROOTCAUSE_COMMON_SHA256": common_sha256,
        "RESCENE_ROOTCAUSE_OBJECTIVE_MODE": objective,
    }
    with _temporary_environment(environment), initialize_config_dir(
        version_base="1.2", config_dir=str(PROJECT_ROOT / "conf")
    ):
        config = compose(
            config_name=ROOTCAUSE_CONFIG_NAME,
            overrides=list(variant_overrides(variant)),
        )
        resolved = OmegaConf.to_container(config, resolve=True)
    if not isinstance(resolved, dict):
        raise RootCauseContractError("resolved root-cause config is invalid")
    validate_full_schedule(resolved)
    return resolved


def portable_variant_config(
    config: Mapping[str, Any],
    *,
    variant: str,
    pretrained_reference: str,
    common_reference: str,
) -> dict[str, Any]:
    """Replace only runtime locations while preserving the exact config."""

    result = copy.deepcopy(dict(config))
    result["backbone"]["name"] = pretrained_reference
    result["model"]["config"]["backbone"]["name"] = pretrained_reference
    result["general"]["rootcause_common_initialization"] = common_reference
    result["general"]["save_dir"] = (
        f"external:checkpoint/rootcause_short/{variant}"
    )
    validate_portable_payload(result)
    return result


def isolation_view(config: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize non-scientific run identity before a resolved-config diff."""

    result = copy.deepcopy(dict(config))
    result["general"]["experiment_name"] = "ROOTCAUSE_VARIANT"
    result["general"]["save_dir"] = "external:checkpoint/rootcause_short/VARIANT"
    for callback in result["callbacks"]:
        if "dirpath" in callback:
            callback["dirpath"] = "external:checkpoint/rootcause_short/VARIANT"
        if "output_dir" in callback:
            callback["output_dir"] = "external:checkpoint/rootcause_short/VARIANT"
    for logger in result["logging"]:
        if "save_dir" in logger:
            logger["save_dir"] = "external:checkpoint/rootcause_short/VARIANT"
    result["data"]["train_dataloader"]["batch_size"] = "DERIVED_FROM_DATA_BATCH_SIZE"
    result["data"]["train_collation"]["filter_out_classes"] = (
        "DERIVED_FROM_TRAIN_FILTER_OUT_CLASSES"
    )
    return result


def build_variant_records(
    configs: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, object]]:
    if tuple(configs) != SELECTED_SHORT_VARIANTS:
        raise RootCauseContractError("variant config order or membership differs")
    control = isolation_view(configs["R0"])
    records: dict[str, dict[str, object]] = {}
    for variant in SELECTED_SHORT_VARIANTS:
        config = copy.deepcopy(dict(configs[variant]))
        validate_portable_payload(config)
        records[variant] = {
            "config_sha256": canonical_sha256(config),
            "resolved_config": config,
            "isolation": validate_variant_isolation(
                variant, control, isolation_view(config), world_size=2
            ),
        }
    return records


def _runtime_environment() -> dict[str, object]:
    gpu_models = [
        torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
    ]
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "pytorch_lightning": pytorch_lightning.__version__,
        "hydra": hydra.__version__,
        "cuda": torch.version.cuda,
        "gpu_count": len(gpu_models),
        "gpu_models": gpu_models,
    }


def _require_equal(observed: object, expected: object, *, name: str) -> None:
    if observed != expected:
        raise RootCauseContractError(f"{name} differs from frozen evidence")


def build_formal_variant_manifest(
    *, common_state: Path, pretrained: Path
) -> dict[str, object]:
    """Build and verify the complete fail-closed RC3 launch authorization."""

    start = _load_json(START_STATE, name="root-cause start state")
    initialization = _load_json(
        INITIALIZATION_MANIFEST, name="common initialization manifest"
    )
    audits = {
        name: _load_json(AUDIT_ROOT / filename, name=name)
        for name, filename in (
            ("objective", "upstream_local_diff.json"),
            ("data", "data_semantics.json"),
            ("ddp_sampler", "ddp_sampler_summary.json"),
            ("encoder_stochasticity", "encoder_stochasticity_summary.json"),
            ("physical_batch", "physical_batch_summary.json"),
        )
    }
    common_identity = _stable_file_identity(common_state)
    pretrained_identity = _stable_file_identity(pretrained)
    _require_equal(
        common_identity,
        {
            "bytes": initialization["common_state"]["bytes"],
            "sha256": initialization["common_state"]["sha256"],
        },
        name="common initialization",
    )
    _require_equal(
        pretrained_identity,
        {
            "bytes": initialization["concerto_pretrained"]["bytes"],
            "sha256": initialization["concerto_pretrained"]["sha256"],
        },
        name="Concerto pretrained encoder",
    )

    from utils.p2_preflight import build_p2_input_manifest

    data_inputs = build_p2_input_manifest(repo_root=PROJECT_ROOT)
    if data_inputs.get("status") != "pass":
        raise RootCauseContractError("formal data content manifest failed")
    for dataset in ("rio", "scannet"):
        expected = start["data"][dataset]
        for field in ("file_count", "total_bytes", "content_sha256"):
            _require_equal(
                data_inputs[dataset][field],
                expected[field],
                name=f"{dataset} {field}",
            )

    gate_results = {
        "R2": audits["physical_batch"]["gate"]["authorized"] is True
        and 8
        in audits["physical_batch"]["gate"]["authorized_physical_global_batches"],
        "R3": audits["encoder_stochasticity"]["gate"]["authorized"] is True,
        "R4": audits["data"]["filter255"]["gate"]["material"] is True,
        "R5": audits["objective"]["eos"]["gate"]["authorized"] is True,
    }
    authorize_short_curve_variants(
        SELECTED_SHORT_VARIANTS, gate_results=gate_results
    )

    pretrained_reference = initialization["concerto_pretrained"]["reference"]
    common_reference = initialization["common_state"]["reference"]
    configs: dict[str, dict[str, Any]] = {}
    for variant in SELECTED_SHORT_VARIANTS:
        composed = compose_variant_config(
            variant,
            pretrained=pretrained_reference,
            common_state=common_reference,
            common_sha256=common_identity["sha256"],
            output=f"external:checkpoint/rootcause_short/{variant}",
        )
        configs[variant] = portable_variant_config(
            composed,
            variant=variant,
            pretrained_reference=pretrained_reference,
            common_reference=common_reference,
        )

    expected_runtime = start["environment"]
    runtime = _runtime_environment()
    for field in ("python", "torch", "pytorch_lightning", "hydra", "cuda"):
        _require_equal(runtime[field], expected_runtime[field], name=f"runtime {field}")
    _require_equal(
        runtime["gpu_count"], expected_runtime["gpu_count"], name="GPU count"
    )
    if any(model != expected_runtime["gpu_model"] for model in runtime["gpu_models"]):
        raise RootCauseContractError("GPU model differs from frozen evidence")

    metric_path = PROJECT_ROOT / "conf/metrics/tmap.yaml"
    evaluator_path = PROJECT_ROOT / "scripts/evaluate_rescan_persist4d.py"
    rio_sequence_path = PROJECT_ROOT / "data/processed/rio/sequence_database_sliding_2.yaml"
    scannet_split_root = PROJECT_ROOT / "third_party/ScanNet/Tasks/Benchmark"
    payload: dict[str, object] = {
        "schema_version": 1,
        "status": "authorized",
        "experiment": EXPERIMENT_ID,
        "source_commit": _git("rev-parse", "HEAD"),
        "selected_variants": list(SELECTED_SHORT_VARIANTS),
        "selection_reason": (
            "R2 and R4 are gate-passed released-runtime/data recipe differences; "
            "R3 remains an authorized diagnostic control but was not selected under "
            "the two-conditional-variant cap; R5 failed its materiality gate."
        ),
        "gates": {
            "results": gate_results,
            "selected_conditionals": ["R2", "R4"],
            "authorized_not_selected": ["R3"],
            "not_authorized": ["R5"],
        },
        "initialization": {
            "manifest_sha256": _stable_file_identity(INITIALIZATION_MANIFEST)[
                "sha256"
            ],
            "common_state": initialization["common_state"],
            "tensor_state": initialization["tensor_state"],
            "trainable_state": initialization["trainable_state"],
            "frozen_encoder_state": initialization["frozen_encoder_state"],
            "pretrained": initialization["concerto_pretrained"],
        },
        "data": {
            "rio": data_inputs["rio"],
            "scannet": data_inputs["scannet"],
            "rio_t2_sequence_database_sha256": _stable_file_identity(
                rio_sequence_path
            )["sha256"],
            "scannet_train_split_sha256": _stable_file_identity(
                scannet_split_root / "scannetv2_train.txt"
            )["sha256"],
            "scannet_validation_split_sha256": _stable_file_identity(
                scannet_split_root / "scannetv2_val.txt"
            )["sha256"],
        },
        "metrics": {
            "config_reference": "repo:conf/metrics/tmap.yaml",
            "config_sha256": _stable_file_identity(metric_path)["sha256"],
            "evaluator_reference": "repo:scripts/evaluate_rescan_persist4d.py",
            "evaluator_sha256": _stable_file_identity(evaluator_path)["sha256"],
            "stmetrics_commit": _git(
                "rev-parse", "HEAD", cwd=PROJECT_ROOT / "third_party/stmetrics"
            ),
            "official_like_validation_sequences": 154,
            "paired_seeds": [45, 46, 47],
        },
        "runtime": runtime,
        "sampler": {
            "sampler_seed": 45,
            "num_samples": audits["ddp_sampler"]["global_sampler"]["num_samples"],
            "contract_sha256": audits["ddp_sampler"]["global_sampler"][
                "contract_sha256"
            ],
            "runtime_audit_sha256": _stable_file_identity(
                AUDIT_ROOT / "ddp_sampler_summary.json"
            )["sha256"],
            "correctly_sharded": audits["ddp_sampler"]["analysis"][
                "correctly_sharded"
            ],
        },
        "audit_bindings": {
            name: _stable_file_identity(AUDIT_ROOT / filename)["sha256"]
            for name, filename in (
                ("objective", "upstream_local_diff.json"),
                ("data", "data_semantics.json"),
                ("ddp_sampler", "ddp_sampler_summary.json"),
                ("encoder_stochasticity", "encoder_stochasticity_summary.json"),
                ("physical_batch", "physical_batch_summary.json"),
            )
        },
        "schedule": {
            "full_epochs": FULL_EPOCHS,
            "short_horizon_epochs": SHORT_HORIZON_EPOCHS,
            "optimizer_steps_per_epoch": OPTIMIZER_STEPS_PER_EPOCH,
            "total_optimizer_steps": TOTAL_OPTIMIZER_STEPS,
            "validation_epochs": [15, 30, 45, 60, 75, 90],
            "external_evaluation_epochs": [60, 90],
        },
        "variants": build_variant_records(configs),
    }
    payload["authorization_sha256"] = canonical_sha256(payload)
    validate_portable_payload(payload)
    return payload


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
    parser.add_argument(
        "--formal-variants",
        action="store_true",
        help="verify formal inputs and emit the RC3 variant authorization",
    )
    parser.add_argument("--common-state", type=Path)
    parser.add_argument("--pretrained", type=Path)
    args = parser.parse_args()
    if args.formal_variants:
        if args.common_state is None or args.pretrained is None:
            parser.error("--formal-variants requires --common-state and --pretrained")
        payload = build_formal_variant_manifest(
            common_state=args.common_state, pretrained=args.pretrained
        )
    else:
        payload = build_core_contract()
    encoded = (
        json.dumps(
            payload,
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
