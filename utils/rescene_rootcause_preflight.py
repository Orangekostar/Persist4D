"""Fail-closed contracts for the ReScene task-learning root-cause study."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

EXPERIMENT_ID = "rescene_task_learning_root_cause_v1"
FULL_EPOCHS = 450
SHORT_HORIZON_EPOCHS = 90
OPTIMIZER_STEPS_PER_EPOCH = 66
TOTAL_OPTIMIZER_STEPS = FULL_EPOCHS * OPTIMIZER_STEPS_PER_EPOCH
MAX_SHORT_CURVE_VARIANTS = 4
MANDATORY_VARIANTS = ("R0", "R1")

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_NAMESPACE_RE = re.compile(r"[a-z0-9][a-z0-9._/-]*")
_IPV4_RE = re.compile(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)")
_PRIVATE_PATH_MARKERS = ("/home/", "/mnt/", "file://")


class RootCauseContractError(ValueError):
    """Raised when a root-cause experiment contract is not exact."""


@dataclass(frozen=True)
class VariantContract:
    name: str
    description: str
    allowed_paths: tuple[str, ...]
    gate: str | None


ROOTCAUSE_VARIANTS = {
    "R0": VariantContract("R0", "weighted formal control", (), None),
    "R1": VariantContract(
        "R1",
        "raw released-code objective",
        ("general.rootcause_objective_mode",),
        None,
    ),
    "R2": VariantContract(
        "R2",
        "larger physical batch",
        ("data.batch_size", "trainer.accumulate_grad_batches"),
        "physical_batch_gradient",
    ),
    "R3": VariantContract(
        "R3",
        "frozen encoder stochastic-depth control",
        ("general.rootcause_frozen_encoder_stochastic_policy",),
        "encoder_stochasticity",
    ),
    "R4": VariantContract(
        "R4",
        "released-code label filtering",
        ("data.train_dataset.filter_out_classes",),
        "filter255_materiality",
    ),
    "R5": VariantContract(
        "R5",
        "released-code EOS coefficient",
        ("loss.eos_coef",),
        "eos_gradient_materiality",
    ),
}


def canonical_sha256(payload: object) -> str:
    """Hash one JSON-compatible value with a stable ASCII encoding."""

    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def portable_reference(namespace: str, sha256: str) -> str:
    """Build a path-free content-addressed external reference."""

    if (
        not isinstance(namespace, str)
        or not _NAMESPACE_RE.fullmatch(namespace)
        or namespace.startswith("/")
        or namespace.endswith("/")
        or ".." in namespace.split("/")
    ):
        raise RootCauseContractError("portable reference namespace is invalid")
    if not isinstance(sha256, str) or not _SHA256_RE.fullmatch(sha256):
        raise RootCauseContractError("portable reference SHA-256 is invalid")
    return f"external:{namespace}/{sha256}"


def _portable_string(value: str) -> bool:
    lowered = value.lower()
    if any(marker in lowered for marker in _PRIVATE_PATH_MARKERS):
        return False
    return not (value.startswith(("/", "~")) or _IPV4_RE.search(value))


def validate_portable_payload(payload: object) -> object:
    """Reject machine paths and network locations from committed payloads."""

    def visit(value: object) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                visit(str(key))
                visit(item)
            return
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for item in value:
                visit(item)
            return
        if isinstance(value, str) and not _portable_string(value):
            raise RootCauseContractError(
                "committed payload contains a non-portable location"
            )

    visit(payload)
    return payload


def _stable_file_hash(path: Path) -> tuple[int, str]:
    path = Path(path)
    try:
        metadata = path.lstat()
    except OSError as error:
        raise RootCauseContractError("external file is unavailable") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise RootCauseContractError("external input must be a regular non-symlink file")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        digest = hashlib.sha256()
        byte_size = 0
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
                byte_size += len(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if identity_before != identity_after or byte_size != after.st_size:
        raise RootCauseContractError("external input changed while hashing")
    return byte_size, digest.hexdigest()


def build_external_file_manifest(
    path: str | Path,
    *,
    logical_name: str,
    reference: str,
    creating_commit: str,
    config_sha256: str,
    upstream_checkpoint_sha256: str,
    selected_epoch: int | None = None,
    selected_step: int | None = None,
) -> dict[str, object]:
    """Bind a non-Git file without serializing its local location."""

    if not logical_name or not isinstance(logical_name, str):
        raise RootCauseContractError("external logical name is invalid")
    if not isinstance(reference, str) or not reference.startswith("external:"):
        raise RootCauseContractError("external reference is invalid")
    for name, value, pattern in (
        ("creating commit", creating_commit, _COMMIT_RE),
        ("config SHA-256", config_sha256, _SHA256_RE),
        ("upstream checkpoint SHA-256", upstream_checkpoint_sha256, _SHA256_RE),
    ):
        if not isinstance(value, str) or not pattern.fullmatch(value):
            raise RootCauseContractError(f"{name} is invalid")
    for name, value in (
        ("selected epoch", selected_epoch),
        ("selected step", selected_step),
    ):
        if value is not None and (type(value) is not int or value < 0):
            raise RootCauseContractError(f"{name} is invalid")
    byte_size, sha256 = _stable_file_hash(Path(path))
    manifest: dict[str, object] = {
        "schema_version": 1,
        "logical_name": logical_name,
        "reference": reference,
        "sha256": sha256,
        "bytes": byte_size,
        "creating_commit": creating_commit,
        "config_sha256": config_sha256,
        "upstream_checkpoint_sha256": upstream_checkpoint_sha256,
        "selected_epoch": selected_epoch,
        "selected_step": selected_step,
    }
    validate_portable_payload(manifest)
    return manifest


def validate_exact_bindings(
    observed: Mapping[str, object], expected: Mapping[str, object]
) -> dict[str, object]:
    """Require recursively exact source, data, runtime, and config bindings."""

    if not isinstance(observed, Mapping) or dict(observed) != dict(expected):
        raise RootCauseContractError("root-cause bindings differ from authorization")
    return copy.deepcopy(dict(observed))


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    if tensor.layout != torch.strided:
        raise RootCauseContractError("tensor state must use strided tensors")
    value = tensor.detach().cpu().contiguous()
    return value.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")


def _tensor_record(
    name: str, tensor: torch.Tensor, *, trainable: bool
) -> dict[str, object]:
    if not isinstance(name, str) or not name or not isinstance(tensor, torch.Tensor):
        raise RootCauseContractError("tensor state entry is invalid")
    return {
        "name": name,
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "numel": int(tensor.numel()),
        "trainable": trainable,
        "sha256": hashlib.sha256(_tensor_bytes(tensor)).hexdigest(),
    }


def build_tensor_state_manifest(
    state: Mapping[str, torch.Tensor],
    *,
    trainable_names: set[str] | frozenset[str] | None = None,
) -> dict[str, object]:
    """Build stable schema and byte-content hashes for one tensor state."""

    if not isinstance(state, Mapping) or not state:
        raise RootCauseContractError("tensor state is empty")
    trainable = set(trainable_names or ())
    if not trainable.issubset(state):
        raise RootCauseContractError("trainable tensor names are absent from state")
    entries = [
        _tensor_record(name, state[name], trainable=name in trainable)
        for name in sorted(state)
    ]
    schema = [
        {
            key: entry[key]
            for key in ("name", "dtype", "shape", "numel", "trainable")
        }
        for entry in entries
    ]
    trainable_schema = [entry for entry in schema if entry["trainable"]]
    content = [
        {
            "name": entry["name"],
            "dtype": entry["dtype"],
            "shape": entry["shape"],
            "sha256": entry["sha256"],
        }
        for entry in entries
    ]
    return {
        "schema_version": 1,
        "tensor_count": len(entries),
        "total_elements": sum(int(entry["numel"]) for entry in entries),
        "schema_sha256": canonical_sha256(schema),
        "trainable_schema_sha256": canonical_sha256(trainable_schema),
        "content_sha256": canonical_sha256(content),
        "tensors": entries,
    }


def validate_common_tensor_state(
    observed: Mapping[str, torch.Tensor],
    expected: Mapping[str, torch.Tensor],
    *,
    allowed_new_prefixes: tuple[str, ...] = (),
) -> dict[str, object]:
    """Require all common tensors to match and bound any architecture additions."""

    missing = sorted(set(expected) - set(observed))
    if missing:
        raise RootCauseContractError(
            "common tensor state has missing tensors: " + ", ".join(missing)
        )
    unexpected = sorted(set(observed) - set(expected))
    unauthorized = [
        name
        for name in unexpected
        if not any(name.startswith(prefix) for prefix in allowed_new_prefixes)
    ]
    if unauthorized:
        raise RootCauseContractError(
            "common tensor state has unexpected tensors: " + ", ".join(unauthorized)
        )
    for name in sorted(expected):
        expected_tensor = expected[name]
        observed_tensor = observed[name]
        if (
            not isinstance(expected_tensor, torch.Tensor)
            or not isinstance(observed_tensor, torch.Tensor)
            or expected_tensor.dtype != observed_tensor.dtype
            or expected_tensor.shape != observed_tensor.shape
        ):
            raise RootCauseContractError(f"common tensor schema differs: {name}")
        if _tensor_bytes(expected_tensor) != _tensor_bytes(observed_tensor):
            raise RootCauseContractError(f"common tensor content differs: {name}")
    return {
        "status": "pass",
        "shared_tensor_count": len(expected),
        "new_tensor_names": unexpected,
        "shared_content_sha256": build_tensor_state_manifest(expected)[
            "content_sha256"
        ],
    }


def load_common_initialization(
    module: Any,
    path: str | Path,
    *,
    expected_sha256: str,
    allowed_new_prefixes: tuple[str, ...] = (),
) -> dict[str, object]:
    """Strictly restore the registered pre-optimizer tensor state."""

    byte_size, observed_sha256 = _stable_file_hash(Path(path))
    if (
        not isinstance(expected_sha256, str)
        or not _SHA256_RE.fullmatch(expected_sha256)
        or observed_sha256 != expected_sha256
    ):
        raise RootCauseContractError("common initialization SHA-256 differs")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise RootCauseContractError("common initialization is unreadable") from error
    state = payload.get("state_dict") if isinstance(payload, Mapping) else None
    if not isinstance(state, Mapping) or not state or any(
        not isinstance(name, str) or not isinstance(tensor, torch.Tensor)
        for name, tensor in state.items()
    ):
        raise RootCauseContractError("common initialization tensor state is invalid")
    observed_state = module.state_dict()
    missing_in_module = sorted(set(state) - set(observed_state))
    if missing_in_module:
        raise RootCauseContractError(
            "common initialization has unexpected tensors: "
            + ", ".join(missing_in_module)
        )
    new_names = sorted(set(observed_state) - set(state))
    unauthorized_new = [
        name
        for name in new_names
        if not any(name.startswith(prefix) for prefix in allowed_new_prefixes)
    ]
    if unauthorized_new:
        raise RootCauseContractError(
            "common initialization is missing tensors: "
            + ", ".join(unauthorized_new)
        )
    new_tensor_bytes = {
        name: _tensor_bytes(observed_state[name]) for name in new_names
    }
    for name, expected_tensor in state.items():
        observed_tensor = observed_state[name]
        if (
            expected_tensor.dtype != observed_tensor.dtype
            or expected_tensor.shape != observed_tensor.shape
        ):
            raise RootCauseContractError(
                f"common initialization tensor schema differs: {name}"
            )
    incompatible = module.load_state_dict(state, strict=False)
    reported_missing = set(incompatible.missing_keys)
    if incompatible.unexpected_keys or not reported_missing.issubset(new_names):
        raise RootCauseContractError("common initialization strict load differs")
    reloaded_state = module.state_dict()
    if any(
        _tensor_bytes(reloaded_state[name]) != new_tensor_bytes[name]
        for name in new_names
    ):
        raise RootCauseContractError("common initialization altered new tensors")
    validation = validate_common_tensor_state(
        reloaded_state,
        state,
        allowed_new_prefixes=allowed_new_prefixes,
    )
    return {
        **validation,
        "bytes": byte_size,
        "sha256": observed_sha256,
        "reference": portable_reference("checkpoint/rootcause_common", observed_sha256),
    }


def _mapping_value(mapping: Mapping[str, object], path: str) -> object:
    value: object = mapping
    for component in path.split("."):
        if not isinstance(value, Mapping) or component not in value:
            raise RootCauseContractError(f"configuration field is missing: {path}")
        value = value[component]
    return value


def validate_full_schedule(config: Mapping[str, object]) -> dict[str, object]:
    """Require every short run to retain the full 450-epoch trajectory."""

    epochs = _mapping_value(config, "trainer.max_epochs")
    target = _mapping_value(config, "scheduler.scheduler._target_")
    total_steps = _mapping_value(config, "scheduler.scheduler.total_steps")
    if epochs != FULL_EPOCHS:
        raise RootCauseContractError("root-cause trainer must retain 450 epochs")
    if target != "torch.optim.lr_scheduler.OneCycleLR":
        raise RootCauseContractError("root-cause scheduler must be OneCycleLR")
    if total_steps != TOTAL_OPTIMIZER_STEPS:
        raise RootCauseContractError(
            "root-cause OneCycle schedule must use explicit 29,700 total steps"
        )
    return {
        "status": "pass",
        "epochs": FULL_EPOCHS,
        "steps_per_epoch": OPTIMIZER_STEPS_PER_EPOCH,
        "total_steps": TOTAL_OPTIMIZER_STEPS,
        "short_horizon_epochs": SHORT_HORIZON_EPOCHS,
        "short_horizon_steps": SHORT_HORIZON_EPOCHS
        * OPTIMIZER_STEPS_PER_EPOCH,
    }


def onecycle_lr_trace(
    config: Mapping[str, object],
    *,
    selected_steps: Sequence[int],
    execution_limit_steps: int | None = None,
) -> dict[int, float]:
    """Simulate selected pre-step learning rates from the full schedule."""

    validate_full_schedule(config)
    if not selected_steps or any(type(step) is not int or step < 0 for step in selected_steps):
        raise RootCauseContractError("selected scheduler steps are invalid")
    selected = tuple(sorted(set(selected_steps)))
    limit = TOTAL_OPTIMIZER_STEPS if execution_limit_steps is None else execution_limit_steps
    if type(limit) is not int or limit <= 0 or limit > TOTAL_OPTIMIZER_STEPS:
        raise RootCauseContractError("scheduler execution limit is invalid")
    if selected[-1] >= limit:
        raise RootCauseContractError("selected scheduler step exceeds execution limit")
    scheduler_config = _mapping_value(config, "scheduler.scheduler")
    if not isinstance(scheduler_config, Mapping):
        raise RootCauseContractError("scheduler configuration is invalid")
    learning_rate = _mapping_value(config, "optimizer.lr")
    if isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float)):
        raise RootCauseContractError("optimizer learning rate is invalid")
    parameter = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.AdamW([parameter], lr=float(learning_rate))
    keyword_names = (
        "max_lr",
        "pct_start",
        "anneal_strategy",
        "cycle_momentum",
        "base_momentum",
        "max_momentum",
        "div_factor",
        "final_div_factor",
        "three_phase",
        "last_epoch",
    )
    kwargs = {
        name: scheduler_config[name]
        for name in keyword_names
        if name in scheduler_config
    }
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        total_steps=TOTAL_OPTIMIZER_STEPS,
        **kwargs,
    )
    trace: dict[int, float] = {}
    wanted = set(selected)
    for step in range(selected[-1] + 1):
        if step in wanted:
            value = float(optimizer.param_groups[0]["lr"])
            if not math.isfinite(value):
                raise RootCauseContractError("OneCycle trace contains a non-finite value")
            trace[step] = value
        optimizer.step()
        scheduler.step()
    return trace


_MISSING = object()


def _normalized_diff_value(value: object) -> object:
    return "<missing>" if value is _MISSING else copy.deepcopy(value)


def _config_diff(
    left: object, right: object, *, path: str, output: list[dict[str, object]]
) -> None:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        for key in sorted(set(left) | set(right)):
            child = f"{path}.{key}" if path else str(key)
            _config_diff(
                left.get(key, _MISSING),
                right.get(key, _MISSING),
                path=child,
                output=output,
            )
        return
    if left != right:
        output.append(
            {
                "path": path,
                "control": _normalized_diff_value(left),
                "variant": _normalized_diff_value(right),
            }
        )


def resolved_config_diff(
    control: Mapping[str, object], variant: Mapping[str, object]
) -> list[dict[str, object]]:
    """Return deterministic leaf-level differences between resolved configs."""

    differences: list[dict[str, object]] = []
    _config_diff(control, variant, path="", output=differences)
    return differences


def validate_variant_isolation(
    variant_name: str,
    control: Mapping[str, object],
    variant: Mapping[str, object],
    *,
    world_size: int = 2,
) -> dict[str, object]:
    """Require one variant's resolved diff to equal its registered allowlist."""

    if variant_name not in ROOTCAUSE_VARIANTS:
        raise RootCauseContractError("root-cause variant is not registered")
    contract = ROOTCAUSE_VARIANTS[variant_name]
    differences = resolved_config_diff(control, variant)
    changed_paths = tuple(item["path"] for item in differences)
    unauthorized = sorted(set(changed_paths) - set(contract.allowed_paths))
    if unauthorized:
        raise RootCauseContractError(
            "root-cause variant has unauthorized changes: " + ", ".join(unauthorized)
        )
    if changed_paths != contract.allowed_paths:
        raise RootCauseContractError(
            "root-cause variant did not change exactly its allowed fields"
        )
    if variant_name == "R1" and _mapping_value(
        variant, "general.rootcause_objective_mode"
    ) != "raw_sum":
        raise RootCauseContractError("R1 objective mode must be raw_sum")
    if variant_name == "R2":
        if type(world_size) is not int or world_size <= 0:
            raise RootCauseContractError("R2 world size is invalid")
        base_effective = (
            world_size
            * int(_mapping_value(control, "data.batch_size"))
            * int(_mapping_value(control, "trainer.accumulate_grad_batches"))
        )
        effective = (
            world_size
            * int(_mapping_value(variant, "data.batch_size"))
            * int(_mapping_value(variant, "trainer.accumulate_grad_batches"))
        )
        if effective != base_effective:
            raise RootCauseContractError("R2 must preserve effective global batch")
    else:
        effective = (
            world_size
            * int(_mapping_value(variant, "data.batch_size"))
            * int(_mapping_value(variant, "trainer.accumulate_grad_batches"))
        )
    return {
        "status": "pass",
        "variant": variant_name,
        "changed_paths": list(changed_paths),
        "effective_global_batch": effective,
        "diff_sha256": canonical_sha256(differences),
    }


def authorize_short_curve_variants(
    requested: Sequence[str], *, gate_results: Mapping[str, bool]
) -> tuple[str, ...]:
    """Authorize mandatory curves plus no more than two gate-passed variants."""

    names = tuple(requested)
    if len(names) != len(set(names)) or any(name not in ROOTCAUSE_VARIANTS for name in names):
        raise RootCauseContractError("short-curve variant list is invalid")
    if not set(MANDATORY_VARIANTS).issubset(names):
        raise RootCauseContractError("mandatory R0/R1 variants are absent")
    if len(names) > MAX_SHORT_CURVE_VARIANTS:
        raise RootCauseContractError("no more than four short-curve variants are allowed")
    for name in names:
        gate = ROOTCAUSE_VARIANTS[name].gate
        if gate is not None and gate_results.get(name) is not True:
            raise RootCauseContractError(f"variant {name} did not pass its gate")
    return names
