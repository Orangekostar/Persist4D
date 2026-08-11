import os
import sys
import glob
import hashlib
import json
import math
import re
import tempfile
import typing
import warnings
from collections import defaultdict
from collections.abc import Mapping
from numbers import Real
from pathlib import Path

import torch
import hydra
import wandb
from omegaconf import DictConfig, ListConfig, OmegaConf
from omegaconf.base import ContainerMetadata, Metadata
from omegaconf.nodes import AnyNode
from hydra.core.hydra_config import HydraConfig
from trainer.trainer import InstanceSegmentation
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from utils.utils import (
    flatten_dict,
    load_checkpoint_with_missing_or_exsessive_keys,
    load_backbone_checkpoint_with_missing_or_exsessive_keys,
)
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from utils.p2_preflight import (
    P2_CONFIG_NAME,
    P2_EXPERIMENT_NAME,
    P2_RUNTIME_ENVIRONMENT_VERSIONS,
    P2_SAVE_DIR,
    P2_TARGET,
    require_p2_preflight_authorization as _require_p2_preflight_authorization,
)

# Fix W&B args before imports that use Hydra decorators
sys.argv[1:] = [arg.lstrip('-') for arg in sys.argv[1:] if not arg.startswith('---')] if any('--' in arg and '=' in arg for arg in sys.argv[1:]) else sys.argv[1:]


_TAP_CHECKPOINT_RE = re.compile(
    r"(?:^|-)val_mean_t-AP=(?P<tap>[+-]?(?:\d+(?:\.\d+)?|\.\d+))"
    r"(?=(?:-v\d+)?\.ckpt$)"
)
_EPOCH_CHECKPOINT_RE = re.compile(r"(?:^|-)epoch=(?P<epoch>\d+)(?=-|\.ckpt$)")
_LEGACY_EPOCH_CHECKPOINT_RE = re.compile(r"^(?P<epoch>\d+)(?=-|\.ckpt$)")
_CHECKPOINT_VERSION_RE = re.compile(r"-v(?P<version>\d+)\.ckpt$")
_LAST_CHECKPOINT_RE = re.compile(r"^last(?:-v\d+)?\.ckpt$")
_LAST_EPOCH_CHECKPOINT_RE = re.compile(r"^last-epoch(?:-v\d+)?\.ckpt$")
_P2_FINAL_CHECKPOINT_BASENAME = f"{P2_EXPERIMENT_NAME}.ckpt"
_P2_TRAIN_SAMPLER_CHECKPOINT_KEY = "p2_train_sampler_generator"
_P2_TRAIN_SAMPLER_CHECKPOINT_SCHEMA_VERSION = 1
_P2_TRAIN_SAMPLER_RESUME_SCOPE = "completed_epoch_boundary_only"
_P2_OPTIMIZER_PARAMETER_CONTRACT_KEY = "p2_optimizer_parameter_contract"
_P2_OPTIMIZER_PARAMETER_CONTRACT_SCHEMA_VERSION = 1
_P2_FORMAL_MODEL_STATE_SCHEMA_SHA256 = (
    "4dc8e5e8d455cec6a9f1ecd25653cd8a2736debb2e94c138a5fae6744562e069"
)
_P2_FORMAL_TRAINABLE_PARAMETER_SCHEMA_SHA256 = (
    "d1c7fc483b1f217ab5734bec15292897eafff11b2dd86019a6e8e55e71513073"
)
_P2_MODEL_CHECKPOINT_TARGET = "pytorch_lightning.callbacks.ModelCheckpoint"
_P2_ADAMW_TARGET = "torch.optim.AdamW"
_P2_ONECYCLE_TARGET = "torch.optim.lr_scheduler.OneCycleLR"
_P2_FORMAL_MODEL_CHECKPOINT_COUNT = 3
# 2112 samples / 2 ranks / batch 4 / accumulation 4 * 450 epochs.
_P2_FORMAL_OPTIMIZER_STEPS_PER_EPOCH = 66
_P2_FORMAL_TRAIN_BATCHES_PER_EPOCH = _P2_FORMAL_OPTIMIZER_STEPS_PER_EPOCH * 4
_P2_FORMAL_ONECYCLE_TOTAL_STEPS = _P2_FORMAL_OPTIMIZER_STEPS_PER_EPOCH * 450
P2_CONCERTO_CHECKPOINT_SHA256 = (
    "845ec7dec97a5fabff8fadb5d9858ac6734347b612d1a4b574213419c139de07"
)
P2_CONCERTO_CHECKPOINT_BYTES = 433_987_358
_REPO_ROOT = Path(__file__).resolve().parent


def _matches_repo_path(value, expected_relative_path):
    if value is None:
        return False
    try:
        candidate = Path(str(value)).expanduser()
        if not candidate.is_absolute():
            candidate = _REPO_ROOT / candidate
        expected = _REPO_ROOT / expected_relative_path
        return candidate.resolve() == expected.resolve()
    except (OSError, RuntimeError, TypeError, ValueError):
        return False


def _has_formal_p2_identity(cfg):
    general = cfg.get("general") if hasattr(cfg, "get") else None
    if isinstance(general, Mapping):
        if general.get("experiment_name") == P2_EXPERIMENT_NAME:
            return True
        if general.get("project_name") == P2_EXPERIMENT_NAME:
            return True
        if _matches_repo_path(general.get("save_dir"), P2_SAVE_DIR):
            return True

    callbacks = cfg.get("callbacks") if hasattr(cfg, "get") else None
    if callbacks is None:
        return False
    try:
        callback_entries = iter(callbacks)
    except TypeError:
        return False
    for callback in callback_entries:
        if not isinstance(callback, Mapping):
            continue
        callback_dir = callback.get("dirpath")
        if _matches_repo_path(callback_dir, P2_SAVE_DIR):
            return True
        callback_filename = str(callback.get("filename", "")).removesuffix(".ckpt")
        if (
            callback_filename == P2_EXPERIMENT_NAME
            and _matches_repo_path(callback_dir, Path(P2_SAVE_DIR).parent)
        ):
            return True
    return False


def _is_formal_p2_training(cfg):
    marker = cfg.get("p2_preflight") if hasattr(cfg, "get") else None
    if isinstance(marker, Mapping) and marker.get("target") == P2_TARGET:
        return True
    general = cfg.get("general") if hasattr(cfg, "get") else None
    if isinstance(general, Mapping) and any(
        general.get(flag) is True
        for flag in ("p2_weighted_objective", "p2_fail_closed_runtime")
    ):
        return True
    if _has_formal_p2_identity(cfg):
        return True
    try:
        config_name = (
            HydraConfig.get().job.config_name if HydraConfig.initialized() else None
        )
    except (AttributeError, ValueError):
        config_name = None
    if config_name is None:
        return False
    return str(config_name).removesuffix(".yaml") == P2_CONFIG_NAME


def require_p2_preflight_authorization(cfg, *, artifact_path=None, now=None):
    return _require_p2_preflight_authorization(
        cfg,
        artifact_path=artifact_path,
        now=now,
    )


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


_SAFE_CHECKPOINT_GLOBALS = [
    DictConfig,
    ListConfig,
    ContainerMetadata,
    Metadata,
    AnyNode,
    typing.Any,
    defaultdict,
    dict,
    list,
    tuple,
    set,
    str,
    int,
    float,
    bool,
    type(None),
]


def _safe_torch_load_checkpoint(path):
    with torch.serialization.safe_globals(_SAFE_CHECKPOINT_GLOBALS):
        return torch.load(path, map_location="cpu", weights_only=True)


def _resolved_local_path(value):
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = _REPO_ROOT / path
    return path.resolve()


def require_p2_repository_cwd(*, cwd=None, repo_root=None):
    expected = Path(repo_root or _REPO_ROOT).resolve()
    observed = Path(cwd or Path.cwd()).resolve()
    if observed != expected:
        raise RuntimeError(
            "Formal P2 training requires the repository working directory: "
            f"expected {expected}, got {observed}"
        )
    return expected


def _snapshot_verified_file(
    source,
    snapshot_dir,
    *,
    prefix,
    suffix,
    expected_bytes=None,
    expected_sha256=None,
):
    source = Path(source)
    if not source.is_file() or source.is_symlink():
        raise RuntimeError(f"verified snapshot source is not a regular file: {source}")
    snapshot_dir = Path(snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    source_fd = os.open(
        source,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
    )
    snapshot_path = None
    try:
        before = os.fstat(source_fd)
        output_fd, output_name = tempfile.mkstemp(
            prefix=prefix,
            suffix=suffix,
            dir=snapshot_dir,
        )
        snapshot_path = Path(output_name)
        digest = hashlib.sha256()
        copied_bytes = 0
        with os.fdopen(source_fd, "rb", closefd=False) as source_handle, os.fdopen(
            output_fd,
            "wb",
        ) as output_handle:
            for block in iter(lambda: source_handle.read(1024 * 1024), b""):
                digest.update(block)
                copied_bytes += len(block)
                output_handle.write(block)
            output_handle.flush()
            os.fsync(output_handle.fileno())
        after = os.fstat(source_fd)
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
        if identity_before != identity_after:
            raise RuntimeError("verified snapshot source changed while copying")
        observed_sha256 = digest.hexdigest()
        if expected_bytes is not None and copied_bytes != expected_bytes:
            raise RuntimeError(
                "Formal P2 Concerto checkpoint byte-size mismatch: expected "
                f"{expected_bytes}, got {copied_bytes}"
            )
        if expected_sha256 is not None and observed_sha256 != expected_sha256:
            raise RuntimeError(
                "Formal P2 Concerto checkpoint SHA256 mismatch: expected "
                f"{expected_sha256}, got {observed_sha256}"
            )
        snapshot_path.chmod(0o400)
        return snapshot_path.resolve()
    except Exception:
        if snapshot_path is not None:
            snapshot_path.unlink(missing_ok=True)
        raise
    finally:
        os.close(source_fd)


def require_p2_concerto_checkpoint(cfg):
    checkpoint = _resolved_local_path(cfg.backbone.name)
    if not checkpoint.is_file():
        raise RuntimeError(
            "Formal P2 training requires backbone.name to be a local file; "
            f"refusing remote or missing checkpoint: {cfg.backbone.name}"
        )
    snapshot_dir = _resolved_local_path(cfg.general.save_dir) / ".verified_inputs"
    snapshot = _snapshot_verified_file(
        checkpoint,
        snapshot_dir,
        prefix="concerto-",
        suffix=".pth",
        expected_bytes=P2_CONCERTO_CHECKPOINT_BYTES,
        expected_sha256=P2_CONCERTO_CHECKPOINT_SHA256,
    )
    cfg.backbone.name = str(snapshot)
    return snapshot


def require_p2_resume_checkpoint(cfg, checkpoint_path):
    snapshot_dir = _resolved_local_path(cfg.general.save_dir) / ".verified_inputs"
    try:
        snapshot = _snapshot_verified_file(
            checkpoint_path,
            snapshot_dir,
            prefix="resume-",
            suffix=".ckpt",
        )
        checkpoint = _safe_torch_load_checkpoint(snapshot)
    except Exception as error:
        raise RuntimeError(
            f"Formal P2 resume checkpoint is unreadable: {type(error).__name__}"
        ) from error
    if not isinstance(checkpoint, Mapping):
        raise RuntimeError("Formal P2 resume checkpoint payload is not a mapping")
    validation_error = _resume_checkpoint_validation_error(
        checkpoint,
        formal_p2=True,
        cfg=cfg,
    )
    if validation_error is not None:
        raise RuntimeError(
            "Formal P2 resume checkpoint is not fully resumable: "
            f"{validation_error}"
        )
    return snapshot


def _enforce_formal_p2_training(cfg):
    if not _is_formal_p2_training(cfg):
        return
    require_p2_repository_cwd()
    require_p2_preflight_authorization(cfg)
    require_p2_concerto_checkpoint(cfg)


def _checkpoint_tap(checkpoint_path):
    match = _TAP_CHECKPOINT_RE.search(os.path.basename(checkpoint_path))
    return float(match.group("tap")) if match else None


def _checkpoint_filename_epoch(checkpoint_path):
    basename = os.path.basename(checkpoint_path)
    match = _EPOCH_CHECKPOINT_RE.search(basename)
    if match is None:
        match = _LEGACY_EPOCH_CHECKPOINT_RE.search(basename)
    return int(match.group("epoch")) if match else None


def _checkpoint_version(checkpoint_path):
    match = _CHECKPOINT_VERSION_RE.search(os.path.basename(checkpoint_path))
    return int(match.group("version")) if match else 0


def _formal_p2_checkpoint_config_validation_error(checkpoint, cfg):
    hyper_parameters = checkpoint.get("hyper_parameters")
    if not (
        OmegaConf.is_config(hyper_parameters)
        or isinstance(hyper_parameters, Mapping)
    ):
        return "has no compatible hyper_parameters"
    marker = hyper_parameters.get("p2_preflight")
    if not isinstance(marker, Mapping) or marker.get("target") != P2_TARGET:
        return "has no matching P2 profile provenance"
    try:
        from utils.p2_preflight import p2_training_semantic_sha256

        checkpoint_config_sha256 = p2_training_semantic_sha256(hyper_parameters)
        current_config_sha256 = p2_training_semantic_sha256(cfg)
    except Exception:
        return "config_sha256 is unavailable"
    if checkpoint_config_sha256 != current_config_sha256:
        return (
            "config_sha256 mismatch: expected "
            f"{current_config_sha256}, got {checkpoint_config_sha256}"
        )
    return None


def _load_resume_candidate(checkpoint_path, *, formal_p2=False, cfg=None):
    if formal_p2:
        candidate_path = Path(os.fspath(checkpoint_path))
        try:
            is_regular_file = (
                not candidate_path.is_symlink() and candidate_path.is_file()
            )
        except OSError:
            is_regular_file = False
        if not is_regular_file:
            print(f"Skipping non-regular checkpoint {checkpoint_path}")
            return False, None
    try:
        checkpoint = (
            _safe_torch_load_checkpoint(checkpoint_path)
            if formal_p2
            else torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
        )
    except Exception as error:
        print(f"Skipping unreadable checkpoint {checkpoint_path}: {error}")
        return False, None

    try:
        validation_error = _resume_checkpoint_validation_error(
            checkpoint,
            formal_p2=formal_p2,
            cfg=cfg,
        )
        if validation_error is None and formal_p2:
            validation_error = _formal_p2_callback_reference_validation_error(
                checkpoint,
                cfg,
            )
    except Exception as error:
        print(
            f"Skipping checkpoint with invalid resume state {checkpoint_path}: "
            f"{type(error).__name__}: {error}"
        )
        return False, None
    if validation_error is not None:
        print(
            f"Skipping non-resumable checkpoint {checkpoint_path}: "
            f"{validation_error}"
        )
        return False, None
    return True, checkpoint


def _p2_sampler_checkpoint_validation_error(checkpoint):
    payload = checkpoint.get(_P2_TRAIN_SAMPLER_CHECKPOINT_KEY)
    if not isinstance(payload, Mapping):
        return "missing or invalid P2 sampler generator payload"

    schema_version = payload.get("schema_version")
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version != _P2_TRAIN_SAMPLER_CHECKPOINT_SCHEMA_VERSION
    ):
        return "missing or invalid P2 sampler schema_version"

    if payload.get("resume_scope") != _P2_TRAIN_SAMPLER_RESUME_SCOPE:
        return "missing or invalid P2 sampler resume_scope"

    generator_state = payload.get("generator_state")
    if (
        not isinstance(generator_state, torch.Tensor)
        or generator_state.dtype != torch.uint8
        or generator_state.ndim != 1
        or generator_state.numel() == 0
    ):
        return "missing or invalid P2 sampler generator state"
    try:
        torch.Generator().set_state(generator_state.detach().cpu())
    except Exception:
        return "missing or invalid P2 sampler generator state"
    return None


def _is_checkpoint_scalar(value, *, finite=False):
    if isinstance(value, torch.Tensor):
        if value.numel() != 1 or value.dtype == torch.bool:
            return False
        try:
            scalar = float(value.detach().cpu().item())
        except (RuntimeError, TypeError, ValueError):
            return False
        return not finite or math.isfinite(scalar)
    if not isinstance(value, Real) or isinstance(value, bool):
        return False
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return False
    return not finite or math.isfinite(scalar)


def _p2_parameter_schema_sha256(entries):
    payload = json.dumps(
        entries,
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _formal_p2_optimizer_parameter_contract_validation_error(checkpoint):
    contract = checkpoint.get(_P2_OPTIMIZER_PARAMETER_CONTRACT_KEY)
    if not isinstance(contract, Mapping):
        return "missing or invalid formal P2 optimizer parameter contract"
    if contract.get("schema_version") != _P2_OPTIMIZER_PARAMETER_CONTRACT_SCHEMA_VERSION:
        return "unsupported formal P2 optimizer parameter contract schema"

    model_state_contract = contract.get("state_dict")
    state_dict = checkpoint["state_dict"]
    if (
        not isinstance(model_state_contract, Mapping)
        or not model_state_contract
        or set(model_state_contract) != set(state_dict)
    ):
        return "formal P2 model state contract coverage is incomplete"
    model_state_entries = []
    for name, metadata in sorted(model_state_contract.items()):
        if not isinstance(name, str) or not isinstance(metadata, Mapping):
            return "invalid formal P2 model state contract entry"
        shape = metadata.get("shape")
        dtype = metadata.get("dtype")
        value = state_dict.get(name)
        if (
            not isinstance(shape, list)
            or any(
                not isinstance(dimension, int)
                or isinstance(dimension, bool)
                or dimension < 0
                for dimension in shape
            )
            or not isinstance(dtype, str)
            or not dtype
            or not isinstance(value, torch.Tensor)
            or value.layout != torch.strided
            or list(value.shape) != shape
            or str(value.dtype) != dtype
            or (
                (value.is_floating_point() or value.is_complex())
                and not bool(torch.isfinite(value).all().item())
            )
        ):
            return "formal P2 model state contract does not match state_dict"
        model_state_entries.append([name, shape, dtype])
    model_state_payload = json.dumps(
        model_state_entries,
        ensure_ascii=True,
        separators=(",", ":"),
    )
    model_state_schema_sha256 = hashlib.sha256(
        model_state_payload.encode("ascii")
    ).hexdigest()
    if contract.get("state_dict_schema_sha256") != model_state_schema_sha256:
        return "formal P2 model state contract SHA256 mismatch"
    if model_state_schema_sha256 != _P2_FORMAL_MODEL_STATE_SCHEMA_SHA256:
        return "formal P2 model state schema SHA256 mismatch"

    parameters = contract.get("parameters")
    if not isinstance(parameters, Mapping) or not parameters:
        return "missing or invalid formal P2 optimizer parameter contract parameters"

    optimizer_state = checkpoint["optimizer_states"][0]
    saved_parameter_groups = [
        list(group["params"]) for group in optimizer_state["param_groups"]
    ]
    contract_parameter_groups = contract.get("param_groups")
    if (
        not isinstance(contract_parameter_groups, list)
        or any(not isinstance(group, list) for group in contract_parameter_groups)
        or contract_parameter_groups != saved_parameter_groups
    ):
        return "formal P2 optimizer parameter group order does not match contract"
    parameter_ids = {
        parameter_id
        for group in optimizer_state["param_groups"]
        for parameter_id in group["params"]
    }
    if set(parameters) != parameter_ids:
        return "formal P2 optimizer parameter contract coverage is incomplete"

    parameter_names = set()
    parameter_shapes = {}
    for parameter_id, metadata in parameters.items():
        if not isinstance(metadata, Mapping):
            return "invalid formal P2 optimizer parameter contract entry"
        name = metadata.get("name")
        shape = metadata.get("shape")
        dtype = metadata.get("dtype")
        if (
            not isinstance(parameter_id, int)
            or isinstance(parameter_id, bool)
            or not isinstance(name, str)
            or not name
            or name in parameter_names
            or not isinstance(shape, list)
            or any(
                not isinstance(dimension, int)
                or isinstance(dimension, bool)
                or dimension < 0
                for dimension in shape
            )
            or not isinstance(dtype, str)
            or not dtype
        ):
            return "invalid formal P2 optimizer parameter contract entry"
        parameter = state_dict.get(name)
        if (
            not isinstance(parameter, torch.Tensor)
            or list(parameter.shape) != shape
            or str(parameter.dtype) != dtype
        ):
            return "formal P2 optimizer parameter contract does not match state_dict"
        parameter_names.add(name)
        parameter_shapes[parameter_id] = tuple(shape)

    trainable_parameters = contract.get("trainable_parameters")
    if (
        not isinstance(trainable_parameters, list)
        or not trainable_parameters
        or any(
            not isinstance(entry, list)
            or len(entry) != 3
            or not isinstance(entry[0], str)
            or not entry[0]
            or not isinstance(entry[1], list)
            or any(
                not isinstance(dimension, int)
                or isinstance(dimension, bool)
                or dimension < 0
                for dimension in entry[1]
            )
            or not isinstance(entry[2], str)
            or not entry[2]
            for entry in trainable_parameters
        )
    ):
        return "missing or invalid formal P2 trainable parameter schema"
    if len(trainable_parameters) != len(parameter_ids):
        return "formal P2 trainable parameter schema coverage is incomplete"

    observed_trainable_parameters = [
        [
            parameters[parameter_id]["name"],
            parameters[parameter_id]["shape"],
            parameters[parameter_id]["dtype"],
        ]
        for group in saved_parameter_groups
        for parameter_id in group
    ]
    if trainable_parameters != observed_trainable_parameters:
        return "formal P2 trainable parameter schema does not match optimizer order"
    trainable_schema_sha256 = contract.get("trainable_parameter_schema_sha256")
    observed_trainable_schema_sha256 = _p2_parameter_schema_sha256(
        trainable_parameters
    )
    if trainable_schema_sha256 != observed_trainable_schema_sha256:
        return "formal P2 trainable parameter schema SHA256 mismatch"
    if trainable_schema_sha256 != _P2_FORMAL_TRAINABLE_PARAMETER_SCHEMA_SHA256:
        return "formal P2 trainable parameter schema SHA256 mismatch"
    return parameter_shapes


def _formal_p2_adamw_validation_error(checkpoint, cfg):
    optimizer_cfg = cfg.get("optimizer") if hasattr(cfg, "get") else None
    if (
        not isinstance(optimizer_cfg, Mapping)
        or optimizer_cfg.get("_target_") != _P2_ADAMW_TARGET
    ):
        return "formal P2 config does not define the required AdamW optimizer"

    optimizer_states = checkpoint["optimizer_states"]
    if len(optimizer_states) != 1:
        return "formal P2 checkpoint requires exactly one AdamW optimizer state"
    optimizer_state = optimizer_states[0]
    param_groups = optimizer_state["param_groups"]
    if len(param_groups) != 1:
        return "formal P2 checkpoint requires one AdamW parameter group"
    try:
        expected_parameter = torch.nn.Parameter(torch.ones(1))
        expected_optimizer = hydra.utils.instantiate(
            optimizer_cfg,
            params=[expected_parameter],
        )
        expected_group = expected_optimizer.state_dict()["param_groups"][0]
    except Exception:
        return "formal P2 config does not define a valid AdamW parameter group"
    group = param_groups[0]
    static_fields = (
        "eps",
        "weight_decay",
        "amsgrad",
        "maximize",
        "foreach",
        "capturable",
        "differentiable",
        "fused",
    )
    if any(
        field not in group or group[field] != expected_group.get(field)
        for field in static_fields
    ):
        return "formal P2 AdamW parameter group does not match config"
    betas = group.get("betas")
    expected_betas = expected_group.get("betas")
    if (
        not isinstance(betas, (list, tuple))
        or len(betas) != 2
        or not _is_checkpoint_scalar(betas[0], finite=True)
        or not _is_checkpoint_scalar(betas[1], finite=True)
        or float(betas[1]) != float(expected_betas[1])
    ):
        return "formal P2 AdamW parameter group has invalid betas"

    state = optimizer_state.get("state")
    if not isinstance(state, Mapping) or not state:
        return "missing or empty formal P2 AdamW optimizer state"

    parameter_ids = []
    params = group.get("params")
    if not params:
        return "formal P2 AdamW parameter group has no parameters"
    for parameter_id in params:
        if (
            not isinstance(parameter_id, int)
            or isinstance(parameter_id, bool)
            or parameter_id < 0
            or parameter_id in parameter_ids
        ):
            return "invalid formal P2 AdamW parameter slot identity"
        parameter_ids.append(parameter_id)

    parameter_id_set = set(parameter_ids)
    if set(state) != parameter_id_set:
        return "formal P2 AdamW parameter slot coverage is incomplete"
    parameter_shapes = _formal_p2_optimizer_parameter_contract_validation_error(
        checkpoint
    )
    if isinstance(parameter_shapes, str):
        return parameter_shapes
    global_step = checkpoint["global_step"]
    amsgrad = bool(optimizer_cfg.get("amsgrad", False))
    for parameter_id, slot in state.items():
        if parameter_id not in parameter_id_set or not isinstance(slot, Mapping):
            return "invalid formal P2 AdamW parameter slot"
        required_fields = {"step", "exp_avg", "exp_avg_sq"}
        if amsgrad:
            required_fields.add("max_exp_avg_sq")
        if required_fields.difference(slot):
            return "incomplete formal P2 AdamW parameter slot"

        step = slot["step"]
        if not _is_checkpoint_scalar(step, finite=True):
            return "invalid formal P2 AdamW parameter slot step"
        step_value = float(step.item() if isinstance(step, torch.Tensor) else step)
        if not step_value.is_integer() or step_value != global_step:
            return "invalid formal P2 AdamW parameter slot step"

        exp_avg = slot["exp_avg"]
        exp_avg_sq = slot["exp_avg_sq"]
        if (
            not isinstance(exp_avg, torch.Tensor)
            or not isinstance(exp_avg_sq, torch.Tensor)
            or not exp_avg.is_floating_point()
            or not exp_avg_sq.is_floating_point()
            or exp_avg.numel() == 0
            or exp_avg.shape != exp_avg_sq.shape
            or tuple(exp_avg.shape) != parameter_shapes[parameter_id]
            or not bool(torch.isfinite(exp_avg).all().item())
            or not bool(torch.isfinite(exp_avg_sq).all().item())
            or bool((exp_avg_sq < 0).any().item())
        ):
            return "invalid formal P2 AdamW parameter slot moments"
        if amsgrad:
            max_exp_avg_sq = slot["max_exp_avg_sq"]
            if (
                not isinstance(max_exp_avg_sq, torch.Tensor)
                or max_exp_avg_sq.shape != exp_avg_sq.shape
                or not max_exp_avg_sq.is_floating_point()
                or not bool(torch.isfinite(max_exp_avg_sq).all().item())
                or bool((max_exp_avg_sq < 0).any().item())
            ):
                return "invalid formal P2 AdamW parameter slot moments"
    return None


def _formal_p2_onecycle_scheduler(cfg):
    scheduler_section = cfg.get("scheduler") if hasattr(cfg, "get") else None
    scheduler_cfg = (
        scheduler_section.get("scheduler")
        if isinstance(scheduler_section, Mapping)
        else None
    )
    lightning_params = (
        scheduler_section.get("pytorch_lightning_params")
        if isinstance(scheduler_section, Mapping)
        else None
    )
    if (
        not isinstance(scheduler_cfg, Mapping)
        or scheduler_cfg.get("_target_") != _P2_ONECYCLE_TARGET
        or not isinstance(lightning_params, Mapping)
        or lightning_params.get("interval") != "step"
    ):
        return None
    try:
        parameter = torch.nn.Parameter(torch.ones(1))
        optimizer = hydra.utils.instantiate(cfg.optimizer, params=[parameter])
        resolved_scheduler_cfg = scheduler_cfg.copy()
        resolved_scheduler_cfg["total_steps"] = _P2_FORMAL_ONECYCLE_TOTAL_STEPS
        scheduler = hydra.utils.instantiate(
            resolved_scheduler_cfg,
            optimizer=optimizer,
        )
    except Exception:
        return None
    return scheduler


def _formal_p2_onecycle_validation_error(checkpoint, cfg):
    expected_scheduler = _formal_p2_onecycle_scheduler(cfg)
    if expected_scheduler is None:
        return "formal P2 config does not define the required OneCycleLR scheduler"
    contract = expected_scheduler.state_dict()

    scheduler_states = checkpoint["lr_schedulers"]
    if len(scheduler_states) != 1:
        return "formal P2 checkpoint requires exactly one OneCycleLR scheduler state"
    state = scheduler_states[0]
    missing_fields = set(contract).difference(state)
    if missing_fields:
        return "missing or invalid formal P2 OneCycleLR scheduler state"

    if state.get("total_steps") != _P2_FORMAL_ONECYCLE_TOTAL_STEPS:
        return "formal P2 OneCycleLR total_steps does not match 29700"
    for field in (
        "_schedule_phases",
        "_anneal_func_type",
        "cycle_momentum",
        "use_beta1",
        "base_lrs",
    ):
        if state.get(field) != contract[field]:
            return f"formal P2 OneCycleLR scheduler state has invalid {field}"

    last_epoch = state.get("last_epoch")
    if (
        not isinstance(last_epoch, int)
        or isinstance(last_epoch, bool)
        or last_epoch != checkpoint["global_step"]
    ):
        return "formal P2 OneCycleLR last_epoch does not match global_step"
    step_count = state.get("_step_count")
    if (
        not isinstance(step_count, int)
        or isinstance(step_count, bool)
        or step_count != last_epoch + 1
    ):
        return "formal P2 OneCycleLR _step_count does not match last_epoch"
    if not isinstance(state.get("_get_lr_called_within_step"), bool):
        return "missing or invalid formal P2 OneCycleLR scheduler state"

    last_lrs = state.get("_last_lr")
    optimizer_group = checkpoint["optimizer_states"][0]["param_groups"][0]
    if (
        not isinstance(last_lrs, list)
        or len(last_lrs) != 1
        or any(not _is_checkpoint_scalar(value, finite=True) for value in last_lrs)
    ):
        return "missing or invalid formal P2 OneCycleLR scheduler state"

    expected_group = expected_scheduler.optimizer.param_groups[0]
    for field in (
        "initial_lr",
        "max_lr",
        "min_lr",
        "max_momentum",
        "base_momentum",
    ):
        if optimizer_group.get(field) != expected_group.get(field):
            return f"formal P2 OneCycleLR optimizer parameter group has invalid {field}"
    expected_scheduler.last_epoch = last_epoch
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            expected_lrs = expected_scheduler.get_lr()
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return "invalid formal P2 OneCycleLR scheduler progress"
    expected_lr = float(expected_lrs[0])
    observed_lr = optimizer_group.get("lr")
    scheduler_lr = last_lrs[0]
    if (
        not _is_checkpoint_scalar(observed_lr, finite=True)
        or not math.isclose(
            float(observed_lr),
            expected_lr,
            rel_tol=1e-12,
            abs_tol=1e-15,
        )
        or not math.isclose(
            float(scheduler_lr),
            expected_lr,
            rel_tol=1e-12,
            abs_tol=1e-15,
        )
    ):
        return "formal P2 OneCycleLR current learning rate is invalid"

    observed_betas = optimizer_group.get("betas")
    expected_betas = expected_scheduler.optimizer.param_groups[0].get("betas")
    if (
        not isinstance(observed_betas, (list, tuple))
        or len(observed_betas) != 2
        or not math.isclose(
            float(observed_betas[0]),
            float(expected_betas[0]),
            rel_tol=1e-12,
            abs_tol=1e-15,
        )
    ):
        return "formal P2 OneCycleLR current momentum is invalid"
    return None


def _formal_p2_model_checkpoint_callbacks(cfg):
    callback_configs = cfg.get("callbacks") if hasattr(cfg, "get") else None
    if callback_configs is None:
        return None
    callbacks = []
    try:
        callback_entries = iter(callback_configs)
    except TypeError:
        return None
    for callback_cfg in callback_entries:
        if (
            not isinstance(callback_cfg, Mapping)
            or callback_cfg.get("_target_") != _P2_MODEL_CHECKPOINT_TARGET
        ):
            continue
        try:
            callback = hydra.utils.instantiate(callback_cfg)
        except Exception:
            return None
        if not isinstance(callback, ModelCheckpoint):
            return None
        callbacks.append(callback)
    state_keys = [callback.state_key for callback in callbacks]
    if (
        len(callbacks) != _P2_FORMAL_MODEL_CHECKPOINT_COUNT
        or len(set(state_keys)) != _P2_FORMAL_MODEL_CHECKPOINT_COUNT
    ):
        return None
    return callbacks


def _checkpoint_paths_match(left, right):
    if not isinstance(left, str) or not left or right is None:
        return False
    try:
        return Path(left).expanduser().resolve() == Path(right).expanduser().resolve()
    except (OSError, RuntimeError, TypeError, ValueError):
        return False


def _checkpoint_path_is_in_dir(path, directory):
    if not isinstance(path, str) or not path or directory is None:
        return False
    try:
        return (
            Path(path).expanduser().resolve().parent
            == Path(directory).expanduser().resolve()
        )
    except (OSError, RuntimeError, TypeError, ValueError):
        return False


def _checkpoint_path_is_regular_file(path):
    if not isinstance(path, str) or not path:
        return False
    try:
        checkpoint_path = Path(path).expanduser()
        return not checkpoint_path.is_symlink() and checkpoint_path.is_file()
    except (OSError, RuntimeError, TypeError, ValueError):
        return False


def _formal_p2_callback_reference_validation_error(checkpoint, cfg):
    expected_callbacks = _formal_p2_model_checkpoint_callbacks(cfg)
    if expected_callbacks is None:
        return "formal P2 config does not define three unique ModelCheckpoint callbacks"
    callback_states = checkpoint["callbacks"]
    for callback in expected_callbacks:
        state = callback_states[callback.state_key]
        referenced_paths = {state["best_model_path"]}
        referenced_paths.add(state["kth_best_model_path"])
        referenced_paths.add(state["last_model_path"])
        referenced_paths.update(state["best_k_models"])
        for referenced_path in referenced_paths:
            if referenced_path and not _checkpoint_path_is_regular_file(
                referenced_path
            ):
                return (
                    "formal P2 ModelCheckpoint callback history references "
                    f"a missing or non-regular file: {callback.state_key}"
                )
    return None


def _checkpoint_scalars_match(left, right):
    return (
        _is_checkpoint_scalar(left, finite=True)
        and _is_checkpoint_scalar(right, finite=True)
        and math.isclose(
            float(left.item() if isinstance(left, torch.Tensor) else left),
            float(right.item() if isinstance(right, torch.Tensor) else right),
            rel_tol=1e-12,
            abs_tol=1e-15,
        )
    )


def _formal_p2_callback_history_validation_error(
    callback,
    state,
    checkpoint,
    *,
    first_validation_epoch,
):
    epoch = checkpoint["epoch"]
    interval = callback._every_n_epochs
    completed_epochs = epoch + 1
    if interval is None:
        return None
    if callback.monitor is None:
        if completed_epochs < interval:
            return None
        if completed_epochs == interval:
            epoch_processed = _nested_checkpoint_value(
                checkpoint["loops"]["fit_loop"],
                ("epoch_progress", "total", "processed"),
            )
            if epoch_processed == completed_epochs:
                return None
    if callback.monitor is not None and completed_epochs < interval:
        return None

    best_path = state["best_model_path"]
    if not _checkpoint_path_is_in_dir(best_path, callback.dirpath):
        return f"invalid formal P2 ModelCheckpoint callback history: {callback.state_key}"
    if callback.monitor is None:
        return None

    best_score = state["best_model_score"]
    current_score = state["current_score"]
    best_k_models = state["best_k_models"]
    if (
        not _is_checkpoint_scalar(best_score, finite=True)
        or not _is_checkpoint_scalar(current_score, finite=True)
        or not isinstance(best_k_models, Mapping)
        or not best_k_models
        or best_path not in best_k_models
        or not _checkpoint_scalars_match(best_k_models[best_path], best_score)
        or state["kth_best_model_path"] != best_path
        or not _checkpoint_scalars_match(state["kth_value"], best_score)
        or any(
            not _checkpoint_path_is_in_dir(path, callback.dirpath)
            or not _is_checkpoint_scalar(score, finite=True)
            for path, score in best_k_models.items()
        )
    ):
        return f"invalid formal P2 ModelCheckpoint callback history: {callback.state_key}"
    if callback.save_last:
        last_path = state["last_model_path"]
        first_top_before_last = (
            epoch == first_validation_epoch and last_path == ""
        )
        if not first_top_before_last and not _checkpoint_path_is_in_dir(
            last_path,
            callback.dirpath,
        ):
            return (
                "invalid formal P2 ModelCheckpoint callback history: "
                f"{callback.state_key}"
            )
    return None


def _formal_p2_callback_validation_error(checkpoint, cfg):
    expected_callbacks = _formal_p2_model_checkpoint_callbacks(cfg)
    if expected_callbacks is None:
        return "formal P2 config does not define three unique ModelCheckpoint callbacks"
    trainer_cfg = cfg.get("trainer") if hasattr(cfg, "get") else None
    validation_cadence = (
        trainer_cfg.get("check_val_every_n_epoch")
        if isinstance(trainer_cfg, Mapping)
        else None
    )
    if (
        not isinstance(validation_cadence, int)
        or isinstance(validation_cadence, bool)
        or validation_cadence < 1
    ):
        return "formal P2 config has invalid validation cadence"
    first_validation_epoch = validation_cadence - 1

    callback_states = checkpoint["callbacks"]
    for callback in expected_callbacks:
        if (
            callback.monitor == "val_mean_t-AP"
            and callback._save_on_train_epoch_end is not True
        ):
            return (
                "formal P2 monitored ModelCheckpoint must save on "
                "train_epoch_end"
            )
        state = callback_states.get(callback.state_key)
        if not isinstance(state, Mapping):
            return f"missing formal P2 ModelCheckpoint callback state: {callback.state_key}"
        required_fields = set(callback.state_dict())
        if required_fields.difference(state):
            return f"incomplete formal P2 ModelCheckpoint callback state: {callback.state_key}"
        if state.get("monitor") != callback.monitor:
            return f"invalid formal P2 ModelCheckpoint callback state: {callback.state_key}"
        if not _checkpoint_paths_match(state.get("dirpath"), callback.dirpath):
            return f"invalid formal P2 ModelCheckpoint callback state: {callback.state_key}"
        if any(
            not isinstance(state.get(field), str)
            for field in (
                "best_model_path",
                "kth_best_model_path",
                "last_model_path",
            )
        ):
            return f"invalid formal P2 ModelCheckpoint callback state: {callback.state_key}"
        if any(
            state.get(field) is not None
            and not _is_checkpoint_scalar(state[field])
            for field in ("best_model_score", "current_score")
        ) or not _is_checkpoint_scalar(state.get("kth_value")):
            return f"invalid formal P2 ModelCheckpoint callback state: {callback.state_key}"
        best_k_models = state.get("best_k_models")
        if not isinstance(best_k_models, Mapping) or any(
            not isinstance(path, str) or not _is_checkpoint_scalar(score)
            for path, score in best_k_models.items()
        ):
            return f"invalid formal P2 ModelCheckpoint callback state: {callback.state_key}"
        history_error = _formal_p2_callback_history_validation_error(
            callback,
            state,
            checkpoint,
            first_validation_epoch=first_validation_epoch,
        )
        if history_error is not None:
            return history_error
    return None


def _nested_checkpoint_value(mapping, path):
    value = mapping
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    return value


def _formal_p2_loop_validation_error(checkpoint, _cfg):
    loops = checkpoint["loops"]
    loop_names = {"fit_loop", "validate_loop", "test_loop", "predict_loop"}
    if set(loops) != loop_names or any(
        not isinstance(loops[name], Mapping) for name in loop_names
    ):
        return "missing or invalid formal P2 Lightning loop state"

    for loop_name in ("validate_loop", "test_loop", "predict_loop"):
        loop = loops[loop_name]
        batch_progress = loop.get("batch_progress")
        if (
            not isinstance(loop.get("state_dict"), Mapping)
            or not isinstance(batch_progress, Mapping)
            or not isinstance(batch_progress.get("total"), Mapping)
            or not isinstance(batch_progress.get("current"), Mapping)
        ):
            return f"missing or invalid formal P2 {loop_name} state"
        for progress in (batch_progress["total"], batch_progress["current"]):
            if any(
                not isinstance(progress.get(field), int)
                or isinstance(progress.get(field), bool)
                or progress[field] < 0
                for field in ("ready", "completed", "started", "processed")
            ):
                return f"missing or invalid formal P2 {loop_name} state"

    fit_loop = loops["fit_loop"]
    completed_epochs = checkpoint["epoch"] + 1
    global_step = checkpoint["global_step"]
    total_batches = completed_epochs * _P2_FORMAL_TRAIN_BATCHES_PER_EPOCH
    expected_progress = {
        ("epoch_progress", "total", "ready"): {completed_epochs},
        ("epoch_progress", "total", "completed"): {completed_epochs},
        ("epoch_progress", "total", "started"): {completed_epochs},
        ("epoch_progress", "total", "processed"): {completed_epochs},
        ("epoch_progress", "current", "ready"): {completed_epochs},
        ("epoch_progress", "current", "completed"): {completed_epochs},
        ("epoch_progress", "current", "started"): {completed_epochs},
        ("epoch_progress", "current", "processed"): {completed_epochs},
        ("epoch_loop.state_dict", "_batches_that_stepped"): {global_step},
        (
            "epoch_loop.automatic_optimization.optim_progress",
            "optimizer",
            "step",
            "total",
            "ready",
        ): {global_step},
        (
            "epoch_loop.automatic_optimization.optim_progress",
            "optimizer",
            "step",
            "total",
            "completed",
        ): {global_step},
        (
            "epoch_loop.automatic_optimization.optim_progress",
            "optimizer",
            "step",
            "current",
            "ready",
        ): {0},
        (
            "epoch_loop.automatic_optimization.optim_progress",
            "optimizer",
            "step",
            "current",
            "completed",
        ): {0},
        (
            "epoch_loop.automatic_optimization.optim_progress",
            "optimizer",
            "zero_grad",
            "total",
            "ready",
        ): {global_step},
        (
            "epoch_loop.automatic_optimization.optim_progress",
            "optimizer",
            "zero_grad",
            "total",
            "completed",
        ): {global_step},
        (
            "epoch_loop.automatic_optimization.optim_progress",
            "optimizer",
            "zero_grad",
            "total",
            "started",
        ): {global_step},
        (
            "epoch_loop.automatic_optimization.optim_progress",
            "optimizer",
            "zero_grad",
            "current",
            "ready",
        ): {0},
        (
            "epoch_loop.automatic_optimization.optim_progress",
            "optimizer",
            "zero_grad",
            "current",
            "completed",
        ): {0},
        (
            "epoch_loop.automatic_optimization.optim_progress",
            "optimizer",
            "zero_grad",
            "current",
            "started",
        ): {0},
        ("epoch_loop.scheduler_progress", "total", "ready"): {global_step},
        (
            "epoch_loop.scheduler_progress",
            "total",
            "completed",
        ): {global_step},
        ("epoch_loop.scheduler_progress", "current", "ready"): {0},
        (
            "epoch_loop.scheduler_progress",
            "current",
            "completed",
        ): {0},
    }
    for progress_scope, expected_value in (
        ("total", total_batches),
        ("current", 0),
    ):
        for field in ("ready", "completed", "started", "processed"):
            expected_progress[
                ("epoch_loop.batch_progress", progress_scope, field)
            ] = {expected_value}
    if not isinstance(fit_loop.get("state_dict"), Mapping):
        return "missing or invalid formal P2 fit_loop state"
    for path, expected_values in expected_progress.items():
        value = _nested_checkpoint_value(fit_loop, path)
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value not in expected_values
        ):
            return "missing or inconsistent formal P2 fit_loop progress"
    if _nested_checkpoint_value(
        fit_loop,
        ("epoch_loop.batch_progress", "is_last_batch"),
    ) is not False:
        return "missing or inconsistent formal P2 fit_loop progress"
    val_batch_progress = fit_loop.get("epoch_loop.val_loop.batch_progress")
    if (
        not isinstance(val_batch_progress, Mapping)
        or val_batch_progress.get("is_last_batch") is not False
    ):
        return "missing or inconsistent formal P2 fit_loop validation progress"
    for field in ("total", "current"):
        progress = val_batch_progress.get(field)
        if (
            not isinstance(progress, Mapping)
            or any(progress.get(counter) != 0 for counter in (
                "ready", "completed", "started", "processed"
            ))
        ):
            return "missing or inconsistent formal P2 fit_loop validation progress"
    return None


def _formal_p2_epoch_boundary_validation_error(checkpoint, _cfg):
    if (
        checkpoint["epoch"] >= 450
        or checkpoint["global_step"] > _P2_FORMAL_ONECYCLE_TOTAL_STEPS
    ):
        return "formal P2 sampler state is outside the completed epoch range"
    expected_global_step = (
        checkpoint["epoch"] + 1
    ) * _P2_FORMAL_OPTIMIZER_STEPS_PER_EPOCH
    if checkpoint["global_step"] != expected_global_step:
        return (
            "formal P2 sampler state is not at a completed epoch boundary: "
            f"expected global_step {expected_global_step}, got "
            f"{checkpoint['global_step']}"
        )
    return None


def _formal_p2_checkpoint_validation_error(checkpoint, cfg):
    for validator in (
        _formal_p2_epoch_boundary_validation_error,
        _formal_p2_loop_validation_error,
        _formal_p2_adamw_validation_error,
        _formal_p2_onecycle_validation_error,
    ):
        validation_error = validator(checkpoint, cfg)
        if validation_error is not None:
            return validation_error
    validation_error = _formal_p2_callback_validation_error(checkpoint, cfg)
    if validation_error is not None:
        return validation_error
    validation_error = _formal_p2_callback_reference_validation_error(
        checkpoint,
        cfg,
    )
    if validation_error is not None:
        return validation_error
    return _p2_sampler_checkpoint_validation_error(checkpoint)


def _resume_checkpoint_validation_error(checkpoint, *, formal_p2=False, cfg=None):
    """Validate static state required before delegating restore to Lightning."""
    if not isinstance(checkpoint, Mapping):
        return "checkpoint payload is not a mapping"

    lightning_version = checkpoint.get("pytorch-lightning_version")
    if not isinstance(lightning_version, str) or not lightning_version:
        return "missing or invalid pytorch-lightning_version"

    for field in ("state_dict", "loops", "callbacks"):
        state = checkpoint.get(field)
        if not isinstance(state, Mapping) or not state:
            return f"missing, empty, or invalid {field}"

    optimizer_states = checkpoint.get("optimizer_states")
    if not isinstance(optimizer_states, list) or not optimizer_states:
        return "missing, empty, or invalid optimizer_states"
    for optimizer_state in optimizer_states:
        if not isinstance(optimizer_state, Mapping):
            return "invalid optimizer state mapping"
        if not isinstance(optimizer_state.get("state"), Mapping):
            return "missing or invalid optimizer state"
        param_groups = optimizer_state.get("param_groups")
        if not isinstance(param_groups, list) or not param_groups:
            return "missing, empty, or invalid optimizer param_groups"
        if any(
            not isinstance(group, Mapping)
            or not isinstance(group.get("params"), list)
            for group in param_groups
        ):
            return "optimizer param_groups require params lists"

    scheduler_states = checkpoint.get("lr_schedulers")
    if (
        not isinstance(scheduler_states, list)
        or not scheduler_states
        or any(
            not isinstance(scheduler_state, Mapping) or not scheduler_state
            for scheduler_state in scheduler_states
        )
    ):
        return "missing, empty, or invalid lr_schedulers"

    for field in ("epoch", "global_step"):
        value = checkpoint.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            return f"missing or invalid {field}"

    if formal_p2:
        if cfg is None:
            return "formal P2 checkpoint validation requires current config"
        expected_lightning_version = P2_RUNTIME_ENVIRONMENT_VERSIONS[
            "pytorch_lightning"
        ]
        if lightning_version != expected_lightning_version:
            return (
                "formal P2 checkpoint pytorch-lightning_version mismatch: "
                f"expected {expected_lightning_version}, got {lightning_version}"
            )
        config_error = _formal_p2_checkpoint_config_validation_error(
            checkpoint,
            cfg,
        )
        if config_error is not None:
            return f"formal P2 checkpoint {config_error}"
        return _formal_p2_checkpoint_validation_error(checkpoint, cfg)
    return None


def find_best_tap_checkpoint(save_dir):
    """Find the best fully resumable t-AP checkpoint."""
    # PyTorch Lightning formats: epoch=X-val_mean_t-AP=Y.ckpt
    patterns = [
        os.path.join(save_dir, "epoch=*-val_mean_t-AP=*.ckpt"),
        os.path.join(save_dir, "*val_mean_t-AP=*.ckpt"),
    ]
    checkpoints = []
    for pattern in patterns:
        checkpoints.extend(glob.glob(pattern))
    checkpoints = sorted(set(checkpoints))
    
    if not checkpoints:
        return None
    
    best_ckpt, best_tap = None, float("-inf")
    for ckpt in checkpoints:
        tap = _checkpoint_tap(ckpt)
        is_valid, _ = _load_resume_candidate(ckpt)
        if is_valid and tap is not None and tap > best_tap:
            best_tap, best_ckpt = tap, ckpt
    return best_ckpt


def find_resume_checkpoint(save_dir, *, formal_p2=False, cfg=None):
    """Return the highest-priority fully resumable checkpoint in ``save_dir``."""
    if formal_p2 and cfg is None:
        raise RuntimeError("Formal P2 checkpoint selection requires current config")
    save_dir = os.fspath(save_dir)
    discovered_paths = set(glob.glob(os.path.join(save_dir, "*.ckpt")))
    if formal_p2:
        expected_callbacks = _formal_p2_model_checkpoint_callbacks(cfg)
        if expected_callbacks is not None:
            for callback in expected_callbacks:
                if (
                    callback.monitor is None
                    and callback._every_n_epochs == 450
                    and callback.filename == P2_EXPERIMENT_NAME
                    and callback.dirpath is not None
                ):
                    final_path = os.path.join(
                        os.fspath(callback.dirpath),
                        _P2_FINAL_CHECKPOINT_BASENAME,
                    )
                    if os.path.isfile(final_path):
                        discovered_paths.add(final_path)
    discovered_paths = sorted(discovered_paths)
    checkpoint_paths = (
        [
            path
            for path in discovered_paths
            if _checkpoint_filename_epoch(path) is not None
            or _LAST_CHECKPOINT_RE.fullmatch(os.path.basename(path)) is not None
            or _LAST_EPOCH_CHECKPOINT_RE.fullmatch(os.path.basename(path)) is not None
            or os.path.basename(path) == _P2_FINAL_CHECKPOINT_BASENAME
            or _checkpoint_tap(path) is not None
        ]
        if formal_p2
        else discovered_paths
    )
    latest_candidates = []
    tap_candidates = []
    for checkpoint_path in checkpoint_paths:
        is_valid, checkpoint = _load_resume_candidate(
            checkpoint_path,
            formal_p2=formal_p2,
            cfg=cfg,
        )
        if not is_valid:
            continue

        basename = os.path.basename(checkpoint_path)
        filename_epoch = _checkpoint_filename_epoch(checkpoint_path)
        version = _checkpoint_version(checkpoint_path)
        is_latest_candidate = (
            filename_epoch is not None
            or _LAST_CHECKPOINT_RE.fullmatch(basename) is not None
            or _LAST_EPOCH_CHECKPOINT_RE.fullmatch(basename) is not None
            or basename == _P2_FINAL_CHECKPOINT_BASENAME
        )
        if is_latest_candidate:
            latest_candidates.append(
                (
                    checkpoint["epoch"],
                    checkpoint["global_step"],
                    version,
                    checkpoint_path,
                )
            )

        tap = _checkpoint_tap(checkpoint_path)
        if tap is not None:
            tap_candidates.append(
                (tap, checkpoint["global_step"], version, checkpoint_path)
            )

    if latest_candidates:
        return max(latest_candidates)[-1]
    if tap_candidates:
        return max(tap_candidates)[-1]
    if checkpoint_paths:
        raise RuntimeError(
            f"Found {len(checkpoint_paths)} checkpoint file(s) in {save_dir}, "
            "but none are fully resumable. Refusing to start from scratch."
        )
    return None


def get_parameters(cfg: DictConfig):    
    # Environment setup for optimal performance
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    # os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "0")  # Disable for better performance
    # os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    
    # parsing input parameters
    seed_everything(cfg.general.seed)

    # getting basic configuration
    if cfg.general.get("gpus", None) is None:
        cfg.general.gpus = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    
    # Only rank 0 creates directories
    if rank_zero_only.rank == 0:
        os.makedirs(cfg.general.save_dir, exist_ok=True)
    
    model = InstanceSegmentation(cfg)
    
    # Load checkpoints
    if cfg.general.backbone_checkpoint:
        print("loading backbone checkpoint")
        cfg, model = load_backbone_checkpoint_with_missing_or_exsessive_keys(cfg, model)
    if cfg.general.checkpoint:
        print("loading checkpoint")
        cfg, model = load_checkpoint_with_missing_or_exsessive_keys(cfg, model)
    
    return cfg, model

@hydra.main(version_base="1.2", config_path="conf", config_name="config_base_instance_segmentation")
def train(cfg: DictConfig):
    formal_p2_training = _is_formal_p2_training(cfg)
    _enforce_formal_p2_training(cfg)
    formal_resume_checkpoint = None
    if formal_p2_training:
        formal_resume_checkpoint = find_resume_checkpoint(
            cfg.general.save_dir,
            formal_p2=True,
            cfg=cfg,
        )
        if formal_resume_checkpoint is not None:
            formal_resume_checkpoint = require_p2_resume_checkpoint(
                cfg,
                formal_resume_checkpoint,
            )

    # Create logger - Lightning handles distributed initialization
    loggers = [hydra.utils.instantiate(logger_cfg) for logger_cfg in cfg.logging]

    # Get the W&B generated name and update config
    # W&B auto-generates meaningful names for sweep runs, only reset if the name is at default 
    if rank_zero_only.rank == 0:
        wandb_logger = next(
            (
                logger
                for logger in loggers
                if getattr(getattr(logger, "experiment", None), "sweep_id", None)
            ),
            None,
        )
        if wandb_logger and cfg.general.experiment_name == "DEBUG":
            try:
                run_name = wandb_logger.experiment.name
                cfg.general.experiment_name = run_name
                # save heirachical for organization
                cfg.general.save_dir = f"saved/{wandb_logger.experiment.sweep_id}/{run_name}"
            except: 
                pass

            print(f"Experiment name: {cfg.general.experiment_name}")
            print(f"Save Dir: {cfg.general.save_dir}")

    cfg, model = get_parameters(cfg)

    # update the save dir to the exp[eriment name created by the logger]
    if rank_zero_only.rank == 0:
        config_dict = flatten_dict(OmegaConf.to_container(cfg, resolve=True))
        for logger in loggers:
            if hasattr(logger, 'log_hyperparams'):
                logger.log_hyperparams(config_dict)
    
    # Callbacks - use only built-in callbacks for DDP safety
    callbacks = [hydra.utils.instantiate(cb) for cb in cfg.callbacks]
    
    trainer = Trainer(
        logger=loggers,
        accelerator='gpu',
        devices=cfg.general.gpus,
        callbacks=callbacks,
        default_root_dir=cfg.general.save_dir,
        **cfg.trainer
    )
    
    # Resume from checkpoint if exists - DDP-safe way
    ckpt_path = (
        formal_resume_checkpoint
        if formal_p2_training
        else find_resume_checkpoint(cfg.general.save_dir)
    )
    
    if ckpt_path:
        print(f"Resuming from checkpoint: {ckpt_path}")
    else:
        print("No checkpoint found, starting from scratch")
    
    fit_kwargs = {"ckpt_path": ckpt_path}
    if ckpt_path is not None:
        fit_kwargs["weights_only"] = False
    trainer.fit(model, **fit_kwargs)

@hydra.main(version_base="1.2", config_path="conf", config_name="config_base_instance_segmentation")
def test(cfg: DictConfig):
    cfg, model = get_parameters(cfg)
    
    # Ensure model is frozen for evaluation
    model.eval()
    model.model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    
    loggers = [hydra.utils.instantiate(logger_cfg) for logger_cfg in cfg.logging]

    if rank_zero_only.rank == 0:
        config_dict = flatten_dict(OmegaConf.to_container(cfg, resolve=True))
        for logger in loggers:
            if hasattr(logger, 'log_hyperparams'):
                logger.log_hyperparams(config_dict)
    
    trainer = Trainer(
        accelerator='gpu',
        devices=cfg.general.gpus,
        logger=loggers,
        default_root_dir=cfg.general.save_dir,
        **cfg.trainer
    )
    
    trainer.test(model)

@hydra.main(version_base="1.2", config_path="conf", config_name="config_base_instance_segmentation")
def main(cfg: DictConfig):
    train(cfg) if cfg.general.train_mode else test(cfg)

if __name__ == "__main__":
    main()
