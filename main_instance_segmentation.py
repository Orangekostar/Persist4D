import os
import sys
import glob
import hashlib
import re
import tempfile
import typing
from collections import defaultdict
from collections.abc import Mapping
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
from utils.utils import (
    flatten_dict,
    load_checkpoint_with_missing_or_exsessive_keys,
    load_backbone_checkpoint_with_missing_or_exsessive_keys,
)
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from utils.p2_preflight import (
    P2_CONFIG_NAME,
    P2_EXPERIMENT_NAME,
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
_P2_TRAIN_SAMPLER_CHECKPOINT_KEY = "p2_train_sampler_generator"
_P2_TRAIN_SAMPLER_CHECKPOINT_SCHEMA_VERSION = 1
_P2_TRAIN_SAMPLER_RESUME_SCOPE = "completed_epoch_boundary_only"
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
    )
    if validation_error is not None:
        raise RuntimeError(
            "Formal P2 resume checkpoint is not fully resumable: "
            f"{validation_error}"
        )
    hyper_parameters = checkpoint.get("hyper_parameters")
    if not (OmegaConf.is_config(hyper_parameters) or isinstance(hyper_parameters, Mapping)):
        raise RuntimeError(
            "Formal P2 resume checkpoint has no compatible hyper_parameters"
        )
    marker = hyper_parameters.get("p2_preflight")
    if not isinstance(marker, Mapping) or marker.get("target") != P2_TARGET:
        raise RuntimeError(
            "Formal P2 resume checkpoint has no matching P2 profile provenance"
        )
    try:
        from utils.p2_preflight import p2_training_semantic_sha256

        checkpoint_config_sha256 = p2_training_semantic_sha256(hyper_parameters)
        current_config_sha256 = p2_training_semantic_sha256(cfg)
    except Exception as error:
        raise RuntimeError(
            "Formal P2 resume checkpoint config_sha256 is unavailable"
        ) from error
    if checkpoint_config_sha256 != current_config_sha256:
        raise RuntimeError(
            "Formal P2 resume checkpoint config_sha256 mismatch: expected "
            f"{current_config_sha256}, got {checkpoint_config_sha256}"
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


def _load_resume_candidate(checkpoint_path, *, formal_p2=False):
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

    validation_error = _resume_checkpoint_validation_error(
        checkpoint,
        formal_p2=formal_p2,
    )
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


def _resume_checkpoint_validation_error(checkpoint, *, formal_p2=False):
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
        return _p2_sampler_checkpoint_validation_error(checkpoint)
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


def find_resume_checkpoint(save_dir, *, formal_p2=False):
    """Return the highest-priority fully resumable checkpoint in ``save_dir``."""
    save_dir = os.fspath(save_dir)
    discovered_paths = sorted(set(glob.glob(os.path.join(save_dir, "*.ckpt"))))
    checkpoint_paths = (
        [
            path
            for path in discovered_paths
            if _checkpoint_filename_epoch(path) is not None
            or _LAST_CHECKPOINT_RE.fullmatch(os.path.basename(path)) is not None
            or _LAST_EPOCH_CHECKPOINT_RE.fullmatch(os.path.basename(path)) is not None
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
