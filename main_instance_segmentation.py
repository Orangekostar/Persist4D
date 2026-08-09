import os
import sys
import glob
import re
from collections.abc import Mapping

import torch
import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
from trainer.trainer import InstanceSegmentation
from pytorch_lightning import Trainer, seed_everything
from utils.utils import (
    flatten_dict,
    load_checkpoint_with_missing_or_exsessive_keys,
    load_backbone_checkpoint_with_missing_or_exsessive_keys,
)
from pytorch_lightning.utilities.rank_zero import rank_zero_only

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


def _load_resume_candidate(checkpoint_path):
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
    except Exception as error:
        print(f"Skipping unreadable checkpoint {checkpoint_path}: {error}")
        return False, None

    validation_error = _resume_checkpoint_validation_error(checkpoint)
    if validation_error is not None:
        print(
            f"Skipping non-resumable checkpoint {checkpoint_path}: "
            f"{validation_error}"
        )
        return False, None
    return True, checkpoint


def _resume_checkpoint_validation_error(checkpoint):
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


def find_resume_checkpoint(save_dir):
    """Return the highest-priority fully resumable checkpoint in ``save_dir``."""
    save_dir = os.fspath(save_dir)
    checkpoint_paths = sorted(set(glob.glob(os.path.join(save_dir, "*.ckpt"))))
    latest_candidates = []
    tap_candidates = []
    for checkpoint_path in checkpoint_paths:
        is_valid, checkpoint = _load_resume_candidate(checkpoint_path)
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
    # Create logger - Lightning handles distributed initialization
    loggers = [hydra.utils.instantiate(logger_cfg) for logger_cfg in cfg.logging]

    # Get the W&B generated name and update config
    # W&B auto-generates meaningful names for sweep runs, only reset if the name is at default 
    if rank_zero_only.rank == 0:
        wandb_logger = next((l for l in loggers if hasattr(l, 'experiment')), None)
        if wandb_logger and wandb_logger.experiment.sweep_id and cfg.general.experiment_name == "DEBUG":
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
    ckpt_path = find_resume_checkpoint(cfg.general.save_dir)
    
    if ckpt_path:
        print(f"Resuming from checkpoint: {ckpt_path}")
    else:
        print("No checkpoint found, starting from scratch")
    
    trainer.fit(model, ckpt_path=ckpt_path)

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
