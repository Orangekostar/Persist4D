import os
import sys
import glob
import re
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


def find_best_tap_checkpoint(save_dir):
    """Find best t-AP checkpoint as fallback if last checkpoints are corrupted."""
    # PyTorch Lightning formats: epoch=X-val_mean_t-AP=Y.ckpt
    patterns = [
        os.path.join(save_dir, "epoch=*-val_mean_t-AP=*.ckpt"),
        os.path.join(save_dir, "*val_mean_t-AP=*.ckpt"),
    ]
    checkpoints = []
    for pattern in patterns:
        checkpoints.extend(glob.glob(pattern))
    checkpoints = list(set(checkpoints))
    
    if not checkpoints:
        return None
    
    best_ckpt, best_tap = None, -1.0
    for ckpt in checkpoints:
        match = re.search(r'val_mean_t-AP=([\d.]+)', os.path.basename(ckpt))
        if match:
            try:
                tap = float(match.group(1))
                if tap > best_tap:
                    best_tap, best_ckpt = tap, ckpt
            except ValueError:
                continue
    return best_ckpt


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
    ckpt_path = None
    try:
        if os.path.exists(f"{cfg.general.save_dir}/last.ckpt"):
            torch.load(f"{cfg.general.save_dir}/last.ckpt", map_location="cpu", weights_only=False)
            ckpt_path = f"{cfg.general.save_dir}/last.ckpt"
        elif os.path.exists(f"{cfg.general.save_dir}/last-epoch.ckpt"):
            torch.load(f"{cfg.general.save_dir}/last-epoch.ckpt", map_location="cpu", weights_only=False)
            ckpt_path = f"{cfg.general.save_dir}/last-epoch.ckpt"
    except Exception as e:
        # If corrupted, fallback to best t-AP checkpoint
        print(f"Checkpoint corrupted ({e}), trying best t-AP checkpoint...")
        ckpt_path = find_best_tap_checkpoint(cfg.general.save_dir)
        if ckpt_path:
            # Verify the fallback checkpoint can be loaded
            try:
                torch.load(ckpt_path, map_location="cpu", weights_only=False)
            except Exception as e2:
                print(f"Best t-AP checkpoint also corrupted ({e2}), starting from scratch")
                ckpt_path = None
    
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