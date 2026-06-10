import sys

if sys.version_info[:2] >= (3, 8):
    from collections.abc import MutableMapping
else:
    from collections import MutableMapping

import torch
from loguru import logger
import collections


def flatten_dict(d, parent_key="", sep="_"):
    """
    https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def load_baseline_model(cfg, model):
    # if it is Minkoski weights
    cfg.model.in_channels = 3
    cfg.model.config.conv1_kernel_size = 5
    cfg.data.add_normals = False
    cfg.data.train_dataset.color_mean_std = [(0.5, 0.5, 0.5), (1, 1, 1)]
    cfg.data.validation_dataset.color_mean_std = [(0.5, 0.5, 0.5), (1, 1, 1)]
    cfg.data.test_dataset.color_mean_std = [(0.5, 0.5, 0.5), (1, 1, 1)]
    cfg.data.voxel_size = 0.02
    model = model(cfg)
    state_dict = torch.load(cfg.general.checkpoint, weights_only=False)["state_dict"]
    model.model.load_state_dict(state_dict)
    return cfg, model


def load_backbone_checkpoint_with_missing_or_exsessive_keys(cfg, model):
    """Load backbone checkpoint with handling for missing, excessive, and shape-mismatched keys.
    
    Handles both Minkowski and Sonata backbones:
    - Minkowski: backbone.conv0, backbone.bn0, backbone.block1, etc.
    - Sonata: backbone.model.embedding.mask_token, backbone.model.enc.*, etc.
    - Old configs: model.backbone.* -> backbone.*
    """
    state_dict = torch.load(cfg.general.backbone_checkpoint, weights_only=False)["state_dict"]
    model_dict = model.state_dict()
    
    # Handle old config format: model.backbone.* -> backbone.*
    if any(key.startswith('model.backbone.') for key in state_dict.keys()):
        logger.info("Converting old format 'model.backbone.*' keys to 'backbone.*'")
        state_dict = {key.replace("model.backbone.", "backbone."): value 
                     for key, value in state_dict.items() 
                     if key.startswith('model.backbone.')}
    
    # Build final state dict for backbone parameters
    final_state_dict = {}
    
    # Process model keys (missing backbone keys will use model's random initialization)
    for key in model_dict:
        if key.startswith('backbone.'):
            backbone_key = key.replace("backbone.", "")
            if backbone_key not in state_dict:
                logger.warning(f"Backbone key not found in checkpoint, using random initialization: {key}")
                final_state_dict[key] = model_dict[key]
            elif state_dict[backbone_key].shape != model_dict[key].shape:
                logger.warning(f"Backbone shape mismatch for {key}: {state_dict[backbone_key].shape} vs {model_dict[key].shape}")
                final_state_dict[key] = model_dict[key]
            else:
                final_state_dict[key] = state_dict[backbone_key]
        else:
            # Keep non-backbone parameters as-is
            final_state_dict[key] = model_dict[key]
    
    # Report excessive keys from checkpoint
    for key in state_dict:
        if f"backbone.{key}" not in model_dict:
            logger.warning(f"Excessive backbone key in checkpoint (ignored): {key}")
    
    model.load_state_dict(final_state_dict)
    return cfg, model


def load_checkpoint_with_missing_or_exsessive_keys(cfg, model):
    """Load checkpoint with handling for missing, excessive, and shape-mismatched keys."""
    state_dict = torch.load(cfg.general.checkpoint, weights_only=False)["state_dict"]
    model_dict = model.state_dict()
    
    # Handle parameter name mismatches
    state_dict = handle_mismatch(state_dict, model_dict)
    
    # Build final state dict, handling missing/excessive keys and shape mismatches
    final_state_dict = {}
    
    # Process model keys (missing keys will use model's random initialization)
    for key in model_dict:
        if key not in state_dict:
            logger.warning(f"Key not found in checkpoint, using random initialization: {key}")
            final_state_dict[key] = model_dict[key]
        elif state_dict[key].shape != model_dict[key].shape:
            logger.warning(f"Shape mismatch for {key}: {state_dict[key].shape} vs {model_dict[key].shape}")
            final_state_dict[key] = model_dict[key]
        else:
            final_state_dict[key] = state_dict[key]
    
    # Report excessive keys from checkpoint
    for key in state_dict:
        if key not in model_dict:
            logger.warning(f"Excessive key in checkpoint (ignored): {key}")
    
    model.load_state_dict(final_state_dict)
    return cfg, model

def handle_mismatch(state_dict, model_state_dict):
    # Handle parameter name changes and shape mismatches
    final_state_dict = dict(state_dict)  # Start with original state_dict
    
    # Handle kernel->weight rename and transpose
    if "model.mask_features_head.kernel" in state_dict and "model.mask_features_head.weight" in model_state_dict:
        kernel = state_dict["model.mask_features_head.kernel"]
        final_state_dict["model.mask_features_head.weight"] = kernel.transpose(0, 1)  # [96,128] -> [128,96]
        del final_state_dict["model.mask_features_head.kernel"]
    
    # Handle bias squeeze
    if "model.mask_features_head.bias" in state_dict:
        bias = state_dict["model.mask_features_head.bias"]
        if bias.shape != model_state_dict["model.mask_features_head.bias"].shape:
            final_state_dict["model.mask_features_head.bias"] = bias.squeeze()  # [1,128] -> [128]
    
    return final_state_dict
    


def freeze_until(net, param_name: str = None):
    """
    Freeze net until param_name
    https://opendatascience.slack.com/archives/CGK4KQBHD/p1588373239292300?thread_ts=1588105223.275700&cid=CGK4KQBHD
    Args:
        net:
        param_name:
    Returns:
    """
    found_name = False
    for name, params in net.named_parameters():
        if name == param_name:
            found_name = True
        params.requires_grad = found_name
