"""
Auto-collation function that automatically selects the appropriate collation function
based on the backbone configuration.
"""

import hydra
from omegaconf import DictConfig

class AutoCollate:
    """
    Automatically selects the appropriate collation function based on the backbone.
    - Uses datasets.minkowski_utils.VoxelizeCollate for minkowski backbone
    - Uses datasets.pointcept_utils.VoxelizeCollate for pointcept backbone
    """
    
    def __init__(self, backbone_target=None, **kwargs):
        self.kwargs = kwargs
        
        # Determine which collation function to use based on backbone
        if backbone_target is None:
            # Try to get from hydra config if available
            try:
                from hydra import compose, initialize_config_dir
                import os
                # This is a fallback - in practice, backbone_target should be passed
                backbone_target = "models.Res16UNet34C"  # default to minkowski
            except:
                backbone_target = "models.Res16UNet34C"  # default to minkowski
        
        if "PointceptBackbone" in backbone_target:
            from datasets.pointcept_utils import VoxelizeCollate as VoxelizeCollateSonata
            self.collate_fn = VoxelizeCollateSonata(**kwargs)
        else:
            # Default to regular VoxelizeCollate for minkowski and other backbones
            from datasets.minkowski_utils import VoxelizeCollate
            self.collate_fn = VoxelizeCollate(**kwargs)
    
    def __call__(self, batch):
        return self.collate_fn(batch)


def create_auto_collate(backbone_target=None, **kwargs):
    """
    Factory function to create an AutoCollate instance.
    This can be used in Hydra configs.
    """
    return AutoCollate(backbone_target, **kwargs)
