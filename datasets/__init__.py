"""
Datasets package for Mask3D.

This package contains various dataset implementations and utilities for 3D instance segmentation.
"""

# Import key classes for easier access
from .auto_collate import create_auto_collate, AutoCollate
from .semseg import SemanticSegmentationDataset
from .multi_dataset import MultiDataset

__all__ = [
    'create_auto_collate',
    'AutoCollate',
    'SemanticSegmentationDataset',
    'MultiDataset'
]
