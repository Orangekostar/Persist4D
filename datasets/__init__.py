"""
Datasets package for Mask3D.

This package contains various dataset implementations and utilities for 3D instance segmentation.
"""

# Import key classes for easier access
from .auto_collate import AutoCollate, create_auto_collate
from .multi_dataset import MultiDataset
from .semseg import SemanticSegmentationDataset
from .task_memory_episode import (
    NativeEpisodeMaster,
    StageMeta,
    TaskMemoryEpisodeBatch,
    TaskMemoryEpisodeCollator,
    TaskMemoryEpisodeDataset,
    TaskMemoryEpisodeSpec,
    build_native_episode_masters,
    build_task_memory_draw_plan,
)

__all__ = [
    'AutoCollate',
    'MultiDataset',
    'NativeEpisodeMaster',
    'SemanticSegmentationDataset',
    'StageMeta',
    'TaskMemoryEpisodeBatch',
    'TaskMemoryEpisodeCollator',
    'TaskMemoryEpisodeDataset',
    'TaskMemoryEpisodeSpec',
    'build_native_episode_masters',
    'build_task_memory_draw_plan',
    'create_auto_collate',
]
