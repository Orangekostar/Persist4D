import logging
import math
from numbers import Integral, Real
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset, ConcatDataset, WeightedRandomSampler
from datasets.semseg import SemanticSegmentationDataset

logger = logging.getLogger(__name__)


class MultiDataset(ConcatDataset):
    """
    A ConcatDataset with weighted sampling using WeightedRandomSampler.
    
    This class combines multiple datasets and provides weighted sampling across them.
    Can be instantiated from configuration files.
    """
    
    def __init__(
        self,
        datasets: List[Dataset],
        weights: List[float] = None,
        epoch_sample_multiple: int = None,
        sampler_seed: int = None,
        fail_closed: bool = False,
    ):
        """
        Initialize the multi-dataset with weighted sampling.
        
        Args:
            datasets: List of dataset instances to combine.
            weights: Optional weights for each dataset. If None, equal weights are used.
            epoch_sample_multiple: Optional multiple used to align the sampler's
                per-epoch sample count downward.
            sampler_seed: Optional seed for reproducible sampler index streams.
            fail_closed: Whether to reject incomplete or invalid dataset mixes.
        """
        if fail_closed:
            if datasets is None:
                raise ValueError("MultiDataset requires at least one dataset")
            datasets = list(datasets)
            if not datasets:
                raise ValueError("MultiDataset requires at least one dataset")

            if weights is None:
                weights = [1.0] * len(datasets)
            else:
                weights = list(weights)
            if len(weights) != len(datasets):
                raise ValueError("datasets and weights must have the same length")

            for index, dataset in enumerate(datasets):
                if len(dataset) == 0:
                    raise ValueError(f"dataset at index {index} has zero length")
            for index, weight in enumerate(weights):
                if (
                    isinstance(weight, bool)
                    or not isinstance(weight, Real)
                    or not math.isfinite(weight)
                    or weight <= 0
                ):
                    raise ValueError(
                        f"weight at index {index} must be a finite positive number"
                    )
        if epoch_sample_multiple is not None and (
            isinstance(epoch_sample_multiple, bool)
            or not isinstance(epoch_sample_multiple, Integral)
            or epoch_sample_multiple <= 0
        ):
            raise ValueError("epoch_sample_multiple must be a positive integer")
        if sampler_seed is not None and (
            isinstance(sampler_seed, bool)
            or not isinstance(sampler_seed, Integral)
        ):
            raise ValueError("sampler_seed must be an integer")

        super().__init__(datasets)
        self.weights = weights if fail_closed else weights or [1.0] * len(datasets)
        self.epoch_sample_multiple = (
            None if epoch_sample_multiple is None else int(epoch_sample_multiple)
        )
        self.sampler_seed = None if sampler_seed is None else int(sampler_seed)
        self.fail_closed = fail_closed
        self._setup_sampler()
    
    @classmethod
    def from_config(cls, **kwargs) -> 'MultiDataset':
        """
        Create MultiDataset from configuration.
        
        Args:
            **kwargs: Alternative way to pass configuration as keyword arguments
                - datasets: List of dataset configurations
                - weights: Optional weights for each dataset
                - Common configs that will be applied to all datasets
        
        Returns:
            MultiDataset instance
        """
        # Use keyword arguments (for Hydra compatibility)
        datasets_config = kwargs.get('datasets', [])
        weights = kwargs.get('weights', None)
        epoch_sample_multiple = kwargs.get('epoch_sample_multiple', None)
        sampler_seed = kwargs.get('sampler_seed', None)
        fail_closed = kwargs.get('fail_closed', False)
        common_configs = {
            k: v
            for k, v in kwargs.items()
            if k not in [
                'datasets',
                'weights',
                'epoch_sample_multiple',
                'sampler_seed',
            ]
        }
        
        # Create individual datasets
        datasets = []
        for dataset_config in datasets_config:
            # Create a copy of the config to avoid modifying the original
            config_copy = dataset_config.copy()
            
            # Start with common configs as base
            merged_config = common_configs.copy()
            
            # Override with dataset-specific configs (dataset config takes precedence)
            merged_config.update({k: v for k, v in config_copy.items() if k not in ['target', '_target_']})
            
            # Instantiate the dataset
            target = config_copy.get('target') or config_copy.get('_target_')
            if target == 'datasets.semseg.SemanticSegmentationDataset':
                dataset = SemanticSegmentationDataset(**merged_config)
            else:
                raise ValueError(f"Unsupported dataset target: {target}")
            
            datasets.append(dataset)
        
        return cls(
            datasets=datasets,
            weights=weights,
            epoch_sample_multiple=epoch_sample_multiple,
            sampler_seed=sampler_seed,
            fail_closed=fail_closed,
        )
    
    def _setup_sampler(self):
        """Set up the WeightedRandomSampler for weighted sampling across datasets.
        
        Weights are normalized by dataset size so that they represent the desired
        ratio of samples per epoch, not per sample. For example, if dataset A has 
        weight 1.0 and dataset B has weight 0.05, dataset B will contribute 5% 
        as many samples as dataset A per epoch, regardless of their sizes.
        
        The num_samples is set based on the primary dataset (first dataset) to ensure
        epoch length is consistent regardless of secondary dataset sizes.
        """
        sample_weights = []
        primary_dataset_size = len(self.datasets[0]) if len(self.datasets) > 0 else 0

        for dataset, weight in zip(self.datasets, self.weights):
            dataset_size = len(dataset)
            if dataset_size > 0:
                # Normalize weight by dataset size so weights represent desired ratio per epoch
                # This ensures a dataset with weight 0.05 contributes 5% as much as one with weight 1.0
                normalized_weight = weight / dataset_size
                sample_weights.extend([normalized_weight] * dataset_size)
            else:
                logger.warning("Dataset has zero size, skipping")
        
        # Calculate num_samples based on weighted contributions
        # This ensures the epoch length reflects the weighted combination
        # For each dataset, its contribution = weight (since weights are normalized to represent epoch ratio)
        if len(self.datasets) > 0 and len(self.weights) > 0:
            primary_weight = self.weights[0]
            # Total samples = primary_size + (sum of other weights / primary_weight) * primary_size
            # This way, if scannet has weight 0.05, it contributes 5% as many samples as rio
            other_weight_sum = sum(self.weights[1:]) if len(self.weights) > 1 else 0
            num_samples = int(primary_dataset_size * (1 + other_weight_sum / primary_weight))
        else:
            num_samples = len(sample_weights)
        if self.epoch_sample_multiple is not None:
            if self.epoch_sample_multiple > num_samples:
                raise ValueError(
                    "epoch_sample_multiple cannot exceed the unaligned epoch "
                    f"sample count {num_samples}"
                )
            num_samples -= num_samples % self.epoch_sample_multiple
        
        generator = None
        if self.sampler_seed is not None:
            generator = torch.Generator()
            generator.manual_seed(self.sampler_seed)
        self.sampler = WeightedRandomSampler(
            sample_weights,
            num_samples,
            generator=generator,
        )
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about the datasets."""
        info = {
            'num_datasets': len(self.datasets),
            'dataset_sizes': [len(dataset) for dataset in self.datasets],
            'total_size': len(self),
            'weights': self.weights,
            'sampling_method': 'weighted_random_sampler'
        }
        return info
