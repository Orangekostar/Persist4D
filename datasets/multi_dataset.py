import logging
from typing import List, Dict, Any
from torch.utils.data import Dataset, ConcatDataset, WeightedRandomSampler
from datasets.semseg import SemanticSegmentationDataset

logger = logging.getLogger(__name__)


class MultiDataset(ConcatDataset):
    """
    A ConcatDataset with weighted sampling using WeightedRandomSampler.
    
    This class combines multiple datasets and provides weighted sampling across them.
    Can be instantiated from configuration files.
    """
    
    def __init__(self, datasets: List[Dataset], weights: List[float] = None):
        """
        Initialize the multi-dataset with weighted sampling.
        
        Args:
            datasets: List of dataset instances to combine.
            weights: Optional weights for each dataset. If None, equal weights are used.
        """
        super().__init__(datasets)
        self.weights = weights or [1.0] * len(datasets)
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
        common_configs = {k: v for k, v in kwargs.items() if k not in ['datasets', 'weights']}
        
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
        
        return cls(datasets=datasets, weights=weights)
    
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
                logger.warning(f"Dataset has zero size, skipping")
        
        # Calculate num_samples based on weighted contributions
        # This ensures the epoch length reflects the weighted combination
        # For each dataset, its contribution = weight (since weights are normalized to represent epoch ratio)
        if len(self.datasets) > 0 and len(self.weights) > 0:
            primary_dataset_size = len(self.datasets[0])
            primary_weight = self.weights[0]
            # Total samples = primary_size + (sum of other weights / primary_weight) * primary_size
            # This way, if scannet has weight 0.05, it contributes 5% as many samples as rio
            other_weight_sum = sum(self.weights[1:]) if len(self.weights) > 1 else 0
            num_samples = int(primary_dataset_size * (1 + other_weight_sum / primary_weight))
        else:
            num_samples = len(sample_weights)
        
        self.sampler = WeightedRandomSampler(sample_weights, num_samples)
    
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
