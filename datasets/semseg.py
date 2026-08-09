import logging
from pathlib import Path
from random import random
from typing import List, Optional, Tuple, Union


import numpy
import torch

import albumentations as A
import numpy as np
import scipy
import volumentations as V
import yaml

from yaml import CLoader as Loader
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class SemanticSegmentationDataset(Dataset):
    """Docstring for SemanticSegmentationDataset."""

    def __init__(
        self,
        dataset_name="scannet",
        data_dir: Optional[Union[str, Tuple[str]]] = "data/processed/scannet",
        label_db_filepath: Optional[
            str
        ] = "configs/scannet_preprocessing/label_database.yaml",
        # mean std values from scannet
        color_mean_std: Optional[Union[str, Tuple[Tuple[float]]]] = (
            (0.47793125906962, 0.4303257521323044, 0.3749598901421883),
            (0.2834475483823543, 0.27566157565723015, 0.27018971370874995),
        ),
        mode: Optional[str] = "train",
        add_colors: Optional[bool] = True,
        add_normals: Optional[bool] = True,
        add_raw_coordinates: Optional[bool] = False,
        add_instance: Optional[bool] = False,
        num_labels: Optional[int] = -1,
        ignore_label: Optional[Union[int, Tuple[int]]] = 255,
        volume_augmentations_path: Optional[str] = None,
        image_augmentations_path: Optional[str] = None,
        filter_out_classes=[],
        label_offset=0,
        temporal_window=2,
        num_changes: Optional[int] = -1,
        change_label_db_filepath: Optional[
            str
        ] = "data/rio/change_label_database.yaml",
        max_points_per_sample: Optional[int] = None,
    ):

        self.dataset_name = dataset_name

        # used by voxelizer
        self.filter_out_classes = filter_out_classes
        self.label_offset = label_offset

        self.mode = mode
        self.data_dir = data_dir
        if type(data_dir) == str:
            self.data_dir = [self.data_dir]
        self.ignore_label = ignore_label
        self.add_colors = add_colors
        self.add_normals = add_normals
        self.add_instance = add_instance
        self.add_raw_coordinates = add_raw_coordinates
        self.temporal_window = temporal_window
        self.max_points_per_sample = max_points_per_sample

        # loading database files
        self._data = []
        for database_path in self.data_dir:
            database_path = Path(database_path)
            database_filepath = database_path / f"{mode}_database.yaml"
            if not database_filepath.exists():
                raise FileNotFoundError(
                    "Required dataset database does not exist: "
                    f"{database_filepath}. Generate it before loading the dataset."
                )
            
            # Load data from this directory
            data_from_dir = self._load_yaml(database_filepath)
            if not isinstance(data_from_dir, list) or not data_from_dir:
                raise ValueError(
                    f"Dataset split database {database_filepath} must be a "
                    "non-empty list"
                )
            
            # For multi-dataset configurations, add dataset source information
            if len(self.data_dir) > 1:
                # Extract dataset name from the directory path
                dataset_source = database_path.name
                for item in data_from_dir:
                    item["dataset_source"] = dataset_source
            
            self._data.extend(data_from_dir)
        labels = self._load_yaml(Path(label_db_filepath))
        self._labels = self._select_correct_labels(labels, num_labels)
        self.color_map = {int(k): tuple(v["color"]) for k, v in labels.items()}
        self.color_map[self.ignore_label] = (255, 255, 255)

        if num_changes != -1:
            change_labels = self._load_yaml(Path(change_label_db_filepath))
            self._change_labels = self._select_correct_labels(change_labels, num_changes)


        if Path(str(color_mean_std)).exists():
            color_mean_std = self._load_yaml(color_mean_std)
            color_mean, color_std = (
                tuple(color_mean_std["mean"]),
                tuple(color_mean_std["std"]),
            )
        elif len(color_mean_std[0]) == 3 and len(color_mean_std[1]) == 3:
            color_mean, color_std = color_mean_std[0], color_mean_std[1]
        else:
            logger.error(
                "pass mean and std as tuple of tuples, or as an .yaml file"
            )

        # augmentations
        self.volume_augmentations = V.NoOp()
        if (volume_augmentations_path is not None) and (
            volume_augmentations_path != "none"
        ):
            self.volume_augmentations = V.load(
                Path(volume_augmentations_path), data_format="yaml"
            )
        self.image_augmentations = A.NoOp()
        if (image_augmentations_path is not None) and (
            image_augmentations_path != "none"
        ):
            self.image_augmentations = A.load(
                Path(image_augmentations_path), data_format="yaml"
            )
        # mandatory color augmentation
        if add_colors:
            self.normalize_color = A.Normalize(mean=color_mean, std=color_std)

        name_idx_mapping = {self._name_scene(scan_idx):scan_idx for scan_idx in range(len(self.data))}
        if self.temporal_window <= 1:
            self.sequence_indices = np.expand_dims(np.arange(len(self.data)), axis=1)
            self.sequence_names = list(name_idx_mapping.keys())
            # empty list of change file locations 
            self.change_files = [None] * len(self.data)
            self.ambiguities = [None] * len(self.data)

        else:
            sequence_type = "sliding"

            self.sequence_names = []
            all_sequence_indices = []
            self.change_files = []
            self.ambiguities = []

            # Process each data directory
            for dir_path in self.data_dir:
                database_path = (
                    Path(dir_path)
                    / f"sequence_database_{sequence_type}_{self.temporal_window}.yaml"
                )
                sequence_db = self._load_yaml(database_path)
                if not isinstance(sequence_db, dict) or not sequence_db:
                    raise ValueError(
                        f"Temporal sequence database {database_path} must be a "
                        "non-empty mapping"
                    )

                dir_mode = []
                for sequence, entry in sequence_db.items():
                    if not isinstance(entry, dict):
                        raise ValueError(
                            f"Temporal sequence database {database_path}: "
                            f"sequence '{sequence}' must map to a record"
                        )
                    if entry.get("type") == mode:
                        dir_mode.append((sequence, entry))
                if not dir_mode:
                    raise ValueError(
                        f"Temporal sequence database {database_path} has no "
                        f"sequences for mode '{mode}'"
                    )

                dir_sequence_names = []
                dir_change_files = []
                dir_ambiguities = []
                dir_sequence_indices = []
                for sequence, entry in dir_mode:
                    if not isinstance(sequence, str):
                        raise ValueError(
                            f"Temporal sequence database {database_path}: "
                            f"sequence key {sequence!r} must be a string"
                        )
                    names = sequence.split("-")
                    if len(names) != self.temporal_window:
                        raise ValueError(
                            f"Temporal sequence database {database_path}: "
                            f"sequence '{sequence}' expected {self.temporal_window} "
                            f"scan names, got {len(names)}"
                        )

                    resolved_indices = []
                    for name in names:
                        lookup_name = name
                        if lookup_name not in name_idx_mapping and len(self.data_dir) > 1:
                            prefixed_name = f"{Path(dir_path).name}_{name}"
                            if prefixed_name in name_idx_mapping:
                                lookup_name = prefixed_name
                        if lookup_name not in name_idx_mapping:
                            raise KeyError(
                                f"Temporal sequence database {database_path}: "
                                f"sequence '{sequence}' references unknown scan "
                                f"'{name}'"
                            )
                        resolved_indices.append(name_idx_mapping[lookup_name])

                    dir_sequence_names.append(sequence)
                    dir_change_files.append(entry.get("filepath"))
                    dir_ambiguities.append(entry.get("ambiguities"))
                    dir_sequence_indices.append(resolved_indices)

                # Add only fully validated sequences to our collections.
                self.sequence_names.extend(dir_sequence_names)
                all_sequence_indices.append(
                    np.asarray(dir_sequence_indices, dtype=int)
                )
                self.change_files.extend(dir_change_files)
                self.ambiguities.extend(dir_ambiguities)

            # Combine sequence indices from all directories
            self.sequence_indices = np.vstack(all_sequence_indices)


    def map2color(self, labels):
        output_colors = list()

        for label in labels:
            output_colors.append(self.color_map[label])

        return torch.tensor(output_colors)
    
    
    def _name_scene(self, idx):
        # determine id name 
        #modified to get file name from instance file instead for consistency with 3RScan 
        name = Path(self.data[idx]["instance_gt_filepath"]).stem
        
        # For multi-dataset configurations, add dataset prefix to avoid name conflicts
        if len(self.data_dir) > 1 and "dataset_source" in self.data[idx]:
            dataset_source = self.data[idx]["dataset_source"]
            name = f"{dataset_source}_{name}"

        return name

    def __len__(self):
        return len(self.sequence_indices)

    def __getitem__(self, idx: int):
        idx = idx % len(self.sequence_indices)

        scan_indices = self.sequence_indices[idx]
        total_points = sum(self.data[scan_idx]["file_len"] for scan_idx in scan_indices)

        # Pre-allocate arrays with exact size
        coordinates = np.zeros((total_points, 3 + (1 if self.temporal_window > 0 else 0)))
        color = np.zeros((total_points, 3))
        normals = np.zeros((total_points, 3))
        segments = np.zeros(total_points, dtype=np.int32)
        labels = np.zeros((total_points, 2), dtype=np.int32)

        max_segment = 0 
        start_slice = 0 
        for i, scan_idx in enumerate(scan_indices):
            end_slice = start_slice + self.data[scan_idx]["file_len"]
            
            points = np.load(self.data[scan_idx]["filepath"].replace("../../", ""))

            if self.temporal_window > 0:
                # add i as an additional dimension to coordinates 
                coordinates[start_slice:end_slice, :3] = points[:, :3]
                coordinates[start_slice:end_slice, 3] = i  # Temporal information
            else: 
                coordinates[start_slice:end_slice] = points[:, :3]

            # append all scans as one sequence
            color[start_slice:end_slice] = points[:, 3:6]
            normals[start_slice:end_slice] = points[:, 6:9]
            labels[start_slice:end_slice] = points[:, 10:]

            # ensure segments are mapped to a new id number if joining shift ids 
            segments[start_slice:end_slice] = points[:, 9] + max_segment
            # update the new max 
            max_segment = np.max(segments[start_slice:end_slice]) + 1

            # skip problematic scans robust to dataset name changes
            scene_source = self.data[scan_idx]['filepath'].split("/")[-3]
            scene_id = self.data[scan_idx]['filepath'].split("/")[-1].replace('.npy', '')
            if scene_id in [
                "scene0636_00",
                "scene0154_00",
            ] and (scene_source == "scannet" or scene_source == "scannet200"):
                return self.__getitem__(0)
            if scene_id in [
                "0171_01"
            ] and (scene_source == "rio" or scene_source == "rio200"):
                return self.__getitem__(0)
            
            start_slice = end_slice
        
        #open(filename).read().splitlines()
        if self.change_files[idx] is not None:
            changes = np.genfromtxt(self.change_files[idx], dtype=int)
            # One change label per point. Future: per-transition labels (N, T-1) when T > 2 sequences.
            if changes.ndim == 2:
                changes = changes[:, 0]
        else:
            # if test sequence or not a temporal sequence, no change information
            changes = np.zeros(total_points)
        
        # Apply max_points_per_sample limit if specified
        if self.max_points_per_sample is not None and total_points > self.max_points_per_sample:
            # Randomly sample indices to downsample
            indices = np.random.choice(total_points, size=self.max_points_per_sample, replace=False)
            indices = np.sort(indices)  # Maintain spatial/temporal order for better performance
            
            coordinates = coordinates[indices]
            color = color[indices]
            normals = normals[indices]
            segments = segments[indices]
            labels = labels[indices]
            changes = changes[indices]
            total_points = self.max_points_per_sample
        
        # add the change information to the labels
        # Note: now labels has a variable number of columns dependant on number of stages
        labels = np.hstack((labels, changes[:, None]))

        raw_coordinates = coordinates.copy()
        raw_color = color
        raw_normals = normals

        if not self.add_colors:
            color = np.ones((len(color), 3))

        # volume and image augmentations for train
        if "train" in self.mode:
            points = np.hstack((coordinates, color, normals, labels, segments[..., None]))

            coordinates -= coordinates.mean(0)

            try:
                coordinates += (
                    np.random.uniform(coordinates.min(0), coordinates.max(0))
                    / 2
                )
            except OverflowError as err:
                print(coordinates)
                print(coordinates.shape)
                raise err

            for i in (0, 1):
                if random() < 0.5:
                    coord_max = np.max(points[:, i])
                    coordinates[:, i] = coord_max - coordinates[:, i]

            if random() < 0.95:
                for granularity, magnitude in ((0.2, 0.4), (0.8, 1.6)):
                    coordinates = elastic_distortion(
                        coordinates, granularity, magnitude
                    )
            aug = self.volume_augmentations(
                points=coordinates,
                normals=normals,
                features=color,
                labels=labels,
            )
            coordinates, color, normals, labels = (
                aug["points"],
                aug["features"],
                aug["normals"],
                aug["labels"],
            )
            pseudo_image = color.astype(np.uint8)[np.newaxis, :, :]
            color = np.squeeze(
                self.image_augmentations(image=pseudo_image)["image"]
            )

        # normalize color information
        pseudo_image = color.astype(np.uint8)[np.newaxis, :, :]
        color = np.squeeze(self.normalize_color(image=pseudo_image)["image"])

        # prepare labels and map from 0 to 20(40)
        labels = labels.astype(np.int32)
        if labels.size > 0:
            labels[:, 0] = self._remap_from_zero(labels[:, 0])
            if not self.add_instance:
                # taking only first column, which is segmentation label, not instance
                labels = labels[:, 0].flatten()[..., None]

        labels = np.hstack((labels, segments[..., None].astype(np.int32)))

        features = color
        if self.add_normals:
            features = np.hstack((features, normals))
        if self.add_raw_coordinates:
            if len(features.shape) == 1:
                features = np.hstack((features[None, ...], coordinates))
            else:
                features = np.hstack((features, coordinates))
                
        # only augment the xyz coordinates, not the temporal dim
        # revert to original temporal dim 
        coordinates[:, 3:] = raw_coordinates[:, 3:]
                
        return (
            coordinates,
            features,
            labels,
            self.sequence_names[idx],
            raw_color,
            raw_normals,
            raw_coordinates,
            idx,
            self.ambiguities[idx],
        )


    @property
    def data(self):
        """database file containing information about preproscessed dataset"""
        return self._data

    @property
    def label_info(self):
        """database file containing information labels used by dataset"""
        return self._labels
    
    @property
    def change_info(self):
        """database file containing information change labels used by dataset"""
        return self._labels

    @staticmethod
    def _load_yaml(filepath):
        with open(filepath) as f:
            file = yaml.load(f, Loader=Loader)
            # file = yaml.load(f)
        return file

    def _select_correct_labels(self, labels, num_labels):
        number_of_validation_labels = 0
        number_of_all_labels = 0
        for (
            k,
            v,
        ) in labels.items():
            number_of_all_labels += 1
            if v["validation"]:
                number_of_validation_labels += 1

        if num_labels == number_of_all_labels:
            return labels
        elif num_labels == number_of_validation_labels:
            valid_labels = dict()
            for (
                k,
                v,
            ) in labels.items():
                if v["validation"]:
                    valid_labels.update({k: v})
            return valid_labels
        else:
            msg = f"""not available number labels, {num_labels} select from:
            {number_of_validation_labels}, {number_of_all_labels}"""
            raise ValueError(msg)

    def _remap_from_zero(self, labels):
        labels[
            ~np.isin(labels, list(self.label_info.keys()))
        ] = self.ignore_label
        # remap to the range from 0
        for i, k in enumerate(self.label_info.keys()):
            labels[labels == k] = i
        return labels

    def _remap_model_output(self, output):
        # output = np.array(output)
        output_remapped = output.clone()
        for i, k in enumerate(self.label_info.keys()):
            output_remapped[output == i] = k
        return output_remapped


def elastic_distortion(pointcloud, granularity, magnitude):
    """Apply elastic distortion on sparse coordinate space.

    pointcloud: numpy array of (number of points, at least 3 spatial dims)
    granularity: size of the noise grid (in same scale[m/cm] as the voxel grid)
    magnitude: noise multiplier
    """
    blurx = np.ones((3, 1, 1, 1)).astype("float32") / 3
    blury = np.ones((1, 3, 1, 1)).astype("float32") / 3
    blurz = np.ones((1, 1, 3, 1)).astype("float32") / 3
    coords = pointcloud[:, :3]
    coords_min = coords.min(0)

    # Create Gaussian noise tensor of the size given by granularity.
    noise_dim = ((coords - coords_min).max(0) // granularity).astype(int) + 3
    noise = np.random.randn(*noise_dim, 3).astype(np.float32)

    # Smoothing.
    for _ in range(2):
        noise = scipy.ndimage.filters.convolve(
            noise, blurx, mode="constant", cval=0
        )
        noise = scipy.ndimage.filters.convolve(
            noise, blury, mode="constant", cval=0
        )
        noise = scipy.ndimage.filters.convolve(
            noise, blurz, mode="constant", cval=0
        )

    # Trilinear interpolate noise filters for each spatial dimensions.
    ax = [
        np.linspace(d_min, d_max, d)
        for d_min, d_max, d in zip(
            coords_min - granularity,
            coords_min + granularity * (noise_dim - 2),
            noise_dim,
        )
    ]
    interp = scipy.interpolate.RegularGridInterpolator(
        ax, noise, bounds_error=0, fill_value=0
    )
    pointcloud[:, :3] = coords + interp(coords) * magnitude
    return pointcloud
