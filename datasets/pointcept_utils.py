import sonata
import numpy as np
import torch
from random import random


class VoxelizeCollate:
    def __init__(
        self,
        ignore_label=255,
        voxel_size=1,
        mode="test",
        small_crops=False,
        very_small_crops=False,
        batch_instance=False,
        probing=False,
        ignore_class_threshold=100,
        filter_out_classes=[],
        label_offset=0,
        num_queries=None,
    ):
        self.filter_out_classes = filter_out_classes
        self.label_offset = label_offset
        self.voxel_size = voxel_size
        self.ignore_label = ignore_label
        self.mode = mode
        self.batch_instance = batch_instance
        self.small_crops = small_crops
        self.very_small_crops = very_small_crops
        self.probing = probing
        self.ignore_class_threshold = ignore_class_threshold

        self.num_queries = num_queries
        
        transform_config = [
            # dict(type="CenterShift", apply_z=True), # center shift applied manually to be consistent across stages
            dict(
                type="Update",
                keys_dict=dict(
                    index_valid_keys=["coord", "color", "normal", "labels", "grid_coord", "batch_idx", "t"],
                )
            ),
            dict(
                type="GridSample",
                grid_size=self.voxel_size,
                hash_type="fnv",
                mode="test" if self.mode=="test" else "train",
                return_grid_coord=True,
                return_inverse=True,
            ),
            # colors are normalized in the preprocessing step
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "color", "inverse", "labels", "batch_idx", "t"),
                feat_keys=("coord", "color", "normal"),
            ),
        ]
        
        self.transform = sonata.transform.Compose(transform_config)

    def __call__(self, batch):
        if ("train" in self.mode) and (
            self.small_crops or self.very_small_crops
        ):
            batch = make_crops(batch)
        if ("train" in self.mode) and self.very_small_crops:
            batch = make_crops(batch)
        return voxelize(
            batch,
            self.voxel_size,
            self.mode,
            ignore_class_threshold=self.ignore_class_threshold,
            filter_out_classes=self.filter_out_classes,
            label_offset=self.label_offset,
            transform = self.transform
        )
        
def voxelize(
    batch,
    voxel_size,
    mode,
    ignore_class_threshold,
    filter_out_classes,
    label_offset,
    transform,
):
    (
        original_labels,
        original_colors,
        original_normals,
        original_coordinates,
        idx,
        full_res_coords, 
        ambiguities,
    ) = ([], [], [], [], [], [], [])

    has_labels = False
    
    points = []
    for b, sample in enumerate(batch):
        #sample[1] stored the semseg features --overwrite with the required sonata features 
        idx.append(sample[7])
        original_coordinates.append(sample[6])
        original_labels.append(sample[2])
        full_res_coords.append(sample[0])
        original_colors.append(sample[4])
        original_normals.append(sample[5])
        ambiguities.append(sample[8])
        batch_idx = np.full((sample[0].shape[0]), b, dtype=np.int32)
        
        if len(sample[2]) > 0: 
            has_labels = True
        else: 
            # if no labels, create dummy labels
            has_labels = False
            sample[2] = np.zeros((sample[0].shape[0], 1), dtype=np.int32)
        
        coords = sample[0]
        D = coords.shape[1]
        if D == 3: 
            # D=3: no temporal stages 
            temporal_stages = np.zeros((coords.shape[0]), dtype=np.int32)  # dummy temporal stage
        else: 
            # D=4: with temporal stages, transform each temporal stage separately
            temporal_stages = np.unique(sample[0][:, 3], axis=0, return_inverse=True)[1]
            
        offset = sonata.utils.batch2offset(torch.from_numpy(temporal_stages))
        
        # center shift coords as a temporal group for consistent alignment (z shift true)
        x_min, y_min, z_min = coords[:, :3].min(axis=0)
        x_max, y_max, _ = coords[:, :3].max(axis=0)
        shift = [(x_min + x_max) / 2, (y_min + y_max) / 2, z_min]
        coords[:, :3] -= shift
        
        start = 0
        inverse_shift = 0
        for end in offset:
            end = end.item()
            point = {
                "coord": coords[start:end, :3],
                "color": sample[4][start:end],
                "normal": sample[5][start:end],
                "labels": sample[2][start:end],
                "batch_idx": batch_idx[start:end], # cannot be named "batch" as it is a reserved for sample handling in sonata
                "t": temporal_stages[start:end],
            }
        
            point = transform(point)
            # store full 4D coordinates
            # overwrite the raw coordinates to keep temporal stage information and original batch information
            # sonata model will not maintain other custom properties but does not use raw coords 
            # (N, 1+4) if D=4 or (N, 1+3) if D=3 where index 0 indicates batch sample
            point["coord"] = torch.concat((point["batch_idx"][:, None], point["coord"], point['t'][:, None]), dim=-1)
            point['coord'] = point['coord'][:, : (D + 1)]  # remove batch_idx and t from coordinates
            
            # shift the inverse maps to reflect temporal stages joined as samples (rather than individual samples)
            # sonata will want inverse per temporal sample, store in an additional key
            point["batch_inverse"] = point["inverse"] + inverse_shift
            inverse_shift += point['coord'].shape[0]
                     
            points.append(point)
            start = end

    # Concatenate all lists
    point = sonata.data.collate_fn(points)
    
    # coordinates expected to be (B, X, Y, Z, T) in grid coord format
    # CURRENT: do not voxelize the temporal information (already discretized)
    # OLD: temp divide by voxel size to be consistent with Minkowski, but grid coordinates not centered the same as minkowski
    point["coordinates"] = torch.concat((point["batch_idx"][:, None], point["grid_coord"], point["t"][:, None]), dim=-1).int()
    point['coordinates'] = point['coordinates'][:, : (D + 1)]  # remove batch_idx and t from coordinates
    
    # expected names by Minkowski original structure
    # split required to be list per batch
    batch_offset = sonata.utils.batch2offset((point["batch_idx"]).int())[:-1].detach().cpu()
    point["labels"] = torch.tensor_split(point["labels"], batch_offset, dim=0)
    point["inverse_maps"] = torch.tensor_split(point["batch_inverse"], batch_offset, dim=0)
    point["temporal_stages"] = torch.tensor_split(point["t"], batch_offset, dim=0)
    point['features'] = point['feat']
            
    # store original coordinates in the point dictionary 
    point['idx'] = idx
    point['original_labels'] = original_labels
    point['full_res_coords'] = full_res_coords
    point['original_colors'] = original_colors
    point['original_normals'] = original_normals
    point['original_coordinates'] = original_coordinates

        
    # segment ID remapping 
    if mode == "test":
        for i in range(len(point["labels"])):
            _, ret_index, ret_inv = np.unique(
                point["labels"][i][:, 0],
                return_index=True,
                return_inverse=True,
            )
            point["labels"][i][:, 0] = torch.from_numpy(ret_inv)
            # input_dict["segment2label"].append(input_dict["labels"][i][ret_index][:, :-1])
    else:
        point["segment2label"] = []
        for i in range(len(point["labels"])):
            _, ret_index, ret_inv = np.unique(
                point["labels"][i][:, -1],
                return_index=True,
                return_inverse=True,
            )
            point["labels"][i][:, -1] = torch.from_numpy(ret_inv)
            point["segment2label"].append(
                point["labels"][i][ret_index][:, :-1]
            )

    if has_labels:
        list_labels = point["labels"]
        target = []
        target_full = []
        original_temporal_stages = torch.from_numpy(original_coordinates[i][:, -1]) if original_coordinates[i].shape[1] == 4 else torch.zeros_like(original_coordinates[i][:, -1])

        if len(list_labels[0].shape) == 1:
            for batch_id in range(len(list_labels)):
                label_ids = list_labels[batch_id].unique()
                if 255 in label_ids:
                    label_ids = label_ids[:-1]

                target.append(
                    {
                        "labels": label_ids,
                        "masks": list_labels[batch_id]
                        == label_ids.unsqueeze(1),
                    }
                )
        else:
            if mode == "test":
                for i in range(len(point["labels"])):
                    target.append(
                        {"point2segment": point["labels"][i][:, 0]}
                    )
                    target_full.append(
                        {
                            "point2segment": torch.from_numpy(
                                original_labels[i][:, 0]
                            ).long(),
                            'temporal_stages': original_temporal_stages,
                            "ambiguities": ambiguities[i],
                        }
                    )
            else:
                target = get_instance_masks(
                    list_labels,
                    list_segments=point["segment2label"],
                    ignore_class_threshold=ignore_class_threshold,
                    filter_out_classes=filter_out_classes,
                    label_offset=label_offset,
                )
                for i in range(len(target)):
                    target[i]["point2segment"] = point["labels"][i][:, -1]
                    target[i]['temporal_stages'] = point['temporal_stages'][i]
                if "train" not in mode:
                    target_full = get_instance_masks(
                        [torch.from_numpy(l) for l in original_labels],
                        ignore_class_threshold=ignore_class_threshold,
                        filter_out_classes=filter_out_classes,
                        label_offset=label_offset,
                    )
                    for i in range(len(target_full)):
                        target_full[i]["point2segment"] = torch.from_numpy(
                            original_labels[i][:, -1]
                        ).long()
                        target_full[i]['temporal_stages'] =torch.from_numpy(original_coordinates[i][:, -1]) if original_coordinates[i].shape[1] == 4 else torch.zeros_like(original_coordinates[i][:, -1])
                        target_full[i]["ambiguities"] = ambiguities[i]
    else:
        target = []
        target_full = []
        point["labels"] = []
        point["coordinates"] = []
        point["features"] = []
        point["segment2label"] = []
        
    point["target_full"] = target_full
        
    if "train" in mode: 
        # remove unneeded keys 
        point.pop("target_full", None)
        point.pop("original_colors", None)
        point.pop("original_normals", None)
        point.pop("original_coordinates", None)
        point.pop("idx", None)
        
    return (
            AttrDict(**point),
            target,
            [sample[3] for sample in batch], # scene names
        )


def get_instance_masks(
    list_labels,
    list_segments=None,
    ignore_class_threshold=100,
    filter_out_classes=[],
    label_offset=0,
):
    target = []

    for batch_id in range(len(list_labels)):
        label_ids = []
        change_ids = []
        masks = []
        ids = []
        segment_masks = []
        instance_ids = list_labels[batch_id][:, 1].unique()

        for instance_id in instance_ids:
            if instance_id == -1:
                continue

            # TODO is it possible that a ignore class (255) is an instance???
            # instance == -1 ???
            tmp = list_labels[batch_id][
                list_labels[batch_id][:, 1] == instance_id
            ]
            label_id = tmp[0, 0]
            # Design: one change label per instance. Future: per-transition support (T-1 labels) when T > 2.
            change_id = tmp[0, 2]

            if (
                label_id in filter_out_classes
            ):  # floor, wall, undefined==255 is not included
                continue

            if (
                255 in filter_out_classes
                and label_id.item() == 255
                and tmp.shape[0] < ignore_class_threshold
            ):
                continue

            label_ids.append(label_id)
            change_ids.append(change_id)
            masks.append(list_labels[batch_id][:, 1] == instance_id)
            ids.append(instance_id)

            if list_segments:
                segment_mask = torch.zeros(
                    list_segments[batch_id].shape[0]
                ).bool()
                segment_mask[
                    list_labels[batch_id][
                        list_labels[batch_id][:, 1] == instance_id
                    ][:, -1].unique()
                ] = True
                segment_masks.append(segment_mask)

        if len(label_ids) == 0:
            return list()

        label_ids = torch.stack(label_ids)
        change_ids = torch.stack(change_ids)
        masks = torch.stack(masks)
        instance_ids = torch.stack(ids)
        if list_segments:
            segment_masks = torch.stack(segment_masks)

        l = torch.clamp(label_ids - label_offset, min=0)

        if list_segments:
            target.append(
                {
                    "labels": l,
                    "changes": change_ids,
                    "ids": instance_ids,
                    "masks": masks,
                    "segment_mask": segment_masks,
                }
            )
        else:
            target.append({"labels": l, "changes": change_ids, "ids": instance_ids,"masks": masks})
    return target


def make_crops(batch):
    new_batch = []
    # detupling
    for scene in batch:
        new_batch.append([scene[0], scene[1], scene[2]])
    batch = new_batch
    new_batch = []
    for scene in batch:
        # move to center for better quadrant split
        scene[0][:, :3] -= scene[0][:, :3].mean(0)

        # BUGFIX - there always would be a point in every quadrant
        scene[0] = np.vstack(
            (
                scene[0],
                np.array(
                    [
                        [0.1, 0.1, 0.1],
                        [0.1, -0.1, 0.1],
                        [-0.1, 0.1, 0.1],
                        [-0.1, -0.1, 0.1],
                    ]
                ),
            )
        )
        scene[1] = np.vstack((scene[1], np.zeros((4, scene[1].shape[1]))))
        scene[2] = np.concatenate(
            (scene[2], np.full_like((scene[2]), 255)[:4])
        )

        crop = scene[0][:, 0] > 0
        crop &= scene[0][:, 1] > 0
        if crop.size > 1:
            new_batch.append([scene[0][crop], scene[1][crop], scene[2][crop]])

        crop = scene[0][:, 0] > 0
        crop &= scene[0][:, 1] < 0
        if crop.size > 1:
            new_batch.append([scene[0][crop], scene[1][crop], scene[2][crop]])

        crop = scene[0][:, 0] < 0
        crop &= scene[0][:, 1] > 0
        if crop.size > 1:
            new_batch.append([scene[0][crop], scene[1][crop], scene[2][crop]])

        crop = scene[0][:, 0] < 0
        crop &= scene[0][:, 1] < 0
        if crop.size > 1:
            new_batch.append([scene[0][crop], scene[1][crop], scene[2][crop]])

    # moving all of them to center
    for i in range(len(new_batch)):
        new_batch[i][0][:, :3] -= new_batch[i][0][:, :3].mean(0)
    return new_batch


class AttrDict(dict):    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._convert_nested()
    
    def _convert_nested(self):
        """Convert nested dictionaries to AttrDict instances"""
        for key, value in list(self.items()):
            if isinstance(value, dict) and not isinstance(value, AttrDict):
                self[key] = AttrDict(value)
            elif isinstance(value, list):
                self[key] = [AttrDict(item) if isinstance(item, dict) and not isinstance(item, AttrDict) 
                           else item for item in value]
    
    def __getattr__(self, key):
        if key.startswith('_'):  # Avoid recursion with private attributes
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")
    
    def __setattr__(self, key, value):
        if key.startswith('_'):  # Handle private attributes normally
            super(dict, self).__setattr__(key, value)
        else:
            self[key] = self._convert_value(value)
    
    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")
    
    def __setitem__(self, key, value):
        super().__setitem__(key, self._convert_value(value))
    
    def _convert_value(self, value):
        """Convert dict values to AttrDict instances"""
        if isinstance(value, dict) and not isinstance(value, AttrDict):
            return AttrDict(value)
        elif isinstance(value, list):
            return [AttrDict(item) if isinstance(item, dict) and not isinstance(item, AttrDict) 
                   else item for item in value]
        return value
    
    def __len__(self):
        """Lightning auto-detection support - prioritize common batch keys"""
        for key in ['inverse_maps', 'coordinates', 'features', 'targets']:
            if key in self and (val := self[key]) is not None:
                return len(val) if hasattr(val, '__len__') else getattr(val, 'shape', [1])[0]
        return 1
    
    @property 
    def batch_size(self):
        """Explicit batch size for Lightning compatibility"""
        return len(self)