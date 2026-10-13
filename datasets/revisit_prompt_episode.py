"""UUID-native causal episodes and CPU-only common-prefix target projection."""

from __future__ import annotations

import copy
from dataclasses import dataclass, fields, replace
from pathlib import Path
from uuid import UUID

import numpy as np
import torch

from datasets.native_long_dataset import canonical_orders, isolated_rng, stable_seed
from datasets.task_memory_episode import StageMeta


@dataclass(frozen=True)
class RevisitEpisodeSpec:
    input_id: str
    reference_id: str
    role: str
    dataset_scope: str
    scan_ids: tuple[str, ...]
    scan_indices: tuple[int, ...]
    horizon: int
    ambiguities: object = None

    def __post_init__(self):
        if self.role not in {"TRAIN", "CAL", "SEL", "PB"}:
            raise ValueError("unknown MPR data role")
        if self.dataset_scope != ("validation" if self.role == "PB" else "train"):
            raise ValueError("dataset scope differs from the fixed role")
        if (not self.input_id or not self.reference_id or not 1 <= self.horizon <= 5
                or len(self.scan_ids) != self.horizon or len(self.scan_indices) != self.horizon
                or len(set(self.scan_ids)) != self.horizon
                or any(type(i) is not int or i < 0 for i in self.scan_indices)):
            raise ValueError("episode identity, horizon or native indices differ")
        for identity in (self.reference_id, *self.scan_ids):
            UUID(identity)  # Validate whole UUIDs; never split them on '-'.

    @classmethod
    def from_record(cls, record):
        if record.get("domain", "rio") != "rio":
            raise ValueError("MPR adaptation uses rio only")
        return cls(input_id=record["input_id"], reference_id=record["reference_id"],
                   role=record["role"], dataset_scope=record.get(
                       "dataset_scope", "validation" if record["role"] == "PB" else "train"),
                   scan_ids=tuple(record["scan_ids"]), scan_indices=tuple(record["scan_indices"]),
                   horizon=record["horizon"], ambiguities=copy.deepcopy(record.get("ambiguities")))

    @property
    def stage_scan_ids(self):
        return tuple(self.scan_ids[max(0, j-1):j+1] for j in range(self.horizon))

    @property
    def stage_scan_indices(self):
        return tuple(self.scan_indices[max(0, j-1):j+1] for j in range(self.horizon))


class RevisitDrawPlan:
    """Exactly H2/H3/H4/H5, uniform eligible references, no old curriculum."""

    def __init__(self, references, *, seed):
        self.seed = int(seed)
        self.references = sorted((r for r in references if r["role"] == "TRAIN"),
                                 key=lambda r: r["reference_id"])
        self.eligible = {h: [r for r in self.references if r["Tmax"] >= h]
                         for h in (2, 3, 4, 5)}
        if any(not refs for refs in self.eligible.values()):
            raise ValueError("every required horizon needs eligible TRAIN references")

    def draw_update(self, update):
        if type(update) is not int or update < 0:
            raise ValueError("update must be a nonnegative optimizer cursor")
        records = []
        for h in (2, 3, 4, 5):
            seed = stable_seed(self.seed, update, h)
            rng = np.random.default_rng(seed)
            ref = self.eligible[h][int(rng.integers(len(self.eligible[h])))]
            order_index = int(rng.integers(3))
            order = canonical_orders(ref)[order_index]
            start = int(rng.integers(len(order)-h+1))
            positions = order[start:start+h]
            records.append(dict(input_id=f"TRAIN:seed{self.seed}:update{update}:H{h}",
                role="TRAIN", dataset_scope="train", domain="rio", seed=self.seed,
                update=update, draw_seed=seed, reference_id=ref["reference_id"],
                horizon=h, order_index=order_index, contiguous_start=start,
                positions=positions, scan_ids=[ref["scan_ids"][i] for i in positions],
                scan_indices=[ref["scan_indices"][i] for i in positions]))
        return records


def causal_prefix_truth(labels, temporal_stages, *, filter_out_classes, label_offset,
                        ignore_class_threshold=100, ambiguities=None):
    from datasets.pointcept_utils import get_instance_masks

    # Native raw labels are Int32; the parent criterion requires Int64 class IDs.
    labels = torch.as_tensor(labels).cpu().long()
    target = get_instance_masks([labels], filter_out_classes=filter_out_classes,
                               label_offset=label_offset,
                               ignore_class_threshold=ignore_class_threshold,
                               preserve_empty_targets=True)[0]
    target.update(temporal_stages=torch.as_tensor(temporal_stages).cpu().long(),
                  point2segment=labels[:, -1].long(), ambiguities=copy.deepcopy(ambiguities))
    return target


def project_causal_target(common, local_labels, point2segment, temporal_stages, *,
                          segment_masks=False, ambiguities=None):
    """Project accepted prefix IDs/classes onto vertices, including filtered local rows."""
    labels = torch.as_tensor(local_labels).cpu()
    p2s = torch.as_tensor(point2segment).cpu().long()
    stages = torch.as_tensor(temporal_stages).cpu().long()
    if labels.shape[0] != p2s.numel() or stages.numel() != p2s.numel():
        raise ValueError("causal target projection must preserve vertex/voxel order")
    masks = common["ids"][:, None] == labels[:, 1][None, :]
    visible = masks.any(1)
    result = {k: common[k][visible].clone() for k in ("ids", "labels", "changes")}
    result.update(masks=masks[visible], point2segment=p2s, temporal_stages=stages,
                  ambiguities=copy.deepcopy(ambiguities))
    if segment_masks:
        count = int(p2s.max())+1 if p2s.numel() else 0
        segment_mask = torch.zeros((int(visible.sum()), count), dtype=torch.bool)
        for index, mask in enumerate(result["masks"]):
            segment_mask[index, p2s[mask].unique()] = True
        result["segment_mask"] = segment_mask
    return result


@dataclass(frozen=True)
class RevisitStageData:
    batch: tuple
    stage_meta: tuple[StageMeta, ...]
    coordinate_offsets: torch.Tensor
    common_truth: dict
    loss_labels: torch.Tensor

    def to(self, device):
        from scripts.native_long_runtime import move_batch

        metas = tuple(replace(meta, **{field.name: getattr(meta, field.name).to(device)
                       for field in fields(meta) if isinstance(getattr(meta, field.name), torch.Tensor)},
                       original_vertex_ids=tuple(ids.to(device) for ids in meta.original_vertex_ids))
                      for meta in self.stage_meta)
        return replace(self, batch=move_batch(self.batch, device), stage_meta=metas,
                       coordinate_offsets=self.coordinate_offsets.to(device))


class RevisitEpisodeLoader:
    """Reuse native loading/voxelization; keep TRAIN and PB indices separate."""

    def __init__(self, config, data_root, formal_manifest):
        from datasets.native_long_dataset import NativeLongCollator

        self.config, self.data_root = config, Path(data_root)
        self.formal_manifest = formal_manifest
        self.datasets = {}
        # Validation collation retains full vertex metadata; augmentation is disabled.
        self.collate = NativeLongCollator(config, training=False)

    def dataset(self, scope):
        if scope not in self.datasets:
            if scope == "train":
                from datasets.native_long_dataset import build_native_datasets
                config = copy.deepcopy(self.config)
                config.data.train_dataset.datasets = [child for child in config.data.train_dataset.datasets
                                                      if child.dataset_name == "rio"]
                self.datasets[scope] = build_native_datasets(
                    config, self.data_root, augmentation=False)["rio"]
            elif scope == "validation":
                from scripts.native_long_confirmation_inference import formal_dataset
                self.datasets[scope] = formal_dataset(self.config, self.data_root, self.formal_manifest)
            else:
                raise ValueError("unknown native dataset scope")
            dataset = self.datasets[scope]
            if dataset.max_points_per_sample is not None or dataset.apply_training_augmentation:
                raise ValueError("MPR cannot shrink inputs or augment independent windows")
        return self.datasets[scope]

    def _sample(self, spec, start, end):
        dataset = self.dataset(spec.dataset_scope)
        scans, indices = spec.scan_ids[start:end], spec.scan_indices[start:end]
        for scan, index in zip(scans, indices, strict=True):
            if Path(dataset.data[index]["raw_filepath"]).parent.name != scan:
                raise ValueError("native index does not locate the bound physical scan UUID")
        record = dict(domain="rio", input_id=f"{spec.input_id}:prefix{end}:start{start}",
                      scan_indices=list(indices), scan_ids=list(scans),
                      ambiguities=copy.deepcopy(spec.ambiguities))
        if spec.dataset_scope == "validation":
            from scripts.native_long_confirmation_inference import formal_samples
            return list(formal_samples(dataset, record)[0])
        with isolated_rng(stable_seed(9001, record["input_id"])):
            sample = list(dataset.load_scan_indices(indices[0], indices, change_file=None))
        sample[3], sample[7] = record["input_id"], stable_seed(record["input_id"])
        if spec.ambiguities is not None:
            sample[8] = copy.deepcopy(spec.ambiguities)
        return sample

    def stages(self, spec):
        from models.perception_gain import derive_segment_stage_ids

        collation = self.collate.base.collate_fn
        for j in range(spec.horizon):
            prefix = self._sample(spec, 0, j+1)
            common = causal_prefix_truth(prefix[2], prefix[6][:, 3],
                filter_out_classes=collation.filter_out_classes,
                label_offset=collation.label_offset,
                ignore_class_threshold=collation.ignore_class_threshold,
                ambiguities=prefix[8])
            sample = self._sample(spec, max(0, j-1), j+1)
            sample[3] = f"{spec.input_id}:stage{j}"
            xyz = sample[0][:, :3]
            if not np.array_equal(xyz, sample[6][:, :3]):
                raise ValueError("unaugmented reference coordinates changed before collation")
            low, high = xyz.min(0), xyz.max(0)
            shift = torch.tensor([[(low[0]+high[0])/2, (low[1]+high[1])/2, low[2]]], dtype=torch.float32)
            data, targets, names = self.collate([tuple(sample)])
            if len(targets) != 1 or len(data.target_full) != 1:
                raise ValueError("native collation dropped an empty target")
            p2s = targets[0]["point2segment"].cpu().long()
            low_stages = targets[0]["temporal_stages"].cpu().long()
            full_stages = torch.as_tensor(sample[6][:, 3]).long()
            inverse = data.inverse_maps[0].cpu().long()
            if not torch.equal(low_stages[inverse], full_stages):
                raise ValueError("local/full vertices disagree on the causal time coordinate")
            low_labels = data.labels[0].cpu().clone()
            targets[0] = project_causal_target(common, low_labels, p2s, low_stages,
                                              segment_masks=True, ambiguities=sample[8])
            full_labels = torch.as_tensor(sample[2]).clone()
            data.target_full[0] = project_causal_target(
                common, full_labels, full_labels[:, -1], full_stages, ambiguities=sample[8])
            sizes = torch.bincount(full_stages, minlength=min(j+1, 2))
            offsets = torch.cat((torch.zeros(1, dtype=torch.long), sizes.cumsum(0)))
            meta = StageMeta(reference_id=spec.reference_id, episode_id=spec.input_id,
                scan_ids_in_window=spec.stage_scan_ids[j], absolute_stage_index=j,
                local_stage_ids=full_stages, original_vertex_ids=tuple(torch.arange(int(n)) for n in sizes),
                scan_vertex_offsets=offsets, point2segment=p2s,
                segment_stage_ids=derive_segment_stage_ids(p2s, low_stages),
                augmentation_transform_id="identity:no-random-augmentation",
                coordinate_frame_id=f"3rscan-reference:{spec.reference_id}",
                voxel_inverse=inverse, full_resolution_point2segment=p2s[inverse])
            yield RevisitStageData((data, targets, names), (meta,), shift, common, full_labels)
