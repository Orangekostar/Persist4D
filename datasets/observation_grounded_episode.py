"""Explicit OGR data roles and local-pair/full-prefix causal native loading."""

import copy
from dataclasses import dataclass
from uuid import UUID

import numpy as np
import torch

from datasets.revisit_prompt_episode import (
    RevisitDrawPlan, RevisitEpisodeLoader, RevisitEpisodeSpec, RevisitStageData,
    causal_prefix_truth, project_causal_target,
)
from datasets.task_memory_episode import StageMeta


@dataclass(frozen=True)
class ObservationEpisodeSpec(RevisitEpisodeSpec):
    window_policy: str = 'local_pair'

    def __post_init__(self):
        if self.role not in {'TRAIN', 'CAL', 'SEL', 'PB', 'CONFIRM'}:
            raise ValueError('unknown OGR data role')
        required = {'TRAIN': 'train', 'CAL': 'train', 'SEL': 'train', 'PB': 'validation'}
        if (self.dataset_scope not in ('train', 'validation')
                or (self.role in required and self.dataset_scope != required[self.role])):
            raise ValueError('OGR dataset scope differs from the explicit role')
        if (self.window_policy not in ('local_pair', 'full_prefix')
                or not self.input_id or not 1 <= self.horizon <= 5
                or len(self.scan_ids) != self.horizon or len(self.scan_indices) != self.horizon
                or len(set(self.scan_ids)) != self.horizon
                or any(type(index) is not int or index < 0 for index in self.scan_indices)):
            raise ValueError('OGR episode identity, horizon, indices or window policy differs')
        for identity in (self.reference_id, *self.scan_ids):
            UUID(identity)

    @classmethod
    def from_record(cls, record, *, window_policy='local_pair'):
        if record.get('domain', 'rio') != 'rio' or 'dataset_scope' not in record:
            raise ValueError('OGR requires rio and explicit dataset_scope, including CONFIRM')
        return cls(input_id=record['input_id'], reference_id=record['reference_id'], role=record['role'],
            dataset_scope=record['dataset_scope'], scan_ids=tuple(record['scan_ids']),
            scan_indices=tuple(record['scan_indices']), horizon=record['horizon'],
            ambiguities=copy.deepcopy(record.get('ambiguities')), window_policy=window_policy)

    def window_start(self, stage):
        if type(stage) is not int or not 0 <= stage < self.horizon:
            raise ValueError('stage outside causal episode')
        return 0 if self.window_policy == 'full_prefix' else max(0, stage - 1)

    @property
    def stage_scan_ids(self):
        return tuple(self.scan_ids[self.window_start(j):j+1] for j in range(self.horizon))

    @property
    def stage_scan_indices(self):
        return tuple(self.scan_indices[self.window_start(j):j+1] for j in range(self.horizon))


class ObservationEpisodeLoader(RevisitEpisodeLoader):
    """Native collation and targets are unchanged; only explicit window start varies."""
    def stages(self, spec):
        from models.perception_gain import derive_segment_stage_ids

        if not isinstance(spec, ObservationEpisodeSpec):
            raise TypeError('OGR loader requires an explicit OGR episode contract')
        collation = self.collate.base.collate_fn
        for j in range(spec.horizon):
            prefix = self._sample(spec, 0, j+1)
            common = causal_prefix_truth(prefix[2], prefix[6][:, 3],
                filter_out_classes=collation.filter_out_classes, label_offset=collation.label_offset,
                ignore_class_threshold=collation.ignore_class_threshold, ambiguities=prefix[8])
            start = spec.window_start(j)
            sample = self._sample(spec, start, j+1)
            sample[3] = f'{spec.input_id}:stage{j}'
            xyz = sample[0][:, :3]
            if not np.array_equal(xyz, sample[6][:, :3]):
                raise ValueError('unaugmented reference coordinates changed before collation')
            low, high = xyz.min(0), xyz.max(0)
            shift = torch.tensor([[(low[0]+high[0])/2, (low[1]+high[1])/2, low[2]]], dtype=torch.float32)
            data, targets, names = self.collate([tuple(sample)])
            if len(targets) != 1 or len(data.target_full) != 1:
                raise ValueError('native collation dropped an empty target')
            p2s = targets[0]['point2segment'].cpu().long()
            low_stages = targets[0]['temporal_stages'].cpu().long()
            full_stages = torch.as_tensor(sample[6][:, 3]).long()
            inverse = data.inverse_maps[0].cpu().long()
            if not torch.equal(low_stages[inverse], full_stages):
                raise ValueError('local/full vertices disagree on causal local-t')
            low_labels = data.labels[0].cpu().clone()
            targets[0] = project_causal_target(common, low_labels, p2s, low_stages,
                                              segment_masks=True, ambiguities=sample[8])
            full_labels = torch.as_tensor(sample[2]).clone()
            data.target_full[0] = project_causal_target(common, full_labels,
                full_labels[:, -1], full_stages, ambiguities=sample[8])
            # The full-prefix baseline retains every causal scan, offset and raw
            # vertex index. No pair-window metadata is reused for its j>=2 inputs.
            count = j + 1 - start
            sizes = torch.bincount(full_stages, minlength=count)
            offsets = torch.cat((torch.zeros(1, dtype=torch.long), sizes.cumsum(0)))
            meta = StageMeta(reference_id=spec.reference_id, episode_id=spec.input_id,
                scan_ids_in_window=spec.stage_scan_ids[j], absolute_stage_index=j,
                local_stage_ids=full_stages, original_vertex_ids=tuple(torch.arange(int(n)) for n in sizes),
                scan_vertex_offsets=offsets, point2segment=p2s,
                segment_stage_ids=derive_segment_stage_ids(p2s, low_stages),
                augmentation_transform_id='identity:no-random-augmentation',
                coordinate_frame_id=f'3rscan-reference:{spec.reference_id}',
                voxel_inverse=inverse, full_resolution_point2segment=p2s[inverse])
            yield RevisitStageData((data, targets, names), (meta,), shift, common, full_labels)
