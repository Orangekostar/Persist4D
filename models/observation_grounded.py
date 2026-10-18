"""Fresh M4 prompting with one differentiable observation-feedback hook."""

import torch

from models.observation_grounded_modules import IdentityEvidenceHead, ObservationFeedback, pool_observation
from models.revisit_prompt import RevisitPromptReScene


VARIANTS = ('R-TARGET', 'R-ID', 'R-QFB', 'R-OBS')


def current_point_counts(meta, segment_count, *, device):
    """Count actual current vertices, rather than voxel counts or GT areas."""
    mapping = meta.full_resolution_point2segment.to(device)
    local_stages = meta.local_stage_ids.to(device)
    if mapping.shape != local_stages.shape or meta.segment_stage_ids.numel() != segment_count:
        raise ValueError("OGR current point mappings do not cover the valid segments")
    current = local_stages == len(meta.scan_ids_in_window) - 1
    return torch.bincount(mapping[current], minlength=segment_count).float()


class ObservationGroundedReScene(RevisitPromptReScene):
    def __init__(self, *args, ogr_variant='R-OBS', mpr_seed=246, **kwargs):
        if ogr_variant not in (*VARIANTS, 'M0', 'M4'):
            raise ValueError("unknown OGR variant")
        kwargs.pop('mpr_variant', None)
        super().__init__(*args, mpr_variant='M0' if ogr_variant == 'M0' else 'M4',
                         mpr_seed=mpr_seed, **kwargs)
        self.ogr_variant = ogr_variant
        self.ogr_identity = IdentityEvidenceHead() if ogr_variant in VARIANTS[1:] else None
        self.ogr_feedback = ObservationFeedback(seed=mpr_seed) if ogr_variant in VARIANTS[2:] else None
        self.ogr_feedback_disabled = False
        self._ogr_observation_state = None
        self._ogr_features = self._ogr_counts = self._ogr_mid = None
        self._ogr_sources = None
        self.train(False)

    def train(self, mode=True):
        super().train(mode)
        for module in (getattr(self, 'ogr_identity', None), getattr(self, 'ogr_feedback', None)):
            if module is not None:
                module.train(mode)
        return self

    def before_first_mask(self, queries, query_pos, sampled_coords, *, dense_features,
                          padding_mask, raw_coordinates, point2segment, stage_meta=None, coords=None):
        if queries.shape[0] != 1 or len(stage_meta) != 1:
            raise ValueError("OGR uses one real episode per GPU")
        self._ogr_features, self._ogr_counts = [], []
        for batch, meta in enumerate(stage_meta):
            count = meta.segment_stage_ids.numel()
            if (dense_features.shape[-1] != 128 or count > dense_features.shape[1]
                    or padding_mask[batch, :count].any() or not padding_mask[batch, count:].all()):
                raise ValueError("OGR dense rows and valid segment padding differ")
            self._ogr_features.append(dense_features[batch, :count].detach())
            self._ogr_counts.append(current_point_counts(meta, count, device=dense_features.device))
        result = super().before_first_mask(queries, query_pos, sampled_coords,
            dense_features=dense_features, padding_mask=padding_mask, raw_coordinates=raw_coordinates,
            point2segment=point2segment, stage_meta=stage_meta, coords=coords)
        self._ogr_sources = []
        for batch, slots in enumerate(self._mpr_selected):
            source_slots = self._source_slots(batch, slots)
            self._ogr_sources.append(dict(target_slots=slots, source_slots=source_slots,
                source_logical_ids=self._mpr_state.logical_ids[batch, source_slots],
                source_generations=self._mpr_state.generations[batch, source_slots]))
        for retrieval in self._mpr_retrieval:
            binding = self._ogr_sources[retrieval['batch_index']]
            retrieval.update(binding)
        return result

    def _source_slots(self, batch, slots):
        if self._mpr_permutation is None:
            return slots
        permutation = self._mpr_permutation[batch]
        if (permutation.shape != slots.shape or permutation.dtype != torch.long
                or not torch.equal(permutation.sort().values,
                                   torch.arange(slots.numel(), device=slots.device))):
            raise ValueError('prompt source intervention must permute the complete selected source rows')
        return slots[permutation]

    def _regional(self, masks, classes, state):
        regional = []
        for batch, (features, counts, logits) in enumerate(zip(
                self._ogr_features, self._ogr_counts, masks, strict=True)):
            values, valid, mass = pool_observation(features, logits, counts)
            slots = (state.occupied[batch] & self._ogr_observation_state.valid[batch]).nonzero(as_tuple=True)[0]
            evidence = dict(values=values, valid=valid, mass=mass, slots=slots, features=features)
            if self.ogr_identity is not None:
                evidence.update(self.ogr_identity(values, valid,
                    self._ogr_observation_state.prototype[batch, slots], classes[batch].softmax(-1),
                    state.class_prob[batch, slots]))
            regional.append(evidence)
        return regional

    def after_decoder_stage(self, queries, *, execution_stage_idx, shared_parameter_idx,
                            decoder_features=None, decoder_padding_mask=None, point2segment=None):
        if execution_stage_idx != len(self.hlevels) - 1 or self.ogr_identity is None:
            return queries
        classes, changes, logits = self.mask_module(queries, decoder_features)
        masks = self.unstack_batched(logits, decoder_padding_mask)
        regional = self._regional(masks, classes, self._mpr_state)
        self._ogr_mid = dict(pred_logits=classes, pred_changes=changes, pred_masks=masks,
                             regional=regional, queries=self.decoder_norm(queries),
                             feedback_execution_stage_idx=execution_stage_idx,
                             first_affected_aux_index=execution_stage_idx + 1)
        if self.ogr_feedback is None or self.ogr_feedback_disabled:
            return queries
        result = queries.clone()
        for batch, evidence in enumerate(regional):
            selected = self._mpr_selected[batch]
            slots = evidence['slots']
            if not selected.numel() or not slots.numel():
                continue
            size = selected.numel()
            probability = evidence['probabilities'][:size]
            reliability = 1 - probability[:, -1]
            history = probability[:, :-1] @ self._ogr_observation_state.prototype[batch, slots]
            history = history / reliability.clamp_min(1e-6)[:, None]
            direct = (self.decoder_norm(queries[batch, :size]) if self.ogr_variant == 'R-QFB'
                      else evidence['values'][:size])
            source = self._mpr_state.embedding[batch, self._source_slots(batch, selected)].detach()
            result[batch, :size] = queries[batch, :size] + reliability[:, None] * self.ogr_feedback(
                direct, history, source)
        return result

    def forward(self, *args, observation_state=None, **kwargs):
        state = kwargs.get('task_state')
        if state is None or observation_state is None:
            raise ValueError("OGR requires a prediction-only observation sidecar")
        observation_state.assert_synchronized(state)
        self._ogr_observation_state, self._ogr_mid = observation_state, None
        try:
            output = super().forward(*args, **kwargs)
            output['ogr_mid'] = self._ogr_mid
            output['ogr_prompt_sources'] = self._ogr_sources
            output['ogr_final'] = self._regional(output['pred_masks'], output['pred_logits'], state)
            return output
        finally:
            self._ogr_observation_state = self._ogr_features = self._ogr_counts = self._ogr_mid = None
            self._ogr_sources = None
