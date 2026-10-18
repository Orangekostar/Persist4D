"""OGR regional partial matching with original committed route semantics."""

import torch
from scipy.optimize import linear_sum_assignment

from models.task_memory_routing import (
    EntityRoute, _metadata_sha256, _route_sha256, _state_sha256,
    _validate_observation_state, _validate_stage_meta, route_entities, commit_entities,
)
from scripts.revisit_prompt_runtime import RevisitEpisodeRuntime, track_step, trajectory_sidecar
from models.observation_evidence_state import ObservationEvidenceState


@torch.no_grad()
def observation_route(observation, state, stage_meta, regional):
    _validate_observation_state(observation, state)
    stage_indices = _validate_stage_meta(stage_meta, state)
    if len(regional) != state.batch_size:
        raise ValueError("OGR regional route batch differs")
    shape = observation.valid.shape
    q2s = torch.full(shape, -1, device=state.embedding.device, dtype=torch.long)
    s2q = torch.full_like(state.logical_ids, -1)
    scores = torch.full_like(observation.confidence, -float('inf'))
    prior_ids, prior_generations = torch.full_like(q2s, -1), torch.full_like(q2s, -1)
    for batch, evidence in enumerate(regional):
        probabilities, slots = evidence['probabilities'].detach(), evidence['slots']
        queries, count = shape[1], slots.numel()
        if (slots.ndim != 1 or slots.dtype != torch.long
                or probabilities.shape != (queries, count + 1)
                or not torch.isfinite(probabilities).all()
                or ((probabilities < 0) | (probabilities > 1)).any()
                or (slots[1:] <= slots[:-1]).any()
                or (count and (int(slots[0]) < 0 or int(slots[-1]) >= state.capacity))
                or not state.occupied[batch, slots].all()):
            raise ValueError("OGR route requires sorted occupied slots and finite row probabilities")
        p, null = probabilities[:, :-1], probabilities[:, -1:]
        allowed = ((p >= .5) & (p > null)
                   & (observation.valid[batch] & observation.current_supported[batch])[:, None])
        cost = probabilities.new_full((queries, count + queries), 1e6)
        cost[:, :count] = torch.where(allowed, (null + 1e-6).log() - (p + 1e-6).log(), 1e6)
        indices = torch.arange(queries, device=cost.device)
        cost[indices, count + indices] = 0
        rows, columns = linear_sum_assignment(cost.cpu().double().numpy())
        for query, column in zip(rows.tolist(), columns.tolist()):
            if column >= count or not bool(allowed[query, column]):
                continue
            slot = slots[column]
            q2s[batch, query], s2q[batch, slot] = slot, query
            scores[batch, query] = p[query, column]
            prior_ids[batch, query] = state.logical_ids[batch, slot]
            prior_generations[batch, query] = state.generations[batch, slot]
    values = dict(query_to_slot=q2s, slot_to_query=s2q, route_score=scores,
        prior_logical_id=prior_ids, prior_generation=prior_generations,
        current_supported=observation.current_supported.clone(),
        previous_supported=observation.previous_supported.clone(), valid=observation.valid.clone(),
        stage_indices=stage_indices, source_state_sha256=_state_sha256(state),
        metadata_sha256=_metadata_sha256(stage_meta))
    provisional = EntityRoute(content_sha256='0' * 64, **values)
    route = EntityRoute(content_sha256=_route_sha256(provisional), **values)
    route.validate()
    return route


class ObservationEpisodeRuntime(RevisitEpisodeRuntime):
    def __init__(self, spec, *, class_count, device):
        super().__init__(spec, class_count=class_count, device=device)
        self.spec = spec
        self.observation_state = ObservationEvidenceState.empty(self.state)

    def _validate_stage(self, stage):
        meta = super()._validate_stage(stage)
        self.observation_state.assert_synchronized(self.state)
        return meta

    def forward(self, system, stage, *, backbone_outputs=None, history_permutation=None):
        self._validate_stage(stage)
        data, targets, _ = stage.batch
        return system.model(data, point2segment=[targets[0]['point2segment']],
            raw_coordinates=system._process_raw_coordinates(data), is_eval=True,
            task_state=self.state, geometry_state=self.geometry, stage_meta=stage.stage_meta,
            observation_state=self.observation_state, coordinate_offsets=stage.coordinate_offsets,
            backbone_outputs=backbone_outputs, history_permutation=history_permutation)

    @torch.no_grad()
    def finish(self, system, output, stage, *, class_mapper=None, publish=True, measure=False):
        import time
        from trainer.task_memory_trainer import build_final_prediction_observation

        meta = self._validate_stage(stage)
        if measure and self.state.embedding.is_cuda:
            torch.cuda.synchronize()
        started = time.perf_counter()
        observation = build_final_prediction_observation(output, stage.stage_meta,
            background_class=system.model.task_background_class, confidence_threshold=.5,
            mask_threshold=.5, minimum_mask_support=1)
        if system.model.ogr_identity is None:
            route = route_entities(observation, self.state, stage.stage_meta)
        else:
            route = observation_route(observation, self.state, stage.stage_meta, output['ogr_final'])
        commit = commit_entities(observation, route, self.state, stage.stage_meta)
        if measure and self.state.embedding.is_cuda:
            torch.cuda.synchronize()
        associated = time.perf_counter()
        result = dict(observation=observation, commit=commit, route=route,
                      track_step=track_step(commit, meta.absolute_stage_index))
        if publish:
            from scripts.rescene_task_postprocess import extract_official_task_prediction

            data, targets, _ = stage.batch
            # Identity P never scales candidate scores or changes raw query lineage.
            official = extract_official_task_prediction(system=system, output=output,
                target_low_resolution=targets[0], target_full_resolution=data.target_full[0],
                data=data, class_mapper=class_mapper, latest_stage_index=len(meta.scan_ids_in_window)-1)
            result.update(official=official, original_official=official)
        postprocessed = time.perf_counter()
        self.geometry.update(commit, observation, output['pred_masks'], output['mpr_contexts'], meta.absolute_stage_index)
        self.observation_state.update(commit, observation, output['ogr_final'], self.state, meta.absolute_stage_index)
        self.state = commit.state.detach()
        self.geometry.assert_synchronized(self.state)
        self.observation_state.assert_synchronized(self.state)
        self.history.append(meta.scan_ids_in_window[-1])
        if measure:
            if self.state.embedding.is_cuda:
                torch.cuda.synchronize()
            result['timings'] = dict(association_seconds=associated-started,
                postprocess_seconds=postprocessed-associated,
                state_geometry_observation_seconds=time.perf_counter()-postprocessed)
        return result

    def state_nbytes(self):
        return dict(common=sum(t.numel() * t.element_size() for t in self.state.tensors()),
                    geometry=self.geometry.nbytes(), observation=self.observation_state.nbytes())
