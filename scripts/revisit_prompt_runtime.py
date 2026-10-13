"""One common prediction-only final route/commit policy for all seven variants."""

from dataclasses import replace
from types import SimpleNamespace
import time

import torch

from models.revisit_prompt_state import RevisitGeometryState
from models.task_memory_routing import route_entities, commit_entities
from models.task_memory_state import TaskMemoryConfig, TaskMemoryState


def empty_common_state(*, class_count, device, dtype=torch.float32):
    return TaskMemoryState.empty(batch_size=1, capacity=100, feature_dim=128,
        class_count=class_count, device=device, dtype=dtype,
        config=TaskMemoryConfig(class_weight=.25, association_threshold=.5,
            update_mode="confidence_ema", update_rate=.2, max_update_rate=.2))


@torch.no_grad()
def common_commit(output, stage_meta, state, *, background_class):
    from trainer.task_memory_trainer import build_final_prediction_observation

    observation = build_final_prediction_observation(output, stage_meta,
        background_class=background_class, confidence_threshold=.5,
        mask_threshold=.5, minimum_mask_support=1)
    probabilities = output["mpr_support_probabilities"].detach()
    if (probabilities.shape != observation.confidence.shape
            or not torch.isfinite(probabilities).all()
            or ((probabilities < 0) | (probabilities > 1)).any()):
        raise ValueError("support probabilities must align with the final query confidence")
    confidence = observation.confidence * probabilities
    observation = replace(observation, confidence=confidence,
                          valid=(confidence >= .5) & (observation.current_supported | observation.previous_supported))
    route = route_entities(observation, state, stage_meta)
    return observation, commit_entities(observation, route, state, stage_meta)


def scale_candidates(official, probabilities):
    """Score original extracted candidates once; keep their masks and lineage."""
    official.validate()
    probabilities = probabilities.detach().cpu().to(official.pred_scores)
    if probabilities.shape != (100,) or not torch.isfinite(probabilities).all():
        raise ValueError("candidate coefficients must cover the unchanged Q100 queries")
    result = replace(official, pred_scores=official.pred_scores * probabilities[official.source_query_ids])
    result.validate()
    return result


def track_step(commit, stage_index):
    identities = commit.query_to_logical_id[0].detach().cpu().tolist()
    generations = commit.query_to_generation[0].detach().cpu().tolist()
    if any(g != 0 for i, g in zip(identities, generations, strict=True) if i >= 0):
        raise ValueError("the fixed no-eviction common policy cannot reuse a logical ID")
    return SimpleNamespace(stage_id=stage_index,
                           track_ids=tuple(i if i >= 0 else None for i in identities),
                           generations=tuple(generations))


class RevisitEpisodeRuntime:
    def __init__(self, spec, *, class_count, device):
        self.input_id, self.reference_id = spec.input_id, spec.reference_id
        self.state = empty_common_state(class_count=class_count, device=device)
        self.geometry = RevisitGeometryState.empty(self.state)
        self.history = []

    def _validate_stage(self, stage):
        if len(stage.stage_meta) != 1:
            raise ValueError("MPR runtime uses one episode per device")
        meta = stage.stage_meta[0]
        if (meta.episode_id != self.input_id or meta.reference_id != self.reference_id
                or meta.absolute_stage_index != int(self.state.stage_watermark[0])+1):
            raise ValueError("episode identity or sequential stage watermark differs")
        self.geometry.assert_synchronized(self.state)
        return meta

    def forward(self, system, stage, *, backbone_outputs=None, history_permutation=None):
        self._validate_stage(stage)
        data, targets, _ = stage.batch
        return system.model(data, point2segment=[targets[0]["point2segment"]],
            raw_coordinates=system._process_raw_coordinates(data), is_eval=True,
            task_state=self.state, geometry_state=self.geometry, stage_meta=stage.stage_meta,
            coordinate_offsets=stage.coordinate_offsets, backbone_outputs=backbone_outputs,
            history_permutation=history_permutation)

    def finish(self, system, output, stage, *, class_mapper, measure=False):
        from scripts.rescene_task_postprocess import extract_official_task_prediction

        meta = self._validate_stage(stage)
        started = time.perf_counter()
        observation, commit = common_commit(output, stage.stage_meta, self.state,
                                           background_class=system.model.task_background_class)
        if measure and observation.confidence.is_cuda:
            torch.cuda.synchronize()
        associated = time.perf_counter()
        data, targets, _ = stage.batch
        # Extract from raw class/mask logits, then apply exactly one score coefficient.
        original = extract_official_task_prediction(system=system, output=output,
            target_low_resolution=targets[0], target_full_resolution=data.target_full[0],
            data=data, class_mapper=class_mapper, latest_stage_index=len(meta.scan_ids_in_window)-1)
        official = scale_candidates(original, output["mpr_support_probabilities"][0])
        step = track_step(commit, meta.absolute_stage_index)
        postprocessed = time.perf_counter()
        # The current output binding exists before geometry changes; never rewrite a past mask.
        result = dict(observation=observation, commit=commit, official=official,
                      original_official=original, track_step=step)
        self.geometry.update(commit, observation, output["pred_masks"],
                             output["mpr_contexts"], meta.absolute_stage_index)
        self.state = commit.state.detach()
        self.geometry.assert_synchronized(self.state)
        self.history.append(meta.scan_ids_in_window[-1])
        if measure:
            if observation.confidence.is_cuda:
                torch.cuda.synchronize()
            result["timings"] = dict(association_seconds=associated-started,
                postprocess_seconds=postprocessed-associated,
                state_geometry_seconds=time.perf_counter()-postprocessed)
        return result


def trajectory_sidecar(official, meta, *, history, provenance, order_id="canonical"):
    """Transport local latest output under absolute j to the frozen accumulator.

    The legacy order enum is transport metadata: for CAL/SEL, 'canonical'
    denotes the already bound input order, without asserting physical timestamps.
    Actual consumed UUIDs, episode identity and absolute stage remain explicit.
    """
    from scripts.system_comparison_v2_cache import validate_task_sidecar

    official.validate()
    task = official.prediction(latest_only=True)
    task["latest_stage_index"] = meta.absolute_stage_index
    sidecar = dict(schema_version=1,
        key=dict(master_sequence_id=meta.episode_id, reference_scene_id=meta.reference_id,
                 order_id=order_id, stage_index=meta.absolute_stage_index,
                 history_scan_ids=list(history), local_window_scan_ids=list(meta.scan_ids_in_window)),
        provenance=provenance, task_prediction=task)
    validate_task_sidecar(sidecar)
    return sidecar
