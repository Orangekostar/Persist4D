"""Frozen ReScene with label-free historical prompts before its first mask."""

import copy

import torch

from datasets.native_long_dataset import stable_seed
from models.rescene import ReScene
from models.revisit_prompt_modules import (
    PromptAdapter, SupportHead, geometry6, nearest_indices, segment_context,
    select_history_slots, tensor_summary,
)
from models.task_memory_read import TaskMemoryRead
from models.task_memory_routing import route_entities


class RevisitPromptReScene(ReScene):
    def __init__(self, *args, mpr_variant="M0", mpr_seed=146, **kwargs):
        super().__init__(*args, **kwargs)
        if mpr_variant not in {f"M{i}" for i in range(7)} or (self.num_queries, self.mask_dim, self.D) != (100, 128, 4):
            raise ValueError("MPR requires M0–M6, Q100, D128 and real XYZ+local-t")
        self.requires_grad_(False)
        self.mpr_variant = mpr_variant
        self.mpr_seed = int(mpr_seed)
        self.mpr_prompt = (PromptAdapter(retrieval=mpr_variant != "M3", seed=mpr_seed)
                           if mpr_variant in {"M1", "M3", "M4", "M5", "M6"} else None)
        self.mpr_read = None
        if mpr_variant == "M2":
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(stable_seed(mpr_seed, "mpr-read"))
                self.mpr_read = TaskMemoryRead()
        self.mpr_support = SupportHead(seed=mpr_seed) if mpr_variant == "M6" else None
        self.return_query_features = True
        if self.save_segment_info or self.native_long_feedback is not None or self.native_long_sampling:
            raise ValueError("MPR must disable GT export and old native-long interventions")
        self.task_background_class = self.num_classes - 1
        self.task_confidence_threshold = .5
        self.task_mask_threshold = .5
        self.task_minimum_mask_support = 1
        self._mpr_state = self._mpr_geometry = self._mpr_meta = None
        self._mpr_coordinate_offsets = None
        self._mpr_permutation = None
        self._mpr_contexts = []
        self._mpr_selected = []
        self._mpr_retrieval = []
        self._mpr_support_probabilities = None
        self.mpr_capture_probe = False
        self._mpr_probe = None
        self.train(False)

    def train(self, mode=True):
        super().train(False)
        self.training = mode
        for module in (getattr(self, "mpr_prompt", None), getattr(self, "mpr_read", None),
                       getattr(self, "mpr_support", None)):
            if module is not None:
                module.train(mode)
        return self

    def _position(self, xyz, local_stage, batch, coords):
        if self._mpr_coordinate_offsets is not None:
            xyz = xyz - self._mpr_coordinate_offsets[batch].to(xyz)
        position = torch.cat((xyz, xyz.new_full((xyz.shape[0], 1), local_stage)), dim=-1)
        low = coords[-1][batch]
        encoded = self.pos_enc(position[None].float(), input_range=[
            low.min(0).values[None], low.max(0).values[None]])
        return self.query_projection(encoded).permute(0, 2, 1)[0], position

    def before_first_mask(self, queries, query_pos, sampled_coords, *, dense_features,
                          padding_mask, raw_coordinates, point2segment, stage_meta=None,
                          coords=None):
        if self._mpr_state is None:
            raise ValueError("MPR requires a prediction-only old state")
        self._mpr_selected = [select_history_slots(self._mpr_state, b, meta.absolute_stage_index)
                              for b, meta in enumerate(stage_meta)]
        # Empty state is a strict no-op, including query positions and coordinates.
        sizes = [p.numel() for p in point2segment]
        raw_batches = list(raw_coordinates.split(sizes)) if isinstance(raw_coordinates, torch.Tensor) else raw_coordinates
        if self._mpr_coordinate_offsets is not None:
            raw_batches = [raw.clone() for raw in raw_batches]
            for batch, raw in enumerate(raw_batches):
                raw[:, :3] += self._mpr_coordinate_offsets[batch].to(raw)
        self._mpr_contexts = [segment_context(raw, p2s, dense_features[b], padding_mask[b], meta)
                              for b, (raw, p2s, meta) in enumerate(zip(raw_batches, point2segment, stage_meta, strict=True))]
        if self.mpr_capture_probe:
            self._mpr_probe = dict(dense_features=tensor_summary(dense_features),
                padding=tensor_summary(padding_mask),parent_queries=tensor_summary(queries),
                parent_positions=tensor_summary(query_pos),
                current_segments=[c["current_indices"].detach().cpu().tolist() for c in self._mpr_contexts])
        if self.mpr_prompt is None or not any(s.numel() for s in self._mpr_selected):
            return queries, query_pos, sampled_coords
        result_queries, result_positions = queries.clone(), query_pos.clone()
        result_coords = sampled_coords.clone() if sampled_coords is not None else None
        for batch, (slots, context) in enumerate(zip(self._mpr_selected, self._mpr_contexts, strict=True)):
            m = slots.numel()
            current = context["current_indices"]
            if not m or not current.numel():
                continue
            source_slots = slots if self._mpr_permutation is None else slots[self._mpr_permutation[batch]]
            current_features, centers = context["features"][current], context["centers"][current]
            if self.mpr_variant == "M1":
                if sampled_coords is None:
                    raise ValueError("M1 needs the parent's actual FPS query coordinates")
                fps = sampled_coords[batch, :m, :3]
                if self._mpr_coordinate_offsets is not None:
                    fps = fps + self._mpr_coordinate_offsets[batch].to(fps)
                closest = torch.stack([nearest_indices(centers, xyz, 1)[0]
                                       for xyz in fps])
                source = current_features[closest]
                gh = torch.stack([geometry6(centers[nearest_indices(centers, centers[a])])[0] for a in closest])
                source_valid = torch.ones(m, dtype=torch.bool, device=queries.device)
            else:
                source = self._mpr_state.embedding[batch, source_slots].detach()
                gh = self._mpr_geometry.geometry[batch, source_slots].detach()
                source_valid = self._mpr_geometry.valid[batch, source_slots]
            if self.mpr_variant == "M3":
                z, gc = source, gh
                current_valid = source_valid
                prompt_xyz = self._mpr_geometry.centroid[batch, source_slots]
            else:
                retrieved = self.mpr_prompt.retrieve(source, current_features, centers)
                z, gc = retrieved["z"], retrieved["geometry"]
                current_valid = retrieved["geometry_valid"]
                prompt_xyz = retrieved["centers"]
                retrieved.update(batch_index=batch, source_slots=slots,
                                 current_segment_indices=current,
                                 anchor_segment_indices=current[retrieved["anchor_indices"]])
                self._mpr_retrieval.append(retrieved)
            flags = torch.stack((source_valid, current_valid), -1).to(queries.dtype)
            result_queries[batch, :m] = queries[batch, :m] + self.mpr_prompt(source, z, gh, gc, flags)
            if self.mpr_variant != "M3" or source_valid.any():
                encoded, position = self._position(prompt_xyz, context["latest_local_stage"], batch, coords)
                selector = source_valid if self.mpr_variant == "M3" else torch.ones_like(source_valid)
                result_positions[batch, :m] = torch.where(selector[:, None], encoded, query_pos[batch, :m])
                if result_coords is not None:
                    result_coords[batch, :m] = torch.where(selector[:, None], position, sampled_coords[batch, :m])
            if self.mpr_support is not None:
                top = retrieved["logits"].topk(min(2, current.numel()), dim=1).values
                margin = top[:, 0] - top[:, 1] if top.shape[1] == 2 else torch.zeros_like(top[:, 0])
                support_input = torch.cat((source, z, gh, gc, flags, top[:, :1], margin[:, None]), -1)
                probabilities = self.mpr_support(support_input)
                self._mpr_support_probabilities[batch, :m] = probabilities
                retrieved["support_probabilities"] = probabilities
        return result_queries, result_positions, result_coords

    def after_decoder_stage(self, queries, *, execution_stage_idx, shared_parameter_idx,
                            decoder_features=None, decoder_padding_mask=None, point2segment=None):
        if (self.mpr_read is None or execution_stage_idx != len(self.hlevels) - 1
                or not any(s.numel() for s in self._mpr_selected)):
            return queries
        from models.persist4d_task_memory import Persist4DTaskMemory

        logits, _, masks = self.mask_module(queries, decoder_features)
        observation = Persist4DTaskMemory._build_pre_observation(
            self, queries, logits, masks, decoder_padding_mask, self._mpr_meta)
        route = route_entities(observation, self._mpr_state, self._mpr_meta)
        read = self.mpr_read(queries, self._mpr_state, route)
        allowed = torch.stack([torch.isin(route.query_to_slot[b], slots)
                               for b, slots in enumerate(self._mpr_selected)])
        return queries + (read - queries) * allowed[..., None]

    def forward(self, x, point2segment=None, raw_coordinates=None, is_eval=False, *,
                task_state=None, geometry_state=None, stage_meta=None,
                backbone_outputs=None, history_permutation=None, coordinate_offsets=None):
        if task_state is None or geometry_state is None or stage_meta is None:
            raise ValueError("MPR requires old state, synchronized geometry and label-free StageMeta")
        geometry_state.assert_synchronized(task_state)
        self._mpr_state, self._mpr_geometry, self._mpr_meta = task_state, geometry_state, stage_meta
        if coordinate_offsets is not None and coordinate_offsets.shape != (task_state.batch_size, 3):
            raise ValueError("window coordinate offsets must be label-free [batch,3] translations")
        self._mpr_coordinate_offsets = coordinate_offsets
        self._mpr_permutation = history_permutation
        self._mpr_retrieval = []
        self._mpr_support_probabilities = task_state.embedding.new_ones((task_state.batch_size, 100))
        x.raw_coordinates, x.point2segment = raw_coordinates, point2segment
        try:
            if backbone_outputs is None:
                # Parent encoder and task backbone decoder are both frozen and eval.
                with torch.no_grad():
                    backbone_outputs = self.backbone(x)
            else:
                backbone_outputs = copy.deepcopy(backbone_outputs)
            output = super().forward(x, point2segment, raw_coordinates, is_eval,
                                     stage_meta=stage_meta, backbone_outputs=backbone_outputs)
            output.update(mpr_retrieval=self._mpr_retrieval, mpr_contexts=self._mpr_contexts,
                          mpr_selected_slots=self._mpr_selected,
                          mpr_support_probabilities=self._mpr_support_probabilities)
            if self.mpr_capture_probe:
                output["mpr_pre_mask_probe"] = self._mpr_probe
            return output
        finally:
            self._mpr_state = self._mpr_geometry = self._mpr_meta = None
            self._mpr_coordinate_offsets = None
            self._mpr_permutation = None
            self._mpr_retrieval = []
            self._mpr_contexts = []
            self._mpr_selected = []
            self._mpr_support_probabilities = None
            self._mpr_probe = None
