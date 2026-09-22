"""Fixed-U equivalence audit for the bounded CrossWindow objective."""

from __future__ import annotations

import itertools
from collections import Counter
from collections.abc import Mapping

import torch

from models.crosswindow_state import CrossWindowState
from models.overlap_entity_association import (
    AssociationConfig,
    associate,
    build_evidence,
    commit_observation,
)
from scripts.crosswindow_cache import build_canonical_frame
from scripts.replay_crosswindow_association import (
    _publication_identities,
    _scan_from_frame,
    canonicalize_stage_target,
    evaluator_prediction,
    publish,
    select_revision_scan,
)


def _partial_injections(size: int):
    choices = (*range(size), None)
    for assignment in itertools.product(choices, repeat=size):
        real = [value for value in assignment if value is not None]
        if len(real) == len(set(real)):
            yield assignment


def _expanded_objective(
    assignment: tuple[int | None, ...],
    *,
    base: tuple[tuple[float, ...], ...],
    overlap: tuple[tuple[float, ...], ...],
    fixed_u: tuple[tuple[int, ...], ...],
    lambda_: float,
) -> float:
    value = 0.0
    for query, anchor in enumerate(assignment):
        if anchor is None:
            continue
        value += base[query][anchor]
        for old_group in range(len(fixed_u)):
            value += lambda_ * fixed_u[old_group][anchor] * overlap[query][old_group]
    return value


def _collapsed_objective(
    assignment: tuple[int | None, ...],
    *,
    base: tuple[tuple[float, ...], ...],
    overlap: tuple[tuple[float, ...], ...],
    fixed_u: tuple[tuple[int, ...], ...],
    lambda_: float,
) -> float:
    weights = tuple(
        tuple(
            base[query][anchor]
            + lambda_
            * sum(
                fixed_u[old_group][anchor] * overlap[query][old_group]
                for old_group in range(len(fixed_u))
            )
            for anchor in range(len(base))
        )
        for query in range(len(base))
    )
    return sum(
        weights[query][anchor]
        for query, anchor in enumerate(assignment)
        if anchor is not None
    )


def exhaustive_fixed_u_equivalence() -> dict[str, object]:
    """Exhaustively compare expanded and collapsed objectives for 2-3 groups."""
    checked = 0
    maximum_error = 0.0
    per_size = {}
    for size in (2, 3):
        base = tuple(
            tuple((query + 2 * anchor + 1) / 17.0 for anchor in range(size))
            for query in range(size)
        )
        overlap = tuple(
            tuple((3 * query + old_group + 1) / 19.0 for old_group in range(size))
            for query in range(size)
        )
        fixed_u = tuple(
            tuple(int(anchor == (old_group + 1) % size) for anchor in range(size))
            for old_group in range(size)
        )
        count = 0
        for assignment in _partial_injections(size):
            expanded = _expanded_objective(
                assignment,
                base=base,
                overlap=overlap,
                fixed_u=fixed_u,
                lambda_=0.25,
            )
            collapsed = _collapsed_objective(
                assignment,
                base=base,
                overlap=overlap,
                fixed_u=fixed_u,
                lambda_=0.25,
            )
            maximum_error = max(maximum_error, abs(expanded - collapsed))
            checked += 1
            count += 1
        per_size[str(size)] = count
    return {
        "schema_version": "crosswindow-fixed-u-equivalence-v1",
        "group_sizes": [2, 3],
        "assignment_count": checked,
        "assignments_by_group_size": per_size,
        "tolerance": 1e-12,
        "maximum_absolute_error": maximum_error,
        "status": "PASS" if maximum_error <= 1e-12 else "FAIL",
    }


def _select_legacy_revision(old, new, *, selector: str, score_mode: str):
    from scripts.task_memory_output import _DenseCandidate, _DenseScan

    if selector not in {"M-new", "M-old", "M-score"}:
        raise ValueError("legacy revision selector differs")
    if score_mode not in {"MASK_ONLY_FIXED_SCORE", "SYSTEM_SELECTED_SCORE"}:
        raise ValueError("legacy revision score mode differs")
    if old.scan_id != new.scan_id or not torch.equal(old.vertex_ids, new.vertex_ids):
        raise ValueError("legacy revision scans are not aligned")
    old_by_identity = {candidate.identity: candidate for candidate in old.candidates}
    selected = []
    choices = Counter()
    for new_candidate in new.candidates:
        old_candidate = old_by_identity.get(new_candidate.identity)
        if old_candidate is None:
            selected.append(new_candidate)
            choices["new_only"] += 1
            continue
        mask_equal = torch.equal(old_candidate.mask, new_candidate.mask)
        choose_new = selector == "M-new" or (
            selector == "M-score"
            and (mask_equal or new_candidate.score > old_candidate.score + 1e-6)
        )
        chosen = new_candidate if choose_new else old_candidate
        score = (
            new_candidate.score
            if score_mode == "MASK_ONLY_FIXED_SCORE"
            else chosen.score
        )
        selected.append(
            _DenseCandidate(
                candidate_index=chosen.candidate_index,
                identity=new_candidate.identity,
                source_query_id=chosen.source_query_id,
                score=score,
                mask=chosen.mask.clone(),
            )
        )
        choices["new" if choose_new else "old"] += 1
    new_identities = {candidate.identity for candidate in new.candidates}
    choices["old_only_dropped"] = sum(
        candidate.identity not in new_identities for candidate in old.candidates
    )
    return (
        _DenseScan(
            scan_id=new.scan_id,
            absolute_stage_index=new.absolute_stage_index,
            vertex_ids=new.vertex_ids.clone(),
            candidates=tuple(selected),
        ),
        choices,
    )


def _candidate_class(candidate: object) -> int:
    identity = candidate.identity
    if hasattr(identity, "class_id"):
        return int(identity.class_id)
    return int(identity[2])


def _best_gt_iou(scan: object, target: Mapping[str, object], class_mapping):
    gt_ids = target["gt_ids"].detach().cpu().long()
    gt_classes = target["gt_classes"].detach().cpu().long()
    gt_masks = target["gt_masks"].detach().cpu().bool()
    result = {}
    for index, gt_id in enumerate(gt_ids.tolist()):
        gt_mask = gt_masks[index]
        if not gt_mask.any().item():
            continue
        gt_class = class_mapping[int(gt_classes[index].item())]
        best = 0.0
        for candidate in scan.candidates:
            if _candidate_class(candidate) != gt_class:
                continue
            union = int((gt_mask | candidate.mask).sum().item())
            if union:
                best = max(best, int((gt_mask & candidate.mask).sum().item()) / union)
        result[int(gt_id)] = best
    return result


def _gt_revision_rows(
    *,
    logical_unit_id: str,
    reference_id: str,
    parent: str,
    selector: str,
    score_mode: str,
    absolute_stage: int,
    old: object,
    new: object,
    selected: object,
    target: Mapping[str, object],
    class_mapping: tuple[int, ...],
    choices: Mapping[str, int],
) -> list[dict[str, object]]:
    old_iou = _best_gt_iou(old, target, class_mapping)
    new_iou = _best_gt_iou(new, target, class_mapping)
    selected_iou = _best_gt_iou(selected, target, class_mapping)
    rows = []
    for gt_id in sorted(set(old_iou) | set(new_iou)):
        for threshold in (0.25, 0.5, 0.75):
            old_pass = old_iou.get(gt_id, 0.0) > threshold
            new_pass = new_iou.get(gt_id, 0.0) > threshold
            transition = (
                "weak_to_strong"
                if not old_pass and new_pass
                else (
                    "strong_to_weak"
                    if old_pass and not new_pass
                    else "both_strong" if old_pass and new_pass else "both_weak"
                )
            )
            rows.append(
                {
                    "event_type": "GT_REVISION",
                    "logical_unit_id": logical_unit_id,
                    "reference_id": reference_id,
                    "parent": parent,
                    "selector": selector,
                    "score_mode": score_mode,
                    "absolute_stage": absolute_stage,
                    "gt_id": gt_id,
                    "tau": threshold,
                    "old_best_iou": old_iou.get(gt_id, 0.0),
                    "new_best_iou": new_iou.get(gt_id, 0.0),
                    "selected_best_iou": selected_iou.get(gt_id, 0.0),
                    "selected_minus_new_iou": (
                        selected_iou.get(gt_id, 0.0) - new_iou.get(gt_id, 0.0)
                    ),
                    "transition": transition,
                    "old_choice_count": int(choices.get("old", 0)),
                    "new_choice_count": int(choices.get("new", 0)),
                    "new_only_count": int(choices.get("new_only", 0)),
                    "old_only_dropped_count": int(choices.get("old_only_dropped", 0)),
                }
            )
    return rows


class E4RevisionAccumulator:
    """Evaluate the fixed D0 and A identity trajectories under six revisions."""

    horizons = (2, 3, 4, 5)
    selectors = ("M-new", "M-old", "M-score")
    score_modes = ("MASK_ONLY_FIXED_SCORE", "SYSTEM_SELECTED_SCORE")

    def __init__(
        self,
        *,
        dataset_spec: str,
        class_mapping: tuple[int, ...],
        checkpoint_sha256: str,
        source_commit: str,
        association_method: str,
        association_config: AssociationConfig,
        selectors: tuple[str, ...] | None = None,
        score_modes: tuple[str, ...] | None = None,
    ) -> None:
        from scripts.analyze_persist4d_allt import AllTBaselineAccumulator

        if len(class_mapping) != 18 or len(set(class_mapping)) != 18:
            raise ValueError("E4 class mapping differs")
        self.dataset_spec = dataset_spec
        self.class_mapping = class_mapping
        self.checkpoint_sha256 = checkpoint_sha256
        self.source_commit = source_commit
        self.association_method = association_method
        self.association_config = association_config
        self.selectors = selectors or type(self).selectors
        self.score_modes = score_modes or type(self).score_modes
        if not self.selectors or not set(self.selectors) <= set(type(self).selectors):
            raise ValueError("E4 selector subset differs")
        if not self.score_modes or not set(self.score_modes) <= set(
            type(self).score_modes
        ):
            raise ValueError("E4 score-mode subset differs")
        self.parents = ("D0", association_method)
        self.metrics = {
            (parent, selector, score_mode, horizon): AllTBaselineAccumulator(
                dataset_spec=dataset_spec
            )
            for parent in self.parents
            for selector in self.selectors
            for score_mode in self.score_modes
            for horizon in self.horizons
        }
        self.revision_events: list[dict[str, object]] = []
        self.unit_count = 0
        self.reference_ids: set[str] = set()

    def _class_mapper(self, value: int) -> int:
        if isinstance(value, bool) or not 0 <= value < len(self.class_mapping):
            raise ValueError("E4 model class is outside the fixed mapping")
        return self.class_mapping[value]

    def _legacy_prefixes_and_events(
        self,
        *,
        logical_unit_id: str,
        reference_id: str,
        predictions,
        identity_maps,
        metas,
        targets,
    ):
        from scripts.task_memory_output import (
            LagOnePublisher,
            _archive_scan,
            _materialize,
        )

        standard = LagOnePublisher(score_reducer="mean", iou_threshold=0.5)
        records = []
        for prediction, identity_map, meta in zip(
            predictions, identity_maps, metas, strict=True
        ):
            standard.update(prediction, identity_map, meta)
            pair = standard.revision_pair
            old, revised = pair if pair is not None else (None, None)
            records.append((old, revised, standard.current_dense_scan))
        outputs = {}
        for selector in self.selectors:
            for score_mode in self.score_modes:
                archive = []
                prefixes = []
                for stage_index, (old, revised, current) in enumerate(records):
                    if old is not None:
                        selected, choices = _select_legacy_revision(
                            old,
                            revised,
                            selector=selector,
                            score_mode=score_mode,
                        )
                        archive.append(_archive_scan(selected))
                        self.revision_events.extend(
                            _gt_revision_rows(
                                logical_unit_id=logical_unit_id,
                                reference_id=reference_id,
                                parent="D0",
                                selector=selector,
                                score_mode=score_mode,
                                absolute_stage=stage_index,
                                old=old,
                                new=revised,
                                selected=selected,
                                target=targets[stage_index - 1],
                                class_mapping=self.class_mapping,
                                choices=choices,
                            )
                        )
                    prefixes.append(
                        _materialize(
                            archive=tuple(archive),
                            buffer=current,
                            revision_log=(),
                            score_reducer="mean",
                            provisional_scan_id=current.scan_id,
                        )
                    )
                outputs[(selector, score_mode)] = prefixes
        return outputs

    def _association_prefixes_and_events(
        self,
        *,
        logical_unit_id: str,
        reference_id: str,
        frames,
        targets,
        feature_dim: int,
        class_count: int,
    ):
        state = CrossWindowState.empty(
            capacity=100,
            feature_dim=feature_dim,
            class_count=class_count,
        )
        buffer = None
        plans = []
        for frame in frames:
            plan = associate(
                build_evidence(frame, state, buffer), self.association_config
            )
            state, committed = commit_observation(frame, state, plan)
            buffer = committed.buffer
            plans.append(plan)
        outputs = {}
        for selector in self.selectors:
            for score_mode in self.score_modes:
                boundary = None
                prefixes = []
                for stage_index, (frame, plan) in enumerate(
                    zip(frames, plans, strict=True)
                ):
                    if boundary is not None:
                        identities, _ = _publication_identities(frame, plan)
                        new_revision = _scan_from_frame(
                            frame,
                            plan,
                            scan_id=boundary.provisional.scan_id,
                            publication_identities=identities,
                        )
                        selected, choice_rows = select_revision_scan(
                            boundary.provisional,
                            new_revision,
                            selector=selector,
                            score_mode=score_mode,
                        )
                        choices = Counter(row["choice"] for row in choice_rows)
                        self.revision_events.extend(
                            _gt_revision_rows(
                                logical_unit_id=logical_unit_id,
                                reference_id=reference_id,
                                parent=self.association_method,
                                selector=selector,
                                score_mode=score_mode,
                                absolute_stage=stage_index,
                                old=boundary.provisional,
                                new=new_revision,
                                selected=selected,
                                target=targets[stage_index - 1],
                                class_mapping=self.class_mapping,
                                choices=choices,
                            )
                        )
                    prefix = publish(
                        frame,
                        plan,
                        mask_selection=selector,
                        score_mode=score_mode,
                        history_boundary=boundary,
                    )
                    boundary = prefix.history_boundary
                    prefixes.append(prefix)
                outputs[(selector, score_mode)] = prefixes
        return outputs

    def update(self, *, logical_unit_id: str, base, supplement) -> None:
        from scripts.run_task_memory_controls import (
            prediction_observation_from_payload,
            run_control_trajectory,
        )
        from scripts.run_task_memory_policy_baseline import (
            _meta_from_payload,
            _prediction_from_payload,
            _target_for_prefix,
        )
        from scripts.system_comparison_metrics import validate_causal_prefix_pair

        base_stages = base["stages"]
        supplement_stages = supplement["stages"]
        episode = base["episode"]
        if len(base_stages) != 5 or len(supplement_stages) != 5:
            raise ValueError("E4 cache stage coverage differs")
        reference_id = str(episode["reference_id"])
        scan_ids = tuple(str(value) for value in episode["scan_ids"])
        metas = [_meta_from_payload(stage["stage_meta"]) for stage in base_stages]
        observations = [
            prediction_observation_from_payload(stage["observation"])
            for stage in supplement_stages
        ]
        predictions = [
            _prediction_from_payload(stage["prediction"]) for stage in supplement_stages
        ]
        frames = [
            build_canonical_frame(
                producer_id="R1-B4-policy",
                order_id=logical_unit_id,
                observation=observation,
                prediction=prediction,
                stage_meta=meta,
            )
            for observation, prediction, meta in zip(
                observations, predictions, metas, strict=True
            )
        ]
        original_targets = [stage["target"] for stage in base_stages]
        canonical_targets = [
            canonicalize_stage_target(
                target,
                original_vertex_ids=meta.original_vertex_ids[-1],
            )
            for target, meta in zip(original_targets, metas, strict=True)
        ]
        trajectory = run_control_trajectory(
            observations,
            metas,
            update_mode="last",
            capacity=100,
            class_weight=0.25,
            association_threshold=0.5,
            update_rate=0.2,
        )
        legacy = self._legacy_prefixes_and_events(
            logical_unit_id=logical_unit_id,
            reference_id=reference_id,
            predictions=predictions,
            identity_maps=trajectory.identity_maps,
            metas=metas,
            targets=original_targets,
        )
        association = self._association_prefixes_and_events(
            logical_unit_id=logical_unit_id,
            reference_id=reference_id,
            frames=frames,
            targets=canonical_targets,
            feature_dim=observations[0].features.shape[-1],
            class_count=observations[0].class_prob.shape[-1],
        )
        for horizon in self.horizons:
            stage_index = horizon - 1
            original_target = _target_for_prefix(
                original_targets,
                horizon=horizon,
                class_mapper=self._class_mapper,
            )
            canonical_target = _target_for_prefix(
                canonical_targets,
                horizon=horizon,
                class_mapper=self._class_mapper,
            )
            for selector in self.selectors:
                for score_mode in self.score_modes:
                    d0_prefix = legacy[(selector, score_mode)][stage_index]
                    association_prefix = association[(selector, score_mode)][
                        stage_index
                    ]
                    d0_pair = validate_causal_prefix_pair(
                        prediction=d0_prefix.prediction,
                        target=original_target,
                        horizon=horizon,
                        observed_scan_ids=scan_ids[:horizon],
                    )
                    association_pair = validate_causal_prefix_pair(
                        prediction=evaluator_prediction(association_prefix),
                        target=canonical_target,
                        horizon=horizon,
                        observed_scan_ids=scan_ids[:horizon],
                    )
                    self.metrics[("D0", selector, score_mode, horizon)].update(d0_pair)
                    self.metrics[
                        (
                            self.association_method,
                            selector,
                            score_mode,
                            horizon,
                        )
                    ].update(association_pair)
        self.unit_count += 1
        self.reference_ids.add(reference_id)

    def finalize(self) -> dict[str, object]:
        rows = []
        for parent in self.parents:
            for selector in self.selectors:
                for score_mode in self.score_modes:
                    for horizon in self.horizons:
                        values = self.metrics[
                            (parent, selector, score_mode, horizon)
                        ].compute()
                        rows.append(
                            {
                                "population_id": "development",
                                "data_role": "DEV-SEL",
                                "reference_count": len(self.reference_ids),
                                "logical_unit_count": self.unit_count,
                                "parent": parent,
                                "association_config_id": (
                                    "D0"
                                    if parent == "D0"
                                    else self.association_config.config_id
                                ),
                                "selector": selector,
                                "score_mode": score_mode,
                                "score_reducer": "mean",
                                "T": horizon,
                                "t_mAP": values["t_mAP"],
                                "t_mAP50": values["t_mAP50"],
                                "t_mAP25": values["t_mAP25"],
                                "t_REC": values["t_REC"],
                                "prefix_overall_mAP": values["prefix_overall_mAP"],
                                "published_current_AP": values["local_current_AP"],
                                "source_commit": self.source_commit,
                                "R1_SHA": self.checkpoint_sha256,
                                "status": "MEASURED",
                                "reason": "",
                            }
                        )
        return {
            "metric_rows": rows,
            "revision_events": self.revision_events,
            "status": "PASS",
        }


__all__ = ["E4RevisionAccumulator", "exhaustive_fixed_u_equivalence"]
