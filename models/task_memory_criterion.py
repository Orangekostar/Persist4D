"""Q-INDEP and tracklet-aware TALA loss composition."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from datasets.task_memory_episode import StageMeta
from models.criterion import SetCriterion
from models.misc import is_dist_avail_and_initialized
from models.task_memory_routing import EntityRoute
from models.task_memory_supervision import (
    QualifiedIdentity,
    TaskMemoryAssignment,
    TrainingIdentityLedger,
    build_independent_assignment,
    build_tala_assignment,
    prediction_stage_ids,
)


class TaskMemoryCriterionError(ValueError):
    """Raised when task-memory loss inputs violate the training contract."""


@dataclass(frozen=True)
class TaskMemoryLossResult:
    losses: Mapping[str, Tensor]
    final_assignment: TaskMemoryAssignment
    aux_assignments: tuple[TaskMemoryAssignment, ...]
    diagnostics: Mapping[str, int | str]


def _normalized_num_masks(
    targets: Sequence[Mapping[str, object]], device: torch.device
) -> Tensor:
    count = sum(int(target["labels"].numel()) for target in targets)
    value = torch.tensor([count], dtype=torch.float32, device=device)
    if is_dist_avail_and_initialized():
        torch.distributed.all_reduce(value)
        value /= torch.distributed.get_world_size()
    return value.clamp_min(1).detach()


def _connected_mask_zero(outputs: Mapping[str, object]) -> Tensor:
    masks = outputs.get("pred_masks")
    if isinstance(masks, (str, bytes)) or not isinstance(masks, Sequence) or not masks:
        raise TaskMemoryCriterionError("prediction masks are unavailable")
    tensors = [mask for mask in masks if isinstance(mask, Tensor)]
    if len(tensors) != len(masks):
        raise TaskMemoryCriterionError("prediction masks must be tensors")
    return sum((mask.sum() * 0.0 for mask in tensors), tensors[0].new_zeros(()))


class TaskMemoryCriterion(nn.Module):
    """Compose existing SetCriterion losses with explicit temporal assignments."""

    def __init__(
        self,
        base_criterion: SetCriterion,
        *,
        matcher_mode: str,
        post_conditioning_aux_start: int,
        empty_current_weight: float = 1.0,
    ) -> None:
        super().__init__()
        if not isinstance(base_criterion, SetCriterion):
            raise TaskMemoryCriterionError("base criterion must be SetCriterion")
        if matcher_mode not in {"independent", "tala"}:
            raise TaskMemoryCriterionError("matcher_mode must be independent or tala")
        if (
            isinstance(post_conditioning_aux_start, bool)
            or not isinstance(post_conditioning_aux_start, int)
            or post_conditioning_aux_start < 0
        ):
            raise TaskMemoryCriterionError("post-conditioning aux boundary is invalid")
        if (
            isinstance(empty_current_weight, bool)
            or not isinstance(empty_current_weight, (int, float))
            or not 0.0 <= float(empty_current_weight) < float("inf")
        ):
            raise TaskMemoryCriterionError("empty-current weight is invalid")
        self.base_criterion = base_criterion
        self.matcher_mode = matcher_mode
        self.post_conditioning_aux_start = post_conditioning_aux_start
        self.empty_current_weight = float(empty_current_weight)
        self.weight_dict = dict(base_criterion.weight_dict)
        self.weight_dict["loss_tala_empty_current"] = self.empty_current_weight

    def _assignment(
        self,
        outputs: Mapping[str, object],
        targets: Sequence[Mapping[str, object]],
        *,
        mask_type: str,
        matcher_mode: str,
        route: EntityRoute | None,
        ledgers: Sequence[TrainingIdentityLedger] | None,
        identity_keys: Sequence[Sequence[QualifiedIdentity]] | None,
        ambiguity_metadata: Sequence[object] | None,
        stage_meta: Sequence[StageMeta] | None,
    ) -> TaskMemoryAssignment:
        if matcher_mode == "independent":
            return build_independent_assignment(
                outputs=outputs,
                targets=targets,
                matcher=self.base_criterion.matcher,
                mask_type=mask_type,
            )
        if (
            route is None
            or ledgers is None
            or identity_keys is None
            or ambiguity_metadata is None
            or stage_meta is None
        ):
            raise TaskMemoryCriterionError(
                "TALA requires route, ledger, and stage metadata"
            )
        return build_tala_assignment(
            outputs=outputs,
            targets=targets,
            matcher=self.base_criterion.matcher,
            mask_type=mask_type,
            route=route,
            ledgers=ledgers,
            identity_keys=identity_keys,
            ambiguity_metadata=ambiguity_metadata,
            stage_meta=stage_meta,
        )

    def _safe_mask_losses(
        self,
        outputs: Mapping[str, object],
        targets: Sequence[Mapping[str, object]],
        assignment: TaskMemoryAssignment,
        *,
        num_masks: Tensor,
        mask_type: str,
    ) -> dict[str, Tensor]:
        zero = _connected_mask_zero(outputs)
        totals = {"loss_mask": zero, "loss_dice": zero}
        for batch_index, pair in enumerate(assignment.indices):
            if pair[0].numel() == 0:
                continue
            sample_outputs = {"pred_masks": [outputs["pred_masks"][batch_index]]}
            sample_losses = self.base_criterion.get_loss(
                "masks",
                sample_outputs,
                [targets[batch_index]],
                [pair],
                num_masks,
                mask_type,
            )
            for name, value in totals.items():
                totals[name] = value + sample_losses[name]
        return totals

    def _empty_current_loss(
        self,
        outputs: Mapping[str, object],
        assignment: TaskMemoryAssignment,
        stage_meta: Sequence[StageMeta] | None,
    ) -> Tensor:
        zero = _connected_mask_zero(outputs)
        if assignment.matcher_mode != "tala":
            return zero
        if stage_meta is None or len(stage_meta) != len(assignment.indices):
            raise TaskMemoryCriterionError("TALA empty-mask metadata is misaligned")
        terms = []
        for batch_index, queries in enumerate(assignment.empty_current_queries):
            if queries.numel() == 0:
                continue
            mask_logits = outputs["pred_masks"][batch_index]
            stage_ids = prediction_stage_ids(
                stage_meta[batch_index], mask_logits.shape[0]
            )
            latest = int(stage_ids.max().item())
            current = (stage_ids == latest).to(device=mask_logits.device)
            if not current.any().item():
                raise TaskMemoryCriterionError("current stage has no mask rows")
            query_index = queries.to(device=mask_logits.device)
            selected = mask_logits[current][:, query_index]
            terms.append(
                F.binary_cross_entropy_with_logits(
                    selected,
                    torch.zeros_like(selected),
                    reduction="mean",
                )
            )
        return zero if not terms else torch.stack(terms).mean()

    def _layer_losses(
        self,
        outputs: Mapping[str, object],
        targets: Sequence[Mapping[str, object]],
        assignment: TaskMemoryAssignment,
        *,
        num_masks: Tensor,
        mask_type: str,
        stage_meta: Sequence[StageMeta] | None,
    ) -> dict[str, Tensor]:
        losses: dict[str, Tensor] = {}
        for loss_name in self.base_criterion.losses:
            if loss_name == "masks":
                losses.update(
                    self._safe_mask_losses(
                        outputs,
                        targets,
                        assignment,
                        num_masks=num_masks,
                        mask_type=mask_type,
                    )
                )
            else:
                losses.update(
                    self.base_criterion.get_loss(
                        loss_name,
                        outputs,
                        targets,
                        assignment.indices,
                        num_masks,
                        mask_type,
                    )
                )
        losses["loss_tala_empty_current"] = self._empty_current_loss(
            outputs, assignment, stage_meta
        )
        return losses

    def compute_with_assignments(
        self,
        outputs: Mapping[str, object],
        targets: Sequence[Mapping[str, object]],
        *,
        mask_type: str,
        route: EntityRoute | None = None,
        ledgers: Sequence[TrainingIdentityLedger] | None = None,
        identity_keys: Sequence[Sequence[QualifiedIdentity]] | None = None,
        ambiguity_metadata: Sequence[object] | None = None,
        stage_meta: Sequence[StageMeta] | None = None,
    ) -> TaskMemoryLossResult:
        if (
            not isinstance(outputs, Mapping)
            or not isinstance(mask_type, str)
            or not mask_type
        ):
            raise TaskMemoryCriterionError("criterion inputs are invalid")
        logits = outputs.get("pred_logits")
        if not isinstance(logits, Tensor) or logits.ndim != 3:
            raise TaskMemoryCriterionError("criterion logits are invalid")
        outputs_without_aux = {
            key: value for key, value in outputs.items() if key != "aux_outputs"
        }
        final_assignment = self._assignment(
            outputs_without_aux,
            targets,
            mask_type=mask_type,
            matcher_mode=self.matcher_mode,
            route=route,
            ledgers=ledgers,
            identity_keys=identity_keys,
            ambiguity_metadata=ambiguity_metadata,
            stage_meta=stage_meta,
        )
        num_masks = _normalized_num_masks(targets, logits.device)
        losses = self._layer_losses(
            outputs_without_aux,
            targets,
            final_assignment,
            num_masks=num_masks,
            mask_type=mask_type,
            stage_meta=stage_meta,
        )
        if self.base_criterion.use_contrastive_loss:
            losses.update(
                self.base_criterion.loss_segment_contrastive(
                    outputs,
                    targets,
                    final_assignment.indices,
                    num_masks,
                    mask_type,
                )
            )
            losses.update(
                self.base_criterion.loss_aux_contrastive(
                    outputs,
                    targets,
                    final_assignment.indices,
                    num_masks,
                    mask_type,
                )
            )

        aux_outputs = outputs.get("aux_outputs", [])
        if isinstance(aux_outputs, (str, bytes)) or not isinstance(
            aux_outputs, Sequence
        ):
            raise TaskMemoryCriterionError("aux outputs must be a sequence")
        aux_assignments = []
        for index, aux_output in enumerate(aux_outputs):
            mode = (
                "tala"
                if self.matcher_mode == "tala"
                and index >= self.post_conditioning_aux_start
                else "independent"
            )
            assignment = self._assignment(
                aux_output,
                targets,
                mask_type=mask_type,
                matcher_mode=mode,
                route=route,
                ledgers=ledgers,
                identity_keys=identity_keys,
                ambiguity_metadata=ambiguity_metadata,
                stage_meta=stage_meta,
            )
            aux_assignments.append(assignment)
            layer_losses = self._layer_losses(
                aux_output,
                targets,
                assignment,
                num_masks=num_masks,
                mask_type=mask_type,
                stage_meta=stage_meta,
            )
            for name, value in layer_losses.items():
                suffixed = f"{name}_{index}"
                losses[suffixed] = value
                if name == "loss_tala_empty_current":
                    self.weight_dict[suffixed] = self.empty_current_weight

        diagnostics: dict[str, int | str] = {
            "aux_layer_count": len(aux_assignments),
            "matcher_mode": self.matcher_mode,
            "post_conditioning_aux_start": self.post_conditioning_aux_start,
            **dict(final_assignment.diagnostics),
        }
        return TaskMemoryLossResult(
            losses=MappingProxyType(losses),
            final_assignment=final_assignment,
            aux_assignments=tuple(aux_assignments),
            diagnostics=MappingProxyType(diagnostics),
        )

    def forward(
        self,
        outputs: Mapping[str, object],
        targets: Sequence[Mapping[str, object]],
        *,
        mask_type: str,
        route: EntityRoute | None = None,
        ledgers: Sequence[TrainingIdentityLedger] | None = None,
        identity_keys: Sequence[Sequence[QualifiedIdentity]] | None = None,
        ambiguity_metadata: Sequence[object] | None = None,
        stage_meta: Sequence[StageMeta] | None = None,
    ) -> Mapping[str, Tensor]:
        return self.compute_with_assignments(
            outputs,
            targets,
            mask_type=mask_type,
            route=route,
            ledgers=ledgers,
            identity_keys=identity_keys,
            ambiguity_metadata=ambiguity_metadata,
            stage_meta=stage_meta,
        ).losses


__all__ = [
    "TaskMemoryCriterion",
    "TaskMemoryCriterionError",
    "TaskMemoryLossResult",
]
