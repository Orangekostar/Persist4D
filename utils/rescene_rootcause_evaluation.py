"""Checkpoint and result contracts for the ReScene root-cause study."""

from __future__ import annotations

import hashlib
import math
import statistics
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from utils.rescene_rootcause_preflight import (
    OPTIMIZER_STEPS_PER_EPOCH,
    RootCauseContractError,
    canonical_sha256,
    portable_reference,
    validate_portable_payload,
)

EVALUATION_SEEDS = (45, 46, 47)
METRIC_NAMES = (
    "t_mAP",
    "t_mAP50",
    "t_mAP25",
    "overall_mAP",
    "stage1_mAP",
    "stage2_mAP",
)
SAMPLER_CHECKPOINT_KEY = "p2_train_sampler_generator"


class RootCauseEvaluationError(ValueError):
    """Raised when checkpoint or metric evidence is not exact."""


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def validate_checkpoint_payload(
    payload: Mapping[str, Any],
    *,
    completed_epoch: int,
    expected_state_dict_entries: int = 798,
) -> dict[str, object]:
    """Require one full, epoch-boundary, exactly resumable Lightning state."""

    if completed_epoch not in (60, 90, 450):
        raise RootCauseEvaluationError("checkpoint epoch is outside the contract")
    if not isinstance(payload, Mapping):
        raise RootCauseEvaluationError("checkpoint payload is invalid")
    expected_step = completed_epoch * OPTIMIZER_STEPS_PER_EPOCH
    if payload.get("epoch") != completed_epoch - 1 or payload.get(
        "global_step"
    ) != expected_step:
        raise RootCauseEvaluationError("checkpoint is not at the exact epoch boundary")
    state = payload.get("state_dict")
    if not isinstance(state, Mapping) or len(state) != expected_state_dict_entries:
        raise RootCauseEvaluationError("checkpoint model state is incomplete")
    for name, tensor in state.items():
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(tensor, torch.Tensor)
            or tensor.layout != torch.strided
            or (
                (tensor.is_floating_point() or tensor.is_complex())
                and not torch.isfinite(tensor).all().item()
            )
        ):
            raise RootCauseEvaluationError("checkpoint model state is invalid")
    optimizers = payload.get("optimizer_states")
    schedulers = payload.get("lr_schedulers")
    if (
        not isinstance(optimizers, list)
        or len(optimizers) != 1
        or not isinstance(optimizers[0], Mapping)
        or not optimizers[0]
        or not isinstance(schedulers, list)
        or len(schedulers) != 1
        or not isinstance(schedulers[0], Mapping)
        or not schedulers[0]
    ):
        raise RootCauseEvaluationError("checkpoint optimizer or scheduler state is missing")
    sampler = payload.get(SAMPLER_CHECKPOINT_KEY)
    if (
        not isinstance(sampler, Mapping)
        or sampler.get("schema_version") != 1
        or sampler.get("resume_scope") != "completed_epoch_boundary_only"
        or sampler.get("mid_epoch_resume_supported") is not False
        or sampler.get("dataloader_prefetch_state_checkpointed") is not False
    ):
        raise RootCauseEvaluationError("checkpoint sampler contract is invalid")
    generator_state = sampler.get("generator_state")
    if (
        not isinstance(generator_state, torch.Tensor)
        or generator_state.dtype != torch.uint8
        or generator_state.ndim != 1
        or generator_state.numel() == 0
    ):
        raise RootCauseEvaluationError("checkpoint sampler state is invalid")
    try:
        torch.Generator().set_state(generator_state.detach().cpu())
    except RuntimeError as error:
        raise RootCauseEvaluationError("checkpoint sampler state is invalid") from error
    if not isinstance(payload.get("hyper_parameters"), Mapping):
        raise RootCauseEvaluationError("checkpoint config state is missing")
    return {
        "selected_epoch": completed_epoch,
        "selected_step": expected_step,
        "state_dict_entry_count": len(state),
        "optimizer_state_count": len(optimizers),
        "scheduler_state_count": len(schedulers),
        "sampler_generator_state_sha256": _tensor_sha256(generator_state),
        "sampler_resume_scope": sampler["resume_scope"],
        "mid_epoch_resume_supported": False,
        "dataloader_prefetch_state_checkpointed": False,
    }


def _require_candidate_binding(
    *,
    variant: str,
    authorization: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> None:
    variants = authorization.get("variants")
    if (
        authorization.get("status") != "authorized"
        or variant not in authorization.get("selected_variants", ())
        or not isinstance(variants, Mapping)
        or not isinstance(variants.get(variant), Mapping)
    ):
        raise RootCauseEvaluationError("variant authorization is invalid")
    initialization = authorization.get("initialization")
    if not isinstance(initialization, Mapping):
        raise RootCauseEvaluationError("variant initialization binding is invalid")
    expected = {
        "variant": variant,
        "source_commit": authorization.get("source_commit"),
        "variant_authorization_sha256": authorization.get(
            "authorization_sha256"
        ),
        "config_sha256": variants[variant].get("config_sha256"),
        "common_initialization_sha256": initialization.get("common_state", {}).get(
            "sha256"
        ),
        "pretrained_sha256": initialization.get("pretrained", {}).get("sha256"),
    }
    if any(candidate.get(key) != value for key, value in expected.items()):
        raise RootCauseEvaluationError("candidate binding differs from authorization")
    candidate_id = candidate.get("candidate_id")
    unsigned_candidate = dict(candidate)
    unsigned_candidate.pop("candidate_id", None)
    if (
        not isinstance(candidate_id, str)
        or len(candidate_id) != 64
        or canonical_sha256(unsigned_candidate) != candidate_id
    ):
        raise RootCauseEvaluationError("candidate binding is incomplete")


def build_checkpoint_manifest(
    *,
    variant: str,
    completed_epoch: int,
    authorization: Mapping[str, Any],
    candidate: Mapping[str, Any],
    file_identity: Mapping[str, Any],
    checkpoint_facts: Mapping[str, Any],
) -> dict[str, object]:
    """Bind a short-curve checkpoint without serializing its machine path."""

    _require_candidate_binding(
        variant=variant, authorization=authorization, candidate=candidate
    )
    byte_size = file_identity.get("bytes")
    sha256 = file_identity.get("sha256")
    if (
        not isinstance(byte_size, int)
        or isinstance(byte_size, bool)
        or byte_size <= 0
        or not isinstance(sha256, str)
        or len(sha256) != 64
    ):
        raise RootCauseEvaluationError("checkpoint file identity is invalid")
    expected_step = completed_epoch * OPTIMIZER_STEPS_PER_EPOCH
    if (
        checkpoint_facts.get("selected_epoch") != completed_epoch
        or checkpoint_facts.get("selected_step") != expected_step
    ):
        raise RootCauseEvaluationError("checkpoint facts differ from requested boundary")
    variants = authorization["variants"]
    initialization = authorization["initialization"]
    payload: dict[str, object] = {
        "schema_version": 1,
        "status": "pass",
        "experiment": "rescene_task_learning_root_cause_v1",
        "variant": variant,
        "checkpoint": {
            "logical_name": f"rootcause_{variant.lower()}_epoch_{completed_epoch:03d}",
            "reference": portable_reference(
                f"checkpoint/rootcause_short/{variant.lower()}/epoch{completed_epoch:03d}",
                sha256,
            ),
            "sha256": sha256,
            "bytes": byte_size,
            "creating_commit": authorization["source_commit"],
            "config_sha256": variants[variant]["config_sha256"],
            "upstream_checkpoint_sha256": initialization["pretrained"]["sha256"],
            "selected_epoch": completed_epoch,
            "selected_step": expected_step,
        },
        "resume_state": dict(checkpoint_facts),
        "bindings": {
            "candidate_id": candidate["candidate_id"],
            "variant_authorization_sha256": authorization["authorization_sha256"],
            "common_initialization_sha256": initialization["common_state"][
                "sha256"
            ],
            "pretrained_sha256": initialization["pretrained"]["sha256"],
        },
    }
    payload["content_sha256"] = canonical_sha256(payload)
    try:
        validate_portable_payload(payload)
    except RootCauseContractError as error:
        raise RootCauseEvaluationError(str(error)) from error
    return payload


def _validated_metrics(value: object) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != set(METRIC_NAMES):
        raise RootCauseEvaluationError("official-like metric schema differs")
    result = {}
    for name in METRIC_NAMES:
        try:
            number = float(value[name])
        except (TypeError, ValueError) as error:
            raise RootCauseEvaluationError("official-like metric is invalid") from error
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            raise RootCauseEvaluationError("official-like metric is invalid")
        result[name] = number
    return result


def summarize_epoch_runs(
    runs: Sequence[Mapping[str, Any]],
    *,
    variants: Sequence[str],
    completed_epoch: int,
) -> dict[str, dict[str, object]]:
    """Validate an exact variant-by-seed matrix and compute paired summaries."""

    variant_names = tuple(variants)
    expected = {
        (variant, seed) for variant in variant_names for seed in EVALUATION_SEEDS
    }
    observed = [(run.get("variant"), run.get("seed")) for run in runs]
    if len(observed) != len(expected) or set(observed) != expected:
        raise RootCauseEvaluationError("official-like run matrix differs")
    contract_hashes = set()
    grouped: dict[str, dict[int, dict[str, float]]] = {
        variant: {} for variant in variant_names
    }
    checkpoint_hashes: dict[str, set[object]] = {
        variant: set() for variant in variant_names
    }
    for run in runs:
        variant = str(run["variant"])
        seed = int(run["seed"])
        if (
            run.get("status") != "pass"
            or run.get("scope") != "official_like_t2"
            or run.get("completed_epoch") != completed_epoch
            or run.get("validation_sequence_count") != 154
        ):
            raise RootCauseEvaluationError("official-like run contract differs")
        metrics = _validated_metrics(run.get("metrics"))
        metrics["SpatialStageMean"] = (
            metrics["stage1_mAP"] + metrics["stage2_mAP"]
        ) / 2.0
        grouped[variant][seed] = metrics
        checkpoint_hashes[variant].add(run.get("checkpoint_sha256"))
        contract_hashes.add(run.get("contract_sha256"))
    if len(contract_hashes) != 1 or None in contract_hashes or any(
        len(values) != 1 or None in values for values in checkpoint_hashes.values()
    ):
        raise RootCauseEvaluationError("official-like provenance binding differs")
    control = grouped.get("R0")
    if control is None:
        raise RootCauseEvaluationError("R0 is required for paired summaries")
    summary: dict[str, dict[str, object]] = {}
    for variant in variant_names:
        rows = grouped[variant]
        record: dict[str, object] = {"seed_count": len(rows)}
        for metric in (*METRIC_NAMES, "SpatialStageMean"):
            values = [rows[seed][metric] for seed in EVALUATION_SEEDS]
            record[f"{metric}_mean"] = statistics.mean(values)
            record[f"{metric}_std"] = statistics.stdev(values)
        deltas = [
            rows[seed]["SpatialStageMean"]
            - control[seed]["SpatialStageMean"]
            for seed in EVALUATION_SEEDS
        ]
        record["paired_spatial_deltas"] = deltas
        record["paired_spatial_delta_mean"] = statistics.mean(deltas)
        record["paired_spatial_positive_seed_count"] = sum(
            value > 0 for value in deltas
        )
        summary[variant] = record
    return summary


def decide_full_candidate(
    summary: Mapping[str, Mapping[str, Any]],
    *,
    validation_leads: Mapping[str, Mapping[int, bool]],
    contract_integrity: Mapping[str, bool],
) -> dict[str, object]:
    """Apply the preregistered RC3 gate and select no more than one candidate."""

    control = summary.get("R0")
    if not isinstance(control, Mapping):
        raise RootCauseEvaluationError("R0 summary is missing")
    decisions: dict[str, dict[str, object]] = {}
    authorized = []
    for variant, record in summary.items():
        if variant == "R0":
            continue
        leads = validation_leads.get(variant, {})
        gates = {
            "positive_for_all_paired_seeds": record.get(
                "paired_spatial_positive_seed_count"
            )
            == len(EVALUATION_SEEDS),
            "mean_spatial_gain_at_least_one_point": float(
                record.get("paired_spatial_delta_mean", float("-inf"))
            )
            >= 0.01,
            "overall_map_not_lower_than_r0": float(
                record.get("overall_mAP_mean", float("-inf"))
            )
            >= float(control.get("overall_mAP_mean", float("inf"))),
            "leads_validation_at_75_and_90": leads.get(75) is True
            and leads.get(90) is True,
            "contract_integrity": contract_integrity.get(variant) is True,
        }
        all_pass = all(gates.values())
        decisions[variant] = {"gates": gates, "all_gates_pass": all_pass}
        if all_pass:
            authorized.append(variant)
    ranked = sorted(
        authorized,
        key=lambda name: (
            float(summary[name]["SpatialStageMean_mean"]),
            float(summary[name]["overall_mAP_mean"]),
            float(summary[name]["t_mAP_mean"]),
        ),
        reverse=True,
    )
    return {
        "status": "pass",
        "authorized_variants": ranked,
        "selected_variant": ranked[0] if ranked else None,
        "decisions": decisions,
        "selection_metric": "highest mean SpatialStageMean at epoch 90",
        "tie_breaks": ["mean overall_mAP", "mean t_mAP"],
    }
