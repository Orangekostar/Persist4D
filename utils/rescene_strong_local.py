"""Fail-closed contracts for ReScene-native strong-local variants."""

from __future__ import annotations

import copy
import math
import statistics
from collections.abc import Mapping
from typing import Any

from utils.rescene_rootcause_evaluation import EVALUATION_SEEDS, METRIC_NAMES
from utils.rescene_rootcause_preflight import (
    RootCauseContractError,
    resolved_config_diff,
)

STRONG_VARIANTS = ("A1", "A2", "A1+A2")
_STRUCTURAL_VALUES = {
    "A1": {"use_np_features": True, "scatter_type": "mean"},
    "A2": {"use_np_features": False, "scatter_type": "adaptive"},
    "A1+A2": {"use_np_features": True, "scatter_type": "adaptive"},
}
_STRUCTURAL_PATHS = {
    "A1": ("model.use_np_features",),
    "A2": ("model.scatter_type",),
    "A1+A2": ("model.scatter_type", "model.use_np_features"),
}
_NEW_PREFIXES = {
    "A1": ("model.np_feature_projection.",),
    "A2": ("model.scatter_fn.",),
    "A1+A2": ("model.np_feature_projection.", "model.scatter_fn."),
}
_NEW_STATE_DICT_ENTRIES = {"A1": 4, "A2": 41, "A1+A2": 45}


def _require_variant(variant: str) -> None:
    if variant not in STRONG_VARIANTS:
        raise RootCauseContractError("strong-local variant is not registered")


def strong_variant_new_prefixes(variant: str) -> tuple[str, ...]:
    _require_variant(variant)
    return _NEW_PREFIXES[variant]


def strong_variant_state_dict_entries(common_entries: int, variant: str) -> int:
    _require_variant(variant)
    if (
        isinstance(common_entries, bool)
        or not isinstance(common_entries, int)
        or common_entries <= 0
    ):
        raise RootCauseContractError("common state entry count is invalid")
    return common_entries + _NEW_STATE_DICT_ENTRIES[variant]


def _operationally_normalized(config: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(config))
    preflight = result.get("rootcause_preflight")
    if isinstance(preflight, dict):
        for field in ("target", "variant_manifest"):
            if field in preflight:
                preflight[field] = "OPERATIONAL"
    general = result.get("general")
    if isinstance(general, dict):
        for field in ("project_name", "experiment_name", "save_dir"):
            if field in general:
                general[field] = "OPERATIONAL"
    for callback in result.get("callbacks", []):
        if isinstance(callback, dict):
            for field in ("dirpath", "output_dir"):
                if field in callback:
                    callback[field] = "OPERATIONAL"
    for logger in result.get("logging", []):
        if isinstance(logger, dict) and "save_dir" in logger:
            logger["save_dir"] = "OPERATIONAL"
    return result


def materialize_strong_config(
    base_config: Mapping[str, Any],
    *,
    variant: str,
    output: str,
) -> dict[str, Any]:
    """Apply one registered native switch and separate operational identity."""

    _require_variant(variant)
    result = copy.deepcopy(dict(base_config))
    model = result.get("model")
    general = result.get("general")
    if (
        not isinstance(model, dict)
        or model.get("use_np_features") is not False
        or model.get("scatter_type") != "mean"
        or not isinstance(general, dict)
    ):
        raise RootCauseContractError("strong-local base structure is invalid")
    model.update(_STRUCTURAL_VALUES[variant])
    general.update(
        {
            "project_name": "rescene_strong_local_v1",
            "experiment_name": variant,
            "save_dir": output,
        }
    )
    preflight = result.get("rootcause_preflight")
    if isinstance(preflight, dict):
        preflight["target"] = "rescene_strong_local_v1"
        preflight["variant_manifest"] = (
            "artifacts/rescene_task_learning_root_cause_v1/strong_local/"
            f"{variant}/variant_manifest.json"
        )
    for callback in result.get("callbacks", []):
        if isinstance(callback, dict):
            for field in ("dirpath", "output_dir"):
                if field in callback:
                    callback[field] = output
    for logger in result.get("logging", []):
        if isinstance(logger, dict) and "save_dir" in logger:
            logger["save_dir"] = output
    validate_strong_variant_isolation(base_config, result, variant=variant)
    return result


def validate_strong_variant_isolation(
    base_config: Mapping[str, Any],
    observed_config: Mapping[str, Any],
    *,
    variant: str,
) -> dict[str, object]:
    """Require the scientific diff to equal the variant's native switches."""

    _require_variant(variant)
    observed_model = observed_config.get("model")
    if not isinstance(observed_model, Mapping) or any(
        observed_model.get(field) != value
        for field, value in _STRUCTURAL_VALUES[variant].items()
    ):
        raise RootCauseContractError("strong-local structural value differs")
    changes = resolved_config_diff(
        _operationally_normalized(base_config),
        _operationally_normalized(observed_config),
    )
    paths = tuple(change["path"] for change in changes)
    if paths != _STRUCTURAL_PATHS[variant]:
        raise RootCauseContractError(
            "strong-local variant has unauthorized configuration changes"
        )
    return {
        "status": "pass",
        "variant": variant,
        "changed_paths": list(paths),
        "new_parameter_prefixes": list(strong_variant_new_prefixes(variant)),
    }


def strong_variant_gate(
    variant: str,
    *,
    diagnostics: Mapping[str, Any],
    a1_result: Mapping[str, Any] | None = None,
    a2_result: Mapping[str, Any] | None = None,
) -> dict[str, object]:
    """Apply the preregistered A1, A2, and combination hierarchy."""

    _require_variant(variant)
    gates = diagnostics.get("gates")
    if diagnostics.get("status") != "pass" or not isinstance(gates, Mapping):
        raise RootCauseContractError("decoder diagnostic decision is invalid")
    a1_gate = gates.get("A1")
    a2_gate = gates.get("A2")
    if not isinstance(a1_gate, Mapping) or not isinstance(a2_gate, Mapping):
        raise RootCauseContractError("decoder diagnostic gates are incomplete")

    authorized = False
    reason = "upstream gate failed"
    if variant == "A1":
        authorized = a1_gate.get("authorized") is True
        reason = "required first native structural experiment after SD0"
    elif variant == "A2":
        diagnostic_pass = a2_gate.get("diagnostic_evidence_pass") is True
        a1_complete = a1_result is not None and a1_result.get("status") == "pass"
        a1_gates = a1_result.get("gates") if a1_complete else None
        a1_stable_spatial_benefit = (
            isinstance(a1_gates, Mapping)
            and a1_gates.get("positive_for_all_paired_seeds") is True
        )
        a1_insufficient = (
            a1_stable_spatial_benefit
            and a1_result.get("all_gates_pass") is False
        )
        authorized = diagnostic_pass and a1_insufficient
        if not diagnostic_pass:
            reason = "superpoint diagnostic evidence gate failed"
        elif not a1_complete:
            reason = "A1 result is pending"
        elif not a1_stable_spatial_benefit:
            reason = "A1 had no stable paired spatial benefit"
        elif not a1_insufficient:
            reason = "A1 passed; architecture expansion stopped"
        else:
            reason = "superpoint evidence passed and A1 was insufficient"
    else:
        a1_pass = (
            a1_result is not None
            and a1_result.get("status") == "pass"
            and a1_result.get("all_gates_pass") is True
        )
        a2_pass = (
            a2_result is not None
            and a2_result.get("status") == "pass"
            and a2_result.get("all_gates_pass") is True
        )
        authorized = a1_pass and a2_pass
        reason = (
            "A1 and A2 independently passed"
            if authorized
            else "A1 and A2 independent gates have not both passed"
        )
    return {
        "status": "authorized" if authorized else "gate_skipped",
        "authorized": authorized,
        "variant": variant,
        "reason": reason,
    }


def decide_strong_result(
    *,
    variant: str,
    base_runs: Mapping[int, Mapping[str, Any]],
    variant_runs: Mapping[int, Mapping[str, Any]],
    validation_leads: Mapping[int, bool],
    contract_integrity: bool,
) -> dict[str, object]:
    """Apply the RC3 spatial gate to one native structural variant."""

    _require_variant(variant)
    expected_seeds = set(EVALUATION_SEEDS)
    if set(base_runs) != expected_seeds or set(variant_runs) != expected_seeds:
        raise RootCauseContractError("strong-local evaluation seed matrix differs")

    def validated(
        rows: Mapping[int, Mapping[str, Any]],
    ) -> dict[int, dict[str, float]]:
        result = {}
        for seed in EVALUATION_SEEDS:
            record = rows[seed]
            values: dict[str, float] = {}
            for metric in METRIC_NAMES:
                try:
                    value = float(record[metric])
                except (KeyError, TypeError, ValueError) as error:
                    raise RootCauseContractError(
                        "strong-local evaluation metric is invalid"
                    ) from error
                if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                    raise RootCauseContractError(
                        "strong-local evaluation metric is invalid"
                    )
                values[metric] = value
            values["SpatialStageMean"] = (
                values["stage1_mAP"] + values["stage2_mAP"]
            ) / 2.0
            result[seed] = values
        return result

    base = validated(base_runs)
    observed = validated(variant_runs)

    def summary(rows: Mapping[int, Mapping[str, float]]) -> dict[str, object]:
        record: dict[str, object] = {"seed_count": len(EVALUATION_SEEDS)}
        for metric in (*METRIC_NAMES, "SpatialStageMean"):
            values = [rows[seed][metric] for seed in EVALUATION_SEEDS]
            record[f"{metric}_mean"] = statistics.mean(values)
            record[f"{metric}_std"] = statistics.stdev(values)
        return record

    base_summary = summary(base)
    variant_summary = summary(observed)
    deltas = [
        observed[seed]["SpatialStageMean"] - base[seed]["SpatialStageMean"]
        for seed in EVALUATION_SEEDS
    ]
    delta_mean = statistics.mean(deltas)
    positive_count = sum(delta > 0.0 for delta in deltas)
    gates = {
        "positive_for_all_paired_seeds": positive_count == len(EVALUATION_SEEDS),
        "mean_spatial_gain_at_least_one_point": delta_mean >= 0.01,
        "overall_map_not_lower_than_base": variant_summary["overall_mAP_mean"]
        >= base_summary["overall_mAP_mean"],
        "leads_validation_at_75_and_90": validation_leads.get(75) is True
        and validation_leads.get(90) is True,
        "contract_integrity": contract_integrity is True,
    }
    all_pass = all(gates.values())
    return {
        "schema_version": 1,
        "status": "pass",
        "variant": variant,
        "gates": gates,
        "all_gates_pass": all_pass,
        "full_training_authorized": all_pass,
        "full_training_status": "authorized" if all_pass else "gate_skipped",
        "paired_spatial_deltas": deltas,
        "paired_spatial_delta_mean": delta_mean,
        "paired_spatial_positive_seed_count": positive_count,
        "base_summary": base_summary,
        "variant_summary": variant_summary,
    }


def choose_diagnostic_base_variant(
    short_decision: Mapping[str, Any], diagnostics: Mapping[str, Any]
) -> str:
    """Bind SP0 to the root-cause semantics used by the formal diagnostics."""

    summaries = short_decision.get("epoch90_summary")
    if not isinstance(summaries, Mapping) or not summaries:
        raise RootCauseContractError("root-cause epoch-90 summary is invalid")
    selected = short_decision.get("selected_variant")
    if selected is not None:
        if not isinstance(selected, str) or selected not in summaries:
            raise RootCauseContractError("root-cause full selection is invalid")
        expected = selected
    else:
        ranked: list[tuple[tuple[float, float, float], str]] = []
        for name, record in summaries.items():
            if not isinstance(name, str) or not isinstance(record, Mapping):
                raise RootCauseContractError("root-cause epoch-90 summary is invalid")
            try:
                score = tuple(
                    float(record[field])
                    for field in (
                        "SpatialStageMean_mean",
                        "overall_mAP_mean",
                        "t_mAP_mean",
                    )
                )
            except (KeyError, TypeError, ValueError) as error:
                raise RootCauseContractError(
                    "root-cause epoch-90 summary is invalid"
                ) from error
            if any(not math.isfinite(value) for value in score):
                raise RootCauseContractError("root-cause epoch-90 summary is invalid")
            ranked.append((score, name))
        ranked.sort(reverse=True)
        if len(ranked) > 1 and ranked[0][0] == ranked[1][0]:
            raise RootCauseContractError("root-cause base selection is tied")
        expected = ranked[0][1]

    provenance = diagnostics.get("provenance")
    bindings = provenance.get("bindings") if isinstance(provenance, Mapping) else None
    observed = bindings.get("variant") if isinstance(bindings, Mapping) else None
    completed_epoch = (
        bindings.get("completed_epoch") if isinstance(bindings, Mapping) else None
    )
    if (
        observed != expected
        or isinstance(completed_epoch, bool)
        or not isinstance(completed_epoch, int)
        or completed_epoch < 90
        or completed_epoch > 450
        or completed_epoch % 15 != 0
    ):
        raise RootCauseContractError(
            "diagnostic base differs from root-cause selection"
        )
    return expected
