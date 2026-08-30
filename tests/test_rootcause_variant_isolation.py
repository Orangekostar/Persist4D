from __future__ import annotations

import copy

import pytest

from utils.rescene_rootcause_preflight import (
    MANDATORY_VARIANTS,
    MAX_SHORT_CURVE_VARIANTS,
    ROOTCAUSE_VARIANTS,
    RootCauseContractError,
    authorize_short_curve_variants,
    resolved_config_diff,
    validate_variant_isolation,
)


def _control() -> dict:
    return {
        "general": {
            "rootcause_objective_mode": "weighted",
            "rootcause_frozen_encoder_stochastic_policy": "current",
        },
        "data": {
            "batch_size": 2,
            "train_dataset": {"filter_out_classes": [0, 1, 255]},
        },
        "trainer": {"accumulate_grad_batches": 8, "max_epochs": 450},
        "loss": {"eos_coef": 0.2},
        "scheduler": {"scheduler": {"total_steps": 29_700}},
    }


def _changed(path: str, value: object) -> dict:
    result = copy.deepcopy(_control())
    cursor = result
    parts = path.split(".")
    for part in parts[:-1]:
        cursor = cursor[part]
    cursor[parts[-1]] = value
    return result


def test_variant_registry_has_exact_single_factor_allowlists() -> None:
    assert MANDATORY_VARIANTS == ("R0", "R1")
    assert MAX_SHORT_CURVE_VARIANTS == 4
    assert set(ROOTCAUSE_VARIANTS) == {"R0", "R1", "R2", "R3", "R4", "R5"}
    assert ROOTCAUSE_VARIANTS["R0"].allowed_paths == ()
    assert ROOTCAUSE_VARIANTS["R1"].allowed_paths == (
        "general.rootcause_objective_mode",
    )
    assert set(ROOTCAUSE_VARIANTS["R2"].allowed_paths) == {
        "data.batch_size",
        "trainer.accumulate_grad_batches",
    }
    assert ROOTCAUSE_VARIANTS["R3"].allowed_paths == (
        "general.rootcause_frozen_encoder_stochastic_policy",
    )
    assert ROOTCAUSE_VARIANTS["R4"].allowed_paths == (
        "data.train_dataset.filter_out_classes",
    )
    assert ROOTCAUSE_VARIANTS["R5"].allowed_paths == ("loss.eos_coef",)


@pytest.mark.parametrize(
    ("variant", "config"),
    [
        ("R0", _control()),
        ("R1", _changed("general.rootcause_objective_mode", "raw_sum")),
        (
            "R3",
            _changed(
                "general.rootcause_frozen_encoder_stochastic_policy",
                "drop_path_disabled",
            ),
        ),
        ("R4", _changed("data.train_dataset.filter_out_classes", [0, 1])),
        ("R5", _changed("loss.eos_coef", 0.1)),
    ],
)
def test_registered_single_factor_diff_passes(variant: str, config: dict) -> None:
    result = validate_variant_isolation(variant, _control(), config)
    assert result["variant"] == variant
    assert result["changed_paths"] == list(ROOTCAUSE_VARIANTS[variant].allowed_paths)


def test_r2_requires_both_batch_fields_and_preserves_effective_batch() -> None:
    variant = _changed("data.batch_size", 4)
    variant["trainer"]["accumulate_grad_batches"] = 4

    result = validate_variant_isolation("R2", _control(), variant, world_size=2)
    assert result["effective_global_batch"] == 32
    with pytest.raises(RootCauseContractError, match="allowed fields"):
        validate_variant_isolation("R2", _control(), _changed("data.batch_size", 4))


def test_variant_isolation_rejects_extra_change() -> None:
    variant = _changed("general.rootcause_objective_mode", "raw_sum")
    variant["loss"]["eos_coef"] = 0.1

    assert [item["path"] for item in resolved_config_diff(_control(), variant)] == [
        "general.rootcause_objective_mode",
        "loss.eos_coef",
    ]
    with pytest.raises(RootCauseContractError, match="unauthorized"):
        validate_variant_isolation("R1", _control(), variant)


def test_optional_variants_require_passed_gates_and_cap() -> None:
    assert authorize_short_curve_variants(
        ["R0", "R1", "R3"], gate_results={"R3": True}
    ) == ("R0", "R1", "R3")
    with pytest.raises(RootCauseContractError, match="gate"):
        authorize_short_curve_variants(
            ["R0", "R1", "R3"], gate_results={"R3": False}
        )
    with pytest.raises(RootCauseContractError, match="four"):
        authorize_short_curve_variants(
            ["R0", "R1", "R2", "R3", "R4"],
            gate_results={"R2": True, "R3": True, "R4": True},
        )
    with pytest.raises(RootCauseContractError, match="mandatory"):
        authorize_short_curve_variants(["R1"], gate_results={})
