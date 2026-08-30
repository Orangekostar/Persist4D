from __future__ import annotations

import copy
from pathlib import Path

import pytest

from scripts.prepare_rescene_strong_local import build_strong_authorization
from scripts.run_rescene_strong_local_training import (
    build_strong_launch_command,
    build_strong_launch_environment,
)
from utils.rescene_rootcause_preflight import RootCauseContractError, canonical_sha256
from utils.rescene_strong_local import (
    STRONG_VARIANTS,
    choose_diagnostic_base_variant,
    decide_strong_result,
    materialize_strong_config,
    strong_variant_gate,
    strong_variant_new_prefixes,
    strong_variant_state_dict_entries,
    validate_strong_variant_isolation,
)


def _base_config() -> dict[str, object]:
    return {
        "rootcause_preflight": {
            "target": "rescene_task_learning_root_cause_v1",
            "variant_manifest": "artifacts/rootcause/variant_manifest.json",
        },
        "general": {
            "project_name": "rescene_task_learning_root_cause_v1",
            "experiment_name": "R1",
            "save_dir": "external:checkpoint/rootcause_short/R1",
            "rootcause_objective_mode": "raw_sum",
        },
        "model": {"use_np_features": False, "scatter_type": "mean"},
        "data": {"batch_size": 2},
        "trainer": {"max_epochs": 450, "accumulate_grad_batches": 8},
        "scheduler": {"scheduler": {"total_steps": 29_700}},
        "callbacks": [
            {"dirpath": "external:checkpoint/rootcause_short/R1"},
            {"output_dir": "external:checkpoint/rootcause_short/R1"},
        ],
        "logging": [{"save_dir": "external:checkpoint/rootcause_short/R1"}],
    }


@pytest.mark.parametrize(
    ("variant", "changed_paths", "prefixes"),
    [
        ("A1", ["model.use_np_features"], ("model.np_feature_projection.",)),
        ("A2", ["model.scatter_type"], ("model.scatter_fn.",)),
        (
            "A1+A2",
            ["model.scatter_type", "model.use_np_features"],
            ("model.np_feature_projection.", "model.scatter_fn."),
        ),
    ],
)
def test_strong_variants_are_exact_native_switches(
    variant: str,
    changed_paths: list[str],
    prefixes: tuple[str, ...],
) -> None:
    base = _base_config()
    observed = materialize_strong_config(
        base,
        variant=variant,
        output="external:checkpoint/rescene_strong_local/" + variant,
    )

    isolation = validate_strong_variant_isolation(base, observed, variant=variant)

    assert STRONG_VARIANTS == ("A1", "A2", "A1+A2")
    assert isolation["changed_paths"] == changed_paths
    assert strong_variant_new_prefixes(variant) == prefixes
    assert (
        strong_variant_state_dict_entries(798, variant)
        == {
            "A1": 802,
            "A2": 839,
            "A1+A2": 843,
        }[variant]
    )
    assert observed["general"]["rootcause_objective_mode"] == "raw_sum"
    assert observed["trainer"]["accumulate_grad_batches"] == 8
    assert observed["scheduler"]["scheduler"]["total_steps"] == 29_700


def test_strong_variant_isolation_rejects_unregistered_change() -> None:
    base = _base_config()
    observed = materialize_strong_config(
        base,
        variant="A1",
        output="external:checkpoint/rescene_strong_local/A1",
    )
    observed["data"]["batch_size"] = 4

    with pytest.raises(RootCauseContractError, match="unauthorized"):
        validate_strong_variant_isolation(base, observed, variant="A1")


def test_strong_variant_gates_enforce_hierarchy() -> None:
    diagnostics = {
        "status": "pass",
        "gates": {
            "A1": {"authorized": True},
            "A2": {"diagnostic_evidence_pass": True},
        },
    }
    assert strong_variant_gate("A1", diagnostics=diagnostics)["authorized"] is True

    a1_failed = {"status": "pass", "all_gates_pass": False}
    assert (
        strong_variant_gate("A2", diagnostics=diagnostics, a1_result=a1_failed)[
            "authorized"
        ]
        is True
    )

    a1_passed = {"status": "pass", "all_gates_pass": True}
    assert (
        strong_variant_gate("A2", diagnostics=diagnostics, a1_result=a1_passed)[
            "status"
        ]
        == "gate_skipped"
    )

    a2_passed = {"status": "pass", "all_gates_pass": True}
    assert (
        strong_variant_gate(
            "A1+A2",
            diagnostics=diagnostics,
            a1_result=a1_passed,
            a2_result=a2_passed,
        )["authorized"]
        is True
    )


def test_a2_gate_requires_diagnostic_evidence_and_a1_result() -> None:
    diagnostics = {
        "status": "pass",
        "gates": {
            "A1": {"authorized": True},
            "A2": {"diagnostic_evidence_pass": False},
        },
    }
    assert strong_variant_gate("A2", diagnostics=diagnostics)["status"] == (
        "gate_skipped"
    )
    assert (
        strong_variant_gate(
            "A2",
            diagnostics=diagnostics,
            a1_result={"status": "pass", "all_gates_pass": False},
        )["authorized"]
        is False
    )


def _metric_runs(gain: float) -> dict[int, dict[str, float]]:
    return {
        seed: {
            "t_mAP": 0.20 + gain + offset,
            "t_mAP50": 0.30 + gain + offset,
            "t_mAP25": 0.40 + gain + offset,
            "overall_mAP": 0.35 + gain + offset,
            "stage1_mAP": 0.25 + gain + offset,
            "stage2_mAP": 0.27 + gain + offset,
        }
        for seed, offset in zip((45, 46, 47), (0.0, 0.001, -0.001))
    }


def test_strong_result_uses_the_rootcause_spatial_gate() -> None:
    result = decide_strong_result(
        variant="A1",
        base_runs=_metric_runs(0.0),
        variant_runs=_metric_runs(0.015),
        validation_leads={75: True, 90: True},
        contract_integrity=True,
    )

    assert result["all_gates_pass"] is True
    assert result["full_training_authorized"] is True
    assert result["paired_spatial_delta_mean"] == pytest.approx(0.015)
    assert result["paired_spatial_positive_seed_count"] == 3


def test_strong_result_fails_on_seed_overall_or_curve_gate() -> None:
    base = _metric_runs(0.0)
    variant = _metric_runs(0.015)
    variant[47]["stage1_mAP"] -= 0.04
    variant[47]["stage2_mAP"] -= 0.04
    variant[45]["overall_mAP"] = 0.1

    result = decide_strong_result(
        variant="A1",
        base_runs=base,
        variant_runs=variant,
        validation_leads={75: True, 90: False},
        contract_integrity=True,
    )

    assert result["all_gates_pass"] is False
    assert result["full_training_authorized"] is False
    assert result["gates"]["positive_for_all_paired_seeds"] is False
    assert result["gates"]["overall_map_not_lower_than_base"] is False
    assert result["gates"]["leads_validation_at_75_and_90"] is False


def test_diagnostic_base_is_bound_to_preregistered_rootcause_selection() -> None:
    short_decision = {
        "selected_variant": None,
        "epoch90_summary": {
            "R0": {
                "SpatialStageMean_mean": 0.25,
                "overall_mAP_mean": 0.30,
                "t_mAP_mean": 0.20,
            },
            "R1": {
                "SpatialStageMean_mean": 0.27,
                "overall_mAP_mean": 0.29,
                "t_mAP_mean": 0.21,
            },
        },
    }
    diagnostics = {"provenance": {"bindings": {"variant": "R1", "completed_epoch": 90}}}

    assert choose_diagnostic_base_variant(short_decision, diagnostics) == "R1"

    stale = copy.deepcopy(diagnostics)
    stale["provenance"]["bindings"]["variant"] = "R0"
    with pytest.raises(RootCauseContractError, match="diagnostic base"):
        choose_diagnostic_base_variant(short_decision, stale)


def _signed(payload: dict[str, object], field: str) -> dict[str, object]:
    payload[field] = canonical_sha256(payload)
    return payload


def _authorization_inputs() -> tuple[dict, dict, dict]:
    root = {
        "schema_version": 1,
        "status": "authorized",
        "source_commit": "1" * 40,
        "selected_variants": ["R0", "R1"],
        "variants": {
            "R0": {
                "config_sha256": "",
                "resolved_config": _base_config(),
            },
            "R1": {
                "config_sha256": "",
                "resolved_config": _base_config(),
            },
        },
        "initialization": {
            "common_state": {
                "bytes": 100,
                "sha256": "4" * 64,
                "reference": "external:checkpoint/rootcause_common/" + "4" * 64,
            },
            "pretrained": {
                "bytes": 200,
                "sha256": "5" * 64,
                "reference": "external:checkpoint/concerto/" + "5" * 64,
            },
            "tensor_state": {"tensor_count": 798},
        },
        "schedule": {
            "optimizer_steps_per_epoch": 66,
            "total_optimizer_steps": 29_700,
        },
    }
    for record in root["variants"].values():
        record["config_sha256"] = canonical_sha256(record["resolved_config"])
    _signed(root, "authorization_sha256")
    short = {
        "schema_version": 1,
        "status": "pass",
        "selected_variant": None,
        "epoch90_summary": {
            "R0": {
                "SpatialStageMean_mean": 0.25,
                "overall_mAP_mean": 0.30,
                "t_mAP_mean": 0.20,
            },
            "R1": {
                "SpatialStageMean_mean": 0.27,
                "overall_mAP_mean": 0.29,
                "t_mAP_mean": 0.21,
            },
        },
    }
    _signed(short, "content_sha256")
    diagnostics = {
        "schema_version": 1,
        "status": "pass",
        "gates": {
            "A1": {"authorized": True},
            "A2": {"diagnostic_evidence_pass": True},
        },
        "provenance": {"bindings": {"variant": "R1", "completed_epoch": 90}},
    }
    _signed(diagnostics, "content_sha256")
    return root, short, diagnostics


def _upstream_identities() -> dict[str, dict[str, object]]:
    return {
        name: {"bytes": index, "sha256": character * 64}
        for index, (name, character) in enumerate(
            (
                ("root_authorization", "7"),
                ("short_decision", "8"),
                ("diagnostics", "9"),
                ("root_learning_curves", "a"),
                ("root_official_like_epoch60", "b"),
                ("root_official_like_epoch90", "c"),
            ),
            start=10,
        )
    }


def test_strong_authorization_binds_gate_base_config_and_initialization() -> None:
    root, short, diagnostics = _authorization_inputs()

    authorization = build_strong_authorization(
        variant="A1",
        root_authorization=root,
        short_decision=short,
        diagnostics=diagnostics,
        source_commit="6" * 40,
        common_identity={"bytes": 100, "sha256": "4" * 64},
        pretrained_identity={"bytes": 200, "sha256": "5" * 64},
        input_identities=_upstream_identities(),
    )

    record = authorization["variants"]["A1"]
    assert authorization["status"] == "authorized"
    assert authorization["base_variant"] == "R1"
    assert authorization["checkpoint_namespace"] == "rescene_strong_local"
    assert record["expected_state_dict_entries"] == 802
    assert record["resolved_config"]["model"]["use_np_features"] is True
    assert record["resolved_config"]["model"]["scatter_type"] == "mean"
    assert (
        canonical_sha256(
            {
                key: value
                for key, value in authorization.items()
                if key != "authorization_sha256"
            }
        )
        == authorization["authorization_sha256"]
    )


def test_strong_authorization_rejects_stale_evidence_or_external_state() -> None:
    root, short, diagnostics = _authorization_inputs()
    short["epoch90_summary"]["R1"]["SpatialStageMean_mean"] = 0.1
    with pytest.raises(RootCauseContractError, match="hash"):
        build_strong_authorization(
            variant="A1",
            root_authorization=root,
            short_decision=short,
            diagnostics=diagnostics,
            source_commit="6" * 40,
            common_identity={"bytes": 100, "sha256": "4" * 64},
            pretrained_identity={"bytes": 200, "sha256": "5" * 64},
            input_identities={},
        )


def test_a2_authorization_binds_signed_a1_failure() -> None:
    root, short, diagnostics = _authorization_inputs()
    a1_result = {"status": "pass", "all_gates_pass": False}
    _signed(a1_result, "content_sha256")
    identities = _upstream_identities()
    identities["a1_result"] = {"bytes": 16, "sha256": "d" * 64}

    authorization = build_strong_authorization(
        variant="A2",
        root_authorization=root,
        short_decision=short,
        diagnostics=diagnostics,
        source_commit="6" * 40,
        common_identity={"bytes": 100, "sha256": "4" * 64},
        pretrained_identity={"bytes": 200, "sha256": "5" * 64},
        input_identities=identities,
        a1_result=a1_result,
    )

    assert authorization["status"] == "authorized"
    assert authorization["variants"]["A2"]["expected_state_dict_entries"] == 839

    a1_result["all_gates_pass"] = True
    with pytest.raises(RootCauseContractError, match="hash"):
        build_strong_authorization(
            variant="A2",
            root_authorization=root,
            short_decision=short,
            diagnostics=diagnostics,
            source_commit="6" * 40,
            common_identity={"bytes": 100, "sha256": "4" * 64},
            pretrained_identity={"bytes": 200, "sha256": "5" * 64},
            input_identities=identities,
            a1_result=a1_result,
        )


def test_strong_launcher_preserves_base_semantics_and_applies_native_switches(
    tmp_path,
) -> None:
    root, short, diagnostics = _authorization_inputs()
    authorization = build_strong_authorization(
        variant="A1",
        root_authorization=root,
        short_decision=short,
        diagnostics=diagnostics,
        source_commit="6" * 40,
        common_identity={"bytes": 100, "sha256": "4" * 64},
        pretrained_identity={"bytes": 200, "sha256": "5" * 64},
        input_identities=_upstream_identities(),
    )

    environment = build_strong_launch_environment(
        authorization=authorization,
        pretrained=tmp_path / "concerto.pth",
        common_state=tmp_path / "common.pt",
        output_dir=tmp_path / "A1",
        devices=(0, 1),
        inherited={"PATH": "/bin"},
    )
    command = build_strong_launch_command(authorization)

    assert environment["RESCENE_ROOTCAUSE_VARIANT"] == "A1"
    assert environment["RESCENE_ROOTCAUSE_OBJECTIVE_MODE"] == "raw_sum"
    assert environment["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert "model.use_np_features=true" in command
    assert "model.scatter_type=mean" in command
    assert "general.project_name=rescene_strong_local_v1" in command
    assert "data.batch_size=4" not in command

    root, short, diagnostics = _authorization_inputs()
    with pytest.raises(RootCauseContractError, match="pretrained"):
        build_strong_authorization(
            variant="A1",
            root_authorization=root,
            short_decision=short,
            diagnostics=diagnostics,
            source_commit="6" * 40,
            common_identity={"bytes": 100, "sha256": "4" * 64},
            pretrained_identity={"bytes": 201, "sha256": "5" * 64},
            input_identities={},
        )


def test_strong_local_pipeline_has_no_persist4d_selection_inputs() -> None:
    root = Path(__file__).resolve().parents[1]
    source = "\n".join(
        (root / path).read_text(encoding="utf-8")
        for path in (
            "utils/rescene_strong_local.py",
            "scripts/prepare_rescene_strong_local.py",
            "scripts/run_rescene_strong_local_training.py",
            "scripts/finalize_rescene_strong_local.py",
            "scripts/finalize_rescene_strong_local_full_evaluation.py",
        )
    )
    forbidden = (
        "artifacts/P6A",
        "artifacts/system_comparison",
        "artifacts/reviewer_closure_v3",
        "gap_recovery",
        "Protocol-B",
    )
    assert all(fragment not in source for fragment in forbidden)
