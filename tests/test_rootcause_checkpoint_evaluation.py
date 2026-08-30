from __future__ import annotations

import copy
import json

import pytest
import torch

from scripts.evaluate_rescene_rootcause_checkpoint import (
    compose_evaluation_config,
    evaluation_contract,
    portable_evaluation_config,
    summarize_outputs,
)
from utils.rescene_rootcause_evaluation import (
    RootCauseEvaluationError,
    build_checkpoint_manifest,
    build_full_checkpoint_manifest,
    decide_full_candidate,
    summarize_epoch_runs,
    validate_checkpoint_manifest_binding,
    validate_checkpoint_payload,
    validate_full_checkpoint_payload,
)
from utils.rescene_rootcause_preflight import canonical_sha256


def test_evaluation_config_is_shared_and_portable(tmp_path) -> None:
    pretrained = tmp_path / "concerto.pth"
    config = compose_evaluation_config(pretrained)
    portable = portable_evaluation_config(
        config,
        pretrained=pretrained,
        pretrained_reference="external:checkpoint/concerto/" + "a" * 64,
    )

    assert config.general.train_mode is False
    assert config.general.rootcause_fail_closed_runtime is False
    assert config.data.batch_size == 1
    assert config.data.test_batch_size == 1
    assert config.data.num_workers == 4
    assert config.trainer.precision == "32-true"
    assert str(tmp_path) not in str(portable)
    contract = evaluation_contract()
    assert contract["seeds"] == [45, 46, 47]
    assert contract["validation_sequence_count"] == 154
    assert len(contract["sha256"]) == 64


def _checkpoint(completed_epoch: int = 60) -> dict[str, object]:
    generator = torch.Generator().manual_seed(45)
    return {
        "epoch": completed_epoch - 1,
        "global_step": completed_epoch * 66,
        "state_dict": {"model.weight": torch.ones(2)},
        "optimizer_states": [{"state": {0: {}}, "param_groups": [{"params": [0]}]}],
        "lr_schedulers": [{"last_epoch": completed_epoch * 66}],
        "p2_train_sampler_generator": {
            "schema_version": 1,
            "resume_scope": "completed_epoch_boundary_only",
            "mid_epoch_resume_supported": False,
            "dataloader_prefetch_state_checkpointed": False,
            "generator_state": generator.get_state(),
        },
        "hyper_parameters": {"general": {"seed": 45}},
    }


def _authorization() -> dict[str, object]:
    return {
        "status": "authorized",
        "source_commit": "1" * 40,
        "authorization_sha256": "2" * 64,
        "selected_variants": ["R0", "R1", "R2", "R4"],
        "variants": {
            name: {"config_sha256": str(index) * 64}
            for index, name in enumerate(("R0", "R1", "R2", "R4"), start=3)
        },
        "initialization": {
            "common_state": {"sha256": "7" * 64},
            "pretrained": {"sha256": "8" * 64},
        },
        "schedule": {
            "optimizer_steps_per_epoch": 66,
            "total_optimizer_steps": 29_700,
        },
    }


def _candidate(variant: str = "R0") -> dict[str, object]:
    authorization = _authorization()
    candidate = {
        "variant": variant,
        "source_commit": authorization["source_commit"],
        "variant_authorization_sha256": authorization["authorization_sha256"],
        "config_sha256": authorization["variants"][variant]["config_sha256"],
        "common_initialization_sha256": "7" * 64,
        "pretrained_sha256": "8" * 64,
    }
    candidate["candidate_id"] = canonical_sha256(candidate)
    return candidate


def _checkpoint_facts() -> dict[str, object]:
    facts = validate_checkpoint_payload(
        _checkpoint(), completed_epoch=60, expected_state_dict_entries=1
    )
    facts["training_config_sha256"] = _authorization()["variants"]["R0"][
        "config_sha256"
    ]
    return facts


def test_checkpoint_payload_requires_full_resume_state() -> None:
    facts = validate_checkpoint_payload(
        _checkpoint(), completed_epoch=60, expected_state_dict_entries=1
    )

    assert facts["selected_epoch"] == 60
    assert facts["selected_step"] == 3_960
    assert facts["state_dict_entry_count"] == 1
    assert facts["optimizer_state_count"] == 1
    assert facts["scheduler_state_count"] == 1
    assert len(facts["sampler_generator_state_sha256"]) == 64

    for missing in (
        "state_dict",
        "optimizer_states",
        "lr_schedulers",
        "p2_train_sampler_generator",
    ):
        invalid = _checkpoint()
        invalid.pop(missing)
        with pytest.raises(RootCauseEvaluationError, match="checkpoint"):
            validate_checkpoint_payload(
                invalid, completed_epoch=60, expected_state_dict_entries=1
            )


def test_checkpoint_payload_rejects_wrong_boundary_and_sampler_scope() -> None:
    invalid = _checkpoint()
    invalid["global_step"] = 3_959
    with pytest.raises(RootCauseEvaluationError, match="boundary"):
        validate_checkpoint_payload(
            invalid, completed_epoch=60, expected_state_dict_entries=1
        )

    invalid = _checkpoint()
    invalid["p2_train_sampler_generator"]["resume_scope"] = "mid_epoch"
    with pytest.raises(RootCauseEvaluationError, match="sampler"):
        validate_checkpoint_payload(
            invalid, completed_epoch=60, expected_state_dict_entries=1
        )


def test_full_checkpoint_payload_allows_validation_selected_boundary() -> None:
    payload = _checkpoint(completed_epoch=315)

    facts = validate_full_checkpoint_payload(
        payload, completed_epoch=315, expected_state_dict_entries=1
    )

    assert facts["selected_epoch"] == 315
    assert facts["selected_step"] == 20_790
    with pytest.raises(RootCauseEvaluationError, match="outside the contract"):
        validate_checkpoint_payload(
            payload, completed_epoch=315, expected_state_dict_entries=1
        )

def test_checkpoint_manifest_binds_authorization_candidate_and_file() -> None:
    manifest = build_checkpoint_manifest(
        variant="R0",
        completed_epoch=60,
        authorization=_authorization(),
        candidate=_candidate(),
        file_identity={"bytes": 123, "sha256": "a" * 64},
        checkpoint_facts=_checkpoint_facts(),
    )

    assert manifest["status"] == "pass"
    assert manifest["checkpoint"]["reference"].endswith("/" + "a" * 64)
    assert manifest["checkpoint"]["selected_epoch"] == 60
    assert manifest["checkpoint"]["selected_step"] == 3_960
    assert manifest["bindings"]["candidate_id"] == _candidate()["candidate_id"]
    assert len(manifest["content_sha256"]) == 64
    assert validate_checkpoint_manifest_binding(
        manifest, authorization=_authorization()
    ) == "R0"

    candidate = _candidate()
    candidate["variant_authorization_sha256"] = "b" * 64
    with pytest.raises(RootCauseEvaluationError, match="candidate"):
        build_checkpoint_manifest(
            variant="R0",
            completed_epoch=60,
            authorization=_authorization(),
            candidate=candidate,
            file_identity={"bytes": 123, "sha256": "a" * 64},
            checkpoint_facts=_checkpoint_facts(),
        )


def test_full_checkpoint_manifest_binds_completed_training_and_selected_epoch() -> None:
    facts = validate_full_checkpoint_payload(
        _checkpoint(completed_epoch=315),
        completed_epoch=315,
        expected_state_dict_entries=1,
    )
    facts["training_config_sha256"] = _authorization()["variants"]["R0"][
        "config_sha256"
    ]

    manifest = build_full_checkpoint_manifest(
        variant="R0",
        completed_epoch=315,
        authorization=_authorization(),
        candidate=_candidate(),
        file_identity={"bytes": 123, "sha256": "a" * 64},
        checkpoint_facts=facts,
        full_training_manifest_sha256="b" * 64,
        full_training_completed_epoch=450,
    )

    assert manifest["stage"] == "full_candidate"
    assert manifest["checkpoint"]["selected_epoch"] == 315
    assert manifest["full_training"]["completed_epoch"] == 450
    assert validate_checkpoint_manifest_binding(
        manifest, authorization=_authorization()
    ) == "R0"


def test_checkpoint_manifest_rejects_stale_or_malformed_authorization_binding() -> None:
    authorization = _authorization()
    manifest = build_checkpoint_manifest(
        variant="R0",
        completed_epoch=60,
        authorization=authorization,
        candidate=_candidate(),
        file_identity={"bytes": 123, "sha256": "a" * 64},
        checkpoint_facts=_checkpoint_facts(),
    )

    stale = copy.deepcopy(manifest)
    stale["bindings"]["variant_authorization_sha256"] = "b" * 64
    with pytest.raises(RootCauseEvaluationError, match="authorization"):
        validate_checkpoint_manifest_binding(stale, authorization=authorization)

    malformed = copy.deepcopy(manifest)
    malformed["checkpoint"] = []
    with pytest.raises(RootCauseEvaluationError, match="manifest"):
        validate_checkpoint_manifest_binding(malformed, authorization=authorization)

    malformed_authorization = copy.deepcopy(authorization)
    malformed_authorization["initialization"]["common_state"] = []
    with pytest.raises(RootCauseEvaluationError, match="authorization"):
        validate_checkpoint_manifest_binding(
            manifest, authorization=malformed_authorization
        )


def _run(
    variant: str,
    seed: int,
    *,
    stage1: float,
    stage2: float,
    overall: float,
    t_map: float,
    epoch: int = 90,
) -> dict[str, object]:
    return {
        "status": "pass",
        "scope": "official_like_t2",
        "variant": variant,
        "completed_epoch": epoch,
        "seed": seed,
        "source_commit": "d" * 40,
        "contract_sha256": "c" * 64,
        "variant_authorization_sha256": "e" * 64,
        "checkpoint_manifest_sha256": variant.encode().hex().ljust(64, "1"),
        "checkpoint_sha256": variant.encode().hex().ljust(64, "0"),
        "evaluation_config_sha256": "f" * 64,
        "validation_sequence_count": 154,
        "elapsed_seconds": 10.0,
        "metrics": {
            "t_mAP": t_map,
            "t_mAP50": t_map + 0.1,
            "t_mAP25": t_map + 0.2,
            "overall_mAP": overall,
            "stage1_mAP": stage1,
            "stage2_mAP": stage2,
        },
        "SpatialStageMean": (stage1 + stage2) / 2.0,
    }


def _epoch_runs() -> list[dict[str, object]]:
    runs = []
    for seed, offset in zip((45, 46, 47), (0.0, 0.001, -0.001)):
        runs.append(
            _run(
                "R0",
                seed,
                stage1=0.30 + offset,
                stage2=0.32 + offset,
                overall=0.36 + offset,
                t_map=0.25 + offset,
            )
        )
        runs.append(
            _run(
                "R1",
                seed,
                stage1=0.32 + offset,
                stage2=0.33 + offset,
                overall=0.37 + offset,
                t_map=0.24 + offset,
            )
        )
    return runs


def test_epoch_summary_uses_paired_spatial_stage_mean() -> None:
    summary = summarize_epoch_runs(
        _epoch_runs(), variants=("R0", "R1"), completed_epoch=90
    )

    assert summary["R0"]["SpatialStageMean_mean"] == pytest.approx(0.31)
    assert summary["R1"]["SpatialStageMean_mean"] == pytest.approx(0.325)
    assert summary["R1"]["paired_spatial_delta_mean"] == pytest.approx(0.015)
    assert summary["R1"]["paired_spatial_positive_seed_count"] == 3

    duplicated = _epoch_runs()
    duplicated.append(copy.deepcopy(duplicated[0]))
    with pytest.raises(RootCauseEvaluationError, match="matrix"):
        summarize_epoch_runs(
            duplicated, variants=("R0", "R1"), completed_epoch=90
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("source_commit", "0" * 40),
        ("variant_authorization_sha256", "0" * 64),
        ("evaluation_config_sha256", "0" * 64),
        ("checkpoint_manifest_sha256", "0" * 64),
        ("SpatialStageMean", 0.99),
    ),
)
def test_epoch_summary_rejects_mixed_or_inconsistent_provenance(
    field: str, replacement: object
) -> None:
    runs = _epoch_runs()
    runs[-1][field] = replacement

    with pytest.raises(RootCauseEvaluationError, match="provenance|metric"):
        summarize_epoch_runs(runs, variants=("R0", "R1"), completed_epoch=90)


def test_epoch_outputs_render_fixed_schema_csv(tmp_path) -> None:
    paths = []
    for index, run in enumerate(_epoch_runs()):
        path = tmp_path / f"run-{index}.json"
        path.write_text(json.dumps(run), encoding="utf-8")
        paths.append(path)

    outputs = summarize_outputs(
        run_paths=paths, variants=("R0", "R1"), completed_epoch=90
    )

    assert set(outputs) == {"per_seed.csv", "summary.csv", "summary.json"}
    assert outputs["per_seed.csv"].decode("ascii").count("\n") == 7
    assert outputs["summary.csv"].decode("ascii").count("\n") == 3


def test_full_candidate_decision_applies_all_five_local_gates() -> None:
    summary = summarize_epoch_runs(
        _epoch_runs(), variants=("R0", "R1"), completed_epoch=90
    )
    decision = decide_full_candidate(
        summary,
        validation_leads={"R1": {75: True, 90: True}},
        contract_integrity={"R1": True},
    )

    assert decision["selected_variant"] == "R1"
    assert decision["authorized_variants"] == ["R1"]
    assert decision["decisions"]["R1"]["all_gates_pass"] is True

    for failed in (
        {75: False, 90: True},
        {75: True, 90: False},
    ):
        rejected = decide_full_candidate(
            summary,
            validation_leads={"R1": failed},
            contract_integrity={"R1": True},
        )
        assert rejected["selected_variant"] is None

    rejected = decide_full_candidate(
        summary,
        validation_leads={"R1": {75: True, 90: True}},
        contract_integrity={"R1": False},
    )
    assert rejected["selected_variant"] is None
