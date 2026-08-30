from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.rescene_rootcause_preflight import (
    SELECTED_SHORT_VARIANTS,
    build_variant_records,
    compose_variant_config,
    isolation_view,
    portable_variant_config,
)
from scripts.run_rescene_rootcause_training import (
    CANDIDATE_RECORD_NAME,
    RootCauseLaunchError,
    authorize_unique_candidate,
    build_launch_command,
    build_launch_environment,
    parse_devices,
)
from utils.rescene_rootcause_preflight import validate_variant_isolation

PRETRAINED_REF = "external:checkpoint/concerto_pretrained/" + "1" * 64
COMMON_REF = "external:checkpoint/rootcause_common/" + "2" * 64


def _configs() -> dict[str, dict[str, object]]:
    configs = {}
    for variant in SELECTED_SHORT_VARIANTS:
        composed = compose_variant_config(
            variant,
            pretrained=PRETRAINED_REF,
            common_state=COMMON_REF,
            common_sha256="2" * 64,
            output=f"external:checkpoint/rootcause_short/{variant}",
        )
        configs[variant] = portable_variant_config(
            composed,
            variant=variant,
            pretrained_reference=PRETRAINED_REF,
            common_reference=COMMON_REF,
        )
    return configs


def test_selected_variants_have_exact_resolved_config_isolation() -> None:
    configs = _configs()
    control = isolation_view(configs["R0"])
    records = build_variant_records(configs)

    assert SELECTED_SHORT_VARIANTS == ("R0", "R1", "R2", "R4")
    assert set(records) == set(SELECTED_SHORT_VARIANTS)
    for variant in SELECTED_SHORT_VARIANTS:
        observed = validate_variant_isolation(
            variant, control, isolation_view(configs[variant]), world_size=2
        )
        assert records[variant]["isolation"] == observed
        assert len(records[variant]["config_sha256"]) == 64
    assert records["R0"]["isolation"]["changed_paths"] == []
    assert records["R1"]["isolation"]["changed_paths"] == [
        "general.rootcause_objective_mode"
    ]
    assert set(records["R2"]["isolation"]["changed_paths"]) == {
        "data.batch_size",
        "trainer.accumulate_grad_batches",
    }
    assert records["R2"]["isolation"]["effective_global_batch"] == 32
    assert records["R4"]["isolation"]["changed_paths"] == [
        "data.train_dataset.filter_out_classes"
    ]


def test_composed_variants_retain_full_schedule_and_short_horizon() -> None:
    configs = _configs()
    for config in configs.values():
        assert config["trainer"]["max_epochs"] == 450
        assert config["trainer"]["check_val_every_n_epoch"] == 15
        assert config["scheduler"]["scheduler"]["total_steps"] == 29_700
        callbacks = config["callbacks"]
        assert callbacks[1]["completed_epochs"] == [60, 90, 450]
        assert callbacks[2]["completed_epoch"] == 90


def test_runtime_locations_normalize_to_same_portable_config(tmp_path: Path) -> None:
    runtime = compose_variant_config(
        "R0",
        pretrained=tmp_path / "concerto.pth",
        common_state=tmp_path / "common.pt",
        common_sha256="2" * 64,
        output=tmp_path / "R0",
    )

    portable = portable_variant_config(
        runtime,
        variant="R0",
        pretrained_reference=PRETRAINED_REF,
        common_reference=COMMON_REF,
    )

    assert portable == _configs()["R0"]


def test_portable_config_supports_separate_strong_local_namespace(
    tmp_path: Path,
) -> None:
    runtime = compose_variant_config(
        "R0",
        pretrained=tmp_path / "concerto.pth",
        common_state=tmp_path / "common.pt",
        common_sha256="2" * 64,
        output=tmp_path / "A1",
    )

    portable = portable_variant_config(
        runtime,
        variant="A1",
        pretrained_reference=PRETRAINED_REF,
        common_reference=COMMON_REF,
        output_namespace="rescene_strong_local",
    )

    assert portable["general"]["save_dir"] == (
        "external:checkpoint/rescene_strong_local/A1"
    )
    assert all(
        callback.get("dirpath", portable["general"]["save_dir"])
        == portable["general"]["save_dir"]
        for callback in portable["callbacks"]
    )


def test_candidate_record_is_immutable_and_resume_exact(tmp_path: Path) -> None:
    candidate = {
        "schema_version": 1,
        "variant": "R0",
        "candidate_id": "a" * 64,
    }
    assert authorize_unique_candidate(tmp_path, candidate) == "fresh"
    assert authorize_unique_candidate(tmp_path, candidate) == "resume"
    record = tmp_path / CANDIDATE_RECORD_NAME
    assert json.loads(record.read_text(encoding="ascii")) == candidate

    changed = dict(candidate, candidate_id="b" * 64)
    with pytest.raises(RootCauseLaunchError, match="candidate contract"):
        authorize_unique_candidate(tmp_path, changed)


def test_candidate_rejects_unowned_checkpoint(tmp_path: Path) -> None:
    (tmp_path / "epoch=090.ckpt").write_bytes(b"unowned")
    with pytest.raises(RootCauseLaunchError, match="unowned checkpoint"):
        authorize_unique_candidate(
            tmp_path,
            {"schema_version": 1, "variant": "R0", "candidate_id": "a" * 64},
        )


@pytest.mark.parametrize("value", ["0", "0,0", "-1,2", "0,1,2", "a,1"])
def test_devices_require_two_distinct_nonnegative_indices(value: str) -> None:
    with pytest.raises((RootCauseLaunchError, ValueError)):
        parse_devices(value)
    assert parse_devices("0,2") == (0, 2)


def test_launch_environment_and_command_bind_exact_variant(tmp_path: Path) -> None:
    environment = build_launch_environment(
        variant="R2",
        devices=(0, 2),
        pretrained=tmp_path / "concerto.pth",
        common_state=tmp_path / "common.pt",
        common_sha256="2" * 64,
        output_dir=tmp_path / "R2",
        inherited={"PATH": "/usr/bin", "PYTHONPATH": "existing"},
    )
    command = build_launch_command("R2")

    assert environment["CUDA_VISIBLE_DEVICES"] == "0,2"
    assert environment["RESCENE_ROOTCAUSE_VARIANT"] == "R2"
    assert environment["RESCENE_ROOTCAUSE_OBJECTIVE_MODE"] == "weighted"
    assert environment["RESCENE_ROOTCAUSE_COMMON_SHA256"] == "2" * 64
    assert command[-2:] == [
        "data.batch_size=4",
        "trainer.accumulate_grad_batches=4",
    ]
    assert build_launch_command("R1")[-2:] == [
        "--config-name",
        "config_rescene4d_concerto_rootcause",
    ]
    assert build_launch_environment(
        variant="R1",
        devices=(0, 1),
        pretrained=tmp_path / "concerto.pth",
        common_state=tmp_path / "common.pt",
        common_sha256="2" * 64,
        output_dir=tmp_path / "R1",
        inherited={},
    )["RESCENE_ROOTCAUSE_OBJECTIVE_MODE"] == "raw_sum"
    assert build_launch_command("R4")[-1] == (
        "data.train_dataset.filter_out_classes=[0,1]"
    )


def test_training_sources_do_not_read_persist4d_selection_evidence() -> None:
    root = Path(__file__).resolve().parents[1]
    source = "\n".join(
        (root / path).read_text(encoding="utf-8")
        for path in (
            "scripts/rescene_rootcause_preflight.py",
            "scripts/run_rescene_rootcause_training.py",
            "utils/rescene_rootcause_preflight.py",
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
