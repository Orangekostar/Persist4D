from __future__ import annotations

from pathlib import Path

import pytest

from scripts.crosswindow_cache import build_data_roles, resolve_assets
from scripts.crosswindow_campaign import (
    CampaignError,
    atomic_write_run_state,
    load_resume_state,
    new_run_state,
)

PARENT = "96edca52d9d1cab7781bfa2a273baf9fe52b6a7d"
INSTRUCTION_SHA = "f7ee448782b0e8e4d2940a5b4e60748ce46d3dd07457eaf2460ab0f9006c01c2"


def _data_contract() -> dict[str, object]:
    inventory = [
        {"reference_id": reference, "role": "development"}
        for reference in ("alpha", "beta", "gamma", "delta", "epsilon")
    ]
    inventory.extend(
        [
            {"reference_id": "train-a", "role": "adaptation"},
            {"reference_id": "native-a", "role": "additional_native_refs"},
        ]
    )
    return {"native_reference_inventory": inventory}


def test_development_split_is_reference_stable_and_disjoint() -> None:
    expected = {
        "DEV-CAL": ["delta", "beta"],
        "DEV-SEL": ["gamma", "alpha", "epsilon"],
        "adaptation": ["train-a"],
        "additional_native_refs": ["native-a"],
    }

    forward = build_data_roles(
        _data_contract(),
        available_references=["alpha", "beta", "gamma", "delta", "epsilon"],
    )
    reverse = build_data_roles(
        _data_contract(),
        available_references=["epsilon", "delta", "gamma", "beta", "alpha"],
    )

    assert forward == expected
    assert reverse == expected
    assert set(forward["DEV-CAL"]).isdisjoint(forward["DEV-SEL"])


def test_asset_resolution_uses_cli_then_environment_then_fallback() -> None:
    resolved = resolve_assets(
        explicit={"data_root": "/cli/data"},
        environ={
            "PERSIST4D_DATA_ROOT": "/env/data",
            "PERSIST4D_R1_CHECKPOINT": "/env/r1.ckpt",
        },
        fallback={
            "external:data_root": "/fallback/data",
            "external:r1_checkpoint": "/fallback/r1.ckpt",
            "external:rio_metadata": "/fallback/3RScan.json",
        },
    )

    assert resolved.values["data_root"] == "/cli/data"
    assert resolved.sources["data_root"] == "cli"
    assert resolved.values["r1_checkpoint"] == "/env/r1.ckpt"
    assert resolved.sources["r1_checkpoint"] == "environment"
    assert resolved.values["rio_metadata"] == "/fallback/3RScan.json"
    assert resolved.sources["rio_metadata"] == "fallback"
    assert resolved.values["pb_base_cache_root"] is None
    assert resolved.sources["pb_base_cache_root"] == "unresolved"


def test_asset_resolution_derives_metric_spec_from_data_directory(
    tmp_path: Path,
) -> None:
    specification = tmp_path / "processed/rio/rio.yaml"
    specification.parent.mkdir(parents=True)
    specification.write_text("data: rio\n", encoding="utf-8")

    resolved = resolve_assets(
        explicit={"data_root": str(tmp_path)},
        environ={},
        fallback={},
    )

    assert resolved.values["metric_dataset_spec"] == str(specification)
    assert resolved.sources["metric_dataset_spec"] == "derived:data_root"


def test_resume_rejects_changed_identity(tmp_path: Path) -> None:
    path = tmp_path / "RUN_STATE.json"
    state = new_run_state(
        parent=PARENT,
        instruction_sha=INSTRUCTION_SHA,
        config_sha="a" * 64,
        data_sha="b" * 64,
    )
    atomic_write_run_state(path, state)

    with pytest.raises(CampaignError, match="identity"):
        load_resume_state(
            path,
            parent=PARENT,
            instruction_sha=INSTRUCTION_SHA,
            config_sha="c" * 64,
            data_sha="b" * 64,
        )


def test_resume_round_trip_preserves_completed_units(tmp_path: Path) -> None:
    path = tmp_path / "RUN_STATE.json"
    state = new_run_state(
        parent=PARENT,
        instruction_sha=INSTRUCTION_SHA,
        config_sha="a" * 64,
        data_sha="b" * 64,
    )
    state["completed_units"] = ["dev:alpha:order-0"]
    atomic_write_run_state(path, state)

    loaded = load_resume_state(
        path,
        parent=PARENT,
        instruction_sha=INSTRUCTION_SHA,
        config_sha="a" * 64,
        data_sha="b" * 64,
    )

    assert loaded == state
