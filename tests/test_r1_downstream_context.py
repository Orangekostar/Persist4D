from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = REPO_ROOT / "configs/r1_downstream_validation/default.yaml"
PROTOCOL_PATH = REPO_ROOT / "artifacts/P6A/protocol_b_manifest.json"
R1_CHECKPOINT = Path(
    "/mnt/shared/ww/persist4d-rescene-finalization/checkpoints/"
    "629ff7624dcac15e6022906e808e2e05b3ec61c60a1116ab0e278f0cfd2368dd.ckpt"
)
CONCERTO_PRETRAIN = Path(
    "/mnt/shared/ww/persist4d-node1-7-migration/checkpoints/concerto_base.pth"
)
METADATA_PATH = Path("/home/ww/3RScan.json")


def _context_module():
    return importlib.import_module("scripts.r1_downstream_context")


def test_contract_freezes_only_r1_checkpoint_and_downstream_semantics() -> None:
    context = _context_module()

    contract = context.load_r1_contract(CONTRACT_PATH)

    assert contract["checkpoint"] == {
        "bytes": 754813672,
        "completed_epoch": 390,
        "completed_step": 25740,
        "reference": (
            "external:checkpoint/rootcause_full/r1/epoch390/"
            "629ff7624dcac15e6022906e808e2e05b3ec61c60a1116ab0e278f0cfd2368dd"
        ),
        "sha256": (
            "629ff7624dcac15e6022906e808e2e05b3ec61c60a1116ab0e278f0cfd2368dd"
        ),
        "state_dict_entries": 798,
        "training_config_sha256": (
            "3846415b6de3ea12b7b08d39ea20398916dd685c99ce7ad33e42df17d9ac6536"
        ),
    }
    assert contract["methods"] == ["FullHistory", "B2", "B4"]
    assert contract["tracker_methods"] == ["B2", "B4"]
    assert contract["state_update_horizons"] == [1, 2, 3, 4, 5]
    assert contract["report_horizons"] == [2, 4, 5]
    assert contract["score_reducers"] == ["mean", "latest", "max"]
    assert contract["profile"] == {
        "measured_repeats": 10,
        "report_horizons": [2, 4, 5],
        "subset_rule": "first_master_per_reference_scene_canonical",
        "warmup_repeats": 5,
    }


def test_protocol_validation_requires_exact_645_entry_common_prefix() -> None:
    context = _context_module()
    contract = context.load_r1_contract(CONTRACT_PATH)

    result = context.validate_protocol_b(PROTOCOL_PATH, contract)

    assert result == {
        "cache_entry_count": 645,
        "master_count": 43,
        "order_count": 3,
        "reference_scene_cluster_count": 6,
        "sequence_count": 129,
        "sha256": (
            "246497165612699b103d0d79d5503025cb2cd14466aad3ab149d4fe82884ecbe"
        ),
        "status": "pass",
    }

    tampered = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    tampered["protocol"]["seed"] = 46
    changed = REPO_ROOT / ".superpowers/r1_downstream_tampered_protocol.json"
    changed.parent.mkdir(parents=True, exist_ok=True)
    changed.write_text(json.dumps(tampered), encoding="ascii")
    try:
        with pytest.raises(context.R1ContextError, match="Protocol-B SHA256"):
            context.validate_protocol_b(changed, contract)
    finally:
        changed.unlink(missing_ok=True)


def test_checkpoint_identity_rejects_wrong_size_name_and_symlink(tmp_path: Path) -> None:
    context = _context_module()
    expected_sha = "a" * 64
    checkpoint = tmp_path / f"{expected_sha}.ckpt"
    checkpoint.write_bytes(b"r1")

    resolved = context.validate_external_file_identity(
        checkpoint,
        expected_sha256=expected_sha,
        expected_bytes=2,
        require_digest_filename=True,
        label="R1 checkpoint",
    )
    assert resolved == checkpoint.resolve()

    with pytest.raises(context.R1ContextError, match="byte size"):
        context.validate_external_file_identity(
            checkpoint,
            expected_sha256=expected_sha,
            expected_bytes=3,
            require_digest_filename=True,
            label="R1 checkpoint",
        )
    wrong_name = tmp_path / "checkpoint.ckpt"
    wrong_name.write_bytes(b"r1")
    with pytest.raises(context.R1ContextError, match="filename"):
        context.validate_external_file_identity(
            wrong_name,
            expected_sha256=expected_sha,
            expected_bytes=2,
            require_digest_filename=True,
            label="R1 checkpoint",
        )
    link = tmp_path / f"link-{expected_sha}.ckpt"
    link.symlink_to(checkpoint)
    with pytest.raises(context.R1ContextError, match="regular non-symlink"):
        context.validate_external_file_identity(
            link,
            expected_sha256=expected_sha,
            expected_bytes=2,
            require_digest_filename=False,
            label="R1 checkpoint",
        )


def test_runtime_composition_is_inference_only_and_uses_registered_pretrain(
    tmp_path: Path,
) -> None:
    context = _context_module()
    pretrain = tmp_path / "concerto_base.pth"
    pretrain.write_bytes(b"pretrain")

    runtime, _memory = context.compose_r1_runtime_config(pretrain)

    assert runtime.general.seed == 45
    assert runtime.general.gpus == 1
    assert runtime.general.train_mode is False
    assert runtime.model.return_query_features is True
    assert runtime.model.config.temporal_window == 2
    assert runtime.backbone.name == str(pretrain.resolve())


def test_tracker_settings_must_match_frozen_p6a_exactly() -> None:
    context = _context_module()
    contract = context.load_r1_contract(CONTRACT_PATH)
    p6a = yaml.safe_load((REPO_ROOT / "conf/p6a/default.yaml").read_text())

    assert context.validate_tracker_settings(contract, p6a) == {
        "methods": ["B2", "B4"],
        "status": "pass",
    }

    p6a["baselines"]["b4"]["association_threshold"] = 0.49
    with pytest.raises(context.R1ContextError, match="B4 settings"):
        context.validate_tracker_settings(contract, p6a)


def test_cache_provenance_uses_frozen_r1_digest_without_checkpoint_io() -> None:
    context = _context_module()
    contract = context.load_r1_contract(CONTRACT_PATH)

    local, full = context.build_r1_cache_provenance(
        contract=contract,
        source_commit="a" * 40,
        config_documents={"p6a": b"a", "runtime": b"x: b\n"},
        protocol_manifest={"protocol": "b"},
    )

    assert local == {
        "checkpoint_sha256": (
            "629ff7624dcac15e6022906e808e2e05b3ec61c60a1116ab0e278f0cfd2368dd"
        ),
        "config_sha256": (
            "36e2ef9028ab3e2667bd2a516682b98e87cd29dc0635283dad5729896293cbce"
        ),
        "dataset_sha256": (
            "a6aa565a1cd6cfa4350fe96f9990ac21e3c0e7bed73993dbc77b6c39ca413dbf"
        ),
        "source_commit": "a" * 40,
    }
    assert full == {
        "checkpoint_sha256": local["checkpoint_sha256"],
        "config_sha256": local["config_sha256"],
        "protocol_sha256": (
            "246497165612699b103d0d79d5503025cb2cd14466aad3ab149d4fe82884ecbe"
        ),
        "source_commit": "a" * 40,
    }

    with pytest.raises(context.R1ContextError, match="source commit"):
        context.build_r1_cache_provenance(
            contract=contract,
            source_commit="dirty",
            config_documents={"p6a": b"a", "runtime": b"x: b\n"},
            protocol_manifest={"protocol": "b"},
        )


@pytest.mark.skipif(
    not all(path.is_file() for path in (R1_CHECKPOINT, CONCERTO_PRETRAIN, METADATA_PATH)),
    reason="registered external experiment inputs are unavailable",
)
def test_cpu_setup_assembles_exact_protocol_without_loading_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context_module()
    from scripts.evaluate_persist4d_p6a import expected_cache_keys

    monkeypatch.chdir("/home/ww/paper5")
    setup = context.build_r1_setup(
        contract_path=CONTRACT_PATH,
        protocol_path=PROTOCOL_PATH,
        checkpoint_path=R1_CHECKPOINT,
        pretrained_path=CONCERTO_PRETRAIN,
        metadata_path=METADATA_PATH,
        data_root=Path("/home/ww/paper5"),
        source_commit="a" * 40,
        device_name=None,
    )

    assert setup.device is None
    assert setup.system is None
    assert len(expected_cache_keys(setup.protocol)) == 645
    assert setup.local_provenance["checkpoint_sha256"] == (
        "629ff7624dcac15e6022906e808e2e05b3ec61c60a1116ab0e278f0cfd2368dd"
    )
    assert setup.full_provenance["protocol_sha256"] == (
        "246497165612699b103d0d79d5503025cb2cd14466aad3ab149d4fe82884ecbe"
    )
    assert setup.runtime_config.general.train_mode is False
    assert setup.runtime_config.backbone.name == str(CONCERTO_PRETRAIN.resolve())


def test_data_working_directory_must_equal_explicit_data_root(tmp_path: Path) -> None:
    context = _context_module()
    data_root = tmp_path / "data-repository"
    other = tmp_path / "worktree"
    data_root.mkdir()
    other.mkdir()

    assert context.validate_data_working_directory(data_root, cwd=data_root) == (
        data_root.resolve()
    )
    with pytest.raises(context.R1ContextError, match="working directory"):
        context.validate_data_working_directory(data_root, cwd=other)


def test_checkpoint_payload_requires_exact_r1_epoch_step_and_state_count() -> None:
    context = _context_module()
    contract = context.load_r1_contract(CONTRACT_PATH)
    payload = {
        "epoch": 389,
        "global_step": 25740,
        "state_dict": {f"parameter.{index}": index for index in range(798)},
    }

    assert context.validate_r1_checkpoint_payload(payload, contract) == {
        "completed_epoch": 390,
        "completed_step": 25740,
        "state_dict_entries": 798,
        "status": "pass",
    }

    payload["global_step"] = 25739
    with pytest.raises(context.R1ContextError, match="epoch/step"):
        context.validate_r1_checkpoint_payload(payload, contract)
