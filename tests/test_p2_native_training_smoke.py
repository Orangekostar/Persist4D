import ast
import hashlib
import inspect
import json
import os
import subprocess
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts import run_p2_native_smoke as smoke
from utils import p2_preflight

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT = Path(
    os.environ.get(
        "P2_NATIVE_SMOKE_REPORT",
        REPO_ROOT / "artifacts" / "P2" / "native_smoke_report.json",
    )
)

RUNTIME_ENVIRONMENT_COMPONENTS = {
    "cumm",
    "flash_attn",
    "hydra",
    "nvidia_cuda_libraries",
    "omegaconf",
    "pointnet2",
    "pytorch_lightning",
    "python",
    "spconv",
    "torch",
    "torch_scatter",
}
RUNTIME_ENVIRONMENT_PYTHON_SOURCE_COMPONENTS = {
    "hydra",
    "omegaconf",
    "pytorch_lightning",
}

EXPECTED_INPUT_PROVENANCE = {
    "dataset": "3RScan",
    "sample_name": "scene0112_00-scene0112_01",
    "temporal_window": 2,
    "processed_point_clouds": [
        {
            "reference": "repo:data/processed/rio/train/0112_00.npy",
            "sha256": "ecbbbc3fbcd52a3f752c43b8c983d3fb204206bc22549219d294be240a27b362",
        },
        {
            "reference": "repo:data/processed/rio/train/0112_01.npy",
            "sha256": "213b69bffc52948f7934f0a21c3e6c48be7180ff68487cb79bf5b64ffd75cd78",
        },
    ],
    "instance_ground_truth": [
        {
            "reference": ("repo:data/processed/rio/instance_gt/train/scene0112_00.txt"),
            "sha256": "f5f9df1505fc72d23ca6c70b5180e6dffe6588a72d474feae9f072673b37f7a3",
        },
        {
            "reference": ("repo:data/processed/rio/instance_gt/train/scene0112_01.txt"),
            "sha256": "1aa55e3012f1cfe6e19def7caa0af67383d8311710c98465b3f47b5feb921b5a",
        },
    ],
    "change_ground_truth": {
        "reference": (
            "repo:data/processed/rio/change_gt/train/" "scene0112_00-scene0112_01.txt"
        ),
        "sha256": "75baf0a2d41956bd7d8c27b2a4257f5b8e606ab7a43a3436cc1bce07cbe0003c",
    },
    "sequence_database": {
        "reference": "repo:data/processed/rio/sequence_database_sliding_2.yaml",
        "sha256": "974299916db02d2ee0233564eb7b36e314dada93e997584d9dfef21ad70d0416",
    },
    "split_database": {
        "reference": "repo:data/processed/rio/train_database.yaml",
        "sha256": "9bb033e9e1e81c5f89e6e9f170392f95e930c9362ac83ac04c549a8b9a9c0ab0",
    },
    "semantic_label_database": {
        "reference": "repo:data/processed/rio/label_database.yaml",
        "sha256": "b03b15ecd0791a0ecd05912e9fe5617dd29a466d117cf5c2188f28638293a063",
    },
    "change_label_database": {
        "reference": "repo:data/processed/rio/change_label_database.yaml",
        "sha256": "593e8c5d18883b9bd97a926e786b000425320b6d746f055d1aab5e54439814f1",
    },
    "color_statistics": {
        "reference": "repo:data/processed/rio/color_mean_std.yaml",
        "sha256": "c1a388cf2fbfd60703ec06bc81864e103895120c37a477cc6ad51f393dd7eb78",
    },
    "train_augmentations": {
        "image": {
            "reference": "repo:conf/augmentation/albumentations_aug.yaml",
            "sha256": "0ee794f06cfa1552423ea6e51cec64d905e32dd27e63cc48defba9b17753d587",
        },
        "volume": {
            "reference": "repo:conf/augmentation/volumentations_aug.yaml",
            "sha256": "1ff353485c25fb37cf89776a60921c40d257bf627d9f89f8665ef49cc1e85cd4",
        },
    },
    "resolved_composed_config": {
        "format": "canonical-json-sort-keys-v1",
        "portable_references": True,
        "serialized_bytes": 9307,
        "sha256": "0f9e61ada901ba416ea66022bed3be90f6a5f43316f2c6983d1f4c38e0086a3a",
    },
}


def _source_tree_fixture(
    commit: str,
    *,
    dirty_paths: list[str] | None = None,
    disallowed_dirty_paths: list[str] | None = None,
) -> dict[str, object]:
    dirty = list(dirty_paths or [])
    disallowed = list(disallowed_dirty_paths or [])
    errors = ["non_artifact_worktree_changes"] if disallowed else []
    return {
        "schema_version": 1,
        "status": "fail" if errors else "pass",
        "source_commit": commit,
        "observed_head": commit,
        "allowed_dirty_prefixes": ["artifacts/P2/"],
        "committed_paths_since_source": [],
        "dirty_paths": dirty,
        "index_flag_paths": [],
        "expected_tracked_tree_sha256": "a" * 64,
        "observed_tracked_tree_sha256": "a" * 64,
        "disallowed_committed_paths": [],
        "disallowed_dirty_paths": disallowed,
        "errors": errors,
    }


def _runtime_source_fixture(
    marker: str = "stable",
    *,
    status: str = "pass",
) -> dict[str, object]:
    errors = [] if status == "pass" else ["concerto:commit_mismatch"]
    return {
        "schema_version": 1,
        "status": status,
        "repositories": {
            "fixture": {
                "marker": marker,
                "status": "pass",
                "errors": [],
                "index_flag_paths": [],
                "expected_tracked_tree_sha256": "b" * 64,
                "observed_tracked_tree_sha256": "b" * 64,
            }
        },
        "errors": errors,
    }


def _runtime_environment_fixture(
    marker: str = "stable",
    *,
    status: str = "pass",
) -> dict[str, object]:
    errors = [] if status == "pass" else ["runtime_versions_mismatch"]
    components = {}
    for name in RUNTIME_ENVIRONMENT_COMPONENTS:
        manifest_field = (
            "python_source_manifest"
            if name in RUNTIME_ENVIRONMENT_PYTHON_SOURCE_COMPONENTS
            else "native_manifest"
        )
        components[name] = {
            "status": "pass",
            "origin_refs": [f"env:{name}"],
            manifest_field: {
                "file_count": 1,
                "total_bytes": 1,
                "content_sha256": "d" * 64,
            },
            "errors": [],
        }
    contract = {
        "schema_version": (p2_preflight.P2_RUNTIME_ENVIRONMENT_CONTRACT_SCHEMA_VERSION),
        "status": status,
        "versions": dict(p2_preflight.P2_RUNTIME_ENVIRONMENT_VERSIONS),
        "components": components,
        "optional_modules": {
            "pointops": {"required": False, "status": "absent"},
        },
        "fixture_marker": marker,
        "errors": errors,
    }
    if status == "pass":
        smoke._require_passing_runtime_environment_contract(contract)
    return contract


def _patch_runtime_environment_contract(
    monkeypatch: pytest.MonkeyPatch,
    contracts: list[dict[str, object]] | None = None,
) -> None:
    observed = list(
        contracts or [_runtime_environment_fixture(), _runtime_environment_fixture()]
    )
    monkeypatch.setattr(
        smoke,
        "_build_runtime_environment_contract",
        lambda: observed.pop(0),
        raising=False,
    )


def _patch_runtime_source_contract(
    monkeypatch: pytest.MonkeyPatch,
    contracts: list[dict[str, object]] | None = None,
) -> None:
    observed = list(contracts or [_runtime_source_fixture(), _runtime_source_fixture()])
    monkeypatch.setattr(
        smoke,
        "_build_runtime_source_contract",
        lambda: observed.pop(0),
    )
    _patch_runtime_environment_contract(monkeypatch)


def _git_nul_paths(*args: str) -> list[str]:
    output = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    ).stdout
    return sorted(os.fsdecode(path) for path in output.split(b"\0") if path)


def _assert_artifact_source_tree_contract(payload: dict[str, object]) -> None:
    contract = payload["source_tree_contract"]
    source_commit = payload["source_commit"]
    assert contract["schema_version"] == 1
    assert contract["status"] == "pass"
    assert contract["source_commit"] == source_commit
    assert contract["observed_head"] == source_commit
    assert contract["generation_head_unchanged"] is True
    assert contract["allowed_dirty_prefixes"] == ["artifacts/P2/"]
    assert contract["committed_paths_since_source"] == []
    assert contract["disallowed_committed_paths"] == []
    assert contract["disallowed_dirty_paths"] == []
    assert contract["index_flag_paths"] == []
    assert len(contract["expected_tracked_tree_sha256"]) == 64
    assert set(contract["expected_tracked_tree_sha256"]) <= set("0123456789abcdef")
    assert (
        contract["observed_tracked_tree_sha256"]
        == contract["expected_tracked_tree_sha256"]
    )
    assert contract["errors"] == []
    assert all(
        path.startswith("artifacts/P2/")
        for path in contract["dirty_paths_before_generation"]
    )
    assert all(path.startswith("artifacts/P2/") for path in contract["dirty_paths"])

    current_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    current_contract = smoke._build_source_tree_contract(source_commit)
    smoke._require_passing_source_tree_contract(current_contract)
    assert current_contract["observed_head"] == current_head
    assert current_contract["index_flag_paths"] == []
    assert (
        current_contract["expected_tracked_tree_sha256"]
        == contract["expected_tracked_tree_sha256"]
    )
    assert (
        current_contract["observed_tracked_tree_sha256"]
        == contract["observed_tracked_tree_sha256"]
    )
    assert (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", source_commit, current_head],
            cwd=REPO_ROOT,
            check=False,
        ).returncode
        == 0
    )
    assert all(
        path.startswith("artifacts/P2/")
        for path in _git_nul_paths(
            "diff",
            "--name-only",
            "--no-renames",
            "-z",
            f"{source_commit}..{current_head}",
            "--",
        )
    )
    current_dirty_paths = sorted(
        set(_git_nul_paths("diff", "--name-only", "--no-renames", "-z", "--"))
        | set(
            _git_nul_paths(
                "diff",
                "--cached",
                "--name-only",
                "--no-renames",
                "-z",
                "--",
            )
        )
        | set(_git_nul_paths("ls-files", "--others", "--exclude-standard", "-z", "--"))
    )
    assert all(path.startswith("artifacts/P2/") for path in current_dirty_paths)


def _assert_artifact_runtime_source_contract(payload: dict[str, object]) -> None:
    current = smoke._build_runtime_source_contract()
    assert current["status"] == "pass"
    assert current["errors"] == []
    for record in current["repositories"].values():
        assert record["index_flag_paths"] == []
        assert len(record["expected_tracked_tree_sha256"]) == 64
        assert set(record["expected_tracked_tree_sha256"]) <= set("0123456789abcdef")
        assert (
            record["observed_tracked_tree_sha256"]
            == record["expected_tracked_tree_sha256"]
        )
    assert payload["runtime_source_contract"] == current


def _assert_artifact_runtime_environment_contract(
    payload: dict[str, object],
) -> None:
    current = smoke._build_runtime_environment_contract()
    validation_errors = []
    observed = p2_preflight._validate_runtime_environment_contract(
        {"runtime_environment_contract": current},
        validation_errors,
    )
    assert observed is current
    assert validation_errors == []
    assert payload["runtime_environment_contract"] == current


def test_source_tree_contract_rejects_hidden_index_flags() -> None:
    contract = _source_tree_fixture("a" * 40)
    contract["index_flag_paths"] = ["scripts/run_p2_native_smoke.py"]

    with pytest.raises(RuntimeError, match="source tree contract"):
        smoke._require_passing_source_tree_contract(contract)


def test_runtime_source_contract_rejects_tracked_tree_mismatch() -> None:
    contract = _runtime_source_fixture()
    contract["repositories"]["fixture"]["observed_tracked_tree_sha256"] = "c" * 64

    with pytest.raises(RuntimeError, match="runtime source contract"):
        smoke._require_passing_runtime_source_contract(contract)


def test_checkpoint_provenance_is_portable_and_pinned() -> None:
    provenance = smoke.checkpoint_provenance(
        Path("/" + "home" + "/fixture/.cache/persist4d/concerto/concerto_base.pth"),
        sha256=smoke.CONCERTO_CHECKPOINT_SHA256,
    )

    assert provenance == {
        "reference": "local_cache:persist4d/concerto/concerto_base.pth",
        "sha256": "845ec7dec97a5fabff8fadb5d9858ac6734347b612d1a4b574213419c139de07",
    }
    assert "/" + "home" + "/" not in json.dumps(provenance)


def test_source_tree_contract_allows_only_tracked_artifact_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_runtime_source_contract(monkeypatch)
    commit = "a" * 40
    observed_contracts = [
        _source_tree_fixture(
            commit,
            dirty_paths=["artifacts/P2/scannet_preflight.json"],
        ),
        _source_tree_fixture(
            commit,
            dirty_paths=[
                "artifacts/P2/native_smoke_report.json",
                "artifacts/P2/scannet_preflight.json",
            ],
        ),
    ]
    monkeypatch.setattr(smoke, "_git_commit", lambda: commit)
    monkeypatch.setattr(
        smoke,
        "_build_source_tree_contract",
        lambda source_commit=None: observed_contracts.pop(0),
    )

    guard = smoke._begin_source_tree_contract()
    contract = smoke._finalize_source_tree_contract(guard)

    assert contract == {
        "schema_version": 1,
        "status": "pass",
        "source_commit": commit,
        "observed_head": commit,
        "allowed_dirty_prefixes": ["artifacts/P2/"],
        "committed_paths_since_source": [],
        "dirty_paths": [
            "artifacts/P2/native_smoke_report.json",
            "artifacts/P2/scannet_preflight.json",
        ],
        "index_flag_paths": [],
        "expected_tracked_tree_sha256": "a" * 64,
        "observed_tracked_tree_sha256": "a" * 64,
        "disallowed_committed_paths": [],
        "disallowed_dirty_paths": [],
        "errors": [],
        "generation_head_unchanged": True,
        "dirty_paths_before_generation": ["artifacts/P2/scannet_preflight.json"],
    }


def test_source_tree_contract_rejects_tracked_source_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_runtime_source_contract(monkeypatch, [_runtime_source_fixture()])
    commit = "b" * 40
    monkeypatch.setattr(
        smoke,
        "_build_source_tree_contract",
        lambda source_commit=None: _source_tree_fixture(
            commit,
            dirty_paths=[
                "artifacts/P2/scannet_preflight.json",
                "scripts/run_p2_native_smoke.py",
            ],
            disallowed_dirty_paths=["scripts/run_p2_native_smoke.py"],
        ),
    )

    with pytest.raises(RuntimeError, match="source tree contract is not pass"):
        smoke._begin_source_tree_contract()


def test_source_tree_contract_rejects_untracked_source_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_runtime_source_contract(monkeypatch, [_runtime_source_fixture()])
    commit = "e" * 40
    monkeypatch.setattr(
        smoke,
        "_build_source_tree_contract",
        lambda source_commit=None: _source_tree_fixture(
            commit,
            dirty_paths=["scratch_runtime_override.py"],
            disallowed_dirty_paths=["scratch_runtime_override.py"],
        ),
    )

    with pytest.raises(RuntimeError, match="scratch_runtime_override.py"):
        smoke._begin_source_tree_contract()


def test_source_tree_contract_rejects_head_change_during_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_runtime_source_contract(monkeypatch)
    commit = "c" * 40
    commits = iter([commit, "d" * 40])
    monkeypatch.setattr(smoke, "_git_commit", lambda: next(commits))
    monkeypatch.setattr(
        smoke,
        "_build_source_tree_contract",
        lambda source_commit=None: _source_tree_fixture(commit),
    )

    guard = smoke._begin_source_tree_contract()

    with pytest.raises(RuntimeError, match="HEAD changed during generation"):
        smoke._finalize_source_tree_contract(guard)


def test_source_tree_contract_rejects_source_change_after_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_runtime_source_contract(monkeypatch)
    commit = "f" * 40
    observed_contracts = [
        _source_tree_fixture(commit),
        _source_tree_fixture(
            commit,
            dirty_paths=["conf/config_p2_rescene4d_concerto_t2.yaml"],
            disallowed_dirty_paths=["conf/config_p2_rescene4d_concerto_t2.yaml"],
        ),
    ]
    monkeypatch.setattr(smoke, "_git_commit", lambda: commit)
    monkeypatch.setattr(
        smoke,
        "_build_source_tree_contract",
        lambda source_commit=None: observed_contracts.pop(0),
    )

    guard = smoke._begin_source_tree_contract()

    with pytest.raises(RuntimeError, match="config_p2_rescene4d_concerto_t2"):
        smoke._finalize_source_tree_contract(guard)


def test_runtime_source_contract_rejects_failed_generation_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commit = "1" * 40
    monkeypatch.setattr(smoke, "_git_commit", lambda: commit)
    monkeypatch.setattr(
        smoke,
        "_build_source_tree_contract",
        lambda source_commit=None: _source_tree_fixture(commit),
    )
    _patch_runtime_source_contract(
        monkeypatch,
        [_runtime_source_fixture(status="fail")],
    )

    with pytest.raises(RuntimeError, match="runtime source contract is not pass"):
        smoke._begin_source_tree_contract()


def test_runtime_source_contract_rejects_change_during_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commit = "2" * 40
    monkeypatch.setattr(smoke, "_git_commit", lambda: commit)
    monkeypatch.setattr(
        smoke,
        "_build_source_tree_contract",
        lambda source_commit=None: _source_tree_fixture(commit),
    )
    _patch_runtime_source_contract(
        monkeypatch,
        [
            _runtime_source_fixture("before"),
            _runtime_source_fixture("after"),
        ],
    )

    guard = smoke._begin_source_tree_contract()

    with pytest.raises(RuntimeError, match="runtime source changed"):
        smoke._finalize_source_tree_contract(guard)


def test_runtime_source_contract_matches_current_runtime_exactly() -> None:
    first = smoke._build_runtime_source_contract()
    second = smoke._build_runtime_source_contract()

    assert first == second
    assert first["schema_version"] == 1
    assert first["status"] == "pass"
    assert first["errors"] == []
    assert set(first["repositories"]) == {
        "concerto",
        "detectron2",
        "sonata",
        "stmetrics",
    }
    for record in first["repositories"].values():
        assert record["status"] == "pass"
        assert record["errors"] == []
        assert record["dirty_paths"] == []
        assert record["index_flag_paths"] == []
        assert len(record["expected_tracked_tree_sha256"]) == 64
        assert set(record["expected_tracked_tree_sha256"]) <= set("0123456789abcdef")
        assert (
            record["observed_tracked_tree_sha256"]
            == record["expected_tracked_tree_sha256"]
        )
        assert record["observed_commit"] == record["expected_commit"]
        assert record["module_origin_ref"].startswith(record["reference"] + "/")
        for native in record["native_extensions"].values():
            assert native["status"] == "pass"
            assert native["observed_byte_size"] == native["expected_byte_size"]
            assert native["observed_sha256"] == native["expected_sha256"]
            assert native["origin_ref"].startswith(record["reference"] + "/")


def test_runtime_environment_contract_rejects_failed_generation_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commit = "3" * 40
    monkeypatch.setattr(smoke, "_git_commit", lambda: commit)
    monkeypatch.setattr(
        smoke,
        "_build_source_tree_contract",
        lambda source_commit=None: _source_tree_fixture(commit),
    )
    _patch_runtime_source_contract(monkeypatch, [_runtime_source_fixture()])
    _patch_runtime_environment_contract(
        monkeypatch,
        [_runtime_environment_fixture(status="fail")],
    )

    with pytest.raises(RuntimeError, match="runtime environment contract"):
        smoke._begin_source_tree_contract()


def test_runtime_environment_contract_rejects_change_during_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commit = "4" * 40
    monkeypatch.setattr(smoke, "_git_commit", lambda: commit)
    monkeypatch.setattr(
        smoke,
        "_build_source_tree_contract",
        lambda source_commit=None: _source_tree_fixture(commit),
    )
    _patch_runtime_source_contract(monkeypatch)
    _patch_runtime_environment_contract(
        monkeypatch,
        [
            _runtime_environment_fixture("before"),
            _runtime_environment_fixture("after"),
        ],
    )

    guard = smoke._begin_source_tree_contract()

    with pytest.raises(RuntimeError, match="runtime environment changed"):
        smoke._finalize_source_tree_contract(guard)


def test_runtime_environment_contract_matches_current_runtime_exactly() -> None:
    first = smoke._build_runtime_environment_contract()
    second = smoke._build_runtime_environment_contract()

    assert first == second
    assert first["schema_version"] == (
        p2_preflight.P2_RUNTIME_ENVIRONMENT_CONTRACT_SCHEMA_VERSION
    )
    assert first["status"] == "pass"
    assert first["errors"] == []
    assert first["versions"] == p2_preflight.P2_RUNTIME_ENVIRONMENT_VERSIONS
    assert set(first["components"]) == RUNTIME_ENVIRONMENT_COMPONENTS
    for name, record in first["components"].items():
        assert record["status"] == "pass"
        assert record["errors"] == []
        assert record["origin_refs"]
        manifest_field = (
            "python_source_manifest"
            if name in RUNTIME_ENVIRONMENT_PYTHON_SOURCE_COMPONENTS
            else "native_manifest"
        )
        manifest = record[manifest_field]
        assert manifest["file_count"] >= 1
        assert manifest["total_bytes"] >= 1
        assert len(manifest["content_sha256"]) == 64
        assert set(manifest["content_sha256"]) <= set("0123456789abcdef")


def test_portable_config_references_fail_closed_outside_repo() -> None:
    portable = smoke._portable_config_value(
        {
            "checkpoint": str(smoke.DEFAULT_CHECKPOINT),
            "dataset": str(REPO_ROOT / "data" / "processed" / "rio"),
            "relative": "artifacts/P2/scannet_preflight.json",
        },
        smoke.DEFAULT_CHECKPOINT,
    )

    assert portable == {
        "checkpoint": smoke.CONCERTO_CHECKPOINT_REFERENCE,
        "dataset": "repo:data/processed/rio",
        "relative": "artifacts/P2/scannet_preflight.json",
    }
    with pytest.raises(ValueError, match="non-portable absolute path"):
        smoke._portable_config_value(
            {"external": "/tmp/external-config-input"},
            smoke.DEFAULT_CHECKPOINT,
        )


def test_real_3rscan_input_provenance_is_portable_and_pinned() -> None:
    import hydra

    config = smoke._compose_config(smoke.DEFAULT_CHECKPOINT)
    assert config.general.p2_fail_closed_runtime is False
    assert config.general.p2_weighted_objective is True
    assert config.data.train_dataset.fail_closed is True
    assert "epoch_sample_multiple" not in config.data.train_dataset
    assert "sampler_seed" not in config.data.train_dataset
    assert all(
        config.data[split].known_empty_scan_policy == "official_substitute"
        for split in ("train_dataset", "validation_dataset", "test_dataset")
    )
    dataset = hydra.utils.instantiate(config.data.train_dataset)
    dataset_index = dataset.sequence_names.index(smoke.TINY_SAMPLE_NAME)

    provenance = smoke._input_provenance(
        config,
        dataset,
        dataset_index,
        smoke.DEFAULT_CHECKPOINT,
    )

    assert provenance == EXPECTED_INPUT_PROVENANCE
    assert "/" + "home" + "/" not in json.dumps(provenance, sort_keys=True)


def test_lightning_resume_runtime_uses_single_rio_weighted_sampler() -> None:
    import hydra
    from torch.utils.data import WeightedRandomSampler

    config = smoke._compose_config(smoke.DEFAULT_CHECKPOINT)
    smoke._configure_lightning_weighted_sampler(config)
    dataset = hydra.utils.instantiate(config.data.train_dataset)

    assert config.general.p2_fail_closed_runtime is True
    assert type(dataset).__module__ == "datasets.multi_dataset"
    assert type(dataset).__name__ == "MultiDataset"
    assert len(dataset.datasets) == 1
    assert dataset.datasets[0].dataset_name == "rio"
    assert isinstance(dataset.sampler, WeightedRandomSampler)
    assert dataset.sampler.generator is not None
    assert dataset.sampler_seed == smoke.SEED


def test_lightning_resume_uses_all_formal_checkpoint_callbacks(
    tmp_path: Path,
) -> None:
    from omegaconf import open_dict

    config = smoke._compose_config(smoke.DEFAULT_CHECKPOINT)
    with open_dict(config):
        config.general.save_dir = str(tmp_path)

    callbacks = smoke._instantiate_formal_checkpoint_callbacks(config)
    state_keys = [callback.state_key for callback in callbacks]

    assert len(callbacks) == 3
    assert len(set(state_keys)) == 3
    assert all(
        type(callback).__module__ == "pytorch_lightning.callbacks.model_checkpoint"
        and type(callback).__name__ == "ModelCheckpoint"
        for callback in callbacks
    )
    assert [callback.monitor for callback in callbacks] == [
        "val_mean_t-AP",
        None,
        None,
    ]
    assert all(callback.save_weights_only is False for callback in callbacks)
    assert sum(bool(callback.save_last) for callback in callbacks) == 1
    assert all(
        set(callback.state_dict())
        == {
            "monitor",
            "best_model_score",
            "best_model_path",
            "current_score",
            "dirpath",
            "best_k_models",
            "kth_best_model_path",
            "kth_value",
            "last_model_path",
        }
        for callback in callbacks
    )


def test_lightning_resume_wiring_requests_full_state_ckpt_path() -> None:
    source = textwrap.dedent(inspect.getsource(smoke._run_lightning_checkpoint_resume))
    tree = ast.parse(source)
    fit_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "fit"
    ]
    resume_calls = [
        call
        for call in fit_calls
        if any(keyword.arg == "ckpt_path" for keyword in call.keywords)
    ]

    assert len(resume_calls) == 1
    keywords = {keyword.arg: keyword.value for keyword in resume_calls[0].keywords}
    assert isinstance(keywords["ckpt_path"], ast.Name)
    assert keywords["ckpt_path"].id == "verified_resume_checkpoint"
    assert isinstance(keywords["weights_only"], ast.Constant)
    assert keywords["weights_only"].value is False
    selector_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "find_resume_checkpoint"
    ]
    assert len(selector_calls) == 1
    selector_keywords = {
        keyword.arg: keyword.value for keyword in selector_calls[0].keywords
    }
    assert isinstance(selector_keywords["formal_p2"], ast.Constant)
    assert selector_keywords["formal_p2"].value is True
    assert isinstance(selector_keywords["cfg"], ast.Name)
    assert selector_keywords["cfg"].id == "source_config"
    verified_snapshot_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "require_p2_resume_checkpoint"
    ]
    assert len(verified_snapshot_calls) == 1
    assert "_P2_FORMAL_OPTIMIZER_STEPS_PER_EPOCH" in source
    assert "_P2_FORMAL_TRAINABLE_PARAMETER_SCHEMA_SHA256" in source
    assert "trainable_parameters" in source
    assert "trainable_parameter_schema_sha256" in source
    assert "def on_train_batch_start" in source
    assert "val_dataloaders" in source
    assert "def setup_with_pinned_dataset" in source
    assert "def on_load_checkpoint_with_identity_check" in source
    assert "find_resume_checkpoint" in source
    assert "WeightedRandomSampler" in source
    assert "_TRAIN_SAMPLER_CHECKPOINT_KEY" in source
    assert '"sampler_state_restored"' in source
    assert '"sampler_state_advanced"' in source
    assert '"sampler_stream_continuous"' in source
    assert "state_dict().items()" in source
    assert "ModelCheckpoint" in source
    assert "TemporaryDirectory" in source
    assert 'details["temporary_checkpoint_removed"]' in source
    assert 'details["verified_resume_snapshot_removed"]' in source


def test_main_resume_fit_explicitly_loads_trusted_full_checkpoint() -> None:
    source = (REPO_ROOT / "main_instance_segmentation.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    train_functions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "train"
    ]
    assert len(train_functions) == 1
    fit_calls = [
        node
        for node in ast.walk(train_functions[0])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "fit"
    ]
    assert len(fit_calls) == 1
    assert len(fit_calls[0].keywords) == 1
    fit_kwargs_expansion = fit_calls[0].keywords[0]
    assert fit_kwargs_expansion.arg is None
    assert isinstance(fit_kwargs_expansion.value, ast.Name)
    assert fit_kwargs_expansion.value.id == "fit_kwargs"
    fit_kwargs_initializers = [
        node
        for node in ast.walk(train_functions[0])
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "fit_kwargs"
            for target in node.targets
        )
        and isinstance(node.value, ast.Dict)
    ]
    assert len(fit_kwargs_initializers) == 1
    initializer = fit_kwargs_initializers[0].value
    assert len(initializer.keys) == len(initializer.values) == 1
    assert isinstance(initializer.keys[0], ast.Constant)
    assert initializer.keys[0].value == "ckpt_path"
    assert isinstance(initializer.values[0], ast.Name)
    assert initializer.values[0].id == "ckpt_path"

    guarded_weights_only_assignments = []
    for node in ast.walk(train_functions[0]):
        if not isinstance(node, ast.If):
            continue
        if not (
            isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "ckpt_path"
            and len(node.test.ops) == 1
            and isinstance(node.test.ops[0], ast.IsNot)
            and len(node.test.comparators) == 1
            and isinstance(node.test.comparators[0], ast.Constant)
            and node.test.comparators[0].value is None
        ):
            continue
        guarded_weights_only_assignments.extend(
            child
            for child in node.body
            if isinstance(child, ast.Assign)
            and isinstance(child.value, ast.Constant)
            and child.value.value is False
            and any(
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and target.value.id == "fit_kwargs"
                and isinstance(target.slice, ast.Constant)
                and target.slice.value == "weights_only"
                for target in child.targets
            )
        )
    assert len(guarded_weights_only_assignments) == 1


def test_gpu_artifact_cli_guards_source_tree_before_and_after_generation() -> None:
    source = textwrap.dedent(inspect.getsource(smoke.main))
    tree = ast.parse(source)
    begin_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_begin_source_tree_contract"
    ]
    finalize_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_finalize_source_tree_contract"
    ]
    guarded_runs = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"run_native_smoke", "run_tiny_overfit"}
    ]

    assert len(begin_calls) == 1
    assert len(finalize_calls) == 1
    assert any(
        isinstance(node, ast.Try)
        and any(
            finalize in list(ast.walk(final_node))
            for final_node in node.finalbody
            for finalize in finalize_calls
        )
        for node in ast.walk(tree)
    )
    assert {call.func.id for call in guarded_runs} == {
        "run_native_smoke",
        "run_tiny_overfit",
    }
    assert all(
        any(
            keyword.arg == "source_tree_guard"
            and isinstance(keyword.value, ast.Name)
            and keyword.value.id == "source_tree_guard"
            for keyword in call.keywords
        )
        for call in guarded_runs
    )
    assert '"runtime_source_contract"' in inspect.getsource(smoke.run_native_smoke)
    assert '"runtime_source_contract"' in inspect.getsource(smoke.run_tiny_overfit)
    assert '"runtime_environment_contract"' in inspect.getsource(smoke.run_native_smoke)
    assert '"runtime_environment_contract"' in inspect.getsource(smoke.run_tiny_overfit)


def test_lightning_completed_epoch_boundary_restores_before_next_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hydra
    import pytorch_lightning as pl
    from omegaconf import open_dict
    from pytorch_lightning.callbacks import Callback
    from torch.utils.data import (
        DataLoader,
        Dataset,
        TensorDataset,
        WeightedRandomSampler,
    )

    import main_instance_segmentation as training_entrypoint
    from main_instance_segmentation import (
        _P2_FORMAL_ONECYCLE_TOTAL_STEPS,
        _P2_FORMAL_TRAIN_BATCHES_PER_EPOCH,
        find_resume_checkpoint,
        require_p2_resume_checkpoint,
    )

    events: list[str] = []

    class TinySystem(pl.LightningModule):
        def __init__(
            self,
            sampler_generator: torch.Generator,
            observed_events: list[str] | None = None,
        ) -> None:
            super().__init__()
            self.save_hyperparameters(config)
            self.layer = torch.nn.Linear(2, 1)
            self.sampler_generator = sampler_generator
            self.observed_events = observed_events

        def setup(self, stage: str | None = None) -> None:
            if self.observed_events is not None and stage == "fit":
                self.observed_events.append("setup")

        def on_load_checkpoint(self, checkpoint: dict[str, object]) -> None:
            payload = checkpoint["p2_train_sampler_generator"]
            self.sampler_generator.set_state(payload["generator_state"])
            if self.observed_events is not None:
                self.observed_events.append("on_load_checkpoint")

        def on_save_checkpoint(self, checkpoint: dict[str, object]) -> None:
            checkpoint["p2_train_sampler_generator"] = {
                "schema_version": 1,
                "resume_scope": "completed_epoch_boundary_only",
                "mid_epoch_resume_supported": False,
                "dataloader_prefetch_state_checkpointed": False,
                "generator_state": self.sampler_generator.get_state().clone(),
            }
            optimizer_state = checkpoint["optimizer_states"][0]
            live_optimizer = self.trainer.optimizers[0]
            model_state = {
                name: {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                }
                for name, value in checkpoint["state_dict"].items()
            }
            model_state_entries = [
                [name, metadata["shape"], metadata["dtype"]]
                for name, metadata in sorted(model_state.items())
            ]
            model_state_payload = json.dumps(
                model_state_entries,
                ensure_ascii=True,
                separators=(",", ":"),
            ).encode("ascii")
            trainable_parameters = [
                [
                    name,
                    list(parameter.shape),
                    str(parameter.dtype),
                ]
                for name, parameter in self.named_parameters()
                if parameter.requires_grad
            ]
            trainable_parameter_payload = json.dumps(
                trainable_parameters,
                ensure_ascii=True,
                separators=(",", ":"),
            ).encode("ascii")
            names_by_identity = {
                id(parameter): name for name, parameter in self.named_parameters()
            }
            parameters = {}
            for saved_group, live_group in zip(
                optimizer_state["param_groups"],
                live_optimizer.param_groups,
                strict=True,
            ):
                for parameter_id, parameter in zip(
                    saved_group["params"],
                    live_group["params"],
                    strict=True,
                ):
                    name = names_by_identity[id(parameter)]
                    saved_parameter = checkpoint["state_dict"][name]
                    parameters[parameter_id] = {
                        "name": name,
                        "shape": list(saved_parameter.shape),
                        "dtype": str(saved_parameter.dtype),
                    }
            checkpoint["p2_optimizer_parameter_contract"] = {
                "schema_version": 1,
                "state_dict": model_state,
                "state_dict_schema_sha256": hashlib.sha256(
                    model_state_payload
                ).hexdigest(),
                "param_groups": [
                    list(group["params"]) for group in optimizer_state["param_groups"]
                ],
                "parameters": parameters,
                "trainable_parameters": trainable_parameters,
                "trainable_parameter_schema_sha256": hashlib.sha256(
                    trainable_parameter_payload
                ).hexdigest(),
            }

        def training_step(self, batch: object, batch_idx: int) -> torch.Tensor:
            features, labels = batch
            return torch.nn.functional.mse_loss(self.layer(features), labels)

        def validation_step(self, batch: object, batch_idx: int) -> None:
            self.log(
                "val_mean_t-AP",
                torch.tensor(0.5),
                on_step=False,
                on_epoch=True,
            )

        def configure_optimizers(self) -> dict[str, object]:
            optimizer = hydra.utils.instantiate(
                config.optimizer,
                params=self.parameters(),
            )
            scheduler_config = config.scheduler.scheduler.copy()
            scheduler_config.total_steps = _P2_FORMAL_ONECYCLE_TOTAL_STEPS
            scheduler = hydra.utils.instantiate(
                scheduler_config,
                optimizer=optimizer,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    **config.scheduler.pytorch_lightning_params,
                },
            }

    class RecordingDataset(Dataset):
        def __init__(self) -> None:
            self.sampled_indices: list[int] = []

        def __len__(self) -> int:
            return 4

        def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
            self.sampled_indices.append(index)
            return torch.ones(2), torch.zeros(1)

    class RestoredStateProbe(Callback):
        def __init__(self) -> None:
            self.snapshot = None

        def on_train_batch_start(
            self,
            trainer: object,
            pl_module: object,
            batch: object,
            batch_idx: int,
        ) -> None:
            if self.snapshot is not None:
                return
            events.append("on_train_batch_start")
            self.snapshot = {
                "global_step": int(trainer.global_step),
                "optimizer": smoke._cpu_snapshot(trainer.optimizers[0].state_dict()),
                "scheduler": smoke._cpu_snapshot(
                    trainer.lr_scheduler_configs[0].scheduler.state_dict()
                ),
                "checkpoint_callbacks": {
                    callback.state_key: smoke._cpu_snapshot(callback.state_dict())
                    for callback in trainer.checkpoint_callbacks
                },
            }

    def loader() -> tuple[RecordingDataset, torch.Generator, DataLoader]:
        dataset = RecordingDataset()
        generator = torch.Generator().manual_seed(smoke.SEED)
        sampler = WeightedRandomSampler(
            torch.ones(len(dataset), dtype=torch.double),
            num_samples=_P2_FORMAL_TRAIN_BATCHES_PER_EPOCH,
            replacement=True,
            generator=generator,
        )
        return (
            dataset,
            generator,
            DataLoader(
                dataset,
                batch_size=1,
                sampler=sampler,
                num_workers=0,
            ),
        )

    def trainer(
        *,
        max_epochs: int,
        max_steps: int,
        callbacks: list[Callback],
    ) -> pl.Trainer:
        return pl.Trainer(
            accelerator="cpu",
            devices=1,
            max_epochs=max_epochs,
            max_steps=max_steps,
            accumulate_grad_batches=4,
            limit_train_batches=_P2_FORMAL_TRAIN_BATCHES_PER_EPOCH,
            limit_val_batches=1,
            num_sanity_val_steps=0,
            logger=False,
            callbacks=callbacks,
            default_root_dir=tmp_path,
            enable_progress_bar=False,
            enable_model_summary=False,
            use_distributed_sampler=False,
        )

    checkpoint = tmp_path / "last.ckpt"
    config = smoke._compose_config(smoke.DEFAULT_CHECKPOINT)
    with open_dict(config):
        config.general.save_dir = str(tmp_path)
        config.scheduler.scheduler.total_steps = _P2_FORMAL_ONECYCLE_TOTAL_STEPS
    source_checkpoint_callbacks = smoke._instantiate_formal_checkpoint_callbacks(config)
    source_dataset, source_generator, source_loader = loader()
    validation_loader = DataLoader(
        TensorDataset(torch.ones(1, 2), torch.zeros(1, 1)),
        batch_size=1,
        num_workers=0,
    )
    source_trainer = trainer(
        max_epochs=1,
        max_steps=-1,
        callbacks=source_checkpoint_callbacks,
    )
    source_trainer.fit(
        TinySystem(source_generator),
        train_dataloaders=source_loader,
        val_dataloaders=validation_loader,
    )
    assert checkpoint.is_file()
    assert len(source_dataset.sampled_indices) == 264
    raw_saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    monkeypatch.setattr(
        training_entrypoint,
        "_P2_FORMAL_MODEL_STATE_SCHEMA_SHA256",
        raw_saved["p2_optimizer_parameter_contract"]["state_dict_schema_sha256"],
    )
    monkeypatch.setattr(
        training_entrypoint,
        "_P2_FORMAL_TRAINABLE_PARAMETER_SCHEMA_SHA256",
        raw_saved["p2_optimizer_parameter_contract"][
            "trainable_parameter_schema_sha256"
        ],
    )

    selected = find_resume_checkpoint(tmp_path, formal_p2=True, cfg=config)
    assert selected == str(checkpoint)
    verified = require_p2_resume_checkpoint(config, selected)
    saved = torch.load(verified, map_location="cpu", weights_only=False)
    assert saved["epoch"] == 0
    assert saved["global_step"] == 66
    assert saved["lr_schedulers"][0]["last_epoch"] == 66
    fit_loop = saved["loops"]["fit_loop"]
    assert fit_loop["epoch_progress"]["total"] == {
        "ready": 1,
        "completed": 0,
        "started": 1,
        "processed": 0,
    }
    assert (
        fit_loop["epoch_loop.automatic_optimization.optim_progress"]["optimizer"][
            "step"
        ]["total"]["completed"]
        == 66
    )
    assert fit_loop["epoch_loop.scheduler_progress"]["total"]["completed"] == 66
    assert fit_loop["epoch_loop.batch_progress"]["total"]["completed"] == 264
    assert fit_loop["epoch_loop.state_dict"]["_batches_that_stepped"] == 65
    assert fit_loop["epoch_loop.val_loop.batch_progress"]["total"]["completed"] == 1
    saved_optimizer = saved["optimizer_states"][0]
    optimizer_parameter_ids = {
        parameter_id
        for group in saved_optimizer["param_groups"]
        for parameter_id in group["params"]
    }
    assert optimizer_parameter_ids == set(saved_optimizer["state"])
    assert optimizer_parameter_ids == set(
        saved["p2_optimizer_parameter_contract"]["parameters"]
    )
    assert saved["p2_optimizer_parameter_contract"]["param_groups"] == [
        list(group["params"]) for group in saved_optimizer["param_groups"]
    ]
    assert len(optimizer_parameter_ids) == sum(
        parameter.requires_grad
        for parameter in source_trainer.lightning_module.parameters()
    )
    monitored_state = saved["callbacks"][source_checkpoint_callbacks[0].state_key]
    assert monitored_state["monitor"] == "val_mean_t-AP"
    assert monitored_state["best_model_score"] is not None
    assert monitored_state["current_score"] is not None
    assert monitored_state["best_model_path"]
    assert monitored_state["best_k_models"]
    assert monitored_state["last_model_path"] == str(checkpoint)

    expected_generator = torch.Generator()
    expected_generator.set_state(saved["p2_train_sampler_generator"]["generator_state"])
    expected_stream = iter(
        WeightedRandomSampler(
            torch.ones(4, dtype=torch.double),
            num_samples=_P2_FORMAL_TRAIN_BATCHES_PER_EPOCH,
            replacement=True,
            generator=expected_generator,
        )
    )
    expected_next_indices = [next(expected_stream) for _ in range(4)]

    probe = RestoredStateProbe()
    resumed_checkpoint_callbacks = smoke._instantiate_formal_checkpoint_callbacks(
        config
    )
    resumed_dataset, resumed_generator, resumed_loader = loader()
    resumed_trainer = trainer(
        max_epochs=2,
        max_steps=67,
        callbacks=[*resumed_checkpoint_callbacks, probe],
    )
    resumed_trainer.fit(
        TinySystem(resumed_generator, events),
        train_dataloaders=resumed_loader,
        val_dataloaders=validation_loader,
        ckpt_path=verified,
        weights_only=False,
    )

    assert events == ["setup", "on_load_checkpoint", "on_train_batch_start"]
    assert probe.snapshot is not None
    assert probe.snapshot["global_step"] == saved["global_step"] == 66
    assert smoke._recursive_equal(
        probe.snapshot["optimizer"],
        saved["optimizer_states"][0],
    )
    assert smoke._recursive_equal(
        probe.snapshot["scheduler"],
        saved["lr_schedulers"][0],
    )
    assert set(saved["callbacks"]) == {
        callback.state_key for callback in source_checkpoint_callbacks
    }
    assert len(saved["callbacks"]) == 3
    assert smoke._recursive_equal(
        smoke._persistent_checkpoint_callback_states(
            probe.snapshot["checkpoint_callbacks"]
        ),
        smoke._persistent_checkpoint_callback_states(saved["callbacks"]),
    )
    resumed_monitored_state = probe.snapshot["checkpoint_callbacks"][
        source_checkpoint_callbacks[0].state_key
    ]
    assert resumed_monitored_state["current_score"] is None
    assert monitored_state["current_score"] is not None
    assert resumed_dataset.sampled_indices == expected_next_indices
    assert resumed_trainer.global_step == 67
    assert resumed_trainer.lr_scheduler_configs[0].scheduler.last_epoch == 67


def test_objective_breakdown_weights_segmentation_and_excludes_diagnostics() -> None:
    losses = {
        "loss_ce": torch.tensor(1.0),
        "loss_mask": torch.tensor(2.0),
        "loss_dice": torch.tensor(3.0),
        "loss_ce_0": torch.tensor(4.0),
        "loss_segment_contrastive": torch.tensor(5.0),
        "loss_aux_contrastive": torch.tensor(6.0),
        "loss_segment_contrastive_layer0": torch.tensor(100.0),
        "loss_aux_contrastive_layer_0": torch.tensor(200.0),
    }
    weight_dict = {
        "loss_ce": 2.0,
        "loss_mask": 5.0,
        "loss_dice": 2.0,
        "loss_ce_0": 2.0,
    }

    breakdown = smoke.objective_breakdown(losses, weight_dict)

    assert breakdown["final_head_segmentation"].item() == 18.0
    assert breakdown["all_segmentation"].item() == 26.0
    assert breakdown["aggregate_contrastive"].item() == 11.0
    assert breakdown["objective"].item() == 37.0
    assert set(breakdown["diagnostic_keys"]) == {
        "loss_segment_contrastive_layer0",
        "loss_aux_contrastive_layer_0",
    }


def test_parameter_groups_identify_frozen_concerto_and_trainable_heads() -> None:
    named_parameters = [
        (
            "model.backbone.model.embedding.proj.weight",
            torch.nn.Parameter(torch.ones(1)),
        ),
        ("model.backbone.model.enc.block.weight", torch.nn.Parameter(torch.ones(1))),
        ("model.backbone.model.dec.block.weight", torch.nn.Parameter(torch.ones(1))),
        ("model.class_embed_head.weight", torch.nn.Parameter(torch.ones(1))),
        ("model.mask_features_head.weight", torch.nn.Parameter(torch.ones(1))),
        ("model.cross_attention.0.weight", torch.nn.Parameter(torch.ones(1))),
        ("criterion.temperature", torch.nn.Parameter(torch.ones(1))),
    ]
    for name, parameter in named_parameters:
        if ".embedding." in name or ".enc." in name:
            parameter.requires_grad_(False)

    groups = smoke.classify_parameters(named_parameters)

    assert groups["frozen_encoder"] == [
        "model.backbone.model.embedding.proj.weight",
        "model.backbone.model.enc.block.weight",
    ]
    assert groups["trainable_concerto_decoder"] == [
        "model.backbone.model.dec.block.weight"
    ]
    assert groups["trainable_rescene_heads"] == [
        "model.class_embed_head.weight",
        "model.mask_features_head.weight",
    ]
    assert groups["trainable_rescene_decoder"] == ["model.cross_attention.0.weight"]
    assert groups["trainable_objective"] == ["criterion.temperature"]


def test_parameter_groups_do_not_hide_trainable_encoder_parameters() -> None:
    encoder = torch.nn.Parameter(torch.ones(1), requires_grad=True)

    groups = smoke.classify_parameters(
        [("model.backbone.model.enc.block.weight", encoder)]
    )

    assert groups["frozen_encoder"] == ["model.backbone.model.enc.block.weight"]
    assert groups["trainable_rescene_decoder"] == []


def test_required_tmap_schema_accepts_real_metric_keys() -> None:
    keys = {
        "val_mean_t-AP",
        "val_mean_t-AP_50",
        "val_mean_t-AP_25",
        "val_mean_AP",
        "val_mean_stage1-AP",
        "val_mean_stage2-AP",
    }

    assert smoke.validate_tmap_schema(keys) == sorted(keys)


def test_required_tmap_schema_rejects_missing_head() -> None:
    with pytest.raises(ValueError, match="val_mean_stage2-AP"):
        smoke.validate_tmap_schema(
            {
                "val_mean_t-AP",
                "val_mean_t-AP_50",
                "val_mean_t-AP_25",
                "val_mean_AP",
                "val_mean_stage1-AP",
            }
        )


def test_matcher_classification_accuracy_excludes_no_object_logit() -> None:
    class Matcher:
        def __call__(self, outputs, targets, mask_type):
            return [(torch.tensor([0]), torch.tensor([0]))]

    system = SimpleNamespace(
        criterion=SimpleNamespace(matcher=Matcher()),
        mask_type="segment_mask",
    )
    output = {
        "pred_logits": torch.tensor([[[0.1, 2.0, 4.0]]]),
        "pred_masks": [torch.tensor([[8.0]])],
        "aux_outputs": [],
    }
    targets = [
        {
            "labels": torch.tensor([1]),
            "segment_mask": torch.tensor([[True]]),
        }
    ]

    quality = smoke._matching_quality(system, output, targets)

    assert quality["classification_accuracy"] == 1.0
    assert quality["mean_dice"] > 0.99


@pytest.mark.gpu
@pytest.mark.skipif(
    os.environ.get("P2_VERIFY_GPU_ARTIFACTS") != "1",
    reason="set P2_VERIFY_GPU_ARTIFACTS=1 after the real single-A40 run",
)
def test_real_native_smoke_artifact_passes_all_gates() -> None:
    payload = json.loads(REPORT.read_text(encoding="utf-8"))

    _assert_artifact_source_tree_contract(payload)
    _assert_artifact_runtime_source_contract(payload)
    _assert_artifact_runtime_environment_contract(payload)
    assert payload["scope"] == "preflight-only"
    assert payload["verification_mode"] == "artifact_contract_not_reexecution"
    assert payload["official_mixed_data_reproduction"] is False
    assert payload["g2_evidence"] is False
    assert payload["hardware"]["device_alias"] == "device-0"
    assert "uuid" not in payload["hardware"]
    assert payload["smoke"]["passed"] is True
    assert payload["smoke"]["first_step_model_bitwise_changed"] is True
    assert payload["smoke"]["encoder_bitwise_unchanged"] is True
    assert payload["smoke"]["decoder_head_changed"] is True
    assert payload["smoke"]["segment_contrastive_positive"] is True
    assert payload["checkpoint_roundtrip"]["passed"] is True
    assert payload["checkpoint_roundtrip"]["kind"] == (
        "native_model_optimizer_scheduler_state_roundtrip"
    )
    assert payload["checkpoint_roundtrip"]["lightning_full_resume_validation"] is False
    assert payload["checkpoint_roundtrip"]["advanced_model_bitwise_changed"] is True
    assert payload["checkpoint_roundtrip"]["advanced_optimizer_state_changed"] is True
    lightning_resume = payload["lightning_checkpoint_resume"]
    assert lightning_resume["passed"] is True
    assert lightning_resume["lightning_full_resume_validation"] is True
    assert lightning_resume["kind"] == "pytorch_lightning_full_checkpoint_resume"
    assert lightning_resume["model_class"] == "trainer.trainer.InstanceSegmentation"
    assert lightning_resume["sample_name"] == smoke.TINY_SAMPLE_NAME
    assert lightning_resume["architecture"] == {
        "model": "ReScene",
        "backbone": "Concerto",
        "backbone_class": "concerto.model.PointTransformerV3",
    }
    assert lightning_resume["data_scope"] == "single_real_3rscan_t2_window"
    assert lightning_resume["scannet_used"] is False
    assert lightning_resume["mixed_sampler_resume_validation"] is False
    assert lightning_resume["sampler_scope"] == (
        "single_real_3rscan_preflight_weighted_sampler"
    )
    assert lightning_resume["sampler_class"] == (
        "torch.utils.data.sampler.WeightedRandomSampler"
    )
    assert lightning_resume["sampler_generator_checkpointed"] is True
    assert lightning_resume["sampler_resume_scope"] == ("completed_epoch_boundary_only")
    assert lightning_resume["sampler_source_state_advanced"] is True
    assert lightning_resume["sampler_state_restored"] is True
    assert lightning_resume["sampler_state_advanced"] is True
    assert lightning_resume["sampler_state_restored_to_actual_loader_generator"] is True
    assert lightning_resume["sampler_stream_continuous"] is True
    assert lightning_resume["sampler_restore_event_order"] == [
        "setup_configured_sampler_verified",
        "setup_actual_loader_sampler_bound",
        "on_load_checkpoint_actual_loader_sampler_verified",
        "on_load_checkpoint_actual_loader_sampler_restored",
        "on_train_batch_start_state_observed",
    ]
    assert lightning_resume["main_formal_resume_checkpoint_selected"] is True
    assert lightning_resume["main_verified_resume_snapshot_used"] is True
    assert lightning_resume["resume_config_matches_checkpoint"] is True
    assert lightning_resume["checkpoint_callbacks_nonempty"] is True
    assert lightning_resume["formal_model_checkpoint_callback_count"] == 3
    assert lightning_resume["formal_model_checkpoint_state_keys_unique"] is True
    assert (
        lightning_resume["formal_model_checkpoint_persistent_states_restored"] is True
    )
    assert (
        lightning_resume["formal_model_checkpoint_transient_current_score_policy"]
        == "not_restored_by_pytorch_lightning_2_6_5"
    )
    assert (
        lightning_resume["formal_model_checkpoint_transient_current_score_not_restored"]
        is True
    )
    assert lightning_resume["source_real_validation_path_executed"] is True
    assert lightning_resume["source_validation_batch_count"] == 1
    assert lightning_resume["source_validation_dataset"] == "rio"
    assert lightning_resume["source_validation_sample"]["name"]
    assert lightning_resume["monitored_callback_history_populated"] is True
    assert lightning_resume["monitored_topk_checkpoint_bytes"] > 0
    assert lightning_resume["monitored_topk_checkpoint_removed"] is True
    assert (
        lightning_resume["source_checkpoint_generated_at_real_validation_epoch_end"]
        is True
    )
    assert lightning_resume["source_fit_loop_progress"] == {
        "epoch_total": {
            "ready": 1,
            "completed": 0,
            "started": 1,
            "processed": 0,
        },
        "batches_that_stepped": 65,
        "optimizer_steps_completed": 66,
        "scheduler_steps_completed": 66,
        "train_batches_completed": 264,
        "validation_batches_completed": 1,
    }
    assert lightning_resume["lightning_ckpt_path_restore"] is True
    assert lightning_resume["lightning_fit_weights_only"] is False
    assert lightning_resume["formal_optimizer_steps_per_epoch"] == 66
    assert lightning_resume["formal_train_batches_per_epoch"] == 264
    assert lightning_resume["gradient_accumulation_steps"] == 4
    assert lightning_resume["formal_completed_epoch_boundary"] is True
    assert lightning_resume["trainable_parameter_tensor_count"] > 0
    assert (
        lightning_resume["optimizer_param_group_parameter_count"]
        == lightning_resume["trainable_parameter_tensor_count"]
    )
    assert (
        lightning_resume["optimizer_parameter_contract_count"]
        == lightning_resume["trainable_parameter_tensor_count"]
    )
    assert lightning_resume["optimizer_parameter_group_order_matches_contract"] is True
    assert lightning_resume["trainable_parameter_schema_order_matches_contract"] is True
    assert lightning_resume["trainable_parameter_schema_matches_formal_contract"] is True
    assert lightning_resume["trainable_parameter_schema_sha256"] == (
        "d1c7fc483b1f217ab5734bec15292897eafff11b2dd86019a6e8e55e71513073"
    )
    assert (
        lightning_resume["optimizer_state_slot_count"]
        == lightning_resume["trainable_parameter_tensor_count"]
    )
    slot_step_values = lightning_resume["optimizer_state_slot_step_unique_values"]
    assert slot_step_values
    assert lightning_resume["optimizer_state_slot_step_min"] == slot_step_values[0]
    assert lightning_resume["optimizer_state_slot_step_max"] == slot_step_values[-1]
    assert 0 < slot_step_values[0] <= slot_step_values[-1] <= 66
    assert isinstance(
        lightning_resume["optimizer_state_slots_all_at_global_step"],
        bool,
    )
    assert (
        lightning_resume["optimizer_state_slots_cover_all_trainable_parameters"] is True
    )
    assert lightning_resume["model_state_schema_entry_count"] == 798
    assert lightning_resume["model_state_schema_serialized_bytes"] == 61441
    assert lightning_resume["model_state_schema_sha256"] == (
        "4dc8e5e8d455cec6a9f1ecd25653cd8a2736debb2e94c138a5fae6744562e069"
    )
    assert lightning_resume["model_state_schema_matches_formal_contract"] is True
    assert lightning_resume["saved_epoch"] == 0
    assert lightning_resume["saved_global_step"] == 66
    assert lightning_resume["restored_global_step"] == 66
    assert lightning_resume["advanced_global_step"] == 67
    assert lightning_resume["model_state_scope"] == "lightning_module_state_dict"
    assert lightning_resume["model_state_restored"] is True
    assert lightning_resume["optimizer_state_restored"] is True
    assert lightning_resume["scheduler_state_restored"] is True
    assert lightning_resume["model_state_advanced"] is True
    assert lightning_resume["optimizer_state_advanced"] is True
    assert lightning_resume["scheduler_state_advanced"] is True
    assert lightning_resume["saved_scheduler_last_epoch"] == 66
    assert lightning_resume["restored_scheduler_last_epoch"] == 66
    assert lightning_resume["advanced_scheduler_last_epoch"] == 67
    assert lightning_resume["elapsed_seconds"] > 0
    assert lightning_resume["peak_allocated_vram_mib"] > 0
    assert lightning_resume["temporary_checkpoint_removed"] is True
    assert lightning_resume["verified_resume_snapshot_removed"] is True
    assert payload["input_provenance"] == EXPECTED_INPUT_PROVENANCE
    assert payload["validation_evaluator"]["pipeline_executed"] is True
    assert payload["validation_evaluator"]["g2_metric_evidence"] is False
    assert payload["validation_evaluator"]["model_state"] == (
        "pretrained_concerto_encoder_with_seeded_randomly_initialized_decoder_"
        "and_heads_after_two_native_smoke_steps"
    )
    smoke.validate_tmap_schema(payload["validation_evaluator"]["schema_keys"])
    serialized = json.dumps(payload, sort_keys=True)
    assert "/" + "home" + "/" not in serialized
    assert "GPU" + "-" not in serialized
