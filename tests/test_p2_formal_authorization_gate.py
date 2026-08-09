import copy
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict

import main_instance_segmentation as training_entrypoint
from utils import p2_preflight

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "conf"
P2_CONFIG_NAME = "config_p2_rescene4d_concerto_t2"
OFFICIAL_SPLIT_COUNTS = {"train": 1201, "validation": 312, "test": 100}
NYU40_INSTANCE_IDS = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]
NYU40_INSTANCE_LABELS = [
    "cabinet",
    "bed",
    "chair",
    "sofa",
    "table",
    "door",
    "window",
    "bookshelf",
    "picture",
    "counter",
    "desk",
    "curtain",
    "refrigerator",
    "shower curtain",
    "toilet",
    "sink",
    "bathtub",
    "otherfurniture",
]
SCANNET_OFFICIAL_COMMIT = "3830fce7f8b2e48ef047ef7fd76ea5f62903f51c"
OFFICIAL_SPLIT_SHA256 = {
    "train": "96acca299b7855f02824c496b19077904d80996e7ced1bb9f0dac98f7dd4d0c8",
    "validation": "d75d4971c3fa7128c643695840e279042c212ef904fe933bd00cf9918c61b083",
    "test": "0214c6a3b1ee516ad653393b0321e7c0394c7662a4b3702eac1ddd7fbc00f7e0",
}
FORMAL_INPUT_MANIFEST = {
    "schema_version": 1,
    "status": "pass",
    "roots": {
        "scannet": "repo:data/processed/scannet",
        "rio": "repo:data/processed/rio",
    },
    "scannet": {
        "file_count": 4842,
        "total_bytes": 1,
        "content_sha256": "1" * 64,
    },
    "rio": {
        "file_count": 5855,
        "total_bytes": 1,
        "content_sha256": "2" * 64,
    },
}


@pytest.fixture(autouse=True)
def _current_input_manifest_matches_the_formal_fixture(monkeypatch) -> None:
    monkeypatch.setattr(
        p2_preflight,
        "build_p2_input_manifest",
        lambda *args, **kwargs: copy.deepcopy(FORMAL_INPUT_MANIFEST),
    )
    monkeypatch.setattr(
        p2_preflight,
        "build_scannet_official_split_identity",
        lambda *args, **kwargs: {
            "status": "pass",
            "repository_ref": "external:github/ScanNet/ScanNet",
            "expected_commit": SCANNET_OFFICIAL_COMMIT,
            "observed_commit": SCANNET_OFFICIAL_COMMIT,
            "files": {
                split: {
                    "reference": (
                        "repo:third_party/ScanNet/Tasks/Benchmark/"
                        + (
                            "scannetv2_val.txt"
                            if split == "validation"
                            else f"scannetv2_{split}.txt"
                        )
                    ),
                    "expected_sha256": OFFICIAL_SPLIT_SHA256[split],
                    "observed_sha256": OFFICIAL_SPLIT_SHA256[split],
                    "expected_scene_count": expected,
                    "observed_scene_count": expected,
                    "status": "pass",
                }
                for split, expected in OFFICIAL_SPLIT_COUNTS.items()
            },
        },
    )


def _compose(config_name: str):
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.2"):
        return compose(config_name=config_name)


def _canonical_sha256(value) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _config_sha256(cfg) -> str:
    payload = OmegaConf.to_container(cfg, resolve=True)
    return _canonical_sha256(payload)


def _artifact_sha256(artifact: dict) -> str:
    payload = json.loads(json.dumps(artifact))
    payload["authorization"].pop("artifact_payload_sha256", None)
    return _canonical_sha256(payload)


def _formal_artifact(cfg, *, issued_at: datetime | None = None) -> dict:
    issued_at = issued_at or datetime.now(timezone.utc)
    total_scenes = sum(OFFICIAL_SPLIT_COUNTS.values())
    instance_scenes = (
        OFFICIAL_SPLIT_COUNTS["train"] + OFFICIAL_SPLIT_COUNTS["validation"]
    )
    artifact = {
        "schema_version": 2,
        "status": "pass",
        "formal_p2_training_authorized": True,
        "official_source_commit": "fb2fe42eb8f1e926567c48eea9acb874e608ee10",
        "expected_split_counts": OFFICIAL_SPLIT_COUNTS,
        "split_metadata_status": "pass",
        "split_metadata": {
            split: {
                "expected": expected,
                "observed": expected,
                "unique": expected,
                "status": "pass",
            }
            for split, expected in OFFICIAL_SPLIT_COUNTS.items()
        },
        "official_split_identity": {
            "status": "pass",
            "repository_ref": "external:github/ScanNet/ScanNet",
            "expected_commit": SCANNET_OFFICIAL_COMMIT,
            "observed_commit": SCANNET_OFFICIAL_COMMIT,
            "files": {
                split: {
                    "reference": (
                        "repo:third_party/ScanNet/Tasks/Benchmark/"
                        + (
                            "scannetv2_val.txt"
                            if split == "validation"
                            else f"scannetv2_{split}.txt"
                        )
                    ),
                    "expected_sha256": OFFICIAL_SPLIT_SHA256[split],
                    "observed_sha256": OFFICIAL_SPLIT_SHA256[split],
                    "expected_scene_count": expected,
                    "observed_scene_count": expected,
                    "status": "pass",
                }
                for split, expected in OFFICIAL_SPLIT_COUNTS.items()
            },
        },
        "model_checkpoint": {
            "reference": "local_cache:persist4d/concerto/concerto_base.pth",
            "expected_sha256": "845ec7dec97a5fabff8fadb5d9858ac6734347b612d1a4b574213419c139de07",
            "observed_sha256": "845ec7dec97a5fabff8fadb5d9858ac6734347b612d1a4b574213419c139de07",
            "expected_byte_size": 433987358,
            "observed_byte_size": 433987358,
            "status": "pass",
        },
        "raw_assets": {
            "status": "pass",
            "expected_scene_count": total_scenes,
            "complete_scene_count": total_scenes,
            "missing_asset_count": 0,
        },
        "processed_assets": {
            "status": "pass",
            "expected_scene_count": total_scenes,
            "database_scene_count": total_scenes,
            "npy_scene_count": total_scenes,
            "instance_gt_scene_count": instance_scenes,
            "by_split": {
                split: {
                    "expected_scene_count": expected,
                    "database_record_count": expected,
                    "database_scene_count": expected,
                    "npy_scene_count": expected,
                    "instance_gt_scene_count": 0 if split == "test" else expected,
                    "status": "pass",
                }
                for split, expected in OFFICIAL_SPLIT_COUNTS.items()
            },
        },
        "class_taxonomy": {
            "status": "pass",
            "name": "scannet",
            "valid_class_ids": list(NYU40_INSTANCE_IDS),
            "class_labels": list(NYU40_INSTANCE_LABELS),
            "class_count": 18,
        },
        "rio_class_taxonomy": {
            "status": "pass",
            "name": "rio",
            "valid_class_ids": list(NYU40_INSTANCE_IDS),
            "class_labels": list(NYU40_INSTANCE_LABELS),
            "class_count": 18,
        },
        "known_empty_scan_substitutions": {
            "status": "pass",
            "dataset": "rio",
            "temporal_window": 2,
            "known_empty_scan_id": "0171_01",
            "policy": "official_substitute",
            "sequence_database_ref": (
                "repo:data/processed/rio/sequence_database_sliding_2.yaml"
            ),
            "expected_sequence_database_sha256": (
                "974299916db02d2ee0233564eb7b36e314dada93e997584d9dfef21ad70d0416"
            ),
            "observed_sequence_database_sha256": (
                "974299916db02d2ee0233564eb7b36e314dada93e997584d9dfef21ad70d0416"
            ),
            "fail_closed": {
                "train": True,
                "validation": True,
                "test": True,
            },
            "affected_sequences": [
                "scene0171_00-scene0171_01",
                "scene0171_01-scene0171_02",
            ],
            "scannet_known_empty_scan_ids": [
                "scene0154_00",
                "scene0636_00",
            ],
        },
        "data_root_bindings": {
            "status": "pass",
            "expected": {
                "raw_scannet": "repo:data/raw/scannet/scannet",
                "scannet": "repo:data/processed/scannet",
                "rio": "repo:data/processed/rio",
                "split_metadata": (
                    "repo:third_party/ScanNet/Tasks/Benchmark"
                ),
                "test_segments": "repo:data/raw/scannet_test_segments",
            },
            "observed": {
                "raw_scannet": "repo:data/raw/scannet/scannet",
                "scannet": "repo:data/processed/scannet",
                "rio": "repo:data/processed/rio",
                "split_metadata": (
                    "repo:third_party/ScanNet/Tasks/Benchmark"
                ),
                "test_segments": "repo:data/raw/scannet_test_segments",
            },
        },
        "rio_path_integrity": {
            "status": "pass",
            "database_record_counts": {
                "train": 1178,
                "validation": 157,
            },
            "sequence_record_count": 1482,
            "content_validation": "pass",
            "supervised_record_count": 1326,
            "unsupervised_sequences": [],
        },
        "input_manifest": copy.deepcopy(FORMAL_INPUT_MANIFEST),
        "mix_instantiation": {
            "attempted": True,
            "status": "pass",
            "implementation": "datasets.multi_dataset.MultiDataset",
            "dataset_names": ["rio", "scannet"],
            "dataset_sizes": [1178, 1201],
            "weights": [1.0, 0.8],
            "temporal_windows": [2, 1],
            "sampler": "WeightedRandomSampler",
        },
        "errors": [],
        "config_contract": {
            "schema_version": 1,
            "status": "pass",
            "errors": [],
            "expected_semantic_sha256": (
                "4e6532a02bb67e1c1a9f990010d1ba89f4d40d596b9790f91b79ff70566565bc"
            ),
            "observed_semantic_sha256": (
                "4e6532a02bb67e1c1a9f990010d1ba89f4d40d596b9790f91b79ff70566565bc"
            ),
        },
        "authorization": {
            "schema_version": 1,
            "status": "issued",
            "config_ref": "repo:conf/config_p2_rescene4d_concerto_t2.yaml",
            "config_sha256": _config_sha256(cfg),
            "expected_split_counts": OFFICIAL_SPLIT_COUNTS,
            "issued_at_utc": issued_at.isoformat().replace("+00:00", "Z"),
            "max_age_seconds": 86400,
        },
    }
    artifact["authorization"]["artifact_payload_sha256"] = _artifact_sha256(
        artifact
    )
    return artifact


def _write_artifact(path: Path, artifact: dict) -> None:
    path.write_text(json.dumps(artifact), encoding="utf-8")


def _sampler_generator_checkpoint_payload() -> dict:
    generator = torch.Generator()
    generator.manual_seed(45)
    return {
        "schema_version": 1,
        "resume_scope": "completed_epoch_boundary_only",
        "mid_epoch_resume_supported": False,
        "dataloader_prefetch_state_checkpointed": False,
        "generator_state": generator.get_state(),
    }


def _formal_resume_payload(cfg) -> dict:
    return {
        "pytorch-lightning_version": "2.6.5",
        "state_dict": {"model.weight": torch.ones(1)},
        "optimizer_states": [
            {"state": {}, "param_groups": [{"params": []}]}
        ],
        "lr_schedulers": [{"last_epoch": 0}],
        "loops": {"fit_loop": {"state": 1}},
        "callbacks": {"ModelCheckpoint": {"best_model_path": ""}},
        "epoch": 0,
        "global_step": 0,
        "hyper_parameters": cfg,
        "p2_train_sampler_generator": _sampler_generator_checkpoint_payload(),
    }


def test_fresh_bound_preflight_authorizes_the_exact_p2_config(tmp_path: Path) -> None:
    cfg = _compose(P2_CONFIG_NAME)
    artifact_path = tmp_path / "scannet_preflight.json"
    _write_artifact(artifact_path, _formal_artifact(cfg))

    training_entrypoint.require_p2_preflight_authorization(
        cfg,
        artifact_path=artifact_path,
    )


def test_missing_preflight_fails_closed_before_formal_p2_training(
    tmp_path: Path,
) -> None:
    cfg = _compose(P2_CONFIG_NAME)

    with pytest.raises(RuntimeError, match="P2 preflight.*missing"):
        training_entrypoint.require_p2_preflight_authorization(
            cfg,
            artifact_path=tmp_path / "missing.json",
        )


def test_stale_preflight_fails_closed(tmp_path: Path) -> None:
    cfg = _compose(P2_CONFIG_NAME)
    artifact_path = tmp_path / "scannet_preflight.json"
    stale_time = datetime.now(timezone.utc) - timedelta(seconds=86401)
    _write_artifact(artifact_path, _formal_artifact(cfg, issued_at=stale_time))

    with pytest.raises(RuntimeError, match="stale"):
        training_entrypoint.require_p2_preflight_authorization(
            cfg,
            artifact_path=artifact_path,
        )


def test_preflight_bound_to_another_resolved_config_fails_closed(
    tmp_path: Path,
) -> None:
    cfg = _compose(P2_CONFIG_NAME)
    artifact_path = tmp_path / "scannet_preflight.json"
    _write_artifact(artifact_path, _formal_artifact(cfg))
    cfg.optimizer.lr = 0.0001

    with pytest.raises(RuntimeError, match="config_sha256"):
        training_entrypoint.require_p2_preflight_authorization(
            cfg,
            artifact_path=artifact_path,
        )


def test_artifact_and_config_cannot_drift_together_from_the_p2_contract(
    tmp_path: Path,
) -> None:
    cfg = _compose(P2_CONFIG_NAME)
    cfg.optimizer.lr = 0.0001
    artifact_path = tmp_path / "scannet_preflight.json"
    _write_artifact(artifact_path, _formal_artifact(cfg))

    with pytest.raises(RuntimeError, match="optimizer.lr"):
        training_entrypoint.require_p2_preflight_authorization(
            cfg,
            artifact_path=artifact_path,
        )


def test_unlisted_model_and_trainer_behavior_drift_is_rejected(
    tmp_path: Path,
) -> None:
    cfg = _compose(P2_CONFIG_NAME)
    cfg.model._target_ = "models.UnreviewedReplacement"
    with open_dict(cfg.trainer):
        cfg.trainer.fast_dev_run = True
        cfg.trainer.limit_train_batches = 1
    artifact_path = tmp_path / "scannet_preflight.json"
    _write_artifact(artifact_path, _formal_artifact(cfg))

    with pytest.raises(RuntimeError, match="semantic_sha256"):
        training_entrypoint.require_p2_preflight_authorization(
            cfg,
            artifact_path=artifact_path,
        )


def test_current_training_input_manifest_is_recomputed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cfg = _compose(P2_CONFIG_NAME)
    artifact_path = tmp_path / "scannet_preflight.json"
    _write_artifact(artifact_path, _formal_artifact(cfg))
    changed = copy.deepcopy(FORMAL_INPUT_MANIFEST)
    changed["rio"]["content_sha256"] = "3" * 64
    monkeypatch.setattr(
        p2_preflight,
        "build_p2_input_manifest",
        lambda *args, **kwargs: changed,
    )

    with pytest.raises(RuntimeError, match="current input_manifest"):
        training_entrypoint.require_p2_preflight_authorization(
            cfg,
            artifact_path=artifact_path,
        )


def test_freshness_is_rechecked_after_expensive_input_validation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cfg = _compose(P2_CONFIG_NAME)
    base = datetime.now(timezone.utc)
    artifact_path = tmp_path / "scannet_preflight.json"
    _write_artifact(
        artifact_path,
        _formal_artifact(
            cfg,
            issued_at=base - timedelta(seconds=86400 - 1),
        ),
    )

    class AdvancingDateTime(datetime):
        calls = 0

        @classmethod
        def now(cls, tz=None):
            cls.calls += 1
            value = base if cls.calls == 1 else base + timedelta(seconds=2)
            return value if tz is not None else value.replace(tzinfo=None)

    monkeypatch.setattr(p2_preflight, "datetime", AdvancingDateTime)

    with pytest.raises(RuntimeError, match="stale"):
        training_entrypoint.require_p2_preflight_authorization(
            cfg,
            artifact_path=artifact_path,
        )


def test_semantically_tampered_preflight_fails_even_with_recomputed_digest(
    tmp_path: Path,
) -> None:
    cfg = _compose(P2_CONFIG_NAME)
    artifact_path = tmp_path / "scannet_preflight.json"
    artifact = _formal_artifact(cfg)
    artifact["raw_assets"]["complete_scene_count"] = 0
    artifact["authorization"]["artifact_payload_sha256"] = _artifact_sha256(
        artifact
    )
    _write_artifact(artifact_path, artifact)

    with pytest.raises(RuntimeError, match="raw_assets"):
        training_entrypoint.require_p2_preflight_authorization(
            cfg,
            artifact_path=artifact_path,
        )


def test_unverified_checkpoint_evidence_fails_even_with_recomputed_digest(
    tmp_path: Path,
) -> None:
    cfg = _compose(P2_CONFIG_NAME)
    artifact_path = tmp_path / "scannet_preflight.json"
    artifact = _formal_artifact(cfg)
    artifact["model_checkpoint"]["status"] = "fail"
    artifact["authorization"]["artifact_payload_sha256"] = _artifact_sha256(
        artifact
    )
    _write_artifact(artifact_path, artifact)

    with pytest.raises(RuntimeError, match="model_checkpoint"):
        training_entrypoint.require_p2_preflight_authorization(
            cfg,
            artifact_path=artifact_path,
        )


def test_tampered_known_empty_substitution_evidence_fails_closed(
    tmp_path: Path,
) -> None:
    cfg = _compose(P2_CONFIG_NAME)
    artifact_path = tmp_path / "scannet_preflight.json"
    artifact = _formal_artifact(cfg)
    artifact["known_empty_scan_substitutions"]["affected_sequences"] = []
    artifact["authorization"]["artifact_payload_sha256"] = _artifact_sha256(
        artifact
    )
    _write_artifact(artifact_path, artifact)

    with pytest.raises(RuntimeError, match="known_empty_scan_substitutions"):
        training_entrypoint.require_p2_preflight_authorization(
            cfg,
            artifact_path=artifact_path,
        )


def test_tampered_rio_taxonomy_fails_closed_even_with_a_new_digest(
    tmp_path: Path,
) -> None:
    cfg = _compose(P2_CONFIG_NAME)
    artifact_path = tmp_path / "scannet_preflight.json"
    artifact = _formal_artifact(cfg)
    artifact["rio_class_taxonomy"]["class_labels"][0] = "not-cabinet"
    artifact["authorization"]["artifact_payload_sha256"] = _artifact_sha256(
        artifact
    )
    _write_artifact(artifact_path, artifact)

    with pytest.raises(RuntimeError, match="rio_class_taxonomy"):
        training_entrypoint.require_p2_preflight_authorization(
            cfg,
            artifact_path=artifact_path,
        )


def test_formal_mix_size_must_match_the_exact_pinned_t2_baseline(
    tmp_path: Path,
) -> None:
    cfg = _compose(P2_CONFIG_NAME)
    artifact_path = tmp_path / "scannet_preflight.json"
    artifact = _formal_artifact(cfg)
    artifact["mix_instantiation"]["dataset_sizes"] = [1, 1201]
    artifact["authorization"]["artifact_payload_sha256"] = _artifact_sha256(
        artifact
    )
    _write_artifact(artifact_path, artifact)

    with pytest.raises(RuntimeError, match="dataset_sizes"):
        training_entrypoint.require_p2_preflight_authorization(
            cfg,
            artifact_path=artifact_path,
        )


def test_formal_p2_checkpoint_requires_an_exact_local_sha256(
    tmp_path: Path,
    monkeypatch,
) -> None:
    checkpoint = tmp_path / "concerto_base.pth"
    checkpoint.write_bytes(b"verified-local-concerto")
    expected_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    monkeypatch.setattr(
        training_entrypoint,
        "P2_CONCERTO_CHECKPOINT_SHA256",
        expected_sha256,
        raising=False,
    )
    monkeypatch.setattr(
        training_entrypoint,
        "P2_CONCERTO_CHECKPOINT_BYTES",
        checkpoint.stat().st_size,
        raising=False,
    )
    monkeypatch.setattr(training_entrypoint, "_REPO_ROOT", tmp_path)
    cfg = _compose(P2_CONFIG_NAME)
    cfg.backbone.name = checkpoint.name

    snapshot = training_entrypoint.require_p2_concerto_checkpoint(cfg)
    assert Path(cfg.backbone.name) == snapshot
    assert snapshot != checkpoint.resolve()
    assert snapshot.read_bytes() == b"verified-local-concerto"

    checkpoint.write_bytes(b"changed-after-verification")
    assert snapshot.read_bytes() == b"verified-local-concerto"

    cfg.backbone.name = "concerto_base"
    with pytest.raises(RuntimeError, match="local file"):
        training_entrypoint.require_p2_concerto_checkpoint(cfg)


def test_formal_p2_requires_repository_working_directory(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="repository working directory"):
        training_entrypoint.require_p2_repository_cwd(
            cwd=tmp_path,
            repo_root=REPO_ROOT,
        )


def test_formal_p2_checkpoint_rejects_wrong_local_sha256(
    tmp_path: Path,
    monkeypatch,
) -> None:
    checkpoint = tmp_path / "concerto_base.pth"
    checkpoint.write_bytes(b"tampered-concerto")
    monkeypatch.setattr(
        training_entrypoint,
        "P2_CONCERTO_CHECKPOINT_SHA256",
        "0" * 64,
        raising=False,
    )
    monkeypatch.setattr(
        training_entrypoint,
        "P2_CONCERTO_CHECKPOINT_BYTES",
        checkpoint.stat().st_size,
        raising=False,
    )
    cfg = _compose(P2_CONFIG_NAME)
    cfg.general.save_dir = str(tmp_path / "verified-snapshots")
    cfg.backbone.name = str(checkpoint)

    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        training_entrypoint.require_p2_concerto_checkpoint(cfg)


def test_formal_p2_resume_rejects_a_checkpoint_from_another_config(
    tmp_path: Path,
) -> None:
    cfg = _compose(P2_CONFIG_NAME)
    other_cfg = _compose(P2_CONFIG_NAME)
    cfg.general.save_dir = str(tmp_path / "verified-snapshots")
    other_cfg.general.save_dir = cfg.general.save_dir
    other_cfg.optimizer.lr = 0.0001
    checkpoint = tmp_path / "other-config.ckpt"
    torch.save(_formal_resume_payload(other_cfg), checkpoint)

    with pytest.raises(RuntimeError, match="config_sha256"):
        training_entrypoint.require_p2_resume_checkpoint(cfg, checkpoint)


def test_formal_p2_resume_accepts_matching_p2_hyperparameters(
    tmp_path: Path,
) -> None:
    cfg = _compose(P2_CONFIG_NAME)
    cfg.general.save_dir = str(tmp_path / "verified-snapshots")
    checkpoint = tmp_path / "matching-config.ckpt"
    torch.save(_formal_resume_payload(cfg), checkpoint)

    training_entrypoint.require_p2_resume_checkpoint(cfg, checkpoint)


def test_formal_p2_resume_rejects_missing_sampler_generator_payload(
    tmp_path: Path,
) -> None:
    cfg = _compose(P2_CONFIG_NAME)
    cfg.general.save_dir = str(tmp_path / "verified-snapshots")
    checkpoint = tmp_path / "missing-sampler-state.ckpt"
    payload = _formal_resume_payload(cfg)
    payload.pop("p2_train_sampler_generator")
    torch.save(payload, checkpoint)

    with pytest.raises(RuntimeError, match="sampler generator payload"):
        training_entrypoint.require_p2_resume_checkpoint(cfg, checkpoint)


@pytest.mark.parametrize(
    "missing_field",
    ["schema_version", "resume_scope", "generator_state"],
)
def test_formal_p2_resume_rejects_incomplete_sampler_generator_payload(
    tmp_path: Path,
    missing_field: str,
) -> None:
    cfg = _compose(P2_CONFIG_NAME)
    cfg.general.save_dir = str(tmp_path / "verified-snapshots")
    checkpoint = tmp_path / f"missing-{missing_field}.ckpt"
    payload = _formal_resume_payload(cfg)
    payload["p2_train_sampler_generator"].pop(missing_field)
    torch.save(payload, checkpoint)

    with pytest.raises(RuntimeError, match=missing_field.replace("_", ".*")):
        training_entrypoint.require_p2_resume_checkpoint(cfg, checkpoint)


def test_formal_p2_resume_rejects_invalid_sampler_generator_state(
    tmp_path: Path,
) -> None:
    cfg = _compose(P2_CONFIG_NAME)
    cfg.general.save_dir = str(tmp_path / "verified-snapshots")
    checkpoint = tmp_path / "invalid-sampler-state.ckpt"
    payload = _formal_resume_payload(cfg)
    payload["p2_train_sampler_generator"]["generator_state"] = torch.ones(
        1,
        dtype=torch.uint8,
    )
    torch.save(payload, checkpoint)

    with pytest.raises(RuntimeError, match="sampler generator state"):
        training_entrypoint.require_p2_resume_checkpoint(cfg, checkpoint)


def test_formal_p2_resume_revalidates_full_state_after_snapshot(
    tmp_path: Path,
) -> None:
    cfg = _compose(P2_CONFIG_NAME)
    cfg.general.save_dir = str(tmp_path / "verified-snapshots")
    checkpoint = tmp_path / "hyperparameters-only.ckpt"
    torch.save({"hyper_parameters": cfg}, checkpoint)

    with pytest.raises(RuntimeError, match="not fully resumable"):
        training_entrypoint.require_p2_resume_checkpoint(cfg, checkpoint)


class _ReachedTraining(Exception):
    pass


def _record_entrypoint_fit(monkeypatch):
    model = object()
    fit_calls = []

    class RecordingTrainer:
        def __init__(self, **_kwargs) -> None:
            pass

        def fit(self, candidate_model, **kwargs) -> None:
            fit_calls.append((candidate_model, kwargs))

    monkeypatch.setattr(training_entrypoint, "Trainer", RecordingTrainer)
    monkeypatch.setattr(
        training_entrypoint,
        "get_parameters",
        lambda cfg: (cfg, model),
    )
    return model, fit_calls


def test_csv_logger_does_not_enter_wandb_sweep_handling(monkeypatch) -> None:
    cfg = _compose(P2_CONFIG_NAME)
    csv_logger = type("FakeCSVLogger", (), {"experiment": object()})()
    monkeypatch.setattr(
        training_entrypoint,
        "_enforce_formal_p2_training",
        lambda cfg: None,
        raising=False,
    )
    monkeypatch.setattr(
        training_entrypoint.hydra.utils,
        "instantiate",
        lambda logger_cfg: csv_logger,
    )
    monkeypatch.setattr(
        training_entrypoint,
        "get_parameters",
        lambda cfg: (_ for _ in ()).throw(_ReachedTraining()),
    )

    with pytest.raises(_ReachedTraining):
        training_entrypoint.train.__wrapped__(cfg)


def test_p2_train_calls_formal_gate_before_logger_instantiation(monkeypatch) -> None:
    cfg = _compose(P2_CONFIG_NAME)
    calls = []

    def fail_gate(candidate_cfg):
        calls.append(candidate_cfg)
        raise RuntimeError("formal P2 gate called")

    monkeypatch.setattr(
        training_entrypoint,
        "_enforce_formal_p2_training",
        fail_gate,
        raising=False,
    )
    monkeypatch.setattr(
        training_entrypoint.hydra.utils,
        "instantiate",
        lambda *_args, **_kwargs: pytest.fail("logger instantiated before P2 gate"),
    )

    with pytest.raises(RuntimeError, match="formal P2 gate called"):
        training_entrypoint.train.__wrapped__(cfg)

    assert calls == [cfg]


def test_formal_resume_explicitly_disables_weights_only_for_trainer_fit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cfg = _compose(P2_CONFIG_NAME)
    with open_dict(cfg):
        cfg.general.save_dir = str(tmp_path)
        cfg.logging = []
        cfg.callbacks = []
    selected = tmp_path / "last.ckpt"
    verified_snapshot = tmp_path / "verified-resume.ckpt"
    selected.touch()
    verified_snapshot.touch()
    monkeypatch.setattr(
        training_entrypoint,
        "_enforce_formal_p2_training",
        lambda _cfg: None,
    )
    monkeypatch.setattr(
        training_entrypoint,
        "find_resume_checkpoint",
        lambda _save_dir, *, formal_p2=False: (
            str(selected) if formal_p2 else pytest.fail("formal selector not used")
        ),
    )
    monkeypatch.setattr(
        training_entrypoint,
        "require_p2_resume_checkpoint",
        lambda _cfg, checkpoint: (
            str(verified_snapshot)
            if checkpoint == str(selected)
            else pytest.fail("unexpected resume candidate")
        ),
    )
    model, fit_calls = _record_entrypoint_fit(monkeypatch)

    training_entrypoint.train.__wrapped__(cfg)

    assert fit_calls == [
        (
            model,
            {"ckpt_path": str(verified_snapshot), "weights_only": False},
        )
    ]


def test_fresh_train_keeps_default_checkpoint_loading_behavior(monkeypatch) -> None:
    cfg = _compose("config_base_instance_segmentation")
    with open_dict(cfg):
        cfg.logging = []
        cfg.callbacks = []
    monkeypatch.setattr(
        training_entrypoint,
        "find_resume_checkpoint",
        lambda _save_dir: None,
    )
    model, fit_calls = _record_entrypoint_fit(monkeypatch)

    training_entrypoint.train.__wrapped__(cfg)

    assert fit_calls == [(model, {"ckpt_path": None})]


def _assert_formal_identity_consumes_preflight(cfg, monkeypatch) -> None:
    monkeypatch.setattr(
        training_entrypoint,
        "require_p2_repository_cwd",
        lambda: None,
    )
    monkeypatch.setattr(
        training_entrypoint,
        "require_p2_preflight_authorization",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("formal P2 identity consumed preflight")
        ),
    )

    with pytest.raises(RuntimeError, match="formal P2 identity consumed preflight"):
        training_entrypoint._enforce_formal_p2_training(cfg)


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("experiment_name", p2_preflight.P2_EXPERIMENT_NAME),
        ("project_name", p2_preflight.P2_EXPERIMENT_NAME),
        ("save_dir", p2_preflight.P2_SAVE_DIR),
    ],
)
def test_formal_p2_general_identity_cannot_bypass_gate_without_marker(
    field_name: str,
    field_value: str,
    monkeypatch,
) -> None:
    cfg = _compose("config_base_instance_segmentation")
    assert "p2_preflight" not in cfg
    setattr(cfg.general, field_name, field_value)

    _assert_formal_identity_consumes_preflight(cfg, monkeypatch)


@pytest.mark.parametrize(
    "callback_identity",
    [
        {
            "dirpath": p2_preflight.P2_SAVE_DIR,
            "filename": "periodic-epoch={epoch:03d}",
        },
        {
            "dirpath": "checkpoints",
            "filename": p2_preflight.P2_EXPERIMENT_NAME,
        },
    ],
)
def test_formal_p2_callback_identity_cannot_bypass_gate_without_marker(
    callback_identity: dict,
    monkeypatch,
) -> None:
    cfg = _compose("config_base_instance_segmentation")
    assert "p2_preflight" not in cfg
    cfg.callbacks.append(
        {
            "_target_": "pytorch_lightning.callbacks.ModelCheckpoint",
            **callback_identity,
        }
    )

    _assert_formal_identity_consumes_preflight(cfg, monkeypatch)


def test_non_p2_config_does_not_require_the_formal_p2_gate(monkeypatch) -> None:
    cfg = _compose("config_base_instance_segmentation")
    monkeypatch.setattr(
        training_entrypoint,
        "require_p2_preflight_authorization",
        lambda *_args, **_kwargs: pytest.fail("P2 preflight consumed for non-P2 config"),
        raising=False,
    )
    monkeypatch.setattr(
        training_entrypoint,
        "require_p2_concerto_checkpoint",
        lambda *_args, **_kwargs: pytest.fail("P2 checkpoint checked for non-P2 config"),
        raising=False,
    )
    monkeypatch.setattr(
        training_entrypoint.hydra.utils,
        "instantiate",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(_ReachedTraining()),
    )

    with pytest.raises(_ReachedTraining):
        training_entrypoint.train.__wrapped__(cfg)
