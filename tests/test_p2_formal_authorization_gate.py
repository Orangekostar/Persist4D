import copy
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import hydra
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset

import main_instance_segmentation as training_entrypoint
from utils import p2_preflight

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "conf"
P2_CONFIG_NAME = "config_p2_rescene4d_concerto_t2"
OFFICIAL_SPLIT_COUNTS = {"train": 1201, "validation": 312, "test": 100}
_FORMAL_DATA_ROOT_CONTRACT = p2_preflight.p2_data_root_reference_contract()
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
P2_EXCLUDED_SEQUENCE_NAMES = {
    "train": [
        "scene0242_00-scene0242_01",
        "scene0242_01-scene0242_02",
        "scene0242_02-scene0242_00",
        "scene0245_01-scene0245_02",
    ],
    "validation": [
        "scene0439_00-scene0439_02",
        "scene0439_01-scene0439_00",
        "scene0439_02-scene0439_01",
    ],
}
P2_EXCLUDED_SEQUENCE_NAMES["test"] = list(
    P2_EXCLUDED_SEQUENCE_NAMES["validation"]
)
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
        "scannet": _FORMAL_DATA_ROOT_CONTRACT["expected_resolved"][
            "scannet"
        ],
        "rio": _FORMAL_DATA_ROOT_CONTRACT["expected_resolved"]["rio"],
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
LOCAL_SOURCE_COMMIT = "a" * 40
FORMAL_TEST_MODEL_STATE_SCHEMA = {
    "model.bias": {"shape": [2], "dtype": "torch.float32"},
    "model.running_mean": {"shape": [3], "dtype": "torch.float32"},
    "model.weight": {"shape": [1], "dtype": "torch.float32"},
}


def _model_state_schema_sha256(schema: dict) -> str:
    entries = [
        [name, metadata["shape"], metadata["dtype"]]
        for name, metadata in sorted(schema.items())
    ]
    payload = json.dumps(entries, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _parameter_schema_sha256(entries: list[list[object]]) -> str:
    payload = json.dumps(entries, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


FORMAL_TEST_MODEL_STATE_SCHEMA_SHA256 = _model_state_schema_sha256(
    FORMAL_TEST_MODEL_STATE_SCHEMA
)
FORMAL_TEST_TRAINABLE_PARAMETER_SCHEMA = [
    ["model.weight", [1], "torch.float32"],
    ["model.bias", [2], "torch.float32"],
]
FORMAL_TEST_TRAINABLE_PARAMETER_SCHEMA_SHA256 = _parameter_schema_sha256(
    FORMAL_TEST_TRAINABLE_PARAMETER_SCHEMA
)


def _passing_source_tree_contract(source_commit: str | None = None) -> dict:
    pinned_commit = source_commit or LOCAL_SOURCE_COMMIT
    return {
        "schema_version": 1,
        "status": "pass",
        "source_commit": pinned_commit,
        "observed_head": pinned_commit,
        "allowed_dirty_prefixes": ["artifacts/P2/"],
        "committed_paths_since_source": [],
        "dirty_paths": ["artifacts/P2/scannet_preflight.json"],
        "index_flag_paths": [],
        "expected_tracked_tree_sha256": "3" * 64,
        "observed_tracked_tree_sha256": "3" * 64,
        "disallowed_committed_paths": [],
        "disallowed_dirty_paths": [],
        "errors": [],
    }


def _passing_runtime_source_contract() -> dict:
    repositories = {}
    for name, definition in p2_preflight.P2_RUNTIME_SOURCE_REPOSITORIES.items():
        relative_root = definition["relative_root"]
        native_extensions = {}
        for module, extension in definition.get("native_extensions", {}).items():
            native_extensions[module] = {
                "origin_ref": (
                    "repo:third_party/detectron2/detectron2/"
                    "_C.cpython-310-x86_64-linux-gnu.so"
                ),
                "expected_byte_size": extension["expected_byte_size"],
                "observed_byte_size": extension["expected_byte_size"],
                "expected_sha256": extension["expected_sha256"],
                "observed_sha256": extension["expected_sha256"],
                "status": "pass",
            }
        repositories[name] = {
            "reference": f"repo:{relative_root}",
            "module": definition["module"],
            "expected_commit": definition["expected_commit"],
            "observed_commit": definition["expected_commit"],
            "module_origin_ref": (
                f"repo:{relative_root}/{definition['module']}/__init__.py"
            ),
            "dirty_paths": [],
            "index_flag_paths": [],
            "expected_tracked_tree_sha256": "4" * 64,
            "observed_tracked_tree_sha256": "4" * 64,
            "native_extensions": native_extensions,
            "status": "pass",
            "errors": [],
        }
    return {
        "schema_version": 1,
        "status": "pass",
        "repositories": repositories,
        "errors": [],
    }


def _passing_runtime_environment_contract() -> dict:
    python_source_components = {"pytorch_lightning", "hydra", "omegaconf"}
    components = {
        name: {
            "status": "pass",
            "origin_refs": [f"env:{name}"],
            (
                "python_source_manifest"
                if name in python_source_components
                else "native_manifest"
            ): {
                "file_count": 1,
                "total_bytes": 1,
                "content_sha256": "5" * 64,
            },
            "errors": [],
        }
        for name in (
            "python",
            "torch",
            "spconv",
            "cumm",
            "flash_attn",
            "torch_scatter",
            "pointnet2",
            "nvidia_cuda_libraries",
            "pytorch_lightning",
            "hydra",
            "omegaconf",
        )
    }
    return {
        "schema_version": (
            p2_preflight.P2_RUNTIME_ENVIRONMENT_CONTRACT_SCHEMA_VERSION
        ),
        "status": "pass",
        "versions": copy.deepcopy(p2_preflight.P2_RUNTIME_ENVIRONMENT_VERSIONS),
        "components": components,
        "optional_modules": {
            "pointops": {"required": False, "status": "absent"}
        },
        "errors": [],
    }


@pytest.fixture(autouse=True)
def _current_input_manifest_matches_the_formal_fixture(monkeypatch) -> None:
    monkeypatch.setattr(
        training_entrypoint,
        "_P2_FORMAL_MODEL_STATE_SCHEMA_SHA256",
        FORMAL_TEST_MODEL_STATE_SCHEMA_SHA256,
    )
    monkeypatch.setattr(
        training_entrypoint,
        "_P2_FORMAL_TRAINABLE_PARAMETER_SCHEMA_SHA256",
        FORMAL_TEST_TRAINABLE_PARAMETER_SCHEMA_SHA256,
    )
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
    monkeypatch.setattr(
        p2_preflight,
        "build_p2_source_tree_contract",
        lambda *args, source_commit=None, **kwargs: _passing_source_tree_contract(
            source_commit
        ),
        raising=False,
    )
    monkeypatch.setattr(
        p2_preflight,
        "build_p2_runtime_source_contract",
        lambda *args, **kwargs: _passing_runtime_source_contract(),
        raising=False,
    )
    monkeypatch.setattr(
        p2_preflight,
        "build_p2_runtime_environment_contract",
        lambda *args, **kwargs: _passing_runtime_environment_contract(),
        raising=False,
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
    filter_by_split = {
        "train": {
            "sequence_count": 1178,
            "excluded_count": 4,
            "retained_count": 1174,
            "excluded_sequences": list(P2_EXCLUDED_SEQUENCE_NAMES["train"]),
        },
        "validation": {
            "sequence_count": 157,
            "excluded_count": 3,
            "retained_count": 154,
            "excluded_sequences": list(
                P2_EXCLUDED_SEQUENCE_NAMES["validation"]
            ),
        },
        "test": {
            "sequence_count": 157,
            "excluded_count": 3,
            "retained_count": 154,
            "excluded_sequences": list(P2_EXCLUDED_SEQUENCE_NAMES["test"]),
        },
    }
    filter_names_payload = {
        split: value["excluded_sequences"]
        for split, value in filter_by_split.items()
    }
    artifact = {
        "schema_version": p2_preflight.P2_PREFLIGHT_SCHEMA_VERSION,
        "status": "pass",
        "formal_p2_training_authorized": True,
        "official_source_commit": "fb2fe42eb8f1e926567c48eea9acb874e608ee10",
        "local_source_commit": LOCAL_SOURCE_COMMIT,
        "source_tree_contract": _passing_source_tree_contract(),
        "runtime_source_contract": _passing_runtime_source_contract(),
        "runtime_environment_contract": (
            _passing_runtime_environment_contract()
        ),
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
            "expected": _FORMAL_DATA_ROOT_CONTRACT["expected"],
            "observed": _FORMAL_DATA_ROOT_CONTRACT["expected"],
            "expected_resolved": _FORMAL_DATA_ROOT_CONTRACT[
                "expected_resolved"
            ],
            "observed_resolved": _FORMAL_DATA_ROOT_CONTRACT[
                "expected_resolved"
            ],
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
            "excluded_unsupervised_sequences": filter_names_payload,
            "filtered_sequence_counts": {
                "train": 1174,
                "validation": 154,
            },
        },
        "unsupervised_sequence_filter": {
            "schema_version": 1,
            "status": "pass",
            "enabled": True,
            "source": "real_npy",
            "taxonomy_label_ids": list(NYU40_INSTANCE_IDS),
            "by_split": filter_by_split,
            "sequence_name_sha256": p2_preflight.P2_RIO_SEQUENCE_FILTER_SHA256,
        },
        "input_manifest": copy.deepcopy(FORMAL_INPUT_MANIFEST),
        "mix_instantiation": {
            "attempted": True,
            "status": "pass",
            "implementation": "datasets.multi_dataset.MultiDataset",
            "dataset_names": ["rio", "scannet"],
            "dataset_sizes": [1174, 1199],
            "weights": [1.0, 0.8],
            "temporal_windows": [2, 1],
            "sampler": "WeightedRandomSampler",
            "sampler_num_samples": 2112,
            "epoch_sample_multiple": 32,
        },
        "errors": [],
        "config_contract": {
            "schema_version": 1,
            "status": "pass",
            "errors": [],
            "expected_semantic_sha256": (
                p2_preflight.P2_TRAINING_SEMANTIC_SHA256
            ),
            "observed_semantic_sha256": (
                p2_preflight.P2_TRAINING_SEMANTIC_SHA256
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


def _optimizer_scheduler_checkpoint_states(
    cfg,
    *,
    global_step: int,
) -> tuple[dict, dict]:
    parameters = [
        torch.nn.Parameter(torch.ones(1)),
        torch.nn.Parameter(torch.ones(2)),
    ]
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=parameters)
    scheduler_cfg = cfg.scheduler.scheduler.copy()
    scheduler_cfg.total_steps = 29_700
    scheduler = hydra.utils.instantiate(scheduler_cfg, optimizer=optimizer)
    for _ in range(global_step):
        for parameter in parameters:
            parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
    return optimizer.state_dict(), scheduler.state_dict()


def _model_checkpoint_callbacks(cfg) -> list[ModelCheckpoint]:
    callbacks = [
        hydra.utils.instantiate(callback_cfg)
        for callback_cfg in cfg.callbacks
        if callback_cfg.get("_target_")
        == "pytorch_lightning.callbacks.ModelCheckpoint"
    ]
    assert len(callbacks) == 3
    assert all(isinstance(callback, ModelCheckpoint) for callback in callbacks)
    return callbacks


def _isolate_formal_callback_paths(cfg) -> None:
    save_dir = Path(str(cfg.general.save_dir)).expanduser()
    if not save_dir.is_absolute():
        return
    for callback_index, callback_cfg in enumerate(cfg.callbacks):
        if callback_cfg.get("_target_") != (
            "pytorch_lightning.callbacks.ModelCheckpoint"
        ):
            continue
        dirpath = callback_cfg.get("dirpath")
        if not isinstance(dirpath, str) or Path(dirpath).is_absolute():
            continue
        with open_dict(callback_cfg):
            callback_cfg.dirpath = str(save_dir / f"callback-{callback_index}")


def _fresh_model_checkpoint_callback_states(cfg) -> dict:
    callbacks = _model_checkpoint_callbacks(cfg)
    return {callback.state_key: callback.state_dict() for callback in callbacks}


def _model_checkpoint_callback_states(cfg, *, epoch: int) -> dict:
    _isolate_formal_callback_paths(cfg)
    callbacks = _model_checkpoint_callbacks(cfg)
    completed_epochs = epoch + 1
    for callback in callbacks:
        interval = callback._every_n_epochs
        if interval is None or completed_epochs < interval:
            continue
        checkpoint_path = str(
            Path(callback.dirpath) / f"fixture-epoch={epoch:03d}.ckpt"
        )
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        Path(checkpoint_path).touch()
        callback.best_model_path = checkpoint_path
        if callback.monitor is not None:
            score = torch.tensor(0.5)
            callback.best_model_score = score
            callback.current_score = score.clone()
            callback.best_k_models = {checkpoint_path: score.clone()}
            callback.kth_best_model_path = checkpoint_path
            callback.kth_value = score.clone()
            last_path = Path(callback.dirpath) / "last.ckpt"
            last_path.parent.mkdir(parents=True, exist_ok=True)
            last_path.touch()
            callback.last_model_path = str(last_path)
    return {callback.state_key: callback.state_dict() for callback in callbacks}


def _optimizer_parameter_contract(optimizer_state: dict, state_dict: dict) -> dict:
    parameter_ids = optimizer_state["param_groups"][0]["params"]
    parameter_names = ["model.weight", "model.bias"]
    assert len(parameter_ids) == len(parameter_names)
    model_state = {
        name: {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
        }
        for name, tensor in state_dict.items()
    }
    return {
        "schema_version": 1,
        "state_dict": model_state,
        "state_dict_schema_sha256": _model_state_schema_sha256(model_state),
        "param_groups": [
            list(group["params"]) for group in optimizer_state["param_groups"]
        ],
        "parameters": {
            parameter_id: {
                "name": name,
                "shape": list(state_dict[name].shape),
                "dtype": str(state_dict[name].dtype),
            }
            for parameter_id, name in zip(parameter_ids, parameter_names)
        },
        "trainable_parameters": FORMAL_TEST_TRAINABLE_PARAMETER_SCHEMA.copy(),
        "trainable_parameter_schema_sha256": (
            FORMAL_TEST_TRAINABLE_PARAMETER_SCHEMA_SHA256
        ),
    }


def _formal_loop_states(*, epoch: int, global_step: int) -> dict:
    epoch_total = {
        "ready": epoch + 1,
        "completed": epoch + 1,
        "started": epoch + 1,
        "processed": epoch + 1,
    }
    optimizer_steps = {"ready": global_step, "completed": global_step}
    current_optimizer_steps = {
        "ready": 0,
        "completed": 0,
    }
    optimizer_zero_grads = {
        "ready": global_step,
        "completed": global_step,
        "started": global_step,
    }
    current_optimizer_zero_grads = {
        "ready": 0,
        "completed": 0,
        "started": 0,
    }
    batches = (epoch + 1) * 528
    batch_total = {
        "ready": batches,
        "completed": batches,
        "started": batches,
        "processed": batches,
    }
    idle_batch_progress = {
        "total": {key: 0 for key in batch_total},
        "current": {key: 0 for key in batch_total},
    }
    return {
        "fit_loop": {
            "state_dict": {},
            "epoch_progress": {"total": epoch_total, "current": epoch_total.copy()},
            "epoch_loop.state_dict": {"_batches_that_stepped": global_step},
            "epoch_loop.batch_progress": {
                "total": batch_total,
                "current": {field: 0 for field in batch_total},
                "is_last_batch": False,
            },
            "epoch_loop.scheduler_progress": {
                "total": optimizer_steps.copy(),
                "current": current_optimizer_steps.copy(),
            },
            "epoch_loop.automatic_optimization.state_dict": {},
            "epoch_loop.automatic_optimization.optim_progress": {
                "optimizer": {
                    "step": {
                        "total": optimizer_steps.copy(),
                        "current": current_optimizer_steps.copy(),
                    },
                    "zero_grad": {
                        "total": optimizer_zero_grads,
                        "current": current_optimizer_zero_grads,
                    },
                }
            },
            "epoch_loop.val_loop.state_dict": {},
            "epoch_loop.val_loop.batch_progress": {
                **copy.deepcopy(idle_batch_progress),
                "is_last_batch": False,
            },
        },
        "validate_loop": {
            "state_dict": {},
            "batch_progress": {**copy.deepcopy(idle_batch_progress), "is_last_batch": False},
        },
        "test_loop": {
            "state_dict": {},
            "batch_progress": {**copy.deepcopy(idle_batch_progress), "is_last_batch": False},
        },
        "predict_loop": {
            "state_dict": {},
            "batch_progress": copy.deepcopy(idle_batch_progress),
        },
    }


def _formal_resume_payload(cfg, *, epoch: int = 0) -> dict:
    global_step = (epoch + 1) * 66
    optimizer_state, scheduler_state = _optimizer_scheduler_checkpoint_states(
        cfg,
        global_step=global_step,
    )
    state_dict = {
        "model.weight": torch.ones(1),
        "model.bias": torch.ones(2),
        "model.running_mean": torch.zeros(3),
    }
    payload = {
        "pytorch-lightning_version": "2.6.5",
        "state_dict": state_dict,
        "optimizer_states": [optimizer_state],
        "lr_schedulers": [scheduler_state],
        "loops": _formal_loop_states(epoch=epoch, global_step=global_step),
        "callbacks": _model_checkpoint_callback_states(cfg, epoch=epoch),
        "epoch": epoch,
        "global_step": global_step,
        "hyper_parameters": cfg,
        "p2_train_sampler_generator": _sampler_generator_checkpoint_payload(),
    }
    payload["p2_optimizer_parameter_contract"] = _optimizer_parameter_contract(
        optimizer_state,
        state_dict,
    )
    return payload


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


def test_preflight_without_local_source_binding_fails_closed(tmp_path: Path) -> None:
    cfg = _compose(P2_CONFIG_NAME)
    artifact_path = tmp_path / "scannet_preflight.json"
    artifact = _formal_artifact(cfg)
    artifact.pop("local_source_commit")
    artifact.pop("source_tree_contract")
    artifact["authorization"]["artifact_payload_sha256"] = _artifact_sha256(
        artifact
    )
    _write_artifact(artifact_path, artifact)

    with pytest.raises(RuntimeError, match="source_tree_contract"):
        training_entrypoint.require_p2_preflight_authorization(
            cfg,
            artifact_path=artifact_path,
        )


def test_current_non_artifact_dirty_source_tree_fails_closed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cfg = _compose(P2_CONFIG_NAME)
    artifact_path = tmp_path / "scannet_preflight.json"
    _write_artifact(artifact_path, _formal_artifact(cfg))
    failed_contract = _passing_source_tree_contract()
    failed_contract.update(
        {
            "status": "fail",
            "dirty_paths": ["trainer/trainer.py"],
            "disallowed_dirty_paths": ["trainer/trainer.py"],
            "errors": ["non_artifact_worktree_changes"],
        }
    )
    monkeypatch.setattr(
        p2_preflight,
        "build_p2_source_tree_contract",
        lambda *args, **kwargs: copy.deepcopy(failed_contract),
        raising=False,
    )

    with pytest.raises(RuntimeError, match="current source_tree_contract"):
        training_entrypoint.require_p2_preflight_authorization(
            cfg,
            artifact_path=artifact_path,
        )


def test_preflight_without_runtime_source_binding_fails_closed(
    tmp_path: Path,
) -> None:
    cfg = _compose(P2_CONFIG_NAME)
    artifact_path = tmp_path / "scannet_preflight.json"
    artifact = _formal_artifact(cfg)
    artifact.pop("runtime_source_contract")
    artifact["authorization"]["artifact_payload_sha256"] = _artifact_sha256(
        artifact
    )
    _write_artifact(artifact_path, artifact)

    with pytest.raises(RuntimeError, match="runtime_source_contract"):
        training_entrypoint.require_p2_preflight_authorization(
            cfg,
            artifact_path=artifact_path,
        )


def test_current_nested_runtime_source_drift_fails_closed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cfg = _compose(P2_CONFIG_NAME)
    artifact_path = tmp_path / "scannet_preflight.json"
    _write_artifact(artifact_path, _formal_artifact(cfg))
    failed_contract = _passing_runtime_source_contract()
    failed_contract["status"] = "fail"
    failed_contract["errors"] = ["detectron2:native_extension_mismatch"]
    failed_contract["repositories"]["detectron2"]["status"] = "fail"
    failed_contract["repositories"]["detectron2"]["errors"] = [
        "native_extension_mismatch"
    ]
    monkeypatch.setattr(
        p2_preflight,
        "build_p2_runtime_source_contract",
        lambda *args, **kwargs: copy.deepcopy(failed_contract),
    )

    with pytest.raises(RuntimeError, match="current runtime_source_contract"):
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


def test_training_inputs_are_rechecked_after_the_authorization_scan(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cfg = _compose(P2_CONFIG_NAME)
    artifact_path = tmp_path / "scannet_preflight.json"
    _write_artifact(artifact_path, _formal_artifact(cfg))
    changed = copy.deepcopy(FORMAL_INPUT_MANIFEST)
    changed["rio"]["content_sha256"] = "9" * 64
    manifests = iter([copy.deepcopy(FORMAL_INPUT_MANIFEST), changed])
    monkeypatch.setattr(
        p2_preflight,
        "build_p2_input_manifest",
        lambda *args, **kwargs: next(manifests),
    )

    with pytest.raises(RuntimeError, match="training inputs changed"):
        training_entrypoint.require_p2_preflight_authorization(
            cfg,
            artifact_path=artifact_path,
        )


def test_official_splits_are_rechecked_after_the_authorization_scan(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cfg = _compose(P2_CONFIG_NAME)
    artifact_path = tmp_path / "scannet_preflight.json"
    artifact = _formal_artifact(cfg)
    _write_artifact(artifact_path, artifact)
    matching = copy.deepcopy(artifact["official_split_identity"])
    changed = copy.deepcopy(matching)
    changed["files"]["validation"]["observed_sha256"] = "9" * 64
    identities = iter([matching, changed])
    monkeypatch.setattr(
        p2_preflight,
        "build_scannet_official_split_identity",
        lambda *args, **kwargs: next(identities),
    )

    with pytest.raises(RuntimeError, match="official split changed"):
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
    artifact["mix_instantiation"]["dataset_sizes"] = [1, 1199]
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


def test_formal_checkpoint_validation_requires_the_current_config(
    tmp_path: Path,
) -> None:
    cfg = _compose(P2_CONFIG_NAME)
    cfg.general.save_dir = str(tmp_path / "verified-snapshots")
    payload = _formal_resume_payload(cfg)

    assert training_entrypoint._resume_checkpoint_validation_error(
        payload,
        formal_p2=True,
    ) == "formal P2 checkpoint validation requires current config"


def test_formal_p2_resume_rejects_learning_rate_monitor_only_callback_state(
    tmp_path: Path,
) -> None:
    cfg = _compose(P2_CONFIG_NAME)
    cfg.general.save_dir = str(tmp_path / "verified-snapshots")
    checkpoint = tmp_path / "learning-rate-monitor-only.ckpt"
    payload = _formal_resume_payload(cfg)
    payload["callbacks"] = {"LearningRateMonitor": {}}
    torch.save(payload, checkpoint)

    with pytest.raises(RuntimeError, match="ModelCheckpoint callback state"):
        training_entrypoint.require_p2_resume_checkpoint(cfg, checkpoint)


@pytest.mark.parametrize("callback_index", range(3))
def test_formal_p2_resume_requires_every_configured_model_checkpoint_state(
    tmp_path: Path,
    callback_index: int,
) -> None:
    cfg = _compose(P2_CONFIG_NAME)
    cfg.general.save_dir = str(tmp_path / "verified-snapshots")
    checkpoint = tmp_path / f"missing-callback-{callback_index}.ckpt"
    payload = _formal_resume_payload(cfg)
    state_key = tuple(payload["callbacks"])[callback_index]
    payload["callbacks"].pop(state_key)
    torch.save(payload, checkpoint)

    with pytest.raises(RuntimeError, match="ModelCheckpoint callback state"):
        training_entrypoint.require_p2_resume_checkpoint(cfg, checkpoint)


def test_formal_p2_resume_rejects_malformed_model_checkpoint_state(
    tmp_path: Path,
) -> None:
    cfg = _compose(P2_CONFIG_NAME)
    cfg.general.save_dir = str(tmp_path / "verified-snapshots")
    checkpoint = tmp_path / "malformed-callback-state.ckpt"
    payload = _formal_resume_payload(cfg)
    state_key = next(iter(payload["callbacks"]))
    payload["callbacks"][state_key] = {"best_model_path": ""}
    torch.save(payload, checkpoint)

    with pytest.raises(RuntimeError, match="ModelCheckpoint callback state"):
        training_entrypoint.require_p2_resume_checkpoint(cfg, checkpoint)


def test_formal_p2_resume_rejects_fresh_state_for_triggered_monitor_callback(
    tmp_path: Path,
) -> None:
    cfg = _compose(P2_CONFIG_NAME)
    cfg.general.save_dir = str(tmp_path / "verified-snapshots")
    checkpoint = tmp_path / "fresh-monitor-callback-state.ckpt"
    payload = _formal_resume_payload(cfg)
    payload["callbacks"] = _fresh_model_checkpoint_callback_states(cfg)
    torch.save(payload, checkpoint)

    with pytest.raises(RuntimeError, match="ModelCheckpoint callback history"):
        training_entrypoint.require_p2_resume_checkpoint(cfg, checkpoint)


def test_formal_p2_resume_rejects_validation_end_monitored_checkpoint_contract(
    tmp_path: Path,
) -> None:
    cfg = _compose(P2_CONFIG_NAME)
    cfg.general.save_dir = str(tmp_path / "verified-snapshots")
    with open_dict(cfg.callbacks[0]):
        cfg.callbacks[0].save_on_train_epoch_end = False
    checkpoint = tmp_path / "validation-end-monitored-callback.ckpt"
    torch.save(_formal_resume_payload(cfg), checkpoint)

    with pytest.raises(RuntimeError, match="train_epoch_end"):
        training_entrypoint.require_p2_resume_checkpoint(cfg, checkpoint)


@pytest.mark.parametrize(
    ("epoch", "callback_index"),
    [(25, 1)],
)
def test_formal_p2_resume_rejects_fresh_state_after_prior_periodic_trigger(
    tmp_path: Path,
    epoch: int,
    callback_index: int,
) -> None:
    cfg = _compose(P2_CONFIG_NAME)
    cfg.general.save_dir = str(tmp_path / "verified-snapshots")
    checkpoint = tmp_path / f"fresh-periodic-callback-{epoch}.ckpt"
    payload = _formal_resume_payload(cfg, epoch=epoch)
    callbacks = _model_checkpoint_callbacks(cfg)
    callback = callbacks[callback_index]
    payload["callbacks"][callback.state_key] = callback.state_dict()
    torch.save(payload, checkpoint)

    with pytest.raises(RuntimeError, match="ModelCheckpoint callback history"):
        training_entrypoint.require_p2_resume_checkpoint(cfg, checkpoint)


@pytest.mark.parametrize(
    ("epoch", "callback_index"),
    [(24, 1), (449, 2)],
)
def test_formal_p2_resume_accepts_fresh_periodic_state_at_first_trigger_boundary(
    tmp_path: Path,
    epoch: int,
    callback_index: int,
) -> None:
    cfg = _compose(P2_CONFIG_NAME)
    cfg.general.save_dir = str(tmp_path / "verified-snapshots")
    checkpoint = tmp_path / f"pre-periodic-callback-{epoch}.ckpt"
    payload = _formal_resume_payload(cfg, epoch=epoch)
    callback = _model_checkpoint_callbacks(cfg)[callback_index]
    payload["callbacks"][callback.state_key] = callback.state_dict()
    torch.save(payload, checkpoint)

    training_entrypoint.require_p2_resume_checkpoint(cfg, checkpoint)


def test_formal_p2_resume_accepts_first_top_checkpoint_before_save_last(
    tmp_path: Path,
) -> None:
    cfg = _compose(P2_CONFIG_NAME)
    cfg.general.save_dir = str(tmp_path / "verified-snapshots")
    checkpoint = tmp_path / "first-top-before-last.ckpt"
    payload = _formal_resume_payload(cfg, epoch=14)
    monitor_state = next(iter(payload["callbacks"].values()))
    monitor_state["last_model_path"] = ""
    torch.save(payload, checkpoint)

    training_entrypoint.require_p2_resume_checkpoint(cfg, checkpoint)


def test_formal_p2_resume_rejects_empty_last_before_first_validation(
    tmp_path: Path,
) -> None:
    cfg = _compose(P2_CONFIG_NAME)
    cfg.general.save_dir = str(tmp_path / "verified-snapshots")
    checkpoint = tmp_path / "empty-last-before-first-validation.ckpt"
    payload = _formal_resume_payload(cfg)
    monitor_state = next(iter(payload["callbacks"].values()))
    monitor_state["last_model_path"] = ""
    torch.save(payload, checkpoint)

    with pytest.raises(RuntimeError, match="ModelCheckpoint callback history"):
        training_entrypoint.require_p2_resume_checkpoint(cfg, checkpoint)


def test_formal_p2_resume_accepts_real_lightning_callback_save_order(
    tmp_path: Path,
) -> None:
    class CallbackTimingModule(LightningModule):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))

        def training_step(self, _batch, _batch_idx):
            return self.weight.square()

        def validation_step(self, _batch, _batch_idx):
            self.log("val_mean_t-AP", torch.tensor(0.5), on_epoch=True)

        def configure_optimizers(self):
            return torch.optim.AdamW(self.parameters(), lr=1e-3)

    cfg = _compose(P2_CONFIG_NAME)
    cfg.general.save_dir = str(tmp_path / "callbacks")
    with open_dict(cfg.callbacks[2]):
        cfg.callbacks[2].dirpath = str(tmp_path / "final")
    callbacks = _model_checkpoint_callbacks(cfg)
    loader = DataLoader(TensorDataset(torch.ones(1)), batch_size=1)
    Trainer(
        max_epochs=15,
        check_val_every_n_epoch=15,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        callbacks=callbacks,
    ).fit(CallbackTimingModule(), loader, loader)

    top_checkpoint = torch.load(
        callbacks[0].best_model_path,
        map_location="cpu",
        weights_only=False,
    )
    monitor_key = callbacks[0].state_key
    assert top_checkpoint["callbacks"][monitor_key]["last_model_path"] == ""

    epoch = 14
    payload = _formal_resume_payload(cfg, epoch=epoch)
    payload["callbacks"] = top_checkpoint["callbacks"]
    checkpoint = tmp_path / f"formal-callback-timing-{epoch}.ckpt"
    torch.save(payload, checkpoint)
    training_entrypoint.require_p2_resume_checkpoint(cfg, checkpoint)


def test_formal_p2_resume_accepts_none_current_score_at_epoch399_periodic_boundary(
    tmp_path: Path,
) -> None:
    cfg = _compose(P2_CONFIG_NAME)
    cfg.general.save_dir = str(tmp_path / "verified-snapshots")
    checkpoint = tmp_path / "periodic-epoch=399.ckpt"
    payload = _formal_resume_payload(cfg, epoch=399)
    monitor_state = next(
        state
        for state in payload["callbacks"].values()
        if state["monitor"] == "val_mean_t-AP"
    )
    monitor_state["current_score"] = None
    torch.save(payload, checkpoint)

    verified = training_entrypoint.require_p2_resume_checkpoint(cfg, checkpoint)

    assert Path(verified).is_file()


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("best_model_path", ""),
        ("best_model_score", torch.tensor(float("nan"))),
        ("current_score", None),
        ("best_k_models", {}),
    ],
)
def test_formal_p2_resume_rejects_incomplete_monitor_callback_history(
    tmp_path: Path,
    field: str,
    invalid_value,
) -> None:
    cfg = _compose(P2_CONFIG_NAME)
    cfg.general.save_dir = str(tmp_path / "verified-snapshots")
    checkpoint = tmp_path / f"invalid-monitor-callback-{field}.ckpt"
    payload = _formal_resume_payload(cfg)
    monitor_state = next(iter(payload["callbacks"].values()))
    monitor_state[field] = invalid_value
    torch.save(payload, checkpoint)

    with pytest.raises(RuntimeError, match="ModelCheckpoint callback history"):
        training_entrypoint.require_p2_resume_checkpoint(cfg, checkpoint)


def test_formal_p2_resume_rejects_missing_last_history_after_first_epoch(
    tmp_path: Path,
) -> None:
    cfg = _compose(P2_CONFIG_NAME)
    cfg.general.save_dir = str(tmp_path / "verified-snapshots")
    checkpoint = tmp_path / "missing-last-after-first-epoch.ckpt"
    payload = _formal_resume_payload(cfg, epoch=1)
    monitor_state = next(iter(payload["callbacks"].values()))
    monitor_state["last_model_path"] = ""
    torch.save(payload, checkpoint)

    with pytest.raises(RuntimeError, match="ModelCheckpoint callback history"):
        training_entrypoint.require_p2_resume_checkpoint(cfg, checkpoint)


def test_formal_p2_resume_rejects_dangling_last_callback_reference(
    tmp_path: Path,
) -> None:
    cfg = _compose(P2_CONFIG_NAME)
    cfg.general.save_dir = str(tmp_path / "verified-snapshots")
    checkpoint = tmp_path / "dangling-last-callback-reference.ckpt"
    payload = _formal_resume_payload(cfg, epoch=1)
    monitor_state = next(
        state
        for state in payload["callbacks"].values()
        if state["monitor"] is not None
    )
    Path(monitor_state["last_model_path"]).unlink()
    torch.save(payload, checkpoint)

    with pytest.raises(RuntimeError, match="callback history references"):
        training_entrypoint.require_p2_resume_checkpoint(cfg, checkpoint)


def test_formal_p2_resume_rejects_inconsistent_monitor_callback_score(
    tmp_path: Path,
) -> None:
    cfg = _compose(P2_CONFIG_NAME)
    cfg.general.save_dir = str(tmp_path / "verified-snapshots")
    checkpoint = tmp_path / "inconsistent-monitor-callback-score.ckpt"
    payload = _formal_resume_payload(cfg)
    monitor_state = next(iter(payload["callbacks"].values()))
    best_path = monitor_state["best_model_path"]
    monitor_state["best_k_models"][best_path] = torch.tensor(-1.0)
    torch.save(payload, checkpoint)

    with pytest.raises(RuntimeError, match="ModelCheckpoint callback history"):
        training_entrypoint.require_p2_resume_checkpoint(cfg, checkpoint)


def test_formal_p2_resume_requires_optimizer_parameter_contract(
    tmp_path: Path,
) -> None:
    cfg = _compose(P2_CONFIG_NAME)
    cfg.general.save_dir = str(tmp_path / "verified-snapshots")
    checkpoint = tmp_path / "missing-optimizer-parameter-contract.ckpt"
    payload = _formal_resume_payload(cfg)
    payload.pop("p2_optimizer_parameter_contract")
    torch.save(payload, checkpoint)

    with pytest.raises(RuntimeError, match="optimizer parameter contract"):
        training_entrypoint.require_p2_resume_checkpoint(cfg, checkpoint)


def test_formal_p2_resume_rejects_missing_nonoptimizer_model_state(
    tmp_path: Path,
) -> None:
    cfg = _compose(P2_CONFIG_NAME)
    cfg.general.save_dir = str(tmp_path / "verified-snapshots")
    checkpoint = tmp_path / "missing-model-buffer.ckpt"
    payload = _formal_resume_payload(cfg)
    payload["state_dict"].pop("model.running_mean")
    torch.save(payload, checkpoint)

    with pytest.raises(RuntimeError, match="model state contract"):
        training_entrypoint.require_p2_resume_checkpoint(cfg, checkpoint)


def test_formal_p2_resume_rejects_model_state_deleted_with_self_contract(
    tmp_path: Path,
) -> None:
    cfg = _compose(P2_CONFIG_NAME)
    cfg.general.save_dir = str(tmp_path / "verified-snapshots")
    checkpoint = tmp_path / "self-consistent-incomplete-model.ckpt"
    payload = _formal_resume_payload(cfg)
    payload["state_dict"].pop("model.running_mean")
    model_contract = payload["p2_optimizer_parameter_contract"]["state_dict"]
    model_contract.pop("model.running_mean")
    payload["p2_optimizer_parameter_contract"]["state_dict_schema_sha256"] = (
        _model_state_schema_sha256(model_contract)
    )
    torch.save(payload, checkpoint)

    with pytest.raises(RuntimeError, match="model state schema"):
        training_entrypoint.require_p2_resume_checkpoint(cfg, checkpoint)


@pytest.mark.parametrize(
    ("mutation", "value"),
    [
        ("shape", torch.zeros(4)),
        ("dtype", torch.zeros(3, dtype=torch.float64)),
        ("extra", torch.ones(1)),
    ],
)
def test_formal_p2_resume_rejects_tampered_model_state_tensor_contract(
    tmp_path: Path,
    mutation: str,
    value: torch.Tensor,
) -> None:
    cfg = _compose(P2_CONFIG_NAME)
    cfg.general.save_dir = str(tmp_path / "verified-snapshots")
    checkpoint = tmp_path / f"tampered-model-state-{mutation}.ckpt"
    payload = _formal_resume_payload(cfg)
    if mutation == "extra":
        payload["state_dict"]["model.unexpected"] = value
    else:
        payload["state_dict"]["model.running_mean"] = value
    torch.save(payload, checkpoint)

    with pytest.raises(RuntimeError, match="model state contract"):
        training_entrypoint.require_p2_resume_checkpoint(cfg, checkpoint)


def test_formal_p2_resume_rejects_adamw_moment_shape_mismatch(
    tmp_path: Path,
) -> None:
    cfg = _compose(P2_CONFIG_NAME)
    cfg.general.save_dir = str(tmp_path / "verified-snapshots")
    checkpoint = tmp_path / "adamw-moment-shape-mismatch.ckpt"
    payload = _formal_resume_payload(cfg)
    payload["optimizer_states"][0]["state"][0]["exp_avg"] = torch.zeros(3)
    payload["optimizer_states"][0]["state"][0]["exp_avg_sq"] = torch.zeros(3)
    torch.save(payload, checkpoint)

    with pytest.raises(RuntimeError, match="AdamW parameter slot moments"):
        training_entrypoint.require_p2_resume_checkpoint(cfg, checkpoint)


def test_formal_p2_resume_rejects_stale_adamw_parameter_step(
    tmp_path: Path,
) -> None:
    cfg = _compose(P2_CONFIG_NAME)
    cfg.general.save_dir = str(tmp_path / "verified-snapshots")
    checkpoint = tmp_path / "stale-adamw-parameter-step.ckpt"
    payload = _formal_resume_payload(cfg)
    step = payload["optimizer_states"][0]["state"][0]["step"]
    payload["optimizer_states"][0]["state"][0]["step"] = step.new_tensor(1)
    torch.save(payload, checkpoint)

    with pytest.raises(RuntimeError, match="AdamW parameter slot step"):
        training_entrypoint.require_p2_resume_checkpoint(cfg, checkpoint)


def test_formal_p2_resume_rejects_shallow_fit_loop_state(tmp_path: Path) -> None:
    cfg = _compose(P2_CONFIG_NAME)
    cfg.general.save_dir = str(tmp_path / "verified-snapshots")
    checkpoint = tmp_path / "shallow-loop-state.ckpt"
    payload = _formal_resume_payload(cfg)
    payload["loops"]["fit_loop"] = {"state": 1}
    torch.save(payload, checkpoint)

    with pytest.raises(RuntimeError, match="fit_loop"):
        training_entrypoint.require_p2_resume_checkpoint(cfg, checkpoint)


@pytest.mark.parametrize(
    ("path", "invalid_value"),
    [
        (("epoch_progress", "total", "ready"), 0),
        (("epoch_progress", "total", "processed"), 2),
        (("epoch_progress", "total", "started"), 0),
        (("epoch_progress", "total", "completed"), 0),
        (("epoch_progress", "current", "processed"), 2),
        (("epoch_loop.state_dict", "_batches_that_stepped"), 64),
        (
            (
                "epoch_loop.automatic_optimization.optim_progress",
                "optimizer",
                "step",
                "total",
                "completed",
            ),
            65,
        ),
        (
            (
                "epoch_loop.automatic_optimization.optim_progress",
                "optimizer",
                "step",
                "current",
                "completed",
            ),
            65,
        ),
        (
            (
                "epoch_loop.automatic_optimization.optim_progress",
                "optimizer",
                "zero_grad",
                "total",
                "started",
            ),
            65,
        ),
        (
            (
                "epoch_loop.automatic_optimization.optim_progress",
                "optimizer",
                "zero_grad",
                "current",
                "ready",
            ),
            65,
        ),
        (("epoch_loop.scheduler_progress", "total", "completed"), 65),
        (("epoch_loop.scheduler_progress", "current", "completed"), 65),
        (("epoch_loop.batch_progress", "total", "ready"), 263),
        (("epoch_loop.batch_progress", "total", "completed"), 263),
        (("epoch_loop.batch_progress", "total", "started"), 263),
        (("epoch_loop.batch_progress", "total", "processed"), 263),
        (("epoch_loop.batch_progress", "current", "processed"), 263),
        (("epoch_loop.batch_progress", "is_last_batch"), True),
    ],
)
def test_formal_p2_resume_rejects_inconsistent_fit_loop_progress(
    tmp_path: Path,
    path: tuple[str, ...],
    invalid_value: int,
) -> None:
    cfg = _compose(P2_CONFIG_NAME)
    cfg.general.save_dir = str(tmp_path / "verified-snapshots")
    checkpoint = tmp_path / f"bad-fit-progress-{path[-1]}-{invalid_value}.ckpt"
    payload = _formal_resume_payload(cfg)
    target = payload["loops"]["fit_loop"]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = invalid_value
    torch.save(payload, checkpoint)

    with pytest.raises(RuntimeError, match="fit_loop"):
        training_entrypoint.require_p2_resume_checkpoint(cfg, checkpoint)


@pytest.mark.parametrize(
    "loop_name",
    ["fit_loop", "validate_loop", "test_loop", "predict_loop"],
)
def test_formal_p2_resume_requires_all_lightning_loop_states(
    tmp_path: Path,
    loop_name: str,
) -> None:
    cfg = _compose(P2_CONFIG_NAME)
    cfg.general.save_dir = str(tmp_path / "verified-snapshots")
    checkpoint = tmp_path / f"missing-{loop_name}.ckpt"
    payload = _formal_resume_payload(cfg)
    payload["loops"].pop(loop_name)
    torch.save(payload, checkpoint)

    with pytest.raises(RuntimeError, match="loop state"):
        training_entrypoint.require_p2_resume_checkpoint(cfg, checkpoint)


def test_formal_p2_resume_rejects_validation_end_epoch_progress(
    tmp_path: Path,
) -> None:
    cfg = _compose(P2_CONFIG_NAME)
    cfg.general.save_dir = str(tmp_path / "verified-snapshots")
    checkpoint = tmp_path / "validation-end-loop-state.ckpt"
    payload = _formal_resume_payload(cfg)
    epoch_progress = payload["loops"]["fit_loop"]["epoch_progress"]
    epoch_progress["total"]["processed"] = 0
    epoch_progress["current"]["processed"] = 0
    payload["loops"]["fit_loop"]["epoch_loop.state_dict"][
        "_batches_that_stepped"
    ] = 65
    torch.save(payload, checkpoint)

    with pytest.raises(RuntimeError, match="fit_loop"):
        training_entrypoint.require_p2_resume_checkpoint(cfg, checkpoint)


def test_formal_p2_resume_rejects_empty_adamw_optimizer_state(
    tmp_path: Path,
) -> None:
    cfg = _compose(P2_CONFIG_NAME)
    cfg.general.save_dir = str(tmp_path / "verified-snapshots")
    checkpoint = tmp_path / "empty-adamw-state.ckpt"
    payload = _formal_resume_payload(cfg)
    payload["optimizer_states"][0]["state"] = {}
    torch.save(payload, checkpoint)

    with pytest.raises(RuntimeError, match="AdamW optimizer state"):
        training_entrypoint.require_p2_resume_checkpoint(cfg, checkpoint)


@pytest.mark.parametrize("missing_field", ["step", "exp_avg", "exp_avg_sq"])
def test_formal_p2_resume_rejects_incomplete_adamw_parameter_slot(
    tmp_path: Path,
    missing_field: str,
) -> None:
    cfg = _compose(P2_CONFIG_NAME)
    cfg.general.save_dir = str(tmp_path / "verified-snapshots")
    checkpoint = tmp_path / f"adamw-slot-without-{missing_field}.ckpt"
    payload = _formal_resume_payload(cfg)
    payload["optimizer_states"][0]["state"][0].pop(missing_field)
    torch.save(payload, checkpoint)

    with pytest.raises(RuntimeError, match="AdamW parameter slot"):
        training_entrypoint.require_p2_resume_checkpoint(cfg, checkpoint)


def test_formal_p2_resume_rejects_adamw_state_for_unknown_parameter(
    tmp_path: Path,
) -> None:
    cfg = _compose(P2_CONFIG_NAME)
    cfg.general.save_dir = str(tmp_path / "verified-snapshots")
    checkpoint = tmp_path / "unknown-adamw-parameter.ckpt"
    payload = _formal_resume_payload(cfg)
    payload["optimizer_states"][0]["state"] = {
        1: payload["optimizer_states"][0]["state"].pop(0)
    }
    torch.save(payload, checkpoint)

    with pytest.raises(RuntimeError, match="AdamW parameter slot"):
        training_entrypoint.require_p2_resume_checkpoint(cfg, checkpoint)


def test_formal_p2_resume_requires_every_adamw_parameter_slot(
    tmp_path: Path,
) -> None:
    cfg = _compose(P2_CONFIG_NAME)
    cfg.general.save_dir = str(tmp_path / "verified-snapshots")
    checkpoint = tmp_path / "missing-adamw-parameter-slot.ckpt"
    payload = _formal_resume_payload(cfg)
    payload["optimizer_states"][0]["state"].pop(1)
    torch.save(payload, checkpoint)

    with pytest.raises(RuntimeError, match="AdamW parameter slot coverage"):
        training_entrypoint.require_p2_resume_checkpoint(cfg, checkpoint)


def test_formal_p2_resume_rejects_reordered_optimizer_parameter_ids(
    tmp_path: Path,
) -> None:
    cfg = _compose(P2_CONFIG_NAME)
    cfg.general.save_dir = str(tmp_path / "verified-snapshots")
    checkpoint = tmp_path / "reordered-optimizer-parameters.ckpt"
    payload = _formal_resume_payload(cfg)
    params = payload["optimizer_states"][0]["param_groups"][0]["params"]
    params.reverse()
    torch.save(payload, checkpoint)

    with pytest.raises(RuntimeError, match="optimizer parameter group order"):
        training_entrypoint.require_p2_resume_checkpoint(cfg, checkpoint)


def test_formal_p2_resume_rejects_same_shape_parameter_name_swap(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cfg = _compose(P2_CONFIG_NAME)
    cfg.general.save_dir = str(tmp_path / "verified-snapshots")
    checkpoint = tmp_path / "same-shape-parameter-name-swap.ckpt"
    payload = _formal_resume_payload(cfg)
    optimizer_state = payload["optimizer_states"][0]
    for parameter_id in (0, 1):
        optimizer_state["state"][parameter_id]["exp_avg"] = torch.ones(1)
        optimizer_state["state"][parameter_id]["exp_avg_sq"] = torch.ones(1)
    payload["state_dict"]["model.bias"] = torch.ones(1)
    contract = payload["p2_optimizer_parameter_contract"]
    contract["state_dict"]["model.bias"] = {
        "shape": [1],
        "dtype": "torch.float32",
    }
    contract["parameters"][1]["shape"] = [1]
    contract["state_dict_schema_sha256"] = _model_state_schema_sha256(
        contract["state_dict"]
    )
    monkeypatch.setattr(
        training_entrypoint,
        "_P2_FORMAL_MODEL_STATE_SCHEMA_SHA256",
        contract["state_dict_schema_sha256"],
    )
    parameter_zero = contract["parameters"][0]
    parameter_one = contract["parameters"][1]
    contract["parameters"][0] = parameter_one
    contract["parameters"][1] = parameter_zero
    torch.save(payload, checkpoint)

    with pytest.raises(RuntimeError, match="trainable parameter schema"):
        training_entrypoint.require_p2_resume_checkpoint(cfg, checkpoint)


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("eps", 1.0),
        ("weight_decay", 999.0),
        ("amsgrad", True),
        ("maximize", True),
    ],
)
def test_formal_p2_resume_rejects_tampered_adamw_parameter_group(
    tmp_path: Path,
    field: str,
    invalid_value,
) -> None:
    cfg = _compose(P2_CONFIG_NAME)
    cfg.general.save_dir = str(tmp_path / "verified-snapshots")
    checkpoint = tmp_path / f"tampered-adamw-{field}.ckpt"
    payload = _formal_resume_payload(cfg)
    payload["optimizer_states"][0]["param_groups"][0][field] = invalid_value
    torch.save(payload, checkpoint)

    with pytest.raises(RuntimeError, match="AdamW parameter group"):
        training_entrypoint.require_p2_resume_checkpoint(cfg, checkpoint)


def test_formal_p2_resume_rejects_missing_adamw_default_group_field(
    tmp_path: Path,
) -> None:
    cfg = _compose(P2_CONFIG_NAME)
    cfg.general.save_dir = str(tmp_path / "verified-snapshots")
    checkpoint = tmp_path / "adamw-group-without-foreach.ckpt"
    payload = _formal_resume_payload(cfg)
    payload["optimizer_states"][0]["param_groups"][0].pop("foreach")
    torch.save(payload, checkpoint)

    with pytest.raises(RuntimeError, match="AdamW parameter group"):
        training_entrypoint.require_p2_resume_checkpoint(cfg, checkpoint)


def test_formal_p2_resume_rejects_wrong_current_onecycle_learning_rate(
    tmp_path: Path,
) -> None:
    cfg = _compose(P2_CONFIG_NAME)
    cfg.general.save_dir = str(tmp_path / "verified-snapshots")
    checkpoint = tmp_path / "tampered-current-lr.ckpt"
    payload = _formal_resume_payload(cfg)
    payload["optimizer_states"][0]["param_groups"][0]["lr"] = 123.0
    payload["lr_schedulers"][0]["_last_lr"] = [123.0]
    torch.save(payload, checkpoint)

    with pytest.raises(RuntimeError, match="OneCycleLR current learning rate"):
        training_entrypoint.require_p2_resume_checkpoint(cfg, checkpoint)


def test_formal_p2_resume_rejects_wrong_current_onecycle_momentum(
    tmp_path: Path,
) -> None:
    cfg = _compose(P2_CONFIG_NAME)
    cfg.general.save_dir = str(tmp_path / "verified-snapshots")
    checkpoint = tmp_path / "tampered-current-momentum.ckpt"
    payload = _formal_resume_payload(cfg)
    group = payload["optimizer_states"][0]["param_groups"][0]
    group["betas"] = (0.1, group["betas"][1])
    torch.save(payload, checkpoint)

    with pytest.raises(RuntimeError, match="OneCycleLR current momentum"):
        training_entrypoint.require_p2_resume_checkpoint(cfg, checkpoint)


def test_formal_p2_optimizer_excludes_frozen_parameters() -> None:
    cfg = _compose(P2_CONFIG_NAME)
    cfg.scheduler.scheduler.total_steps = 2
    trainable = torch.nn.Parameter(torch.ones(1), requires_grad=True)
    frozen = torch.nn.Parameter(torch.ones(1), requires_grad=False)

    class OptimizerOwner:
        config = cfg

        @staticmethod
        def parameters():
            return iter((trainable, frozen))

    optimizers, _ = training_entrypoint.InstanceSegmentation.configure_optimizers(
        OptimizerOwner()
    )

    assert optimizers[0].param_groups[0]["params"] == [trainable]


def test_nonformal_optimizer_keeps_frozen_parameter_compatibility() -> None:
    cfg = _compose(P2_CONFIG_NAME)
    cfg.general.p2_fail_closed_runtime = False
    cfg.scheduler.scheduler.total_steps = 2
    trainable = torch.nn.Parameter(torch.ones(1), requires_grad=True)
    frozen = torch.nn.Parameter(torch.ones(1), requires_grad=False)

    class OptimizerOwner:
        config = cfg

        @staticmethod
        def parameters():
            return iter((trainable, frozen))

    optimizers, _ = training_entrypoint.InstanceSegmentation.configure_optimizers(
        OptimizerOwner()
    )

    params = optimizers[0].param_groups[0]["params"]
    assert len(params) == 2
    assert params[0] is trainable
    assert params[1] is frozen


def test_formal_p2_resume_rejects_arbitrary_nonempty_scheduler_state(
    tmp_path: Path,
) -> None:
    cfg = _compose(P2_CONFIG_NAME)
    cfg.general.save_dir = str(tmp_path / "verified-snapshots")
    checkpoint = tmp_path / "fake-scheduler-state.ckpt"
    payload = _formal_resume_payload(cfg)
    payload["lr_schedulers"] = [{"last_epoch": payload["global_step"]}]
    torch.save(payload, checkpoint)

    with pytest.raises(RuntimeError, match="OneCycleLR scheduler state"):
        training_entrypoint.require_p2_resume_checkpoint(cfg, checkpoint)


def test_formal_p2_resume_requires_scheduler_step_to_match_global_step(
    tmp_path: Path,
) -> None:
    cfg = _compose(P2_CONFIG_NAME)
    cfg.general.save_dir = str(tmp_path / "verified-snapshots")
    checkpoint = tmp_path / "scheduler-step-mismatch.ckpt"
    payload = _formal_resume_payload(cfg)
    payload["lr_schedulers"][0]["last_epoch"] = payload["global_step"] - 1
    torch.save(payload, checkpoint)

    with pytest.raises(RuntimeError, match="last_epoch.*global_step"):
        training_entrypoint.require_p2_resume_checkpoint(cfg, checkpoint)


def test_formal_p2_resume_requires_the_planned_onecycle_total_steps(
    tmp_path: Path,
) -> None:
    cfg = _compose(P2_CONFIG_NAME)
    cfg.general.save_dir = str(tmp_path / "verified-snapshots")
    checkpoint = tmp_path / "scheduler-total-steps-mismatch.ckpt"
    payload = _formal_resume_payload(cfg)
    payload["lr_schedulers"][0]["total_steps"] = 10
    torch.save(payload, checkpoint)

    with pytest.raises(RuntimeError, match="OneCycleLR total_steps"):
        training_entrypoint.require_p2_resume_checkpoint(cfg, checkpoint)


def test_formal_p2_resume_requires_a_completed_epoch_boundary(
    tmp_path: Path,
) -> None:
    cfg = _compose(P2_CONFIG_NAME)
    cfg.general.save_dir = str(tmp_path / "verified-snapshots")
    checkpoint = tmp_path / "mid-epoch-sampler-state.ckpt"
    payload = _formal_resume_payload(cfg)
    payload["epoch"] = 400
    torch.save(payload, checkpoint)

    with pytest.raises(RuntimeError, match="completed epoch boundary"):
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


def test_csv_logger_does_not_enter_wandb_sweep_handling(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cfg = _compose(P2_CONFIG_NAME)
    with open_dict(cfg):
        cfg.general.save_dir = str(tmp_path)
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
    expected_cfg = cfg
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
        lambda _save_dir, *, formal_p2=False, cfg=None: (
            str(selected)
            if formal_p2 and cfg is expected_cfg
            else pytest.fail("formal selector did not receive current config")
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
    "flag_name",
    ["p2_weighted_objective", "p2_fail_closed_runtime"],
)
def test_formal_p2_runtime_flag_cannot_bypass_gate_without_marker(
    flag_name: str,
    monkeypatch,
) -> None:
    cfg = _compose("config_base_instance_segmentation")
    assert "p2_preflight" not in cfg
    with open_dict(cfg.general):
        setattr(cfg.general, flag_name, True)

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
