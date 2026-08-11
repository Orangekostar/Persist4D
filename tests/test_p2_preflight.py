import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from scripts import audit_p2_reproduction as audit
from utils import p2_preflight

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "audit_p2_reproduction.py"
OFFICIAL_SPLIT_COUNTS = {"train": 1201, "validation": 312, "test": 100}
SCANNET_OFFICIAL_COMMIT = "3830fce7f8b2e48ef047ef7fd76ea5f62903f51c"
OFFICIAL_SPLIT_SHA256 = {
    "train": "96acca299b7855f02824c496b19077904d80996e7ced1bb9f0dac98f7dd4d0c8",
    "validation": "d75d4971c3fa7128c643695840e279042c212ef904fe933bd00cf9918c61b083",
    "test": "0214c6a3b1ee516ad653393b0321e7c0394c7662a4b3702eac1ddd7fbc00f7e0",
}
REQUIRED_ARTIFACTS = {
    "config_audit.md",
    "environment_manifest.json",
    "reproduction_target.yaml",
    "official_vs_repro_config_diff.json",
    "scannet_preflight.json",
    "BLOCKED_MISSING_SCANNET.md",
}
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


def _run_audit(
    *,
    raw_dir: Path,
    processed_dir: Path,
    output_dir: Path,
    split_dir: Path | None = None,
    test_segments_dir: Path | None = None,
    expected_counts: tuple[int, int, int] | None = None,
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(SCRIPT),
        "--raw-scannet-dir",
        str(raw_dir),
        "--processed-scannet-dir",
        str(processed_dir),
        "--output-dir",
        str(output_dir),
    ]
    if split_dir is not None:
        command.extend(["--split-dir", str(split_dir)])
    if test_segments_dir is not None:
        command.extend(["--test-segments-dir", str(test_segments_dir)])
    if expected_counts is not None:
        command.extend(
            [
                "--expected-train",
                str(expected_counts[0]),
                "--expected-validation",
                str(expected_counts[1]),
                "--expected-test",
                str(expected_counts[2]),
            ]
        )
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_artifacts_private(output_dir: Path) -> None:
    artifact_text = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in sorted(output_dir.iterdir())
        if path.is_file()
    )
    linux_home_marker = "/" + "ho" + "me" + "/"
    macos_home_marker = "/" + "Us" + "ers" + "/"
    assert linux_home_marker not in artifact_text
    assert macos_home_marker not in artifact_text
    assert not re.search(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", artifact_text)
    assert "GPU" + "-" not in artifact_text

    references = re.findall(
        r'(?:(?:path|reference|source|config)_ref)["\']?\s*[:=]\s*["\']([^"\'\n]+)',
        artifact_text,
    )
    assert all(
        reference.startswith(("repo:", "external:", "local_cache:"))
        for reference in references
    )


def _write_yaml(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def _make_complete_scannet_fixture(tmp_path: Path) -> dict[str, Path]:
    raw_dir = tmp_path / "private-input" / "scannet"
    processed_dir = tmp_path / "private-output" / "scannet"
    split_dir = tmp_path / "private-metadata" / "Benchmark"
    test_segments_dir = tmp_path / "private-output" / "test-segments"
    output_dir = tmp_path / "artifacts"
    scenes = {
        "train": "scene0000_00",
        "validation": "scene0001_00",
        "test": "scene0002_00",
    }

    split_dir.mkdir(parents=True)
    (split_dir / "scannetv2_train.txt").write_text(
        scenes["train"] + "\n", encoding="utf-8"
    )
    (split_dir / "scannetv2_val.txt").write_text(
        scenes["validation"] + "\n", encoding="utf-8"
    )
    (split_dir / "scannetv2_test.txt").write_text(
        scenes["test"] + "\n", encoding="utf-8"
    )
    raw_dir.mkdir(parents=True)
    (raw_dir / "scannetv2-labels.combined.tsv").write_text(
        "id\traw_category\tnyu40id\tnyu40class\n", encoding="utf-8"
    )

    for split, scene in scenes.items():
        parent = raw_dir / ("scans_test" if split == "test" else "scans") / scene
        parent.mkdir(parents=True)
        (parent / f"{scene}_vh_clean_2.ply").touch()
        if split != "test":
            (parent / f"{scene}_vh_clean_2.labels.ply").touch()
            (parent / f"{scene}_vh_clean_2.0.010000.segs.json").write_text(
                '{"segIndices": []}\n', encoding="utf-8"
            )
            (parent / f"{scene}.aggregation.json").write_text(
                '{"segGroups": []}\n', encoding="utf-8"
            )
            (parent / f"{scene}.txt").write_text(
                "sceneType = test room\n", encoding="utf-8"
            )
        else:
            test_segments_dir.mkdir(parents=True)
            (test_segments_dir / f"{scene}_vh_clean_2.0.010000.segs.json").write_text(
                '{"segIndices": []}\n', encoding="utf-8"
            )

    labels = {
        class_id: {
            "name": "wall" if class_id == 1 else "floor" if class_id == 2 else label,
            "validation": True,
            "color": [class_id, class_id, class_id],
        }
        for class_id, label in zip(
            [1, 2, *NYU40_INSTANCE_IDS],
            ["wall", "floor", *NYU40_INSTANCE_LABELS],
        )
    }
    _write_yaml(processed_dir / "label_database.yaml", labels)
    _write_yaml(
        processed_dir / "color_mean_std.yaml",
        {"mean": [0.5, 0.5, 0.5], "std": [0.25, 0.25, 0.25]},
    )
    _write_yaml(
        processed_dir / "scannet.yaml",
        {
            "name": "scannet",
            "class_labels": NYU40_INSTANCE_LABELS,
            "valid_class_ids": NYU40_INSTANCE_IDS,
            "aux": "changes",
            "aux_labels": [
                "static",
                "rigid",
                "nonrigid",
                "ambiguities",
                "added",
                "removed",
            ],
            "valid_aux_ids": [0, 1, 2, 3, 4, 5],
        },
    )

    database_names = {
        "train": "train_database.yaml",
        "validation": "validation_database.yaml",
        "test": "test_database.yaml",
    }
    for split, scene in scenes.items():
        numeric_name = scene.removeprefix("scene")
        npy_path = processed_dir / split / f"{numeric_name}.npy"
        npy_path.parent.mkdir(parents=True, exist_ok=True)
        column_count = 10 if split == "test" else 12
        points = np.zeros((1, column_count), dtype=np.float32)
        points[0, :3] = [0.1, 0.2, 0.3]
        points[0, 3:6] = [10.0, 20.0, 30.0]
        points[0, 9] = 0
        if split != "test":
            points[0, 10] = 3
            points[0, 11] = 0
        np.save(npy_path, points)
        record = {
            "scene": int(scene[5:9]),
            "sub_scene": int(scene[10:12]),
            "file_len": 1,
            "filepath": str(npy_path),
        }
        if split != "test":
            instance_path = (
                processed_dir / "instance_gt" / split / f"{scene}.txt"
            )
            instance_path.parent.mkdir(parents=True, exist_ok=True)
            instance_path.write_text("3001\n", encoding="utf-8")
            record["instance_gt_filepath"] = str(instance_path)
        _write_yaml(processed_dir / database_names[split], [record])

    return {
        "raw_dir": raw_dir,
        "processed_dir": processed_dir,
        "split_dir": split_dir,
        "test_segments_dir": test_segments_dir,
        "output_dir": output_dir,
    }


def test_missing_scannet_writes_all_blocked_artifacts_and_exits_two(tmp_path: Path) -> None:
    output_dir = tmp_path / "private-user-name" / "artifacts"
    result = _run_audit(
        raw_dir=tmp_path / "private-user-name" / "missing-raw",
        processed_dir=tmp_path / "private-user-name" / "missing-processed",
        output_dir=output_dir,
    )

    assert result.returncode == 2, result.stderr
    assert {path.name for path in output_dir.iterdir()} == REQUIRED_ARTIFACTS

    preflight = _load_json(output_dir / "scannet_preflight.json")
    assert preflight["local_source_commit"] == subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert preflight["source_tree_contract"]["schema_version"] == 1
    assert preflight["source_tree_contract"]["allowed_dirty_prefixes"] == [
        "artifacts/P2/"
    ]
    assert preflight["source_tree_contract"]["source_commit"] == preflight[
        "local_source_commit"
    ]
    assert preflight["runtime_source_contract"]["status"] == "pass"
    assert set(preflight["runtime_source_contract"]["repositories"]) == {
        "concerto",
        "sonata",
        "detectron2",
        "stmetrics",
    }
    assert preflight["status"] == "blocked_missing_scannet"
    assert preflight["formal_p2_training_authorized"] is False
    assert preflight["expected_split_counts"] == OFFICIAL_SPLIT_COUNTS
    assert preflight["mix_instantiation"]["status"] == "blocked_prerequisites"
    assert preflight["mix_instantiation"]["attempted"] is False
    assert preflight["errors"]
    assert len((output_dir / "scannet_preflight.json").read_bytes()) < 50_000
    assert {error["code"] for error in preflight["errors"]} >= {
        "scannet_raw_root_missing",
        "scannet_processed_root_missing",
    }

    blocked = (output_dir / "BLOCKED_MISSING_SCANNET.md").read_text(
        encoding="utf-8"
    )
    for unexecuted in (
        "formal topology benchmark",
        "official smoke test",
        "official tiny overfit",
        "formal 450-epoch training",
        "formal checkpoint",
        "G2 metrics",
    ):
        assert unexecuted in blocked
    assert "G2 =" not in blocked
    assert "P2 complete" not in blocked
    assert "GO" not in blocked

    _assert_artifacts_private(output_dir)


def test_split_metadata_rejects_cross_partition_overlap(tmp_path: Path) -> None:
    split_dir = tmp_path / "Benchmark"
    split_dir.mkdir()
    for filename in (
        "scannetv2_train.txt",
        "scannetv2_val.txt",
        "scannetv2_test.txt",
    ):
        (split_dir / filename).write_text("scene0000_00\n", encoding="utf-8")

    _, _, errors = audit._read_split_metadata(
        split_dir,
        {"train": 1, "validation": 1, "test": 1},
    )

    assert "split_cross_partition_overlap" in {
        error["code"] for error in errors
    }


def test_official_split_identity_requires_pinned_files_and_commit(
    tmp_path: Path,
) -> None:
    split_dir = tmp_path / "Benchmark"
    split_dir.mkdir()
    for filename in (
        "scannetv2_train.txt",
        "scannetv2_val.txt",
        "scannetv2_test.txt",
    ):
        (split_dir / filename).write_text("scene0000_00\n", encoding="utf-8")

    identity, errors = audit._audit_official_split_identity(split_dir)

    assert identity["status"] == "fail"
    assert identity["expected_commit"] == SCANNET_OFFICIAL_COMMIT
    assert {
        split: record["expected_sha256"]
        for split, record in identity["files"].items()
    } == OFFICIAL_SPLIT_SHA256
    assert {error["code"] for error in errors} >= {
        "scannet_official_commit_mismatch",
        "official_split_sha256_mismatch",
    }


def test_official_split_identity_is_derived_from_the_parsed_split_snapshot() -> None:
    split_dir = REPO_ROOT / "third_party" / "ScanNet" / "Tasks" / "Benchmark"

    _, records, read_errors = audit._read_split_metadata(
        split_dir,
        OFFICIAL_SPLIT_COUNTS,
    )
    identity, identity_errors = audit._audit_official_split_identity(
        split_dir,
        split_records=records,
    )

    assert read_errors == []
    assert identity_errors == []
    assert identity == p2_preflight.build_scannet_official_split_identity(
        split_dir=split_dir
    )
    assert all(
        record["observed_sha256"] == OFFICIAL_SPLIT_SHA256[split]
        for split, record in records.items()
    )


def test_semantic_snapshot_stability_rejects_input_or_split_mutation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    initial_manifest = {"schema_version": 1, "status": "pass", "value": "before"}
    final_manifest = {"schema_version": 1, "status": "pass", "value": "after"}
    initial_split = {"status": "pass", "value": "before"}
    final_split = {"status": "pass", "value": "after"}
    monkeypatch.setattr(
        audit,
        "build_p2_input_manifest",
        lambda **kwargs: final_manifest,
    )
    monkeypatch.setattr(
        audit,
        "build_scannet_official_split_identity",
        lambda **kwargs: final_split,
    )

    observed_manifest, observed_split, errors = (
        audit._audit_semantic_snapshot_stability(
            initial_input_manifest=initial_manifest,
            initial_split_identity=initial_split,
            processed_scannet_dir=tmp_path / "scannet",
            rio_processed_dir=tmp_path / "rio",
            split_dir=tmp_path / "Benchmark",
        )
    )

    assert observed_manifest == final_manifest
    assert observed_split == final_split
    assert {error["code"] for error in errors} == {
        "processed_input_changed_during_semantic_audit",
        "official_split_changed_during_semantic_audit",
    }


def test_source_snapshot_stability_rejects_main_or_nested_runtime_mutation(
    monkeypatch,
) -> None:
    initial_source = {"schema_version": 1, "status": "pass", "value": "before"}
    final_source = {"schema_version": 1, "status": "pass", "value": "after"}
    initial_runtime = {
        "schema_version": 1,
        "status": "pass",
        "value": "before",
    }
    final_runtime = {
        "schema_version": 1,
        "status": "pass",
        "value": "after",
    }
    initial_environment = {
        "schema_version": 1,
        "status": "pass",
        "value": "before",
    }
    final_environment = {
        "schema_version": 1,
        "status": "pass",
        "value": "after",
    }
    monkeypatch.setattr(
        audit,
        "build_p2_source_tree_contract",
        lambda: final_source,
    )
    monkeypatch.setattr(
        audit,
        "build_p2_runtime_source_contract",
        lambda: final_runtime,
    )
    monkeypatch.setattr(
        audit,
        "build_p2_runtime_environment_contract",
        lambda: final_environment,
    )

    observed_source, observed_runtime, observed_environment, errors = (
        audit._audit_source_contract_stability(
            initial_source_tree_contract=initial_source,
            initial_runtime_source_contract=initial_runtime,
            initial_runtime_environment_contract=initial_environment,
        )
    )

    assert observed_source == final_source
    assert observed_runtime == final_runtime
    assert observed_environment == final_environment
    assert {error["code"] for error in errors} == {
        "source_tree_changed_during_audit",
        "runtime_source_changed_during_audit",
        "runtime_environment_changed_during_audit",
    }


def test_environment_manifest_runtime_probe_fails_closed_on_oserror(
    monkeypatch,
) -> None:
    real_import = __import__

    def unavailable_torch_import(name, *args, **kwargs):
        if name == "torch":
            raise OSError("runtime shared library unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", unavailable_torch_import)

    observed = audit._measure_runtime_environment(
        {"torch": "fallback-torch", "cuda": "fallback-cuda", "gpu": {}}
    )

    assert observed["torch"] == "fallback-torch"
    assert observed["cuda"] == "fallback-cuda"
    assert observed["cudnn"] == "unknown"
    assert observed["nccl"] == "unknown"


@pytest.mark.parametrize(
    "failed_contract",
    ["source_tree", "runtime_source", "runtime_environment"],
)
def test_initial_input_manifest_gate_requires_all_runtime_contracts_to_pass(
    failed_contract: str,
) -> None:
    contracts = {
        "source_tree": {"status": "pass"},
        "runtime_source": {"status": "pass"},
        "runtime_environment": {"status": "pass"},
    }

    assert audit._initial_input_manifest_allowed(**contracts) is True
    contracts[failed_contract]["status"] = "fail"

    assert audit._initial_input_manifest_allowed(**contracts) is False


def test_artifacts_bind_the_exact_p2_target_and_current_reproduction_config(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "artifacts"
    result = _run_audit(
        raw_dir=tmp_path / "missing-raw",
        processed_dir=tmp_path / "missing-processed",
        output_dir=output_dir,
    )
    assert result.returncode == 2, result.stderr

    target = yaml.safe_load(
        (output_dir / "reproduction_target.yaml").read_text(encoding="utf-8")
    )
    assert target["stage"] == "P2"
    assert target["model"] == {
        "backbone": "Concerto",
        "encoder": "PTv3 frozen",
        "decoder": "train from scratch",
        "queries": 100,
        "query_initialization": "FPS non-parametric",
        "temporal_window": 2,
        "contrastive": True,
        "st_serialization": True,
        "st_masking": False,
    }
    assert target["data"]["mix"] == [
        {"dataset": "3RScan", "temporal_window": 2, "weight": 1.0},
        {"dataset": "ScanNet", "temporal_window": 1, "weight": 0.8},
    ]
    assert target["data"]["voxel_size_m"] == 0.02
    assert target["training"] == {
        "epochs": 450,
        "effective_batch_size": 32,
        "optimizer": "AdamW",
        "scheduler": "OneCycleLR",
        "max_lr": 0.0005,
        "loss_weights": {"class": 2.0, "mask_bce": 5.0, "dice": 2.0},
        "no_object_weight": 0.2,
        "seed": 45,
        "precision": "32-true",
    }
    assert target["local_recommended_topology"] == {
        "gpus": 2,
        "batch_per_gpu": 4,
        "gradient_accumulation": 4,
        "physical_global_batch": 8,
        "effective_batch": 32,
    }
    assert target["reproduction_choices"]["sequence_database"] == {
        "reference": "repo:data/processed/rio/sequence_database_sliding_2.yaml",
        "sha256": "974299916db02d2ee0233564eb7b36e314dada93e997584d9dfef21ad70d0416",
    }
    assert target["reproduction_choices"]["frozen_encoder_runtime"] == {
        "parameters_require_grad": False,
        "module_mode": "train",
        "concerto_drop_path_rate": 0.3,
        "decoder_and_head_trainable": True,
    }

    diff = _load_json(output_dir / "official_vs_repro_config_diff.json")
    assert diff["config_ref"] == "repo:conf/config_p2_rescene4d_concerto_t2.yaml"
    assert diff["config_composed"] is True
    assert diff["settings"]["precision"]["status"] == "explicit_reproduction_choice"
    assert diff["settings"]["precision"]["official"] == "not reported"
    assert diff["settings"]["backbone_checkpoint"]["status"] == (
        "verified_reproduction_choice"
    )
    assert diff["settings"]["backbone_checkpoint"]["reproduction"] == {
        "reference": "local_cache:persist4d/concerto/concerto_base.pth",
        "revision": "c31f993a56129f2ba9c5d06a35957e3f05bff710",
        "sha256": "845ec7dec97a5fabff8fadb5d9858ac6734347b612d1a4b574213419c139de07",
        "license": "CC-BY-NC-4.0",
    }
    assert diff["settings"]["frozen_encoder_runtime"] == {
        "official": {
            "parameters_require_grad": False,
            "module_mode": "not reported",
            "drop_path_rate": "not reported",
        },
        "reproduction": {
            "parameters_require_grad": False,
            "module_mode": "train",
            "drop_path_rate": 0.3,
            "decoder_and_head_trainable": True,
        },
        "repository_default": {
            "parameters_require_grad": True,
            "module_mode": "train",
            "drop_path_rate": 0.3,
        },
        "status": "repository_behavior_risk",
    }
    assert diff["settings"]["sequence_database"]["reproduction"] == {
        "reference": "repo:data/processed/rio/sequence_database_sliding_2.yaml",
        "sha256": "974299916db02d2ee0233564eb7b36e314dada93e997584d9dfef21ad70d0416",
        "mode": "sliding",
        "temporal_window": 2,
    }
    assert diff["settings"]["adamw_implicit_defaults"]["status"] == (
        "verified_reproduction_choice"
    )
    assert diff["settings"]["onecycle_implicit_defaults"]["status"] == (
        "verified_reproduction_choice"
    )
    assert diff["settings"]["augmentations"]["status"] == (
        "paper_exactness_unverified"
    )
    for setting in (
        "backbone",
        "encoder_freeze",
        "num_queries",
        "query_initialization",
        "temporal_window",
        "contrastive",
        "st_serialization",
        "st_masking",
        "voxel_size_m",
        "loss_weights",
        "no_object_weight",
        "optimizer",
        "scheduler",
        "max_lr",
        "epochs",
        "effective_batch_size",
        "dataset_mix",
    ):
        assert diff["settings"][setting]["status"] == "match"

    deviation_settings = {entry["setting"] for entry in diff["declared_deviations"]}
    assert deviation_settings >= {
        "backbone_checkpoint",
        "frozen_encoder_runtime",
        "adamw_implicit_defaults",
        "onecycle_implicit_defaults",
        "augmentations",
        "precision",
        "hardware_topology",
    }
    assert diff["reproduction_code_relation"] == {
        "official_code_commit": "fb2fe42eb8f1e926567c48eea9acb874e608ee10",
        "local_fix_commit": "3c6b11a3af600aa98c93128361c2ecb4900ea186",
        "runtime_safety_fix_commit": "611ba161454cdfde7fe047fcae1e0d7b81387bf2",
        "official_code_used_unchanged": False,
        "status": "local_alignment_and_safety_patch_set",
    }
    fixes = {fix["id"]: fix for fix in diff["implementation_fixes"]}
    paper_fix_ids = {
        "weighted_segmentation_objective",
        "contrastive_diagnostic_deduplication",
        "hydra_contrastive_override_order",
    }
    safety_fix_ids = {
        "fail_closed_dataset_sequence_mix_validation",
        "ddp_batch_contract_consensus",
        "full_state_checkpoint_resume_selection",
    }
    assert set(fixes) == paper_fix_ids | safety_fix_ids
    assert fixes["weighted_segmentation_objective"]["upstream_semantic_effect"] == (
        "configured class/mask_bce/dice weights 2/5/2 were not applied to the "
        "optimized objective; effective weights were 1/1/1"
    )
    assert fixes["weighted_segmentation_objective"]["local_behavior"] == (
        "training and validation use criterion.weight_dict, applying 2/5/2 to final "
        "and auxiliary segmentation losses"
    )
    assert fixes["contrastive_diagnostic_deduplication"]["upstream_semantic_effect"] == (
        "aggregate contrastive losses and their per-layer diagnostic losses were both "
        "summed, double-counting the contrastive objective"
    )
    assert fixes["contrastive_diagnostic_deduplication"]["local_behavior"] == (
        "aggregate contrastive objectives are optimized exactly once; per-layer values "
        "remain logging-only diagnostics"
    )
    assert fixes["contrastive_diagnostic_deduplication"]["paper_alignment"] == (
        "counts each aggregate temporal contrastive loss exactly once while preserving "
        "per-layer observability"
    )
    assert fixes["hydra_contrastive_override_order"]["upstream_semantic_effect"] == (
        "loss/contrastive=infoNCE was overwritten by the later set_criterion default, "
        "leaving loss.contrastive_loss=false"
    )
    assert all(
        fix["classification"] == "local_paper_alignment_fix"
        and fix["official_code_commit"]
        == "fb2fe42eb8f1e926567c48eea9acb874e608ee10"
        and fix["local_fix_commit"]
        == "3c6b11a3af600aa98c93128361c2ecb4900ea186"
        for fix_id, fix in fixes.items()
        if fix_id in paper_fix_ids
    )
    assert all(
        fix["classification"] == "local_reproduction_safety_fix"
        and fix["official_code_commit"]
        == "fb2fe42eb8f1e926567c48eea9acb874e608ee10"
        and fix["local_fix_commit"]
        == "611ba161454cdfde7fe047fcae1e0d7b81387bf2"
        and "paper_alignment" not in fix
        for fix_id, fix in fixes.items()
        if fix_id in safety_fix_ids
    )
    assert fixes["fail_closed_dataset_sequence_mix_validation"][
        "safety_effect"
    ] == (
        "prevents missing ScanNet from degrading the required mix to RIO-only and "
        "prevents unknown temporal scans from becoming zero indices"
    )
    assert fixes["ddp_batch_contract_consensus"]["collective_contract"] == {
        "normal_train_microbatch": {
            "safety_int32_max_all_reduce_count": 3,
            "criterion_float_num_masks_all_reduce_count": 1,
            "total_all_reduce_count": 4,
            "all_gather_object_count": 0,
        },
        "normal_validation_microbatch": {
            "safety_int32_max_all_reduce_count": 4,
            "criterion_float_num_masks_all_reduce_count": 1,
            "total_all_reduce_count": 5,
            "all_gather_object_count": 0,
        },
        "normal_test_microbatch": {
            "safety_int32_max_all_reduce_count": 3,
            "criterion_float_num_masks_all_reduce_count": 0,
            "total_all_reduce_count": 3,
            "all_gather_object_count": 0,
        },
        "covered_stage_failure": {
            "additional_all_gather_object_count": 1,
        },
        "train_optimizer_step_accumulation_4": {
            "safety_int32_max_all_reduce_count": 13,
            "optimizer_gradient_int32_max_all_reduce_count": 1,
            "criterion_float_num_masks_all_reduce_count": 4,
            "total_all_reduce_count": 17,
        },
    }
    assert fixes["ddp_batch_contract_consensus"]["performance_cost"] == (
        "three blocking scalar int32 MAX all_reduce operations per normal train "
        "DDP microbatch, four per validation microbatch, and three per test "
        "microbatch; train "
        "accumulation=4 costs twelve microbatch safety all-reduces, one "
        "optimizer-gradient safety all-reduce, and four criterion float num_masks "
        "all-reduces per optimizer step (17 total); all_gather_object adds one "
        "call only on a covered stage failure"
    )
    assert fixes["full_state_checkpoint_resume_selection"]["local_behavior"] == (
        "statically validates required Lightning full-state fields, selects the latest "
        "valid candidate by checkpoint epoch/global_step and numeric filename version, "
        "and refuses to start from scratch when checkpoint files exist but all are invalid"
    )
    assert fixes["full_state_checkpoint_resume_selection"]["restore_boundary"] == (
        "static validation is not a real Lightning restore; trainer.fit is attempted once "
        "with the selected checkpoint and does not automatically retry another candidate "
        "after a Lightning restore failure"
    )
    audit_markdown = (output_dir / "config_audit.md").read_text(encoding="utf-8")
    assert "drop_path=0.3" in audit_markdown
    assert "module remains in train mode" in audit_markdown
    assert "974299916db02d2ee0233564eb7b36e314dada93e997584d9dfef21ad70d0416" in audit_markdown
    assert "Code-Level Paper Alignment Fixes" in audit_markdown
    assert "not an unchanged checkout" in audit_markdown
    assert "effective weights were `1/1/1`" in audit_markdown
    assert "optimized exactly once" in audit_markdown
    assert "loss.contrastive_loss=false" in audit_markdown
    assert "Local Reproduction Safety Fixes" in audit_markdown
    assert "not paper-alignment loss fixes" in audit_markdown
    assert "missing ScanNet" in audit_markdown
    assert "RIO-only" in audit_markdown
    assert (
        "three scalar int32 MAX all-reduces per train microbatch"
        in audit_markdown
    )
    assert "four per validation and three per test microbatch" in audit_markdown
    assert (
        "twelve microbatch safety plus one optimizer-gradient safety and four "
        "criterion float num_masks all-reduces"
        in audit_markdown
    )
    assert "all_gather_object only on a covered failure" in audit_markdown
    assert "does not automatically retry another candidate" in audit_markdown

    environment = _load_json(output_dir / "environment_manifest.json")
    assert environment["official_source"]["commit"] == (
        "fb2fe42eb8f1e926567c48eea9acb874e608ee10"
    )
    assert {
        name: source["source_ref"]
        for name, source in environment["third_party_sources"].items()
    } == {
        "concerto": "external:github/Pointcept/Concerto",
        "sonata": "external:github/facebookresearch/sonata",
        "detectron2": "external:github/facebookresearch/detectron2",
        "stmetrics": "external:github/GradientSpaces/stmetrics",
        "scannet_tools": "external:github/ScanNet/ScanNet",
    }
    assert environment["model_weights"]["concerto"] == {
        "reference": "local_cache:persist4d/concerto/concerto_base.pth",
        "revision": "c31f993a56129f2ba9c5d06a35957e3f05bff710",
        "expected_sha256": "845ec7dec97a5fabff8fadb5d9858ac6734347b612d1a4b574213419c139de07",
        "observed_sha256": "845ec7dec97a5fabff8fadb5d9858ac6734347b612d1a4b574213419c139de07",
        "expected_byte_size": 433987358,
        "observed_byte_size": 433987358,
        "status": "pass",
        "license": "CC-BY-NC-4.0",
    }
    assert environment["runtime_environment"] == {
        "python": "3.10.20",
        "torch": "2.6.0+cu126",
        "cuda": "12.6",
        "cudnn": "9.5.1",
        "nccl": "2.21.5",
        "runtime_packages": {
            "pytorch_lightning": "2.6.5",
            "hydra_core": "1.3.4",
            "spconv": "2.3.8",
            "flash_attn": "2.8.3",
            "torch_scatter": "2.1.2+pt26cu126",
            "sonata": "1.0",
            "detectron2": "0.6",
            "concerto": "1.0",
            "stmetrics": "0.1.0",
            "wandb": "0.28.0",
        },
        "gpu": {
            "count": 3,
            "model": "NVIDIA A40",
            "memory_mib": 46068,
            "driver": "595.71.05",
        },
    }
    _assert_artifacts_private(output_dir)


def test_complete_injected_fixture_is_diagnostic_only_with_non_official_counts(
    tmp_path: Path,
) -> None:
    fixture = _make_complete_scannet_fixture(tmp_path)
    result = _run_audit(
        raw_dir=fixture["raw_dir"],
        processed_dir=fixture["processed_dir"],
        split_dir=fixture["split_dir"],
        test_segments_dir=fixture["test_segments_dir"],
        output_dir=fixture["output_dir"],
        expected_counts=(1, 1, 1),
    )

    assert result.returncode == 0, result.stderr
    preflight = _load_json(fixture["output_dir"] / "scannet_preflight.json")
    assert preflight["status"] == "diagnostic_pass"
    assert preflight["formal_p2_training_authorized"] is False
    assert preflight["authorization"]["status"] == "not_issued"
    assert preflight["authorization"]["reason"] == (
        "non_official_expected_split_counts"
    )
    assert preflight["errors"] == []
    assert preflight["raw_assets"]["complete_scene_count"] == 3
    assert preflight["processed_assets"]["database_scene_count"] == 3
    assert preflight["processed_assets"]["npy_scene_count"] == 3
    assert preflight["class_taxonomy"] == {
        "status": "pass",
        "name": "scannet",
        "valid_class_ids": NYU40_INSTANCE_IDS,
        "class_labels": NYU40_INSTANCE_LABELS,
        "class_count": 18,
    }
    assert preflight["known_empty_scan_substitutions"] == {
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
    }
    assert preflight["mix_instantiation"] == {
        "attempted": True,
        "status": "pass",
        "implementation": "datasets.multi_dataset.MultiDataset",
        "dataset_names": ["rio", "scannet"],
        "dataset_sizes": [1178, 1],
        "weights": [1.0, 0.8],
        "temporal_windows": [2, 1],
        "sampler": "WeightedRandomSampler",
    }
    blocker = fixture["output_dir"] / "BLOCKED_MISSING_SCANNET.md"
    assert blocker.is_file()
    blocker_text = blocker.read_text(encoding="utf-8")
    assert "diagnostic-only" in blocker_text
    assert "1201/312/100" in blocker_text
    _assert_artifacts_private(fixture["output_dir"])


def test_empty_metric_taxonomy_blocks_formal_authorization(tmp_path: Path) -> None:
    fixture = _make_complete_scannet_fixture(tmp_path)
    metric_path = fixture["processed_dir"] / "scannet.yaml"
    metric = yaml.safe_load(metric_path.read_text(encoding="utf-8"))
    metric["valid_class_ids"] = []
    metric["class_labels"] = []
    _write_yaml(metric_path, metric)

    result = _run_audit(
        raw_dir=fixture["raw_dir"],
        processed_dir=fixture["processed_dir"],
        split_dir=fixture["split_dir"],
        test_segments_dir=fixture["test_segments_dir"],
        output_dir=fixture["output_dir"],
        expected_counts=(1, 1, 1),
    )

    assert result.returncode == 2, result.stderr
    preflight = _load_json(fixture["output_dir"] / "scannet_preflight.json")
    assert preflight["formal_p2_training_authorized"] is False
    assert preflight["class_taxonomy"]["status"] == "fail"
    assert {error["code"] for error in preflight["errors"]} >= {
        "metric_class_ids_mismatch",
        "metric_class_labels_mismatch",
    }


def test_empty_label_database_validation_mapping_fails_taxonomy(
    tmp_path: Path,
) -> None:
    fixture = _make_complete_scannet_fixture(tmp_path)
    label_db_path = fixture["processed_dir"] / "label_database.yaml"
    labels = yaml.safe_load(label_db_path.read_text(encoding="utf-8"))
    for entry in labels.values():
        entry["validation"] = False
    _write_yaml(label_db_path, labels)

    taxonomy, errors = audit._audit_taxonomy(fixture["processed_dir"])

    assert taxonomy["status"] == "fail"
    assert "label_database_validation_mapping_mismatch" in {
        error["code"] for error in errors
    }


def test_wrong_label_database_id_label_mapping_fails_taxonomy(
    tmp_path: Path,
) -> None:
    fixture = _make_complete_scannet_fixture(tmp_path)
    label_db_path = fixture["processed_dir"] / "label_database.yaml"
    labels = yaml.safe_load(label_db_path.read_text(encoding="utf-8"))
    labels[NYU40_INSTANCE_IDS[0]]["name"] = "wrong-label"
    _write_yaml(label_db_path, labels)

    taxonomy, errors = audit._audit_taxonomy(fixture["processed_dir"])

    assert taxonomy["status"] == "fail"
    assert "label_database_validation_mapping_mismatch" in {
        error["code"] for error in errors
    }


def test_preprocessor_underscore_label_maps_to_the_exact_nyu40_taxonomy(
    tmp_path: Path,
) -> None:
    fixture = _make_complete_scannet_fixture(tmp_path)
    label_db_path = fixture["processed_dir"] / "label_database.yaml"
    labels = yaml.safe_load(label_db_path.read_text(encoding="utf-8"))
    labels[28]["name"] = "shower_curtain"
    _write_yaml(label_db_path, labels)

    taxonomy, errors = audit._audit_taxonomy(fixture["processed_dir"])

    assert errors == []
    assert taxonomy["status"] == "pass"


def test_label_database_validation_mapping_order_is_exact(
    tmp_path: Path,
) -> None:
    fixture = _make_complete_scannet_fixture(tmp_path)
    label_db_path = fixture["processed_dir"] / "label_database.yaml"
    labels = yaml.safe_load(label_db_path.read_text(encoding="utf-8"))
    items = list(labels.items())
    items[0], items[1] = items[1], items[0]
    _write_yaml(label_db_path, dict(items))

    taxonomy, errors = audit._audit_taxonomy(fixture["processed_dir"])

    assert taxonomy["status"] == "fail"
    assert "label_database_validation_order_mismatch" in {
        error["code"] for error in errors
    }


def test_taxonomy_rejects_coercible_non_integer_ids_and_non_boolean_flags(
    tmp_path: Path,
) -> None:
    fixture = _make_complete_scannet_fixture(tmp_path)
    label_db_path = fixture["processed_dir"] / "label_database.yaml"
    labels = yaml.safe_load(label_db_path.read_text(encoding="utf-8"))
    rewritten = {
        (3.0 if class_id == 3 else class_id): entry
        for class_id, entry in labels.items()
    }
    rewritten[4]["validation"] = "true"
    _write_yaml(label_db_path, rewritten)
    metric_path = fixture["processed_dir"] / "scannet.yaml"
    metric = yaml.safe_load(metric_path.read_text(encoding="utf-8"))
    metric["valid_class_ids"][0] = "3"
    _write_yaml(metric_path, metric)

    taxonomy, errors = audit._audit_taxonomy(fixture["processed_dir"])

    assert taxonomy["status"] == "fail"
    assert {error["code"] for error in errors} >= {
        "label_database_validation_id_type_invalid",
        "label_database_validation_flag_type_invalid",
        "metric_class_id_type_invalid",
    }


def test_rio_label_database_requires_the_exact_canonical_taxonomy(
    tmp_path: Path,
) -> None:
    fixture = _make_complete_scannet_fixture(tmp_path)
    processed_dir = fixture["processed_dir"]
    metric = yaml.safe_load(
        (processed_dir / "scannet.yaml").read_text(encoding="utf-8")
    )
    metric["name"] = "rio"
    _write_yaml(processed_dir / "rio.yaml", metric)
    labels = yaml.safe_load(
        (processed_dir / "label_database.yaml").read_text(encoding="utf-8")
    )
    labels[24]["name"] = "refridgerator"
    _write_yaml(processed_dir / "label_database.yaml", labels)

    taxonomy, errors = audit._audit_taxonomy(processed_dir, "rio")

    assert taxonomy["status"] == "fail"
    assert "label_database_validation_mapping_mismatch" in {
        error["code"] for error in errors
    }


def test_rio_record_audit_rejects_paths_outside_the_processed_root(
    tmp_path: Path,
) -> None:
    rio_root = tmp_path / "rio"
    outside = tmp_path / "outside"
    outside.mkdir()
    np.save(outside / "0001_00.npy", np.zeros((1, 12), dtype=np.float32))
    (outside / "scene0001_00.txt").write_text("3001\n", encoding="utf-8")
    record = {
        "scene": 1,
        "sub_scene": 0,
        "file_len": 1,
        "filepath": str(outside / "0001_00.npy"),
        "instance_gt_filepath": str(outside / "scene0001_00.txt"),
    }
    _write_yaml(rio_root / "train_database.yaml", [record])
    _write_yaml(rio_root / "validation_database.yaml", [record])
    _write_yaml(
        rio_root / "sequence_database_sliding_2.yaml",
        {
            "scene0001_00-scene0001_00": {
                "type": "train",
                "filepath": str(outside / "change.txt"),
            }
        },
    )

    evidence, errors = audit._audit_rio_record_paths(rio_root)

    assert evidence["status"] == "fail"
    assert {error["code"] for error in errors} >= {
        "rio_processed_npy_path_outside_split",
        "rio_instance_gt_path_outside_split",
        "rio_change_gt_path_outside_root",
    }


def test_scannet_record_path_does_not_fall_back_to_a_matching_basename(
    tmp_path: Path,
) -> None:
    fixture = _make_complete_scannet_fixture(tmp_path)
    database_path = fixture["processed_dir"] / "train_database.yaml"
    records = yaml.safe_load(database_path.read_text(encoding="utf-8"))
    basename = Path(records[0]["filepath"]).name
    records[0]["filepath"] = str(tmp_path / "wrong-location" / basename)
    _write_yaml(database_path, records)

    _, _, errors = audit._audit_processed_assets(
        fixture["processed_dir"],
        {
            "train": ["scene0000_00"],
            "validation": ["scene0001_00"],
            "test": ["scene0002_00"],
        },
    )

    assert "processed_npy_path_outside_split" in {
        error["code"] for error in errors
    }


def test_rio_active_t2_sequence_without_instance_supervision_is_a_blocker(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(audit, "REPO_ROOT", tmp_path)
    root = tmp_path / "data" / "processed" / "rio"

    def record(split: str, scene: int, semantic: int) -> dict:
        stem = f"{scene:04d}_00"
        npy_ref = Path("data/processed/rio") / split / f"{stem}.npy"
        gt_ref = (
            Path("data/processed/rio")
            / "instance_gt"
            / split
            / f"scene{stem}.txt"
        )
        points = np.zeros((1, 12), dtype=np.float32)
        points[0, 10] = semantic
        points[0, 11] = 0
        (tmp_path / npy_ref).parent.mkdir(parents=True, exist_ok=True)
        np.save(tmp_path / npy_ref, points)
        (tmp_path / gt_ref).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / gt_ref).write_text(f"{semantic * 1000}\n", encoding="utf-8")
        return {
            "scene": scene,
            "sub_scene": 0,
            "file_len": 1,
            "filepath": npy_ref.as_posix(),
            "instance_gt_filepath": gt_ref.as_posix(),
        }

    train = [record("train", 1, 1), record("train", 2, 2)]
    validation = [
        record("validation", 3, 3),
        record("validation", 4, 3),
    ]
    _write_yaml(root / "train_database.yaml", train)
    _write_yaml(root / "validation_database.yaml", validation)
    _write_yaml(root / "train_validation_database.yaml", train + validation)
    sequences = {}
    for split, first, second in (
        ("train", 1, 2),
        ("validation", 3, 4),
    ):
        name = f"scene{first:04d}_00-scene{second:04d}_00"
        change_ref = Path("data/processed/rio/change_gt") / split / f"{name}.txt"
        (tmp_path / change_ref).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / change_ref).write_text("0\n", encoding="utf-8")
        sequences[name] = {
            "type": split,
            "scene": first,
            "sub_scenes": [0, 0],
            "filepath": change_ref.as_posix(),
        }
    _write_yaml(root / "sequence_database_sliding_2.yaml", sequences)

    evidence, errors = audit._audit_rio_record_paths(
        root,
        validate_content=True,
    )

    assert evidence["unsupervised_sequences"] == [
        "scene0001_00-scene0002_00"
    ]
    assert "rio_active_sequence_supervision_empty" in {
        error["code"] for error in errors
    }


def test_model_checkpoint_audit_rejects_an_unverified_local_file(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "concerto_base.pth"
    checkpoint.write_bytes(b"not-the-pinned-concerto-checkpoint")

    evidence, errors = audit._audit_model_checkpoint(checkpoint)

    assert evidence["status"] == "fail"
    assert evidence["observed_byte_size"] == checkpoint.stat().st_size
    assert evidence["observed_sha256"] != evidence["expected_sha256"]
    assert {error["code"] for error in errors} >= {
        "model_checkpoint_size_mismatch",
        "model_checkpoint_sha256_mismatch",
    }


def test_known_empty_substitution_audit_requires_the_pinned_sequence_database(
    tmp_path: Path,
) -> None:
    sequence_database = tmp_path / "sequence_database_sliding_2.yaml"
    _write_yaml(
        sequence_database,
        {
            "scene0171_00-scene0171_01": {
                "scene": 171,
                "sub_scenes": [0, 1],
                "type": "train",
            }
        },
    )
    *_, p2_config = audit._compose_config_snapshot()

    evidence, errors = audit._audit_known_empty_scan_substitutions(
        p2_config,
        sequence_database,
    )

    assert evidence["status"] == "fail"
    assert evidence["affected_sequences"] == [
        "scene0171_00-scene0171_01"
    ]
    assert {error["code"] for error in errors} >= {
        "rio_sequence_database_sha256_mismatch",
        "known_empty_scan_sequences_mismatch",
    }


def test_formal_audit_rejects_data_roots_other_than_the_resolved_p2_config(
    tmp_path: Path,
) -> None:
    *_, p2_config = audit._compose_config_snapshot()

    evidence, errors = audit._audit_formal_data_roots(
        p2_config,
        tmp_path / "scannet",
        tmp_path / "rio",
        raw_scannet_dir=tmp_path / "raw-scannet",
        split_dir=tmp_path / "Benchmark",
        test_segments_dir=tmp_path / "test-segments",
    )

    assert evidence["status"] == "fail"
    assert evidence["expected"] == {
        "raw_scannet": "repo:data/raw/scannet/scannet",
        "scannet": "repo:data/processed/scannet",
        "rio": "repo:data/processed/rio",
        "split_metadata": "repo:third_party/ScanNet/Tasks/Benchmark",
        "test_segments": "repo:data/raw/scannet_test_segments",
    }
    assert {error["code"] for error in errors} == {
        "formal_raw_scannet_data_root_mismatch",
        "formal_scannet_data_root_mismatch",
        "formal_rio_data_root_mismatch",
        "formal_split_metadata_data_root_mismatch",
        "formal_test_segments_data_root_mismatch",
    }


def test_input_manifest_digest_changes_when_a_training_input_changes(
    tmp_path: Path,
) -> None:
    scannet_root = tmp_path / "scannet"
    rio_root = tmp_path / "rio"
    scannet_root.mkdir()
    rio_root.mkdir()
    training_input = scannet_root / "train_database.yaml"
    training_input.write_text("[]\n", encoding="utf-8")
    (rio_root / "train_database.yaml").write_text("[]\n", encoding="utf-8")

    before = p2_preflight.build_p2_input_manifest(
        scannet_root=scannet_root,
        rio_root=rio_root,
        repo_root=tmp_path,
    )
    training_input.write_text("[changed]\n", encoding="utf-8")
    after = p2_preflight.build_p2_input_manifest(
        scannet_root=scannet_root,
        rio_root=rio_root,
        repo_root=tmp_path,
    )

    assert before["status"] == "pass"
    assert after["status"] == "pass"
    assert before["scannet"]["content_sha256"] != (
        after["scannet"]["content_sha256"]
    )


def test_all_zero_train_validation_npy_and_gt_are_rejected(tmp_path: Path) -> None:
    fixture = _make_complete_scannet_fixture(tmp_path)
    validation_npy = fixture["processed_dir"] / "validation" / "0001_00.npy"
    np.save(validation_npy, np.zeros((1, 12), dtype=np.float32))
    validation_gt = (
        fixture["processed_dir"]
        / "instance_gt"
        / "validation"
        / "scene0001_00.txt"
    )
    validation_gt.write_text("0\n", encoding="utf-8")

    result = _run_audit(
        raw_dir=fixture["raw_dir"],
        processed_dir=fixture["processed_dir"],
        split_dir=fixture["split_dir"],
        test_segments_dir=fixture["test_segments_dir"],
        output_dir=fixture["output_dir"],
        expected_counts=(1, 1, 1),
    )

    assert result.returncode == 2, result.stderr
    preflight = _load_json(fixture["output_dir"] / "scannet_preflight.json")
    assert preflight["formal_p2_training_authorized"] is False
    assert {error["code"] for error in preflight["errors"]} >= {
        "processed_npy_supervision_empty",
        "processed_instance_gt_supervision_empty",
    }


def test_test_npy_requires_exactly_ten_columns(tmp_path: Path) -> None:
    fixture = _make_complete_scannet_fixture(tmp_path)
    test_npy = fixture["processed_dir"] / "test" / "0002_00.npy"
    np.save(test_npy, np.zeros((1, 12), dtype=np.float32))

    result = _run_audit(
        raw_dir=fixture["raw_dir"],
        processed_dir=fixture["processed_dir"],
        split_dir=fixture["split_dir"],
        test_segments_dir=fixture["test_segments_dir"],
        output_dir=fixture["output_dir"],
        expected_counts=(1, 1, 1),
    )

    assert result.returncode == 2, result.stderr
    preflight = _load_json(fixture["output_dir"] / "scannet_preflight.json")
    assert "processed_npy_shape_invalid" in {
        error["code"] for error in preflight["errors"]
    }


def test_processed_npy_requires_float32_dtype(tmp_path: Path) -> None:
    fixture = _make_complete_scannet_fixture(tmp_path)
    validation_npy = fixture["processed_dir"] / "validation" / "0001_00.npy"
    points = np.load(validation_npy).astype(np.float64)
    np.save(validation_npy, points)

    result = _run_audit(
        raw_dir=fixture["raw_dir"],
        processed_dir=fixture["processed_dir"],
        split_dir=fixture["split_dir"],
        test_segments_dir=fixture["test_segments_dir"],
        output_dir=fixture["output_dir"],
        expected_counts=(1, 1, 1),
    )

    assert result.returncode == 2, result.stderr
    preflight = _load_json(fixture["output_dir"] / "scannet_preflight.json")
    assert "processed_npy_dtype_invalid" in {
        error["code"] for error in preflight["errors"]
    }


def test_string_npy_fails_closed_without_crashing_the_audit(tmp_path: Path) -> None:
    fixture = _make_complete_scannet_fixture(tmp_path)
    validation_npy = fixture["processed_dir"] / "validation" / "0001_00.npy"
    np.save(validation_npy, np.full((1, 12), "x"))

    result = _run_audit(
        raw_dir=fixture["raw_dir"],
        processed_dir=fixture["processed_dir"],
        split_dir=fixture["split_dir"],
        test_segments_dir=fixture["test_segments_dir"],
        output_dir=fixture["output_dir"],
        expected_counts=(1, 1, 1),
    )

    assert result.returncode == 2, result.stderr
    preflight = _load_json(fixture["output_dir"] / "scannet_preflight.json")
    assert preflight["formal_p2_training_authorized"] is False
    assert "processed_npy_dtype_invalid" in {
        error["code"] for error in preflight["errors"]
    }


def test_wall_only_train_validation_data_is_not_instance_supervision(
    tmp_path: Path,
) -> None:
    fixture = _make_complete_scannet_fixture(tmp_path)
    validation_npy = fixture["processed_dir"] / "validation" / "0001_00.npy"
    points = np.load(validation_npy)
    points[:, 10] = 1
    points[:, 11] = 0
    np.save(validation_npy, points)
    validation_gt = (
        fixture["processed_dir"]
        / "instance_gt"
        / "validation"
        / "scene0001_00.txt"
    )
    validation_gt.write_text("1001\n", encoding="utf-8")

    result = _run_audit(
        raw_dir=fixture["raw_dir"],
        processed_dir=fixture["processed_dir"],
        split_dir=fixture["split_dir"],
        test_segments_dir=fixture["test_segments_dir"],
        output_dir=fixture["output_dir"],
        expected_counts=(1, 1, 1),
    )

    assert result.returncode == 2, result.stderr
    preflight = _load_json(fixture["output_dir"] / "scannet_preflight.json")
    assert "processed_npy_instance_supervision_empty" in {
        error["code"] for error in preflight["errors"]
    }


def test_processed_record_paths_are_contained_scene_matched_and_unique(
    tmp_path: Path,
) -> None:
    fixture = _make_complete_scannet_fixture(tmp_path)
    validation_db_path = fixture["processed_dir"] / "validation_database.yaml"
    validation_db = yaml.safe_load(
        validation_db_path.read_text(encoding="utf-8")
    )
    validation_db[0]["filepath"] = str(
        fixture["processed_dir"] / "train" / "0000_00.npy"
    )
    _write_yaml(validation_db_path, validation_db)

    result = _run_audit(
        raw_dir=fixture["raw_dir"],
        processed_dir=fixture["processed_dir"],
        split_dir=fixture["split_dir"],
        test_segments_dir=fixture["test_segments_dir"],
        output_dir=fixture["output_dir"],
        expected_counts=(1, 1, 1),
    )

    assert result.returncode == 2, result.stderr
    preflight = _load_json(fixture["output_dir"] / "scannet_preflight.json")
    assert {error["code"] for error in preflight["errors"]} >= {
        "processed_npy_path_outside_split",
        "processed_npy_path_reused",
        "processed_npy_scene_stem_mismatch",
    }


def test_missing_processed_npy_blocks_before_mix_instantiation(tmp_path: Path) -> None:
    fixture = _make_complete_scannet_fixture(tmp_path)
    missing_npy = fixture["processed_dir"] / "validation" / "0001_00.npy"
    missing_npy.unlink()

    result = _run_audit(
        raw_dir=fixture["raw_dir"],
        processed_dir=fixture["processed_dir"],
        split_dir=fixture["split_dir"],
        test_segments_dir=fixture["test_segments_dir"],
        output_dir=fixture["output_dir"],
        expected_counts=(1, 1, 1),
    )

    assert result.returncode == 2, result.stderr
    preflight = _load_json(fixture["output_dir"] / "scannet_preflight.json")
    assert preflight["formal_p2_training_authorized"] is False
    assert preflight["mix_instantiation"] == {
        "attempted": False,
        "status": "blocked_prerequisites",
    }
    assert any(
        error["code"] == "processed_npy_missing"
        and error["split"] == "validation"
        and error["scene"] == "scene0001_00"
        for error in preflight["errors"]
    )


def test_blocked_artifacts_are_byte_deterministic(tmp_path: Path) -> None:
    output_dir = tmp_path / "artifacts"
    arguments = {
        "raw_dir": tmp_path / "missing-raw",
        "processed_dir": tmp_path / "missing-processed",
        "output_dir": output_dir,
    }
    first = _run_audit(**arguments)
    assert first.returncode == 2, first.stderr
    first_contents = {
        path.name: path.read_bytes() for path in sorted(output_dir.iterdir())
    }

    second = _run_audit(**arguments)
    assert second.returncode == 2, second.stderr
    second_contents = {
        path.name: path.read_bytes() for path in sorted(output_dir.iterdir())
    }

    assert second_contents == first_contents


def test_audit_invalidates_an_old_pass_before_any_fallible_work(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output_dir = tmp_path / "artifacts"
    output_dir.mkdir()
    (output_dir / "scannet_preflight.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "formal_p2_training_authorized": True,
            }
        ),
        encoding="utf-8",
    )

    def fail_compose():
        raise RuntimeError("injected audit crash")

    monkeypatch.setattr(audit, "_compose_config_snapshot", fail_compose)
    with pytest.raises(RuntimeError, match="injected audit crash"):
        audit.run_audit(
            raw_scannet_dir=tmp_path / "raw",
            processed_scannet_dir=tmp_path / "scannet",
            split_dir=tmp_path / "Benchmark",
            test_segments_dir=tmp_path / "test-segments",
            rio_processed_dir=tmp_path / "rio",
            output_dir=output_dir,
            expected_split_counts=OFFICIAL_SPLIT_COUNTS,
        )

    invalidated = _load_json(output_dir / "scannet_preflight.json")
    assert invalidated["status"] == "audit_in_progress"
    assert invalidated["formal_p2_training_authorized"] is False
