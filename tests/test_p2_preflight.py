import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "audit_p2_reproduction.py"
OFFICIAL_SPLIT_COUNTS = {"train": 1201, "validation": 312, "test": 100}
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
        np.save(npy_path, np.zeros((1, 12), dtype=np.float32))
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
    assert preflight["status"] == "blocked_missing_scannet"
    assert preflight["formal_p2_training_authorized"] is False
    assert preflight["expected_split_counts"] == OFFICIAL_SPLIT_COUNTS
    assert preflight["mix_instantiation"]["status"] == "blocked_prerequisites"
    assert preflight["mix_instantiation"]["attempted"] is False
    assert len(preflight["errors"]) < 30
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
        "runtime_safety_fix_commit": "973629172cc01ae0998bc785ac0ea2979756b72c",
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
        == "973629172cc01ae0998bc785ac0ea2979756b72c"
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
        "normal_ddp_microbatch": {
            "int32_max_all_reduce_count": 2,
            "all_gather_object_count": 0,
        },
        "covered_preflight_failure": {
            "int32_max_all_reduce_count": 1,
            "all_gather_object_count": 1,
        },
        "covered_forward_failure": {
            "int32_max_all_reduce_count": 2,
            "all_gather_object_count": 1,
        },
    }
    assert fixes["ddp_batch_contract_consensus"]["performance_cost"] == (
        "two blocking scalar int32 MAX all_reduce operations per normal DDP "
        "microbatch, or eight per optimizer step at accumulation=4; "
        "all_gather_object is failure-only"
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
    assert "two scalar int32 MAX all-reduces per normal DDP microbatch" in audit_markdown
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
        "sha256": "845ec7dec97a5fabff8fadb5d9858ac6734347b612d1a4b574213419c139de07",
        "byte_size": 433987358,
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


def test_complete_injected_fixture_passes_and_instantiates_real_mix(tmp_path: Path) -> None:
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
    assert preflight["status"] == "pass"
    assert preflight["formal_p2_training_authorized"] is True
    assert preflight["errors"] == []
    assert preflight["raw_assets"]["complete_scene_count"] == 3
    assert preflight["processed_assets"]["database_scene_count"] == 3
    assert preflight["processed_assets"]["npy_scene_count"] == 3
    assert preflight["class_taxonomy"] == {
        "status": "pass",
        "name": "scannet",
        "valid_class_ids": NYU40_INSTANCE_IDS,
        "class_count": 18,
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
    assert not (fixture["output_dir"] / "BLOCKED_MISSING_SCANNET.md").exists()
    _assert_artifacts_private(fixture["output_dir"])


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
