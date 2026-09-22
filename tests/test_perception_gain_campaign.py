from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml


def _campaign_module():
    try:
        return importlib.import_module("scripts.perception_gain_campaign")
    except ModuleNotFoundError as error:
        pytest.fail(f"perception campaign module is unavailable: {error}")


def test_campaign_config_and_resume_identity_are_fail_closed(tmp_path: Path) -> None:
    campaign = _campaign_module()
    source = Path("configs/perception_gain_v1.yaml")
    config = campaign.load_campaign_config(source)
    assert config["identity"]["parent_commit"] == (
        "6ef77620aa20926311eff3124a794a6ca2e32727"
    )
    assert config["runtime"] == {
        "batch_size_per_gpu": 2,
        "cpu_workers": 8,
        "devices": 2,
        "gradient_accumulation": 8,
        "precision": "32-true",
        "train_seed": 45,
        "eval_seed": 45,
    }
    assert config["budget"]["total_gpu_hours"] == 192

    altered = yaml.safe_load(source.read_text(encoding="utf-8"))
    altered["runtime"]["gradient_accumulation"] = 4
    altered_path = tmp_path / "altered.yaml"
    altered_path.write_text(yaml.safe_dump(altered), encoding="utf-8")
    with pytest.raises(campaign.CampaignError, match="runtime contract differs"):
        campaign.load_campaign_config(altered_path)

    state_path = tmp_path / "RUN_STATE.json"
    state = campaign.new_run_state(
        parent=config["identity"]["parent_commit"],
        instruction_sha=config["identity"]["instruction_sha256"],
        config_sha="1" * 64,
        data_sha="2" * 64,
    )
    state_path.write_text(json.dumps(state), encoding="utf-8")
    assert (
        campaign.load_resume_state(
            state_path,
            parent=config["identity"]["parent_commit"],
            instruction_sha=config["identity"]["instruction_sha256"],
            config_sha="1" * 64,
            data_sha="2" * 64,
        )["stage"]
        == "bootstrap"
    )
    with pytest.raises(campaign.CampaignError, match="resume identity differs"):
        campaign.load_resume_state(
            state_path,
            parent=config["identity"]["parent_commit"],
            instruction_sha=config["identity"]["instruction_sha256"],
            config_sha="3" * 64,
            data_sha="2" * 64,
        )


def test_bootstrap_writes_portable_public_contracts_and_local_asset_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = _campaign_module()
    project = tmp_path / "repo"
    artifact_root = project / "artifacts/perception_gain_v1"
    task_root = project / "artifacts/task_memory_retention_v2"
    cross_root = project / "artifacts/crosswindow_evidence_v1"
    config_root = project / "configs"
    for directory in (artifact_root, task_root, cross_root, config_root):
        directory.mkdir(parents=True, exist_ok=True)

    instruction = artifact_root / "EXECUTION_INSTRUCTION.md"
    instruction.write_text("approved instruction\n", encoding="utf-8")
    task_contract = {
        "roles": {
            "adaptation_reference_ids": ["train", "shared"],
            "additional_native_reference_ids": ["local"],
            "protocol_b_reference_ids": ["shared", "pb"],
        }
    }
    (task_root / "DATA_CONTRACT.json").write_text(
        json.dumps(task_contract), encoding="utf-8"
    )
    (cross_root / "DATA_ROLES.json").write_text(
        json.dumps({"roles": {"DEV-CAL": ["cal"], "DEV-SEL": ["sel"]}}),
        encoding="utf-8",
    )
    r1 = tmp_path / "r1.ckpt"
    concerto = tmp_path / "concerto.pth"
    metadata = tmp_path / "3RScan.json"
    data_root = tmp_path / "data"
    metric_spec = data_root / "processed/rio/rio.yaml"
    metric_spec.parent.mkdir(parents=True)
    r1.write_bytes(b"r1")
    concerto.write_bytes(b"concerto")
    metadata.write_text("[]", encoding="utf-8")
    metric_spec.write_text("labels: []\n", encoding="utf-8")
    locator = tmp_path / "old-assets.json"
    locator.write_text(
        json.dumps(
            {
                "external:r1_checkpoint": str(r1),
                "external:concerto_pretrained": str(concerto),
                "external:data_root": str(data_root),
                "external:rio_metadata": str(metadata),
            }
        ),
        encoding="utf-8",
    )
    config = yaml.safe_load(Path("configs/perception_gain_v1.yaml").read_text())
    config["identity"]["instruction_sha256"] = hashlib.sha256(
        instruction.read_bytes()
    ).hexdigest()
    config["identity"]["r1_checkpoint_sha256"] = hashlib.sha256(
        r1.read_bytes()
    ).hexdigest()
    config["identity"]["r1_checkpoint_bytes"] = r1.stat().st_size
    config["identity"]["concerto_sha256"] = hashlib.sha256(
        concerto.read_bytes()
    ).hexdigest()
    config_path = config_root / "perception_gain_v1.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    monkeypatch.setattr(campaign, "PROJECT_ROOT", project)
    external = tmp_path / "external"
    assert (
        campaign._bootstrap(
            SimpleNamespace(
                config=str(config_path),
                external_root=str(external),
                assets_from=str(locator),
            )
        )
        == 0
    )

    run_config = json.loads((artifact_root / "RUN_CONFIG.json").read_text())
    public_text = json.dumps(run_config)
    assert str(tmp_path) not in public_text
    assert run_config["inputs"]["r1_checkpoint"]["status"] == "VERIFIED"
    assert run_config["inputs"]["concerto_pretrained"]["status"] == "VERIFIED"
    roles = json.loads((artifact_root / "DATA_ROLES.json").read_text())["roles"]
    assert roles["TRAIN"] == ["train"]
    assert roles["removed_train_overlap"] == ["shared"]
    assert json.loads((external / "assets.local.json").read_text())[
        "r1_checkpoint"
    ] == str(r1)
    assert (artifact_root / "CODE_BINDINGS.md").is_file()
    assert (artifact_root / "EXECUTION_LOG.jsonl").is_file()


def test_pending_stage_prefix_is_strict_and_resume_safe() -> None:
    campaign = _campaign_module()

    assert campaign.pending_stages(["bootstrap"], through="foundation") == (
        "bind",
        "foundation",
    )
    assert (
        campaign.pending_stages(
            ["bootstrap", "bind", "foundation"], through="foundation"
        )
        == ()
    )
    with pytest.raises(campaign.CampaignError, match="exact prefix"):
        campaign.pending_stages(["bootstrap", "foundation"], through="foundation")


def test_validate_scorer_stage_binds_external_checkpoint_hash(tmp_path: Path) -> None:
    campaign = _campaign_module()
    checkpoint = tmp_path / "training/scorer/model/update=0500.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"frozen scorer")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    shard = tmp_path / "training/scorer/data/scorer-0000.pt"
    shard.parent.mkdir(parents=True)
    shard.write_bytes(b"records")
    data_manifest = {
        "status": "PASS",
        "completed_pair_count": 64,
        "reference_count": 64,
        "gpu_hours": 0.01,
        "shards": [
            {
                "external_reference": "external:training/scorer/data/scorer-0000.pt",
                "sha256": hashlib.sha256(shard.read_bytes()).hexdigest(),
                "bytes": shard.stat().st_size,
                "record_count": 64,
            }
        ],
    }
    training_summary = {
        "status": "COMPLETE",
        "completed_updates": 500,
        "checkpoint": "update=0500.ckpt",
        "checkpoint_sha256": digest,
        "gpu_hours": 0.001,
    }

    result = campaign.validate_scorer_stage(
        data_manifest=data_manifest,
        training_summary=training_summary,
        external_root=tmp_path,
    )

    assert result["status"] == "PASS"
    assert result["checkpoint"] == "external:training/scorer/model/update=0500.ckpt"
    assert result["gpu_hours"] == pytest.approx(0.011)
    with pytest.raises(campaign.CampaignError, match="hash"):
        campaign.validate_scorer_stage(
            data_manifest=data_manifest,
            training_summary={**training_summary, "checkpoint_sha256": "0" * 64},
            external_root=tmp_path,
        )


def test_cached_bind_and_foundation_evidence_are_reconciled_without_external_reads() -> (
    None
):
    campaign = _campaign_module()
    binding = {
        "schema_version": "perception-gain-foundation-binding-v1",
        "status": "PASS",
        "r1": {
            "checkpoint_bytes": 754_813_672,
            "checkpoint_sha256": "a" * 64,
        },
    }

    assert (
        campaign.validate_cached_bind_stage(
            binding,
            checkpoint_sha256="a" * 64,
            checkpoint_bytes=754_813_672,
        )
        == binding
    )

    foundation = campaign.validate_cached_foundation_stage(
        corrected_e1={"status": "PASS"},
        live_smoke={"status": "PASS", "gpu_hours": 0.2},
        diagnostic_panel={"status": "PASS", "gpu_hours": 0.3},
        native_smoke={"status": "PASS", "gpu_hours": 0.1},
    )
    assert foundation["status"] == "PASS"
    assert foundation["gpu_hours"] == pytest.approx(0.6)
    with pytest.raises(campaign.CampaignError, match="foundation evidence"):
        campaign.validate_cached_foundation_stage(
            corrected_e1={"status": "PASS"},
            live_smoke={"status": "PASS", "gpu_hours": 0.2},
            diagnostic_panel={"status": "PASS", "gpu_hours": 0.3},
            native_smoke={"status": "PARTIAL", "gpu_hours": 0.1},
        )


def test_validate_perception_pilot_binds_training_evaluation_and_selection(
    tmp_path: Path,
) -> None:
    campaign = _campaign_module()
    variants = ("C0", "S-BAL", "S-WORST", "Q-SEM", "A-OPEN")
    training = {}
    evaluations = {}
    values = {
        "C0": 0.201,
        "S-BAL": 0.202,
        "S-WORST": 0.203,
        "Q-SEM": 0.190,
        "A-OPEN": 0.180,
    }
    for ordinal, variant in enumerate(variants):
        run_dir = tmp_path / "training" / variant
        run_dir.mkdir(parents=True)
        checkpoints = []
        for update in (250, 750):
            path = run_dir / f"update={update:04d}.ckpt"
            path.write_bytes(f"{variant}:{update}".encode())
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            checkpoints.append(
                {"bytes": path.stat().st_size, "name": path.name, "sha256": digest}
            )
            metric = values[variant] if update == 750 else values[variant] - 0.001
            evaluations[(variant, update)] = {
                "status": "PASS",
                "variant": variant,
                "optimizer_update": update,
                "data_role": "CAL",
                "expected_logical_unit_count": 23,
                "completed_logical_unit_count": 23,
                "gpu_hours": 0.2,
                "t2_same_forward_parity": {"status": "PASS"},
                "candidate": {
                    "method_id": variant,
                    "optimizer_update": update,
                    "new_parameter_count": ordinal,
                    "metrics": {str(horizon): metric for horizon in (2, 3, 4, 5)},
                    "coverage_status": "COMPLETE",
                    "checkpoint": f"external:training/{variant}/{path.name}",
                    "checkpoint_sha256": digest,
                    "expected_logical_units": 23,
                    "completed_logical_units": 23,
                },
            }
        last = run_dir / "last.ckpt"
        last.write_bytes(f"{variant}:last".encode())
        checkpoints.append(
            {
                "bytes": last.stat().st_size,
                "name": last.name,
                "sha256": hashlib.sha256(last.read_bytes()).hexdigest(),
            }
        )
        training[variant] = {
            "status": "COMPLETE",
            "variant": variant,
            "completed_global_step": 750,
            "gpu_hours": 1.0,
            "progress": {
                "schema_version": "perception-gain-exact-resume-v1",
                "completed_local_batches": 6000,
                "completed_optimizer_updates": 750,
                "next_global_draw_index": 24000,
            },
            "checkpoints": checkpoints,
        }

    result = campaign.validate_perception_pilot_stage(
        training_summaries=training,
        evaluation_summaries=evaluations,
        baseline_metrics={str(horizon): 0.2 for horizon in (2, 3, 4, 5)},
        external_root=tmp_path,
    )

    assert result["status"] == "PASS"
    assert result["selection"]["full_training_arms"] == [
        "C0",
        "S-WORST",
        "S-BAL",
    ]
    assert result["training_gpu_hours"] == pytest.approx(5.0)
    assert result["evaluation_gpu_hours"] == pytest.approx(2.0)

    evaluations[("S-BAL", 750)]["candidate"]["checkpoint_sha256"] = "0" * 64
    with pytest.raises(campaign.CampaignError, match="checkpoint identity"):
        campaign.validate_perception_pilot_stage(
            training_summaries=training,
            evaluation_summaries=evaluations,
            baseline_metrics={str(horizon): 0.2 for horizon in (2, 3, 4, 5)},
            external_root=tmp_path,
        )


def test_validate_perception_full_selects_cal_points_and_pairs_c0(
    tmp_path: Path,
) -> None:
    campaign = _campaign_module()
    arms = ("C0", "S-BAL")
    updates = (0, 250, 750, 1500, 2250, 3000)
    training = {}
    evaluations = {}
    for variant in arms:
        run_dir = tmp_path / "training" / variant
        run_dir.mkdir(parents=True)
        checkpoints = []
        for update in updates[1:]:
            path = run_dir / f"update={update:04d}.ckpt"
            path.write_bytes(f"{variant}:{update}".encode())
            checkpoints.append(
                {
                    "bytes": path.stat().st_size,
                    "name": path.name,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
        last = run_dir / "last.ckpt"
        last.write_bytes(f"{variant}:last".encode())
        checkpoints.append(
            {
                "bytes": last.stat().st_size,
                "name": last.name,
                "sha256": hashlib.sha256(last.read_bytes()).hexdigest(),
            }
        )
        training[variant] = {
            "status": "COMPLETE",
            "variant": variant,
            "completed_global_step": 3000,
            "gpu_hours": 4.0,
            "progress": {
                "schema_version": "perception-gain-exact-resume-v1",
                "completed_local_batches": 24000,
                "completed_optimizer_updates": 3000,
                "next_global_draw_index": 96000,
            },
            "checkpoints": checkpoints,
        }
        records = {record["name"]: record for record in checkpoints}
        for update in updates:
            value = 0.2
            if variant == "C0" and update == 3000:
                value = 0.205
            if variant == "S-BAL" and update == 2250:
                value = 0.208
            checkpoint_sha = (
                "a" * 64
                if update == 0
                else records[f"update={update:04d}.ckpt"]["sha256"]
            )
            evaluations[(variant, update)] = {
                "status": "PASS",
                "variant": variant,
                "optimizer_update": update,
                "data_role": "CAL",
                "expected_logical_unit_count": 23,
                "completed_logical_unit_count": 23,
                "gpu_hours": 0.1,
                "t2_same_forward_parity": {"status": "PASS"},
                "candidate": {
                    "method_id": variant,
                    "optimizer_update": update,
                    "new_parameter_count": 0,
                    "metrics": {str(horizon): value for horizon in (2, 3, 4, 5)},
                    "coverage_status": "COMPLETE",
                    "checkpoint": (
                        "external:r1_checkpoint"
                        if update == 0
                        else f"external:training/{variant}/update={update:04d}.ckpt"
                    ),
                    "checkpoint_sha256": checkpoint_sha,
                    "expected_logical_units": 23,
                    "completed_logical_units": 23,
                },
            }

    result = campaign.validate_perception_full_stage(
        completed_full_arms=arms,
        training_summaries=training,
        evaluation_summaries=evaluations,
        baseline_metrics={str(horizon): 0.2 for horizon in (2, 3, 4, 5)},
        external_root=tmp_path,
    )

    assert result["status"] == "PASS"
    assert result["cal_checkpoint_selection"]["C0"]["optimizer_update"] == 3000
    assert result["cal_checkpoint_selection"]["S-BAL"]["optimizer_update"] == 2250
    assert result["incremental_evaluation_gpu_hours"] == pytest.approx(0.8)
    paired = result["paired_vs_c0"]
    assert len(paired) == len(updates)
    assert next(row for row in paired if row["optimizer_update"] == 2250)[
        "S_mean"
    ] == pytest.approx(0.008)


def test_validate_perception_selection_locks_exact_sel_parent() -> None:
    campaign = _campaign_module()

    def candidate(method: str, update: int, value: float) -> dict[str, object]:
        return {
            "method_id": method,
            "optimizer_update": update,
            "new_parameter_count": 0,
            "metrics": {str(horizon): value for horizon in (2, 3, 4, 5)},
            "coverage_status": "COMPLETE",
            "checkpoint": (
                "external:r1_checkpoint"
                if update == 0
                else f"external:training/{method}/update={update:04d}.ckpt"
            ),
            "checkpoint_sha256": hashlib.sha256(method.encode()).hexdigest(),
            "expected_logical_units": 24,
            "completed_logical_units": 24,
        }

    selected = {
        "C0": candidate("C0", 3000, 0.205),
        "S-BAL": candidate("S-BAL", 2250, 0.208),
    }
    baseline = candidate("C0", 0, 0.2)
    evaluations = {}
    for method, row in {"R1": baseline, **selected}.items():
        summary_method = "C0" if method == "R1" else method
        evaluations[method] = {
            "status": "PASS",
            "variant": summary_method,
            "optimizer_update": row["optimizer_update"],
            "data_role": "SEL",
            "expected_logical_unit_count": 24,
            "completed_logical_unit_count": 24,
            "gpu_hours": 0.1,
            "t2_same_forward_parity": {"status": "PASS"},
            "candidate": row,
        }

    result = campaign.validate_perception_selection_stage(
        cal_selection={
            "status": "PASS",
            "completed_full_arms": ["C0", "S-BAL"],
            "cal_checkpoint_selection": selected,
        },
        evaluation_summaries=evaluations,
    )

    assert result["status"] == "PASS"
    assert result["selection"]["status"] == "NEW_PERCEPTION_SELECTED"
    assert result["lock"]["selected_method_id"] == "S-BAL"
    assert result["lock"]["parent_candidate"]["optimizer_update"] == 2250
    assert result["evaluation_gpu_hours"] == pytest.approx(0.3)

    evaluations["S-BAL"]["candidate"] = candidate("S-BAL", 3000, 0.208)
    with pytest.raises(campaign.CampaignError, match="CAL-selected identity"):
        campaign.validate_perception_selection_stage(
            cal_selection={
                "status": "PASS",
                "completed_full_arms": ["C0", "S-BAL"],
                "cal_checkpoint_selection": selected,
            },
            evaluation_summaries=evaluations,
        )


def test_validate_refinement_selects_cal_checkpoint_then_applies_sel_gate(
    tmp_path: Path,
) -> None:
    campaign = _campaign_module()
    parent_sha = hashlib.sha256(b"parent").hexdigest()
    parent = {
        "method_id": "S-BAL",
        "source_variant": "S-BAL",
        "optimizer_update": 2250,
        "new_parameter_count": 0,
        "metrics": {str(horizon): 0.2 for horizon in (2, 3, 4, 5)},
        "coverage_status": "COMPLETE",
        "checkpoint": "external:training/S-BAL/update=2250.ckpt",
        "checkpoint_sha256": parent_sha,
        "expected_logical_units": 24,
        "completed_logical_units": 24,
    }
    lock = {
        "schema_version": "perception-lock-v1",
        "status": "LOCKED",
        "selected_method_id": "S-BAL",
        "parent_candidate": parent,
        "parent_cal_candidate": {
            **parent,
            "expected_logical_units": 23,
            "completed_logical_units": 23,
        },
    }
    shard = tmp_path / "training/refiner/data/refiner-0000.pt"
    shard.parent.mkdir(parents=True)
    shard.write_bytes(b"soft records")
    data_manifest = {
        "status": "PASS",
        "variant": "S-BAL",
        "optimizer_update": 2250,
        "checkpoint": parent["checkpoint"],
        "checkpoint_sha256": parent_sha,
        "selected_reference_count": 8,
        "completed_reference_count": 8,
        "candidate_count": 16,
        "cache_bytes": shard.stat().st_size,
        "gpu_hours": 0.2,
        "shards": [
            {
                "external_reference": "external:training/refiner/data/refiner-0000.pt",
                "bytes": shard.stat().st_size,
                "sha256": hashlib.sha256(shard.read_bytes()).hexdigest(),
                "record_count": 16,
            }
        ],
    }
    model_root = tmp_path / "training/refiner/model"
    model_root.mkdir(parents=True)
    records = {}
    for update in (0, 500, 1000, 1500):
        checkpoint = model_root / f"update={update:04d}.ckpt"
        checkpoint.write_bytes(f"refiner:{update}".encode())
        records[update] = {
            "path": checkpoint,
            "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        }
    training = {
        "status": "COMPLETE",
        "completed_updates": 1500,
        "checkpoint": "update=1500.ckpt",
        "checkpoint_sha256": records[1500]["sha256"],
        "reference_count": 8,
        "candidate_count": 16,
        "gpu_hours": 0.1,
    }

    def evaluation(role: str, update: int, value: float) -> dict[str, object]:
        logical_units = 23 if role == "CAL" else 24
        return {
            "status": "PASS",
            "data_role": role,
            "parent": {
                "variant": "S-BAL",
                "optimizer_update": 2250,
                "checkpoint": parent["checkpoint"],
                "checkpoint_sha256": parent_sha,
            },
            "refiner": {
                "updates": update,
                "checkpoint": (
                    f"external:training/refiner/model/update={update:04d}.ckpt"
                ),
                "sha256": records[update]["sha256"],
            },
            "candidate": {
                "method_id": "R-REFINE",
                "optimizer_update": update,
                "new_parameter_count": 16641,
                "metrics": {str(horizon): value for horizon in (2, 3, 4, 5)},
                "coverage_status": "COMPLETE",
                "checkpoint": (
                    f"external:training/refiner/model/update={update:04d}.ckpt"
                ),
                "checkpoint_sha256": records[update]["sha256"],
                "expected_logical_units": logical_units,
                "completed_logical_units": logical_units,
            },
            "parent_metrics": {str(horizon): 0.2 for horizon in (2, 3, 4, 5)},
            "expected_logical_unit_count": logical_units,
            "completed_logical_unit_count": logical_units,
            "mask_change_audit": {
                "candidate_count": 20,
                "changed_candidate_count": 0 if update == 0 else 2,
                "changed_point_count": 0 if update == 0 else 10,
            },
            "gpu_hours": 0.05,
        }

    cal = {
        0: evaluation("CAL", 0, 0.2),
        500: evaluation("CAL", 500, 0.202),
        1000: evaluation("CAL", 1000, 0.204),
        1500: evaluation("CAL", 1500, 0.203),
    }
    sel = evaluation("SEL", 1000, 0.204)

    result = campaign.validate_refinement_stage(
        perception_lock=lock,
        data_manifest=data_manifest,
        training_summary=training,
        cal_evaluations=cal,
        sel_evaluation=sel,
        external_root=tmp_path,
    )

    assert result["status"] == "PASS"
    assert result["cal_selected"]["optimizer_update"] == 1000
    assert result["selection"]["status"] == "REFINER_SELECTED"
    assert result["selection"]["enabled"] is True
    assert result["cache_bytes"] == shard.stat().st_size
    assert result["refinement_gpu_hours"] == pytest.approx(0.3)
    assert result["evaluation_gpu_hours"] == pytest.approx(0.25)

    cal[0]["mask_change_audit"]["changed_point_count"] = 1
    with pytest.raises(campaign.CampaignError, match="no-op"):
        campaign.validate_refinement_stage(
            perception_lock=lock,
            data_manifest=data_manifest,
            training_summary=training,
            cal_evaluations=cal,
            sel_evaluation=sel,
            external_root=tmp_path,
        )


def test_build_final_lock_freezes_recipe_and_refiner_decision() -> None:
    campaign = _campaign_module()
    parent = {
        "method_id": "C0",
        "source_variant": "C0",
        "optimizer_update": 3000,
        "checkpoint": "external:training/C0/update=3000.ckpt",
        "checkpoint_sha256": "a" * 64,
    }
    perception_lock = {
        "schema_version": "perception-lock-v1",
        "status": "LOCKED",
        "selected_method_id": "C0",
        "selection_status": "CONTINUATION_ONLY",
        "parent_candidate": parent,
    }
    refinement = {
        "schema_version": "perception-gain-refinement-stage-v1",
        "status": "PASS",
        "cal_selected": {
            "method_id": "R-REFINE",
            "optimizer_update": 1000,
            "checkpoint": "external:training/refiner/model/update=1000.ckpt",
            "checkpoint_sha256": "b" * 64,
        },
        "selection": {
            "status": "KEEP_PARENT",
            "enabled": False,
            "selected": None,
            "comparison": {"S_mean": 0.001},
        },
    }

    lock = campaign.build_final_lock(
        perception_lock=perception_lock,
        refinement=refinement,
        campaign_identity={"parent_commit": "c" * 40},
        campaign_config_sha256="d" * 64,
        data_roles_sha256="e" * 64,
    )

    assert lock["schema_version"] == "perception-gain-final-lock-v1"
    assert lock["status"] == "LOCKED"
    assert lock["recipe"]["parent_variant"] == "C0"
    assert lock["recipe"]["refiner_enabled"] is False
    assert lock["recipe"]["refiner_checkpoint"] is None
    assert lock["evaluated_refiner_candidate"]["optimizer_update"] == 1000


def test_confirmation_summary_requires_frozen_populations_and_computes_gates() -> None:
    campaign = _campaign_module()

    def result(
        method: str,
        population: str,
        metrics: dict[int, float],
        *,
        units: int,
        prefixes: int,
        physical: str,
        by_horizon: dict[str, dict[str, int]],
    ) -> dict[str, object]:
        return {
            "status": "MEASURED",
            "method_id": method,
            "population_id": population,
            "population_hash": "a" * 64,
            "metrics": {str(key): value for key, value in metrics.items()},
            "coverage": {
                "expected_units": units,
                "completed_units": units,
                "expected_prefixes": prefixes,
                "completed_prefixes": prefixes,
            },
            "population_by_horizon": by_horizon,
            "physical_evaluation_id": physical,
            "gpu_hours": 0.1,
        }

    pb_population = {
        str(horizon): {"reference_count": 6, "logical_unit_count": 129}
        for horizon in (2, 3, 4, 5)
    }
    pb_metrics = {
        "FH-R1-native": {2: 0.20, 3: 0.19, 4: 0.18, 5: 0.17},
        "D0-R1": {2: 0.18, 3: 0.17, 4: 0.16, 5: 0.15},
        "C0-best-D0": {2: 0.19, 3: 0.18, 4: 0.17, 5: 0.16},
        "FH-P*-native": {2: 0.21, 3: 0.20, 4: 0.19, 5: 0.18},
        "P*-D0": {2: 0.20, 3: 0.19, 4: 0.18, 5: 0.17},
        "FINAL": {2: 0.22, 3: 0.21, 4: 0.20, 5: 0.19},
    }
    pb = {
        method: result(
            method,
            "PB-129",
            metrics,
            units=129,
            prefixes=516,
            physical=f"pb-{method}",
            by_horizon=pb_population,
        )
        for method, metrics in pb_metrics.items()
    }
    local_population = {"2": {"reference_count": 41, "logical_unit_count": 154}}
    local = {
        method: result(
            method,
            "LOCAL-T2-154",
            {2: value},
            units=154,
            prefixes=154,
            physical=f"local-{method}",
            by_horizon=local_population,
        )
        for method, value in {
            "R1-native": 0.20,
            "C0-native": 0.21,
            "P*-native": 0.22,
        }.items()
    }
    additional_population = {
        "2": {"reference_count": 40, "logical_unit_count": 111},
        "3": {"reference_count": 23, "logical_unit_count": 77},
        "4": {"reference_count": 8, "logical_unit_count": 32},
    }
    additional = {
        method: result(
            method,
            "ADDITIONAL-native-terminal",
            {2: 0.2, 3: 0.19, 4: 0.18},
            units=220,
            prefixes=220,
            physical=f"additional-{method}",
            by_horizon=additional_population,
        )
        for method in ("FH-R1-native", "P*-D0", "FINAL")
    }

    summary = campaign.build_confirmation_summary(
        pb=pb,
        local_t2=local,
        additional=additional,
    )

    assert summary["status"] == "PASS"
    assert summary["execution_status"] == "COMPLETE"
    assert summary["comparisons"]["local_t2_gain"] is True
    assert summary["comparisons"]["pb_all_t_vs_native_fh"] is True
    assert summary["comparisons"]["pb_all_t_vs_d0"] is True
    assert summary["deployment"]["recommended_method"] == "FINAL"
    assert summary["gpu_hours"] == pytest.approx(1.2)

    pb["FINAL"]["coverage"]["completed_prefixes"] = 515
    partial = campaign.build_confirmation_summary(
        pb=pb,
        local_t2=local,
        additional=additional,
    )
    assert partial["execution_status"] == "PARTIAL_WITH_BLOCKERS"
    assert partial["comparisons"]["pb_all_t_vs_native_fh"] is None
    assert partial["comparisons"]["pb_all_t_vs_d0"] is None


def test_confirmation_normalizer_handles_native_d0_local_and_refiner_outputs() -> None:
    campaign = _campaign_module()
    pooled_rows = [
        {"method": "raw", "reference": "all", "T": horizon, "t_mAP": 0.1}
        for horizon in (2, 3, 4, 5)
    ]
    native = campaign.normalize_confirmation_evaluation(
        {
            "schema_version": "perception-gain-native-evaluation-v1",
            "status": "PASS",
            "data_role": "PB",
            "population_manifest_sha256": "a" * 64,
            "expected_logical_unit_count": 129,
            "expected_prefix_count": 516,
            "completed_prefix_count": 516,
            "completed_units_by_horizon": {
                str(horizon): [f"unit-{index}" for index in range(129)]
                for horizon in (2, 3, 4, 5)
            },
            "population_by_horizon": {
                str(horizon): {
                    "reference_count": 6,
                    "logical_unit_count": 129,
                }
                for horizon in (2, 3, 4, 5)
            },
            "metric_rows": pooled_rows,
            "gpu_hours": 0.2,
        },
        method_id="FH-R1-native",
        population_id="PB-129",
        physical_evaluation_id="native-r1",
    )
    d0 = campaign.normalize_confirmation_evaluation(
        {
            "schema_version": "perception-gain-checkpoint-evaluation-v1",
            "status": "PASS",
            "data_role": "PB",
            "population_manifest_sha256": "a" * 64,
            "expected_logical_unit_count": 129,
            "completed_logical_unit_count": 129,
            "candidate": {"metrics": {str(horizon): 0.1 for horizon in (2, 3, 4, 5)}},
            "metric_rows": pooled_rows,
            "gpu_hours": 0.3,
        },
        method_id="D0-R1",
        population_id="PB-129",
        physical_evaluation_id="d0-r1",
    )
    local = campaign.normalize_confirmation_evaluation(
        {
            "schema_version": "perception-gain-local-t2-v1",
            "status": "PASS",
            "population_manifest_sha256": "b" * 64,
            "validation_sequence_count": 154,
            "metrics": {"t_mAP": 0.2},
            "gpu_hours": 0.4,
        },
        method_id="R1-native",
        population_id="LOCAL-T2-154",
        physical_evaluation_id="local-r1",
    )
    refiner = campaign.normalize_confirmation_evaluation(
        {
            "schema_version": "perception-refiner-evaluation-v1",
            "status": "PASS",
            "data_role": "ADDITIONAL",
            "population_manifest_sha256": "c" * 64,
            "expected_logical_unit_count": 220,
            "completed_logical_unit_count": 220,
            "candidate": {"metrics": {"2": 0.2, "3": 0.2, "4": 0.2}},
            "parent_metrics": {"2": 0.1, "3": 0.1, "4": 0.1},
            "population_by_horizon": {
                "2": {"reference_count": 40, "logical_unit_count": 111},
                "3": {"reference_count": 23, "logical_unit_count": 77},
                "4": {"reference_count": 8, "logical_unit_count": 32},
            },
            "metric_rows": [
                {
                    "method": method,
                    "reference": "all",
                    "T": horizon,
                    "t_mAP": value,
                }
                for method, value in (("PARENT", 0.1), ("R-REFINE", 0.2))
                for horizon in (2, 3, 4)
            ],
            "gpu_hours": 0.5,
        },
        method_id="FINAL",
        population_id="ADDITIONAL-native-terminal",
        physical_evaluation_id="additional-final",
        source_method="R-REFINE",
    )

    assert native["coverage"]["completed_units"] == 129
    assert native["metrics"]["5"] == pytest.approx(0.1)
    assert d0["coverage"]["completed_prefixes"] == 516
    assert local["metrics"] == {"2": pytest.approx(0.2)}
    assert refiner["coverage"]["expected_prefixes"] == 220
    assert refiner["metrics"]["4"] == pytest.approx(0.2)


def test_replication_decision_is_not_applicable_or_budget_gated() -> None:
    campaign = _campaign_module()
    r1_lock = {
        "status": "LOCKED",
        "recipe": {
            "parent_method_id": "R1",
            "parent_optimizer_update": 0,
            "refiner_enabled": False,
        },
    }
    selected_lock = {
        "status": "LOCKED",
        "recipe": {
            "parent_method_id": "C0",
            "parent_optimizer_update": 3000,
            "refiner_enabled": False,
        },
    }

    assert (
        campaign.build_replication_decision(
            final_lock=r1_lock,
            estimated_perception_gpu_hours=0.0,
            estimated_refinement_gpu_hours=0.0,
            remaining_perception_gpu_hours=100.0,
            remaining_refinement_gpu_hours=10.0,
        )["replication_status"]
        == "NOT_APPLICABLE"
    )
    assert (
        campaign.build_replication_decision(
            final_lock=selected_lock,
            estimated_perception_gpu_hours=20.0,
            estimated_refinement_gpu_hours=0.0,
            remaining_perception_gpu_hours=25.0,
            remaining_refinement_gpu_hours=10.0,
        )["decision"]
        == "RUN_REQUIRED"
    )
    assert (
        campaign.build_replication_decision(
            final_lock=selected_lock,
            estimated_perception_gpu_hours=20.0,
            estimated_refinement_gpu_hours=0.0,
            remaining_perception_gpu_hours=19.0,
            remaining_refinement_gpu_hours=10.0,
        )["replication_status"]
        == "NOT_RUN_BUDGET"
    )


def test_partial_finalization_preserves_stage_prefix_and_records_exact_blocker() -> (
    None
):
    campaign = _campaign_module()
    state = {
        "completed_stages": ["bootstrap", "bind", "foundation", "scorer"],
        "failures": [],
        "next_command": "resume command",
    }
    interruption = {
        "schema_version": "perception-gain-training-interruption-v1",
        "status": "RESUMABLE_EXTERNAL_INTERRUPTION",
        "checkpoint": "external:training/S-BAL/last.ckpt",
        "checkpoint_sha256": "a" * 64,
        "resume_progress": {"completed_optimizer_updates": 350},
        "resource_accounting": {"accounted_gpu_hours": 3.25},
        "failure": {
            "external_dependency": "192.168.100.102:2049",
            "external_dependency_status": "CONNECTION_REFUSED",
            "reason": "NFS unavailable",
        },
    }

    finalized = campaign.build_partial_finalization_state(
        state, interruption=interruption
    )

    assert finalized["completed_stages"] == state["completed_stages"]
    assert finalized["execution_status"] == "PARTIAL_WITH_BLOCKERS"
    assert finalized["stage"] == "perception_pilot"
    assert finalized["stage_status"] == "BLOCKED"
    assert finalized["failures"][0]["completed_optimizer_updates"] == 350
    assert finalized["failures"][0]["resume_checkpoint"] == (
        "external:training/S-BAL/last.ckpt"
    )
    assert finalized["failures"][0]["accounted_gpu_hours"] == pytest.approx(3.25)

    with pytest.raises(campaign.CampaignError, match="interruption"):
        campaign.build_partial_finalization_state(
            state, interruption={**interruption, "status": "COMPLETE"}
        )


def test_partial_cost_reconciliation_counts_completed_and_interrupted_work_once(
    tmp_path: Path,
) -> None:
    campaign = _campaign_module()
    artifacts = tmp_path / "artifacts"
    evaluations = artifacts / "training/pilot/evaluation/cal/C0"
    evaluations.mkdir(parents=True)
    (evaluations / "update=0250.json").write_text(
        json.dumps({"status": "PASS", "gpu_hours": 0.75}), encoding="utf-8"
    )
    external = tmp_path / "external"
    summary = external / "training/C0/run_summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text(
        json.dumps(
            {
                "status": "COMPLETE",
                "variant": "C0",
                "completed_global_step": 750,
                "gpu_hours": 4.5,
            }
        ),
        encoding="utf-8",
    )
    state = {
        "budget_used": {
            "perception_training_gpu_hours": 0.01,
            "foundation_and_evaluation_gpu_hours": 0.02,
        }
    }
    interruption = {"resource_accounting": {"accounted_gpu_hours": 3.25}}

    result = campaign.reconcile_partial_costs(
        state,
        artifact_root=artifacts,
        external_root=external,
        interruption=interruption,
    )

    assert result["budget_used"]["perception_training_gpu_hours"] == pytest.approx(7.76)
    assert result["budget_used"][
        "foundation_and_evaluation_gpu_hours"
    ] == pytest.approx(0.77)
    assert (
        campaign.reconcile_partial_costs(
            result,
            artifact_root=artifacts,
            external_root=external,
            interruption=interruption,
        )
        == result
    )
