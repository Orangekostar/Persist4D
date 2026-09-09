import csv
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from scripts.task_memory_contracts import (
    AssetBindings,
    TaskMemoryContractError,
    freeze_task_memory_contracts,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REVIEWED_PARENT = "32a51e11b51043ec5a5825215669ef7b0ea03bc2"
BRANCH = "research/persist4d-task-memory-retention-v2"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_scan(
    data_root: Path,
    *,
    split: str,
    scene: int,
    sub_scene: int,
    readable: bool = True,
) -> dict[str, object]:
    scan_id = f"scene{scene:04d}_{sub_scene:02d}"
    relative_scan = Path("processed/rio") / split / f"{scene:04d}_{sub_scene:02d}.npy"
    relative_instance = (
        Path("processed/rio/instance_gt") / split / f"{scan_id}.txt"
    )
    if readable:
        (data_root / relative_scan).parent.mkdir(parents=True, exist_ok=True)
        (data_root / relative_scan).write_bytes(scan_id.encode("ascii"))
        (data_root / relative_instance).parent.mkdir(parents=True, exist_ok=True)
        (data_root / relative_instance).write_text("1\n", encoding="ascii")
    return {
        "filepath": f"data/{relative_scan.as_posix()}",
        "instance_gt_filepath": f"data/{relative_instance.as_posix()}",
        "scene": scene,
        "sub_scene": sub_scene,
    }


def _fixture(tmp_path: Path) -> tuple[Path, AssetBindings]:
    data_root = tmp_path / "data"
    rio_root = data_root / "processed/rio"
    rio_root.mkdir(parents=True)
    train_records = [
        _write_scan(data_root, split="train", scene=0, sub_scene=0),
        _write_scan(
            data_root,
            split="train",
            scene=0,
            sub_scene=1,
            readable=False,
        ),
        _write_scan(data_root, split="train", scene=1, sub_scene=0),
        _write_scan(data_root, split="train", scene=1, sub_scene=1),
    ]
    validation_records = [
        *[
            _write_scan(data_root, split="validation", scene=2, sub_scene=index)
            for index in range(5)
        ],
        *[
            _write_scan(data_root, split="validation", scene=3, sub_scene=index)
            for index in range(3)
        ],
    ]
    (rio_root / "train_database.yaml").write_text(
        yaml.safe_dump(train_records, sort_keys=True), encoding="utf-8"
    )
    (rio_root / "validation_database.yaml").write_text(
        yaml.safe_dump(validation_records, sort_keys=True), encoding="utf-8"
    )

    metadata = tmp_path / "3RScan.json"
    _write_json(
        metadata,
        [
            {
                "reference": "ref-adapt",
                "scene": 0,
                "scans": [{"reference": "scan-adapt-1"}],
                "type": "train",
            },
            {
                "reference": "ref-dev",
                "scene": 1,
                "scans": [{"reference": "scan-dev-1"}],
                "type": "train",
            },
            {
                "reference": "ref-protocol",
                "scene": 2,
                "scans": [
                    {"reference": f"scan-protocol-{index}"} for index in range(1, 5)
                ],
                "type": "validation",
            },
            {
                "reference": "ref-native",
                "scene": 3,
                "scans": [
                    {"reference": f"scan-native-{index}"} for index in range(1, 3)
                ],
                "type": "validation",
            },
        ],
    )
    split = tmp_path / "split.json"
    _write_json(
        split,
        {
            "reference_split": {
                "adaptation_reference_ids": ["ref-adapt"],
                "development_reference_ids": ["ref-dev"],
            },
            "protocol_b_reference_ids": ["ref-protocol"],
        },
    )
    protocol = tmp_path / "protocol.json"
    _write_json(
        protocol,
        {
            "masters": [
                {
                    "master_sequence_id": "-".join(
                        f"scene0002_{index:02d}" for index in range(5)
                    ),
                    "reference_scene_id": "ref-protocol",
                    "scan_ids": [f"scene0002_{index:02d}" for index in range(5)],
                }
            ],
            "protocol": {"order_variants": ["canonical", "reverse", "sha256_seed45"]},
        },
    )
    r1 = tmp_path / "r1.ckpt"
    concerto = tmp_path / "concerto.pth"
    r1.write_bytes(b"r1-test-weight")
    concerto.write_bytes(b"concerto-test-weight")
    run_root = tmp_path / "runs"
    run_root.mkdir()

    evidence = PROJECT_ROOT / "docs/task_memory_v2/01_EVIDENCE_AND_CODE_MAP.md"
    prompt = PROJECT_ROOT / "docs/task_memory_v2/Persist4D_Codex_TaskMemory_Retention_V2.md"
    old_handoff = PROJECT_ROOT / "artifacts/allt_task_superiority_v1/HANDOFF.md"
    config = {
        "schema_version": "task-memory-retention-v2-config-v1",
        "experiment": "task_memory_retention_v2",
        "repository": {"branch": BRANCH, "reviewed_parent": REVIEWED_PARENT},
        "sources": {
            "evidence_map": {
                "reference": "repo:docs/task_memory_v2/01_EVIDENCE_AND_CODE_MAP.md",
                "sha256": _sha256(evidence),
            },
            "task_prompt": {
                "reference": "repo:docs/task_memory_v2/Persist4D_Codex_TaskMemory_Retention_V2.md",
                "sha256": _sha256(prompt),
            },
            "old_handoff": {
                "reference": "repo:artifacts/allt_task_superiority_v1/HANDOFF.md",
                "sha256": _sha256(old_handoff),
            },
            "frozen_split": {"reference": "config:split.json", "sha256": _sha256(split)},
            "protocol_b": {
                "reference": "config:protocol.json",
                "sha256": _sha256(protocol),
            },
        },
        "data": {
            "metadata_reference": "external:rio_metadata",
            "train_database_reference": "external:data_root/processed/rio/train_database.yaml",
            "validation_database_reference": "external:data_root/processed/rio/validation_database.yaml",
            "expected_protocol_masters": 1,
            "expected_protocol_order_units": 3,
            "expected_protocol_references": 1,
        },
        "assets": {
            "r1_checkpoint": {
                "bytes": r1.stat().st_size,
                "sha256": _sha256(r1),
            },
            "concerto_pretrained": {
                "bytes": concerto.stat().st_size,
                "sha256": _sha256(concerto),
            },
        },
        "training": {
            "seed": 45,
            "devices": 2,
            "physical_episode_batch_per_gpu": 1,
            "gradient_accumulation": 4,
            "optimizer_updates": 3000,
            "pilot_updates": 300,
            "evaluation_updates": [0, 750, 1500, 2250, 3000],
            "precision": "32-true",
            "episode_buckets": {
                "single_scan": 0.2,
                "T2": 0.2,
                "T3": 0.2,
                "T4": 0.2,
                "T5": 0.2,
            },
        },
        "budgets": {
            "training_gpu_hours": 120,
            "evaluation_gpu_hours": 30,
            "evaluation_cache_bytes": 40 * 1024**3,
            "permanent_state_bytes": 2 * 1024**2,
        },
        "output": {"primary_policy": "lag1", "control_policy": "commit0"},
    }
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    return config_path, AssetBindings(
        data_root=data_root,
        rio_metadata=metadata,
        r1_checkpoint=r1,
        concerto_pretrained=concerto,
        run_root=run_root,
    )


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    unsigned = dict(value)
    observed = unsigned.pop("content_sha256")
    expected = hashlib.sha256(
        json.dumps(
            unsigned,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()
    assert observed == expected
    return value


def test_freeze_locks_repository_assets_budget_and_output_policy(tmp_path: Path) -> None:
    config, assets = _fixture(tmp_path)
    output = tmp_path / "artifacts"

    bundle = freeze_task_memory_contracts(
        config_path=config, assets=assets, output_root=output
    )

    assert bundle.output_root == output
    start = _load_json(output / "START_STATE.json")
    budget = _load_json(output / "BUDGET_CONTRACT.json")
    output_contract = (output / "OUTPUT_CONTRACT.md").read_text(encoding="utf-8")
    repository = dict(start["repository"])
    start_head = repository.pop("start_head")
    assert isinstance(start_head, str) and len(start_head) == 40
    assert repository == {
        "branch": BRANCH,
        "reviewed_parent": REVIEWED_PARENT,
        "reviewed_parent_is_ancestor": True,
    }
    assert start["assets"]["r1_checkpoint"] == {
        "bytes": len(b"r1-test-weight"),
        "logical_reference": "external:r1_checkpoint",
        "sha256": hashlib.sha256(b"r1-test-weight").hexdigest(),
        "status": "VERIFIED",
    }
    assert start["environment"]["storage"] == {
        "data_root": {
            "free_bytes": pytest.approx(
                start["environment"]["storage"]["data_root"]["free_bytes"]
            ),
            "logical_reference": "external:data_root",
        },
        "run_root": {
            "free_bytes": pytest.approx(
                start["environment"]["storage"]["run_root"]["free_bytes"]
            ),
            "logical_reference": "external:run_root",
        },
    }
    assert start["legacy_inputs"] == {
        "allt_cache_root": "external:allt_task_superiority_v1/evaluation_cache",
        "allt_handoff": "repo:artifacts/allt_task_superiority_v1/HANDOFF.md",
        "allt_handoff_sha256": _sha256(
            PROJECT_ROOT / "artifacts/allt_task_superiority_v1/HANDOFF.md"
        ),
        "policy": "read_only",
    }
    assert budget["training"] == {
        "devices": 2,
        "effective_episode_batch": 8,
        "evaluation_updates": [0, 750, 1500, 2250, 3000],
        "gradient_accumulation": 4,
        "optimizer_updates": 3000,
        "physical_episode_batch_per_gpu": 1,
        "pilot_updates": 300,
        "precision": "32-true",
        "seed": 45,
    }
    assert budget["caps"]["training_gpu_hours"] == 120
    assert budget["caps"]["evaluation_gpu_hours"] == 30
    assert budget["caps"]["evaluation_cache_bytes"] == 40 * 1024**3
    assert budget["caps"]["permanent_state_bytes"] == 2 * 1024**2
    assert budget["episode_buckets"] == [
        {"fraction": 0.2, "name": name}
        for name in ("single_scan", "T2", "T3", "T4", "T5")
    ]
    assert "`commit0`" in output_contract
    assert "`lag1`" in output_contract
    assert "primary" in output_contract
    assert "exactly once" in output_contract

    public_bytes = b"".join(
        path.read_bytes()
        for path in output.iterdir()
        if path.name != "external_assets.local.json"
    )
    assert str(tmp_path).encode() not in public_bytes
    local = json.loads((output / "external_assets.local.json").read_text())
    assert local["external:r1_checkpoint"] == str(assets.r1_checkpoint.resolve())
    assert local["external:run_root"] == str(assets.run_root.resolve())


def test_native_lengths_use_readable_files_and_roles_are_disjoint(tmp_path: Path) -> None:
    config, assets = _fixture(tmp_path)
    output = tmp_path / "artifacts"
    freeze_task_memory_contracts(config_path=config, assets=assets, output_root=output)

    data = _load_json(output / "DATA_CONTRACT.json")
    inventory = {
        row["reference_id"]: row
        for row in data["native_reference_inventory"]
    }
    assert inventory["ref-adapt"]["metadata_scan_count"] == 2
    assert inventory["ref-adapt"]["readable_supervised_scan_count"] == 1
    assert inventory["ref-adapt"]["max_native_horizon"] == 1
    assert inventory["ref-dev"]["readable_supervised_scan_count"] == 2
    assert inventory["ref-dev"]["role"] == "development"
    assert inventory["ref-protocol"]["role"] == "protocol_b_final"
    assert inventory["ref-native"]["role"] == "additional_native_refs"
    assert set(data["roles"]["adaptation_reference_ids"]).isdisjoint(
        data["roles"]["development_reference_ids"]
    )
    assert set(data["roles"]["adaptation_reference_ids"]).isdisjoint(
        data["roles"]["protocol_b_reference_ids"]
    )
    assert data["populations"]["protocol_b_common_129"] == {
        "master_count": 1,
        "order_unit_count": 3,
        "reference_count": 1,
    }

    with (output / "references_inventory.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    missing = next(row for row in rows if row["scan_id"] == "scene0000_01")
    assert missing["processed_readable"] == "False"
    assert missing["instance_gt_readable"] == "False"


def test_unresolvable_logical_reference_fails_closed(tmp_path: Path) -> None:
    config, assets = _fixture(tmp_path)
    value = yaml.safe_load(config.read_text(encoding="utf-8"))
    value["sources"]["protocol_b"]["reference"] = "config:missing.json"
    config.write_text(yaml.safe_dump(value, sort_keys=True), encoding="utf-8")

    with pytest.raises(TaskMemoryContractError, match="cannot resolve"):
        freeze_task_memory_contracts(
            config_path=config, assets=assets, output_root=tmp_path / "artifacts"
        )


def test_overlapping_training_and_frozen_roles_fail_closed(tmp_path: Path) -> None:
    config, assets = _fixture(tmp_path)
    split = tmp_path / "split.json"
    value = json.loads(split.read_text(encoding="utf-8"))
    value["reference_split"]["development_reference_ids"] = ["ref-adapt"]
    _write_json(split, value)
    config_value = yaml.safe_load(config.read_text(encoding="utf-8"))
    config_value["sources"]["frozen_split"]["sha256"] = _sha256(split)
    config.write_text(yaml.safe_dump(config_value, sort_keys=True), encoding="utf-8")

    with pytest.raises(TaskMemoryContractError, match="overlap"):
        freeze_task_memory_contracts(
            config_path=config, assets=assets, output_root=tmp_path / "artifacts"
        )
