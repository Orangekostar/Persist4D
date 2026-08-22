from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.system_comparison_protocol import (
    ProtocolBindingError,
    build_system_comparison_manifest,
    load_incumbent_config,
    validate_incumbent_binding,
    validate_system_comparison_manifest,
    write_canonical_json,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs/system_comparison/persist4d_incumbent.yaml"
PROTOCOL_PATH = REPO_ROOT / "artifacts/P6A/protocol_b_manifest.json"
CHECKPOINT_PATH = REPO_ROOT / "checkpoints/rescene4d_concerto_t2_repro.ckpt"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_incumbent_config_restores_exact_p6a_b4() -> None:
    config = load_incumbent_config(CONFIG_PATH)

    assert config["method"] == {
        "id": "B4",
        "name": "Persist4D Persistent-State",
        "implementation": "frozen_p5_persist4d",
    }
    assert config["memory"] == {
        "capacity": 100,
        "association_threshold": 0.5,
        "class_weight": 0.25,
        "background_class": 18,
        "update_rate": 0.2,
        "max_update_rate": 0.2,
        "confidence_threshold": 0.5,
        "mask_threshold": 0.5,
        "minimum_mask_support": 1,
    }
    assert config["profile"] == {
        "warmup_repeats": 5,
        "measured_repeats": 10,
        "subset_rule": "first_master_per_reference_scene_canonical",
    }
    assert config["statistics"] == {
        "cluster_unit": "reference_scene_id",
        "bootstrap_replicates": 10000,
        "seed": 45,
    }


def test_incumbent_binding_matches_frozen_files_and_results() -> None:
    binding = validate_incumbent_binding(
        CONFIG_PATH,
        repo_root=REPO_ROOT,
        checkpoint_path=CHECKPOINT_PATH,
    )

    assert binding["status"] == "pass"
    assert binding["p6a_source_commit"] == (
        "cee151a9dfc1c9aa038227bc4e179b671e739575"
    )
    assert binding["implementation_base_commit"] == (
        "73b83ced10a59c4ba755e94fad5fbf43c35d90e8"
    )
    assert binding["checkpoint_sha256"] == (
        "85ed1aba60320cd19798536b71b91dbc156b7ea60f838832bc0bbbdba131546e"
    )
    assert binding["p6a_config_sha256"] == (
        "dd7a9fccc098fc7e5faecc059d6202ca7f8a762aaee3e7182738a2d7f800aac2"
    )
    assert binding["p6a_protocol_manifest_sha256"] == _sha256(PROTOCOL_PATH)
    assert binding["reference_metrics"]["T5"]["t_mAP"] == pytest.approx(
        0.04449684917926788
    )


def test_incumbent_binding_fails_closed_on_tampered_reference(tmp_path: Path) -> None:
    config = load_incumbent_config(CONFIG_PATH)
    tampered = tmp_path / "strict_online_results.csv"
    source = REPO_ROOT / "artifacts/P6A/strict_online_results.csv"
    tampered.write_bytes(source.read_bytes() + b"\n")
    config["sources"]["p6a_strict_results"]["reference"] = str(tampered)
    config_path = tmp_path / "incumbent.yaml"

    import yaml

    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    with pytest.raises(ProtocolBindingError, match="p6a_strict_results.*SHA256"):
        validate_incumbent_binding(
            config_path,
            repo_root=REPO_ROOT,
            checkpoint_path=CHECKPOINT_PATH,
        )


def test_system_manifest_is_exact_common_prefix_copy() -> None:
    binding = validate_incumbent_binding(
        CONFIG_PATH,
        repo_root=REPO_ROOT,
        checkpoint_path=CHECKPOINT_PATH,
    )
    manifest = build_system_comparison_manifest(
        PROTOCOL_PATH,
        incumbent_binding=binding,
    )
    validated = validate_system_comparison_manifest(
        manifest,
        source_protocol_path=PROTOCOL_PATH,
    )

    assert validated["comparison_prefix_count"] == 43 * 3 * 4
    assert validated["identity_initialization_count"] == 43 * 3
    assert len(validated["masters"]) == 43
    assert len({row["reference_scene_id"] for row in validated["masters"]}) == 6
    assert validated["protocol"]["order_variants"] == [
        "canonical",
        "reverse",
        "sha256_seed45",
    ]
    assert validated["protocol"]["horizons"] == [2, 3, 4, 5]
    assert validated["source_protocol"]["sha256"] == _sha256(PROTOCOL_PATH)

    original = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    assert validated["masters"] == original["masters"]
    for master in validated["masters"]:
        for order_name in validated["protocol"]["order_variants"]:
            order = master["orders"][order_name]
            for horizon in validated["protocol"]["horizons"]:
                prefix = order["prefixes"][str(horizon)]
                assert prefix["scan_ids"] == order["visit_order"][:horizon]
                assert prefix["scan_indices"] == order["scan_indices"][:horizon]


def test_system_manifest_rejects_prefix_mutation() -> None:
    binding = validate_incumbent_binding(
        CONFIG_PATH,
        repo_root=REPO_ROOT,
        checkpoint_path=CHECKPOINT_PATH,
    )
    manifest = build_system_comparison_manifest(
        PROTOCOL_PATH,
        incumbent_binding=binding,
    )
    tampered = copy.deepcopy(manifest)
    tampered["masters"][0]["orders"]["canonical"]["prefixes"]["3"][
        "scan_ids"
    ][-1] = "scene9999_99"

    with pytest.raises(ProtocolBindingError, match="content digest|exact prefix"):
        validate_system_comparison_manifest(
            tampered,
            source_protocol_path=PROTOCOL_PATH,
        )


def test_canonical_manifest_writer_refuses_overwrite(tmp_path: Path) -> None:
    binding = validate_incumbent_binding(
        CONFIG_PATH,
        repo_root=REPO_ROOT,
        checkpoint_path=CHECKPOINT_PATH,
    )
    manifest = build_system_comparison_manifest(
        PROTOCOL_PATH,
        incumbent_binding=binding,
    )
    output = tmp_path / "manifest.json"
    write_canonical_json(output, manifest)

    payload = output.read_bytes()
    assert payload.endswith(b"\n")
    assert json.loads(payload) == manifest
    with pytest.raises(FileExistsError):
        write_canonical_json(output, manifest)
