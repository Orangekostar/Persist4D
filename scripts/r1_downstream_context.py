"""Frozen input and runtime context for R1 downstream validation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class R1ContextError(ValueError):
    """Raised when an R1 downstream input differs from the frozen contract."""


@dataclass
class R1Setup:
    contract: dict[str, Any]
    checkpoint: Path
    pretrained: Path
    protocol: object
    protocol_manifest: dict[str, object]
    p6a_config: dict[str, Any]
    runtime_config: object
    memory_config: object
    dataset: object
    collate: object
    local_provenance: dict[str, str]
    full_provenance: dict[str, str]
    device: object | None = None
    system: object | None = None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise R1ContextError(f"{name} must be a mapping")
    return dict(value)


def load_r1_contract(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise R1ContextError("R1 contract cannot be decoded") from error
    contract = _mapping(value, name="R1 contract")
    if contract.get("schema_version") != 1 or contract.get("experiment") != (
        "persist4d_r1_downstream_validation_v1"
    ):
        raise R1ContextError("R1 contract identity differs")
    if contract.get("seed") != 45:
        raise R1ContextError("R1 contract seed differs")
    if contract.get("methods") != ["FullHistory", "B2", "B4"] or contract.get(
        "tracker_methods"
    ) != ["B2", "B4"]:
        raise R1ContextError("R1 contract methods differ")
    if contract.get("state_update_horizons") != [1, 2, 3, 4, 5] or contract.get(
        "report_horizons"
    ) != [2, 4, 5]:
        raise R1ContextError("R1 contract horizons differ")
    if contract.get("score_reducers") != ["mean", "latest", "max"]:
        raise R1ContextError("R1 contract score reducers differ")
    return contract


def validate_external_file_identity(
    path: Path,
    *,
    expected_sha256: str,
    expected_bytes: int,
    require_digest_filename: bool,
    label: str,
) -> Path:
    candidate = path.expanduser()
    if candidate.is_symlink() or not candidate.is_file():
        raise R1ContextError(f"{label} must be a regular non-symlink file")
    resolved = candidate.resolve(strict=True)
    if resolved.stat().st_size != expected_bytes:
        raise R1ContextError(f"{label} byte size differs")
    if (
        len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise R1ContextError(f"{label} expected SHA256 is invalid")
    if require_digest_filename and resolved.stem != expected_sha256:
        raise R1ContextError(f"{label} filename must equal its audited SHA256")
    return resolved


def validate_data_working_directory(data_root: Path, *, cwd: Path | None = None) -> Path:
    root = data_root.expanduser().resolve(strict=True)
    current = Path.cwd().resolve() if cwd is None else cwd.expanduser().resolve(strict=True)
    if current != root:
        raise R1ContextError("runtime working directory must equal the explicit data root")
    return root


def validate_r1_checkpoint_payload(
    payload: object, contract: Mapping[str, object]
) -> dict[str, object]:
    checkpoint = _mapping(contract.get("checkpoint"), name="checkpoint contract")
    root = _mapping(payload, name="R1 checkpoint payload")
    state_dict = root.get("state_dict")
    if not isinstance(state_dict, Mapping):
        raise R1ContextError("R1 checkpoint is missing its state_dict")
    stored_epoch = root.get("epoch")
    global_step = root.get("global_step")
    expected_epoch = int(checkpoint["completed_epoch"])
    expected_step = int(checkpoint["completed_step"])
    if (
        isinstance(stored_epoch, bool)
        or not isinstance(stored_epoch, int)
        or stored_epoch + 1 != expected_epoch
        or isinstance(global_step, bool)
        or not isinstance(global_step, int)
        or global_step != expected_step
    ):
        raise R1ContextError("R1 checkpoint epoch/step differs from contract")
    expected_entries = int(checkpoint["state_dict_entries"])
    if len(state_dict) != expected_entries:
        raise R1ContextError("R1 checkpoint state_dict entry count differs")
    return {
        "status": "pass",
        "completed_epoch": expected_epoch,
        "completed_step": expected_step,
        "state_dict_entries": expected_entries,
    }


def _load_r1_system(
    config: object,
    checkpoint: Path,
    device: object,
    contract: Mapping[str, object],
) -> object:
    import torch

    from trainer.trainer import InstanceSegmentation

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    validate_r1_checkpoint_payload(payload, contract)
    state_dict = payload["state_dict"]
    system = InstanceSegmentation(config)
    system.load_state_dict(state_dict, strict=True)
    system.to(device)
    system.eval()
    return system


def validate_protocol_b(path: Path, contract: Mapping[str, object]) -> dict[str, object]:
    protocol_contract = _mapping(contract.get("protocol"), name="protocol contract")
    if path.is_symlink() or not path.is_file():
        raise R1ContextError("Protocol-B manifest must be a regular non-symlink file")
    observed_sha256 = _sha256_file(path)
    if observed_sha256 != protocol_contract.get("sha256"):
        raise R1ContextError("Protocol-B SHA256 differs")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise R1ContextError("Protocol-B manifest cannot be decoded") from error
    root = _mapping(manifest, name="Protocol-B manifest")
    protocol = _mapping(root.get("protocol"), name="Protocol-B settings")
    masters = root.get("masters")
    if isinstance(masters, (str, bytes)) or not isinstance(masters, Sequence):
        raise R1ContextError("Protocol-B masters must be a sequence")
    orders = protocol_contract.get("orders")
    references = {
        master.get("reference_scene_id")
        for master in masters
        if isinstance(master, Mapping)
    }
    expected = {
        "master_count": 43,
        "order_count": 3,
        "sequence_count": 129,
        "reference_scene_cluster_count": 6,
        "cache_entry_count": 645,
    }
    observed = {
        "master_count": len(masters),
        "order_count": len(orders) if isinstance(orders, Sequence) else -1,
        "sequence_count": len(masters) * 3,
        "reference_scene_cluster_count": len(references),
        "cache_entry_count": len(masters) * 3 * 5,
    }
    if (
        observed != expected
        or protocol.get("seed") != 45
        or protocol.get("order_variants") != orders
        or protocol.get("horizons") != [2, 3, 4, 5]
        or any(
            not isinstance(master, Mapping)
            or not isinstance(master.get("orders"), Mapping)
            or list(master["orders"]) != list(orders)
            for master in masters
        )
    ):
        raise R1ContextError("Protocol-B structure differs")
    return {**observed, "sha256": observed_sha256, "status": "pass"}


def compose_r1_runtime_config(pretrained_path: Path) -> tuple[Any, Any]:
    from scripts.evaluate_persist4d import _compose_runtime_config

    runtime, memory = _compose_runtime_config()
    runtime.backbone.name = str(pretrained_path.expanduser().resolve(strict=True))
    return runtime, memory


def validate_tracker_settings(
    contract: Mapping[str, object], p6a_config: Mapping[str, object]
) -> dict[str, object]:
    expected = _mapping(contract.get("tracker_settings"), name="tracker settings")
    baselines = _mapping(p6a_config.get("baselines"), name="P6-A baselines")
    for method in ("B2", "B4"):
        actual = _mapping(baselines.get(method.lower()), name=f"P6-A {method}")
        if actual != _mapping(expected.get(method), name=f"contract {method}"):
            raise R1ContextError(f"P6-A {method} settings differ from contract")
    return {"methods": ["B2", "B4"], "status": "pass"}


def build_r1_cache_provenance(
    *,
    contract: Mapping[str, object],
    source_commit: str,
    config_documents: Mapping[str, bytes],
    protocol_manifest: Mapping[str, object],
) -> tuple[dict[str, str], dict[str, str]]:
    from scripts.p6a_cache import config_documents_sha256

    if len(source_commit) != 40 or any(
        character not in "0123456789abcdef" for character in source_commit
    ):
        raise R1ContextError("source commit must be a lowercase SHA-1")
    checkpoint = _mapping(contract.get("checkpoint"), name="checkpoint contract")
    protocol = _mapping(contract.get("protocol"), name="protocol contract")
    protocol_bytes = json.dumps(
        protocol_manifest,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    local = {
        "source_commit": source_commit,
        "checkpoint_sha256": str(checkpoint["sha256"]),
        "config_sha256": config_documents_sha256(config_documents),
        "dataset_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
    }
    full = {
        "source_commit": source_commit,
        "checkpoint_sha256": local["checkpoint_sha256"],
        "config_sha256": local["config_sha256"],
        "protocol_sha256": str(protocol["sha256"]),
    }
    return local, full


def build_r1_setup(
    *,
    contract_path: Path,
    protocol_path: Path,
    checkpoint_path: Path,
    pretrained_path: Path,
    metadata_path: Path,
    data_root: Path,
    source_commit: str,
    device_name: str | None,
) -> R1Setup:
    import hydra
    from omegaconf import OmegaConf

    from scripts.evaluate_persist4d import _validate_cuda_device
    from scripts.evaluate_persist4d_p6a import expected_cache_keys
    from scripts.p6a_protocol import build_protocol_b
    from scripts.system_comparison_inference import deterministic_inference_runtime

    contract = load_r1_contract(contract_path)
    validate_protocol_b(protocol_path, contract)
    checkpoint_contract = _mapping(
        contract.get("checkpoint"), name="checkpoint contract"
    )
    pretrained_contract = _mapping(
        contract.get("pretrained"), name="pretrained contract"
    )
    checkpoint = validate_external_file_identity(
        checkpoint_path,
        expected_sha256=str(checkpoint_contract["sha256"]),
        expected_bytes=int(checkpoint_contract["bytes"]),
        require_digest_filename=True,
        label="R1 checkpoint",
    )
    pretrained = validate_external_file_identity(
        pretrained_path,
        expected_sha256=str(pretrained_contract["sha256"]),
        expected_bytes=int(pretrained_contract["bytes"]),
        require_digest_filename=False,
        label="Concerto pretrain",
    )
    data_repository = validate_data_working_directory(data_root)
    p6a_path = Path(__file__).resolve().parents[1] / "conf/p6a/default.yaml"
    p6a_bytes = p6a_path.read_bytes()
    p6a_config = _mapping(
        yaml.safe_load(p6a_bytes), name="P6-A runtime configuration"
    )
    protocol_config = _mapping(
        p6a_config.get("protocol_b"), name="P6-A protocol configuration"
    )
    sources = _mapping(protocol_config.get("sources"), name="P6-A protocol sources")
    sequence_database = data_repository / (
        "data/processed/rio/sequence_database_sliding_5.yaml"
    )
    scan_metadata = data_repository / "data/processed/rio/validation_database.yaml"
    metadata = metadata_path.expanduser().resolve(strict=True)
    for name, path, digest_key in (
        ("sequence database", sequence_database, "sequence_database_sha256"),
        ("scan metadata", scan_metadata, "scan_metadata_sha256"),
        ("3RScan metadata", metadata, "metadata_sha256"),
    ):
        if _sha256_file(path) != sources.get(digest_key):
            raise R1ContextError(f"frozen {name} SHA256 differs")
    protocol = build_protocol_b(
        sequence_database,
        scan_metadata,
        metadata_path=metadata,
        expected_split=str(protocol_config["split"]),
        expected_master_count=int(protocol_config["expected_master_count"]),
        expected_cluster_count=int(
            protocol_config["expected_reference_scene_clusters"]
        ),
        horizons=tuple(int(value) for value in protocol_config["horizons"]),
        seed=int(protocol_config["seed"]),
        require_supervised=bool(protocol_config["require_supervised"]),
        substitution_policy=str(protocol_config["substitution_policy"]),
    )
    if len(expected_cache_keys(protocol)) != 645:
        raise R1ContextError("Protocol-B runtime key coverage differs")
    protocol_manifest = json.loads(protocol_path.read_text(encoding="utf-8"))
    validate_tracker_settings(contract, p6a_config)
    runtime_config, memory_config = compose_r1_runtime_config(pretrained)
    runtime_bytes = OmegaConf.to_yaml(
        runtime_config,
        resolve=True,
        sort_keys=True,
    ).encode("utf-8")
    local_provenance, full_provenance = build_r1_cache_provenance(
        contract=contract,
        source_commit=source_commit,
        config_documents={"p6a": p6a_bytes, "runtime": runtime_bytes},
        protocol_manifest=protocol_manifest,
    )
    dataset_config = OmegaConf.create(
        OmegaConf.to_container(
            runtime_config.data.validation_dataset,
            resolve=True,
        )
    )
    dataset_config.temporal_window = 5
    rio_root = data_repository / "data/processed/rio"
    dataset_config.data_dir = str(rio_root)
    dataset_config.label_db_filepath = str(rio_root / "label_database.yaml")
    dataset_config.color_mean_std = str(rio_root / "color_mean_std.yaml")
    dataset = hydra.utils.instantiate(dataset_config)
    collate = hydra.utils.instantiate(runtime_config.data.validation_collation)
    setup = R1Setup(
        contract=contract,
        checkpoint=checkpoint,
        pretrained=pretrained,
        protocol=protocol,
        protocol_manifest=dict(protocol_manifest),
        p6a_config=p6a_config,
        runtime_config=runtime_config,
        memory_config=memory_config,
        dataset=dataset,
        collate=collate,
        local_provenance=local_provenance,
        full_provenance=full_provenance,
    )
    if device_name is None:
        return setup
    device = _validate_cuda_device(device_name)
    with deterministic_inference_runtime(int(contract["seed"]), device):
        system = _load_r1_system(runtime_config, checkpoint, device, contract)
    setup.device = device
    setup.system = system
    return setup


__all__ = [
    "R1ContextError",
    "R1Setup",
    "build_r1_cache_provenance",
    "build_r1_setup",
    "compose_r1_runtime_config",
    "load_r1_contract",
    "validate_external_file_identity",
    "validate_data_working_directory",
    "validate_protocol_b",
    "validate_r1_checkpoint_payload",
    "validate_tracker_settings",
]
