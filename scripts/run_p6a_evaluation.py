"""Build the complete P6-A evidence bundle from one frozen prediction cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml

from scripts.evaluate_persist4d import (
    _begin_source_tree_contract,
    _compose_runtime_config,
    _finalize_source_tree_contract,
    _resolve_checkpoint,
)
from scripts.evaluate_persist4d_p6a import (
    EXPECTED_RESCENE_CHECKPOINT_SHA256,
    _external_cache_directory,
    _file_sha256,
    _frozen_protocol_bundle,
    _repository_path,
    build_cache_provenance,
    build_rio_class_mapper,
    build_tracker_factories,
    evaluate_cached_task_metrics,
    load_cached_protocol_sequences,
)
from scripts.p6a_artifacts import (
    REPORT_PATH,
    REQUIRED_CSV_PATHS,
    REQUIRED_JSON_PATHS,
    REQUIRED_MARKDOWN_PATHS,
    REQUIRED_SVG_PATHS,
    REQUIRED_YAML_PATHS,
    ROOT_ARTIFACT_PATH,
    publish_root_artifact,
)
from scripts.p6a_builder import _expected_cache_keys, build_p6a_root_artifact
from scripts.p6a_cache import validate_cache_manifest
from scripts.p6a_efficiency import validate_efficiency_manifest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _canonical_json_bytes(value: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _canonical_json_mapping(path: Path, *, name: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} must be a regular non-symlink file")
    try:
        payload = path.read_bytes()
        value = json.loads(payload)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} cannot be decoded") from error
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must decode to a mapping")
    normalized = dict(value)
    try:
        canonical = _canonical_json_bytes(normalized)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} is not finite canonical JSON") from error
    if payload != canonical:
        raise ValueError(f"{name} must use canonical P6-A JSON bytes")
    return normalized


def _declared_output_paths(output_root: Path) -> tuple[Path, ...]:
    relative = {
        ROOT_ARTIFACT_PATH,
        REPORT_PATH,
        *REQUIRED_CSV_PATHS,
        *REQUIRED_JSON_PATHS,
        *REQUIRED_MARKDOWN_PATHS,
        *REQUIRED_SVG_PATHS,
        *REQUIRED_YAML_PATHS,
    }
    return tuple(output_root / path for path in sorted(relative))


def _runtime_config_text(config: object) -> str:
    from omegaconf import OmegaConf

    return OmegaConf.to_yaml(config, resolve=True, sort_keys=True)


def _instantiate_validation_dataset(config: object) -> object:
    import hydra
    from omegaconf import OmegaConf

    data = getattr(config, "data", None)
    validation = getattr(data, "validation_dataset", None)
    if validation is None:
        raise TypeError("runtime config must define data.validation_dataset")
    dataset_config = OmegaConf.create(
        OmegaConf.to_container(validation, resolve=True)
    )
    dataset_config.temporal_window = 5
    return hydra.utils.instantiate(dataset_config)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _validate_frozen_inputs(
    *,
    source_commit: str,
    checkpoint_path: Path,
    p6a_bytes: bytes,
    runtime_text: str,
    protocol_manifest: Mapping[str, object],
    protocol_path: Path,
    cache_manifest: Mapping[str, object],
    cache_manifest_path: Path,
    efficiency_manifest: Mapping[str, object],
) -> None:
    if _file_sha256(checkpoint_path) != EXPECTED_RESCENE_CHECKPOINT_SHA256:
        raise ValueError("formal ReScene checkpoint SHA-256 differs from P6-A")
    expected_cache_provenance = build_cache_provenance(
        source_commit=source_commit,
        checkpoint_path=checkpoint_path,
        config_documents={
            "p6a": p6a_bytes,
            "runtime": runtime_text.encode("utf-8"),
        },
        protocol_manifest=protocol_manifest,
    )
    validate_cache_manifest(
        cache_manifest,
        expected_keys=_expected_cache_keys(protocol_manifest),
        expected_provenance=expected_cache_provenance,
    )
    if cache_manifest.get("provenance") != expected_cache_provenance:
        raise ValueError("cache provenance differs from the frozen P6-A inputs")

    validate_efficiency_manifest(efficiency_manifest)
    expected_efficiency_provenance = {
        "source_commit": source_commit,
        "checkpoint_sha256": EXPECTED_RESCENE_CHECKPOINT_SHA256,
        "config_sha256": expected_cache_provenance["config_sha256"],
        "protocol_sha256": _file_sha256(protocol_path),
        "cache_manifest_sha256": _file_sha256(cache_manifest_path),
    }
    if efficiency_manifest.get("provenance") != expected_efficiency_provenance:
        raise ValueError("efficiency provenance differs from the frozen P6-A inputs")


def run_p6a_evaluation(
    *,
    cache_directory: Path,
    metadata_path: Path,
    checkpoint_path: Path,
    output_root: Path,
    efficiency_manifest_path: Path | None = None,
) -> dict[str, object]:
    """Evaluate cached P6-A methods and atomically publish one evidence root."""

    cache_root = _external_cache_directory(cache_directory)
    output = output_root.expanduser()
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite output root: {output}")
    metadata = _repository_path(metadata_path)
    protocol_path = cache_root / "protocol_b_manifest.json"
    cache_manifest_path = cache_root / "cache_manifest.json"
    efficiency_path = (
        cache_root / "efficiency_raw_manifest.json"
        if efficiency_manifest_path is None
        else efficiency_manifest_path.expanduser().resolve()
    )
    if efficiency_path.parent != cache_root or efficiency_path.name != (
        "efficiency_raw_manifest.json"
    ):
        raise ValueError("efficiency manifest must remain inside prediction cache root")

    guard = _begin_source_tree_contract(
        repo_root=PROJECT_ROOT,
        output_paths=_declared_output_paths(output),
    )
    protocol, current_protocol_manifest, p6a_bytes = _frozen_protocol_bundle(
        metadata_path=metadata
    )
    stored_protocol_manifest = _canonical_json_mapping(
        protocol_path,
        name="Protocol B manifest",
    )
    if stored_protocol_manifest != current_protocol_manifest:
        raise ValueError("stored Protocol B manifest differs from frozen inputs")
    cache_manifest = _canonical_json_mapping(
        cache_manifest_path,
        name="cache manifest",
    )
    efficiency_manifest = _canonical_json_mapping(
        efficiency_path,
        name="efficiency manifest",
    )

    config, _memory_config = _compose_runtime_config()
    runtime_text = _runtime_config_text(config)
    checkpoint = _resolve_checkpoint(checkpoint_path)
    _validate_frozen_inputs(
        source_commit=guard.source_commit,
        checkpoint_path=checkpoint,
        p6a_bytes=p6a_bytes,
        runtime_text=runtime_text,
        protocol_manifest=current_protocol_manifest,
        protocol_path=protocol_path,
        cache_manifest=cache_manifest,
        cache_manifest_path=cache_manifest_path,
        efficiency_manifest=efficiency_manifest,
    )
    sequences = load_cached_protocol_sequences(
        protocol=protocol,
        cache_directory=cache_root / "entries",
        manifest_path=cache_manifest_path,
    )
    if len(sequences) != 129:
        raise ValueError("P6-A evaluation requires exactly 129 cached sequences")

    dataset = _instantiate_validation_dataset(config)
    class_mapper = build_rio_class_mapper(dataset)
    try:
        p6a_config = yaml.safe_load(p6a_bytes)
    except yaml.YAMLError as error:
        raise ValueError("P6-A config cannot be decoded") from error
    if not isinstance(p6a_config, Mapping):
        raise TypeError("P6-A config must decode to a mapping")
    tracker_factories = build_tracker_factories(p6a_config)
    background_class = int(p6a_config["baselines"]["b4"]["background_class"])
    evaluation = evaluate_cached_task_metrics(
        sequences,
        tracker_factories=tracker_factories,
        class_mapper=class_mapper,
        background_class=background_class,
    )
    artifact = build_p6a_root_artifact(
        evaluation=evaluation,
        protocol_manifest=current_protocol_manifest,
        cache_manifest=cache_manifest,
        efficiency_manifest=efficiency_manifest,
        source_commit=guard.source_commit,
        p6a_config_text=p6a_bytes.decode("utf-8"),
        runtime_config_text=runtime_text,
    )

    before_publish = _finalize_source_tree_contract(guard)
    published = False
    try:
        publish_root_artifact(output, artifact)
        published = True
        after_publish = _finalize_source_tree_contract(guard)
        if after_publish != before_publish:
            raise RuntimeError("source tree contract changed during publication")
    except Exception:
        if published and output.exists() and not output.is_symlink():
            shutil.rmtree(output)
        raise
    return artifact


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen P6-A cache and publish the complete evidence bundle."
    )
    parser.add_argument("--cache-directory", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/rescene4d_concerto_t2_repro.ckpt"),
    )
    parser.add_argument(
        "--efficiency-manifest",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/P6A"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _argument_parser().parse_args(argv)
    artifact = run_p6a_evaluation(
        cache_directory=args.cache_directory,
        metadata_path=args.metadata,
        checkpoint_path=args.checkpoint,
        output_root=args.output_root,
        efficiency_manifest_path=args.efficiency_manifest,
    )
    gate_results = artifact["gate_results"]
    decision = (
        "P6A_GO"
        if all(record["passed"] for record in gate_results.values())
        else "P6A_STOP"
    )
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "source_commit": artifact["source_commit"],
                "decision": decision,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
