"""Versioned orchestration for official-candidate Persist4D evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

from scripts.evaluate_persist4d_p6a import (
    build_rio_class_mapper,
    expected_cache_keys,
)
from scripts.p6a_cache import validate_cache_entry, write_cache_entry
from scripts.rescene_task_postprocess import (
    OfficialTaskPrediction,
    extract_official_task_prediction,
)
from scripts.run_system_comparison import (
    REPRODUCIBILITY_BINDING,
    SOURCE_PROTOCOL,
    _build_frozen_setup,
    _local_producer,
    _publish_exact_bytes,
)
from scripts.system_comparison_inference import deterministic_inference_runtime
from scripts.system_comparison_v2_cache import (
    build_task_sidecar,
    load_task_sidecar,
    observation_fingerprint,
    task_sidecar_digest,
    write_task_sidecar,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts/system_comparison_v2"
DEFAULT_METADATA = Path("/home/ww/3RScan.json")
DEFAULT_CACHE_ROOT = Path(
    "/mnt/shared/ww/persist4d-tmap-root-cause-v2/system_comparison_v2_full"
)
EXPECTED_ENTRY_COUNT = 43 * 3 * 5
SCORE_REDUCER = "mean"


class V2RunError(RuntimeError):
    """Raised when a V2 cache or evaluation gate fails closed."""


def _read_json(path: Path) -> Mapping[str, object]:
    if path.is_symlink() or not path.is_file():
        raise V2RunError(f"required JSON is unavailable: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise V2RunError(f"required JSON must contain a mapping: {path}")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_digest(value: object) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _key_identity(key: Mapping[str, object]) -> str:
    return json.dumps(
        key,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _atomic_replace_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _source_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _new_progress(
    *, source_commit: str, checkpoint_sha256: str, protocol_sha256: str
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "in_progress",
        "source_commit": source_commit,
        "checkpoint_sha256": checkpoint_sha256,
        "protocol_manifest_sha256": protocol_sha256,
        "score_reducer": SCORE_REDUCER,
        "records": [],
    }


def _validate_progress_binding(
    progress: Mapping[str, object],
    *,
    source_commit: str,
    checkpoint_sha256: str,
    protocol_sha256: str,
) -> Sequence[Mapping[str, object]]:
    expected = {
        "schema_version": 1,
        "source_commit": source_commit,
        "checkpoint_sha256": checkpoint_sha256,
        "protocol_manifest_sha256": protocol_sha256,
        "score_reducer": SCORE_REDUCER,
    }
    if any(progress.get(key) != value for key, value in expected.items()):
        raise V2RunError("V2 cache progress binding differs")
    records = progress.get("records")
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise V2RunError("V2 cache progress records must be a sequence")
    if progress.get("status") not in {"in_progress", "pass"}:
        raise V2RunError("V2 cache progress status is invalid")
    return records


def _validate_pair_record(
    record: Mapping[str, object],
    *,
    raw_directory: Path,
    sidecar_directory: Path,
) -> None:
    expected_fields = {
        "key",
        "raw_entry",
        "sidecar_entry",
        "raw_observation_fingerprint",
    }
    if set(record) != expected_fields:
        raise V2RunError("V2 cache pair record fields differ")
    raw_entry = record["raw_entry"]
    sidecar_entry = record["sidecar_entry"]
    if not isinstance(raw_entry, Mapping) or not isinstance(sidecar_entry, Mapping):
        raise V2RunError("V2 cache pair entries must be mappings")
    if raw_entry.get("key") != record["key"] or sidecar_entry.get("key") != record["key"]:
        raise V2RunError("V2 cache pair keys differ")
    raw = validate_cache_entry(
        raw_directory / str(raw_entry["filename"]), raw_entry
    )
    sidecar_path = sidecar_directory / str(sidecar_entry["filename"])
    sidecar = load_task_sidecar(sidecar_path)
    if sidecar_path.stat().st_size != sidecar_entry.get("file_bytes"):
        raise V2RunError("V2 sidecar byte size differs")
    if _file_sha256(sidecar_path) != sidecar_entry.get("file_sha256"):
        raise V2RunError("V2 sidecar file hash differs")
    if task_sidecar_digest(sidecar) != sidecar_entry.get("content_sha256"):
        raise V2RunError("V2 sidecar content hash differs")
    fingerprint = observation_fingerprint(raw)
    if (
        fingerprint != record["raw_observation_fingerprint"]
        or sidecar["provenance"]["source_raw_observation_fingerprint"]
        != fingerprint
    ):
        raise V2RunError("V2 raw/sidecar observation binding differs")


def run_v2_cache(
    *,
    metadata_path: Path,
    device_name: str,
    cache_root: Path,
    artifact_root: Path = ARTIFACT_ROOT,
) -> Mapping[str, object]:
    source_commit = _source_commit()
    binding = _read_json(REPRODUCIBILITY_BINDING)
    protocol_sha256 = _file_sha256(SOURCE_PROTOCOL)
    checkpoint_sha256 = str(binding["checkpoint_sha256"])
    if binding.get("protocol_sha256") != protocol_sha256:
        raise V2RunError("Protocol-B manifest differs from frozen binding")

    raw_directory = cache_root / "raw_predictions/entries"
    sidecar_directory = cache_root / "task_sidecars/entries"
    progress_path = cache_root / "cache_progress.json"
    if progress_path.exists():
        progress = dict(_read_json(progress_path))
    else:
        progress = _new_progress(
            source_commit=source_commit,
            checkpoint_sha256=checkpoint_sha256,
            protocol_sha256=protocol_sha256,
        )
        _atomic_replace_json(progress_path, progress)
    raw_records = _validate_progress_binding(
        progress,
        source_commit=source_commit,
        checkpoint_sha256=checkpoint_sha256,
        protocol_sha256=protocol_sha256,
    )
    records: list[Mapping[str, object]] = []
    records_by_key: dict[str, Mapping[str, object]] = {}
    for raw_record in raw_records:
        if not isinstance(raw_record, Mapping) or not isinstance(
            raw_record.get("key"), Mapping
        ):
            raise V2RunError("V2 cache progress record is invalid")
        identity = _key_identity(raw_record["key"])
        if identity in records_by_key:
            raise V2RunError("V2 cache progress contains duplicate keys")
        _validate_pair_record(
            raw_record,
            raw_directory=raw_directory,
            sidecar_directory=sidecar_directory,
        )
        record = dict(raw_record)
        records.append(record)
        records_by_key[identity] = record

    setup = _build_frozen_setup(
        binding=binding,
        metadata_path=metadata_path,
        device_name=device_name,
    )
    keys = expected_cache_keys(setup.protocol)
    if len(keys) != EXPECTED_ENTRY_COUNT:
        raise V2RunError("Protocol B does not contain 645 cache keys")
    expected_identities = {_key_identity(key) for key in keys}
    if not set(records_by_key) <= expected_identities:
        raise V2RunError("V2 cache progress contains an unexpected key")

    producer = _local_producer(setup)
    producer.provenance = {
        **setup.local_provenance,
        "source_commit": source_commit,
    }
    class_mapper = build_rio_class_mapper(setup.dataset)
    seed = int(setup.p6a_config["protocol_b"]["seed"])
    if setup.device is None:
        raise V2RunError("CUDA setup did not expose a device")
    with deterministic_inference_runtime(seed, setup.device):
        for position, key in enumerate(keys, start=1):
            identity = _key_identity(key)
            if identity in records_by_key:
                continue
            produced = producer.produce_bundle(
                key,
                task_prediction_builder=extract_official_task_prediction,
                class_mapper=class_mapper,
            )
            if not isinstance(produced.task_prediction, OfficialTaskPrediction):
                raise V2RunError("local forward did not produce official candidates")
            sidecar = write_task_sidecar(
                sidecar_directory,
                payload=build_task_sidecar(
                    raw_cache_payload=produced.payload,
                    official_prediction=produced.task_prediction,
                    protocol_manifest_sha256=protocol_sha256,
                ),
            )
            raw_entry = write_cache_entry(raw_directory, produced.payload)
            raw_fingerprint = observation_fingerprint(produced.payload)
            record = {
                "key": key,
                "raw_entry": raw_entry,
                "sidecar_entry": sidecar,
                "raw_observation_fingerprint": raw_fingerprint,
            }
            _validate_pair_record(
                record,
                raw_directory=raw_directory,
                sidecar_directory=sidecar_directory,
            )
            records.append(record)
            records_by_key[identity] = record
            progress = {
                **progress,
                "status": "in_progress",
                "records": records,
            }
            _atomic_replace_json(progress_path, progress)
            if position % 25 == 0 or position == len(keys):
                print(f"V2 cache progress: {len(records)}/{len(keys)}", flush=True)

    ordered_records = [records_by_key[_key_identity(key)] for key in keys]
    if len(ordered_records) != EXPECTED_ENTRY_COUNT:
        raise V2RunError("V2 cache coverage is incomplete")
    for record in ordered_records:
        _validate_pair_record(
            record,
            raw_directory=raw_directory,
            sidecar_directory=sidecar_directory,
        )
    progress = {
        **progress,
        "status": "pass",
        "records": ordered_records,
    }
    _atomic_replace_json(progress_path, progress)
    manifest = {
        **progress,
        "entry_count": len(ordered_records),
        "records_sha256": _canonical_digest(ordered_records),
        "external_cache_reference": "external:system_comparison_v2_full",
    }
    manifest_bytes = (
        json.dumps(
            manifest,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    _publish_exact_bytes(artifact_root / "cache_manifest.json", manifest_bytes)
    return {
        "status": "pass",
        "entry_count": len(ordered_records),
        "records_sha256": manifest["records_sha256"],
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run System Comparison V2.")
    parser.add_argument("stage", choices=("cache",))
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    result = run_v2_cache(
        metadata_path=arguments.metadata,
        device_name=arguments.device,
        cache_root=arguments.cache_root,
        artifact_root=arguments.artifact_root,
    )
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
