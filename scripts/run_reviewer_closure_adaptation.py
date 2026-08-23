"""Run exact T2-T5 evaluation for the T2-to-T3 horizon-adapted ReScene model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import reviewer_closure_adaptation as adaptation

ARTIFACT_ROOT = PROJECT_ROOT / "artifacts/reviewer_closure"
PHASE_II_CONFIG = PROJECT_ROOT / "configs/reviewer_closure/phase_ii_evaluation.yaml"
TRAINING_MANIFEST = ARTIFACT_ROOT / "rescene_horizon_training_manifest.json"
ADAPTED_CHECKPOINT = (
    PROJECT_ROOT / "checkpoints/rescene4d_t2_to_t3_horizon_adapted.ckpt"
)
SYSTEM_MANIFEST = (
    PROJECT_ROOT / "artifacts/system_comparison/system_comparison_manifest.json"
)
SYSTEM_BINDING = (
    PROJECT_ROOT / "artifacts/system_comparison/reproducibility_binding.json"
)
PREDICTION_ROOT = ARTIFACT_ROOT / "rescene_horizon_adapted_predictions"
PREDICTION_ENTRIES = PREDICTION_ROOT / "entries"
PREDICTION_MANIFEST = PREDICTION_ROOT / "manifest.json"
SIDECAR_ROOT = ARTIFACT_ROOT / "rescene_horizon_adapted_observations_v2"
SIDECAR_ENTRIES = SIDECAR_ROOT / "entries"
SIDECAR_MANIFEST = SIDECAR_ROOT / "manifest.json"
PHASE_I_RESULTS = ARTIFACT_ROOT / "full_history_tracker_results.csv"
SYSTEM_AGGREGATE = PROJECT_ROOT / "artifacts/system_comparison/aggregate_results.csv"
SYSTEM_PER_ORDER = PROJECT_ROOT / "artifacts/system_comparison/per_order_results.csv"
ADAPTATION_PER_SEQUENCE = ARTIFACT_ROOT / "rescene_horizon_adaptation_per_sequence.csv"
ADAPTATION_RESULTS = ARTIFACT_ROOT / "rescene_horizon_adaptation_results.csv"
ADAPTATION_PER_ORDER = ARTIFACT_ROOT / "rescene_horizon_adaptation_per_order.csv"
ADAPTATION_BOOTSTRAP = ARTIFACT_ROOT / "rescene_horizon_adaptation_cluster_bootstrap.csv"
ADAPTATION_ORDER_STATISTICS = (
    ARTIFACT_ROOT / "rescene_horizon_adaptation_order_robustness.csv"
)
ADAPTATION_LOSO = ARTIFACT_ROOT / "rescene_horizon_adaptation_loso.csv"
ADAPTATION_EVALUATION_MANIFEST = (
    ARTIFACT_ROOT / "rescene_horizon_adaptation_evaluation_manifest.json"
)
ADAPTATION_PROFILE_RESULTS = (
    ARTIFACT_ROOT / "rescene_horizon_adaptation_profile_results.csv"
)
ADAPTATION_COMPUTE = ARTIFACT_ROOT / "rescene_horizon_compute.csv"
ADAPTATION_PROFILE_MANIFEST = (
    ARTIFACT_ROOT / "rescene_horizon_profile_manifest.json"
)
GATE_II_TASK_EVIDENCE = ARTIFACT_ROOT / "rescene_horizon_gate_ii_task_evidence.csv"
GATE_II_IDENTITY_EVIDENCE = (
    ARTIFACT_ROOT / "rescene_horizon_gate_ii_identity_evidence.csv"
)
GATE_II_COMPUTE_EVIDENCE = (
    ARTIFACT_ROOT / "rescene_horizon_gate_ii_compute_evidence.csv"
)
GATE_II_PATH = ARTIFACT_ROOT / "rescene_horizon_gate_ii.json"
ADAPTED_CHALLENGE_REPORT = (
    ARTIFACT_ROOT / "LONG_HORIZON_RESCENE_CHALLENGE_REPORT.md"
)
DEFAULT_METADATA = Path(os.environ.get("PERSIST4D_3RSCAN_METADATA", "3RScan.json"))
_WINDOWS_ABSOLUTE_FRAGMENT = re.compile(
    r"(?i)(?<![A-Za-z0-9])(?:[A-Za-z]:[\\/]|\\\\[^\\/\s]+[\\/])"
)


class AdaptationRunError(RuntimeError):
    """Raised when adapted inference would violate the frozen Phase II contract."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def _content_sha256(value: Mapping[str, object]) -> str:
    payload = dict(value)
    payload.pop("content_sha256", None)
    compact = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(compact).hexdigest()


def _portable_runtime_config_text(runtime_config_text: str) -> str:
    document = yaml.safe_load(runtime_config_text)
    if not isinstance(document, Mapping):
        raise TypeError("runtime config must be a YAML mapping")
    portable = dict(document)
    replacements = 0

    def local_path_name(value: str) -> str | None:
        posix_path = PurePosixPath(value)
        if posix_path.is_absolute():
            return posix_path.name
        windows_path = PureWindowsPath(value)
        if windows_path.is_absolute() or windows_path.drive:
            return windows_path.name
        return None

    def visit(value: object) -> None:
        nonlocal replacements
        if isinstance(value, dict):
            if value.get("model_lib") == "concerto":
                checkpoint = value.get("name")
                checkpoint_name = (
                    local_path_name(checkpoint)
                    if isinstance(checkpoint, str)
                    else None
                )
                if checkpoint_name is not None:
                    if not checkpoint_name:
                        raise ValueError("Concerto checkpoint path has no file name")
                    value["name"] = f"external:concerto/{checkpoint_name}"
                    replacements += 1
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
        elif isinstance(value, str) and (
            local_path_name(value) is not None
            or _WINDOWS_ABSOLUTE_FRAGMENT.search(value) is not None
        ):
            raise ValueError("runtime config contains an unportable absolute path")

    visit(portable)
    if replacements == 0:
        return runtime_config_text
    return yaml.safe_dump(portable, allow_unicode=False, sort_keys=True)


def _config_documents_sha256(config_documents: Mapping[str, bytes]) -> str:
    if not isinstance(config_documents, Mapping) or not config_documents:
        raise ValueError("config_documents must be a non-empty mapping")
    hasher = hashlib.sha256()
    for name, raw_content in sorted(config_documents.items()):
        if not isinstance(name, str) or not name or "/" in name or "\\" in name:
            raise ValueError("config document names must be portable identifiers")
        if not isinstance(raw_content, bytes) or not raw_content:
            raise ValueError("config documents must contain non-empty bytes")
        content = raw_content
        if name == "runtime":
            try:
                runtime_text = content.decode("utf-8")
            except UnicodeDecodeError as error:
                raise ValueError("runtime config must use UTF-8") from error
            content = _portable_runtime_config_text(runtime_text).encode("utf-8")
        hasher.update(name.encode("utf-8") + b"\0")
        hasher.update(len(content).to_bytes(8, "big") + content)
    return hasher.hexdigest()


def _load_json(path: str | Path, *, name: str) -> dict[str, Any]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise AdaptationRunError(f"{name} is unavailable")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AdaptationRunError(f"cannot decode {name}") from error
    if not isinstance(value, Mapping):
        raise AdaptationRunError(f"{name} must be a mapping")
    return dict(value)


def _prediction_inference_source_commit(
    path: str | Path = PREDICTION_MANIFEST,
) -> str:
    manifest = _load_json(path, name="adapted prediction manifest")
    provenance = manifest.get("provenance")
    source_commit = (
        provenance.get("source_commit") if isinstance(provenance, Mapping) else None
    )
    if not isinstance(source_commit, str) or re.fullmatch(
        r"[0-9a-f]{40}", source_commit
    ) is None:
        raise AdaptationRunError(
            "adapted prediction manifest source commit is invalid"
        )
    return source_commit


def _publish_exact(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file():
            raise FileExistsError(f"output is not a regular file: {path}")
        if path.read_bytes() == payload:
            return
        raise FileExistsError(f"output already contains different content: {path}")
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
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _git_head() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            check=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise AdaptationRunError("cannot resolve adaptation evaluation commit") from error
    commit = completed.stdout.strip()
    if len(commit) != 40:
        raise AdaptationRunError("adaptation evaluation commit is invalid")
    return commit


def _require_evaluation_tree_clean() -> None:
    completed = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=True,
        text=True,
    )
    disallowed = []
    for line in completed.stdout.splitlines():
        path = line[3:]
        if " -> " in path:
            path = path.split(" -> ", maxsplit=1)[1]
        if not path.startswith("artifacts/reviewer_closure/"):
            disallowed.append(path)
    if disallowed:
        raise AdaptationRunError(
            "adaptation evaluation source must be committed before CUDA inference: "
            + ", ".join(sorted(disallowed))
        )


def build_adapted_provenance(
    *,
    checkpoint_path: str | Path,
    source_commit: str,
    config_documents: Mapping[str, bytes],
    protocol_sha256: str,
) -> dict[str, str]:
    checkpoint = Path(checkpoint_path)
    if checkpoint.is_symlink() or not checkpoint.is_file():
        raise AdaptationRunError("adapted checkpoint must be a regular file")
    if (
        not isinstance(source_commit, str)
        or len(source_commit) != 40
        or any(character not in "0123456789abcdef" for character in source_commit)
    ):
        raise AdaptationRunError("adapted provenance source commit is invalid")
    if (
        not isinstance(protocol_sha256, str)
        or len(protocol_sha256) != 64
        or any(character not in "0123456789abcdef" for character in protocol_sha256)
    ):
        raise AdaptationRunError("adapted provenance protocol digest is invalid")
    try:
        config_sha256 = _config_documents_sha256(config_documents)
    except ValueError as error:
        raise AdaptationRunError("adapted provenance config documents are invalid") from error
    return {
        "source_commit": source_commit,
        "checkpoint_sha256": _sha256_file(checkpoint),
        "config_sha256": config_sha256,
        "protocol_sha256": protocol_sha256,
    }


def validate_training_checkpoint(
    *,
    manifest_path: str | Path = TRAINING_MANIFEST,
    checkpoint_path: str | Path = ADAPTED_CHECKPOINT,
) -> dict[str, object]:
    import torch

    manifest = _load_json(manifest_path, name="T3 training manifest")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("status") != "pass"
        or manifest.get("selected_level") != 2
        or manifest.get("content_sha256") != _content_sha256(manifest)
    ):
        raise AdaptationRunError("T3 training manifest contract differs")
    checkpoints = manifest.get("checkpoints")
    reload_record = manifest.get("canonical_checkpoint_reload")
    if not isinstance(checkpoints, Mapping) or not isinstance(reload_record, Mapping):
        raise AdaptationRunError("T3 training manifest checkpoint records are missing")
    canonical = checkpoints.get("canonical_final")
    checkpoint = Path(checkpoint_path)
    if not isinstance(canonical, Mapping) or checkpoint.is_symlink() or not checkpoint.is_file():
        raise AdaptationRunError("canonical adapted checkpoint is unavailable")
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError) as error:
        raise AdaptationRunError("cannot load canonical adapted checkpoint") from error
    state_dict = payload.get("state_dict") if isinstance(payload, Mapping) else None
    if not isinstance(state_dict, Mapping):
        raise AdaptationRunError("canonical adapted checkpoint lacks a state dict")
    expected_reference = "repo:checkpoints/rescene4d_t2_to_t3_horizon_adapted.ckpt"
    if (
        canonical.get("reference") != expected_reference
        or canonical.get("sha256") != _sha256_file(checkpoint)
        or canonical.get("byte_size") != checkpoint.stat().st_size
        or canonical.get("epoch") != 44
        or canonical.get("global_step") != 2160
        or canonical.get("state_dict_entry_count") != len(state_dict)
        or reload_record.get("strict") is not True
        or reload_record.get("state_dict_entry_count") != len(state_dict)
    ):
        raise AdaptationRunError("canonical adapted checkpoint binding differs")
    source_commit = manifest.get("source_commit")
    if not isinstance(source_commit, str) or len(source_commit) != 40:
        raise AdaptationRunError("T3 training source commit is invalid")
    return {
        "training_manifest_content_sha256": manifest["content_sha256"],
        "training_source_commit": source_commit,
        "checkpoint_sha256": canonical["sha256"],
        "checkpoint_byte_size": canonical["byte_size"],
        "checkpoint_epoch": canonical["epoch"],
        "checkpoint_global_step": canonical["global_step"],
        "state_dict_entry_count": len(state_dict),
    }


def _select_smoke_key(
    expected_keys: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    for key in expected_keys:
        if key.get("order_id") == "canonical" and key.get("horizon") == 2:
            return dict(key)
    raise AdaptationRunError("adapted cache lacks a canonical T2 smoke key")


def produce_adapted_batch(
    *,
    expected_keys: Sequence[Mapping[str, object]],
    existing_prediction_entries: Sequence[Mapping[str, object]],
    existing_sidecar_entries: Sequence[Mapping[str, object]],
    producer: object,
    prediction_writer: Callable[[Mapping[str, object]], Mapping[str, object]],
    sidecar_builder: Callable[..., Mapping[str, object]],
    sidecar_writer: Callable[[Mapping[str, object]], Mapping[str, object]],
    source_commit: str,
    smoke_only: bool,
) -> dict[str, int]:
    pending = adaptation.validate_adapted_resume(
        expected_keys=expected_keys,
        prediction_entries=existing_prediction_entries,
        sidecar_entries=existing_sidecar_entries,
        source_commit=source_commit,
    )
    if smoke_only:
        smoke = _select_smoke_key(expected_keys)
        smoke_identity = json.dumps(smoke, sort_keys=True)
        selected = [key for key in pending if json.dumps(key, sort_keys=True) == smoke_identity]
        expected_count = 1
    else:
        selected = pending
        expected_count = len(expected_keys)
    produce_bundle = getattr(producer, "produce_bundle", None)
    if not callable(produce_bundle):
        raise AdaptationRunError("adapted producer lacks one-forward bundle production")
    for key in selected:
        bundle = produce_bundle(key)
        prediction = getattr(bundle, "payload", None)
        processed = getattr(bundle, "processed", None)
        raw_observation = getattr(processed, "raw_observation", None)
        if not isinstance(prediction, Mapping) or not isinstance(raw_observation, Mapping):
            raise AdaptationRunError("adapted bundle lacks prediction or raw observation")
        prediction_record = dict(prediction_writer(prediction))
        if (
            prediction_record.get("key") != dict(key)
            or prediction_record.get("content_sha256")
            != prediction.get("content_sha256")
        ):
            raise AdaptationRunError("adapted prediction writer changed the bundle")
        sidecar = sidecar_builder(
            key=key,
            raw_observation=raw_observation,
            prediction=prediction,
            source_commit=source_commit,
        )
        sidecar_record = dict(sidecar_writer(sidecar))
        if (
            sidecar_record.get("source_prediction_content_sha256")
            != prediction_record["content_sha256"]
            or sidecar_record.get("reference_prediction_content_sha256")
            != prediction_record["content_sha256"]
            or sidecar_record.get("sidecar_source_commit") != source_commit
        ):
            raise AdaptationRunError("adapted sidecar writer changed the binding")
    return {
        "expected_count": expected_count,
        "reused_count": expected_count - len(selected),
        "produced_count": len(selected),
    }


def build_adapted_sidecar_manifest(
    *,
    entries: Sequence[Mapping[str, object]],
    expected_keys: Sequence[Mapping[str, object]],
    prediction_manifest: Mapping[str, object],
    source_commit: str,
    cache_directory: str | Path | None = None,
) -> dict[str, object]:
    predictions = prediction_manifest.get("entries")
    provenance = prediction_manifest.get("provenance")
    prediction_content = prediction_manifest.get("content_sha256")
    if (
        isinstance(predictions, (str, bytes))
        or not isinstance(predictions, Sequence)
        or not isinstance(provenance, Mapping)
        or not isinstance(prediction_content, str)
        or len(prediction_content) != 64
    ):
        raise adaptation.AdaptationEvidenceError(
            "adapted prediction manifest is incomplete"
        )
    pending = adaptation.validate_adapted_resume(
        expected_keys=expected_keys,
        prediction_entries=predictions,
        sidecar_entries=entries,
        source_commit=source_commit,
    )
    if pending or len(entries) != len(expected_keys):
        raise adaptation.AdaptationEvidenceError(
            "adapted sidecar manifest lacks exact coverage"
        )
    normalized = [dict(entry) for entry in entries]
    normalized.sort(key=lambda row: json.dumps(row["key"], sort_keys=True))
    if cache_directory is not None:
        from scripts.reviewer_closure_sidecar import (
            load_full_history_observation_sidecar_entry,
        )

        directory = Path(cache_directory)
        if directory.is_symlink() or not directory.is_dir():
            raise adaptation.AdaptationEvidenceError(
                "adapted sidecar cache directory is unavailable"
            )
        expected_files = {str(entry["filename"]) for entry in normalized}
        if {path.name for path in directory.iterdir()} != expected_files:
            raise adaptation.AdaptationEvidenceError(
                "adapted sidecar cache directory coverage differs"
            )
        for entry in normalized:
            load_full_history_observation_sidecar_entry(directory, entry)
    manifest: dict[str, object] = {
        "schema_version": "adapted-full-history-observations-v2-manifest",
        "status": "pass",
        "source_commit": source_commit,
        "prediction_manifest_content_sha256": prediction_content,
        "prediction_provenance": dict(provenance),
        "entry_count": len(normalized),
        "entries_sha256": hashlib.sha256(
            _canonical_json_bytes({"entries": normalized})
        ).hexdigest(),
        "entries": normalized,
    }
    manifest["content_sha256"] = _content_sha256(manifest)
    return manifest


def _build_adapted_setup(
    *,
    metadata_path: Path,
    device_name: str | None,
    source_commit: str | None = None,
):
    from omegaconf import OmegaConf

    from scripts import reviewer_closure_training
    from scripts.run_reviewer_closure_t3 import strict_load_adaptation_weights
    from scripts.run_system_comparison import _build_frozen_setup

    training = validate_training_checkpoint()
    system_manifest = _load_json(SYSTEM_MANIFEST, name="system manifest")
    binding = _load_json(SYSTEM_BINDING, name="system reproducibility binding")
    setup = _build_frozen_setup(
        binding=binding,
        metadata_path=metadata_path,
        device_name=device_name,
    )
    if device_name is not None:
        if setup.system is None:
            raise AdaptationRunError("adapted CUDA setup lacks a model")
        strict_load_adaptation_weights(
            setup.system,
            ADAPTED_CHECKPOINT,
            expected_sha256=str(training["checkpoint_sha256"]),
        )
    runtime_bytes = OmegaConf.to_yaml(
        setup.runtime_config,
        resolve=True,
        sort_keys=True,
    ).encode("utf-8")
    documents = {
        "p6a": (
            PROJECT_ROOT / "artifacts/P6A/configs/p6a_default.yaml"
        ).read_bytes(),
        "runtime": runtime_bytes,
        "t3_adaptation_recipe": reviewer_closure_training.RECIPE_PATH.read_bytes(),
        "phase_ii_evaluation": PHASE_II_CONFIG.read_bytes(),
    }
    provenance_source_commit = (
        source_commit if source_commit is not None else _git_head()
    )
    setup.full_provenance = build_adapted_provenance(
        checkpoint_path=ADAPTED_CHECKPOINT,
        source_commit=provenance_source_commit,
        config_documents=documents,
        protocol_sha256=str(setup.full_provenance["protocol_sha256"]),
    )
    if setup.full_provenance["checkpoint_sha256"] != training["checkpoint_sha256"]:
        raise AdaptationRunError("adapted setup checkpoint binding differs")
    return setup, system_manifest, training


def _expected_keys(system_manifest: Mapping[str, object]) -> list[dict[str, object]]:
    from scripts.system_comparison_inference import full_history_cache_keys

    return adaptation.expected_adapted_keys(full_history_cache_keys(system_manifest))


def run_cache(
    *,
    device_name: str,
    metadata_path: Path,
    smoke_only: bool,
) -> dict[str, object]:
    import torch

    from scripts.reviewer_closure_sidecar import (
        build_full_history_observation_sidecar,
        discover_full_history_observation_sidecar_entries,
        sidecar_key_for_source_prediction,
        write_full_history_observation_sidecar_entry,
    )
    from scripts.run_system_comparison import _full_producer
    from scripts.system_comparison_inference import (
        deterministic_inference_runtime,
        discover_full_history_cache_entries,
        write_full_history_cache_entry,
    )

    if not metadata_path.is_file() or metadata_path.is_symlink():
        raise AdaptationRunError("3RScan metadata is unavailable")
    _require_evaluation_tree_clean()
    setup, system_manifest, training = _build_adapted_setup(
        metadata_path=metadata_path,
        device_name=device_name,
    )
    if setup.device is None or setup.system is None:
        raise AdaptationRunError("adapted cache production requires CUDA")
    expected = _expected_keys(system_manifest)
    predictions = discover_full_history_cache_entries(
        PREDICTION_ENTRIES,
        expected_provenance=setup.full_provenance,
    )
    sidecars = discover_full_history_observation_sidecar_entries(SIDECAR_ENTRIES)
    source_commit = _git_head()
    producer = _full_producer(setup)

    def write_prediction(payload: Mapping[str, object]) -> Mapping[str, object]:
        return write_full_history_cache_entry(PREDICTION_ENTRIES, payload)

    def build_sidecar(
        *,
        key: Mapping[str, object],
        raw_observation: Mapping[str, object],
        prediction: Mapping[str, object],
        source_commit: str,
    ) -> Mapping[str, object]:
        return build_full_history_observation_sidecar(
            key=sidecar_key_for_source_prediction(key),
            raw_observation=raw_observation,
            source_prediction=prediction,
            reference_prediction_content_sha256=str(prediction["content_sha256"]),
            sidecar_source_commit=source_commit,
        )

    def write_sidecar(payload: Mapping[str, object]) -> Mapping[str, object]:
        return write_full_history_observation_sidecar_entry(SIDECAR_ENTRIES, payload)

    with deterministic_inference_runtime(45, setup.device):
        result = produce_adapted_batch(
            expected_keys=expected,
            existing_prediction_entries=predictions,
            existing_sidecar_entries=sidecars,
            producer=producer,
            prediction_writer=write_prediction,
            sidecar_builder=build_sidecar,
            sidecar_writer=write_sidecar,
            source_commit=source_commit,
            smoke_only=smoke_only,
        )
    torch.cuda.empty_cache()
    return {
        "status": "pass",
        "mode": "smoke" if smoke_only else "full",
        **result,
        "source_commit": source_commit,
        "checkpoint_sha256": training["checkpoint_sha256"],
    }


def run_finalize(*, metadata_path: Path) -> dict[str, object]:
    from scripts.reviewer_closure_sidecar import (
        discover_full_history_observation_sidecar_entries,
    )
    from scripts.system_comparison_inference import (
        build_full_history_cache_manifest,
        discover_full_history_cache_entries,
        write_full_history_cache_manifest,
    )

    inference_source_commit = (
        _prediction_inference_source_commit()
        if PREDICTION_MANIFEST.is_file() and not PREDICTION_MANIFEST.is_symlink()
        else _git_head()
    )
    setup, system_manifest, training = _build_adapted_setup(
        metadata_path=metadata_path,
        device_name=None,
        source_commit=inference_source_commit,
    )
    expected = _expected_keys(system_manifest)
    predictions = discover_full_history_cache_entries(
        PREDICTION_ENTRIES,
        expected_provenance=setup.full_provenance,
    )
    prediction_manifest = build_full_history_cache_manifest(
        predictions,
        expected_keys=expected,
        expected_provenance=setup.full_provenance,
        cache_directory=PREDICTION_ENTRIES,
    )
    write_full_history_cache_manifest(
        PREDICTION_MANIFEST,
        prediction_manifest,
        expected_keys=expected,
        expected_provenance=setup.full_provenance,
        cache_directory=PREDICTION_ENTRIES,
    )
    sidecars = discover_full_history_observation_sidecar_entries(SIDECAR_ENTRIES)
    sidecar_manifest = build_adapted_sidecar_manifest(
        entries=sidecars,
        expected_keys=expected,
        prediction_manifest=prediction_manifest,
        source_commit=inference_source_commit,
        cache_directory=SIDECAR_ENTRIES,
    )
    _publish_exact(SIDECAR_MANIFEST, _canonical_json_bytes(sidecar_manifest))
    return {
        "status": "pass",
        "prediction_entry_count": len(predictions),
        "sidecar_entry_count": len(sidecars),
        "checkpoint_sha256": training["checkpoint_sha256"],
        "prediction_manifest_content_sha256": prediction_manifest["content_sha256"],
        "sidecar_manifest_content_sha256": sidecar_manifest["content_sha256"],
    }


def _csv_bytes(
    rows: Sequence[Mapping[str, object]], fields: Sequence[str]
) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(fields), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        if set(row) != set(fields):
            raise AdaptationRunError("adaptation CSV row fields differ")
        writer.writerow({field: "" if row[field] is None else row[field] for field in fields})
    return stream.getvalue().encode("utf-8")


def _read_typed_csv(path: str | Path) -> list[dict[str, object]]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise AdaptationRunError(f"required CSV is unavailable: {source}")
    string_fields = {
        "method_id",
        "method",
        "metric",
        "challenger_method_id",
        "baseline_method_id",
        "reference_scene_id",
        "dropped_reference_scene_id",
        "master_sequence_id",
        "order_id",
        "task_metric_source",
        "status",
        "error_type",
        "error_message",
    }
    boolean_fields = {"complete_cluster_population", "sign_consistent"}
    integer_fields = {
        "horizon",
        "sequence_count",
        "tracker_initialization_horizon",
        "deployment_id_switches",
        "identity_transition_opportunities",
        "fragmentation_count",
        "fragmentation_opportunities",
        "merge_count",
        "merge_opportunities",
        "gap_opportunities",
        "recovery_attempts",
        "correct_recoveries",
        "update_scan_count",
        "update_point_count",
        "cumulative_scan_count",
        "cumulative_point_count",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
        "model_input_bytes",
        "cumulative_model_input_bytes",
        "persistent_state_bytes",
        "explicit_history_input_bytes",
        "reference_scene_count",
        "cluster_count",
        "missing_cluster_count",
        "pair_count",
        "bootstrap_replicates",
        "seed",
        "remaining_cluster_count",
    }
    try:
        with source.open("r", encoding="utf-8", newline="") as handle:
            raw_rows = list(csv.DictReader(handle))
    except (OSError, UnicodeError, csv.Error) as error:
        raise AdaptationRunError(f"cannot decode CSV: {source}") from error
    if not raw_rows:
        raise AdaptationRunError(f"required CSV is empty: {source}")
    rows = []
    for raw in raw_rows:
        row: dict[str, object] = {}
        for field, raw_value in raw.items():
            if field in string_fields:
                if not raw_value and field not in {"error_type", "error_message"}:
                    raise AdaptationRunError(f"CSV field {field} is empty")
                row[field] = raw_value
            elif field in boolean_fields:
                if raw_value not in {"True", "False"}:
                    raise AdaptationRunError(f"CSV field {field} is not boolean")
                row[field] = raw_value == "True"
            elif raw_value == "":
                row[field] = None
            elif field in integer_fields:
                try:
                    row[field] = int(raw_value)
                except ValueError as error:
                    raise AdaptationRunError(
                        f"CSV field {field} is not an integer"
                    ) from error
            else:
                try:
                    row[field] = float(raw_value)
                except ValueError as error:
                    raise AdaptationRunError(
                        f"CSV field {field} is not numeric"
                    ) from error
        rows.append(row)
    return rows


def _publish_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise AdaptationRunError("cannot publish an empty adaptation CSV")
    fields = tuple(rows[0])
    _publish_exact(path, _csv_bytes(rows, fields))


def _manifest_identity(key: Mapping[str, object]) -> str:
    return json.dumps(
        dict(key),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _validate_adapted_manifests(
    *,
    expected_keys: Sequence[Mapping[str, object]],
    expected_provenance: Mapping[str, object],
    source_commit: str,
) -> tuple[dict[str, object], dict[str, object]]:
    from scripts.system_comparison_inference import build_full_history_cache_manifest

    prediction = _load_json(PREDICTION_MANIFEST, name="adapted prediction manifest")
    entries = prediction.get("entries")
    if not isinstance(entries, list):
        raise AdaptationRunError("adapted prediction manifest lacks entries")
    rebuilt_prediction = build_full_history_cache_manifest(
        entries,
        expected_keys=expected_keys,
        expected_provenance=expected_provenance,
        cache_directory=PREDICTION_ENTRIES,
    )
    if prediction != rebuilt_prediction:
        raise AdaptationRunError("adapted prediction manifest differs")
    sidecar = _load_json(SIDECAR_MANIFEST, name="adapted sidecar manifest")
    sidecar_entries = sidecar.get("entries")
    if not isinstance(sidecar_entries, list):
        raise AdaptationRunError("adapted sidecar manifest lacks entries")
    rebuilt_sidecar = build_adapted_sidecar_manifest(
        entries=sidecar_entries,
        expected_keys=expected_keys,
        prediction_manifest=prediction,
        source_commit=source_commit,
        cache_directory=SIDECAR_ENTRIES,
    )
    if sidecar != rebuilt_sidecar:
        raise AdaptationRunError("adapted sidecar manifest differs")
    return prediction, sidecar


def _frozen_phase_ii_rows() -> list[dict[str, object]]:
    rows = _read_typed_csv(PHASE_I_RESULTS)
    selected = [
        row
        for row in rows
        if row.get("method_id") in {"FullHistoryNative", "B2", "Persist4D"}
    ]
    if len(selected) != 3 * 129 * 4:
        raise AdaptationRunError("frozen Phase II per-sequence coverage differs")
    return selected


def _frozen_pooled_task_rows() -> list[dict[str, object]]:
    from scripts.reviewer_closure_analysis import TASK_FIELDS

    target_methods = {
        "FullHistory": ("FullHistoryFrozenNative", "FullHistoryFrozenB2"),
        "Persist4D": ("Persist4D",),
    }
    result = []
    for path in (SYSTEM_AGGREGATE, SYSTEM_PER_ORDER):
        for row in _read_typed_csv(path):
            source_method = row.get("method")
            if source_method not in target_methods:
                continue
            order = row.get("order_id")
            horizon = row.get("horizon")
            sequence_count = row.get("sequence_count")
            if (
                order not in {"all", "canonical", "reverse", "sha256_seed45"}
                or horizon not in adaptation.HORIZONS
                or sequence_count != (129 if order == "all" else 43)
            ):
                raise AdaptationRunError("frozen pooled task cell differs")
            for method_id in target_methods[str(source_method)]:
                result.append(
                    {
                        "method_id": method_id,
                        "order_id": order,
                        "horizon": horizon,
                        "sequence_count": sequence_count,
                        "task_metric_source": (
                            "frozen_persist4d_cache"
                            if method_id == "Persist4D"
                            else "frozen_full_history_cache"
                        ),
                        **{field: row[field] for field in TASK_FIELDS},
                    }
                )
    if len(result) != 3 * 4 * 4:
        raise AdaptationRunError("frozen pooled task coverage differs")
    return result


def run_evaluate(*, metadata_path: Path) -> dict[str, object]:
    from scripts.reviewer_closure_analysis import TASK_FIELDS
    from scripts.reviewer_closure_sidecar import (
        load_full_history_observation_sidecar_entry,
        sidecar_key_for_source_prediction,
    )
    from scripts.reviewer_closure_tracking import (
        FullHistoryTrackerSequence,
        _tracker_stage_from_payloads,
        build_full_history_tracker_factories,
        evaluate_full_history_tracker_sequences,
        full_history_tracker_identity_rows,
    )
    from scripts.system_comparison_inference import load_full_history_cache_entry
    from scripts.system_comparison_metrics import (
        CausalTaskAccumulator,
        causal_prefix_pair_from_payload,
        compute_causal_task_metrics,
    )

    inference_source_commit = _prediction_inference_source_commit()
    setup, system_manifest, training = _build_adapted_setup(
        metadata_path=metadata_path,
        device_name=None,
        source_commit=inference_source_commit,
    )
    expected = _expected_keys(system_manifest)
    source_commit = _git_head()
    prediction_manifest, sidecar_manifest = _validate_adapted_manifests(
        expected_keys=expected,
        expected_provenance=setup.full_provenance,
        source_commit=inference_source_commit,
    )
    prediction_entries = {
        _manifest_identity(entry["key"]): entry
        for entry in prediction_manifest["entries"]
    }
    sidecar_entries = {
        _manifest_identity(entry["key"]): entry
        for entry in sidecar_manifest["entries"]
    }
    scopes: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    for key in expected:
        scope = (str(key["master_sequence_id"]), str(key["order_id"]))
        scopes.setdefault(scope, []).append(key)
    if len(scopes) != 129:
        raise AdaptationRunError("adapted evaluation sequence coverage differs")

    accumulators = {
        (order, horizon): CausalTaskAccumulator()
        for order in ("all", *adaptation.ORDERS)
        for horizon in adaptation.HORIZONS
    }
    adapted_task_rows: list[dict[str, object]] = []

    def sequences():
        for scope in sorted(scopes):
            keys = sorted(scopes[scope], key=lambda value: int(value["horizon"]))
            stages = []
            for key in keys:
                prediction_entry = prediction_entries.get(_manifest_identity(key))
                sidecar_key = sidecar_key_for_source_prediction(key)
                sidecar_entry = sidecar_entries.get(_manifest_identity(sidecar_key))
                if prediction_entry is None or sidecar_entry is None:
                    raise AdaptationRunError("adapted evaluation cache binding differs")
                prediction = load_full_history_cache_entry(
                    PREDICTION_ENTRIES,
                    prediction_entry,
                    expected_provenance=prediction_manifest["provenance"],
                )
                sidecar = load_full_history_observation_sidecar_entry(
                    SIDECAR_ENTRIES,
                    sidecar_entry,
                )
                pair = causal_prefix_pair_from_payload(prediction)
                horizon = int(key["horizon"])
                order = str(key["order_id"])
                accumulators[("all", horizon)].update(pair)
                accumulators[(order, horizon)].update(pair)
                adapted_task_rows.append(
                    {
                        "reference_scene_id": str(key["reference_scene_id"]),
                        "master_sequence_id": str(key["master_sequence_id"]),
                        "order_id": order,
                        "horizon": horizon,
                        **compute_causal_task_metrics([pair]),
                    }
                )
                stages.append(_tracker_stage_from_payloads(sidecar, prediction))
            yield FullHistoryTrackerSequence(stages=tuple(stages))

    tracking = evaluate_full_history_tracker_sequences(
        sequences(),
        tracker_factories={"B2": build_full_history_tracker_factories()["B2"]},
    )
    adapted_identity_rows = full_history_tracker_identity_rows(tracking)
    per_sequence = adaptation.merge_phase_ii_per_sequence(
        adapted_task_rows=adapted_task_rows,
        adapted_identity_rows=adapted_identity_rows,
        frozen_rows=_frozen_phase_ii_rows(),
    )
    pooled_tasks = _frozen_pooled_task_rows()
    for order in ("all", *adaptation.ORDERS):
        for horizon in adaptation.HORIZONS:
            metrics = accumulators[(order, horizon)].compute()
            for method_id in (
                "FullHistoryAdaptedNative",
                "FullHistoryAdaptedB2",
            ):
                pooled_tasks.append(
                    {
                        "method_id": method_id,
                        "order_id": order,
                        "horizon": horizon,
                        "sequence_count": 129 if order == "all" else 43,
                        "task_metric_source": "adapted_checkpoint_cache",
                        **{field: metrics[field] for field in TASK_FIELDS},
                    }
                )
    summaries = adaptation.aggregate_phase_ii_results(
        per_sequence_rows=per_sequence,
        pooled_task_rows=pooled_tasks,
    )
    statistics = adaptation.paired_phase_ii_statistics(
        per_sequence,
        challenger_method_id="FullHistoryAdaptedB2",
        baseline_method_id="Persist4D",
    )
    outputs = {
        ADAPTATION_PER_SEQUENCE: per_sequence,
        ADAPTATION_RESULTS: summaries["results"],
        ADAPTATION_PER_ORDER: summaries["per_order"],
        ADAPTATION_BOOTSTRAP: statistics["bootstrap"],
        ADAPTATION_ORDER_STATISTICS: statistics["order_robustness"],
        ADAPTATION_LOSO: statistics["leave_one_scene_out"],
    }
    for path, rows in outputs.items():
        _publish_csv(path, rows)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "status": "pass",
        "paper_name": "ReScene4D T2-to-T3 Horizon-Adapted",
        "selected_level": 2,
        "source_commit": source_commit,
        "inference_source_commit": inference_source_commit,
        "checkpoint_sha256": training["checkpoint_sha256"],
        "training_manifest_content_sha256": training[
            "training_manifest_content_sha256"
        ],
        "prediction_manifest_content_sha256": prediction_manifest["content_sha256"],
        "sidecar_manifest_content_sha256": sidecar_manifest["content_sha256"],
        "coverage": {
            "reference_scene_count": 6,
            "sequence_count": 129,
            "horizons": list(adaptation.HORIZONS),
            "per_sequence_row_count": len(per_sequence),
            "result_row_count": len(summaries["results"]),
            "per_order_row_count": len(summaries["per_order"]),
        },
        "statistics": {
            "pooled_task_metrics": "official_accumulator_over_all_sequences",
            "cluster_statistics": "cluster_macro_of_per_sequence_official_metrics",
            "cluster_count": 6,
            "bootstrap_replicates": 10_000,
            "seed": 45,
        },
        "artifacts": {
            path.name: {"sha256": _sha256_file(path), "row_count": len(rows)}
            for path, rows in outputs.items()
        },
    }
    manifest["content_sha256"] = _content_sha256(manifest)
    _publish_exact(
        ADAPTATION_EVALUATION_MANIFEST,
        _canonical_json_bytes(manifest),
    )
    return {
        "status": "pass",
        "per_sequence_row_count": len(per_sequence),
        "result_row_count": len(summaries["results"]),
        "manifest_content_sha256": manifest["content_sha256"],
    }


def run_profile(*, device_name: str, metadata_path: Path) -> dict[str, object]:
    import torch
    from torch import Tensor

    from scripts.evaluate_persist4d import (
        _move_data_to_device,
        _move_targets_to_device,
        _segment_stages,
    )
    from scripts.profile_system_comparison import (
        PROTOCOL_MEASURED_REPEATS,
        PROTOCOL_WARMUP_REPEATS,
        ProfilingError,
        build_profile_subset,
        measure_cuda_repeats,
    )
    from scripts.system_comparison_inference import (
        deterministic_inference_runtime,
        model_input_storage_bytes,
    )

    setup, system_manifest, training = _build_adapted_setup(
        metadata_path=metadata_path,
        device_name=device_name,
    )
    if not isinstance(setup.device, torch.device) or setup.system is None:
        raise AdaptationRunError("adapted profiling setup did not initialize CUDA")
    device = setup.device
    units = build_profile_subset(system_manifest)

    def field(value: object, name: str) -> object:
        if isinstance(value, Mapping):
            return value.get(name)
        return getattr(value, name, None)

    def prepare(unit, indices: tuple[int, ...]):
        from types import SimpleNamespace

        sample = setup.dataset.load_scan_indices(
            unit.context_index,
            indices,
            change_file=None,
        )
        data, targets, names = setup.collate([sample])
        if list(names) != [unit.master_sequence_id] or len(targets) != 1:
            raise ProfilingError("adapted profile collator changed sequence identity")
        target_full_values = field(data, "target_full")
        if (
            isinstance(target_full_values, (str, bytes))
            or not isinstance(target_full_values, Sequence)
            or len(target_full_values) != 1
            or not isinstance(target_full_values[0], Mapping)
        ):
            raise ProfilingError("adapted profile input lacks one full target")
        temporal_stages = target_full_values[0].get("temporal_stages")
        if not isinstance(temporal_stages, Tensor) or temporal_stages.ndim != 1:
            raise ProfilingError("adapted profile target stages are invalid")
        input_bytes = model_input_storage_bytes(data)
        data = _move_data_to_device(data, device)
        targets = _move_targets_to_device(targets, device)
        target = targets[0]
        stages = _segment_stages(target)
        raw_coordinates = setup.system._process_raw_coordinates(data)
        return SimpleNamespace(
            data=data,
            target=target,
            stages=stages,
            raw_coordinates=raw_coordinates,
            point_count=int(temporal_stages.numel()),
            input_bytes=input_bytes,
        )

    rows: list[dict[str, object]] = []
    with deterministic_inference_runtime(45, device):
        for unit in units:
            cumulative_points = 0
            cumulative_bytes = 0
            for horizon in adaptation.HORIZONS:
                base = {
                    "method": "FullHistoryAdapted",
                    "reference_scene_id": unit.reference_scene_id,
                    "master_sequence_id": unit.master_sequence_id,
                    "order_id": unit.order_id,
                    "horizon": horizon,
                }
                try:
                    torch.cuda.empty_cache()
                    prepared = prepare(unit, unit.scan_indices[:horizon])

                    def operation_factory(prepared=prepared):
                        def operation(prepared=prepared):
                            with torch.inference_mode():
                                return setup.system(
                                    prepared.data,
                                    point2segment=[prepared.target["point2segment"]],
                                    raw_coordinates=prepared.raw_coordinates,
                                    is_eval=True,
                                )

                        return operation

                    cumulative_points += prepared.point_count
                    cumulative_bytes += prepared.input_bytes
                    profile = measure_cuda_repeats(
                        operation_factory,
                        device=device,
                        warmup_repeats=PROTOCOL_WARMUP_REPEATS,
                        measured_repeats=PROTOCOL_MEASURED_REPEATS,
                        enforce_protocol=True,
                    )
                    rows.append(
                        {
                            **base,
                            "status": "pass",
                            "error_type": "",
                            "error_message": "",
                            "median_latency_ms": profile.median_ms,
                            "mean_latency_ms": profile.mean_ms,
                            "std_latency_ms": profile.std_ms,
                            "peak_allocated_bytes": profile.peak_allocated_bytes,
                            "peak_reserved_bytes": profile.peak_reserved_bytes,
                            "peak_allocated_mib": profile.peak_allocated_bytes
                            / (1024**2),
                            "peak_reserved_mib": profile.peak_reserved_bytes
                            / (1024**2),
                            "update_scan_count": horizon,
                            "cumulative_scan_count": horizon * (horizon + 1) // 2 - 1,
                            "update_point_count": prepared.point_count,
                            "cumulative_point_count": cumulative_points,
                            "model_input_bytes": prepared.input_bytes,
                            "cumulative_model_input_bytes": cumulative_bytes,
                            "persistent_state_bytes": None,
                            "explicit_history_input_bytes": prepared.input_bytes,
                        }
                    )
                except Exception as error:  # noqa: BLE001 - preserve failed cell.
                    torch.cuda.empty_cache()
                    rows.append(
                        {
                            **base,
                            "status": "fail",
                            "error_type": type(error).__name__,
                            "error_message": str(error).replace("\n", " ")[:500],
                            "median_latency_ms": None,
                            "mean_latency_ms": None,
                            "std_latency_ms": None,
                            "peak_allocated_bytes": None,
                            "peak_reserved_bytes": None,
                            "peak_allocated_mib": None,
                            "peak_reserved_mib": None,
                            "update_scan_count": None,
                            "cumulative_scan_count": None,
                            "update_point_count": None,
                            "cumulative_point_count": None,
                            "model_input_bytes": None,
                            "cumulative_model_input_bytes": None,
                            "persistent_state_bytes": None,
                            "explicit_history_input_bytes": None,
                        }
                    )
    rows.sort(
        key=lambda row: (
            str(row["reference_scene_id"]),
            int(row["horizon"]),
        )
    )
    _publish_csv(ADAPTATION_PROFILE_RESULTS, rows)
    failures = sum(row["status"] != "pass" for row in rows)
    if failures:
        raise ProfilingError(f"adapted profile retained {failures} failed cells")
    compute_rows = adaptation.build_phase_ii_compute_rows(
        adapted_profile_rows=rows,
        frozen_profile_rows=_read_typed_csv(
            PROJECT_ROOT / "artifacts/system_comparison/profile_results.csv"
        ),
    )
    _publish_csv(ADAPTATION_COMPUTE, compute_rows)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "status": "pass",
        "source_commit": _git_head(),
        "checkpoint_sha256": training["checkpoint_sha256"],
        "profile_cluster_count": 6,
        "horizons": list(adaptation.HORIZONS),
        "warmup_repeats": PROTOCOL_WARMUP_REPEATS,
        "measured_repeats": PROTOCOL_MEASURED_REPEATS,
        "raw_profile_sha256": _sha256_file(ADAPTATION_PROFILE_RESULTS),
        "compute_table_sha256": _sha256_file(ADAPTATION_COMPUTE),
        "raw_profile_row_count": len(rows),
        "compute_row_count": len(compute_rows),
    }
    manifest["content_sha256"] = _content_sha256(manifest)
    _publish_exact(ADAPTATION_PROFILE_MANIFEST, _canonical_json_bytes(manifest))
    return {
        "status": "pass",
        "raw_profile_row_count": len(rows),
        "compute_row_count": len(compute_rows),
        "manifest_content_sha256": manifest["content_sha256"],
    }


def _validate_content_manifest(path: Path, *, name: str) -> dict[str, object]:
    manifest = _load_json(path, name=name)
    if manifest.get("status") != "pass" or manifest.get("content_sha256") != (
        _content_sha256(manifest)
    ):
        raise AdaptationRunError(f"{name} content binding differs")
    return manifest


def run_gate() -> dict[str, object]:
    evaluation = _validate_content_manifest(
        ADAPTATION_EVALUATION_MANIFEST,
        name="adaptation evaluation manifest",
    )
    profile = _validate_content_manifest(
        ADAPTATION_PROFILE_MANIFEST,
        name="adaptation profile manifest",
    )
    evidence = adaptation.build_gate_ii_evidence(
        result_rows=_read_typed_csv(ADAPTATION_RESULTS),
        bootstrap_rows=_read_typed_csv(ADAPTATION_BOOTSTRAP),
        order_rows=_read_typed_csv(ADAPTATION_ORDER_STATISTICS),
        loso_rows=_read_typed_csv(ADAPTATION_LOSO),
        compute_rows=_read_typed_csv(ADAPTATION_COMPUTE),
    )
    gate = adaptation.derive_gate_ii(
        task_evidence=evidence["task"],
        identity_evidence=evidence["identity"],
        compute_evidence=evidence["compute"],
        config=adaptation.load_phase_ii_evaluation_config(PHASE_II_CONFIG),
    )
    gate["source_commit"] = _git_head()
    gate["evaluation_manifest_content_sha256"] = evaluation["content_sha256"]
    gate["profile_manifest_content_sha256"] = profile["content_sha256"]
    gate["content_sha256"] = _content_sha256(gate)
    for path, rows in (
        (GATE_II_TASK_EVIDENCE, evidence["task"]),
        (GATE_II_IDENTITY_EVIDENCE, evidence["identity"]),
        (GATE_II_COMPUTE_EVIDENCE, evidence["compute"]),
    ):
        _publish_csv(path, rows)
    _publish_exact(GATE_II_PATH, _canonical_json_bytes(gate))
    if gate["classification"] == "FULL_HISTORY_DOMINANT":
        lines = [
            "# Full-History Adapted Challenge",
            "",
            "Gate II: `FULL_HISTORY_DOMINANT`.",
            "",
            "The preregistered T4/T5 task, identity, and compute conditions all passed.",
            "The adapted deployment is therefore the strongest challenger in the frozen comparison scope.",
            "No additional training or post-result threshold changes were used.",
            "",
        ]
        _publish_exact(ADAPTED_CHALLENGE_REPORT, "\n".join(lines).encode("ascii"))
    return {
        "status": "pass",
        "classification": gate["classification"],
        "content_sha256": gate["content_sha256"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("smoke", "cache", "finalize", "evaluate", "profile", "gate"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "smoke":
        result = run_cache(
            device_name=arguments.device,
            metadata_path=arguments.metadata,
            smoke_only=True,
        )
    elif arguments.command == "cache":
        result = run_cache(
            device_name=arguments.device,
            metadata_path=arguments.metadata,
            smoke_only=False,
        )
    elif arguments.command == "finalize":
        result = run_finalize(metadata_path=arguments.metadata)
    elif arguments.command == "evaluate":
        result = run_evaluate(metadata_path=arguments.metadata)
    elif arguments.command == "profile":
        result = run_profile(
            device_name=arguments.device,
            metadata_path=arguments.metadata,
        )
    else:
        result = run_gate()
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AdaptationRunError",
    "build_adapted_provenance",
    "build_adapted_sidecar_manifest",
    "build_parser",
    "produce_adapted_batch",
    "run_cache",
    "run_evaluate",
    "run_finalize",
    "run_gate",
    "run_profile",
    "validate_training_checkpoint",
]
