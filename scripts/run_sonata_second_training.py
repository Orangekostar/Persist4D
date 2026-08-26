"""Validate SP0 and launch the unique formal Sonata training candidate."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.sonata_second_preflight import (
    EXPECTED_RIO_CONTENT,
    EXPECTED_SCANNET_CONTENT,
    SS1_ARTIFACT_DIR,
    _compose_config,
    _weight_flash_attn_active,
)
from utils.sonata_second_preflight import (
    SONATA_BRANCH,
    SONATA_CONFIG_NAME,
    SonataSecondPreflightError,
    build_sonata_environment_manifest,
    build_sonata_source_tree_contract,
    build_sonata_training_semantics,
    canonical_sha256,
    directory_content_manifest,
    file_sha256,
    portable_resolved_config,
    validate_formal_resource_blocker,
    validate_sonata_preflight_authorization,
    validate_sonata_training_config_contract,
)
from utils.sonata_training_evidence import append_runtime_event
from utils.sonata_weight_provenance import (
    OFFICIAL_SONATA_WEIGHT_SPEC,
    validate_sonata_weight_manifest,
)

DEFAULT_ARTIFACT_DIR = (
    PROJECT_ROOT / "artifacts" / "sonata_second_perception_v1" / "preflight"
)
DEFAULT_DEVICES = (1, 2)
CANDIDATE_RECORD_NAME = ".sonata_second_candidate.json"
RESOURCE_BLOCKER_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "sonata_second_perception_v1"
    / "training"
    / "RESOURCE_BLOCKER.json"
)


def _load_json(path: Path, *, name: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SonataSecondPreflightError(f"{name} is not readable JSON") from error
    if not isinstance(payload, dict):
        raise SonataSecondPreflightError(f"{name} must contain an object")
    return payload


def _require_ancestor(commit: str) -> None:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
    )
    if result.returncode != 0:
        raise SonataSecondPreflightError(
            "authorized Sonata source commit is not an ancestor of HEAD"
        )


def require_formal_authorization(
    *,
    weight_path: Path,
    training_output_dir: Path,
    artifact_dir: Path,
    allow_stale_resume: bool = False,
) -> Path:
    if Path.cwd().resolve() != PROJECT_ROOT.resolve():
        raise SonataSecondPreflightError("training launcher requires repository root")
    artifact_dir = Path(artifact_dir)
    authorization = _load_json(
        artifact_dir / "preflight_authorization.json",
        name="Sonata preflight authorization",
    )
    resolved_config = _load_json(
        artifact_dir / "resolved_config.json", name="resolved Sonata config"
    )
    data_manifest = _load_json(
        artifact_dir / "data_manifest.json", name="Sonata data manifest"
    )
    environment_manifest = _load_json(
        artifact_dir / "environment_manifest.json",
        name="Sonata environment manifest",
    )
    training_semantics = _load_json(
        artifact_dir / "training_semantics.json",
        name="Sonata training semantics",
    )
    ss1_manifest_path = SS1_ARTIFACT_DIR / "sonata_weight_manifest.json"
    load_audit_path = SS1_ARTIFACT_DIR / "sonata_load_key_audit.json"
    ss1_manifest = _load_json(ss1_manifest_path, name="SS1 weight manifest")
    validate_sonata_weight_manifest(
        ss1_manifest,
        weight_path,
        spec=OFFICIAL_SONATA_WEIGHT_SPEC,
        require_official=True,
    )

    source_contract = build_sonata_source_tree_contract(
        PROJECT_ROOT, require_clean=True
    )
    if source_contract["branch"] != SONATA_BRANCH:
        raise SonataSecondPreflightError("formal Sonata branch mismatch")
    authorized_source_commit = authorization.get("bindings", {}).get("source_commit")
    if not isinstance(authorized_source_commit, str):
        raise SonataSecondPreflightError("authorization source commit is missing")
    _require_ancestor(authorized_source_commit)

    verified_weight = (
        Path(training_output_dir)
        / ".verified_inputs"
        / f"sonata-{OFFICIAL_SONATA_WEIGHT_SPEC.sha256}.pth"
    )
    validate_sonata_weight_manifest(
        ss1_manifest,
        verified_weight,
        spec=OFFICIAL_SONATA_WEIGHT_SPEC,
        require_official=True,
    )
    cfg = _compose_config(verified_weight, training_output_dir)
    config_errors = validate_sonata_training_config_contract(
        cfg,
        expected_weight_path=verified_weight,
        expected_output_dir=training_output_dir,
    )
    if config_errors:
        raise SonataSecondPreflightError("; ".join(config_errors))
    current_portable_config = portable_resolved_config(
        cfg,
        expected_weight_path=verified_weight,
        expected_output_dir=training_output_dir,
        weight_sha256=OFFICIAL_SONATA_WEIGHT_SPEC.sha256,
    )
    if current_portable_config != resolved_config:
        raise SonataSecondPreflightError("resolved Sonata config is stale")

    for name, root, expected in (
        ("rio", PROJECT_ROOT / "data" / "processed" / "rio", EXPECTED_RIO_CONTENT),
        (
            "scannet",
            PROJECT_ROOT / "data" / "processed" / "scannet",
            EXPECTED_SCANNET_CONTENT,
        ),
    ):
        observed = directory_content_manifest(root)
        if observed != expected or data_manifest.get("datasets", {}).get(name, {}).get(
            "content"
        ) != observed:
            raise SonataSecondPreflightError(f"authorized {name} data is stale")

    environment_source = dict(source_contract)
    environment_source["source_commit"] = environment_manifest.get("source", {}).get(
        "commit"
    )
    current_environment = build_sonata_environment_manifest(
        source_tree_contract=environment_source,
        flash_attn_active=_weight_flash_attn_active(verified_weight),
    )
    if current_environment != environment_manifest:
        raise SonataSecondPreflightError("authorized runtime environment is stale")

    config_sha256 = canonical_sha256(current_portable_config)
    weight_manifest_sha256 = file_sha256(ss1_manifest_path)
    current_semantics = build_sonata_training_semantics(
        cfg,
        config_sha256=config_sha256,
        weight_manifest_sha256=weight_manifest_sha256,
        load_key_audit_sha256=file_sha256(load_audit_path),
    )
    current_semantics["verified_weight"] = training_semantics.get("verified_weight")
    if current_semantics != training_semantics:
        raise SonataSecondPreflightError("authorized training semantics are stale")

    bindings = {
        "source_tree_sha256": source_contract["content_sha256"],
        "source_commit": authorized_source_commit,
        "config_sha256": config_sha256,
        "weight_manifest_sha256": weight_manifest_sha256,
        "data_manifest_sha256": file_sha256(artifact_dir / "data_manifest.json"),
        "environment_manifest_sha256": file_sha256(
            artifact_dir / "environment_manifest.json"
        ),
        "training_semantics_sha256": file_sha256(
            artifact_dir / "training_semantics.json"
        ),
    }
    validate_sonata_preflight_authorization(
        authorization,
        expected_bindings=bindings,
        enforce_age=not allow_stale_resume,
    )
    return verified_weight.resolve()


def _parse_devices(value: str) -> tuple[int, ...]:
    try:
        devices = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "devices must be comma-separated integers"
        ) from error
    if len(devices) != 2 or len(set(devices)) != 2 or any(item < 0 for item in devices):
        raise argparse.ArgumentTypeError("devices must identify two distinct GPUs")
    return devices


def authorize_unique_candidate(
    training_output_dir: Path,
    contract: dict[str, Any],
) -> str:
    """Create the immutable candidate record or validate an identical resume."""

    output_dir = Path(training_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    record_path = output_dir / CANDIDATE_RECORD_NAME
    if record_path.exists() or record_path.is_symlink():
        if record_path.is_symlink() or not record_path.is_file():
            raise SonataSecondPreflightError("candidate record is not a regular file")
        observed = _load_json(record_path, name="Sonata candidate record")
        if observed != contract:
            raise SonataSecondPreflightError("candidate contract mismatch")
        return "resume"

    if any(output_dir.glob("*.ckpt")):
        raise SonataSecondPreflightError(
            "formal training directory contains an unowned checkpoint"
        )
    runtime_events = output_dir / ".sonata_runtime_events.jsonl"
    if runtime_events.exists() or runtime_events.is_symlink():
        raise SonataSecondPreflightError(
            "formal training directory contains unowned runtime events"
        )
    encoded = (
        json.dumps(contract, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True)
        + "\n"
    ).encode("ascii")
    descriptor = os.open(
        record_path,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY,
        0o444,
    )
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return "fresh"


def _candidate_contract(
    *,
    artifact_dir: Path,
    smoke_authorization: dict[str, Any],
    devices: tuple[int, ...],
) -> dict[str, Any]:
    preflight = _load_json(
        Path(artifact_dir) / "preflight_authorization.json",
        name="Sonata preflight authorization",
    )
    data_manifest = _load_json(
        Path(artifact_dir) / "data_manifest.json",
        name="Sonata data manifest",
    )
    training_semantics = _load_json(
        Path(artifact_dir) / "training_semantics.json",
        name="Sonata training semantics",
    )
    resource_blocker = _load_json(
        RESOURCE_BLOCKER_PATH, name="Sonata formal resource blocker"
    )
    blocker_evidence = validate_formal_resource_blocker(resource_blocker)
    blocker_sha256 = file_sha256(RESOURCE_BLOCKER_PATH)
    smoke_bindings = smoke_authorization.get("bindings", {})
    if smoke_bindings.get("resource_blocker_sha256") != blocker_sha256:
        raise SonataSecondPreflightError("SSMOKE resource blocker binding differs")
    samples_per_epoch = data_manifest.get("mixed_runtime", {}).get(
        "sampler_num_samples"
    )
    effective_batch = training_semantics.get("effective_global_batch")
    microbatch = training_semantics.get("physical_batch_per_device")
    accumulation = training_semantics.get("accumulate_grad_batches")
    if (
        samples_per_epoch != 2112
        or microbatch != 2
        or accumulation != 8
        or effective_batch != 32
    ):
        raise SonataSecondPreflightError("formal training budget inputs mismatch")
    bindings = dict(preflight.get("bindings", {}))
    bindings["preflight_authorization_sha256"] = preflight.get(
        "authorization_sha256"
    )
    bindings["smoke_authorization_sha256"] = smoke_authorization.get(
        "authorization_sha256"
    )
    bindings["weight_sha256"] = OFFICIAL_SONATA_WEIGHT_SPEC.sha256
    bindings["resource_blocker_sha256"] = blocker_sha256
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "active",
        "bindings": bindings,
        "recipe": {
            "seed": 45,
            "epochs": 450,
            "devices": list(devices),
            "microbatch_per_gpu": microbatch,
            "accumulate_grad_batches": accumulation,
            "effective_global_batch": effective_batch,
            "samples_per_epoch": samples_per_epoch,
            "optimizer_steps_per_epoch": (
                samples_per_epoch + effective_batch - 1
            )
            // effective_batch,
            "checkpoint_selection": "highest val_mean_t-AP",
        },
        "reauthorization": {
            "basis": (
                "user_authorized_recommended_configuration_after_resource_failure"
            ),
            "reason_gate": blocker_evidence["gate"],
            "resource_blocker_sha256": blocker_sha256,
            "supersedes_candidate_id": blocker_evidence[
                "superseded_candidate_id"
            ],
        },
    }
    payload["candidate_id"] = canonical_sha256(payload)
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weight-path", type=Path, required=True)
    parser.add_argument("--training-output-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--devices", type=_parse_devices, default=DEFAULT_DEVICES)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="launch training after authorization; otherwise validate only",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    candidate_path = Path(args.training_output_dir) / CANDIDATE_RECORD_NAME
    allow_stale_resume = (
        args.execute and candidate_path.is_file() and not candidate_path.is_symlink()
    )
    verified_weight = require_formal_authorization(
        weight_path=args.weight_path,
        training_output_dir=args.training_output_dir,
        artifact_dir=args.artifact_dir,
        allow_stale_resume=allow_stale_resume,
    )
    if not args.execute:
        print(json.dumps({"gate": "SP0-PASS", "training_launched": False}))
        return 0
    from scripts.sonata_second_smoke import require_smoke_authorization

    cfg = _compose_config(verified_weight, args.training_output_dir)
    smoke_authorization = require_smoke_authorization(
        expected_microbatch_per_gpu=int(cfg.data.batch_size),
        expected_accumulation=int(cfg.trainer.accumulate_grad_batches),
        expected_devices=args.devices,
    )
    candidate = _candidate_contract(
        artifact_dir=args.artifact_dir,
        smoke_authorization=smoke_authorization,
        devices=args.devices,
    )
    launch_mode = authorize_unique_candidate(args.training_output_dir, candidate)
    append_runtime_event(
        args.training_output_dir,
        {
            "schema_version": 1,
            "event": "launch_authorized",
            "candidate_id": candidate["candidate_id"],
            "launch_mode": launch_mode,
            "checkpoint_count": len(list(Path(args.training_output_dir).glob("*.ckpt"))),
        },
    )
    environment = os.environ.copy()
    environment["SONATA_CHECKPOINT"] = str(verified_weight)
    environment["SONATA_OUTPUT_DIR"] = str(Path(args.training_output_dir).resolve())
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(
        str(device) for device in args.devices
    )
    command = [
        sys.executable,
        str(PROJECT_ROOT / "main_instance_segmentation.py"),
        "--config-name",
        SONATA_CONFIG_NAME,
    ]
    os.execve(sys.executable, command, environment)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
