#!/usr/bin/env python3
"""Validate and launch one formally authorized ReScene root-cause curve."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.rescene_rootcause_preflight import (
    AUDIT_ROOT,
    INITIALIZATION_MANIFEST,
    ROOTCAUSE_CONFIG_NAME,
    SELECTED_SHORT_VARIANTS,
    _git,
    _load_json,
    _runtime_environment,
    _stable_file_identity,
    compose_variant_config,
    portable_variant_config,
    variant_overrides,
)
from utils.rescene_rootcause_preflight import (
    RootCauseContractError,
    canonical_sha256,
    validate_portable_payload,
)

BRANCH = "research/persist4d-rescene-task-learning-root-cause-v1"
CANDIDATE_RECORD_NAME = ".rootcause_candidate.json"
DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "artifacts/rescene_task_learning_root_cause_v1/short_curves/variant_manifest.json"
)


class RootCauseLaunchError(RuntimeError):
    """Raised when a formal root-cause training launch is not exact."""


def parse_devices(value: str) -> tuple[int, int]:
    try:
        devices = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise RootCauseLaunchError(
            "devices must be two comma-separated integers"
        ) from error
    if len(devices) != 2 or len(set(devices)) != 2 or any(item < 0 for item in devices):
        raise RootCauseLaunchError("devices must identify two distinct GPUs")
    return devices


def _atomic_candidate_record(path: Path, payload: Mapping[str, object]) -> None:
    encoded = (
        json.dumps(
            payload, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True
        )
        + "\n"
    ).encode("ascii")
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def authorize_unique_candidate(
    output_dir: Path, candidate: Mapping[str, object]
) -> str:
    """Create an immutable owner record or require an exact resume."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    record = output_dir / CANDIDATE_RECORD_NAME
    if record.exists() or record.is_symlink():
        if record.is_symlink() or not record.is_file():
            raise RootCauseLaunchError("candidate record is not a regular file")
        observed = _load_json(record, name="root-cause candidate record")
        if observed != dict(candidate):
            raise RootCauseLaunchError("candidate contract mismatch")
        return "resume"
    if any(output_dir.glob("*.ckpt")):
        raise RootCauseLaunchError("training directory contains an unowned checkpoint")
    if any(output_dir.iterdir()):
        raise RootCauseLaunchError("training directory is not empty or owned")
    _atomic_candidate_record(record, candidate)
    return "fresh"


def build_launch_environment(
    *,
    variant: str,
    devices: tuple[int, int],
    pretrained: Path,
    common_state: Path,
    common_sha256: str,
    output_dir: Path,
    inherited: Mapping[str, str] | None = None,
) -> dict[str, str]:
    if variant not in SELECTED_SHORT_VARIANTS:
        raise RootCauseLaunchError("variant is not formally selected")
    environment = dict(os.environ if inherited is None else inherited)
    required_python_paths = [
        PROJECT_ROOT / "third_party/concerto",
        PROJECT_ROOT / "third_party/detectron2",
        PROJECT_ROOT / "third_party/sonata",
        PROJECT_ROOT / "third_party/stmetrics",
    ]
    prior_python_path = environment.get("PYTHONPATH")
    python_paths = [str(path) for path in required_python_paths]
    if prior_python_path:
        python_paths.append(prior_python_path)
    environment.update(
        {
            "PYTHONPATH": os.pathsep.join(python_paths),
            "CUDA_VISIBLE_DEVICES": ",".join(str(device) for device in devices),
            "CONCERTO_CHECKPOINT": str(Path(pretrained).resolve()),
            "RESCENE_ROOTCAUSE_VARIANT": variant,
            "RESCENE_ROOTCAUSE_OUTPUT_DIR": str(Path(output_dir).resolve()),
            "RESCENE_ROOTCAUSE_COMMON_STATE": str(Path(common_state).resolve()),
            "RESCENE_ROOTCAUSE_COMMON_SHA256": common_sha256,
            "RESCENE_ROOTCAUSE_OBJECTIVE_MODE": (
                "raw_sum" if variant == "R1" else "weighted"
            ),
        }
    )
    return environment


def build_launch_command(variant: str) -> list[str]:
    if variant not in SELECTED_SHORT_VARIANTS:
        raise RootCauseLaunchError("variant is not formally selected")
    return [
        sys.executable,
        str(PROJECT_ROOT / "main_instance_segmentation.py"),
        "--config-name",
        ROOTCAUSE_CONFIG_NAME,
        *variant_overrides(variant),
    ]


def _require_equal(observed: object, expected: object, *, name: str) -> None:
    if observed != expected:
        raise RootCauseLaunchError(f"{name} differs from formal authorization")


def _validate_authorization_hash(manifest: Mapping[str, Any]) -> None:
    expected = manifest.get("authorization_sha256")
    payload = dict(manifest)
    payload.pop("authorization_sha256", None)
    if not isinstance(expected, str) or canonical_sha256(payload) != expected:
        raise RootCauseLaunchError("variant authorization hash differs")


def _validate_source(manifest: Mapping[str, Any]) -> None:
    if Path.cwd().resolve() != PROJECT_ROOT.resolve():
        raise RootCauseLaunchError("training launcher requires repository root")
    branch = _git("branch", "--show-current")
    if branch != BRANCH:
        raise RootCauseLaunchError("formal root-cause branch differs")
    source_commit = manifest.get("source_commit")
    if not isinstance(source_commit, str):
        raise RootCauseLaunchError("authorized source commit is missing")
    try:
        _git("merge-base", "--is-ancestor", source_commit, "HEAD")
    except RootCauseContractError as error:
        raise RootCauseLaunchError(
            "authorized source commit is not an ancestor of HEAD"
        ) from error
    changed = [
        path
        for path in _git("diff", "--name-only", f"{source_commit}..HEAD").splitlines()
        if path
    ]
    artifact_prefix = "artifacts/rescene_task_learning_root_cause_v1/"
    if any(not path.startswith(artifact_prefix) for path in changed):
        raise RootCauseLaunchError("training source changed after authorization")
    if _git("status", "--porcelain", "--untracked-files=all"):
        raise RootCauseLaunchError("training worktree is not clean")


def _validate_data(manifest: Mapping[str, Any]) -> None:
    from utils.p2_preflight import build_p2_input_manifest

    observed = build_p2_input_manifest(repo_root=PROJECT_ROOT)
    if observed.get("status") != "pass":
        raise RootCauseLaunchError("formal data content hashing failed")
    for dataset in ("rio", "scannet"):
        _require_equal(observed[dataset], manifest["data"][dataset], name=dataset)


def _validate_file_bindings(manifest: Mapping[str, Any]) -> None:
    for name, filename in (
        ("objective", "upstream_local_diff.json"),
        ("data", "data_semantics.json"),
        ("ddp_sampler", "ddp_sampler_summary.json"),
        ("encoder_stochasticity", "encoder_stochasticity_summary.json"),
        ("physical_batch", "physical_batch_summary.json"),
    ):
        _require_equal(
            _stable_file_identity(AUDIT_ROOT / filename)["sha256"],
            manifest["audit_bindings"][name],
            name=f"{name} audit",
        )
    _require_equal(
        _stable_file_identity(INITIALIZATION_MANIFEST)["sha256"],
        manifest["initialization"]["manifest_sha256"],
        name="initialization manifest",
    )
    for key, path in (
        ("config_sha256", PROJECT_ROOT / "conf/metrics/tmap.yaml"),
        (
            "evaluator_sha256",
            PROJECT_ROOT / "scripts/evaluate_rescan_persist4d.py",
        ),
    ):
        _require_equal(
            _stable_file_identity(path)["sha256"],
            manifest["metrics"][key],
            name=f"metric {key}",
        )
    _require_equal(
        _git("rev-parse", "HEAD", cwd=PROJECT_ROOT / "third_party/stmetrics"),
        manifest["metrics"]["stmetrics_commit"],
        name="stmetrics revision",
    )


def require_formal_authorization(
    *,
    variant: str,
    pretrained: Path,
    common_state: Path,
    output_dir: Path,
    manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Recompute every launch-critical binding and return the exact config."""

    try:
        manifest = _load_json(manifest_path, name="root-cause variant authorization")
        validate_portable_payload(manifest)
    except RootCauseContractError as error:
        raise RootCauseLaunchError(str(error)) from error
    _validate_authorization_hash(manifest)
    if (
        manifest.get("status") != "authorized"
        or variant not in manifest.get("selected_variants", ())
    ):
        raise RootCauseLaunchError("variant is not formally authorized")
    _validate_source(manifest)
    _validate_file_bindings(manifest)
    _validate_data(manifest)
    _require_equal(
        _runtime_environment(), manifest["runtime"], name="runtime environment"
    )

    common_identity = _stable_file_identity(common_state)
    pretrained_identity = _stable_file_identity(pretrained)
    expected_common = manifest["initialization"]["common_state"]
    expected_pretrained = manifest["initialization"]["pretrained"]
    for field in ("bytes", "sha256"):
        _require_equal(
            common_identity[field], expected_common[field], name="common initialization"
        )
        _require_equal(
            pretrained_identity[field],
            expected_pretrained[field],
            name="Concerto pretrained encoder",
        )

    runtime_config = compose_variant_config(
        variant,
        pretrained=pretrained,
        common_state=common_state,
        common_sha256=common_identity["sha256"],
        output=output_dir,
    )
    portable_config = portable_variant_config(
        runtime_config,
        variant=variant,
        pretrained_reference=expected_pretrained["reference"],
        common_reference=expected_common["reference"],
    )
    expected_variant = manifest["variants"][variant]
    _require_equal(
        portable_config,
        expected_variant["resolved_config"],
        name="resolved variant config",
    )
    _require_equal(
        canonical_sha256(portable_config),
        expected_variant["config_sha256"],
        name="variant config hash",
    )
    return manifest, portable_config


def build_candidate_contract(
    *,
    variant: str,
    devices: tuple[int, int],
    manifest_path: Path,
    manifest: Mapping[str, Any],
    portable_config: Mapping[str, Any],
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "status": "active",
        "experiment": manifest["experiment"],
        "variant": variant,
        "devices": list(devices),
        "source_commit": manifest["source_commit"],
        "variant_authorization_sha256": manifest["authorization_sha256"],
        "variant_manifest_file_sha256": _stable_file_identity(manifest_path)[
            "sha256"
        ],
        "config_sha256": canonical_sha256(portable_config),
        "common_initialization_sha256": manifest["initialization"]["common_state"][
            "sha256"
        ],
        "pretrained_sha256": manifest["initialization"]["pretrained"]["sha256"],
        "schedule": manifest["schedule"],
    }
    payload["candidate_id"] = canonical_sha256(payload)
    validate_portable_payload(payload)
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=SELECTED_SHORT_VARIANTS, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--common-state", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--devices", type=parse_devices, default=(0, 1))
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    manifest, portable_config = require_formal_authorization(
        variant=args.variant,
        pretrained=args.pretrained,
        common_state=args.common_state,
        output_dir=args.output_dir,
        manifest_path=args.manifest,
    )
    if not args.execute:
        print(
            json.dumps(
                {
                    "gate": "RC3-PASS",
                    "variant": args.variant,
                    "training_launched": False,
                },
                sort_keys=True,
            )
        )
        return 0
    candidate = build_candidate_contract(
        variant=args.variant,
        devices=args.devices,
        manifest_path=args.manifest,
        manifest=manifest,
        portable_config=portable_config,
    )
    launch_mode = authorize_unique_candidate(args.output_dir, candidate)
    environment = build_launch_environment(
        variant=args.variant,
        devices=args.devices,
        pretrained=args.pretrained,
        common_state=args.common_state,
        common_sha256=manifest["initialization"]["common_state"]["sha256"],
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "gate": "RC3-PASS",
                "variant": args.variant,
                "training_launched": True,
                "launch_mode": launch_mode,
                "candidate_id": candidate["candidate_id"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    command = build_launch_command(args.variant)
    os.execve(sys.executable, command, environment)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
