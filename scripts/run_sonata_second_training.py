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
    validate_sonata_preflight_authorization,
    validate_sonata_training_config_contract,
)
from utils.sonata_weight_provenance import (
    OFFICIAL_SONATA_WEIGHT_SPEC,
    validate_sonata_weight_manifest,
)

DEFAULT_ARTIFACT_DIR = (
    PROJECT_ROOT / "artifacts" / "sonata_second_perception_v1" / "preflight"
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
    )
    return verified_weight.resolve()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weight-path", type=Path, required=True)
    parser.add_argument("--training-output-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="launch training after authorization; otherwise validate only",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    verified_weight = require_formal_authorization(
        weight_path=args.weight_path,
        training_output_dir=args.training_output_dir,
        artifact_dir=args.artifact_dir,
    )
    if not args.execute:
        print(json.dumps({"gate": "SP0-PASS", "training_launched": False}))
        return 0
    environment = os.environ.copy()
    environment["SONATA_CHECKPOINT"] = str(verified_weight)
    environment["SONATA_OUTPUT_DIR"] = str(Path(args.training_output_dir).resolve())
    command = [
        sys.executable,
        str(PROJECT_ROOT / "main_instance_segmentation.py"),
        f"--config-name={SONATA_CONFIG_NAME}",
    ]
    os.execve(sys.executable, command, environment)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
