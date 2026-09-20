"""Resumable command line entry point for CrossWindow V1."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from scripts.crosswindow_cache import ASSET_KEYS, build_data_roles, resolve_assets

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/crosswindow_evidence_v1.yaml"
RUN_STATE_SCHEMA = "crosswindow-run-state-v1"


class CampaignError(RuntimeError):
    """Raised when campaign identity, state, or command contracts differ."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CampaignError(f"cannot read JSON: {path}") from error
    if not isinstance(value, dict):
        raise CampaignError(f"JSON root must be an object: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def load_campaign_config(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise CampaignError(f"cannot read campaign config: {path}") from error
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != "crosswindow-evidence-v1"
    ):
        raise CampaignError("campaign config schema differs")
    for section in ("identity", "paths", "runtime", "budget", "data_split"):
        if not isinstance(value.get(section), dict):
            raise CampaignError(f"campaign config lacks {section}")
    return value


def new_run_state(
    *, parent: str, instruction_sha: str, config_sha: str, data_sha: str
) -> dict[str, object]:
    identities = {
        "parent": parent,
        "instruction_sha256": instruction_sha,
        "config_sha256": config_sha,
        "data_sha256": data_sha,
    }
    if (
        not isinstance(parent, str)
        or len(parent) != 40
        or any(character not in "0123456789abcdef" for character in parent)
        or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in (instruction_sha, config_sha, data_sha)
        )
    ):
        raise CampaignError("run-state identities must be Git/SHA256 hex strings")
    return {
        "schema_version": RUN_STATE_SCHEMA,
        "identity": identities,
        "stage": "BOOTSTRAP",
        "stage_status": "PASS",
        "completed_units": [],
        "methods": [],
        "budget_used": {
            "inference_gpu_hours": 0.0,
            "head_training_gpu_hours": 0.0,
            "profile_gpu_hours": 0.0,
            "cpu_core_hours": 0.0,
            "new_cache_bytes": 0,
        },
        "failures": [],
        "next_command": (
            "python -m scripts.crosswindow_campaign preflight "
            "--config configs/crosswindow_evidence_v1.yaml --read-only"
        ),
    }


def atomic_write_run_state(path: Path, state: Mapping[str, object]) -> None:
    if state.get("schema_version") != RUN_STATE_SCHEMA:
        raise CampaignError("run-state schema differs")
    _atomic_json(path, state)


def load_resume_state(
    path: Path,
    *,
    parent: str,
    instruction_sha: str,
    config_sha: str,
    data_sha: str,
) -> dict[str, Any]:
    state = _read_json(path)
    if state.get("schema_version") != RUN_STATE_SCHEMA:
        raise CampaignError("run-state schema differs")
    expected = {
        "parent": parent,
        "instruction_sha256": instruction_sha,
        "config_sha256": config_sha,
        "data_sha256": data_sha,
    }
    if state.get("identity") != expected:
        raise CampaignError("resume identity differs")
    return state


def _git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _available_dev_references(manifest: Mapping[str, object]) -> list[str]:
    records = manifest.get("records")
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise CampaignError("development manifest lacks records")
    references = {
        record.get("reference_id")
        for record in records
        if isinstance(record, Mapping) and isinstance(record.get("reference_id"), str)
    }
    return sorted(references)


def _infer_old_cache_roots(
    values: dict[str, str | None],
    sources: dict[str, str],
    config: Mapping[str, Any],
) -> None:
    old_root_value = values.get("external_run_root")
    if old_root_value is None:
        return
    old_root = Path(old_root_value)
    paths = config["paths"]
    mappings = (
        ("dev_base_cache_root", "dev_base_manifest", "baseline/cache"),
        (
            "dev_supplement_root",
            "dev_supplement_manifest",
            "baseline/control_observations",
        ),
        ("pb_base_cache_root", "pb_base_manifest", "baseline/cache"),
        (
            "pb_supplement_root",
            "pb_supplement_manifest",
            "baseline/control_observations",
        ),
    )
    for asset_key, manifest_key, relative_root in mappings:
        if values[asset_key] is not None:
            continue
        manifest = _read_json(PROJECT_ROOT / paths[manifest_key])
        config_sha = manifest.get("config_sha256")
        if not isinstance(config_sha, str) or len(config_sha) != 64:
            raise CampaignError(f"{manifest_key} lacks config SHA")
        candidate = old_root / relative_root / config_sha[:16]
        if candidate.is_dir():
            values[asset_key] = str(candidate)
            sources[asset_key] = "known_manifest"


def _tracked_asset_status(
    values: Mapping[str, str | None], sources: Mapping[str, str]
) -> dict[str, dict[str, object]]:
    result = {}
    for key in ASSET_KEYS:
        value = values[key]
        path = Path(value) if value is not None else None
        result[key] = {
            "available": bool(path is not None and path.exists()),
            "kind": (
                "directory"
                if path is not None and path.is_dir()
                else "file"
                if path is not None and path.is_file()
                else "missing"
            ),
            "source": sources[key],
        }
    return result


def _bootstrap(args: argparse.Namespace) -> int:
    config_path = Path(args.config).resolve()
    config = load_campaign_config(config_path)
    expected_parent = config["identity"]["parent_commit"]
    if _git_head() != expected_parent:
        raise CampaignError("worktree HEAD differs from fixed parent")
    paths = config["paths"]
    instruction_path = PROJECT_ROOT / paths["instruction"]
    instruction_sha = _sha256(instruction_path)
    if instruction_sha != config["identity"]["instruction_sha256"]:
        raise CampaignError("execution instruction SHA differs")
    data_contract_path = PROJECT_ROOT / paths["data_contract"]
    fallback = _read_json(Path(args.assets_from).resolve()) if args.assets_from else {}
    explicit = {
        key: getattr(args, key)
        for key in ASSET_KEYS
        if getattr(args, key, None) is not None
    }
    resolved = resolve_assets(explicit=explicit, environ=os.environ, fallback=fallback)
    values, sources = dict(resolved.values), dict(resolved.sources)
    _infer_old_cache_roots(values, sources, config)
    if args.external_root:
        values["external_run_root"] = str(Path(args.external_root).resolve())
        sources["external_run_root"] = "cli"
    external_root_value = values["external_run_root"]
    if external_root_value is None:
        raise CampaignError("bootstrap requires a distinct external run root")
    external_root = Path(external_root_value)
    external_root.mkdir(parents=True, exist_ok=True)
    _atomic_json(
        external_root / "assets.local.json",
        {key: values[key] for key in ASSET_KEYS},
    )

    data_contract = _read_json(data_contract_path)
    dev_manifest_path = PROJECT_ROOT / paths["dev_base_manifest"]
    dev_manifest = _read_json(dev_manifest_path)
    roles = build_data_roles(
        data_contract,
        available_references=_available_dev_references(dev_manifest),
    )
    artifact_root = PROJECT_ROOT / paths["artifact_root"]
    _atomic_json(
        artifact_root / "DATA_ROLES.json",
        {
            "schema_version": "crosswindow-data-roles-v1",
            "source": paths["data_contract"],
            "roles": roles,
            "dev_reference_count": len(roles["DEV-CAL"]) + len(roles["DEV-SEL"]),
            "dev_search_authorized": (
                len(roles["DEV-CAL"]) + len(roles["DEV-SEL"])
                >= config["data_split"]["minimum_dev_references_for_search"]
            ),
        },
    )
    source_paths = (
        "scripts/task_memory_output.py",
        "models/task_memory_routing.py",
        "scripts/run_task_memory_controls.py",
        "scripts/rescene_task_postprocess.py",
        "scripts/analyze_persist4d_allt.py",
        "scripts/system_comparison_metrics.py",
        paths["data_contract"],
        paths["dev_base_manifest"],
        paths["dev_supplement_manifest"],
        paths["pb_base_manifest"],
        paths["pb_supplement_manifest"],
        paths["fh_native_manifest"],
    )
    _atomic_json(
        artifact_root / "SOURCE_MANIFEST.json",
        {
            "schema_version": "crosswindow-source-manifest-v1",
            "source_commit": expected_parent,
            "files": [
                {
                    "path": relative,
                    "bytes": (PROJECT_ROOT / relative).stat().st_size,
                    "sha256": _sha256(PROJECT_ROOT / relative),
                }
                for relative in source_paths
            ],
            "assets": _tracked_asset_status(values, sources),
        },
    )
    state = new_run_state(
        parent=expected_parent,
        instruction_sha=instruction_sha,
        config_sha=_sha256(config_path),
        data_sha=_sha256(data_contract_path),
    )
    state["external_assets_file"] = "external:assets.local.json"
    atomic_write_run_state(artifact_root / "RUN_STATE.json", state)
    print(
        json.dumps(
            {
                "status": "PASS",
                "stage": "BOOTSTRAP",
                "dev_reference_count": len(roles["DEV-CAL"]) + len(roles["DEV-SEL"]),
                "asset_status": _tracked_asset_status(values, sources),
                "next_command": state["next_command"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    bootstrap = subparsers.add_parser("bootstrap")
    bootstrap.add_argument("--config", default=str(DEFAULT_CONFIG))
    bootstrap.add_argument("--external-root")
    bootstrap.add_argument("--assets-from")
    bootstrap.add_argument("--resume", action="store_true")
    for key in ASSET_KEYS:
        if key != "external_run_root":
            bootstrap.add_argument(f"--{key.replace('_', '-')}", dest=key)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "bootstrap":
        return _bootstrap(args)
    raise CampaignError(f"unsupported campaign command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CampaignError",
    "atomic_write_run_state",
    "load_campaign_config",
    "load_resume_state",
    "main",
    "new_run_state",
]
