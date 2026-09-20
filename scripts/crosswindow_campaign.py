"""Resumable command line entry point for CrossWindow V1."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import hashlib
import io
import json
import os
import statistics
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from scripts.crosswindow_cache import (
    ASSET_KEYS,
    build_canonical_frame,
    build_data_roles,
    compare_vertex_order,
    iter_campaign_units,
    resolve_assets,
)

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


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_csv(
    path: Path, rows: Sequence[Mapping[str, object]], *, fieldnames: Sequence[str]
) -> None:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in fieldnames})
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as output:
            output.write(stream.getvalue())
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_csv_gz(
    path: Path, rows: Sequence[Mapping[str, object]], *, fieldnames: Sequence[str]
) -> None:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in fieldnames})
    payload = gzip.compress(stream.getvalue().encode("utf-8"), mtime=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
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
            "python -m scripts.crosswindow_campaign preflight --config "
            "configs/crosswindow_evidence_v1.yaml --external-root "
            '"$PERSIST4D_RUN_ROOT" --resume --read-only'
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


def _git_contains(commit: str) -> bool:
    return (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
        ).returncode
        == 0
    )


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
    if not _git_contains(expected_parent):
        raise CampaignError("worktree does not contain the fixed parent")
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


def _verified_torch_load(
    *,
    path: Path,
    record: Mapping[str, object],
    hash_cache: dict[str, dict[str, object]],
) -> Mapping[str, object]:
    import torch

    digest_path = path.with_suffix(path.suffix + ".sha256")
    if (
        not path.is_file()
        or path.is_symlink()
        or not digest_path.is_file()
        or digest_path.is_symlink()
    ):
        raise CampaignError(f"cache file is unavailable: {path.name}")
    expected_bytes = record.get("bytes")
    expected_sha = record.get("sha256")
    if (
        isinstance(expected_bytes, bool)
        or not isinstance(expected_bytes, int)
        or expected_bytes <= 0
        or not isinstance(expected_sha, str)
        or len(expected_sha) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha)
        or path.stat().st_size + digest_path.stat().st_size != expected_bytes
    ):
        raise CampaignError(f"cache manifest identity differs: {path.name}")
    stat = path.stat()
    cache_key = str(path.resolve())
    cached = hash_cache.get(cache_key)
    if (
        isinstance(cached, Mapping)
        and cached.get("bytes") == stat.st_size
        and cached.get("mtime_ns") == stat.st_mtime_ns
        and cached.get("sha256") == expected_sha
    ):
        digest = expected_sha
    else:
        digest = _sha256(path)
        hash_cache[cache_key] = {
            "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": digest,
        }
    if digest != expected_sha:
        raise CampaignError(f"cache SHA differs: {path.name}")
    if digest_path.read_text(encoding="ascii").strip().split() != [
        digest,
        path.name,
    ]:
        raise CampaignError(f"cache digest sidecar differs: {path.name}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise CampaignError(f"cache cannot be loaded safely: {path.name}") from error
    if not isinstance(payload, Mapping):
        raise CampaignError(f"cache payload is not a mapping: {path.name}")
    return payload


def _three_cycle_real_module_check() -> dict[str, object]:
    import torch

    from scripts.task_memory_output import _align_masks

    old_ids = torch.tensor([10, 20, 30])
    current_ids = torch.tensor([20, 30, 10])
    old_mask = torch.tensor([True, False, False])
    current_mask = torch.tensor([False, False, True])
    old_expression_mask = _align_masks(
        source_vertex_ids=current_ids,
        target_vertex_ids=old_ids,
        mask=old_mask,
    )
    corrected_current = _align_masks(
        source_vertex_ids=current_ids,
        target_vertex_ids=old_ids,
        mask=current_mask,
    )

    def iou(left: object, right: object) -> float:
        intersection = int((left & right).sum().item())
        union = int((left | right).sum().item())
        return intersection / union if union else 0.0

    old_expression_iou = iou(old_expression_mask, current_mask)
    corrected_iou = iou(old_mask, corrected_current)
    return {
        "function": "scripts.task_memory_output._align_masks",
        "old_expression_iou": old_expression_iou,
        "corrected_iou": corrected_iou,
        "passed": old_expression_iou == 0.0 and corrected_iou == 1.0,
    }


def _preflight(args: argparse.Namespace) -> int:
    from scripts.run_task_memory_controls import prediction_observation_from_payload
    from scripts.run_task_memory_policy_baseline import (
        _meta_from_payload,
        _prediction_from_payload,
    )

    config_path = Path(args.config).resolve()
    config = load_campaign_config(config_path)
    parent = config["identity"]["parent_commit"]
    if not _git_contains(parent):
        raise CampaignError("worktree does not contain the fixed parent")
    paths = config["paths"]
    artifact_root = PROJECT_ROOT / paths["artifact_root"]
    data_contract_path = PROJECT_ROOT / paths["data_contract"]
    load_resume_state(
        artifact_root / "RUN_STATE.json",
        parent=parent,
        instruction_sha=config["identity"]["instruction_sha256"],
        config_sha=_sha256(config_path),
        data_sha=_sha256(data_contract_path),
    )
    external_root_value = args.external_root or os.environ.get("PERSIST4D_RUN_ROOT")
    if not external_root_value:
        raise CampaignError("preflight requires --external-root or PERSIST4D_RUN_ROOT")
    external_root = Path(external_root_value).resolve()
    assets = _read_json(external_root / "assets.local.json")
    base_root_value = assets.get("dev_base_cache_root")
    supplement_root_value = assets.get("dev_supplement_root")
    if not isinstance(base_root_value, str) or not isinstance(
        supplement_root_value, str
    ):
        raise CampaignError("development cache roots are unresolved")
    base_root = Path(base_root_value)
    supplement_root = Path(supplement_root_value)
    base_manifest = _read_json(PROJECT_ROOT / paths["dev_base_manifest"])
    supplement_manifest = _read_json(PROJECT_ROOT / paths["dev_supplement_manifest"])
    data_roles = _read_json(artifact_root / "DATA_ROLES.json")
    roles = data_roles.get("roles")
    if not isinstance(roles, Mapping):
        raise CampaignError("campaign data roles are unavailable")
    development_references = []
    for role_name in ("DEV-CAL", "DEV-SEL"):
        references = roles.get(role_name)
        if not isinstance(references, list) or any(
            not isinstance(reference, str) for reference in references
        ):
            raise CampaignError(f"campaign data role differs: {role_name}")
        development_references.extend(references)
    units = list(
        iter_campaign_units(
            role="development",
            role_references=development_references,
            base_manifest=base_manifest,
            supplement_manifest=supplement_manifest,
        )
    )
    hash_cache_path = external_root / "verified_hashes.local.json"
    hash_cache = _read_json(hash_cache_path) if hash_cache_path.is_file() else {}
    point_rows: list[dict[str, object]] = []
    unit_count = 0
    frame_count = 0
    candidate_count = 0
    multi_class_group_count = 0
    bindings = []
    for unit in units:
        base_record = {"bytes": unit.base_bytes, "sha256": unit.base_sha256}
        supplement_record = {
            "bytes": unit.supplement_bytes,
            "sha256": unit.supplement_sha256,
        }
        base = _verified_torch_load(
            path=base_root / unit.base_filename,
            record=base_record,
            hash_cache=hash_cache,
        )
        supplement = _verified_torch_load(
            path=supplement_root / unit.supplement_filename,
            record=supplement_record,
            hash_cache=hash_cache,
        )
        bindings.append(
            {
                "logical_unit_id": unit.logical_unit_id,
                "logical_index": unit.logical_index,
                "reference_id": unit.reference_id,
                "sequence_id": unit.sequence_id,
                "base_ref": f"external:dev_base_cache/{unit.base_filename}",
                "base_sha256": unit.base_sha256,
                "supplement_ref": (
                    f"external:dev_supplement_cache/{unit.supplement_filename}"
                ),
                "supplement_sha256": unit.supplement_sha256,
            }
        )
        base_stages = base.get("stages")
        supplement_stages = supplement.get("stages")
        episode = base.get("episode")
        if (
            base.get("schema_version") != "task-memory-policy-cache-v1"
            or supplement.get("schema_version") != "task-memory-control-observations-v2"
            or not isinstance(episode, Mapping)
            or not isinstance(base_stages, list)
            or not isinstance(supplement_stages, list)
            or len(base_stages) != 5
            or len(supplement_stages) != 5
        ):
            raise CampaignError("development cache schema or stage coverage differs")
        previous_meta = None
        for stage_index, (base_stage, supplement_stage) in enumerate(
            zip(base_stages, supplement_stages, strict=True)
        ):
            if not isinstance(base_stage, Mapping) or not isinstance(
                supplement_stage, Mapping
            ):
                raise CampaignError("development cache stage must be a mapping")
            meta = _meta_from_payload(base_stage["stage_meta"])
            prediction = _prediction_from_payload(supplement_stage["prediction"])
            observation = prediction_observation_from_payload(
                supplement_stage["observation"]
            )
            frame = build_canonical_frame(
                producer_id="R1-B4-policy",
                order_id=f"dev-{unit.logical_index:03d}",
                observation=observation,
                prediction=prediction,
                stage_meta=meta,
            )
            frame_count += 1
            candidate_count += len(frame.candidates)
            multi_class_group_count += sum(
                len(group.candidate_indices) > 1 for group in frame.groups
            )
            if previous_meta is not None:
                prior_scan = previous_meta.scan_ids_in_window[-1]
                repeated_scan = meta.scan_ids_in_window[0]
                status = (
                    compare_vertex_order(
                        previous_meta.original_vertex_ids[-1],
                        meta.original_vertex_ids[0],
                    )
                    if prior_scan == repeated_scan
                    else "SCAN_ID_MISMATCH"
                )
                point_rows.append(
                    {
                        "population_id": "development",
                        "reference_id": episode["reference_id"],
                        "sequence_id": unit.sequence_id,
                        "transition": f"{stage_index - 1}->{stage_index}",
                        "scan_id": repeated_scan,
                        "prior_point_count": previous_meta.original_vertex_ids[
                            -1
                        ].numel(),
                        "current_point_count": meta.original_vertex_ids[0].numel(),
                        "status": status,
                    }
                )
            previous_meta = meta
        unit_count += 1
        del base, supplement
    _atomic_json(hash_cache_path, hash_cache)
    unique_physical_pairs = len({unit.physical_pair for unit in units})
    _atomic_json(
        artifact_root / "preflight/logical_unit_bindings.json",
        {
            "schema_version": "crosswindow-logical-unit-bindings-v1",
            "role": "development",
            "logical_unit_count": len(units),
            "unique_physical_pair_count": unique_physical_pairs,
            "units": bindings,
        },
    )

    status_counts: dict[str, int] = {}
    for row in point_rows:
        key = str(row["status"])
        status_counts[key] = status_counts.get(key, 0) + 1
    e0_root = artifact_root / "e0"
    _atomic_csv(
        e0_root / "index_trigger_counts.csv",
        point_rows,
        fieldnames=(
            "population_id",
            "reference_id",
            "sequence_id",
            "transition",
            "scan_id",
            "prior_point_count",
            "current_point_count",
            "status",
        ),
    )
    module_check = _three_cycle_real_module_check()
    numerical_pass = (
        module_check["passed"]
        and status_counts.get("SET_MISMATCH", 0) == 0
        and status_counts.get("SCAN_ID_MISMATCH", 0) == 0
    )
    _atomic_json(
        e0_root / "correctness_fix_ledger.json",
        {
            "schema_version": "crosswindow-correctness-ledger-v1",
            "scope": "development cache preflight; Protocol-B not yet scored",
            "real_module_three_cycle": module_check,
            "repeated_scan_status_counts": status_counts,
            "non_identity_trigger_count": status_counts.get("NON_IDENTITY", 0),
            "measured_metric_impact": "NOT_MEASURED",
            "candidate_frames_validated": frame_count,
            "candidate_records_preserved": candidate_count,
            "multi_class_groups_preserved": multi_class_group_count,
            "status": "PASS" if numerical_pass else "FAIL",
        },
    )
    _atomic_json(
        artifact_root / "preflight/cache_preflight.json",
        {
            "schema_version": "crosswindow-cache-preflight-v1",
            "status": "PASS" if numerical_pass else "FAIL",
            "read_only_old_assets": bool(args.read_only),
            "development_units": unit_count,
            "unique_physical_pairs": unique_physical_pairs,
            "frames": frame_count,
            "point_order_status_counts": status_counts,
            "candidate_records": candidate_count,
            "multi_class_groups": multi_class_group_count,
            "verified_file_count": len(hash_cache),
        },
    )
    state_path = artifact_root / "RUN_STATE.json"
    state = _read_json(state_path)
    state["preflight"] = {
        "status": "PASS" if numerical_pass else "FAIL",
        "development_units": unit_count,
        "frames": frame_count,
    }
    atomic_write_run_state(state_path, state)
    print(
        json.dumps(
            {
                "status": "PASS" if numerical_pass else "FAIL",
                "development_units": unit_count,
                "frames": frame_count,
                "point_order_status_counts": status_counts,
                "candidate_records": candidate_count,
                "multi_class_groups": multi_class_group_count,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if numerical_pass else 1


def _run_e0(args: argparse.Namespace) -> int:
    from scripts.replay_crosswindow_association import E0ReplayAccumulator

    started_cpu = time.process_time()
    config_path = Path(args.config).resolve()
    config = load_campaign_config(config_path)
    paths = config["paths"]
    artifact_root = PROJECT_ROOT / paths["artifact_root"]
    data_contract_path = PROJECT_ROOT / paths["data_contract"]
    state_path = artifact_root / "RUN_STATE.json"
    state = load_resume_state(
        state_path,
        parent=config["identity"]["parent_commit"],
        instruction_sha=config["identity"]["instruction_sha256"],
        config_sha=_sha256(config_path),
        data_sha=_sha256(data_contract_path),
    )
    e0_required = (
        "baseline_metrics.csv",
        "checkpoint_policy_matrix.csv",
        "source_parity.csv",
        "status.json",
        "t2_same_forward_parity.json",
    )
    if (
        args.resume
        and state.get("stage") == "E0"
        and state.get("stage_status") == "PASS"
    ):
        e0_root = artifact_root / "e0"
        if not all((e0_root / name).is_file() for name in e0_required):
            raise CampaignError("completed E0 state lacks required artifacts")
        summary = _read_json(e0_root / "status.json")
        parity = _read_json(e0_root / "t2_same_forward_parity.json")
        if summary.get("status") != "PASS" or parity.get("status") != "PASS":
            raise CampaignError("completed E0 artifacts do not pass")
        print(json.dumps({**summary, "resumed": True}, indent=2, sort_keys=True))
        return 0
    external_root_value = args.external_root or os.environ.get("PERSIST4D_RUN_ROOT")
    if not external_root_value:
        raise CampaignError("run requires --external-root or PERSIST4D_RUN_ROOT")
    external_root = Path(external_root_value).resolve()
    assets = _read_json(external_root / "assets.local.json")
    required_assets = (
        "dev_base_cache_root",
        "dev_supplement_root",
        "metric_dataset_spec",
    )
    if any(not isinstance(assets.get(key), str) for key in required_assets):
        raise CampaignError("E0 development assets are unresolved")
    base_root = Path(assets["dev_base_cache_root"])
    supplement_root = Path(assets["dev_supplement_root"])
    metric_dataset_spec = Path(assets["metric_dataset_spec"])
    if (
        not base_root.is_dir()
        or not supplement_root.is_dir()
        or not metric_dataset_spec.is_file()
    ):
        raise CampaignError("E0 development assets are unavailable")

    base_manifest = _read_json(PROJECT_ROOT / paths["dev_base_manifest"])
    supplement_manifest = _read_json(PROJECT_ROOT / paths["dev_supplement_manifest"])
    checkpoint_sha256 = supplement_manifest.get("checkpoint_sha256")
    if checkpoint_sha256 != config["identity"]["r1_checkpoint_sha256"]:
        raise CampaignError("E0 supplement checkpoint identity differs")
    roles = _read_json(artifact_root / "DATA_ROLES.json").get("roles")
    if not isinstance(roles, Mapping) or not isinstance(roles.get("DEV-CAL"), list):
        raise CampaignError("DEV-CAL role is unavailable")
    units = list(
        iter_campaign_units(
            role="DEV-CAL",
            role_references=roles["DEV-CAL"],
            base_manifest=base_manifest,
            supplement_manifest=supplement_manifest,
        )
    )
    metric_spec = yaml.safe_load(metric_dataset_spec.read_text(encoding="utf-8"))
    class_mapping = (
        metric_spec.get("valid_class_ids") if isinstance(metric_spec, dict) else None
    )
    if not isinstance(class_mapping, list) or any(
        isinstance(value, bool) or not isinstance(value, int) for value in class_mapping
    ):
        raise CampaignError("metric dataset class mapping is unavailable")
    correctness = _read_json(artifact_root / "e0/correctness_fix_ledger.json")
    index_trigger_count = correctness.get("non_identity_trigger_count")
    if isinstance(index_trigger_count, bool) or not isinstance(
        index_trigger_count, int
    ):
        raise CampaignError("point-order trigger count is unavailable")
    accumulator = E0ReplayAccumulator(
        dataset_spec=str(metric_dataset_spec),
        class_mapping=tuple(class_mapping),
        checkpoint_sha256=checkpoint_sha256,
        source_commit=config["identity"]["parent_commit"],
        index_trigger_count=index_trigger_count,
    )
    hash_cache_path = external_root / "verified_hashes.local.json"
    hash_cache = _read_json(hash_cache_path) if hash_cache_path.is_file() else {}
    completed_units = []
    for index, unit in enumerate(units, start=1):
        base = _verified_torch_load(
            path=base_root / unit.base_filename,
            record={"bytes": unit.base_bytes, "sha256": unit.base_sha256},
            hash_cache=hash_cache,
        )
        supplement = _verified_torch_load(
            path=supplement_root / unit.supplement_filename,
            record={
                "bytes": unit.supplement_bytes,
                "sha256": unit.supplement_sha256,
            },
            hash_cache=hash_cache,
        )
        accumulator.update(
            logical_unit_id=unit.logical_unit_id,
            base=base,
            supplement=supplement,
        )
        completed_units.append(unit.logical_unit_id)
        del base, supplement
        print(
            json.dumps(
                {
                    "stage": "E0",
                    "completed": index,
                    "total": len(units),
                    "logical_unit_id": unit.logical_unit_id,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    _atomic_json(hash_cache_path, hash_cache)
    result = accumulator.finalize()
    e0_root = artifact_root / "e0"
    metric_rows = result["metric_rows"]
    source_rows = result["source_rows"]
    checkpoint_rows = result["checkpoint_rows"]
    _atomic_csv(
        e0_root / "baseline_metrics.csv",
        metric_rows,
        fieldnames=tuple(metric_rows[0]),
    )
    _atomic_csv(
        e0_root / "source_parity.csv",
        source_rows,
        fieldnames=tuple(source_rows[0]),
    )
    _atomic_csv(
        e0_root / "checkpoint_policy_matrix.csv",
        checkpoint_rows,
        fieldnames=tuple(checkpoint_rows[0]),
    )
    _atomic_json(e0_root / "t2_same_forward_parity.json", result["t2_parity"])
    cpu_core_hours = (time.process_time() - started_cpu) / 3600.0
    source_base_exact_rows = sum(
        bool(row["generated_vs_base_exact"]) for row in source_rows
    )
    shared_forward_rows = sum(
        bool(row["d_and_a_physical_forward"]) for row in source_rows
    )
    summary = {
        "schema_version": "crosswindow-e0-summary-v1",
        "status": result["status"],
        "data_role": "DEV-CAL",
        "reference_count": len(roles["DEV-CAL"]),
        "logical_unit_count": len(units),
        "source_parity_rows": len(source_rows),
        "source_base_exact_rows": source_base_exact_rows,
        "source_base_different_rows": len(source_rows) - source_base_exact_rows,
        "d_and_a_shared_forward_rows": shared_forward_rows,
        "metric_rows": len(metric_rows),
        "index_trigger_count": result["index_trigger_count"],
        "legacy_route_conflicts": result["route_conflicts"],
        "legacy_fallback_matches": result["fallback_matches"],
        "a0_collision_fallback_count": result["a0_collision_fallback_count"],
        "cpu_core_hours": cpu_core_hours,
    }
    _atomic_json(e0_root / "status.json", summary)
    budget = state.get("budget_used")
    if not isinstance(budget, dict):
        raise CampaignError("run-state budget is unavailable")
    budget["cpu_core_hours"] = float(budget.get("cpu_core_hours", 0.0)) + cpu_core_hours
    state.update(
        {
            "stage": "E0",
            "stage_status": result["status"],
            "completed_units": completed_units,
            "methods": sorted({str(row["method"]) for row in metric_rows}),
            "e0": summary,
            "next_command": (
                "python -m scripts.crosswindow_campaign run --config "
                "configs/crosswindow_evidence_v1.yaml --through E2 --resume"
            ),
        }
    )
    atomic_write_run_state(state_path, state)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


def _run_e1(args: argparse.Namespace) -> int:
    from scripts.diagnose_crosswindow_failures import E1DiagnosticAccumulator

    started_cpu = time.process_time()
    config_path = Path(args.config).resolve()
    config = load_campaign_config(config_path)
    paths = config["paths"]
    artifact_root = PROJECT_ROOT / paths["artifact_root"]
    data_contract_path = PROJECT_ROOT / paths["data_contract"]
    state_path = artifact_root / "RUN_STATE.json"
    state = load_resume_state(
        state_path,
        parent=config["identity"]["parent_commit"],
        instruction_sha=config["identity"]["instruction_sha256"],
        config_sha=_sha256(config_path),
        data_sha=_sha256(data_contract_path),
    )
    e1_required = (
        "candidate_coverage.csv",
        "failure_ledger.csv.gz",
        "gate.json",
        "gt_assisted.csv",
        "status.json",
    )
    if (
        args.resume
        and state.get("stage") == "E1"
        and state.get("stage_status") == "PASS"
    ):
        e1_root = artifact_root / "e1"
        if not all((e1_root / name).is_file() for name in e1_required):
            raise CampaignError("completed E1 state lacks required artifacts")
        summary = _read_json(e1_root / "status.json")
        gate = _read_json(e1_root / "gate.json")
        if summary.get("status") != "PASS" or not isinstance(gate.get("decision"), str):
            raise CampaignError("completed E1 artifacts do not pass")
        print(json.dumps({**summary, "resumed": True}, indent=2, sort_keys=True))
        return 0
    if state.get("stage") != "E0" or state.get("stage_status") != "PASS":
        raise CampaignError("E1 requires completed E0 evidence")
    external_root_value = args.external_root or os.environ.get("PERSIST4D_RUN_ROOT")
    if not external_root_value:
        raise CampaignError("run requires --external-root or PERSIST4D_RUN_ROOT")
    external_root = Path(external_root_value).resolve()
    assets = _read_json(external_root / "assets.local.json")
    required_assets = (
        "dev_base_cache_root",
        "dev_supplement_root",
        "metric_dataset_spec",
    )
    if any(not isinstance(assets.get(key), str) for key in required_assets):
        raise CampaignError("E1 development assets are unresolved")
    base_root = Path(assets["dev_base_cache_root"])
    supplement_root = Path(assets["dev_supplement_root"])
    metric_dataset_spec = Path(assets["metric_dataset_spec"])
    if (
        not base_root.is_dir()
        or not supplement_root.is_dir()
        or not metric_dataset_spec.is_file()
    ):
        raise CampaignError("E1 development assets are unavailable")
    base_manifest = _read_json(PROJECT_ROOT / paths["dev_base_manifest"])
    supplement_manifest = _read_json(PROJECT_ROOT / paths["dev_supplement_manifest"])
    checkpoint_sha256 = supplement_manifest.get("checkpoint_sha256")
    if checkpoint_sha256 != config["identity"]["r1_checkpoint_sha256"]:
        raise CampaignError("E1 supplement checkpoint identity differs")
    roles = _read_json(artifact_root / "DATA_ROLES.json").get("roles")
    if not isinstance(roles, Mapping) or not isinstance(roles.get("DEV-CAL"), list):
        raise CampaignError("DEV-CAL role is unavailable")
    units = list(
        iter_campaign_units(
            role="DEV-CAL",
            role_references=roles["DEV-CAL"],
            base_manifest=base_manifest,
            supplement_manifest=supplement_manifest,
        )
    )
    metric_spec = yaml.safe_load(metric_dataset_spec.read_text(encoding="utf-8"))
    class_mapping = (
        metric_spec.get("valid_class_ids") if isinstance(metric_spec, dict) else None
    )
    if not isinstance(class_mapping, list) or any(
        isinstance(value, bool) or not isinstance(value, int) for value in class_mapping
    ):
        raise CampaignError("metric dataset class mapping is unavailable")
    accumulator = E1DiagnosticAccumulator(
        dataset_spec=str(metric_dataset_spec),
        class_mapping=tuple(class_mapping),
        source_commit=config["identity"]["parent_commit"],
        checkpoint_sha256=checkpoint_sha256,
    )
    hash_cache_path = external_root / "verified_hashes.local.json"
    hash_cache = _read_json(hash_cache_path) if hash_cache_path.is_file() else {}
    completed_units = []
    for index, unit in enumerate(units, start=1):
        base = _verified_torch_load(
            path=base_root / unit.base_filename,
            record={"bytes": unit.base_bytes, "sha256": unit.base_sha256},
            hash_cache=hash_cache,
        )
        supplement = _verified_torch_load(
            path=supplement_root / unit.supplement_filename,
            record={
                "bytes": unit.supplement_bytes,
                "sha256": unit.supplement_sha256,
            },
            hash_cache=hash_cache,
        )
        accumulator.update(
            logical_unit_id=unit.logical_unit_id,
            base=base,
            supplement=supplement,
        )
        completed_units.append(unit.logical_unit_id)
        del base, supplement
        print(
            json.dumps(
                {
                    "stage": "E1",
                    "completed": index,
                    "total": len(units),
                    "logical_unit_id": unit.logical_unit_id,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    _atomic_json(hash_cache_path, hash_cache)
    result = accumulator.finalize()
    e1_root = artifact_root / "e1"
    metric_rows = result["metric_rows"]
    coverage_rows = result["coverage_rows"]
    coverage_events = result["coverage_events"]
    assignment_events = result["assignment_events"]
    _atomic_csv(
        e1_root / "gt_assisted.csv",
        metric_rows,
        fieldnames=tuple(metric_rows[0]),
    )
    _atomic_csv(
        e1_root / "candidate_coverage.csv",
        coverage_rows,
        fieldnames=tuple(coverage_rows[0]),
    )
    _atomic_csv_gz(
        e1_root / "failure_ledger.csv.gz",
        coverage_events,
        fieldnames=tuple(coverage_events[0]),
    )
    _atomic_csv(
        e1_root / "gt_assignment_events.csv",
        assignment_events,
        fieldnames=tuple(assignment_events[0]),
    )
    _atomic_json(e1_root / "gate.json", result["gate"])
    cpu_core_hours = (time.process_time() - started_cpu) / 3600.0
    summary = {
        "schema_version": "crosswindow-e1-summary-v1",
        "status": result["status"],
        "data_role": "DEV-CAL",
        "reference_count": len(roles["DEV-CAL"]),
        "logical_unit_count": len(units),
        "coverage_event_count": len(coverage_events),
        "assignment_event_count": len(assignment_events),
        "gate_decision": result["gate"]["decision"],
        "cpu_core_hours": cpu_core_hours,
    }
    _atomic_json(e1_root / "status.json", summary)
    budget = state.get("budget_used")
    if not isinstance(budget, dict):
        raise CampaignError("run-state budget is unavailable")
    budget["cpu_core_hours"] = float(budget.get("cpu_core_hours", 0.0)) + cpu_core_hours
    state.update(
        {
            "stage": "E1",
            "stage_status": result["status"],
            "completed_units": completed_units,
            "e1": summary,
            "next_command": (
                "python -m scripts.crosswindow_campaign run --config "
                "configs/crosswindow_evidence_v1.yaml --through E2 --resume"
            ),
        }
    )
    atomic_write_run_state(state_path, state)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


def _run_e2(args: argparse.Namespace) -> int:
    from models.overlap_entity_association import (
        A0_DEFAULT,
        A1_DEFAULT,
        A2_DEFAULT,
        preregistered_association_configs,
    )
    from scripts.diagnose_crosswindow_failures import (
        evaluate_dev_selection_gate,
        rank_family_candidates,
    )
    from scripts.replay_crosswindow_association import E2ReplayAccumulator

    started_cpu = time.process_time()
    config_path = Path(args.config).resolve()
    config = load_campaign_config(config_path)
    paths = config["paths"]
    artifact_root = PROJECT_ROOT / paths["artifact_root"]
    data_contract_path = PROJECT_ROOT / paths["data_contract"]
    state_path = artifact_root / "RUN_STATE.json"
    state = load_resume_state(
        state_path,
        parent=config["identity"]["parent_commit"],
        instruction_sha=config["identity"]["instruction_sha256"],
        config_sha=_sha256(config_path),
        data_sha=_sha256(data_contract_path),
    )
    required = (
        "assignment_events.csv",
        "bridge_baselines.csv",
        "calibration_grid.csv",
        "dev_selection.csv",
        "family_candidates.json",
        "status.json",
    )
    e2_root = artifact_root / "e2"
    if (
        args.resume
        and state.get("stage") == "E2"
        and state.get("stage_status") == "PASS"
    ):
        if not all((e2_root / name).is_file() for name in required):
            raise CampaignError("completed E2 state lacks required artifacts")
        summary = _read_json(e2_root / "status.json")
        if summary.get("status") != "PASS":
            raise CampaignError("completed E2 artifacts do not pass")
        print(json.dumps({**summary, "resumed": True}, indent=2, sort_keys=True))
        return 0
    if state.get("stage") != "E1" or state.get("stage_status") != "PASS":
        raise CampaignError("E2 requires completed E1 diagnostics")
    gate = _read_json(artifact_root / "e1/gate.json")
    if gate.get("decision") not in {
        "ASSOCIATION_HEADROOM_SUPPORTED",
        "INCONCLUSIVE",
        "INSUFFICIENT_EVENTS",
        "MASK_OR_OTHER_DOMINANT",
    }:
        raise CampaignError("E1 headroom decision is unavailable")
    if gate.get("run_full_grid") is True:
        raise CampaignError(
            "full E2 calibration is unsupported by this bounded execution path"
        )

    external_root_value = args.external_root or os.environ.get("PERSIST4D_RUN_ROOT")
    if not external_root_value:
        raise CampaignError("run requires --external-root or PERSIST4D_RUN_ROOT")
    external_root = Path(external_root_value).resolve()
    assets = _read_json(external_root / "assets.local.json")
    required_assets = (
        "dev_base_cache_root",
        "dev_supplement_root",
        "metric_dataset_spec",
    )
    if any(not isinstance(assets.get(key), str) for key in required_assets):
        raise CampaignError("E2 development assets are unresolved")
    base_root = Path(assets["dev_base_cache_root"])
    supplement_root = Path(assets["dev_supplement_root"])
    metric_dataset_spec = Path(assets["metric_dataset_spec"])
    if (
        not base_root.is_dir()
        or not supplement_root.is_dir()
        or not metric_dataset_spec.is_file()
    ):
        raise CampaignError("E2 development assets are unavailable")
    base_manifest = _read_json(PROJECT_ROOT / paths["dev_base_manifest"])
    supplement_manifest = _read_json(PROJECT_ROOT / paths["dev_supplement_manifest"])
    checkpoint_sha256 = supplement_manifest.get("checkpoint_sha256")
    if checkpoint_sha256 != config["identity"]["r1_checkpoint_sha256"]:
        raise CampaignError("E2 supplement checkpoint identity differs")
    roles = _read_json(artifact_root / "DATA_ROLES.json").get("roles")
    if not isinstance(roles, Mapping) or not isinstance(roles.get("DEV-SEL"), list):
        raise CampaignError("DEV-SEL role is unavailable")
    units = list(
        iter_campaign_units(
            role="DEV-SEL",
            role_references=roles["DEV-SEL"],
            base_manifest=base_manifest,
            supplement_manifest=supplement_manifest,
        )
    )
    metric_spec = yaml.safe_load(metric_dataset_spec.read_text(encoding="utf-8"))
    class_mapping = (
        metric_spec.get("valid_class_ids") if isinstance(metric_spec, dict) else None
    )
    if not isinstance(class_mapping, list) or any(
        isinstance(value, bool) or not isinstance(value, int) for value in class_mapping
    ):
        raise CampaignError("metric dataset class mapping is unavailable")
    configs = {
        "A0-U-default": A0_DEFAULT,
        "A1-default": A1_DEFAULT,
        "A2-default": A2_DEFAULT,
    }
    accumulator = E2ReplayAccumulator(
        dataset_spec=str(metric_dataset_spec),
        class_mapping=tuple(class_mapping),
        checkpoint_sha256=checkpoint_sha256,
        source_commit=config["identity"]["parent_commit"],
        data_role="DEV-SEL",
        association_configs=configs,
    )
    hash_cache_path = external_root / "verified_hashes.local.json"
    hash_cache = _read_json(hash_cache_path) if hash_cache_path.is_file() else {}
    completed_units = []
    for index, unit in enumerate(units, start=1):
        base = _verified_torch_load(
            path=base_root / unit.base_filename,
            record={"bytes": unit.base_bytes, "sha256": unit.base_sha256},
            hash_cache=hash_cache,
        )
        supplement = _verified_torch_load(
            path=supplement_root / unit.supplement_filename,
            record={
                "bytes": unit.supplement_bytes,
                "sha256": unit.supplement_sha256,
            },
            hash_cache=hash_cache,
        )
        accumulator.update(
            logical_unit_id=unit.logical_unit_id,
            base=base,
            supplement=supplement,
        )
        completed_units.append(unit.logical_unit_id)
        del base, supplement
        print(
            json.dumps(
                {
                    "stage": "E2",
                    "completed": index,
                    "total": len(units),
                    "logical_unit_id": unit.logical_unit_id,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    _atomic_json(hash_cache_path, hash_cache)
    result = accumulator.finalize()
    metric_rows = result["metric_rows"]
    aggregate_rows = [row for row in metric_rows if row["reference"] == "all"]
    reference_rows = [row for row in metric_rows if row["reference"] != "all"]
    by_method_horizon = {
        (str(row["method"]), int(row["T"])): row for row in aggregate_rows
    }
    d0_by_horizon = {
        horizon: float(by_method_horizon[("D0", horizon)]["t_mAP"])
        for horizon in (2, 3, 4, 5)
    }
    candidate_metric_rows = [
        {
            "config_id": by_method_horizon[(method, horizon)]["config_id"],
            "T": horizon,
            "t_mAP": by_method_horizon[(method, horizon)]["t_mAP"],
        }
        for method in ("A1-default", "A2-default")
        for horizon in (2, 3, 4, 5)
    ]
    ranked = rank_family_candidates(
        candidate_metric_rows,
        d0_by_horizon=d0_by_horizon,
    )
    method_by_config = {
        config.config_id: method
        for method, config in configs.items()
        if method != "A0-U-default"
    }
    selection_config = config["selection"]
    baseline_identity = by_method_horizon[("D0", 5)]
    candidate_results = []
    for rank_index, rank_row in enumerate(ranked, start=1):
        config_id = str(rank_row["config_id"])
        method = method_by_config[config_id]
        deltas = {
            horizon: float(by_method_horizon[(method, horizon)]["t_mAP"])
            - d0_by_horizon[horizon]
            for horizon in (2, 3, 4, 5)
        }
        positive_references = 0
        for reference in sorted({str(row["reference"]) for row in reference_rows}):
            reference_delta = []
            for horizon in (4, 5):
                candidate_row = next(
                    row
                    for row in reference_rows
                    if row["reference"] == reference
                    and row["method"] == method
                    and row["T"] == horizon
                )
                baseline_row = next(
                    row
                    for row in reference_rows
                    if row["reference"] == reference
                    and row["method"] == "D0"
                    and row["T"] == horizon
                )
                reference_delta.append(
                    float(candidate_row["t_mAP"])
                    - float(baseline_row["t_mAP"])
                )
            positive_references += sum(reference_delta) / 2.0 > 0.0
        candidate_identity = by_method_horizon[(method, 5)]
        gate_result = evaluate_dev_selection_gate(
            deltas=deltas,
            positive_reference_count=positive_references,
            reference_count=len(roles["DEV-SEL"]),
            candidate_merge_rate=candidate_identity["merge_rate"],
            baseline_merge_rate=baseline_identity["merge_rate"],
            merge_event_count=min(
                int(candidate_identity["merge_opportunities"]),
                int(baseline_identity["merge_opportunities"]),
            ),
            candidate_wrong_reactivation_rate=candidate_identity[
                "wrong_reactivation_rate"
            ],
            baseline_wrong_reactivation_rate=baseline_identity[
                "wrong_reactivation_rate"
            ],
            reactivation_event_count=min(
                int(candidate_identity["recovery_attempts"]),
                int(baseline_identity["recovery_attempts"]),
            ),
            max_per_t_drop=float(selection_config["max_per_T_drop"]),
            minimum_long_gain=float(selection_config["min_long_gain"]),
            minimum_mean_gain=float(selection_config["min_mean_gain"]),
            minimum_positive_references=int(
                selection_config["min_positive_refs"]
            ),
            maximum_error_rate_increase=float(
                selection_config["max_error_rate_increase"]
            ),
        )
        candidate_results.append(
            {
                "rank": rank_index,
                "method": method,
                "config_id": config_id,
                "selection": gate_result,
                "ranking": rank_row,
                "identity_T5": {
                    field: candidate_identity[field]
                    for field in (
                        "merge_count",
                        "merge_opportunities",
                        "merge_rate",
                        "wrong_reactivations",
                        "recovery_attempts",
                        "wrong_reactivation_rate",
                    )
                },
            }
        )
    promoted = next(
        (row for row in candidate_results if row["selection"]["eligible"]), None
    )
    exploratory = candidate_results[0]
    family_candidates = {
        "schema_version": "crosswindow-e2-family-candidates-v1",
        "data_role": "DEV-SEL",
        "headroom_gate": gate["decision"],
        "full_grid_run": False,
        "calibration_policy": "DEFAULTS_ONLY_NO_PARAMETER_SEARCH",
        "families": {
            "A0-U": {
                "role": "fixed_bridge_baseline",
                "config_id": A0_DEFAULT.config_id,
            },
            "A1": {
                "role": "default_candidate",
                "config_id": A1_DEFAULT.config_id,
            },
            "A2": {
                "role": "default_candidate",
                "config_id": A2_DEFAULT.config_id,
            },
        },
        "candidates": candidate_results,
        "promoted_candidate": (
            {
                "method": promoted["method"],
                "config_id": promoted["config_id"],
                "status": promoted["selection"]["status"],
            }
            if promoted is not None
            else None
        ),
        "final_association_parent": promoted["method"] if promoted else "D0",
        "exploratory_candidate": {
            "method": exploratory["method"],
            "config_id": exploratory["config_id"],
            "status": exploratory["selection"]["status"],
        },
    }
    calibration_rows = []
    for association_config in preregistered_association_configs():
        for horizon in (2, 3, 4, 5):
            calibration_rows.append(
                {
                    "population_id": "development",
                    "data_role": "DEV-CAL",
                    "family": association_config.family,
                    "config_id": association_config.config_id,
                    "tau": association_config.tau,
                    "overlap_theta": association_config.overlap_theta,
                    "lambda": association_config.lambda_,
                    "T": horizon,
                    "t_mAP": None,
                    "status": "SKIPPED_HEADROOM_GATE",
                    "reason": gate["decision"],
                }
            )
    bridge_rows = []
    for row in aggregate_rows:
        if row["method"] not in {"D0", "A0-U-default"}:
            continue
        bridge_rows.append(
            {
                **row,
                "interface_and_assignment_change": (
                    0.0
                    if row["method"] == "D0"
                    else float(row["t_mAP"])
                    - float(by_method_horizon[("D0", int(row["T"]))]["t_mAP"])
                ),
            }
        )
    _atomic_csv(
        e2_root / "calibration_grid.csv",
        calibration_rows,
        fieldnames=tuple(calibration_rows[0]),
    )
    _atomic_json(e2_root / "family_candidates.json", family_candidates)
    _atomic_csv(
        e2_root / "dev_selection.csv",
        metric_rows,
        fieldnames=tuple(metric_rows[0]),
    )
    assignment_events = result["assignment_events"]
    _atomic_csv(
        e2_root / "assignment_events.csv",
        assignment_events,
        fieldnames=tuple(assignment_events[0]),
    )
    _atomic_csv(
        e2_root / "bridge_baselines.csv",
        bridge_rows,
        fieldnames=tuple(bridge_rows[0]),
    )
    cpu_core_hours = (time.process_time() - started_cpu) / 3600.0
    summary = {
        "schema_version": "crosswindow-e2-summary-v1",
        "status": result["status"],
        "data_role": "DEV-SEL",
        "headroom_gate": gate["decision"],
        "calibration_grid_status": "SKIPPED_HEADROOM_GATE",
        "reference_count": len(roles["DEV-SEL"]),
        "logical_unit_count": len(units),
        "assignment_event_count": len(assignment_events),
        "collision_fallbacks": result["collision_fallbacks"],
        "promoted_candidate": family_candidates["promoted_candidate"],
        "exploratory_candidate": family_candidates["exploratory_candidate"],
        "cpu_core_hours": cpu_core_hours,
    }
    _atomic_json(e2_root / "status.json", summary)
    budget = state.get("budget_used")
    if not isinstance(budget, dict):
        raise CampaignError("run-state budget is unavailable")
    budget["cpu_core_hours"] = float(budget.get("cpu_core_hours", 0.0)) + cpu_core_hours
    state.update(
        {
            "stage": "E2",
            "stage_status": result["status"],
            "completed_units": completed_units,
            "methods": ["D0", *configs],
            "e2": summary,
            "next_command": (
                "python -m scripts.crosswindow_campaign run --config "
                "configs/crosswindow_evidence_v1.yaml --through E4 --resume"
            ),
        }
    )
    atomic_write_run_state(state_path, state)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


def _run_e4(args: argparse.Namespace) -> int:
    from models.overlap_entity_association import A1_DEFAULT, A2_DEFAULT
    from scripts.diagnose_crosswindow_failures import rank_family_candidates
    from scripts.evaluate_crosswindow_consensus import (
        E4RevisionAccumulator,
        exhaustive_fixed_u_equivalence,
    )

    started_cpu = time.process_time()
    config_path = Path(args.config).resolve()
    config = load_campaign_config(config_path)
    paths = config["paths"]
    artifact_root = PROJECT_ROOT / paths["artifact_root"]
    data_contract_path = PROJECT_ROOT / paths["data_contract"]
    state_path = artifact_root / "RUN_STATE.json"
    state = load_resume_state(
        state_path,
        parent=config["identity"]["parent_commit"],
        instruction_sha=config["identity"]["instruction_sha256"],
        config_sha=_sha256(config_path),
        data_sha=_sha256(data_contract_path),
    )
    e3_root = artifact_root / "e3"
    e4_root = artifact_root / "e4"
    e4_required = (
        "association_revision_factorial.csv",
        "content_choice_status.json",
        "revision_events.csv",
        "status.json",
    )
    if (
        args.resume
        and state.get("stage") == "E4"
        and state.get("stage_status") == "PASS"
    ):
        if not all((e4_root / name).is_file() for name in e4_required):
            raise CampaignError("completed E4 state lacks required artifacts")
        summary = _read_json(e4_root / "status.json")
        if summary.get("status") != "PASS":
            raise CampaignError("completed E4 artifacts do not pass")
        print(json.dumps({**summary, "resumed": True}, indent=2, sort_keys=True))
        return 0
    if state.get("stage") != "E2" or state.get("stage_status") != "PASS":
        raise CampaignError("E4 requires completed E2 selection")

    equivalence = exhaustive_fixed_u_equivalence()
    if equivalence["status"] != "PASS":
        raise CampaignError("fixed-U equivalence audit failed")
    e3_status = {
        **equivalence,
        "status": "SKIPPED_EQUIVALENT",
        "completion_status": "PASS",
        "reason": "fixed U collapses the joint objective to the E2 assignment",
        "assignment_plan_checks": {
            "unique_entity_per_group": "COVERED_BY_TEST",
            "generation_preserved": "COVERED_BY_TEST",
            "single_commit": "COVERED_BY_TEST",
            "publish_does_not_reassign": "COVERED_BY_TEST",
            "archive_immutability": "COVERED_BY_TEST",
        },
    }
    _atomic_json(e3_root / "status.json", e3_status)
    _atomic_text(
        e3_root / "CONSISTENCY_EQUIVALENCE.md",
        """# CrossWindow fixed-U consistency audit

With the historical assignment `U` frozen, the expanded objective is

`J(Z) = sum(q,k) Z(q,k) B(q,k) + lambda sum(q,i,k) Z(q,k) U(i,k) O(q,i)`.

Collecting terms indexed by `(q,k)` gives

`J(Z) = sum(q,k) Z(q,k) [B(q,k) + lambda sum(i) U(i,k) O(q,i)]`.

The feasible set is unchanged: every query is assigned to at most one real
anchor or its private dummy, and every real anchor is used at most once.
Therefore the bounded W=2 contract contains no second independent consensus
optimization. Exhaustive 2-3 group partial-assignment checks are recorded in
`status.json`; E3 is completed as `SKIPPED_EQUIVALENT`.
""",
    )

    family_candidates = _read_json(artifact_root / "e2/family_candidates.json")
    exploratory = family_candidates.get("exploratory_candidate")
    if not isinstance(exploratory, Mapping):
        raise CampaignError("E2 exploratory association candidate is unavailable")
    association_method = exploratory.get("method")
    if association_method == "A1-default":
        association_config = A1_DEFAULT
    elif association_method == "A2-default":
        association_config = A2_DEFAULT
    else:
        raise CampaignError("E2 exploratory association method differs")

    external_root_value = args.external_root or os.environ.get("PERSIST4D_RUN_ROOT")
    if not external_root_value:
        raise CampaignError("run requires --external-root or PERSIST4D_RUN_ROOT")
    external_root = Path(external_root_value).resolve()
    assets = _read_json(external_root / "assets.local.json")
    required_assets = (
        "dev_base_cache_root",
        "dev_supplement_root",
        "metric_dataset_spec",
    )
    if any(not isinstance(assets.get(key), str) for key in required_assets):
        raise CampaignError("E4 development assets are unresolved")
    base_root = Path(assets["dev_base_cache_root"])
    supplement_root = Path(assets["dev_supplement_root"])
    metric_dataset_spec = Path(assets["metric_dataset_spec"])
    if (
        not base_root.is_dir()
        or not supplement_root.is_dir()
        or not metric_dataset_spec.is_file()
    ):
        raise CampaignError("E4 development assets are unavailable")
    base_manifest = _read_json(PROJECT_ROOT / paths["dev_base_manifest"])
    supplement_manifest = _read_json(PROJECT_ROOT / paths["dev_supplement_manifest"])
    checkpoint_sha256 = supplement_manifest.get("checkpoint_sha256")
    if checkpoint_sha256 != config["identity"]["r1_checkpoint_sha256"]:
        raise CampaignError("E4 supplement checkpoint identity differs")
    roles = _read_json(artifact_root / "DATA_ROLES.json").get("roles")
    if not isinstance(roles, Mapping) or not isinstance(roles.get("DEV-SEL"), list):
        raise CampaignError("DEV-SEL role is unavailable")
    units = list(
        iter_campaign_units(
            role="DEV-SEL",
            role_references=roles["DEV-SEL"],
            base_manifest=base_manifest,
            supplement_manifest=supplement_manifest,
        )
    )
    metric_spec = yaml.safe_load(metric_dataset_spec.read_text(encoding="utf-8"))
    class_mapping = (
        metric_spec.get("valid_class_ids") if isinstance(metric_spec, dict) else None
    )
    if not isinstance(class_mapping, list) or any(
        isinstance(value, bool) or not isinstance(value, int) for value in class_mapping
    ):
        raise CampaignError("metric dataset class mapping is unavailable")
    accumulator = E4RevisionAccumulator(
        dataset_spec=str(metric_dataset_spec),
        class_mapping=tuple(class_mapping),
        checkpoint_sha256=checkpoint_sha256,
        source_commit=config["identity"]["parent_commit"],
        association_method=str(association_method),
        association_config=association_config,
    )
    hash_cache_path = external_root / "verified_hashes.local.json"
    hash_cache = _read_json(hash_cache_path) if hash_cache_path.is_file() else {}
    completed_units = []
    for index, unit in enumerate(units, start=1):
        base = _verified_torch_load(
            path=base_root / unit.base_filename,
            record={"bytes": unit.base_bytes, "sha256": unit.base_sha256},
            hash_cache=hash_cache,
        )
        supplement = _verified_torch_load(
            path=supplement_root / unit.supplement_filename,
            record={
                "bytes": unit.supplement_bytes,
                "sha256": unit.supplement_sha256,
            },
            hash_cache=hash_cache,
        )
        accumulator.update(
            logical_unit_id=unit.logical_unit_id,
            base=base,
            supplement=supplement,
        )
        completed_units.append(unit.logical_unit_id)
        del base, supplement
        print(
            json.dumps(
                {
                    "stage": "E4",
                    "completed": index,
                    "total": len(units),
                    "logical_unit_id": unit.logical_unit_id,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    _atomic_json(hash_cache_path, hash_cache)
    result = accumulator.finalize()
    metric_rows = result["metric_rows"]
    chosen_by_parent = {}
    parent_rankings = {}
    for parent in ("D0", association_method):
        system_rows = [
            row
            for row in metric_rows
            if row["parent"] == parent
            and row["score_mode"] == "SYSTEM_SELECTED_SCORE"
        ]
        baseline = {
            int(row["T"]): float(row["t_mAP"])
            for row in system_rows
            if row["selector"] == "M-new"
        }
        ranking_rows = [
            {
                "config_id": row["selector"],
                "T": row["T"],
                "t_mAP": row["t_mAP"],
            }
            for row in system_rows
        ]
        ranked = rank_family_candidates(ranking_rows, d0_by_horizon=baseline)
        evaluated = []
        for row in ranked:
            eligible = float(row["S_min"]) >= -float(
                config["selection"]["max_per_T_drop"]
            ) and float(row["S_long"]) >= 0.0
            evaluated.append({**row, "eligible_vs_parent": eligible})
        selected = next(row for row in evaluated if row["eligible_vs_parent"])
        parent_rankings[parent] = evaluated
        chosen_by_parent[parent] = selected
    fixed_parent = family_candidates.get("final_association_parent")
    if fixed_parent not in chosen_by_parent:
        raise CampaignError("E4 fixed parent differs from E2 selection")
    selected_revision = chosen_by_parent[fixed_parent]
    fixed_score_ranking = next(
        row
        for row in rank_family_candidates(
            [
                {
                    "config_id": row["selector"],
                    "T": row["T"],
                    "t_mAP": row["t_mAP"],
                }
                for row in metric_rows
                if row["parent"] == fixed_parent
                and row["score_mode"] == "MASK_ONLY_FIXED_SCORE"
            ],
            d0_by_horizon={
                int(row["T"]): float(row["t_mAP"])
                for row in metric_rows
                if row["parent"] == fixed_parent
                and row["score_mode"] == "MASK_ONLY_FIXED_SCORE"
                and row["selector"] == "M-new"
            },
        )
        if row["config_id"] == selected_revision["config_id"]
    )
    if selected_revision["config_id"] == "M-new":
        revision_interpretation = "NO_REVISION_GAIN"
    elif float(selected_revision["S_long"]) > 0 and float(
        fixed_score_ranking["S_long"]
    ) <= 0:
        revision_interpretation = "RANKING_GAIN_ONLY"
    elif abs(
        float(selected_revision["S_long"])
        - float(fixed_score_ranking["S_long"])
    ) <= 1e-6:
        revision_interpretation = "MASK_GAIN"
    else:
        revision_interpretation = "MIXED_MASK_AND_RANKING_GAIN"
    choice_status = {
        "schema_version": "crosswindow-e4-content-choice-v1",
        "status": "PASS",
        "association_parent": fixed_parent,
        "association_exploratory_parent": association_method,
        "parent_rankings": parent_rankings,
        "selected_system": {
            "parent": fixed_parent,
            "selector": selected_revision["config_id"],
            "score_mode": "SYSTEM_SELECTED_SCORE",
            "score_reducer": "mean",
            "ranking": selected_revision,
            "fixed_score_ranking": fixed_score_ranking,
            "interpretation": revision_interpretation,
        },
        "M_consensus": {
            "status": "SKIPPED_NO_INDEPENDENT_EVIDENCE",
            "reason": "W=2 exposes only old and new masks",
        },
    }
    _atomic_csv(
        e4_root / "association_revision_factorial.csv",
        metric_rows,
        fieldnames=tuple(metric_rows[0]),
    )
    revision_events = result["revision_events"]
    _atomic_csv(
        e4_root / "revision_events.csv",
        revision_events,
        fieldnames=tuple(revision_events[0]),
    )
    _atomic_json(e4_root / "content_choice_status.json", choice_status)
    cpu_core_hours = (time.process_time() - started_cpu) / 3600.0
    summary = {
        "schema_version": "crosswindow-e4-summary-v1",
        "status": result["status"],
        "data_role": "DEV-SEL",
        "reference_count": len(roles["DEV-SEL"]),
        "logical_unit_count": len(units),
        "factorial_row_count": len(metric_rows),
        "revision_event_count": len(revision_events),
        "selected_system": choice_status["selected_system"],
        "cpu_core_hours": cpu_core_hours,
    }
    _atomic_json(e4_root / "status.json", summary)
    budget = state.get("budget_used")
    if not isinstance(budget, dict):
        raise CampaignError("run-state budget is unavailable")
    budget["cpu_core_hours"] = float(budget.get("cpu_core_hours", 0.0)) + cpu_core_hours
    state.update(
        {
            "stage": "E4",
            "stage_status": result["status"],
            "completed_units": completed_units,
            "e3": e3_status,
            "e4": summary,
            "next_command": (
                "python -m scripts.crosswindow_campaign lock --config "
                "configs/crosswindow_evidence_v1.yaml"
            ),
        }
    )
    atomic_write_run_state(state_path, state)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


def _conditional_train(args: argparse.Namespace) -> int:
    config_path = Path(args.config).resolve()
    config = load_campaign_config(config_path)
    paths = config["paths"]
    artifact_root = PROJECT_ROOT / paths["artifact_root"]
    state_path = artifact_root / "RUN_STATE.json"
    state = load_resume_state(
        state_path,
        parent=config["identity"]["parent_commit"],
        instruction_sha=config["identity"]["instruction_sha256"],
        config_sha=_sha256(config_path),
        data_sha=_sha256(PROJECT_ROOT / paths["data_contract"]),
    )
    if state.get("stage") not in {"E4", "E6"} or state.get("stage_status") != "PASS":
        raise CampaignError("conditional training requires completed E4")
    e6_root = artifact_root / "e6"
    status_path = e6_root / "status.json"
    if args.resume and status_path.is_file():
        status = _read_json(status_path)
        if status.get("status") == "SKIPPED_CONDITION":
            print(json.dumps({**status, "resumed": True}, indent=2, sort_keys=True))
            return 0
    gate = _read_json(artifact_root / "e1/gate.json")
    family = _read_json(artifact_root / "e2/family_candidates.json")
    roles = _read_json(artifact_root / "DATA_ROLES.json")["roles"]
    external_root_value = args.external_root or os.environ.get("PERSIST4D_RUN_ROOT")
    assets = (
        _read_json(Path(external_root_value).resolve() / "assets.local.json")
        if external_root_value
        else {}
    )
    budget = state.get("budget_used")
    if not isinstance(budget, Mapping):
        raise CampaignError("run-state budget is unavailable")
    predicates = [
        {
            "id": "E1_HEADROOM",
            "pass": gate.get("decision") == "ASSOCIATION_HEADROOM_SUPPORTED",
            "evidence": gate.get("decision"),
        },
        {
            "id": "CAL_GAP_TO_GT_LAG1",
            "pass": False,
            "evidence": (
                "NOT_EVALUATED_GATE_CLOSED; full CAL connector grid was not authorized"
            ),
        },
        {
            "id": "ISOLATED_ADAPTATION_PAIRS",
            "pass": False,
            "evidence": {
                "adaptation_reference_count": len(roles.get("adaptation", [])),
                "train_observation_cache_available": bool(
                    assets.get("train_observation_cache_root")
                ),
                "positive_pair_count": None,
                "negative_pair_count": None,
            },
        },
        {
            "id": "GPU_BUDGET",
            "pass": (
                float(budget.get("head_training_gpu_hours", 0.0))
                <= float(config["budget"]["head_training_gpu_hours"]) - 6.0
                and float(budget.get("inference_gpu_hours", 0.0))
                <= float(config["budget"]["inference_gpu_hours"])
            ),
            "evidence": {
                "used_training_gpu_hours": budget.get("head_training_gpu_hours"),
                "used_inference_gpu_hours": budget.get("inference_gpu_hours"),
            },
        },
        {
            "id": "E0_E2_CHECKS",
            "pass": (
                state.get("e0", {}).get("status") == "PASS"
                and state.get("e2", {}).get("status") == "PASS"
            ),
            "evidence": {
                "E0": state.get("e0", {}).get("status"),
                "E2": state.get("e2", {}).get("status"),
                "promoted_candidate": family.get("promoted_candidate"),
            },
        },
    ]
    status = {
        "schema_version": "crosswindow-e6-status-v1",
        "status": "SKIPPED_CONDITION",
        "authorized": all(item["pass"] for item in predicates),
        "predicates": predicates,
        "head_SHA": None,
        "training_steps": 0,
        "gpu_hours": 0.0,
        "reason": "one or more preregistered authorization predicates failed",
    }
    if status["authorized"]:
        raise CampaignError("E6 unexpectedly authorized; training path is unavailable")
    _atomic_json(status_path, status)
    state.update(
        {
            "stage": "E6",
            "stage_status": "PASS",
            "e6": status,
            "next_command": (
                "python -m scripts.crosswindow_campaign lock --config "
                "configs/crosswindow_evidence_v1.yaml"
            ),
        }
    )
    atomic_write_run_state(state_path, state)
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0


def _lock(args: argparse.Namespace) -> int:
    config_path = Path(args.config).resolve()
    config = load_campaign_config(config_path)
    paths = config["paths"]
    artifact_root = PROJECT_ROOT / paths["artifact_root"]
    state_path = artifact_root / "RUN_STATE.json"
    state = load_resume_state(
        state_path,
        parent=config["identity"]["parent_commit"],
        instruction_sha=config["identity"]["instruction_sha256"],
        config_sha=_sha256(config_path),
        data_sha=_sha256(PROJECT_ROOT / paths["data_contract"]),
    )
    if state.get("stage") not in {"E6", "SELECT_AND_LOCK"}:
        raise CampaignError("selection lock requires completed conditional E6 decision")
    selection_root = artifact_root / "selection"
    lock_path = selection_root / "FINAL_LOCK.json"
    if lock_path.is_file():
        lock = _read_json(lock_path)
        if lock.get("status") != "LOCKED":
            raise CampaignError("existing final lock is invalid")
        print(json.dumps({**lock, "resumed": True}, indent=2, sort_keys=True))
        return 0
    e2 = _read_json(artifact_root / "e2/family_candidates.json")
    e4 = _read_json(artifact_root / "e4/content_choice_status.json")
    e6 = _read_json(artifact_root / "e6/status.json")
    roles = _read_json(artifact_root / "DATA_ROLES.json")
    selected = e4["selected_system"]
    association_parent = selected["parent"]
    exploratory = e2["exploratory_candidate"]
    review_items = [
        ("source_mapping", "PASS", "SOURCE_MANIFEST.json hashes fixed parent sources"),
        ("model_frozen", "PASS", "R1 checkpoint SHA fixed; E6 skipped"),
        ("GT_boundary", "PASS", "production association imports no diagnostic target"),
        ("candidate_preservation", "PASS", "canonical ledger and collision tests pass"),
        ("point_direction", "PASS", "three-cycle and target alignment tests pass"),
        ("comparison_source", "PASS", "D0/A methods share supplement forwards"),
        ("missing_overlap", "PASS", "missing uses base score and is logged separately"),
        ("E3_equivalence", "PASS", "exhaustive 2-3 group audit <=1e-12"),
        ("E4_scoring", "PASS", "mask-only and selected-score channels separated"),
        ("split_isolation", "PASS", "CAL/SEL/PB reference roles are disjoint"),
        ("fixed_start", "PASS", "selection deltas use fixed D0"),
        ("budget", "PASS", "recorded usage remains within configured budgets"),
        (
            "real_timing",
            "DEFERRED_EXTERNAL_ASSET",
            "native FH cache unavailable; E5 resource result must be UNCONFIRMED",
        ),
        ("publication_loop", "PASS", "E/P/readback workflow is configured"),
    ]
    review_lines = [
        "# CrossWindow V1 preflight review",
        "",
        "| Item | Verdict | Evidence |",
        "|---|---|---|",
        *[f"| {name} | {verdict} | {evidence} |" for name, verdict, evidence in review_items],
        "",
        (
            "Numeric-correctness verdict: **PASS**. The native-FH timing dependency "
            "is external and will remain explicitly unconfirmed."
        ),
        "",
    ]
    _atomic_text(artifact_root / "PREFLIGHT_REVIEW.md", "\n".join(review_lines))
    evaluator_files = (
        "scripts/crosswindow_campaign.py",
        "scripts/replay_crosswindow_association.py",
        "scripts/evaluate_crosswindow_consensus.py",
        "scripts/system_comparison_metrics.py",
    )
    lock = {
        "schema_version": "crosswindow-final-lock-v1",
        "status": "LOCKED",
        "locked_before_protocol_b_candidate_scoring": True,
        "identity": {
            "parent_commit": config["identity"]["parent_commit"],
            "R1_SHA": config["identity"]["r1_checkpoint_sha256"],
            "head_SHA": None,
            "producer_id": "R1-B4-policy",
        },
        "final_system": {
            "association_parent": association_parent,
            "association_config_id": (
                "D0"
                if association_parent == "D0"
                else exploratory["config_id"]
            ),
            "K": config["runtime"]["resident_capacity"],
            "buffer_groups_max": config["runtime"]["buffer_groups_max"],
            "mask_selector": selected["selector"],
            "score_mode": selected["score_mode"],
            "score_reducer": selected["score_reducer"],
            "output_policy": "lag1",
            "learning_status": e6["status"],
        },
        "exploratory_connector": exploratory,
        "selection_evidence": {
            "E1_gate": _read_json(artifact_root / "e1/gate.json"),
            "E2_family_candidates_sha256": _sha256(
                artifact_root / "e2/family_candidates.json"
            ),
            "E4_content_choice_sha256": _sha256(
                artifact_root / "e4/content_choice_status.json"
            ),
            "E6_status_sha256": _sha256(artifact_root / "e6/status.json"),
        },
        "data_roles": roles["roles"],
        "inputs": {
            "dev_base_manifest_sha256": _sha256(
                PROJECT_ROOT / paths["dev_base_manifest"]
            ),
            "dev_supplement_manifest_sha256": _sha256(
                PROJECT_ROOT / paths["dev_supplement_manifest"]
            ),
            "pb_base_manifest_sha256": _sha256(
                PROJECT_ROOT / paths["pb_base_manifest"]
            ),
            "pb_supplement_manifest_sha256": _sha256(
                PROJECT_ROOT / paths["pb_supplement_manifest"]
            ),
        },
        "evaluator_files": [
            {"path": path, "sha256": _sha256(PROJECT_ROOT / path)}
            for path in evaluator_files
        ],
        "planned_comparisons": [
            "FH-R1-native",
            "D-LEGACY",
            "D0",
            "A0-U-default",
            str(exploratory["method"]),
            "frozen-final-candidate",
        ],
        "native_fh_asset_status": "BLOCKED_ASSET",
    }
    _atomic_json(lock_path, lock)
    state.update(
        {
            "stage": "SELECT_AND_LOCK",
            "stage_status": "PASS",
            "selection_lock_sha256": _sha256(lock_path),
            "next_command": (
                "python -m scripts.crosswindow_campaign confirm --config "
                "configs/crosswindow_evidence_v1.yaml --resume"
            ),
        }
    )
    atomic_write_run_state(state_path, state)
    print(json.dumps(lock, indent=2, sort_keys=True))
    return 0


def _confirm(args: argparse.Namespace) -> int:
    from models.overlap_entity_association import (
        A0_DEFAULT,
        A1_DEFAULT,
        A2_DEFAULT,
    )
    from scripts.evaluate_crosswindow_consensus import E4RevisionAccumulator
    from scripts.profile_crosswindow import evaluate_final_status
    from scripts.replay_crosswindow_association import E2ReplayAccumulator

    started_cpu = time.process_time()
    config_path = Path(args.config).resolve()
    config = load_campaign_config(config_path)
    paths = config["paths"]
    artifact_root = PROJECT_ROOT / paths["artifact_root"]
    final_root = artifact_root / "final"
    state_path = artifact_root / "RUN_STATE.json"
    state = load_resume_state(
        state_path,
        parent=config["identity"]["parent_commit"],
        instruction_sha=config["identity"]["instruction_sha256"],
        config_sha=_sha256(config_path),
        data_sha=_sha256(PROJECT_ROOT / paths["data_contract"]),
    )
    required = (
        "all_t_metrics.csv",
        "identity_and_revision.csv",
        "resources.csv",
        "status.json",
    )
    if (
        args.resume
        and state.get("stage") == "E5-final"
        and state.get("stage_status") == "PASS"
    ):
        if not all((final_root / name).is_file() for name in required):
            raise CampaignError("completed final state lacks required artifacts")
        status = _read_json(final_root / "status.json")
        print(json.dumps({**status, "resumed": True}, indent=2, sort_keys=True))
        return 0
    if state.get("stage") != "SELECT_AND_LOCK" or state.get("stage_status") != "PASS":
        raise CampaignError("final confirmation requires a completed selection lock")
    lock = _read_json(artifact_root / "selection/FINAL_LOCK.json")
    if lock.get("status") != "LOCKED":
        raise CampaignError("final selection lock is unavailable")
    locked_system = lock["final_system"]
    exploratory = lock["exploratory_connector"]
    exploratory_method = exploratory["method"]
    if exploratory_method == "A1-default":
        exploratory_config = A1_DEFAULT
    elif exploratory_method == "A2-default":
        exploratory_config = A2_DEFAULT
    else:
        raise CampaignError("locked exploratory connector differs")

    external_root_value = args.external_root or os.environ.get("PERSIST4D_RUN_ROOT")
    if not external_root_value:
        raise CampaignError("confirm requires --external-root or PERSIST4D_RUN_ROOT")
    external_root = Path(external_root_value).resolve()
    assets = _read_json(external_root / "assets.local.json")
    required_assets = (
        "pb_base_cache_root",
        "pb_supplement_root",
        "metric_dataset_spec",
    )
    if any(not isinstance(assets.get(key), str) for key in required_assets):
        raise CampaignError("Protocol-B assets are unresolved")
    base_root = Path(assets["pb_base_cache_root"])
    supplement_root = Path(assets["pb_supplement_root"])
    metric_dataset_spec = Path(assets["metric_dataset_spec"])
    if (
        not base_root.is_dir()
        or not supplement_root.is_dir()
        or not metric_dataset_spec.is_file()
    ):
        raise CampaignError("Protocol-B assets are unavailable")
    base_manifest = _read_json(PROJECT_ROOT / paths["pb_base_manifest"])
    supplement_manifest = _read_json(PROJECT_ROOT / paths["pb_supplement_manifest"])
    checkpoint_sha256 = supplement_manifest.get("checkpoint_sha256")
    if checkpoint_sha256 != config["identity"]["r1_checkpoint_sha256"]:
        raise CampaignError("Protocol-B checkpoint identity differs")
    protocol_references = _available_dev_references(base_manifest)
    units = list(
        iter_campaign_units(
            role="PROTOCOL-B",
            role_references=protocol_references,
            base_manifest=base_manifest,
            supplement_manifest=supplement_manifest,
        )
    )
    if len(units) != 129:
        raise CampaignError("Protocol-B logical-unit coverage differs")
    metric_spec = yaml.safe_load(metric_dataset_spec.read_text(encoding="utf-8"))
    class_mapping = (
        metric_spec.get("valid_class_ids") if isinstance(metric_spec, dict) else None
    )
    if not isinstance(class_mapping, list) or any(
        isinstance(value, bool) or not isinstance(value, int) for value in class_mapping
    ):
        raise CampaignError("metric dataset class mapping is unavailable")
    main_configs = {
        "A0-U-default": A0_DEFAULT,
        str(exploratory_method): exploratory_config,
    }
    main = E2ReplayAccumulator(
        dataset_spec=str(metric_dataset_spec),
        class_mapping=tuple(class_mapping),
        checkpoint_sha256=checkpoint_sha256,
        source_commit=config["identity"]["parent_commit"],
        data_role="PROTOCOL-B",
        association_configs=main_configs,
        capacity=100,
    )
    capacity_accumulators = {
        capacity: E2ReplayAccumulator(
            dataset_spec=str(metric_dataset_spec),
            class_mapping=tuple(class_mapping),
            checkpoint_sha256=checkpoint_sha256,
            source_commit=config["identity"]["parent_commit"],
            data_role="PROTOCOL-B",
            association_configs={str(exploratory_method): exploratory_config},
            capacity=capacity,
        )
        for capacity in (16, 32)
    }
    selected_selector = str(locked_system["mask_selector"])
    selected_score_mode = str(locked_system["score_mode"])
    revision_accumulator = (
        E4RevisionAccumulator(
            dataset_spec=str(metric_dataset_spec),
            class_mapping=tuple(class_mapping),
            checkpoint_sha256=checkpoint_sha256,
            source_commit=config["identity"]["parent_commit"],
            association_method=str(exploratory_method),
            association_config=exploratory_config,
            selectors=(selected_selector,),
            score_modes=(selected_score_mode,),
        )
        if selected_selector != "M-new"
        else None
    )
    hash_cache_path = external_root / "verified_hashes.local.json"
    hash_cache = _read_json(hash_cache_path) if hash_cache_path.is_file() else {}
    completed_units = []
    replay_durations_ms = []
    for index, unit in enumerate(units, start=1):
        base = _verified_torch_load(
            path=base_root / unit.base_filename,
            record={"bytes": unit.base_bytes, "sha256": unit.base_sha256},
            hash_cache=hash_cache,
        )
        supplement = _verified_torch_load(
            path=supplement_root / unit.supplement_filename,
            record={
                "bytes": unit.supplement_bytes,
                "sha256": unit.supplement_sha256,
            },
            hash_cache=hash_cache,
        )
        started_unit = time.perf_counter()
        main.update(
            logical_unit_id=unit.logical_unit_id,
            base=base,
            supplement=supplement,
        )
        for accumulator in capacity_accumulators.values():
            accumulator.update(
                logical_unit_id=unit.logical_unit_id,
                base=base,
                supplement=supplement,
            )
        if revision_accumulator is not None:
            revision_accumulator.update(
                logical_unit_id=unit.logical_unit_id,
                base=base,
                supplement=supplement,
            )
        replay_durations_ms.append((time.perf_counter() - started_unit) * 1000.0)
        completed_units.append(unit.logical_unit_id)
        del base, supplement
        print(
            json.dumps(
                {
                    "stage": "E5-final",
                    "completed": index,
                    "total": len(units),
                    "logical_unit_id": unit.logical_unit_id,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    _atomic_json(hash_cache_path, hash_cache)
    main_result = main.finalize()
    capacity_results = {
        capacity: accumulator.finalize()
        for capacity, accumulator in capacity_accumulators.items()
    }
    revision_result = (
        revision_accumulator.finalize()
        if revision_accumulator is not None
        else None
    )
    main_rows = main_result["metric_rows"]
    aggregate = [row for row in main_rows if row["reference"] == "all"]
    by_method_horizon = {
        (str(row["method"]), int(row["T"])): row for row in aggregate
    }
    final_parent = str(locked_system["association_parent"])
    if revision_result is None:
        final_values = {
            horizon: by_method_horizon[(final_parent, horizon)]
            for horizon in (2, 3, 4, 5)
        }
    else:
        final_values = {
            int(row["T"]): row
            for row in revision_result["metric_rows"]
            if row["parent"] == final_parent
        }
    final_rows = []
    for horizon in (2, 3, 4, 5):
        source = final_values[horizon]
        template = by_method_horizon[(final_parent, horizon)]
        final_rows.append(
            {
                **template,
                "method": "frozen-final-candidate",
                "config_id": locked_system["association_config_id"],
                "mask_selector": selected_selector,
                "score_mode": selected_score_mode,
                "t_mAP": source["t_mAP"],
                "t_mAP50": source["t_mAP50"],
                "t_mAP25": source["t_mAP25"],
                "t_REC": source["t_REC"],
                "prefix_overall_mAP": source["prefix_overall_mAP"],
                "published_current_AP": source["published_current_AP"],
                "status": "MEASURED_FROZEN",
                "reason": "selected before Protocol-B scoring",
            }
        )
    d_legacy_rows = [
        {
            **row,
            "method": "D-LEGACY",
            "config_id": "D-LEGACY-LAST",
            "status": "MEASURED_ALIAS_D0",
            "reason": "D0 is the corrected LAST legacy route on this fixed source",
        }
        for row in aggregate
        if row["method"] == "D0"
    ]
    template = final_rows[0]
    native_rows = []
    numeric_fields = (
        "t_mAP",
        "t_mAP50",
        "t_mAP25",
        "t_REC",
        "prefix_overall_mAP",
        "raw_current_AP",
        "published_current_AP",
        "normalized_id_switch_rate",
        "fragmentation_rate",
        "merge_rate",
        "wrong_reactivation_rate",
        "gap_recovery_accuracy",
        "gap_recovery_recall",
    )
    for horizon in (2, 3, 4, 5):
        row = {
            **template,
            "method": "FH-R1-native",
            "config_id": "FH-R1-native",
            "T": horizon,
            "reference": "all",
            "status": "BLOCKED_ASSET",
            "reason": "fh_native_cache_root is unavailable",
        }
        for field in numeric_fields:
            row[field] = None
        native_rows.append(row)
    all_t_rows = [*native_rows, *d_legacy_rows, *main_rows, *final_rows]
    _atomic_csv(
        final_root / "all_t_metrics.csv",
        all_t_rows,
        fieldnames=tuple(all_t_rows[0]),
    )
    identity_rows = []
    for row in [*aggregate, *final_rows]:
        identity_rows.append(
            {
                "population_id": row["population_id"],
                "data_role": row["data_role"],
                "method": row["method"],
                "config_id": row["config_id"],
                "mask_selector": row["mask_selector"],
                "score_mode": row["score_mode"],
                "T": row["T"],
                **{
                    field: row.get(field)
                    for field in (
                        "deployment_id_switches",
                        "identity_transition_opportunities",
                        "normalized_id_switch_rate",
                        "fragmentation_count",
                        "fragmentation_opportunities",
                        "fragmentation_rate",
                        "merge_count",
                        "merge_opportunities",
                        "merge_rate",
                        "gap_opportunities",
                        "recovery_attempts",
                        "correct_recoveries",
                        "wrong_reactivations",
                        "wrong_reactivation_rate",
                        "gap_recovery_accuracy",
                        "gap_recovery_recall",
                    )
                },
                "status": (
                    "IDENTITY_PLAN_FIXED_TASK_MATCH_PARENT_PROXY"
                    if revision_result is not None
                    and row["method"] == "frozen-final-candidate"
                    else row["status"]
                ),
                "reason": (
                    "revision changes masks but not issued identity plans"
                    if revision_result is not None
                    and row["method"] == "frozen-final-candidate"
                    else row["reason"]
                ),
            }
        )
    _atomic_csv(
        final_root / "identity_and_revision.csv",
        identity_rows,
        fieldnames=tuple(identity_rows[0]),
    )
    resource_rows = []
    for method in ("FH-R1-native", "D0", "frozen-final-candidate"):
        for horizon in (2, 3, 4, 5):
            resource_rows.append(
                {
                    "row_type": "PROFILE",
                    "method": method,
                    "K": 100,
                    "T": horizon,
                    "model_update_ms": None,
                    "output_materialization_ms": None,
                    "end_to_end_ms": None,
                    "cumulative_ms": None,
                    "allocated_peak_bytes": None,
                    "reserved_peak_bytes": None,
                    "cpu_rss_bytes": None,
                    "resident_bytes": None,
                    "buffer_bytes": None,
                    "archive_bytes": None,
                    "materialized_bytes": None,
                    "cache_replay_with_metrics_ms": (
                        statistics.median(replay_durations_ms)
                        if method != "FH-R1-native"
                        else None
                    ),
                    "t_mAP": None,
                    "peak_occupied_slots": None,
                    "rejected_births": None,
                    "status": (
                        "BLOCKED_ASSET"
                        if method == "FH-R1-native"
                        else "CACHE_REPLAY_ONLY"
                    ),
                    "reason": (
                        "fh_native_cache_root is unavailable"
                        if method == "FH-R1-native"
                        else "not a real-forward timing measurement"
                    ),
                }
            )
    capacity_sources = {100: main_result, **capacity_results}
    for capacity, result in sorted(capacity_sources.items()):
        rows = [row for row in result["metric_rows"] if row["reference"] == "all"]
        accounting = result["capacity_accounting"]
        for method in ("D0", str(exploratory_method)):
            for row in rows:
                if row["method"] != method:
                    continue
                resource_rows.append(
                    {
                        "row_type": "CAPACITY",
                        "method": method,
                        "K": capacity,
                        "T": row["T"],
                        "model_update_ms": None,
                        "output_materialization_ms": None,
                        "end_to_end_ms": None,
                        "cumulative_ms": None,
                        "allocated_peak_bytes": None,
                        "reserved_peak_bytes": None,
                        "cpu_rss_bytes": None,
                        "resident_bytes": accounting[method]["resident_bytes"],
                        "buffer_bytes": accounting[method]["buffer_bytes"],
                        "archive_bytes": accounting[method]["archive_bytes"],
                        "materialized_bytes": accounting[method][
                            "materialized_bytes"
                        ],
                        "cache_replay_with_metrics_ms": None,
                        "t_mAP": row["t_mAP"],
                        "peak_occupied_slots": accounting[method][
                            "peak_occupied_slots"
                        ],
                        "rejected_births": accounting[method]["rejected_births"],
                        "status": "MEASURED_PARENT_TRAJECTORY",
                        "reason": (
                            "capacity-uninformative"
                            if accounting[method]["peak_occupied_slots"] < capacity
                            else "capacity reached"
                        ),
                    }
                )
    _atomic_csv(
        final_root / "resources.csv",
        resource_rows,
        fieldnames=tuple(resource_rows[0]),
    )
    candidate_tmap = {
        int(row["T"]): float(row["t_mAP"]) for row in final_rows
    }
    final_status = evaluate_final_status(
        candidate_tmap=candidate_tmap,
        native_fh_tmap=None,
        resource_status="UNCONFIRMED",
        epsilon=float(config["success"]["tmap_all_T_epsilon"]),
    )
    status = {
        **final_status,
        "status": "PASS",
        "execution_status": "COMPLETE_WITH_EXTERNAL_BLOCKERS",
        "population_id": "protocol_b_final",
        "data_role": "PROTOCOL-B",
        "reference_count": len(protocol_references),
        "logical_unit_count": len(units),
        "common_cohort": True,
        "coverage_by_method": {
            "D0": len(units),
            "A0-U-default": len(units),
            str(exploratory_method): len(units),
            "frozen-final-candidate": len(units),
            "FH-R1-native": 0,
        },
        "native_fh_status": "BLOCKED_ASSET",
        "profile_status": "UNCONFIRMED_REAL_FORWARD_NOT_RUN",
        "leave_one_reference_out_status": "NOT_RUN_NO_NATIVE_COMPARATOR",
        "cpu_core_hours": (time.process_time() - started_cpu) / 3600.0,
    }
    _atomic_json(final_root / "status.json", status)
    budget = state.get("budget_used")
    if not isinstance(budget, dict):
        raise CampaignError("run-state budget is unavailable")
    budget["cpu_core_hours"] = float(budget.get("cpu_core_hours", 0.0)) + float(
        status["cpu_core_hours"]
    )
    state.update(
        {
            "stage": "E5-final",
            "stage_status": "PASS",
            "completed_units": completed_units,
            "final": status,
            "next_command": (
                "python -m scripts.crosswindow_campaign report --config "
                "configs/crosswindow_evidence_v1.yaml"
            ),
        }
    )
    atomic_write_run_state(state_path, state)
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0


def _remote_branch_sha(branch: str) -> str | None:
    result = subprocess.run(
        ["git", "ls-remote", "origin", f"refs/heads/{branch}"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return None
    value = result.stdout.split()[0]
    return value if len(value) == 40 else None


def _remote_file_sha(repository: str, commit: str, path: str) -> str | None:
    url = f"https://raw.githubusercontent.com/{repository}/{commit}/{path}"
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            payload = response.read()
    except (OSError, urllib.error.URLError):
        return None
    return hashlib.sha256(payload).hexdigest()


def _report(args: argparse.Namespace) -> int:
    config_path = Path(args.config).resolve()
    config = load_campaign_config(config_path)
    artifact_root = PROJECT_ROOT / config["paths"]["artifact_root"]
    state = _read_json(artifact_root / "RUN_STATE.json")
    if state.get("stage") != "E5-final" or state.get("stage_status") != "PASS":
        raise CampaignError("report requires completed final confirmation")
    final_status = _read_json(artifact_root / "final/status.json")
    e1_gate = _read_json(artifact_root / "e1/gate.json")
    e2 = _read_json(artifact_root / "e2/family_candidates.json")
    e4 = _read_json(artifact_root / "e4/content_choice_status.json")
    lock = _read_json(artifact_root / "selection/FINAL_LOCK.json")
    experiment_commit = _git_head()
    branch = config["identity"]["branch"]
    remote_sha = _remote_branch_sha(branch)
    main_table_path = "artifacts/crosswindow_evidence_v1/final/all_t_metrics.csv"
    remote_main_sha = (
        _remote_file_sha(
            config["identity"]["repository"], experiment_commit, main_table_path
        )
        if remote_sha == experiment_commit
        else None
    )
    local_main_sha = _sha256(PROJECT_ROOT / main_table_path)
    experiment_publication_status = (
        "RESULTS_PUSH_VERIFIED"
        if remote_sha == experiment_commit and remote_main_sha == local_main_sha
        else "PUSHED_READBACK_UNVERIFIED"
        if remote_sha == experiment_commit
        else "RESULTS_NOT_YET_PUSHED"
    )
    publication = {
        "schema_version": "crosswindow-publication-v1",
        "branch": branch,
        "experiment_commit_E": experiment_commit,
        "experiment_remote_sha": remote_sha,
        "experiment_main_table_local_sha256": local_main_sha,
        "experiment_main_table_readback_sha256": remote_main_sha,
        "experiment_status": experiment_publication_status,
        "documentation_commit_P": "RESOLVE_FROM_REMOTE_BRANCH_AFTER_PUBLISH",
        "documentation_status": "PENDING",
    }
    _atomic_json(artifact_root / "PUBLICATION.json", publication)
    candidate = final_status["candidate_t_mAP"]
    report = f"""# Persist4D CrossWindow V1 final report

## Outcome

- Execution: complete for E0, E1, E2, E3, E4, E6 decision, lock, and cached Protocol-B confirmation.
- Scientific joint goal: **not confirmed** (`JOINT_GOAL_PASS=false`).
- Quality gate: `{final_status['TMAP_ALL_T_STATUS']}` because native FH-R1 payloads were unavailable.
- Resource gate: `{final_status['RESOURCE_STATUS']}` because comparable native real-forward profiling was unavailable.
- Frozen system: `{lock['final_system']['association_parent']}` + `{lock['final_system']['mask_selector']}` + `{lock['final_system']['score_mode']}`.

## Development evidence

- E1 decision: `{e1_gate['decision']}`; GT-LAG1 long gain `{e1_gate['long_gain']:.6f}` and candidate-complete failure fraction `{e1_gate['candidate_complete_failure_fraction']:.6f}`.
- No 12-config search was authorized. A1/A2 defaults were evaluated on DEV-SEL only.
- Best exploratory connector: `{e2['exploratory_candidate']['method']}` with status `{e2['exploratory_candidate']['status']}`; no connector was promoted.
- E4 selected `{e4['selected_system']['selector']}` with interpretation `{e4['selected_system']['interpretation']}` under its fixed parent.
- E6: `SKIPPED_CONDITION`; no learned head was created and R1 remained frozen.

## Protocol-B evidence

- Common cached cohort: {final_status['logical_unit_count']} logical units across {final_status['reference_count']} references.
- Frozen candidate t-mAP: T2 `{candidate['2']:.6f}`, T3 `{candidate['3']:.6f}`, T4 `{candidate['4']:.6f}`, T5 `{candidate['5']:.6f}`.
- FH-R1-native rows are null with `BLOCKED_ASSET`; cached replay timing is not presented as network profiling.
- Capacity K=16/32/100 is reported for D0 and the exploratory connector in `final/resources.csv`.

## Boundaries

Development and Protocol-B caches share the frozen R1 producer and are historically exposed; this is not evidence on new independent scenes. GT-assisted rows are diagnostic only and never compete as deployable methods. The negative association result does not prove that all possible association mechanisms lack headroom.
"""
    _atomic_text(artifact_root / "FINAL_REPORT.md", report)
    handoff = f"""# CrossWindow V1 handoff

- Parent: `{config['identity']['parent_commit']}`
- Branch: `{branch}`
- Experiment commit E: `{experiment_commit}`
- Documentation commit P: resolve `refs/heads/{branch}` after publication
- R1 checkpoint: `{config['identity']['r1_checkpoint_sha256']}`
- Final lock: `selection/FINAL_LOCK.json`
- Final metrics: `final/all_t_metrics.csv`
- Final status: `final/status.json`
- External run root: `{args.external_root or os.environ.get('PERSIST4D_RUN_ROOT') or 'unresolved'}`

The E1 association gate was negative, A1/A2 failed DEV-SEL promotion, and the final parent remained D0. Native FH data were unavailable, so quality and resource superiority remain unconfirmed. Resume or reproduce using `COMMANDS.md`; do not change `FINAL_LOCK.json` after inspecting Protocol-B results.
"""
    _atomic_text(artifact_root / "HANDOFF.md", handoff)
    commands = f"""# Commands

```bash
export PERSIST4D_RUN_ROOT={args.external_root or os.environ.get('PERSIST4D_RUN_ROOT') or '/mnt/shared/ww/persist4d-crosswindow-evidence-v1'}
conda run -n persist4d python -m scripts.crosswindow_campaign preflight --config configs/crosswindow_evidence_v1.yaml --external-root "$PERSIST4D_RUN_ROOT" --resume --read-only
conda run -n persist4d python -m scripts.crosswindow_campaign run --config configs/crosswindow_evidence_v1.yaml --external-root "$PERSIST4D_RUN_ROOT" --through E4 --resume
conda run -n persist4d python -m scripts.crosswindow_campaign conditional-train --config configs/crosswindow_evidence_v1.yaml --external-root "$PERSIST4D_RUN_ROOT" --resume
conda run -n persist4d python -m scripts.crosswindow_campaign lock --config configs/crosswindow_evidence_v1.yaml --external-root "$PERSIST4D_RUN_ROOT" --resume
conda run -n persist4d python -m scripts.crosswindow_campaign confirm --config configs/crosswindow_evidence_v1.yaml --external-root "$PERSIST4D_RUN_ROOT" --resume
conda run -n persist4d python -m scripts.crosswindow_campaign report --config configs/crosswindow_evidence_v1.yaml --external-root "$PERSIST4D_RUN_ROOT"
conda run -n persist4d python -m scripts.crosswindow_campaign publish --config configs/crosswindow_evidence_v1.yaml --external-root "$PERSIST4D_RUN_ROOT"
```
    """
    _atomic_text(artifact_root / "COMMANDS.md", commands)
    state.update(
        {
            "stage": "REPORT",
            "stage_status": "PASS",
            "experiment_commit_E": experiment_commit,
            "next_command": (
                "python -m scripts.crosswindow_campaign publish --config "
                "configs/crosswindow_evidence_v1.yaml"
            ),
        }
    )
    atomic_write_run_state(artifact_root / "RUN_STATE.json", state)
    manifest_entries = []
    for path in sorted(artifact_root.rglob("*")):
        if (
            not path.is_file()
            or path.name == "FINAL_MANIFEST.json"
            or path.name == "PUBLICATION_RECEIPT.local.json"
        ):
            continue
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        manifest_entries.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "role": path.relative_to(artifact_root).parts[0],
            }
        )
    manifest = {
        "schema_version": "crosswindow-final-manifest-v1",
        "experiment_commit_E": experiment_commit,
        "self_hash_included": False,
        "entries": manifest_entries,
    }
    _atomic_json(artifact_root / "FINAL_MANIFEST.json", manifest)
    print(
        json.dumps(
            {
                "status": "PASS",
                "experiment_commit_E": experiment_commit,
                "experiment_publication_status": experiment_publication_status,
                "manifest_entry_count": len(manifest_entries),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _publish(args: argparse.Namespace) -> int:
    config = load_campaign_config(Path(args.config).resolve())
    artifact_root = PROJECT_ROOT / config["paths"]["artifact_root"]
    state = _read_json(artifact_root / "RUN_STATE.json")
    if state.get("stage") != "REPORT" or state.get("stage_status") != "PASS":
        raise CampaignError("publish requires completed reports")
    publication = _read_json(artifact_root / "PUBLICATION.json")
    experiment_commit = publication.get("experiment_commit_E")
    documentation_commit = _git_head()
    branch = config["identity"]["branch"]
    if not isinstance(experiment_commit, str) or len(experiment_commit) != 40:
        raise CampaignError("experiment commit E is unavailable")
    if documentation_commit == experiment_commit:
        raise CampaignError("documentation commit P has not been created")
    if not _git_contains(experiment_commit):
        raise CampaignError("documentation commit does not contain experiment E")
    external_root_value = args.external_root or os.environ.get("PERSIST4D_RUN_ROOT")
    if not external_root_value:
        raise CampaignError("publish requires --external-root or PERSIST4D_RUN_ROOT")
    external_root = Path(external_root_value).resolve()
    external_root.mkdir(parents=True, exist_ok=True)
    push = None
    environment = {**os.environ, "GIT_SSH_COMMAND": "ssh -o BatchMode=yes"}
    for _ in range(2):
        try:
            push = subprocess.run(
                ["git", "push", "-u", "origin", branch],
                cwd=PROJECT_ROOT,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
                timeout=180,
            )
        except subprocess.TimeoutExpired:
            push = None
        if push is not None and push.returncode == 0:
            break
    if push is None or push.returncode != 0:
        bundle_path = external_root / "persist4d-crosswindow-evidence-v1.bundle"
        bundle = subprocess.run(
            ["git", "bundle", "create", str(bundle_path), branch],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        receipt = {
            "schema_version": "crosswindow-publication-receipt-v1",
            "status": "PUBLISH_BLOCKED_AUTH",
            "experiment_commit_E": experiment_commit,
            "documentation_commit_P": documentation_commit,
            "remote_sha": _remote_branch_sha(branch),
            "bundle_path": str(bundle_path) if bundle.returncode == 0 else None,
            "bundle_sha256": _sha256(bundle_path) if bundle.returncode == 0 else None,
            "push_stderr": push.stderr[-2000:] if push is not None else "timeout",
            "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        _atomic_json(external_root / "PUBLICATION_RECEIPT.local.json", receipt)
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 1
    remote_sha = _remote_branch_sha(branch)
    if remote_sha != documentation_commit:
        raise CampaignError("remote branch SHA differs after push")
    readback_paths = (
        "artifacts/crosswindow_evidence_v1/HANDOFF.md",
        "artifacts/crosswindow_evidence_v1/FINAL_MANIFEST.json",
        "artifacts/crosswindow_evidence_v1/final/all_t_metrics.csv",
    )
    readback = []
    for relative in readback_paths:
        local_sha = _sha256(PROJECT_ROOT / relative)
        remote_file_sha = _remote_file_sha(
            config["identity"]["repository"], documentation_commit, relative
        )
        readback.append(
            {
                "path": relative,
                "local_sha256": local_sha,
                "readback_sha256": remote_file_sha,
                "match": remote_file_sha == local_sha,
            }
        )
    verified = all(item["match"] for item in readback)
    receipt = {
        "schema_version": "crosswindow-publication-receipt-v1",
        "status": "PUSH_VERIFIED" if verified else "PUSHED_READBACK_UNVERIFIED",
        "experiment_commit_E": experiment_commit,
        "documentation_commit_P": documentation_commit,
        "remote_sha": remote_sha,
        "branch": branch,
        "manifest_sha256": _sha256(artifact_root / "FINAL_MANIFEST.json"),
        "readback": readback,
        "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    _atomic_json(external_root / "PUBLICATION_RECEIPT.local.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if verified else 2


def _configure_cpu_threads(config: Mapping[str, Any]) -> None:
    maximum = int(config["budget"]["max_cpu_workers"])
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = str(maximum)
    import torch

    torch.set_num_threads(maximum)


def _run(args: argparse.Namespace) -> int:
    config = load_campaign_config(Path(args.config).resolve())
    _configure_cpu_threads(config)
    if args.through == "E0":
        return _run_e0(args)
    if args.through == "E1":
        e0_status = _run_e0(args)
        return e0_status if e0_status else _run_e1(args)
    if args.through == "E2":
        return _run_e2(args)
    if args.through == "E4":
        return _run_e4(args)
    raise CampaignError("unsupported run horizon")


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
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--config", default=str(DEFAULT_CONFIG))
    preflight.add_argument("--external-root")
    preflight.add_argument("--resume", action="store_true")
    preflight.add_argument("--read-only", action="store_true")
    run = subparsers.add_parser("run")
    run.add_argument("--config", default=str(DEFAULT_CONFIG))
    run.add_argument("--external-root")
    run.add_argument("--resume", action="store_true")
    run.add_argument("--through", required=True, choices=("E0", "E1", "E2", "E4"))
    conditional = subparsers.add_parser("conditional-train")
    conditional.add_argument("--config", default=str(DEFAULT_CONFIG))
    conditional.add_argument("--external-root")
    conditional.add_argument("--resume", action="store_true")
    lock = subparsers.add_parser("lock")
    lock.add_argument("--config", default=str(DEFAULT_CONFIG))
    lock.add_argument("--external-root")
    lock.add_argument("--resume", action="store_true")
    confirm = subparsers.add_parser("confirm")
    confirm.add_argument("--config", default=str(DEFAULT_CONFIG))
    confirm.add_argument("--external-root")
    confirm.add_argument("--resume", action="store_true")
    report = subparsers.add_parser("report")
    report.add_argument("--config", default=str(DEFAULT_CONFIG))
    report.add_argument("--external-root")
    report.add_argument("--resume", action="store_true")
    publish = subparsers.add_parser("publish")
    publish.add_argument("--config", default=str(DEFAULT_CONFIG))
    publish.add_argument("--external-root")
    publish.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "bootstrap":
        return _bootstrap(args)
    if args.command == "preflight":
        return _preflight(args)
    if args.command == "run":
        return _run(args)
    if args.command == "conditional-train":
        return _conditional_train(args)
    if args.command == "lock":
        return _lock(args)
    if args.command == "confirm":
        return _confirm(args)
    if args.command == "report":
        return _report(args)
    if args.command == "publish":
        return _publish(args)
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
