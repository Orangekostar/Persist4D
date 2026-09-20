"""Resumable command line entry point for CrossWindow V1."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import subprocess
import tempfile
import time
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


def _run(args: argparse.Namespace) -> int:
    if args.through != "E0":
        raise CampaignError("this implementation stage currently supports --through E0")
    return _run_e0(args)


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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "bootstrap":
        return _bootstrap(args)
    if args.command == "preflight":
        return _preflight(args)
    if args.command == "run":
        return _run(args)
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
