#!/usr/bin/env python3
"""Require and, when requested, stage the exact authorized epoch-90 resume."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.rescene_rootcause_evaluation import (
    RootCauseEvaluationError,
    validate_candidate_binding,
    validate_checkpoint_manifest_binding,
    validate_checkpoint_payload,
)
from utils.rescene_rootcause_preflight import canonical_sha256

LAST_CHECKPOINT = re.compile(r"last(?:-v\d+)?\.ckpt")


def _load_json(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RootCauseEvaluationError(f"{name} is not readable JSON") from error
    if not isinstance(value, dict):
        raise RootCauseEvaluationError(f"{name} must contain an object")
    return value


def _file_identity(path: Path) -> dict[str, object]:
    try:
        before = path.lstat()
        if path.is_symlink() or not stat.S_ISREG(before.st_mode):
            raise OSError
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                size += len(block)
                digest.update(block)
        after = path.lstat()
    except OSError as error:
        raise RootCauseEvaluationError("resume input is unavailable") from error
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ) or size != after.st_size:
        raise RootCauseEvaluationError("resume input changed while hashing")
    return {"bytes": size, "sha256": digest.hexdigest()}


def _validate_hash(payload: Mapping[str, Any], field: str, *, name: str) -> None:
    expected = payload.get(field)
    unsigned = dict(payload)
    unsigned.pop(field, None)
    if not isinstance(expected, str) or canonical_sha256(unsigned) != expected:
        raise RootCauseEvaluationError(f"{name} hash differs")


def _core_resume_plan(
    *,
    variant: str,
    authorization_path: Path,
    candidate_path: Path,
    decision_path: Path,
    checkpoint_manifest_path: Path,
    exact_checkpoint_path: Path,
) -> dict[str, object]:
    authorization = _load_json(authorization_path, name="variant authorization")
    _validate_hash(authorization, "authorization_sha256", name="variant authorization")
    candidate = _load_json(candidate_path, name="candidate record")
    validate_candidate_binding(
        variant=variant, authorization=authorization, candidate=candidate
    )
    decision = _load_json(decision_path, name="short-curve decision")
    _validate_hash(decision, "content_sha256", name="short-curve decision")
    if (
        decision.get("status") != "pass"
        or decision.get("selected_variant") != variant
        or decision.get("full_training_authorized") is not True
        or decision.get("full_training_status") != "authorized"
        or decision.get("variant_authorization_sha256")
        != authorization["authorization_sha256"]
    ):
        raise RootCauseEvaluationError("full candidate is not authorized")

    manifest = _load_json(checkpoint_manifest_path, name="checkpoint manifest")
    _validate_hash(manifest, "content_sha256", name="checkpoint manifest")
    if (
        validate_checkpoint_manifest_binding(manifest, authorization=authorization)
        != variant
        or manifest["checkpoint"]["selected_epoch"] != 90
        or manifest["checkpoint"]["selected_step"] != 5_940
        or manifest["bindings"]["candidate_id"] != candidate["candidate_id"]
    ):
        raise RootCauseEvaluationError("epoch-90 checkpoint binding differs")
    exact_identity = _file_identity(exact_checkpoint_path)
    if any(
        exact_identity[field] != manifest["checkpoint"][field]
        for field in ("bytes", "sha256")
    ):
        raise RootCauseEvaluationError(
            "exact epoch-90 checkpoint differs from manifest"
        )
    from scripts.evaluate_rescene_rootcause_checkpoint import (
        _expected_state_dict_entries,
    )

    return {
        "schema_version": 1,
        "status": "pass",
        "experiment": authorization.get(
            "experiment", "rescene_task_learning_root_cause_v1"
        ),
        "variant": variant,
        "completed_epoch": 90,
        "selected_step": 5_940,
        "resume_scope": "completed_epoch_boundary_only",
        "exact_checkpoint_name": exact_checkpoint_path.name,
        "exact_checkpoint_bytes": exact_identity["bytes"],
        "exact_checkpoint_sha256": exact_identity["sha256"],
        "candidate_id": candidate["candidate_id"],
        "variant_authorization_sha256": authorization["authorization_sha256"],
        "short_decision_sha256": decision["content_sha256"],
        "checkpoint_manifest_sha256": manifest["content_sha256"],
        "config_sha256": manifest["checkpoint"]["config_sha256"],
        "common_initialization_sha256": manifest["bindings"][
            "common_initialization_sha256"
        ],
        "pretrained_sha256": manifest["bindings"]["pretrained_sha256"],
        "expected_state_dict_entries": _expected_state_dict_entries(
            authorization, variant
        ),
    }


def validate_exact_resume_evidence(
    *,
    variant: str,
    authorization_path: Path,
    candidate_path: Path,
    decision_path: Path,
    checkpoint_manifest_path: Path,
    exact_checkpoint_path: Path,
    selected_resume_checkpoint: Path,
) -> dict[str, object]:
    """Return a signed plan only when the runtime selector chooses epoch=090."""

    plan = _core_resume_plan(
        variant=variant,
        authorization_path=authorization_path,
        candidate_path=candidate_path,
        decision_path=decision_path,
        checkpoint_manifest_path=checkpoint_manifest_path,
        exact_checkpoint_path=exact_checkpoint_path,
    )
    try:
        exact = exact_checkpoint_path.resolve(strict=True)
        selected = selected_resume_checkpoint.resolve(strict=True)
    except OSError as error:
        raise RootCauseEvaluationError(
            "selected resume checkpoint is unavailable"
        ) from error
    if selected != exact:
        raise RootCauseEvaluationError(
            "runtime selector did not choose the exact epoch-90 checkpoint"
        )
    plan["runtime_selector_exact_match"] = True
    plan["content_sha256"] = canonical_sha256(plan)
    return plan


def archive_conflicting_last_checkpoint(
    *,
    exact_checkpoint_path: Path,
    selected_checkpoint_path: Path,
    selected_checkpoint_facts: Mapping[str, Any],
    archive_directory: Path,
) -> dict[str, object]:
    """Move only a validated same-boundary last checkpoint to recoverable storage."""

    exact_parent = exact_checkpoint_path.resolve(strict=True).parent
    selected = selected_checkpoint_path.resolve(strict=True)
    if (
        selected.parent != exact_parent
        or selected == exact_checkpoint_path.resolve(strict=True)
        or LAST_CHECKPOINT.fullmatch(selected.name) is None
    ):
        raise RootCauseEvaluationError(
            "resume conflict is not an archivable last-checkpoint"
        )
    if (
        selected_checkpoint_facts.get("selected_epoch") != 90
        or selected_checkpoint_facts.get("selected_step") != 5_940
    ):
        raise RootCauseEvaluationError(
            "last-checkpoint is not at the epoch-90 boundary"
        )
    archive = archive_directory.resolve()
    if archive.parent != exact_parent:
        raise RootCauseEvaluationError(
            "resume archive must remain inside the run directory"
        )
    archive.mkdir(mode=0o755, parents=False, exist_ok=True)
    target = archive / selected.name
    if target.exists() or target.is_symlink():
        raise RootCauseEvaluationError("resume archive target already exists")
    identity = _file_identity(selected)
    os.replace(selected, target)
    if selected.exists() or _file_identity(target) != identity:
        raise RootCauseEvaluationError(
            "recoverable checkpoint archive verification failed"
        )
    return {
        "original_name": selected.name,
        "archive_name": f"{archive.name}/{selected.name}",
        "bytes": identity["bytes"],
        "sha256": identity["sha256"],
        "selected_epoch": 90,
        "selected_step": 5_940,
        "recoverable": True,
    }


def _runtime_selected_checkpoint(output_directory: Path) -> Path:
    from main_instance_segmentation import find_resume_checkpoint

    selected = find_resume_checkpoint(output_directory)
    if selected is None:
        raise RootCauseEvaluationError("runtime selector found no resume checkpoint")
    return Path(selected)


def _checkpoint_facts(
    path: Path, *, expected_state_dict_entries: int
) -> dict[str, object]:
    try:
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise RootCauseEvaluationError("selected checkpoint is unreadable") from error
    return validate_checkpoint_payload(
        payload,
        completed_epoch=90,
        expected_state_dict_entries=expected_state_dict_entries,
    )


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True)
        + "\n"
    ).encode("ascii")


def _publish(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_file() and not path.is_symlink() and path.read_bytes() == payload:
            return
        raise RootCauseEvaluationError("refusing to overwrite full-resume output")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--exact-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage-conflicting-last", action="store_true")
    arguments = parser.parse_args(argv)

    exact = arguments.exact_checkpoint.resolve(strict=True)
    initial_plan = _core_resume_plan(
        variant=arguments.variant,
        authorization_path=arguments.authorization,
        candidate_path=arguments.candidate,
        decision_path=arguments.decision,
        checkpoint_manifest_path=arguments.checkpoint_manifest,
        exact_checkpoint_path=exact,
    )
    selected = _runtime_selected_checkpoint(exact.parent)
    archived = []
    if selected.resolve(strict=True) != exact:
        if not arguments.stage_conflicting_last:
            raise RootCauseEvaluationError(
                "runtime selector did not choose the exact epoch-90 checkpoint"
            )
        facts = _checkpoint_facts(
            selected,
            expected_state_dict_entries=initial_plan["expected_state_dict_entries"],
        )
        archive_directory = exact.parent / "pre_full_resume_checkpoints"
        archived.append(
            archive_conflicting_last_checkpoint(
                exact_checkpoint_path=exact,
                selected_checkpoint_path=selected,
                selected_checkpoint_facts=facts,
                archive_directory=archive_directory,
            )
        )
        selected = _runtime_selected_checkpoint(exact.parent)

    plan = validate_exact_resume_evidence(
        variant=arguments.variant,
        authorization_path=arguments.authorization,
        candidate_path=arguments.candidate,
        decision_path=arguments.decision,
        checkpoint_manifest_path=arguments.checkpoint_manifest,
        exact_checkpoint_path=exact,
        selected_resume_checkpoint=selected,
    )
    plan["archived_conflicts"] = archived
    plan.pop("content_sha256")
    plan["content_sha256"] = canonical_sha256(plan)
    _publish(arguments.output, _json_bytes(plan))
    print(
        json.dumps(
            {
                "archived_conflict_count": len(archived),
                "content_sha256": plan["content_sha256"],
                "selected_checkpoint": plan["exact_checkpoint_name"],
                "status": "pass",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
