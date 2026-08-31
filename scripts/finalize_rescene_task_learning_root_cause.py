#!/usr/bin/env python3
"""Assemble the final ReScene task-learning evidence package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.verify_rescene_rootcause_handoff import (
    MANIFEST_NAME,
    build_final_manifest,
    verify_final_artifacts,
)
from utils.rescene_rootcause_preflight import canonical_sha256

STRONG_VARIANTS = ("A1", "A2", "A1+A2")
SHORT_FILES = (
    "learning_curves.csv",
    "official_like_epoch60.csv",
    "official_like_epoch90.csv",
    "rootcause_per_seed.csv",
    "rootcause_summary.csv",
    "ROOTCAUSE_SHORT_DECISION.json",
    "ROOTCAUSE_SHORT_DECISION.md",
    "ROOTCAUSE_SHORT_PROVENANCE.json",
)
DIAGNOSTIC_FILES = (
    "query_initialization.csv",
    "query_conflicts.csv",
    "attention_mask_recall.csv",
    "superpoint_features.csv",
    "DECODER_DIAGNOSTICS.json",
    "DECODER_DIAGNOSTICS.md",
)
FULL_TRAINING_FILES = (
    "FULL_TRAINING_REPORT.md",
    "FULL_TRAINING_MANIFEST.json",
    "learning_curve.csv",
    "checkpoint_inventory.csv",
    "selected_checkpoint_manifest.json",
)
FULL_EVALUATION_FILES = (
    "official_like_per_seed.csv",
    "official_like_summary.csv",
    "ROOT_CAUSE_FULL_VERDICT.json",
    "ROOT_CAUSE_FULL_VERDICT.md",
    "FULL_EVALUATION_PROVENANCE.json",
)
STRONG_FULL_EVALUATION_FILES = (
    "official_like_per_seed.csv",
    "official_like_summary.csv",
    "STRONG_LOCAL_FULL_VERDICT.json",
    "STRONG_LOCAL_FULL_VERDICT.md",
    "STRONG_LOCAL_FULL_PROVENANCE.json",
)


class FinalizationError(RuntimeError):
    """Raised when final-stage evidence is incomplete or contradictory."""


def classify_principal_outcome(
    *,
    short_decision: Mapping[str, Any],
    full_verdict: Mapping[str, Any] | None,
    strong_verdicts: Sequence[Mapping[str, Any]],
) -> str:
    """Map completed root-cause and structural evidence to the TLRC outcome."""

    selected = short_decision.get("selected_variant")
    full_label = full_verdict.get("verdict") if full_verdict is not None else None
    if selected is not None and full_verdict is None:
        raise FinalizationError("authorized full candidate result is missing")
    if selected is None and full_verdict is not None:
        raise FinalizationError("full candidate exists without short authorization")
    if full_label == "ROOTCAUSE-CONFIRMED":
        return "TLRC-GREEN"
    if full_label not in {None, "ROOTCAUSE-PARTIAL", "ROOTCAUSE-NOT-CONFIRMED"}:
        raise FinalizationError("root-cause full verdict is invalid")

    structural_gain = any(
        verdict.get("status") == "pass" and verdict.get("all_gates_pass") is True
        for verdict in strong_verdicts
    )
    if full_label == "ROOTCAUSE-PARTIAL" or structural_gain:
        return "TLRC-YELLOW"
    return "TLRC-RED"


@dataclass(frozen=True)
class StrongStudy:
    """Paths for one independently finalized ReScene-Strong curve."""

    variant: str
    authorization_path: Path
    output_directory: Path
    full_directory: Path | None = None


@dataclass(frozen=True)
class StrongSkip:
    """One signed structural-variant gate that prevented a training run."""

    variant: str
    status_path: Path


@dataclass(frozen=True)
class FinalPackageInputs:
    """Explicit evidence roots needed to publish the final portable package."""

    artifact_root: Path
    short_directory: Path
    diagnostics_directory: Path
    strong_studies: tuple[StrongStudy, ...]
    final_report_path: Path
    handoff_path: Path
    repository: Mapping[str, object]
    external_files: tuple[Mapping[str, object], ...]
    full_training_directory: Path | None = None
    full_evaluation_directory: Path | None = None
    strong_skips: tuple[StrongSkip, ...] = ()


def _load_json(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FinalizationError(f"{name} is not readable JSON") from error
    if not isinstance(value, dict):
        raise FinalizationError(f"{name} must contain an object")
    return value


def _validate_signed(payload: Mapping[str, Any], *, field: str, name: str) -> None:
    expected = payload.get(field)
    unsigned = dict(payload)
    unsigned.pop(field, None)
    if not isinstance(expected, str) or canonical_sha256(unsigned) != expected:
        raise FinalizationError(f"{name} content hash differs")


def _file_identity(path: Path) -> dict[str, object]:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise FinalizationError("strong-local evidence is unavailable") from error
    return {
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _read_csv(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    try:
        with path.open(encoding="ascii", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = tuple(reader.fieldnames or ())
            rows = [dict(row) for row in reader]
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise FinalizationError("strong-local CSV is unreadable") from error
    if not fields or not rows:
        raise FinalizationError("strong-local CSV is empty")
    return fields, rows


def _merge_csv(sources: Sequence[Path], *, key_fields: tuple[str, ...]) -> bytes:
    expected_fields: tuple[str, ...] | None = None
    merged: dict[tuple[str, ...], dict[str, str]] = {}
    for source in sources:
        fields, rows = _read_csv(source)
        if expected_fields is None:
            expected_fields = fields
        if fields != expected_fields or any(
            field not in fields for field in key_fields
        ):
            raise FinalizationError("strong-local CSV schema differs")
        for row in rows:
            key = tuple(row[field] for field in key_fields)
            previous = merged.get(key)
            if previous is not None and previous != row:
                raise FinalizationError("strong-local CSV has conflicting duplicate")
            if previous is None:
                merged[key] = row
    if expected_fields is None:
        raise FinalizationError("strong-local CSV source list is empty")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=expected_fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(merged.values())
    return stream.getvalue().encode("ascii")


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True)
        + "\n"
    ).encode("ascii")


def _required_bytes(path: Path) -> bytes:
    try:
        if path.is_symlink() or not path.is_file():
            raise OSError
        return path.read_bytes()
    except OSError as error:
        raise FinalizationError(
            f"finalization input is unavailable: {path.name}"
        ) from error


def _stage_payload(root: Path, relative: str, payload: bytes) -> None:
    destination = root / relative
    if destination.exists() or destination.is_symlink():
        if (
            destination.is_file()
            and not destination.is_symlink()
            and destination.read_bytes() == payload
        ):
            return
        raise FinalizationError(f"refusing to replace staged artifact: {relative}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)


def _stage_directory_files(
    *, root: Path, source: Path, destination: str, names: Sequence[str]
) -> None:
    for name in names:
        _stage_payload(root, f"{destination}/{name}", _required_bytes(source / name))


def _full_gate_skip(short_decision: Mapping[str, Any]) -> bytes:
    if (
        short_decision.get("selected_variant") is not None
        or short_decision.get("full_training_authorized") is not False
        or short_decision.get("full_training_status") != "gate_skipped"
    ):
        raise FinalizationError("full-candidate skip differs from short decision")
    payload: dict[str, object] = {
        "schema_version": 1,
        "status": "gate_skipped",
        "reason": "no short-curve candidate passed every full-run authorization gate",
        "upstream_gate": "ROOTCAUSE_SHORT_DECISION",
        "short_decision_content_sha256": short_decision["content_sha256"],
    }
    payload["content_sha256"] = canonical_sha256(payload)
    return _json_bytes(payload)


def _publish_atomic(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        if path.is_file() and not path.is_symlink() and path.read_bytes() == payload:
            return
        raise FinalizationError(f"refusing to overwrite final artifact: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
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


def publish_final_package(inputs: FinalPackageInputs) -> dict[str, object]:
    """Assemble, verify, and atomically publish the complete final package."""

    if not inputs.artifact_root.is_dir() or inputs.artifact_root.is_symlink():
        raise FinalizationError("artifact root is unavailable")
    short_decision = _load_json(
        inputs.short_directory / "ROOTCAUSE_SHORT_DECISION.json",
        name="root-cause short decision",
    )
    _validate_signed(
        short_decision,
        field="content_sha256",
        name="root-cause short decision",
    )
    if short_decision.get("status") != "pass":
        raise FinalizationError("root-cause short decision is incomplete")
    short_provenance = _load_json(
        inputs.short_directory / "ROOTCAUSE_SHORT_PROVENANCE.json",
        name="root-cause short provenance",
    )
    _validate_signed(
        short_provenance,
        field="content_sha256",
        name="root-cause short provenance",
    )
    if (
        short_provenance.get("decision_content_sha256")
        != short_decision["content_sha256"]
    ):
        raise FinalizationError("root-cause short provenance binding differs")
    diagnostics = _load_json(
        inputs.diagnostics_directory / "DECODER_DIAGNOSTICS.json",
        name="decoder diagnostics",
    )
    _validate_signed(diagnostics, field="content_sha256", name="decoder diagnostics")
    if diagnostics.get("status") != "pass":
        raise FinalizationError("decoder diagnostics are incomplete")
    considered_strong_variants = {
        *(study.variant for study in inputs.strong_studies),
        *(skip.variant for skip in inputs.strong_skips),
    }
    if not {"A1", "A2"}.issubset(considered_strong_variants):
        raise FinalizationError("strong-local A2 decision is missing")

    full_verdict: Mapping[str, Any] | None = None
    selected = short_decision.get("selected_variant")
    if selected is None:
        if (
            inputs.full_training_directory is not None
            or inputs.full_evaluation_directory is not None
        ):
            raise FinalizationError("full outputs exist without short authorization")
    else:
        if (
            inputs.full_training_directory is None
            or inputs.full_evaluation_directory is None
        ):
            raise FinalizationError("authorized full candidate outputs are missing")
        full_verdict = _load_json(
            inputs.full_evaluation_directory / "ROOT_CAUSE_FULL_VERDICT.json",
            name="root-cause full verdict",
        )
        _validate_signed(
            full_verdict,
            field="content_sha256",
            name="root-cause full verdict",
        )
        full_provenance = _load_json(
            inputs.full_evaluation_directory / "FULL_EVALUATION_PROVENANCE.json",
            name="root-cause full provenance",
        )
        _validate_signed(
            full_provenance,
            field="content_sha256",
            name="root-cause full provenance",
        )
        if (
            full_provenance.get("result_content_sha256")
            != full_verdict["content_sha256"]
        ):
            raise FinalizationError("root-cause full provenance binding differs")
        if (
            full_verdict.get("status") != "pass"
            or full_verdict.get("variant") != selected
            or full_verdict.get("verdict_prefix") != "ROOTCAUSE"
        ):
            raise FinalizationError("root-cause full verdict binding differs")
        selected_manifest = _load_json(
            inputs.full_training_directory / "selected_checkpoint_manifest.json",
            name="selected full checkpoint manifest",
        )
        _validate_signed(
            selected_manifest,
            field="content_sha256",
            name="selected full checkpoint manifest",
        )
        training_manifest = _load_json(
            inputs.full_training_directory / "FULL_TRAINING_MANIFEST.json",
            name="full-training manifest",
        )
        _validate_signed(
            training_manifest,
            field="content_sha256",
            name="full-training manifest",
        )
        checkpoint = selected_manifest.get("checkpoint")
        checkpoint_bindings = selected_manifest.get("bindings")
        selected_full_training = selected_manifest.get("full_training")
        training_selection = training_manifest.get("selection")
        budget = training_manifest.get("budget")
        authorization_sha256 = short_decision.get("variant_authorization_sha256")
        if (
            selected_manifest.get("status") != "pass"
            or selected_manifest.get("stage") != "full_candidate"
            or selected_manifest.get("variant") != selected
            or not isinstance(checkpoint, Mapping)
            or not isinstance(checkpoint_bindings, Mapping)
            or checkpoint_bindings.get("variant_authorization_sha256")
            != authorization_sha256
            or checkpoint.get("sha256")
            != full_verdict.get("selected_checkpoint_sha256")
            or selected_manifest.get("content_sha256")
            != full_verdict.get("checkpoint_manifest_sha256")
        ):
            raise FinalizationError("full checkpoint binding differs")
        if (
            training_manifest.get("status") != "pass"
            or training_manifest.get("variant") != selected
            or training_manifest.get("variant_authorization_sha256")
            != authorization_sha256
            or not isinstance(training_selection, Mapping)
            or not isinstance(budget, Mapping)
            or budget.get("completed_epoch") != 450
            or budget.get("optimizer_steps") != 29_700
            or training_selection.get("selected_checkpoint_sha256")
            != checkpoint.get("sha256")
            or not isinstance(selected_full_training, Mapping)
            or selected_full_training.get("completed_epoch") != 450
            or selected_full_training.get("manifest_sha256")
            != training_manifest.get("content_sha256")
            or full_verdict.get("full_training_manifest_sha256")
            != training_manifest.get("content_sha256")
        ):
            raise FinalizationError("full training manifest binding differs")

    strong_verdicts = [
        _load_json(
            study.output_directory / "STRONG_LOCAL_VERDICT.json",
            name=f"{study.variant} verdict",
        )
        for study in inputs.strong_studies
    ]
    if any(
        verdict.get("all_gates_pass") is True and study.full_directory is None
        for study, verdict in zip(inputs.strong_studies, strong_verdicts, strict=True)
    ):
        raise FinalizationError("authorized strong-local full result is missing")
    expected_outcome = classify_principal_outcome(
        short_decision=short_decision,
        full_verdict=full_verdict,
        strong_verdicts=strong_verdicts,
    )
    final_report = _required_bytes(inputs.final_report_path)
    if (
        final_report.count(f"Principal outcome: `{expected_outcome}`".encode("ascii"))
        != 1
    ):
        raise FinalizationError("final report outcome differs from evidence")
    handoff = _required_bytes(inputs.handoff_path)
    strong_outputs = aggregate_strong_outputs(
        inputs.strong_studies, inputs.strong_skips
    )

    inputs.artifact_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=inputs.artifact_root.parent, prefix=".rescene-final."
    ) as temporary_directory:
        staged_root = Path(temporary_directory) / inputs.artifact_root.name
        shutil.copytree(inputs.artifact_root, staged_root, symlinks=True)
        manifest_path = staged_root / MANIFEST_NAME
        manifest_path.unlink(missing_ok=True)

        _stage_directory_files(
            root=staged_root,
            source=inputs.short_directory,
            destination="short_curves",
            names=SHORT_FILES,
        )
        _stage_directory_files(
            root=staged_root,
            source=inputs.diagnostics_directory,
            destination="decoder_diagnostics",
            names=DIAGNOSTIC_FILES,
        )
        if selected is None:
            _stage_payload(
                staged_root,
                "full_candidate/STATUS.json",
                _full_gate_skip(short_decision),
            )
        else:
            assert inputs.full_training_directory is not None
            assert inputs.full_evaluation_directory is not None
            _stage_directory_files(
                root=staged_root,
                source=inputs.full_training_directory,
                destination="full_candidate",
                names=FULL_TRAINING_FILES,
            )
            _stage_directory_files(
                root=staged_root,
                source=inputs.full_evaluation_directory,
                destination="full_candidate",
                names=FULL_EVALUATION_FILES,
            )
        for name, payload in strong_outputs.items():
            _stage_payload(staged_root, f"strong_local/{name}", payload)
        _stage_payload(staged_root, "FINAL_REPORT.md", final_report)
        _stage_payload(staged_root, "HANDOFF.md", handoff)

        manifest = build_final_manifest(
            artifact_root=staged_root,
            repository=inputs.repository,
            external_files=inputs.external_files,
        )
        _stage_payload(staged_root, MANIFEST_NAME, _json_bytes(manifest))
        expected_result = verify_final_artifacts(staged_root)

        publications = [
            (path.relative_to(staged_root), path.read_bytes())
            for path in staged_root.rglob("*")
            if path.is_file() and not path.is_symlink()
        ]
        publications.sort(key=lambda item: item[0].as_posix() == MANIFEST_NAME)
        for relative, payload in publications:
            target = inputs.artifact_root / relative
            if (target.exists() or target.is_symlink()) and (
                not target.is_file()
                or target.is_symlink()
                or target.read_bytes() != payload
            ):
                raise FinalizationError(
                    f"existing final artifact differs: {relative.as_posix()}"
                )
        for relative, payload in publications:
            _publish_atomic(inputs.artifact_root / relative, payload)

    observed = verify_final_artifacts(inputs.artifact_root)
    if observed != expected_result:
        raise FinalizationError("published final package verification differs")
    return observed


def _spec_path(spec: Mapping[str, Any], name: str) -> Path:
    value = spec.get(name)
    if not isinstance(value, str) or not value:
        raise FinalizationError(f"finalization spec field is invalid: {name}")
    return Path(value)


def load_finalization_inputs(spec_path: Path) -> FinalPackageInputs:
    """Load the external, machine-local finalization path specification."""

    spec = _load_json(spec_path, name="finalization spec")
    required = {
        "schema_version",
        "artifact_root",
        "short_directory",
        "diagnostics_directory",
        "strong_studies",
        "final_report_path",
        "handoff_path",
        "repository",
        "external_files",
    }
    allowed = required | {
        "full_training_directory",
        "full_evaluation_directory",
        "strong_skips",
    }
    if (
        spec.get("schema_version") != 1
        or set(spec) - allowed
        or not required.issubset(spec)
    ):
        raise FinalizationError("finalization spec schema differs")
    raw_studies = spec["strong_studies"]
    if not isinstance(raw_studies, list) or not raw_studies:
        raise FinalizationError("finalization spec strong studies differ")
    studies = []
    for value in raw_studies:
        required_study_fields = {
            "variant",
            "authorization_path",
            "output_directory",
        }
        allowed_study_fields = required_study_fields | {"full_directory"}
        if (
            not isinstance(value, Mapping)
            or not required_study_fields.issubset(value)
            or set(value) - allowed_study_fields
        ):
            raise FinalizationError("finalization spec strong study is invalid")
        variant = value.get("variant")
        authorization_path = value.get("authorization_path")
        output_directory = value.get("output_directory")
        full_directory = value.get("full_directory")
        if (
            not isinstance(variant, str)
            or not isinstance(authorization_path, str)
            or not authorization_path
            or not isinstance(output_directory, str)
            or not output_directory
            or full_directory is not None
            and (not isinstance(full_directory, str) or not full_directory)
        ):
            raise FinalizationError("finalization spec strong study is invalid")
        studies.append(
            StrongStudy(
                variant=variant,
                authorization_path=Path(authorization_path),
                output_directory=Path(output_directory),
                full_directory=(
                    Path(full_directory) if full_directory is not None else None
                ),
            )
        )
    repository = spec["repository"]
    external_files = spec["external_files"]
    if not isinstance(repository, Mapping) or not isinstance(external_files, list):
        raise FinalizationError("finalization spec evidence inventory differs")
    raw_skips = spec.get("strong_skips", [])
    if not isinstance(raw_skips, list):
        raise FinalizationError("finalization spec strong skips differ")
    skips = []
    for value in raw_skips:
        if not isinstance(value, Mapping) or set(value) != {
            "variant",
            "status_path",
        }:
            raise FinalizationError("finalization spec strong skip is invalid")
        variant = value.get("variant")
        status_path = value.get("status_path")
        if (
            not isinstance(variant, str)
            or not isinstance(status_path, str)
            or not status_path
        ):
            raise FinalizationError("finalization spec strong skip is invalid")
        skips.append(StrongSkip(variant=variant, status_path=Path(status_path)))

    def optional_path(name: str) -> Path | None:
        value = spec.get(name)
        if value is None:
            return None
        if not isinstance(value, str) or not value:
            raise FinalizationError(f"finalization spec field is invalid: {name}")
        return Path(value)

    return FinalPackageInputs(
        artifact_root=_spec_path(spec, "artifact_root"),
        short_directory=_spec_path(spec, "short_directory"),
        diagnostics_directory=_spec_path(spec, "diagnostics_directory"),
        strong_studies=tuple(studies),
        final_report_path=_spec_path(spec, "final_report_path"),
        handoff_path=_spec_path(spec, "handoff_path"),
        repository=dict(repository),
        external_files=tuple(
            dict(record) if isinstance(record, Mapping) else record
            for record in external_files
        ),
        full_training_directory=optional_path("full_training_directory"),
        full_evaluation_directory=optional_path("full_evaluation_directory"),
        strong_skips=tuple(skips),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    arguments = parser.parse_args(argv)
    result = publish_final_package(load_finalization_inputs(arguments.spec))
    print(json.dumps(result, sort_keys=True))
    return 0


def _strong_full_outputs(
    *,
    study: StrongStudy,
    authorization: Mapping[str, Any],
    short_verdict: Mapping[str, Any],
) -> tuple[dict[str, object], dict[str, bytes]] | None:
    directory = study.full_directory
    if directory is None:
        if short_verdict.get("all_gates_pass") is True:
            return None
        status: dict[str, object] = {
            "schema_version": 1,
            "status": "gate_skipped",
            "experiment": "rescene_strong_local_v1",
            "variant": study.variant,
            "reason": "short spatial gate did not authorize full training",
            "upstream_gate": "STRONG_LOCAL_VERDICT",
            "short_verdict_content_sha256": short_verdict["content_sha256"],
        }
        status["content_sha256"] = canonical_sha256(status)
        return (
            {
                "variant": study.variant,
                "status": "gate_skipped",
                "content_sha256": status["content_sha256"],
            },
            {f"variants/{study.variant}/full/STATUS.json": _json_bytes(status)},
        )
    if short_verdict.get("all_gates_pass") is not True:
        raise FinalizationError(
            "strong-local full result exists without short authorization"
        )

    training = _load_json(
        directory / "FULL_TRAINING_MANIFEST.json",
        name=f"{study.variant} full-training manifest",
    )
    checkpoint = _load_json(
        directory / "selected_checkpoint_manifest.json",
        name=f"{study.variant} full checkpoint manifest",
    )
    verdict = _load_json(
        directory / "STRONG_LOCAL_FULL_VERDICT.json",
        name=f"{study.variant} full verdict",
    )
    provenance = _load_json(
        directory / "STRONG_LOCAL_FULL_PROVENANCE.json",
        name=f"{study.variant} full provenance",
    )
    for payload, name in (
        (training, "full-training manifest"),
        (checkpoint, "full checkpoint manifest"),
        (verdict, "full verdict"),
        (provenance, "full provenance"),
    ):
        _validate_signed(
            payload,
            field="content_sha256",
            name=f"{study.variant} {name}",
        )

    checkpoint_record = checkpoint.get("checkpoint")
    checkpoint_bindings = checkpoint.get("bindings")
    checkpoint_training = checkpoint.get("full_training")
    training_budget = training.get("budget")
    training_selection = training.get("selection")
    authorization_sha256 = authorization["authorization_sha256"]
    if (
        checkpoint.get("status") != "pass"
        or checkpoint.get("stage") != "full_candidate"
        or checkpoint.get("variant") != study.variant
        or not isinstance(checkpoint_record, Mapping)
        or not isinstance(checkpoint_bindings, Mapping)
        or checkpoint_bindings.get("variant_authorization_sha256")
        != authorization_sha256
        or checkpoint_record.get("sha256") != verdict.get("selected_checkpoint_sha256")
        or checkpoint.get("content_sha256") != verdict.get("checkpoint_manifest_sha256")
    ):
        raise FinalizationError("strong-local full checkpoint binding differs")
    if (
        training.get("status") != "pass"
        or training.get("variant") != study.variant
        or training.get("variant_authorization_sha256") != authorization_sha256
        or not isinstance(training_budget, Mapping)
        or training_budget.get("completed_epoch") != 450
        or training_budget.get("optimizer_steps") != 29_700
        or not isinstance(training_selection, Mapping)
        or training_selection.get("selected_checkpoint_sha256")
        != checkpoint_record.get("sha256")
        or not isinstance(checkpoint_training, Mapping)
        or checkpoint_training.get("completed_epoch") != 450
        or checkpoint_training.get("manifest_sha256") != training.get("content_sha256")
        or verdict.get("full_training_manifest_sha256")
        != training.get("content_sha256")
    ):
        raise FinalizationError("strong-local full training manifest binding differs")
    if (
        verdict.get("status") != "pass"
        or verdict.get("variant") != study.variant
        or verdict.get("verdict_prefix") != "STRONG-LOCAL"
        or verdict.get("verdict")
        not in {
            "STRONG-LOCAL-CONFIRMED",
            "STRONG-LOCAL-PARTIAL",
            "STRONG-LOCAL-NOT-CONFIRMED",
        }
    ):
        raise FinalizationError("strong-local full verdict binding differs")
    if provenance.get("result_content_sha256") != verdict["content_sha256"]:
        raise FinalizationError("strong-local full provenance binding differs")

    prefix = f"variants/{study.variant}/full"
    outputs = {
        f"{prefix}/{name}": _required_bytes(directory / name)
        for name in (*FULL_TRAINING_FILES, *STRONG_FULL_EVALUATION_FILES)
    }
    result = {
        "variant": study.variant,
        "status": "pass",
        "content_sha256": verdict["content_sha256"],
        "verdict": verdict["verdict"],
    }
    return result, outputs


def aggregate_strong_outputs(
    studies: Sequence[StrongStudy],
    skips: Sequence[StrongSkip] = (),
) -> dict[str, bytes]:
    """Merge independently signed strong-local outputs without duplicate baselines."""

    variants = [study.variant for study in studies]
    skipped_variants = [skip.variant for skip in skips]
    considered_variants = variants + skipped_variants
    if (
        not variants
        or len(variants) != len(set(variants))
        or any(variant not in STRONG_VARIANTS for variant in variants)
        or variants != sorted(variants, key=STRONG_VARIANTS.index)
        or variants[0] != "A1"
        or len(considered_variants) != len(set(considered_variants))
        or any(variant not in STRONG_VARIANTS for variant in skipped_variants)
        or considered_variants != sorted(considered_variants, key=STRONG_VARIANTS.index)
    ):
        raise FinalizationError("strong-local studies differ from registered order")

    authorizations = []
    verdicts = []
    full_results = []
    full_statuses = []
    curve_paths = []
    per_seed_paths = []
    source_outputs: dict[str, bytes] = {}
    for study in studies:
        authorization = _load_json(
            study.authorization_path, name=f"{study.variant} authorization"
        )
        _validate_signed(
            authorization,
            field="authorization_sha256",
            name=f"{study.variant} authorization",
        )
        if authorization.get("status") != "authorized" or authorization.get(
            "selected_variants"
        ) != [study.variant]:
            raise FinalizationError("strong-local authorization differs")
        verdict_path = study.output_directory / "STRONG_LOCAL_VERDICT.json"
        verdict = _load_json(verdict_path, name=f"{study.variant} verdict")
        _validate_signed(
            verdict, field="content_sha256", name=f"{study.variant} verdict"
        )
        if (
            verdict.get("status") != "pass"
            or verdict.get("variant") != study.variant
            or verdict.get("variant_authorization_sha256")
            != authorization["authorization_sha256"]
            or verdict.get("selection_used_persist4d") is not False
        ):
            raise FinalizationError("strong-local verdict binding differs")
        provenance_path = study.output_directory / "STRONG_LOCAL_PROVENANCE.json"
        provenance = _load_json(provenance_path, name=f"{study.variant} provenance")
        _validate_signed(
            provenance,
            field="content_sha256",
            name=f"{study.variant} provenance",
        )
        if provenance.get("decision_content_sha256") != verdict["content_sha256"]:
            raise FinalizationError("strong-local provenance binding differs")
        authorizations.append(
            {
                "variant": study.variant,
                "authorization_sha256": authorization["authorization_sha256"],
                "file": _file_identity(study.authorization_path),
                "verdict_file": _file_identity(verdict_path),
                "provenance_file": _file_identity(provenance_path),
            }
        )
        verdicts.append(verdict)
        curve_paths.append(study.output_directory / "learning_curves.csv")
        per_seed_paths.append(study.output_directory / "official_like_per_seed.csv")
        prefix = f"variants/{study.variant}"
        source_outputs[f"{prefix}/variant_manifest.json"] = _required_bytes(
            study.authorization_path
        )
        source_outputs[f"{prefix}/STRONG_LOCAL_VERDICT.json"] = _required_bytes(
            verdict_path
        )
        source_outputs[f"{prefix}/STRONG_LOCAL_PROVENANCE.json"] = _required_bytes(
            provenance_path
        )
        full_result = _strong_full_outputs(
            study=study,
            authorization=authorization,
            short_verdict=verdict,
        )
        if full_result is not None:
            result, outputs = full_result
            if result["status"] == "pass":
                full_results.append(result)
            else:
                full_statuses.append(result)
            source_outputs.update(outputs)

    skip_records = []
    for skip in skips:
        payload = _load_json(skip.status_path, name=f"{skip.variant} skip")
        _validate_signed(
            payload,
            field="content_sha256",
            name=f"{skip.variant} skip",
        )
        gate = payload.get("gate")
        evidence = payload.get("upstream_evidence")
        if (
            payload.get("status") != "gate_skipped"
            or payload.get("experiment") != "rescene_strong_local_v1"
            or payload.get("variant") != skip.variant
            or not isinstance(gate, Mapping)
            or gate.get("status") != "gate_skipped"
            or gate.get("authorized") is not False
            or not isinstance(gate.get("reason"), str)
            or not gate["reason"]
            or not isinstance(evidence, Mapping)
            or not evidence
        ):
            raise FinalizationError("strong-local skipped variant binding differs")
        skip_records.append(
            {
                "variant": skip.variant,
                "content_sha256": payload["content_sha256"],
                "reason": gate["reason"],
                "file": _file_identity(skip.status_path),
            }
        )
        source_outputs[f"variants/{skip.variant}/variant_manifest.json"] = (
            _required_bytes(skip.status_path)
        )

    manifest: dict[str, object] = {
        "schema_version": 1,
        "status": "pass",
        "experiment": "rescene_strong_local_v1",
        "authorizations": authorizations,
        "skips": skip_records,
    }
    manifest["content_sha256"] = canonical_sha256(manifest)
    aggregate: dict[str, object] = {
        "schema_version": 1,
        "status": "pass",
        "experiment": "rescene_strong_local_v1",
        "variants_run": variants,
        "variants_considered": considered_variants,
        "skipped_variants": [
            {
                "variant": record["variant"],
                "content_sha256": record["content_sha256"],
                "reason": record["reason"],
            }
            for record in skip_records
        ],
        "full_training_authorized_variants": [
            verdict["variant"]
            for verdict in verdicts
            if verdict.get("all_gates_pass") is True
        ],
        "selection_used_persist4d": False,
        "full_results": full_results,
        "full_statuses": full_statuses,
        "results": [
            {
                "variant": verdict["variant"],
                "content_sha256": verdict["content_sha256"],
                "all_gates_pass": verdict.get("all_gates_pass") is True,
                "paired_spatial_delta_mean": verdict.get("paired_spatial_delta_mean"),
            }
            for verdict in verdicts
        ],
    }
    aggregate["content_sha256"] = canonical_sha256(aggregate)
    report_lines = [
        "# ReScene-Strong Verdict",
        "",
        *[
            (
                f"- `{verdict['variant']}`: full gate "
                f"`{'PASS' if verdict.get('all_gates_pass') is True else 'FAIL'}`, "
                "paired SpatialStageMean delta "
                f"`{verdict.get('paired_spatial_delta_mean')}`"
            )
            for verdict in verdicts
        ],
        *[
            (f"- `{result['variant']}` full: `{result['verdict']}`")
            for result in full_results
        ],
        *[
            f"- `{record['variant']}`: `SKIPPED` ({record['reason']})"
            for record in skip_records
        ],
        "",
        "Persist4D metrics were not used for selection.",
        "",
    ]
    return {
        "variant_manifest.json": _json_bytes(manifest),
        "learning_curves.csv": _merge_csv(
            curve_paths, key_fields=("variant", "completed_epoch")
        ),
        "official_like_per_seed.csv": _merge_csv(
            per_seed_paths,
            key_fields=("variant", "completed_epoch", "seed"),
        ),
        "STRONG_LOCAL_VERDICT.json": _json_bytes(aggregate),
        "STRONG_LOCAL_VERDICT.md": "\n".join(report_lines).encode("ascii"),
        **source_outputs,
    }


if __name__ == "__main__":
    raise SystemExit(main())
