"""Build and verify the lightweight reviewer-closure evidence package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts/reviewer_closure"
DEFAULT_OUTPUT = DEFAULT_ARTIFACT_ROOT / "final_evidence_manifest.json"

FINAL_CLASSIFICATIONS = (
    "FINAL_LOCK",
    "FINAL_PARETO_LOCK",
    "ASSOCIATION_REOPEN",
    "PERCEPTION_REOPEN",
)
FINAL_REPORT_SECTIONS = (
    "1. Frozen method definition",
    "2. Strongest trivial alternative",
    "3. Temporal-horizon adaptation challenge",
    "4. Why t-mAP is near-parity",
    "5. Oracle headroom",
    "6. External geometry matcher",
    "7. Compute and memory Pareto result",
    "8. Statistical robustness",
    "9. Claims supported",
    "10. Claims not supported",
    "11. Final classification",
    "12. Paper-ready next action",
)
FIGURE_STEMS = (
    "iou_threshold_curve",
    "observation_coverage",
    "failure_decomposition",
    "strong_baseline_identity_scaling",
    "horizon_adaptation_task_scaling",
    "oracle_association_gain",
    "performance_decomposition",
)
REQUIRED_ARTIFACTS = (
    "REVIEWER_CLOSURE_SUMMARY.md",
    "FINAL_METHOD_LOCK_REPORT.md",
    "FULL_HISTORY_TRACKER_AUDIT.md",
    "full_history_tracker_results.csv",
    "full_history_tracker_cluster_bootstrap.csv",
    "full_history_tracker_loso.csv",
    "full_history_tracker_order_robustness.csv",
    "gate_i.json",
    "REScene_HORIZON_TRAINING_AUDIT.md",
    "t3_smoke_report.json",
    "rescene_horizon_training_manifest.json",
    "rescene_horizon_adaptation_results.csv",
    "rescene_horizon_adaptation_profile_results.csv",
    "rescene_horizon_compute.csv",
    "rescene_horizon_adaptation_evaluation_manifest.json",
    "rescene_horizon_profile_manifest.json",
    "rescene_horizon_gate_ii.json",
    "tmap_iou_sweep.csv",
    "observation_coverage.csv",
    "oracle_association_results.csv",
    "failure_decomposition.csv",
    "phase_iii_manifest.json",
    "METRIC_AGGREGATION_NOTE.md",
    *(
        f"figures/{stem}.{suffix}"
        for stem in FIGURE_STEMS
        for suffix in ("svg", "pdf", "png")
    ),
)
CONDITIONAL_LIVINGSCENES_ARTIFACTS = (
    "LIVINGSCENES_CODE_COMPATIBILITY_AUDIT.md",
    "living_scenes_supported_subset_manifest.json",
    "living_scenes_adapted_results.csv",
)


class FinalEvidenceError(ValueError):
    """Raised when final reviewer-closure evidence is incomplete or inconsistent."""


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


def content_sha256(value: Mapping[str, object]) -> str:
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


def _pretty_content_sha256(value: Mapping[str, object]) -> str:
    payload = dict(value)
    payload.pop("content_sha256", None)
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path, *, name: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FinalEvidenceError(f"{name} is unavailable")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FinalEvidenceError(f"cannot decode {name}") from error
    if not isinstance(value, Mapping):
        raise FinalEvidenceError(f"{name} must be a mapping")
    return dict(value)


def _validate_content_manifest(path: Path, *, name: str) -> dict[str, Any]:
    manifest = _load_json(path, name=name)
    if manifest.get("status") != "pass":
        raise FinalEvidenceError(f"{name} status differs")
    digest = manifest.get("content_sha256")
    if not isinstance(digest, str) or digest != content_sha256(manifest):
        raise FinalEvidenceError(f"{name} content binding differs")
    return manifest


def _validate_pretty_content_manifest(path: Path, *, name: str) -> dict[str, Any]:
    manifest = _load_json(path, name=name)
    if manifest.get("status") != "pass":
        raise FinalEvidenceError(f"{name} status differs")
    digest = manifest.get("content_sha256")
    if not isinstance(digest, str) or digest != _pretty_content_sha256(manifest):
        raise FinalEvidenceError(f"{name} content binding differs")
    return manifest


def _git_tree(repo_root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD:artifacts/system_comparison"],
            cwd=repo_root,
            capture_output=True,
            check=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise FinalEvidenceError("cannot resolve system-comparison tree") from error
    tree = completed.stdout.strip()
    if re.fullmatch(r"[0-9a-f]{40}", tree) is None:
        raise FinalEvidenceError("system-comparison tree is invalid")
    return tree


def _csv_row_count(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.reader(handle))
    except (OSError, UnicodeError, csv.Error) as error:
        raise FinalEvidenceError(f"cannot decode CSV: {path.name}") from error
    if len(rows) < 2 or not rows[0]:
        raise FinalEvidenceError(f"CSV is empty: {path.name}")
    return len(rows) - 1


def validate_final_report(report: str) -> str:
    if not isinstance(report, str) or not report.strip():
        raise FinalEvidenceError("final report is empty")
    sections = tuple(
        line.removeprefix("## ").strip()
        for line in report.splitlines()
        if line.startswith("## ")
    )
    if sections != FINAL_REPORT_SECTIONS:
        raise FinalEvidenceError("final report must contain exactly 12 sections")
    classifications = re.findall(
        r"`(" + "|".join(FINAL_CLASSIFICATIONS) + r")`",
        report,
    )
    if len(classifications) != 1:
        raise FinalEvidenceError("final report must contain exactly one classification")
    return classifications[0]


def validate_t3_smoke_evidence(smoke: Mapping[str, object]) -> None:
    if (
        smoke.get("status") != "pass"
        or smoke.get("formal_training_started") is not False
    ):
        raise FinalEvidenceError("T3 smoke status differs")
    if smoke.get("recipe_changed") is not False:
        raise FinalEvidenceError("T3 smoke recipe changed")
    sample = smoke.get("sample")
    if not isinstance(sample, Mapping) or sample.get("temporal_stages") != [0, 1, 2]:
        raise FinalEvidenceError("T3 smoke stages differ")

    losses = smoke.get("losses")
    if (
        not isinstance(losses, Mapping)
        or not losses
        or not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in losses.values()
        )
    ):
        raise FinalEvidenceError("T3 smoke requires finite losses")
    objective = smoke.get("weighted_objective")
    if (
        not isinstance(objective, (int, float))
        or isinstance(objective, bool)
        or not math.isfinite(float(objective))
    ):
        raise FinalEvidenceError("T3 smoke objective is not finite")

    gradients = smoke.get("gradients")
    trainable = (
        gradients.get("trainable_parameters")
        if isinstance(gradients, Mapping)
        else None
    )
    frozen = gradients.get("frozen_encoder") if isinstance(gradients, Mapping) else None
    if (
        not isinstance(trainable, Mapping)
        or trainable.get("finite") is not True
        or not isinstance(trainable.get("nonzero_grad_tensors"), int)
        or trainable["nonzero_grad_tensors"] <= 0
        or trainable.get("missing_grad_tensors") != 0
    ):
        raise FinalEvidenceError("T3 smoke trainable gradients differ")
    if (
        not isinstance(frozen, Mapping)
        or frozen.get("finite") is not True
        or frozen.get("nonzero_grad_tensors") != 0
        or frozen.get("max_grad_norm") != 0.0
    ):
        raise FinalEvidenceError("T3 smoke frozen gradients differ")

    checkpoint_reload = smoke.get("checkpoint_reload")
    if (
        not isinstance(checkpoint_reload, Mapping)
        or checkpoint_reload.get("strict") is not True
        or not isinstance(checkpoint_reload.get("state_dict_entry_count"), int)
        or checkpoint_reload["state_dict_entry_count"] <= 0
    ):
        raise FinalEvidenceError("T3 smoke checkpoint reload differs")
    if (
        smoke.get("fresh_optimizer_state_before_step") is not True
        or smoke.get("fresh_scheduler_last_epoch_after_step") != 1
    ):
        raise FinalEvidenceError("T3 smoke optimizer step differs")

    source_checkpoint = smoke.get("source_checkpoint")
    if (
        not isinstance(source_checkpoint, Mapping)
        or source_checkpoint.get("optimizer_state_resumed") is not False
        or source_checkpoint.get("scheduler_state_resumed") is not False
    ):
        raise FinalEvidenceError("T3 smoke initialization differs")
    digest = smoke.get("content_sha256")
    if not isinstance(digest, str) or digest != content_sha256(smoke):
        raise FinalEvidenceError("T3 smoke content binding differs")


def validate_gate_classifications(
    gate_i: Mapping[str, object],
    gate_ii: Mapping[str, object],
    phase_iii: Mapping[str, object],
) -> tuple[str, str, str]:
    manifests = (("Gate I", gate_i), ("Gate II", gate_ii), ("Phase III", phase_iii))
    for name, manifest in manifests:
        if manifest.get("status") != "pass":
            raise FinalEvidenceError(f"{name} status differs")

    phase_i_classification = gate_i.get("classification")
    if phase_i_classification not in {
        "TRACKER_REJECTED",
        "TRACKER_EXPLAINS_IDENTITY",
    }:
        raise FinalEvidenceError("Gate-I classification differs")
    phase_ii_classification = gate_ii.get("classification")
    if phase_ii_classification not in {
        "HORIZON_ROBUST",
        "FULL_HISTORY_DOMINANT",
        "ACCURACY_ADVANTAGE_BUT_COSTLY",
    }:
        raise FinalEvidenceError("Gate-II classification differs")
    phase_iii_classification = phase_iii.get("ceiling_classification")
    if phase_iii_classification not in {
        "ASSOCIATION_CEILING",
        "PERCEPTION_CEILING",
    }:
        raise FinalEvidenceError("Phase-III classification differs")
    return (
        phase_i_classification,
        phase_ii_classification,
        phase_iii_classification,
    )


def _validate_required_artifacts(artifact_root: Path) -> None:
    for relative in REQUIRED_ARTIFACTS:
        path = artifact_root / relative
        if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
            raise FinalEvidenceError(f"required artifact is unavailable: {relative}")


def _validate_evaluation_artifacts(
    artifact_root: Path,
    evaluation: Mapping[str, object],
) -> None:
    artifacts = evaluation.get("artifacts")
    if not isinstance(artifacts, Mapping) or not artifacts:
        raise FinalEvidenceError("adaptation evaluation artifact bindings are missing")
    for filename, record in artifacts.items():
        if not isinstance(filename, str) or not isinstance(record, Mapping):
            raise FinalEvidenceError(
                "adaptation evaluation artifact binding is invalid"
            )
        path = artifact_root / filename
        if (
            path.is_symlink()
            or not path.is_file()
            or record.get("sha256") != _sha256_file(path)
            or record.get("row_count") != _csv_row_count(path)
        ):
            raise FinalEvidenceError(
                f"adaptation evaluation artifact differs: {filename}"
            )


def _validate_profile_artifacts(
    artifact_root: Path,
    profile: Mapping[str, object],
) -> None:
    bindings = (
        (
            "rescene_horizon_adaptation_profile_results.csv",
            "raw_profile_sha256",
            "raw_profile_row_count",
        ),
        (
            "rescene_horizon_compute.csv",
            "compute_table_sha256",
            "compute_row_count",
        ),
    )
    for filename, digest_field, count_field in bindings:
        path = artifact_root / filename
        if profile.get(digest_field) != _sha256_file(path) or profile.get(
            count_field
        ) != _csv_row_count(path):
            raise FinalEvidenceError(f"adaptation profile artifact differs: {filename}")


def _validate_conditional_artifacts(
    artifact_root: Path,
    *,
    phase_i: str,
    phase_ii: str,
    phase_iii: str,
    living_scenes_triggered: object,
) -> str:
    tracker_challenge = artifact_root / "TRIVIAL_TRACKER_CHALLENGE_REPORT.md"
    if (phase_i == "TRACKER_EXPLAINS_IDENTITY") != tracker_challenge.is_file():
        raise FinalEvidenceError("Gate-I challenge report condition differs")
    horizon_challenge = artifact_root / "LONG_HORIZON_RESCENE_CHALLENGE_REPORT.md"
    if (phase_ii == "FULL_HISTORY_DOMINANT") != horizon_challenge.is_file():
        raise FinalEvidenceError("Gate-II challenge report condition differs")
    conditional_paths = [
        artifact_root / name for name in CONDITIONAL_LIVINGSCENES_ARTIFACTS
    ]
    if phase_iii == "ASSOCIATION_CEILING":
        if living_scenes_triggered is not True or not all(
            path.is_file() and not path.is_symlink() for path in conditional_paths
        ):
            raise FinalEvidenceError("LivingScenes artifacts are required by Phase III")
        return "triggered"
    if phase_iii != "PERCEPTION_CEILING" or living_scenes_triggered is not False:
        raise FinalEvidenceError("Phase-III conditional state is invalid")
    if any(path.exists() or path.is_symlink() for path in conditional_paths):
        raise FinalEvidenceError("untriggered LivingScenes artifacts must be absent")
    return "not_triggered"


def _inventory_artifacts(artifact_root: Path) -> dict[str, dict[str, object]]:
    records = {}
    for directory, child_directories, filenames in os.walk(artifact_root):
        child_directories[:] = sorted(
            name for name in child_directories if name != "entries"
        )
        for filename in sorted(filenames):
            path = Path(directory) / filename
            relative = path.relative_to(artifact_root).as_posix()
            if relative == "final_evidence_manifest.json":
                continue
            if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
                raise FinalEvidenceError(f"lightweight artifact is invalid: {relative}")
            records[relative] = {
                "byte_size": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
    return dict(sorted(records.items()))


def build_final_evidence_manifest(
    *,
    artifact_root: str | Path,
    repo_root: str | Path,
    source_commit: str,
    expected_system_tree: str,
) -> dict[str, object]:
    root = Path(artifact_root)
    repository = Path(repo_root)
    if root.is_symlink() or not root.is_dir():
        raise FinalEvidenceError("reviewer-closure artifact root is unavailable")
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise FinalEvidenceError("final evidence source commit is invalid")
    if re.fullmatch(r"[0-9a-f]{40}", expected_system_tree) is None:
        raise FinalEvidenceError("expected system-comparison tree is invalid")
    actual_tree = _git_tree(repository)
    if actual_tree != expected_system_tree:
        raise FinalEvidenceError("immutable system-comparison tree differs")
    _validate_required_artifacts(root)

    final_classification = validate_final_report(
        (root / "FINAL_METHOD_LOCK_REPORT.md").read_text(encoding="utf-8")
    )
    summary = (root / "REVIEWER_CLOSURE_SUMMARY.md").read_text(encoding="utf-8")
    if summary.count(f"`{final_classification}`") != 1:
        raise FinalEvidenceError("summary final classification differs")

    gate_i = _validate_content_manifest(root / "gate_i.json", name="Gate I")
    evaluation = _validate_content_manifest(
        root / "rescene_horizon_adaptation_evaluation_manifest.json",
        name="adaptation evaluation manifest",
    )
    profile = _validate_content_manifest(
        root / "rescene_horizon_profile_manifest.json",
        name="adaptation profile manifest",
    )
    gate_ii = _validate_content_manifest(
        root / "rescene_horizon_gate_ii.json",
        name="Gate II",
    )
    phase_iii = _validate_pretty_content_manifest(
        root / "phase_iii_manifest.json",
        name="Phase III",
    )
    training = _validate_content_manifest(
        root / "rescene_horizon_training_manifest.json",
        name="adaptation training manifest",
    )
    smoke = _validate_content_manifest(
        root / "t3_smoke_report.json",
        name="T3 smoke",
    )
    validate_t3_smoke_evidence(smoke)
    _validate_evaluation_artifacts(root, evaluation)
    _validate_profile_artifacts(root, profile)
    if (
        gate_ii.get("evaluation_manifest_content_sha256")
        != evaluation["content_sha256"]
        or gate_ii.get("profile_manifest_content_sha256") != profile["content_sha256"]
    ):
        raise FinalEvidenceError("Gate-II manifest bindings differ")
    canonical_reload = training.get("canonical_checkpoint_reload")
    if (
        not isinstance(canonical_reload, Mapping)
        or canonical_reload.get("strict") is not True
    ):
        raise FinalEvidenceError("canonical adaptation checkpoint reload is not strict")

    (
        phase_i_classification,
        phase_ii_classification,
        phase_iii_classification,
    ) = validate_gate_classifications(gate_i, gate_ii, phase_iii)
    phase_iv = _validate_conditional_artifacts(
        root,
        phase_i=phase_i_classification,
        phase_ii=phase_ii_classification,
        phase_iii=phase_iii_classification,
        living_scenes_triggered=phase_iii.get("living_scenes_triggered"),
    )
    artifacts = _inventory_artifacts(root)
    if not set(REQUIRED_ARTIFACTS).issubset(artifacts):
        raise FinalEvidenceError("final evidence inventory lacks required artifacts")
    manifest: dict[str, object] = {
        "schema_version": 1,
        "status": "pass",
        "source_commit": source_commit,
        "final_classification": final_classification,
        "gates": {
            "phase_i": phase_i_classification,
            "phase_ii": phase_ii_classification,
            "phase_iii": phase_iii_classification,
            "phase_iv": phase_iv,
        },
        "immutable_system_comparison_tree": actual_tree,
        "required_artifact_count": len(REQUIRED_ARTIFACTS),
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }
    manifest["content_sha256"] = content_sha256(manifest)
    return manifest


def verify_final_evidence_manifest(
    manifest: Mapping[str, object],
    *,
    artifact_root: str | Path,
    repo_root: str | Path,
) -> None:
    if not isinstance(manifest, Mapping) or manifest.get("status") != "pass":
        raise FinalEvidenceError("final evidence manifest status differs")
    source_commit = manifest.get("source_commit")
    expected_tree = manifest.get("immutable_system_comparison_tree")
    if not isinstance(source_commit, str) or not isinstance(expected_tree, str):
        raise FinalEvidenceError("final evidence manifest provenance is missing")
    rebuilt = build_final_evidence_manifest(
        artifact_root=artifact_root,
        repo_root=repo_root,
        source_commit=source_commit,
        expected_system_tree=expected_tree,
    )
    if dict(manifest) != rebuilt:
        raise FinalEvidenceError("final evidence manifest differs from artifacts")


def publish_final_evidence_manifest(
    path: str | Path,
    manifest: Mapping[str, object],
) -> None:
    output = Path(path)
    payload = _canonical_json_bytes(manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        if output.is_symlink() or not output.is_file():
            raise FileExistsError(f"output is not a regular file: {output}")
        if output.read_bytes() == payload:
            return
        raise FileExistsError(f"output already contains different content: {output}")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.link(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _git_head(repo_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        check=True,
        text=True,
    )
    return completed.stdout.strip()


def _expected_system_tree(repo_root: Path) -> str:
    config = yaml.safe_load(
        (repo_root / "configs/reviewer_closure/protocol.yaml").read_text(
            encoding="utf-8"
        )
    )
    if not isinstance(config, Mapping):
        raise FinalEvidenceError("reviewer-closure protocol config is invalid")
    baseline = config.get("baseline")
    tree = baseline.get("artifact_tree") if isinstance(baseline, Mapping) else None
    if not isinstance(tree, str):
        raise FinalEvidenceError("reviewer-closure baseline tree is missing")
    return tree


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("build", "verify"))
    parser.add_argument("--repo-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args(argv)
    if arguments.mode == "build":
        manifest = build_final_evidence_manifest(
            artifact_root=arguments.artifact_root,
            repo_root=arguments.repo_root,
            source_commit=_git_head(arguments.repo_root),
            expected_system_tree=_expected_system_tree(arguments.repo_root),
        )
        publish_final_evidence_manifest(arguments.output, manifest)
    else:
        manifest = _load_json(arguments.output, name="final evidence manifest")
        verify_final_evidence_manifest(
            manifest,
            artifact_root=arguments.artifact_root,
            repo_root=arguments.repo_root,
        )
    print(
        json.dumps(
            {
                "artifact_count": manifest["artifact_count"],
                "content_sha256": manifest["content_sha256"],
                "final_classification": manifest["final_classification"],
                "status": "pass",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "REQUIRED_ARTIFACTS",
    "FinalEvidenceError",
    "build_final_evidence_manifest",
    "content_sha256",
    "publish_final_evidence_manifest",
    "validate_final_report",
    "validate_gate_classifications",
    "validate_t3_smoke_evidence",
    "verify_final_evidence_manifest",
]
