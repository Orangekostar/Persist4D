"""Strict, deterministic artifact contract for Persist4D P6-B."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path, PurePosixPath

import yaml

from scripts.p6b_figures import render_identity_figure, render_reactivation_figure

P6B_ARTIFACT_SCHEMA_VERSION = 1

_ROOT_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "decision",
        "source_commit",
        "source_tree_contract",
        "provenance",
        "split_manifest",
        "selection",
        "sweep_rows",
        "final_results",
        "per_sequence_results",
        "failure_analysis",
        "gate_results",
        "claims_supported",
        "claims_not_supported",
        "next_action",
        "artifact_manifest",
    }
)
_PROVENANCE_KEYS = frozenset(
    {
        "checkpoint",
        "p5",
        "p6a",
        "p6a_protocol_manifest",
        "p6a_cache_manifest",
    }
)
_STAGES = (
    "assignment",
    "reactivation",
    "class_compatibility",
    "consolidation",
    "birth_gate",
    "joint_neighbors",
)
_STAGE_PATHS = {
    "assignment": "assignment_ablation.csv",
    "reactivation": "reactivation_threshold_sweep.csv",
    "class_compatibility": "class_compatibility_ablation.csv",
    "consolidation": "consolidation_ablation.csv",
    "birth_gate": "birth_gate_sweep.csv",
    "joint_neighbors": "joint_validation_sweep.csv",
}
_SWEEP_COLUMNS = (
    "config_id",
    "config_json",
    "stage",
    "T",
    "identity_switches",
    "wrong_reactivations",
    "false_births",
    "reactivation_accuracy",
    "reactivation_recall",
    "accepted_valid_observations",
    "total_valid_observations",
    "strict_online_tmap",
    "strict_online_trec",
    "eligible",
    "eligibility_reasons",
)
_FINAL_COLUMNS = (
    "method",
    "T",
    "t_mAP",
    "t_REC",
    "identity_switches",
    "identity_switch_rate",
    "reactivation_accuracy",
    "reactivation_recall",
    "false_births",
)
_PER_SEQUENCE_COLUMNS = (
    "method",
    "reference_scene_id",
    "master_sequence_id",
    "order_id",
    "T",
    "identity_switches",
    "transition_opportunities",
    "identity_switch_rate",
    "wrong_reactivations",
    "false_births",
    "reactivation_accuracy",
    "reactivation_recall",
)
_FAILURE_COLUMNS = ("method", "T", "failure_category", "count")
_REPORT_SECTIONS = (
    "1. What was changed",
    "2. Why it was changed",
    "3. Experimental protocol",
    "4. Reproducibility binding",
    "5. Main results",
    "6. Statistical evidence",
    "7. Failure analysis",
    "8. What claims are supported",
    "9. What claims are NOT supported",
    "10. GO / NO-GO decision",
    "11. Exact next action",
)
_PRIVATE_PATH = re.compile(
    r"(?:/home/|/Users/|/mnt/|[A-Za-z]:[\\/]Users[\\/])"
)
_SHA40 = re.compile(r"[0-9a-f]{40}")
_SHA64 = re.compile(r"[0-9a-f]{64}")


def _exact_keys(value: object, expected: frozenset[str], *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    if set(value) != expected:
        raise ValueError(f"{name} keys differ from the schema")
    return value


def _finite_tree(value: object, *, name: str = "artifact") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{name} contains a non-finite number")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _finite_tree(item, name=f"{name}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            _finite_tree(item, name=f"{name}[{index}]")


def _plain_ref(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.startswith(("repo:", "external:")):
        raise ValueError(f"{name} must be a portable repo: or external: reference")
    if _PRIVATE_PATH.search(value):
        raise ValueError(f"{name} must be portable")
    return value


def _validate_rows(
    rows: object, columns: Sequence[str], *, name: str, nonempty: bool = True
) -> Sequence[Mapping[str, object]]:
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise TypeError(f"{name} must be a sequence")
    if nonempty and not rows:
        raise ValueError(f"{name} must not be empty")
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or tuple(row) != tuple(columns):
            raise ValueError(f"{name}[{index}] columns differ from the schema")
    return rows


def _validate_base(root: Mapping[str, object]) -> None:
    _exact_keys(root, _ROOT_KEYS, name="P6-B root")
    if root["schema_version"] != P6B_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("P6-B schema_version differs")
    if root["status"] != "pass":
        raise ValueError("P6-B artifact status must be pass")
    if root["decision"] not in {"P6B_GO", "P6B_STOP"}:
        raise ValueError("P6-B decision is invalid")
    source_commit = root["source_commit"]
    if not isinstance(source_commit, str) or _SHA40.fullmatch(source_commit) is None:
        raise ValueError("source_commit must be a lowercase Git SHA")
    source_contract = _exact_keys(
        root["source_tree_contract"],
        frozenset({"status", "source_commit"}),
        name="source_tree_contract",
    )
    if source_contract != {"status": "pass", "source_commit": source_commit}:
        raise ValueError("source_tree_contract differs from source_commit")
    provenance = _exact_keys(root["provenance"], _PROVENANCE_KEYS, name="provenance")
    for key, raw_record in provenance.items():
        record = _exact_keys(
            raw_record, frozenset({"ref", "sha256"}), name=f"provenance.{key}"
        )
        _plain_ref(record["ref"], name=f"provenance.{key}.ref")
        if not isinstance(record["sha256"], str) or _SHA64.fullmatch(record["sha256"]) is None:
            raise ValueError(f"provenance.{key}.sha256 is invalid")
    split = root["split_manifest"]
    if not isinstance(split, Mapping):
        raise TypeError("split_manifest must be a mapping")
    tuning = split.get("tuning_reference_scene_ids")
    heldout = split.get("heldout_reference_scene_ids")
    tuning_masters = split.get("tuning_master_sequence_ids")
    heldout_masters = split.get("heldout_master_sequence_ids")
    if not all(isinstance(value, list) for value in (tuning, heldout, tuning_masters, heldout_masters)):
        raise ValueError("split partitions must be lists")
    if len(tuning) != 4 or len(heldout) != 2 or len(tuning_masters) != 32 or len(heldout_masters) != 11:
        raise ValueError("split partition counts differ from the frozen protocol")
    if set(tuning) & set(heldout) or set(tuning_masters) & set(heldout_masters):
        raise ValueError("split tuning/heldout overlap is forbidden")
    selection = _exact_keys(
        root["selection"],
        frozenset(
            {
                "config_id",
                "config_sha256",
                "config",
                "ranking_key",
                "tuning_reference_scene_ids",
            }
        ),
        name="selection",
    )
    if selection["tuning_reference_scene_ids"] != tuning:
        raise ValueError("selection tuning partition differs from split")
    canonical_config = json.dumps(
        selection["config"], sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    expected_config_sha = hashlib.sha256(canonical_config).hexdigest()
    if selection["config_sha256"] != expected_config_sha:
        raise ValueError("selection config_sha256 differs from canonical config")
    sweep_rows = _validate_rows(root["sweep_rows"], _SWEEP_COLUMNS, name="sweep_rows")
    if {row["stage"] for row in sweep_rows} != set(_STAGES):
        raise ValueError("sweep_rows must cover every required ablation stage")
    _validate_rows(root["final_results"], _FINAL_COLUMNS, name="final_results")
    per_sequence = _validate_rows(
        root["per_sequence_results"], _PER_SEQUENCE_COLUMNS, name="per_sequence_results"
    )
    if any(row["reference_scene_id"] not in heldout for row in per_sequence):
        raise ValueError("final per-sequence rows contain tuning data")
    _validate_rows(root["failure_analysis"], _FAILURE_COLUMNS, name="failure_analysis")
    gates = _exact_keys(
        root["gate_results"],
        frozenset(f"G6B-{index}" for index in range(1, 6)),
        name="gate_results",
    )
    for gate, raw_record in gates.items():
        record = _exact_keys(
            raw_record, frozenset({"passed", "evidence"}), name=f"gate_results.{gate}"
        )
        if not isinstance(record["passed"], bool) or not isinstance(record["evidence"], str) or not record["evidence"]:
            raise ValueError(f"gate_results.{gate} is invalid")
    expected_decision = "P6B_GO" if all(record["passed"] for record in gates.values()) else "P6B_STOP"
    if root["decision"] != expected_decision:
        raise ValueError("decision differs from gate results")
    for key in ("claims_supported", "claims_not_supported"):
        claims = root[key]
        if not isinstance(claims, list) or not claims or any(not isinstance(item, str) or not item for item in claims):
            raise ValueError(f"{key} must contain nonempty claims")
    supported_text = " ".join(root["claims_supported"]).casefold()
    if "sota" in supported_text or "state-of-the-art" in supported_text:
        raise ValueError("unsupported claim in claims_supported")
    if not isinstance(root["next_action"], str) or not root["next_action"]:
        raise ValueError("next_action must be nonempty")
    serialized = json.dumps(root, sort_keys=True, allow_nan=False)
    if _PRIVATE_PATH.search(serialized):
        raise ValueError("artifact contains a private absolute path")
    _finite_tree(root)


def _render_csv(rows: Sequence[Mapping[str, object]], columns: Sequence[str]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: "" if row[key] is None else row[key] for key in columns})
    return output.getvalue().encode("utf-8")


def _render_report(root: Mapping[str, object]) -> bytes:
    results = root["final_results"]
    p6b_t5 = next(
        row for row in results if row["method"] == "P6B" and row["T"] == "T5"
    )
    content = ["# Persist4D P6-B GO / NO-GO Report", ""]
    bodies = (
        "Added threshold-aware assignment and quality-gated persistent memory without changing frozen local predictions.",
        "P6-A isolated association, reactivation, and birth quality as the actionable method bottlenecks.",
        "Candidates used four tuning reference clusters; the selected config was frozen before one evaluation on two held-out clusters.",
        f"Source `{root['source_commit']}`; selected config `{root['selection']['config_sha256']}`; split `{root['split_manifest']['sha256']}`.",
        f"Held-out P6B T5 t-mAP={p6b_t5['t_mAP']:.6f}, t-REC={p6b_t5['t_REC']:.6f}, ID switches={p6b_t5['identity_switches']}.",
        "Paired per-sequence rows and deterministic eligibility/ranking evidence are included in the bundle.",
        "Failure categories are reported separately and remain bounded to frozen local predictions plus P6-B association decisions.",
        "\n".join(f"- {claim}" for claim in root["claims_supported"]),
        "\n".join(f"- {claim}" for claim in root["claims_not_supported"]),
        "All five preregistered gates determine the terminal decision below.",
        root["next_action"],
    )
    for heading, body in zip(_REPORT_SECTIONS, bodies, strict=True):
        content.extend((f"## {heading}", "", body, ""))
    content.append(root["decision"])
    return "\n".join(content).encode("utf-8")


def _render_derived(root: Mapping[str, object]) -> dict[str, bytes]:
    sweep_rows = root["sweep_rows"]
    rendered = {
        path: _render_csv(
            [row for row in sweep_rows if row["stage"] == stage], _SWEEP_COLUMNS
        )
        for stage, path in _STAGE_PATHS.items()
    }
    rendered.update(
        {
            "split_manifest.json": (json.dumps(root["split_manifest"], sort_keys=True, indent=2, allow_nan=False) + "\n").encode(),
            "selected_config.yaml": yaml.safe_dump(root["selection"]["config"], sort_keys=True).encode(),
            "final_results.csv": _render_csv(root["final_results"], _FINAL_COLUMNS),
            "per_sequence_results.csv": _render_csv(root["per_sequence_results"], _PER_SEQUENCE_COLUMNS),
            "failure_analysis.csv": _render_csv(root["failure_analysis"], _FAILURE_COLUMNS),
            "P6B_GO_NOGO_REPORT.md": _render_report(root),
            "figures/identity_comparison.svg": render_identity_figure(root["final_results"]).encode(),
            "figures/reactivation_comparison.svg": render_reactivation_figure(root["final_results"]).encode(),
        }
    )
    return dict(sorted(rendered.items()))


def _manifest_for(files: Mapping[str, bytes]) -> list[dict[str, object]]:
    return [
        {"path": path, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
        for path, payload in sorted(files.items())
    ]


def validate_p6b_artifact(root: Mapping[str, object]) -> None:
    _validate_base(root)
    manifest = root["artifact_manifest"]
    if not isinstance(manifest, list) or not manifest:
        raise ValueError("artifact_manifest must not be empty")
    expected = _manifest_for(_render_derived(root))
    if manifest != expected:
        raise ValueError("artifact_manifest does not bind rendered bytes")


def finalize_p6b_artifact(root: Mapping[str, object]) -> dict[str, object]:
    candidate = deepcopy(dict(root))
    candidate["artifact_manifest"] = []
    _validate_base(candidate)
    candidate["artifact_manifest"] = _manifest_for(_render_derived(candidate))
    validate_p6b_artifact(candidate)
    return candidate


def render_p6b_bundle(root: Mapping[str, object]) -> dict[str, bytes]:
    validate_p6b_artifact(root)
    files = _render_derived(root)
    files["artifact_manifest.json"] = (
        json.dumps(root["artifact_manifest"], sort_keys=True, indent=2) + "\n"
    ).encode()
    files["p6b_eval.json"] = (
        json.dumps(root, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()
    return dict(sorted(files.items()))


def publish_p6b_artifact(output_root: Path, root: Mapping[str, object]) -> Path:
    output = Path(output_root)
    if output.exists() or output.is_symlink():
        raise FileExistsError("P6-B output root already exists")
    if output.name in {"", ".", ".."}:
        raise ValueError("P6-B output root must be a named directory")
    files = render_p6b_bundle(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        for relative, payload in files.items():
            path = PurePosixPath(relative)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("artifact path must be repository-relative")
            target = stage.joinpath(*path.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
        if output.exists() or output.is_symlink():
            raise FileExistsError("P6-B output root appeared during publication")
        os.replace(stage, output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return output
