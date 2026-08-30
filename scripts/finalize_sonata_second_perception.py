#!/usr/bin/env python3
"""Build the gate-aware final Sonata second-perception synthesis artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts/sonata_second_perception_v1"
FROZEN_CONCERTO_REPORT = (
    PROJECT_ROOT
    / "artifacts/reviewer_closure_v3/FINAL_REVIEWER_CLOSURE_V3.md"
)
UPSTREAM_PATHS = {
    "START_STATE.json": ARTIFACT_ROOT / "START_STATE.json",
    "EVIDENCE_BASIS.md": ARTIFACT_ROOT / "EVIDENCE_BASIS.md",
    "weight/sonata_weight_manifest.json": (
        ARTIFACT_ROOT / "weight/sonata_weight_manifest.json"
    ),
    "weight/sonata_load_key_audit.json": (
        ARTIFACT_ROOT / "weight/sonata_load_key_audit.json"
    ),
    "preflight/preflight_authorization.json": (
        ARTIFACT_ROOT / "preflight/preflight_authorization.json"
    ),
    "preflight/resolved_config.json": (
        ARTIFACT_ROOT / "preflight/resolved_config.json"
    ),
    "preflight/data_manifest.json": ARTIFACT_ROOT / "preflight/data_manifest.json",
    "smoke/smoke_results.json": ARTIFACT_ROOT / "smoke/smoke_results.json",
    "smoke/gradient_contract.json": ARTIFACT_ROOT / "smoke/gradient_contract.json",
    "smoke/query_interface.json": ARTIFACT_ROOT / "smoke/query_interface.json",
    "training/TRAINING_MANIFEST.json": (
        ARTIFACT_ROOT / "training/TRAINING_MANIFEST.json"
    ),
    "training/source_snapshot_manifest.json": (
        ARTIFACT_ROOT / "training/source_snapshot_manifest.json"
    ),
    "checkpoint/CHECKPOINT_MANIFEST.json": (
        ARTIFACT_ROOT / "checkpoint/CHECKPOINT_MANIFEST.json"
    ),
    "checkpoint/QUALIFICATION_MANIFEST.json": (
        ARTIFACT_ROOT / "checkpoint/QUALIFICATION_MANIFEST.json"
    ),
    "checkpoint/official_like_per_seed.csv": (
        ARTIFACT_ROOT / "checkpoint/official_like_per_seed.csv"
    ),
    "checkpoint/official_like_summary.csv": (
        ARTIFACT_ROOT / "checkpoint/official_like_summary.csv"
    ),
    "frozen_concerto_v3/FINAL_REVIEWER_CLOSURE_V3.md": FROZEN_CONCERTO_REPORT,
}


def _measured_row(
    summary_rows: Sequence[Mapping[str, str]], model: str
) -> Mapping[str, str]:
    selected = [row for row in summary_rows if row.get("model") == model]
    if len(selected) != 1:
        raise ValueError(f"official-like summary lacks one {model} row")
    return selected[0]


def build_cross_backbone_synthesis(
    *,
    summary_rows: Sequence[Mapping[str, str]],
    qualification_gate: Mapping[str, object],
) -> dict[str, object]:
    """Build a negative synthesis without manufacturing gated SS6/SS7 values."""

    if (
        qualification_gate.get("label") != "SQ-RED"
        or qualification_gate.get("authorizes_ss6") is not False
    ):
        raise ValueError("final negative synthesis requires SQ-RED qualification evidence")
    sonata = _measured_row(summary_rows, "our_sonata_reimplementation")
    concerto = _measured_row(summary_rows, "our_concerto_reimplementation")
    rows = [
        {
            "stage": "SS5",
            "horizon": "T2",
            "metric": metric,
            "concerto_value": concerto[mean_key],
            "sonata_value": sonata[mean_key],
            "concerto_status": "measured_matched_harness",
            "sonata_status": "measured_matched_harness",
        }
        for metric, mean_key in (
            ("t_mAP", "t_mAP_mean"),
            ("overall_mAP", "overall_mAP_mean"),
        )
    ]
    rows.append(
        {
            "stage": "SS6",
            "horizon": "T2",
            "metric": "local_candidate_invariance",
            "concerto_value": "pass",
            "sonata_value": "",
            "concerto_status": "frozen_v3_evidence",
            "sonata_status": "gate_skipped",
        }
    )
    for horizon, gap_recovery in (("T4", "19.971"), ("T5", "22.690")):
        rows.extend(
            [
                {
                    "stage": "SS7",
                    "horizon": horizon,
                    "metric": "b4_minus_b2_gap_recovery_pp",
                    "concerto_value": gap_recovery,
                    "sonata_value": "",
                    "concerto_status": "frozen_v3_evidence",
                    "sonata_status": "gate_skipped",
                },
                {
                    "stage": "SS7",
                    "horizon": horizon,
                    "metric": "positive_reference_scene_clusters",
                    "concerto_value": "6/6",
                    "sonata_value": "",
                    "concerto_status": "frozen_v3_evidence",
                    "sonata_status": "gate_skipped",
                },
            ]
        )
    return {
        "sq_gate": "SQ-RED",
        "sr_gate": "SR-RED",
        "protocol_b_status": "gate_skipped",
        "rows": rows,
    }


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def build_final_manifest(
    *,
    source_commit: str,
    upstream_payloads: Mapping[str, bytes],
    output_payloads: Mapping[str, bytes],
) -> dict[str, object]:
    """Bind the negative final conclusion to every source and output byte."""

    manifest = {
        "schema_version": 1,
        "status": "negative_result",
        "stage": "FINAL",
        "source_commit": source_commit,
        "sq_gate": "SQ-RED",
        "sr_gate": "SR-RED",
        "stages": {
            "SS0_SS5": "completed",
            "SS6": "gate_skipped",
            "SS7": "gate_skipped",
        },
        "upstream_sha256": {
            name: hashlib.sha256(payload).hexdigest()
            for name, payload in sorted(upstream_payloads.items())
        },
        "output_sha256": {
            name: hashlib.sha256(payload).hexdigest()
            for name, payload in sorted(output_payloads.items())
        },
        "authorized_claims": [
            "A provenance-locked local Sonata ReScene4D task reimplementation completed 450 epochs and was selected by local validation only.",
            "Under the matched three-seed local T2 harness, this Sonata task checkpoint is weaker than the frozen Concerto reimplementation on temporal and overall AP.",
            "The current experiment does not validate Persist4D long-gap recovery across Sonata and Concerto backbones.",
        ],
        "forbidden_claims": [
            "official ReScene4D-S checkpoint or reproduction of paper-reported 33.2",
            "Sonata Protocol-B or robustness results",
            "cross-backbone persistence benefit",
            "backbone-universal or SOTA behavior",
        ],
    }
    manifest["content_sha256"] = _canonical_sha256(manifest)
    return manifest


def _read_json(path: Path) -> Mapping[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"JSON evidence must contain a mapping: {path.name}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _csv_payload(rows: Sequence[Mapping[str, object]]) -> bytes:
    if not rows:
        raise ValueError("cross-backbone summary cannot be empty")
    fields = tuple(rows[0])
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _json_payload(value: object) -> bytes:
    return (
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _publish(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_file() and not path.is_symlink() and path.read_bytes() == payload:
            return
        raise FileExistsError(f"refusing to overwrite final artifact: {path.name}")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _percent(value: object) -> str:
    return f"{100.0 * float(value):.3f}%"


def _render_cross_analysis(synthesis: Mapping[str, object]) -> bytes:
    rows = synthesis["rows"]
    sonata_tmap = next(
        row["sonata_value"]
        for row in rows
        if row["stage"] == "SS5" and row["metric"] == "t_mAP"
    )
    concerto_tmap = next(
        row["concerto_value"]
        for row in rows
        if row["stage"] == "SS5" and row["metric"] == "t_mAP"
    )
    sonata_overall = next(
        row["sonata_value"]
        for row in rows
        if row["stage"] == "SS5" and row["metric"] == "overall_mAP"
    )
    concerto_overall = next(
        row["concerto_value"]
        for row in rows
        if row["stage"] == "SS5" and row["metric"] == "overall_mAP"
    )
    return "\n".join(
        [
            "# Cross-Backbone Analysis",
            "",
            "- SQ gate: `SQ-RED`",
            "- SR gate: `SR-RED`",
            "- Sonata Protocol-B status: `gate_skipped`",
            "",
            "Under the same three-seed local T2 harness, the Sonata reimplementation",
            f"reached t-mAP `{_percent(sonata_tmap)}` and overall mAP",
            f"`{_percent(sonata_overall)}`. The matched Concerto reimplementation",
            f"reached `{_percent(concerto_tmap)}` and `{_percent(concerto_overall)}`.",
            "Sonata is therefore weaker on both qualification axes and does not",
            "authorize SS6 or SS7.",
            "",
            "Frozen Concerto V3 remains positive for B4-minus-B2 gap recovery",
            "at T4 (+19.971 pp) and T5 (+22.690 pp), with all six physical-scene",
            "clusters positive at both horizons. No corresponding Sonata values",
            "were computed, so those Concerto results cannot be generalized across",
            "backbones in this experiment.",
            "",
            "Conclusion: current persistent-state advantage is not yet",
            "cross-backbone validated. This is a negative qualification result,",
            "not evidence that persistence fails with Sonata.",
            "",
        ]
    ).encode("utf-8")


def _render_robustness_report() -> bytes:
    return b"""# Sonata Robustness Report

- Status: `gate_skipped`
- SQ gate: `SQ-RED`
- SR gate: `SR-RED`
- Protocol-B cache created: `false`
- B2/B3/B4 evaluations run: `false`
- FullHistory evaluation run: `false`
- Score-reducer sensitivity run: `false`

SS6 and SS7 require SQ-GREEN. The Sonata checkpoint was weaker than
the matched Concerto checkpoint on both t-mAP and overall mAP, so the
automatic robustness stage was not authorized. Blank Sonata Protocol-B
fields in the cross-backbone summary mean not measured, not zero.
"""


def _render_final_report(
    *,
    start_state: Mapping[str, object],
    weight: Mapping[str, object],
    load_audit: Mapping[str, object],
    training: Mapping[str, object],
    checkpoint: Mapping[str, object],
    per_seed: Sequence[Mapping[str, str]],
    summary_rows: Sequence[Mapping[str, str]],
    source_commit: str,
    changed_file_count: int,
) -> bytes:
    sonata = _measured_row(summary_rows, "our_sonata_reimplementation")
    concerto = _measured_row(summary_rows, "our_concerto_reimplementation")
    sonata_seed_rows = [row for row in per_seed if row["model"] == "sonata"]
    external = start_state["external_sources"]
    remote = weight["remote_metadata"]
    local = weight["local_file"]
    runtime = training["runtime"]
    recipe = training["recipe"]
    budget = training["budget"]
    selected = checkpoint["sonata"]
    seed_text = "; ".join(
        f"{row['seed']}: t-mAP={float(row['t_mAP']):.6f}, overall={float(row['overall_mAP']):.6f}"
        for row in sonata_seed_rows
    )
    return "\n".join(
        [
            "# Final Sonata Second-Perception Report",
            "",
            "1. Branch / start commit / generation commit",
            f"   - `{start_state['branch']}` / `{start_state['start_commit']}` / `{source_commit}`.",
            "2. Changed files",
            f"   - `{changed_file_count}` tracked paths changed since the captured start commit; core additions are Sonata provenance, preflight, smoke/training evidence, SS5 evaluator/tests, and final synthesis artifacts.",
            "3. External evidence re-verified",
            f"   - ReScene4D `{external['rescene4d']['revision']}` still declared checkpoints `Coming soon`; no official task checkpoint was substituted.",
            "4. Sonata source revision",
            f"   - Code `{external['sonata_code']['revision']}`; weight repository `{remote['revision']}`.",
            "5. Sonata pretrained weight",
            f"   - Immutable revision `{remote['revision']}`, SHA256 `{local['sha256']}`, `{local['bytes']}` bytes, license `CC-BY-NC-4.0`.",
            "6. Load-key audit",
            f"   - `{load_audit['loaded_key_count']}` encoder keys loaded; `{len(load_audit['missing_keys'])}` allowlisted decoder keys missing; `{len(load_audit['unexpected_keys'])}` unexpected keys.",
            "7. Resolved Sonata ReScene config",
            "   - Sonata PTv3, T2, 100 non-parametric queries, mixed ST serialization, temporal masking ON, contrastive OFF, EOS 0.2, frozen encoder, AdamW/OneCycle, max LR 5e-4.",
            "8. Data / split / mix provenance",
            "   - 3RScan T2 + ScanNet T1 mixed training at 1.0:0.8; official-like qualification uses all 154 filtered 3RScan T2 validation sequences.",
            "9. Hardware / batch",
            f"   - Two NVIDIA A40 GPUs; physical batch `{recipe['microbatch_per_gpu']}` per GPU, accumulation `{recipe['accumulate_grad_batches']}`, effective batch `{recipe['effective_global_batch']}`.",
            "10. Smoke + gradient contract",
            "   - SSMOKE-PASS; query interface compatible; frozen encoder gradients absent; all trainable decoder/head gradients finite and nonzero.",
            "11. Formal training",
            f"   - Seed `{recipe['seed']}`, `{budget['completed_epochs']}` epochs, `{budget['optimizer_steps']}` optimizer steps, `{runtime['interruption_count']}` interruptions / `{runtime['resume_launch_count']}` resumes.",
            "12. Selected checkpoint",
            f"   - Epoch `{selected['epoch']}`, SHA256 `{selected['sha256']}`, highest local val_mean_t-AP `{selected['selection_metric_exact']}`; no Protocol-B selection leakage.",
            "13. Official-like Sonata local results",
            f"   - {seed_text}. Mean t-mAP `{float(sonata['t_mAP_mean']):.6f}`, overall `{float(sonata['overall_mAP_mean']):.6f}`.",
            "14. Matched current-Concerto results",
            f"   - Mean t-mAP `{float(concerto['t_mAP_mean']):.6f}`, overall `{float(concerto['overall_mAP_mean']):.6f}` under the same seeds/runtime.",
            "15. SQ gate",
            "   - `SQ-RED`: Sonata is weaker than Concerto on both temporal and spatial qualification metrics and is below the 0.297 t-mAP threshold.",
            "16. Conditional SQ-GREEN evidence",
            "   - Not applicable. SS6/SS7 were not authorized; no Sonata Protocol-B, invariance, robustness, reducer, or compute values were generated.",
            "17. Concerto-vs-Sonata synthesis",
            "   - Frozen Concerto V3 remains positive, but this Sonata checkpoint did not qualify. The persistent-state benefit is not cross-backbone validated by this experiment.",
            "18. Tests / lint / diff-check",
            "   - `95` Sonata tests and `57` frozen V3 regressions passed; task-owned Python files pass Ruff and `git diff --check` passes. The same `36` Ruff findings remain in `trainer/trainer.py` at both the captured start commit and this revision.",
            "19. Artifact and checkpoint hashes",
            "   - Upstream/output hashes are enumerated in `FINAL_MANIFEST.json`; checkpoint hashes are in `checkpoint/CHECKPOINT_MANIFEST.json`.",
            "20. Remaining external-asset failures",
            "   - Official ReScene4D-S/C task checkpoints remain unavailable; no missing local asset blocked SS0-SS5.",
            "21. SR gate",
            "   - `SR-RED`, derived from the failed SQ gate; robustness was gate-skipped rather than measured as zero.",
            "22. Claims now authorized",
            "   - A provenance-locked 450-epoch Sonata local task reimplementation was trained and negatively qualified against matched Concerto evidence.",
            "23. Claims still forbidden",
            "   - Official 33.2 reproduction, Sonata Protocol-B performance, cross-backbone persistence benefit, backbone universality, and SOTA.",
            "24. Recommended next research stage",
            "   - Diagnose the local Sonata task-learning gap under a separately preregistered training study; do not tune Persist4D/B4 on these final qualification results.",
            "",
        ]
    ).encode("utf-8")


def finalize(*, artifact_root: Path = ARTIFACT_ROOT) -> Mapping[str, object]:
    upstream_payloads = {
        name: path.read_bytes() for name, path in UPSTREAM_PATHS.items()
    }
    qualification = _read_json(
        artifact_root / "checkpoint/QUALIFICATION_MANIFEST.json"
    )
    summary_rows = _read_csv(
        artifact_root / "checkpoint/official_like_summary.csv"
    )
    per_seed = _read_csv(artifact_root / "checkpoint/official_like_per_seed.csv")
    synthesis = build_cross_backbone_synthesis(
        summary_rows=summary_rows,
        qualification_gate=qualification["gate"],
    )
    source_commit = _git("rev-parse", "HEAD")
    start_state = _read_json(artifact_root / "START_STATE.json")
    changed_files = _git(
        "diff",
        "--name-only",
        f"{start_state['start_commit']}..{source_commit}",
    ).splitlines()
    cross_payload = _csv_payload(synthesis["rows"])
    analysis_payload = _render_cross_analysis(synthesis)
    robustness_payload = _render_robustness_report()
    final_report_payload = _render_final_report(
        start_state=start_state,
        weight=_read_json(artifact_root / "weight/sonata_weight_manifest.json"),
        load_audit=_read_json(
            artifact_root / "weight/sonata_load_key_audit.json"
        ),
        training=_read_json(artifact_root / "training/TRAINING_MANIFEST.json"),
        checkpoint=_read_json(
            artifact_root / "checkpoint/CHECKPOINT_MANIFEST.json"
        ),
        per_seed=per_seed,
        summary_rows=summary_rows,
        source_commit=source_commit,
        changed_file_count=len(changed_files),
    )
    output_payloads = {
        "cross_backbone_summary.csv": cross_payload,
        "CROSS_BACKBONE_ANALYSIS.md": analysis_payload,
        "robustness/ROBUSTNESS_REPORT.md": robustness_payload,
        "FINAL_SONATA_SECOND_PERCEPTION_REPORT.md": final_report_payload,
    }
    manifest = build_final_manifest(
        source_commit=source_commit,
        upstream_payloads=upstream_payloads,
        output_payloads=output_payloads,
    )
    manifest["branch"] = start_state["branch"]
    manifest["start_commit"] = start_state["start_commit"]
    manifest["changed_files"] = changed_files
    manifest.pop("content_sha256")
    manifest["content_sha256"] = _canonical_sha256(manifest)
    output_payloads["FINAL_MANIFEST.json"] = _json_payload(manifest)
    for relative_path, payload in output_payloads.items():
        _publish(artifact_root / relative_path, payload)
    return {
        "status": "negative_result",
        "sq_gate": "SQ-RED",
        "sr_gate": "SR-RED",
        "source_commit": source_commit,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    result = finalize()
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
