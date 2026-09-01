from __future__ import annotations

import csv
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from scripts.finalize_rescene_task_learning_root_cause import (
    FinalizationError,
    FinalPackageInputs,
    StrongSkip,
    StrongStudy,
    aggregate_strong_outputs,
    classify_principal_outcome,
    publish_final_package,
)
from scripts.finalize_rescene_task_learning_root_cause import (
    main as finalization_main,
)
from scripts.verify_rescene_rootcause_handoff import (
    HANDOFF_SECTION_TITLES,
    REQUIRED_ARTIFACTS,
    FinalArtifactError,
    build_final_manifest,
    verify_final_artifacts,
)
from utils.rescene_rootcause_preflight import canonical_sha256


def test_principal_outcome_requires_full_rootcause_confirmation_for_green() -> None:
    assert (
        classify_principal_outcome(
            short_decision={"selected_variant": "R1"},
            full_verdict={"verdict": "ROOTCAUSE-CONFIRMED"},
            strong_verdicts=[],
        )
        == "TLRC-GREEN"
    )


@pytest.mark.parametrize(
    ("short_decision", "full_verdict", "strong_verdicts"),
    [
        (
            {"selected_variant": "R1"},
            {"verdict": "ROOTCAUSE-PARTIAL"},
            [],
        ),
        (
            {"selected_variant": None},
            None,
            [
                {
                    "status": "pass",
                    "variant": "A1",
                    "all_gates_pass": True,
                }
            ],
        ),
    ],
)
def test_principal_outcome_is_yellow_for_incomplete_or_structural_gain(
    short_decision: dict[str, object],
    full_verdict: dict[str, object] | None,
    strong_verdicts: list[dict[str, object]],
) -> None:
    assert (
        classify_principal_outcome(
            short_decision=short_decision,
            full_verdict=full_verdict,
            strong_verdicts=strong_verdicts,
        )
        == "TLRC-YELLOW"
    )


def test_principal_outcome_is_red_when_all_controlled_interventions_fail() -> None:
    assert (
        classify_principal_outcome(
            short_decision={"selected_variant": None},
            full_verdict=None,
            strong_verdicts=[
                {
                    "status": "pass",
                    "variant": "A1",
                    "all_gates_pass": False,
                }
            ],
        )
        == "TLRC-RED"
    )


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _signed(payload: dict[str, object], field: str) -> dict[str, object]:
    payload[field] = canonical_sha256(payload)
    return payload


def _strong_study(tmp_path: Path, variant: str, value: float) -> StrongStudy:
    root = tmp_path / variant
    authorization = root / "variant_manifest.json"
    output = root / "finalized"
    output.mkdir(parents=True)
    authorization.write_text(
        json.dumps(
            _signed(
                {
                    "schema_version": 1,
                    "status": "authorized",
                    "selected_variants": [variant],
                },
                "authorization_sha256",
            )
        ),
        encoding="ascii",
    )
    _write_csv(
        output / "learning_curves.csv",
        [
            {"variant": "R1", "completed_epoch": 90, "metric": 0.3},
            {"variant": variant, "completed_epoch": 90, "metric": value},
        ],
    )
    _write_csv(
        output / "official_like_per_seed.csv",
        [
            {
                "variant": "R1",
                "completed_epoch": 90,
                "seed": 45,
                "metric": 0.3,
            },
            {
                "variant": variant,
                "completed_epoch": 90,
                "seed": 45,
                "metric": value,
            },
        ],
    )
    authorization_payload = json.loads(authorization.read_text(encoding="ascii"))
    verdict = _signed(
        {
            "schema_version": 1,
            "status": "pass",
            "variant": variant,
            "variant_authorization_sha256": authorization_payload[
                "authorization_sha256"
            ],
            "all_gates_pass": value >= 0.32,
            "paired_spatial_delta_mean": value - 0.3,
            "selection_used_persist4d": False,
        },
        "content_sha256",
    )
    (output / "STRONG_LOCAL_VERDICT.json").write_text(
        json.dumps(verdict), encoding="ascii"
    )
    provenance = _signed(
        {
            "schema_version": 1,
            "decision_content_sha256": verdict["content_sha256"],
        },
        "content_sha256",
    )
    (output / "STRONG_LOCAL_PROVENANCE.json").write_text(
        json.dumps(provenance), encoding="ascii"
    )
    return StrongStudy(
        variant=variant,
        authorization_path=authorization,
        output_directory=output,
    )


def _with_strong_full(
    study: StrongStudy,
    *,
    checkpoint_sha256: str = "a" * 64,
    verdict_training_sha256: str | None = None,
) -> StrongStudy:
    full = study.output_directory.parent / "full" / "finalized"
    full.mkdir(parents=True)
    authorization = json.loads(study.authorization_path.read_text(encoding="ascii"))
    (full / "FULL_TRAINING_REPORT.md").write_text(
        "# Full Training\n", encoding="ascii"
    )
    training = _signed(
        {
            "schema_version": 1,
            "status": "pass",
            "variant": study.variant,
            "variant_authorization_sha256": authorization[
                "authorization_sha256"
            ],
            "budget": {"completed_epoch": 450, "optimizer_steps": 29_700},
            "selection": {"selected_checkpoint_sha256": "a" * 64},
        },
        "content_sha256",
    )
    (full / "FULL_TRAINING_MANIFEST.json").write_text(
        json.dumps(training), encoding="ascii"
    )
    (full / "learning_curve.csv").write_text("status\npass\n", encoding="ascii")
    (full / "checkpoint_inventory.csv").write_text(
        "status\npass\n", encoding="ascii"
    )
    checkpoint = _signed(
        {
            "schema_version": 1,
            "status": "pass",
            "stage": "full_candidate",
            "variant": study.variant,
            "checkpoint": {"sha256": "a" * 64},
            "bindings": {
                "variant_authorization_sha256": authorization[
                    "authorization_sha256"
                ]
            },
            "full_training": {
                "completed_epoch": 450,
                "manifest_sha256": training["content_sha256"],
            },
        },
        "content_sha256",
    )
    (full / "selected_checkpoint_manifest.json").write_text(
        json.dumps(checkpoint), encoding="ascii"
    )
    for name in ("official_like_per_seed.csv", "official_like_summary.csv"):
        (full / name).write_text("status\npass\n", encoding="ascii")
    verdict = _signed(
        {
            "schema_version": 1,
            "status": "pass",
            "variant": study.variant,
            "verdict_prefix": "STRONG-LOCAL",
            "verdict": "STRONG-LOCAL-CONFIRMED",
            "selected_checkpoint_sha256": checkpoint_sha256,
            "checkpoint_manifest_sha256": checkpoint["content_sha256"],
            "full_training_manifest_sha256": (
                verdict_training_sha256 or training["content_sha256"]
            ),
        },
        "content_sha256",
    )
    (full / "STRONG_LOCAL_FULL_VERDICT.json").write_text(
        json.dumps(verdict), encoding="ascii"
    )
    (full / "STRONG_LOCAL_FULL_VERDICT.md").write_text(
        "# Strong Local Full Verdict\n", encoding="ascii"
    )
    provenance = _signed(
        {
            "schema_version": 1,
            "result_content_sha256": verdict["content_sha256"],
        },
        "content_sha256",
    )
    (full / "STRONG_LOCAL_FULL_PROVENANCE.json").write_text(
        json.dumps(provenance), encoding="ascii"
    )
    return replace(study, full_directory=full)


def _strong_skip(tmp_path: Path, variant: str) -> StrongSkip:
    path = tmp_path / variant / "variant_manifest.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            _signed(
                {
                    "schema_version": 1,
                    "status": "gate_skipped",
                    "experiment": "rescene_strong_local_v1",
                    "variant": variant,
                    "gate": {
                        "status": "gate_skipped",
                        "authorized": False,
                        "reason": "upstream structural gate did not pass",
                    },
                    "upstream_evidence": {
                        "a1_result": {"bytes": 1, "sha256": "a" * 64}
                    },
                },
                "content_sha256",
            )
        ),
        encoding="ascii",
    )
    return StrongSkip(variant=variant, status_path=path)


def test_strong_aggregation_deduplicates_bound_root_rows(tmp_path: Path) -> None:
    outputs = aggregate_strong_outputs(
        [_strong_study(tmp_path, "A1", 0.32), _strong_study(tmp_path, "A2", 0.31)]
    )

    curves = list(
        csv.DictReader(outputs["learning_curves.csv"].decode("ascii").splitlines())
    )
    per_seed = list(
        csv.DictReader(
            outputs["official_like_per_seed.csv"].decode("ascii").splitlines()
        )
    )
    manifest = json.loads(outputs["variant_manifest.json"])
    verdict = json.loads(outputs["STRONG_LOCAL_VERDICT.json"])

    assert [row["variant"] for row in curves] == ["R1", "A1", "A2"]
    assert [row["variant"] for row in per_seed] == ["R1", "A1", "A2"]
    assert [row["variant"] for row in manifest["authorizations"]] == ["A1", "A2"]
    assert verdict["variants_run"] == ["A1", "A2"]
    assert verdict["full_training_authorized_variants"] == ["A1"]
    assert verdict["selection_used_persist4d"] is False
    assert "variants/A1/variant_manifest.json" in outputs
    assert "variants/A1/STRONG_LOCAL_VERDICT.json" in outputs
    assert "variants/A1/STRONG_LOCAL_PROVENANCE.json" in outputs
    assert "variants/A2/variant_manifest.json" in outputs


def test_strong_aggregation_rejects_conflicting_duplicate_baseline(
    tmp_path: Path,
) -> None:
    a1 = _strong_study(tmp_path, "A1", 0.32)
    a2 = _strong_study(tmp_path, "A2", 0.31)
    rows = list(
        csv.DictReader((a2.output_directory / "learning_curves.csv").open())
    )
    rows[0]["metric"] = "0.29"
    _write_csv(a2.output_directory / "learning_curves.csv", rows)

    with pytest.raises(FinalizationError, match="conflicting duplicate"):
        aggregate_strong_outputs([a1, a2])


def test_strong_aggregation_preserves_bound_full_training_result(
    tmp_path: Path,
) -> None:
    study = _with_strong_full(_strong_study(tmp_path, "A1", 0.32))

    outputs = aggregate_strong_outputs([study])

    verdict = json.loads(outputs["STRONG_LOCAL_VERDICT.json"])
    assert verdict["full_results"] == [
        {
            "variant": "A1",
            "status": "pass",
            "content_sha256": json.loads(
                outputs[
                    "variants/A1/full/STRONG_LOCAL_FULL_VERDICT.json"
                ]
            )["content_sha256"],
            "verdict": "STRONG-LOCAL-CONFIRMED",
        }
    ]
    assert "variants/A1/full/FULL_TRAINING_MANIFEST.json" in outputs
    assert "variants/A1/full/selected_checkpoint_manifest.json" in outputs
    assert "variants/A1/full/STRONG_LOCAL_FULL_PROVENANCE.json" in outputs


def test_strong_aggregation_rejects_unbound_full_checkpoint(tmp_path: Path) -> None:
    study = _with_strong_full(
        _strong_study(tmp_path, "A1", 0.32), checkpoint_sha256="b" * 64
    )

    with pytest.raises(FinalizationError, match="full checkpoint binding"):
        aggregate_strong_outputs([study])


def test_strong_aggregation_rejects_full_result_without_short_gate(
    tmp_path: Path,
) -> None:
    study = _with_strong_full(_strong_study(tmp_path, "A1", 0.31))

    with pytest.raises(FinalizationError, match="without short authorization"):
        aggregate_strong_outputs([study])


def test_strong_aggregation_records_signed_full_gate_skip(tmp_path: Path) -> None:
    outputs = aggregate_strong_outputs([_strong_study(tmp_path, "A1", 0.31)])

    status = json.loads(outputs["variants/A1/full/STATUS.json"])
    aggregate = json.loads(outputs["STRONG_LOCAL_VERDICT.json"])
    assert status["status"] == "gate_skipped"
    assert status["upstream_gate"] == "STRONG_LOCAL_VERDICT"
    assert status["short_verdict_content_sha256"] == aggregate["results"][0][
        "content_sha256"
    ]
    assert canonical_sha256(
        {key: value for key, value in status.items() if key != "content_sha256"}
    ) == status["content_sha256"]
    assert aggregate["full_statuses"] == [
        {
            "variant": "A1",
            "status": "gate_skipped",
            "content_sha256": status["content_sha256"],
        }
    ]


def test_strong_aggregation_preserves_signed_skipped_variant(tmp_path: Path) -> None:
    outputs = aggregate_strong_outputs(
        [_strong_study(tmp_path, "A1", 0.31)],
        [_strong_skip(tmp_path, "A2")],
    )

    manifest = json.loads(outputs["variant_manifest.json"])
    aggregate = json.loads(outputs["STRONG_LOCAL_VERDICT.json"])
    assert manifest["skips"][0]["variant"] == "A2"
    assert aggregate["variants_considered"] == ["A1", "A2"]
    assert aggregate["skipped_variants"][0]["reason"] == (
        "upstream structural gate did not pass"
    )
    assert "variants/A2/variant_manifest.json" in outputs


def test_strong_aggregation_rejects_tampered_skipped_variant(tmp_path: Path) -> None:
    skipped = _strong_skip(tmp_path, "A2")
    payload = json.loads(skipped.status_path.read_text(encoding="ascii"))
    payload["gate"]["reason"] = "changed"
    skipped.status_path.write_text(json.dumps(payload), encoding="ascii")

    with pytest.raises(FinalizationError, match="A2 skip content hash"):
        aggregate_strong_outputs(
            [_strong_study(tmp_path, "A1", 0.31)], [skipped]
        )


def _handoff() -> str:
    lines = ["# ReScene Task-Learning Root-Cause Handoff", ""]
    for index, title in enumerate(HANDOFF_SECTION_TITLES, start=1):
        lines.extend([f"## {index}. {title}", "", "Verified evidence.", ""])
    return "\n".join(lines)


def _external_file() -> dict[str, object]:
    return {
        "logical_name": "R0 epoch-90 checkpoint",
        "external_reference": "external:checkpoint/" + "a" * 64,
        "sha256": "a" * 64,
        "bytes": 754_813_736,
        "creating_commit": "b" * 40,
        "config_sha256": "c" * 64,
        "selected_epoch": 90,
        "selected_step": 5_940,
    }


def _artifact_tree(tmp_path: Path, *, full_skipped: bool = False) -> Path:
    root = tmp_path / "artifacts" / "rescene_task_learning_root_cause_v1"
    for relative in REQUIRED_ARTIFACTS:
        if full_skipped and relative.startswith("full_candidate/"):
            continue
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.name == "HANDOFF.md":
            path.write_text(_handoff(), encoding="ascii")
        elif path.name == "FINAL_REPORT.md":
            path.write_text(
                "# Final Report\n\nPrincipal outcome: `TLRC-YELLOW`\n",
                encoding="ascii",
            )
        elif path.suffix == ".json":
            path.write_text('{"status": "pass"}\n', encoding="ascii")
        elif path.suffix == ".csv":
            path.write_text("status\npass\n", encoding="ascii")
        else:
            path.write_text("# Evidence\n\nVerified.\n", encoding="ascii")
    if full_skipped:
        status = {
            "schema_version": 1,
            "status": "gate_skipped",
            "reason": "no short-curve candidate passed every gate",
            "upstream_gate": "ROOTCAUSE_SHORT_DECISION",
        }
        from utils.rescene_rootcause_preflight import canonical_sha256

        status["content_sha256"] = canonical_sha256(status)
        path = root / "full_candidate" / "STATUS.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(status), encoding="ascii")
    return root


def _write_manifest(root: Path) -> None:
    manifest = build_final_manifest(
        artifact_root=root,
        repository={
            "branch": "research/persist4d-rescene-task-learning-root-cause-v1",
            "start_commit": "1" * 40,
            "evidence_commit": "2" * 40,
            "head_reference": (
                "refs/heads/research/"
                "persist4d-rescene-task-learning-root-cause-v1"
            ),
            "remote_reference": (
                "refs/remotes/origin/research/"
                "persist4d-rescene-task-learning-root-cause-v1"
            ),
        },
        external_files=[_external_file()],
    )
    (root / "FINAL_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )


def _repository() -> dict[str, object]:
    return {
        "branch": "research/persist4d-rescene-task-learning-root-cause-v1",
        "start_commit": "1" * 40,
        "evidence_commit": "2" * 40,
        "head_reference": (
            "refs/heads/research/persist4d-rescene-task-learning-root-cause-v1"
        ),
        "remote_reference": (
            "refs/remotes/origin/research/"
            "persist4d-rescene-task-learning-root-cause-v1"
        ),
    }


def _finalization_inputs(tmp_path: Path) -> FinalPackageInputs:
    root = _artifact_tree(tmp_path, full_skipped=True)
    for relative in (
        "FINAL_REPORT.md",
        "HANDOFF.md",
        "full_candidate/STATUS.json",
        "short_curves/learning_curves.csv",
        "short_curves/official_like_epoch60.csv",
        "short_curves/official_like_epoch90.csv",
        "short_curves/rootcause_per_seed.csv",
        "short_curves/rootcause_summary.csv",
        "short_curves/ROOTCAUSE_SHORT_DECISION.md",
        "decoder_diagnostics/query_initialization.csv",
        "decoder_diagnostics/query_conflicts.csv",
        "decoder_diagnostics/attention_mask_recall.csv",
        "decoder_diagnostics/superpoint_features.csv",
        "decoder_diagnostics/DECODER_DIAGNOSTICS.md",
        "strong_local/variant_manifest.json",
        "strong_local/learning_curves.csv",
        "strong_local/official_like_per_seed.csv",
        "strong_local/STRONG_LOCAL_VERDICT.md",
    ):
        (root / relative).unlink()

    short = tmp_path / "short"
    short.mkdir()
    for name in (
        "learning_curves.csv",
        "official_like_epoch60.csv",
        "official_like_epoch90.csv",
        "rootcause_per_seed.csv",
        "rootcause_summary.csv",
    ):
        (short / name).write_text("status\ngate_skipped\n", encoding="ascii")
    decision = _signed(
        {
            "schema_version": 1,
            "status": "pass",
            "selected_variant": None,
            "full_training_authorized": False,
            "full_training_status": "gate_skipped",
        },
        "content_sha256",
    )
    (short / "ROOTCAUSE_SHORT_DECISION.json").write_text(
        json.dumps(decision), encoding="ascii"
    )
    (short / "ROOTCAUSE_SHORT_DECISION.md").write_text(
        "# Short Decision\n\nStatus: `gate_skipped`\n", encoding="ascii"
    )
    (short / "ROOTCAUSE_SHORT_PROVENANCE.json").write_text(
        json.dumps(
            _signed(
                {
                    "schema_version": 1,
                    "decision_content_sha256": decision["content_sha256"],
                },
                "content_sha256",
            )
        ),
        encoding="ascii",
    )

    diagnostics = tmp_path / "diagnostics"
    diagnostics.mkdir()
    for name in (
        "query_initialization.csv",
        "query_conflicts.csv",
        "attention_mask_recall.csv",
        "superpoint_features.csv",
    ):
        (diagnostics / name).write_text("status\npass\n", encoding="ascii")
    (diagnostics / "DECODER_DIAGNOSTICS.md").write_text(
        "# Decoder Diagnostics\n\nVerified.\n", encoding="ascii"
    )
    (diagnostics / "DECODER_DIAGNOSTICS.json").write_text(
        json.dumps(
            _signed(
                {"schema_version": 1, "status": "pass"}, "content_sha256"
            )
        ),
        encoding="ascii",
    )

    final_report = tmp_path / "FINAL_REPORT.md"
    final_report.write_text(
        "# Final Report\n\nPrincipal outcome: `TLRC-RED`\n", encoding="ascii"
    )
    handoff = tmp_path / "HANDOFF.md"
    handoff.write_text(_handoff(), encoding="ascii")
    return FinalPackageInputs(
        artifact_root=root,
        short_directory=short,
        diagnostics_directory=diagnostics,
        strong_studies=(_strong_study(tmp_path, "A1", 0.31),),
        final_report_path=final_report,
        handoff_path=handoff,
        repository=_repository(),
        external_files=(_external_file(),),
        strong_skips=(_strong_skip(tmp_path, "A2"),),
    )


def test_finalizer_publishes_verified_package_and_signed_full_skip(
    tmp_path: Path,
) -> None:
    inputs = _finalization_inputs(tmp_path)

    result = publish_final_package(inputs)

    assert result["status"] == "pass"
    assert result["principal_outcome"] == "TLRC-RED"
    assert result["full_candidate_status"] == "gate_skipped"
    status = json.loads(
        (inputs.artifact_root / "full_candidate/STATUS.json").read_text(
            encoding="ascii"
        )
    )
    assert status["status"] == "gate_skipped"
    assert status["upstream_gate"] == "ROOTCAUSE_SHORT_DECISION"
    assert verify_final_artifacts(inputs.artifact_root)["status"] == "pass"


def test_finalizer_does_not_publish_partial_tree_when_input_is_missing(
    tmp_path: Path,
) -> None:
    inputs = _finalization_inputs(tmp_path)
    (inputs.short_directory / "rootcause_summary.csv").unlink()

    with pytest.raises(FinalizationError, match="unavailable"):
        publish_final_package(inputs)

    assert not (inputs.artifact_root / "FINAL_REPORT.md").exists()
    assert not (inputs.artifact_root / "FINAL_MANIFEST.json").exists()


def test_finalizer_requires_an_explicit_A2_run_or_skip(tmp_path: Path) -> None:
    inputs = replace(_finalization_inputs(tmp_path), strong_skips=())

    with pytest.raises(FinalizationError, match="A2 decision is missing"):
        publish_final_package(inputs)


def test_finalizer_rejects_short_provenance_from_a_different_decision(
    tmp_path: Path,
) -> None:
    inputs = _finalization_inputs(tmp_path)
    provenance_path = inputs.short_directory / "ROOTCAUSE_SHORT_PROVENANCE.json"
    provenance = json.loads(provenance_path.read_text(encoding="ascii"))
    provenance.pop("content_sha256")
    provenance["decision_content_sha256"] = "f" * 64
    provenance_path.write_text(
        json.dumps(_signed(provenance, "content_sha256")), encoding="ascii"
    )

    with pytest.raises(FinalizationError, match="short provenance binding"):
        publish_final_package(inputs)


def _authorized_full_inputs(
    tmp_path: Path,
    *,
    verdict_checkpoint_sha256: str = "a" * 64,
    verdict_training_sha256: str | None = None,
    provenance_verdict_sha256: str | None = None,
) -> FinalPackageInputs:
    inputs = _finalization_inputs(tmp_path)
    decision_path = inputs.short_directory / "ROOTCAUSE_SHORT_DECISION.json"
    decision = json.loads(decision_path.read_text(encoding="ascii"))
    decision.pop("content_sha256")
    decision.update(
        {
            "selected_variant": "R1",
            "full_training_authorized": True,
            "full_training_status": "authorized",
            "variant_authorization_sha256": "6" * 64,
        }
    )
    decision = _signed(decision, "content_sha256")
    decision_path.write_text(json.dumps(decision), encoding="ascii")
    (inputs.short_directory / "ROOTCAUSE_SHORT_PROVENANCE.json").write_text(
        json.dumps(
            _signed(
                {
                    "schema_version": 1,
                    "decision_content_sha256": decision["content_sha256"],
                },
                "content_sha256",
            )
        ),
        encoding="ascii",
    )
    inputs.final_report_path.write_text(
        "# Final Report\n\nPrincipal outcome: `TLRC-GREEN`\n", encoding="ascii"
    )

    training = tmp_path / "full-training"
    evaluation = tmp_path / "full-evaluation"
    training.mkdir()
    evaluation.mkdir()
    (training / "FULL_TRAINING_REPORT.md").write_text(
        "# Full Training\n", encoding="ascii"
    )
    training_manifest = _signed(
        {
            "schema_version": 1,
            "status": "pass",
            "variant": "R1",
            "variant_authorization_sha256": "6" * 64,
            "budget": {"completed_epoch": 450, "optimizer_steps": 29_700},
            "selection": {"selected_checkpoint_sha256": "a" * 64},
        },
        "content_sha256",
    )
    (training / "FULL_TRAINING_MANIFEST.json").write_text(
        json.dumps(training_manifest), encoding="ascii"
    )
    (training / "learning_curve.csv").write_text(
        "status\npass\n", encoding="ascii"
    )
    (training / "checkpoint_inventory.csv").write_text(
        "status\npass\n", encoding="ascii"
    )
    selected = _signed(
        {
            "schema_version": 1,
            "status": "pass",
            "stage": "full_candidate",
            "variant": "R1",
            "checkpoint": {"sha256": "a" * 64},
            "bindings": {"variant_authorization_sha256": "6" * 64},
            "full_training": {
                "completed_epoch": 450,
                "manifest_sha256": training_manifest["content_sha256"],
            },
        },
        "content_sha256",
    )
    (training / "selected_checkpoint_manifest.json").write_text(
        json.dumps(selected), encoding="ascii"
    )
    for name in ("official_like_per_seed.csv", "official_like_summary.csv"):
        (evaluation / name).write_text("status\npass\n", encoding="ascii")
    verdict = _signed(
        {
            "schema_version": 1,
            "status": "pass",
            "variant": "R1",
            "verdict": "ROOTCAUSE-CONFIRMED",
            "verdict_prefix": "ROOTCAUSE",
            "selected_checkpoint_sha256": verdict_checkpoint_sha256,
            "checkpoint_manifest_sha256": selected["content_sha256"],
            "full_training_manifest_sha256": (
                verdict_training_sha256 or training_manifest["content_sha256"]
            ),
        },
        "content_sha256",
    )
    (evaluation / "ROOT_CAUSE_FULL_VERDICT.json").write_text(
        json.dumps(verdict), encoding="ascii"
    )
    (evaluation / "ROOT_CAUSE_FULL_VERDICT.md").write_text(
        "# Full Verdict\n", encoding="ascii"
    )
    (evaluation / "FULL_EVALUATION_PROVENANCE.json").write_text(
        json.dumps(
            _signed(
                {
                    "schema_version": 1,
                    "result_content_sha256": (
                        provenance_verdict_sha256 or verdict["content_sha256"]
                    ),
                },
                "content_sha256",
            )
        ),
        encoding="ascii",
    )
    return replace(
        inputs,
        full_training_directory=training,
        full_evaluation_directory=evaluation,
    )


def test_finalizer_rejects_full_verdict_from_a_different_checkpoint(
    tmp_path: Path,
) -> None:
    inputs = _authorized_full_inputs(
        tmp_path, verdict_checkpoint_sha256="b" * 64
    )

    with pytest.raises(FinalizationError, match="checkpoint binding"):
        publish_final_package(inputs)


def test_finalizer_rejects_full_verdict_from_a_different_training_manifest(
    tmp_path: Path,
) -> None:
    inputs = _authorized_full_inputs(
        tmp_path, verdict_training_sha256="f" * 64
    )

    with pytest.raises(FinalizationError, match="training manifest binding"):
        publish_final_package(inputs)


def test_finalizer_rejects_full_provenance_from_a_different_verdict(
    tmp_path: Path,
) -> None:
    inputs = _authorized_full_inputs(
        tmp_path, provenance_verdict_sha256="f" * 64
    )

    with pytest.raises(FinalizationError, match="full provenance binding"):
        publish_final_package(inputs)


def test_finalizer_publishes_completed_full_candidate(tmp_path: Path) -> None:
    inputs = _authorized_full_inputs(tmp_path)

    result = publish_final_package(inputs)

    assert result["principal_outcome"] == "TLRC-GREEN"
    assert result["full_candidate_status"] == "completed"
    assert (
        inputs.artifact_root / "full_candidate/FULL_TRAINING_MANIFEST.json"
    ).is_file()
    assert (
        inputs.artifact_root / "full_candidate/FULL_EVALUATION_PROVENANCE.json"
    ).is_file()


def _finalization_spec(inputs: FinalPackageInputs) -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_root": str(inputs.artifact_root),
        "short_directory": str(inputs.short_directory),
        "diagnostics_directory": str(inputs.diagnostics_directory),
        "strong_studies": [
            {
                "variant": study.variant,
                "authorization_path": str(study.authorization_path),
                "output_directory": str(study.output_directory),
                **(
                    {"full_directory": str(study.full_directory)}
                    if study.full_directory is not None
                    else {}
                ),
            }
            for study in inputs.strong_studies
        ],
        "strong_skips": [
            {
                "variant": skip.variant,
                "status_path": str(skip.status_path),
            }
            for skip in inputs.strong_skips
        ],
        "final_report_path": str(inputs.final_report_path),
        "handoff_path": str(inputs.handoff_path),
        "repository": inputs.repository,
        "external_files": list(inputs.external_files),
    }


def test_finalizer_cli_loads_explicit_external_spec(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    inputs = _finalization_inputs(tmp_path)
    spec_path = tmp_path / "finalization-spec.json"
    spec_path.write_text(json.dumps(_finalization_spec(inputs)), encoding="ascii")

    assert finalization_main(["--spec", str(spec_path)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "pass"
    assert result["principal_outcome"] == "TLRC-RED"


def test_finalizer_script_entrypoint_runs_complete_spec(tmp_path: Path) -> None:
    inputs = _finalization_inputs(tmp_path)
    spec_path = tmp_path / "finalization-spec.json"
    spec_path.write_text(json.dumps(_finalization_spec(inputs)), encoding="ascii")
    project_root = Path(__file__).resolve().parents[1]

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/finalize_rescene_task_learning_root_cause.py",
            "--spec",
            str(spec_path),
        ],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["status"] == "pass"


def test_final_artifact_verifier_hashes_tree_and_accepts_complete_handoff(
    tmp_path: Path,
) -> None:
    root = _artifact_tree(tmp_path)
    _write_manifest(root)

    result = verify_final_artifacts(root)

    assert result["status"] == "pass"
    assert result["principal_outcome"] == "TLRC-YELLOW"
    assert result["handoff_section_count"] == 30
    assert result["external_file_count"] == 1
    assert result["artifact_count"] == len(
        [path for path in root.rglob("*") if path.is_file()]
    ) - 1


def test_final_artifact_verifier_accepts_signed_full_gate_skip(tmp_path: Path) -> None:
    root = _artifact_tree(tmp_path, full_skipped=True)
    _write_manifest(root)

    result = verify_final_artifacts(root)

    assert result["full_candidate_status"] == "gate_skipped"


def test_final_manifest_accepts_pretraining_external_state(tmp_path: Path) -> None:
    root = _artifact_tree(tmp_path)
    external = _external_file()
    external.update(
        {
            "logical_name": "common initialization state",
            "selected_epoch": 0,
            "selected_step": 0,
        }
    )

    manifest = build_final_manifest(
        artifact_root=root,
        repository={
            "branch": "research/persist4d-rescene-task-learning-root-cause-v1",
            "start_commit": "1" * 40,
            "evidence_commit": "2" * 40,
            "head_reference": (
                "refs/heads/research/"
                "persist4d-rescene-task-learning-root-cause-v1"
            ),
            "remote_reference": (
                "refs/remotes/origin/research/"
                "persist4d-rescene-task-learning-root-cause-v1"
            ),
        },
        external_files=[external],
    )

    assert manifest["external_files"][0]["selected_step"] == 0


def test_final_artifact_verifier_rejects_private_paths(tmp_path: Path) -> None:
    root = _artifact_tree(tmp_path)
    _write_manifest(root)
    (root / "CODE_AUDIT.md").write_text(
        "checkpoint: /home/researcher/run/model.ckpt\n", encoding="ascii"
    )

    with pytest.raises(FinalArtifactError, match="private path"):
        verify_final_artifacts(root)


def test_final_artifact_verifier_rejects_private_mnt_paths(
    tmp_path: Path,
) -> None:
    root = _artifact_tree(tmp_path)
    _write_manifest(root)
    (root / "CODE_AUDIT.md").write_text(
        "checkpoint: /mnt/node8-persist4d/run/model.ckpt\n", encoding="ascii"
    )

    with pytest.raises(FinalArtifactError, match="private path"):
        verify_final_artifacts(root)


def test_final_artifact_verifier_rejects_changed_artifact(tmp_path: Path) -> None:
    root = _artifact_tree(tmp_path)
    _write_manifest(root)
    with (root / "CODE_AUDIT.md").open("a", encoding="ascii") as handle:
        handle.write("changed after manifest\n")

    with pytest.raises(FinalArtifactError, match="identity differs"):
        verify_final_artifacts(root)


def test_final_artifact_verifier_rejects_missing_handoff_section(
    tmp_path: Path,
) -> None:
    root = _artifact_tree(tmp_path)
    _write_manifest(root)
    handoff = root / "HANDOFF.md"
    handoff.write_text(
        handoff.read_text(encoding="ascii").replace(
            "## 17. Query initialization diagnostics\n", ""
        ),
        encoding="ascii",
    )
    with pytest.raises(FinalArtifactError, match="HANDOFF sections differ"):
        verify_final_artifacts(root)
