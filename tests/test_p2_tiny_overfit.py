import json
import os
import subprocess
from pathlib import Path

import pytest

from scripts import run_p2_native_smoke as smoke
from utils import p2_preflight

REPO_ROOT = Path(__file__).resolve().parents[1]
JSON_REPORT = Path(
    os.environ.get(
        "P2_TINY_OVERFIT_JSON_REPORT",
        REPO_ROOT / "artifacts" / "P2" / "tiny_overfit_report.json",
    )
)
MARKDOWN_REPORT = Path(
    os.environ.get(
        "P2_TINY_OVERFIT_MARKDOWN_REPORT",
        REPO_ROOT / "artifacts" / "P2" / "tiny_overfit_report.md",
    )
)


def _git_nul_paths(*args: str) -> list[str]:
    output = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    ).stdout
    return sorted(os.fsdecode(path) for path in output.split(b"\0") if path)


def _assert_artifact_source_tree_contract(payload: dict[str, object]) -> None:
    contract = payload["source_tree_contract"]
    source_commit = payload["source_commit"]
    assert contract["schema_version"] == 1
    assert contract["status"] == "pass"
    assert contract["source_commit"] == source_commit
    assert contract["observed_head"] == source_commit
    assert contract["generation_head_unchanged"] is True
    assert contract["allowed_dirty_prefixes"] == ["artifacts/P2/"]
    assert contract["committed_paths_since_source"] == []
    assert contract["disallowed_committed_paths"] == []
    assert contract["disallowed_dirty_paths"] == []
    assert contract["index_flag_paths"] == []
    assert len(contract["expected_tracked_tree_sha256"]) == 64
    assert set(contract["expected_tracked_tree_sha256"]) <= set("0123456789abcdef")
    assert (
        contract["observed_tracked_tree_sha256"]
        == contract["expected_tracked_tree_sha256"]
    )
    assert contract["errors"] == []
    assert all(
        path.startswith("artifacts/P2/")
        for path in contract["dirty_paths_before_generation"]
    )
    assert all(path.startswith("artifacts/P2/") for path in contract["dirty_paths"])

    current_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    current_contract = smoke._build_source_tree_contract(source_commit)
    smoke._require_passing_source_tree_contract(current_contract)
    assert current_contract["observed_head"] == current_head
    assert current_contract["index_flag_paths"] == []
    assert (
        current_contract["expected_tracked_tree_sha256"]
        == contract["expected_tracked_tree_sha256"]
    )
    assert (
        current_contract["observed_tracked_tree_sha256"]
        == contract["observed_tracked_tree_sha256"]
    )
    assert (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", source_commit, current_head],
            cwd=REPO_ROOT,
            check=False,
        ).returncode
        == 0
    )
    assert all(
        path.startswith("artifacts/P2/")
        for path in _git_nul_paths(
            "diff",
            "--name-only",
            "--no-renames",
            "-z",
            f"{source_commit}..{current_head}",
            "--",
        )
    )
    current_dirty_paths = sorted(
        set(_git_nul_paths("diff", "--name-only", "--no-renames", "-z", "--"))
        | set(
            _git_nul_paths(
                "diff",
                "--cached",
                "--name-only",
                "--no-renames",
                "-z",
                "--",
            )
        )
        | set(_git_nul_paths("ls-files", "--others", "--exclude-standard", "-z", "--"))
    )
    assert all(path.startswith("artifacts/P2/") for path in current_dirty_paths)


def _assert_artifact_runtime_source_contract(payload: dict[str, object]) -> None:
    current = smoke._build_runtime_source_contract()
    assert current["status"] == "pass"
    assert current["errors"] == []
    for record in current["repositories"].values():
        assert record["index_flag_paths"] == []
        assert len(record["expected_tracked_tree_sha256"]) == 64
        assert set(record["expected_tracked_tree_sha256"]) <= set("0123456789abcdef")
        assert (
            record["observed_tracked_tree_sha256"]
            == record["expected_tracked_tree_sha256"]
        )
    assert payload["runtime_source_contract"] == current


def _assert_artifact_runtime_environment_contract(
    payload: dict[str, object],
) -> None:
    current = smoke._build_runtime_environment_contract()
    validation_errors = []
    observed = p2_preflight._validate_runtime_environment_contract(
        {"runtime_environment_contract": current},
        validation_errors,
    )
    assert observed is current
    assert validation_errors == []
    assert payload["runtime_environment_contract"] == current


def _passing_history() -> list[dict[str, float]]:
    rows = []
    for step in range(1, smoke.TINY_OVERFIT_STEPS + 1):
        fraction = (step - 1) / (smoke.TINY_OVERFIT_STEPS - 1)
        rows.append(
            {
                "step": step,
                "final_head_segmentation": 8.0 - 7.5 * fraction,
                "aggregate_contrastive": 4.0 - 2.4 * fraction,
                "classification_accuracy": 0.5 + 0.5 * fraction,
                "mean_dice": 0.2 + 0.78 * fraction,
            }
        )
    return rows


def test_tiny_overfit_gate_contract_passes_calibrated_history() -> None:
    result = smoke.evaluate_tiny_overfit_gates(_passing_history())

    assert result["passed"] is True
    assert result["steps"] == 128
    assert result["gates"] == {
        "final_segmentation_median_ratio_le_0.25": True,
        "final_contrastive_ratio_le_0.50": True,
        "contrastive_positive_and_finite": True,
        "classification_accuracy_ge_0.75": True,
        "mean_dice_ge_0.90": True,
    }


@pytest.mark.parametrize(
    ("field", "value", "failed_gate"),
    [
        ("final_head_segmentation", 9.0, "final_segmentation_median_ratio_le_0.25"),
        ("aggregate_contrastive", 3.0, "final_contrastive_ratio_le_0.50"),
        ("aggregate_contrastive", 0.0, "contrastive_positive_and_finite"),
        ("classification_accuracy", 0.5, "classification_accuracy_ge_0.75"),
        ("mean_dice", 0.5, "mean_dice_ge_0.90"),
    ],
)
def test_tiny_overfit_gate_contract_rejects_failed_measurement(
    field: str, value: float, failed_gate: str
) -> None:
    history = _passing_history()
    if field == "final_head_segmentation":
        for row in history[-10:]:
            row[field] = value
    elif field == "aggregate_contrastive" and value == 0.0:
        history[63][field] = value
    else:
        history[-1][field] = value

    result = smoke.evaluate_tiny_overfit_gates(history)

    assert result["passed"] is False
    assert result["gates"][failed_gate] is False


def test_tiny_overfit_markdown_discloses_non_official_scope() -> None:
    result = smoke.evaluate_tiny_overfit_gates(_passing_history())
    markdown = smoke.render_tiny_overfit_markdown(
        result,
        sample_name=smoke.TINY_SAMPLE_NAME,
        elapsed_seconds=12.5,
        peak_vram_mib=1234.0,
    )

    assert "P2 Preflight-only Tiny Overfit" in markdown
    assert "not an official mixed-data reproduction" in markdown
    assert "not G2 evidence" in markdown
    assert smoke.TINY_SAMPLE_NAME in markdown
    assert "/" + "home" + "/" not in markdown


@pytest.mark.gpu
@pytest.mark.skipif(
    os.environ.get("P2_VERIFY_GPU_ARTIFACTS") != "1",
    reason="set P2_VERIFY_GPU_ARTIFACTS=1 after the real single-A40 run",
)
def test_real_tiny_overfit_artifacts_pass_all_gates() -> None:
    payload = json.loads(JSON_REPORT.read_text(encoding="utf-8"))
    markdown = MARKDOWN_REPORT.read_text(encoding="utf-8")

    _assert_artifact_source_tree_contract(payload)
    _assert_artifact_runtime_source_contract(payload)
    _assert_artifact_runtime_environment_contract(payload)
    assert payload["scope"] == "preflight-only"
    assert payload["verification_mode"] == "artifact_contract_not_reexecution"
    assert payload["official_mixed_data_reproduction"] is False
    assert payload["g2_evidence"] is False
    assert payload["sample_name"] == smoke.TINY_SAMPLE_NAME
    assert payload["steps"] == smoke.TINY_OVERFIT_STEPS
    assert payload["passed"] is True
    assert payload["encoder_bitwise_unchanged"] is True
    assert payload["decoder_head_changed"] is True
    provenance = payload["input_provenance"]
    assert provenance["dataset"] == "3RScan"
    assert provenance["sample_name"] == smoke.TINY_SAMPLE_NAME
    assert provenance["resolved_composed_config"] == {
        "format": "canonical-json-sort-keys-v1",
        "portable_references": True,
        "serialized_bytes": 9420,
        "sha256": "c04291fd18ac761e44d545e615639c88054cd625d5af96ece09dd5b70c03eec6",
    }
    assert len(provenance["processed_point_clouds"]) == 2
    assert len(provenance["instance_ground_truth"]) == 2
    assert provenance["change_ground_truth"]["sha256"] == (
        "75baf0a2d41956bd7d8c27b2a4257f5b8e606ab7a43a3436cc1bce07cbe0003c"
    )
    assert provenance["sequence_database"]["sha256"] == (
        "974299916db02d2ee0233564eb7b36e314dada93e997584d9dfef21ad70d0416"
    )
    assert "not an official mixed-data reproduction" in markdown
    assert "not G2 evidence" in markdown
    serialized = json.dumps(payload, sort_keys=True)
    assert "/" + "home" + "/" not in serialized
    assert "GPU" + "-" not in serialized
