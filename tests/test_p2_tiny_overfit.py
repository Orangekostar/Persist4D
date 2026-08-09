import json
import os
from pathlib import Path

import pytest

from scripts import run_p2_native_smoke as smoke

REPO_ROOT = Path(__file__).resolve().parents[1]
JSON_REPORT = REPO_ROOT / "artifacts" / "P2" / "tiny_overfit_report.json"
MARKDOWN_REPORT = REPO_ROOT / "artifacts" / "P2" / "tiny_overfit_report.md"


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

    assert payload["scope"] == "preflight-only"
    assert payload["official_mixed_data_reproduction"] is False
    assert payload["g2_evidence"] is False
    assert payload["sample_name"] == smoke.TINY_SAMPLE_NAME
    assert payload["steps"] == smoke.TINY_OVERFIT_STEPS
    assert payload["passed"] is True
    assert payload["encoder_bitwise_unchanged"] is True
    assert payload["decoder_head_changed"] is True
    assert "not an official mixed-data reproduction" in markdown
    assert "not G2 evidence" in markdown
    serialized = json.dumps(payload, sort_keys=True)
    assert "/" + "home" + "/" not in serialized
    assert "GPU" + "-" not in serialized
