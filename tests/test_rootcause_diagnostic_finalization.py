from __future__ import annotations

import pytest

from scripts.finalize_rescene_rootcause_diagnostics import (
    summarize_decoder_diagnostics,
)


def _tables(*, weak_superpoints: bool, conflict: bool, starvation: bool):
    return {
        "query_initialization": [
            {
                "record_type": "scene_summary",
                "file_name": "scene0",
                "num_queries": "100",
                "foreground_query_fraction": "0.4",
                "background_query_fraction": "0.6",
                "gt_instance_count": "2",
                "gt_instance_coverage": "0.5",
                "query_content_norm_mean": "0.0",
                "query_content_norm_max": "0.0",
                "query_content_zero_fraction": "1.0",
                "gt_instance_id": "",
                "gt_label": "",
                "size_points": "",
                "size_bin": "",
                "query_count": "",
                "covered_by_fps_query": "",
            },
            {
                "record_type": "gt_instance",
                "file_name": "scene0",
                "num_queries": "",
                "foreground_query_fraction": "",
                "background_query_fraction": "",
                "gt_instance_count": "",
                "gt_instance_coverage": "",
                "query_content_norm_mean": "",
                "query_content_norm_max": "",
                "query_content_zero_fraction": "",
                "gt_instance_id": "1",
                "gt_label": "2",
                "size_points": "50",
                "size_bin": "small_lt100",
                "query_count": "0",
                "covered_by_fps_query": "False",
            },
        ],
        "query_conflicts": [
            {
                "file_name": "scene0",
                "decoder_prediction_layer": "0",
                "feeds_next_attention": "True",
                "query_count": "100",
                "active_query_count": "60",
                "gt_instance_count": "2",
                "gt_coverage_iou25": "0.5",
                "gt_coverage_iou50": "0.25",
                "mean_queries_per_gt_iou25": "2.5" if conflict else "1.0",
                "mean_queries_per_gt_iou50": "1.0",
                "competed_active_query_fraction": "0.3" if conflict else "0.05",
                "competing_query_pairwise_iou_mean": "0.6",
                "distinct_gt_covered_iou25": "1",
                "query_utilization_iou25": "0.2",
                "distinct_gt_per_utilized_query": "0.1",
            }
        ],
        "attention_mask_recall": [
            {
                "file_name": "scene0",
                "decoder_prediction_layer": "0",
                "query_id": "1",
                "gt_instance_id": "1",
                "match_iou": "0.4",
                "gt_point_count": "50",
                "allowed_gt_fraction": "0.2" if starvation else "0.8",
                "masked_gt_fraction": "0.8" if starvation else "0.2",
                "post_sample_all_masked_reset_count": "20" if starvation else "1",
                "post_sample_query_count": "100",
                "post_sample_reset_fraction": "0.2" if starvation else "0.01",
            }
        ],
        "superpoint_features": [
            {
                "file_name": "scene0",
                "gt_instance_id": "1",
                "gt_label": "2",
                "size_points": "50",
                "size_bin": "small_lt100",
                "segments_per_gt": "3",
                "within_instance_feature_variance": "0.08"
                if weak_superpoints
                else "0.01",
                "nearest_instance_cosine_margin": "0.05" if weak_superpoints else "0.4",
                "mean_segment_purity": "0.8" if weak_superpoints else "0.98",
                "mean_gt_instances_per_segment": "1.2" if weak_superpoints else "1.0",
            }
        ],
    }


def test_diagnostic_summary_preregisters_evidence_gates_without_starting_a2() -> None:
    result = summarize_decoder_diagnostics(
        _tables(weak_superpoints=True, conflict=True, starvation=True)
    )

    assert result["query_initialization"]["query_content_zero_fraction_mean"] == 1.0
    assert result["query_initialization"]["small_lt100_coverage"] == 0.0
    assert result["gates"]["A1"]["authorized"] is True
    assert result["gates"]["A2"]["diagnostic_evidence_pass"] is True
    assert result["gates"]["A2"]["authorized"] is False
    assert result["gates"]["A2"]["status"] == "pending_A1_result"
    assert result["gates"]["query_competition_design"]["supported"] is True
    assert result["gates"]["attention_relaxation_design"]["supported"] is True


def test_diagnostic_summary_closes_unsupported_high_risk_gates() -> None:
    result = summarize_decoder_diagnostics(
        _tables(weak_superpoints=False, conflict=False, starvation=False)
    )

    assert result["superpoint_features"]["within_variance_mean"] == pytest.approx(0.01)
    assert result["gates"]["A2"]["diagnostic_evidence_pass"] is False
    assert result["gates"]["query_competition_design"]["supported"] is False
    assert result["gates"]["attention_relaxation_design"]["supported"] is False
