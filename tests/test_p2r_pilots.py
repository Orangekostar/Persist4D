from __future__ import annotations

from scripts.p2r_pilots import choose_full_candidate, stratified_indices


def test_stratified_indices_include_population_endpoints() -> None:
    assert stratified_indices(10, 4) == (0, 3, 6, 9)
    assert stratified_indices(4, 4) == (0, 1, 2, 3)


def test_full_candidate_requires_pareto_improvement_over_control() -> None:
    results = {
        "P2R-0": {"metrics": {"t_mAP": 0.20, "stage1_mAP": 0.30, "stage2_mAP": 0.40}},
        "P2R-A": {"metrics": {"t_mAP": 0.22, "stage1_mAP": 0.31, "stage2_mAP": 0.41}},
        "P2R-B": {"metrics": {"t_mAP": 0.23, "stage1_mAP": 0.29, "stage2_mAP": 0.42}},
        "P2R-C": {"metrics": {"t_mAP": 0.21, "stage1_mAP": 0.31, "stage2_mAP": 0.41}},
    }

    decision = choose_full_candidate(results)

    assert decision["authorized"] is True
    assert decision["selected_variant"] == "P2R-A"
    assert decision["dominating_variants"] == ["P2R-A", "P2R-C"]


def test_full_candidate_is_not_authorized_without_pareto_dominance() -> None:
    results = {
        "P2R-0": {"metrics": {"t_mAP": 0.20, "stage1_mAP": 0.30, "stage2_mAP": 0.40}},
        "P2R-A": {"metrics": {"t_mAP": 0.21, "stage1_mAP": 0.29, "stage2_mAP": 0.41}},
        "P2R-B": {"metrics": {"t_mAP": 0.19, "stage1_mAP": 0.31, "stage2_mAP": 0.41}},
        "P2R-C": {"metrics": {"t_mAP": 0.20, "stage1_mAP": 0.30, "stage2_mAP": 0.40}},
    }

    decision = choose_full_candidate(results)

    assert decision["authorized"] is False
    assert decision["selected_variant"] is None
