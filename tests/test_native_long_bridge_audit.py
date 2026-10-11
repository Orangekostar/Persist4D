import pytest


def test_reference_bootstrap_pools_counts_and_keeps_zero_denominators_missing():
    from scripts.native_long_bridge_audit import paired_count_bootstrap

    a = [{'reference_id': 'a', 'n': 1, 'd': 1}, {'reference_id': 'b', 'n': 0, 'd': 9}]
    b = [{**r, 'n': 0} for r in a]
    result = paired_count_bootstrap(a, b, 'n', 'd', replicates=100)
    assert result['point_delta'] == .1
    assert result['valid_replicates'] == 100
    assert result == paired_count_bootstrap(a, b, 'n', 'd', replicates=100)
    empty = [{**r, 'n': 0, 'd': 0} for r in a]
    missing = paired_count_bootstrap(empty, empty, 'n', 'd', replicates=100)
    assert missing['point_delta'] is None and missing['ci95'] is None
    with pytest.raises(ValueError, match='reference'):
        paired_count_bootstrap(a, b[:1], 'n', 'd', replicates=100)


def test_identity_assignment_recompute_detects_altered_aggregate_counts():
    from scripts.native_long_bridge_audit import (
        restore_identity_updates,
        verify_identity_rows,
    )
    from scripts.system_comparison_metrics import compute_deployment_identity_metrics

    raw = [{'horizon': 1, 'visible_gt_ids': [10], 'assignments': {'10': 2}},
           {'horizon': 2, 'visible_gt_ids': [10], 'assignments': {'10': 3}}]
    updates = restore_identity_updates(raw)
    expected = compute_deployment_identity_metrics(updates)
    verify_identity_rows(expected, expected)
    assert expected['deployment_id_switches'] == 1
    with pytest.raises(ValueError, match='identity'):
        verify_identity_rows({**expected, 'deployment_id_switches': 0}, expected)
