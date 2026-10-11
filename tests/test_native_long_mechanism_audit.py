import pytest

from scripts.native_long_mechanism_audit import (
    candidate_delta,
    nested_candidate_trace,
    pair_alignment,
    summarize_sampling,
    validate_gradients,
)


def test_pair_alignment_preserves_upstream_drift_as_confounded_measurement():
    from copy import deepcopy

    active = {'input_sha256': {'coords': 'raw'}, 'target_sha256': {'masks': 'gt'},
              'feedback': [{'input_sha256': {'features': 'same'}, 'update_norm': 1.}],
              'stage_sha256': [{'logits': 'same', 'masks': 'same'} for _ in range(13)]}
    off = deepcopy(active)
    off['stage_sha256'][8]['masks'] = 'expected-intervention-change'
    assert pair_alignment(active, off)['attribution_eligible']
    off['stage_sha256'][0]['masks'] = 'upstream-drift'
    off['feedback'][0]['input_sha256']['features'] = 'upstream-drift'
    result = pair_alignment(active, off)
    assert result['status'] == 'RUNTIME_CONFOUNDED'
    assert not result['attribution_eligible']
    assert result['changed_upstream_stages'] == [0]
    assert result['input_exact'] and result['target_exact']
    off['target_sha256']['masks'] = 'different-gt'
    with pytest.raises(ValueError, match='input, target or capture'):
        pair_alignment(active, off)


def test_pair_alignment_requires_computed_feedback_and_all_captures_exact():
    from copy import deepcopy

    active = {'input_sha256': {}, 'target_sha256': {},
              'feedback': [{'input_sha256': {}, 'update_norm': 1.}],
              'stage_sha256': [{} for _ in range(13)]}
    off = deepcopy(active)
    off['feedback'][0]['update_norm'] = 1.00000001
    assert not pair_alignment(active, off)['attribution_eligible']
    off['feedback'] = []
    with pytest.raises(ValueError, match='input, target or capture'):
        pair_alignment(active, off)


def candidate(identity, matched, query=7, label=2):
    return {'gt_id': identity, 'gt_class': label, 'matched': matched,
            'best_query_id': query, 'high_iou_candidates': int(matched), 'class_candidates': 1}


def test_candidate_delta_reports_losses_as_well_as_gains():
    control = [candidate(1, True), candidate(2, False), candidate(3, True)]
    active = [candidate(1, False), candidate(2, True), candidate(3, True)]
    assert candidate_delta(active, control) == {
        'gt_objects': 3, 'matched_with_feedback': 2, 'matched_without_feedback': 2,
        'new_high_iou_objects': [2], 'lost_high_iou_objects': [1], 'preserved_high_iou_objects': [3],
        'high_iou_candidate_count_delta': 0}
    with pytest.raises(ValueError, match='ground truth'):
        candidate_delta([candidate(1, True, label=3)], [candidate(1, True)])


def test_nested_absence_and_class_change_do_not_count_as_failure_or_query_switch():
    rows = [[candidate(1, True, 7)], [], [candidate(1, False, 7)],
            [candidate(1, True, 8)], [candidate(1, True, 9, label=3)]]
    result = nested_candidate_trace(rows)
    item = result[0]
    assert item['statuses'] == ['matched', 'absent', 'missed', 'matched', 'matched']
    assert item['first_observed_failure_horizon'] == 3
    assert item['recovered_after_visible_miss'] == 1
    assert item['class_change_count'] == 1
    assert item['matched_query_changes'] == 0
    assert item['matched_query_transition_opportunities'] == 0


def test_sampling_summary_preserves_denominators_and_validates_cap():
    row = {'execution_stage': 0, 'cap': 2, 'available': 100, 'selected': 2,
           'cap_triggered': True, 'stages': [0, 1], 'available_per_stage': [1, 99],
           'uniform': {'counts': [0, 2], 'missing_available_stages': 1},
           'stratified': {'counts': [1, 1], 'missing_available_stages': 0}}
    result = summarize_sampling([row])
    assert result['available_stage_opportunities'] == 2
    assert result['uniform_missing'] == 1 and result['stratified_missing'] == 0
    assert result['uniform_missing_past'] == 1 and result['stratified_missing_past'] == 0
    assert result['uniform_mean_stage_share_spread'] == 1
    assert result['stratified_mean_stage_share_spread'] == 0
    with pytest.raises(ValueError, match='sampling'):
        summarize_sampling([{**row, 'selected': 3}])


def test_gradient_audit_accepts_float_summation_roundoff_but_rejects_missing_or_zero_groups():
    groups = {'feedback': 'model.native_long_feedback.', 'mask_head': 'model.mask_embed_head.',
              'cross_attention': 'model.cross_attention.', 'self_attention': 'model.self_attention.',
              'decoder_ffn': 'model.ffn_attention.'}
    row = {'no_optimizer_step': True, 'parameters_unchanged': True,
           'frozen_encoder_gradients_absent': True, 'auxiliary_count': 12,
           'feedback_checkpoint_chunks': False, 'total_loss': 1.,
           'gradient_norms': {prefix + 'weight': .3 for prefix in groups.values()},
           'gradient_groups': {name: {'tensor_count': 1, 'l2': .30000000000000004} for name in groups}}
    validate_gradients(row)
    row['gradient_groups']['feedback']['tensor_count'] = 0
    with pytest.raises(ValueError, match='gradient group'):
        validate_gradients(row)
    row['gradient_groups']['feedback'] = {'tensor_count': 1, 'l2': 0.}
    with pytest.raises(ValueError, match='gradient group'):
        validate_gradients(row)
