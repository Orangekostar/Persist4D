import pytest
import torch


def test_panel_uses_all_cal_t5_and_only_causal_prefixes():
    from scripts.native_long_mechanism import build_panel

    records = [{'input_id': f'CAL:{i}', 'horizon': 5, 'reference_id': str(i // 3),
                'domain': 'rio', 'scan_ids': list('abcde'), 'scan_indices': list(range(5))}
               for i in range(6)]
    panel = build_panel({'evaluation_inputs': {'CAL': records}})
    assert len(panel) == 30
    for i, row in enumerate(panel):
        h = i % 5 + 1
        assert row['scan_ids'] == list('abcde')[:h]
        assert row['horizon'] == h
        assert row['panel_unit'] == f'CAL:{i // 5}'
    with pytest.raises(ValueError, match='six'):
        build_panel({'evaluation_inputs': {'CAL': records[:-1]}})
    records[0]['input_id'] = 'SEL:0'
    with pytest.raises(ValueError, match='CAL'):
        build_panel({'evaluation_inputs': {'CAL': records}})


def test_feedback_off_executes_trained_block_but_preserves_input_and_weights():
    from scripts.native_long_mechanism import feedback_capture

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(2.))
            self.calls = 0

        def forward(self, queries, features, padding):
            self.calls += 1
            return features + queries.mean(1, keepdim=True) * self.weight

    block = Block()
    q, f, pad = torch.ones(1, 2, 3), torch.ones(1, 4, 3), torch.zeros(1, 4, dtype=torch.bool)
    before = f.clone()
    rows = []
    with feedback_capture(block, 'off', rows):
        result = block(q, f, pad)
    assert torch.equal(result, before) and torch.equal(f, before)
    assert block.calls == 1 and block.weight.item() == 2
    assert rows[0]['update_norm'] > 0
    assert not torch.equal(block(q, f, pad), before)
    assert not block._forward_hooks


def test_candidate_diagnostic_preserves_missing_classes_and_empty_targets():
    from scripts.native_long_mechanism import candidate_diagnostics

    prediction = {'pred_masks': torch.tensor([[1, 0], [1, 0], [0, 1]], dtype=torch.bool),
                  'pred_classes': torch.tensor([2, 9]), 'source_query_ids': torch.tensor([8, 4])}
    target = {'ids': torch.tensor([10, 20, 30]), 'labels': torch.tensor([2, 3, 2]),
              'masks': torch.tensor([[1, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=torch.bool)}
    rows = candidate_diagnostics(prediction, target)
    assert len(rows) == 2
    assert rows[0]['best_iou'] == 1 and rows[0]['best_query_id'] == 8
    assert rows[0]['high_iou_candidates'] == 1
    assert rows[1]['best_iou'] == 0 and rows[1]['best_query_id'] is None
    assert rows[1]['class_candidates'] == 0
    assert candidate_diagnostics(prediction, {k: v[:0] for k, v in target.items()}) == []


def test_undefined_single_input_metric_is_missing_and_not_zero():
    from types import SimpleNamespace

    from scripts.native_long_mechanism import nullable_single_input_metrics

    raw = {'val_mean_t-' + k: torch.tensor(float('nan'))
           for k in ('AP', 'AP_50', 'AP_25', 'REC', 'REC_50', 'REC_25')}
    raw['val_mean_t-AP_50'] = torch.tensor(.5)
    metrics = nullable_single_input_metrics(SimpleNamespace(_metric=SimpleNamespace(compute=lambda: raw)))
    assert metrics['online_t-mAP'] is None
    assert metrics['online_t-mAP50'] == .5


def test_sampling_capture_preserves_uncapped_and_hierarchical_native_calls():
    from models.rescene import ReScene
    from scripts.native_long_mechanism import capture_sampler, sampling_panel

    model = ReScene.__new__(ReScene)
    torch.nn.Module.__init__(model)
    model.max_sample_size = False
    model.native_long_sampling = False
    model.native_long_sampling_seed = 45
    model.native_long_draw_ids = [0]
    model.native_long_sampling_stats = []
    features = [torch.arange(30.).reshape(15, 2)]
    stages = [torch.arange(5).repeat_interleave(3)]
    rows = []
    wrapped = capture_sampler(model.sample_and_batch_features, rows)
    expected = model.sample_and_batch_features(features)
    assert all(torch.equal(a, b) for a, b in zip(wrapped(features), expected))
    assert rows == []
    kwargs = {'curr_sample_max': 8, 'is_eval': True, 'extra': [features],
              'stage_ids': stages, 'sampling_stage_idx': 0}
    expected = model.sample_and_batch_features(features, **kwargs)
    actual = wrapped(features, **kwargs)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))
    assert len(rows) == 1 and rows[0][:2] == (0, 8)
    panel = sampling_panel(rows, 'CAL:test')
    assert panel[0]['cap_triggered'] and panel[0]['selected'] == 8
    assert panel[0]['stratified']['counts'] == [2, 2, 2, 1, 1]
