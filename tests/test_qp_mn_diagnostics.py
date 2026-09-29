import importlib

import pytest
import torch


def api():
    return importlib.import_module('scripts.qp_mn_diagnostics')


def test_ranking_reports_destroyed_pairs_and_preserves_unknown_default():
    target = torch.tensor([.9, .1, .5, .0])
    parent = torch.tensor([.9, .1, .5, .8])
    current = torch.tensor([-.9, -.1, -.5, -.8])
    rows = api().ranking_counts(target, parent, current, torch.tensor([True, True, True, False]), torch.ones(4, dtype=torch.long))
    assert rows == [{'class_id': 1, 'comparable_pairs': 3, 'inversions': 3,
                     'ties': 0, 'original_correct_destroyed': 3, 'original_correct_tied': 0}]
    labels = {'temporal': target, 'status': ['MATCHED','VALID_NEGATIVE','AMBIGUOUS','IGNORE']}
    assert torch.equal(api().gt_quality_scores(parent, labels), torch.tensor([.9,.1,.5,.8]))


def test_loose_reachability_keeps_forced_errors_and_rejects_nonmonotone_bounds():
    lower = torch.tensor([True, False, False, False])
    upper = torch.tensor([True, True, True, False])
    original = torch.tensor([True, False, True, False])
    gt = torch.tensor([False, True, False, True])
    result = api().loose_reachable(lower, upper, original, gt)
    assert torch.equal(result, torch.tensor([True, True, False, False]))
    with pytest.raises(ValueError):
        api().loose_reachable(lower, upper, ~original, gt)
