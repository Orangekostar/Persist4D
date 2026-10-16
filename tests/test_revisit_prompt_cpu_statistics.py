"""CPU bootstrap acceleration must preserve the original official arithmetic."""

import importlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch
from stmetrics.instances import ap_math, evaluator


def _acceleration():
    name = 'scripts.revisit_prompt_cpu_statistics'
    assert importlib.util.find_spec(name) is not None, 'CPU statistics accelerator is missing'
    return importlib.import_module(name)


def test_reciprocal_multiplication_keeps_original_float32_rounding():
    acceleration = _acceleration()
    truth = torch.tensor([1., 1., 1., 0., 0., 0., 0.])
    scores = torch.full((7,), .5)
    precision, recall = acceleration.precision_recall_cpu(truth, scores, torch.tensor(0.), 'cpu')
    # The official scalar expression is reciprocal(7) * 3, not rounded 3 / 7.
    assert precision.tolist() == [0.4285714626312256, 1.]
    assert recall.tolist() == [1., 0.]


@pytest.mark.parametrize('length', [0, 1, 2, 17, 128, 1024])
@pytest.mark.parametrize('hard_fn', [0., 7.])
def test_cpu_curves_match_official_bit_for_bit(length, hard_fn):
    acceleration = _acceleration()
    generator = torch.Generator().manual_seed(146 + length)
    truth = torch.randint(0, 2, (length,), generator=generator).float()
    scores = torch.randint(0, 11, (length,), generator=generator).float() / 10
    false_negatives = torch.tensor(hard_fn)
    expected = ap_math.compute_precision_recall(truth, scores, false_negatives, 'cpu')
    actual = acceleration.precision_recall_cpu(truth, scores, false_negatives, 'cpu')
    for got, want in zip(actual, expected):
        assert got.dtype == want.dtype and got.device == want.device
        assert torch.equal(got, want)


def test_float64_inputs_keep_original_scalar_semantics():
    acceleration = _acceleration()
    truth = torch.tensor([1., 0., 1.], dtype=torch.float64)
    scores = torch.tensor([.1, .9, .9], dtype=torch.float64)
    false_negatives = torch.tensor(3., dtype=torch.float64)
    expected = ap_math.compute_precision_recall(truth, scores, false_negatives, 'cpu')
    actual = acceleration.precision_recall_cpu(truth, scores, false_negatives, 'cpu')
    assert all(torch.equal(got, want) for got, want in zip(actual, expected))


def _reference_states():
    from scripts.p6a_metrics import OfficialMetricAccumulator

    taxonomy = Path(__file__).resolve().parents[1] / 'artifacts/native_long_postselection_v1/deployment/rio.yaml'
    states = []
    for objects, predicted in ((1, 1), (3, 1)):
        masks = torch.zeros(objects, objects * 300, dtype=torch.bool)
        for i in range(objects):
            masks[i, i * 150:(i + 1) * 150] = True
            masks[i, objects * 150 + i * 150:objects * 150 + (i + 1) * 150] = True
        target = dict(ids=torch.arange(objects) + 42, labels=torch.full((objects,), 5), masks=masks,
                      changes=torch.zeros(objects, dtype=torch.long),
                      temporal_stages=torch.cat((torch.zeros(objects * 150, dtype=torch.long),
                                                 torch.ones(objects * 150, dtype=torch.long))))
        prediction = dict(pred_masks=masks[:predicted].T, pred_scores=torch.full((predicted,), .9),
                          pred_classes=torch.full((predicted,), 5))
        metric = OfficialMetricAccumulator(mode='strict_online', dataset_spec=taxonomy)
        metric.update(prediction, target)
        states.append(metric.export_evidence())
    return taxonomy, states


def test_context_preserves_pooled_metrics_and_restores_bindings_on_error():
    acceleration = _acceleration()
    from scripts.evaluate_revisit_prompt import pool_official_states

    taxonomy, states = _reference_states()
    expected = pool_official_states(states, dataset_spec=taxonomy)
    original_math = ap_math.compute_precision_recall
    original_evaluator = evaluator.compute_precision_recall
    with pytest.raises(RuntimeError, match='deliberate exit'):
        with acceleration.accelerated_precision_recall():
            assert pool_official_states(states, dataset_spec=taxonomy) == expected
            with acceleration.accelerated_precision_recall():
                assert pool_official_states([states[0], states[0]], dataset_spec=taxonomy)['online_t-mAP'] == 1.
            raise RuntimeError('deliberate exit')
    assert ap_math.compute_precision_recall is original_math
    assert evaluator.compute_precision_recall is original_evaluator


def test_context_preserves_official_state_hash_validation():
    acceleration = _acceleration()
    from scripts.evaluate_revisit_prompt import pool_official_states

    taxonomy, states = _reference_states()
    states[0]['updates'] += 1
    with acceleration.accelerated_precision_recall():
        with pytest.raises(ValueError, match='byte identity differs'):
            pool_official_states(states, dataset_spec=taxonomy)


def _launcher():
    name = 'scripts.continue_revisit_prompt_statistics'
    assert importlib.util.find_spec(name) is not None, 'Scoped statistics continuation is missing'
    return importlib.import_module(name)


def test_statistics_launcher_dry_run_does_not_start_controller(monkeypatch, capsys):
    launcher = _launcher()
    from scripts import continue_revisit_prompt_sparse as controller

    def unexpected_start():
        pytest.fail('dry-run started the experiment controller')

    monkeypatch.setattr(controller, 'main', unexpected_start)
    assert launcher.main(['--dry-run']) == 0
    assert json.loads(capsys.readouterr().out)['status'] == 'DRY_RUN_NOT_STARTED'


@pytest.mark.parametrize('raise_after_pool', [False, True])
def test_statistics_scope_restores_bootstrap_and_curve_bindings(monkeypatch, raise_after_pool):
    launcher = _launcher()
    acceleration = _acceleration()
    from scripts import evaluate_revisit_prompt as evaluation

    taxonomy, states = _reference_states()
    expected = evaluation.pool_official_states(states, dataset_spec=taxonomy)
    original_curve = evaluator.compute_precision_recall
    assets, control, candidate = object(), object(), object()

    # Replace only the expensive upstream campaign, retaining the real pooling
    # and PR calculation inside the callback whose context is being tested.
    def bootstrap(bound_assets, *, role, control, candidate):
        assert bound_assets is assets and role == 'SEL'
        assert control is original_control and candidate is original_candidate
        assert evaluator.compute_precision_recall is acceleration.precision_recall_cpu
        result = evaluation.pool_official_states(states, dataset_spec=taxonomy)
        if raise_after_pool:
            raise RuntimeError('upstream bootstrap failed')
        return result

    original_control, original_candidate = control, candidate
    monkeypatch.setattr(evaluation, 'bootstrap_locked', bootstrap)
    if raise_after_pool:
        with pytest.raises(RuntimeError, match='upstream bootstrap failed'):
            with launcher.accelerated_bootstrap():
                evaluation.bootstrap_locked(assets, role='SEL', control=control, candidate=candidate)
    else:
        with launcher.accelerated_bootstrap():
            assert evaluation.bootstrap_locked(assets, role='SEL', control=control, candidate=candidate) == expected
            assert evaluator.compute_precision_recall is original_curve
    assert evaluation.bootstrap_locked is bootstrap
    assert evaluator.compute_precision_recall is original_curve
