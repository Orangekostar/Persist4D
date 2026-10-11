import pytest


def test_profile_inputs_are_fixed_cal_only_with_every_horizon():
    from scripts.native_long_deployment_validation import fixed_cal_inputs

    rows = [{'horizon': h, 'input_id': f'CAL:{h}:{suffix}'}
            for h in range(1, 6) for suffix in ('b', 'a')]
    population = {'evaluation_inputs': {'CAL': rows, 'SEL': []}}
    assert [r['input_id'] for r in fixed_cal_inputs(population)] == [f'CAL:{h}:a' for h in range(1, 6)]
    population['evaluation_inputs']['CAL'] = rows[:-2]
    with pytest.raises(ValueError, match='horizons'):
        fixed_cal_inputs(population)
    population['evaluation_inputs']['CAL'] = rows + [{'horizon': 2, 'input_id': 'SEL:a'}]
    with pytest.raises(ValueError, match='CAL'):
        fixed_cal_inputs(population)


def test_tensor_comparison_reports_shape_and_numeric_changes():
    import torch

    from scripts.native_long_deployment_validation import tensor_difference

    assert tensor_difference(torch.ones(2), torch.ones(2)) == {
        'same_shape': True, 'exact': True, 'max_abs': 0., 'mean_abs': 0., 'changed_fraction': 0.}
    result = tensor_difference(torch.tensor([0., 1.]), torch.tensor([0., 2.]))
    assert result['max_abs'] == 1. and result['changed_fraction'] == .5
    assert tensor_difference(torch.ones(2), torch.ones(3))['same_shape'] is False
