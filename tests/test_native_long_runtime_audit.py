import pytest
import torch

from scripts.native_long_runtime_audit import tensor_delta


def test_tensor_delta_distinguishes_exact_numeric_and_shape_changes():
    a = torch.tensor([1., 2.])
    assert tensor_delta(a, a.clone())['exact']
    row = tensor_delta(a, torch.tensor([1., 2.5]))
    assert not row['exact']
    assert row['max_abs'] == .5
    assert row['different_fraction'] == .5
    assert tensor_delta(a, a[:1])['shape_equal'] is False
    assert tensor_delta(a.bool(), torch.tensor([True, False]))['different_fraction'] == .5
    with pytest.raises(ValueError, match='finite'):
        tensor_delta(a, torch.tensor([1., torch.nan]))
