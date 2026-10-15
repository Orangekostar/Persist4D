"""Known neighborhoods and strict inference scope for sparse reference math."""
import pytest
import torch

from models.revisit_prompt_sparse_reference import (
    evaluation_sparse_reference, ordered_subm_convolution,
)


@pytest.fixture(autouse=True)
def inference_only():
    with torch.no_grad():
        yield


def example():
    features = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])
    weights = torch.tensor([[[1., 0.], [0., 1.], [1., 1.]],
                            [[0., 2.], [1., -1.], [-1., 1.]]])
    pairs = torch.full((2, 3, 3), -12345, dtype=torch.int32)
    pairs[:, 0, :2] = torch.tensor([[0, 1], [1, 2]])
    pairs[:, 2, :2] = torch.tensor([[1, 2], [0, 1]])
    counts = torch.tensor([2, 0, 0], dtype=torch.int32)
    return features, weights, pairs, counts


def run(features, weights, pairs, counts, **kwargs):
    return ordered_subm_convolution(features, weights, pairs, counts,
                                    features.shape[0], subm=True, **kwargs)


def test_known_neighbors_include_center_mirrored_offsets_and_bias():
    expected = torch.tensor([[9.5, -2.], [16.5, 2.], [9.5, 5.]])
    inputs = example()
    copies = [x.clone() for x in inputs]
    actual = run(*inputs, bias=torch.tensor([.5, -2.]))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert all(torch.equal(x, y) for x, y in zip(inputs, copies))


def test_neighbor_storage_permutation_does_not_change_results():
    f, w, p, c = example()
    reference = run(f, w, p, c)
    p[:, 0, :2] = p[:, 0, :2].flip(1)
    p[:, 2, :2] = p[:, 2, :2].flip(1)
    torch.testing.assert_close(run(f, w, p, c), reference, rtol=0, atol=0)


def test_center_only_and_empty_inputs():
    f, w, _, _ = example()
    p = torch.empty(2, 1, 0, dtype=torch.int64)
    c = torch.zeros(1, dtype=torch.int32)
    torch.testing.assert_close(run(f, w[:, 1:2], p, c), f @ w[:, 1].T, rtol=0, atol=0)
    assert run(f[:0], w[:, 1:2], p, c).shape == (0, 2)


@pytest.mark.parametrize('case', ['half', 'even_kernel', 'count', 'duplicate'])
def test_unsupported_or_ambiguous_tensor_inputs_are_rejected(case):
    f, w, p, c = example()
    if case == 'half': f, w = f.half(), w.half()
    if case == 'even_kernel': w, p, c = w[:, :2], p[:, :2], c[:2]
    if case == 'count': c[0] = 4
    if case == 'duplicate': p[1, 0, :2] = 1
    with pytest.raises(ValueError):
        run(f, w, p, c)


@pytest.mark.parametrize('kwargs', [{'subm': False}, {'inverse': True}, {'act_type': 1}])
def test_unsupported_operations_fail_closed(kwargs):
    f, w, p, c = example()
    options = dict(subm=True)
    options.update(kwargs)
    with pytest.raises(ValueError):
        ordered_subm_convolution(f, w, p, c, 3, **options)


def test_reference_rejects_training_and_output_shape_changes():
    values = example()
    with torch.enable_grad(), pytest.raises(ValueError, match='inference'):
        run(*values)
    with pytest.raises(ValueError):
        ordered_subm_convolution(*values, 4, subm=True)


def test_context_restores_original_operation_on_error():
    from spconv.pytorch import ops
    original = ops.indice_conv
    with pytest.raises(RuntimeError, match='owned failure'):
        with evaluation_sparse_reference():
            assert ops.indice_conv is ordered_subm_convolution
            raise RuntimeError('owned failure')
    assert ops.indice_conv is original


def test_matches_installed_spconv_cpu_submanifold_convolution():
    import spconv.pytorch as spconv
    from models.revisit_prompt_native_numerics import pin_native_convolutions
    f, _, _, _ = example()
    coords = torch.tensor([[0, 1, 1, 0], [0, 1, 1, 1], [0, 1, 1, 2]], dtype=torch.int32)
    layer = spconv.SubMConv3d(2, 3, kernel_size=3, bias=False)
    pin_native_convolutions(layer)
    layer.weight.copy_(torch.arange(layer.weight.numel()).reshape_as(layer.weight) / 128)
    layer.eval()
    native = layer(spconv.SparseConvTensor(f, coords, [3, 3, 3], 1)).features
    with evaluation_sparse_reference():
        reference = layer(spconv.SparseConvTensor(f, coords, [3, 3, 3], 1)).features
    torch.testing.assert_close(reference, native, rtol=1e-6, atol=1e-6)
