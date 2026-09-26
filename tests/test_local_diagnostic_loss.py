import pytest
import torch


def test_diagnostic_loss_restores_strict_determinism_even_after_error():
    from scripts.perception_gain_local_evaluation import diagnostic_loss_forward

    previous = torch.are_deterministic_algorithms_enabled()
    warning = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True)

        def loss(value):
            assert torch.are_deterministic_algorithms_enabled()
            assert torch.is_deterministic_algorithms_warn_only_enabled()
            if value is None:
                raise ValueError("diagnostic failure")
            return value.square()

        wrapped = diagnostic_loss_forward(loss)
        with pytest.raises(RuntimeError, match="no-grad"):
            wrapped(torch.tensor(3.))
        with torch.no_grad():
            assert wrapped(torch.tensor(3.)).item() == 9
            assert not torch.is_deterministic_algorithms_warn_only_enabled()
            with pytest.raises(ValueError, match="diagnostic failure"):
                wrapped(None)
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.is_deterministic_algorithms_warn_only_enabled()
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=warning)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_nll_diagnostic_keeps_prediction_runtime_strict():
    from scripts.perception_gain_local_evaluation import diagnostic_loss_forward

    previous = torch.are_deterministic_algorithms_enabled()
    warning = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        logits = torch.ones(1, 3, 2, 2, device="cuda")
        target = torch.zeros(1, 2, 2, dtype=torch.long, device="cuda")
        with torch.no_grad():
            with pytest.raises(RuntimeError, match="deterministic"):
                torch.nn.functional.cross_entropy(logits, target)
            with pytest.warns(UserWarning, match="deterministic"):
                loss = diagnostic_loss_forward(torch.nn.functional.cross_entropy)(logits, target)
        assert loss.item() == pytest.approx(1.0986123)
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.is_deterministic_algorithms_warn_only_enabled()
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=warning)
