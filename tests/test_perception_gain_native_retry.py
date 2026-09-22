import pytest
import torch

from scripts import perception_gain_native_evaluation as evaluation


def test_native_oom_retries_once_with_identical_prefix(monkeypatch):
    calls, cleared, retries = [], [], []
    key = {"horizon": 5, "scan_indices": [0, 1, 2, 3, 4]}

    class Producer:
        def produce_bundle(self, actual):
            assert actual is key
            calls.append(dict(actual))
            if len(calls) == 1:
                raise torch.OutOfMemoryError("test OOM")
            return "complete"

    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: cleared.append(True))
    assert (
        evaluation.produce_native_with_retry(Producer(), key, on_retry=retries.append)
        == "complete"
    )
    assert calls == [key, key] and cleared == [True] and len(retries) == 1


@pytest.mark.parametrize(
    "failure,expected", [(ValueError, 1), (torch.OutOfMemoryError, 2)]
)
def test_native_retry_does_not_repeat_other_errors_or_a_second_oom(
    monkeypatch, failure, expected
):
    calls = []

    class Producer:
        def produce_bundle(self, key):
            calls.append(key)
            raise failure("test failure")

    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    with pytest.raises(failure):
        evaluation.produce_native_with_retry(Producer(), {}, on_retry=lambda _: None)
    assert len(calls) == expected
