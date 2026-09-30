import torch

from models.rescene import ReScene


def sampler(enabled):
    model = ReScene.__new__(ReScene)
    torch.nn.Module.__init__(model)
    model.max_sample_size = False
    model.native_long_sampling = enabled
    model.native_long_sampling_seed = 45
    model.native_long_draw_ids = [10, 11]
    model.native_long_sampling_stats = []
    return model


def test_stratified_features_extras_and_padding_share_indices():
    model = sampler(True)
    features = [torch.arange(15.).reshape(15, 1), torch.arange(4.).reshape(4, 1)]
    stages = [torch.arange(15) // 3, torch.zeros(4, dtype=torch.long)]
    extra = [x * 10 for x in features]
    selected, selected_extra, padding = model.sample_and_batch_features(
        features, 8, extra=[extra], stage_ids=stages, sampling_stage_idx=0)
    assert selected.shape == (2, 8, 1)
    assert torch.equal(selected_extra, selected * 10)
    assert padding.sum(1).tolist() == [0, 4]
    assert model.native_long_sampling_stats[0]["counts"] == [2, 2, 2, 1, 1]
    assert torch.equal(selected[1, :4], features[1])


def test_single_stage_and_uncapped_eval_preserve_native_sampling():
    features = [torch.arange(15.).reshape(15, 1), torch.arange(4.).reshape(4, 1)]
    stages = [torch.zeros(15, dtype=torch.long), torch.zeros(4, dtype=torch.long)]
    for cap, is_eval in ((8, False), (30, False), (8, True)):
        first = sampler(False).sample_and_batch_features(
            features, cap, is_eval=is_eval, stage_ids=stages, sampling_stage_idx=0)
        second = sampler(True).sample_and_batch_features(
            features, cap, is_eval=is_eval, stage_ids=stages, sampling_stage_idx=0)
        assert all(torch.equal(a, b) for a, b in zip(first, second, strict=True))
