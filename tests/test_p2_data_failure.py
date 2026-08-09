import re

import numpy as np
import pytest
import yaml
from lightning_fabric.utilities.distributed import DistributedSamplerWrapper
from torch.utils.data import DataLoader, Dataset

from datasets.multi_dataset import MultiDataset
from datasets.semseg import SemanticSegmentationDataset


class SizedDataset(Dataset):
    def __init__(self, size: int) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> int:
        return index


def _write_yaml(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def test_missing_split_database_raises_file_not_found_error_with_path(
    tmp_path,
) -> None:
    processed_dir = tmp_path / "processed" / "scannet"
    expected_database = processed_dir / "train_database.yaml"

    with pytest.raises(FileNotFoundError, match=re.escape(str(expected_database))):
        SemanticSegmentationDataset(
            data_dir=str(processed_dir),
            mode="train",
            fail_closed=True,
        )


def test_missing_split_database_default_preserves_official_system_exit(
    tmp_path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    processed_dir = tmp_path / "processed" / "scannet"
    expected_database = processed_dir / "train_database.yaml"

    with pytest.raises(SystemExit):
        SemanticSegmentationDataset(
            data_dir=str(processed_dir),
            mode="train",
        )

    assert f"generate {expected_database} first" in capsys.readouterr().out


@pytest.mark.parametrize("payload", [[], {}, None, {"unexpected": "mapping"}, "scalar"])
def test_split_database_requires_a_non_empty_list_with_path_context(
    tmp_path,
    payload,
) -> None:
    processed_dir = tmp_path / "processed" / "scannet"
    database_path = processed_dir / "train_database.yaml"
    _write_yaml(database_path, payload)

    with pytest.raises(
        ValueError,
        match=rf"{re.escape(str(database_path))}.*non-empty list",
    ):
        SemanticSegmentationDataset(
            data_dir=str(processed_dir),
            mode="train",
            fail_closed=True,
        )


def test_every_configured_data_directory_requires_non_empty_split(
    tmp_path,
) -> None:
    first_dir = tmp_path / "processed" / "first"
    second_dir = tmp_path / "processed" / "second"
    _write_yaml(
        first_dir / "train_database.yaml",
        [{"instance_gt_filepath": "first.txt"}],
    )
    second_database = second_dir / "train_database.yaml"
    _write_yaml(second_database, [])

    with pytest.raises(ValueError, match=re.escape(str(second_database))):
        SemanticSegmentationDataset(
            data_dir=(str(first_dir), str(second_dir)),
            mode="train",
            fail_closed=True,
        )


def test_multi_dataset_rejects_empty_dataset_list() -> None:
    with pytest.raises(ValueError, match="at least one dataset"):
        MultiDataset([], fail_closed=True)


@pytest.mark.parametrize("weights", [[], [1.0], [1.0, 0.8, 0.5]])
def test_multi_dataset_rejects_dataset_weight_length_mismatch(weights) -> None:
    with pytest.raises(ValueError, match="same length"):
        MultiDataset(
            [SizedDataset(2), SizedDataset(3)],
            weights=weights,
            fail_closed=True,
        )


def test_multi_dataset_rejects_empty_child_dataset() -> None:
    with pytest.raises(ValueError, match="dataset at index 1.*zero length"):
        MultiDataset(
            [SizedDataset(2), SizedDataset(0)],
            weights=[1.0, 0.8],
            fail_closed=True,
        )


@pytest.mark.parametrize("invalid_weight", [0.0, -0.1, float("nan"), float("inf"), "1.0", True])
def test_multi_dataset_rejects_invalid_weights(invalid_weight) -> None:
    with pytest.raises(ValueError, match="finite positive number"):
        MultiDataset(
            [SizedDataset(2)],
            weights=[invalid_weight],
            fail_closed=True,
        )


def test_multi_dataset_default_preserves_upstream_empty_child_skip(
    caplog: pytest.LogCaptureFixture,
) -> None:
    dataset = MultiDataset(
        [SizedDataset(2), SizedDataset(0)],
        weights=[1.0, 0.8],
    )

    assert dataset.sampler.num_samples == 3
    assert len(dataset.sampler.weights) == 2
    assert "Dataset has zero size, skipping" in caplog.text


def test_multi_dataset_default_preserves_upstream_falsy_weights() -> None:
    dataset = MultiDataset(
        [SizedDataset(2), SizedDataset(3)],
        weights=[],
    )

    assert dataset.weights == [1.0, 1.0]
    assert dataset.sampler.num_samples == 4


def test_multi_dataset_default_preserves_upstream_weight_length_behavior() -> None:
    dataset = MultiDataset(
        [SizedDataset(2), SizedDataset(3)],
        weights=[1.0],
    )

    assert dataset.sampler.num_samples == 2
    assert len(dataset.sampler.weights) == 2


def test_multi_dataset_aligns_epoch_sample_count_without_changing_weights() -> None:
    dataset = MultiDataset(
        [SizedDataset(1178), SizedDataset(1201)],
        weights=[1.0, 0.8],
        epoch_sample_multiple=32,
        fail_closed=True,
    )

    assert dataset.sampler.num_samples == 2112
    assert dataset.sampler.weights[:1178].sum().item() == pytest.approx(1.0)
    assert dataset.sampler.weights[1178:].sum().item() == pytest.approx(0.8)


def test_multi_dataset_default_epoch_sample_count_is_unchanged() -> None:
    dataset = MultiDataset(
        [SizedDataset(1178), SizedDataset(1201)],
        weights=[1.0, 0.8],
    )

    assert dataset.sampler.num_samples == 2120
    assert dataset.sampler.generator is None
    assert dataset.fail_closed is False


def test_aligned_sampler_yields_264_microbatches_per_ddp_rank() -> None:
    dataset = MultiDataset(
        [SizedDataset(1178), SizedDataset(1201)],
        weights=[1.0, 0.8],
        epoch_sample_multiple=32,
        sampler_seed=45,
        fail_closed=True,
    )
    rank_sampler = DistributedSamplerWrapper(
        dataset.sampler,
        num_replicas=2,
        rank=0,
        shuffle=True,
        seed=45,
        drop_last=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=4,
        sampler=rank_sampler,
        drop_last=False,
    )

    assert len(rank_sampler) == 1056
    assert len(loader) == 264
    assert len(loader) % 4 == 0
    assert len(loader) // 4 == 66
    assert len(loader) // 4 * 450 == 29700


def test_multi_dataset_from_config_routes_epoch_sample_multiple(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child_kwargs = []

    class ConfiguredSizedDataset(SizedDataset):
        def __init__(self, size: int, **kwargs) -> None:
            super().__init__(size)
            child_kwargs.append(kwargs)

    monkeypatch.setattr(
        "datasets.multi_dataset.SemanticSegmentationDataset",
        ConfiguredSizedDataset,
    )
    dataset = MultiDataset.from_config(
        datasets=[
            {
                "target": "datasets.semseg.SemanticSegmentationDataset",
                "size": 1178,
            },
            {
                "target": "datasets.semseg.SemanticSegmentationDataset",
                "size": 1201,
            },
        ],
        weights=[1.0, 0.8],
        epoch_sample_multiple=32,
        fail_closed=True,
    )

    assert dataset.sampler.num_samples == 2112
    assert child_kwargs == [{"fail_closed": True}, {"fail_closed": True}]


def test_multi_dataset_sampler_seed_reproduces_fresh_sample_streams() -> None:
    datasets = [SizedDataset(10), SizedDataset(12)]
    first = MultiDataset(
        datasets,
        weights=[1.0, 0.8],
        sampler_seed=45,
        fail_closed=True,
    )
    recreated = MultiDataset(
        datasets,
        weights=[1.0, 0.8],
        sampler_seed=45,
        fail_closed=True,
    )

    first_epoch = list(first.sampler)
    recreated_first_epoch = list(recreated.sampler)

    assert first.sampler.generator.initial_seed() == 45
    assert first_epoch == recreated_first_epoch
    assert list(first.sampler) != first_epoch


def test_multi_dataset_from_config_routes_sampler_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child_kwargs = []

    class ConfiguredSizedDataset(SizedDataset):
        def __init__(self, size: int, **kwargs) -> None:
            super().__init__(size)
            child_kwargs.append(kwargs)

    monkeypatch.setattr(
        "datasets.multi_dataset.SemanticSegmentationDataset",
        ConfiguredSizedDataset,
    )
    dataset = MultiDataset.from_config(
        datasets=[
            {
                "target": "datasets.semseg.SemanticSegmentationDataset",
                "size": 10,
            },
            {
                "target": "datasets.semseg.SemanticSegmentationDataset",
                "size": 12,
            },
        ],
        weights=[1.0, 0.8],
        sampler_seed=45,
        fail_closed=True,
    )

    assert child_kwargs == [{"fail_closed": True}, {"fail_closed": True}]
    assert dataset.sampler.generator.initial_seed() == 45


@pytest.mark.parametrize("invalid_seed", [1.5, "45", True])
def test_multi_dataset_rejects_non_integer_sampler_seed(invalid_seed) -> None:
    with pytest.raises(ValueError, match="sampler_seed.*integer"):
        MultiDataset(
            [SizedDataset(10), SizedDataset(12)],
            weights=[1.0, 0.8],
            sampler_seed=invalid_seed,
            fail_closed=True,
        )


def test_multi_dataset_from_config_routes_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child_kwargs = []

    class ConfiguredSizedDataset(SizedDataset):
        def __init__(self, size: int, **kwargs) -> None:
            super().__init__(size)
            child_kwargs.append(kwargs)

    monkeypatch.setattr(
        "datasets.multi_dataset.SemanticSegmentationDataset",
        ConfiguredSizedDataset,
    )

    with pytest.raises(ValueError, match="dataset at index 1.*zero length"):
        MultiDataset.from_config(
            datasets=[
                {
                    "target": "datasets.semseg.SemanticSegmentationDataset",
                    "size": 2,
                },
                {
                    "target": "datasets.semseg.SemanticSegmentationDataset",
                    "size": 0,
                },
            ],
            weights=[1.0, 0.8],
            fail_closed=True,
        )

    assert child_kwargs == [{"fail_closed": True}, {"fail_closed": True}]


@pytest.mark.parametrize("invalid_multiple", [0, -1, 1.5, "32", True])
def test_multi_dataset_rejects_invalid_epoch_sample_multiple(
    invalid_multiple,
) -> None:
    with pytest.raises(ValueError, match="epoch_sample_multiple.*positive integer"):
        MultiDataset(
            [SizedDataset(10), SizedDataset(12)],
            weights=[1.0, 0.8],
            epoch_sample_multiple=invalid_multiple,
            fail_closed=True,
        )


def test_multi_dataset_rejects_epoch_sample_multiple_larger_than_epoch() -> None:
    with pytest.raises(ValueError, match="epoch_sample_multiple.*18"):
        MultiDataset(
            [SizedDataset(10), SizedDataset(12)],
            weights=[1.0, 0.8],
            epoch_sample_multiple=32,
            fail_closed=True,
        )


def test_multi_dataset_normalizes_integral_epoch_sample_multiple() -> None:
    dataset = MultiDataset(
        [SizedDataset(1178), SizedDataset(1201)],
        weights=[1.0, 0.8],
        epoch_sample_multiple=np.int64(32),
        fail_closed=True,
    )

    assert dataset.epoch_sample_multiple == 32
    assert type(dataset.epoch_sample_multiple) is int
    assert dataset.sampler.num_samples == 2112
