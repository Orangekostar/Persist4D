import re

import pytest
import yaml
from torch.utils.data import Dataset

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
        )


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
        )


def test_multi_dataset_rejects_empty_dataset_list() -> None:
    with pytest.raises(ValueError, match="at least one dataset"):
        MultiDataset([])


@pytest.mark.parametrize("weights", [[], [1.0], [1.0, 0.8, 0.5]])
def test_multi_dataset_rejects_dataset_weight_length_mismatch(weights) -> None:
    with pytest.raises(ValueError, match="same length"):
        MultiDataset([SizedDataset(2), SizedDataset(3)], weights=weights)


def test_multi_dataset_rejects_empty_child_dataset() -> None:
    with pytest.raises(ValueError, match="dataset at index 1.*zero length"):
        MultiDataset([SizedDataset(2), SizedDataset(0)], weights=[1.0, 0.8])


@pytest.mark.parametrize("invalid_weight", [0.0, -0.1, float("nan"), float("inf"), "1.0", True])
def test_multi_dataset_rejects_invalid_weights(invalid_weight) -> None:
    with pytest.raises(ValueError, match="finite positive number"):
        MultiDataset([SizedDataset(2)], weights=[invalid_weight])
