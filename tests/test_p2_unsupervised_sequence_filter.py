import copy
from pathlib import Path

import numpy as np
import pytest
import yaml
from lightning_fabric.utilities.distributed import DistributedSamplerWrapper
from torch.utils.data import DataLoader, Dataset

from datasets.multi_dataset import MultiDataset
from datasets.semseg import SemanticSegmentationDataset
from scripts import audit_p2_reproduction as audit
from utils import p2_preflight


def _write_yaml(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def _points(semantic: int, instance: int) -> np.ndarray:
    points = np.zeros((2, 12), dtype=np.float32)
    points[:, :3] = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
    points[:, 3:6] = 128.0
    points[:, 9] = [0.0, 1.0]
    points[:, 10] = semantic
    points[:, 11] = instance
    return points


class _SizedDataset(Dataset):
    def __init__(self, size: int) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> int:
        return index


@pytest.fixture()
def sequence_fixture(tmp_path: Path) -> dict[str, Path]:
    root = tmp_path / "rio"
    labels = {
        1: {"color": [1, 1, 1], "name": "wall", "validation": True},
        2: {"color": [2, 2, 2], "name": "floor", "validation": True},
        3: {"color": [3, 3, 3], "name": "chair", "validation": True},
        4: {"color": [4, 4, 4], "name": "table", "validation": True},
    }
    _write_yaml(root / "label_database.yaml", labels)
    _write_yaml(root / "rio.yaml", {"valid_class_ids": [3, 4]})

    records = {}
    for mode, scenes in {
        "train": ("scene0001_00", "scene0001_01", "scene0002_00", "scene0002_01"),
        "validation": (
            "scene0003_00",
            "scene0003_01",
            "scene0004_00",
            "scene0004_01",
        ),
    }.items():
        for scene in scenes:
            numeric = scene.removeprefix("scene")
            npy = root / mode / f"{numeric}.npy"
            gt = root / "instance_gt" / mode / f"{scene}.txt"
            # Scenes 2 and 4 contain only background taxonomy labels.
            semantic = 2 if scene.startswith(("scene0002", "scene0004")) else 3
            npy.parent.mkdir(parents=True, exist_ok=True)
            gt.parent.mkdir(parents=True, exist_ok=True)
            np.save(npy, _points(semantic, 7 if semantic == 3 else 1))
            gt.write_text("1\n1\n", encoding="utf-8")
            records[scene] = {
                "filepath": str(npy),
                "instance_gt_filepath": str(gt),
                "file_len": 2,
            }
        _write_yaml(
            root / f"{mode}_database.yaml",
            [records[scene] for scene in scenes],
        )

    sequence_db = {
        "scene0001_00-scene0001_01": {
            "type": "train",
            "filepath": str(root / "train-change.txt"),
            "ambiguities": [[7]],
        },
        "scene0002_00-scene0002_01": {
            "type": "train",
            "filepath": str(root / "train-empty-change.txt"),
            "ambiguities": [[8]],
        },
        "scene0003_00-scene0003_01": {
            "type": "validation",
            "filepath": str(root / "validation-change.txt"),
            "ambiguities": [[9]],
        },
        "scene0004_00-scene0004_01": {
            "type": "validation",
            "filepath": str(root / "validation-empty-change.txt"),
            "ambiguities": [[10]],
        },
    }
    for change_name in (
        "train-change.txt",
        "train-empty-change.txt",
        "validation-change.txt",
        "validation-empty-change.txt",
    ):
        (root / change_name).write_text("0\n0\n0\n0\n", encoding="utf-8")
    _write_yaml(root / "sequence_database_sliding_2.yaml", sequence_db)
    return {"root": root, "labels": root / "label_database.yaml"}


def _dataset(
    root: Path,
    mode: str,
    *,
    exclude: bool = False,
    temporal_window: int = 2,
    fail_closed: bool = False,
):
    return SemanticSegmentationDataset(
        dataset_name="rio",
        data_dir=str(root),
        label_db_filepath=str(root / "label_database.yaml"),
        color_mean_std=((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        mode=mode,
        add_colors=True,
        add_normals=False,
        add_raw_coordinates=False,
        add_instance=True,
        num_labels=4,
        filter_out_classes=[0, 1],
        label_offset=2,
        temporal_window=temporal_window,
        fail_closed=fail_closed,
        exclude_unsupervised_sequences=exclude,
    )


def test_unsupervised_sequence_filter_is_disabled_by_default(sequence_fixture):
    dataset = _dataset(sequence_fixture["root"], "train")

    assert dataset.sequence_names == [
        "scene0001_00-scene0001_01",
        "scene0002_00-scene0002_01",
    ]
    assert dataset.excluded_unsupervised_sequences == []


def test_unsupervised_sequence_filter_is_deterministic_and_keeps_metadata_aligned(
    sequence_fixture,
):
    first = _dataset(
        sequence_fixture["root"], "train", exclude=True, fail_closed=True
    )
    second = _dataset(
        sequence_fixture["root"], "train", exclude=True, fail_closed=True
    )

    assert first.sequence_names == ["scene0001_00-scene0001_01"]
    assert first.excluded_unsupervised_sequences == ["scene0002_00-scene0002_01"]
    assert first.sequence_names == second.sequence_names
    assert first.excluded_unsupervised_sequences == second.excluded_unsupervised_sequences
    assert first.change_files == [str(sequence_fixture["root"] / "train-change.txt")]
    assert first.ambiguities == [[[7]]]
    assert first.sequence_indices.shape == (1, 2)

    for sequence_index in range(len(first)):
        sample = first[sequence_index]
        labels = sample[2]
        assert np.any((labels[:, 0] >= 2) & (labels[:, 1] >= 0))


def test_validation_filter_uses_same_real_target_rule(sequence_fixture):
    dataset = _dataset(
        sequence_fixture["root"],
        "validation",
        exclude=True,
        fail_closed=True,
    )

    assert dataset.sequence_names == ["scene0003_00-scene0003_01"]
    assert dataset.excluded_unsupervised_sequences == [
        "scene0004_00-scene0004_01"
    ]


def test_temporal_window_one_keeps_all_scans_when_common_flag_is_true(
    sequence_fixture,
):
    dataset = _dataset(
        sequence_fixture["root"],
        "train",
        exclude=True,
        temporal_window=1,
        fail_closed=True,
    )

    assert len(dataset) == 4
    assert dataset.unsupervised_sequence_filter["enabled"] is False
    assert dataset.excluded_unsupervised_sequences == []


def test_enabled_filter_rejects_taxonomy_outside_active_label_database(
    sequence_fixture,
):
    labels_path = sequence_fixture["labels"]
    labels = yaml.safe_load(labels_path.read_text(encoding="utf-8"))
    labels[3]["validation"] = False
    _write_yaml(labels_path, labels)
    _write_yaml(sequence_fixture["root"] / "rio.yaml", {"valid_class_ids": [3, 4]})

    with pytest.raises(ValueError, match="outside the active validation"):
        _dataset(
            sequence_fixture["root"],
            "train",
            exclude=True,
            fail_closed=True,
        )


def test_filtered_dataset_preserves_multidataset_epoch_alignment(sequence_fixture):
    rio = _dataset(
        sequence_fixture["root"], "train", exclude=True, fail_closed=True
    )
    other = _dataset(sequence_fixture["root"], "validation", exclude=False)
    mixed = MultiDataset(
        datasets=[rio, other],
        weights=[1.0, 1.0],
        epoch_sample_multiple=2,
        sampler_seed=45,
        fail_closed=True,
    )

    assert [len(dataset) for dataset in mixed.datasets] == [1, 2]
    assert mixed.sampler.num_samples == 2
    assert mixed.sampler.num_samples % 2 == 0


def test_p2_filtered_mix_keeps_formal_effective_epoch_batch():
    mixed = MultiDataset(
        datasets=[_SizedDataset(1174), _SizedDataset(1201)],
        weights=[1.0, 0.8],
        epoch_sample_multiple=32,
        sampler_seed=45,
        fail_closed=True,
    )
    rank_sampler = DistributedSamplerWrapper(
        mixed.sampler,
        num_replicas=2,
        rank=0,
        shuffle=True,
        seed=45,
        drop_last=False,
    )
    loader = DataLoader(mixed, batch_size=4, sampler=rank_sampler)

    assert mixed.sampler.num_samples == 2112
    assert len(rank_sampler) == 1056
    assert len(loader) == 264
    assert len(loader) // 4 == 66


def test_real_audit_binds_exact_filter_names_and_rejects_drift():
    evidence, errors = audit._audit_rio_record_paths(
        Path("data/processed/rio"),
        validate_content=True,
        exclude_unsupervised_sequences=True,
    )

    assert errors == []
    filter_evidence = evidence["unsupervised_sequence_filter"]
    assert filter_evidence["enabled"] is True
    assert filter_evidence["source"] == "real_npy"
    assert filter_evidence["by_split"]["train"]["excluded_count"] == 4
    assert filter_evidence["by_split"]["validation"]["excluded_count"] == 3
    assert filter_evidence["by_split"]["test"]["excluded_count"] == 3
    assert filter_evidence["by_split"]["train"]["excluded_sequences"] == [
        "scene0242_00-scene0242_01",
        "scene0242_01-scene0242_02",
        "scene0242_02-scene0242_00",
        "scene0245_01-scene0245_02",
    ]
    assert filter_evidence["by_split"]["validation"]["excluded_sequences"] == [
        "scene0439_00-scene0439_02",
        "scene0439_01-scene0439_00",
        "scene0439_02-scene0439_01",
    ]
    artifact = {
        "unsupervised_sequence_filter": filter_evidence,
        "rio_path_integrity": evidence,
    }

    validation_errors = []
    p2_preflight._validate_unsupervised_sequence_filter(
        artifact,
        validation_errors,
    )
    assert validation_errors == []

    drifted = copy.deepcopy(artifact)
    drifted["unsupervised_sequence_filter"]["by_split"]["train"][
        "excluded_sequences"
    ][0] = "scene9999_00-scene9999_01"
    drift_errors = []
    p2_preflight._validate_unsupervised_sequence_filter(drifted, drift_errors)
    assert "unsupervised_sequence_filter.sequence_name_sha256 mismatch" in drift_errors

    count_drifted = copy.deepcopy(artifact)
    count_drifted["unsupervised_sequence_filter"]["by_split"]["train"][
        "excluded_count"
    ] = 5
    count_errors = []
    p2_preflight._validate_unsupervised_sequence_filter(
        count_drifted,
        count_errors,
    )
    assert "unsupervised_sequence_filter.by_split.train.excluded_count mismatch" in count_errors


def test_audit_default_off_does_not_apply_filter_contract():
    evidence, errors = audit._audit_rio_record_paths(
        Path("data/processed/rio"),
        validate_content=False,
    )

    assert errors == []
    assert evidence["unsupervised_sequence_filter"]["enabled"] is False
    assert evidence["unsupervised_sequence_filter"]["status"] == (
        "not_run_diagnostic"
    )
