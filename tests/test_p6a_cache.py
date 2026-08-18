from __future__ import annotations

import copy
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import torch

from scripts.p6a_cache import (
    ORDER_IDS,
    build_cache_manifest,
    cache_payload_digest,
    load_cache_entry,
    load_cache_manifest,
    validate_cache_entry,
    validate_cache_manifest,
    validate_cache_payload,
    write_cache_entry,
    write_cache_manifest,
)

PROVENANCE = {
    "source_commit": "1" * 40,
    "checkpoint_sha256": "2" * 64,
    "config_sha256": "3" * 64,
    "dataset_sha256": "4" * 64,
}


def _payload(*, stage_index: int = 1) -> dict[str, object]:
    history = [f"scene0001_{index:02d}" for index in range(5)][: stage_index + 1]
    local_window = history if stage_index == 0 else history[-2:]
    masks = torch.tensor(
        [[True, True, False, False], [False, False, True, True]],
        dtype=torch.bool,
    )
    return {
        "schema_version": 1,
        "key": {
            "master_sequence_id": "scene0001_00-scene0001_01",
            "reference_scene_id": "uuid-1",
            "order_id": "canonical",
            "stage_index": stage_index,
            "history_scan_ids": history,
            "local_window_scan_ids": local_window,
        },
        "provenance": dict(PROVENANCE),
        "observation": {
            "features": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            "class_prob": torch.tensor([[0.8, 0.1, 0.1], [0.1, 0.7, 0.2]]),
            "confidence": torch.tensor([0.8, 0.7]),
            "valid": torch.tensor([True, True]),
            "masks": masks,
            "mask_support": torch.tensor([2, 2], dtype=torch.long),
            "local_query_ids": torch.tensor([0, 1], dtype=torch.long),
        },
        "target": {
            "gt_ids": torch.tensor([10, 20], dtype=torch.long),
            "gt_classes": torch.tensor([0, 1], dtype=torch.long),
            "gt_masks": masks.clone(),
            "changes": torch.tensor([0, 1], dtype=torch.long),
        },
    }


def test_cache_digest_is_mapping_order_independent_and_tensor_sensitive():
    payload = _payload()
    reordered = {key: payload[key] for key in reversed(payload)}

    assert cache_payload_digest(payload) == cache_payload_digest(reordered)
    changed = copy.deepcopy(payload)
    changed["observation"]["features"][0, 0] = 0.5
    assert cache_payload_digest(changed) != cache_payload_digest(payload)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(extra=True),
        lambda value: value["key"].update(history_scan_ids=["scene0001_00"]),
        lambda value: value["key"].update(local_window_scan_ids=["scene0001_00"]),
        lambda value: value["observation"].update(mask_support=torch.tensor([1, 2])),
        lambda value: value["observation"]["features"].fill_(float("nan")),
        lambda value: value["observation"].update(valid=torch.tensor([True])),
        lambda value: value["observation"].update(local_query_ids=torch.tensor([1, 0])),
        lambda value: value["target"].update(gt_ids=torch.tensor([10, 10])),
    ],
)
def test_cache_payload_fails_closed_on_shape_content_or_schema_drift(mutation):
    payload = _payload()
    mutation(payload)

    with pytest.raises(ValueError):
        validate_cache_payload(payload)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["key"].update(stage_index=5),
        lambda value: value["key"].update(order_id="unregistered"),
        lambda value: value["target"].update(
            changes=torch.tensor([0, -1], dtype=torch.long)
        ),
        lambda value: value["target"].update(
            gt_classes=torch.tensor([0, -1], dtype=torch.long)
        ),
    ],
)
def test_cache_payload_rejects_unregistered_stage_order_and_negative_labels(mutation):
    payload = _payload()
    mutation(payload)

    with pytest.raises(ValueError):
        validate_cache_payload(payload)


def test_cache_write_load_roundtrip_and_provenance_binding(tmp_path: Path):
    payload = _payload()

    entry = write_cache_entry(tmp_path, payload)
    loaded = load_cache_entry(
        tmp_path / entry["filename"],
        expected_provenance=payload["provenance"],
    )

    assert cache_payload_digest(loaded) == entry["content_sha256"]
    assert entry["filename"] == f"{entry['content_sha256']}.pt"
    assert entry["file_bytes"] > 0
    assert len(entry["file_sha256"]) == 64
    resumed = write_cache_entry(tmp_path, payload)
    assert resumed == entry

    wrong = dict(payload["provenance"])
    wrong["config_sha256"] = "f" * 64
    with pytest.raises(ValueError):
        load_cache_entry(tmp_path / entry["filename"], expected_provenance=wrong)


def test_validate_cache_entry_rechecks_file_evidence_and_key(tmp_path: Path):
    payload = _payload()
    entry = write_cache_entry(tmp_path, payload)

    validated = validate_cache_entry(tmp_path / entry["filename"], entry, PROVENANCE)
    assert cache_payload_digest(validated) == entry["content_sha256"]

    wrong_file_sha = dict(entry)
    wrong_file_sha["file_sha256"] = "f" * 64
    with pytest.raises(ValueError):
        validate_cache_entry(tmp_path / entry["filename"], wrong_file_sha, PROVENANCE)

    (tmp_path / entry["filename"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        write_cache_entry(tmp_path, payload)


def test_cache_write_is_no_clobber_under_concurrent_same_payload_writers(
    tmp_path: Path,
):
    payload = _payload()

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(lambda _: write_cache_entry(tmp_path, payload), range(2))
        )

    assert results[0] == results[1]
    assert list(tmp_path.glob("*.pt")) == [
        tmp_path / f"{results[0]['content_sha256']}.pt"
    ]


def test_load_cache_entry_returns_isolated_cpu_snapshots(tmp_path: Path):
    payload = _payload()
    entry = write_cache_entry(tmp_path, payload)

    first = load_cache_entry(
        tmp_path / entry["filename"], expected_provenance=PROVENANCE
    )
    second = load_cache_entry(
        tmp_path / entry["filename"], expected_provenance=PROVENANCE
    )
    first["observation"]["features"][0, 0] = 99.0
    first["target"]["gt_ids"][0] = 999

    assert second["observation"]["features"][0, 0].item() == 1.0
    assert second["target"]["gt_ids"][0].item() == 10


def test_cache_loader_rejects_symlink_and_corrupt_file(tmp_path: Path):
    entry = write_cache_entry(tmp_path, _payload())
    path = tmp_path / entry["filename"]
    alias = tmp_path / "alias.pt"
    alias.symlink_to(path)

    with pytest.raises(ValueError):
        load_cache_entry(alias)
    path.write_bytes(b"not-a-cache")
    with pytest.raises(ValueError):
        load_cache_entry(path)


def test_manifest_requires_exact_expected_unique_cache_keys(tmp_path: Path):
    first = write_cache_entry(tmp_path, _payload(stage_index=0))
    second = write_cache_entry(tmp_path, _payload(stage_index=1))
    expected = [first["key"], second["key"]]

    manifest = build_cache_manifest(
        [second, first], expected_keys=expected, expected_provenance=PROVENANCE
    )

    assert manifest["status"] == "pass"
    assert manifest["entry_count"] == 2
    assert [entry["key"]["stage_index"] for entry in manifest["entries"]] == [0, 1]
    with pytest.raises(ValueError):
        build_cache_manifest(
            [first], expected_keys=expected, expected_provenance=PROVENANCE
        )
    with pytest.raises(ValueError):
        build_cache_manifest(
            [first, first], expected_keys=[first["key"]], expected_provenance=PROVENANCE
        )


def test_manifest_revalidates_cache_directory_entries(tmp_path: Path):
    first = write_cache_entry(tmp_path, _payload(stage_index=0))
    second = write_cache_entry(tmp_path, _payload(stage_index=1))
    expected = [first["key"], second["key"]]

    manifest = build_cache_manifest(
        [first, second],
        expected_keys=expected,
        expected_provenance=PROVENANCE,
        cache_directory=tmp_path,
    )
    validate_cache_manifest(
        manifest,
        expected_keys=expected,
        expected_provenance=PROVENANCE,
        cache_directory=tmp_path,
    )

    stale = copy.deepcopy(first)
    stale["file_sha256"] = "f" * 64
    with pytest.raises(ValueError):
        build_cache_manifest(
            [stale, second],
            expected_keys=expected,
            expected_provenance=PROVENANCE,
            cache_directory=tmp_path,
        )


def _logical_key(
    master_index: int, order_id: str, stage_index: int
) -> dict[str, object]:
    history = [
        f"scene{master_index:04d}_{visit:02d}" for visit in range(stage_index + 1)
    ]
    return {
        "master_sequence_id": "-".join(history),
        "reference_scene_id": f"uuid-{master_index:02d}",
        "order_id": order_id,
        "stage_index": stage_index,
        "history_scan_ids": history,
        "local_window_scan_ids": history if stage_index == 0 else history[-2:],
    }


def _synthetic_entry(key: dict[str, object]) -> dict[str, object]:
    content_sha256 = hashlib.sha256(
        json.dumps(key, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "filename": f"{content_sha256}.pt",
        "content_sha256": content_sha256,
        "file_sha256": hashlib.sha256(content_sha256.encode()).hexdigest(),
        "file_bytes": 1,
        "key": key,
    }


def test_manifest_locks_645_logical_coverage_without_materializing_tensors():
    expected = [
        _logical_key(master, order, stage)
        for master in range(43)
        for order in ORDER_IDS
        for stage in range(5)
    ]
    entries = [_synthetic_entry(key) for key in expected]

    manifest = build_cache_manifest(
        entries, expected_keys=expected, expected_provenance=PROVENANCE
    )

    assert manifest["entry_count"] == 645
    assert len(manifest["entries"]) == 645


def test_manifest_write_load_is_atomic_exact_and_idempotent(tmp_path: Path):
    cache_directory = tmp_path / "cache"
    cache_directory.mkdir()
    entry = write_cache_entry(cache_directory, _payload())
    expected = [entry["key"]]
    manifest = build_cache_manifest(
        [entry],
        expected_keys=expected,
        expected_provenance=PROVENANCE,
        cache_directory=cache_directory,
    )
    path = tmp_path / "cache_manifest.json"

    assert (
        write_cache_manifest(
            path,
            manifest,
            expected_keys=expected,
            expected_provenance=PROVENANCE,
            cache_directory=cache_directory,
        )
        == manifest
    )
    assert (
        load_cache_manifest(
            path,
            expected_keys=expected,
            expected_provenance=PROVENANCE,
            cache_directory=cache_directory,
        )
        == manifest
    )
    assert (
        write_cache_manifest(
            path,
            manifest,
            expected_keys=expected,
            expected_provenance=PROVENANCE,
            cache_directory=cache_directory,
        )
        == manifest
    )

    malformed = copy.deepcopy(manifest)
    malformed.pop("provenance")
    with pytest.raises(ValueError):
        validate_cache_manifest(
            malformed,
            expected_keys=expected,
            expected_provenance=PROVENANCE,
            cache_directory=cache_directory,
        )

    changed = copy.deepcopy(manifest)
    changed["entries_sha256"] = "f" * 64
    with pytest.raises(ValueError):
        write_cache_manifest(
            path,
            changed,
            expected_keys=expected,
            expected_provenance=PROVENANCE,
            cache_directory=cache_directory,
        )

    different = copy.deepcopy(manifest)
    different["provenance"] = {
        **PROVENANCE,
        "config_sha256": "f" * 64,
    }
    with pytest.raises(FileExistsError):
        write_cache_manifest(
            path,
            different,
            expected_keys=expected,
            expected_provenance=different["provenance"],
        )

    assert not list(tmp_path.glob(".cache_manifest.json.*.tmp"))
