from __future__ import annotations

import copy
import importlib
import json
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]


def _runner():
    return importlib.import_module("scripts.run_r1_downstream_validation")


def _provenance() -> dict[str, str]:
    return {
        "checkpoint_sha256": "b" * 64,
        "config_sha256": "c" * 64,
        "dataset_sha256": "d" * 64,
        "source_commit": "a" * 40,
    }


def test_cache_progress_resumes_only_missing_exact_keys() -> None:
    runner = _runner()
    keys = [{"key": index} for index in range(4)]
    progress = runner.new_cache_progress(
        cache_kind="local",
        provenance=_provenance(),
        protocol_sha256="e" * 64,
    )
    progress["records"] = [{"key": keys[0]}, {"key": keys[2]}]

    pending = runner.resume_pending_keys(
        expected_keys=keys,
        progress=progress,
        cache_kind="local",
        provenance=_provenance(),
        protocol_sha256="e" * 64,
    )

    assert pending == (keys[1], keys[3])
    duplicate = copy.deepcopy(progress)
    duplicate["records"].append({"key": keys[0]})
    with pytest.raises(runner.R1RunError, match="duplicate"):
        runner.resume_pending_keys(
            expected_keys=keys,
            progress=duplicate,
            cache_kind="local",
            provenance=_provenance(),
            protocol_sha256="e" * 64,
        )
    foreign = copy.deepcopy(progress)
    foreign["records"].append({"key": {"key": 99}})
    with pytest.raises(runner.R1RunError, match="unexpected"):
        runner.resume_pending_keys(
            expected_keys=keys,
            progress=foreign,
            cache_kind="local",
            provenance=_provenance(),
            protocol_sha256="e" * 64,
        )


def test_cache_progress_rejects_checkpoint_or_code_binding_drift() -> None:
    runner = _runner()
    progress = runner.new_cache_progress(
        cache_kind="local",
        provenance=_provenance(),
        protocol_sha256="e" * 64,
    )

    changed = _provenance()
    changed["checkpoint_sha256"] = "f" * 64
    with pytest.raises(runner.R1RunError, match="binding"):
        runner.resume_pending_keys(
            expected_keys=[{"key": 0}],
            progress=progress,
            cache_kind="local",
            provenance=changed,
            protocol_sha256="e" * 64,
        )
    changed = _provenance()
    changed["source_commit"] = "9" * 40
    with pytest.raises(runner.R1RunError, match="binding"):
        runner.resume_pending_keys(
            expected_keys=[{"key": 0}],
            progress=progress,
            cache_kind="local",
            provenance=changed,
            protocol_sha256="e" * 64,
        )


def test_final_cache_manifest_requires_both_exact_645_entry_caches() -> None:
    runner = _runner()
    local_keys = [{"kind": "local", "key": index} for index in range(645)]
    full_keys = [{"kind": "full", "key": index} for index in range(645)]
    local = runner.new_cache_progress(
        cache_kind="local",
        provenance=_provenance(),
        protocol_sha256="e" * 64,
    )
    full_provenance = {
        key: value for key, value in _provenance().items() if key != "dataset_sha256"
    }
    full_provenance["protocol_sha256"] = "e" * 64
    full = runner.new_cache_progress(
        cache_kind="full_history",
        provenance=full_provenance,
        protocol_sha256="e" * 64,
    )
    local["status"] = "pass"
    full["status"] = "pass"
    local["records"] = [{"key": key} for key in local_keys]
    full["records"] = [{"key": key} for key in full_keys]

    manifest = runner.finalize_cache_manifest(
        local_progress=local,
        full_progress=full,
        expected_local_keys=local_keys,
        expected_full_keys=full_keys,
        local_provenance=_provenance(),
        full_provenance=full_provenance,
        protocol_sha256="e" * 64,
    )

    assert manifest["status"] == "pass"
    assert manifest["checkpoint_sha256"] == "b" * 64
    assert manifest["source_commit"] == "a" * 40
    assert manifest["local"]["entry_count"] == 645
    assert manifest["full_history"]["entry_count"] == 645
    assert len(manifest["local"]["records_sha256"]) == 64
    assert len(manifest["full_history"]["records_sha256"]) == 64

    full["records"].pop()
    with pytest.raises(runner.R1RunError, match="coverage"):
        runner.finalize_cache_manifest(
            local_progress=local,
            full_progress=full,
            expected_local_keys=local_keys,
            expected_full_keys=full_keys,
            local_provenance=_provenance(),
            full_provenance=full_provenance,
            protocol_sha256="e" * 64,
        )


def test_generation_is_single_process_cuda_zero_only() -> None:
    runner = _runner()

    assert runner.validate_cache_execution("cuda:0") == "cuda:0"
    for device in ("cpu", "cuda", "cuda:1"):
        with pytest.raises(runner.R1RunError, match="cuda:0"):
            runner.validate_cache_execution(device)


def test_smoke_pairs_are_six_cluster_distinct_canonical_t2_prefixes() -> None:
    runner = _runner()
    from scripts.system_comparison_inference import full_history_cache_keys

    manifest = json.loads(
        (REPO_ROOT / "artifacts/P6A/protocol_b_manifest.json").read_text()
    )
    pairs = runner.select_smoke_pairs(
        runner.local_cache_keys(manifest),
        full_history_cache_keys(manifest),
    )
    from scripts.profile_system_comparison import build_profile_subset

    assert len(pairs) == 6
    assert len({local["reference_scene_id"] for local, _full in pairs}) == 6
    assert [local["master_sequence_id"] for local, _full in pairs] == [
        unit.master_sequence_id for unit in build_profile_subset(manifest)
    ]
    for local, full in pairs:
        assert local["order_id"] == full["order_id"] == "canonical"
        assert local["stage_index"] == 1
        assert full["horizon"] == 2
        assert local["master_sequence_id"] == full["master_sequence_id"]
        assert local["history_scan_ids"] == full["history_scan_ids"]


def test_smoke_repeat_plan_adds_one_forward_to_only_two_inputs() -> None:
    runner = _runner()

    assert [runner.smoke_repeat_count(index) for index in range(6)] == [
        2,
        2,
        1,
        1,
        1,
        1,
    ]


def test_query_feature_export_parity_preserves_nonfeature_outputs() -> None:
    runner = _runner()
    disabled = {
        "pred_logits": torch.tensor([[[1.0, 2.0]]]),
        "pred_masks": [torch.tensor([[3.0]])],
        "aux_outputs": [{"pred_logits": torch.tensor([[[4.0, 5.0]]])}],
    }
    enabled = {
        **disabled,
        "query_features": torch.ones(1, 1, 128),
    }

    assert runner.validate_query_feature_export_parity(
        disabled, enabled
    ) == {
        "status": "pass",
        "legacy_predictions_unchanged": True,
        "query_feature_shape": [1, 1, 128],
    }
    changed = {**enabled, "pred_logits": torch.tensor([[[1.0, 3.0]]])}
    with pytest.raises(runner.R1RunError, match="query feature export"):
        runner.validate_query_feature_export_parity(disabled, changed)


def test_t2_parity_metric_uses_explicit_data_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _runner()
    specification = tmp_path / "data/processed/rio/rio.yaml"
    specification.parent.mkdir(parents=True)
    specification.write_text("dataset: rio\n", encoding="utf-8")
    captured = {}

    def metric(prediction, target, *, dataset_spec):
        captured["prediction"] = prediction
        captured["target"] = target
        captured["dataset_spec"] = dataset_spec
        return {"raw_local_AP": 1.0}

    monkeypatch.setattr(
        "scripts.p6a_metrics.compute_official_raw_local_metrics", metric
    )
    function = runner.build_t2_parity_metric(tmp_path)

    assert function("prediction", "target") == {"raw_local_AP": 1.0}
    assert captured["dataset_spec"] == specification.resolve()


def test_resumable_materializer_publishes_each_record_once(tmp_path: Path) -> None:
    runner = _runner()
    keys = [{"key": index} for index in range(3)]
    progress_path = tmp_path / "progress.json"
    produced: list[int] = []

    def produce(key: dict[str, int]) -> dict[str, object]:
        produced.append(key["key"])
        return {"key": key, "value": key["key"] * 10}

    def validate(record: dict[str, object]) -> None:
        assert record["value"] == record["key"]["key"] * 10

    progress = runner.materialize_records(
        cache_kind="local",
        expected_keys=keys,
        provenance=_provenance(),
        protocol_sha256="e" * 64,
        progress_path=progress_path,
        produce_record=produce,
        validate_record=validate,
    )

    assert progress["status"] == "pass"
    assert produced == [0, 1, 2]
    assert json.loads(progress_path.read_text()) == progress

    produced.clear()
    resumed = runner.materialize_records(
        cache_kind="local",
        expected_keys=keys,
        provenance=_provenance(),
        protocol_sha256="e" * 64,
        progress_path=progress_path,
        produce_record=produce,
        validate_record=validate,
    )
    assert resumed == progress
    assert produced == []


def test_cli_defaults_to_registered_single_a40_and_shared_cache() -> None:
    runner = _runner()

    arguments = runner.argument_parser().parse_args(["cache-local"])

    assert arguments.stage == "cache-local"
    assert arguments.device == "cuda:0"
    assert arguments.data_root == Path("/home/ww/paper5")
    assert arguments.cache_root == Path(
        "/mnt/shared/ww/persist4d-r1-downstream-validation-v1/cache"
    )
    assert arguments.checkpoint.name == (
        "629ff7624dcac15e6022906e808e2e05b3ec61c60a1116ab0e278f0cfd2368dd.ckpt"
    )
    assert runner.argument_parser().parse_args(["smoke"]).stage == "smoke"
    assert runner.argument_parser().parse_args(["cache-parity"]).stage == (
        "cache-parity"
    )
    assert runner.argument_parser().parse_args(["cache-full"]).stage == "cache-full"
    assert runner.argument_parser().parse_args(["finalize-cache"]).stage == (
        "finalize-cache"
    )
