from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_reviewer_closure as runner
from scripts.reviewer_closure_protocol import full_history_observation_keys

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPO_ROOT / "scripts/run_reviewer_closure.py"


def test_reviewer_closure_runner_exists() -> None:
    assert RUNNER_PATH.is_file()


def _api(name: str):
    value = getattr(runner, name, None)
    assert value is not None, f"missing reviewer-closure runner API: {name}"
    return value


def _manifest() -> dict[str, object]:
    return _api("build_bound_reviewer_manifest")()


def test_sidecar_execution_is_single_process_cuda_zero_only() -> None:
    validate = _api("validate_sidecar_execution")
    validate("cuda:0")

    error = _api("ReviewerClosureGateFailure")
    with pytest.raises(error, match="cuda:0|single"):
        validate("cuda:1")
    with pytest.raises(error, match="cuda:0|single"):
        validate("cpu")


def test_smoke_key_is_first_canonical_t2_and_resume_is_exact() -> None:
    keys = full_history_observation_keys(_manifest())
    smoke = _api("select_sidecar_smoke_key")(keys)

    assert smoke == next(
        key for key in keys if key["order_id"] == "canonical" and key["horizon"] == 2
    )
    existing = [{"key": keys[0]}, {"key": keys[7]}]
    pending = _api("select_pending_sidecar_keys")(keys, existing)
    assert len(pending) == len(keys) - 2
    assert keys[0] not in pending
    assert keys[7] not in pending

    unexpected = copy.deepcopy(existing)
    unexpected[0]["key"]["history_scan_ids"][-1] = "future_scan"
    error = _api("ReviewerClosureGateFailure")
    with pytest.raises(error, match="existing|expected|prefix"):
        _api("select_pending_sidecar_keys")(keys, unexpected)


def test_resume_rejects_sidecars_from_a_different_code_commit() -> None:
    validate = _api("validate_sidecar_resume_commit")
    validate([{"sidecar_source_commit": "a" * 40}], "a" * 40)
    validate([], "a" * 40)

    error = _api("ReviewerClosureGateFailure")
    with pytest.raises(error, match="commit|mixed"):
        validate([{"sidecar_source_commit": "b" * 40}], "a" * 40)


def test_bind_publishes_exact_reviewer_manifest_atomically(tmp_path: Path) -> None:
    output = tmp_path / "reviewer_closure_manifest.json"
    first = _api("run_bind")(output_path=output)
    second = _api("run_bind")(output_path=output)

    assert first == second
    assert json.loads(output.read_text(encoding="utf-8")) == first
    changed = json.loads(output.read_text(encoding="utf-8"))
    changed["name"] = "tampered"
    output.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(FileExistsError, match="different content"):
        _api("run_bind")(output_path=output)


def test_cli_exposes_only_gate_ordered_sidecar_commands() -> None:
    parser = _api("build_parser")()

    for command in ("bind", "sidecar-smoke", "cache-sidecars", "finalize-sidecars"):
        arguments = parser.parse_args([command])
        assert arguments.command == command
    with pytest.raises(SystemExit):
        parser.parse_args(["all"])


def test_cli_default_metadata_path_is_portable() -> None:
    arguments = _api("build_parser")().parse_args(["sidecar-smoke"])

    assert "/home/" not in str(arguments.metadata)
    assert "/Users/" not in str(arguments.metadata)


def test_sidecar_batch_processes_each_pending_key_once_and_smoke_is_bounded() -> None:
    keys = full_history_observation_keys(_manifest())[:3]
    loaded: list[str] = []
    produced: list[str] = []
    written: list[str] = []

    def identity(key):
        return json.dumps(key, sort_keys=True)

    def lookup(key):
        return {"key": key, "content_sha256": f"{len(loaded) + 1:064x}"}

    def load(record):
        loaded.append(identity(record["key"]))
        return {
            "key": record["key"],
            "content_sha256": record["content_sha256"],
        }

    class Producer:
        def produce_bundle(self, key):
            produced.append(identity(key))
            return SimpleNamespace(payload={}, processed=SimpleNamespace())

    def build(producer, source_prediction, sidecar_key, sidecar_source_commit):
        producer.produce_bundle(source_prediction["key"])
        assert sidecar_source_commit == "a" * 40
        return {"sidecar": {"key": sidecar_key}}

    def write(pair):
        written.append(identity(pair["sidecar"]["key"]))
        return {"key": pair["sidecar"]["key"]}

    result = _api("produce_sidecar_batch")(
        expected_keys=keys,
        existing_entries=[{"key": keys[0]}],
        source_lookup=lookup,
        source_loader=load,
        producer=Producer(),
        sidecar_builder=build,
        pair_writer=write,
        sidecar_source_commit="a" * 40,
        smoke_only=False,
    )
    assert result == {
        "expected_count": 3,
        "reused_count": 1,
        "produced_count": 2,
    }
    assert loaded == produced == written == [identity(keys[1]), identity(keys[2])]

    loaded.clear()
    produced.clear()
    written.clear()
    smoke_keys = full_history_observation_keys(_manifest())
    smoke = _api("select_sidecar_smoke_key")(smoke_keys)
    result = _api("produce_sidecar_batch")(
        expected_keys=smoke_keys,
        existing_entries=[],
        source_lookup=lookup,
        source_loader=load,
        producer=Producer(),
        sidecar_builder=build,
        pair_writer=write,
        sidecar_source_commit="a" * 40,
        smoke_only=True,
    )
    assert result["produced_count"] == 1
    assert written == [identity(smoke)]


def test_completed_replay_pairs_require_exact_keys_and_content_binding() -> None:
    keys = full_history_observation_keys(_manifest())[:2]
    source_entries = json.loads(
        (
            REPO_ROOT
            / "artifacts/system_comparison/full_history_predictions/manifest.json"
        ).read_text(encoding="utf-8")
    )["entries"]

    def full_key(key):
        return next(
            entry["key"]
            for entry in source_entries
            if all(
                entry["key"][name] == key[name]
                for name in (
                    "reference_scene_id",
                    "master_sequence_id",
                    "order_id",
                    "horizon",
                )
            )
        )

    sidecars = [
        {
            "key": key,
            "source_prediction_content_sha256": f"{index + 1:064x}",
        }
        for index, key in enumerate(keys)
    ]
    replays = [
        {
            "key": full_key(key),
            "content_sha256": f"{index + 1:064x}",
        }
        for index, key in enumerate(keys)
    ]
    assert _api("validate_completed_replay_pairs")(sidecars, replays) == sidecars

    error = _api("ReviewerClosureGateFailure")
    with pytest.raises(error, match="coverage|pair"):
        _api("validate_completed_replay_pairs")(sidecars, replays[:1])
    changed = copy.deepcopy(replays)
    changed[0]["content_sha256"] = "f" * 64
    with pytest.raises(error, match="content|pair"):
        _api("validate_completed_replay_pairs")(sidecars, changed)


def test_real_sidecar_command_fails_before_gpu_when_inputs_are_missing(
    tmp_path: Path,
) -> None:
    error = _api("ReviewerClosureGateFailure")
    with pytest.raises(error, match="metadata|source cache"):
        _api("run_sidecar_cache")(
            device_name="cuda:0",
            metadata_path=tmp_path / "missing-metadata.json",
            source_entry_root=tmp_path / "missing-source-cache",
            sidecar_entry_root=tmp_path / "sidecars",
            smoke_only=True,
        )


def test_finalize_refuses_incomplete_sidecar_coverage(tmp_path: Path) -> None:
    entries = tmp_path / "entries"
    entries.mkdir()
    reviewer_manifest = tmp_path / "reviewer_closure_manifest.json"
    _api("run_bind")(output_path=reviewer_manifest)
    error = _api("FullHistoryObservationSidecarError")
    with pytest.raises(error, match="exact coverage"):
        _api("run_finalize_sidecars")(
            sidecar_entry_root=entries,
            output_path=tmp_path / "manifest.json",
            reviewer_manifest_path=reviewer_manifest,
        )
