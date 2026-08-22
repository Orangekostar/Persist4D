from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

import scripts.run_system_comparison as runner
from scripts.run_system_comparison import (
    GateFailure,
    _argument_parser,
    build_reproducibility_binding,
    oracle_attribution_required,
    publish_exact_json,
    run_stage_pipeline,
    select_disjoint_shard,
    select_pending_keys,
    select_smoke_keys,
    verify_determinism_repeats,
    verify_incumbent_regression,
    verify_t2_regression_pairs,
)
from scripts.system_comparison_inference import (
    build_full_history_cache_manifest,
    full_history_cache_keys,
)
from scripts.system_comparison_protocol import build_system_comparison_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = REPO_ROOT / "artifacts/P6A/protocol_b_manifest.json"


def _system_manifest() -> dict[str, object]:
    digest = hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest()
    return build_system_comparison_manifest(
        PROTOCOL_PATH,
        incumbent_binding={
            "status": "pass",
            "p6a_protocol_manifest_sha256": digest,
        },
    )


def _provenance() -> dict[str, str]:
    return {
        "source_commit": "a" * 40,
        "checkpoint_sha256": "b" * 64,
        "config_sha256": "c" * 64,
        "protocol_sha256": "d" * 64,
    }


def test_disjoint_shards_have_exact_ordered_coverage() -> None:
    values = tuple(range(29))
    shards = [select_disjoint_shard(values, index, 3) for index in range(3)]

    assert [value for value in values if value in set().union(*map(set, shards))] == list(
        values
    )
    assert sum(len(shard) for shard in shards) == len(values)
    assert not (set(shards[0]) & set(shards[1]))
    assert not (set(shards[0]) & set(shards[2]))
    assert not (set(shards[1]) & set(shards[2]))
    with pytest.raises(ValueError, match="shard"):
        select_disjoint_shard(values, 3, 3)


def test_stage_pipeline_resumes_completed_prefix_and_stops_on_failure() -> None:
    calls: list[str] = []

    def action(name: str, *, fail: bool = False):
        def run() -> str:
            calls.append(name)
            if fail:
                raise GateFailure(name)
            return name

        return run

    completed = run_stage_pipeline(
        (("bind", action("bind")), ("smoke", action("smoke"))),
        completed=("bind",),
    )
    assert completed == ("bind", "smoke")
    assert calls == ["smoke"]

    with pytest.raises(GateFailure, match="evaluate"):
        run_stage_pipeline(
            (
                ("bind", action("bind-again")),
                ("evaluate", action("evaluate", fail=True)),
                ("profile", action("profile")),
            )
        )
    assert "profile" not in calls


def test_pending_cache_keys_resume_without_recomputing_published_entries() -> None:
    keys = ({"key": 0}, {"key": 1}, {"key": 2})
    assert select_pending_keys(keys, (keys[0], keys[2])) == (keys[1],)
    with pytest.raises(GateFailure, match="unexpected"):
        select_pending_keys(keys, ({"key": 4},))


def test_full_history_manifest_requires_exact_coverage_and_provenance() -> None:
    keys = full_history_cache_keys(_system_manifest())
    entries = [
        {
            "key": key,
            "filename": (
                hashlib.sha256(
                    json.dumps(
                        key,
                        ensure_ascii=True,
                        separators=(",", ":"),
                        sort_keys=True,
                    ).encode("utf-8")
                ).hexdigest()
                + ".pt"
            ),
            "sha256": f"{index + 1:064x}",
            "byte_size": index + 1,
            "content_sha256": f"{index + 2:064x}",
        }
        for index, key in enumerate(keys)
    ]
    manifest = build_full_history_cache_manifest(
        entries,
        expected_keys=keys,
        expected_provenance=_provenance(),
    )
    assert manifest["entry_count"] == 645
    assert manifest["provenance"] == _provenance()

    with pytest.raises(ValueError, match="coverage"):
        build_full_history_cache_manifest(
            entries[:-1],
            expected_keys=keys,
            expected_provenance=_provenance(),
        )
    mismatched = copy.deepcopy(entries)
    mismatched[0]["provenance"] = {**_provenance(), "source_commit": "e" * 40}
    with pytest.raises(ValueError, match="fields"):
        build_full_history_cache_manifest(
            mismatched,
            expected_keys=keys,
            expected_provenance=_provenance(),
        )


def test_publish_exact_json_resumes_equal_content_and_refuses_mismatch(
    tmp_path: Path,
) -> None:
    target = tmp_path / "binding.json"
    publish_exact_json(target, {"status": "pass", "value": 1})
    publish_exact_json(target, {"value": 1, "status": "pass"})

    with pytest.raises(FileExistsError, match="different content"):
        publish_exact_json(target, {"status": "pass", "value": 2})


def test_binding_hashes_every_frozen_input(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.ckpt"
    config = tmp_path / "incumbent.yaml"
    protocol = tmp_path / "protocol.json"
    checkpoint.write_bytes(b"checkpoint")
    config.write_bytes(b"config")
    protocol.write_bytes(b"protocol")
    binding = build_reproducibility_binding(
        source_commit="a" * 40,
        checkpoint_path=checkpoint,
        config_path=config,
        protocol_path=protocol,
        system_manifest={"content_sha256": "e" * 64},
    )

    assert binding["source_commit"] == "a" * 40
    assert binding["checkpoint_sha256"] == hashlib.sha256(b"checkpoint").hexdigest()
    assert binding["config_sha256"] == hashlib.sha256(b"config").hexdigest()
    assert binding["protocol_sha256"] == hashlib.sha256(b"protocol").hexdigest()
    assert binding["system_manifest_sha256"] == "e" * 64


def test_incumbent_regression_blocks_any_metric_outside_absolute_tolerance() -> None:
    reference = {
        f"T{horizon}": {
            "t_mAP": horizon / 100,
            "t_mAP50": horizon / 50,
            "t_mAP25": horizon / 25,
            "t_REC": horizon / 80,
        }
        for horizon in range(2, 6)
    }
    assert verify_incumbent_regression(reference, reference, tolerance=1e-12)[
        "status"
    ] == "pass"

    observed = copy.deepcopy(reference)
    observed["T5"]["t_mAP"] += 2e-12
    with pytest.raises(GateFailure, match="T5.t_mAP"):
        verify_incumbent_regression(observed, reference, tolerance=1e-12)


def test_t2_regression_and_three_repeat_determinism_are_blocking() -> None:
    matched = {"combined": "a" * 64}
    full = [{"observation_fingerprints": matched}]
    local = [{"observation_fingerprints": matched}]
    assert verify_t2_regression_pairs(full, local)["status"] == "pass"

    with pytest.raises(GateFailure, match="T2"):
        verify_t2_regression_pairs(
            full,
            [{"observation_fingerprints": {"combined": "b" * 64}}],
        )

    repeats = [["a" * 64, "b" * 64]] * 3
    assert verify_determinism_repeats(repeats)["status"] == "pass"
    with pytest.raises(GateFailure, match="determinism"):
        verify_determinism_repeats([repeats[0], repeats[1], ["c" * 64, "b" * 64]])


def test_smoke_selection_and_oracle_trigger_follow_preregistration() -> None:
    selected = select_smoke_keys(full_history_cache_keys(_system_manifest()))
    assert len(selected) == 3
    assert {row["order_id"] for row in selected} == {"canonical"}
    assert {row["horizon"] for row in selected} == {5}
    assert len({row["reference_scene_id"] for row in selected}) == 3

    assert oracle_attribution_required(
        persist4d={"T4": 0.10, "T5": 0.09},
        full_history={"T4": 0.111, "T5": 0.10},
        paired_ci={"T4": (-0.02, -0.001), "T5": (-0.02, 0.0)},
        minimum_advantage=0.01,
    )
    assert not oracle_attribution_required(
        persist4d={"T4": 0.10, "T5": 0.09},
        full_history={"T4": 0.111, "T5": 0.10},
        paired_ci={"T4": (-0.02, 0.001), "T5": (-0.02, 0.0)},
        minimum_advantage=0.01,
    )


def test_all_pipeline_ends_with_final_artifact_gate(monkeypatch, capsys) -> None:
    captured: list[str] = []

    def fake_pipeline(stages, *, completed=()):
        del completed
        captured.extend(name for name, _action in stages)
        return tuple(captured)

    monkeypatch.setattr(runner, "run_stage_pipeline", fake_pipeline)

    assert runner.main(["all"]) == 0
    assert captured[-2:] == ["profile", "artifacts"]
    assert _argument_parser().parse_args(["artifacts"]).command == "artifacts"
    assert '"status": "pass"' in capsys.readouterr().out
