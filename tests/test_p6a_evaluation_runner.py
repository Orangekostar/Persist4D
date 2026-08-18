from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.run_p6a_evaluation import (
    _argument_parser,
    _canonical_json_mapping,
    _declared_output_paths,
    run_p6a_evaluation,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def test_declared_outputs_cover_root_report_and_every_derived_file(
    tmp_path: Path,
) -> None:
    paths = _declared_output_paths(tmp_path / "P6A")
    relative = {path.relative_to(tmp_path / "P6A").as_posix() for path in paths}

    assert "p6a_eval.json" in relative
    assert "P6A_GO_NOGO_REPORT.md" in relative
    assert "efficiency_results.csv" in relative
    assert "figures/figure_e_latency.svg" in relative
    assert "configs/resolved_runtime.yaml" in relative
    assert len(relative) == len(paths) == 28


def test_canonical_json_loader_rejects_symlink_and_noncanonical_bytes(
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "canonical.json"
    _write_json(canonical, {"status": "pass", "value": 1})
    assert _canonical_json_mapping(canonical, name="input")["status"] == "pass"

    noncanonical = tmp_path / "noncanonical.json"
    noncanonical.write_text('{"status":"pass","value":1}\n')
    with pytest.raises(ValueError, match="canonical"):
        _canonical_json_mapping(noncanonical, name="input")

    linked = tmp_path / "linked.json"
    linked.symlink_to(canonical)
    with pytest.raises(ValueError, match="regular non-symlink"):
        _canonical_json_mapping(linked, name="input")


def test_runner_binds_inputs_evaluates_cache_and_publishes_only_after_finalize(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.run_p6a_evaluation as runner

    cache_root = tmp_path / "external-cache"
    output_root = tmp_path / "artifacts" / "P6A"
    metadata = tmp_path / "3RScan.json"
    checkpoint = tmp_path / "model.ckpt"
    metadata.write_text("metadata\n")
    checkpoint.write_bytes(b"checkpoint")
    protocol = SimpleNamespace(name="protocol")
    protocol_manifest = {"protocol": {"name": "common-prefix"}}
    cache_manifest = {
        "provenance": {
            "source_commit": "a" * 40,
            "checkpoint_sha256": "b" * 64,
            "config_sha256": "c" * 64,
            "dataset_sha256": "d" * 64,
        }
    }
    efficiency_manifest = {
        "status": "pass",
        "provenance": {
            "source_commit": "a" * 40,
            "checkpoint_sha256": "b" * 64,
            "config_sha256": "c" * 64,
            "protocol_sha256": "e" * 64,
            "cache_manifest_sha256": "f" * 64,
        },
    }
    _write_json(cache_root / "protocol_b_manifest.json", protocol_manifest)
    _write_json(cache_root / "cache_manifest.json", cache_manifest)
    _write_json(cache_root / "efficiency_raw_manifest.json", efficiency_manifest)
    (cache_root / "entries").mkdir()

    events: list[str] = []
    guard = SimpleNamespace(source_commit="a" * 40)
    config = SimpleNamespace(data=SimpleNamespace(validation_dataset={}))
    dataset = SimpleNamespace(name="dataset")
    evaluation = SimpleNamespace(sequence_count=129)
    artifact = {"status": "pass", "source_commit": "a" * 40}

    monkeypatch.setattr(runner, "_external_cache_directory", lambda path: path)
    monkeypatch.setattr(runner, "_repository_path", lambda path: path)
    monkeypatch.setattr(
        runner,
        "_begin_source_tree_contract",
        lambda **kwargs: events.append("begin") or guard,
    )
    monkeypatch.setattr(
        runner,
        "_finalize_source_tree_contract",
        lambda received: events.append("finalize")
        or {"status": "pass", "source_commit": received.source_commit},
    )
    monkeypatch.setattr(
        runner,
        "_frozen_protocol_bundle",
        lambda **kwargs: (
            protocol,
            protocol_manifest,
            b"baselines:\n  b4:\n    background_class: 18\n",
        ),
    )
    monkeypatch.setattr(
        runner,
        "_compose_runtime_config",
        lambda: (config, SimpleNamespace()),
    )
    monkeypatch.setattr(
        runner,
        "_runtime_config_text",
        lambda _config: "runtime: true\n",
    )
    monkeypatch.setattr(runner, "_resolve_checkpoint", lambda path: path)
    monkeypatch.setattr(
        runner,
        "_validate_frozen_inputs",
        lambda **kwargs: events.append("validate_inputs"),
    )
    sequences = tuple(SimpleNamespace(index=index) for index in range(129))
    monkeypatch.setattr(
        runner,
        "load_cached_protocol_sequences",
        lambda **kwargs: events.append("load_cache") or sequences,
    )
    monkeypatch.setattr(
        runner,
        "_instantiate_validation_dataset",
        lambda _config: events.append("dataset") or dataset,
    )
    monkeypatch.setattr(
        runner,
        "build_rio_class_mapper",
        lambda received: events.append("class_mapper")
        or (lambda value: value)
        if received is dataset
        else None,
    )
    monkeypatch.setattr(
        runner,
        "build_tracker_factories",
        lambda settings: events.append("trackers") or {"B4": object()},
    )
    monkeypatch.setattr(
        runner,
        "evaluate_cached_task_metrics",
        lambda received, **kwargs: events.append("evaluate") or evaluation,
    )
    monkeypatch.setattr(
        runner,
        "build_p6a_root_artifact",
        lambda **kwargs: events.append("build") or artifact,
    )
    monkeypatch.setattr(
        runner,
        "publish_root_artifact",
        lambda root, received: events.append("publish") or [root / "p6a_eval.json"],
    )

    result = run_p6a_evaluation(
        cache_directory=cache_root,
        metadata_path=metadata,
        checkpoint_path=checkpoint,
        output_root=output_root,
    )

    assert result is artifact
    assert events == [
        "begin",
        "validate_inputs",
        "load_cache",
        "dataset",
        "class_mapper",
        "trackers",
        "evaluate",
        "build",
        "finalize",
        "publish",
        "finalize",
    ]


def test_runner_rejects_repository_cache_before_source_setup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.run_p6a_evaluation as runner

    called = False

    def begin(**_kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(runner, "_begin_source_tree_contract", begin)
    with pytest.raises(ValueError, match="outside the repository"):
        run_p6a_evaluation(
            cache_directory=Path("artifacts/P6A/cache"),
            metadata_path=tmp_path / "metadata.json",
            checkpoint_path=tmp_path / "checkpoint.ckpt",
            output_root=tmp_path / "P6A",
        )
    assert called is False


def test_cli_defaults_to_repository_artifact_root() -> None:
    args = _argument_parser().parse_args(
        ["--cache-directory", "/external/cache", "--metadata", "/external/meta.json"]
    )

    assert args.output_root == Path("artifacts/P6A")
    assert args.efficiency_manifest is None
