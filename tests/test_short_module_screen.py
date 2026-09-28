"""Runner configuration cannot broaden the frozen single-module experiment."""

from pathlib import Path

import pytest
import yaml

from scripts.short_module_screen import read_config, run_pipeline


def test_config_rejects_multiple_modules_or_training_parent(tmp_path):
    source = Path(__file__).resolve().parents[1] / "configs/short_module_screen_v1.yaml"
    config = read_config(source)
    assert config["scope"]["train_arms"] == ["Q1", "Q2", "Q3", "M0", "M1", "M2"]
    config["parent"]["frozen"] = False
    modified = tmp_path / "bad.yaml"
    modified.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="frozen protocol"):
        read_config(modified)


def test_run_keeps_six_arm_stage_order_and_partial_delivery(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from scripts import short_module_screen as runner

    monkeypatch.setattr(runner, "ARTIFACTS", tmp_path / "public")
    commands = []

    def execute(command, **kwargs):
        commands.append(command)
        assert kwargs["env"]["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
        return SimpleNamespace(returncode=1 if command[3] == "train" else 0)

    monkeypatch.setattr(runner.subprocess, "run", execute)
    args = SimpleNamespace(root=tmp_path, config=tmp_path / "config.yaml", device="cuda:0", resume=True)
    status = run_pipeline(args)
    assert [c[3] for c in commands] == ["prepare", "export", "baseline", "train", "evaluate-cal",
                                     "screen", "replicate", "diagnostics", "profile", "report", "publish"]
    assert status["train"] == "EXIT_1"
    assert status["publish"] == "COMPLETE"
    assert all("--module" not in c for c in commands)
