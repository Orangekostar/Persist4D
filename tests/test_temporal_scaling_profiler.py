import csv
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.profile_temporal_scaling as profiler
from scripts.profile_temporal_scaling import (
    CSV_REQUIRED_FIELDS,
    find_max_batch_size,
    gpu_provenance,
    render_plots_from_csv,
    select_reference_windows,
    summarize_rows,
    write_measurement_csv,
)


def _measurement_row(**overrides):
    row = {
        "run_id": "fixture",
        "mode": "inference",
        "T": 2,
        "batch_size": 1,
        "trial": 0,
        "precision": "fp32",
        "seed": 45,
        "sequence_names": "scene0219_00-scene0219_01",
        "num_points": 100,
        "num_voxels": 50,
        "peak_gpu_memory_mb": 1000.0,
        "wall_time_ms": 20.0,
        "samples_per_second": 50.0,
        "forward_backward_ms": 18.0,
        "max_batch_size_without_oom": "",
        "oom_observed": "",
        "gpu_name": "NVIDIA A40",
        "gpu_uuid": "redacted",
        "driver_version": "fixture",
        "torch_version": "fixture",
        "cuda_version": "fixture",
        "backbone": "concerto_base",
        "voxel_size": 0.02,
        "freeze_mode": "backbone_encoder",
        "source_commit": "fb2fe42eb8f1e926567c48eea9acb874e608ee10",
    }
    row.update(overrides)
    return row


def test_summary_uses_median_and_computes_throughput():
    rows = [
        {"T": 2, "mode": "inference", "batch_size": 1, "wall_time_ms": 10.0},
        {"T": 2, "mode": "inference", "batch_size": 1, "wall_time_ms": 30.0},
        {"T": 2, "mode": "inference", "batch_size": 1, "wall_time_ms": 20.0},
    ]

    summary = summarize_rows(rows)[0]

    assert summary["wall_time_ms"] == 20.0
    assert summary["wall_time_ms_min"] == 10.0
    assert summary["wall_time_ms_max"] == 30.0
    assert summary["samples_per_second"] == 50.0


def test_oom_search_reports_right_censoring_at_configured_cap():
    result = find_max_batch_size(lambda size: True, maximum=8)

    assert result == {"max_batch_size_without_oom": 8, "oom_observed": False}


def test_oom_search_doubles_then_binary_searches_true_maximum():
    import torch

    attempted = []

    def fits(batch_size):
        if batch_size > 5:
            raise torch.cuda.OutOfMemoryError("fixture oom")
        return True

    result = find_max_batch_size(fits, maximum=16, attempts=attempted)

    assert result == {"max_batch_size_without_oom": 5, "oom_observed": True}
    assert [attempt["batch_size"] for attempt in attempted if attempt["repetition"] == 1] == [1, 2, 4, 8, 6, 5]
    assert all(
        sum(a["outcome"] == "success" for a in attempted if a["batch_size"] == size) == 2
        for size in (1, 2, 4, 5)
    )


def test_oom_search_reports_failure_at_batch_one():
    import torch

    def never_fits(batch_size):
        raise torch.cuda.OutOfMemoryError("fixture oom")

    assert find_max_batch_size(never_fits, maximum=8) == {
        "max_batch_size_without_oom": 0,
        "oom_observed": True,
    }


def test_oom_search_does_not_swallow_regular_runtime_errors():
    def broken(batch_size):
        raise RuntimeError("not a CUDA OOM")

    with pytest.raises(RuntimeError, match="not a CUDA OOM"):
        find_max_batch_size(broken, maximum=8)


def test_csv_contains_required_measurement_schema(tmp_path):
    output = tmp_path / "profile.csv"

    write_measurement_csv([_measurement_row()], output)

    assert b"\r\n" not in output.read_bytes()
    with output.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames or []) == CSV_REQUIRED_FIELDS
        assert len(list(reader)) == 1


def test_csv_rejects_rows_missing_required_fields(tmp_path):
    row = _measurement_row()
    row.pop("gpu_uuid")

    with pytest.raises(ValueError, match="gpu_uuid"):
        write_measurement_csv([row], tmp_path / "profile.csv")


def test_csv_rejects_fields_outside_locked_schema(tmp_path):
    row = _measurement_row(extra_runtime_value=1)

    with pytest.raises(ValueError, match="unexpected CSV fields"):
        write_measurement_csv([row], tmp_path / "profile.csv")


def test_gpu_provenance_redacts_uuid_and_keeps_separate_device_alias(monkeypatch):
    import torch

    properties = SimpleNamespace(
        name="NVIDIA A40",
        total_memory=46_068 * 1024**2,
        major=8,
        minor=6,
    )
    commands = []

    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda device_index: properties
    )

    def fake_run(command, **kwargs):
        commands.append(command)
        if "uuid" in command[2]:
            stdout = "NVIDIA A40, GPU-private-machine-id, 595.71.05, 46068\n"
        else:
            stdout = "NVIDIA A40, 595.71.05, 46068\n"
        return SimpleNamespace(returncode=0, stdout=stdout)

    monkeypatch.setattr("scripts.profile_temporal_scaling.subprocess.run", fake_run)

    hardware = gpu_provenance(0)

    assert hardware["uuid"] == "redacted"
    assert hardware["device_alias"] == "device-0"
    assert hardware["uuid_redacted"] is True
    assert not hardware["uuid"].startswith("GPU-")
    assert commands[0][2] == "--query-gpu=name,driver_version,memory.total"


def test_run_profile_applies_and_records_official_matmul_precision(
    tmp_path, monkeypatch
):
    import torch

    precision_state = {"value": "highest"}
    events = []
    config = {
        "seed": 45,
        "horizons": [2],
        "sequence_type": "sliding",
        "precision": "fp32",
        "gpu_index": 0,
        "freeze_mode": "backbone_encoder",
        "voxel_size": 0.02,
        "warmup_iterations": 0,
        "measurement_iterations": 1,
        "profile_scenes": 5,
        "max_batch_search": 1,
        "oom_safety_margin_mb": 512,
    }

    def set_matmul_precision(value):
        events.append(("set_matmul_precision", value))
        precision_state["value"] = value

    def fake_profile_horizon(**kwargs):
        events.append(("profile_horizon", precision_state["value"]))
        return {
            "rows": [],
            "batch_search": {},
            "selected_windows": [],
            "sequence_database": {},
            "model": {},
        }

    monkeypatch.setattr(profiler, "load_yaml", lambda path: config)
    monkeypatch.setattr(profiler, "verify_official_source_tree", dict)
    monkeypatch.setattr(
        profiler, "validate_checkpoint", lambda path: {"sha256": "fixture-sha256"}
    )
    monkeypatch.setattr(
        profiler,
        "gpu_provenance",
        lambda device_index: {
            "name": "fixture",
            "uuid": "redacted",
            "device_alias": "device-0",
            "uuid_redacted": True,
        },
    )
    monkeypatch.setattr(profiler, "profile_horizon", fake_profile_horizon)
    monkeypatch.setattr(
        profiler,
        "render_plots_from_csv",
        lambda *args, **kwargs: {"summary": [], "plot_paths": []},
    )
    monkeypatch.setattr(profiler, "artifact_hashes", lambda paths: {})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device_index: None)
    monkeypatch.setattr(torch, "set_float32_matmul_precision", set_matmul_precision)
    monkeypatch.setattr(
        torch, "get_float32_matmul_precision", lambda: precision_state["value"]
    )
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", True)

    manifest = profiler.run_profile(
        SimpleNamespace(
            config=tmp_path / "config.yaml",
            processed_dir=tmp_path / "processed",
            checkpoint=tmp_path / "checkpoint.pth",
            horizons=[2],
            warmup_iterations=0,
            measurement_iterations=1,
            max_batch_search=1,
            reference_scene=219,
            run_id="fixture",
            skip_batch_search=True,
            output_dir=tmp_path / "output",
        )
    )

    assert events == [
        ("set_matmul_precision", "high"),
        ("profile_horizon", "high"),
    ]
    assert manifest["configuration"]["precision_contract"] == {
        "outer_execution": "fp32_without_autocast",
        "float32_matmul_precision": "high",
        "cuda_matmul_allow_tf32": True,
        "upstream_concerto_flash_attention_qkv": "explicit_fp16_cast",
    }
    assert manifest["hardware"]["uuid"] == "redacted"
    assert manifest["hardware"]["device_alias"] == "device-0"
    assert manifest["hardware"]["uuid_redacted"] is True


def test_batch_search_horizons_include_t2_and_skip_t3():
    assert getattr(profiler, "BATCH_SEARCH_HORIZONS", ()) == (2, 4, 5)


def test_markdown_discloses_t2_batch_search_and_non_strict_fp32():
    markdown = profiler.render_markdown(
        [],
        measurement_iterations=10,
        reference_scene=219,
        checkpoint_sha256="fixture",
        safety_margin_mb=512,
    )

    assert "## T=2/4/5 Maximum Batch Search" in markdown
    assert "FP32 outer tensors (TF32-eligible matmul high)" in markdown
    assert "TF32-eligible" in markdown
    assert "not strict IEEE FP32" in markdown


def test_selects_five_sorted_windows_from_same_validation_reference_scene():
    names = [f"window-{index}" for index in (4, 1, 3, 0, 2)]
    sequence_database = {
        name: {"type": "validation", "scene": 219, "sub_scenes": [0, 1, 2]}
        for name in names
    }
    sequence_database["train-window"] = {"type": "train", "scene": 219}
    sequence_database["other-scene"] = {"type": "validation", "scene": 220}
    dataset_sequence_names = ["other-scene", *names, "train-window"]

    selected = select_reference_windows(
        sequence_database,
        dataset_sequence_names,
        reference_scene=219,
        expected_count=5,
    )

    assert [item["sequence_name"] for item in selected] == sorted(names)
    assert [dataset_sequence_names[item["dataset_index"]] for item in selected] == sorted(names)


def test_select_reference_windows_rejects_non_cyclic_fixture_count():
    sequence_database = {
        f"window-{index}": {"type": "validation", "scene": 219}
        for index in range(4)
    }

    with pytest.raises(ValueError, match="exactly 5"):
        select_reference_windows(
            sequence_database,
            list(sequence_database),
            reference_scene=219,
            expected_count=5,
        )


def test_small_csv_generates_three_nonempty_pngs_and_exact_summary(tmp_path):
    rows = []
    for mode, memory_offset in (("inference", 0.0), ("training", 500.0)):
        for horizon in (2, 3):
            for trial, wall_time in enumerate((10.0, 30.0, 20.0)):
                rows.append(
                    _measurement_row(
                        mode=mode,
                        T=horizon,
                        trial=trial,
                        wall_time_ms=wall_time,
                        samples_per_second=1000.0 / wall_time,
                        forward_backward_ms=wall_time - 1.0,
                        peak_gpu_memory_mb=memory_offset + horizon * 100.0 + trial,
                    )
                )
    csv_path = tmp_path / "re_scene4d_scaling.csv"
    write_measurement_csv(rows, csv_path)

    result = render_plots_from_csv(csv_path, tmp_path, measurement_iterations=3)

    summary = next(
        row for row in result["summary"] if row["T"] == 2 and row["mode"] == "inference"
    )
    assert summary["wall_time_ms"] == 20.0
    assert summary["wall_time_ms_min"] == 10.0
    assert summary["wall_time_ms_max"] == 30.0
    assert summary["samples_per_second"] == 50.0
    assert result["measurement_count"] == 3
    for png_path in result["plot_paths"]:
        data = Path(png_path).read_bytes()
        assert data.startswith(b"\x89PNG\r\n\x1a\n")
        assert len(data) > 3000


def test_plot_render_rejects_missing_t_mode_group(tmp_path):
    rows = []
    group_counts = {
        (2, "inference"): 3,
        (2, "training"): 3,
        (3, "inference"): 3,
    }
    for (horizon, mode), count in group_counts.items():
        for trial in range(count):
            rows.append(_measurement_row(T=horizon, mode=mode, trial=trial))
    csv_path = tmp_path / "missing-group.csv"
    write_measurement_csv(rows, csv_path)

    with pytest.raises(
        ValueError, match=r"T=3/mode=training expected 3, found 0"
    ):
        render_plots_from_csv(csv_path, tmp_path / "plots", measurement_iterations=3)


def test_plot_render_rejects_uneven_t_mode_group_counts(tmp_path):
    rows = []
    for mode, count in (("inference", 3), ("training", 2)):
        for trial in range(count):
            rows.append(_measurement_row(mode=mode, trial=trial))
    csv_path = tmp_path / "uneven-groups.csv"
    write_measurement_csv(rows, csv_path)

    with pytest.raises(
        ValueError, match=r"T=2/mode=training expected 3, found 2"
    ):
        render_plots_from_csv(csv_path, tmp_path / "plots", measurement_iterations=3)
