from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts.sonata_second_smoke import (
    build_load_interface_contract,
    classify_sonata_parameters,
    select_batch_configuration,
    summarize_gradients,
    validate_query_interface,
    validate_tiny_optimization,
)


def _parameter(value: float, *, trainable: bool, gradient: float | None):
    parameter = torch.nn.Parameter(torch.tensor([value]), requires_grad=trainable)
    if gradient is not None:
        parameter.grad = torch.tensor([gradient])
    return parameter


def test_freeze_and_gradient_contract_separates_encoder_decoder_and_heads() -> None:
    parameters = [
        (
            "model.backbone.model.embedding.stem.weight",
            _parameter(1.0, trainable=False, gradient=None),
        ),
        (
            "model.backbone.model.enc.blocks.0.weight",
            _parameter(1.0, trainable=False, gradient=None),
        ),
        (
            "model.backbone.model.dec.blocks.0.weight",
            _parameter(1.0, trainable=True, gradient=0.25),
        ),
        (
            "model.cross_attention.0.0.weight",
            _parameter(1.0, trainable=True, gradient=0.5),
        ),
        (
            "model.class_embed_head.weight",
            _parameter(1.0, trainable=True, gradient=0.75),
        ),
    ]

    groups = classify_sonata_parameters(parameters)
    summaries = {
        name: summarize_gradients(dict(parameters), members)
        for name, members in groups.items()
    }

    assert len(groups["frozen_encoder_embedding"]) == 2
    assert groups["trainable_sonata_decoder"]
    assert groups["trainable_rescene_decoder"]
    assert groups["trainable_rescene_heads"]
    assert summaries["frozen_encoder_embedding"]["nonzero_grad_tensors"] == 0
    assert summaries["trainable_sonata_decoder"]["finite"] is True
    assert summaries["trainable_sonata_decoder"]["nonzero_grad_tensors"] == 1
    assert summaries["trainable_rescene_heads"]["nonzero_grad_tensors"] == 1


def test_batch_selection_uses_largest_stable_divisor_without_accuracy() -> None:
    records = [
        {
            "microbatch_per_gpu": 1,
            "status": "stable",
            "finite_loss": True,
            "finite_gradients": True,
            "peak_vram_mib": 9000.0,
            "memory_total_mib": 46068.0,
            "samples_per_second": 1.0,
        },
        {
            "microbatch_per_gpu": 2,
            "status": "stable",
            "finite_loss": True,
            "finite_gradients": True,
            "peak_vram_mib": 18000.0,
            "memory_total_mib": 46068.0,
            "samples_per_second": 1.5,
        },
        {
            "microbatch_per_gpu": 4,
            "status": "oom",
            "finite_loss": False,
            "finite_gradients": False,
            "peak_vram_mib": 46000.0,
            "memory_total_mib": 46068.0,
            "samples_per_second": 0.0,
        },
    ]

    selected = select_batch_configuration(records, gpu_count=2)

    assert selected == {
        "gpu_count": 2,
        "microbatch_per_gpu": 2,
        "physical_global_batch": 4,
        "accumulate_grad_batches": 8,
        "effective_global_batch": 32,
        "selection_uses_validation_accuracy": False,
    }

    records[1]["finite_gradients"] = False
    selected = select_batch_configuration(records, gpu_count=2)
    assert selected["microbatch_per_gpu"] == 1


def test_query_interface_and_tiny_optimization_contract() -> None:
    output = {
        "pred_logits": torch.zeros(1, 100, 19),
        "pred_masks": [torch.zeros(12, 100)],
        "query_features": torch.zeros(1, 100, 128),
        "backbone_features": SimpleNamespace(F=torch.zeros(12, 96)),
        "aux_outputs": [{} for _ in range(12)],
    }

    interface = validate_query_interface(output, expected_batch_size=1)

    assert interface["query_feature_shape"] == [1, 100, 128]
    assert interface["mask_shapes"] == [[12, 100]]
    assert interface["official_output_extraction_compatible"] is True

    optimization = validate_tiny_optimization([9.0, 8.5, 8.7, 7.9])
    assert optimization["passed"] is True
    assert optimization["minimum_after_initial"] == 7.9

    with pytest.raises(ValueError, match="decrease"):
        validate_tiny_optimization([9.0, 9.1, 9.2, 9.3])


def test_load_interface_uses_ss1_audit_schema_without_null_counts() -> None:
    contract = build_load_interface_contract(
        {
            "gate": "SW0-PASS",
            "loaded_encoder_key_count": 453,
            "expected_decoder_missing_key_count": 248,
            "unexpected_keys": [],
        },
        decoder_parameter_tensor_count=248,
    )

    assert contract["loaded_encoder_key_count"] == 453
    assert contract["missing_decoder_key_count"] == 248
    assert contract["unexpected_key_count"] == 0
    assert contract["decoder_initialized"] is True


def test_batch_feasibility_csv_uses_repository_lf_endings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import scripts.sonata_second_smoke as smoke

    output = Path(tmp_path)
    monkeypatch.setattr(smoke, "PREFLIGHT_DIR", output)
    smoke._write_batch_artifacts(
        [
            {
                "schema_version": 1,
                "probe_scope": "real",
                "microbatch_per_gpu": 4,
                "gpu_count": 2,
                "physical_global_batch": 8,
                "status": "stable",
                "finite_loss": True,
                "finite_gradients": True,
                "loss": 1.0,
                "peak_allocated_vram_mib": 10.0,
                "peak_reserved_vram_mib": 12.0,
                "memory_total_mib": 100.0,
                "safe_headroom": True,
                "step_latency_seconds": 1.0,
                "samples_per_second": 4.0,
                "raw_points": 100,
                "voxels": 80,
                "validation_accuracy_inspected": False,
            }
        ],
        {
            "gpu_count": 2,
            "microbatch_per_gpu": 4,
            "physical_global_batch": 8,
            "accumulate_grad_batches": 4,
            "effective_global_batch": 32,
        },
        {
            "devices": [{"model": "NVIDIA A40"}],
            "selected_same_numa": True,
            "links": [{"interconnect": "NODE"}],
        },
        {},
    )

    assert b"\r" not in (output / "batch_feasibility.csv").read_bytes()
