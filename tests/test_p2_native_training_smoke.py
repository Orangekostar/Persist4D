import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts import run_p2_native_smoke as smoke

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT = REPO_ROOT / "artifacts" / "P2" / "native_smoke_report.json"


def test_checkpoint_provenance_is_portable_and_pinned() -> None:
    provenance = smoke.checkpoint_provenance(
        Path("/" + "home" + "/fixture/.cache/persist4d/concerto/concerto_base.pth"),
        sha256=smoke.CONCERTO_CHECKPOINT_SHA256,
    )

    assert provenance == {
        "reference": "local_cache:persist4d/concerto/concerto_base.pth",
        "sha256": "845ec7dec97a5fabff8fadb5d9858ac6734347b612d1a4b574213419c139de07",
    }
    assert "/" + "home" + "/" not in json.dumps(provenance)


def test_objective_breakdown_weights_segmentation_and_excludes_diagnostics() -> None:
    losses = {
        "loss_ce": torch.tensor(1.0),
        "loss_mask": torch.tensor(2.0),
        "loss_dice": torch.tensor(3.0),
        "loss_ce_0": torch.tensor(4.0),
        "loss_segment_contrastive": torch.tensor(5.0),
        "loss_aux_contrastive": torch.tensor(6.0),
        "loss_segment_contrastive_layer0": torch.tensor(100.0),
        "loss_aux_contrastive_layer_0": torch.tensor(200.0),
    }
    weight_dict = {
        "loss_ce": 2.0,
        "loss_mask": 5.0,
        "loss_dice": 2.0,
        "loss_ce_0": 2.0,
    }

    breakdown = smoke.objective_breakdown(losses, weight_dict)

    assert breakdown["final_head_segmentation"].item() == 18.0
    assert breakdown["all_segmentation"].item() == 26.0
    assert breakdown["aggregate_contrastive"].item() == 11.0
    assert breakdown["objective"].item() == 37.0
    assert set(breakdown["diagnostic_keys"]) == {
        "loss_segment_contrastive_layer0",
        "loss_aux_contrastive_layer_0",
    }


def test_parameter_groups_identify_frozen_concerto_and_trainable_heads() -> None:
    named_parameters = [
        (
            "model.backbone.model.embedding.proj.weight",
            torch.nn.Parameter(torch.ones(1)),
        ),
        ("model.backbone.model.enc.block.weight", torch.nn.Parameter(torch.ones(1))),
        ("model.backbone.model.dec.block.weight", torch.nn.Parameter(torch.ones(1))),
        ("model.class_embed_head.weight", torch.nn.Parameter(torch.ones(1))),
        ("model.mask_features_head.weight", torch.nn.Parameter(torch.ones(1))),
        ("model.cross_attention.0.weight", torch.nn.Parameter(torch.ones(1))),
        ("criterion.temperature", torch.nn.Parameter(torch.ones(1))),
    ]
    for name, parameter in named_parameters:
        if ".embedding." in name or ".enc." in name:
            parameter.requires_grad_(False)

    groups = smoke.classify_parameters(named_parameters)

    assert groups["frozen_encoder"] == [
        "model.backbone.model.embedding.proj.weight",
        "model.backbone.model.enc.block.weight",
    ]
    assert groups["trainable_concerto_decoder"] == [
        "model.backbone.model.dec.block.weight"
    ]
    assert groups["trainable_rescene_heads"] == [
        "model.class_embed_head.weight",
        "model.mask_features_head.weight",
    ]
    assert groups["trainable_rescene_decoder"] == ["model.cross_attention.0.weight"]
    assert groups["trainable_objective"] == ["criterion.temperature"]


def test_parameter_groups_do_not_hide_trainable_encoder_parameters() -> None:
    encoder = torch.nn.Parameter(torch.ones(1), requires_grad=True)

    groups = smoke.classify_parameters(
        [("model.backbone.model.enc.block.weight", encoder)]
    )

    assert groups["frozen_encoder"] == ["model.backbone.model.enc.block.weight"]
    assert groups["trainable_rescene_decoder"] == []


def test_required_tmap_schema_accepts_real_metric_keys() -> None:
    keys = {
        "val_mean_t-AP",
        "val_mean_t-AP_50",
        "val_mean_t-AP_25",
        "val_mean_AP",
        "val_mean_stage1-AP",
        "val_mean_stage2-AP",
    }

    assert smoke.validate_tmap_schema(keys) == sorted(keys)


def test_required_tmap_schema_rejects_missing_head() -> None:
    with pytest.raises(ValueError, match="val_mean_stage2-AP"):
        smoke.validate_tmap_schema(
            {
                "val_mean_t-AP",
                "val_mean_t-AP_50",
                "val_mean_t-AP_25",
                "val_mean_AP",
                "val_mean_stage1-AP",
            }
        )


def test_matcher_classification_accuracy_excludes_no_object_logit() -> None:
    class Matcher:
        def __call__(self, outputs, targets, mask_type):
            return [(torch.tensor([0]), torch.tensor([0]))]

    system = SimpleNamespace(
        criterion=SimpleNamespace(matcher=Matcher()),
        mask_type="segment_mask",
    )
    output = {
        "pred_logits": torch.tensor([[[0.1, 2.0, 4.0]]]),
        "pred_masks": [torch.tensor([[8.0]])],
        "aux_outputs": [],
    }
    targets = [
        {
            "labels": torch.tensor([1]),
            "segment_mask": torch.tensor([[True]]),
        }
    ]

    quality = smoke._matching_quality(system, output, targets)

    assert quality["classification_accuracy"] == 1.0
    assert quality["mean_dice"] > 0.99


@pytest.mark.gpu
@pytest.mark.skipif(
    os.environ.get("P2_VERIFY_GPU_ARTIFACTS") != "1",
    reason="set P2_VERIFY_GPU_ARTIFACTS=1 after the real single-A40 run",
)
def test_real_native_smoke_artifact_passes_all_gates() -> None:
    payload = json.loads(REPORT.read_text(encoding="utf-8"))

    assert payload["scope"] == "preflight-only"
    assert payload["official_mixed_data_reproduction"] is False
    assert payload["g2_evidence"] is False
    assert payload["hardware"]["device_alias"] == "device-0"
    assert "uuid" not in payload["hardware"]
    assert payload["smoke"]["passed"] is True
    assert payload["smoke"]["first_step_model_bitwise_changed"] is True
    assert payload["smoke"]["encoder_bitwise_unchanged"] is True
    assert payload["smoke"]["decoder_head_changed"] is True
    assert payload["smoke"]["segment_contrastive_positive"] is True
    assert payload["checkpoint_roundtrip"]["passed"] is True
    assert payload["checkpoint_roundtrip"]["kind"] == (
        "native_model_optimizer_scheduler_state_roundtrip"
    )
    assert payload["checkpoint_roundtrip"]["lightning_full_resume_validation"] is False
    assert payload["checkpoint_roundtrip"]["advanced_model_bitwise_changed"] is True
    assert payload["checkpoint_roundtrip"]["advanced_optimizer_state_changed"] is True
    assert payload["validation_evaluator"]["pipeline_executed"] is True
    assert payload["validation_evaluator"]["g2_metric_evidence"] is False
    assert payload["validation_evaluator"]["model_state"] == (
        "pretrained_concerto_encoder_with_seeded_randomly_initialized_decoder_"
        "and_heads_after_two_native_smoke_steps"
    )
    smoke.validate_tmap_schema(payload["validation_evaluator"]["schema_keys"])
    serialized = json.dumps(payload, sort_keys=True)
    assert "/" + "home" + "/" not in serialized
    assert "GPU" + "-" not in serialized
