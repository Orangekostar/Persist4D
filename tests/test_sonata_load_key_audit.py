from __future__ import annotations

import pytest

from utils.sonata_weight_provenance import (
    SonataLoadKeyError,
    build_sonata_load_key_audit,
)

MODEL_KEYS = {
    "embedding.stem.weight",
    "embedding.norm.weight",
    "enc.enc0.block.weight",
    "enc.enc1.block.weight",
    "dec.dec0.block.weight",
    "dec.dec1.block.weight",
}
ENCODER_CHECKPOINT_KEYS = {
    "embedding.stem.weight",
    "embedding.norm.weight",
    "enc.enc0.block.weight",
    "enc.enc1.block.weight",
}
DECODER_MISSING_KEYS = {
    "dec.dec0.block.weight",
    "dec.dec1.block.weight",
}


def test_encoder_complete_decoder_missing_audit_passes() -> None:
    audit = build_sonata_load_key_audit(
        checkpoint_keys=ENCODER_CHECKPOINT_KEYS,
        model_keys=MODEL_KEYS,
        missing_keys=DECODER_MISSING_KEYS,
        unexpected_keys=(),
        weight_sha256="a" * 64,
    )

    assert audit["status"] == "pass"
    assert audit["gate"] == "SW0-PASS"
    assert audit["loaded_encoder_key_count"] == 4
    assert audit["expected_decoder_missing_key_count"] == 2
    assert audit["missing_keys"] == sorted(DECODER_MISSING_KEYS)
    assert audit["unexpected_keys"] == []
    assert audit["critical_encoder_missing_keys"] == []


@pytest.mark.parametrize(
    "missing_key",
    ["embedding.stem.weight", "enc.enc1.block.weight"],
)
def test_missing_encoder_or_embedding_key_fails_closed(missing_key: str) -> None:
    with pytest.raises(SonataLoadKeyError, match="critical encoder"):
        build_sonata_load_key_audit(
            checkpoint_keys=ENCODER_CHECKPOINT_KEYS - {missing_key},
            model_keys=MODEL_KEYS,
            missing_keys=DECODER_MISSING_KEYS | {missing_key},
            unexpected_keys=(),
            weight_sha256="a" * 64,
        )


def test_non_decoder_missing_key_fails_closed() -> None:
    with pytest.raises(SonataLoadKeyError, match="non-decoder missing"):
        build_sonata_load_key_audit(
            checkpoint_keys=ENCODER_CHECKPOINT_KEYS,
            model_keys=MODEL_KEYS | {"head.weight"},
            missing_keys=DECODER_MISSING_KEYS | {"head.weight"},
            unexpected_keys=(),
            weight_sha256="a" * 64,
        )


def test_unexplained_unexpected_key_fails_closed() -> None:
    with pytest.raises(SonataLoadKeyError, match="unexpected"):
        build_sonata_load_key_audit(
            checkpoint_keys=ENCODER_CHECKPOINT_KEYS | {"mystery.weight"},
            model_keys=MODEL_KEYS,
            missing_keys=DECODER_MISSING_KEYS,
            unexpected_keys=("mystery.weight",),
            weight_sha256="a" * 64,
        )


def test_empty_encoder_checkpoint_fails_closed() -> None:
    with pytest.raises(SonataLoadKeyError, match="no encoder"):
        build_sonata_load_key_audit(
            checkpoint_keys=(),
            model_keys=MODEL_KEYS,
            missing_keys=MODEL_KEYS,
            unexpected_keys=(),
            weight_sha256="a" * 64,
        )
