import pytest


def test_unlocked_checkpoint_or_role_cannot_run():
    from scripts.native_long_confirmation_inference import validate_request

    lock = {"training_seed": 45, "evaluation_seed": 9001, "rows": [{"arm": "E3", "step": 2970}]}
    manifest = {"records": {"PB": [{"input_id": "a", "horizon": 2}]}}
    validate_request(lock, manifest, "E3", 2970, "PB")
    with pytest.raises(ValueError, match="locked"):
        validate_request(lock, manifest, "E3", 29700, "PB")
    with pytest.raises(ValueError, match="role"):
        validate_request(lock, manifest, "E3", 2970, "CAL")


def test_inference_rejects_checkpoint_identity_or_step_drift():
    from scripts.native_long_confirmation_inference import validate_checkpoint

    saved = {"global_step": 2970, "native_long_resume": {"identity": {"population": "sha"}}}
    validate_checkpoint(saved, {"population": "sha"}, 2970)
    with pytest.raises(ValueError, match="identity"):
        validate_checkpoint(saved, {"population": "different"}, 2970)
    with pytest.raises(ValueError, match="step"):
        validate_checkpoint(saved, {"population": "sha"}, 5940)
