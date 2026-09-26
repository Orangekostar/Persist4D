import pytest
import torch

from scripts.perception_gain_v2_bundle import (
    reassemble_asset,
    replacement_state,
    restore_state,
    split_asset,
    state_digest,
)


def test_replacement_preserves_trained_weights_and_changed_buffers_exactly():
    base = {
        "weight": torch.tensor([1e20]),
        "frozen": torch.tensor([3.0]),
        "count": torch.tensor(2),
        "signed_zero": torch.tensor([0.0]),
    }
    final = {
        "weight": torch.tensor([1.0]),
        "frozen": torch.tensor([3.0]),
        "count": torch.tensor(9),
        "signed_zero": torch.tensor([-0.0]),
    }
    replacements, kinds = replacement_state(
        base, final, parameter_names={"weight", "frozen"}, trainable_names={"weight"}
    )
    assert set(replacements) == {"weight", "count", "signed_zero"}
    assert kinds["count"] == "changed_buffer"
    restored = restore_state(base, replacements)
    assert state_digest(restored) == state_digest(final)
    # Floating subtraction followed by addition would lose this trained value.
    assert not torch.equal(
        base["weight"] + (final["weight"] - base["weight"]), final["weight"]
    )
    assert torch.equal(base["count"], torch.tensor(2))


def test_unchanged_trainable_and_new_scorer_are_bundled_but_frozen_changes_fail():
    base = {"weight": torch.ones(2), "encoder": torch.zeros(1)}
    final = {**base, "model.semantic_query_scorer.weight": torch.ones(3)}
    replacements, kinds = replacement_state(
        base, final, parameter_names=set(final), trainable_names={"weight"}
    )
    assert set(replacements) == {"weight", "model.semantic_query_scorer.weight"}
    assert kinds["model.semantic_query_scorer.weight"] == "added_parameter"
    changed = {**final, "encoder": torch.ones(1)}
    with pytest.raises(ValueError, match="Frozen parameter changed"):
        replacement_state(
            base, changed, parameter_names=set(final), trainable_names={"weight"}
        )


def test_restore_rejects_shape_changes_in_a_base_tensor():
    with pytest.raises(ValueError, match="shape or dtype"):
        restore_state({"weight": torch.zeros(1)}, {"weight": torch.zeros(2)})


def test_asset_parts_reconstruct_exact_bytes_and_preserve_existing_files(tmp_path):
    source = tmp_path / "deployment.pt"
    source.write_bytes(bytes(range(53)))
    manifest = split_asset(source, maximum_bytes=16)
    assert [part["bytes"] for part in manifest["ordered_parts"]] == [16, 16, 16, 5]
    target = tmp_path / "download/reconstructed.pt"
    reassemble_asset(source.with_suffix(".pt.manifest.json"), output=target)
    assert target.read_bytes() == source.read_bytes()
    assert split_asset(source, maximum_bytes=16) == manifest
    damaged = tmp_path / manifest["ordered_parts"][0]["name"]
    damaged.write_bytes(b"x" * 16)
    with pytest.raises(ValueError, match="different content"):
        split_asset(source, maximum_bytes=16)
    assert damaged.read_bytes() == b"x" * 16
