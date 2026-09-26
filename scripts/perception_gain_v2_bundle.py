"""Exact replacement-tensor deployment bundles; base weights remain external."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from scripts.perception_gain_v2 import file_hash, read_json, write_json
from scripts.perception_gain_v2_config import R1_SHA256

SCHEMA = "perception-gain-replacement-bundle-v2"


def _bytes(tensor):
    import torch

    return (
        tensor.detach()
        .cpu()
        .contiguous()
        .reshape(-1)
        .view(torch.uint8)
        .numpy()
        .tobytes()
    )


def state_digest(state: dict) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(
            json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode()
        )
        digest.update(_bytes(tensor))
    return digest.hexdigest()


def replacement_state(
    base: dict, final: dict, *, parameter_names: set, trainable_names: set
) -> tuple[dict, dict]:
    import torch

    if not set(base).issubset(final) or not trainable_names.issubset(parameter_names):
        raise ValueError(
            "Deployment state removes a base tensor or mislabels a parameter"
        )
    replacements, kinds = {}, {}
    for name, tensor in final.items():
        if not isinstance(tensor, torch.Tensor):
            # Preserve the malformed-deployment validation exception contract.
            raise ValueError("Deployment state contains a non-tensor")  # noqa: TRY004
        added = name not in base
        if not added and (
            base[name].shape != tensor.shape or base[name].dtype != tensor.dtype
        ):
            raise ValueError(f"Base tensor shape or dtype changed: {name}")
        changed = added or _bytes(base[name]) != _bytes(tensor)
        if (
            name in parameter_names
            and changed
            and not added
            and name not in trainable_names
        ):
            raise ValueError(f"Frozen parameter changed: {name}")
        if changed or name in trainable_names:
            if tensor.is_floating_point() and not torch.isfinite(tensor).all():
                raise ValueError(f"Deployment tensor is non-finite: {name}")
            replacements[name] = tensor.detach().cpu().contiguous().clone()
            kinds[name] = (
                "added_parameter"
                if added and name in parameter_names
                else (
                    "trainable_parameter"
                    if name in trainable_names
                    else "added_buffer" if added else "changed_buffer"
                )
            )
    return replacements, kinds


def restore_state(base: dict, replacements: dict) -> dict:
    state = dict(base)
    for name, tensor in replacements.items():
        if name in base and (
            base[name].shape != tensor.shape or base[name].dtype != tensor.dtype
        ):
            raise ValueError(f"Replacement shape or dtype differs: {name}")
        state[name] = tensor
    return state


def load_payload(bundle: Path, *, base_checkpoint: Path) -> tuple[dict, dict]:
    import torch

    payload = torch.load(bundle, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != SCHEMA or payload.get("operation") != "REPLACE":
        raise ValueError("Unsupported deployment bundle")
    if file_hash(base_checkpoint) != payload["base_weight_sha256"]:
        raise ValueError("Deployment bundle requires its exact original R1 base")
    base = torch.load(base_checkpoint, map_location="cpu", weights_only=False)[
        "state_dict"
    ]
    state = restore_state(base, payload["model_replacements"])
    if state_digest(state) != payload["restored_state_sha256"]:
        raise ValueError("Reconstructed model tensors differ from the bundled method")
    return payload, state


def instantiate_bundle(
    bundle: Path,
    *,
    base_checkpoint: Path,
    pretrained: Path,
    runtime_root: Path,
    device="cpu",
) -> tuple[object, object | None, dict]:
    """Load a complete parent and optional head, using replacement rather than addition."""
    from models.perception_gain import CausalMaskRefiner
    from scripts.train_perception_gain import compose_variant_config
    from trainer.perception_gain_trainer import PerceptionGainTrainer

    payload, state = load_payload(bundle, base_checkpoint=base_checkpoint)
    method = payload["method"]
    cfg = compose_variant_config(
        method["architecture_variant"],
        pretrained=pretrained,
        run_dir=runtime_root,
        recipe_config=method["recipe"],
    )
    cfg.model.return_query_features = True
    system = PerceptionGainTrainer(cfg)
    system.load_state_dict(state, strict=True)
    system.to(device).eval().requires_grad_(False)
    head = None
    if payload["refiner"] is not None:
        stored = payload["refiner"]
        if (
            stored["binding"] != method["refiner"]["binding"]
            or stored["input_mode"] != method["refiner"]["input_mode"]
        ):
            raise ValueError("Deployment head mode or parent binding differs")
        head = CausalMaskRefiner(input_mode=stored["input_mode"])
        head.load_state_dict(stored["state_dict"], strict=True)
        if state_digest(head.state_dict()) != stored["state_sha256"]:
            raise ValueError("Deployment head tensors changed")
        head.to(device).eval().requires_grad_(False)
    return system, head, payload


def build_method_bundle(method: dict, *, external_root: Path, artifacts: Path) -> dict:
    import torch

    from scripts.perception_gain_evaluation import _load_evaluation_weights
    from scripts.perception_gain_v2_confirmation import method_inputs
    from scripts.perception_gain_v2_lock import external_path
    from scripts.perception_refiner_evaluation import _load_refiner
    from scripts.train_perception_gain import compose_variant_config
    from scripts.train_perception_refiner import _atomic_torch_save
    from trainer.perception_gain_trainer import PerceptionGainTrainer

    inputs = method_inputs(method, external_root=external_root, artifacts=artifacts)
    assets = read_json(external_root / "assets.local.json")
    base_path = Path(assets["r1_checkpoint"])
    if file_hash(base_path) != R1_SHA256:
        raise ValueError("Bundle base is not the authorized original R1")
    cfg = compose_variant_config(
        method["architecture_variant"],
        pretrained=Path(assets["concerto_pretrained"]),
        run_dir=external_root / "publication/bundle-runtime",
        recipe_config=method["recipe"],
    )
    cfg.model.return_query_features = True
    system = PerceptionGainTrainer(cfg)
    parameter_names = {name for name, _ in system.named_parameters()}
    trainable_names = {
        name
        for name, value in system.named_parameters()
        if value.requires_grad and method["parent_optimizer_update"] > 0
    }
    _, observed, _, _, _ = _load_evaluation_weights(
        system=system,
        variant=method["architecture_variant"],
        optimizer_update=method["parent_optimizer_update"],
        r1_checkpoint=base_path,
        checkpoint=inputs["checkpoint"],
        scorer_checkpoint=inputs["scorer_checkpoint"],
    )
    if observed != method["parent_weight_sha256"]:
        raise ValueError("Bundle source is not the locked parent")
    base = torch.load(base_path, map_location="cpu", weights_only=False)["state_dict"]
    final = system.state_dict()
    replacements, kinds = replacement_state(
        base, final, parameter_names=parameter_names, trainable_names=trainable_names
    )
    restored = restore_state(base, replacements)
    original_digest = state_digest(final)
    model_tensor_count = len(final)
    if state_digest(restored) != original_digest:
        raise ValueError(
            "Replacement bundle does not reconstruct the exact model state"
        )
    head_payload = None
    if descriptor := method.get("refiner"):
        head, identity = _load_refiner(
            external_path(descriptor["checkpoint"], external_root),
            optimizer_update=descriptor["optimizer_update"],
            device="cpu",
            expected_binding=descriptor["binding"],
        )
        if identity["sha256"] != descriptor["checkpoint_sha256"]:
            raise ValueError("Deployment head differs from the final lock")
        head_payload = {
            "input_mode": descriptor["input_mode"],
            "binding": descriptor["binding"],
            "optimizer_update": descriptor["optimizer_update"],
            "state_dict": {
                name: tensor.detach().cpu()
                for name, tensor in head.state_dict().items()
            },
            "state_sha256": state_digest(head.state_dict()),
        }
    payload = {
        "schema_version": SCHEMA,
        "operation": "REPLACE",
        "method": method,
        "base_weight_sha256": R1_SHA256,
        "base_acquisition": "Use the original ReScene R1 checkpoint identified by this SHA; base weights and source data are not redistributed.",
        "model_replacements": replacements,
        "replacement_kinds": kinds,
        "restored_state_sha256": original_digest,
        "refiner": head_payload,
        "association_config": method.get("association_config"),
        "publisher": method["publisher"],
        "eval_seed": method["eval_seed"],
        "scorer_sha256": method.get("scorer_sha256"),
        "source_module_sha256": file_hash(Path(__file__)),
    }
    del restored, base, final, system
    safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", method["method_id"])
    path = external_root / f"publication/bundles/{safe_name}.pt"
    if path.exists():
        previous, previous_state = load_payload(path, base_checkpoint=base_path)
        if (
            previous["method"] != method
            or state_digest(previous_state) != original_digest
        ):
            raise ValueError("Existing deployment bundle belongs to another method")
        del previous, previous_state
    else:
        _atomic_torch_save(path, payload)
    # A separate load verifies file serialization, not just the in-memory dict.
    loaded, loaded_state = load_payload(path, base_checkpoint=base_path)
    if (
        loaded["refiner"] is not None
        and state_digest(loaded["refiner"]["state_dict"])
        != head_payload["state_sha256"]
    ):
        raise ValueError("Serialized deployment head differs")
    manifest = {
        "schema_version": SCHEMA,
        "status": "TENSORS_VERIFIED",
        "method_id": method["method_id"],
        "inference_identity": method["inference_identity"],
        "logical_reference": "external:" + str(path.relative_to(external_root)),
        "bytes": path.stat().st_size,
        "sha256": file_hash(path),
        "base_weight_sha256": R1_SHA256,
        "restored_state_sha256": state_digest(loaded_state),
        "replaced_parameter_count": sum(
            replacements[name].numel()
            for name in replacements
            if name in parameter_names
        ),
        "replacement_kinds": kinds,
        "model_tensor_count": model_tensor_count,
        "input_mode": descriptor["input_mode"] if descriptor else None,
        "live_panel_reload": "NOT_RUN",
        "method": method,
    }
    manifest_path = artifacts / f"publication/bundles/{safe_name}.json"
    if manifest_path.exists():
        previous = read_json(manifest_path)
        if (
            previous.get("sha256") == manifest["sha256"]
            and previous.get("method") == method
        ):
            for key in ("live_panel_reload", "live_panel_evidence"):
                if key in previous:
                    manifest[key] = previous[key]
    write_json(manifest_path, manifest)
    return manifest


def split_asset(path: Path, *, maximum_bytes: int = 1024**3) -> dict:
    """Only oversize assets are split; ordered hashes make reassembly verifiable."""
    if maximum_bytes <= 0:
        raise ValueError("Asset byte limit must be positive")
    whole_sha = file_hash(path)
    if path.stat().st_size <= maximum_bytes:
        parts = [{"name": path.name, "bytes": path.stat().st_size, "sha256": whole_sha}]
    else:
        parts = []
        with path.open("rb") as source:
            remaining = path.stat().st_size
            while remaining:
                part_path = path.with_name(f"{path.name}.part-{len(parts):03d}")
                size = min(maximum_bytes, remaining)
                existing = part_path.exists()
                if existing and part_path.stat().st_size != size:
                    raise ValueError("Existing asset part has a different size")
                with part_path.open("rb" if existing else "xb") as output:
                    left = size
                    while left:
                        block = source.read(min(left, 8 * 1024**2))
                        if not block:
                            raise ValueError(
                                "Deployment asset ended before its declared size"
                            )
                        if existing:
                            if output.read(len(block)) != block:
                                raise ValueError(
                                    "Existing asset part has different content"
                                )
                        else:
                            output.write(block)
                        left -= len(block)
                parts.append(
                    {
                        "name": part_path.name,
                        "bytes": size,
                        "sha256": file_hash(part_path),
                    }
                )
                remaining -= size
    manifest = {
        "filename": path.name,
        "bytes": path.stat().st_size,
        "sha256": whole_sha,
        "ordered_parts": parts,
    }
    write_json(path.with_suffix(path.suffix + ".manifest.json"), manifest)
    return manifest


def reassemble_asset(manifest_path: Path, *, output: Path) -> None:
    manifest = read_json(manifest_path)
    parts = []
    for part in manifest["ordered_parts"]:
        path = manifest_path.parent / part["name"]
        if (
            path.parent.resolve() != manifest_path.parent.resolve()
            or path.stat().st_size != part["bytes"]
            or file_hash(path) != part["sha256"]
        ):
            raise ValueError("Deployment asset part identity differs")
        parts.append(path)
    if output.exists():
        if (
            output.stat().st_size != manifest["bytes"]
            or file_hash(output) != manifest["sha256"]
        ):
            raise ValueError("Refusing to overwrite a different reconstructed asset")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as target:
        for path in parts:
            with path.open("rb") as source:
                while block := source.read(8 * 1024**2):
                    target.write(block)
    if file_hash(output) != manifest["sha256"]:
        raise ValueError("Reconstructed asset digest differs")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify-tensors", "reassemble"))
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.command == "reassemble":
        if args.output is None:
            parser.error("reassemble requires --output")
        reassemble_asset(args.bundle, output=args.output)
        print(json.dumps({"status": "VERIFIED", "sha256": file_hash(args.output)}))
    else:
        if args.base_checkpoint is None:
            parser.error("verify-tensors requires --base-checkpoint")
        payload, _ = load_payload(args.bundle, base_checkpoint=args.base_checkpoint)
        print(
            json.dumps(
                {
                    "status": "TENSORS_VERIFIED",
                    "method_id": payload["method"]["method_id"],
                    "live_panel_reload": "NOT_RUN",
                }
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
