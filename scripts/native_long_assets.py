"""Exact nonencoder task/buffer export with a separately bound Concerto encoder."""

from pathlib import Path

import torch
from omegaconf import OmegaConf

from datasets.native_long_dataset import isolated_rng
from scripts.native_long_campaign import ARTIFACTS, code_identity
from scripts.short_module_screen import read_json, sha256, write_json
from trainer.native_long_trainer import NativeLongTrainer

ENCODER_PREFIXES = ("model.backbone.model.embedding.", "model.backbone.model.enc.")


def export_task_bundle(root, system, *, step, destination):
    source = read_json(root / "SOURCE_AND_INITIALIZATION.json")
    seed = str(system.config.general.seed)
    initial = torch.load(source["initialization"][seed]["path"], map_location="cpu", weights_only=True)["state_dict"]
    parameters = dict(system.named_parameters())
    omitted = {k for k in parameters if k.startswith(ENCODER_PREFIXES)}
    state = {k: v.detach().cpu() for k, v in system.state_dict().items()}
    if any(not torch.equal(state[k], initial[k]) for k in omitted):
        raise ValueError("frozen pretrained encoder parameters changed")
    buffers = [k for k, _ in system.named_buffers() if k.startswith(ENCODER_PREFIXES)]
    payload = {"schema": 1, "state_dict": {k: v for k, v in state.items() if k not in omitted},
               "omitted_encoder_parameter_names": sorted(omitted), "included_encoder_buffers": buffers,
               "config": OmegaConf.to_container(system.config, resolve=True),
               "metadata": {"seed": int(seed), "step": step, "arm": system.config.native_long.arm,
                            "role": "TRAINED_TASK" if step else "INITIALIZATION_ONLY",
                            "common_initialization_sha256": source["initialization"][seed]["sha256"],
                            "pretrained_encoder_sha256": source["weights"]["concerto_pretrained"]["sha256"],
                            "population_sha256": sha256(root / "POPULATION.json"), "code": code_identity()}}
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)
    return {"path": str(destination), "sha256": sha256(destination), "bytes": destination.stat().st_size,
            "metadata": payload["metadata"], "encoder_buffers": len(buffers),
            "task_state_entries": len(payload["state_dict"])}


def load_task_bundle(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["schema"] != 1 or payload["metadata"]["code"] != code_identity():
        raise ValueError("task bundle schema/numeric code identity differs")
    config = OmegaConf.create(payload["config"])
    if sha256(Path(config.backbone.name)) != payload["metadata"]["pretrained_encoder_sha256"]:
        raise ValueError("separate encoder bytes differ from bound pretrained asset")
    with isolated_rng(config.general.seed):
        system = NativeLongTrainer(config)
    current = system.state_dict()
    omitted = set(payload["omitted_encoder_parameter_names"])
    actual_omitted = {k for k, _ in system.named_parameters() if k.startswith(ENCODER_PREFIXES)}
    if omitted != actual_omitted or set(payload["state_dict"]) != set(current) - omitted:
        raise ValueError("bundle omitted a task weight or mutable buffer")
    current.update(payload["state_dict"])
    system.load_state_dict(current, strict=True)
    return system


def audit_initialization_reload(root, config):
    from scripts.native_long_runtime import system_from_common

    original, _ = system_from_common(root, config)
    # Preserve any registered encoder buffers, including updated running statistics.
    changed = []
    for name, buffer in original.named_buffers():
        if name.startswith(ENCODER_PREFIXES) and name.endswith(("running_mean", "num_batches_tracked")):
            buffer.add_(1)
            changed.append(name)
    row = export_task_bundle(root, original, step=0,
                            destination=root / "initialization/task_reload_initialization.pt")
    restored = load_task_bundle(row["path"])
    if any(not torch.equal(v, restored.state_dict()[k]) for k, v in original.state_dict().items()):
        raise ValueError("actual task/encoder-buffer reconstruction changed a tensor")
    row.update(status="EXACT_TENSOR_RELOAD", deliberately_changed_buffers=changed,
               scope="initialized state tensor roundtrip; actual encoder has no registered buffers in this configuration; no trained deployment claim")
    write_json(ARTIFACTS / "initialization/TASK_BUNDLE_RELOAD_AUDIT.json", row)
