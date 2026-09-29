"""Head-only fixed-plan training; every arm starts independently from frozen R1 features."""

import csv
import hashlib
import json
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from models.short_module_heads import Module, QualityHead, build_head, quality_inputs
from scripts.short_module_data import geometry_loss, threshold_quality_loss
from scripts.short_module_identity import source_identity
from scripts.short_module_screen import ARTIFACTS, read_json, sha256, write_json


def build_sample_plan(records: list[dict], counts: dict[str, int], *, seed: int,
                      updates: int = 1500) -> list[list[dict]]:
    pools = {1: {}, 2: {}}
    for record in records:
        if record["role"] != "TRAIN" or counts.get(record["input_id"], 0) == 0:
            continue
        pools[record["horizon"]].setdefault(record["reference_id"], set()).add(record["input_id"])
    if not all(pools.values()):
        raise ValueError("training requires nonempty real H1 and H2 records")
    generator = torch.Generator().manual_seed(seed)
    plan = []
    for _ in range(updates):
        batch = []
        for horizon in (2, 2, 2, 1):
            references = sorted(pools[horizon])
            reference = references[int(torch.randint(len(references), (1,), generator=generator))]
            inputs = sorted(pools[horizon][reference])
            identity = inputs[int(torch.randint(len(inputs), (1,), generator=generator))]
            n = counts[identity]
            candidates = (torch.randperm(n, generator=generator)[:4] if n >= 4 else
                          torch.randint(n, (4,), generator=generator)).tolist()
            batch.append({"input_id": identity, "reference_id": reference, "horizon": horizon,
                          "candidates": candidates,
                          "segment_seed": int(torch.randint(2**31 - 1, (1,), generator=generator))})
        plan.append(batch)
    return plan


def parameter_digest(head) -> str:
    digest = hashlib.sha256()
    for name, value in head.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_training_records(root: Path, index: dict) -> dict:
    """Keep only small segment-level training fields in RAM, never full masks."""
    cache = Path(index["cache"])
    records = {}
    for entry in index["entries"]:
        if entry["record"]["role"] != "TRAIN" or entry["audit"]["candidate_count"] == 0:
            continue
        for field in ("prediction", "targets"):
            if sha256(cache / entry[field]["file"]) != entry[field]["sha256"]:
                raise ValueError(f"training input changed: {entry['input_id']} {field}")
        prediction = torch.load(cache / entry["prediction"]["file"], map_location="cpu", weights_only=False)
        target = torch.load(cache / entry["targets"]["file"], map_location="cpu", weights_only=False)
        if target["quality"].get("schema") != "quality-threshold-validity-v2":
            raise ValueError("legacy labels require explicit v2 relabel before training")
        soft = prediction["parent"]["soft"]
        records[entry["input_id"]] = {
            "features": soft["segment_features"], "query": soft["query_features"],
            "logits": soft["segment_logits"], "class_probabilities": soft["class_probabilities"],
            "class_ids": soft["source_class_ids"], "scores": prediction["parent"]["scores"],
            "descriptor": prediction["descriptor"], "quality": target["quality"],
            "assignment": target["assignment"], "targets": target["segment_targets"],
            "weights": target["segment_weights"]}
    if not records:
        raise ValueError("training cache contains no eligible candidate records")
    return records


def _save_checkpoint(path: Path, head, optimizer, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save({"head": {k: v.detach().cpu() for k, v in head.state_dict().items()},
                "optimizer": optimizer.state_dict(), "metadata": metadata}, temporary)
    temporary.replace(path)


def train_arm(module: str, *, seed: int, updates: int, root: Path, device: str,
              training_records: dict | None = None, audit_replay: bool = False) -> dict:
    from trainer.perception_gain_trainer import perception_lr_multiplier

    module = Module(module)
    if module is Module.B0 or seed not in (45, 46) or not 1 <= updates <= 1500:
        raise ValueError("invalid fixed head training request")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    resolved = read_json(ARTIFACTS / "RUN_CONFIG.json")
    protocol = resolved["protocol"]
    index = read_json(root / "EXPORT_INDEX.json")
    if index["status"] != "COMPLETE":
        raise ValueError("full shared export is required before training")
    population = read_json(ARTIFACTS / "SHORT_POPULATION.json")
    counts = {e["input_id"]: e["audit"]["candidate_count"] for e in index["entries"]}
    plan = build_sample_plan(population["records"], counts, seed=seed)
    plan_sha = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    plan_path = root / "training" / f"sample_plan_seed{seed}.json"
    if plan_path.exists() and read_json(plan_path) != plan:
        raise ValueError("common training sample plan changed")
    if not plan_path.exists():
        write_json(plan_path, plan)
    records = training_records if training_records is not None else load_training_records(root, index)
    dimensions = resolved["parent_dimensions"]
    head = build_head(module, **dimensions, thresholds=tuple(resolved["official_thresholds"]), seed=seed)
    initial_digest = parameter_digest(head)
    head.to(device).train()
    optimizer = torch.optim.AdamW(head.parameters(), lr=.001, weight_decay=.0001,
                                  betas=(.9, .999), eps=1e-8)
    public = ARTIFACTS / "training" / module.value
    if seed != 45:
        public /= f"seed{seed}"
    directory = root / "training" / module.value / f"seed{seed}"
    original_directory = directory
    if audit_replay:
        directory = root / "training_log_replay" / module.value / f"seed{seed}"
        public = ARTIFACTS / "training_log_replay" / module.value / f"seed{seed}"
    public.mkdir(parents=True, exist_ok=True)
    metadata = {"module": module.value, "seed": seed, "base_sha256": protocol["parent"]["sha256"],
                "dimensions": dimensions, "thresholds": resolved["official_thresholds"],
                "sample_plan_sha256": plan_sha, "cache_identity_sha256": index["cache_identity_sha256"],
                "initial_parameter_sha256": initial_digest, "parameter_count": sum(p.numel() for p in head.parameters()),
                "source_sha256": sha256(Path(__file__)), "optimizer_updates": 0,
                "loss_source_sha256": sha256(Path(__file__).with_name("short_module_data.py")),
                "label_identity_sha256": index.get("label_identity_sha256"),
                "training_sources": source_identity([Path(__file__),
                    Path(__file__).with_name("short_module_data.py"),
                    Path(__file__).parents[1] / "models/short_module_heads.py",
                    Path(__file__).parents[1] / "trainer/perception_gain_trainer.py"]),
                "schedule_horizon": 1500, "target_updates": updates,
                "descriptor": resolved["descriptor_normalization"],
                "supported_horizons": [1, 2], "postprocess": "original retained candidates, no new filter",
                "supervised_gradient_checks": [], "parent_loaded_in_training": False}
    write_json(public / "resolved_config.json", {**metadata, "training": protocol["training"],
               "segment_sampling": "Uniform <=2048 known segments per record/stage, common subset for sampled candidate slots; same plan all arms",
               "zero_candidate_records": [key for key, count in counts.items() if count == 0]})
    start = 0
    last = directory / "last.pt"
    if last.exists():
        checkpoint = torch.load(last, map_location=device, weights_only=False)
        old = checkpoint["metadata"]
        for name in ("module", "seed", "base_sha256", "sample_plan_sha256", "cache_identity_sha256",
                     "initial_parameter_sha256", "label_identity_sha256", "training_sources"):
            if old.get(name) != metadata[name]:
                raise ValueError(f"resume identity changed: {name}")
        head.load_state_dict(checkpoint["head"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start = old["optimizer_updates"]
        metadata["supervised_gradient_checks"] = old["supervised_gradient_checks"]
        for field in ("first20_seconds", "estimated_1500_seconds"):
            if field in old:
                metadata[field] = old[field]
    else:
        _save_checkpoint(directory / "update=0000.pt", head, optimizer, metadata)
    log_path = public / "metrics.csv"
    started = time.perf_counter()
    losses, valid_counts, detail_rows = [], [], []
    for step in range(start, updates):
        lr = .001 * perception_lr_multiplier(step, total_updates=1500, warmup_updates=75,
                                              minimum_fraction=.1)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        batch_losses, q_inputs, q_targets, q_valid = [], [], [], []
        supervised = 0
        detail = {"bce": 0., "dice": 0., "regularization": 0., "quality": 0.,
                  "positive": 0, "negative": 0, "unknown": 0}
        for draw in plan[step]:
            record = records[draw["input_id"]]
            selected = torch.tensor(draw["candidates"], dtype=torch.long)
            desc = record["descriptor"]
            query = record["query"][selected].to(device)
            local_h = desc["h"][selected].to(device)
            one_hot = F.one_hot(record["class_ids"][selected].to(device), dimensions["classes"]).float()
            eligibility = "valid" if isinstance(head, QualityHead) else "geometry_valid"
            valid = record["quality"][eligibility][selected].to(device)
            matched = (record["assignment"][selected] >= 0).to(device)
            positive = ((record["quality"]["temporal"][selected].to(device) > 0)
                        if isinstance(head, QualityHead) else matched)
            detail["positive"] += int((valid & positive).sum())
            detail["negative"] += int((valid & ~positive).sum())
            detail["unknown"] += int((~valid).sum())
            if isinstance(head, QualityHead):
                x = quality_inputs(query, local_h, record["class_probabilities"][selected].to(device),
                                   one_hot, record["scores"][selected].to(device),
                                   desc["support"][selected].to(device),
                                   desc["foreground_probability"][selected].to(device))
                target_name = {Module.Q1: "concat", Module.Q2: "temporal", Module.Q3: "events"}[module]
                q_inputs.append(x)
                q_targets.append(record["quality"][target_name][selected].to(device).float())
                q_valid.append(record["quality"]["threshold_valid"][selected].to(device)
                               if module is Module.Q3 else valid)
                supervised += int(valid.sum())
            else:
                features = record["features"].to(device)
                segment_stages = desc["segment_stages"].to(device)
                delta = head(features, query, local_h, one_hot, segment_stages)
                logits = record["logits"][:, selected].to(device) + delta
                weights = record["weights"].to(device)
                generator = torch.Generator().manual_seed(draw["segment_seed"])
                samples = []
                for stage in range(draw["horizon"]):
                    eligible = torch.where((desc["segment_stages"] == stage) & (record["weights"] > 0))[0]
                    samples.append(eligible[torch.randperm(len(eligible), generator=generator)[:2048]].to(device))
                supervised += int((valid & (matched if module is not Module.M0 else True)).sum())
                components = {}
                batch_losses.append(geometry_loss(module, logits=logits, delta=delta,
                    targets=record["targets"][:, selected].to(device), weights=weights,
                    segment_stages=segment_stages, matched=matched, usable=valid, sample_indices=samples,
                    details=components))
                for key, value in components.items():
                    detail[key] += value / len(plan[step])
        if isinstance(head, QualityHead):
            x, y, valid = torch.cat(q_inputs), torch.cat(q_targets), torch.cat(q_valid)
            if valid.any():
                loss = (threshold_quality_loss(head.probabilities(x), y, valid) if module is Module.Q3
                        else F.mse_loss(head(x[valid]), y[valid]))
            else:
                loss = head(x).sum() * 0
            detail["quality"] = loss.detach().item()
        else:
            loss = torch.stack(batch_losses).mean()
        if not torch.isfinite(loss):
            raise ValueError(f"nonfinite head loss at update {step + 1}")
        check_gradient = supervised > 0 and len(metadata["supervised_gradient_checks"]) < 2
        before = {k: p.detach().clone() for k, p in head.named_parameters()} if check_gradient else None
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if check_gradient:
            changed = any(not torch.equal(before[k], p) for k, p in head.named_parameters())
            if not changed or norm.item() <= 0:
                raise ValueError("valid supervision did not update independent head")
            metadata["supervised_gradient_checks"].append({"update": step + 1, "gradient_norm": norm.item(),
                                                          "head_changed": changed, "parent_in_optimizer": False})
        metadata["optimizer_updates"] = step + 1
        losses.append(loss.item())
        valid_counts.append(supervised)
        detail_rows.append(detail)
        if step + 1 == 20:
            metadata["first20_seconds"] = time.perf_counter() - started
            metadata["estimated_1500_seconds"] = metadata["first20_seconds"] * 75
        if (step + 1) % 100 == 0 or step + 1 == updates:
            row = {"optimizer_update": step + 1, "loss": sum(losses) / len(losses), "lr": lr,
                   "valid_supervised_candidates": sum(valid_counts),
                   "elapsed_seconds": time.perf_counter() - started}
            row.update({key: sum(d[key] for d in detail_rows) / (len(detail_rows) if key in
                        ("bce", "dice", "regularization", "quality") else 1) for key in detail})
            exists = log_path.exists()
            with log_path.open("a", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                if not exists:
                    writer.writeheader()
                writer.writerow(row)
            losses, valid_counts, detail_rows = [], [], []
            _save_checkpoint(last, head, optimizer, metadata)
            print(json.dumps({"module": module.value, "seed": seed, **row}), flush=True)
        if step + 1 in (500, 1000, 1500) or step + 1 == updates:
            if audit_replay:
                original = torch.load(original_directory / f"update={step + 1:04d}.pt",
                                      map_location=device, weights_only=False)
                if not all(torch.equal(value, original["head"][key]) for key, value in head.state_dict().items()):
                    raise ValueError("logging replay changed trained parameters")
            _save_checkpoint(directory / f"update={step + 1:04d}.pt", head, optimizer, metadata)
    checkpoints = [{"step": int(path.stem.split("=")[1]), "path": str(path),
                    "sha256": sha256(path), "bytes": path.stat().st_size}
                   for path in sorted(directory.glob("update=*.pt"))]
    manifest = {**metadata, "optimizer_updates": max(start, updates), "status": "COMPLETE",
                "checkpoints": checkpoints, "training_seconds": time.perf_counter() - started}
    write_json(public / "checkpoint_manifest.json", manifest)
    if audit_replay:
        import shutil

        canonical = ARTIFACTS / "training" / module.value
        if seed != 45:
            canonical /= f"seed{seed}"
        archived = public / "original_metrics.csv"
        if not archived.exists():
            shutil.copyfile(canonical / "metrics.csv", archived)
        shutil.copyfile(log_path, canonical / "metrics.csv")
        write_json(canonical / "LOG_REPLAY.json", {
            "status": "EXACT_PARAMETER_REPLAY", "checked_steps": [500, 1000, 1500],
            "source_sha256": metadata["source_sha256"], "loss_source_sha256": metadata["loss_source_sha256"],
            "checkpoint_parameters_unchanged": True,
            "elapsed_seconds_in_metrics": "Replay measured time; original measured time preserved in original_metrics.csv",
            "replay_manifest": str((public / "checkpoint_manifest.json").relative_to(ARTIFACTS))})
    return manifest
