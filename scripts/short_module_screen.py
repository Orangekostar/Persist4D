"""Fixed ReScene short-module experiment runner; artifacts describe actual work."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

PROJECT = Path(__file__).resolve().parents[1]
ARTIFACTS = Path(os.environ.get("SHORT_MODULE_ARTIFACTS", PROJECT / "artifacts/short_module_screen_v1"))
PACKAGE = PROJECT / "docs/0928/ReScene_Round1_Single_Module"
V2_ROOT = Path("/home/ww/persist4d_runs/perception_gain_v2")
V2_LIVE = Path("/home/ww/paper5/.worktrees/persist4d-perception-gain-v2/artifacts/perception_gain_v2")


def read_config(path: Path) -> dict:
    value = yaml.safe_load(path.read_text())
    frozen = yaml.safe_load((PACKAGE / "short_module_screen_v1.protocol.yaml").read_text())
    if value != frozen:
        raise ValueError("configuration differs from the frozen protocol")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text())


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def append_event(path: Path, event: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def native_config(root: Path, assets: dict):
    from scripts.train_perception_gain import compose_variant_config

    config = compose_variant_config("C0", pretrained=Path(assets["concerto_pretrained"]),
                                    run_dir=root / "parent")
    config.model.return_query_features = True
    # Necessary for true single scans without evaluable instances. Does not add,
    # remove or substitute points; the parent executes the same native forward.
    from omegaconf import open_dict
    with open_dict(config.data.validation_collation):
        config.data.validation_collation.preserve_empty_targets = True
    return config


def build_datasets(config, assets: dict):
    from copy import deepcopy

    from omegaconf import open_dict

    from scripts.perception_gain_evaluation import build_role_base_dataset

    calibration = build_role_base_dataset(config=config, data_root=Path(assets["data_root"]),
                                           role="CAL", horizon=5)
    train_config = deepcopy(config)
    # TRAIN eligibility must be metadata-only, not an inspection of GT difficulty.
    with open_dict(train_config.data.train_dataset):
        train_config.data.train_dataset.exclude_unsupervised_sequences = False
        train_config.data.train_dataset.known_empty_scan_policy = "allow_actual"
    training = build_role_base_dataset(config=train_config, data_root=Path(assets["data_root"]),
                                       role="CAL", horizon=2)
    return {2: training, 5: calibration}


def prepare(config: dict, root: Path) -> dict:
    from datasets.task_memory_episode import _scan_scene
    from scripts.perception_gain_evaluation import (
        DATA_CONTRACT,
        select_live_population_units,
    )
    from scripts.preflight_task_memory_episode import load_reference_by_scene
    from scripts.run_task_memory_policy_baseline import (
        DEVELOPMENT_POPULATION_ID,
        _episode_specs,
        build_baseline_population,
    )
    from scripts.short_module_data import select_population

    assets = read_json(V2_ROOT / "assets.local.json")
    roles_path = PROJECT / config["data"]["inherit_reference_roles_from"]
    roles = read_json(roles_path)["roles"]
    parent = Path(assets["r1_checkpoint"])
    if parent.stat().st_size != config["parent"]["bytes"] or sha256(parent) != config["parent"]["sha256"]:
        raise ValueError("parent checkpoint differs from fixed R1")
    if sha256(Path(assets["concerto_pretrained"])) != "845ec7dec97a5fabff8fadb5d9858ac6734347b612d1a4b574213419c139de07":
        raise ValueError("Concerto checkpoint identity differs")
    ledger_path = V2_LIVE / "budget/LEDGER.jsonl"
    prior_events = [json.loads(line) for line in ledger_path.read_text().splitlines() if line.strip()]
    if len({e["event_id"] for e in prior_events}) != len(prior_events):
        raise ValueError("prior ledger contains duplicate event identities")
    prior = sum(e["gpu_hours"] for e in prior_events)
    if abs(prior - read_json(V2_ROOT / "publication/FINAL_DELIVERY_AUDIT.json")["cost"]["settled_gpu_hours"]) > 1e-8:
        raise ValueError("prior final receipt and live ledger disagree")
    local_audit = read_json(V2_LIVE / "foundation/LOCAL_POPULATION_AUDIT.json")
    excluded = set(local_audit["validation_reference_ids"])
    for role in ("CAL", "SEL", "PB", "LOCAL-T2", "ADDITIONAL"):
        excluded.update(roles[role])
    if set(roles["TRAIN"]) & excluded:
        raise ValueError("inherited TRAIN overlaps an exposed physical reference")
    native = native_config(root, assets)
    datasets = build_datasets(native, assets)
    _base, masters, _ = build_baseline_population(
        datasets[5], data_contract=read_json(DATA_CONTRACT),
        metadata_path=Path(assets["rio_metadata"]), population_id=DEVELOPMENT_POPULATION_ID,
        protocol_b_manifest_path=PROJECT / "artifacts/P6A/protocol_b_manifest.json")
    specs = _episode_specs(masters)
    heldout = {}
    for role, expected in (("CAL", 23), ("SEL", 24)):
        units = select_live_population_units(role=role, role_references=roles[role],
                                            episode_specs=specs, expected_logical_units=expected)
        heldout[role] = []
        for unit in units:
            spec = specs[unit.spec_index]
            heldout[role].append({"reference_id": unit.reference_id,
                                     "scan_ids": list(spec.scan_ids[:2]),
                                     "scan_indices": list(spec.scan_indices[:2]),
                                     "context_index": spec.context_index, "dataset_horizon": 5,
                                     "logical_unit_id": unit.logical_unit_id,
                                     "master_sequence_id": unit.sequence_id})
    by_scene = load_reference_by_scene(Path(assets["rio_metadata"]))
    train_records = []
    train = datasets[2]
    for index, name in enumerate(train.sequence_names):
        scans = name.split("-")
        references = {by_scene[_scan_scene(scan)] for scan in scans}
        if len(references) != 1:
            raise ValueError("training pair crosses references")
        reference = next(iter(references))
        if reference not in roles["TRAIN"]:
            continue
        indices = [int(i) for i in train.sequence_indices[index]]
        if len(scans) != 2 or any(not Path(train.data[i]["filepath"].replace("../../", "")).is_file()
                                   for i in indices):
            continue
        train_records.append({"reference_id": reference, "scan_ids": scans, "scan_indices": indices,
                                  "context_index": index, "dataset_horizon": 2,
                                  "logical_unit_id": f"TRAIN:{index}", "master_sequence_id": name})
    population = select_population(train_records, heldout["CAL"], heldout["SEL"],
                                   excluded_references=excluded)
    write_json(ARTIFACTS / "SHORT_POPULATION.json", population)
    sources = {"roles": roles_path, "v2_inputs": PROJECT / "artifacts/perception_gain_v2/INPUT_MANIFEST.json",
               "staging": PROJECT / "artifacts/perception_gain_v2/data/STAGING_MANIFEST.json",
               "prior_ledger": ledger_path, "local_population_audit": V2_LIVE / "foundation/LOCAL_POPULATION_AUDIT.json",
               "metadata": Path(assets["rio_metadata"])}
    manifest = {"sources": {k: {"path": str(p), "sha256": sha256(p)} for k, p in sources.items()},
                "assets": assets, "population_sha256": sha256(ARTIFACTS / "SHORT_POPULATION.json"),
                "raw_staging": {"shared_existing_data": True, "new_bytes": 0,
                                "limitation": "Inherited hardlink staging manifest has null per-file SHA; not byte-immutable"}}
    write_json(ARTIFACTS / "INPUT_MANIFEST.json", manifest)
    resolved = {"protocol": config, "status": "PREPARED",
                "prior_gpu_hours": prior, "round_cap_gpu_hours": min(48., 192. - prior),
                "numeric_environment": {"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
                                        "OPENBLAS_NUM_THREADS": "2", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                                        "precision": "float32", "TF32": False, "eval_seed": 45},
                "parent_dimensions": None,
                "parent_dimensions_status": "PENDING_REAL_NATIVE_EXPORT",
                "data_mode": {str(k): d.mode for k, d in datasets.items()},
                "training_eligibility": "Metadata-only native directed pairs; no supervised-content filter",
                "descriptor_normalization": "Q: functional LN(q), LN(mean_h); raw absolute h difference; no concatenation LN. M: parent-normalized q and raw h.",
                "runtime_data_symlink": str(Path(assets["data_root"]).resolve())}
    write_json(ARTIFACTS / "RUN_CONFIG.json", resolved)
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "assets.local.json", assets)
    budget_path = ARTIFACTS / "BUDGET_LEDGER.jsonl"
    if not budget_path.exists():
        append_event(budget_path, {"event_id": "inherited-v2-final", "gpu_hours": prior,
                                  "scope": "PRIOR", "source_sha256": sha256(ledger_path),
                                  "prior_event_count": len(prior_events)})
    return {"status": "PREPARED", "counts": population["counts"], "prior_gpu_hours": prior}


def export(config: dict, root: Path, *, device: str, limit: int | None) -> dict:
    export_started = time.monotonic()
    import importlib.metadata
    import inspect
    import shutil

    import hydra
    import stmetrics.instances.evaluator
    import stmetrics.instances.matcher
    import torch
    from omegaconf import OmegaConf

    from scripts.short_module_data import (
        geometry_assignment,
        low_segment_targets,
        quality_labels,
    )
    from scripts.short_module_native import NativeSession, unpack_prediction
    from scripts.system_comparison_inference import pack_bool_matrix

    assets = read_json(root / "assets.local.json")
    resolved = read_json(ARTIFACTS / "RUN_CONFIG.json")
    numeric = resolved["numeric_environment"]
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "CUBLAS_WORKSPACE_CONFIG"):
        if os.environ.get(name) != numeric[name]:
            raise ValueError(f"native runtime requires {name}={numeric[name]}")
    population = read_json(ARTIFACTS / "SHORT_POPULATION.json")
    native = native_config(root, assets)
    datasets = build_datasets(native, assets)
    source_files = [PROJECT / p for p in (
        "scripts/short_module_screen.py", "scripts/short_module_native.py", "models/short_module_heads.py", "models/rescene.py",
        "models/perception_gain.py", "scripts/rescene_task_postprocess.py", "trainer/trainer.py",
        "datasets/semseg.py", "datasets/pointcept_utils.py", "datasets/auto_collate.py",
        "scripts/system_comparison_inference.py", "scripts/short_module_data.py")]
    source_files += [Path(inspect.getfile(m)) for m in
                     (stmetrics.instances.matcher, stmetrics.instances.evaluator)]
    source_files.append(Path(assets["metric_dataset_spec"]))
    source_files += [Path(inspect.getfile(hydra.utils.get_class(native.backbone._target_))),
                     PROJECT / "scripts/evaluate_persist4d.py",
                     PROJECT / "scripts/evaluate_persist4d_p6a.py"]
    identity = {"parent_sha256": config["parent"]["sha256"],
                "sources": {str(p): sha256(p) for p in source_files},
                "input_manifest_sha256": sha256(ARTIFACTS / "INPUT_MANIFEST.json"),
                "native_config": OmegaConf.to_container(native, resolve=True),
                "numeric_environment": numeric,
                "versions": {name: importlib.metadata.version(name) for name in ("torch", "numpy", "scipy")},
                "postprocess": "original topk/filter once; native low>0/inverse/full-partition majority",
                "cache_representation": "FP32 segment tensors and packed native full bool; prediction/GT separate"}
    identity_sha = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache = root / "cache" / identity_sha
    cache.mkdir(parents=True, exist_ok=True)
    write_json(cache / "IDENTITY.json", identity)
    session = NativeSession(config=native, datasets=datasets, assets=assets, device=device)
    # CAL T2 then CAL T1 guarantees the smoke covers a real pair and single scan.
    ordered = sorted(population["records"], key=lambda r: (
        ("CAL", "SEL", "TRAIN").index(r["role"]), -r["horizon"], r["logical_unit_id"]))
    if limit is not None:
        # Smoke explicitly alternates first CAL T2 and first CAL T1. No claim of
        # complete coverage is derived from this subset.
        ordered = ([next(r for r in ordered if r["role"] == "CAL" and r["horizon"] == h)
                    for h in (2, 1)] + ordered)[:limit]
    entries = []
    seen = set()
    for record in ordered:
        key = record["input_id"]
        if key in seen:
            continue
        seen.add(key)
        entry_path = cache / f"{key}.json"
        if entry_path.exists():
            entry = read_json(entry_path)
            if all(sha256(cache / entry[k]["file"]) == entry[k]["sha256"] for k in ("prediction", "targets")):
                entries.append(entry)
                continue
            raise ValueError("existing cache shard checksum differs")
        consumed = sum(json.loads(line)["gpu_hours"] for line in
                       (ARTIFACTS / "BUDGET_LEDGER.jsonl").read_text().splitlines())
        if consumed + (time.monotonic() - export_started) / 3600 >= min(192., resolved["prior_gpu_hours"] + 48.) - .05:
            raise ValueError("GPU budget exhausted")
        if shutil.disk_usage(root).free < 2 * 1024**3:
            raise ValueError("less than 2 GiB free disk headroom")
        prediction, targets, audit = session.produce(record, compare_native=(not entries))
        parent = unpack_prediction(prediction["parent"])
        labels = quality_labels(parent.prediction(), targets["target"],
                                dataset_spec=Path(assets["metric_dataset_spec"]))
        assignment = geometry_assignment(parent.prediction(), targets["target"], labels,
                                         candidate_keys=prediction["candidate_keys"])
        s, n = parent.soft_evidence.segment_logits.shape
        segment_targets = torch.zeros(s, n)
        weights = None
        for candidate, entity in enumerate(assignment.tolist()):
            mask = (targets["target"]["masks"][targets["target"]["ids"] == entity].any(0)
                    if entity >= 0 else torch.zeros_like(targets["semantic_labels"], dtype=torch.bool))
            value, weights = low_segment_targets(
                low_point2segment=parent.soft_evidence.low_point2segment,
                voxel_inverse=parent.soft_evidence.voxel_inverse, gt_mask=mask,
                semantic_labels=targets["semantic_labels"], segment_count=s)
            segment_targets[:, candidate] = value
        if weights is None:
            _, weights = low_segment_targets(
                low_point2segment=parent.soft_evidence.low_point2segment,
                voxel_inverse=parent.soft_evidence.voxel_inverse,
                gt_mask=torch.zeros_like(targets["semantic_labels"], dtype=torch.bool),
                semantic_labels=targets["semantic_labels"], segment_count=s)
        targets.update(quality=labels, assignment=assignment, segment_targets=segment_targets,
                       segment_weights=weights)
        targets["target"]["masks"] = pack_bool_matrix(targets["target"]["masks"])
        partition = hashlib.sha256()
        for value in (parent.soft_evidence.low_point2segment, parent.soft_evidence.voxel_inverse,
                      parent.soft_evidence.full_point2segment):
            partition.update(value.numpy().tobytes())
        entry = {"input_id": key, "record": record, "audit": audit,
                 "partition_sha256": partition.hexdigest(), "cache_identity_sha256": identity_sha}
        for name, value in (("prediction", prediction), ("targets", targets)):
            destination = cache / f"{key}.{name}.pt"
            temporary = destination.with_suffix(".tmp")
            torch.save(value, temporary)
            temporary.replace(destination)
            entry[name] = {"file": destination.name, "sha256": sha256(destination),
                           "bytes": destination.stat().st_size}
        total_cache_bytes = sum(p.stat().st_size for p in (root / "cache").rglob("*.pt"))
        if total_cache_bytes > config["data"]["new_feature_cache_limit_gib"] * 1024**3:
            raise ValueError("new cache exceeds 32 GiB; shard cleanup required before continuing")
        write_json(entry_path, entry)
        entries.append(entry)
        dimensions = prediction["dimensions"]
        if resolved["parent_dimensions"] is not None and resolved["parent_dimensions"] != dimensions:
            raise ValueError("actual parent dimensions changed across inputs")
        resolved["parent_dimensions"] = dimensions
        resolved["parent_dimensions_status"] = "BOUND_REAL_NATIVE_TENSORS"
        resolved["official_thresholds"] = labels["thresholds"]
        write_json(ARTIFACTS / "RUN_CONFIG.json", resolved)
        append_event(ARTIFACTS / "EXECUTION_LOG.jsonl", {"stage": "export-input", **audit,
                     "cache_identity_sha256": identity_sha, "cache_bytes": total_cache_bytes})
        print(json.dumps({"exported": len(entries), "role": record["role"], "H": record["horizon"],
                          "seconds": audit["elapsed_seconds"], "candidates": audit["candidate_count"]}), flush=True)
    expected = len({r["input_id"] for r in population["records"]})
    result = {"status": "COMPLETE" if len(entries) == expected else "PARTIAL",
              "cache": str(cache), "cache_identity_sha256": identity_sha,
              "completed_unique_inputs": len(entries), "expected_unique_inputs": expected,
              "entries": entries}
    write_json(root / "EXPORT_INDEX.json", result)
    write_json(ARTIFACTS / "EXPORT_STATUS.json", {k: v for k, v in result.items() if k != "entries"})
    return {k: v for k, v in result.items() if k != "entries"}


def run_pipeline(args) -> dict:
    """Fixed stage order, child-owned cost accounting, recoverable partial delivery."""
    environment = {**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
                   "OPENBLAS_NUM_THREADS": "2", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
    statuses = {}
    stages = ("prepare", "export", "baseline", "train", "evaluate-cal", "screen",
              "replicate", "diagnostics", "profile", "report", "publish")
    if args.resume and (args.root / "EXPORT_INDEX.json").exists():
        from scripts.short_module_identity import (
            assert_export_current,
            assert_sources_current,
        )
        index = assert_export_current(args.root)
        stored = read_json(Path(index["cache"]) / "IDENTITY.json")
        producer_project = next(Path(p).parents[1] for p in stored["sources"]
                                if p.endswith("/scripts/short_module_native.py"))
        # Imported manifests contain absolute paths into their producing
        # checkout. Also compare THIS checkout, not just the preserved producer.
        current_sources = {str(PROJECT / Path(p).relative_to(producer_project))
                           if Path(p).is_relative_to(producer_project) else p: digest
                           for p, digest in stored["sources"].items()}
        assert_sources_current({"sources": current_sources})
        if stored["input_manifest_sha256"] != sha256(ARTIFACTS / "INPUT_MANIFEST.json"):
            raise ValueError("resume input manifest changed")
    # Publication recovery is an explicit `publish` operation, never a reason
    # for `run --resume` to silently bypass computation freshness checks.
    for stage in stages:
        if args.resume:
            if stage == "prepare" and (ARTIFACTS / "RUN_CONFIG.json").exists():
                statuses[stage] = "REUSED"
                continue
            if (stage == "export" and (args.root / "EXPORT_INDEX.json").exists()
                    and read_json(args.root / "EXPORT_INDEX.json")["status"] == "COMPLETE"):
                statuses[stage] = "REUSED"
                continue
            # train_arm validates the complete current training identity before
            # resuming; checkpoint presence/hashes alone cannot skip that check.
        command = [sys.executable, "-m", "scripts.short_module_screen", stage,
                   "--config", str(args.config), "--root", str(args.root), "--device", args.device]
        if args.resume:
            command.append("--resume")
        completed = subprocess.run(command, cwd=PROJECT, env=environment, check=False)
        statuses[stage] = "COMPLETE" if completed.returncode == 0 else f"EXIT_{completed.returncode}"
        write_json(args.root / "PIPELINE_STATUS.json", statuses)
    return statuses


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "prepare", "export", "baseline", "train", "evaluate-cal", "screen", "replicate", "diagnostics", "profile", "report", "publish", "status"))
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/short_module_screen_v1.yaml")
    parser.add_argument("--root", type=Path, default=Path(os.environ.get(
        "RESCENE_SHORT_MODULE_ROOT", "/home/ww/persist4d_runs/short_module_screen_v1")))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--module", choices=("Q1", "Q2", "Q3", "M0", "M1", "M2"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--audit-replay", action="store_true", help="Replay training only to recover logs; require exact checkpoint parameter equality")
    args = parser.parse_args(argv)
    config = read_config(args.config)
    if args.command == "status":
        paths = {"pipeline": args.root / "PIPELINE_STATUS.json", "training": args.root / "TRAINING_STATUS.json",
                 "report": ARTIFACTS / "REPORT_STATUS.json", "publication": args.root / "publication/PUBLICATION_RECEIPT.json"}
        print(json.dumps({name: read_json(path) if path.exists() else {"status": "NOT_RUN"}
                          for name, path in paths.items()}, indent=2))
        return 0
    started = time.monotonic()
    event = {"command": sys.argv, "stage": args.command,
             "start_utc": datetime.now(timezone.utc).isoformat(),
             "gpu_count": 0 if args.command in ("run", "prepare", "report", "publish") or (args.command == "diagnostics" and args.baseline_only) else 1, "pid": os.getpid(),
             "source_sha256": sha256(Path(__file__))}
    event["event_id"] = f"{args.command}:{event['start_utc']}"
    log_path = (args.root / "publication/EXECUTION_LOG.jsonl" if args.command in ("run", "publish")
                else ARTIFACTS / "EXECUTION_LOG.jsonl")
    append_event(log_path, {**event, "status": "STARTED"})
    try:
        if event["gpu_count"]:
            ledger = [json.loads(line) for line in (ARTIFACTS / "BUDGET_LEDGER.jsonl").read_text().splitlines()]
            total = sum(row["gpu_hours"] for row in ledger)
            incremental = sum(row["gpu_hours"] for row in ledger if row["scope"] == "ROUND1")
            # Preserve capacity for full-population evaluation and delivery;
            # export and replication also check their observed running forecast.
            if min(192. - total, 48. - incremental) < .5:
                raise ValueError("NOT_RUN_BUDGET: insufficient reserved half-hour for complete stage")
        if args.command == "run":
            result = run_pipeline(args)
        elif args.command in ("report", "publish"):
            from scripts.short_module_delivery import publish, report
            result = {"report": report, "publish": publish}[args.command](args.root)
        elif args.command == "prepare":
            result = prepare(config, args.root)
        elif args.command == "export":
            result = export(config, args.root, device=args.device, limit=args.limit)
        elif args.command == "train":
            from scripts.short_module_training import load_training_records, train_arm
            base = read_json(ARTIFACTS / "BASELINE.json")
            if any(v["status"] != "COMPLETE" for v in base.values()):
                raise ValueError("complete B0 must precede head training")
            records = load_training_records(args.root, read_json(args.root / "EXPORT_INDEX.json"))
            status_path = args.root / "TRAINING_STATUS.json"
            result = read_json(status_path) if status_path.exists() else {}
            for arm in ([args.module] if args.module else config["scope"]["train_arms"]):
                try:
                    result[arm] = train_arm(arm, seed=45, updates=1500, root=args.root,
                                           device=args.device, training_records=records, audit_replay=args.audit_replay)
                except (ValueError, RuntimeError, OSError, KeyError, TypeError) as error:
                    result[arm] = {"status": "FAILED", "error": f"{type(error).__name__}: {error}"}
                    append_event(ARTIFACTS / "EXECUTION_LOG.jsonl", {"stage": "train-arm", "module": arm, **result[arm]})
            write_json(args.root / ("TRAINING_LOG_REPLAY_STATUS.json" if args.audit_replay else "TRAINING_STATUS.json"), result)
            if any(row["status"] != "COMPLETE" for row in result.values()):
                raise ValueError("one or more training arms failed; successful arms remain saved")
        elif args.command == "profile":
            from scripts.short_module_profile import profile
            result = profile(args.root, device=args.device)
        elif args.command == "replicate":
            from scripts.short_module_results import replicate
            result = replicate(args.root, device=args.device)
        elif args.command == "diagnostics":
            from scripts.short_module_diagnostics import diagnostics
            result = diagnostics(args.root, device=args.device, include_transitions=not args.baseline_only)
        else:
            from scripts.short_module_evaluation import baseline, evaluate_cal, screen
            function = {"baseline": baseline, "evaluate-cal": evaluate_cal, "screen": screen}[args.command]
            result = function(args.root, device=args.device)
        event["status"] = "COMPLETE"
        if args.command == "run" and any(value.startswith("EXIT_") for value in result.values()):
            event["status"] = "PARTIAL"
        print(json.dumps({"stage": args.command, "status": event["status"]}, indent=2))
        return 0 if event["status"] == "COMPLETE" else 1
    except BaseException as error:
        event.update(status="INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "FAILED",
                     error=f"{type(error).__name__}: {error}")
        raise
    finally:
        event["elapsed_seconds"] = time.monotonic() - started
        event["gpu_hours"] = event["elapsed_seconds"] * event["gpu_count"] / 3600
        append_event(log_path, event)
        if event["gpu_count"]:
            append_event(ARTIFACTS / "BUDGET_LEDGER.jsonl", {**event, "scope": "ROUND1",
                         "measurement": "MEASURED_PROCESS_GPU_RESERVATION_WALLTIME"})


if __name__ == "__main__":
    raise SystemExit(main())
