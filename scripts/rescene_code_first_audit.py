"""Bounded 0929 repair campaign; immutable historical predictions are explicit inputs."""

import argparse
import fcntl
import importlib.metadata
import inspect
import json
import os
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[1]
PUBLIC = PROJECT / "artifacts/rescene_code_first_audit_v1"
OLD_PUBLIC = PROJECT / "artifacts/short_module_screen_v1"
OLD_ROOT = Path("/home/ww/persist4d_runs/short_module_screen_v1")
ROOT = Path("/home/ww/persist4d_runs/rescene_code_first_audit_v1")


def export_noaug(device):
    from omegaconf import OmegaConf

    from scripts.short_module_identity import identity_digest, source_identity
    from scripts.short_module_native import NativeSession
    from scripts.short_module_screen import (
        build_datasets,
        native_config,
        read_json,
        sha256,
        write_json,
    )
    from scripts.system_comparison_inference import pack_bool_matrix, unpack_bool_matrix

    root = ROOT / "noaug"
    assets = read_json(OLD_ROOT / "assets.local.json")
    resolved = read_json(PUBLIC / "RUN_CONFIG.json")
    if sha256(Path(assets["r1_checkpoint"])) != resolved["protocol"]["parent"]["sha256"]:
        raise ValueError("parent weight identity changed")
    old_index = read_json(OLD_ROOT / "EXPORT_INDEX.json")
    old_cache = Path(old_index["cache"])
    old_identity = read_json(old_cache / "IDENTITY.json")
    old_project = next(Path(p).parents[1] for p in old_identity["sources"]
                       if p.endswith("/scripts/short_module_native.py"))
    sources = [PROJECT / "scripts/short_module_screen.py"]
    for value in old_identity["sources"]:
        path = Path(value)
        if path.name == "short_module_data.py":
            continue
        sources.append(PROJECT / path.relative_to(old_project) if path.is_relative_to(old_project) else path)
    native = native_config(root, assets)
    datasets = build_datasets(native, assets)
    discovery = {h: (list(ds.sequence_names), len(ds.data)) for h, ds in datasets.items()}
    for dataset in datasets.values():
        dataset.apply_training_augmentation = False
    if discovery != {h: (list(ds.sequence_names), len(ds.data)) for h, ds in datasets.items()}:
        raise ValueError("augmentation switch changed discovery")
    population = read_json(PUBLIC / "SHORT_POPULATION.json")
    records = [entry["record"] for entry in old_index["entries"] if entry["record"]["role"] in ("CAL", "SEL")]
    identity = {"schema": "no-training-augmentation-predictions-v1",
                "sources": source_identity(sources),
                "export_function": identity_digest(inspect.getsource(export_noaug)),
                "native_config": OmegaConf.to_container(native, resolve=True),
                "parent_sha256": resolved["protocol"]["parent"]["sha256"],
                "records": records, "dataset_modes": {str(h): ds.mode for h, ds in datasets.items()},
                "apply_training_augmentation": False,
                "numeric_environment": resolved["numeric_environment"],
                "reference_prediction_identity": old_index["cache_identity_sha256"]}
    digest = identity_digest(identity)
    cache = root / "cache" / digest
    write_json(cache / "IDENTITY.json", identity)
    write_json(root / "assets.local.json", assets)
    session = None
    entries = []
    old_entries = {entry["input_id"]: entry for entry in old_index["entries"]}
    for record in records:
        key = record["input_id"]
        entry_path = cache / f"{key}.json"
        if entry_path.exists():
            entry = read_json(entry_path)
            if all(sha256(cache / entry[name]["file"]) == entry[name]["sha256"]
                   for name in ("prediction", "targets")):
                entries.append(entry)
                continue
        if session is None:
            session = NativeSession(config=native, datasets=datasets, assets=assets, device=device)
        prediction, targets, audit = session.produce(record, compare_native=(not entries))
        original = torch.load(old_cache / old_entries[key]["targets"]["file"],
                              map_location="cpu", weights_only=False)["target"]
        original["masks"] = unpack_bool_matrix(original["masks"])
        for field in ("masks", "labels", "ids", "temporal_stages"):
            if not torch.equal(targets["target"][field], original[field]):
                raise ValueError(f"noaug changed target {field} for {key}")
        targets["target"]["masks"] = pack_bool_matrix(targets["target"]["masks"])
        entry = {"input_id": key, "record": record, "audit": audit, "target_parity": True}
        for name, value in (("prediction", prediction), ("targets", targets)):
            path = cache / f"{key}.{name}.pt"
            torch.save(value, path)
            entry[name] = {"file": path.name, "sha256": sha256(path), "bytes": path.stat().st_size}
        write_json(entry_path, entry)
        entries.append(entry)
        print(f"Noaug native inputs {len(entries)}/{len(records)}", flush=True)
    expected = sum(r["role"] in ("CAL", "SEL") for r in population["records"])
    if len(entries) != expected:
        raise ValueError("noaug population incomplete")
    result = {"status": "COMPLETE", "cache": str(cache), "cache_identity_sha256": digest,
              "entries": entries, "condition": "no-training-augmentation",
              "completed_unique_inputs": len(entries), "expected_unique_inputs": expected}
    write_json(root / "EXPORT_INDEX.json", result)
    write_json(PUBLIC / "NOAUG_EXPORT.json", {k: v for k, v in result.items() if k != "entries"})
    return result


def evaluate_noaug(device):
    from scripts.short_module_evaluation import evaluate_point
    from scripts.short_module_screen import read_json, write_json

    lock = read_json(PUBLIC / "selection/CAL_LOCK.json")["selected_updates"]
    results = []
    for role in ("CAL", "SEL"):
        for arm in ("B0", "Q1", "Q2", "Q3", "M0", "M1", "M2"):
            results.append(evaluate_point(ROOT / "noaug", module=arm, seed=45,
                                          step=lock.get(arm, 0), role=role, device=device,
                                          checkpoint_root=OLD_ROOT))
            write_json(PUBLIC / "NOAUG_FIXED_HEADS.json", results)
    return results


def train_q(device):
    from scripts.short_module_identity import assert_sources_current
    from scripts.short_module_screen import read_json, write_json
    from scripts.short_module_training import load_training_records, train_arm

    index = read_json(ROOT / "EXPORT_INDEX.json")
    assert_sources_current(index["label_identity"])
    records = load_training_records(ROOT, index)
    results = {}
    for arm in ("Q1", "Q2", "Q3"):
        results[arm] = train_arm(arm, seed=45, updates=1500, root=ROOT, device=device,
                                 training_records=records)
        write_json(PUBLIC / "Q_TRAINING_STATUS.json", results)
    return results


def evaluate_q(device):
    from scripts.short_module_evaluation import evaluate_point
    from scripts.short_module_screen import read_json, write_json

    lock = read_json(PUBLIC / "selection/CAL_LOCK.json")["selected_updates"]
    results = []
    for role in ("CAL", "SEL"):
        base = evaluate_point(ROOT, module="B0", seed=45, step=0, role=role, device=device)
        for arm in ("Q1", "Q2", "Q3"):
            old = evaluate_point(ROOT, module=arm, seed=45, step=lock[arm], role=role,
                                 device=device, checkpoint_root=OLD_ROOT)
            new = evaluate_point(ROOT, module=arm, seed=45, step=lock[arm], role=role, device=device)
            results.append({"role": role, "arm": arm, "locked_step": lock[arm],
                            "baseline": base, "original": old, "repaired": new,
                            "repair_delta": {h: new["metrics"][h] - old["metrics"][h]
                                             for h in ("T1", "T2")}})
            write_json(PUBLIC / "Q_MATCHED_COMPARISON.json", results)
    return results


def gpu_command(command, device):
    from scripts.rescene_code_first_geometry import evaluate as evaluate_m
    from scripts.rescene_code_first_geometry import train as train_m
    from scripts.short_module_screen import append_event, read_json, sha256, write_json

    ROOT.mkdir(parents=True, exist_ok=True)
    reservation = (ROOT / "gpu-reservation.lock").open("a")
    fcntl.flock(reservation, fcntl.LOCK_EX | fcntl.LOCK_NB)
    source_sha = sha256(Path(__file__))
    numeric = read_json(PUBLIC / "RUN_CONFIG.json")["numeric_environment"]
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "CUBLAS_WORKSPACE_CONFIG"):
        if os.environ.get(name) != numeric[name]:
            raise ValueError(f"controlled runtime requires {name}={numeric[name]}")
    original_index = read_json(OLD_ROOT / "EXPORT_INDEX.json")
    original_identity = read_json(Path(original_index["cache"]) / "IDENTITY.json")
    from scripts.short_module_identity import identity_digest
    if identity_digest(original_identity) != original_index["cache_identity_sha256"]:
        raise ValueError("historical native configuration/identity content changed")
    if sha256(PUBLIC / "SHORT_POPULATION.json") != sha256(OLD_PUBLIC / "SHORT_POPULATION.json"):
        raise ValueError("audit scan population or evaluation order changed")
    for name, version in original_identity["versions"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f"controlled library version changed: {name}")
    index = read_json(ROOT / "EXPORT_INDEX.json")
    spec = Path(read_json(ROOT / "assets.local.json")["metric_dataset_spec"])
    if sha256(spec) != index["label_identity"]["dataset_spec_sha256"]:
        raise ValueError("dataset spec changed; relabel in a new version before continuing")
    prior_path = OLD_PUBLIC / "BUDGET_LEDGER.jsonl"
    prior_events = [json.loads(line) for line in prior_path.read_text().splitlines() if line.strip()]
    prior = sum(event["gpu_hours"] for event in prior_events)
    if abs(prior - read_json(OLD_PUBLIC / "REPORT_STATUS.json")["cost"]["cumulative_gpu_hours"]) > 1e-8:
        raise ValueError("historical cost ledger disagrees with settled report")
    ledger = PUBLIC / "BUDGET_LEDGER.jsonl"
    events = [json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []
    spent = sum(event["gpu_hours"] for event in events)
    if spent >= 4 or prior + spent >= 192:
        raise ValueError("bounded audit budget exhausted")
    write_json(PUBLIC / "BUDGET_CONTRACT.json", {
        "inherited_gpu_hours": prior, "inherited_ledger_sha256": sha256(prior_path),
        "campaign_cap_gpu_hours": 4, "cumulative_cap_gpu_hours": 192,
        "forecast_gpu_hours": 1.5, "max_concurrent_GPUs": 1,
        "measurement": "whole process GPU reservation wall time; failures included"})
    start = time.monotonic()
    utc = datetime.now(timezone.utc).isoformat()
    status = "FAILED"
    try:
        result = {"train-q": train_q, "evaluate-q": evaluate_q,
                  "export-noaug": export_noaug, "evaluate-noaug": evaluate_noaug,
                  "train-m": train_m, "evaluate-m": evaluate_m}[command](device)
        status = "COMPLETE"
        return result
    finally:
        elapsed = time.monotonic() - start
        append_event(ledger, {"event_id": f"{command}:{utc}", "command": command,
                             "start_utc": utc, "gpu_count": 1, "device": device,
                             "elapsed_seconds": elapsed, "gpu_hours": elapsed / 3600,
                             "status": status, "source_sha256": source_sha})
        reservation.close()


def relabel():
    from scripts.short_module_data import geometry_assignment, quality_labels
    from scripts.short_module_identity import identity_digest, label_identity
    from scripts.short_module_native import unpack_prediction
    from scripts.short_module_screen import read_json, sha256, write_json
    from scripts.system_comparison_inference import pack_bool_matrix, unpack_bool_matrix

    old_index = read_json(OLD_ROOT / "EXPORT_INDEX.json")
    if old_index["status"] != "COMPLETE":
        raise ValueError("historical prediction export must be complete")
    assets = read_json(OLD_ROOT / "assets.local.json")
    cache = Path(old_index["cache"])
    prediction_identity = {"kind": "immutable-baseline-import",
                           "baseline_commit": "4bf00902c9428795f7f47547bf462540aa304cc6",
                           "historical_cache_identity": old_index["cache_identity_sha256"],
                           "historical_index_sha256": sha256(OLD_ROOT / "EXPORT_INDEX.json")}
    identity = label_identity(prediction_identity=prediction_identity,
                              dataset_spec=assets["metric_dataset_spec"])
    label_sha = identity_digest(identity)
    directory = ROOT / "labels" / label_sha
    write_json(directory / "IDENTITY.json", identity)
    groups = defaultdict(Counter)
    entries = []
    for number, old_entry in enumerate(old_index["entries"]):
        for field in ("prediction", "targets"):
            if sha256(cache / old_entry[field]["file"]) != old_entry[field]["sha256"]:
                raise ValueError(f"historical shard identity changed: {old_entry['input_id']} {field}")
        saved = torch.load(cache / old_entry["prediction"]["file"], map_location="cpu", weights_only=False)
        targets = torch.load(cache / old_entry["targets"]["file"], map_location="cpu", weights_only=False)
        old_quality = targets["quality"]
        targets["target"]["masks"] = unpack_bool_matrix(targets["target"]["masks"])
        parent = unpack_prediction(saved["parent"])
        labels = quality_labels(parent.prediction(), targets["target"],
                                dataset_spec=Path(assets["metric_dataset_spec"]))
        if not torch.equal(labels["geometry_valid"], old_quality["valid"]):
            raise ValueError("Q-only repair altered legacy geometry eligibility")
        assignment = geometry_assignment(parent.prediction(), targets["target"], labels,
                                         candidate_keys=saved["candidate_keys"])
        if not torch.equal(assignment, targets["assignment"]):
            raise ValueError("Q-only repair altered legacy M assignment")
        expanded = geometry_assignment(parent.prediction(), targets["target"],
                                       {**labels, "geometry_valid": labels["valid"]},
                                       candidate_keys=saved["candidate_keys"])
        record = old_entry["record"]
        counts = groups[record["role"], record["horizon"]]
        counts["candidates"] += len(labels["valid"])
        counts["new_quality_eligible"] += int((labels["valid"] & ~old_quality["valid"]).sum())
        counts["expanded_M_assignment_changes"] += int((expanded != assignment).sum())
        counts["expanded_M_new_matches"] += int(((expanded >= 0) & (assignment < 0)).sum())
        counts["legacy_M_assignment_changes"] += int((assignment != targets["assignment"]).sum())
        targets["quality"] = labels
        targets["expanded_assignment"] = expanded
        targets["target"]["masks"] = pack_bool_matrix(targets["target"]["masks"])
        destination = directory / f"{old_entry['input_id']}.targets.pt"
        torch.save(targets, destination)
        entries.append({**old_entry,
                        "prediction": {**old_entry["prediction"],
                                       "file": str(cache / old_entry["prediction"]["file"])},
                        "targets": {"file": str(destination), "sha256": sha256(destination),
                                    "bytes": destination.stat().st_size}})
        if (number + 1) % 50 == 0:
            print(f"Relabeled and verified {number + 1}/{len(old_index['entries'])}", flush=True)
    index = {**old_index, "entries": entries, "prediction_identity": prediction_identity,
             "label_identity_sha256": label_sha, "label_identity": identity}
    write_json(ROOT / "EXPORT_INDEX.json", index)
    write_json(ROOT / "assets.local.json", assets)
    for name in ("RUN_CONFIG.json", "SHORT_POPULATION.json", "INPUT_MANIFEST.json", "selection/CAL_LOCK.json"):
        write_json(PUBLIC / name, read_json(OLD_PUBLIC / name))
    report = {"status": "COMPLETE", "label_identity": identity, "label_identity_sha256": label_sha,
              "prediction_files_reused": len(entries), "R1_forwards": 0,
              "groups": [{"role": role, "H": horizon, **counts}
                         for (role, horizon), counts in sorted(groups.items())],
              "Q_comparison": "original inputs, seed45/sample plan/1500 updates; original locked steps",
              "geometry_policy": "legacy eligibility, assignment, segment targets and weights unchanged"}
    write_json(PUBLIC / "RELABEL_AUDIT.json", report)
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("relabel", "train-q", "evaluate-q", "export-noaug", "evaluate-noaug",
                                             "train-m", "evaluate-m", "evaluate-all"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    os.environ["SHORT_MODULE_ARTIFACTS"] = str(PUBLIC)
    torch.set_num_threads(2)
    if args.command == "relabel":
        relabel()
    elif args.command == "evaluate-all":
        for command in ("evaluate-q", "evaluate-m", "evaluate-noaug"):
            gpu_command(command, args.device)
        from scripts.rescene_code_first_report import main as report
        report()
    else:
        gpu_command(args.command, args.device)


if __name__ == "__main__":
    main()
