"""Evaluate fixed trained checkpoints on separately locked formal populations."""

import argparse
import collections
import json
import time
from pathlib import Path

from scripts.native_long_budget import write_json_atomic
from scripts.native_long_postselection import read_json, sha256


def validate_request(lock, manifest, arm, step, role):
    if (lock["training_seed"] != 45 or lock["evaluation_seed"] != 9001
            or not any(r["arm"] == arm and r["step"] == step for r in lock["rows"])):
        raise ValueError("checkpoint is not locked for formal confirmation")
    if role not in {"PB", "LOCAL-T2", "ADDITIONAL"} or not manifest["records"].get(role):
        raise ValueError("formal role is missing or unapproved")


def validate_checkpoint(saved, identity, step):
    if saved["global_step"] != step:
        raise ValueError("checkpoint step differs from the formal lock")
    if saved["native_long_resume"]["identity"] != identity:
        raise ValueError("checkpoint training identity differs")


def formal_dataset(config, data_root, manifest):
    import hydra
    from omegaconf import OmegaConf

    folder = data_root / "processed/rio"
    cfg = OmegaConf.to_container(config.data.validation_dataset, resolve=True)
    cfg.update(data_dir=str(folder), temporal_window=1, mode="validation",
               apply_training_augmentation=False, known_empty_scan_policy="allow_actual",
               exclude_unsupervised_sequences=False)
    for key in ("label_db_filepath", "change_label_db_filepath", "color_mean_std"):
        if key in cfg and cfg[key] is not None:
            cfg[key] = str(folder / Path(cfg[key]).name)
    dataset = hydra.utils.instantiate(cfg)
    if len(dataset.data) != len(manifest["validation_aliases"]):
        raise ValueError("formal validation scan population differs")
    for index, row in enumerate(dataset.data):
        alias = Path(row["instance_gt_filepath"]).stem
        expected = manifest["validation_aliases"][alias]
        if expected["index"] != index or Path(row["raw_filepath"]).parent.name != expected["physical_scan_id"]:
            raise ValueError("formal validation scan identity/index differs")
        for key in ("filepath", "instance_gt_filepath"):
            path = Path(row[key].replace("../../", ""))
            if path.is_absolute():
                relative = path.relative_to(manifest["source_data_root"])
            else:
                relative = path.relative_to("data")
            row[key] = str(data_root / relative)
    return dataset


def formal_samples(dataset, record):
    from scripts.native_long_runtime import load_samples

    sample = list(load_samples({"rio": dataset}, [record])[0])
    # PB prefixes retain the original H5 context's ambiguity annotation; other
    # roles use their own source sequence. A temporal-window-1 loader has none.
    sample[8] = record["ambiguities"]
    return [tuple(sample)]


def evaluate(root, directory, data_root, arm, step, role):
    import torch

    from datasets.native_long_dataset import (
        NativeLongCollator,
        isolated_rng,
        stable_seed,
    )
    from scripts.evaluate_persist4d import _metric_target
    from scripts.evaluate_persist4d_p6a import build_rio_class_mapper
    from scripts.native_long_campaign import PROJECT, compose_config
    from scripts.native_long_execution import native_training_identity
    from scripts.native_long_runtime import (
        configure_numeric,
        move_batch,
        system_from_common,
    )
    from scripts.p6a_metrics import OfficialMetricAccumulator
    from scripts.rescene_task_postprocess import extract_official_task_prediction

    started = time.monotonic()
    configure_numeric()
    manifest_path, lock_path = directory / "FORMAL_POPULATION.json", directory / "FORMAL_LOCK.json"
    manifest, lock = read_json(manifest_path), read_json(lock_path)
    validate_request(lock, manifest, arm, step, role)
    if (sha256(manifest_path) != lock["FORMAL_POPULATION_sha256"]
            or sha256(root / "selection/CAL_LOCK.json") != lock["CAL_LOCK_sha256"]):
        raise ValueError("formal population or CAL lock SHA changed")
    for item in manifest["data_files"]:
        path = data_root / item["relative_path"]
        if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise ValueError("formal staged data bytes/SHA differ")
    config = compose_config(PROJECT / "conf/config_native_long_retrain.yaml", root=root, arm=arm, seed=45)
    training_identity = native_training_identity(root, config)
    checkpoint = root / "training" / arm / "seed45" / f"update={step:05d}.ckpt"
    records = manifest["records"][role]
    if records != sorted(records, key=lambda r: (r["horizon"], r["input_id"])):
        raise ValueError("formal input order changed")
    dependencies = {"training": training_identity, "checkpoint_sha256": sha256(checkpoint),
                    "formal_lock_sha256": sha256(lock_path), "formal_population_sha256": sha256(manifest_path),
                    "evaluation_seed": 9001, "input_ids": [r["input_id"] for r in records],
                    "native_process_protocol": manifest["native_process_protocol"],
                    "producer_sha256": {Path(p).name: sha256(p) for p in (
                        Path(__file__), Path(__file__).with_name("native_long_postselection.py"))},
                    "metric_sources": read_json(root / "SOURCE_AND_INITIALIZATION.json")["installed_libraries"]["stmetrics"]}
    result_path = directory / "evaluation" / arm / "seed45" / f"{role}-{step:05d}.json"
    if result_path.exists():
        prior = read_json(result_path)
        if prior["identity"] != dependencies:
            raise ValueError("existing formal result has a different producer/input identity")
        if prior["status"] == "COMPLETE":
            for item in (*prior["evidence"], *prior["reference_evidence"]):
                if sha256(item["path"]) != item["sha256"]:
                    raise ValueError("cached formal evidence SHA differs")
            return prior
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    validate_checkpoint(saved, training_identity, step)
    system, _ = system_from_common(root, config)
    system.load_state_dict(saved["state_dict"], strict=True)
    del saved
    system.to("cuda:0").eval()
    dataset = formal_dataset(config, data_root, manifest)
    collate = NativeLongCollator(config, training=False)
    mapper = build_rio_class_mapper(dataset)
    expected = dict(collections.Counter(f"T{r['horizon']}" for r in records))

    def metric():
        return OfficialMetricAccumulator(mode="strict_online", dataset_spec=config.instance_metric.dataset)

    pooled = {h: metric() for h in sorted({r["horizon"] for r in records})}
    references = {}
    result = {"arm": arm, "seed": 45, "step": step, "role": role, "status": "INCOMPLETE",
              "completed": dict.fromkeys(expected, 0), "expected": expected, "identity": dependencies,
              "metrics": {}, "official_metrics": {}, "evidence": [], "by_reference": [],
              "reference_evidence": [], "setup_seconds": time.monotonic() - started}
    write_json_atomic(result_path, result)
    for record in records:
        h, ref = record["horizon"], record["reference_id"]
        with isolated_rng(stable_seed(9001, record["input_id"]), cuda_devices=(0,)):
            data, targets, _ = move_batch(collate(formal_samples(dataset, record)), torch.device("cuda:0"))
            with torch.inference_mode():
                outputs = system(data, point2segment=[targets[0]["point2segment"]],
                                 raw_coordinates=system._process_raw_coordinates(data), is_eval=True)
            prediction = extract_official_task_prediction(
                system=system, output=outputs, target_low_resolution=targets[0],
                target_full_resolution=data.target_full[0], data=data, class_mapper=mapper,
                latest_stage_index=h - 1).prediction()
            target = _metric_target(data.target_full[0], dataset)
        pooled[h].update(prediction, target)
        references.setdefault((h, ref), metric()).update(prediction, target)
        del outputs, data, targets, prediction, target
        result["completed"][f"T{h}"] += 1
        result["elapsed_seconds"] = time.monotonic() - started
        write_json_atomic(result_path, result)
    if result["completed"] != expected:
        raise ValueError("formal evaluation is incomplete")
    for h, accumulator in pooled.items():
        values = accumulator.compute()
        result["metrics"][f"T{h}"] = values["online_t-mAP"]
        result["official_metrics"][f"T{h}"] = values
        path = result_path.with_name(f"{role}-{step:05d}-T{h}-official-state.json")
        write_json_atomic(path, accumulator.export_evidence())
        result["evidence"].append({"path": str(path), "sha256": sha256(path)})
    result["metrics"]["S_long"] = (sum(result["metrics"][f"T{h}"] for h in (3, 4, 5)) / 3
                                       if all(h in pooled for h in (3, 4, 5)) else None)
    counts = collections.Counter((r["horizon"], r["reference_id"]) for r in records)
    for (h, ref), accumulator in sorted(references.items()):
        values = accumulator.compute()
        if accumulator._updates != counts[h, ref]:
            raise ValueError("formal per-reference denominator differs")
        result["by_reference"].append({"horizon": h, "reference_id": ref,
            "expected": counts[h, ref], "completed": accumulator._updates, "AP": values["online_t-mAP"]})
        path = result_path.with_name(f"{role}-{step:05d}-T{h}-{ref}-official-state.json")
        write_json_atomic(path, accumulator.export_evidence())
        result["reference_evidence"].append({"horizon": h, "reference_id": ref,
                                             "path": str(path), "sha256": sha256(path)})
    result["elapsed_seconds"] = time.monotonic() - started
    result["status"] = "COMPLETE"
    write_json_atomic(result_path, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--arm", choices=("E0", "E1", "E3"), required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--role", choices=("PB", "LOCAL-T2", "ADDITIONAL"), required=True)
    args = parser.parse_args()
    result = evaluate(args.root, args.directory, args.data_root, args.arm, args.step, args.role)
    print(json.dumps({k: result[k] for k in ("arm", "step", "role", "status", "completed", "metrics")}))


if __name__ == "__main__":
    main()
