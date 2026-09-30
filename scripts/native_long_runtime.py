"""Finite real native initialization, correctness and cost measurements."""

import argparse
import copy
import hashlib
import inspect
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from datasets.native_long_dataset import (
    NativeLongCollator,
    build_native_datasets,
    canonical_orders,
    isolated_rng,
    stable_seed,
)
from scripts.native_long_campaign import (
    ARTIFACTS,
    PROJECT,
    code_identity,
    compose_config,
)
from scripts.short_module_screen import append_event, read_json, sha256, write_json
from trainer.native_long_trainer import NativeLongTrainer
from utils.rescene_rootcause_preflight import (
    build_tensor_state_manifest,
    load_common_initialization,
    validate_common_tensor_state,
)


def system_from_common(root, config):
    with isolated_rng(int(config.general.seed)):
        system = NativeLongTrainer(config)
    source = read_json(root / "SOURCE_AND_INITIALIZATION.json")
    seed = str(config.general.seed)
    entry = source["initialization"][seed]
    audit = load_common_initialization(
        system, entry["path"], expected_sha256=entry["sha256"],
        allowed_new_prefixes=("model.native_long_feedback.",) if config.native_long.arm == "E3" else ())
    audit["reference"] = f"external:native-long/common/seed{seed}/{entry['sha256']}"
    if system.model.native_long_feedback is not None:
        system.model.native_long_feedback.checkpoint_chunks = bool(
            config.native_long.get("feedback_checkpoint", False))
    return system, audit


def load_samples(datasets, records):
    samples = []
    for record in records:
        dataset = datasets[record["domain"]]
        indices = tuple(record["scan_indices"])
        with isolated_rng(stable_seed(9001, record["input_id"])):
            sample = list(dataset.load_scan_indices(indices[0], indices, change_file=None))
        sample[3] = record["input_id"]
        sample[7] = stable_seed(record["input_id"])
        samples.append(tuple(sample))
    return samples


def validate_batch(batch, samples, records):
    data, targets, names = batch
    if len(targets) != len(samples) or names != [r["input_id"] for r in records]:
        raise ValueError("collator changed batch/identity or dropped an empty target")
    summary = []
    for index, (sample, record, target) in enumerate(zip(samples, records, targets, strict=True)):
        inverse = data.inverse_maps[index].long()
        stages = target["temporal_stages"].long()
        full_stages = torch.from_numpy(sample[6][:, 3]).long()
        full = data.target_full[index]
        if not torch.equal(stages[inverse], full_stages):
            raise ValueError("mixed batch inverse does not preserve full temporal stages")
        from models.perception_gain import derive_segment_stage_ids

        low_segment_stages = derive_segment_stage_ids(target["point2segment"], stages)
        if not torch.equal(low_segment_stages[target["point2segment"][inverse]], full_stages):
            raise ValueError("LOW segment projection crossed a temporal stage")
        # Full partitions are independently native-remapped; voxel selection can
        # cross a spatial superpoint boundary, so their numeric IDs need not match LOW.
        for segment in full["point2segment"].unique():
            if full_stages[full["point2segment"] == segment].unique().numel() != 1:
                raise ValueError("FULL partition spans different physical scans")
        if not torch.equal(full["temporal_stages"].long(), full_stages):
            raise ValueError("full GT temporal labels differ")
        if set(stages.tolist()) != set(range(record["horizon"])):
            raise ValueError("real input horizon differs from collated labels")
        if target["masks"].shape[1] != stages.numel() or full["masks"].shape[1] != inverse.numel():
            raise ValueError("GT mask columns do not match low/full vertices")
        summary.append({"input_id": record["input_id"], "horizon": record["horizon"],
                        "low_points": stages.numel(), "full_points": inverse.numel(),
                        "segments": int(target["point2segment"].max()) + 1,
                        "gt_instances": target["labels"].numel(),
                        "empty_stage_instance_pairs": int((full["masks"].sum(1) == 0).sum()),
                        "inverse_and_time": "PASS"})
    return summary


def initialize(root, config):
    datasets = build_native_datasets(config, config.native_long.data_root, augmentation=False)
    population = read_json(root / "POPULATION.json")
    train = [r for r in population["references"] if r["role"] == "TRAIN"]
    long = [r for r in train if r["Tmax"] >= 5]
    short = min(train, key=lambda r: (sum(datasets["rio"].data[i]["file_len"]
                                         for i in r["scan_indices"][:2]), r["reference_id"]))
    largest = max(long, key=lambda r: sum(sorted(
        [datasets["rio"].data[i]["file_len"] for i in r["scan_indices"]], reverse=True)[:5]))

    def record(ref, h, *, large=False):
        order = canonical_orders(ref)[0]
        if large:
            order = sorted(order, key=lambda i: datasets["rio"].data[ref["scan_indices"][i]]["file_len"],
                           reverse=True)
        order = order[:h]
        return {"input_id": f"PREFLIGHT:{ref['reference_id']}:T{h}", "domain": "rio", "horizon": h,
                "reference_id": ref["reference_id"], "scan_ids": [ref["scan_ids"][i] for i in order],
                "scan_indices": [ref["scan_indices"][i] for i in order]}

    records = {"A": record(short, 2), "B": record(largest, 5, large=True),
               "SCANNET": {"input_id": "PREFLIGHT:SCANNET:0", "domain": "scannet", "horizon": 1,
                           "reference_id": "scannet:0", "scan_ids": ["scannet:0"], "scan_indices": [0]}}
    for h in range(1, 6):
        records[f"COST_T{h}"] = record(largest if h >= 3 else short, h, large=h >= 3)
    write_json(root / "PREFLIGHT_INPUTS.json", records)
    collate = NativeLongCollator(config, training=False)
    checks = {}
    for group in (("A", "B"), ("B", "SCANNET")):
        selected = [records[k] for k in group]
        samples = load_samples(datasets, selected)
        batch = collate(copy.deepcopy(samples))
        checks["+".join(group)] = validate_batch(batch, samples, selected)
    write_json(ARTIFACTS / "initialization/REAL_MIXED_BATCH.json", checks)
    # Inspect actual annotation mappings, never infer entities from local ID coincidence.
    gt = []
    for ref in train:
        per_id_classes = {}
        for scan, index in zip(ref["scan_ids"], ref["scan_indices"], strict=True):
            row = datasets["rio"].data[index]
            annotation = read_json(Path(row["raw_instance_filepath"]))
            if annotation["scan_id"] != scan:
                raise ValueError("raw annotation scan does not match metadata")
            mappings = {int(g["id"]): int(g["objectId"]) for g in annotation["segGroups"]}
            if any(k != v for k, v in mappings.items()):
                raise ValueError("preprocessed ID is not the verified reference object ID")
            points = np.load(row["filepath"], mmap_mode="r")
            present = set(points[:, 11].astype(np.int64)) - {-1}
            if not present <= set(mappings):
                raise ValueError("processed instance ID is absent from annotation mapping")
            for identity, label in np.unique(points[:, 10:12].astype(np.int64), axis=0):
                # Columns are semantic class then global instance ID.
                per_id_classes.setdefault(int(label), set()).add(int(identity))
        conflicts = {str(i): sorted(c) for i, c in per_id_classes.items() if i >= 0 and len(c) > 1}
        gt.append({"reference_id": ref["reference_id"], "scans": len(ref["scan_ids"]),
                   "id_objectId_equal": True, "class_conflicts": conflicts})
    write_json(ARTIFACTS / "initialization/GT_IDENTITY_AUDIT.json",
               {"status": "VERIFIED_EXISTING_GLOBAL_IDS", "references": gt,
                "conflict_policy": "preserve existing annotation/criterion semantics; no prediction relabeling"})
    with isolated_rng(int(config.general.seed)):
        system = NativeLongTrainer(config)
    state = system.state_dict()
    source = read_json(root / "SOURCE_AND_INITIALIZATION.json")
    path = root / "initialization" / f"seed{config.general.seed}" / "common_initial_state.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        saved = torch.load(path, weights_only=True)["state_dict"]
        validate_common_tensor_state(state, saved)
    else:
        torch.save({"state_dict": state, "seed": int(config.general.seed),
                    "pretrained_scope": "encoder_only"}, path)
    manifest = build_tensor_state_manifest(state, trainable_names={
        k for k, p in system.named_parameters() if p.requires_grad})
    source["initialization"][str(config.general.seed)] = {
        "path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size,
        "tensor_state": manifest, "pretrained_load": system.model.backbone.pretrained_load_audit,
        "encoder_training_mode": "frozen weights, train mode retained; eval during evaluation",
        "mutable_buffers": [k for k, _ in system.named_buffers() if k.startswith("model.backbone.model.enc")],
    }
    source["code"] = code_identity()
    write_json(root / "SOURCE_AND_INITIALIZATION.json", source)
    write_json(ARTIFACTS / "SOURCE_AND_INITIALIZATION.json", source)
    print("INITIALIZED", records["A"]["scan_ids"], records["B"]["scan_ids"], flush=True)


def configure_numeric():
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def move_batch(batch, device):
    from scripts.evaluate_persist4d import _move_data_to_device, _move_targets_to_device

    data, targets, names = batch
    return _move_data_to_device(data, device), _move_targets_to_device(targets, device), names


def native_forward(system, datasets, collate, record, *, device, trace=False, deep=False,
                   capture_aux=False):
    traces, handles = {}, []

    def hook(name):
        def save(module, args, output):
            if isinstance(output, dict):
                traces[name] = {k: v.detach().cpu().clone() for k, v in output.items()
                                if k in ("feat", "coord", "grid_coord", "serialized_order")
                                and isinstance(v, torch.Tensor)}
            elif isinstance(output, torch.Tensor):
                traces[name] = {"tensor": output.detach().cpu().clone()}
            elif hasattr(output, "features"):
                traces[name] = {"features": output.features.detach().cpu().clone()}
        return save

    if trace:
        for name, module in (("embedding", system.model.backbone.model.embedding),
                             ("encoder", system.model.backbone.model.enc),
                             ("decoder", system.model.backbone.model.dec),
                             ("maskF", system.model.mask_features_head),
                             ("query", system.model.decoder_norm)):
            handles.append(module.register_forward_hook(hook(name)))
        if deep:
            for name, module in system.model.backbone.model.enc.enc0.block0.named_modules():
                if name:
                    handles.append(module.register_forward_hook(hook("enc0.block0." + name)))
    try:
        with isolated_rng(stable_seed(9001, record["input_id"]), cuda_devices=(device.index,)):
            samples = load_samples(datasets, [record])
            batch = collate(samples)
            data, targets, _ = move_batch(batch, device)
            if trace:
                traces["input"] = {k: data[k].detach().cpu().clone() for k in ("coord", "feat", "grid_coord")}
                traces["mapping"] = {"inverse": data.inverse_maps[0].cpu().clone(),
                                      "point2segment": targets[0]["point2segment"].cpu().clone()}
            system.model.native_long_draw_ids = [record["input_id"]]
            with torch.inference_mode():
                output = system(data, point2segment=[targets[0]["point2segment"]],
                                raw_coordinates=system._process_raw_coordinates(data), is_eval=True)
            if len(output["aux_outputs"]) != 12:
                raise ValueError("native forward changed the 12 auxiliary outputs")
            traces["final"] = {"classes": output["pred_logits"].cpu().clone(),
                               "masks": output["pred_masks"][0].cpu().clone()}
            if capture_aux:
                traces["aux"] = [{"classes": item["pred_logits"].cpu().clone(),
                                  "masks": item["pred_masks"][0].cpu().clone()}
                                 for item in output["aux_outputs"]]
            if any(not torch.isfinite(v).all() for v in traces["final"].values()):
                raise ValueError("native output is nonfinite")
            return traces
    finally:
        for handle in handles:
            handle.remove()


def order_check(root, config, order, repeat, *, deep=False, native_sparse=False):
    configure_numeric()
    device = torch.device("cuda:0")
    system, _ = system_from_common(root, config)
    if native_sparse:
        import spconv.constants
        from spconv.core import ConvAlgo
        from spconv.pytorch.conv import SparseConvolution

        if not spconv.constants.ALL_WEIGHT_IS_KRSC:
            raise ValueError("sparse algorithm probe requires unchanged KRSC weight layout")
        for module in system.modules():
            if isinstance(module, SparseConvolution):
                module.algo = ConvAlgo.Native
    system.to(device).eval()
    datasets = build_native_datasets(config, config.native_long.data_root, augmentation=False)
    collate = NativeLongCollator(config, training=False)
    records = read_json(root / "PREFLIGHT_INPUTS.json")
    last_a = None
    for letter in order:
        result = native_forward(system, datasets, collate, records[letter], device=device,
                                trace=letter == "A", deep=deep)
        if letter == "A":
            last_a = result
    prefix = "native-" if native_sparse else "deep-" if deep else ""
    path = root / "runtime" / f"{prefix}{order}-{repeat}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(last_a, path)
    print("ORDER_COMPLETE", order, repeat, flush=True)


def run_child(root, phase, *, order=None, repeat=None):
    command = [sys.executable, "-m", "scripts.native_long_runtime", phase, "--root", str(root)]
    if order:
        command += ["--order", order, "--repeat", str(repeat)]
    environment = {**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
                   "OPENBLAS_NUM_THREADS": "2", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                   "CUDA_VISIBLE_DEVICES": "0"}
    cpu_only = phase in {"initialize", "io", "runtime-metric", "assets"}
    if cpu_only:
        environment["CUDA_VISIBLE_DEVICES"] = ""
    ledger = ARTIFACTS / "COST_LEDGER.jsonl"
    used = sum(json.loads(line)["gpu_hours"] for line in ledger.read_text().splitlines()) if ledger.exists() else 0.
    numerical = phase in {"order", "deep", "native"}
    numerical_used = sum(json.loads(line)["gpu_hours"] for line in ledger.read_text().splitlines()
                         if json.loads(line)["phase"] in {"order", "deep", "native"}) if ledger.exists() else 0.
    available = min(1. - used, .5 - numerical_used) if numerical else 1. - used
    if not cpu_only and available <= 0:
        raise RuntimeError("bounded preflight GPU budget exhausted")
    started = time.perf_counter()
    exit_code = -1
    try:
        result = subprocess.run(command, cwd=PROJECT, env=environment, check=False,
                                timeout=1200 if cpu_only else available * 3600)
        exit_code = result.returncode
    finally:
        elapsed = time.perf_counter() - started
        event = {"event_id": f"native-long:{phase}:{order}:{repeat}:{time.time_ns()}",
                 "phase": phase, "order": order, "repeat": repeat, "seconds": elapsed,
                 "cards": 0 if cpu_only else 1,
                 "gpu_hours": 0. if cpu_only else elapsed / 3600,
                 "cpu_wall_seconds": elapsed if cpu_only else None, "exit_code": exit_code}
        append_event(ledger, event)
        append_event(root / "COST_LEDGER.jsonl", event)
    if exit_code:
        raise RuntimeError(f"real {phase} failed; charged event {event['event_id']}")


def compare_orders(root):
    records = {f"{order}-{repeat}": torch.load(root / "runtime" / f"{order}-{repeat}.pt", weights_only=True)
               for order in ("A", "BA", "ABA") for repeat in (0, 1)}
    baseline = records["A-0"]
    rows = []
    for run, observed in records.items():
        changes = {}
        for stage in ("input", "mapping", "embedding", "encoder", "decoder", "maskF", "query", "final"):
            changes[stage] = {}
            for key, expected in baseline[stage].items():
                value = observed[stage][key]
                same = expected.shape == value.shape
                changes[stage][key] = {"equal": same and torch.equal(expected, value),
                                       "max_abs": float((expected.double() - value.double()).abs().max())
                                       if same and value.numel() else None}
        rows.append({"run": run, "differences_from_A0": changes})
    within_protocol = all(all(torch.equal(records[f"{order}-0"][stage][key],
                                          records[f"{order}-1"][stage][key])
                              for key in records[f"{order}-0"][stage])
                          for order in ("A", "BA", "ABA") for stage in ("input", "mapping", "final"))
    across_orders = all(value["equal"] for row in rows for stage in row["differences_from_A0"].values()
                        for value in stage.values())
    status = "REPRODUCIBLE" if across_orders else "RUNTIME_CONDITIONAL" if within_protocol else "BLOCKED_NATIVE_CORRECTNESS"
    result = {"status": status, "protocol": "per-input seed hash(9001,input_id); construction independent",
              "within_protocol_repeatable": within_protocol, "rows": rows,
              "deterministic_algorithms": "warn_only", "TF32": False, "precision": "FP32"}
    deep_paths = [root / "runtime" / f"deep-{order}-0.pt" for order in ("A", "BA")]
    if all(p.exists() for p in deep_paths):
        a, ba = (torch.load(p, weights_only=True) for p in deep_paths)
        differences = []
        for name in a:
            for key in a[name]:
                if name in ba and key in ba[name] and isinstance(a[name][key], torch.Tensor):
                    left, right = a[name][key], ba[name][key]
                    if left.shape == right.shape and not torch.equal(left, right):
                        differences.append({"module": name, "value": key,
                                            "max_abs": float((left.double() - right.double()).abs().max())})
        write_json(ARTIFACTS / "runtime/FIRST_DIFFERENCE.json", {
            "scope": "bounded original native deep A/BA probe", "first": differences[0],
            "differences": differences, "raw_trace_sha256": [sha256(p) for p in deep_paths]})
    write_json(ARTIFACTS / "runtime/ORDER_AUDIT.json", result)
    write_json(root / "runtime/ORDER_AUDIT.json", result)
    return result


def preflight_identity(root):
    code = code_identity()
    return {"numeric_sources": {k: v for k, v in code.items() if k.startswith(("models/", "datasets/"))},
            "entry_functions": {f.__name__: hashlib.sha256(inspect.getsource(f).encode()).hexdigest()
                                for f in (system_from_common, native_forward, load_samples, configure_numeric, order_check, feedback_check)},
            "resolved_config": OmegaConf.to_container(
                compose_config(PROJECT / "conf/config_native_long_retrain.yaml", root=root), resolve=True),
            "population_sha256": sha256(root / "POPULATION.json"),
            "data_content_sha256": sha256(root / "DATA_CONTENT_BINDING.json"),
            "libraries": read_json(root / "SOURCE_AND_INITIALIZATION.json")["installed_libraries"]}


def run_preflight(root, config, *, resume=False):
    proof_path = root / "PREFLIGHT_PROOF.json"
    identity = preflight_identity(root)
    if resume and proof_path.exists():
        proof = read_json(proof_path)
        if proof["identity"] == identity and all(sha256(Path(k)) == v for k, v in proof["evidence"].items()):
            assets = ARTIFACTS / "initialization/TASK_BUNDLE_RELOAD_AUDIT.json"
            if read_json(assets)["metadata"]["code"] != code_identity():
                run_child(root, "assets")
                proof["evidence"][str(assets)] = sha256(assets)
                write_json(proof_path, proof)
            print("PREFLIGHT_REUSED_CURRENT_IDENTITY", flush=True)
            return read_json(ARTIFACTS / "runtime/ORDER_AUDIT.json")
    if "45" not in read_json(root / "SOURCE_AND_INITIALIZATION.json")["initialization"]:
        run_child(root, "initialize")
    for order in ("A", "BA", "ABA"):
        for repeat in (0, 1):
            run_child(root, "order", order=order, repeat=repeat)
    result = compare_orders(root)
    run_child(root, "feedback")
    if not (root / "resources/COST_MEASUREMENTS.json").exists():
        run_child(root, "cost")
        run_child(root, "memory")
        run_child(root, "io")
    run_child(root, "runtime-metric")
    run_child(root, "assets")
    evidence = [ARTIFACTS / "runtime/ORDER_AUDIT.json", ARTIFACTS / "initialization/REAL_FEEDBACK_IDENTITY.json",
                ARTIFACTS / "initialization/TASK_BUNDLE_RELOAD_AUDIT.json", root / "resources/COST_MEASUREMENTS.json",
                root / "resources/IO_MEASUREMENTS.json", root / "resources/CHECKPOINT_MEMORY_PROBE.json"]
    write_json(proof_path, {"identity": preflight_identity(root), "evidence": {str(p): sha256(p) for p in evidence}})
    state = read_json(root / "RUN_STATE.json")
    state.update(stage="RUNTIME_CHECKED", native_runtime_status=result["status"])
    write_json(root / "RUN_STATE.json", state)
    write_json(ARTIFACTS / "RUN_STATE.json", state)
    print("RUNTIME_STATUS", result["status"], flush=True)


def feedback_check(root, config):
    configure_numeric()
    device = torch.device("cuda:0")
    base, _ = system_from_common(root, config)
    feedback_config = compose_config(PROJECT / "conf/config_native_long_retrain.yaml", root=root, arm="E3")
    construction_rng = torch.cuda.get_rng_state(device)
    feedback, audit = system_from_common(root, feedback_config)
    if not torch.equal(construction_rng, torch.cuda.get_rng_state(device)):
        raise ValueError("F construction changed the existing CUDA RNG stream")
    base.to(device).eval()
    feedback.to(device).eval()
    datasets = build_native_datasets(config, config.native_long.data_root, augmentation=False)
    collate = NativeLongCollator(config, training=False)
    records = read_json(root / "PREFLIGHT_INPUTS.json")
    results = {}
    for letter in ("A", "B"):
        before = native_forward(base, datasets, collate, records[letter], device=device)
        after = native_forward(feedback, datasets, collate, records[letter], device=device)
        equal = {k: torch.equal(before["final"][k], after["final"][k]) for k in before["final"]}
        if not all(equal.values()):
            raise ValueError("zero-initialized feedback changes the actual native final prediction")
        results[letter] = {"initial_final_equal": equal,
                            "actual_feature_update": feedback.model.native_long_feedback_stats}
    del base
    torch.cuda.empty_cache()
    block = feedback.model.native_long_feedback
    before = native_forward(feedback, datasets, collate, records["A"], device=device,
                            capture_aux=True)
    with isolated_rng(1945, cuda_devices=(device.index,)):
        torch.nn.init.normal_(block.output_projection.weight, std=.01)
        torch.nn.init.normal_(block.ffn[2].weight, std=.01)
    changed = native_forward(feedback, datasets, collate, records["A"], device=device,
                             capture_aux=True)
    aux_equal = [all(torch.equal(a[k], b[k]) for k in a)
                 for a, b in zip(before["aux"], changed["aux"], strict=True)]
    final_delta = float((changed["final"]["masks"] - before["final"]["masks"]).abs().max())
    if not all(aux_equal[:8]) or any(aux_equal[8:]) or final_delta <= 0:
        raise ValueError("feedback failed to update exactly the later native predictions")
    results["active"] = {"update": feedback.model.native_long_feedback_stats,
                         "aux_equal": aux_equal, "final_mask_max_abs": final_delta}
    write_json(ARTIFACTS / "initialization/REAL_FEEDBACK_IDENTITY.json", {
        "common_tensor_audit": audit, "CUDA_construction_rng_unchanged": True, "results": results})
    print("FEEDBACK_REAL_IDENTITY_PASS", flush=True)


def runtime_metric_audit(root, config):
    from scripts.evaluate_persist4d import _metric_target
    from scripts.evaluate_persist4d_p6a import build_rio_class_mapper
    from scripts.p6a_metrics import OfficialMetricAccumulator
    from scripts.rescene_task_postprocess import extract_official_task_prediction

    system, _ = system_from_common(root, config)
    datasets = build_native_datasets(config, config.native_long.data_root, augmentation=False)
    record = read_json(root / "PREFLIGHT_INPUTS.json")["A"]
    data, targets, _ = NativeLongCollator(config, training=False)(load_samples(datasets, [record]))
    target = _metric_target(data.target_full[0], datasets["rio"])
    mapper = build_rio_class_mapper(datasets["rio"])
    rows, predictions = [], {}
    for protocol in ("A", "BA", "ABA"):
        raw = torch.load(root / "runtime" / f"{protocol}-0.pt", weights_only=True)["final"]
        output = {"pred_logits": raw["classes"].clone(), "pred_masks": [raw["masks"].clone()], "aux_outputs": []}
        prediction = extract_official_task_prediction(
            system=system, output=output, target_low_resolution=targets[0],
            target_full_resolution=data.target_full[0], data=data, class_mapper=mapper,
            latest_stage_index=record["horizon"] - 1).prediction()
        accumulator = OfficialMetricAccumulator(mode="strict_online", dataset_spec=config.instance_metric.dataset)
        accumulator.update(prediction, target)
        raw_scores = accumulator._metric.compute()
        scores = {}
        for suffix, official_key in (("mAP", "AP"), ("mAP50", "AP_50"), ("mAP25", "AP_25")):
            value = float(raw_scores["val_mean_t-" + official_key])
            scores["online_t-" + suffix] = value if math.isfinite(value) else None
        predictions[protocol] = prediction
        path = ARTIFACTS / "evidence" / f"INITIALIZED_TRAIN_A_{protocol}_official_state.json"
        write_json(path, accumulator.export_evidence())
        rows.append({"protocol": protocol, "trained_updates": 0, "role": "TRAIN_DIAGNOSTIC",
                     "horizon": 2, "completed": 1, "expected": 1, "metrics": scores,
                     "eligible_GT_region_count": int((target["masks"].sum(1) >= 100).sum()),
                     "metric_status": "FINITE" if all(v is not None for v in scores.values()) else "UNDEFINED_OFFICIAL_METRIC",
                     "candidates": prediction["pred_scores"].numel(),
                     "evidence": {"path": str(path.relative_to(ARTIFACTS)), "sha256": sha256(path)}})
    differences = {}
    for protocol in ("BA", "ABA"):
        before, after = predictions["A"], predictions[protocol]
        same_lineage = torch.equal(before["source_query_ids"], after["source_query_ids"]) and torch.equal(
            before["source_class_ids"], after["source_class_ids"])
        differences[protocol] = {"candidate_lineage_equal": same_lineage,
            "changed_bool_mask_entries": int((before["pred_masks"] != after["pred_masks"]).sum())
            if same_lineage and before["pred_masks"].shape == after["pred_masks"].shape else None,
            "score_max_abs": float((before["pred_scores"] - after["pred_scores"]).abs().max())
            if same_lineage and before["pred_scores"].numel() else None,
            "single_input_t_AP_delta": rows[("A", "BA", "ABA").index(protocol)]["metrics"]["online_t-mAP"]
            - rows[0]["metrics"]["online_t-mAP"] if rows[0]["metrics"]["online_t-mAP"] is not None
            and rows[("A", "BA", "ABA").index(protocol)]["metrics"]["online_t-mAP"] is not None else None}
    write_json(ARTIFACTS / "runtime/INITIALIZED_RUNTIME_METRIC_AUDIT.json", {
        "status": "DIAGNOSTIC_ONLY", "gain_assessment": "NO_TRAINED_CANDIDATE",
        "scope": "one fixed TRAIN T2 at initialization; not CAL/SEL/confirmation", "rows": rows,
        "differences_from_A": differences})
    feedback_config = compose_config(PROJECT / "conf/config_native_long_retrain.yaml", root=root, arm="E3")
    feedback, _ = system_from_common(root, feedback_config)
    path = ARTIFACTS / "initialization/feedback_initial_state.pt"
    torch.save({"state_dict": feedback.model.native_long_feedback.state_dict(),
                "metadata": {"seed": 45, "independent_stream_seed": 1945, "trained_updates": 0,
                             "role": "INITIALIZATION_ONLY", "common_sha256": read_json(
                                 root / "SOURCE_AND_INITIALIZATION.json")["initialization"]["45"]["sha256"]}}, path)


def training_measurement(system, batch, *, draw_seed, optimizer, scheduler, raw_sum_audit=False):
    from trainer.trainer import _configured_objective_loss

    device = next(system.parameters()).device
    data, targets, _ = move_batch(batch, device)
    optimizer.zero_grad(set_to_none=True)
    system.train()
    system.model.native_long_draw_ids = [draw_seed + i for i in range(len(targets))]
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    with isolated_rng(stable_seed(45, "measurement", draw_seed), cuda_devices=(device.index,)):
        output = system(data, point2segment=[t["point2segment"] for t in targets],
                        raw_coordinates=system._process_raw_coordinates(data))
        if len(output["aux_outputs"]) != 12:
            raise ValueError("training forward changed auxiliary output count")
        losses = system.criterion(output, targets, mask_type=system.mask_type)
        total = _configured_objective_loss(system, losses)
        audit = None
        if raw_sum_audit:
            explicit = sum(losses.values())
            parameter = system.model.mask_embed_head[0].weight
            reduced_gradient = torch.autograd.grad(total, parameter, retain_graph=True)[0]
            explicit_gradient = torch.autograd.grad(explicit, parameter, retain_graph=True)[0]
            if not torch.equal(total, explicit) or not torch.equal(reduced_gradient, explicit_gradient):
                raise ValueError("configured raw_sum scalar/gradient differs from explicit original sum")
            audit = {"keys": {k: {"coefficient": 1., "value": float(v.detach())} for k, v in losses.items()},
                     "total": float(total.detach()), "gradient_equal": True,
                     "aux_outputs": 12, "segment_contrastive_inputs": len(output["segment_features"])}
        total.backward()
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in system.parameters()):
        raise ValueError("real native backward produced nonfinite parameter gradients")
    torch.nn.utils.clip_grad_norm_([p for p in system.parameters() if p.requires_grad],
                                  float(system.config.trainer.gradient_clip_val))
    gradients = {name: float(p.grad.detach().norm()) for name, p in system.named_parameters()
                 if p.grad is not None and (name.startswith("model.native_long_feedback.")
                                            or name == "model.mask_embed_head.0.weight")}
    optimizer.step()
    scheduler.step()
    torch.cuda.synchronize(device)
    result = {"seconds": time.perf_counter() - started,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
              "loss": float(total.detach()), "gradients": gradients,
              "sampling": system.model.native_long_sampling_stats,
              "feature_update": system.model.native_long_feedback_stats,
              "low_points": int(data.feat.shape[0]), "raw_sum_audit": audit}
    return result


def select_cost_inputs(datasets, train):
    cost_inputs, point_distributions = {}, {}
    for h in range(1, 6):
        if h == 1:
            ordered = sorted(range(len(datasets["scannet"].data)),
                             key=lambda i: datasets["scannet"].data[i]["file_len"])
            index = ordered[len(ordered) // 2]
            record = {"input_id": f"COST:SCANNET:{index}", "domain": "scannet", "horizon": 1,
                      "reference_id": f"scannet:{index}", "scan_ids": [f"scannet:{index}"], "scan_indices": [index]}
            sizes = [r["file_len"] for r in datasets["scannet"].data]
        else:
            available = [r for r in train if r["Tmax"] >= h]
            expected_size = lambda r, horizon=h: horizon * np.mean(
                [datasets["rio"].data[i]["file_len"] for i in r["scan_indices"]])
            available.sort(key=lambda r: (expected_size(r), r["reference_id"]))
            ref = available[len(available) // 2]
            positions = canonical_orders(ref)[0][:h]
            record = {"input_id": f"COST:{ref['reference_id']}:T{h}", "domain": "rio", "horizon": h,
                      "reference_id": ref["reference_id"], "scan_ids": [ref["scan_ids"][i] for i in positions],
                      "scan_indices": [ref["scan_indices"][i] for i in positions]}
            sizes = [expected_size(r) for r in available]
        cost_inputs[h] = record
        point_distributions[h] = {"eligible_references_or_scans": len(sizes),
                                  "mean_full_points": float(np.mean(sizes)),
                                  "median_full_points": float(np.median(sizes)),
                                  "p95_full_points": float(np.percentile(sizes, 95))}
    return cost_inputs, point_distributions


def io_measurements(root, config):
    datasets = build_native_datasets(config, config.native_long.data_root, augmentation=True)
    train = [r for r in read_json(root / "POPULATION.json")["references"] if r["role"] == "TRAIN"]
    inputs, _ = select_cost_inputs(datasets, train)
    physical_batch = read_json(root / "resources/COST_MEASUREMENTS.json")["per_rank_batch"]
    collate = NativeLongCollator(config, training=True)
    rows = []
    for h, record in inputs.items():
        for repeat in range(2):
            started = time.perf_counter()
            samples = load_samples(datasets, [record] * physical_batch)
            loaded = time.perf_counter()
            batch = collate(samples)
            finished = time.perf_counter()
            rows.append({"horizon": h, "repeat": repeat, "input": record,
                         "per_rank_batch": physical_batch, "full_points": sum(len(s[0]) for s in samples),
                         "load_seconds": loaded - started, "collate_seconds": finished - loaded,
                         "total_seconds": finished - started})
            del batch, samples
    payload = {"rows": rows, "workers": 0, "mean_scan_points_by_train_reference": {
        ref["reference_id"]: float(np.mean([datasets["rio"].data[i]["file_len"]
                                            for i in ref["scan_indices"]])) for ref in train}}
    write_json(root / "resources/IO_MEASUREMENTS.json", payload)
    write_json(ARTIFACTS / "resources/IO_MEASUREMENTS.json", payload)


def memory_checkpoint_probe(root, config):
    from omegaconf import open_dict

    configure_numeric()
    cfg = compose_config(PROJECT / "conf/config_native_long_retrain.yaml", root=root, arm="E3")
    with open_dict(cfg):
        cfg.native_long.feedback_checkpoint = True
    system, _ = system_from_common(root, cfg)
    system.to("cuda:0")
    datasets = build_native_datasets(cfg, cfg.native_long.data_root, augmentation=True)
    batch = NativeLongCollator(cfg, training=True)(load_samples(
        datasets, [read_json(root / "PREFLIGHT_INPUTS.json")["B"]] * 2))
    optimizers, schedules = system.configure_optimizers()
    try:
        result = training_measurement(system, batch, draw_seed=1, optimizer=optimizers[0],
                                      scheduler=schedules[0]["scheduler"])
        result.update(status="PASS", per_rank_batch=2)
    except torch.OutOfMemoryError as error:
        result = {"status": "OOM", "per_rank_batch": 2, "error": str(error)}
    result["exact_memory_strategy"] = "non-reentrant checkpoint for F chunk512, full Q100; unrelated models released"
    write_json(ARTIFACTS / "resources/CHECKPOINT_MEMORY_PROBE.json", result)
    write_json(root / "resources/CHECKPOINT_MEMORY_PROBE.json", result)


def cost_measurements(root, config):
    configure_numeric()
    device = torch.device("cuda:0")
    datasets = build_native_datasets(config, config.native_long.data_root, augmentation=True)
    population = read_json(root / "POPULATION.json")
    train = [r for r in population["references"] if r["role"] == "TRAIN"]
    records = read_json(root / "PREFLIGHT_INPUTS.json")
    cost_inputs, point_distributions = select_cost_inputs(datasets, train)
    collate = NativeLongCollator(config, training=True)
    cfg_f = compose_config(PROJECT / "conf/config_native_long_retrain.yaml", root=root, arm="E3")
    f_system, _ = system_from_common(root, cfg_f)
    f_system.to(device)
    optimizers, schedules = f_system.configure_optimizers()
    optimizer, scheduler = optimizers[0], schedules[0]["scheduler"]
    # Large, complete T5 backward is the physical-batch gate for every arm.
    physical_batch = 2
    feasibility = []
    for attempt_batch in (2, 1):
        batch = collate(load_samples(datasets, [records["B"]] * attempt_batch))
        try:
            result = training_measurement(f_system, batch, draw_seed=1, optimizer=optimizer,
                                          scheduler=scheduler)
            feasibility.append({"per_rank_batch": attempt_batch, "status": "PASS", **result})
            physical_batch = attempt_batch
            break
        except torch.OutOfMemoryError as error:
            feasibility.append({"per_rank_batch": attempt_batch, "status": "OOM", "error": str(error)})
            optimizer.zero_grad(set_to_none=True)
            del batch
            import gc

            gc.collect()
            torch.cuda.empty_cache()
    else:
        write_json(ARTIFACTS / "resources/COST_MEASUREMENTS.json", {"status": "BLOCKED_PHYSICAL_BATCH", "feasibility": feasibility})
        raise RuntimeError("complete largest T5 cannot backward even at batch1")
    measured = []
    for arm, horizons in (("E1", range(1, 6)), ("E2", (5,)), ("E3", (2, 5))):
        del f_system, optimizer, scheduler, optimizers, schedules
        torch.cuda.empty_cache()
        cfg = compose_config(PROJECT / "conf/config_native_long_retrain.yaml", root=root, arm=arm)
        f_system, _ = system_from_common(root, cfg)
        f_system.to(device)
        optimizers, schedules = f_system.configure_optimizers()
        optimizer, scheduler = optimizers[0], schedules[0]["scheduler"]
        for h in horizons:
            samples = load_samples(datasets, [cost_inputs[h]] * physical_batch)
            batch = collate(samples)
            full_points = sum(len(sample[0]) for sample in samples)
            for repeat in range(2):
                result = training_measurement(f_system, batch, draw_seed=100 * h + repeat,
                                              optimizer=optimizer, scheduler=scheduler,
                                              raw_sum_audit=arm == "E1" and h == 2 and repeat == 0)
                measured.append({"arm": arm, "horizon": h, "repeat": repeat,
                                 "full_points": full_points, "per_rank_batch": physical_batch, **result})
                print("MEASURED", arm, h, repeat, round(result["seconds"], 4), flush=True)
    payload = {"status": "MEASURED", "temporary_updates_not_training": True,
               "per_rank_batch": physical_batch, "world_size_planned": 2,
               "accumulation": 16 if physical_batch == 1 else 8,
               "feasibility": feasibility, "measurements": measured,
               "point_distributions": point_distributions,
               "forecast_limit": "finite median inputs with metadata point-count scaling; real trajectory error must be reconciled"}
    write_json(root / "resources/COST_MEASUREMENTS.json", payload)
    write_json(ARTIFACTS / "resources/COST_MEASUREMENTS.json", payload)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("initialize", "order", "deep", "native", "feedback", "cost", "io", "memory", "runtime-metric", "assets"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--order", default="A")
    parser.add_argument("--repeat", type=int, default=0)
    args = parser.parse_args()
    config = compose_config(PROJECT / "conf/config_native_long_retrain.yaml", root=args.root)
    torch.set_num_threads(2)
    if args.phase == "initialize":
        initialize(args.root, config)
    elif args.phase == "feedback":
        feedback_check(args.root, config)
    elif args.phase == "cost":
        cost_measurements(args.root, config)
    elif args.phase == "io":
        io_measurements(args.root, config)
    elif args.phase == "memory":
        memory_checkpoint_probe(args.root, config)
    elif args.phase == "runtime-metric":
        runtime_metric_audit(args.root, config)
    elif args.phase == "assets":
        from scripts.native_long_assets import audit_initialization_reload

        audit_initialization_reload(args.root, config)
    else:
        order_check(args.root, config, args.order, args.repeat, deep=args.phase == "deep",
                    native_sparse=args.phase == "native")


if __name__ == "__main__":
    main()
