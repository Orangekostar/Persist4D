"""Real full-forward single-module deployment profiling and portable head bundles."""

import hashlib
import resource
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from models.perception_gain import derive_segment_stage_ids
from models.short_module_heads import apply_module, describe_candidates, quality_inputs
from scripts.rescene_task_postprocess import (
    extract_official_task_prediction,
    materialize_segment_logits,
)
from scripts.short_module_evaluation import ARMS, load_head, write_csv
from scripts.short_module_native import NativeSession
from scripts.short_module_screen import (
    ARTIFACTS,
    build_datasets,
    native_config,
    read_json,
    sha256,
    write_json,
)


def process_resources() -> dict:
    """Read Linux process RSS and physical disk IO without an extra dependency."""
    io = {key: int(value.strip()) for key, value in
          (line.split(":", 1) for line in Path("/proc/self/io").read_text().splitlines())}
    status = dict(line.split(":", 1) for line in Path("/proc/self/status").read_text().splitlines())
    return {"read_bytes": io["read_bytes"], "write_bytes": io["write_bytes"],
            "rss_bytes": int(status["VmRSS"].split()[0]) * 1024}


def profile(root: Path, *, device: str) -> dict:
    from scripts.evaluate_persist4d import _move_data_to_device, _move_targets_to_device
    from scripts.evaluate_persist4d_p6a import (
        _frozen_inference_seed,
        build_rio_class_mapper,
    )
    from scripts.system_comparison_inference import deterministic_inference_runtime

    assets = read_json(root / "assets.local.json")
    config = native_config(root, assets)
    session = NativeSession(config=config, datasets=build_datasets(config, assets), assets=assets, device=device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    lock = read_json(ARTIFACTS / "selection/CAL_LOCK.json")
    population = read_json(ARTIFACTS / "SHORT_POPULATION.json")
    groups = {}
    for record in population["records"]:
        if record["role"] == "SEL" and record["horizon"] == 2:
            groups.setdefault(record["reference_id"], []).append(record)
    records = [min(group, key=lambda r: (hashlib.sha256(
        ("short-module-v1:" + "-".join(r["scan_ids"])).encode()).hexdigest(), r["logical_unit_id"]))
        for _, group in sorted(groups.items())]
    if len(records) != 4:
        raise ValueError("profile requires the four actual SEL references")
    rows, reload_checks = [], []
    for module in ("B0", *ARMS):
        step = 0 if module == "B0" else lock["selected_updates"][module]
        head, path = load_head(root, module, step, 45, device)
        bundle_path = None
        if head is not None:
            saved = torch.load(path, map_location="cpu", weights_only=False)
            bundle_path = ARTIFACTS / "training" / module / "deployment.pt"
            torch.save({"head": {k: v.detach().cpu() for k, v in head.state_dict().items()},
                        "metadata": saved["metadata"], "slot": "score_only" if module.startswith("Q") else "mask_only",
                        "module_contract": read_json(ARTIFACTS / "RUN_CONFIG.json")["protocol"]["modules"][module],
                        "label_version": saved["metadata"]["cache_identity_sha256"],
                        "inference_api": "models.short_module_heads.apply_module"}, bundle_path)
            if bundle_path.stat().st_size > 20 * 1024**2:
                raise ValueError("small head unexpectedly exceeds Git size limit")
        for record_index, record in enumerate(records):
            dataset = session.datasets[record["dataset_horizon"]]
            mapper = build_rio_class_mapper(dataset)
            for repeat in range(4):
                torch.cuda.synchronize(session.device)
                torch.cuda.reset_peak_memory_stats(session.device)
                io_before = process_resources()
                with deterministic_inference_runtime(45, session.device), _frozen_inference_seed(45, session.device):
                    begin = time.perf_counter()
                    sample = dataset.load_scan_indices(record["context_index"], tuple(record["scan_indices"]), change_file=None)
                    original_xyz = torch.from_numpy(sample[6][:, :3].copy()).float() if head is not None else None
                    data, targets, _ = session.collate([sample])
                    target_full = data.target_full[0]
                    prepared = time.perf_counter()
                    data = _move_data_to_device(data, session.device)
                    targets = _move_targets_to_device(targets, session.device)
                    low = targets[0]
                    raw = session.system._process_raw_coordinates(data)
                    torch.cuda.synchronize(session.device)
                    moved = time.perf_counter()
                    with torch.inference_mode():
                        output = session.system(data, point2segment=[low["point2segment"]], raw_coordinates=raw, is_eval=True)
                    torch.cuda.synchronize(session.device)
                    forwarded = time.perf_counter()
                    parent = extract_official_task_prediction(
                        system=session.system, output=output, target_low_resolution=low,
                        target_full_resolution=target_full, data=data, class_mapper=mapper,
                        latest_stage_index=1, return_soft_evidence=head is not None)
                    postprocessed = time.perf_counter()
                    descriptor = None
                    if head is not None:
                        soft = parent.soft_evidence
                        stages = derive_segment_stage_ids(soft.low_point2segment, low["temporal_stages"].cpu().long())
                        mapping = soft.low_point2segment[soft.voxel_inverse]
                        centroids = torch.zeros(soft.segment_features.shape[0], 3)
                        centroids.index_add_(0, mapping, original_xyz)
                        centroids /= torch.bincount(mapping, minlength=centroids.shape[0])[:, None].clamp_min(1)
                        keys = tuple(f"{record['scan_ids'][int(stage)]}:{i}" for i, stage in enumerate(stages))
                        h, support, probability = describe_candidates(soft.segment_features, soft.segment_logits,
                                                                      centroids, stages, keys, horizon=2)
                        classes = soft.class_probabilities.shape[1] - 1
                        descriptor = {"h": h, "support": support, "foreground_probability": probability,
                                      "segment_stages": stages, "classes": classes}
                        q = soft.query_features.to(device)
                        h_device = h.to(device)
                        one_hot = F.one_hot(soft.source_class_ids.to(device), classes).float()
                        if module.startswith("Q"):
                            x = quality_inputs(q, h_device, soft.class_probabilities.to(device), one_hot,
                                               parent.pred_scores.to(device), support.to(device), probability.to(device))
                        else:
                            features = soft.segment_features.to(device)
                            stage_device = stages.to(device)
                    torch.cuda.synchronize(session.device)
                    described = time.perf_counter()
                    prediction = parent.prediction()
                    if head is not None:
                        with torch.no_grad():
                            if module.startswith("Q"):
                                prediction["pred_scores"] = head(x).cpu()
                            else:
                                delta = head(features, q, h_device, one_hot, stage_device).cpu()
                    torch.cuda.synchronize(session.device)
                    headed = time.perf_counter()
                    if module.startswith("M"):
                        prediction["pred_masks"] = materialize_segment_logits(
                            system=session.system, evidence=soft, segment_logits=soft.segment_logits + delta)
                    torch.cuda.synchronize(session.device)
                    finished = time.perf_counter()
                io_after = process_resources()
                row = {"module": module, "selected_step": step, "reference_id": record["reference_id"],
                       "input_id": record["input_id"], "repeat": repeat, "warmup": repeat == 0,
                       "input_seconds": prepared - begin, "H2D_seconds": moved - prepared,
                       "parent_network_seconds": forwarded - moved,
                       "parent_postprocess_seconds": postprocessed - forwarded,
                       "descriptor_seconds": described - postprocessed, "head_seconds": headed - described,
                       "materialize_seconds": finished - headed, "end_to_end_seconds": finished - begin,
                       "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(session.device),
                       "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(session.device),
                       "cpu_rss_bytes": io_after["rss_bytes"],
                       "cpu_process_highwater_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                       "disk_read_bytes": io_after["read_bytes"] - io_before["read_bytes"],
                       "disk_write_bytes": io_after["write_bytes"] - io_before["write_bytes"],
                       "device": torch.cuda.get_device_name(session.device)}
                rows.append(row)
                if head is not None and record_index == 0 and repeat == 1:
                    # Deliberately outside every timed interval: genuine bundle reload.
                    from models.short_module_heads import build_head
                    bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
                    meta = bundle["metadata"]
                    restored = build_head(module, **meta["dimensions"], thresholds=tuple(meta["thresholds"]), seed=45)
                    restored.load_state_dict(bundle["head"], strict=True)
                    restored.to(device).eval()
                    expected = apply_module(module, head, parent, descriptor, system=session.system)
                    actual = apply_module(module, restored, parent, descriptor, system=session.system)
                    equality = {k: torch.equal(expected[k], actual[k]) and torch.equal(expected[k], prediction[k])
                                for k in expected}
                    if not all(equality.values()):
                        raise ValueError(f"real-input bundle reload/profile output mismatch: {module}")
                    reload_checks.append({"module": module, "input_id": record["input_id"],
                                          "bundle_sha256": sha256(bundle_path), "equal": equality,
                                          "bundle_bytes": bundle_path.stat().st_size})
                    del restored
                # Do not retain a previous forward's live CUDA tensors in the
                # next measurement's allocated-memory baseline.
                del output, data, targets, low, raw
                if head is not None:
                    del q, h_device, one_hot
                    if module.startswith("Q"):
                        del x
                    else:
                        del features, stage_device, delta
        del head
        torch.cuda.empty_cache()
    write_csv(ARTIFACTS / "resources/PROFILE.csv", rows)
    result = {"status": "COMPLETE", "selected_records": records, "measurement_rows": len(rows),
              "warmup_rows": sum(row["warmup"] for row in rows), "reload_checks": reload_checks,
              "timing_scope": "Each method executes the full parent anew. Metrics, hashes and reload checks excluded. Disk IO reported separately."}
    write_json(ARTIFACTS / "resources/PROFILE_SUMMARY.json", result)
    return result
