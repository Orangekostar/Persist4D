"""Post-training repair diagnostics from TRAIN features, with no parent forward."""

from __future__ import annotations

from collections import Counter
from contextlib import nullcontext
from itertools import groupby
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

from scripts.perception_gain_v2 import (
    PROJECT_ROOT,
    _free_gpus,
    append_event,
    file_hash,
    load_config,
    read_json,
    utc_now,
    write_json,
)
from scripts.perception_gain_v2_config import executed_identity


def reconstruct_cached_target(
    target, *, source_class_id: int, gt_masks, gt_classes, full_segment_ids
) -> dict:
    import torch

    if not target.any():
        return {"status": "EMPTY_TARGET", "gt_index": None}
    count = target.numel()
    if (
        full_segment_ids.numel() != gt_masks.shape[1]
        or int(full_segment_ids.max()) >= count
    ):
        raise ValueError("Reconstructed TRAIN partition differs from cached segments")
    totals = torch.zeros((gt_masks.shape[0], count), dtype=torch.float32)
    totals.scatter_add_(
        1, full_segment_ids.expand(gt_masks.shape[0], -1), gt_masks.float()
    )
    counts = torch.bincount(full_segment_ids, minlength=count).float()
    if (counts == 0).any():
        raise ValueError("Reconstructed TRAIN segment contains no vertices")
    fractions = totals / counts[None]
    matches = torch.nonzero(
        (fractions == target[None]).all(1) & (gt_classes == source_class_id),
        as_tuple=False,
    ).flatten()
    return {
        "status": (
            "UNIQUE_EXACT"
            if len(matches) == 1
            else "AMBIGUOUS_EXACT" if len(matches) > 1 else "NO_EXACT_MATCH"
        ),
        "gt_index": int(matches[0]) if len(matches) == 1 else None,
    }


def repair_segment_counts(original, refined, target) -> dict:
    before, after = (original > 0) != (target > 0.5), (refined > 0) != (target > 0.5)
    changed = (original > 0) != (refined > 0)
    return {
        "segments": original.numel(),
        "original_errors": int(before.sum()),
        "refined_errors": int(after.sum()),
        "corrected_original_errors": int((before & ~after).sum()),
        "broken_original_correct": int((~before & after).sum()),
        "changed_strictly_outside_bound": int((changed & (original.abs() > 2)).sum()),
        "changed_at_exact_abs2": int((changed & (original.abs() == 2)).sum()),
    }


def run_diagnostics(
    config: dict, *, external_root: Path, parent_id: str, device_name: str
) -> dict:
    reserved_started = time.monotonic()
    import hydra
    import torch
    from datasets.task_memory_episode import (
        TaskMemoryEpisodeDataset,
        TaskMemoryEpisodeSpec,
        TaskMemoryEpisodeCollator,
    )
    from scripts.evaluate_task_memory import _ProtocolOrderDataset
    from scripts.p6a_metrics import official_temporal_iou_thresholds
    from scripts.perception_gain_v2_evidence import validate_refiner_cache
    from scripts.perception_gain_v2_perception import budget_decision
    from scripts.perception_refiner_evaluation import _load_refiner
    from scripts.preflight_task_memory_episode import _rio_base_dataset
    from scripts.run_task_memory_policy_baseline import NativeEpisodeMaster
    from scripts.train_perception_gain import compose_variant_config
    from scripts.system_comparison_inference import deterministic_inference_runtime
    from trainer.trainer import InstanceSegmentation

    artifacts = PROJECT_ROOT / config["artifact_root"]
    public = artifacts / f"refiner/{parent_id}"
    manifest = read_json(public / "DATA_MANIFEST.json")
    recipe = manifest["recipe"]
    cache_root = external_root / f"cache/refiner/{parent_id}"
    validate_refiner_cache(
        manifest,
        recipe=recipe,
        parent_weight=manifest["checkpoint_sha256"],
        artifacts=artifacts,
        cache_root=cache_root,
        required_inventory=manifest["inventory"],
        external_root=external_root,
    )
    assets = read_json(external_root / "assets.local.json")
    cfg = compose_variant_config(
        recipe["architecture_variant"],
        pretrained=Path(assets["concerto_pretrained"]),
        run_dir=cache_root.parent / "feature_producer",
        recipe_config=recipe,
    )
    rows = manifest["inventory"]["episodes"]
    wrapped = {}
    for horizon in sorted({len(row["scan_ids"]) for row in rows}):
        base = _rio_base_dataset(
            cfg, data_root=Path(assets["data_root"]), horizon=horizon
        )
        wrapped[horizon] = _ProtocolOrderDataset(
            base,
            sequence_names=[row["sequence_id"] for row in rows],
            sequence_indices=[row["scan_indices"] for row in rows],
            source_context_indices=[row["source_context_index"] for row in rows],
        )
    specs = {}
    for index, row in enumerate(rows):
        master = NativeEpisodeMaster(
            reference_id=row["reference_id"],
            sequence_id=row["sequence_id"],
            scan_ids=tuple(row["scan_ids"]),
            scan_indices=tuple(row["scan_indices"]),
            role="TRAIN",
            context_index=index,
        )
        spec = TaskMemoryEpisodeSpec.from_master(
            master,
            horizon=len(master.scan_ids),
            augmentation_seed=45,
            draw_index=index,
            bucket=f"T{len(master.scan_ids)}",
        )
        specs[spec.episode_id] = spec
    collator = TaskMemoryEpisodeCollator(
        hydra.utils.instantiate(cfg.data.validation_collation)
    )
    device = torch.device(device_name)
    heads, head_identity = {}, {}
    for label in ("NEW", "PAIR"):
        training = read_json(public / f"{label}/TRAINING.json")
        checkpoint = (
            external_root / f"training/refiner/{parent_id}/{label}-s45/update=1500.ckpt"
        )
        heads[label], _ = _load_refiner(
            checkpoint,
            optimizer_update=1500,
            device=device,
            expected_binding=training["binding"],
        )
        head_identity[label] = {
            "sha256": file_hash(checkpoint),
            "input_mode": training["binding"]["input_mode"],
            "optimizer_update": 1500,
        }
    binding = {
        "cache_manifest_sha256": file_hash(public / "DATA_MANIFEST.json"),
        "heads": head_identity,
        "source_sha256": file_hash(Path(__file__)),
        "device_type": device.type,
        "input_manifest_sha256": file_hash(artifacts / "data/STAGING_MANIFEST.json"),
    }
    destination = public / "POST_TRAIN_DIAGNOSTICS.json"
    if destination.exists():
        saved = read_json(destination)
        if saved.get("binding") != binding:
            raise ValueError(
                "Post-training diagnostics changed; preserve the previous attempt before recovery"
            )
        return saved
    taus = tuple(
        float(value)
        for value in official_temporal_iou_thresholds(
            dataset_spec=assets["metric_dataset_spec"]
        )
    )
    counts = {label: Counter() for label in heads}
    transitions = {label: {str(tau): Counter() for tau in taus} for label in heads}
    reconstructions = Counter()
    completed, failures, partition_hashes = [], [], {}
    total_records = 0
    mapper = SimpleNamespace(eval_on_segments=bool(cfg.general.eval_on_segments))

    def records():
        for shard in manifest["shards"]:
            payload = torch.load(
                cache_root / shard["path"], map_location="cpu", weights_only=False
            )
            yield from payload

    runtime = (
        deterministic_inference_runtime(45, device)
        if device.type == "cuda"
        else nullcontext()
    )
    with runtime, torch.inference_mode():
        for episode_id, group in groupby(records(), key=lambda row: row["episode_id"]):
            if device.type == "cuda":
                budget = budget_decision(
                    config,
                    external_root=external_root,
                    label=f"post-train-diagnostics:{parent_id}",
                    predicted_gpu_hours=(time.monotonic() - reserved_started) / 3600
                    + 0.01,
                    category="refinement",
                )
                if not budget["affordable"]:
                    failures.append(
                        {
                            "episode_id": episode_id,
                            "reason": "GPU_BUDGET_OR_PROTECTED_RESERVE",
                            "budget": budget,
                        }
                    )
                    break
            if episode_id in completed or episode_id not in specs:
                raise ValueError(
                    "TRAIN cache episode grouping differs from the fixed inventory"
                )
            episode_records = list(group)
            spec = specs[episode_id]
            try:
                episode = TaskMemoryEpisodeDataset(
                    wrapped[spec.horizon], (spec,), apply_augmentation=False
                )[0]
                batch = collator([episode])
                for stage in batch.stage_batches[1:]:
                    data, _, _ = stage.model_batch
                    meta = stage.stage_meta[0]
                    stop = int(meta.scan_vertex_offsets[1])
                    segment_ids = torch.nonzero(
                        meta.segment_stage_ids == 0, as_tuple=False
                    ).flatten()
                    full_partition_ids = meta.full_resolution_point2segment[:stop]
                    low_full_partition = torch.searchsorted(
                        segment_ids, full_partition_ids
                    )
                    if (
                        low_full_partition >= len(segment_ids)
                    ).any() or not torch.equal(
                        segment_ids[low_full_partition], full_partition_ids
                    ):
                        raise ValueError(
                            "Reconstructed previous-scan vertices cross the segment boundary"
                        )
                    target = data.target_full[0]
                    evaluation_partition = target["point2segment"].long().cpu()
                    if set(evaluation_partition[:stop].tolist()) & set(
                        evaluation_partition[stop:].tolist()
                    ):
                        raise ValueError(
                            "Full-resolution majority segments cross scan boundaries"
                        )
                    gt_masks = target["masks"][:, :stop].bool().cpu()
                    gt_classes = target["labels"].long().cpu()
                    digest = hashlib.sha256()
                    for value in (
                        meta.point2segment,
                        meta.voxel_inverse,
                        evaluation_partition,
                    ):
                        digest.update(value.cpu().numpy().tobytes())
                    partition_hashes[f"{episode_id}:{meta.absolute_stage_index}"] = (
                        digest.hexdigest()
                    )
                    for record in (
                        value
                        for value in episode_records
                        if value["absolute_stage_index"] == meta.absolute_stage_index
                    ):
                        if (
                            record["reference_id"] != spec.reference_id
                            or record["scan_id"] != meta.scan_ids_in_window[0]
                            or record["new_logits"].numel() != len(segment_ids)
                        ):
                            raise ValueError(
                                "Cached candidate lineage differs from the reconstructed stage"
                            )
                        reconstruction = reconstruct_cached_target(
                            record["target"],
                            source_class_id=record["source_class_id"],
                            gt_masks=gt_masks,
                            gt_classes=gt_classes,
                            full_segment_ids=low_full_partition,
                        )
                        reconstructions[reconstruction["status"]] += 1
                        if reconstruction["status"] == "NO_EXACT_MATCH":
                            raise ValueError(
                                "Cached target occupancy does not exactly reproduce the original TRAIN labels"
                            )
                        truth = (
                            torch.zeros(stop, dtype=torch.bool)
                            if reconstruction["status"] == "EMPTY_TARGET"
                            else (
                                gt_masks[reconstruction["gt_index"]]
                                if reconstruction["gt_index"] is not None
                                else None
                            )
                        )

                        def materialize(logits):
                            all_logits = torch.zeros(
                                (meta.segment_stage_ids.numel(), 1), dtype=torch.float32
                            )
                            all_logits[segment_ids, 0] = logits.cpu()
                            low_masks = (all_logits[meta.point2segment] > 0).float()
                            return InstanceSegmentation._get_full_res_mask(
                                mapper,
                                low_masks,
                                meta.voxel_inverse,
                                evaluation_partition,
                            )[:stop, 0].bool()

                        original_mask = materialize(record["new_logits"])
                        original_iou = (
                            float((original_mask & truth).sum())
                            / max(1, int((original_mask | truth).sum()))
                            if truth is not None
                            else None
                        )
                        for label, head in heads.items():
                            refined, _ = head(
                                new_features=record["new_features"].to(device),
                                old_logits=record["old_logits"].to(device),
                                new_logits=record["new_logits"].to(device),
                                old_score=record["old_score"],
                                new_score=record["new_score"],
                            )
                            refined = refined.cpu()
                            if not torch.isfinite(refined).all():
                                raise ValueError(
                                    "Frozen diagnostic head produced non-finite logits"
                                )
                            values = repair_segment_counts(
                                record["new_logits"], refined, record["target"]
                            )
                            if values["changed_strictly_outside_bound"]:
                                raise ValueError(
                                    "Bounded repair changed a segment strictly outside abs(logit)>2"
                                )
                            counts[label].update(values)
                            if truth is not None:
                                repaired_mask = materialize(refined)
                                repaired_iou = float(
                                    (repaired_mask & truth).sum()
                                ) / max(1, int((repaired_mask | truth).sum()))
                                for tau in taus:
                                    before, after = (
                                        original_iou > tau,
                                        repaired_iou > tau,
                                    )
                                    transitions[label][str(tau)][
                                        f"{int(before)}_to_{int(after)}"
                                    ] += 1
                        total_records += 1
                completed.append(episode_id)
                print(
                    json.dumps(
                        {
                            "diagnostic_episodes": len(completed),
                            "candidate_records": total_records,
                            "parent_network_forward_count": 0,
                        }
                    ),
                    flush=True,
                )
            except (OSError, RuntimeError, ValueError, KeyError, IndexError) as error:
                failures.append({"episode_id": episode_id, "reason": str(error)})
    for label, values in counts.items():
        if (
            values["segments"] != manifest["residual_diagnostics"]["segments"]
            or values["original_errors"]
            != manifest["residual_diagnostics"]["wrong_segments"]
        ):
            failures.append(
                {
                    "head": label,
                    "reason": "Post-training segment population differs from original cache diagnostics",
                }
            )
    result = {
        "status": (
            "PASS"
            if not failures and total_records == manifest["candidate_count"]
            else "PARTIAL"
        ),
        "binding": binding,
        "data_role": "TRAIN",
        "scope": "Frozen 1500-update heads on original cached features; full-resolution single-scan candidate IoU uses the original majority-mask function. These are diagnostics, not pooled temporal AP or selection results.",
        "parent_network_forward_count": 0,
        "candidate_record_count": total_records,
        "expected_candidate_record_count": manifest["candidate_count"],
        "head_counts": {label: dict(value) for label, value in counts.items()},
        "target_reconstruction": dict(reconstructions),
        "candidate_iou_transitions": {
            label: {tau: dict(values) for tau, values in per_tau.items()}
            for label, per_tau in transitions.items()
        },
        "official_tau_values": taus,
        "candidate_threshold_operator": ">",
        "reconstructed_partition_hashes": partition_hashes,
        "completed_episode_ids": completed,
        "failures": failures,
        "elapsed_seconds": time.monotonic() - reserved_started,
        "device": device.type,
        "gpu_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "limitations": "Target identities are recovered only from exact same-class occupancy matches. Ambiguous recoveries remain excluded from candidate-IoU transitions, with their count explicit. Original partition digests were not stored: the regenerated partition is bound to the same deterministic inputs and cross-checked against cached positive target occupancies. FP32 abs(logit)==2 boundary changes are reported separately.",
    }
    write_json(destination, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-root", type=Path, required=True)
    parser.add_argument("--parent-id", choices=("R1", "P"), default="R1")
    parser.add_argument("--gpu", type=int)
    args = parser.parse_args()
    if args.gpu is not None and args.gpu not in _free_gpus():
        raise ValueError(
            "Diagnostics require an idle A40 when GPU execution is requested"
        )
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu) if args.gpu is not None else ""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    config = load_config(PROJECT_ROOT / "configs/perception_gain_v2.yaml")
    artifacts = PROJECT_ROOT / config["artifact_root"]
    event = {
        "event_id": f"{args.parent_id}-post-train-diagnostics:{utc_now()}",
        "task": f"REPAIR_{args.parent_id}",
        "substep": "POST_TRAIN_DIAGNOSTICS",
        "scope": "V2",
        "start_utc": utc_now(),
        "pid": os.getpid(),
        "gpu_count": int(args.gpu is not None),
        "gpus": [args.gpu] if args.gpu is not None else [],
        "code": executed_identity(PROJECT_ROOT, [Path(__file__)]),
        "argv": [
            "python",
            "-m",
            "scripts.perception_gain_v2_diagnostics",
            "--external-root",
            "$PERSIST4D_GAIN_V2_ROOT",
            "--parent-id",
            args.parent_id,
            *(["--gpu", str(args.gpu)] if args.gpu is not None else []),
        ],
        "cwd": "repo:.",
        "outputs": [
            f"repo:{config['artifact_root']}/refiner/{args.parent_id}/POST_TRAIN_DIAGNOSTICS.json"
        ],
    }
    started = time.monotonic()
    cpu_started = time.process_time()
    try:
        result = run_diagnostics(
            config,
            external_root=args.external_root,
            parent_id=args.parent_id,
            device_name="cuda:0" if args.gpu is not None else "cpu",
        )
        event["status"] = result["status"]
        event["exit_code"] = 0 if result["status"] == "PASS" else 1
    except Exception as error:
        event.update(status="BLOCKED", reason=str(error), exit_code=1)
        raise
    finally:
        elapsed = time.monotonic() - started
        event.update(
            end_utc=utc_now(),
            elapsed_seconds=elapsed,
            gpu_hours=elapsed * event["gpu_count"] / 3600,
            measurement="MEASURED_PROCESS_GPU_RESERVATION_WALLTIME",
            cpu_core_hours=(time.process_time() - cpu_started) / 3600,
        )
        append_event(artifacts / "budget/LEDGER.jsonl", event)
        append_event(artifacts / "EXECUTION_LOG.jsonl", event)
    print(
        json.dumps(
            {
                "status": result["status"],
                "candidate_records": result["candidate_record_count"],
                "gpu_hours": event["gpu_hours"],
            }
        ),
        flush=True,
    )
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
