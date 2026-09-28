"""Actual single/two-scan parent forward and compact prediction-only cache."""

import time
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

from models.perception_gain import derive_segment_stage_ids
from models.short_module_heads import describe_candidates
from scripts.rescene_task_postprocess import (
    OfficialTaskPrediction,
    OfficialTaskSoftEvidence,
    extract_official_task_prediction,
)
from scripts.system_comparison_inference import pack_bool_matrix, unpack_bool_matrix


def pack_prediction(parent: OfficialTaskPrediction) -> dict:
    parent.validate()
    soft = parent.soft_evidence
    if soft is None:
        raise ValueError("native export requires soft evidence")
    omitted = {"low_resolution_logits", "full_resolution_logits", "full_resolution_probabilities"}
    return {"soft": {f.name: getattr(soft, f.name) for f in fields(soft) if f.name not in omitted},
            "masks": pack_bool_matrix(parent.pred_masks), "scores": parent.pred_scores,
            "classes": parent.pred_classes, "temporal_stages": parent.temporal_stages,
            "latest_stage_index": parent.latest_stage_index}


def unpack_prediction(packed: dict) -> OfficialTaskPrediction:
    values = dict(packed["soft"])
    low = values["segment_logits"][values["low_point2segment"]]
    full = low[values["voxel_inverse"]]
    soft = OfficialTaskSoftEvidence(**values, low_resolution_logits=low,
                                    full_resolution_logits=full,
                                    full_resolution_probabilities=full.sigmoid())
    masks = unpack_bool_matrix(packed["masks"])
    stages = packed["temporal_stages"]
    latest = packed["latest_stage_index"]
    result = OfficialTaskPrediction(masks, packed["scores"], packed["classes"],
                                    soft.source_query_ids, soft.source_class_ids,
                                    stages, latest, masks[stages == latest], soft)
    result.validate()
    return result


class NativeSession:
    def __init__(self, *, config, datasets: dict, assets: dict, device: str):
        import hydra

        from scripts.perception_gain_evaluation import _load_evaluation_weights
        from trainer.perception_gain_trainer import PerceptionGainTrainer

        self.device = torch.device(device)
        if self.device.type != "cuda" or self.device.index is None:
            raise ValueError("native session requires an explicit CUDA device")
        self.datasets = datasets
        self.collate = hydra.utils.instantiate(config.data.validation_collation)
        self.system = PerceptionGainTrainer(config)
        self.weight_audit = _load_evaluation_weights(
            system=self.system, variant="C0", optimizer_update=0,
            r1_checkpoint=Path(assets["r1_checkpoint"]), checkpoint=None, scorer_checkpoint=None)[0]
        self.system.to(self.device).eval().requires_grad_(False)
        if any(p.requires_grad for p in self.system.parameters()) or self.system.training:
            raise ValueError("parent must be frozen and eval")

    def produce(self, record: dict, *, compare_native: bool = False) -> tuple[dict, dict, dict]:
        from scripts.evaluate_persist4d import (
            _move_data_to_device,
            _move_targets_to_device,
        )
        from scripts.evaluate_persist4d_p6a import (
            _frozen_inference_seed,
            build_rio_class_mapper,
        )
        from scripts.system_comparison_inference import (
            deterministic_inference_runtime,
            postprocess_full_history_output,
        )

        dataset = self.datasets[record["dataset_horizon"]]
        mapper = build_rio_class_mapper(dataset)
        h = record["horizon"]
        if h not in (1, 2) or h != len(record["scan_indices"]):
            raise ValueError("native record must read exactly H scans")
        expected_paths = [str(Path(dataset.data[i]["filepath"].replace("../../", "")).resolve())
                          for i in record["scan_indices"]]
        loaded_paths = []
        real_load = np.load

        def traced_load(path, *args, **kwargs):
            loaded_paths.append(str(Path(path).resolve()))
            return real_load(path, *args, **kwargs)

        started = time.perf_counter()
        with deterministic_inference_runtime(45, self.device), _frozen_inference_seed(45, self.device):
            np.load = traced_load
            try:
                sample = dataset.load_scan_indices(record["context_index"], tuple(record["scan_indices"]),
                                                   change_file=None)
            finally:
                np.load = real_load
            if loaded_paths != expected_paths:
                raise ValueError("native file reads differ from declared scans (possible substitution)")
            coordinates = torch.from_numpy(sample[6][:, :3].copy()).float()
            semantic_labels = torch.from_numpy(sample[2][:, 0].copy()).long()
            full_stages = torch.from_numpy(sample[6][:, 3].copy()).long()
            if set(full_stages.tolist()) != set(range(h)):
                raise ValueError("native temporal encoding differs, including true T1 time zero")
            data, targets, names = self.collate([sample])
            if len(targets) != 1 or names != [dataset.sequence_names[record["context_index"]]]:
                raise ValueError("native collator changed sample identity")
            target_full = data.target_full[0]
            loaded_seconds = time.perf_counter() - started
            data = _move_data_to_device(data, self.device)
            targets = _move_targets_to_device(targets, self.device)
            low = targets[0]
            raw_coordinates = self.system._process_raw_coordinates(data)
            torch.cuda.synchronize(self.device)
            network_start = time.perf_counter()
            with torch.inference_mode():
                output = self.system(data, point2segment=[low["point2segment"]],
                                     raw_coordinates=raw_coordinates, is_eval=True)
            torch.cuda.synchronize(self.device)
            network_seconds = time.perf_counter() - network_start

            def fresh_output():
                # Official processing may replace aux dict logits. Each call
                # gets separate dictionaries while read-only tensors are shared.
                return {**output, "aux_outputs": [dict(v) for v in output["aux_outputs"]]}

            parent = extract_official_task_prediction(
                system=self.system, output=fresh_output(), target_low_resolution=low,
                target_full_resolution=target_full, data=data, class_mapper=mapper,
                latest_stage_index=h - 1, return_soft_evidence=True)
            equality = None
            if compare_native:
                native = postprocess_full_history_output(
                    system=self.system, output=fresh_output(), target_low_resolution=low,
                    target_full_resolution=target_full, data=data, horizon=h, class_mapper=mapper,
                    background_class=18, confidence_threshold=.5, mask_threshold=.5,
                    minimum_mask_support=1)
                equality = {name: torch.equal(value, parent.prediction()[name])
                            for name, value in native.task_prediction.items()}
                if not all(equality.values()):
                    raise ValueError("native soft output differs from original native prediction")
            soft = parent.soft_evidence
            low_stages = low["temporal_stages"].detach().cpu().long()
            segment_stages = derive_segment_stage_ids(soft.low_point2segment, low_stages)
            full_to_low = soft.low_point2segment[soft.voxel_inverse]
            segment_count = soft.segment_features.shape[0]
            counts = torch.bincount(full_to_low, minlength=segment_count).float()
            centroids = torch.zeros(segment_count, 3)
            centroids.index_add_(0, full_to_low, coordinates)
            centroids /= counts[:, None].clamp_min(1)
            if not torch.equal(segment_stages[full_to_low], full_stages):
                raise ValueError("low segment mappings mix physical scans")
            stable_keys = tuple(f"{record['scan_ids'][int(stage)]}:{i}"
                                for i, stage in enumerate(segment_stages))
            descriptor_start = time.perf_counter()
            local_h, support, probability = describe_candidates(
                soft.segment_features, soft.segment_logits, centroids, segment_stages,
                stable_keys, horizon=h)
            descriptor_seconds = time.perf_counter() - descriptor_start
            dimensions = {"feature_dim": soft.segment_features.shape[1],
                          "query_dim": soft.query_features.shape[1],
                          "classes": output["pred_logits"].shape[-1] - 1,
                          "probability_dim": soft.class_probabilities.shape[1]}
            descriptor = {"h": local_h, "support": support, "foreground_probability": probability,
                          "segment_stages": segment_stages, "classes": dimensions["classes"]}
            packed = pack_prediction(parent)
            prediction = {"input_id": record["input_id"], "parent": packed,
                          "descriptor": descriptor, "dimensions": dimensions,
                          "centroids": centroids, "segment_keys": stable_keys,
                          "scan_ids": record["scan_ids"], "coordinates": coordinates,
                          "vertex_ids": [torch.arange(int((full_stages == t).sum())) for t in range(h)],
                          "candidate_keys": [(record["input_id"], int(q), int(c), i)
                                             for i, (q, c) in enumerate(zip(soft.source_query_ids,
                                                                           soft.source_class_ids))]}
            target = {"masks": target_full["masks"].detach().cpu().bool(),
                      "labels": torch.tensor([mapper(int(v)) for v in target_full["labels"]]),
                      "ids": target_full["ids"].detach().cpu().long(),
                      "changes": torch.zeros_like(target_full["ids"]).cpu(),
                      "temporal_stages": full_stages}
            if target_full.get("ambiguities") is not None:
                target["ambiguities"] = target_full["ambiguities"]
            train_targets = {"target": target, "semantic_labels": semantic_labels,
                             "semantic_source": "dataset sample[2][:,0], preprocessed raw semantic labels; 255 unknown"}
        audit = {"input_id": record["input_id"], "horizon": h, "actual_loaded_paths": loaded_paths,
                 "native_equality": equality, "dimensions": dimensions,
                 "candidate_count": parent.pred_scores.numel(), "segment_count": segment_count,
                 "full_vertex_count": full_stages.numel(), "input_seconds": loaded_seconds,
                 "parent_network_seconds": network_seconds, "descriptor_seconds": descriptor_seconds,
                 "elapsed_seconds": time.perf_counter() - started,
                 "parent_frozen": not any(p.requires_grad for p in self.system.parameters()),
                 "parent_eval": not self.system.training,
                 "cuda_peak_allocated": torch.cuda.max_memory_allocated(self.device)}
        return prediction, train_targets, audit
