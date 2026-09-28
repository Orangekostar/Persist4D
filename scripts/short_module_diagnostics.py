"""Evidence-only candidate and geometry diagnostics, never deployed oracle scores."""

from collections import Counter
from pathlib import Path

import torch

from models.short_module_heads import apply_module
from scripts.p6a_metrics import OfficialMetricAccumulator
from scripts.short_module_evaluation import (
    ARMS,
    load_head,
    materialization_system,
    write_csv,
)
from scripts.short_module_native import unpack_prediction
from scripts.short_module_screen import ARTIFACTS, read_json, write_json
from scripts.system_comparison_inference import unpack_bool_matrix


def pairwise_geometry(prediction: dict, target: dict, *, dataset_spec: Path,
                      min_region_size: int = 100) -> dict:
    from stmetrics.instances.matcher import InstanceMatcher

    class ColumnMatcher(InstanceMatcher):
        def _format_prediction_info(self, pred):
            return dict(enumerate(super()._format_prediction_info(pred).values()))

    matcher = ColumnMatcher(str(dataset_spec), min_region_size=min_region_size,
                            timestep_key="temporal_stages")
    evaluator = OfficialMetricAccumulator(mode="strict_online", dataset_spec=dataset_spec,
                                          min_region_size=min_region_size)._metric.heads[0]
    config = matcher.config
    params = (config.min_region_sizes[0].item(), config.distance_threshes[0].item(),
              config.distance_confs[0].item())
    gt2pred, _ = matcher._assign_instances_for_scan(prediction, target)
    result = {}
    for group in gt2pred.values():
        for gt in group.values():
            if gt["ambiguous"] or not evaluator._valid_gt(gt, *params) or not (gt["vert_count"] > 0).all():
                continue
            overlaps = {int(p["uuid"]): [float(x) for x in p["overlaps"]]
                        for p in gt["matched_pred"]}
            worst = {candidate: min(values) for candidate, values in overlaps.items()}
            joint = max(worst.values(), default=0.)
            separate = min((max((values[t] for values in overlaps.values()), default=0.)
                            for t in range(len(gt["vert_count"]))), default=0.)
            result[int(gt["instance_id"])] = {"U_joint": joint, "U_separate": separate,
                                              "candidate_worst": worst}
    return result


def diagnostics(root: Path, *, device: str, include_transitions: bool = True) -> dict:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    index = read_json(root / "EXPORT_INDEX.json")
    if index["status"] != "COMPLETE":
        raise ValueError("diagnostics require complete shared export")
    cache = Path(index["cache"])
    assets = read_json(root / "assets.local.json")
    thresholds = read_json(ARTIFACTS / "RUN_CONFIG.json")["official_thresholds"]
    failure_rows, presence_rows, upper_rows, transition_rows = [], [], [], []
    counts = Counter()
    heads = {}
    if include_transitions:
        lock = read_json(ARTIFACTS / "selection/CAL_LOCK.json")
        for arm in ARMS:
            heads[arm] = load_head(root, arm, lock["selected_updates"][arm], 45, device)[0]
    system = materialization_system(index)
    for entry in index["entries"]:
        record = entry["record"]
        if record["role"] not in ("TRAIN", "CAL") and not include_transitions:
            continue
        saved = torch.load(cache / entry["prediction"]["file"], map_location="cpu", weights_only=False)
        data = torch.load(cache / entry["targets"]["file"], map_location="cpu", weights_only=False)
        parent = unpack_prediction(saved["parent"])
        target = data["target"]
        target["masks"] = unpack_bool_matrix(target["masks"])
        labels = data["quality"]
        if record["role"] in ("TRAIN", "CAL"):
            reliable = {}
            for candidate in sorted(range(len(labels["status"])),
                                    key=lambda i: (-float(labels["temporal"][i]), saved["candidate_keys"][i])):
                status = labels["status"][candidate]
                identity = labels["gt_ids"][candidate]
                temporal = float(labels["temporal"][candidate])
                category = "unknown_ignore"
                if status == "MATCHED":
                    if temporal <= thresholds[0]:
                        category = "low_overlap"
                    elif identity in reliable:
                        category = "duplicate"
                    else:
                        reliable[identity] = candidate
                        category = "reliable_correspondence"
                elif status == "VALID_NEGATIVE":
                    compatible = any(int(label) == int(parent.pred_classes[candidate]) and int(entity) in labels["valid_gt_ids"]
                                     for label, entity in zip(target["labels"], target["ids"]))
                    category = "low_overlap" if compatible else "no_same_class_gt"
                score = float(parent.pred_scores[candidate])
                failure_rows.append({"role": record["role"], "reference_id": record["reference_id"],
                    "input_id": record["input_id"], "H": record["horizon"], "retained_index": candidate,
                    "source_query_id": int(parent.source_query_ids[candidate]),
                    "source_class_id": int(parent.source_class_ids[candidate]), "category": category,
                    "official_status": status, "parent_score": score,
                    "score_group": "low_le_025" if score <= .25 else "high_ge_075" if score >= .75 else "middle",
                    "concat_iou": float(labels["concat"][candidate]), "temporal_overlap": temporal,
                    "label_valid": bool(labels["valid"][candidate]), "ambiguity_metadata": labels["ambiguity_metadata"]})
                counts[record["role"], category] += 1
            if record["horizon"] == 2:
                for entity in labels["valid_gt_ids"]:
                    mask = target["masks"][target["ids"] == entity].any(0)
                    presence = [bool(mask[target["temporal_stages"] == stage].any()) for stage in (0, 1)]
                    presence_rows.append({"role": record["role"], "reference_id": record["reference_id"],
                        "input_id": record["input_id"], "entity_id": entity,
                        "presence": "both" if all(presence) else "appeared" if presence[1] else "disappeared",
                        "stage1_present": presence[0], "stage2_present": presence[1]})
        if record["horizon"] != 2:
            continue
        original = pairwise_geometry(parent.prediction(), target, dataset_spec=Path(assets["metric_dataset_spec"]))
        if record["role"] in ("TRAIN", "CAL"):
            for threshold in thresholds:
                upper_rows.append({"role": record["role"], "reference_id": record["reference_id"],
                    "input_id": record["input_id"], "threshold": threshold, "valid_both_stage_GT": len(original),
                    "joint_pass": sum(r["U_joint"] > threshold for r in original.values()),
                    "separate_pass": sum(r["U_separate"] > threshold for r in original.values())})
        if not include_transitions or record["role"] not in ("CAL", "SEL"):
            continue
        for arm in ARMS:
            prediction = apply_module(arm, heads[arm], parent, saved["descriptor"], system=system)
            if arm.startswith("Q"):
                if not torch.equal(prediction["pred_masks"], parent.pred_masks):
                    raise ValueError("quality head changed diagnostic geometry")
                current = original
            else:
                current = pairwise_geometry(prediction, target, dataset_spec=Path(assets["metric_dataset_spec"]))
            tracks = [(original[int(entity)]["candidate_worst"].get(candidate, 0.),
                       current.get(int(entity), {}).get("candidate_worst", {}).get(candidate, 0.))
                      for candidate, entity in enumerate(data["assignment"])
                      if int(entity) in original]
            for threshold in thresholds:
                transition_rows.append({"module": arm, "seed": 45, "role": record["role"],
                    "reference_id": record["reference_id"], "input_id": record["input_id"], "threshold": threshold,
                    "valid_both_stage_GT": len(original),
                    "joint_pass_B0": sum(r["U_joint"] > threshold for r in original.values()),
                    "joint_pass_module": sum(r["U_joint"] > threshold for r in current.values()),
                    "original_assigned_tracks": len(tracks),
                    "repaired_tracks": sum(before <= threshold < after for before, after in tracks),
                    "broken_tracks": sum(after <= threshold < before for before, after in tracks),
                    "mean_worst_stage_iou_delta": sum(after - before for before, after in tracks) / len(tracks) if tracks else None})
    write_csv(ARTIFACTS / "diagnostics/CANDIDATE_FAILURES.csv", failure_rows)
    write_csv(ARTIFACTS / "diagnostics/APPEAR_DISAPPEAR.csv", presence_rows)
    write_csv(ARTIFACTS / "diagnostics/JOINT_SEPARATE_COUNTS.csv", upper_rows)
    write_csv(ARTIFACTS / "diagnostics/GEOMETRY_TRANSITIONS.csv", transition_rows)
    quality_groups = {}
    for row in failure_rows:
        key = row["role"], row["H"], row["score_group"]
        quality_groups.setdefault(key, []).append(row)
    distributions = []
    for (role, horizon, score_group), group in sorted(quality_groups.items()):
        values = torch.tensor([r["temporal_overlap"] for r in group if r["label_valid"]])
        quantiles = values.quantile(torch.tensor([.1, .5, .9])).tolist() if len(values) else [None] * 3
        distributions.append({"role": role, "H": horizon, "score_group": score_group,
            "all_candidates": len(group), "valid_quality_labels": len(values),
            "temporal_mean": values.mean().item() if len(values) else None,
            "temporal_q10": quantiles[0], "temporal_median": quantiles[1], "temporal_q90": quantiles[2],
            "reliable_gt050": int((values > .5).sum()),
            "ranking_misalignment_count": int((values > .5).sum()) if score_group == "low_le_025" else
                int((values <= .5).sum()) if score_group == "high_ge_075" else None})
    write_csv(ARTIFACTS / "diagnostics/SCORE_GEOMETRY_DISTRIBUTION.csv", distributions)
    reference_counts = {}
    for row in upper_rows:
        key = row["role"], row["reference_id"], row["threshold"]
        total = reference_counts.setdefault(key, Counter())
        for field in ("valid_both_stage_GT", "joint_pass", "separate_pass"):
            total[field] += row[field]
        total["logical_pairs"] += 1
    write_csv(ARTIFACTS / "diagnostics/JOINT_SEPARATE_BY_REFERENCE.csv", [
        {"role": role, "reference_id": reference, "threshold": threshold, **counts}
        for (role, reference, threshold), counts in sorted(reference_counts.items())])
    summary = {"status": "COMPLETE" if include_transitions else "BASELINE_DIAGNOSTICS_COMPLETE",
               "candidate_denominator": len(failure_rows), "candidate_counts": [
                   {"role": role, "category": category, "count": count}
                   for (role, category), count in sorted(counts.items())],
               "presence_denominator": len(presence_rows), "geometry_transition_rows": len(transition_rows),
               "scope": "Class-compatible official pairwise geometry; reliable means overlap strictly greater than0.5. Duplicate is a subsequent reliable candidate for the same GT in geometry/key order.",
               "track_definition": "Frozen one-to-one concat-IoU>=0.10 assignment, one entity across stages; valid GT present in both stages",
               "limitation": "Diagnostic geometry quantities are not realizable AP upper bounds. Empty-stage rules are the installed official matcher rules."}
    write_json(ARTIFACTS / "diagnostics/SUMMARY.json", summary)
    return summary
