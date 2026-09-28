"""Independent official pooled metrics and CAL-only fixed-checkpoint selection."""

import csv
import hashlib
import inspect
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from models.short_module_heads import Module, apply_module, build_head
from scripts.p6a_metrics import OfficialMetricAccumulator
from scripts.short_module_native import unpack_prediction
from scripts.short_module_screen import ARTIFACTS, read_json, sha256, write_json
from scripts.system_comparison_inference import unpack_bool_matrix

ARMS = ("Q1", "Q2", "Q3", "M0", "M1", "M2")
TOLERANCE = 1e-6
CONTROLS = {"Q2": "Q1", "Q3": "Q2", "M1": "M0", "M2": "M1"}


def build_cal_lock(rows: list[dict]) -> dict:
    selected = {}
    for arm in ARMS:
        points = [r for r in rows if r["module"] == arm and r["seed"] == 45
                  and r["step"] in (500, 1000, 1500)]
        if (len(points) != 3 or {r["step"] for r in points} != {500, 1000, 1500}
                or any(r["status"] != "COMPLETE" or r["metrics"]["T1"] is None
                       or r["metrics"]["T2"] is None for r in points)):
            raise ValueError(f"{arm} lacks all complete trained CAL checkpoints")
        best = min(points, key=lambda r: r["step"])
        for row in sorted(points, key=lambda r: r["step"]):
            d2 = row["metrics"]["T2"] - best["metrics"]["T2"]
            d1 = row["metrics"]["T1"] - best["metrics"]["T1"]
            if d2 > TOLERANCE or (abs(d2) <= TOLERANCE and d1 > TOLERANCE):
                best = row
        selected[arm] = best["step"]
    return {"schema": "short-module-cal-lock-v1", "seed": 45, "selected_updates": selected,
            "selection": "complete CAL T2 descending, CAL T1 descending, earlier update; tolerance1e-6",
            "all_arms_locked_before_SEL": True}


def classify_gain(d2: float, d1: float | None, t1_complete: bool) -> str:
    if d2 < -TOLERANCE:
        return "NEGATIVE"
    if abs(d2) <= TOLERANCE:
        return "NO_GAIN"
    if not t1_complete or d1 is None or d1 < -.002 - TOLERANCE:
        return "T2_ONLY_TRADEOFF"
    return "TARGET_GAIN" if d2 >= .005 - TOLERANCE else "MODEST_GAIN"


def build_shortlist(rows: list[dict]) -> dict:
    positive = {"MODEST_GAIN", "TARGET_GAIN"}
    return {"confirmed": [r for r in rows if r["evidence_level"] == "DEVELOPMENT_REPLICATED"],
            "provisional": [r for r in rows if r["classification"] in positive
                            and r["evidence_level"] != "DEVELOPMENT_REPLICATED"],
            "tradeoff": [r for r in rows if r["classification"] == "T2_ONLY_TRADEOFF"],
            "diagnostic": [r for r in rows if r.get("geometry_signal_only", False)],
            "slot_policy": "ALTERNATIVES_NOT_STACKABLE", "combinations_executed": False,
            "evidence_ceiling": "development_replicated"}


def materialization_system(index: dict):
    from trainer.trainer import InstanceSegmentation

    native = read_json(Path(index["cache"]) / "IDENTITY.json")["native_config"]
    system = SimpleNamespace(eval_on_segments=native["general"]["eval_on_segments"])
    system._get_full_res_mask = InstanceSegmentation._get_full_res_mask.__get__(system)
    return system


def load_head(root: Path, module: str, step: int, seed: int, device: str):
    resolved = read_json(ARTIFACTS / "RUN_CONFIG.json")
    if module == "B0":
        return None, None
    path = root / "training" / module / f"seed{seed}" / f"update={step:04d}.pt"
    saved = torch.load(path, map_location="cpu", weights_only=False)
    meta = saved["metadata"]
    if (meta["module"] != module or meta["seed"] != seed or meta["optimizer_updates"] != step
            or meta["base_sha256"] != resolved["protocol"]["parent"]["sha256"]
            or meta["dimensions"] != resolved["parent_dimensions"]):
        raise ValueError("checkpoint mode, parent, dimensions, seed or update mismatch")
    index = read_json(root / "EXPORT_INDEX.json")
    if meta["cache_identity_sha256"] != index["cache_identity_sha256"]:
        raise ValueError("checkpoint was trained on a different input/cache identity")
    head = build_head(module, **meta["dimensions"], thresholds=tuple(meta["thresholds"]), seed=seed)
    head.load_state_dict(saved["head"], strict=True)
    return head.to(device).eval(), path


def evaluate_point(root: Path, *, module: str, step: int, seed: int,
                   role: str, device: str = "cuda:1") -> dict:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    Module(module)
    if role not in ("CAL", "SEL"):
        raise ValueError("round-one evaluation is restricted to CAL and SEL")
    if module != "B0" and role == "SEL" and seed == 45:
        lock = read_json(ARTIFACTS / "selection/CAL_LOCK.json")
        if set(lock["selected_updates"]) != set(ARMS) or lock["selected_updates"][module] != step:
            raise ValueError("SEL requires every arm locked and this arm's fixed CAL point")
    index = read_json(root / "EXPORT_INDEX.json")
    if index["status"] != "COMPLETE":
        raise ValueError("full export must precede pooled evaluation")
    head, head_path = load_head(root, module, step, seed, device)
    semantic_sources = "\n".join(inspect.getsource(f) for f in (evaluate_point, load_head, materialization_system))
    evaluator_sha = hashlib.sha256(semantic_sources.encode()).hexdigest()
    result_path = root / "evaluation" / evaluator_sha[:16] / module / f"seed{seed}" / f"{role}-{step:04d}.json"
    if result_path.exists():
        previous = read_json(result_path)
        if (previous["status"] == "COMPLETE" and previous["cache_identity_sha256"] == index["cache_identity_sha256"]
                and previous["head_sha256"] == (sha256(head_path) if head_path else None)):
            return previous
    system = materialization_system(index)
    assets = read_json(root / "assets.local.json")
    population = read_json(ARTIFACTS / "SHORT_POPULATION.json")
    records = [r for r in population["records"] if r["role"] == role]
    entries = {entry["input_id"]: entry for entry in index["entries"]}
    cache = Path(index["cache"])

    def metric(h):
        return OfficialMetricAccumulator(mode="raw_local" if h == 1 else "strict_online",
                                         dataset_spec=assets["metric_dataset_spec"], min_region_size=100)

    pooled = {1: metric(1), 2: metric(2)}
    reference_metrics = {}
    stage_metrics = {0: metric(1), 1: metric(1)}
    completed = {1: 0, 2: 0}
    incomplete = []
    isolation = {"mask_unchanged": True, "score_unchanged": True, "class_lineage_unchanged": True}
    started = time.perf_counter()
    for record in records:
        try:
            entry = entries[record["input_id"]]
            saved = torch.load(cache / entry["prediction"]["file"], map_location="cpu", weights_only=False)
            parent = unpack_prediction(saved["parent"])
            targets = torch.load(cache / entry["targets"]["file"], map_location="cpu", weights_only=False)
            target = targets["target"]
            target["masks"] = unpack_bool_matrix(target["masks"])
            prediction = apply_module(module, head, parent, saved["descriptor"], system=system)
            for name in ("pred_classes", "source_query_ids", "source_class_ids"):
                if not torch.equal(prediction[name], getattr(parent, name)):
                    raise ValueError("head changed class or retained candidate lineage")
            if module.startswith("Q") and not torch.equal(prediction["pred_masks"], parent.pred_masks):
                raise ValueError("score module changed geometry")
            if module.startswith("M") and not torch.equal(prediction["pred_scores"], parent.pred_scores):
                raise ValueError("mask module changed scores")
            isolation["mask_unchanged"] &= torch.equal(prediction["pred_masks"], parent.pred_masks)
            isolation["score_unchanged"] &= torch.equal(prediction["pred_scores"], parent.pred_scores)
            h = record["horizon"]
            reference = record["reference_id"]
            pooled[h].update(prediction, target)
            if (h, reference) not in reference_metrics:
                reference_metrics[h, reference] = metric(h)
            reference_metrics[h, reference].update(prediction, target)
            if h == 2:
                for stage, accumulator in stage_metrics.items():
                    keep = target["temporal_stages"] == stage
                    stage_target = {**target, "masks": target["masks"][:, keep],
                                    "temporal_stages": target["temporal_stages"][keep]}
                    stage_prediction = {**prediction, "pred_masks": prediction["pred_masks"][keep]}
                    accumulator.update(stage_prediction, stage_target)
            completed[h] += 1
        except (ValueError, RuntimeError, OSError, KeyError, TypeError, AssertionError) as error:
            incomplete.append({"logical_unit_id": record["logical_unit_id"], "input_id": record["input_id"],
                               "error": f"{type(error).__name__}: {error}"})
    expected = {h: sum(r["horizon"] == h for r in records) for h in (1, 2)}
    values = {}
    for h in (1, 2):
        key = "raw_local_AP" if h == 1 else "online_t-mAP"
        values[f"T{h}"] = pooled[h].compute()[key] if completed[h] == expected[h] and completed[h] else None
    by_reference = []
    for (h, reference), accumulator in sorted(reference_metrics.items()):
        expected_ref = sum(r["horizon"] == h and r["reference_id"] == reference for r in records)
        complete = accumulator._updates == expected_ref
        by_reference.append({"reference_id": reference, "H": h, "completed": accumulator._updates,
                             "expected": expected_ref, "AP": accumulator.compute()["raw_local_AP" if h == 1 else "online_t-mAP"]
                             if complete else None})
    result = {"module": module, "seed": seed, "step": step, "role": role,
              "status": "COMPLETE" if not incomplete and completed == expected else "PARTIAL",
              "metrics": values, "completed": completed, "expected": expected,
              "by_reference": by_reference, "incomplete": incomplete, "isolation": isolation,
              "stage_AP_given_T2": {f"stage{stage + 1}": m.compute()["raw_local_AP"]
                                    if completed[2] == expected[2] else None for stage, m in stage_metrics.items()},
              "head_sha256": sha256(head_path) if head_path else None,
              "cache_identity_sha256": index["cache_identity_sha256"],
              "evaluator_sha256": evaluator_sha, "elapsed_seconds": time.perf_counter() - started,
              "result_path": str(result_path)}
    write_json(result_path, result)
    for h, accumulator in pooled.items():
        if accumulator._updates:
            write_json(result_path.with_name(f"{role}-{step:04d}-H{h}-metric-state.json"), accumulator.export_evidence())
    print(f"{module} seed{seed} update{step} {role}: {values} {result['status']}", flush=True)
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_cal(root: Path, *, device: str) -> dict:
    base = baseline(root, device=device)
    rows = []
    for arm in ARMS:
        for step in (0, 500, 1000, 1500):
            try:
                row = evaluate_point(root, module=arm, step=step, seed=45, role="CAL", device=device)
            except (ValueError, RuntimeError, OSError, KeyError, TypeError) as error:
                row = {"module": arm, "seed": 45, "step": step, "status": "BLOCKED",
                       "metrics": {"T1": None, "T2": None}, "result_path": None,
                       "error": f"{type(error).__name__}: {error}"}
            rows.append(row)
    write_csv(ARTIFACTS / "evaluation/CAL_ALL_CHECKPOINTS.csv",
              [{"module": r["module"], "seed": r["seed"], "step": r["step"], "status": r["status"],
                "T2_AP": r["metrics"]["T2"], "T1_AP": r["metrics"]["T1"],
                "result_path": r["result_path"], "error": r.get("error"),
                "expected_T1": base["CAL"]["expected"]["1" if "1" in base["CAL"]["expected"] else 1],
                "expected_T2": base["CAL"]["expected"]["2" if "2" in base["CAL"]["expected"] else 2]}
               for r in [base["CAL"], *rows]])
    lock = build_cal_lock(rows)
    lock["evidence"] = [{"path": r["result_path"], "sha256": sha256(Path(r["result_path"]))} for r in rows]
    path = ARTIFACTS / "selection/CAL_LOCK.json"
    if path.exists() and read_json(path) != lock:
        raise ValueError("CAL lock is immutable; refusing reselection")
    if not path.exists():
        write_json(path, lock)
    return lock


def baseline(root: Path, *, device: str) -> dict:
    result = {role: evaluate_point(root, module="B0", step=0, seed=45, role=role, device=device)
              for role in ("CAL", "SEL")}
    write_json(ARTIFACTS / "BASELINE.json", result)
    if any(row["status"] != "COMPLETE" for row in result.values()):
        raise ValueError("full B0 CAL/SEL evaluation is incomplete")
    return result


def screen(root: Path, *, device: str) -> list[dict]:
    lock = read_json(ARTIFACTS / "selection/CAL_LOCK.json")
    base = read_json(ARTIFACTS / "BASELINE.json")["SEL"]
    if base["status"] != "COMPLETE":
        raise ValueError("complete native baseline required")
    rows, results = [], {}
    by_reference = [{"module": "B0", "seed": 45, "step": 0, **r, "delta_B0": 0.}
                    for r in base["by_reference"]]
    for arm, step in lock["selected_updates"].items():
        try:
            result = evaluate_point(root, module=arm, step=step, seed=45, role="SEL", device=device)
        except (ValueError, RuntimeError, OSError, KeyError, TypeError) as error:
            result = {"module": arm, "seed": 45, "step": step, "role": "SEL", "status": "BLOCKED",
                      "metrics": {"T1": None, "T2": None}, "expected": base["expected"],
                      "by_reference": [], "error": f"{type(error).__name__}: {error}"}
        results[arm] = result
        if result["status"] != "COMPLETE":
            rows.append({"module": arm, "slot": "score_only" if arm.startswith("Q") else "mask_only",
                         "seed": 45, "actual_updates": 1500, "selected_step": step,
                         "SEL_T2_AP": result["metrics"]["T2"], "delta_T2_B0": None,
                         "SEL_T1_AP": result["metrics"]["T1"], "delta_T1_B0": None,
                         "classification": "BLOCKED", "positive_SEL_references": 0,
                         "reference_consistency": "INCOMPLETE", "evidence_level": "NOT_REPLICATED"})
            continue
        d2 = result["metrics"]["T2"] - base["metrics"]["T2"]
        d1 = result["metrics"]["T1"] - base["metrics"]["T1"]
        base_refs = {(r["H"], r["reference_id"]): r["AP"] for r in base["by_reference"]}
        positives = 0
        for row in result["by_reference"]:
            delta = row["AP"] - base_refs[(row["H"], row["reference_id"])]
            positives += int(row["H"] == 2 and delta > TOLERANCE)
            by_reference.append({"module": arm, "seed": 45, "step": step, **row, "delta_B0": delta})
        rows.append({"module": arm, "slot": "score_only" if arm.startswith("Q") else "mask_only",
                     "seed": 45, "actual_updates": 1500, "selected_step": step,
                     "SEL_T2_AP": result["metrics"]["T2"], "delta_T2_B0": d2,
                     "SEL_T1_AP": result["metrics"]["T1"], "delta_T1_B0": d1,
                     "classification": classify_gain(d2, d1, True), "positive_SEL_references": positives,
                     "reference_consistency": "REFERENCE_CONSISTENT" if positives >= 3 else "REFERENCE_MIXED",
                     "evidence_level": "NOT_REPLICATED"})
    write_csv(ARTIFACTS / "evaluation/SEL_FIXED_CHECKPOINTS.csv", rows)
    write_csv(ARTIFACTS / "evaluation/BY_REFERENCE.csv", by_reference)
    write_csv(ARTIFACTS / "selection/MODULE_SCREENING.csv", rows)
    write_json(ARTIFACTS / "selection/SCREEN_RESULTS.json", results)
    write_json(ARTIFACTS / "selection/SHORTLIST.json", build_shortlist(rows))
    return rows
