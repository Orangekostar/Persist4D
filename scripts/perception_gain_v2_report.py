"""Evidence-only V2 report tables; missing measurements retain fixed denominators."""

from __future__ import annotations

import csv
import datetime as dt
import json
import shlex
import subprocess
from pathlib import Path

from scripts.perception_gain_v2 import (
    PROJECT_ROOT,
    TASKS,
    file_hash,
    read_json,
    utc_now,
    write_json,
)
from scripts.perception_gain_v2_confirmation import POPULATIONS
from scripts.perception_gain_v2_selection import compare


def optional(path: Path) -> dict:
    return read_json(path) if path.is_file() else {}


def json_lines(path: Path) -> list[dict]:
    return (
        [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if path.exists()
        else []
    )


def ledger_totals(events: list[dict]) -> dict:
    unique = {row["event_id"]: row for row in events}
    prior = sum(row["gpu_hours"] for row in unique.values() if row["scope"] == "PRIOR")
    current = sum(row["gpu_hours"] for row in unique.values() if row["scope"] == "V2")
    return {
        "prior_gpu_hours": prior,
        "v2_gpu_hours": current,
        "settled_gpu_hours": prior + current,
        "event_count": len(unique),
    }


def perception_selection_label(result: dict) -> str:
    selected = result.get("selected", {})
    if result.get("fallback") or selected.get("method_id") == "R1-D0":
        return "KEEP_R1"
    if selected.get("architecture_variant") == "C0":
        return "CONTINUATION_ONLY"
    return "NEW_PERCEPTION" if selected else "NOT_RUN"


def confirmation_rows(lock: dict, report: dict) -> tuple[list, list]:
    rows, references = [], []
    defaults = {
        "PB": ["FH-R1-native", "R1-D0", "P-D0", "FH-P-native", "FINAL"],
        "ADDITIONAL": ["FH-R1-native", "FINAL-parent-D0", "FINAL"],
        "LOCAL-T2": ["FH-R1-native", "FH-P-native"],
    }
    for role, population in POPULATIONS.items():
        for name in lock.get("confirmation_methods", {}).get(role, defaults[role]):
            result = report.get("populations", {}).get(role, {}).get(name, {})
            for horizon, expected in population.items():
                coverage = result.get("coverage_by_horizon", {}).get(horizon, {})
                rows.append(
                    {
                        "population": role,
                        "method": name,
                        "T": int(horizon),
                        "eval_seed": 45,
                        "actual_units": coverage.get("actual_logical_units", 0),
                        "expected_units": expected["logical_units"],
                        "actual_references": coverage.get("actual_references", 0),
                        "expected_references": expected["references"],
                        "t_mAP": result.get("metrics", {}).get(horizon),
                        "coverage": coverage.get("status", "NOT_RUN"),
                        "source_status": result.get("source_status", "NOT_RUN"),
                        "artifact": result.get("artifact"),
                    }
                )
            references.extend(
                {"population": role, **row}
                for row in result.get("metric_rows", [])
                if row.get("reference") != "all"
            )
    for name in lock.get("confirmation_methods", {}).get(
        "LOCAL-T2", defaults["LOCAL-T2"]
    ):
        result = report.get("local_t2", {}).get(name, {})
        for seed in (45, 46, 47):
            value = result.get("seeds", {}).get(str(seed), {})
            rows.append(
                {
                    "population": "LOCAL-T2",
                    "method": name,
                    "T": 2,
                    "eval_seed": seed,
                    "actual_units": value.get("validation_sequence_count", 0),
                    "expected_units": 154,
                    "actual_references": value.get("validation_reference_count", 0),
                    "expected_references": lock.get("confirmation_populations", {})
                    .get("LOCAL-T2", {})
                    .get("references", 41),
                    "t_mAP": value.get("metrics", {}).get("t_mAP"),
                    "coverage": result.get("coverage_status", "NOT_RUN"),
                    "source_status": value.get("status", "NOT_RUN"),
                    "artifact": None,
                }
            )
        if result.get("mean_metrics") is not None:
            rows.append(
                {
                    **rows[-1],
                    "eval_seed": "mean(45,46,47)",
                    "t_mAP": result["mean_metrics"]["t_mAP"],
                }
            )
    return rows, references


def write_csv(path: Path, rows: list[dict], *, empty_fields=("status",)) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row)) or list(
        empty_fields
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(value, sort_keys=True)
                        if isinstance(value, (dict, list))
                        else value
                    )
                    for key, value in row.items()
                }
            )


def table(rows: list[dict], fields: list[str]) -> str:
    def value(item):
        if item is None:
            return "—"
        if isinstance(item, float):
            return f"{item:.6f}"
        if isinstance(item, (dict, list)):
            item = json.dumps(item, ensure_ascii=False, sort_keys=True)
        return str(item).replace("|", "\\|").replace("\n", " ")

    return "\n".join(
        [
            "| " + " | ".join(fields) + " |",
            "| " + " | ".join("---" for _ in fields) + " |",
            *[
                "| " + " | ".join(value(row.get(field)) for field in fields) + " |"
                for row in rows
            ],
        ]
    )


def auxiliary_complete(result: dict) -> bool:
    """A finished wrapper can still contain failed pilots or missing CAL evidence."""
    rate = result.get("learning_rate")
    if result.get("status") != "COMPLETE" or rate not in {"H", "L"}:
        return False
    for label in ("H", "L"):
        probes = result.get("probes", {}).get(label, [])
        if len(probes) != 4 or any(
            row.get("coverage_status") != "COMPLETE" for row in probes
        ):
            return False
    arms = result.get("arms", {})
    expected = {f"{variant}-{rate}-s45" for variant in ("A-OPEN", "Q-SEM", "S-WORST")}
    if set(arms) != expected:
        return False
    for name, row in arms.items():
        points = row.get("cal", [])
        if any(point.get("coverage_status") != "COMPLETE" for point in points):
            return False
        steps = {point.get("optimizer_update") for point in points}
        status = row.get("status")
        zero_probe = not name.startswith("S-WORST-")
        if status == "SKIPPED_BUDGET":
            continue
        if status == "SKIPPED_SEVERE_ZERO_STEP" and zero_probe and steps == {0}:
            continue
        required = {0, 250, 750} if zero_probe else {250, 750}
        if (
            status != "COMPLETE"
            or row.get("completed_global_step") != 750
            or not required.issubset(steps)
        ):
            return False
    return True


def observed_tasks(artifacts: Path, external_root: Path, state: dict) -> dict:
    """Report external recovery without mutating a running controller's state."""
    tasks = dict(state["tasks"])
    for event in json_lines(artifacts / "RECOVERY_EVENTS.jsonl"):
        if event["exit_code"] == 0 and tasks[event["task"]]["status"] in {
            "BLOCKED",
            "PENDING",
        }:
            path = external_root / event["result_path"]
            if path.exists() and file_hash(path) == event["result_sha256"]:
                tasks[event["task"]] = {
                    **read_json(path),
                    "observed_from": event["result_path"],
                }
    for path in (external_root / "tasks").glob("*_INVOCATION.json"):
        event = read_json(path)
        if event.get("end_utc") or event.get("task") not in tasks:
            continue
        pid = event.get("pid")
        command = Path(f"/proc/{pid}/cmdline")
        if command.exists() and b"scripts.perception_gain_v2" in command.read_bytes():
            tasks[event["task"]] = {
                **tasks[event["task"]],
                "status": "RUNNING",
                "observed_from": path.name,
                "pid": pid,
            }
    return tasks


def run_report(config: dict, *, external_root: Path) -> dict:
    artifacts = PROJECT_ROOT / config["artifact_root"]
    state = read_json(artifacts / "RUN_STATE.json")
    tasks = observed_tasks(artifacts, external_root, state)
    lock = optional(artifacts / "selection/FINAL_LOCK.json")
    local_population = optional(artifacts / "foundation/LOCAL_POPULATION_AUDIT.json")
    rebaseline = optional(artifacts / "foundation/REBASELINE_STATUS.json")
    supersession = optional(artifacts / "foundation/BASELINE_SUPERSESSION.json")
    confirmation = optional(artifacts / "confirmation/CONFIRMATION.json")
    auxiliary = optional(artifacts / "selection/AUX.json")
    perception = optional(artifacts / "selection/PERCEPTION.json")
    association = optional(artifacts / "association/SELECTION.json")
    replication = optional(artifacts / "replication/REPLICATION.json")
    profile = optional(artifacts / "resources/PROFILE_SUMMARY.json")
    publication = optional(external_root / "publication/PUBLICATION_RECEIPT.json")
    release_plan = optional(artifacts / "RELEASE_PLAN.json")
    cost = ledger_totals(json_lines(artifacts / "budget/LEDGER.jsonl"))
    cost["association_cpu_core_hours"] = sum(
        row["cpu_core_hours"]
        for row in json_lines(artifacts / "budget/ASSOCIATION_CPU.jsonl")
    )
    live = {
        row.get("pid"): row
        for row in state["tasks"].values()
        if row.get("status") == "RUNNING" and Path(f"/proc/{row.get('pid')}").exists()
    }
    for path in (external_root / "tasks").glob("*_INVOCATION.json"):
        row = read_json(path)
        if (
            row.get("start_utc")
            and not row.get("end_utc")
            and Path(f"/proc/{row.get('pid')}").exists()
        ):
            live.setdefault(row["pid"], row)
    cost["unsettled_live_reservation_gpu_hours"] = sum(
        max(
            0.0,
            (
                dt.datetime.now(dt.timezone.utc)
                - dt.datetime.fromisoformat(row["start_utc"])
            ).total_seconds(),
        )
        * (
            len(row["gpus"])
            if "gpus" in row
            else row.get("gpu_count", int(row.get("gpu") is not None))
        )
        / 3600
        for row in live.values()
    )
    runner_source = (PROJECT_ROOT / "scripts/perception_gain_v2.py").read_text()
    unimplemented = {
        name
        for name in ("PROFILE", "PUBLISH")
        if f'if task == "{name}":' not in runner_source
    }
    current_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
    ).strip()
    task_rows = [
        {
            "task": name,
            "status": tasks[name]["status"],
            "dependencies": list(dependencies),
            "reason": tasks[name].get(
                "reason", "CODE_HANDLER_PENDING" if name in unimplemented else None
            ),
            "observed_from": tasks[name].get("observed_from", "RUN_STATE.json"),
        }
        for name, dependencies in TASKS.items()
    ]
    baseline_rows = []
    for role, expected in (("CAL", 23), ("SEL", 24)):
        for method in ("D0", "FH-native"):
            result = optional(artifacts / f"foundation/{role}/{method}.json")
            for horizon in (2, 3, 4, 5):
                metric = next(
                    (
                        row
                        for row in result.get("metric_rows", [])
                        if row["reference"] == "all" and row["T"] == horizon
                    ),
                    {},
                )
                native_units = result.get("completed_units_by_horizon", {}).get(
                    str(horizon)
                )
                observed = result.get("population_by_horizon", {}).get(str(horizon), {})
                baseline_rows.append(
                    {
                        "role": role,
                        "method": method,
                        "T": horizon,
                        "status": result.get("status", "NOT_RUN"),
                        "t_mAP": metric.get("t_mAP"),
                        "actual_units": (
                            len(native_units)
                            if native_units is not None
                            else observed.get("logical_unit_count", 0)
                        ),
                        "expected_units": expected,
                        "actual_references": metric.get("reference_count", 0),
                        "expected_references": len(
                            read_json(artifacts / "DATA_ROLES.json")["roles"][role]
                        ),
                    }
                )
    write_json(artifacts / "foundation/BASELINE_COVERAGE.json", {"rows": baseline_rows})
    write_csv(artifacts / "foundation/BASELINE_METRICS.csv", baseline_rows)
    training = []
    for recipe_path in sorted((artifacts / "training").glob("*/recipe.json")):
        name = recipe_path.parent.name
        summary = optional(
            external_root / f"training/{name}/run_summary.json"
        ) or optional(recipe_path.parent / "run_summary.json")
        progress = optional(external_root / f"training/{name}/PROGRESS.json")
        steps = summary.get(
            "completed_global_step", progress.get("completed_optimizer_updates", 0)
        )
        imported = name == "C0-H-s45" and (external_root / "assets.local.json").exists()
        assets = optional(external_root / "assets.local.json") if imported else {}
        imported_steps = [
            step
            for step in (250, 750)
            if f"C0_H_{step:04d}" in assets
            and Path(assets[f"C0_H_{step:04d}"]).is_file()
        ]
        steps = max([steps, *imported_steps])
        training.append(
            {
                "recipe": name,
                "train_seed": read_json(recipe_path)["train_seed"],
                "completed_updates": steps,
                "status": summary.get(
                    "status",
                    (
                        "IN_PROGRESS"
                        if progress
                        else "IMPORTED"
                        if imported_steps
                        else "NOT_RUN"
                    ),
                ),
                "last_checkpoint": (
                    "external:" + f"training/{name}/last.ckpt"
                    if (external_root / f"training/{name}/last.ckpt").is_file()
                    else None
                ),
                "gpu_hours_in_summary": summary.get("gpu_hours"),
                "attribution_only": True,
            }
        )
        metrics = []
        for path in sorted(recipe_path.parent.glob("metrics-version_*.csv")):
            with path.open(newline="") as stream:
                metrics.extend(
                    {"source_file": path.name, **row} for row in csv.DictReader(stream)
                )
        if metrics:
            write_csv(recipe_path.parent / "metrics.csv", metrics)
    evaluation = []
    for role in ("CAL", "SEL"):
        baseline = optional(artifacts / f"foundation/{role}/D0.json").get(
            "candidate", {}
        )
        for path in sorted((artifacts / f"evaluation/{role}").glob("*/update=*.json")):
            result = read_json(path)
            candidate = result.get("candidate", {})
            comparison = compare(candidate, baseline)
            evaluation.append(
                {
                    "role": role,
                    "recipe": result.get("recipe", {}).get(
                        "recipe_id", path.parent.name
                    ),
                    "step": candidate.get("optimizer_update"),
                    "coverage": candidate.get("coverage_status"),
                    **{
                        f"T{t}": candidate.get("metrics", {}).get(str(t))
                        for t in (2, 3, 4, 5)
                    },
                    "mean_delta_R1": comparison["S_mean"],
                    "min_delta_R1": comparison["S_min"],
                    "artifact": str(path.relative_to(artifacts)),
                }
            )
    pilots = []
    for rate in ("H", "L"):
        for arm in ("C0", "S-BAL"):
            for step in (250, 750):
                name = f"{arm}-{rate}-s45"
                row = next(
                    (
                        item
                        for item in evaluation
                        if item["role"] == "CAL"
                        and item["recipe"] == name
                        and item["step"] == step
                    ),
                    {
                        "role": "CAL",
                        "recipe": name,
                        "step": step,
                        "coverage": "NOT_RUN",
                    },
                )
                control = next(
                    (
                        item
                        for item in evaluation
                        if item["role"] == "CAL"
                        and item["recipe"] == f"C0-{rate}-s45"
                        and item["step"] == step
                    ),
                    {},
                )
                deltas = (
                    [row[f"T{t}"] - control[f"T{t}"] for t in (2, 3, 4, 5)]
                    if row.get("coverage") == control.get("coverage") == "COMPLETE"
                    else None
                )
                pilots.append(
                    {
                        **row,
                        "mean_same_step_delta_C0": sum(deltas) / 4 if deltas else None,
                    }
                )
    repair_rows, coverage_rows, diagnostics = [], [], []
    repairs = {}
    parent_cohorts = []
    for parent in ("R1", "P"):
        folder = artifacts / f"refiner/{parent}"
        repairs[parent] = optional(folder / "SELECTION.json")
        manifest = optional(folder / "DATA_MANIFEST.json")
        post = optional(folder / "POST_TRAIN_DIAGNOSTICS.json")
        coverage_rows.append(
            {
                "parent": parent,
                "status": manifest.get("status", "NOT_RUN"),
                "episodes": manifest.get("episode_count"),
                "references": manifest.get("completed_reference_count"),
                "candidate_records": manifest.get("candidate_count"),
                **manifest.get("pairing_and_target_audit", {}),
            }
        )
        for label in ("NEW", "PAIR"):
            summary = optional(folder / f"{label}/TRAINING.json")
            diagnostics.append(
                {
                    "parent": parent,
                    "mode": label,
                    "trained_updates": summary.get("completed_updates", 0),
                    "loss": summary.get("final_loss"),
                    "post_diagnostic_status": post.get("status", "NOT_RUN"),
                    **post.get("head_counts", {}).get(label, {}),
                }
            )
        for role in ("CAL", "SEL"):
            result = optional(folder / f"{role}.json")
            if parent == "R1" and result.get("status") == "PASS":
                baseline = optional(artifacts / f"foundation/{role}/D0.json")
                paired = compare(
                    result.get("parent_candidate", {}), baseline.get("candidate", {})
                )
                parent_cohorts.append(
                    {
                        "role": role,
                        "comparison_status": paired["comparison_status"],
                        "deltas": paired["deltas"],
                        "published_output_equal": result.get(
                            "method_output_sha256", {}
                        ).get("PARENT")
                        == baseline.get("method_output_sha256", {}).get("PARENT"),
                        "status": (
                            "EXACT"
                            if paired["deltas"] is not None
                            and all(value == 0 for value in paired["deltas"].values())
                            and result.get("method_output_sha256", {}).get("PARENT")
                            == baseline.get("method_output_sha256", {}).get("PARENT")
                            else "PARENT_OUTPUT_DRIFT"
                        ),
                    }
                )
            for name, candidate in {
                "PARENT": result.get("parent_candidate", {}),
                **result.get("candidates", {}),
            }.items():
                compared = compare(candidate, result.get("parent_candidate", {}))
                repair_rows.append(
                    {
                        "parent": parent,
                        "role": role,
                        "method": name,
                        "step": candidate.get("optimizer_update"),
                        "coverage": candidate.get("coverage_status", "NOT_RUN"),
                        **{
                            f"T{t}": candidate.get("metrics", {}).get(str(t))
                            for t in (2, 3, 4, 5)
                        },
                        "mean_delta_parent": compared["S_mean"],
                        "min_delta_parent": compared["S_min"],
                    }
                )
    association_rows = []
    from models.overlap_entity_association import preregistered_association_configs

    for item in preregistered_association_configs():
        result = optional(artifacts / f"association/CAL/{item.config_id}.json")
        candidate = result.get("candidate", {})
        association_rows.append(
            {
                "config": item.config_id,
                "attempt": "CURRENT_PARENT",
                "status": result.get("status", "NOT_RUN"),
                "completed_units": len(result.get("completed_units", [])),
                "expected_units": 23,
                "cpu_core_hours": result.get("cpu_core_hours"),
                **{
                    f"T{t}": candidate.get("metrics", {}).get(str(t))
                    for t in (2, 3, 4, 5)
                },
            }
        )
    for path in sorted(
        (artifacts / "association_invalidated_parent_drift/CAL").glob("*.json")
    ):
        result = read_json(path)
        candidate = result.get("candidate", {})
        association_rows.append(
            {
                "config": path.stem,
                "attempt": "HISTORICAL_PARENT",
                "status": "INVALIDATED_PARENT_RUNTIME",
                "completed_units": len(result.get("completed_units", [])),
                "expected_units": 23,
                "cpu_core_hours": result.get("cpu_core_hours"),
                **{
                    f"T{t}": candidate.get("metrics", {}).get(str(t))
                    for t in (2, 3, 4, 5)
                },
            }
        )
    display_lock = lock
    if not lock and local_population.get("status") == "PASS":
        display_lock = {
            "confirmation_populations": {
                "LOCAL-T2": {
                    "references": local_population["validation_reference_count"]
                }
            }
        }
    formal, by_reference = confirmation_rows(display_lock, confirmation)
    write_json(
        artifacts / "foundation/PARENT_COHORT_COMPARISON.json",
        {
            "rows": parent_cohorts,
            "scope": "Observed equality of unchanged R1 D0 outputs across separate live runs; a mismatch requires diagnosis before cross-run selection comparisons.",
        },
    )
    for row in by_reference:
        aliases = lock.get("aliases", {})
        for label in ("FH-R1-native", "R1-D0"):
            baseline = next(
                (
                    item
                    for item in by_reference
                    if item["population"] == row["population"]
                    and item["reference"] == row["reference"]
                    and item["T"] == row["T"]
                    and item["method"] == aliases.get(label)
                ),
                None,
            )
            row[f"delta_vs_{label}"] = (
                row["t_mAP"] - baseline["t_mAP"]
                if baseline is not None
                and row["logical_unit_count"] == baseline["logical_unit_count"]
                else None
            )
    publication_rows = [
        {
            "asset": row.get("planned_name", row.get("name")),
            "status": row.get("status"),
            "bytes": row.get("bytes"),
            "sha256": row.get("sha256"),
            "url": row.get("url"),
        }
        for row in publication.get("assets", release_plan.get("assets", []))
    ]
    publication_rows.insert(
        0,
        {
            "asset": "Git branch/tag and required Release assets",
            "status": publication.get("publication_status", "NOT_PUBLISHED"),
            "url": publication.get("release_url", publication.get("branch_url")),
        },
    )
    status = {
        "execution_status": (
            "COMPLETE"
            if all(
                tasks[name]["status"] == "COMPLETE"
                for name in (
                    "DATA",
                    "BASELINE",
                    "HIGH_CONT",
                    "LOW_PAIR",
                    "REPAIR_R1",
                    "AUX",
                    "PERCEPTION",
                    "LOCK",
                    "ASSOC",
                    "CONFIRM",
                    "PROFILE",
                )
            )
            and replication.get("status") in {"COMPLETE", "SKIPPED_BUDGET"}
            and auxiliary_complete(auxiliary)
            else "PARTIAL_WITH_BLOCKERS"
        ),
        "perception_selection": perception_selection_label(perception),
        "repair_R1_selection": repairs["R1"]
        .get("selection", {})
        .get("selected_mode", "NOT_RUN"),
        "repair_P_selection": repairs["P"]
        .get("selection", {})
        .get("selected_mode", repairs["P"].get("status", "NOT_RUN")),
        "association_selection": (
            association.get("adopted_candidate", {}).get("method_id")
            if association.get("adopted_candidate")
            else (
                "KEEP_D0"
                if association.get("status") == "COMPLETE"
                else association.get("status", "NOT_RUN")
            )
        ),
        "replication_status": replication.get("replication_status", "NOT_RUN"),
        "pb_all_t_vs_native_fh": confirmation.get("deployment", {}).get(
            "tmap_all_T_vs_native_FH"
        ),
        "pb_all_t_vs_d0": confirmation.get("deployment", {}).get("tmap_all_T_vs_D0"),
        "resource_status": profile.get("resource_status", "UNCONFIRMED"),
        "publication_status": publication.get("publication_status", "NOT_PUBLISHED"),
    }
    local = confirmation.get("local_t2", {})
    aliases = lock.get("aliases", {})
    local_parent = local.get(aliases.get("FH-P-native"), {})
    local_r1 = local.get(aliases.get("FH-R1-native"), {})
    status["local_gain"] = (
        local_parent["mean_metrics"]["t_mAP"] > local_r1["mean_metrics"]["t_mAP"]
        if local_parent.get("coverage_status")
        == local_r1.get("coverage_status")
        == "COMPLETE"
        else None
    )
    if any(row["status"] == "PARENT_OUTPUT_DRIFT" for row in parent_cohorts):
        status["execution_status"] = "PARTIAL_WITH_BLOCKERS"
        status["parent_runtime_comparison"] = "PARENT_OUTPUT_DRIFT"
    # Protocol-named summaries retain the original component evidence rather
    # than inferring completion from the existence of a report.
    write_json(
        artifacts / "selection/CAL_SELECTION.json",
        {
            "perception": perception.get("cal_selected", {}),
            "repair": {
                parent: optional(artifacts / f"refiner/{parent}/CAL_SELECTION.json")
                for parent in ("R1", "P")
            },
            "association": association.get("cal_selected"),
            "source_files": [
                "selection/PERCEPTION.json", "refiner/R1/CAL_SELECTION.json",
                "refiner/P/CAL_SELECTION.json", "association/SELECTION.json",
            ],
        },
    )
    component_rows = [
        {
            "component": "perception",
            "method": row.get("method_id"),
            "step": row.get("optimizer_update"),
            "coverage": row.get("coverage_status"),
            "S_mean": row.get("S_mean"),
            "S_min": row.get("S_min"),
            "S_long": row.get("S_long"),
            "selected": row.get("method_id") == (perception.get("selected") or {}).get("method_id"),
        }
        for row in perception.get("evaluated", [])
    ]
    for parent in ("R1", "P"):
        repair = optional(artifacts / f"refiner/{parent}/SELECTION.json")
        component_rows.append({
            "component": "repair_" + parent,
            "method": repair.get("selected_mode", status["repair_" + parent + "_selection"]),
            "generic_gain": repair.get("base_positive"),
            "history_evidence_gain": repair.get("history_evidence_supported"),
        })
    component_rows.append({
        "component": "association",
        "method": (association.get("sel") or {}).get("method_id"),
        **{key: (association.get("sel") or {}).get(key)
           for key in ("S_mean", "S_min", "S_long", "coverage_status")},
        "selection": status["association_selection"],
    })
    write_csv(artifacts / "selection/SEL_COMPONENTS.csv", component_rows)
    write_json(
        artifacts / "confirmation/CONFIRMATION_SUMMARY.json",
        {
            "status": confirmation.get("status", "NOT_RUN"),
            "decision": confirmation.get("deployment", {}),
            "status_fields": status,
            "expected_pb_units": 129,
            "expected_pb_prefixes": 516,
        },
    )
    for relative, rows in (
        ("tables/TASKS.csv", task_rows),
        ("tables/TRAINING.csv", training),
        ("tables/PILOTS.csv", pilots),
        ("tables/CHECKPOINTS.csv", evaluation),
        ("refinement/PAIR_COVERAGE.csv", coverage_rows),
        ("refinement/REPAIR_DIAGNOSTICS.csv", diagnostics),
        ("tables/REPAIR.csv", repair_rows),
        ("association/CONFIG_RESULTS.csv", association_rows),
        ("confirmation/COVERAGE.csv", formal),
        ("confirmation/METRICS.csv", formal),
        ("confirmation/BY_REFERENCE.csv", by_reference),
        ("tables/PUBLICATION.csv", publication_rows),
    ):
        write_csv(artifacts / relative, rows)
    sections = [
        f"# Persist4D Perception Gain V2\n\n生成时间：{utc_now()}；代码：`{current_commit}`。这是实际文件的当前快照。\n\n所有 AP 与差值使用 0–1 标度；缺失值为 —。只有完整人口才能支持全覆盖胜出结论。\n\n```json\n{json.dumps(status, ensure_ascii=False, indent=2)}\n```"
    ]
    for title, rows, fields in (
        ("1. 任务与训练范围", task_rows, ["task", "status", "reason", "observed_from"]),
        (
            "训练终点（summary GPUh 仅归因，总账不重复相加）",
            training,
            ["recipe", "train_seed", "completed_updates", "status", "last_checkpoint"],
        ),
        (
            "2. H/L × C0/S-BAL 固定比较",
            pilots,
            [
                "recipe",
                "step",
                "coverage",
                "T2",
                "T3",
                "T4",
                "T5",
                "mean_delta_R1",
                "mean_same_step_delta_C0",
            ],
        ),
        (
            "3. 所有已评价感知检查点",
            evaluation,
            [
                "role",
                "recipe",
                "step",
                "coverage",
                "T2",
                "T3",
                "T4",
                "T5",
                "mean_delta_R1",
                "artifact",
            ],
        ),
        (
            "4. NEW/PAIR 与各自父模型",
            repair_rows,
            [
                "parent",
                "role",
                "method",
                "step",
                "coverage",
                "T2",
                "T3",
                "T4",
                "T5",
                "mean_delta_parent",
            ],
        ),
        (
            "修复数据与诊断",
            diagnostics,
            [
                "parent",
                "mode",
                "trained_updates",
                "loss",
                "post_diagnostic_status",
                "corrected_original_errors",
                "broken_original_correct",
            ],
        ),
        (
            "5. 固定 12 个关联配置",
            association_rows,
            [
                "config",
                "attempt",
                "status",
                "completed_units",
                "expected_units",
                "T2",
                "T3",
                "T4",
                "T5",
                "cpu_core_hours",
            ],
        ),
        (
            "6. 正式确认：各人口分别报告",
            formal,
            [
                "population",
                "method",
                "T",
                "eval_seed",
                "actual_units",
                "expected_units",
                "actual_references",
                "expected_references",
                "t_mAP",
                "coverage",
            ],
        ),
        (
            "7. 工程成本（未结算的运行占用未加入本表）",
            [{"quantity": key, "value": value} for key, value in cost.items()],
            ["quantity", "value"],
        ),
        (
            "8. 实际发布覆盖",
            publication_rows,
            ["asset", "status", "bytes", "sha256", "url"],
        ),
    ):
        sections.append(f"## {title}\n\n{table(rows, fields)}")
    sections.append(
        "## 选择与证据边界\n\n"
        + json.dumps(
            {
                "learning_rate": optional(
                    artifacts / "selection/LEARNING_RATE.json"
                ).get("learning_rate_label"),
                "auxiliary_pilots": {
                    name: {
                        **{
                            key: row.get(key)
                            for key in (
                                "status",
                                "reason",
                                "completed_global_step",
                                "budget",
                            )
                        },
                        "cal": [
                            {
                                "step": point.get("optimizer_update"),
                                "coverage": point.get("coverage_status"),
                            }
                            for point in row.get("cal", [])
                        ],
                    }
                    for name, row in auxiliary.get("arms", {}).items()
                },
                "full_promotion": optional(artifacts / "selection/FULL_PROMOTION.json"),
                "perception_failures": perception.get("failures"),
                "final_method": lock.get("final_method_id"),
                "association_CAL": association.get("cal_selected"),
                "association_SEL": association.get("sel"),
                "replication": replication.get("pipeline_replication"),
                "deployment": confirmation.get("deployment"),
                "profile": profile.get("status", "NOT_RUN"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    sections.append(
        "逐 reference 原始行见 `confirmation/BY_REFERENCE.csv`；LOCAL 各 seed 的原始结果见 `confirmation/LOCAL-T2/METHODS.json`。未实现 pooled reference-block bootstrap 时不报告均值 AP 的置信区间。时延只接受真实 live profile；训练吞吐、CPU IoU microbenchmark 与多头共享前向不代表部署时延。"
    )
    if parent_cohorts:
        sections.append(
            "## R1 跨运行一致性\n\n"
            + table(
                parent_cohorts, ["role", "status", "published_output_equal", "deltas"]
            )
            + "\n\n差异诊断与原始运行证据保留在 `foundation/`；同一次前向中的零残差一致性不能替代跨运行一致性。"
        )
    if local_population.get("metadata_correction"):
        sections.append(
            "## LOCAL 人口元数据更正\n\n"
            f"继承的 DATA_ROLES 写为 {local_population['declared_reference_count']} refs；原始 native T2 的 "
            f"{local_population['validation_sequence_count']} 条序列实际覆盖 {local_population['validation_reference_count']} refs。"
            "V1 bootstrap 将 ADDITIONAL reference 列表直接用作 LOCAL 列表。这里完整保留原始序列及 45/46/47 seeds，"
            "按实际 reference 数报告，并在 FINAL_LOCK 中绑定该元数据核对。"
            "核对只读取人口元数据，不含锁定前预测或评分；TRAIN overlap 为 0。"
            "详见 `foundation/LOCAL_POPULATION_AUDIT.json`；原始 DATA_ROLES 与 V1 历史文件保持不变。"
        )
    if rebaseline:
        sections.append(
            "## 父模型运行更正\n\n"
            f"固定 recipe 的 CAL/SEL 重跑状态为 {rebaseline.get('status')}；采用状态为 {rebaseline.get('adoption')}。"
            "原运行的差异原因仍为 UNDETERMINED，不能归因于已排除的线程数、零头或启动上下文。"
            "当前重跑已与修复器父模型逐字节一致；历史父模型上的关联结果保留但不参与当前排名，全部耗时仍计入预算。"
            + (
                "原始四项基线已保存在 `foundation/original-runtime/`，替换记录见 `foundation/BASELINE_SUPERSESSION.json`。"
                if supersession
                else "当前控制器停止前，重跑结果暂存于 `foundation/rebaseline-fixed-runtime/`。"
            )
        )
    interpretation = artifacts / "validation/FINAL_INTERPRETATION.md"
    if interpretation.is_file():
        sections.append(interpretation.read_text())
    (artifacts / "FINAL_REPORT.md").write_text("\n\n".join(sections) + "\n")
    commands = []
    for event in json_lines(artifacts / "EXECUTION_LOG.jsonl"):
        argv = event.get("argv")
        if argv:
            command = " ".join(
                (
                    '"$PERSIST4D_GAIN_V2_ROOT'
                    + arg.split("$PERSIST4D_GAIN_V2_ROOT", 1)[1]
                    + '"'
                    if isinstance(arg, str)
                    and arg.startswith("$PERSIST4D_GAIN_V2_ROOT")
                    else shlex.quote(str(arg))
                )
                for arg in argv
            )
            commands.append(
                {
                    "task": event.get("task"),
                    "command": command,
                    "exit_code": event.get("exit_code"),
                    "utc": event.get("start_utc", event.get("utc")),
                }
            )
    write_csv(artifacts / "tables/EXECUTED_COMMANDS.csv", commands)
    tests = optional(artifacts / "validation/FINAL_TESTS.json")
    audit = optional(artifacts / "validation/REQUIREMENT_AUDIT.json")
    handoff = [
        f"# V2 交接\n\n固定起点：`{config['parent_commit']}`；当前代码：`{current_commit}`；分支：`{config['branch']}`。\n\n实际实验代码见每个评价的 `execution_provenance` 与 `EXECUTION_LOG.jsonl`。资产 SHA/角色见 `INPUT_MANIFEST.json`、`DATA_ROLES.json`；本机绑定为 `$PERSIST4D_GAIN_V2_ROOT/assets.local.json`。源码映射见 `CODE_BINDINGS.md`。\n\n当前执行：`{status['execution_status']}`；候选：`{lock.get('final_method_id', 'NOT_LOCKED')}`；默认部署：`{confirmation.get('deployment', {}).get('default_method', 'UNCONFIRMED')}`。正式结果、每臂步数和失效范围见 `FINAL_REPORT.md` 与各原始 JSON/CSV。",
        "## 已执行命令\n\n命令逐条来自真实日志：`tables/EXECUTED_COMMANDS.csv`。恢复前保留失败结果，并先解决对应 reason；控制器拒绝相同依赖下盲目重试。\n\n```bash\nexport PERSIST4D_GAIN_V2_ROOT=/home/ww/persist4d_runs/perception_gain_v2\ncd /home/ww/paper5/.worktrees/persist4d-perception-gain-v2\nconda run -n persist4d python -m scripts.perception_gain_v2 status\nconda run -n persist4d python -m scripts.perception_gain_v2 report\n```\n\n已有控制器运行时不启动另一控制器；待其退出后，按下表恢复。标为 CODE_HANDLER_PENDING 的任务暂不能运行。",
        "## 检查与待完成证据\n\n"
        + json.dumps(
            {
                "final_test_record": tests or "NOT_RUN",
                "requirement_by_requirement_audit": audit or "NOT_COMPLETED",
                "publication_receipt": publication or "NOT_PUBLISHED",
                "settled_cost": cost,
            },
            ensure_ascii=False,
            indent=2,
        ),
        "## 可恢复训练\n\n"
        + table(
            [row for row in training if row["last_checkpoint"]],
            ["recipe", "completed_updates", "last_checkpoint", "status"],
        ),
        "## 未完成任务\n\n"
        + table(
            [
                {
                    "task": row["task"],
                    "status": row["status"],
                    "reason": row["reason"],
                    "next_command": (
                        None
                        if row["task"] in unimplemented
                        else f"conda run -n persist4d python -m scripts.perception_gain_v2 run --target {row['task']} --resume"
                        + (
                            f" --retry-blocked {row['task']}"
                            if row["status"] == "BLOCKED"
                            else ""
                        )
                    ),
                }
                for row in task_rows
                if row["status"]
                not in {
                    "COMPLETE",
                    "SKIPPED_BUDGET",
                    "NOT_APPLICABLE",
                    "EXCLUDED_NUMERICAL",
                }
            ],
            ["task", "status", "reason", "next_command"],
        ),
    ]
    (artifacts / "HANDOFF.md").write_text("\n\n".join(handoff) + "\n")
    result = {
        "status": "COMPLETE",
        "execution_status": status["execution_status"],
        "source_commit": current_commit,
        "report": "artifacts/perception_gain_v2/FINAL_REPORT.md",
        "handoff": "artifacts/perception_gain_v2/HANDOFF.md",
        "status_fields": status,
        "cost": cost,
        "note": "COMPLETE describes report generation only, not experiment or publication completion.",
    }
    write_json(artifacts / "REPORT_STATUS.json", result)
    return result
