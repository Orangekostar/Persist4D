"""Preregistered CPU association replay on fresh V2 R1 live predictions."""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

from scripts.perception_gain_v2 import (
    PROJECT_ROOT,
    append_event,
    file_hash,
    read_json,
    write_json,
)
from scripts.perception_gain_v2_config import R1_SHA256, executed_identity
from scripts.perception_gain_v2_perception import baseline_candidate
from scripts.perception_gain_v2_selection import compare, gate, rank


def replay_configuration(
    config: dict, *, external_root: Path, association_config, role: str
) -> dict:
    import torch

    from scripts.perception_gain_foundation import _metric_class_mapping
    from scripts.replay_crosswindow_association import E2ReplayAccumulator

    artifacts = PROJECT_ROOT / config["artifact_root"]
    live_path = artifacts / f"foundation/{role}/D0.json"
    live = read_json(live_path)
    assets = read_json(external_root / "assets.local.json")
    binding = live.get("cache_binding", {})
    expected = {"CAL": 23, "SEL": 24, "PB": 129}[role]
    if (
        live.get("status") != "PASS"
        or binding.get("weight_hash") != R1_SHA256
        or binding.get("role") != role
        or binding.get("roles_sha256") != file_hash(artifacts / "DATA_ROLES.json")
        or binding.get("input_manifest_hash")
        != file_hash(artifacts / "data/STAGING_MANIFEST.json")
        or binding.get("point_order_transform")
        != "canonical_vertices/identity_geometry"
        or binding.get("eval_seed") != 45
        or binding.get("publisher") != "D0/LAST/lag1/mean"
        or len(live.get("replay_exports", [])) != expected
    ):
        raise ValueError(
            "Association requires complete, bound, fresh R1 live predictions"
        )
    code = executed_identity(
        PROJECT_ROOT,
        [
            PROJECT_ROOT / name
            for name in (
                "scripts/perception_gain_v2_association.py",
                "models/overlap_entity_association.py",
                "models/crosswindow_state.py",
                "scripts/crosswindow_cache.py",
                "scripts/replay_crosswindow_association.py",
                "scripts/task_memory_output.py",
                "scripts/system_comparison_metrics.py",
                "scripts/p6a_metrics.py",
            )
        ],
    )
    name = association_config.config_id
    output_path = artifacts / f"association/{role}/{name}.json"
    if output_path.exists():
        saved = read_json(output_path)
        if (
            saved.get("status") == "PASS"
            and saved.get("live_artifact_sha256") == file_hash(live_path)
            and saved.get("input_cache_binding") == binding
            and saved.get("execution_provenance", {}).get("relevant_source_digest")
            == code["relevant_source_digest"]
            and saved.get("candidate", {}).get("association_config")
            == dataclasses.asdict(association_config)
        ):
            return {**saved, "reused_evidence": True}
        raise ValueError(
            f"Existing association evidence needs explicit recovery: {name}/{role}"
        )
    accumulator = E2ReplayAccumulator(
        dataset_spec=assets["metric_dataset_spec"],
        class_mapping=_metric_class_mapping(Path(assets["metric_dataset_spec"])),
        checkpoint_sha256=R1_SHA256,
        source_commit=code["executed_code_commit"],
        data_role={"CAL": "DEV-CAL", "SEL": "DEV-SEL", "PB": "PROTOCOL-B"}[role],
        association_configs={name: association_config},
    )
    references = set(read_json(artifacts / "DATA_ROLES.json")["roles"][role])
    completed, failures = [], []
    started, cpu_started = time.monotonic(), time.process_time()
    for entry in live["replay_exports"]:
        reference = entry["path"]
        if not reference.startswith("external:"):
            raise ValueError("Replay file lacks an external artifact identity")
        path = external_root / reference.removeprefix("external:")
        if (
            not path.resolve().is_relative_to((external_root / "cache").resolve())
            or file_hash(path) != entry["sha256"]
        ):
            raise ValueError("Replay file identity differs from live producer export")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        payload_binding = payload.get("cache_binding", {})
        if (
            any(payload_binding.get(key) != value for key, value in binding.items())
            or payload["logical_unit_id"] != entry["logical_unit_id"]
            or payload_binding.get("reference") not in references
        ):
            raise ValueError(
                "Replay unit does not belong to the frozen live producer population"
            )
        try:
            accumulator.update(
                logical_unit_id=payload["logical_unit_id"],
                base=payload["base"],
                supplement=payload["supplement"],
            )
            completed.append(payload["logical_unit_id"])
        except (RuntimeError, ValueError, KeyError, IndexError) as error:
            failures.append(
                {"logical_unit_id": payload["logical_unit_id"], "reason": str(error)}
            )
            # A partial update may have entered accumulators. It cannot be scored.
            break
        finally:
            del payload
    finalized = (
        accumulator.finalize() if not failures and len(completed) == expected else None
    )
    rows = finalized["metric_rows"] if finalized else []
    if role == "PB":
        for row in rows:
            row["population_id"] = "protocol-b"
    pooled = {
        method: {
            int(row["T"]): row["t_mAP"]
            for row in rows
            if row["method"] == method and row["reference"] == "all"
        }
        for method in ("D0", name)
    }
    parent = baseline_candidate(artifacts, role)
    bridge = {
        **parent,
        "metrics": pooled["D0"],
        "coverage_status": "COMPLETE" if finalized else "INCOMPLETE",
    }
    difference = compare(bridge, parent)
    bridge_ok = difference["comparison_status"] == "COMPLETE" and all(
        abs(value) <= 1e-12 for value in difference["deltas"].values()
    )
    candidate = {
        "method_id": name,
        "recipe_id": "C0-L-s45",
        "optimizer_update": 0,
        "new_parameter_count": 0,
        "metrics": pooled[name],
        "coverage_status": "COMPLETE" if bridge_ok else "INCOMPLETE",
        "checkpoint_sha256": R1_SHA256,
        "association_config": dataclasses.asdict(association_config),
    }
    result = {
        "status": "PASS" if bridge_ok else "BLOCKED",
        "candidate": candidate,
        "data_role": role,
        "completed_units": completed,
        "expected_logical_units": expected,
        "failures": failures,
        "d0_bridge": difference,
        "metric_rows": rows,
        "elapsed_seconds": time.monotonic() - started,
        "cpu_core_hours": (time.process_time() - cpu_started) / 3600,
        "execution_provenance": code,
        "live_artifact_sha256": file_hash(live_path),
        "input_cache_binding": binding,
        "association_diagnostics": {
            key: value
            for key, value in (finalized or {}).items()
            if key != "metric_rows"
        },
    }
    write_json(output_path, result)
    print(f"Association {role} {name}: {result['status']}", flush=True)
    return result


def run_association(config: dict, *, external_root: Path) -> dict:
    from models.overlap_entity_association import preregistered_association_configs

    artifacts = PROJECT_ROOT / config["artifact_root"]
    baseline = baseline_candidate(artifacts, "CAL")
    configurations = preregistered_association_configs()
    measured, candidates, failures, pruned = [], [], {}, []
    ledger = artifacts / "budget/ASSOCIATION_CPU.jsonl"
    used = (
        sum(
            json.loads(line)["cpu_core_hours"]
            for line in ledger.read_text().splitlines()
            if line.strip()
        )
        if ledger.exists()
        else 0.0
    )
    limit = float(config["budget"]["association_cpu_core_hours"])
    for index, association in enumerate(configurations):
        # Prune only at configuration boundaries; preserve completed populations.
        projected = max(measured, default=0.0)
        if used >= limit or (projected and used + projected * (1 + 24 / 23) > limit):
            pruned = [value.config_id for value in configurations[index:]]
            break
        started = time.process_time()
        try:
            result = replay_configuration(
                config,
                external_root=external_root,
                association_config=association,
                role="CAL",
            )
            candidates.append(result["candidate"])
        except (OSError, RuntimeError, ValueError, KeyError) as error:
            failures[association.config_id] = str(error)
        elapsed = (time.process_time() - started) / 3600
        used += elapsed
        measured.append(elapsed)
        append_event(
            artifacts / "budget/ASSOCIATION_CPU.jsonl",
            {
                "config_id": association.config_id,
                "role": "CAL",
                "cpu_core_hours": elapsed,
                "cumulative_cpu_core_hours": used,
                "measurement": "PROCESS_CPU_TIME_ALL_THREADS",
            },
        )
    ranking = rank(candidates, baseline)
    choice = ranking[0] if ranking else None
    write_json(
        artifacts / "association/CAL_LOCK.json",
        {"selected": choice, "ranked": ranking, "pruned": pruned, "failures": failures},
    )
    sel = None
    if choice is not None and used + max(measured, default=0.0) * 24 / 23 <= limit:
        selected = next(
            value for value in configurations if value.config_id == choice["method_id"]
        )
        started = time.process_time()
        try:
            result = replay_configuration(
                config,
                external_root=external_root,
                association_config=selected,
                role="SEL",
            )
            sel = compare(result["candidate"], baseline_candidate(artifacts, "SEL"))
        except (OSError, RuntimeError, ValueError, KeyError) as error:
            failures["SEL"] = str(error)
        finally:
            elapsed = (time.process_time() - started) / 3600
            used += elapsed
            append_event(
                ledger,
                {
                    "config_id": selected.config_id,
                    "role": "SEL",
                    "cpu_core_hours": elapsed,
                    "cumulative_cpu_core_hours": used,
                    "measurement": "PROCESS_CPU_TIME_ALL_THREADS",
                },
            )
    elif choice is not None:
        failures["SEL"] = (
            "SKIPPED_BUDGET: remaining CPU allocation cannot cover fixed winner SEL"
        )
    adopted = (
        {**sel, "component_gate_passed": True}
        if sel is not None and gate(sel, mean=0.005, minimum=-0.001)
        else None
    )
    result = {
        "status": (
            "COMPLETE"
            if choice is not None
            and sel is not None
            and sel["comparison_status"] == "COMPLETE"
            and not failures
            else "BLOCKED"
        ),
        "search_status": (
            "SEARCH_BUDGET_PRUNED"
            if pruned
            else "COMPLETE" if len(ranking) == 12 else "PARTIAL"
        ),
        "cal_selected": choice,
        "sel": sel,
        "adopted_candidate": adopted,
        "cpu_core_hours": used,
        "pruned": pruned,
        "failures": failures,
    }
    write_json(artifacts / "association/SELECTION.json", result)
    return result
