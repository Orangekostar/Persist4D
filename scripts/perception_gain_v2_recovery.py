"""Run an explicit independent recovery without overwriting a live controller."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys
import time

from scripts.perception_gain_v2 import (
    PROJECT_ROOT,
    _free_gpus,
    append_event,
    file_hash,
    load_config,
    read_json,
    training_watchdog_deadline,
    utc_now,
    write_json,
)
from scripts.perception_gain_v2_config import executed_identity


def _rows(path: Path) -> list[dict]:
    return (
        [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if path.exists()
        else []
    )


def _child_cpu_seconds(pid: int) -> float:
    fields = Path(f"/proc/{pid}/stat").read_text().split()
    return sum(int(fields[index]) for index in (13, 14, 15, 16)) / os.sysconf(
        "SC_CLK_TCK"
    )


def run_recovery(
    *,
    task: str,
    attempt: str,
    reason: str,
    config_path: Path,
    external_root: Path,
    gpus: list[int],
) -> dict:
    config = load_config(config_path)
    artifacts = PROJECT_ROOT / config["artifact_root"]
    if task not in {"ASSOC", "REPAIR_R1", "REPAIR_P"}:
        raise ValueError("Independent recovery is limited to association and repair")
    if not attempt or any(
        char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        for char in attempt
    ):
        raise ValueError("Recovery attempt must be a simple unique identifier")
    if len(gpus) != (0 if task == "ASSOC" else 1) or not set(gpus).issubset(
        _free_gpus()
    ):
        raise ValueError("Recovery requires its exact number of idle A40 devices")
    state = read_json(artifacts / "RUN_STATE.json")
    if state["tasks"][task]["status"] not in {"BLOCKED", "PENDING"}:
        raise ValueError("Controller already owns or completed this task")
    tasks = external_root / "tasks"
    for previous_path in tasks.glob(f"{task}-*_INVOCATION.json"):
        previous = read_json(previous_path)
        if "end_utc" in previous or "pid" not in previous:
            continue
        proc = Path(f"/proc/{previous['pid']}/cmdline")
        if proc.exists():
            arguments = proc.read_bytes().replace(b"\x00", b" ").decode()
            if (
                "scripts.perception_gain_v2" in arguments
                and f"--target {task}" in arguments
            ):
                raise ValueError("An independent attempt of this task is still running")
    invocation = tasks / f"{task}-{attempt}_INVOCATION.json"
    result_path = tasks / f"{task}-{attempt}.json"
    if invocation.exists() or result_path.exists():
        raise ValueError("Recovery attempt already exists")
    # Preserve previous scientific results before the task writes its next attempt.
    relative = (
        "association"
        if task == "ASSOC"
        else f"refiner/{'R1' if task == 'REPAIR_R1' else 'P'}"
    )
    prior = artifacts / relative
    if prior.exists():
        shutil.copytree(
            prior, external_root / f"attempts/{task}-{attempt}/previous_public_evidence"
        )
    environment = dict(os.environ)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": ",".join(map(str, gpus)),
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
    )
    command = [
        sys.executable,
        "-m",
        "scripts.perception_gain_v2",
        "execute-task",
        "--target",
        task,
        "--config",
        str(config_path),
        "--external-root",
        str(external_root),
    ]
    event = {
        "event_id": f"{task}-{attempt}:{utc_now()}",
        "task": task,
        "scope": "V2",
        "recovery_reason": reason,
        "start_utc": utc_now(),
        "gpus": gpus,
        "gpu_count": len(gpus),
        "argv": command,
        "cwd": str(PROJECT_ROOT),
        "code": executed_identity(
            PROJECT_ROOT,
            [Path(__file__), PROJECT_ROOT / "scripts/perception_gain_v2.py"],
        ),
    }
    cpu_ledger = artifacts / "budget/ASSOCIATION_CPU.jsonl"
    cpu_before = sum(row["cpu_core_hours"] for row in _rows(cpu_ledger))
    usage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    started, started_unix = time.monotonic(), time.time()
    stopped = None
    with (tasks / f"{task}.log").open("a") as stream:
        child = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        event["pid"] = child.pid
        write_json(invocation, event)
        while child.poll() is None:
            now = time.time()
            if task == "ASSOC":
                try:
                    if (
                        cpu_before + _child_cpu_seconds(child.pid) / 3600
                        >= config["budget"]["association_cpu_core_hours"]
                    ):
                        stopped = "ASSOCIATION_CPU_BUDGET_EXHAUSTED"
                except FileNotFoundError:
                    pass
            else:
                if now > training_watchdog_deadline(
                    task, external_root, started_unix, 180
                ):
                    stopped = "NO_COMPLETED_PROGRESS_WITHIN_WATCHDOG"
                current = read_json(artifacts / "RUN_STATE.json")
                settled = {
                    row["event_id"]: row
                    for row in _rows(artifacts / "budget/LEDGER.jsonl")
                }
                charged = sum(row["gpu_hours"] for row in settled.values())
                live = sum(
                    max(
                        0.0,
                        now - dt.datetime.fromisoformat(row["start_utc"]).timestamp(),
                    )
                    * len(row["gpus"])
                    / 3600
                    for row in current["tasks"].values()
                    if row["status"] == "RUNNING"
                )
                own = (time.monotonic() - started) * len(gpus) / 3600
                if (
                    config["cumulative_gpu_hour_cap"] - charged - live - own
                    <= current["confirmation_reserve_gpu_hours"]
                ):
                    stopped = "PROTECTED_CONFIRMATION_RESERVE"
            if stopped:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                break
            time.sleep(5)
        exit_code = child.wait()
    elapsed = time.monotonic() - started
    actual = tasks / f"{task}.json"
    result = (
        read_json(actual)
        if exit_code == 0 and actual.exists()
        else {
            "status": "BLOCKED",
            "reason": stopped or f"Recovery child exited {exit_code}",
        }
    )
    write_json(result_path, result)
    event.update(
        {
            "end_utc": utc_now(),
            "exit_code": exit_code,
            "elapsed_seconds": elapsed,
            "gpu_hours": elapsed * len(gpus) / 3600,
            "measurement": "MEASURED_PROCESS_GPU_RESERVATION_WALLTIME",
            "result_path": str(result_path.relative_to(external_root)),
            "result_sha256": file_hash(result_path),
            "stop_reason": stopped,
        }
    )
    if task == "ASSOC":
        usage_after = resource.getrusage(resource.RUSAGE_CHILDREN)
        measured = (
            usage_after.ru_utime
            + usage_after.ru_stime
            - usage_before.ru_utime
            - usage_before.ru_stime
        ) / 3600
        recorded = sum(row["cpu_core_hours"] for row in _rows(cpu_ledger)) - cpu_before
        append_event(
            cpu_ledger,
            {
                "event_id": event["event_id"],
                "role": "RECOVERY_OVERHEAD_OR_INTERRUPTION",
                "cpu_core_hours": max(0.0, measured - recorded),
                "measurement": "RUSAGE_CHILDREN_MINUS_ALREADY_RECORDED_CONFIG_CPU",
                "measured_attempt_cpu_core_hours": measured,
            },
        )
    if gpus:
        append_event(artifacts / "budget/LEDGER.jsonl", event)
    append_event(artifacts / "EXECUTION_LOG.jsonl", event)
    append_event(artifacts / "RECOVERY_EVENTS.jsonl", event)
    write_json(invocation, event)
    return {
        "task": task,
        "status": result["status"],
        "exit_code": exit_code,
        "gpu_hours": event["gpu_hours"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task", choices=("ASSOC", "REPAIR_R1", "REPAIR_P"), required=True
    )
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument(
        "--config",
        dest="config_path",
        type=Path,
        default=PROJECT_ROOT / "configs/perception_gain_v2.yaml",
    )
    parser.add_argument("--external-root", type=Path, required=True)
    parser.add_argument("--gpus", nargs="*", type=int, default=[])
    result = run_recovery(**vars(parser.parse_args()))
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["exit_code"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
