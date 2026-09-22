"""Independent V2 tasks, local inputs, cumulative accounting and execution CLI."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import signal
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path

import yaml

from scripts.perception_gain_v2_config import (
    PARENT_COMMIT,
    R1_SHA256,
    content_hash,
    executed_identity,
    remaining_gpu_hours,
    resolve_recipe,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASKS = {
    "BIND": (),
    "DATA": ("BIND",),
    "BASELINE": ("DATA",),
    "HIGH_CONT": ("DATA",),
    "LOW_PAIR": ("DATA",),
    "REPAIR_R1": ("DATA",),
    "ASSOC": ("BASELINE",),
    "AUX": ("HIGH_CONT", "LOW_PAIR", "BASELINE"),
    "PERCEPTION": ("BASELINE", "AUX"),
    "REPAIR_P": ("PERCEPTION", "REPAIR_R1"),
    "LOCK": ("PERCEPTION", "REPAIR_R1", "REPAIR_P", "ASSOC", "BASELINE"),
    "REPLICATE": ("LOCK",),
    "CONFIRM": ("LOCK", "DATA"),
    "PROFILE": ("LOCK", "DATA"),
    "REPORT": ("REPLICATE", "CONFIRM", "PROFILE"),
    "PUBLISH": ("REPORT",),
}
TERMINAL = {
    "COMPLETE",
    "BLOCKED",
    "SKIPPED_BUDGET",
    "EXCLUDED_NUMERICAL",
    "NOT_APPLICABLE",
}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def append_event(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")
        stream.flush()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dependency_closure(target: str) -> tuple[str, ...]:
    if target == "all":
        return tuple(TASKS)
    if target == "core":
        return tuple(
            name
            for name in TASKS
            if name
            in {"BIND", "DATA", "BASELINE", "HIGH_CONT", "LOW_PAIR", "REPAIR_R1"}
        )
    needed = {target}
    for name in tuple(needed):
        if name not in TASKS:
            raise ValueError(f"unknown V2 task: {name}")

    def visit(name):
        for dependency in TASKS[name]:
            if dependency not in needed:
                needed.add(dependency)
                visit(dependency)

    visit(target)
    return tuple(name for name in TASKS if name in needed)


def ready_tasks(tasks: Mapping[str, Mapping], *, target: str) -> tuple[str, ...]:
    return tuple(
        name
        for name in dependency_closure(target)
        if tasks[name]["status"] == "PENDING"
        and all(tasks[dependency]["status"] in TERMINAL for dependency in TASKS[name])
    )


def load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text())
    if (
        config["parent_commit"] != PARENT_COMMIT
        or file_hash(PROJECT_ROOT / config["instruction"])
        != config["instruction_sha256"]
    ):
        raise ValueError("V2 instruction or parent identity differs")
    remaining_gpu_hours(float(config["cumulative_gpu_hour_cap"]), 0, [])
    return config


def stage_input_file(source: Path, destination: Path, *, copy_file: bool) -> dict:
    """Keep source content intact; a staged file must never point back to NFS."""
    source = source.resolve(strict=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    size = source.stat().st_size
    if destination.exists():
        if destination.is_symlink() or destination.stat().st_size != size:
            raise ValueError(f"staged input differs: {destination}")
        return {"bytes": size, "sha256": None, "transfer": "EXISTING_SIZE_VERIFIED"}
    if not copy_file and source.stat().st_dev == destination.parent.stat().st_dev:
        os.link(source, destination)
        return {"bytes": size, "sha256": None, "transfer": "LOCAL_HARDLINK_SAME_INODE"}
    digest = hashlib.sha256()
    temporary = destination.with_suffix(destination.suffix + ".copying")
    with source.open("rb") as incoming, temporary.open("wb") as output:
        for block in iter(lambda: incoming.read(4 * 1024 * 1024), b""):
            output.write(block)
            digest.update(block)
    if temporary.stat().st_size != size:
        raise OSError(f"incomplete input copy: {source}")
    temporary.replace(destination)
    return {
        "bytes": size,
        "sha256": digest.hexdigest(),
        "transfer": "COPIED_STREAM_SHA256",
    }


def _prior_costs(prior_root: Path) -> tuple[float, list[dict]]:
    state = read_json(PROJECT_ROOT / "artifacts/perception_gain_v1/RUN_STATE.json")
    recorded = state["budget_used"]
    prior = sum(
        float(value) for key, value in recorded.items() if key.endswith("gpu_hours")
    )
    records = [
        {
            "event_id": "V1-published-total",
            "gpu_hours": prior,
            "source": "repo:artifacts/perception_gain_v1/RUN_STATE.json",
            "measurement": "MIXED_MEASURED_AND_ESTIMATED_INTERRUPTION",
        }
    ]
    # V1 public state explicitly recorded smoke_and_recovery_gpu_hours=0.
    # Reconcile the final run of each known smoke directory and earlier resume slices.
    if not recorded.get("smoke_and_recovery_gpu_hours"):
        for summary_path in sorted(
            (prior_root / "smoke").glob("*/training/C0/run_summary.json")
        ):
            summary = read_json(summary_path)
            hours = float(summary["gpu_hours"])
            versions = sorted(
                (summary_path.parent / "training_metrics").glob("version_*")
            )
            for version in versions[:-1]:
                start = version / "hparams.yaml"
                checkpoints = list((version / "checkpoints").glob("*.ckpt"))
                if start.is_file() and checkpoints:
                    elapsed = (
                        max(p.stat().st_mtime for p in checkpoints)
                        - start.stat().st_mtime
                    )
                    hours += max(0, elapsed) * 2 / 3600
            records.append(
                {
                    "event_id": f"V1-smoke-{summary_path.relative_to(prior_root).parts[1]}",
                    "gpu_hours": hours,
                    "measurement": "SUMMARY_PLUS_ESTIMATED_EARLIER_SLICES",
                    "source": f"prior:{summary_path.relative_to(prior_root).as_posix()}",
                    "limitation": "Earlier slice occupancy estimated from hparams/checkpoint timestamps; setup outside these intervals is not measurable.",
                }
            )
    return sum(row["gpu_hours"] for row in records), records


def bootstrap(
    config: dict, *, config_path: Path, external_root: Path, prior_root: Path
) -> dict:
    artifacts = PROJECT_ROOT / config["artifact_root"]
    identity = {
        "parent_commit": PARENT_COMMIT,
        "instruction_sha256": config["instruction_sha256"],
        "config_sha256": file_hash(config_path),
    }
    state_path = artifacts / "RUN_STATE.json"
    if state_path.exists():
        state = read_json(state_path)
        if state["identity"] != identity:
            raise ValueError("existing V2 run identity differs; cannot overwrite")
        return state
    external_root.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    source_assets = read_json(prior_root / "assets.local.json")
    assets = {
        key: source_assets[key]
        for key in (
            "r1_checkpoint",
            "concerto_pretrained",
            "data_root",
            "rio_metadata",
            "metric_dataset_spec",
        )
    }
    expected = {
        "r1_checkpoint": R1_SHA256,
        "concerto_pretrained": "845ec7dec97a5fabff8fadb5d9858ac6734347b612d1a4b574213419c139de07",
        "scorer_checkpoint": "5bf58cdde393493daeb29bd023cb058db51bec56e76d39ba14ff68a6f911380c",
        "S_BAL_H_resume": "49062639ec7e1f38280c6be2a42be81fe913cd6e1169dff862b9fc8f551b20ce",
        "C0_H_0250": "362a8575fb0d011b079b60af4eb92f57fcb5734484230c76c468bb9718007b87",
        "C0_H_0750": "384884e3e86de79b5ca92661aa887b1f34612d2d8c90e93a66bbf70b87e30143",
    }
    assets.update(
        {
            "scorer_checkpoint": str(
                prior_root / "training/scorer/model/update=0500.ckpt"
            ),
            "S_BAL_H_resume": str(prior_root / "training/S-BAL/last.ckpt"),
            "C0_H_0250": str(prior_root / "training/C0/update=0250.ckpt"),
            "C0_H_0750": str(prior_root / "training/C0/update=0750.ckpt"),
            "S_BAL_H_config": str(prior_root / "training/S-BAL/resolved_config.yaml"),
        }
    )
    inputs = {}
    for key, digest in expected.items():
        path = Path(assets[key])
        if not path.is_file():
            inputs[key] = {"status": "MISSING", "expected_sha256": digest}
            continue
        if key == "r1_checkpoint":
            destination = external_root / "inputs/r1.ckpt"
            transfer = stage_input_file(path, destination, copy_file=True)
            assets[key] = str(destination)
            actual = transfer["sha256"] or file_hash(destination)
            path = destination
        else:
            actual = file_hash(path)
        if actual != digest:
            raise ValueError(f"input weight identity mismatch: {key}")
        inputs[key] = {
            "status": "VERIFIED",
            "bytes": path.stat().st_size,
            "sha256": actual,
            "logical_reference": f"asset:{key}",
        }
    roles = read_json(PROJECT_ROOT / "artifacts/perception_gain_v1/DATA_ROLES.json")
    train = set(roles["roles"]["TRAIN"])
    if any(
        train & set(roles["roles"][key])
        for key in ("CAL", "SEL", "PB", "LOCAL-T2", "ADDITIONAL")
    ):
        raise ValueError("TRAIN overlaps a frozen evaluation role")
    prior, prior_records = _prior_costs(prior_root)
    for record in prior_records:
        append_event(
            artifacts / "budget/LEDGER.jsonl",
            {**record, "scope": "PRIOR", "utc": utc_now()},
        )
    runtime = config["runtime"]
    recipes = {
        f"{variant}-{lr}-s{seed}": resolve_recipe(
            variant,
            learning_rate=lr,
            seed=seed,
            devices=runtime["perception_devices"],
            num_workers=runtime["workers_per_rank"],
            timeout_seconds=runtime["loader_timeout_seconds"],
        )
        for variant in ("C0", "S-BAL", "S-WORST", "Q-SEM", "A-OPEN")
        for lr in ("H", "L")
        for seed in (45, 46)
    }
    for name, recipe in recipes.items():
        write_json(artifacts / f"training/{name}/recipe.json", recipe)
    auth = {
        "release": "UNAVAILABLE",
        "git_read": "UNKNOWN",
        "git_push_dry_run": "UNKNOWN",
    }
    for name, command in (
        ("git_read", ["git", "ls-remote", "origin", f"refs/heads/{config['branch']}"]),
        (
            "git_push_dry_run",
            [
                "git",
                "push",
                "--dry-run",
                "origin",
                f"HEAD:refs/heads/{config['branch']}",
            ],
        ),
    ):
        probe = subprocess.run(
            command, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=30
        )
        auth[name] = "AVAILABLE" if probe.returncode == 0 else "UNAVAILABLE"
    if shutil.which("gh"):
        probe = subprocess.run(
            ["gh", "auth", "status"], capture_output=True, timeout=30
        )
        auth["release"] = "AVAILABLE" if probe.returncode == 0 else "UNAVAILABLE"
    elif os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN"):
        auth["release"] = "API_TOKEN_PRESENT_UNVERIFIED"
    state = {
        "schema_version": "perception-gain-v2-state-v1",
        "identity": identity,
        "execution_status": "IN_PROGRESS",
        "tasks": {name: {"status": "PENDING"} for name in TASKS},
        "prior_gpu_hours": prior,
        "v2_gpu_hours": 0.0,
        "remaining_gpu_hours": remaining_gpu_hours(
            config["cumulative_gpu_hour_cap"], prior, []
        ),
        "confirmation_reserve_gpu_hours": config["budget"]["confirmation_reserve"],
        "auth": auth,
    }
    state["tasks"]["BIND"] = {"status": "COMPLETE", "utc": utc_now(), "inputs": inputs}
    write_json(external_root / "assets.local.json", assets)
    write_json(
        external_root / "run.local.json",
        {"prior_root": str(prior_root), "project_root": str(PROJECT_ROOT)},
    )
    write_json(artifacts / "DATA_ROLES.json", roles)
    write_json(
        artifacts / "INPUT_MANIFEST.json",
        {
            "weights": inputs,
            "role_counts": {key: len(value) for key, value in roles["roles"].items()},
            "local_additional_overlap": sorted(
                set(roles["roles"]["LOCAL-T2"]) & set(roles["roles"]["ADDITIONAL"])
            ),
        },
    )
    write_json(
        artifacts / "RUN_CONFIG.json",
        {
            "identity": identity,
            "resolved_config": config,
            "code": executed_identity(
                PROJECT_ROOT,
                [Path(__file__), PROJECT_ROOT / "scripts/perception_gain_v2_config.py"],
            ),
            "prior_usage_records": prior_records,
        },
    )
    shutil.copyfile(
        PROJECT_ROOT / config["instruction"], artifacts / "EXECUTION_INSTRUCTION.md"
    )
    write_json(state_path, state)
    append_event(
        artifacts / "EXECUTION_LOG.jsonl",
        {
            "task": "BIND",
            "argv": [
                "python",
                "-m",
                "scripts.perception_gain_v2",
                "bootstrap",
                "--config",
                "configs/perception_gain_v2.yaml",
                "--external-root",
                "$PERSIST4D_GAIN_V2_ROOT",
            ],
            "cwd": "repo:.",
            "pid": os.getpid(),
            "utc": utc_now(),
            "exit_code": 0,
            "outputs": ["RUN_CONFIG.json", "INPUT_MANIFEST.json", "RUN_STATE.json"],
        },
    )
    return state


def prepare_local_data(config: dict, *, external_root: Path) -> dict:
    artifacts = PROJECT_ROOT / config["artifact_root"]
    assets = read_json(external_root / "assets.local.json")
    source_root = Path(assets["data_root"])
    local_root = external_root / "data"
    if (
        source_root == local_root
        and (artifacts / "data/STAGING_MANIFEST.json").is_file()
    ):
        return read_json(artifacts / "data/STAGING_MANIFEST.json")
    files: dict[str, dict] = {}
    manifests = {}
    for dataset in ("rio", "scannet"):
        directory = source_root / "processed" / dataset
        for manifest in sorted(directory.glob("*.yaml")):
            relative = manifest.relative_to(source_root).as_posix()
            manifests[relative] = file_hash(manifest)
            files.setdefault(relative, {"manifest": relative})
            if manifest.name.endswith("database.yaml") or manifest.name.startswith(
                "sequence_database"
            ):
                payload = yaml.load(manifest.read_text(), Loader=yaml.CSafeLoader)
                records = (
                    payload
                    if isinstance(payload, list)
                    else payload.values() if isinstance(payload, dict) else ()
                )
                for row in records:
                    if not isinstance(row, dict):
                        continue
                    for key in ("filepath", "instance_gt_filepath"):
                        value = row.get(key)
                        if isinstance(value, str) and value.replace(
                            "../../", ""
                        ).startswith("data/"):
                            relative_input = value.replace("../../", "").removeprefix(
                                "data/"
                            )
                            files.setdefault(relative_input, {"manifest": relative})
    needed = sum(
        (source_root / name).stat().st_size
        for name in files
        if (source_root / name).resolve().is_relative_to("/mnt/shared")
    )
    if needed + 40 * 1024**3 > shutil.disk_usage(external_root).free:
        raise OSError(
            "insufficient local space for referenced data plus checkpoint/cache reserve"
        )
    records = []
    started = time.monotonic()
    for name, source in sorted(files.items()):
        path = source_root / name
        record = stage_input_file(
            path,
            local_root / name,
            copy_file=path.resolve().is_relative_to("/mnt/shared"),
        )
        records.append(
            {
                "relative_path": name,
                **record,
                "source_manifest_sha256": manifests[source["manifest"]],
            }
        )
    link = PROJECT_ROOT / "data"
    if link.exists() or link.is_symlink():
        if not link.is_symlink() or link.resolve() != local_root.resolve():
            raise ValueError("worktree data path already points elsewhere")
    else:
        link.symlink_to(local_root, target_is_directory=True)
    assets["data_root"] = str(local_root)
    assets["metric_dataset_spec"] = str(local_root / "processed/rio/rio.yaml")
    write_json(external_root / "assets.local.json", assets)
    result = {
        "status": "COMPLETE",
        "files": records,
        "source_manifests": manifests,
        "file_count": len(records),
        "bytes": sum(row["bytes"] for row in records),
        "elapsed_seconds": time.monotonic() - started,
        "scope": "All files referenced by existing RIO/ScanNet split/sequence manifests; no split edits",
        "roles": {
            key: "INPUTS_LOCAL"
            for key in ("TRAIN", "CAL", "SEL", "PB", "LOCAL-T2", "ADDITIONAL")
        },
    }
    write_json(artifacts / "data/STAGING_MANIFEST.json", result)
    return result


def train_recipe(
    config: dict,
    *,
    recipe_id: str,
    endpoint: int,
    external_root: Path,
    import_resume: Path | None = None,
    legacy_config: Path | None = None,
) -> dict:
    artifacts = PROJECT_ROOT / config["artifact_root"]
    recipe_path = artifacts / f"training/{recipe_id}/recipe.json"
    recipe = read_json(recipe_path)
    run_dir = external_root / f"training/{recipe_id}"
    summary_path = run_dir / "run_summary.json"
    if summary_path.is_file():
        summary = read_json(summary_path)
        if (
            summary.get("recipe") == recipe
            and summary["completed_global_step"] == endpoint
            and summary["status"] == "COMPLETE"
        ):
            return summary
    resume = run_dir / "last.ckpt"
    resume = resume if resume.is_file() else import_resume
    command = [
        sys.executable,
        "-m",
        "scripts.train_perception_gain",
        "--variant",
        recipe["architecture_variant"],
        "--recipe-config",
        str(recipe_path),
        "--run-subdir",
        recipe_id,
        "--assets",
        str(external_root / "assets.local.json"),
        "--roles",
        str(artifacts / "DATA_ROLES.json"),
        "--external-root",
        str(external_root),
        "--stop-after-updates",
        str(endpoint),
    ]
    if resume is not None:
        command += ["--resume", str(resume)]
    if legacy_config is not None:
        command += ["--legacy-resume-config", str(legacy_config)]
    event = {
        "recipe_id": recipe_id,
        "argv": command,
        "cwd": str(PROJECT_ROOT),
        "start_utc": utc_now(),
        "endpoint": endpoint,
        "resume": str(resume) if resume else None,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    attempt = len(list(run_dir.glob("invocation-*.json")))
    invocation = run_dir / f"invocation-{attempt:02d}.json"
    with (run_dir / f"stdout-{attempt:02d}.log").open("w") as stream:
        process = subprocess.Popen(
            command, cwd=PROJECT_ROOT, stdout=stream, stderr=subprocess.STDOUT
        )
        event["pid"] = process.pid
        write_json(invocation, event)
        exit_code = process.wait()
    event.update({"end_utc": utc_now(), "exit_code": exit_code})
    write_json(invocation, event)
    if exit_code:
        raise RuntimeError(
            f"training {recipe_id} exited {exit_code}; see external:training/{recipe_id}/stdout-{attempt:02d}.log"
        )
    summary = read_json(summary_path)
    # Copy only portable, small training evidence; optimizer checkpoints stay external.
    destination = artifacts / f"training/{recipe_id}"
    for name in ("resolved_config.yaml", "run_plan.json", "run_summary.json"):
        content = (run_dir / name).read_text()
        content = content.replace(
            str(external_root), "$PERSIST4D_GAIN_V2_ROOT"
        ).replace(str(PROJECT_ROOT), "$PERSIST4D_REPO")
        (destination / name).write_text(content)
    for metrics in sorted((run_dir / "training_metrics").glob("version_*/metrics.csv")):
        shutil.copyfile(metrics, destination / f"metrics-{metrics.parent.name}.csv")
    write_json(
        destination / "checkpoint_manifest.json",
        {"checkpoints": summary["checkpoints"]},
    )
    return summary


def execute_task(task: str, config: dict, *, external_root: Path) -> dict:
    # Set before any head training creates a CUDA/cuBLAS context in this process.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    assets = read_json(external_root / "assets.local.json")
    artifacts = PROJECT_ROOT / config["artifact_root"]
    if task == "DATA":
        result = prepare_local_data(config, external_root=external_root)
        return {
            key: value
            for key, value in result.items()
            if key not in {"files", "source_manifests"}
        }
    if task == "BASELINE":
        from scripts.perception_gain_v2_evidence import run_baselines

        return run_baselines(config, external_root=external_root)
    if task == "REPAIR_R1":
        from scripts.perception_gain_v2_evidence import run_repair

        return run_repair(
            config,
            external_root=external_root,
            parent_id="R1",
            recipe=read_json(artifacts / "training/C0-L-s45/recipe.json"),
            parent_update=0,
            parent_checkpoint=None,
        )
    if task in {"AUX", "PERCEPTION"}:
        from scripts.perception_gain_v2_perception import run_aux, run_perception

        handler = run_aux if task == "AUX" else run_perception
        return handler(config, external_root=external_root)
    if task == "ASSOC":
        from scripts.perception_gain_v2_association import run_association

        return run_association(config, external_root=external_root)
    if task == "REPAIR_P":
        from scripts.perception_gain_v2_evidence import run_optional_parent_repair

        return run_optional_parent_repair(config, external_root=external_root)
    if task == "LOCK":
        from scripts.perception_gain_v2_lock import run_lock

        return run_lock(config, external_root=external_root)
    if task == "HIGH_CONT":
        summary = train_recipe(
            config,
            recipe_id="S-BAL-H-s45",
            endpoint=750,
            external_root=external_root,
            import_resume=Path(assets["S_BAL_H_resume"]),
            legacy_config=Path(assets["S_BAL_H_config"]),
        )
        return {
            "status": "COMPLETE",
            "training": {"S-BAL-H-s45": summary},
            "C0_H_weights": ["asset:C0_H_0250", "asset:C0_H_0750"],
        }
    if task == "LOW_PAIR":
        rows = {}
        for recipe_id in ("C0-L-s45", "S-BAL-L-s45"):
            try:
                rows[recipe_id] = train_recipe(
                    config,
                    recipe_id=recipe_id,
                    endpoint=750,
                    external_root=external_root,
                )
            except (OSError, RuntimeError, ValueError) as error:
                rows[recipe_id] = {"status": "BLOCKED", "reason": str(error)}
        return {
            "status": (
                "COMPLETE"
                if all(row["status"] == "COMPLETE" for row in rows.values())
                else "BLOCKED"
            ),
            "training": rows,
        }
    raise RuntimeError(
        f"V2 handler {task} is still under development; no experiment was performed"
    )


def _free_gpus() -> list[int]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return [
        int(parts[0])
        for line in result.stdout.splitlines()
        if len(parts := [value.strip() for value in line.split(",")]) == 4
        and "A40" in parts[1]
        and int(parts[2]) < 256
        and int(parts[3]) == 0
    ]


def training_watchdog_deadline(
    task: str, external_root: Path, started_unix: float, timeout_seconds: float
) -> float:
    """Only a task's current invocation may renew its completion watchdog."""
    patterns = {
        "HIGH_CONT": ("S-BAL-H-s45",),
        "LOW_PAIR": ("C0-L-s45", "S-BAL-L-s45"),
        "AUX": ("A-OPEN-*-s45", "Q-SEM-*-s45", "S-WORST-*-s45"),
        "PERCEPTION": ("*-s45",),
        "REPLICATE": ("*-s46",),
    }
    active = []
    for pattern in patterns.get(task, ()):
        for run_dir in (external_root / "training").glob(pattern):
            for path in run_dir.glob("invocation-*.json"):
                invocation = read_json(path)
                started = dt.datetime.fromisoformat(invocation["start_utc"]).timestamp()
                if started >= started_unix and "end_utc" not in invocation:
                    active.append((started, run_dir))
    if not active:
        # AUX/full tasks also perform live evaluation between training invocations.
        # Each completed unit flushes this task's log; other jobs cannot renew it.
        task_log = external_root / f"tasks/{task}.log"
        updated = task_log.stat().st_mtime if task_log.exists() else started_unix
        if task in {"REPAIR_R1", "REPAIR_P"}:
            parent = "R1" if task == "REPAIR_R1" else "P"
            progress_files = list(
                (external_root / f"training/refiner/{parent}").glob("*/last.ckpt")
            )
            progress_files.append(external_root / "cache/refiner/PROGRESS.json")
            updated = max(
                [updated]
                + [path.stat().st_mtime for path in progress_files if path.exists()]
            )
        return max(started_unix, updated) + 600
    started, run_dir = max(active)
    progress = run_dir / "PROGRESS.json"
    if progress.is_file() and progress.stat().st_mtime >= started:
        return progress.stat().st_mtime + timeout_seconds
    return started + 600


def run_tasks(
    config: dict,
    *,
    config_path: Path,
    external_root: Path,
    target: str,
    retry_blocked: str | None = None,
) -> dict:
    artifacts = PROJECT_ROOT / config["artifact_root"]
    state_path = artifacts / "RUN_STATE.json"
    state = read_json(state_path)
    ledger_path = artifacts / "budget/LEDGER.jsonl"
    settled = [
        json.loads(line)
        for line in ledger_path.read_text().splitlines()
        if line.strip()
    ]
    v2_events = [event for event in settled if event.get("scope") == "V2"]
    state["remaining_gpu_hours"] = remaining_gpu_hours(
        config["cumulative_gpu_hour_cap"],
        state["prior_gpu_hours"],
        v2_events,
    )
    state["v2_gpu_hours"] = (
        config["cumulative_gpu_hour_cap"]
        - state["prior_gpu_hours"]
        - state["remaining_gpu_hours"]
    )
    recovery_path = artifacts / "RECOVERY_EVENTS.jsonl"
    recovered = set(state.get("applied_recovery_events", []))
    for line in (
        recovery_path.read_text().splitlines() if recovery_path.exists() else ()
    ):
        event = json.loads(line)
        if event["event_id"] in recovered or event["exit_code"] != 0:
            continue
        result_path = external_root / event["result_path"]
        if file_hash(result_path) != event["result_sha256"]:
            raise ValueError("External recovery result identity changed")
        name = event["task"]
        previous = state["tasks"][name]
        if previous["status"] not in {"BLOCKED", "PENDING"}:
            raise ValueError("External recovery conflicts with a live/completed task")
        result = read_json(result_path)
        state["tasks"][name] = {
            **result,
            "previous_attempt": previous,
            "recovery_event": event,
            "gpu_hours": previous.get("gpu_hours", 0.0) + event["gpu_hours"],
        }
        recovered.add(event["event_id"])
    state["applied_recovery_events"] = sorted(recovered)
    if state["identity"]["config_sha256"] != file_hash(config_path):
        raise ValueError("V2 resume config changed")
    if any(row["status"] == "RUNNING" for row in state["tasks"].values()):
        raise RuntimeError(
            "existing RUNNING task must be reconciled against its live PID before resuming"
        )
    signature = content_hash(
        {
            "assets": read_json(external_root / "assets.local.json"),
            "code": executed_identity(PROJECT_ROOT, [Path(__file__)]),
        }
    )
    if retry_blocked:
        row = state["tasks"][retry_blocked]
        if row["status"] != "BLOCKED" or row.get("dependency_signature") == signature:
            raise ValueError(
                "retry requires a blocked task and changed dependency/fix signature"
            )
        state["tasks"][retry_blocked] = {"status": "PENDING", "previous_attempt": row}
    write_json(state_path, state)
    running = {}
    task_root = external_root / "tasks"
    task_root.mkdir(exist_ok=True)
    requirements = {name: 1 for name in TASKS}
    requirements.update(
        {
            name: config["runtime"]["perception_devices"]
            for name in ("HIGH_CONT", "LOW_PAIR", "AUX", "PERCEPTION", "REPLICATE")
        }
    )
    requirements.update(
        {name: 0 for name in ("BIND", "DATA", "ASSOC", "LOCK", "REPORT", "PUBLISH")}
    )
    start_cost = float(state["v2_gpu_hours"])
    # The protocol-specific order starts R1 repair independently on the spare GPU.
    order = (
        "BIND",
        "DATA",
        "REPAIR_R1",
        "HIGH_CONT",
        "LOW_PAIR",
        "BASELINE",
        "ASSOC",
        "AUX",
        "PERCEPTION",
        "REPAIR_P",
        "LOCK",
        "REPLICATE",
        "CONFIRM",
        "PROFILE",
        "REPORT",
        "PUBLISH",
    )
    while True:
        live_gpu_cost = sum(
            (time.monotonic() - item["start"]) * len(item["gpus"]) / 3600
            for item in running.values()
        )
        remaining = (
            config["cumulative_gpu_hour_cap"]
            - state["prior_gpu_hours"]
            - state["v2_gpu_hours"]
            - live_gpu_cost
        )
        ready = ready_tasks(state["tasks"], target=target)
        reserved = {gpu for item in running.values() for gpu in item["gpus"]}
        free = (
            [gpu for gpu in _free_gpus() if gpu not in reserved][
                : max(0, config["runtime"]["maximum_gpus"] - len(reserved))
            ]
            if any(requirements[name] for name in ready)
            else []
        )
        for name in order:
            if name not in ready:
                continue
            count = requirements[name]
            if count > len(free) or (
                count
                and sum(bool(item["gpus"]) for item in running.values())
                >= config["runtime"]["maximum_training_jobs"]
            ):
                continue
            reserve = (
                0
                if name in {"CONFIRM", "PROFILE", "REPORT", "PUBLISH"}
                else state["confirmation_reserve_gpu_hours"]
            )
            if count and remaining <= reserve:
                state["tasks"][name] = {
                    "status": "BLOCKED",
                    "reason": "Cumulative budget leaves only protected confirmation reserve",
                    "dependency_signature": signature,
                }
                write_json(state_path, state)
                continue
            gpus, free = free[:count], free[count:]
            argv = [
                sys.executable,
                "-m",
                "scripts.perception_gain_v2",
                "execute-task",
                "--config",
                str(config_path),
                "--external-root",
                str(external_root),
                "--target",
                name,
            ]
            environment = {
                **os.environ,
                "CUDA_VISIBLE_DEVICES": ",".join(map(str, gpus)),
                "OMP_NUM_THREADS": "2",
                "MKL_NUM_THREADS": "2",
                "OPENBLAS_NUM_THREADS": "2",
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            }
            if assets_scorer := read_json(external_root / "assets.local.json").get(
                "scorer_checkpoint"
            ):
                environment["PERSIST4D_Q_SEM_SCORER"] = assets_scorer
            stream = (task_root / f"{name}.log").open("a")
            process = subprocess.Popen(
                argv,
                cwd=PROJECT_ROOT,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            item = {
                "process": process,
                "stream": stream,
                "start": time.monotonic(),
                "gpus": gpus,
                "start_utc": utc_now(),
                "argv": argv,
            }
            running[name] = item
            state["tasks"][name] = {
                "status": "RUNNING",
                "pid": process.pid,
                "gpus": gpus,
                "start_utc": item["start_utc"],
                "dependency_signature": signature,
            }
            write_json(state_path, state)
        for name, item in list(running.items()):
            process = item["process"]
            limit = (
                0
                if name in {"CONFIRM", "PROFILE"}
                else state["confirmation_reserve_gpu_hours"]
            )
            if process.poll() is None and item["gpus"] and remaining <= limit:
                item["terminated_reason"] = "GPU budget/reserve limit"
            if process.poll() is None and name in {
                "HIGH_CONT",
                "LOW_PAIR",
                "AUX",
                "PERCEPTION",
                "REPLICATE",
                "REPAIR_R1",
                "REPAIR_P",
                "BASELINE",
            }:
                deadline = training_watchdog_deadline(
                    name,
                    external_root,
                    dt.datetime.fromisoformat(item["start_utc"]).timestamp(),
                    config["runtime"]["loader_timeout_seconds"],
                )
                if time.time() > deadline:
                    item["terminated_reason"] = (
                        "No completed batch/update within startup/progress watchdog"
                    )
            if process.poll() is None and "terminated_reason" in item:
                if "terminated_at" not in item:
                    os.killpg(process.pid, signal.SIGTERM)
                    item["terminated_at"] = time.monotonic()
                elif time.monotonic() - item["terminated_at"] > 30:
                    os.killpg(process.pid, signal.SIGKILL)
            exit_code = process.poll()
            if exit_code is None:
                continue
            item["stream"].close()
            elapsed = time.monotonic() - item["start"]
            hours = elapsed * len(item["gpus"]) / 3600
            result_path = task_root / f"{name}.json"
            result = (
                read_json(result_path)
                if result_path.is_file() and exit_code == 0
                else {
                    "status": "BLOCKED",
                    "reason": item.get(
                        "terminated_reason",
                        f"Task exited {exit_code}; see external:tasks/{name}.log",
                    ),
                }
            )
            state["tasks"][name] = {
                **result,
                "exit_code": exit_code,
                "gpu_hours": hours,
                "end_utc": utc_now(),
                "dependency_signature": signature,
            }
            event = {
                "event_id": f"{name}:{item['start_utc']}",
                "task": name,
                "gpu_hours": hours,
                "measurement": "MEASURED_PROCESS_GPU_RESERVATION_WALLTIME",
                "elapsed_seconds": elapsed,
                "gpu_count": len(item["gpus"]),
                "start_utc": item["start_utc"],
                "end_utc": utc_now(),
                "scope": "V2",
            }
            append_event(artifacts / "budget/LEDGER.jsonl", event)
            portable_argv = [
                part.replace(str(PROJECT_ROOT), "$PERSIST4D_REPO").replace(
                    str(external_root), "$PERSIST4D_GAIN_V2_ROOT"
                )
                for part in item["argv"]
            ]
            append_event(
                artifacts / "EXECUTION_LOG.jsonl",
                {
                    **event,
                    "argv": portable_argv,
                    "cwd": "repo:.",
                    "pid": process.pid,
                    "gpus": item["gpus"],
                    "exit_code": exit_code,
                    "outputs": [f"external:tasks/{name}.json"],
                },
            )
            state["v2_gpu_hours"] += hours
            state["remaining_gpu_hours"] = max(
                0,
                config["cumulative_gpu_hour_cap"]
                - state["prior_gpu_hours"]
                - state["v2_gpu_hours"],
            )
            write_json(state_path, state)
            del running[name]
        if not running:
            if not ready_tasks(state["tasks"], target=target):
                break
            if all(
                requirements[name] > len(_free_gpus())
                for name in ready_tasks(state["tasks"], target=target)
            ):
                break
        time.sleep(5)
    return {
        "tasks": {
            name: state["tasks"][name]["status"] for name in dependency_closure(target)
        },
        "incremental_gpu_hours": state["v2_gpu_hours"] - start_cost,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("bootstrap", "run", "status", "report", "publish", "execute-task"),
    )
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs/perception_gain_v2.yaml"
    )
    parser.add_argument(
        "--external-root",
        type=Path,
        default=Path(
            os.environ.get(
                "PERSIST4D_GAIN_V2_ROOT",
                Path.home() / "persist4d_runs/perception_gain_v2",
            )
        ),
    )
    parser.add_argument(
        "--prior-root",
        type=Path,
        default=Path.home() / "persist4d_runs/perception_gain_v1",
    )
    parser.add_argument("--target", choices=("all", "core", *TASKS), default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-blocked", choices=tuple(TASKS))
    args = parser.parse_args(argv)
    config = load_config(args.config)
    artifact_root = PROJECT_ROOT / config["artifact_root"]
    if args.command == "bootstrap":
        state = bootstrap(
            config,
            config_path=args.config,
            external_root=args.external_root,
            prior_root=args.prior_root,
        )
        print(
            json.dumps(
                {
                    "status": "COMPLETE",
                    "task": "BIND",
                    "remaining_gpu_hours": state["remaining_gpu_hours"],
                    "auth": state["auth"],
                }
            )
        )
        return 0
    state = read_json(artifact_root / "RUN_STATE.json")
    if args.command == "status":
        print(json.dumps(state, indent=2))
        return 0
    if args.command == "execute-task":
        result = execute_task(args.target, config, external_root=args.external_root)
        write_json(args.external_root / f"tasks/{args.target}.json", result)
        return 0
    if args.command == "run" and args.target == "DATA":
        result = prepare_local_data(config, external_root=args.external_root)
        state["tasks"]["DATA"] = {
            key: value
            for key, value in result.items()
            if key not in {"files", "source_manifests"}
        }
        write_json(artifact_root / "RUN_STATE.json", state)
        print(json.dumps(state["tasks"]["DATA"]))
        return 0
    if args.command == "run":
        print(
            json.dumps(
                run_tasks(
                    config,
                    config_path=args.config.resolve(),
                    external_root=args.external_root,
                    target=args.target,
                    retry_blocked=args.retry_blocked,
                )
            )
        )
        return 0
    raise RuntimeError(
        "requested V2 task handler is not implemented yet; run remains IN_PROGRESS"
    )


if __name__ == "__main__":
    raise SystemExit(main())
