"""Independent V2 tasks, local inputs, cumulative accounting and execution CLI."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
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
    "AUX": ("HIGH_CONT", "LOW_PAIR"),
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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("bootstrap", "run", "status", "report", "publish")
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
    parser.add_argument("--target", choices=("all", *TASKS), default="all")
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
    raise RuntimeError(
        "requested V2 task handler is not implemented yet; run remains IN_PROGRESS"
    )


if __name__ == "__main__":
    raise SystemExit(main())
