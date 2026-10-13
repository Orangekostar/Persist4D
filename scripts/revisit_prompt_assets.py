"""Bound MPR-V1 assets and exact parent/delta reconstruction."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections import Counter
from pathlib import Path

from scripts.native_long_budget import merge_events, write_json_atomic

BASE = "6b1b779172afdeb51e2719a737cd05e7eeeca982"
PROJECT = Path(__file__).resolve().parents[1]
ARTIFACTS = PROJECT / "artifacts/revisit_prompt_v1"
NEW_PREFIXES = ("model.mpr_prompt.", "model.mpr_read.", "model.mpr_support.")
ENCODER_PREFIXES = ("model.backbone.model.embedding.", "model.backbone.model.enc.")
PARENT_RELATIVE = "artifacts/native_long_postselection_v1/deployment/E1-23760.pt"
ENCODER_SHA256 = "845ec7dec97a5fabff8fadb5d9858ac6734347b612d1a4b574213419c139de07"


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_output(*args):
    return subprocess.check_output(["git", *args], cwd=PROJECT, text=True).strip()


def local_assets():
    return read_json(PROJECT / "assets.local.json")


def validate_bound_file(path, expected, cache=None):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    stat = path.stat()
    prior = (cache or {}).get(str(path), {})
    if (prior.get("bytes") == stat.st_size and prior.get("mtime_ns") == stat.st_mtime_ns
            and prior.get("sha256") == expected):
        return prior
    if sha256(path) != expected:
        raise ValueError(f"bound asset byte identity differs: {path.name}")
    return {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": expected}


def load_parent_state(system, payload):
    import torch

    current = system.state_dict()
    parameters, buffers = dict(system.named_parameters()), dict(system.named_buffers())
    omitted = {k for k in parameters if k.startswith(ENCODER_PREFIXES)}
    encoder_buffers = {k for k in buffers if k.startswith(ENCODER_PREFIXES)}
    new = {k for k in current if k.startswith(NEW_PREFIXES)}
    expected_task = set(current) - omitted - new
    if (set(payload["omitted_encoder_parameter_names"]) != omitted
            or set(payload["included_encoder_buffers"]) != encoder_buffers
            or set(payload["state_dict"]) != expected_task):
        raise ValueError("parent task/buffer coverage differs; only MPR namespaces may be new")
    for key, tensor in payload["state_dict"].items():
        if tensor.shape != current[key].shape or tensor.dtype != current[key].dtype:
            raise ValueError(f"parent shape/dtype mismatch: {key}")
        current[key] = tensor
    system.load_state_dict(current, strict=True)
    actual = system.state_dict()
    if any(not torch.equal(actual[k].cpu(), tensor.cpu()) for k, tensor in payload["state_dict"].items()):
        raise ValueError("loaded parent tensor differs")
    return {"loaded_task_entries": len(expected_task), "omitted_encoder_entries": len(omitted),
            "included_encoder_buffers": sorted(encoder_buffers),
            "new_parameter_names": sorted(k for k in parameters if k.startswith(NEW_PREFIXES)),
            "new_state_entries": sorted(new), "status": "EXACT_PARENT_TASK_AND_BUFFER_COVERAGE"}


def parent_payload(assets):
    import torch

    validate_bound_file(assets["parent_bundle"], assets["hashes"]["parent"], assets.get("identity_cache"))
    payload = torch.load(assets["parent_bundle"], map_location="cpu", weights_only=True)
    if payload["schema"] != 1 or payload["metadata"]["arm"] != "E1" or payload["metadata"]["step"] != 23760:
        raise ValueError("parent must be the published E1-23760 task bundle")
    for relative, expected in payload["metadata"]["code"].items():
        original = subprocess.check_output(["git", "show", f"{BASE}:{relative}"], cwd=PROJECT)
        if hashlib.sha256(original).hexdigest() != expected:
            raise ValueError(f"parent original numeric identity differs from BASE: {relative}")
    return payload


def parent_config(assets, *, variant, seed):
    from omegaconf import OmegaConf

    payload = parent_payload(assets)
    for name, key in (("encoder_path", "encoder"), ("taxonomy_path", "taxonomy")):
        validate_bound_file(assets[name], assets["hashes"][key], assets.get("identity_cache"))
    config = OmegaConf.create(payload["config"])
    config.backbone.name = assets["encoder_path"]
    config.model.config.backbone.name = assets["encoder_path"]
    config.instance_metric.dataset = assets["taxonomy_path"]
    config.model._target_ = "models.revisit_prompt.RevisitPromptReScene"
    config.model.mpr_variant, config.model.mpr_seed = variant, seed
    config.model.return_query_features = True
    config.model.save_segment_info = False
    config.model.native_long_sampling = False
    config.model.native_long_feedback = False
    config.general.save_segment_info = False
    config.general.freeze = None  # Explicit whole-parent freeze is owned by the new system.
    config.general.seed = seed
    config.general.gpus = 1
    config.general.compile_model = False
    config.general.matmul_precision = "highest"
    config.general.frozen_encoder_eval = True
    config.data.batch_size, config.data.num_workers = 1, 0
    config.trainer.precision = "32-true"
    return config, payload


def load_revisit_bundle(path, *, assets=None):
    import torch
    from trainer.revisit_prompt_trainer import RevisitPromptTrainer

    assets = local_assets() if assets is None else assets
    delta = torch.load(path, map_location="cpu", weights_only=True)
    if delta["schema"] != 1 or delta["parent_sha256"] != assets["hashes"]["parent"]:
        raise ValueError("delta parent identity/schema differs")
    if (delta["encoder_sha256"] != assets["hashes"]["encoder"]
            or delta["taxonomy_sha256"] != assets["hashes"]["taxonomy"]):
        raise ValueError("delta encoder or taxonomy identity differs")
    for relative, expected in delta["numeric_sources"].items():
        if sha256(PROJECT / relative) != expected:
            raise ValueError(f"delta numerical source identity differs: {relative}")
    config, parent = parent_config(assets, variant=delta["variant"], seed=delta["adapter_seed"])
    system = RevisitPromptTrainer(config)
    load_parent_state(system, parent)
    expected = {k for k in system.state_dict() if k.startswith(NEW_PREFIXES)}
    if set(delta["state_dict"]) != expected:
        raise ValueError("delta misses an adapter parameter/buffer")
    current = system.state_dict()
    for key, value in delta["state_dict"].items():
        if value.shape != current[key].shape or value.dtype != current[key].dtype:
            raise ValueError(f"delta shape/dtype differs: {key}")
        current[key] = value
    system.load_state_dict(current, strict=True)
    return system


def resolve_assets():
    """Search only already referenced run roots and the published parent assets."""
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    if "Orangekostar/Persist4D" not in git_output("remote", "get-url", "origin"):
        raise ValueError("origin must be the authorized Orangekostar/Persist4D repository")
    git_output("merge-base", "--is-ancestor", BASE, "HEAD")
    prior_local = read_json(PROJECT / "assets.local.json") if (PROJECT / "assets.local.json").is_file() else {}
    historical_roots = [PROJECT.parent / "rescene-native-long-retrain-v1",
                        PROJECT.parent / "persist4d-perception-gain-v2"]
    history_root = Path(os.environ.get("RESCENE_NATIVE_LONG_ROOT", Path.home() / "persist4d_runs/native_long_retrain_v1"))
    source_path = history_root / "SOURCE_AND_INITIALIZATION.json"
    source = read_json(source_path)
    original_audit_path = PROJECT.parent / "persist4d_mpr_v1_runs/PARENT_ORIGINAL_LOADER_AUDIT.json"
    original_audit = read_json(original_audit_path) if original_audit_path.is_file() else {}
    cache = {**original_audit.get("identity_cache", {}), **prior_local.get("identity_cache", {})}
    manifest = read_json(PROJECT / "artifacts/native_long_postselection_v1/deployment/MODEL_ASSET_MANIFEST.json")
    parent_info = next(r for r in manifest["assets"] if r["name"] == "E1-23760.pt")
    candidates = [prior_local.get("parent_bundle"), os.environ.get("MPR_PARENT_BUNDLE"),
                  *[str(root / PARENT_RELATIVE) for root in historical_roots], str(PROJECT / PARENT_RELATIVE)]
    matching = []
    for candidate in sorted({str(Path(p).resolve()) for p in candidates if p}):
        path = Path(candidate)
        if path.is_file() and path.stat().st_size == parent_info["bytes"]:
            cache[str(path)] = validate_bound_file(path, parent_info["sha256"], cache)
            matching.append(candidate)
    if not matching:
        raise FileNotFoundError("no exact published E1-23760 entity; fetch its required LFS object")
    assets = {
        "schema": 1, "repo_root": str(PROJECT),
        "run_root": prior_local.get("run_root") or os.environ.get("MPR_RUN_ROOT") or str(PROJECT.parent / "persist4d_mpr_v1_runs"),
        "data_root": prior_local.get("data_root") or os.environ.get("MPR_DATA_ROOT") or source["assets"]["data_root"],
        "encoder_path": prior_local.get("encoder_path") or os.environ.get("CONCERTO_CHECKPOINT") or source["assets"]["concerto_pretrained"],
        "parent_bundle": matching[0],
        "taxonomy_path": str(PROJECT / "artifacts/native_long_postselection_v1/deployment/rio.yaml"),
        "population_path": str(PROJECT / "artifacts/native_long_retrain_v1/POPULATION.json"),
        "formal_population_path": str(PROJECT / "artifacts/native_long_postselection_v1/protocol/FORMAL_POPULATION.json"),
        "python": str(Path(source["installed_libraries"]["pytorch_lightning"]["source"]).parents[4] / "bin/python"),
        "device": prior_local.get("device", "cuda:0"),
        "physical_gpu_index": prior_local.get("physical_gpu_index", 0),
        "original_parent_loader_audit": str(original_audit_path),
        "historical_run_root": str(history_root),
        "selection_lock": str(ARTIFACTS / "CAL_SELECTION_LOCK.json"),
        "hashes": {"parent": parent_info["sha256"], "encoder": ENCODER_SHA256,
                   "taxonomy": manifest["taxonomy_sha256"]},
    }
    # Library roots come from the original verified environment, without installing a new stack.
    if not Path(assets["python"]).is_file():
        raise FileNotFoundError("bound original environment Python is unavailable")
    assets["pythonpath"] = [str(PROJECT), *sorted({str(Path(v["source"]).parents[1])
                           for k, v in source["installed_libraries"].items()
                           if k in {"concerto", "sonata", "stmetrics"}}),
                           str(Path(source["assets"]["data_root"]).parents[2] / "paper5/third_party/detectron2")]
    for name, key in (("encoder_path", "encoder"), ("taxonomy_path", "taxonomy")):
        cache[assets[name]] = validate_bound_file(assets[name], assets["hashes"][key], cache)
    for name, key in (("population_path", "population"), ("formal_population_path", "formal_population")):
        assets["hashes"][key] = sha256(assets[name])
    if not Path(assets["data_root"]).joinpath("processed/rio/train_database.yaml").is_file():
        raise FileNotFoundError("bound native rio training database is unavailable")
    cost = read_json(PROJECT / "artifacts/native_long_postselection_v1/evidence/COST_AUDIT.json")
    receipt = read_json(history_root / "publication/PUBLICATION_RECEIPT.json")
    cost_ledger = PROJECT / "artifacts/native_long_postselection_v1/evidence/COST_LEDGER_POSTSELECTION.jsonl"
    events = merge_events([[json.loads(line) for line in cost_ledger.read_text().splitlines() if line]])
    if (sha256(cost_ledger) != cost["ledger_sha256"]
            or cost["baseline_gpu_hours"] + sum(e["gpu_hours"] for e in events) != cost["total_gpu_hours"]
            or receipt["total_gpu_hours"] != cost["total_gpu_hours"]
            or receipt["live_gpu_hours"] != 0):
        raise ValueError("historical receipt and complete deduplicated budget sources differ")
    if (history_root / "ACTIVE_PROCESS.json").exists():
        raise ValueError("historical campaign still owns an active reservation; resolve it before new spending")
    assets["budget"] = {"lifetime_cap_gpu_hours": cost["cap_gpu_hours"],
                        "prior_gpu_hours": cost["total_gpu_hours"], "reserve_gpu_hours": 8.,
                        "new_cap_gpu_hours": min(78., max(0., cost["remaining_gpu_hours"] - 8.)),
                        "authorization": "user instruction; original cap and audited predecessor receipt",
                        "historical_cost_sha256": sha256(PROJECT / "artifacts/native_long_postselection_v1/evidence/COST_AUDIT.json"),
                        "historical_ledger_sha256": cost["ledger_sha256"]}
    assets["identity_cache"] = cache
    common_dir = Path(git_output("rev-parse", "--git-common-dir"))
    if not common_dir.is_absolute():
        common_dir = (PROJECT / common_dir).resolve()
    exclude = common_dir / "info/exclude"
    excluded = exclude.read_text() if exclude.exists() else ""
    if "/assets.local.json" not in excluded.splitlines():
        with exclude.open("a") as stream:
            stream.write("\n/assets.local.json\n")
    write_json_atomic(PROJECT / "assets.local.json", assets)
    Path(assets["run_root"]).mkdir(parents=True, exist_ok=True)
    population, formal = read_json(assets["population_path"]), read_json(assets["formal_population_path"])
    records = {role: sorted(population["evaluation_inputs"][role], key=lambda x: (x["horizon"], x["input_id"]))
               for role in ("CAL", "SEL")}
    for role in ("CAL", "SEL"):
        records[role] = [{**r, "role": role, "dataset_scope": "train"} for r in records[role]]
    records["PB"] = [{**r, "scan_ids": r["physical_scan_ids"], "dataset_scope": "validation"}
                     for r in formal["records"]["PB"]]
    data_manifest = {"schema": 1, "exposure": "HISTORICALLY_EXPOSED",
                     "source_population_sha256": assets["hashes"]["population"],
                     "formal_population_sha256": assets["hashes"]["formal_population"],
                     "reference_counts": population["counts"], "references": population["references"],
                     "records": records, "counts": {role: dict(Counter(str(r["horizon"]) for r in rows)) for role, rows in records.items()},
                     "stage_forwards": {role: sum(r["horizon"] for r in rows) for role, rows in records.items()},
                     "independent_references": {role: len({r["reference_id"] for r in rows}) for role, rows in records.items()}}
    write_json_atomic(ARTIFACTS / "DATA_MANIFEST.json", data_manifest)
    start = {"schema": 1, "status": "ASSETS_AND_PRIOR_BOUND", "base_commit": BASE,
             "branch": git_output("branch", "--show-current"),
             "instruction_sha256": sha256(ARTIFACTS / "EXECUTION_INSTRUCTION.md"),
             "asset_hashes": assets["hashes"], "budget": assets["budget"],
             "data_counts": data_manifest["counts"], "parent_seed": 45,
             "adapter_seeds": [146, 147], "original_parent_loader_status": original_audit.get("status", "NOT_YET_EXECUTED"),
             "data_scope": "TRAIN/CAL/SEL=train indices; PB=validation indices",
             "local_paths_public": False, "publication_status": "PENDING_EXTERNAL_VERIFICATION"}
    write_json_atomic(ARTIFACTS / "START_STATE.json", start)
    return assets


NUMERIC_FILES = (
    "models/rescene.py", "models/revisit_prompt.py", "models/revisit_prompt_modules.py",
    "models/revisit_prompt_state.py", "models/revisit_prompt_loss.py",
    "datasets/revisit_prompt_episode.py", "trainer/revisit_prompt_trainer.py",
    "scripts/revisit_prompt_assets.py", "scripts/revisit_prompt_runtime.py",
    "scripts/train_revisit_prompt.py",
    "scripts/evaluate_revisit_prompt.py", "scripts/select_revisit_prompt.py",
    "scripts/revisit_prompt_campaign.py", "conf/revisit_prompt/default.yaml",
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def numeric_sources():
    return {relative: sha256(PROJECT / relative) for relative in NUMERIC_FILES}


def validate_run_plan(plan, assets, sources, manifest):
    n = plan.get("N")
    if (plan.get("status") != "LOCKED" or type(n) is not int
            or n not in (400, 800, 1200, 1600)
            or plan.get("checkpoint_steps") != [n//4, n//2, 3*n//4, n]):
        raise ValueError("RUN_LOCK status, N or quarter checkpoints differ")
    if plan.get("budget") != assets["budget"]:
        raise ValueError("RUN_LOCK cannot change the bound lifetime budget")
    if plan.get("numeric_sources") != sources:
        raise ValueError("RUN_LOCK numerical source identity differs")
    if plan.get("expected_inputs") != manifest["counts"]:
        raise ValueError("RUN_LOCK evaluation population counts differ")
    return plan


def read_run_plan(assets):
    private = Path(assets["run_root"])/"RUN_LOCK.json"
    public = ARTIFACTS/"RUN_LOCK.json"
    if not private.is_file() or not public.is_file() or sha256(private) != sha256(public):
        raise ValueError("identical frozen runtime/public RUN_LOCK files are required")
    return validate_run_plan(read_json(private), assets, numeric_sources(),
                             read_json(ARTIFACTS/"DATA_MANIFEST.json"))


def validate_replication(variant, selection):
    replication = (selection or {}).get("replication", {})
    if not replication.get("eligible") or variant not in replication.get("variants_to_train", []):
        raise ValueError("seed147 family is not qualified by the main CAL selection lock")


def validate_evaluation_request(*, variant, seed, step, role, run_plan=None, selection=None):
    if variant not in {f"M{i}" for i in range(7)} or seed not in (146, 147):
        raise ValueError("unknown MPR variant or adapter seed")
    if type(step) is not int or step < 0 or role not in {"CAL", "SEL", "PB"}:
        raise ValueError("invalid evaluation role or checkpoint step")
    if variant == "M0":
        if step != 0 or seed != 146:
            raise ValueError("M0 is the fixed seed146 reference at step0")
    elif run_plan is None or step not in run_plan["checkpoint_steps"]:
        raise ValueError("trained evaluation requires a RUN_LOCK quarter checkpoint step")
    if role == "CAL":
        if seed == 147:
            validate_replication(variant, selection)
        return
    if selection is None or selection.get("status") not in {"COMPLETE", "EXPERIMENT_INCOMPLETE"}:
        raise ValueError("SEL/PB require the frozen selection lock")
    endpoints = [selection.get(key) for key in
                 ("control_star", "candidate_star", "candidate_descriptive_star")]
    if variant == "M0":
        return
    if not any(endpoint is not None and
               (endpoint["variant"], endpoint["adapter_seed"], endpoint["step"]) == (variant, seed, step)
               for endpoint in endpoints):
        raise ValueError("requested confirmation endpoint is not locked")


def require_charged_child(phase):
    event_id = os.environ.get("MPR_CHARGED_EVENT_ID", "")
    if not event_id.startswith(f"mpr:{phase}:"):
        raise ValueError(f"GPU execution requires a charged campaign {phase} child")


def configure_runtime(seed):
    import random
    import numpy as np
    import torch

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")
    torch.set_num_threads(2)


def build_system(assets, *, variant, seed):
    from datasets.native_long_dataset import isolated_rng
    from scripts.revisit_prompt_assets import parent_config, load_parent_state
    from trainer.revisit_prompt_trainer import RevisitPromptTrainer

    config, payload = parent_config(assets, variant=variant, seed=seed)
    with isolated_rng(seed):
        system = RevisitPromptTrainer(config)
    audit = load_parent_state(system, payload)
    return system, config, audit


def configuration_digest(config, assets):
    from omegaconf import OmegaConf

    logical = {assets[k]: f"<{k}>" for k in
               ("repo_root", "data_root", "encoder_path", "parent_bundle", "taxonomy_path")}
    def normalize(value):
        if isinstance(value, dict):
            return {k: normalize(v) for k, v in value.items()}
        if isinstance(value, list):
            return [normalize(v) for v in value]
        if isinstance(value, str):
            for path in sorted(logical, key=len, reverse=True):
                value = value.replace(path, logical[path])
        return value
    return digest(normalize(OmegaConf.to_container(config, resolve=True)))


def parent_tensor_digest(system):
    """Actual original parameters and buffers; adapter tensors are excluded."""
    hasher = hashlib.sha256()
    for name, value in sorted(system.state_dict().items()):
        if name.startswith(NEW_PREFIXES):
            continue
        tensor = value.detach().cpu().contiguous()
        hasher.update(name.encode())
        hasher.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        hasher.update(tensor.reshape(-1).view(__import__("torch").uint8).numpy().tobytes())
    return hasher.hexdigest()


def public_error(error, assets=None):
    message = str(error)
    paths = {value:key for key,value in {"user_home":str(Path.home()),"repo_root":str(PROJECT),**(assets or {})}.items()
             if isinstance(value,str) and value.startswith("/")}
    for path in sorted(paths,key=len,reverse=True):
        message = message.replace(path,f"<{paths[path]}>")
    return message
