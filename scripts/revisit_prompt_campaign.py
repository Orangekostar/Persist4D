"""Thin MPR campaign accounting; historical ledgers and their locks are immutable."""

from __future__ import annotations

import argparse
import copy
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

from scripts.native_long_budget import ledger_events, merge_events, write_json_atomic
from scripts.revisit_prompt_assets import (ARTIFACTS, PROJECT, local_assets, read_json, sha256,
    numeric_sources, digest, public_error)


def ensure_job(assets, command, *, status_path, phase, job_key, complete_status, training=False):
    """Reuse exact completed work; at most two retries, stop repeated errors."""
    root, status_path = Path(assets["run_root"]),Path(status_path)
    journal = root/"jobs"/(digest(job_key)+".json")
    attempts = read_json(journal)["attempts"] if journal.is_file() else []
    while True:
        old = read_json(status_path) if status_path.is_file() else {}
        if old:
            sources = old.get("numeric_sources",old.get("identity",{}).get("numeric_sources"))
            if sources != numeric_sources():
                raise ValueError("recorded job source identity differs; do not mix numerical versions")
        if old.get("status") == complete_status:
            return old
        if len(attempts) >= 3:
            raise RuntimeError("owned job exhausted its two permitted process retries")
        if len(attempts) >= 2 and attempts[-1]["error_fingerprint"] == attempts[-2]["error_fingerprint"]:
            raise RuntimeError("same runtime error repeated; no blind third restart")
        argv = list(command)
        if training and status_path.parent.joinpath("resume-latest.pt").is_file():
            if "--resume" not in argv:
                argv.append("--resume")
        elif not training and old:
            argv += ["--retry",str(old.get("retry",0)+1)]
        try:
            event = charged_job(root,argv,phase=phase,job_key=job_key)
            current = read_json(status_path) if status_path.is_file() else {}
            if current.get("status") != complete_status:
                raise RuntimeError("owned child exited without its required complete result")
            attempts.append(dict(status="COMPLETE",event_id=event["event_id"],error_fingerprint=None))
            write_json_atomic(journal,dict(job_key=job_key,attempts=attempts))
            return current
        except Exception as error:
            current = read_json(status_path) if status_path.is_file() else {}
            # The child's final exception identifies the cause, not changing event/PID text.
            message = current.get("error",public_error(error,assets))
            kind = current.get("error_type",type(error).__name__)
            failed = [e for e in ledger_events(root/"COST_LEDGER.jsonl") if e["job_key"]==job_key and e["status"]=="FAILED"]
            if failed:
                log = root/"logs"/(failed[-1]["event_id"].replace(":","-")+".log")
                if log.is_file():
                    with log.open("rb") as stream:
                        stream.seek(max(0,log.stat().st_size-16384))
                        lines = stream.read().decode(errors="replace").strip().splitlines()
                    if lines:
                        message = lines[-1]
                        kind = message.split(":",1)[0]
            attempts.append(dict(status="FAILED",error_fingerprint=digest([kind,message]),
                error_type=kind,error=public_error(Exception(message),assets)))
            write_json_atomic(journal,dict(job_key=job_key,attempts=attempts))


def child_command(assets, module, *args):
    return [assets["python"],"-m",module,*map(str,args)]


def evaluation_folder(assets, endpoint, role, repeat):
    return Path(assets["run_root"])/"evaluation"/(
        f"{endpoint['variant']}-s{endpoint['adapter_seed']}-u{endpoint['step']:06d}")/role/f"repeat{repeat}"


def evaluate_endpoint(assets, endpoint, role, *, selection_lock=None):
    for repeat in range(3):
        argv = child_command(assets,"scripts.evaluate_revisit_prompt","--variant",endpoint["variant"],
            "--seed",endpoint["adapter_seed"],"--step",endpoint["step"],"--role",role,"--repeat-index",repeat)
        if selection_lock:
            argv += ["--selection-lock",str(selection_lock)]
        phase = "P1" if role == "CAL" and endpoint["variant"] == "M0" else "P3" if role == "CAL" else "P4"
        folder = evaluation_folder(assets,endpoint,role,repeat)
        ensure_job(assets,argv,status_path=folder/"STATUS.json",phase=phase,
            job_key=f"eval:{endpoint['variant']}:s{endpoint['adapter_seed']}:u{endpoint['step']}:{role}:r{repeat}",
            complete_status="COMPLETE")


def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        for row in rows:
            values = {key:row.get(key) for key in fields}
            values = {key:json.dumps(value,sort_keys=True,allow_nan=False) if isinstance(value,(dict,list,tuple))
                      else value for key,value in values.items()}
            writer.writerow(values)
    os.replace(temporary,path)


def collect_evaluations(assets):
    """Preserve every actual repeat/checkpoint, including negative/empty results."""
    rows, references, identities = [],[],[]
    root = Path(assets["run_root"])
    for path in sorted((root/"evaluation").glob("*/*/repeat*/RESULTS.json")):
        result = read_json(path)
        if result["status"] != "COMPLETE":
            continue
        request = result["request"]
        if request["numeric_sources"] != numeric_sources():
            raise ValueError("completed evaluation source differs; preserve and revise the experiment lock")
        scope = {key:request[key] for key in ("variant","adapter_seed","parent_seed","step","role","repeat_index")}
        rows.extend({**row,"parent_sha256":assets["hashes"]["parent"]} for row in result["rows"])
        references.extend({**scope,**row} for row in result["references"])
        identities.extend({**scope,**row} for row in result["identity"])
    fields = ["variant","adapter_seed","parent_seed","parent_sha256","step","role","repeat_index","horizon",
              "status","unit","AP","current_AP","expected_inputs","completed_inputs","identity"]
    for role, name in (("CAL","CAL_ALL"),("SEL","SEL_LOCKED"),("PB","PB_LOCKED")):
        write_csv(ARTIFACTS/(name+".csv"),[r for r in rows if r["role"] == role],fields)
    write_csv(ARTIFACTS/"BY_REFERENCE.csv",references,["variant","adapter_seed","parent_seed","step","role",
        "repeat_index","reference_id","horizon","kind","metrics","evidence","sha256"])
    write_csv(ARTIFACTS/"IDENTITY_EVENTS_SUMMARY.csv",identities,["variant","adapter_seed","parent_seed","step",
        "role","repeat_index","reference_id","input_id","horizon","gap_recovery_recall","gap_recovery_accuracy",
        "normalized_id_switch_rate","gap_opportunities","correct_recoveries","recovery_attempts",
        "deployment_id_switches","identity_transition_opportunities","assignments","current_mask_diagnostics"])
    return rows


def clone_runtime(runtime):
    from models.task_memory_state import TaskMemoryState
    result = copy.copy(runtime)
    result.state = TaskMemoryState(*(t.clone() for t in runtime.state.tensors()),config=runtime.state.config)
    result.geometry = runtime.geometry.clone()
    result.history = list(runtime.history)
    return result


def frozen_backbone(system, stage):
    import torch
    data = copy.deepcopy(stage.batch[0])
    data.raw_coordinates = system._process_raw_coordinates(data)
    data.point2segment = [stage.batch[1][0]["point2segment"]]
    with torch.no_grad():
        return system.model.backbone(data)


def compare_raw_outputs(left, right):
    import torch
    if any(name not in left or name not in right for name in ("pred_logits","pred_masks")):
        raise ValueError("controlled comparison requires both task prediction fields")
    differences = {}
    for name in ("pred_logits","pred_masks","query_features"):
        if name not in left or name not in right:
            continue
        a = left[name] if isinstance(left[name],(list,tuple)) else [left[name]]
        b = right[name] if isinstance(right[name],(list,tuple)) else [right[name]]
        if len(a) != len(b) or any(x.shape != y.shape for x,y in zip(a,b)):
            raise ValueError("controlled raw-output shapes differ")
        diffs = [(x.detach().double()-y.detach().double()).abs() for x,y in zip(a,b)]
        if any(not torch.isfinite(d).all() for d in diffs):
            raise FloatingPointError("controlled output is not finite")
        differences[name] = dict(exact=all(x.dtype==y.dtype and torch.equal(x,y) for x,y in zip(a,b)),
                                max_abs=max((float(d.max()) for d in diffs if d.numel()),default=0.))
    # Original parent predictions have no MPR-only support-probability field.
    return dict(exact=all(row["exact"] for row in differences.values()),tensors=differences)


def controlled_probe_checks(system, loader, record, *, assets):
    """Real T01/T03: fixed backbone/RNG, separate normal and label-only copies."""
    import torch
    from dataclasses import replace
    from omegaconf import OmegaConf
    from datasets.native_long_dataset import isolated_rng,stable_seed
    from datasets.revisit_prompt_episode import RevisitEpisodeSpec
    from scripts.evaluate_persist4d_p6a import build_rio_class_mapper
    from scripts.revisit_prompt_assets import load_parent_state,parent_payload,parent_tensor_digest
    from scripts.revisit_prompt_runtime import RevisitEpisodeRuntime
    from trainer.trainer import InstanceSegmentation

    spec = RevisitEpisodeSpec.from_record(record)
    mapper = build_rio_class_mapper(loader.dataset("train"),foreground_class_count=system.model.num_classes-1)
    runtime = RevisitEpisodeRuntime(spec,class_count=system.model.num_classes,device="cuda:0")
    checks = {}
    for j, cpu_stage in enumerate(loader.stages(spec)):
        stage = cpu_stage.to("cuda:0")
        if j not in (0,spec.horizon-1):
            with torch.no_grad(),isolated_rng(stable_seed(9001,record["input_id"],"controlled",j),cuda_devices=(0,)):
                output = runtime.forward(system,stage)
                runtime.finish(system,output,stage,class_mapper=mapper)
            continue
        backbone = frozen_backbone(system,stage)
        seed = stable_seed(9001,record["input_id"],"controlled",j)
        def forward(local_stage, local_runtime=None):
            with torch.no_grad(),isolated_rng(seed,cuda_devices=(0,)):
                return (local_runtime or runtime).forward(system,copy.deepcopy(local_stage),backbone_outputs=backbone)
        normal = forward(stage)
        if j == 0:
            repeated_empty = forward(stage)
            checks["T01_normal_normal"] = compare_raw_outputs(normal,repeated_empty)
            del repeated_empty
            config = OmegaConf.create(OmegaConf.to_container(system.config,resolve=True))
            config.model._target_ = "models.rescene.ReScene"
            del config.model.mpr_variant
            del config.model.mpr_seed
            with isolated_rng(146,cuda_devices=(0,)):
                parent = InstanceSegmentation(config)
            load_parent_state(parent,parent_payload(assets))
            parent.requires_grad_(False).to("cuda:0").eval()
            data,targets,_ = copy.deepcopy(stage.batch)
            raw = parent._process_raw_coordinates(data)
            with torch.no_grad(),isolated_rng(seed,cuda_devices=(0,)):
                original = parent.model(data,point2segment=[targets[0]["point2segment"]],raw_coordinates=raw,
                    is_eval=True,stage_meta=stage.stage_meta,backbone_outputs=copy.deepcopy(backbone))
            checks["T01_parent_empty_history"] = compare_raw_outputs(original,normal)
            del original,parent
        if j == spec.horizon-1:
            # Change targets only; retain point2segment, raw data and StageMeta.
            def relabel(target):
                target = copy.deepcopy(target)
                for key in ("labels","ids","changes"):
                    if key in target:
                        target[key] = target[key]+1
                if "masks" in target:
                    target["masks"] = target["masks"].flip(0)
                return target
            changed = copy.deepcopy(stage)
            data,targets,names = changed.batch
            data.target_full = [relabel(t) for t in data.target_full]
            changed = replace(changed,batch=(data,[relabel(t) for t in targets],names),
                common_truth=relabel(changed.common_truth),loss_labels=changed.loss_labels+1)
            repeated,swapped = forward(stage),forward(changed)
            checks["T03_normal_normal"] = compare_raw_outputs(normal,repeated)
            checks["T03_normal_swapped_labels"] = compare_raw_outputs(normal,swapped)
            post_states = []
            for output,local_stage in ((normal,stage),(repeated,stage),(swapped,changed)):
                local = clone_runtime(runtime)
                with torch.no_grad():
                    local.finish(system,output,copy.deepcopy(local_stage),class_mapper=mapper)
                from models.revisit_prompt_modules import tensor_summary
                post_states.append(digest([tensor_summary(t) for t in (*local.state.tensors(),*local.geometry.tensors())]))
            checks["T03_state_normal_normal_exact"] = post_states[0]==post_states[1]
            checks["T03_state_normal_swapped_exact"] = post_states[0]==post_states[2]
            checks["T03_selected_history_slots"] = int(normal["mpr_selected_slots"][0].numel())
        # Continue one original path; interventions never update it.
        with torch.no_grad():
            runtime.finish(system,normal,copy.deepcopy(stage),class_mapper=mapper)
        del normal,backbone,stage
    checks["parent_tensor_sha256_after"] = parent_tensor_digest(system)
    return checks


def p5_worker(args, assets):
    from scripts.revisit_prompt_assets import (build_system,configure_runtime,configuration_digest,
        load_revisit_bundle,read_run_plan,require_charged_child,validate_evaluation_request)
    endpoint = dict(variant=args.variant,adapter_seed=args.seed,step=args.step)
    manifest = read_json(ARTIFACTS/"DATA_MANIFEST.json")
    all_records = sorted((r for r in manifest["records"]["CAL"] if r["horizon"] >= 3),key=lambda r:r["input_id"])
    panel_path = Path(assets["run_root"])/"p5/PANEL.json"
    if args.mode == "profile" and panel_path.is_file():
        panel_ids = read_json(panel_path)["input_ids"]
        records = [r for r in all_records if r["input_id"] in panel_ids]
    else:
        records = all_records if args.mode == "mechanism" else all_records[:6]
    if args.dry_run:
        return dict(status="DRY_RUN_NO_P5",mode=args.mode,endpoint=endpoint,input_ids=[r["input_id"] for r in records[:6]],
            panel_policy="first up to six sorted long CAL inputs with occupied history; profiles reuse this fixed panel")
    selection = read_json(ARTIFACTS/("REPLICATION_LOCK.json" if args.seed == 147 else "CAL_SELECTION_LOCK.json"))
    validate_evaluation_request(variant=args.variant,seed=args.seed,step=args.step,role="SEL",
        run_plan=read_run_plan(assets),selection=selection)
    require_charged_child("P5")
    import numpy as np
    import torch
    from datasets.native_long_dataset import isolated_rng,stable_seed
    from datasets.revisit_prompt_episode import RevisitEpisodeLoader,RevisitEpisodeSpec
    from models.revisit_prompt_modules import select_history_slots
    from scripts.evaluate_revisit_prompt import replay_episode,raw_prediction_fingerprint
    from scripts.evaluate_persist4d_p6a import build_rio_class_mapper
    from scripts.revisit_prompt_runtime import RevisitEpisodeRuntime

    configure_runtime(9001)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("P5 requires one visible CUDA device")
    if args.variant == "M0":
        system,config,_ = build_system(assets,variant="M0",seed=146)
        checkpoint_sha = assets["hashes"]["parent"]
    else:
        delta = Path(assets["run_root"])/"trajectories"/f"{args.variant}-s{args.seed}"/f"delta-{args.step:06d}.pt"
        system = load_revisit_bundle(delta,assets=assets)
        config,checkpoint_sha = system.config,sha256(delta)
    system.to("cuda:0").eval()
    loader = RevisitEpisodeLoader(config,assets["data_root"],read_json(assets["formal_population_path"]))
    provenance = dict(checkpoint_sha256=checkpoint_sha,config_hash=configuration_digest(config,assets),
        protocol_manifest_hash=sha256(ARTIFACTS/"DATA_MANIFEST.json"))
    rows = []
    folder = Path(assets["run_root"])/"p5"/f"{args.mode}-{args.variant}-s{args.seed}-u{args.step:06d}"
    from scripts.evaluate_revisit_prompt import prepare_evaluation_folder
    prepare_evaluation_folder(folder,Path(assets["run_root"]),args.retry)
    status = dict(status="RUNNING",numeric_sources=numeric_sources(),mode=args.mode,endpoint=endpoint,
        retry=args.retry,charge_event_id=os.environ["MPR_CHARGED_EVENT_ID"])
    write_json_atomic(folder/"STATUS.json",status)
    with torch.no_grad():
        if args.mode == "profile":
            for repeat in range(4):
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                for record in records:
                    with isolated_rng(stable_seed(9001,record["input_id"]),cuda_devices=(0,)):
                        episode = replay_episode(system,loader,record,provenance=provenance,measure_only=True)
                    rows.extend({**endpoint,"input_id":record["input_id"],"horizon":record["horizon"],
                        "repeat_index":repeat-1,"status":"WARMUP" if repeat==0 else "MEASURED",
                        **structure,**timing,"peak_allocated_bytes":torch.cuda.max_memory_allocated(),
                        "peak_reserved_bytes":torch.cuda.max_memory_reserved()}
                        for structure,timing in zip(episode["structures"],episode["timings"],strict=True))
                for row in rows:
                    if row["repeat_index"] == repeat-1:
                        row.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                   peak_reserved_bytes=torch.cuda.max_memory_reserved(),peak_scope="whole identical CAL panel")
                if repeat == 0 and args.variant != "M0":
                    status["deployment_reload"] = "EXACT_PARENT_ENCODER_DELTA_COVERAGE_AND_REAL_FORWARD"
        else:
            system.model.mpr_capture_probe = True
            mapper = build_rio_class_mapper(loader.dataset("train"),foreground_class_count=system.model.num_classes-1)
            for record in records:
                if len(rows) >= 6:
                    break
                spec = RevisitEpisodeSpec.from_record(record)
                runtime = RevisitEpisodeRuntime(spec,class_count=system.model.num_classes,device="cuda:0")
                samples = iter(loader.stages(spec))
                with isolated_rng(stable_seed(9001,record["input_id"]),cuda_devices=(0,)):
                    for j in range(spec.horizon-1):
                        stage = next(samples).to("cuda:0")
                        output = runtime.forward(system,stage)
                        runtime.finish(system,output,stage,class_mapper=mapper)
                    stage = next(samples).to("cuda:0")
                    slots = select_history_slots(runtime.state,0,spec.horizon-1)
                    if not runtime.state.occupied.any():
                        continue
                    row = dict(**endpoint,input_id=record["input_id"],horizon=spec.horizon,selected_slots=int(slots.numel()))
                    if slots.numel() < 2:
                        rows.append({**row,"status":"NOT_AVAILABLE_FEWER_THAN_TWO_HISTORY_SLOTS"})
                        continue
                    backbone = frozen_backbone(system,stage)
                    rng_seed = stable_seed(9001,record["input_id"],"history-content-intervention")
                    outputs = []
                    for permutation in (None,[torch.arange(slots.numel(),device=slots.device).roll(1)]):
                        local = clone_runtime(runtime)
                        with isolated_rng(rng_seed,cuda_devices=(0,)):
                            output = local.forward(system,copy.deepcopy(stage),backbone_outputs=backbone,
                                history_permutation=permutation)
                        finished = local.finish(system,output,copy.deepcopy(stage),class_mapper=mapper)
                        outputs.append(dict(fingerprint=raw_prediction_fingerprint(output),
                            context=output["mpr_pre_mask_probe"],candidate_count=int(finished["official"].pred_scores.numel()),
                            candidate_score_sum=float(finished["official"].pred_scores.sum())))
                    aligned = outputs[0]["context"] == outputs[1]["context"]
                    rows.append({**row,"status":"CONTROLLED" if aligned else "CONFOUNDED", "outputs":outputs,
                        "raw_prediction_changed":outputs[0]["fingerprint"] != outputs[1]["fingerprint"]})
                    del stage,backbone,output,finished
            panel_ids = [r["input_id"] for r in rows] or [r["input_id"] for r in all_records[:6]]
            panel = dict(input_ids=panel_ids,endpoint=endpoint,numeric_sources=numeric_sources(),
                         history_eligible_inputs=len(rows),fallback_no_history=not bool(rows))
            if panel_path.is_file() and read_json(panel_path) != panel:
                raise ValueError("P5 fixed profile panel changed on retry")
            write_json_atomic(panel_path,panel)
    status.update(status="COMPLETE",row_count=len(rows))
    write_json_atomic(folder/"RESULTS.json",dict(rows=rows,**status))
    write_json_atomic(folder/"STATUS.json",status)
    return status


def p0_inputs(population):
    from datasets.native_long_dataset import canonical_orders

    references = sorted((r for r in population["references"] if r["role"] == "TRAIN"),key=lambda r:r["reference_id"])
    records = []
    for h in (2,5):
        eligible = [r for r in references if r["Tmax"] >= h][:2]
        if len(eligible) != 2:
            raise ValueError("P0 needs two bound TRAIN inputs at each required horizon")
        for ref in eligible:
            order = canonical_orders(ref)[0][:h]
            records.append(dict(input_id=f"P0:T{h}:{ref['reference_id']}:order0:start0",role="TRAIN",
                dataset_scope="train",domain="rio",reference_id=ref["reference_id"],horizon=h,
                scan_ids=[ref["scan_ids"][i] for i in order],scan_indices=[ref["scan_indices"][i] for i in order]))
    return sorted(records,key=lambda r:(r["reference_id"],r["input_id"]))


def probe_worker(args, assets):
    from scripts.revisit_prompt_assets import build_system, configure_runtime, configuration_digest, digest, numeric_sources, require_charged_child

    input_path = Path(assets["run_root"])/"preflight/P0_INPUTS.json"
    records = read_json(input_path)["records"] if input_path.is_file() else p0_inputs(read_json(assets["population_path"]))
    record = next(r for r in records if r["input_id"] == args.input_id)
    if args.dry_run:
        return dict(status="DRY_RUN_NO_GPU_PROBE",input_id=args.input_id,repeat_index=args.repeat_index)
    require_charged_child("P0")
    import numpy as np
    import torch
    from datasets.native_long_dataset import isolated_rng, stable_seed
    from datasets.revisit_prompt_episode import RevisitEpisodeLoader
    from scripts.evaluate_revisit_prompt import replay_episode,official_compute_nullable
    from scripts.evaluate_revisit_prompt import prepare_evaluation_folder
    from scripts.p6a_metrics import OfficialMetricAccumulator

    configure_runtime(9001)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("P0 requires one visible CUDA device")
    folder = Path(assets["run_root"])/"preflight/probes"/digest(args.input_id)/f"repeat{args.repeat_index}"
    prepare_evaluation_folder(folder,Path(assets["run_root"]),args.retry)
    write_json_atomic(folder/"STATUS.json",dict(status="INITIALIZING",numeric_sources=numeric_sources(),
        retry=args.retry,charge_event_id=os.environ["MPR_CHARGED_EVENT_ID"]))
    system,config,_ = build_system(assets,variant="M4",seed=146)
    system.to("cuda:0").eval()
    system.model.mpr_capture_probe = True
    loader = RevisitEpisodeLoader(config,assets["data_root"],read_json(assets["formal_population_path"]))
    provenance = dict(checkpoint_sha256=assets["hashes"]["parent"],config_hash=configuration_digest(config,assets),
                      protocol_manifest_hash=assets["hashes"]["population"])
    with torch.no_grad(),isolated_rng(stable_seed(9001,args.input_id),cuda_devices=(0,)):
        episode = replay_episode(system,loader,record,provenance=provenance)
    metrics = {}
    for kind in ("long","current"):
        mode = "strict_online" if kind == "long" else "raw_local"
        metric = OfficialMetricAccumulator(mode=mode,dataset_spec=assets["taxonomy_path"])
        metric.update(episode["prediction" if kind == "long" else "current_prediction"],
                      episode["target" if kind == "long" else "current_target"])
        metrics[kind] = official_compute_nullable(metric)
    arrays = {}
    for kind in ("long","current"):
        prediction = episode["prediction" if kind == "long" else "current_prediction"]
        arrays[kind+"_mask_shape"] = np.array(prediction["pred_masks"].shape,dtype=np.int64)
        arrays[kind+"_packed_masks"] = np.packbits(prediction["pred_masks"].numpy(),axis=0,bitorder="little")
        arrays[kind+"_scores"] = prediction["pred_scores"].numpy()
        arrays[kind+"_classes"] = prediction["pred_classes"].numpy()
    np.savez_compressed(folder/"candidates.npz",**arrays)
    checks = None
    if record == records[0] and args.repeat_index == 0:
        from scripts.revisit_prompt_assets import parent_tensor_digest
        before = parent_tensor_digest(system)
        checks = controlled_probe_checks(system,loader,record,assets=assets)
        checks["parent_unchanged"] = before == checks["parent_tensor_sha256_after"]
        if not checks["parent_unchanged"]:
            raise ValueError("controlled P0 inference changed the frozen parent")
    result = dict(status="COMPLETE_UNTRAINED_NUMERICAL_PROBE",input_id=args.input_id,record=record,
        repeat_index=args.repeat_index,numeric_sources=numeric_sources(),metrics=metrics,
        structures=episode["structures"],identity=episode["identity"],controlled_checks=checks,
        candidate_sha256=sha256(folder/"candidates.npz"))
    write_json_atomic(folder/"RESULTS.json",result)
    write_json_atomic(folder/"STATUS.json",dict(status=result["status"],numeric_sources=numeric_sources(),
        retry=args.retry,charge_event_id=os.environ["MPR_CHARGED_EVENT_ID"]))
    return dict(status=result["status"],input_id=args.input_id,repeat_index=args.repeat_index)


def forecast_cost_plan(budget, pilots, evaluation, manifest):
    """One largest admissible N; evaluation populations never shrink with N."""
    variants = [f"M{i}" for i in range(1,7)]
    if set(pilots) != set(variants):
        raise ValueError("cost admission needs all six measured adapter pilots")
    values = [pilots[v][key] for v in variants for key in ("update_seconds","startup_seconds")]
    values += [evaluation[key] for key in ("seconds_per_stage","seconds_per_input","startup_seconds")]
    if any(not math.isfinite(float(value)) or value < 0 for value in values):
        raise ValueError("measured cost estimates must be finite and nonnegative")
    if any(pilots[v]["update_seconds"] <= 0 for v in variants) or evaluation["seconds_per_stage"] <= 0:
        raise ValueError("real optimizer and evaluation costs must be positive")
    b = float(budget["new_cap_gpu_hours"])
    if not 0 <= b <= 78:
        raise ValueError("new research pool exceeds the authorized cap")
    def evaluation_seconds(role):
        counts = manifest["counts"][role]
        points = sum(int(h)*count for h,count in counts.items())
        inputs = sum(counts.values())
        return evaluation["startup_seconds"]+points*evaluation["seconds_per_stage"]+inputs*evaluation["seconds_per_input"]
    caps = dict(P0=4*b/78,P1=4*b/78,
                P3=96*1.5*evaluation_seconds("CAL")/3600,
                P4=15*1.5*(evaluation_seconds("SEL")+evaluation_seconds("PB"))/3600,
                P5=4*b/78)
    available = b-sum(caps.values())
    caps["P2"] = max(0.,available)
    estimates, worst = {}, {}
    for n in (400,800,1200,1600):
        costs = {v:1.25*(n*pilots[v]["update_seconds"]+pilots[v]["startup_seconds"])/3600 for v in variants}
        replicas = sorted(variants,key=lambda v:(-costs[v],v))[:2]
        estimates[str(n)] = sum(costs.values())+sum(costs[v] for v in replicas)
        worst[str(n)] = replicas
    accepted = [n for n in (400,800,1200,1600) if estimates[str(n)] <= available]
    n = max(accepted) if accepted else None
    return dict(status="LOCKABLE" if n else "BLOCKED_FULL_MATRIX_BUDGET",N=n,
        checkpoint_steps=[n//4,n//2,3*n//4,n] if n else [],
        phase_caps_gpu_hours=caps,cap_P2_available_gpu_hours=available,
        training_estimates_gpu_hours=estimates,worst_replication_variants=worst,
        measured_pilots=pilots,measured_evaluation=evaluation,
        minimum_additional_gpu_hours=max(0.,estimates["400"]-available),
        forecast_policy="training max six timed updates x1.25; evaluation full populations x1.5; startup included")


def available_budget(root, plan, *, phase):
    budget, caps = plan["budget"], plan["phase_caps_gpu_hours"]
    prior, lifetime, new, reserve = (float(budget[k]) for k in
        ("prior_gpu_hours", "lifetime_cap_gpu_hours", "new_cap_gpu_hours", "reserve_gpu_hours"))
    if (not all(math.isfinite(x) and x >= 0 for x in (prior, lifetime, new, reserve))
            or new > min(78., max(0., lifetime-prior-reserve))+1e-9
            or set(caps) != {f"P{i}" for i in range(6)}
            or any(not math.isfinite(float(v)) or v < 0 for v in caps.values())
            or sum(caps.values()) > new+1e-9 or phase not in caps):
        raise ValueError("campaign phase caps exceed the authorized lifetime balance")
    events = merge_events([ledger_events(Path(root)/"COST_LEDGER.jsonl")])
    used = sum(event["gpu_hours"] for event in events)
    phase_used = sum(event["gpu_hours"] for event in events if event["phase"] == phase)
    return max(0., min(new-used, float(caps[phase])-phase_used,
                       lifetime-prior-reserve-used))


def _job_plan(root, phase, assets):
    lock = Path(root)/"RUN_LOCK.json"
    if lock.is_file():
        plan = read_json(lock)
        if plan["status"] != "LOCKED":
            raise ValueError("research RUN_LOCK is not frozen")
        public = ARTIFACTS/"RUN_LOCK.json"
        if public.is_file() and sha256(public) != sha256(lock):
            raise ValueError("public and runtime RUN_LOCK identities differ")
    else:
        if phase not in {"P0", "P1"}:
            raise ValueError("research jobs require the measured, frozen RUN_LOCK")
        import yaml
        initial = yaml.safe_load((PROJECT/"conf/revisit_prompt/default.yaml").read_text())
        fraction = assets["budget"]["new_cap_gpu_hours"]/78.
        plan = dict(status="PREFLIGHT_ONLY_RESEARCH_NOT_ADMITTED", budget=assets["budget"],
                    phase_caps_gpu_hours={k: v*fraction for k,v in initial["initial_pool_caps"].items()})
    if plan["budget"] != assets["budget"]:
        raise ValueError("RUN_LOCK cannot reset the bound historical GPU consumption")
    return plan


def _append_event(path, event):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(event, sort_keys=True, allow_nan=False)+"\n")
        stream.flush()
        os.fsync(stream.fileno())


def available_gpu(index):
    identifier = subprocess.check_output(["nvidia-smi","-i",str(index),"--query-gpu=uuid",
        "--format=csv,noheader,nounits"],text=True).strip()
    rows = subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid,pid",
        "--format=csv,noheader,nounits"],text=True).splitlines()
    processes = [line.strip().split(",",1)[1].strip() for line in rows
                 if "," in line and line.split(",",1)[0].strip() == identifier]
    if processes:
        raise RuntimeError("bound GPU is occupied by existing compute processes; no job was launched")
    return identifier


def charged_job(root, command, *, phase, job_key, max_gpu_hours=None):
    """Reserve and settle one owned single-GPU child, including failed attempts."""
    if os.environ.get("MPR_CHARGED_EVENT_ID"):
        raise RuntimeError("a charged child cannot launch another charged parent/child pair")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    assets = local_assets()
    if root.resolve() != Path(assets["run_root"]).resolve():
        raise ValueError("charged jobs must use the bound new MPR run root")
    if not isinstance(command, (list, tuple)) or not command or not all(isinstance(v, str) for v in command):
        raise ValueError("job command must be an explicit argv list")
    with (root/"JOB.lock").open("a") as exclusive:
        fcntl.flock(exclusive.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        active = root/"ACTIVE_JOB.json"
        if active.exists():
            raise RuntimeError("unreconciled MPR reservation; inspect its owned PID and ledger")
        plan = _job_plan(root, phase, assets)
        remaining = available_budget(root, plan, phase=phase)
        if max_gpu_hours is not None:
            maximum = float(max_gpu_hours)
            if not math.isfinite(maximum) or maximum <= 0:
                raise ValueError("job GPU-hour reservation must be finite and positive")
            remaining = min(remaining, maximum)
        if remaining <= 0:
            raise RuntimeError("bound campaign/phase GPU budget is exhausted")
        gpu_uuid = available_gpu(assets["physical_gpu_index"])
        started, wall_start = time.perf_counter(), time.time()
        event_id = f"mpr:{phase}:{time.time_ns()}"
        deadline = started+remaining*3600
        log_path = root/"logs"/f"{event_id.replace(':','-')}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        reservation = dict(event_id=event_id, job_key=job_key, phase=phase, gpus=1,
            parent_pid=os.getpid(), started_unix=wall_start, started_monotonic=started,
            reserved_gpu_hours=remaining, deadline_monotonic=deadline,
            command_sha256=hashlib.sha256(json.dumps(list(command)).encode()).hexdigest(),
            log_path=str(log_path), physical_gpu_index=assets["physical_gpu_index"],gpu_uuid=gpu_uuid)
        write_json_atomic(active, reservation)
        process, error, exit_code = None, None, -1
        try:
            with log_path.open("wb") as output:
                process = subprocess.Popen(command, cwd=PROJECT, start_new_session=True,
                    stdout=output, stderr=subprocess.STDOUT,
                    env={**os.environ, "PYTHONPATH": os.pathsep.join(assets["pythonpath"]),
                         "CUDA_VISIBLE_DEVICES": str(assets["physical_gpu_index"]),
                         "OMP_NUM_THREADS":"2", "MKL_NUM_THREADS":"2", "OPENBLAS_NUM_THREADS":"2",
                         "CUBLAS_WORKSPACE_CONFIG":":4096:8", "MPR_CHARGED_EVENT_ID":event_id,
                         "MPR_JOB_DEADLINE_MONOTONIC":str(deadline),
                         "MPR_JOB_STARTED_MONOTONIC":str(started)})
                write_json_atomic(active, {**reservation, "pid":process.pid})
                grace = min(10., remaining*1800)
                exit_code = process.wait(timeout=max(.01, deadline-time.perf_counter()-grace))
        except BaseException as exc:
            error = exc
            if process is not None and process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=max(.01, min(10., deadline-time.perf_counter())))
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            exit_code = process.returncode if process is not None else -1
        finally:
            seconds = time.perf_counter()-started
            event = dict(event_id=event_id, job_key=job_key, phase=phase, gpus=1,
                seconds=seconds, gpu_hours=seconds/3600, exit_code=exit_code,
                started_unix=wall_start, ended_unix=time.time(),
                status="COMPLETE" if exit_code == 0 and error is None else "FAILED",
                command_sha256=reservation["command_sha256"])
            _append_event(root/"COST_LEDGER.jsonl", event)
            _append_event(ARTIFACTS/"COST_LEDGER.jsonl", event)
            active.unlink()
        if error is not None:
            raise RuntimeError(f"{job_key} stopped; failed attempt settled in the new ledger") from error
        if exit_code != 0:
            raise RuntimeError(f"{job_key} exited {exit_code}; failed attempt settled in the new ledger")
        return event


def preflight(assets, *, dry_run=False):
    root = Path(assets["run_root"])
    records = p0_inputs(read_json(assets["population_path"]))
    if dry_run:
        return dict(status="DRY_RUN_NO_PREFLIGHT",cpu_test="tests/test_revisit_prompt.py",
            gradient_updates=3,gradient_variant="M6",probe_inputs=[r["input_id"] for r in records],
            new_process_repeats=3,phase="P0",formal_training_started=False)
    proof_path = ARTIFACTS/"FOCUSED_CPU_TEST.json"
    identity = dict(numeric_sources=numeric_sources(),test_sha256=sha256(PROJECT/"tests/test_revisit_prompt.py"))
    proof = read_json(proof_path) if proof_path.is_file() else {}
    if proof.get("identity") != identity or proof.get("status") != "PASSED":
        started = time.perf_counter()
        try:
            process = subprocess.run(child_command(assets,"pytest","tests/test_revisit_prompt.py","-q"),
                cwd=PROJECT,env={**os.environ,"PYTHONPATH":os.pathsep.join(assets["pythonpath"]),
                "CUDA_VISIBLE_DEVICES":"","OMP_NUM_THREADS":"2","MKL_NUM_THREADS":"2"},
                capture_output=True,text=True,timeout=1200)
            output = process.stdout+process.stderr
            proof = dict(identity=identity,status="PASSED" if process.returncode==0 else "FAILED",
                exit_code=process.returncode,seconds=time.perf_counter()-started,output_tail=public_error(Exception(output[-4000:]),assets))
        except subprocess.TimeoutExpired:
            proof = dict(identity=identity,status="TIMEOUT",seconds=time.perf_counter()-started)
        write_json_atomic(proof_path,proof)
        if proof["status"] != "PASSED":
            raise RuntimeError("focused CPU test failed or timed out; no GPU launch")
    write_json_atomic(root/"preflight/P0_INPUTS.json",dict(records=records,numeric_sources=numeric_sources()))
    folder = root/"preflight/gradients/M6-s146"
    smoke = ensure_job(assets,child_command(assets,"scripts.train_revisit_prompt","--variant","M6","--smoke"),
        status_path=folder/"STATUS.json",phase="P0",job_key="smoke:M6:s146",training=True,
        complete_status="SMOKE_COMPLETE_NOT_A_FORMAL_MODEL")
    updates = ledger_events(folder/"TRAINING.jsonl")
    if (smoke["step"] != 3 or not smoke["frozen_parent_unchanged"]
            or not any(smoke["adapter_tensor_changed"].values())
            or len(updates) != 3 or any(r["stage_count"] != 14 for r in updates)
            or not any(s["selected_slots"] > 0 and s["backward"] for r in updates for s in r["stages"])):
        raise ValueError("real P0 gradient/freeze/causal-stage evidence is incomplete")
    probes = []
    for record in records:
        for repeat in range(3):
            folder = root/"preflight/probes"/digest(record["input_id"])/f"repeat{repeat}"
            ensure_job(assets,child_command(assets,"scripts.revisit_prompt_campaign","probe", "--input-id",
                record["input_id"],"--repeat-index",repeat),status_path=folder/"STATUS.json",phase="P0",
                job_key=f"probe:{record['input_id']}:r{repeat}",complete_status="COMPLETE_UNTRAINED_NUMERICAL_PROBE")
            probes.append(read_json(folder/"RESULTS.json"))
    nonbitwise, comparisons = False,[]
    for record in records:
        group = [p for p in probes if p["input_id"] == record["input_id"]]
        structural = lambda p:[{k:s[k] for k in ("stage","window_scan_ids","points","segments","current_points","input_structure")}
                                for s in p["structures"]]
        if any(structural(p) != structural(group[0]) for p in group[1:]):
            raise ValueError("P0 input structures changed between identical new processes")
        exact = all(p["structures"] == group[0]["structures"] for p in group[1:])
        # NPZ archive bytes can include container metadata; compare array values.
        import numpy as np
        paths = [root/"preflight/probes"/digest(record["input_id"])/f"repeat{r}/candidates.npz" for r in range(3)]
        candidates = [np.load(path) for path in paths]
        array_differences = {}
        try:
            for key in candidates[0].files:
                values = [c[key] for c in candidates]
                equal = all(v.shape==values[0].shape and np.array_equal(v,values[0]) for v in values[1:])
                array_differences[key] = dict(exact=equal,shapes=[list(v.shape) for v in values])
                exact &= equal
        finally:
            for c in candidates:
                c.close()
        nonbitwise |= not exact
        metric_differences = {kind:{key:max(p["metrics"][kind][key] for p in group)-min(p["metrics"][kind][key] for p in group)
            if all(p["metrics"][kind][key] is not None for p in group) else None
            for key in group[0]["metrics"][kind]} for kind in ("long","current")}
        comparisons.append(dict(input_id=record["input_id"],horizon=record["horizon"],structure_exact=True,
            outputs_exact=bool(exact),candidate_differences=array_differences,official_scalar_ranges=metric_differences,
            structures_by_repeat=[p["structures"] for p in group]))
    checks = next(p["controlled_checks"] for p in probes if p["controlled_checks"] is not None)
    for name,baseline in (("T01_parent_empty_history","T01_normal_normal"),
                          ("T03_normal_swapped_labels","T03_normal_normal")):
        if checks[baseline]["exact"] and not checks[name]["exact"]:
            raise ValueError("controlled parent or label-only forward diverged with a stable baseline")
        nonbitwise |= not checks[name]["exact"] or not checks[baseline]["exact"]
    if checks["T03_state_normal_normal_exact"] and not checks["T03_state_normal_swapped_exact"]:
        raise ValueError("label-only exchange changed a stable prediction-driven state")
    result = dict(status="COMPLETE",numeric_sources=numeric_sources(),gradient_status=smoke["status"],
        numerical_status="NONBITWISE_RUNTIME_OBSERVED" if nonbitwise else "BITWISE_ON_P0_PANEL_ONLY",
        root_cause="NUMERICAL_ROOT_CAUSE_UNRESOLVED" if nonbitwise else "NOT_CLAIMED_OUTSIDE_MEASURED_PANEL",
        controlled_checks=checks,comparisons=comparisons,formal_training_started=False)
    write_json_atomic(ARTIFACTS/"RUNTIME_SUMMARY.json",result)
    write_json_atomic(root/"preflight/PREFLIGHT.json",result)
    return result


def plan_campaign(assets, *, dry_run=False):
    root = Path(assets["run_root"])
    manifest = read_json(ARTIFACTS/"DATA_MANIFEST.json")
    if dry_run:
        return dict(status="DRY_RUN_NO_COST_MEASUREMENT",N=None,variant_pilots=[f"M{i}" for i in range(1,7)],
            pilot_updates=8,timed_updates=6,baseline_CAL_processes=3,budget=assets["budget"],
            expected_inputs=manifest["counts"],formal_training_started=False)
    if root.joinpath("RUN_LOCK.json").is_file():
        lock = read_json(root/"RUN_LOCK.json")
        if lock["numeric_sources"] != numeric_sources() or sha256(root/"RUN_LOCK.json") != sha256(ARTIFACTS/"RUN_LOCK.json"):
            raise ValueError("frozen run lock source or public identity differs")
        return lock
    smoke = read_json(root/"preflight/PREFLIGHT.json")
    if smoke["status"] != "COMPLETE" or smoke["numeric_sources"] != numeric_sources():
        raise ValueError("plan requires the completed identical P0 preflight")
    m0 = dict(variant="M0",adapter_seed=146,step=0)
    evaluate_endpoint(assets,m0,"CAL")
    results = [read_json(evaluation_folder(assets,m0,"CAL",r)/"RESULTS.json") for r in range(3)]
    aps = [[next(row["AP"] for row in result["rows"] if row["horizon"]==h) for result in results] for h in range(1,6)]
    if any(a is None for group in aps for a in group):
        raise ValueError("P1 complete CAL contains undefined pooled AP; numerical admission is unavailable")
    epsilon = max(max(values)-min(values) for values in aps)*100
    runtime = read_json(ARTIFACTS/"RUNTIME_SUMMARY.json")
    runtime.update(epsilon_run_pp=epsilon,M0_CAL_repeat_AP=aps,P1_admitted=epsilon<=.2)
    write_json_atomic(ARTIFACTS/"RUNTIME_SUMMARY.json",runtime)
    pilots = {}
    if epsilon <= .2:
        for variant in [f"M{i}" for i in range(1,7)]:
            folder = root/"pilots"/f"{variant}-s146"
            status = ensure_job(assets,child_command(assets,"scripts.train_revisit_prompt","--variant",variant,"--pilot"),
                status_path=folder/"STATUS.json",phase="P1",job_key=f"pilot:{variant}:s146",training=True,
                complete_status="PILOT_COMPLETE_NOT_A_FORMAL_MODEL")
            training = ledger_events(folder/"TRAINING.jsonl")
            by_step = {r["step"]:r for r in training}
            if set(by_step) != set(range(1,9)) or not status["frozen_parent_unchanged"]:
                raise ValueError("cost pilot lacks eight actual frozen-parent optimizer updates")
            events = [e for e in merge_events([ledger_events(root/"COST_LEDGER.jsonl")]) if e["job_key"]==f"pilot:{variant}:s146"]
            pilots[variant] = dict(update_seconds=max(by_step[s]["seconds"] for s in range(3,9)),
                startup_seconds=max(0.,sum(e["seconds"] for e in events)-sum(r["seconds"] for r in training)),
                actual_optimizer_steps=8,attempts=len(events),warmup_updates=2,timed_updates=6,
                adapter_changed=status["adapter_tensor_changed"])
    estimates = []
    for repeat,result in enumerate(results):
        status = read_json(evaluation_folder(assets,m0,"CAL",repeat)/"STATUS.json")
        inputs = result["identity"]
        stage_seconds = sum(t["end_to_end_seconds"] for r in inputs for t in r["timings"])
        extra_seconds = sum(max(0.,r["seconds"]-sum(t["end_to_end_seconds"] for t in r["timings"]))+
                            r["metric_update_seconds"] for r in inputs)+status["metric_serialization_seconds"]
        estimates.append(dict(seconds_per_stage=stage_seconds/manifest["stage_forwards"]["CAL"],
            seconds_per_input=extra_seconds/len(inputs),startup_seconds=status["startup_seconds"]))
    evaluation = {key:max(row[key] for row in estimates) for key in estimates[0]}
    cost = (forecast_cost_plan(assets["budget"],pilots,evaluation,manifest) if epsilon <= .2 else
            dict(status="BLOCKED_NUMERICAL_VARIANCE",N=None,checkpoint_steps=[],phase_caps_gpu_hours=None,
                 measured_evaluation=evaluation,measured_pilots=pilots))
    cost.update(epsilon_run_pp=epsilon,baseline_repeats=3,budget=assets["budget"],numeric_sources=numeric_sources())
    write_json_atomic(ARTIFACTS/"COST_PLAN.json",cost)
    if cost["status"] == "LOCKABLE":
        from datasets.revisit_prompt_episode import RevisitDrawPlan
        from scripts.revisit_prompt_assets import build_system,git_output
        from trainer.revisit_prompt_trainer import adapter_state
        from models.revisit_prompt_modules import tensor_summary
        commit = read_json(root/"IMPLEMENTATION_COMMIT.json")
        if commit["numeric_sources"] != numeric_sources():
            raise ValueError("reviewed implementation commit C is required before formal admission")
        sources_at_commit = {name:hashlib.sha256(subprocess.check_output(["git","show",f"{commit['commit']}:{name}"],cwd=PROJECT)).hexdigest()
                             for name in numeric_sources()}
        if sources_at_commit != numeric_sources():
            raise ValueError("implementation C does not contain the admitted numerical bytes")
        initializations,parameters,draws = {},{},{}
        population = read_json(assets["population_path"])
        for seed in (146,147):
            draw = RevisitDrawPlan(population["references"],seed=seed)
            text = "".join(json.dumps(r,sort_keys=True)+"\n" for cursor in range(cost["N"]) for r in draw.draw_update(cursor))
            draws[str(seed)] = hashlib.sha256(text.encode()).hexdigest()
            for variant in [f"M{i}" for i in range(1,7)]:
                system,config,_ = build_system(assets,variant=variant,seed=seed)
                initializations[f"{variant}-s{seed}"] = digest({k:tensor_summary(v) for k,v in adapter_state(system).items()})
                parameters[variant] = sum(p.numel() for p in system.new_parameters())
                del system
        parameters["M0"] = 0
        lock = {**cost,"status":"LOCKED","schema":1,"implementation_commit":commit["commit"],
            "variants":[f"M{i}" for i in range(7)],"adapter_seeds":[146,147],"parent_seed":45,
            "expected_inputs":manifest["counts"],"data_manifest_sha256":sha256(ARTIFACTS/"DATA_MANIFEST.json"),
            "parent_sha256":assets["hashes"]["parent"],"encoder_sha256":assets["hashes"]["encoder"],
            "taxonomy_sha256":assets["hashes"]["taxonomy"],"initialization_hashes":initializations,
            "draw_hashes":draws,"trainable_parameters":parameters,
            "configuration_overrides":read_json(ARTIFACTS/"START_STATE.json").get("configuration_overrides",{}),
            "recipe":"conf/revisit_prompt/default.yaml","evaluation_processes_per_endpoint":3}
        available_budget(root,lock,phase="P2")
    else:
        lock = {**cost,"status":cost["status"],"schema":1,"variants":[f"M{i}" for i in range(7)],
                "expected_inputs":manifest["counts"],"formal_training_started":False}
    write_json_atomic(root/"RUN_LOCK.json",lock)
    write_json_atomic(ARTIFACTS/"RUN_LOCK.json",lock)
    collect_evaluations(assets)
    return lock


def selection_endpoints(selection):
    candidate = selection.get("candidate_star") or selection.get("candidate_descriptive_star")
    values = [dict(variant="M0",adapter_seed=146,step=0),selection.get("control_star"),candidate]
    unique = {}
    for endpoint in values:
        if endpoint is not None:
            unique[endpoint["variant"],endpoint["adapter_seed"],endpoint["step"]] = endpoint
    return list(unique.values())


def run_campaign(assets, *, resume=False,dry_run=False):
    root = Path(assets["run_root"])
    if dry_run:
        lock = read_json(root/"RUN_LOCK.json") if root.joinpath("RUN_LOCK.json").is_file() else None
        return dict(status="DRY_RUN_NO_EXPERIMENT",admission=lock["status"] if lock else "NOT_ADMITTED",
            N=lock.get("N") if lock else None,main_variants=[f"M{i}" for i in range(1,7)],
            replication="only families qualified by frozen main CAL lock",resume=resume)
    from scripts.revisit_prompt_assets import read_run_plan
    plan = read_run_plan(assets)
    from scripts.select_revisit_prompt import main as select_main
    from scripts.evaluate_revisit_prompt import bootstrap_locked
    for seed in (146,147):
        selection_path = ARTIFACTS/"CAL_SELECTION_LOCK.json"
        if seed == 147:
            main_selection = read_json(selection_path)
            if not main_selection["replication"]["eligible"]:
                write_json_atomic(ARTIFACTS/"REPLICATION_LOCK.json",dict(status="NOT_EXECUTED_CAL_GATE_FAILED",
                    seed=147,reason=main_selection["replication"]["reasons"],main_selection_sha256=sha256(selection_path)))
                break
            variants = main_selection["replication"]["variants_to_train"]
        else:
            variants = [f"M{i}" for i in range(1,7)]
        for variant in variants:
            folder = root/"trajectories"/f"{variant}-s{seed}"
            if folder.exists() and any(folder.iterdir()) and not resume:
                raise ValueError("existing formal work requires run --resume")
            ensure_job(assets,child_command(assets,"scripts.train_revisit_prompt","--variant",variant,"--seed",seed),
                status_path=folder/"STATUS.json",phase="P2",job_key=f"train:{variant}:s{seed}",training=True,
                complete_status="COMPLETE")
            for step in plan["checkpoint_steps"]:
                evaluate_endpoint(assets,dict(variant=variant,adapter_seed=seed,step=step),"CAL",
                    selection_lock=selection_path if seed==147 else None)
        collect_evaluations(assets)
        output = ARTIFACTS/("CAL_SELECTION_LOCK.json" if seed==146 else "REPLICATION_LOCK.json")
        old = read_json(output) if output.is_file() else {}
        if old.get("status") != "COMPLETE":
            if select_main(["--seed",str(seed)]) != 0:
                raise RuntimeError("CAL selector failed; no confirmation model is selected")
        else:
            snapshot = output.parent/old["input_identities"]["results_snapshot"]
            if (sha256(snapshot) != old["input_identities"]["results_sha256"] or
                    sha256(ARTIFACTS/"RUN_LOCK.json") != old["input_identities"]["run_lock_sha256"]):
                raise ValueError("immutable CAL selection snapshot/run identity differs")
        if read_json(output)["status"] != "COMPLETE":
            raise ValueError("required four-point matrix is incomplete; no winner claim")
    main_selection = read_json(ARTIFACTS/"CAL_SELECTION_LOCK.json")
    confirmations = [(main_selection,ARTIFACTS/"CAL_SELECTION_LOCK.json")]
    replication = read_json(ARTIFACTS/"REPLICATION_LOCK.json")
    if replication["status"] == "COMPLETE":
        confirmations.append((replication,ARTIFACTS/"REPLICATION_LOCK.json"))
    done = set()
    for selection,selection_path in confirmations:
        for endpoint in selection_endpoints(selection):
            key = endpoint["variant"],endpoint["adapter_seed"],endpoint["step"]
            if key in done:
                continue
            for role in ("SEL","PB"):
                evaluate_endpoint(assets,endpoint,role,selection_lock=selection_path)
            done.add(key)
        candidate = selection.get("candidate_star") or selection.get("candidate_descriptive_star")
        if candidate is not None and selection.get("control_star") is not None:
            for role in ("SEL","PB"):
                target = ARTIFACTS/f"BOOTSTRAP_{role}_s{selection['seed']}.json"
                if not target.is_file():
                    result = bootstrap_locked(assets,role=role,control=selection["control_star"],candidate=candidate)
                    write_json_atomic(target,result)
    collect_evaluations(assets)
    candidate = main_selection.get("candidate_star") or main_selection.get("candidate_descriptive_star")
    if candidate is not None:
        endpoint = candidate
        folder = root/"p5"/f"mechanism-{endpoint['variant']}-s{endpoint['adapter_seed']}-u{endpoint['step']:06d}"
        ensure_job(assets,child_command(assets,"scripts.revisit_prompt_campaign","p5","--mode","mechanism",
            "--variant",endpoint["variant"],"--seed",endpoint["adapter_seed"],"--step",endpoint["step"]),
            status_path=folder/"STATUS.json",phase="P5",job_key="mechanism:locked-main-candidate",complete_status="COMPLETE")
    for endpoint in selection_endpoints(main_selection):
        folder = root/"p5"/f"profile-{endpoint['variant']}-s{endpoint['adapter_seed']}-u{endpoint['step']:06d}"
        ensure_job(assets,child_command(assets,"scripts.revisit_prompt_campaign","p5","--mode","profile",
            "--variant",endpoint["variant"],"--seed",endpoint["adapter_seed"],"--step",endpoint["step"]),
            status_path=folder/"STATUS.json",phase="P5",
            job_key=f"profile:{endpoint['variant']}:s{endpoint['adapter_seed']}:u{endpoint['step']}",complete_status="COMPLETE")
    return summarize(assets)


def aggregate_endpoint(rows, endpoint, role, expected):
    selected = [r for r in rows if r["role"]==role and
        (r["variant"],r["adapter_seed"],r["step"]) == (endpoint["variant"],endpoint["adapter_seed"],endpoint["step"])]
    horizons = sorted(map(int,expected))
    points = {(r["repeat_index"],r["horizon"]):r for r in selected}
    if (len(points) != 3*len(horizons) or len(points) != len(selected)
            or any((repeat,h) not in points for repeat in range(3) for h in horizons)):
        return dict(status="INCOMPLETE",reason="missing or duplicate actual repeat/horizon rows")
    for (repeat,h),row in points.items():
        if (row["status"] != "COMPLETE" or row["completed_inputs"] != expected[str(h)]
                or row["expected_inputs"] != expected[str(h)] or row["unit"] != "0-1"
                or any(row.get(key) is None or not math.isfinite(row[key]) for key in ("AP","current_AP"))):
            return dict(status="INCOMPLETE",reason="undefined metric or incorrect input denominator")
    ap = {str(h):sum(points[r,h]["AP"] for r in range(3))/3 for h in horizons}
    current = {str(h):sum(points[r,h]["current_AP"] for r in range(3))/3 for h in horizons}
    values = {key:[sum(points[r,h][field] for h in (3,4,5))/3 for r in range(3)]
              for key,field in (("S_long","AP"),("S_current","current_AP"))}
    return dict(status="COMPLETE",endpoint=endpoint,role=role,AP=ap,current_AP=current,
        **{key:sum(v)/3 for key,v in values.items()},repeat_values=values,
        repeat_ranges={key:max(v)-min(v) for key,v in values.items()},expected_inputs=expected)


def confirmation_verdict(main_selection, pairs, *, replication_complete):
    """Apply the preregistered final gate to complete actual endpoint summaries."""
    if main_selection.get("status") != "COMPLETE":
        return "INCONCLUSIVE"
    if not main_selection["replication"]["eligible"]:
        return "SCREEN_NO_GO"
    required = {f"s{seed}:{role}" for seed in ((146,147) if replication_complete else (146,)) for role in ("SEL","PB")}
    if not required.issubset(pairs):
        return "INCONCLUSIVE"
    if not pairs or any(value[side]["status"] != "COMPLETE" for value in pairs.values() for side in ("control","candidate")):
        return "INCONCLUSIVE"
    if not replication_complete:
        return ("PROVISIONAL_SINGLE_ADAPTER_SEED_SIGNAL" if all(
            p["candidate"]["S_long"] > p["control"]["S_long"] for p in pairs.values()) else "INCONCLUSIVE")
    for role in ("SEL","PB"):
        role_pairs = [pairs[f"s{seed}:{role}"] for seed in (146,147)]
        delta_long = [p["candidate"]["S_long"]-p["control"]["S_long"] for p in role_pairs]
        delta_current = [p["candidate"]["S_current"]-p["control"]["S_current"] for p in role_pairs]
        if min(delta_long) <= 0 or sum(delta_long)/2 < .005 or sum(delta_current)/2 < 0:
            return "CONFIRMATION_NO_GO"
        for p in role_pairs:
            candidate,control = p["candidate"],p["control"]
            ranges = [x["repeat_ranges"]["S_long"] for x in (candidate,control)]
            if max(ranges) > .002 or candidate["S_long"]-control["S_long"] <= 2*max(ranges):
                return "CONFIRMATION_NO_GO"
            if any(candidate["AP"][h]-control["AP"][h] < -.002 for h in control["AP"]):
                return "CONFIRMATION_NO_GO"
    return "PROMISING_REPLICATED_ADAPTER_SIGNAL"


def summarize(assets, *, dry_run=False):
    if dry_run:
        return dict(status="DRY_RUN_NO_SUMMARY",scope="actual journals, official rows and frozen locks only")
    import shutil
    import zipfile
    root = Path(assets["run_root"])
    rows = collect_evaluations(assets)
    manifest = read_json(ARTIFACTS/"DATA_MANIFEST.json")
    lock = read_json(root/"RUN_LOCK.json") if root.joinpath("RUN_LOCK.json").is_file() else {}
    main_path,rep_path = ARTIFACTS/"CAL_SELECTION_LOCK.json",ARTIFACTS/"REPLICATION_LOCK.json"
    main = read_json(main_path) if main_path.is_file() else {}
    rep = read_json(rep_path) if rep_path.is_file() else {}
    training = []
    for path in sorted((root/"trajectories").glob("*/TRAINING.jsonl")):
        training += [{**r,"parent_sha256":assets["hashes"]["parent"]} for r in ledger_events(path)]
    write_csv(ARTIFACTS/"TRAINING.csv",training,["variant","adapter_seed","parent_seed","parent_sha256","step",
        "status","episode_count","stage_count","loss","lr","gradient_norm","seconds","process_gpu_hours",
        "attempt","replayed_update","layer_gradients","stages"])
    mechanisms,profiles = [],[]
    for path in sorted((root/"p5").glob("*/RESULTS.json")):
        result = read_json(path)
        (profiles if result["mode"]=="profile" else mechanisms).extend(result["rows"])
    write_csv(ARTIFACTS/"MECHANISM.csv",mechanisms or [dict(status="NOT_EXECUTED")],
        ["variant","adapter_seed","step","input_id","horizon","status","selected_slots","raw_prediction_changed","outputs"])
    write_csv(ARTIFACTS/"PROFILE.csv",profiles or [dict(status="NOT_EXECUTED")],
        ["variant","adapter_seed","step","input_id","horizon","repeat_index","stage","status","points","segments",
         "current_points","occupied_slots","state_bytes","io_seconds","forward_seconds","association_seconds",
         "postprocess_seconds","state_geometry_seconds","end_to_end_seconds","peak_allocated_bytes","peak_reserved_bytes"])
    pairs,diagnoses = {},[]
    for selection in (main,rep):
        if selection.get("status") != "COMPLETE":
            continue
        candidate = selection.get("candidate_star") or selection.get("candidate_descriptive_star")
        control = selection.get("control_star")
        if candidate is None or control is None:
            continue
        for role in ("SEL","PB"):
            pairs[f"s{selection['seed']}:{role}"] = {side:aggregate_endpoint(rows,endpoint,role,manifest["counts"][role])
                for side,endpoint in (("control",control),("candidate",candidate))}
            # Analyse actual official candidate masks; preserve losses as well as recoveries.
            for repeat in range(3):
                sources = {}
                for name,endpoint in (("baseline",dict(variant="M0",adapter_seed=146,step=0)),("candidate",candidate)):
                    path = evaluation_folder(assets,endpoint,role,repeat)/"RESULTS.json"
                    if path.is_file():
                        sources[name] = {r["input_id"]:r for r in read_json(path)["identity"]}
                if set(sources) != {"baseline","candidate"}:
                    continue
                if set(sources["baseline"]) != set(sources["candidate"]):
                    raise ValueError("paired current-mask input populations differ")
                for input_id,baseline in sources["baseline"].items():
                    result = sources["candidate"][input_id]
                    a,b = baseline["current_mask_diagnostics"][-1],result["current_mask_diagnostics"][-1]
                    visible_a = set(a["matched_gt_ids"]+a["unmatched_gt_ids"])
                    visible_b = set(b["matched_gt_ids"]+b["unmatched_gt_ids"])
                    if visible_a != visible_b:
                        raise ValueError("paired current-mask GT populations differ")
                    matched_a,matched_b = set(a["matched_gt_ids"]),set(b["matched_gt_ids"])
                    diagnoses.append(dict(role=role,adapter_seed=selection["seed"],repeat_index=repeat,
                        input_id=input_id,reference_id=baseline["reference_id"],horizon=baseline["horizon"],
                        recovered_gt_ids=sorted(matched_b-matched_a),lost_gt_ids=sorted(matched_a-matched_b),
                        recovered_count=len(matched_b-matched_a),lost_count=len(matched_a-matched_b),gt_count=len(visible_a)))
    write_json_atomic(ARTIFACTS/"CURRENT_MASK_DIAGNOSTICS.json",dict(status="COMPLETE" if diagnoses else "NOT_AVAILABLE",
        matching="all official candidates, class-aware one-to-one IoU>=0.5",rows=diagnoses,
        moving_subset=dict(status="NOT_AVAILABLE",reason="no verified moving-annotation correspondence bound to this diagnostic")))
    scientific = confirmation_verdict(main,pairs,replication_complete=rep.get("status")=="COMPLETE")
    blocked = lock.get("status","").startswith("BLOCKED_")
    eval_complete = bool(pairs) and all(p[side]["status"]=="COMPLETE" for p in pairs.values() for side in ("control","candidate"))
    expected_rep = main.get("replication",{}).get("eligible",False)
    complete = main.get("status")=="COMPLETE" and eval_complete and (not expected_rep or rep.get("status")=="COMPLETE")
    experiment = (lock["status"] if blocked else "COMPLETE" if complete else
                  "EXPERIMENT_INCOMPLETE" if training or rows else "NOT_STARTED")
    if blocked:
        scientific = "INCONCLUSIVE"
    decision = dict(status=scientific,experiment_status=experiment,main_selection_status=main.get("status","NOT_EXECUTED"),
        replication_status=rep.get("status","NOT_EXECUTED"),pairs=pairs,unit="0-1",
        H1="NOT_ESTABLISHED",H2="NOT_AVAILABLE_MOVING_SUBSET",H3="NOT_ESTABLISHED",
        QUALITY_SCOPE="TEMPORAL_ONLY" if any(p["candidate"].get("S_long",0)>p["control"].get("S_long",0) and
            p["candidate"].get("S_current",0)<=p["control"].get("S_current",0) for p in pairs.values()) else "UNCONFIRMED")
    write_json_atomic(ARTIFACTS/"SELECTION_DECISION.json",decision)
    events = merge_events([ledger_events(root/"COST_LEDGER.jsonl")])
    new_hours = sum(e["gpu_hours"] for e in events)
    old = read_json(ARTIFACTS/"STATUS.json") if ARTIFACTS.joinpath("STATUS.json").is_file() else {}
    status = {**old,"status":experiment,"experiment_status":experiment,"scientific_status":scientific,
        "formal_N":lock.get("N"),"gpu_training_started":bool(training),"new_gpu_hours":new_hours,
        "publication_status":old.get("publication_status","NOT_PUBLISHED"),"publication_ready":False,
        "prompt_audit_execution_complete":False,"current_numeric_sources":numeric_sources(),
        "next_task":"Primary requirement-by-requirement review, then complete missing stages and publish",
        "hypothesis_status":{k:decision[k] for k in ("H1","H2","H3")}}
    write_json_atomic(ARTIFACTS/"STATUS.json",status)
    assets_rows = []
    def copy_asset(source,relative,**metadata):
        destination = ARTIFACTS/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.exists() and sha256(source) != sha256(destination):
            raise ValueError("already packaged model asset differs; preserve the prior publication")
        if not destination.exists():
            shutil.copyfile(source,destination)
        assets_rows.append(dict(path=destination.relative_to(PROJECT).as_posix(),bytes=destination.stat().st_size,
                               sha256=sha256(destination),**metadata))
    for selection in (main,rep):
        for endpoint in selection.get("selected_by_variant",{}).values():
            if endpoint is None or endpoint["variant"] == "M0":
                continue
            v,s,u = endpoint["variant"],endpoint["adapter_seed"],endpoint["step"]
            copy_asset(root/"trajectories"/f"{v}-s{s}"/f"delta-{u:06d}.pt",f"deployment/{v}-s{s}-u{u:06d}.pt",
                kind="delta",variant=v,adapter_seed=s,step=u,optimizer_included=False)
        for endpoint in selection_endpoints(selection) if selection else []:
            if endpoint["variant"]=="M0":
                continue
            v,s = endpoint["variant"],endpoint["adapter_seed"]
            relative = f"resume/{v}-s{s}-u{lock['N']:06d}.pt"
            if any(r["path"]==(ARTIFACTS/relative).relative_to(PROJECT).as_posix() for r in assets_rows):
                continue
            copy_asset(root/"trajectories"/f"{v}-s{s}/resume-latest.pt",relative,
                kind="resume",variant=v,adapter_seed=s,step=lock["N"],optimizer_included=True)
    evidence_paths = sorted((root/"evaluation").glob("*/*/repeat*/official-*.json"))
    evidence_paths += sorted((root/"evaluation").glob("*/*/repeat*/references/*.json"))
    if evidence_paths:
        archive = root/"official_states.zip"
        with zipfile.ZipFile(archive,"w",compression=zipfile.ZIP_DEFLATED) as stream:
            for path in evidence_paths:
                stream.write(path,path.relative_to(root/"evaluation").as_posix())
        destination = ARTIFACTS/"evidence/official_states.zip"
        # Evidence can grow before A is frozen; it is generated from actual states.
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(archive,destination)
        assets_rows.append(dict(path=destination.relative_to(PROJECT).as_posix(),bytes=destination.stat().st_size,
            sha256=sha256(destination),kind="official-sufficient-states",file_count=len(evidence_paths)))
    write_json_atomic(ARTIFACTS/"MODEL_MANIFEST.json",dict(schema=1,status="ACTUAL_ASSETS_ONLY",assets=assets_rows,
        formal_models_produced=any(r["kind"]=="delta" for r in assets_rows),dependencies=dict(
            parent=dict(path="artifacts/native_long_postselection_v1/deployment/E1-23760.pt",sha256=assets["hashes"]["parent"]),
            encoder=dict(sha256=assets["hashes"]["encoder"],redistributed=False),taxonomy_sha256=assets["hashes"]["taxonomy"]),
        delta_requires_parent_and_encoder=True))
    if not main_path.is_file():
        write_json_atomic(main_path,dict(status="NOT_EXECUTED",reason=experiment,candidate_star=None,control_star=None))
    if not rep_path.is_file():
        write_json_atomic(rep_path,dict(status="NOT_EXECUTED",reason=experiment))
    if not ARTIFACTS.joinpath("RUNTIME_SUMMARY.json").is_file():
        write_json_atomic(ARTIFACTS/"RUNTIME_SUMMARY.json",dict(status="NOT_EXECUTED",epsilon_run_pp=None))
    runtime = read_json(ARTIFACTS/"RUNTIME_SUMMARY.json")
    model_rows = main.get("selected_by_variant",{})
    lines = ["# MPR-V1 实际实验报告","",f"代码状态：{status.get('code_status','IN_PROGRESS')}。实验状态：{experiment}。科学状态：{scientific}。",
        "", "## 问题与方法边界","",
        "本轮在冻结的 E1-23760 seed45 上适配历史提示，研究首掩码前的当前特征检索、当前位置提示与拒绝学习。",
        "M2 复用首轮解码后读取；M4–M6 在首掩码前检索。不以普通记忆注意力或持久 query 本身声称首次创新。",
        "SAM-V/MOTR/S4VY 为近邻启发，本轮未复现这些外部系统；未采用 RGB 教师或新增基础模型。",
        "", "## 参数与实验政策","",
        "全部父参数和缓冲区保持冻结/eval，只优化新增模块。M0 无新增参数且不训练；M1/M4/M5=84224，M2=99073，M3=51200，M6=101761。",
        "M1 用当前 FPS 局部内容对照；M3 传播历史描述和旧位置；M5 加 0.1 检索损失；M6 再加 0.1 支持 BCE，并在提取后将候选分数乘支持概率一次。",
        "原任务 raw_sum 损失、100 queries、最终共同关联/commit0 输出政策相同。FP32 主训练、TF32 关闭，原 FlashAttention 内部 FP16 如实保留。",
        "seed146/147 是同一 seed45 父模型的新适配种子，不是两个独立父训练。试跑权重不作为正式结果。",
        "", "## 数据、选点与完整性","",
        f"正式 N：{lock.get('N')}；实际正式训练日志行数：{len(training)}（包含收费的恢复重复运算）。",
        f"CAL/SEL/PB 输入数：`{json.dumps(manifest['counts'],sort_keys=True)}`。独立场景：`{json.dumps(manifest['independent_references'],sort_keys=True)}`。",
        "这些是历史暴露面板；PB 只有六个物理场景。LOCAL 不加入本轮，不能与原论文 T2 主表直接相减。",
        "所有四点及三次独立进程行保存在 CAL_ALL/SEL_LOCKED/PB_LOCKED.csv，未知指标为空，单位0–1；AP报告百分数、差值pp需乘100。",
        f"逐变体实际最佳点与负结果：`{json.dumps(model_rows,sort_keys=True)}`。",
        f"主选模状态：{main.get('status','NOT_EXECUTED')}；复验状态：{rep.get('status','NOT_EXECUTED')}。SEL/PB 不重选模型。",
        "", "## 数值、统计与机制","",
        f"P0状态：{runtime.get('numerical_status','NOT_EXECUTED')}；P1 epsilon_run_pp：{runtime.get('epsilon_run_pp')}。未知数值根因不称已解决。",
        "配对 bootstrap 按物理场景重新池化官方状态，不平均场景 AP。PB 区间受六个场景限制；没有定义的重采样保留 NA，不重抽。",
        "H1/H2/H3 当前均未被自动确认为成立。CURRENT_MASK_DIAGNOSTICS 同时保留恢复与丢失；仅 AP 上升不能证明掩码改善。",
        f"质量范围：{decision['QUALITY_SCOPE']}。搬动子集未绑定可靠变化标注时为 NOT_AVAILABLE。",
        "MECHANISM/PROFILE 保存受限干预与真实资源；NOT_EXECUTED 行不代表测量完成。显式状态字节包含几何sidecar，离线输出档案单独存储。",
        "", "## 成本、交付与判断","",
        f"新 GPUh：{new_hours:.9f}；历史GPUh：{assets['budget']['prior_gpu_hours']:.9f}；预留8GPUh保持不动。",
        "delta 部署依赖固定父任务包及编码器；仅实际产出的最佳点和恢复文件列入 MODEL_MANIFEST。原始数据与编码器不公开。",
        "没有完成的训练、评价、真实梯度、部署回载或远端字节核验不得用代码测试代替；最终发布仍需主执行者逐条审阅。",
        f"继续判断：{'预算/数值阻塞，当前不启动正式矩阵。' if blocked else '以完整锁定确认和机制证据决定投入；当前不宣布 CVPR 就绪。'}",""
    ]
    ARTIFACTS.joinpath("REPORT.md").write_text("\n".join(lines))
    return status


def main(argv=None):
    parser = argparse.ArgumentParser(description="MPR-V1 fixed-budget execution campaign")
    sub = parser.add_subparsers(dest="command",required=True)
    resolve = sub.add_parser("resolve",help="bind actual existing assets and historical cost")
    resolve.add_argument("--auto",action="store_true",required=True)
    resolve.add_argument("--dry-run",action="store_true")
    for name in ("preflight","plan","run","summarize","publish"):
        item = sub.add_parser(name)
        item.add_argument("--dry-run",action="store_true")
        if name == "run":
            item.add_argument("--resume",action="store_true")
        if name == "publish":
            item.add_argument("--execute",action="store_true")
    probe = sub.add_parser("probe",help="internal charged P0 child")
    probe.add_argument("--input-id",required=True)
    probe.add_argument("--repeat-index",type=int,required=True,choices=(0,1,2))
    probe.add_argument("--retry",type=int,default=0,choices=(0,1,2))
    probe.add_argument("--dry-run",action="store_true")
    p5 = sub.add_parser("p5",help="internal charged mechanism/profile child")
    p5.add_argument("--mode",required=True,choices=("mechanism","profile"))
    p5.add_argument("--variant",required=True,choices=[f"M{i}" for i in range(7)])
    p5.add_argument("--seed",type=int,default=146,choices=(146,147))
    p5.add_argument("--step",type=int,required=True)
    p5.add_argument("--retry",type=int,default=0,choices=(0,1,2))
    p5.add_argument("--dry-run",action="store_true")
    args = parser.parse_args(argv)
    assets = None
    try:
        if args.command == "resolve":
            from scripts.revisit_prompt_assets import resolve_assets
            if args.dry_run:
                result = dict(status="DRY_RUN_NO_ASSET_REBIND",known_resolver=PROJECT.joinpath("assets.local.json").is_file())
            else:
                assets = resolve_assets()
                result = read_json(ARTIFACTS/"START_STATE.json")
        else:
            assets = local_assets()
            if args.command == "preflight":
                result = preflight(assets,dry_run=args.dry_run)
            elif args.command == "plan":
                result = plan_campaign(assets,dry_run=args.dry_run)
            elif args.command == "run":
                result = run_campaign(assets,resume=args.resume,dry_run=args.dry_run)
            elif args.command == "summarize":
                result = summarize(assets,dry_run=args.dry_run)
            elif args.command == "probe":
                result = probe_worker(args,assets)
            elif args.command == "p5":
                result = p5_worker(args,assets)
            else:
                from scripts.publish_revisit_prompt import publish
                result = publish(assets,execute=args.execute and not args.dry_run)
        print(json.dumps(result,sort_keys=True,allow_nan=False))
        return 0
    except Exception as error:
        if assets and args.command in {"probe","p5"}:
            root = Path(assets["run_root"])
            folder = (root/"preflight/probes"/digest(args.input_id)/f"repeat{args.repeat_index}" if args.command=="probe" else
                root/"p5"/f"{args.mode}-{args.variant}-s{args.seed}-u{args.step:06d}")
            if folder.joinpath("STATUS.json").is_file():
                status = read_json(folder/"STATUS.json")
                if status["status"] not in {"COMPLETE","COMPLETE_UNTRAINED_NUMERICAL_PROBE"}:
                    write_json_atomic(folder/"STATUS.json",{**status,"status":"FAILED",
                        "error_type":type(error).__name__,"error":str(error)})
        write_json_atomic(ARTIFACTS/"CAMPAIGN_ENTRY_STATUS.json",dict(status="FAILED",command=args.command,
            error_type=type(error).__name__,error=public_error(error,assets)))
        print(f"{type(error).__name__}: {public_error(error,assets)}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
