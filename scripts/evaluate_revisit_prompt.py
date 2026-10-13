"""Independent-process MPR evaluation using frozen official candidate/AP semantics."""

from __future__ import annotations

import argparse
import base64
import collections
import hashlib
import json
from pathlib import Path
import time

from scripts.native_long_budget import write_json_atomic, ledger_events
from scripts.revisit_prompt_assets import ARTIFACTS, local_assets, read_json, sha256, public_error
from scripts.revisit_prompt_assets import (
    build_system, configuration_digest, configure_runtime, digest, numeric_sources,
    read_run_plan, require_charged_child, validate_evaluation_request,
)


def current_metric_truth(target, stage_index):
    import torch

    selected = target["temporal_stages"] == stage_index
    masks = target["masks"][:, selected].cpu().bool()
    present = masks.any(1)
    return dict(ids=target["ids"].cpu()[present], labels=target["labels"].cpu()[present],
                changes=target["changes"].cpu()[present], masks=masks[present],
                temporal_stages=torch.zeros(int(selected.sum()), dtype=torch.long),
                ambiguities=target.get("ambiguities", []))


def official_compute_nullable(metric):
    """An observed population with no evaluable GT has undefined AP, not zero."""
    import torch

    head = metric._metric.heads[0]
    gt = 0.
    for name in head.metric_state:
        value = getattr(head,name)
        if name.endswith("_hard_fn"):
            gt += float(value.sum())
        elif name.endswith("_y_true"):
            gt += sum(float(v.sum()) for v in (value if isinstance(value,list) else [value]))
    if gt:
        return metric.compute()
    if metric._updates == 0:
        raise ValueError("official metric population has no input observations")
    prefix = "raw_local" if metric.mode == "raw_local" else "online" if metric.mode == "strict_online" else "offline_reconstructed"
    measures = ("AP","AP50","AP25","REC","REC50","REC25") if metric.mode == "raw_local" else (
                "t-mAP","t-mAP50","t-mAP25","t-REC","t-REC50","t-REC25")
    return {f"{prefix}_{name}":None for name in measures}


def pool_official_states(states, *, dataset_spec):
    """Concatenate detection states and sum count states, then use official AP."""
    import torch
    from scripts.p6a_metrics import OfficialMetricAccumulator

    states = list(states)
    if not states or len({state["mode"] for state in states}) != 1:
        raise ValueError("pooling requires nonempty official states of one metric mode")
    metric = OfficialMetricAccumulator(mode=states[0]["mode"],dataset_spec=dataset_spec)
    head = metric._metric.heads[0]
    merged = {name: [] if isinstance(getattr(head,name),list) else torch.zeros_like(getattr(head,name))
              for name in sorted(head.metric_state)}
    updates = 0
    dtypes = {str(dtype):dtype for dtype in (torch.float32,torch.float64,torch.int32,torch.int64,torch.bool)}
    for state in states:
        core = {k:v for k,v in state.items() if k != "sha256"}
        expected = hashlib.sha256((json.dumps(core,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()).hexdigest()
        if state["schema_version"] != 1 or state["sha256"] != expected:
            raise ValueError("official state byte identity differs")
        if [r["name"] for r in state["states"]] != sorted(head.metric_state):
            raise ValueError("official state variable population differs")
        updates += state["updates"]
        for record in state["states"]:
            raw = base64.b64decode(record["data"],validate=True)
            dtype = dtypes[record["dtype"]]
            tensor = (torch.frombuffer(bytearray(raw),dtype=dtype).clone().reshape(record["shape"])
                      if raw else torch.empty(record["shape"],dtype=dtype))
            value = merged[record["name"]]
            if isinstance(value,list):
                if record["kind"] != "list" or tensor.ndim != 1:
                    raise ValueError("official list state schema differs")
                value.append(tensor)
            else:
                if record["kind"] != "tensor" or tensor.shape != value.shape or tensor.dtype != value.dtype:
                    raise ValueError("official count state schema differs")
                merged[record["name"]] = value+tensor
    for name,value in merged.items():
        setattr(head,name,[torch.cat(value)] if isinstance(value,list) else value)
    metric._updates = metric._metric._update_count = head._update_count = updates
    return official_compute_nullable(metric)


def bootstrap_locked(assets, *, role, control, candidate):
    """Paired reference resampling; all three process differences stay paired."""
    import numpy as np

    if role not in {"SEL","PB"} or control is None or candidate is None:
        return dict(status="NOT_AVAILABLE",reason="no complete locked endpoint pair")
    root = Path(assets["run_root"])
    banks, observed, references = {}, [], None
    for repeat in (0,1,2):
        results = {}
        for name, endpoint in (("control",control),("candidate",candidate)):
            folder = root/"evaluation"/f"{endpoint['variant']}-s{endpoint['adapter_seed']}-u{endpoint['step']:06d}"/role/f"repeat{repeat}"
            result = read_json(folder/"RESULTS.json")
            request = result["request"]
            if (result["status"] != "COMPLETE" or request["role"] != role
                    or request["repeat_index"] != repeat
                    or (request["variant"],request["adapter_seed"],request["step"]) !=
                       (endpoint["variant"],endpoint["adapter_seed"],endpoint["step"])):
                raise ValueError("bootstrap requires complete, identity-matched locked evaluations")
            results[name] = result
            items = {(r["horizon"],r["kind"],r["reference_id"]):r for r in result["references"]
                     if r["horizon"] in (3,4,5)}
            keys = set(items)
            if references is None:
                references = keys
            elif references != keys:
                raise ValueError("paired bootstrap reference/horizon populations differ")
            for key, row in items.items():
                path = folder/row["evidence"]
                if sha256(path) != row["sha256"]:
                    raise ValueError("bootstrap sufficient-state bytes differ")
                banks[(repeat,name,*key)] = read_json(path)
        rowsets = {name:{r["horizon"]:r for r in result["rows"]} for name,result in results.items()}
        observed.append({kind:sum(rowsets["candidate"][h][field]-rowsets["control"][h][field]
                                 for h in (3,4,5))/3
                         for kind,field in (("S_long","AP"),("S_current","current_AP"))})
    reference_ids = sorted({key[2] for key in references})
    rng = np.random.default_rng(9101)
    draws = []
    for sample in range(1000):
        selected = rng.choice(reference_ids,size=len(reference_ids),replace=True).tolist()
        differences = {"S_long":[],"S_current":[]}
        defined = True
        for repeat in (0,1,2):
            repeat_delta = {"S_long":[],"S_current":[]}
            for h in (3,4,5):
                for kind, score, statistic in (("long","online_t-mAP","S_long"),
                                                ("current","raw_local_AP","S_current")):
                    values = {}
                    for name in ("control","candidate"):
                        states = [banks[repeat,name,h,kind,ref] for ref in selected
                                  if (h,kind,ref) in references]
                        if not states:
                            defined = False
                            break
                        values[name] = pool_official_states(states,dataset_spec=assets["taxonomy_path"])[score]
                        if values[name] is None:
                            defined = False
                            break
                    if not defined:
                        break
                    repeat_delta[statistic].append(values["candidate"]-values["control"])
                if not defined:
                    break
            if not defined:
                break
            for key in differences:
                differences[key].append(sum(repeat_delta[key])/3)
        draws.append(dict(index=sample,delta={key:sum(values)/3 if defined else None
                                            for key,values in differences.items()}))
    defined_count = sum(row["delta"]["S_long"] is not None for row in draws)
    intervals = {key:np.percentile([r["delta"][key] for r in draws],[2.5,97.5]).tolist()
                 if defined_count == 1000 else None for key in ("S_long","S_current")}
    result = dict(status="COMPLETE",role=role,bootstrap_seed=9101,samples=1000,
        reference_count=len(reference_ids),reference_ids=reference_ids,defined_draws=defined_count,
        undefined_draws=1000-defined_count,intervals_95=intervals,unit="0-1",
        interval_status="DEFINED" if defined_count==1000 else "NA_UNDEFINED_POOLED_AP_DRAWS",
        panel_warning="INTERVAL_UNSTABLE_FEWER_THAN_TEN_REFERENCES" if len(reference_ids)<10 else None,
        process_differences=observed,
        process_difference_ranges={key:max(r[key] for r in observed)-min(r[key] for r in observed)
                                   for key in ("S_long","S_current")},draws=draws)
    write_json_atomic(ARTIFACTS/f"BOOTSTRAP_{role}.json",result)
    return result


def raw_prediction_fingerprint(output):
    hasher = hashlib.sha256()
    for name in ("pred_logits", "pred_masks", "mpr_support_probabilities"):
        values = output[name]
        values = values if isinstance(values, list) else [values]
        for value in values:
            tensor = value.detach().cpu().contiguous()
            hasher.update(name.encode())
            hasher.update(str((tuple(tensor.shape), tensor.dtype)).encode())
            hasher.update(tensor.reshape(-1).view(__import__("torch").uint8).numpy().tobytes())
    return hasher.hexdigest()


def current_mask_diagnostic(prediction, target):
    """Class-aware one-to-one IoU>=.5 matches of all official candidates."""
    from scripts.p6a_metrics import match_instances_hungarian

    pairs = match_instances_hungarian(target["masks"],prediction["pred_masks"].T.contiguous(),
        gt_classes=target["labels"],pred_classes=prediction["pred_classes"],threshold=.5)
    matched = {int(target["ids"][g]) for g,_ in pairs}
    return dict(matched_gt_ids=sorted(matched),
        unmatched_gt_ids=sorted(set(target["ids"].tolist())-matched),
        gt_count=int(target["ids"].numel()),candidate_count=int(prediction["pred_classes"].numel()),
        matching="class-aware one-to-one Hungarian IoU>=0.5; all official candidates including ephemeral")


def replay_episode(system, loader, record, *, provenance,measure_only=False):
    import torch
    from datasets.revisit_prompt_episode import RevisitEpisodeSpec
    from scripts.evaluate_persist4d import _metric_target
    from scripts.evaluate_persist4d_p6a import build_rio_class_mapper
    from scripts.revisit_prompt_runtime import RevisitEpisodeRuntime, trajectory_sidecar
    from scripts.system_comparison_metrics import match_identity_update, compute_deployment_identity_metrics
    from scripts.system_comparison_v2_inference import OfficialCandidateTrajectoryAccumulator
    from models.revisit_prompt_modules import tensor_summary

    spec = RevisitEpisodeSpec.from_record(record)
    dataset = loader.dataset(spec.dataset_scope)
    mapper = build_rio_class_mapper(dataset, foreground_class_count=system.model.num_classes-1)
    runtime = RevisitEpisodeRuntime(spec, class_count=system.model.num_classes, device="cuda:0")
    trajectory = OfficialCandidateTrajectoryAccumulator(score_reducer="mean")
    identities, timing_rows, structures, mask_diagnostics = [], [], [], []
    samples = iter(loader.stages(spec))
    episode_start = time.perf_counter()
    for j in range(spec.horizon):
        before = time.perf_counter()
        stage = next(samples).to("cuda:0")
        torch.cuda.synchronize()
        loaded = time.perf_counter()
        input_structure = None
        if system.model.mpr_capture_probe:
            meta = stage.stage_meta[0]
            input_structure = dict(coordinates=tensor_summary(stage.batch[0].coordinates),
                features=tensor_summary(stage.batch[0].features),point2segment=tensor_summary(meta.point2segment),
                inverse=tensor_summary(meta.voxel_inverse),local_stages=tensor_summary(meta.local_stage_ids))
        output = runtime.forward(system, stage)
        torch.cuda.synchronize()
        forwarded = time.perf_counter()
        finished = runtime.finish(system, output, stage, class_mapper=mapper,measure=True)
        torch.cuda.synchronize()
        committed = time.perf_counter()
        meta = stage.stage_meta[0]
        raw_fingerprint = raw_prediction_fingerprint(output)
        source_order = record.get("source_unit", "").rsplit(":",1)[-1]
        order = source_order if source_order in {"canonical", "reverse", "sha256_seed45"} else "canonical"
        sidecar = trajectory_sidecar(finished["official"], meta, history=runtime.history,
            provenance={**provenance,"source_raw_observation_fingerprint":raw_fingerprint}, order_id=order)
        trajectory.add_stage(sidecar, finished["track_step"])
        predicted = time.perf_counter()
        target = _metric_target(stage.common_truth, dataset)
        current_target = current_metric_truth(target, j)
        local_current = meta.local_stage_ids == len(meta.scan_ids_in_window)-1
        current_prediction = finished["official"].prediction(latest_only=True)
        current_prediction = {k: current_prediction[k] for k in ("pred_masks","pred_scores","pred_classes")}
        if not measure_only:
            query_ids = (finished["commit"].query_to_logical_id[0] >= 0).nonzero().flatten()
            probabilities = finished["observation"].class_prob[0, query_ids].detach().clone()
            probabilities[:, system.model.task_background_class] = -torch.inf
            classes = torch.tensor([mapper(int(c)) for c in probabilities.argmax(-1).cpu()], dtype=torch.long)
            masks = (output["pred_masks"][0][meta.full_resolution_point2segment][:,query_ids] >= 0)
            masks = masks[local_current].detach().cpu()
            identities.append(match_identity_update(horizon=j+1, gt_ids=current_target["ids"],
                gt_classes=current_target["labels"], gt_masks=current_target["masks"],
                issued_ids=finished["commit"].query_to_logical_id[0,query_ids].detach().cpu(),
                pred_classes=classes, pred_masks=masks, minimum_iou=.5))
            mask_diagnostics.append(dict(stage=j,**current_mask_diagnostic(current_prediction,current_target)))
        structures.append(dict(stage=j, window_scan_ids=list(meta.scan_ids_in_window),
            points=int(meta.local_stage_ids.numel()), segments=int(meta.segment_stage_ids.numel()),
            current_points=int(local_current.sum()), occupied_slots=int(runtime.state.occupied.sum()),
            state_bytes=runtime.state.state_bytes+runtime.geometry.nbytes(),
            raw_prediction_sha256=raw_fingerprint))
        if system.model.mpr_capture_probe:
            structures[-1]["pre_first_mask"] = output["mpr_pre_mask_probe"]
            structures[-1]["input_structure"] = input_structure
        timing_rows.append(dict(stage=j, io_seconds=loaded-before, forward_seconds=forwarded-loaded,
            commit_and_postprocess_seconds=committed-forwarded, **finished["timings"],
            output_accumulation_seconds=predicted-committed,end_to_end_seconds=predicted-before,
            evaluation_only_seconds=time.perf_counter()-predicted))
        del output, finished, stage
    snapshot = trajectory.snapshot()
    if snapshot.prediction["pred_masks"].shape[0] != target["masks"].shape[1]:
        raise ValueError("committed current vertices differ from the common causal prefix")
    return dict(prediction=snapshot.prediction, target=target, current_prediction=current_prediction,
                current_target=current_target, identity=compute_deployment_identity_metrics(identities) if not measure_only else None,
                identity_assignments=[dict(horizon=u.horizon,visible_gt_ids=list(u.visible_gt_ids),
                                           assignments=u.assignments) for u in identities],
                structures=structures, timings=timing_rows,current_mask_diagnostics=mask_diagnostics,
                seconds=time.perf_counter()-episode_start)


def prepare_evaluation_folder(folder, root, retry):
    if folder.exists() and any(folder.iterdir()):
        old = read_json(folder/"STATUS.json")
        ended = any(event["event_id"] == old.get("charge_event_id") and event["status"] == "FAILED"
                    for event in ledger_events(root/"COST_LEDGER.jsonl"))
        if (old["status"] == "COMPLETE" or retry != old.get("retry",0)+1 or retry not in (1,2)
                or not (old["status"] == "FAILED" or ended)):
            raise ValueError("evaluation retry needs a failed owned attempt; completed results remain frozen")
        archived = folder.parent/"failed_attempts"/f"{folder.name}-attempt{retry-1}"
        archived.parent.mkdir(parents=True,exist_ok=True)
        if archived.exists():
            raise ValueError("failed evaluation archive already exists")
        folder.rename(archived)
    elif retry:
        raise ValueError("evaluation retry has no preceding owned attempt")
    folder.mkdir(parents=True,exist_ok=True)


def evaluate(args, assets):
    trained = args.variant != "M0"
    plan = read_run_plan(assets) if trained else None
    selection = None
    if args.role != "CAL" or args.seed == 147:
        path = Path(args.selection_lock or (assets["selection_lock"] if args.seed == 146 or args.role == "CAL"
                                           else ARTIFACTS/"REPLICATION_LOCK.json"))
        selection = read_json(path)
    validate_evaluation_request(variant=args.variant,seed=args.seed,step=args.step,role=args.role,
                                run_plan=plan,selection=selection)
    manifest = read_json(ARTIFACTS/"DATA_MANIFEST.json")
    records = sorted(manifest["records"][args.role], key=lambda r:(r["horizon"],r["input_id"]))
    expected = {int(h):n for h,n in manifest["counts"][args.role].items()}
    request = dict(status="DRY_RUN_NO_EVALUATION",variant=args.variant,adapter_seed=args.seed,
        step=args.step,role=args.role,repeat_index=args.repeat_index,evaluation_seed=9001,
        parent_seed=45,parent_sha256=assets["hashes"]["parent"],numeric_sources=numeric_sources(),
        expected_inputs=expected,input_count=len(records),population_sha256=sha256(ARTIFACTS/"DATA_MANIFEST.json"))
    if args.dry_run:
        return request
    phase = "P1" if args.role == "CAL" and not trained else ("P3" if args.role == "CAL" else "P4")
    require_charged_child(phase)
    import torch
    from datasets.native_long_dataset import isolated_rng, stable_seed
    from datasets.revisit_prompt_episode import RevisitEpisodeLoader
    from scripts.native_long_memory_bridge import pool_identity
    from scripts.p6a_metrics import OfficialMetricAccumulator
    from scripts.revisit_prompt_assets import load_revisit_bundle

    configure_runtime(9001)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("MPR evaluation requires exactly one visible CUDA device")
    root = Path(assets["run_root"])
    folder = root/"evaluation"/f"{args.variant}-s{args.seed}-u{args.step:06d}"/args.role/f"repeat{args.repeat_index}"
    prepare_evaluation_folder(folder,root,args.retry)
    # Construction failures also belong to an explicit owned attempt.
    write_json_atomic(folder/"STATUS.json",{**request,"status":"INITIALIZING","retry":args.retry,
        "charge_event_id":__import__("os").environ["MPR_CHARGED_EVENT_ID"]})
    if trained:
        delta = Path(args.checkpoint or root/"trajectories"/f"{args.variant}-s{args.seed}"/f"delta-{args.step:06d}.pt")
        metadata = torch.load(delta, map_location="cpu",weights_only=True)
        if (metadata["variant"],metadata["adapter_seed"],metadata["step"]) != (args.variant,args.seed,args.step):
            raise ValueError("delta metadata differs from the requested endpoint")
        if metadata["numeric_sources"] != numeric_sources():
            raise ValueError("evaluation delta numerical identity differs")
        system = load_revisit_bundle(delta, assets=assets)
        config = system.config
        checkpoint_sha = sha256(delta)
    else:
        system, config, _ = build_system(assets,variant="M0",seed=146)
        checkpoint_sha = assets["hashes"]["parent"]
    system.to("cuda:0").eval()
    loader = RevisitEpisodeLoader(config,assets["data_root"],read_json(assets["formal_population_path"]))
    provenance = dict(checkpoint_sha256=checkpoint_sha,config_hash=configuration_digest(config,assets),
                      protocol_manifest_hash=request["population_sha256"])
    def metric(h, current=False):
        return OfficialMetricAccumulator(mode="raw_local" if h==1 or current else "strict_online",
            dataset_spec=assets["taxonomy_path"], min_region_size=int(config.instance_metric.get("min_region_size",100)))
    pooled = {(h,kind):metric(h,current=kind=="current") for h in expected for kind in ("long","current")}
    by_reference, counts, identity_rows, evidence_files = {}, collections.Counter(), [], []
    setup_seconds = time.perf_counter()-float(__import__("os").environ["MPR_JOB_STARTED_MONOTONIC"])
    status = {**request,"status":"RUNNING","completed_inputs":{h:0 for h in expected},
              "retry":args.retry,"charge_event_id":__import__("os").environ["MPR_CHARGED_EVENT_ID"],
              "startup_seconds":setup_seconds}
    write_json_atomic(folder/"STATUS.json",status)
    try:
        with torch.no_grad():
            for record in records:
                with isolated_rng(stable_seed(9001,record["input_id"]),cuda_devices=(0,)):
                    episode = replay_episode(system,loader,record,provenance=provenance)
                h, ref = record["horizon"],record["reference_id"]
                metrics_start = time.perf_counter()
                for kind in ("long","current"):
                    prediction = episode["prediction" if kind=="long" else "current_prediction"]
                    target = episode["target" if kind=="long" else "current_target"]
                    pooled[h,kind].update(prediction,target)
                    key = h,kind,ref
                    if key not in by_reference:
                        by_reference[key] = metric(h,current=kind=="current")
                    by_reference[key].update(prediction,target)
                identity_rows.append(dict(input_id=record["input_id"],reference_id=ref,horizon=h,
                    **episode["identity"],assignments=episode["identity_assignments"],
                    structures=episode["structures"],timings=episode["timings"],seconds=episode["seconds"],
                    current_mask_diagnostics=episode["current_mask_diagnostics"],
                    metric_update_seconds=time.perf_counter()-metrics_start))
                counts[h] += 1
                status["completed_inputs"] = dict(counts)
                write_json_atomic(folder/"STATUS.json",status)
        if dict(counts) != expected:
            raise ValueError("completed evaluation input denominator differs from the bound population")
        rows = []
        serialization_start = time.perf_counter()
        for h in sorted(expected):
            long_metrics, current_metrics = official_compute_nullable(pooled[h,"long"]),official_compute_nullable(pooled[h,"current"])
            rows.append(dict(variant=args.variant,adapter_seed=args.seed,parent_seed=45,step=args.step,
                role=args.role,repeat_index=args.repeat_index,horizon=h,status="COMPLETE",unit="0-1",
                AP=long_metrics["raw_local_AP" if h==1 else "online_t-mAP"],
                current_AP=current_metrics["raw_local_AP"],expected_inputs=expected[h],completed_inputs=counts[h],
                official_metrics=long_metrics,official_current_metrics=current_metrics,
                identity=pool_identity([r for r in identity_rows if r["horizon"]==h])))
            for kind in ("long","current"):
                path = folder/f"official-{kind}-T{h}.json"
                write_json_atomic(path,pooled[h,kind].export_evidence())
                evidence_files.append(dict(name=path.name,sha256=sha256(path),horizon=h,kind=kind))
        reference_rows = []
        for (h,kind,ref), accumulator in sorted(by_reference.items()):
            path = folder/"references"/f"{ref}-T{h}-{kind}.json"
            write_json_atomic(path,accumulator.export_evidence())
            reference_rows.append(dict(reference_id=ref,horizon=h,kind=kind,metrics=official_compute_nullable(accumulator),
                evidence=path.relative_to(folder).as_posix(),sha256=sha256(path)))
        write_json_atomic(folder/"RESULTS.json",dict(status="COMPLETE",request=request,rows=rows,
            references=reference_rows,identity=identity_rows,official_evidence=evidence_files))
        status.update(status="COMPLETE",completed_inputs=dict(counts),checkpoint_sha256=checkpoint_sha,
                      metric_serialization_seconds=time.perf_counter()-serialization_start)
        write_json_atomic(folder/"STATUS.json",status)
        return status
    except BaseException as error:
        status.update(status="FAILED",error_type=type(error).__name__,error=str(error))
        write_json_atomic(folder/"STATUS.json",status)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant",default="M0",choices=[f"M{i}" for i in range(7)])
    parser.add_argument("--seed",type=int,default=146,choices=(146,147))
    parser.add_argument("--step",type=int,default=0)
    parser.add_argument("--role",required=True,choices=("CAL","SEL","PB"))
    parser.add_argument("--selection-lock")
    parser.add_argument("--checkpoint")
    parser.add_argument("--repeat-index",type=int,default=0,choices=(0,1,2))
    parser.add_argument("--retry",type=int,default=0,choices=(0,1,2),help="preserve and restart a failed owned attempt")
    parser.add_argument("--dry-run",action="store_true")
    args = parser.parse_args(argv)
    assets = None
    try:
        assets = local_assets()
        result = evaluate(args,assets)
        print(json.dumps(result,sort_keys=True,allow_nan=False))
        return 0
    except Exception as error:
        write_json_atomic(ARTIFACTS/"EVAL_ENTRY_STATUS.json",dict(status="FAILED",variant=args.variant,
            adapter_seed=args.seed,step=args.step,role=args.role,error_type=type(error).__name__,error=public_error(error,assets)))
        print(f"{type(error).__name__}: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
