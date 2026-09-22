"""Live V2 deployment measurements with the frozen producer's exact inputs."""

# Timing closures execute synchronously before the loop advances.
# ruff: noqa: B023

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import statistics
import subprocess
import time
from collections import defaultdict, deque
from functools import partial
from pathlib import Path

from scripts.perception_gain_profile import (
    ProfileError,
    _csv_bytes,
    _current_rss_bytes,
    _elapsed_ms,
    _storage_bytes,
    select_profile_specs,
    validate_profile_rows,
)

COMPONENTS = (
    "input_prepare_ms",
    "network_forward_ms",
    "scorer_ms",
    "refiner_ms",
    "state_update_ms",
    "materialize_ms",
)


def profile_inventory(lock):
    methods = {}
    for label in ("FH-R1-native", "R1-D0", "FINAL", "FH-P-native"):
        method = lock["methods"][lock["aliases"][label]]
        methods.setdefault(method["inference_identity"], method)
    return list(methods.values())


def summarize_measurements(rows, *, methods):
    validate_profile_rows(rows, methods=methods, repeats=3)
    groups = defaultdict(list)
    for row in rows:
        if any(not math.isfinite(row[key]) or row[key] < 0 for key in COMPONENTS):
            raise ProfileError("Profile component is non-finite or negative")
        if row["end_to_end_ms"] + 1e-6 < sum(row[key] for key in COMPONENTS):
            raise ProfileError("Profile end-to-end excludes a deployed component")
        groups[(row["method"], row["reference_scene_id"], row["T"])].append(row)
    summaries = []
    for key, samples in sorted(groups.items()):
        result = {
            "method": key[0],
            "reference_scene_id": key[1],
            "T": key[2],
            "master_sequence_id": samples[0]["master_sequence_id"],
            "measured_repeats": 3,
        }
        for field in (*COMPONENTS, "end_to_end_ms", "prefix_cumulative_ms"):
            values = [row[field] for row in samples]
            result.update(
                {
                    f"median_{field}": statistics.median(values),
                    f"minimum_{field}": min(values),
                    f"maximum_{field}": max(values),
                }
            )
        for field in (
            "peak_allocated_bytes",
            "peak_reserved_bytes",
            "cpu_rss_bytes",
            "resident_state_bytes",
            "soft_buffer_bytes",
            "archive_bytes",
        ):
            result[f"maximum_{field}"] = max(row[field] for row in samples)
        summaries.append(result)
    return summaries


class _ScorerClock:
    def __init__(self, system, device):
        self.elapsed_ms = 0.0
        self.device = device
        self.handles = []
        scorer = getattr(system.model, "semantic_query_scorer", None)
        if scorer is not None:
            self.handles = [
                scorer.register_forward_pre_hook(self._before),
                scorer.register_forward_hook(self._after),
            ]

    def _before(self, module, args):
        import torch

        torch.cuda.synchronize(self.device)
        self.started = time.perf_counter_ns()

    def _after(self, module, args, output):
        import torch

        torch.cuda.synchronize(self.device)
        self.elapsed_ms += (time.perf_counter_ns() - self.started) / 1_000_000

    def close(self):
        for handle in self.handles:
            handle.remove()


def _hash_prediction(digest, prediction, keys=()):
    digest.update(repr(keys).encode())
    for name in ("pred_masks", "pred_scores", "pred_classes"):
        digest.update(prediction[name].detach().cpu().numpy().tobytes())


def _forward(system, data, low, raw_coordinates):
    import torch

    with torch.inference_mode():
        return system(
            data,
            point2segment=[low["point2segment"]],
            raw_coordinates=raw_coordinates,
            is_eval=True,
        )


def _row(
    method,
    spec,
    horizon,
    repeat,
    *,
    timings,
    cumulative,
    device,
    point2segment,
    resident,
    soft,
    archive,
):
    import torch

    return {
        "method": method["method_id"],
        "reference_scene_id": spec.reference_id,
        "master_sequence_id": spec.source_sequence_id,
        "order_id": "canonical",
        "T": horizon,
        "repeat": repeat,
        **timings,
        "prefix_cumulative_ms": cumulative,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "cpu_rss_bytes": _current_rss_bytes(),
        "resident_state_bytes": resident,
        "soft_buffer_bytes": soft,
        "archive_bytes": archive,
        "window_voxel_points": int(point2segment.numel()),
        "window_segments": int(point2segment.max().item()) + 1,
    }


def profile_online_sequence(
    *,
    method,
    system,
    head,
    episode,
    base_collator,
    spec,
    class_mapper,
    observation_settings,
    device,
    repeat,
    logical_unit_id,
    check_budget=None,
):
    """Same episode collation, seeded forward, postprocess and publisher as CAL/PB.

    The producer collates the whole in-memory episode before the first forward;
    charge that preparation at T1 so every prefix includes its real cost. The
    consumed stage batches are released after use. No metric or replay cache is
    read, and only the requested deployed head is evaluated.
    """
    import torch

    from datasets.task_memory_episode import TaskMemoryEpisodeCollator
    from models.persistent_memory import build_local_observation
    from scripts.evaluate_persist4d import (
        _move_data_to_device,
        _move_targets_to_device,
        _segment_stages,
    )
    from scripts.evaluate_persist4d_p6a import _frozen_inference_seed
    from scripts.perception_gain_profile import _TimedTransform
    from scripts.prepare_perception_refiner import advance_d0_identity
    from scripts.rescene_task_postprocess import extract_official_task_prediction
    from scripts.run_task_memory_controls import _window_observation
    from scripts.task_memory_output import LagOnePublisher
    from scripts.train_perception_refiner import CausalRefinerRevisionTransform

    parent = LagOnePublisher(score_reducer="mean", iou_threshold=0.5)
    transform = (
        _TimedTransform(
            CausalRefinerRevisionTransform(system=system, refiner=head, device=device),
            device,
        )
        if head is not None
        else None
    )
    refined = (
        LagOnePublisher(
            score_reducer="mean", iou_threshold=0.5, revision_mask_transform=transform
        )
        if transform
        else None
    )
    association = method.get("association_config")
    association_state, buffer, boundary = None, None, None
    if association:
        from models.crosswindow_state import CrossWindowState
        from models.overlap_entity_association import (
            AssociationConfig,
            associate,
            build_evidence,
            commit_observation,
        )
        from scripts.crosswindow_cache import build_canonical_frame
        from scripts.replay_crosswindow_association import evaluator_prediction, publish

        association = AssociationConfig(**association)
    state, rows, cumulative = None, [], 0.0
    digest, clock = hashlib.sha256(), _ScorerClock(system, device)
    try:
        with _frozen_inference_seed(method["eval_seed"], device):
            batch, collation_ms = _elapsed_ms(
                device, lambda: TaskMemoryEpisodeCollator(base_collator)([episode])
            )
            batches = deque(batch.stage_batches)
            del batch
            while batches:
                if check_budget is not None:
                    check_budget()
                stage_batch = batches.popleft()
                meta = stage_batch.stage_meta[0]
                horizon = meta.absolute_stage_index + 1
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
                started = time.perf_counter_ns()
                prepare_started = started
                data, targets, names = stage_batch.model_batch
                if list(names) != ["-".join(meta.scan_ids_in_window)]:
                    raise ValueError("Profile stage identity changed")
                full = data.target_full[0]
                data = _move_data_to_device(data, device)
                low = _move_targets_to_device(targets, device)[0]
                segment_stages = _segment_stages(low)
                latest = int(segment_stages.max().item())
                raw_coordinates = system._process_raw_coordinates(data)
                torch.cuda.synchronize(device)
                prepare_ms = (time.perf_counter_ns() - prepare_started) / 1_000_000
                if horizon == 1:
                    prepare_ms += collation_ms
                scorer_before = clock.elapsed_ms

                output, network_ms = _elapsed_ms(
                    device, partial(_forward, system, data, low, raw_coordinates)
                )
                scorer_ms = clock.elapsed_ms - scorer_before

                def observe():
                    local = build_local_observation(
                        output,
                        [segment_stages],
                        latest_stage=latest,
                        **observation_settings,
                    )
                    return _window_observation(
                        local_observation=local,
                        output=output,
                        segment_stages=segment_stages,
                        latest_stage=latest,
                        confidence_threshold=observation_settings[
                            "confidence_threshold"
                        ],
                        mask_threshold=observation_settings["mask_threshold"],
                        minimum_mask_support=observation_settings[
                            "minimum_mask_support"
                        ],
                    )

                observation, state_ms = _elapsed_ms(device, observe)
                prediction, materialize_ms = _elapsed_ms(
                    device,
                    lambda: extract_official_task_prediction(
                        system=system,
                        output=output,
                        target_low_resolution=low,
                        target_full_resolution=full,
                        data=data,
                        class_mapper=class_mapper,
                        latest_stage_index=latest,
                        return_soft_evidence=True,
                    ),
                )
                refiner_before = transform.elapsed_ms if transform else 0.0
                if association:

                    def update_association():
                        nonlocal association_state, buffer
                        frame = build_canonical_frame(
                            producer_id="R1-B4-policy",
                            order_id=logical_unit_id,
                            observation=observation,
                            prediction=prediction,
                            stage_meta=meta,
                        )
                        if association_state is None:
                            association_state = CrossWindowState.empty(
                                capacity=100,
                                feature_dim=observation.features.shape[-1],
                                class_count=observation.class_prob.shape[-1],
                            )
                        plan = associate(
                            build_evidence(frame, association_state, buffer),
                            association,
                        )
                        association_state, committed = commit_observation(
                            frame, association_state, plan
                        )
                        buffer = committed.buffer
                        return frame, plan, committed

                    (frame, plan, committed), extra_state_ms = _elapsed_ms(
                        device, update_association
                    )
                    published, publication_ms = _elapsed_ms(
                        device,
                        lambda: publish(
                            frame, plan, mask_selection="new", history_boundary=boundary
                        ),
                    )
                    boundary = published.history_boundary
                    prediction_out, convert_ms = _elapsed_ms(
                        device, lambda: evaluator_prediction(published)
                    )
                    materialize_ms += publication_ms + convert_ms
                    resident, soft, archive = (
                        committed.resident_bytes,
                        committed.buffer_bytes,
                        published.accounting.archive_bytes,
                    )
                    keys = published.keys
                else:
                    (state, identity_map), extra_state_ms = _elapsed_ms(
                        device,
                        lambda: advance_d0_identity(
                            observation=observation, state=state, stage_meta=meta
                        ),
                    )

                    def materialize():
                        original = parent.update(prediction, identity_map, meta)
                        return (
                            refined.update(
                                prediction,
                                identity_map,
                                meta,
                                reference_prefix=original,
                            )
                            if refined
                            else original
                        ), original

                    (published, original), publication_ms = _elapsed_ms(
                        device, materialize
                    )
                    refiner_ms = (
                        transform.elapsed_ms - refiner_before if transform else 0.0
                    )
                    materialize_ms += max(0.0, publication_ms - refiner_ms)
                    resident = _storage_bytes(state)
                    soft, archive = (
                        published.accounting.lag1_buffer_bytes,
                        published.accounting.archive_payload_bytes,
                    )
                    if refined:
                        soft += original.accounting.lag1_buffer_bytes
                        archive += original.accounting.archive_payload_bytes
                    prediction_out, keys = published.prediction, published.keys
                refiner_ms = transform.elapsed_ms - refiner_before if transform else 0.0
                torch.cuda.synchronize(device)
                end_to_end = (time.perf_counter_ns() - started) / 1_000_000 + (
                    collation_ms if horizon == 1 else 0.0
                )
                cumulative += end_to_end
                timings = dict(
                    zip(
                        COMPONENTS,
                        (
                            prepare_ms,
                            max(0.0, network_ms - scorer_ms),
                            scorer_ms,
                            refiner_ms,
                            state_ms + extra_state_ms,
                            materialize_ms,
                        ),
                        strict=True,
                    )
                )
                timings["end_to_end_ms"] = end_to_end
                if horizon >= 2:
                    rows.append(
                        _row(
                            method,
                            spec,
                            horizon,
                            repeat,
                            timings=timings,
                            cumulative=cumulative,
                            device=device,
                            point2segment=low["point2segment"],
                            resident=resident,
                            soft=soft,
                            archive=archive,
                        )
                    )
                # Serialization used for reload validation is outside the timer.
                _hash_prediction(digest, prediction_out, keys)
                if not association:
                    del original
                del (
                    stage_batch,
                    data,
                    targets,
                    low,
                    full,
                    raw_coordinates,
                    output,
                    observation,
                    prediction,
                    published,
                    prediction_out,
                )
    finally:
        clock.close()
    return rows, digest.hexdigest()


def profile_native_sequence(
    *,
    method,
    system,
    dataset,
    base_collator,
    spec,
    class_mapper,
    observation_settings,
    device,
    repeat,
    check_budget=None,
):
    import torch

    from scripts.evaluate_persist4d import _move_data_to_device, _move_targets_to_device
    from scripts.evaluate_persist4d_p6a import _frozen_inference_seed
    from scripts.system_comparison_inference import postprocess_full_history_output

    rows, io_rows, cumulative = [], [], 0.0
    digest, clock = hashlib.sha256(), _ScorerClock(system, device)
    try:
        for horizon in range(1, 6):
            if check_budget is not None:
                check_budget()
            # Matches FullHistoryPredictionProducer: seed before loading as well
            # as collation. File loading stays outside the deployment timer.
            with _frozen_inference_seed(method["eval_seed"], device):
                io_start = time.perf_counter_ns()
                sample = dataset.load_scan_indices(
                    spec.context_index,
                    tuple(spec.scan_indices[:horizon]),
                    change_file=None,
                )
                io_rows.append(
                    {
                        "method": method["method_id"],
                        "reference_scene_id": spec.reference_id,
                        "T": horizon,
                        "repeat": repeat,
                        "input_load_ms": (time.perf_counter_ns() - io_start)
                        / 1_000_000,
                        "filesystem_cache_state": "UNCONTROLLED",
                    }
                )
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
                started = time.perf_counter_ns()
                data, targets, names = base_collator([sample])
                if list(names) != [spec.source_sequence_id]:
                    raise ProfileError("Native profile changed the input prefix")
                full = data.target_full[0]
                data = _move_data_to_device(data, device)
                low = _move_targets_to_device(targets, device)[0]
                raw_coordinates = system._process_raw_coordinates(data)
                torch.cuda.synchronize(device)
                prepare_ms = (time.perf_counter_ns() - started) / 1_000_000
                scorer_before = clock.elapsed_ms

                output, network_ms = _elapsed_ms(
                    device, partial(_forward, system, data, low, raw_coordinates)
                )
                scorer_ms = clock.elapsed_ms - scorer_before
                processed, materialize_ms = _elapsed_ms(
                    device,
                    partial(
                        postprocess_full_history_output,
                        system=system,
                        output=output,
                        target_low_resolution=low,
                        target_full_resolution=full,
                        data=data,
                        horizon=horizon,
                        class_mapper=class_mapper,
                        **observation_settings,
                    ),
                )
                torch.cuda.synchronize(device)
                end_to_end = (time.perf_counter_ns() - started) / 1_000_000
                cumulative += end_to_end
                timings = dict(
                    zip(
                        COMPONENTS,
                        (
                            prepare_ms,
                            max(0.0, network_ms - scorer_ms),
                            scorer_ms,
                            0.0,
                            0.0,
                            materialize_ms,
                        ),
                        strict=True,
                    )
                )
                timings["end_to_end_ms"] = end_to_end
                if horizon >= 2:
                    rows.append(
                        _row(
                            method,
                            spec,
                            horizon,
                            repeat,
                            timings=timings,
                            cumulative=cumulative,
                            device=device,
                            point2segment=low["point2segment"],
                            resident=0,
                            soft=0,
                            archive=0,
                        )
                    )
                _hash_prediction(digest, processed.task_prediction)
                del sample, data, targets, low, full, raw_coordinates, output, processed
    finally:
        clock.close()
    return rows, digest.hexdigest(), io_rows


def load_profile_method(method, *, assets, external_root, artifacts, device):
    from scripts.perception_gain_evaluation import _load_evaluation_weights
    from scripts.perception_gain_v2_confirmation import method_inputs
    from scripts.perception_gain_v2_lock import external_path
    from scripts.perception_refiner_evaluation import _load_refiner
    from scripts.train_perception_gain import compose_variant_config
    from trainer.perception_gain_trainer import PerceptionGainTrainer

    inputs = method_inputs(method, external_root=external_root, artifacts=artifacts)
    config = compose_variant_config(
        method["architecture_variant"],
        pretrained=Path(assets["concerto_pretrained"]),
        run_dir=external_root / "profile/runtime",
        recipe_config=method["recipe"],
    )
    config.model.return_query_features = True
    if inputs["scorer_checkpoint"] is not None:
        config.perception_training.scorer_checkpoint = str(inputs["scorer_checkpoint"])
    system = PerceptionGainTrainer(config)
    _, weight, _, _, _ = _load_evaluation_weights(
        system=system,
        variant=method["architecture_variant"],
        optimizer_update=method["parent_optimizer_update"],
        r1_checkpoint=Path(assets["r1_checkpoint"]),
        checkpoint=inputs["checkpoint"],
        scorer_checkpoint=inputs["scorer_checkpoint"],
    )
    if weight != method["parent_weight_sha256"]:
        raise ProfileError("Profile parent differs from the immutable lock")
    system.to(device).eval().requires_grad_(False)
    head = None
    if descriptor := method.get("refiner"):
        head, identity = _load_refiner(
            external_path(descriptor["checkpoint"], external_root),
            optimizer_update=descriptor["optimizer_update"],
            device=device,
            expected_binding=descriptor["binding"],
        )
        if (
            identity["sha256"] != descriptor["checkpoint_sha256"]
            or head.input_mode != descriptor["input_mode"]
        ):
            raise ProfileError(
                "Profile refiner differs from the locked head or input mode"
            )
    return system, config, head


def profile_population(model_config, *, assets):
    import hydra

    from scripts.evaluate_persist4d_p6a import build_rio_class_mapper
    from scripts.perception_gain_evaluation import (
        DATA_CONTRACT,
        PROJECT_ROOT,
        build_role_base_dataset,
        select_live_population_units,
    )
    from scripts.perception_gain_v2 import read_json
    from scripts.run_task_memory_policy_baseline import (
        PROTOCOL_B_POPULATION_ID,
        _episode_specs,
        build_baseline_population,
    )

    base = build_role_base_dataset(
        config=model_config, data_root=Path(assets["data_root"]), role="PB", horizon=5
    )
    dataset, masters, manifest_sha = build_baseline_population(
        base,
        data_contract=read_json(DATA_CONTRACT),
        metadata_path=Path(assets["rio_metadata"]),
        population_id=PROTOCOL_B_POPULATION_ID,
        protocol_b_manifest_path=PROJECT_ROOT
        / "artifacts/P6A/protocol_b_manifest.json",
    )
    specs = _episode_specs(masters)
    selected = select_profile_specs(specs)
    units = select_live_population_units(
        role="PB",
        role_references=[spec.reference_id for spec in selected],
        episode_specs=specs,
        expected_logical_units=129,
    )
    ids = {unit.spec_index: unit.logical_unit_id for unit in units}
    selected_units = [(spec, ids[specs.index(spec)]) for spec in selected]
    collator = hydra.utils.instantiate(model_config.data.validation_collation)
    return (
        dataset,
        collator,
        build_rio_class_mapper(dataset),
        selected_units,
        manifest_sha,
    )


def _assert_exclusive_device(device):
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    physical = visible[device.index] if visible and visible[0] else str(device.index)
    result = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            physical,
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    owners = {
        int(line.strip())
        for line in result.stdout.splitlines()
        if line.strip().isdigit()
    }
    if owners - {os.getpid()}:
        raise ProfileError("Another process is using the profiling GPU")


def resource_comparison(rows, *, candidate, baseline):
    cells = {}
    for method in (candidate, baseline):
        selected = [
            row for row in rows if row["method"] == method and row["T"] in (4, 5)
        ]
        if len(selected) != 36:
            return {"status": "UNCONFIRMED"}
        cells[method] = {
            "latency": statistics.median(row["end_to_end_ms"] for row in selected),
            "memory": statistics.median(
                row["peak_allocated_bytes"] for row in selected
            ),
        }
    relative = {
        key: cells[candidate][key] / cells[baseline][key]
        for key in ("latency", "memory")
    }
    status = (
        "IMPROVED"
        if all(value <= 1 for value in relative.values())
        and any(value < 1 for value in relative.values())
        else "REGRESSED"
        if all(value > 1 for value in relative.values())
        else "TRADEOFF"
    )
    return {
        "status": status,
        "candidate": candidate,
        "baseline": baseline,
        "ratios": relative,
        "scope": "Median of measured T4/T5 samples; descriptive comparison without a statistical stability claim.",
    }


def run_profile_v2(config, *, external_root):
    import torch
    import yaml

    from datasets.task_memory_episode import TaskMemoryEpisodeDataset
    from scripts.evaluate_persist4d import _validate_cuda_device
    from scripts.perception_gain_v2 import (
        PROJECT_ROOT,
        file_hash,
        read_json,
        utc_now,
        write_json,
    )
    from scripts.perception_gain_v2_bundle import (
        build_method_bundle,
        instantiate_bundle,
    )
    from scripts.perception_gain_v2_confirmation import load_lock
    from scripts.perception_gain_v2_lock import external_path
    from scripts.perception_gain_v2_publication import selected_release_methods
    from scripts.system_comparison_inference import deterministic_inference_runtime

    artifacts = PROJECT_ROOT / config["artifact_root"]
    lock, lock_sha = load_lock(config, artifacts)
    assets = read_json(external_root / "assets.local.json")
    methods = profile_inventory(lock)
    device = _validate_cuda_device("cuda:0")
    if "A40" not in torch.cuda.get_device_name(device):
        raise ProfileError("The fixed profile requires an idle NVIDIA A40")
    _assert_exclusive_device(device)
    previous_cost = {
        row["event_id"]: row
        for line in (artifacts / "budget/LEDGER.jsonl").read_text().splitlines()
        if line.strip()
        for row in (json.loads(line),)
        if row.get("task") == "PROFILE"
    }
    available = max(
        0.0,
        config["budget"]["profile_maximum"]
        - sum(row["gpu_hours"] for row in previous_cost.values()),
    )
    started = time.monotonic()
    deadline = started + available * 3600

    def check_budget():
        if time.monotonic() >= deadline:
            raise ProfileError("Measured profile reached its cumulative GPU-hour cap")
        _assert_exclusive_device(device)

    p6a = yaml.safe_load((PROJECT_ROOT / "conf/p6a/default.yaml").read_text())[
        "baselines"
    ]["b4"]
    observation_settings = {
        key: p6a[key]
        for key in (
            "background_class",
            "confidence_threshold",
            "mask_threshold",
            "minimum_mask_support",
        )
    }
    rows, io_rows, failures, reloads = [], [], [], []
    selected_identity = None
    population_sha = None
    output_root = artifacts / "resources"
    output_root.mkdir(parents=True, exist_ok=True)
    all_methods = {method["inference_identity"]: method for method in methods}
    for method in selected_release_methods(lock):
        all_methods.setdefault(method["inference_identity"], method)
    measured_ids = {method["inference_identity"] for method in methods}
    for identity, method in all_methods.items():
        system, head, reloaded, reloaded_head = None, None, None, None
        try:
            check_budget()
            bundle = build_method_bundle(
                method, external_root=external_root, artifacts=artifacts
            )
            system, model_config, head = load_profile_method(
                method,
                assets=assets,
                external_root=external_root,
                artifacts=artifacts,
                device=device,
            )
            dataset, collator, mapper, selected, manifest_sha = profile_population(
                model_config, assets=assets
            )
            current_identity = [
                {
                    "reference_scene_id": spec.reference_id,
                    "master_sequence_id": spec.source_sequence_id,
                    "context_index": spec.context_index,
                    "logical_unit_id": unit,
                }
                for spec, unit in selected
            ]
            if selected_identity is None:
                selected_identity, population_sha = current_identity, manifest_sha
            elif (
                selected_identity != current_identity or population_sha != manifest_sha
            ):
                raise ProfileError("Profile methods use different canonical PB inputs")

            def sequence(active_system, active_head, spec, unit, repeat):
                if method["kind"] == "NATIVE":
                    values, digest, io = profile_native_sequence(
                        method=method,
                        system=active_system,
                        dataset=dataset,
                        base_collator=collator,
                        spec=spec,
                        class_mapper=mapper,
                        observation_settings=observation_settings,
                        device=device,
                        repeat=repeat,
                        check_budget=check_budget,
                    )
                else:
                    io_start = time.perf_counter_ns()
                    episode = TaskMemoryEpisodeDataset(
                        dataset, (spec,), apply_augmentation=False
                    )[0]
                    io = [
                        {
                            "method": method["method_id"],
                            "reference_scene_id": spec.reference_id,
                            "T": 5,
                            "repeat": repeat,
                            "input_load_ms": (time.perf_counter_ns() - io_start)
                            / 1_000_000,
                            "filesystem_cache_state": "UNCONTROLLED",
                        }
                    ]
                    values, digest = profile_online_sequence(
                        method=method,
                        system=active_system,
                        head=active_head,
                        episode=episode,
                        base_collator=collator,
                        spec=spec,
                        class_mapper=mapper,
                        observation_settings=observation_settings,
                        device=device,
                        repeat=repeat,
                        logical_unit_id=unit,
                        check_budget=check_budget,
                    )
                return values, digest, io

            direct_hash = None
            with deterministic_inference_runtime(45, device):
                for position, (spec, unit) in enumerate(
                    selected if identity in measured_ids else selected[:1]
                ):
                    expected_hash = None
                    for repeat in range(-1, 3) if identity in measured_ids else (-1,):
                        check_budget()
                        values, digest, io = sequence(system, head, spec, unit, repeat)
                        if expected_hash is not None and expected_hash != digest:
                            raise ProfileError("Repeated live profile outputs differ")
                        expected_hash = digest
                        if position == 0:
                            direct_hash = digest
                        if identity in measured_ids:
                            io_rows.extend(io)
                            if repeat >= 0:
                                rows.extend(values)
                                (output_root / "RAW_PROFILE.csv").write_bytes(
                                    _csv_bytes(rows, tuple(rows[0]))
                                )
                        write_json(
                            external_root / "profile/PROGRESS.json",
                            {
                                "method": method["method_id"],
                                "completed_references": position + 1,
                                "repeat": repeat,
                                "measured_rows": len(rows),
                                "updated_utc": utc_now(),
                            },
                        )
            del system, head
            system, head = None, None
            gc.collect()
            torch.cuda.empty_cache()
            check_budget()
            reloaded, reloaded_head, _ = instantiate_bundle(
                external_path(bundle["logical_reference"], external_root),
                base_checkpoint=Path(assets["r1_checkpoint"]),
                pretrained=Path(assets["concerto_pretrained"]),
                runtime_root=external_root / "profile/bundle-reload",
                device=device,
            )
            with deterministic_inference_runtime(45, device):
                _, restored_hash, _ = sequence(
                    reloaded, reloaded_head, *selected[0], -1
                )
            if direct_hash != restored_hash:
                raise ProfileError(
                    "Real panel output differs after deployment bundle reload"
                )
            evidence = {
                "status": "PASS",
                "bundle_sha256": bundle["sha256"],
                "lock_sha256": lock_sha,
                "input_manifest_sha256": lock["input_manifest_sha256"],
                "panel": current_identity[0],
                "direct_output_sha256": direct_hash,
                "reloaded_output_sha256": restored_hash,
                "input_mode": method.get("refiner", {}).get("input_mode")
                if method.get("refiner")
                else None,
            }
            bundle.update(live_panel_reload="PASS", live_panel_evidence=evidence)
            safe = external_path(bundle["logical_reference"], external_root).stem
            write_json(artifacts / f"publication/bundles/{safe}.json", bundle)
            reloads.append({"method": method["method_id"], **evidence})
        except (OSError, RuntimeError, ValueError, KeyError, IndexError) as error:
            failures.append({"method": method["method_id"], "reason": str(error)})
        finally:
            del system, head, reloaded, reloaded_head
            gc.collect()
            torch.cuda.empty_cache()
    summaries = []
    try:
        summaries = summarize_measurements(
            rows, methods=[method["method_id"] for method in methods]
        )
    except (ValueError, RuntimeError) as error:
        failures.append({"scope": "coverage", "reason": str(error)})
    if summaries:
        (output_root / "PROFILE_MEDIANS.csv").write_bytes(
            _csv_bytes(summaries, tuple(summaries[0]))
        )
    if io_rows:
        (output_root / "INPUT_LOAD_DIAGNOSTIC.csv").write_bytes(
            _csv_bytes(io_rows, tuple(io_rows[0]))
        )
    final_id = lock["aliases"]["FINAL"]
    resource = (
        resource_comparison(
            rows, candidate=final_id, baseline=lock["aliases"]["FH-R1-native"]
        )
        if not failures
        else {"status": "UNCONFIRMED"}
    )
    result = {
        "status": "COMPLETE" if not failures else "BLOCKED",
        "schema_version": "perception-gain-profile-v2",
        "resource_status": resource["status"],
        "comparison": resource,
        "lock_sha256": lock_sha,
        "device": torch.cuda.get_device_name(device),
        "warmup_full_sequences": 1,
        "measured_full_sequences": 3,
        "methods": methods,
        "selected_sequences": selected_identity,
        "population_manifest_sha256": population_sha,
        "measurement_rows": len(rows),
        "bundle_reload_checks": reloads,
        "failures": failures,
        "elapsed_seconds": time.monotonic() - started,
        "profile_gpu_hour_limit": available,
        "source_sha256": file_hash(Path(__file__)),
        "scope": {
            "network_input": list(COMPONENTS[:3]),
            "state_and_revision": ["state_update_ms", "refiner_ms"],
            "usable_output": "materialize_ms",
            "end_to_end": "In-memory input preparation and transfer, live networks, state, revisions and output; excludes file loading, metrics and validation hashes.",
            "io_diagnostic": "Observed input loading is separate. OS cache state is uncontrolled; this is not a guaranteed cold-cache benchmark.",
            "memory": "Absolute CUDA peaks, process RSS, actual resident/soft/archive storage. Binary history grows with T.",
        },
    }
    write_json(output_root / "PROFILE_SUMMARY.json", result)
    return result
