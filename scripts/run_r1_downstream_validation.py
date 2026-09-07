"""Resumable execution for the frozen R1 downstream validation."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = PROJECT_ROOT / "configs/r1_downstream_validation/default.yaml"
DEFAULT_PROTOCOL = PROJECT_ROOT / "artifacts/P6A/protocol_b_manifest.json"
DEFAULT_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts/r1_downstream_validation_v1"
DEFAULT_CHECKPOINT = Path(
    "/mnt/shared/ww/persist4d-rescene-finalization/checkpoints/"
    "629ff7624dcac15e6022906e808e2e05b3ec61c60a1116ab0e278f0cfd2368dd.ckpt"
)
DEFAULT_PRETRAINED = Path(
    "/mnt/shared/ww/persist4d-node1-7-migration/checkpoints/concerto_base.pth"
)
DEFAULT_METADATA = Path("/home/ww/3RScan.json")
DEFAULT_DATA_ROOT = Path("/home/ww/paper5")
DEFAULT_CACHE_ROOT = Path(
    "/mnt/shared/ww/persist4d-r1-downstream-validation-v1/cache"
)


class R1RunError(RuntimeError):
    """Raised when R1 execution state differs from the frozen run."""


def validate_cache_execution(device_name: str) -> str:
    if device_name != "cuda:0":
        raise R1RunError("cache generation requires one process on cuda:0")
    return device_name


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _key_identity(value: Mapping[str, object]) -> str:
    return _canonical_bytes(value).decode("utf-8")


def _read_json(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise R1RunError(f"required JSON is unavailable: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise R1RunError(f"required JSON cannot be decoded: {path}") from error
    if not isinstance(value, Mapping):
        raise R1RunError(f"required JSON must contain a mapping: {path}")
    return dict(value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _publish_exact_json(path: Path, value: Mapping[str, object]) -> None:
    payload = (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file():
            raise R1RunError(f"artifact path is not a regular file: {path}")
        if path.read_bytes() == payload:
            return
        raise R1RunError(f"refusing to overwrite different artifact: {path}")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _git_head() -> str:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if len(commit) != 40:
        raise R1RunError("Git HEAD is invalid")
    return commit


def _require_clean_tracked_tree() -> None:
    for arguments in (("diff", "--quiet"), ("diff", "--cached", "--quiet")):
        completed = subprocess.run(
            ["git", *arguments], cwd=PROJECT_ROOT, check=False
        )
        if completed.returncode != 0:
            raise R1RunError("tracked source tree must be clean before execution")


def local_cache_keys(protocol_manifest: Mapping[str, object]) -> list[dict[str, object]]:
    protocol = protocol_manifest.get("protocol")
    masters = protocol_manifest.get("masters")
    if (
        not isinstance(protocol, Mapping)
        or protocol.get("order_variants")
        != ["canonical", "reverse", "sha256_seed45"]
        or isinstance(masters, (str, bytes))
        or not isinstance(masters, Sequence)
        or len(masters) != 43
    ):
        raise R1RunError("Protocol-B manifest structure differs")
    keys: list[dict[str, object]] = []
    for master in masters:
        if not isinstance(master, Mapping) or not isinstance(master.get("orders"), Mapping):
            raise R1RunError("Protocol-B master structure differs")
        for order_id in protocol["order_variants"]:
            order = master["orders"].get(order_id)
            if not isinstance(order, Mapping):
                raise R1RunError("Protocol-B order structure differs")
            visit_order = order.get("visit_order")
            if (
                isinstance(visit_order, (str, bytes))
                or not isinstance(visit_order, Sequence)
                or len(visit_order) != 5
            ):
                raise R1RunError("Protocol-B visit order differs")
            for stage in range(5):
                history = list(visit_order[: stage + 1])
                keys.append(
                    {
                        "master_sequence_id": master["master_sequence_id"],
                        "reference_scene_id": master["reference_scene_id"],
                        "order_id": order_id,
                        "stage_index": stage,
                        "history_scan_ids": history,
                        "local_window_scan_ids": history[-1:] if stage == 0 else history[-2:],
                    }
                )
    if len(keys) != 645 or len({_key_identity(key) for key in keys}) != 645:
        raise R1RunError("local cache key coverage differs")
    return keys


def select_smoke_pairs(
    local_keys: Sequence[Mapping[str, object]],
    full_keys: Sequence[Mapping[str, object]],
) -> tuple[tuple[Mapping[str, object], Mapping[str, object]], ...]:
    local_t2 = {
        (
            key.get("reference_scene_id"),
            key.get("master_sequence_id"),
            tuple(key.get("history_scan_ids", ())),
        ): key
        for key in local_keys
        if key.get("order_id") == "canonical" and key.get("stage_index") == 1
    }
    candidates = [
        key
        for key in full_keys
        if key.get("order_id") == "canonical" and key.get("horizon") == 2
    ]
    candidates.sort(
        key=lambda value: (
            str(value.get("reference_scene_id")),
            str(value.get("master_sequence_id")),
        )
    )
    selected: list[tuple[Mapping[str, object], Mapping[str, object]]] = []
    seen: set[object] = set()
    for full in candidates:
        reference = full.get("reference_scene_id")
        if reference in seen:
            continue
        identity = (
            reference,
            full.get("master_sequence_id"),
            tuple(full.get("history_scan_ids", ())),
        )
        local = local_t2.get(identity)
        if local is None:
            raise R1RunError("smoke local/FullHistory T2 pair differs")
        selected.append((local, full))
        seen.add(reference)
        if len(selected) == 6:
            break
    if len(selected) != 6:
        raise R1RunError("smoke requires six distinct Protocol-B clusters")
    return tuple(selected)


def smoke_repeat_count(selected_index: int) -> int:
    """Repeat only the first two preregistered smoke inputs once."""

    if isinstance(selected_index, bool) or not 0 <= selected_index < 6:
        raise R1RunError("smoke selected index is invalid")
    return 2 if selected_index < 2 else 1


def validate_query_feature_export_parity(
    disabled_output: Mapping[str, object],
    enabled_output: Mapping[str, object],
) -> dict[str, object]:
    """Require feature export to add only finite 128-D query features."""

    import torch

    from scripts.evaluate_persist4d import (
        _legacy_value_snapshot,
        _require_legacy_value_equal,
    )

    if not isinstance(disabled_output, Mapping) or not isinstance(
        enabled_output, Mapping
    ):
        raise R1RunError("query feature export outputs must be mappings")
    if set(enabled_output) != {*disabled_output, "query_features"}:
        raise R1RunError("query feature export changed prediction keys")
    try:
        for key in disabled_output:
            _require_legacy_value_equal(
                _legacy_value_snapshot(enabled_output[key], path=f"enabled.{key}"),
                _legacy_value_snapshot(disabled_output[key], path=f"disabled.{key}"),
                path=key,
            )
    except (TypeError, ValueError, RuntimeError) as error:
        raise R1RunError("query feature export changed legacy predictions") from error
    features = enabled_output["query_features"]
    if (
        not isinstance(features, torch.Tensor)
        or features.ndim != 3
        or features.shape[0] != 1
        or features.shape[1] <= 0
        or features.shape[2] != 128
        or not features.is_floating_point()
        or not torch.isfinite(features).all().item()
    ):
        raise R1RunError("query feature export shape or values differ")
    return {
        "status": "pass",
        "legacy_predictions_unchanged": True,
        "query_feature_shape": list(features.shape),
    }


def new_cache_progress(
    *,
    cache_kind: str,
    provenance: Mapping[str, object],
    protocol_sha256: str,
) -> dict[str, object]:
    if cache_kind not in {"local", "full_history"}:
        raise R1RunError("cache kind is invalid")
    return {
        "schema_version": 1,
        "status": "in_progress",
        "cache_kind": cache_kind,
        "provenance": dict(provenance),
        "protocol_manifest_sha256": protocol_sha256,
        "records": [],
    }


def _validated_progress_records(
    *,
    expected_keys: Sequence[Mapping[str, object]],
    progress: Mapping[str, object],
    cache_kind: str,
    provenance: Mapping[str, object],
    protocol_sha256: str,
) -> tuple[list[Mapping[str, object]], set[str]]:
    expected_binding = new_cache_progress(
        cache_kind=cache_kind,
        provenance=provenance,
        protocol_sha256=protocol_sha256,
    )
    for field in (
        "schema_version",
        "cache_kind",
        "provenance",
        "protocol_manifest_sha256",
    ):
        if progress.get(field) != expected_binding[field]:
            raise R1RunError("cache progress binding differs")
    if progress.get("status") not in {"in_progress", "pass"}:
        raise R1RunError("cache progress status is invalid")
    records = progress.get("records")
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise R1RunError("cache progress records must be a sequence")
    expected_identities = [_key_identity(key) for key in expected_keys]
    if len(set(expected_identities)) != len(expected_identities):
        raise R1RunError("expected cache keys contain duplicate identities")
    record_values: list[Mapping[str, object]] = []
    record_identities: list[str] = []
    for record in records:
        if not isinstance(record, Mapping) or not isinstance(record.get("key"), Mapping):
            raise R1RunError("cache progress record is invalid")
        record_values.append(record)
        record_identities.append(_key_identity(record["key"]))
    if len(set(record_identities)) != len(record_identities):
        raise R1RunError("cache progress contains a duplicate key")
    expected_set = set(expected_identities)
    if not set(record_identities) <= expected_set:
        raise R1RunError("cache progress contains an unexpected key")
    return record_values, set(record_identities)


def resume_pending_keys(
    *,
    expected_keys: Sequence[Mapping[str, object]],
    progress: Mapping[str, object],
    cache_kind: str,
    provenance: Mapping[str, object],
    protocol_sha256: str,
) -> tuple[Mapping[str, object], ...]:
    _records, completed = _validated_progress_records(
        expected_keys=expected_keys,
        progress=progress,
        cache_kind=cache_kind,
        provenance=provenance,
        protocol_sha256=protocol_sha256,
    )
    return tuple(key for key in expected_keys if _key_identity(key) not in completed)


def materialize_records(
    *,
    cache_kind: str,
    expected_keys: Sequence[Mapping[str, object]],
    provenance: Mapping[str, object],
    protocol_sha256: str,
    progress_path: Path,
    produce_record: Callable[[Mapping[str, object]], Mapping[str, object]],
    validate_record: Callable[[Mapping[str, object]], None],
) -> dict[str, object]:
    progress = (
        _read_json(progress_path)
        if progress_path.exists() or progress_path.is_symlink()
        else new_cache_progress(
            cache_kind=cache_kind,
            provenance=provenance,
            protocol_sha256=protocol_sha256,
        )
    )
    records, completed = _validated_progress_records(
        expected_keys=expected_keys,
        progress=progress,
        cache_kind=cache_kind,
        provenance=provenance,
        protocol_sha256=protocol_sha256,
    )
    records_by_key: dict[str, Mapping[str, object]] = {}
    for record in records:
        validate_record(record)
        records_by_key[_key_identity(record["key"])] = record
    for position, key in enumerate(expected_keys, start=1):
        identity = _key_identity(key)
        if identity in completed:
            continue
        record = produce_record(key)
        if not isinstance(record, Mapping) or record.get("key") != key:
            raise R1RunError("cache producer returned a mismatched key")
        validate_record(record)
        records_by_key[identity] = dict(record)
        completed.add(identity)
        progress = {
            **progress,
            "status": "in_progress",
            "records": [
                records_by_key[_key_identity(item)]
                for item in expected_keys
                if _key_identity(item) in records_by_key
            ],
        }
        _atomic_write_json(progress_path, progress)
        if position % 25 == 0 or position == len(expected_keys):
            print(
                f"{cache_kind} cache progress: {len(completed)}/{len(expected_keys)}",
                flush=True,
            )
    if len(completed) != len(expected_keys):
        raise R1RunError("cache materialization coverage is incomplete")
    progress = {
        **progress,
        "status": "pass",
        "records": [records_by_key[_key_identity(key)] for key in expected_keys],
    }
    _atomic_write_json(progress_path, progress)
    return progress


def _cache_summary(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    return {
        "entry_count": len(records),
        "records_sha256": hashlib.sha256(_canonical_bytes(records)).hexdigest(),
    }


def finalize_cache_manifest(
    *,
    local_progress: Mapping[str, object],
    full_progress: Mapping[str, object],
    expected_local_keys: Sequence[Mapping[str, object]],
    expected_full_keys: Sequence[Mapping[str, object]],
    local_provenance: Mapping[str, object],
    full_provenance: Mapping[str, object],
    protocol_sha256: str,
) -> dict[str, object]:
    local_records, local_completed = _validated_progress_records(
        expected_keys=expected_local_keys,
        progress=local_progress,
        cache_kind="local",
        provenance=local_provenance,
        protocol_sha256=protocol_sha256,
    )
    full_records, full_completed = _validated_progress_records(
        expected_keys=expected_full_keys,
        progress=full_progress,
        cache_kind="full_history",
        provenance=full_provenance,
        protocol_sha256=protocol_sha256,
    )
    if (
        local_progress.get("status") != "pass"
        or full_progress.get("status") != "pass"
        or len(local_completed) != 645
        or len(full_completed) != 645
        or len(expected_local_keys) != 645
        or len(expected_full_keys) != 645
    ):
        raise R1RunError("cache coverage is incomplete")
    common = ("source_commit", "checkpoint_sha256", "config_sha256")
    if any(local_provenance.get(key) != full_provenance.get(key) for key in common):
        raise R1RunError("local and FullHistory cache provenance differs")
    return {
        "schema_version": 1,
        "status": "pass",
        "external_cache_reference": "external:r1_downstream_validation_v1/cache",
        "source_commit": local_provenance["source_commit"],
        "checkpoint_sha256": local_provenance["checkpoint_sha256"],
        "config_sha256": local_provenance["config_sha256"],
        "protocol_sha256": protocol_sha256,
        "local": _cache_summary(local_records),
        "full_history": _cache_summary(full_records),
    }


def _build_setup(arguments: argparse.Namespace, *, device_name: str | None):
    from scripts.r1_downstream_context import build_r1_setup

    return build_r1_setup(
        contract_path=arguments.contract,
        protocol_path=arguments.protocol,
        checkpoint_path=arguments.checkpoint,
        pretrained_path=arguments.pretrained,
        metadata_path=arguments.metadata,
        data_root=arguments.data_root,
        source_commit=_git_head(),
        device_name=device_name,
    )


def run_audit(arguments: argparse.Namespace) -> dict[str, object]:
    from omegaconf import OmegaConf

    from scripts.p6a_cache import portable_runtime_config_text
    from scripts.r1_downstream_context import (
        build_algorithm_semantic_identity,
        validate_protocol_b,
    )

    _require_clean_tracked_tree()
    setup = _build_setup(arguments, device_name=None)
    protocol = validate_protocol_b(arguments.protocol, setup.contract)
    checkpoint = setup.contract["checkpoint"]
    pretrained = setup.contract["pretrained"]
    manifest = {
        "schema_version": 1,
        "status": "pass",
        "experiment": setup.contract["experiment"],
        "checkpoint": dict(checkpoint),
        "pretrained": dict(pretrained),
        "protocol": protocol,
        "metadata": {
            "reference": "external:3RScan/3RScan.json",
            "sha256": setup.p6a_config["protocol_b"]["sources"]["metadata_sha256"],
            "bytes": arguments.metadata.stat().st_size,
        },
        "runtime": {
            "seed": 45,
            "precision": "float32",
            "batch_size": 1,
            "device_count": 1,
            "device": "cuda:0",
        },
        "source": {
            "code_commit_at_audit": setup.local_provenance["source_commit"],
            **build_algorithm_semantic_identity(PROJECT_ROOT),
        },
        "checkpoint_sha256_verification": (
            "preverified_once_before_execution_and_bound_by_digest_filename_and_bytes"
        ),
    }
    _publish_exact_json(arguments.artifact_root / "input_manifest.json", manifest)
    runtime_text = OmegaConf.to_yaml(
        setup.runtime_config, resolve=True, sort_keys=True
    )
    portable = portable_runtime_config_text(runtime_text).encode("utf-8")
    runtime_path = arguments.artifact_root / "runtime_config.yaml"
    runtime_path.parent.mkdir(parents=True, exist_ok=True)
    if runtime_path.exists():
        if runtime_path.read_bytes() != portable:
            raise R1RunError("refusing to overwrite different runtime config")
    else:
        runtime_path.write_bytes(portable)
    return manifest


def _release_cuda(*values: object) -> None:
    for value in values:
        del value
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:  # pragma: no cover - runtime always has torch.
        pass


def _run_query_feature_export_smoke(
    producer: object, logical_key: Mapping[str, object]
) -> dict[str, object]:
    import torch

    from scripts.evaluate_persist4d_p6a import (
        _frozen_inference_seed,
        resolve_protocol_cache_request,
    )

    request = resolve_protocol_cache_request(producer.protocol, logical_key)
    system = producer.system
    model = getattr(system, "model", None)
    if model is None or not hasattr(model, "return_query_features"):
        raise R1RunError("R1 model lacks the query feature export switch")

    def materialize() -> tuple[object, object, object]:
        with _frozen_inference_seed(producer.seed, producer.device):
            sample = producer.dataset.load_scan_indices(
                request.context_index,
                request.scan_indices,
                change_file=None,
            )
            data, targets, names = producer.collate([sample])
            if list(names) != [request.master_sequence_id] or len(targets) != 1:
                raise R1RunError("query feature parity sample identity differs")
            data = producer.move_data(data, producer.device)
            targets = producer.move_targets(targets, producer.device)
            raw_coordinates = system._process_raw_coordinates(data)
            return data, targets[0], raw_coordinates

    original = model.return_query_features
    try:
        disabled_data, disabled_target, disabled_raw = materialize()
        enabled_data, enabled_target, enabled_raw = materialize()
        model.return_query_features = False
        with (
            _frozen_inference_seed(producer.seed, producer.device),
            torch.inference_mode(),
        ):
            disabled = system(
                disabled_data,
                point2segment=[disabled_target["point2segment"]],
                raw_coordinates=disabled_raw,
                is_eval=True,
            )
        model.return_query_features = True
        with (
            _frozen_inference_seed(producer.seed, producer.device),
            torch.inference_mode(),
        ):
            enabled = system(
                enabled_data,
                point2segment=[enabled_target["point2segment"]],
                raw_coordinates=enabled_raw,
                is_eval=True,
            )
        return validate_query_feature_export_parity(disabled, enabled)
    finally:
        model.return_query_features = original


def run_smoke(arguments: argparse.Namespace) -> dict[str, object]:
    from scripts.evaluate_persist4d_p6a import (
        build_rio_class_mapper,
        build_tracker_factories,
        cache_payload_to_frozen_observation,
        expected_cache_keys,
    )
    from scripts.rescene_task_postprocess import extract_official_task_prediction
    from scripts.run_system_comparison import _full_producer, _local_producer
    from scripts.system_comparison_inference import (
        assert_t2_observation_regression,
        deterministic_inference_runtime,
        full_history_cache_keys,
        full_history_prediction_fingerprint,
    )
    from scripts.system_comparison_v2_cache import (
        build_task_sidecar,
        observation_fingerprint,
        task_sidecar_digest,
    )
    from scripts.system_comparison_v2_parity import compare_t2_task_predictions
    from scripts.system_comparison_v3_identity import run_fresh_tracker_steps

    _require_clean_tracked_tree()
    validate_cache_execution(arguments.device)
    setup = local_producer = full_producer = None
    try:
        setup = _build_setup(arguments, device_name=arguments.device)
        local_keys = expected_cache_keys(setup.protocol)
        manifest_local_keys = local_cache_keys(setup.protocol_manifest)
        if local_keys != manifest_local_keys:
            raise R1RunError("runtime and manifest local cache keys differ")
        full_keys = full_history_cache_keys(setup.protocol_manifest)
        pairs = select_smoke_pairs(local_keys, full_keys)
        local_producer = _local_producer(setup)
        full_producer = _full_producer(setup)
        class_mapper = build_rio_class_mapper(setup.dataset)
        tracker_factories = build_tracker_factories(setup.p6a_config)
        rows = []
        smoke_observations = []
        with deterministic_inference_runtime(45, setup.device):
            feature_parity = _run_query_feature_export_smoke(
                local_producer, pairs[0][0]
            )
            for pair_index, (local_key, full_key) in enumerate(pairs):
                repeat_count = smoke_repeat_count(pair_index)
                repeats = []
                candidate_rows = []
                for _repeat in range(repeat_count):
                    local = local_producer.produce_bundle(
                        local_key,
                        task_prediction_builder=extract_official_task_prediction,
                        class_mapper=class_mapper,
                    )
                    full = full_producer.produce_bundle(full_key)
                    assert_t2_observation_regression(full.payload, local.payload)
                    sidecar = build_task_sidecar(
                        raw_cache_payload=local.payload,
                        official_prediction=local.task_prediction,
                        protocol_manifest_sha256=setup.full_provenance[
                            "protocol_sha256"
                        ],
                    )
                    candidate = compare_t2_task_predictions(
                        full_payload=full.payload,
                        local_sidecar=sidecar,
                        full_history_content_sha256=str(
                            full.payload["content_sha256"]
                        ),
                        sidecar_content_sha256=task_sidecar_digest(sidecar),
                    )
                    if candidate["parity_pass"] is not True:
                        raise R1RunError("smoke T2 official candidates differ")
                    repeats.append(
                        {
                            "local_observation": observation_fingerprint(local.payload),
                            "local_task_sidecar": task_sidecar_digest(sidecar),
                            "full_prediction": full_history_prediction_fingerprint(
                                full.payload
                            ),
                        }
                    )
                    candidate_rows.append(candidate)
                    if _repeat == 0:
                        smoke_observations.append(
                            cache_payload_to_frozen_observation(local.payload)
                        )
                if any(repeat != repeats[0] for repeat in repeats[1:]):
                    raise R1RunError("smoke prediction fingerprints are not deterministic")
                if any(
                    row != candidate_rows[0] for row in candidate_rows[1:]
                ):
                    raise R1RunError("smoke candidate parity is not deterministic")
                rows.append(
                    {
                        "reference_scene_id": local_key["reference_scene_id"],
                        "master_sequence_id": local_key["master_sequence_id"],
                        "order_id": "canonical",
                        "horizon": 2,
                        "repeat_count": repeat_count,
                        "fingerprints": repeats[0],
                        "t2_observation_parity": "pass",
                        "t2_candidate_parity": "pass",
                        "candidate_count": candidate_rows[0][
                            "candidate_count_local"
                        ],
                        "score_max_abs_diff": candidate_rows[0][
                            "score_max_abs_diff"
                        ],
                    }
                )
        for pair_index, observation in enumerate(smoke_observations):
            for method in ("B2", "B4"):
                run_fresh_tracker_steps(
                    factory=tracker_factories[method],
                    observations=(observation,),
                    sequence_id=f"r1-smoke:{pair_index}:{method}",
                )
        result = {
            "schema_version": 1,
            "status": "pass",
            "source_commit": setup.local_provenance["source_commit"],
            "checkpoint_sha256": setup.local_provenance["checkpoint_sha256"],
            "config_sha256": setup.local_provenance["config_sha256"],
            "protocol_sha256": setup.full_provenance["protocol_sha256"],
            "pair_count": 6,
            "repeated_input_count": 2,
            "total_pair_forward_count": sum(
                int(row["repeat_count"]) for row in rows
            ),
            "query_feature_export_parity": feature_parity,
            "fresh_state_gt_isolation": {
                "status": "pass",
                "method_count": 2,
                "selected_unit_count": 6,
                "tracker_input": "frozen_query_observation_without_gt",
                "diagnostic_timing": "after_tracker_output_only",
            },
            "pairs": rows,
        }
        _publish_exact_json(
            arguments.artifact_root / "smoke_and_parity.json", result
        )
        return result
    finally:
        _release_cuda(local_producer, full_producer, setup)


def run_cache_parity(arguments: argparse.Namespace) -> dict[str, object]:
    from scripts.system_comparison_inference import load_full_history_cache_entry
    from scripts.system_comparison_v2_cache import load_task_sidecar
    from scripts.system_comparison_v2_parity import (
        compare_t2_task_predictions,
        summarize_t2_rows,
    )

    _require_clean_tracked_tree()
    smoke_path = arguments.artifact_root / "smoke_and_parity.json"
    smoke = _read_json(smoke_path)
    if smoke.get("status") != "pass" or smoke.get("pair_count") != 6:
        raise R1RunError("six-cluster smoke must pass before cache parity")
    local_progress = _read_json(arguments.cache_root / "local_progress.json")
    full_progress = _read_json(
        arguments.cache_root / "full_history_progress.json"
    )
    if (
        local_progress.get("status") != "pass"
        or full_progress.get("status") != "pass"
        or len(local_progress.get("records", ())) != 645
        or len(full_progress.get("records", ())) != 645
    ):
        raise R1RunError("complete local and FullHistory caches are required")
    local_records = {
        (
            record["key"]["master_sequence_id"],
            record["key"]["reference_scene_id"],
            record["key"]["order_id"],
        ): record
        for record in local_progress["records"]
        if record["key"]["stage_index"] == 1
    }
    full_records = {
        (
            record["key"]["master_sequence_id"],
            record["key"]["reference_scene_id"],
            record["key"]["order_id"],
        ): record
        for record in full_progress["records"]
        if record["key"]["horizon"] == 2
    }
    if (
        len(local_records) != 129
        or set(local_records) != set(full_records)
    ):
        raise R1RunError("T2 cache parity coverage differs")
    rows = []
    for identity in sorted(local_records):
        local_record = local_records[identity]
        full_record = full_records[identity]
        sidecar_entry = local_record["sidecar_entry"]
        sidecar = load_task_sidecar(
            arguments.cache_root
            / "task_sidecars/entries"
            / str(sidecar_entry["filename"])
        )
        full = load_full_history_cache_entry(
            arguments.cache_root / "full_history/entries",
            full_record,
            expected_provenance=full_progress["provenance"],
        )
        rows.append(
            compare_t2_task_predictions(
                full_payload=full,
                local_sidecar=sidecar,
                full_history_content_sha256=str(full_record["content_sha256"]),
                sidecar_content_sha256=str(sidecar_entry["content_sha256"]),
            )
        )
    summary = summarize_t2_rows(rows, expected_unit_count=129)
    if summary["status"] != "pass":
        raise R1RunError("T2 cache candidate parity failed")
    result = {
        **smoke,
        "cache_parity_source_commit": _git_head(),
        "t2_cache_parity": {
            **summary,
            "rows_sha256": hashlib.sha256(_canonical_bytes(rows)).hexdigest(),
            "rows": rows,
        },
        "local_current_invariance": {
            "status": "pass",
            "shared_raw_entry_count": 645,
            "shared_sidecar_entry_count": 645,
            "tracker_methods": ["B2", "B4"],
        },
    }
    _atomic_write_json(smoke_path, result)
    return {
        "status": "pass",
        "unit_count": summary["unit_count"],
        "pass_count": summary["pass_count"],
        "rows_sha256": result["t2_cache_parity"]["rows_sha256"],
    }


def _validate_local_record(
    record: Mapping[str, object],
    *,
    raw_directory: Path,
    sidecar_directory: Path,
    provenance: Mapping[str, object],
) -> None:
    from scripts.p6a_cache import validate_cache_entry
    from scripts.system_comparison_v2_cache import (
        load_task_sidecar,
        observation_fingerprint,
        task_sidecar_digest,
    )

    if set(record) != {
        "key",
        "raw_entry",
        "sidecar_entry",
        "raw_observation_fingerprint",
    }:
        raise R1RunError("local cache record fields differ")
    raw_entry = record["raw_entry"]
    sidecar_entry = record["sidecar_entry"]
    if not isinstance(raw_entry, Mapping) or not isinstance(sidecar_entry, Mapping):
        raise R1RunError("local cache entry metadata differs")
    if raw_entry.get("key") != record["key"] or sidecar_entry.get("key") != record["key"]:
        raise R1RunError("local cache pair keys differ")
    raw = validate_cache_entry(
        raw_directory / str(raw_entry["filename"]),
        raw_entry,
        expected_provenance=provenance,
    )
    sidecar_path = sidecar_directory / str(sidecar_entry["filename"])
    sidecar = load_task_sidecar(sidecar_path)
    if (
        sidecar_path.stat().st_size != sidecar_entry.get("file_bytes")
        or _sha256_file(sidecar_path) != sidecar_entry.get("file_sha256")
        or task_sidecar_digest(sidecar) != sidecar_entry.get("content_sha256")
    ):
        raise R1RunError("local task sidecar file evidence differs")
    fingerprint = observation_fingerprint(raw)
    if (
        fingerprint != record["raw_observation_fingerprint"]
        or sidecar["provenance"]["source_raw_observation_fingerprint"] != fingerprint
    ):
        raise R1RunError("local raw/sidecar same-forward binding differs")


def run_local_cache(arguments: argparse.Namespace) -> dict[str, object]:
    from scripts.evaluate_persist4d_p6a import (
        build_rio_class_mapper,
        expected_cache_keys,
    )
    from scripts.p6a_cache import write_cache_entry
    from scripts.rescene_task_postprocess import extract_official_task_prediction
    from scripts.run_system_comparison import _local_producer
    from scripts.system_comparison_inference import deterministic_inference_runtime
    from scripts.system_comparison_v2_cache import (
        build_task_sidecar,
        observation_fingerprint,
        write_task_sidecar,
    )

    _require_clean_tracked_tree()
    validate_cache_execution(arguments.device)
    setup = producer = None
    raw_directory = arguments.cache_root / "raw_predictions/entries"
    sidecar_directory = arguments.cache_root / "task_sidecars/entries"
    try:
        setup = _build_setup(arguments, device_name=arguments.device)
        keys = expected_cache_keys(setup.protocol)
        if keys != local_cache_keys(setup.protocol_manifest):
            raise R1RunError("runtime and manifest local cache keys differ")
        producer = _local_producer(setup)
        class_mapper = build_rio_class_mapper(setup.dataset)

        def produce(key: Mapping[str, object]) -> Mapping[str, object]:
            produced = producer.produce_bundle(
                key,
                task_prediction_builder=extract_official_task_prediction,
                class_mapper=class_mapper,
            )
            sidecar = write_task_sidecar(
                sidecar_directory,
                build_task_sidecar(
                    raw_cache_payload=produced.payload,
                    official_prediction=produced.task_prediction,
                    protocol_manifest_sha256=setup.full_provenance[
                        "protocol_sha256"
                    ],
                ),
            )
            raw = write_cache_entry(raw_directory, produced.payload)
            return {
                "key": dict(key),
                "raw_entry": raw,
                "sidecar_entry": sidecar,
                "raw_observation_fingerprint": observation_fingerprint(
                    produced.payload
                ),
            }

        def validate(record: Mapping[str, object]) -> None:
            _validate_local_record(
                record,
                raw_directory=raw_directory,
                sidecar_directory=sidecar_directory,
                provenance=setup.local_provenance,
            )

        with deterministic_inference_runtime(45, setup.device):
            progress = materialize_records(
                cache_kind="local",
                expected_keys=keys,
                provenance=setup.local_provenance,
                protocol_sha256=setup.full_provenance["protocol_sha256"],
                progress_path=arguments.cache_root / "local_progress.json",
                produce_record=produce,
                validate_record=validate,
            )
        return {
            "status": progress["status"],
            "entry_count": len(progress["records"]),
        }
    finally:
        _release_cuda(producer, setup)


def run_full_cache(arguments: argparse.Namespace) -> dict[str, object]:
    from scripts.run_system_comparison import _full_producer
    from scripts.system_comparison_inference import (
        deterministic_inference_runtime,
        full_history_cache_keys,
        load_full_history_cache_entry,
        write_full_history_cache_entry,
    )

    _require_clean_tracked_tree()
    validate_cache_execution(arguments.device)
    setup = producer = None
    directory = arguments.cache_root / "full_history/entries"
    try:
        setup = _build_setup(arguments, device_name=arguments.device)
        keys = full_history_cache_keys(setup.protocol_manifest)
        producer = _full_producer(setup)

        def produce(key: Mapping[str, object]) -> Mapping[str, object]:
            return write_full_history_cache_entry(
                directory, producer.produce_bundle(key).payload
            )

        def validate(record: Mapping[str, object]) -> None:
            load_full_history_cache_entry(
                directory,
                record,
                expected_provenance=setup.full_provenance,
            )

        with deterministic_inference_runtime(45, setup.device):
            progress = materialize_records(
                cache_kind="full_history",
                expected_keys=keys,
                provenance=setup.full_provenance,
                protocol_sha256=setup.full_provenance["protocol_sha256"],
                progress_path=arguments.cache_root / "full_history_progress.json",
                produce_record=produce,
                validate_record=validate,
            )
        return {
            "status": progress["status"],
            "entry_count": len(progress["records"]),
        }
    finally:
        _release_cuda(producer, setup)


def run_finalize_cache(arguments: argparse.Namespace) -> dict[str, object]:
    from scripts.evaluate_persist4d_p6a import expected_cache_keys
    from scripts.p6a_cache import build_cache_manifest
    from scripts.system_comparison_inference import (
        build_full_history_cache_manifest,
        full_history_cache_keys,
    )

    _require_clean_tracked_tree()
    setup = _build_setup(arguments, device_name=None)
    local_progress = _read_json(arguments.cache_root / "local_progress.json")
    full_progress = _read_json(
        arguments.cache_root / "full_history_progress.json"
    )
    local_keys = expected_cache_keys(setup.protocol)
    full_keys = full_history_cache_keys(setup.protocol_manifest)
    for record in local_progress.get("records", []):
        _validate_local_record(
            record,
            raw_directory=arguments.cache_root / "raw_predictions/entries",
            sidecar_directory=arguments.cache_root / "task_sidecars/entries",
            provenance=setup.local_provenance,
        )
    raw_manifest = build_cache_manifest(
        [record["raw_entry"] for record in local_progress["records"]],
        expected_keys=local_keys,
        expected_provenance=setup.local_provenance,
        cache_directory=arguments.cache_root / "raw_predictions/entries",
    )
    full_manifest = build_full_history_cache_manifest(
        full_progress["records"],
        expected_keys=full_keys,
        expected_provenance=setup.full_provenance,
        cache_directory=arguments.cache_root / "full_history/entries",
    )
    result = finalize_cache_manifest(
        local_progress=local_progress,
        full_progress=full_progress,
        expected_local_keys=local_keys,
        expected_full_keys=full_keys,
        local_provenance=setup.local_provenance,
        full_provenance=setup.full_provenance,
        protocol_sha256=setup.full_provenance["protocol_sha256"],
    )
    result["local"]["raw_entries_sha256"] = raw_manifest["entries_sha256"]
    result["full_history"]["entries_sha256"] = full_manifest["entries_sha256"]
    result["full_history"]["content_sha256"] = full_manifest["content_sha256"]
    _atomic_write_json(arguments.cache_root / "cache_manifest.json", result)
    _publish_exact_json(arguments.artifact_root / "cache_manifest.json", result)
    return result


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "audit",
            "smoke",
            "cache-local",
            "cache-full",
            "finalize-cache",
            "cache-parity",
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--pretrained", type=Path, default=DEFAULT_PRETRAINED)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = argument_parser().parse_args(argv)
    actions = {
        "audit": run_audit,
        "smoke": run_smoke,
        "cache-local": run_local_cache,
        "cache-full": run_full_cache,
        "finalize-cache": run_finalize_cache,
        "cache-parity": run_cache_parity,
    }
    result = actions[arguments.stage](arguments)
    print(json.dumps(result, allow_nan=False, sort_keys=True), flush=True)
    return 0


__all__ = [
    "R1RunError",
    "argument_parser",
    "finalize_cache_manifest",
    "local_cache_keys",
    "materialize_records",
    "new_cache_progress",
    "resume_pending_keys",
    "select_smoke_pairs",
    "smoke_repeat_count",
    "validate_cache_execution",
    "validate_query_feature_export_parity",
]


if __name__ == "__main__":
    raise SystemExit(main())
