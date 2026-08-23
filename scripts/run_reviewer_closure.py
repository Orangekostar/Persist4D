"""Gate-driven orchestration for Persist4D reviewer-closure experiments."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Callable

from scripts.reviewer_closure_protocol import (
    build_reviewer_closure_manifest,
    full_history_observation_keys,
    validate_reviewer_closure_binding,
    validate_reviewer_closure_manifest,
)
from scripts.reviewer_closure_sidecar import FullHistoryObservationSidecarError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs/reviewer_closure/protocol.yaml"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints/rescene4d_concerto_t2_repro.ckpt"
SYSTEM_MANIFEST_PATH = (
    PROJECT_ROOT / "artifacts/system_comparison/system_comparison_manifest.json"
)
REVIEWER_ROOT = PROJECT_ROOT / "artifacts/reviewer_closure"
REVIEWER_MANIFEST_PATH = REVIEWER_ROOT / "reviewer_closure_manifest.json"
SIDECAR_ROOT = REVIEWER_ROOT / "full_history_observations_v2"
SIDECAR_ENTRY_ROOT = SIDECAR_ROOT / "entries"
SIDECAR_MANIFEST_PATH = SIDECAR_ROOT / "manifest.json"
REPLAY_ROOT = REVIEWER_ROOT / "full_history_replay_v2"
REPLAY_ENTRY_ROOT = REPLAY_ROOT / "entries"
REPLAY_STAGING_ROOT = REPLAY_ROOT / "staging"
REPLAY_MANIFEST_PATH = REPLAY_ROOT / "manifest.json"
SOURCE_PREDICTION_MANIFEST_PATH = (
    PROJECT_ROOT / "artifacts/system_comparison/full_history_predictions/manifest.json"
)
REPRODUCIBILITY_BINDING_PATH = (
    PROJECT_ROOT / "artifacts/system_comparison/reproducibility_binding.json"
)
DEFAULT_SOURCE_ENTRY_ROOT = (
    PROJECT_ROOT / "artifacts/system_comparison/full_history_predictions/entries"
)
DEFAULT_METADATA_PATH = Path(os.environ.get("PERSIST4D_3RSCAN_METADATA", "3RScan.json"))


class ReviewerClosureGateFailure(RuntimeError):
    """Raised when a reviewer-closure execution gate does not pass."""


def _canonical_json_bytes(value: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def publish_exact_json(path: str | Path, value: Mapping[str, object]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_json_bytes(value)
    if output.exists() or output.is_symlink():
        if output.is_symlink() or not output.is_file():
            raise FileExistsError(f"output is not a regular file: {output}")
        if output.read_bytes() == payload:
            return
        raise FileExistsError(f"output already contains different content: {output}")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.link(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def build_bound_reviewer_manifest() -> dict[str, object]:
    binding = validate_reviewer_closure_binding(
        CONFIG_PATH,
        repo_root=PROJECT_ROOT,
        checkpoint_path=CHECKPOINT_PATH,
    )
    manifest = build_reviewer_closure_manifest(
        CONFIG_PATH,
        system_manifest_path=SYSTEM_MANIFEST_PATH,
        binding=binding,
    )
    validate_reviewer_closure_manifest(
        manifest,
        config_path=CONFIG_PATH,
        system_manifest_path=SYSTEM_MANIFEST_PATH,
    )
    return manifest


def run_bind(*, output_path: str | Path = REVIEWER_MANIFEST_PATH) -> dict[str, object]:
    manifest = build_bound_reviewer_manifest()
    publish_exact_json(output_path, manifest)
    return manifest


def _load_json(path: str | Path, *, name: str) -> dict[str, object]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ReviewerClosureGateFailure(f"{name} is not a regular file")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ReviewerClosureGateFailure(f"{name} cannot be decoded") from error
    if not isinstance(value, Mapping):
        raise ReviewerClosureGateFailure(f"{name} must be a mapping")
    return dict(value)


def _git_head() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            check=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ReviewerClosureGateFailure(
            "cannot resolve sidecar source commit"
        ) from error
    commit = completed.stdout.strip()
    if len(commit) != 40:
        raise ReviewerClosureGateFailure("sidecar source commit is invalid")
    return commit


def _require_source_tree_clean() -> None:
    completed = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=True,
        text=True,
    )
    disallowed: list[str] = []
    for line in completed.stdout.splitlines():
        path = line[3:]
        if " -> " in path:
            path = path.split(" -> ", maxsplit=1)[1]
        if not path.startswith("artifacts/reviewer_closure/"):
            disallowed.append(path)
    if disallowed:
        raise ReviewerClosureGateFailure(
            "tracked sidecar source must be committed before inference: "
            + ", ".join(sorted(disallowed))
        )


def _load_bound_reviewer_manifest(path: str | Path) -> dict[str, object]:
    observed = _load_json(path, name="reviewer-closure manifest")
    expected = build_bound_reviewer_manifest()
    if observed != expected:
        raise ReviewerClosureGateFailure(
            "reviewer-closure manifest differs from frozen inputs"
        )
    validate_reviewer_closure_manifest(
        observed,
        config_path=CONFIG_PATH,
        system_manifest_path=SYSTEM_MANIFEST_PATH,
    )
    return observed


def validate_sidecar_execution(device_name: str) -> None:
    if device_name != "cuda:0":
        raise ReviewerClosureGateFailure(
            "sidecar inference is preregistered as one deterministic cuda:0 process"
        )


def _key_identity(value: Mapping[str, object]) -> str:
    return json.dumps(
        dict(value), ensure_ascii=True, separators=(",", ":"), sort_keys=True
    )


def select_pending_sidecar_keys(
    expected_keys: Sequence[Mapping[str, object]],
    existing_entries: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    if (
        isinstance(expected_keys, (str, bytes))
        or not isinstance(expected_keys, Sequence)
        or not expected_keys
        or isinstance(existing_entries, (str, bytes))
        or not isinstance(existing_entries, Sequence)
    ):
        raise ReviewerClosureGateFailure("sidecar resume inputs are invalid")
    expected = [dict(key) for key in expected_keys]
    expected_identities = [_key_identity(key) for key in expected]
    if len(set(expected_identities)) != len(expected_identities):
        raise ReviewerClosureGateFailure("expected sidecar keys contain duplicates")
    existing_identities: list[str] = []
    for record in existing_entries:
        if not isinstance(record, Mapping) or not isinstance(
            record.get("key"), Mapping
        ):
            raise ReviewerClosureGateFailure("existing sidecar entry is invalid")
        existing_identities.append(_key_identity(record["key"]))
    if len(set(existing_identities)) != len(existing_identities) or not set(
        existing_identities
    ) <= set(expected_identities):
        raise ReviewerClosureGateFailure(
            "existing sidecar keys are not an exact expected-prefix subset"
        )
    existing_set = set(existing_identities)
    return [
        key
        for key, identity in zip(expected, expected_identities, strict=True)
        if identity not in existing_set
    ]


def select_sidecar_smoke_key(
    expected_keys: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    for raw_key in expected_keys:
        key = dict(raw_key)
        if key.get("order_id") == "canonical" and key.get("horizon") == 2:
            return key
    raise ReviewerClosureGateFailure("no canonical T2 sidecar smoke key is available")


def validate_sidecar_resume_commit(
    existing_entries: Sequence[Mapping[str, object]],
    source_commit: str,
) -> None:
    if len(source_commit) != 40:
        raise ReviewerClosureGateFailure("sidecar source commit is invalid")
    for record in existing_entries:
        if record.get("sidecar_source_commit") != source_commit:
            raise ReviewerClosureGateFailure(
                "existing sidecars contain a mixed or different code commit"
            )


def validate_completed_replay_pairs(
    sidecar_entries: Sequence[Mapping[str, object]],
    replay_entries: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    from scripts.reviewer_closure_sidecar import sidecar_key_for_source_prediction

    if (
        isinstance(sidecar_entries, (str, bytes))
        or not isinstance(sidecar_entries, Sequence)
        or isinstance(replay_entries, (str, bytes))
        or not isinstance(replay_entries, Sequence)
    ):
        raise ReviewerClosureGateFailure("completed replay pair inputs are invalid")
    sidecars: dict[str, dict[str, object]] = {}
    for raw_record in sidecar_entries:
        record = dict(raw_record)
        key = record.get("key")
        digest = record.get("source_prediction_content_sha256")
        if not isinstance(key, Mapping) or not isinstance(digest, str):
            raise ReviewerClosureGateFailure("completed sidecar pair is invalid")
        identity = _key_identity(key)
        if identity in sidecars:
            raise ReviewerClosureGateFailure("completed sidecar pair coverage differs")
        sidecars[identity] = record
    replays: dict[str, dict[str, object]] = {}
    for raw_record in replay_entries:
        record = dict(raw_record)
        key = record.get("key")
        digest = record.get("content_sha256")
        if not isinstance(key, Mapping) or not isinstance(digest, str):
            raise ReviewerClosureGateFailure("completed replay pair is invalid")
        try:
            identity = _key_identity(sidecar_key_for_source_prediction(key))
        except FullHistoryObservationSidecarError as error:
            raise ReviewerClosureGateFailure(
                "completed replay pair key is invalid"
            ) from error
        if identity in replays:
            raise ReviewerClosureGateFailure("completed replay pair coverage differs")
        replays[identity] = record
    if set(sidecars) != set(replays):
        raise ReviewerClosureGateFailure("completed replay pair coverage differs")
    for identity, sidecar in sidecars.items():
        if (
            sidecar["source_prediction_content_sha256"]
            != replays[identity]["content_sha256"]
        ):
            raise ReviewerClosureGateFailure("completed replay pair content differs")
    return [dict(record) for record in sidecar_entries]


def produce_sidecar_batch(
    *,
    expected_keys: Sequence[Mapping[str, object]],
    existing_entries: Sequence[Mapping[str, object]],
    source_lookup: Callable[[Mapping[str, object]], Mapping[str, object]],
    source_loader: Callable[[Mapping[str, object]], Mapping[str, object]],
    producer: object,
    sidecar_builder: Callable[..., Mapping[str, object]],
    pair_writer: Callable[[Mapping[str, object]], Mapping[str, object]],
    sidecar_source_commit: str,
    smoke_only: bool,
) -> dict[str, int]:
    pending = select_pending_sidecar_keys(expected_keys, existing_entries)
    if smoke_only:
        smoke = select_sidecar_smoke_key(expected_keys)
        smoke_identity = _key_identity(smoke)
        selected = [key for key in pending if _key_identity(key) == smoke_identity]
        expected_count = 1
    else:
        selected = pending
        expected_count = len(expected_keys)
    for key in selected:
        source_record = source_lookup(key)
        source_prediction = source_loader(source_record)
        pair = sidecar_builder(
            producer=producer,
            source_prediction=source_prediction,
            sidecar_key=key,
            sidecar_source_commit=sidecar_source_commit,
        )
        pair_writer(pair)
    return {
        "expected_count": expected_count,
        "reused_count": expected_count - len(selected),
        "produced_count": len(selected),
    }


def publish_full_history_replay_pair(
    pair: Mapping[str, object],
    *,
    staging_root: str | Path,
    replay_writer: Callable[[Mapping[str, object]], Mapping[str, object]],
    sidecar_writer: Callable[[Mapping[str, object]], Mapping[str, object]],
) -> dict[str, object]:
    from scripts.reviewer_closure_sidecar import (
        remove_full_history_replay_pair_stage,
        write_full_history_replay_pair_stage,
    )

    stage = write_full_history_replay_pair_stage(staging_root, pair)
    replay_record = dict(replay_writer(pair["replay_prediction"]))
    sidecar_record = dict(sidecar_writer(pair["sidecar"]))
    if replay_record.get("content_sha256") != sidecar_record.get(
        "source_prediction_content_sha256"
    ) or stage["key"] != sidecar_record.get("key"):
        raise ReviewerClosureGateFailure("published replay pair content differs")
    remove_full_history_replay_pair_stage(staging_root, stage)
    return sidecar_record


def recover_full_history_replay_pairs(
    *,
    staging_root: str | Path,
    replay_writer: Callable[[Mapping[str, object]], Mapping[str, object]],
    sidecar_writer: Callable[[Mapping[str, object]], Mapping[str, object]],
    sidecar_source_commit: str,
) -> int:
    from scripts.reviewer_closure_sidecar import (
        discover_full_history_replay_pair_stages,
        load_full_history_replay_pair_stage,
    )

    recovered = 0
    for record in discover_full_history_replay_pair_stages(staging_root):
        if record["sidecar_source_commit"] != sidecar_source_commit:
            raise ReviewerClosureGateFailure(
                "staged replay pair uses a different code commit"
            )
        pair = load_full_history_replay_pair_stage(staging_root, record)
        publish_full_history_replay_pair(
            pair,
            staging_root=staging_root,
            replay_writer=replay_writer,
            sidecar_writer=sidecar_writer,
        )
        recovered += 1
    return recovered


def run_sidecar_cache(
    *,
    device_name: str,
    metadata_path: str | Path,
    source_entry_root: str | Path,
    sidecar_entry_root: str | Path = SIDECAR_ENTRY_ROOT,
    replay_entry_root: str | Path = REPLAY_ENTRY_ROOT,
    replay_staging_root: str | Path = REPLAY_STAGING_ROOT,
    reviewer_manifest_path: str | Path = REVIEWER_MANIFEST_PATH,
    smoke_only: bool,
) -> dict[str, object]:
    validate_sidecar_execution(device_name)
    metadata = Path(metadata_path)
    source_root = Path(source_entry_root)
    if metadata.is_symlink() or not metadata.is_file():
        raise ReviewerClosureGateFailure("metadata is not a regular file")
    if source_root.is_symlink() or not source_root.is_dir():
        raise ReviewerClosureGateFailure("source cache is not a regular directory")

    reviewer_manifest = _load_bound_reviewer_manifest(reviewer_manifest_path)
    system_manifest = _load_json(SYSTEM_MANIFEST_PATH, name="system manifest")
    source_manifest = _load_json(
        SOURCE_PREDICTION_MANIFEST_PATH, name="source prediction manifest"
    )
    from scripts.reviewer_closure_sidecar import (
        discover_full_history_observation_sidecar_entries,
        produce_bound_full_history_observation_sidecar,
        source_prediction_entry_for_key,
        validate_source_prediction_manifest,
        write_full_history_observation_sidecar_entry,
    )
    from scripts.system_comparison_inference import (
        discover_full_history_cache_entries,
        write_full_history_cache_entry,
    )

    source_manifest = validate_source_prediction_manifest(
        source_manifest, system_manifest=system_manifest
    )
    expected_keys = full_history_observation_keys(reviewer_manifest)
    source_commit = _git_head()

    def write_replay(payload: Mapping[str, object]) -> Mapping[str, object]:
        return write_full_history_cache_entry(replay_entry_root, payload)

    def write_sidecar(payload: Mapping[str, object]) -> Mapping[str, object]:
        return write_full_history_observation_sidecar_entry(sidecar_entry_root, payload)

    recovered_count = recover_full_history_replay_pairs(
        staging_root=replay_staging_root,
        replay_writer=write_replay,
        sidecar_writer=write_sidecar,
        sidecar_source_commit=source_commit,
    )
    existing = discover_full_history_observation_sidecar_entries(sidecar_entry_root)
    replay_entries = discover_full_history_cache_entries(
        replay_entry_root,
        expected_provenance=source_manifest["provenance"],
    )
    validate_completed_replay_pairs(existing, replay_entries)
    validate_sidecar_resume_commit(existing, source_commit)
    _require_source_tree_clean()
    pending = select_pending_sidecar_keys(expected_keys, existing)
    if smoke_only:
        smoke_identity = _key_identity(select_sidecar_smoke_key(expected_keys))
        pending = [key for key in pending if _key_identity(key) == smoke_identity]
    if not pending:
        return {
            "status": "pass",
            "mode": "smoke" if smoke_only else "full",
            "expected_count": 1 if smoke_only else len(expected_keys),
            "reused_count": 1 if smoke_only else len(expected_keys),
            "produced_count": 0,
            "recovered_count": recovered_count,
            "sidecar_source_commit": source_commit,
        }

    binding = _load_json(
        REPRODUCIBILITY_BINDING_PATH, name="system reproducibility binding"
    )
    from scripts.run_system_comparison import _build_frozen_setup, _full_producer
    from scripts.system_comparison_inference import (
        deterministic_inference_runtime,
        load_full_history_cache_entry,
    )

    setup = _build_frozen_setup(
        binding=binding,
        metadata_path=metadata,
        device_name=device_name,
    )
    producer = _full_producer(setup)

    def lookup(key: Mapping[str, object]) -> Mapping[str, object]:
        return source_prediction_entry_for_key(source_manifest, key)

    def load(record: Mapping[str, object]) -> Mapping[str, object]:
        return load_full_history_cache_entry(
            source_root,
            record,
            expected_provenance=source_manifest["provenance"],
        )

    def write_pair(pair: Mapping[str, object]) -> Mapping[str, object]:
        return publish_full_history_replay_pair(
            pair,
            staging_root=replay_staging_root,
            replay_writer=write_replay,
            sidecar_writer=write_sidecar,
        )

    with deterministic_inference_runtime(45, setup.device):
        result = produce_sidecar_batch(
            expected_keys=expected_keys,
            existing_entries=existing,
            source_lookup=lookup,
            source_loader=load,
            producer=producer,
            sidecar_builder=produce_bound_full_history_observation_sidecar,
            pair_writer=write_pair,
            sidecar_source_commit=source_commit,
            smoke_only=smoke_only,
        )
    return {
        "status": "pass",
        "mode": "smoke" if smoke_only else "full",
        **result,
        "recovered_count": recovered_count,
        "sidecar_source_commit": source_commit,
    }


def run_finalize_sidecars(
    *,
    sidecar_entry_root: str | Path = SIDECAR_ENTRY_ROOT,
    output_path: str | Path = SIDECAR_MANIFEST_PATH,
    replay_entry_root: str | Path = REPLAY_ENTRY_ROOT,
    replay_output_path: str | Path = REPLAY_MANIFEST_PATH,
    reviewer_manifest_path: str | Path = REVIEWER_MANIFEST_PATH,
) -> dict[str, object]:
    from scripts.reviewer_closure_sidecar import (
        build_full_history_observation_sidecar_manifest,
        discover_full_history_observation_sidecar_entries,
        source_prediction_entry_for_key,
        validate_source_prediction_manifest,
    )
    from scripts.system_comparison_inference import (
        FullHistoryCacheError,
        build_full_history_cache_manifest,
        discover_full_history_cache_entries,
    )

    reviewer_manifest = _load_bound_reviewer_manifest(reviewer_manifest_path)
    system_manifest = _load_json(SYSTEM_MANIFEST_PATH, name="system manifest")
    source_manifest = validate_source_prediction_manifest(
        _load_json(SOURCE_PREDICTION_MANIFEST_PATH, name="source prediction manifest"),
        system_manifest=system_manifest,
    )
    entries = discover_full_history_observation_sidecar_entries(sidecar_entry_root)
    expected_keys = full_history_observation_keys(reviewer_manifest)
    replay_entries = discover_full_history_cache_entries(
        replay_entry_root,
        expected_provenance=source_manifest["provenance"],
    )
    try:
        replay_manifest = build_full_history_cache_manifest(
            replay_entries,
            expected_keys=[
                source_prediction_entry_for_key(source_manifest, key)["key"]
                for key in expected_keys
            ],
            expected_provenance=source_manifest["provenance"],
            cache_directory=replay_entry_root,
        )
    except FullHistoryCacheError as error:
        raise FullHistoryObservationSidecarError(
            "replay prediction manifest does not have exact coverage"
        ) from error
    manifest = build_full_history_observation_sidecar_manifest(
        entries,
        expected_keys=expected_keys,
        source_prediction_manifest=source_manifest,
        replay_prediction_manifest=replay_manifest,
        system_manifest=system_manifest,
        reviewer_manifest=reviewer_manifest,
        sidecar_code_commit=_git_head(),
        cache_directory=sidecar_entry_root,
    )
    _require_source_tree_clean()
    publish_exact_json(replay_output_path, replay_manifest)
    publish_exact_json(output_path, manifest)
    return {
        "status": "pass",
        "replay_prediction_manifest": replay_manifest,
        "sidecar_manifest": manifest,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("bind")
    for name in ("sidecar-smoke", "cache-sidecars"):
        command = subparsers.add_parser(name)
        command.add_argument("--device", default="cuda:0")
        command.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_PATH)
        command.add_argument(
            "--source-cache-root", type=Path, default=DEFAULT_SOURCE_ENTRY_ROOT
        )
    subparsers.add_parser("finalize-sidecars")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "bind":
        result = run_bind()
    elif arguments.command in {"sidecar-smoke", "cache-sidecars"}:
        result = run_sidecar_cache(
            device_name=arguments.device,
            metadata_path=arguments.metadata,
            source_entry_root=arguments.source_cache_root,
            smoke_only=arguments.command == "sidecar-smoke",
        )
    elif arguments.command == "finalize-sidecars":
        result = run_finalize_sidecars()
    else:
        raise ReviewerClosureGateFailure(f"unsupported command: {arguments.command}")
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0


__all__ = [
    "ReviewerClosureGateFailure",
    "FullHistoryObservationSidecarError",
    "build_bound_reviewer_manifest",
    "build_parser",
    "publish_exact_json",
    "produce_sidecar_batch",
    "publish_full_history_replay_pair",
    "recover_full_history_replay_pairs",
    "run_bind",
    "run_finalize_sidecars",
    "run_sidecar_cache",
    "select_pending_sidecar_keys",
    "select_sidecar_smoke_key",
    "validate_sidecar_execution",
    "validate_sidecar_resume_commit",
    "validate_completed_replay_pairs",
]


if __name__ == "__main__":
    raise SystemExit(main())
