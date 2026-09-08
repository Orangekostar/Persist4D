#!/usr/bin/env python3
"""Fail-closed contracts for Persist4D All-T Task Superiority V1."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

R1_CHECKPOINT_SHA256 = (
    "629ff7624dcac15e6022906e808e2e05b3ec61c60a1116ab0e278f0cfd2368dd"
)
R1_CHECKPOINT_BYTES = 754_813_672
PROTOCOL_B_SHA256 = (
    "246497165612699b103d0d79d5503025cb2cd14466aad3ab149d4fe82884ecbe"
)
PROTOCOL_B_BYTES = 468_179
CONCERTO_SHA256 = (
    "845ec7dec97a5fabff8fadb5d9858ac6734347b612d1a4b574213419c139de07"
)
CONCERTO_BYTES = 433_987_358
RIO_METADATA_SHA256 = (
    "674a00f50f76b198b9de44efd86c390fea3da37ba8f12cf8ccd00045e265fa64"
)
RIO_METADATA_BYTES = 3_155_995
BASELINE_COMMIT = "2c7494b982eff84886aef3a7274bae43752a1485"


class AllTContractError(RuntimeError):
    """Raised when an All-T experiment contract cannot be proven."""


def canonical_json_sha256(value: object) -> str:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise AllTContractError(
            "contract values must be finite portable JSON"
        ) from error
    return hashlib.sha256(encoded).hexdigest()


def require_file_identity(
    path: str | Path,
    *,
    expected_bytes: int,
    expected_sha256: str,
    label: str,
) -> dict[str, object]:
    candidate = Path(path).expanduser()
    if candidate.is_symlink() or not candidate.is_file():
        raise AllTContractError(f"{label} must be a regular non-symlink file")
    if candidate.stat().st_size != expected_bytes:
        raise AllTContractError(f"{label} byte size differs")
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    observed = digest.hexdigest()
    if observed != expected_sha256:
        raise AllTContractError(f"{label} SHA256 differs")
    return {"bytes": expected_bytes, "sha256": observed, "status": "pass"}


def build_reference_split(
    reference_ids: Sequence[str], *, development_count: int, seed: int = 45
) -> dict[str, object]:
    if isinstance(reference_ids, (str, bytes)):
        raise AllTContractError("reference_ids must be a sequence")
    normalized = list(reference_ids)
    if not normalized or any(
        not isinstance(reference, str) or not reference for reference in normalized
    ):
        raise AllTContractError("reference IDs must be non-empty strings")
    if len(set(normalized)) != len(normalized):
        raise AllTContractError("reference IDs must be unique")
    if (
        isinstance(development_count, bool)
        or not isinstance(development_count, int)
        or development_count <= 0
        or development_count >= len(normalized)
    ):
        raise AllTContractError("development_count must define a non-empty holdout")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise AllTContractError("seed must be an integer")

    namespace = f"allt-v1:{seed}:"
    ranking = sorted(
        (
            hashlib.sha256(f"{namespace}{reference}".encode()).hexdigest(),
            reference,
        )
        for reference in normalized
    )
    development = sorted(reference for _, reference in ranking[:development_count])
    adaptation = sorted(reference for _, reference in ranking[development_count:])
    assignments = {
        reference: (
            "development" if reference in set(development) else "adaptation"
        )
        for reference in sorted(normalized)
    }
    return {
        "adaptation_reference_ids": adaptation,
        "assignment_by_reference": assignments,
        "development_reference_ids": development,
        "hash_namespace": namespace,
        "ranking_sha256": canonical_json_sha256(ranking),
        "seed": seed,
    }


def build_split_manifest(
    masters: Sequence[Mapping[str, object]],
    *,
    protocol_reference_ids: Sequence[str],
    development_count: int,
    expected_reference_count: int,
    expected_master_count: int,
) -> dict[str, object]:
    if isinstance(masters, (str, bytes)) or not isinstance(masters, Sequence):
        raise AllTContractError("masters must be a sequence")
    if len(masters) != expected_master_count:
        raise AllTContractError("T5 train master count differs")

    normalized = []
    for index, master in enumerate(masters):
        if not isinstance(master, Mapping):
            raise AllTContractError(f"master {index} must be a mapping")
        sequence_id = master.get("sequence_id")
        reference_id = master.get("reference_scene_id")
        if not isinstance(sequence_id, str) or not sequence_id:
            raise AllTContractError(f"master {index} lacks sequence_id")
        if not isinstance(reference_id, str) or not reference_id:
            raise AllTContractError(f"master {index} lacks reference_scene_id")
        normalized.append((sequence_id, reference_id))
    sequence_ids = [sequence_id for sequence_id, _ in normalized]
    if len(set(sequence_ids)) != len(sequence_ids):
        raise AllTContractError("T5 train sequence IDs must be unique")

    reference_ids = sorted({reference_id for _, reference_id in normalized})
    if len(reference_ids) != expected_reference_count:
        raise AllTContractError("T5 train reference count differs")
    split = build_reference_split(
        reference_ids, development_count=development_count, seed=45
    )
    validate_role_assignments(
        split,
        protocol_reference_ids=protocol_reference_ids,
        processed_test_available=False,
    )
    assignments = split["assignment_by_reference"]
    master_assignments = [
        {"role": assignments[reference_id], "sequence_id": sequence_id}
        for sequence_id, reference_id in normalized
    ]
    manifest = {
        "data_roles": {
            "adaptation": "rio_train_reference_partition",
            "development": "rio_train_reference_holdout_r1_base_exposed",
            "protocol_b": "final_evaluation_only_historical_benchmark_exposed",
            "processed_test": "unavailable",
        },
        "independent_generalization": "NOT_ESTABLISHED",
        "master_assignments": master_assignments,
        "master_counts": {
            role: sum(item["role"] == role for item in master_assignments)
            for role in ("adaptation", "development")
        },
        "processed_test_available": False,
        "protocol_b_reference_ids": sorted(protocol_reference_ids),
        "reference_counts": {
            "adaptation": len(split["adaptation_reference_ids"]),
            "development": len(split["development_reference_ids"]),
            "protocol_b_final": len(set(protocol_reference_ids)),
        },
        "reference_split": split,
        "schema_version": 1,
    }
    manifest["content_sha256"] = canonical_json_sha256(manifest)
    return manifest


def validate_role_assignments(
    split: Mapping[str, object],
    *,
    protocol_reference_ids: Sequence[str],
    processed_test_available: bool,
) -> None:
    if not isinstance(split, Mapping):
        raise AllTContractError("split must be a mapping")
    development = set(_string_list(split.get("development_reference_ids"), "development"))
    adaptation = set(_string_list(split.get("adaptation_reference_ids"), "adaptation"))
    if development & adaptation:
        raise AllTContractError("adaptation and development references must be disjoint")
    assignments = split.get("assignment_by_reference")
    if not isinstance(assignments, Mapping) or set(assignments) != development | adaptation:
        raise AllTContractError("reference assignment map differs from the split")
    if any(assignments[item] != "development" for item in development) or any(
        assignments[item] != "adaptation" for item in adaptation
    ):
        raise AllTContractError("reference assignment roles differ")
    protocol = set(_string_list(protocol_reference_ids, "Protocol-B"))
    if protocol & (development | adaptation):
        raise AllTContractError("Protocol-B references overlap adaptation/development")
    if type(processed_test_available) is not bool:
        raise AllTContractError("processed_test_available must be a boolean")
    if processed_test_available:
        raise AllTContractError(
            "processed test data appeared and requires a new evidence-role contract"
        )


def _string_list(value: object, label: str) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise AllTContractError(f"{label} references must be a sequence")
    result = list(value)
    if not result or any(not isinstance(item, str) or not item for item in result):
        raise AllTContractError(f"{label} references must be non-empty strings")
    if len(set(result)) != len(result):
        raise AllTContractError(f"{label} references must be unique")
    return result


def apply_budget_amendment(
    budget: Mapping[str, Any],
    *,
    changes: Mapping[str, int],
    reason: str,
    throughput: Mapping[str, float | int],
) -> dict[str, object]:
    if budget.get("status") != "provisional_pre_throughput" or budget.get(
        "amendment_count"
    ) != 0:
        raise AllTContractError("budget is already frozen")
    if budget.get("formal_training_started") is not False:
        raise AllTContractError("budget cannot change after formal training starts")
    if not isinstance(reason, str) or not reason.strip():
        raise AllTContractError("budget amendment reason is required")
    if not isinstance(changes, Mapping) or not changes:
        raise AllTContractError("budget amendment changes are required")
    allowed = {
        "devices",
        "gradient_accumulation",
        "optimizer_updates",
        "physical_episode_batch_per_gpu",
    }
    if set(changes) - allowed:
        raise AllTContractError("budget amendment contains unsupported fields")
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in changes.values()):
        raise AllTContractError("budget amendment values must be positive integers")
    if not isinstance(throughput, Mapping) or not throughput or any(
        isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0
        for value in throughput.values()
    ):
        raise AllTContractError("finite non-negative throughput evidence is required")

    before = dict(budget)
    after = dict(before)
    after.update(changes)
    after["effective_episode_batch"] = (
        int(after["physical_episode_batch_per_gpu"])
        * int(after["gradient_accumulation"])
        * int(after["devices"])
    )
    if after["effective_episode_batch"] != 8:
        raise AllTContractError("effective episode batch must remain 8")
    optimizer_updates = int(after["optimizer_updates"])
    if optimizer_updates % 4:
        raise AllTContractError("optimizer updates must preserve quarter evaluations")
    evaluation_interval = optimizer_updates // 4
    after["evaluation_updates"] = [
        evaluation_interval * index for index in range(5)
    ]
    after["status"] = "frozen_before_formal_training"
    after["amendment_count"] = 1
    after["amendment"] = {
        "after": {key: after[key] for key in sorted(changes)},
        "before": {key: before[key] for key in sorted(changes)},
        "reason": reason.strip(),
        "throughput": dict(sorted(throughput.items())),
    }
    return after


def build_s0_documents(
    *,
    baseline_commit: str,
    identities: Mapping[str, Mapping[str, object]],
    split_manifest: Mapping[str, object],
    cache_counts: Mapping[str, int],
) -> dict[str, object]:
    if len(baseline_commit) != 40 or any(
        character not in "0123456789abcdef" for character in baseline_commit
    ):
        raise AllTContractError("baseline commit must be a lowercase Git SHA-1")
    required_inputs = {
        "concerto_pretrained",
        "protocol_b",
        "r1_checkpoint",
        "rio_metadata",
    }
    if set(identities) != required_inputs:
        raise AllTContractError("input identity set differs")
    for label, identity in identities.items():
        if (
            not isinstance(identity, Mapping)
            or identity.get("status") != "pass"
            or not isinstance(identity.get("bytes"), int)
            or not isinstance(identity.get("sha256"), str)
            or len(identity["sha256"]) != 64
        ):
            raise AllTContractError(f"{label} identity is invalid")
    if dict(cache_counts) != {"local": 645, "full_history": 645}:
        raise AllTContractError("frozen R1 cache counts differ")
    split_payload = dict(split_manifest)
    split_digest = split_payload.get("content_sha256")
    unhashed_split = dict(split_payload)
    unhashed_split.pop("content_sha256", None)
    if split_digest != canonical_json_sha256(unhashed_split):
        raise AllTContractError("split manifest content hash differs")

    budget = {
        "amendment_count": 0,
        "devices": 2,
        "effective_episode_batch": 8,
        "evaluation_updates": [0, 1000, 2000, 3000, 4000],
        "formal_training_started": False,
        "gradient_accumulation": 4,
        "optimizer_updates": 4000,
        "physical_episode_batch_per_gpu": 1,
        "scheduler": {
            "kind": "cosine",
            "minimum_lr_fraction": 0.01,
            "warmup_fraction": 0.05,
        },
        "schema_version": 1,
        "status": "provisional_pre_throughput",
    }
    budget["content_sha256"] = canonical_json_sha256(budget)

    run_contract = {
        "baseline_commit": baseline_commit,
        "cache_counts": dict(sorted(cache_counts.items())),
        "conditional_variants": ["C3", "FH-L"],
        "development_evidence": "rio_train_reference_holdout_r1_base_exposed",
        "evaluation_seeds": [45, 46, 47],
        "experiment": "persist4d_allt_task_superiority_v1",
        "final_population": "protocol_b_43_masters_3_orders_6_references",
        "formal_candidate_metrics_started": False,
        "independent_generalization": "NOT_ESTABLISHED",
        "input_identities": {key: dict(identities[key]) for key in sorted(identities)},
        "primary_metric": "causal_prefix_t_mAP",
        "primary_reducer": "mean",
        "report_horizons": [2, 3, 4, 5],
        "required_variants": ["C0", "C1", "C2", "FH-adapt"],
        "schema_version": 1,
        "split_manifest_sha256": split_digest,
        "training_seed": 45,
    }
    run_contract["content_sha256"] = canonical_json_sha256(run_contract)

    contract_markdown = "# Experiment Contract\n\n"
    contract_markdown += "## Goal\n\n"
    contract_markdown += (
        "One frozen Persist4D checkpoint must strictly exceed the matched "
        "ReScene comparator in causal-prefix t-mAP at T=2,3,4,5. Mean score "
        "reduction is primary; no average, recovery, or efficiency result may "
        "compensate for a failed horizon.\n\n"
    )
    contract_markdown += "## Frozen evidence\n\n"
    contract_markdown += (
        f"- Baseline commit: `{baseline_commit}`\n"
        f"- R1 checkpoint SHA256: `{identities['r1_checkpoint']['sha256']}`\n"
        f"- Protocol-B SHA256: `{identities['protocol_b']['sha256']}`\n"
        f"- Split manifest SHA256: `{split_digest}`\n"
        "- R1 caches: 645 local and 645 FullHistory entries.\n\n"
    )
    contract_markdown += "## Data roles\n\n"
    contract_markdown += (
        "The 36-reference RIO train partition is adaptation data. The "
        "8-reference train holdout is development-only and was exposed to R1 "
        "base training. Protocol-B is final-only. Processed test data is "
        "unavailable, so independent generalization is not established.\n\n"
    )
    contract_markdown += "## Budget and gates\n\n"
    contract_markdown += (
        "The provisional 4000-update, effective-batch-8 budget may be amended "
        "once after real throughput preflight and before formal training. C3 "
        "and FH-L are conditional. Model selection uses only development data.\n"
    )

    source_map = "# Source Map\n\n"
    source_map += "| Logical input | Runtime binding | Git policy |\n"
    source_map += "| --- | --- | --- |\n"
    source_map += (
        "| R1 checkpoint | `R1_CHECKPOINT` | external, read-only |\n"
        "| Concerto pretrained | `CONCERTO_PRETRAINED` | external, read-only |\n"
        "| 3RScan metadata | `RIO_METADATA` | external, read-only |\n"
        "| processed datasets | `PERSIST4D_DATA_ROOT` | external, read-only |\n"
        "| R1 cache | `R1_CACHE_ROOT` | external, read-only |\n"
        "| new checkpoints/cache | `ALLT_EXTERNAL_ROOT` | external, writable |\n"
        "| Protocol-B manifest | `repo:artifacts/P6A/protocol_b_manifest.json` | tracked |\n"
    )

    documents: dict[str, object] = {
        "EXPERIMENT_CONTRACT.md": contract_markdown,
        "budget_and_schedule.json": budget,
        "run_contract.json": run_contract,
        "source_map.md": source_map,
        "split_manifest.json": split_payload,
    }
    serialized = json.dumps(documents, allow_nan=False, ensure_ascii=True)
    if any(marker in serialized for marker in ("/home/", "/mnt/", "192.168.")):
        raise AllTContractError("S0 documents contain private absolute paths")
    return documents


def write_s0_documents(
    output_root: str | Path, documents: Mapping[str, object]
) -> None:
    required = {
        "EXPERIMENT_CONTRACT.md",
        "budget_and_schedule.json",
        "run_contract.json",
        "source_map.md",
        "split_manifest.json",
    }
    if set(documents) != required:
        raise AllTContractError("S0 document set differs")
    root = Path(output_root)
    if root.is_symlink():
        raise AllTContractError("S0 output root must not be a symlink")
    root.mkdir(parents=True, exist_ok=True)
    for name in sorted(required):
        value = documents[name]
        if name.endswith(".json"):
            encoded = (
                json.dumps(
                    value,
                    allow_nan=False,
                    ensure_ascii=True,
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
        elif isinstance(value, str):
            encoded = value
        else:
            raise AllTContractError(f"{name} must be text")
        path = root / name
        if path.is_symlink():
            raise AllTContractError(f"{name} must not be a symlink")
        if path.exists():
            if not path.is_file() or path.read_text(encoding="utf-8") != encoded:
                raise AllTContractError(f"existing S0 document {name} differs")
            continue
        path.write_text(encoded, encoding="utf-8")


def _load_json(path: Path, *, label: str) -> object:
    if path.is_symlink() or not path.is_file():
        raise AllTContractError(f"{label} must be a regular non-symlink file")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AllTContractError(f"{label} cannot be decoded") from error


def _cache_record_count(path: Path, *, expected_kind: str) -> int:
    payload = _load_json(path, label=f"{expected_kind} cache progress")
    if not isinstance(payload, Mapping):
        raise AllTContractError(f"{expected_kind} cache progress must be a mapping")
    records = payload.get("records")
    if (
        payload.get("cache_kind") != expected_kind
        or payload.get("protocol_manifest_sha256") != PROTOCOL_B_SHA256
        or not isinstance(records, list)
        or len(records) != 645
    ):
        raise AllTContractError(f"{expected_kind} cache population differs")
    return len(records)


def generate_s0(
    *,
    checkpoint: Path,
    pretrained: Path,
    metadata: Path,
    data_root: Path,
    protocol: Path,
    cache_root: Path,
    output_root: Path,
) -> dict[str, object]:
    from scripts.p6a_protocol import load_t5_masters

    data_root = data_root.expanduser().resolve(strict=True)
    rio_root = data_root / "processed/rio"
    sequence_database = rio_root / "sequence_database_sliding_5.yaml"
    train_database = rio_root / "train_database.yaml"
    if (rio_root / "test_database.yaml").exists() or (
        rio_root / "test"
    ).exists():
        raise AllTContractError(
            "processed test data appeared and requires a new evidence-role contract"
        )

    identities = {
        "concerto_pretrained": require_file_identity(
            pretrained,
            expected_bytes=CONCERTO_BYTES,
            expected_sha256=CONCERTO_SHA256,
            label="Concerto pretrained",
        ),
        "protocol_b": require_file_identity(
            protocol,
            expected_bytes=PROTOCOL_B_BYTES,
            expected_sha256=PROTOCOL_B_SHA256,
            label="Protocol-B manifest",
        ),
        "r1_checkpoint": require_file_identity(
            checkpoint,
            expected_bytes=R1_CHECKPOINT_BYTES,
            expected_sha256=R1_CHECKPOINT_SHA256,
            label="R1 checkpoint",
        ),
        "rio_metadata": require_file_identity(
            metadata,
            expected_bytes=RIO_METADATA_BYTES,
            expected_sha256=RIO_METADATA_SHA256,
            label="3RScan metadata",
        ),
    }
    protocol_payload = _load_json(protocol, label="Protocol-B manifest")
    if not isinstance(protocol_payload, Mapping) or not isinstance(
        protocol_payload.get("masters"), list
    ):
        raise AllTContractError("Protocol-B masters are unavailable")
    protocol_references = sorted(
        {
            master.get("reference_scene_id")
            for master in protocol_payload["masters"]
            if isinstance(master, Mapping)
            and isinstance(master.get("reference_scene_id"), str)
        }
    )
    if len(protocol_payload["masters"]) != 43 or len(protocol_references) != 6:
        raise AllTContractError("Protocol-B population differs")

    masters = load_t5_masters(
        sequence_database,
        train_database,
        metadata_path=metadata,
        expected_split="train",
        expected_master_count=262,
        expected_cluster_count=44,
        require_supervised=True,
        substitution_policy="reject",
    )
    split_manifest = build_split_manifest(
        [
            {
                "reference_scene_id": master.reference_scene_id,
                "sequence_id": master.sequence_id,
            }
            for master in masters
        ],
        protocol_reference_ids=protocol_references,
        development_count=8,
        expected_reference_count=44,
        expected_master_count=262,
    )
    cache_counts = {
        "full_history": _cache_record_count(
            cache_root / "full_history_progress.json",
            expected_kind="full_history",
        ),
        "local": _cache_record_count(
            cache_root / "local_progress.json", expected_kind="local"
        ),
    }
    documents = build_s0_documents(
        baseline_commit=BASELINE_COMMIT,
        identities=identities,
        split_manifest=split_manifest,
        cache_counts=cache_counts,
    )
    write_s0_documents(output_root, documents)
    return {
        "artifact_count": len(documents),
        "output_reference": "repo:artifacts/allt_task_superiority_v1",
        "split_sha256": split_manifest["content_sha256"],
        "status": "pass",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_ROOT / "artifacts/P6A/protocol_b_manifest.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts/allt_task_superiority_v1",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = generate_s0(
        checkpoint=args.checkpoint,
        pretrained=args.pretrained,
        metadata=args.metadata,
        data_root=args.data_root,
        protocol=args.protocol,
        cache_root=args.cache_root,
        output_root=args.output_root,
    )
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
