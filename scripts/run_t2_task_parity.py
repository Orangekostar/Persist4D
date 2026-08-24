"""Run the blocking T2 current-stage task parity regression."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch

from scripts.evaluate_persist4d_p6a import (
    build_rio_class_mapper,
    expected_cache_keys,
)
from scripts.p6a_cache import cache_payload_digest, write_cache_entry
from scripts.p6a_metrics import OfficialMetricAccumulator
from scripts.rescene_task_postprocess import (
    OfficialTaskPrediction,
    extract_official_task_prediction,
)
from scripts.run_system_comparison import (
    FULL_CACHE_MANIFEST,
    LOCAL_CACHE_MANIFEST,
    REPRODUCIBILITY_BINDING,
    SOURCE_PROTOCOL,
    _build_frozen_setup,
    _full_producer,
    _local_producer,
    _publish_exact_bytes,
)
from scripts.system_comparison_inference import (
    assert_t2_observation_regression,
    deterministic_inference_runtime,
)
from scripts.system_comparison_metrics import (
    causal_prefix_pair_from_payload,
    current_stage_pair,
)
from scripts.system_comparison_v2_cache import (
    build_task_sidecar,
    write_task_sidecar,
)
from scripts.system_comparison_v2_parity import (
    T2ParityError,
    compare_t2_task_predictions,
    summarize_t2_rows,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts/tmap_root_cause_v2"
DEFAULT_METADATA = Path("/home/ww/3RScan.json")
DEFAULT_SIDECAR_ENTRIES = Path(
    "/mnt/shared/ww/persist4d-tmap-root-cause-v2/"
    "system_comparison_v2/task_sidecars/entries"
)
DEFAULT_RAW_ENTRIES = Path(
    "/mnt/shared/ww/persist4d-tmap-root-cause-v2/"
    "system_comparison_v2/raw_predictions/entries"
)
EXPECTED_T2_UNITS = 43 * 3
CSV_FIELDS = (
    "master_sequence_id",
    "reference_scene_id",
    "order_id",
    "history_scan_ids",
    "candidate_count_full",
    "candidate_count_local",
    "masks_equal",
    "classes_equal",
    "score_max_abs_diff",
    "scores_allclose",
    "full_current_stage_AP",
    "local_current_stage_AP",
    "AP_abs_diff",
    "parity_pass",
    "raw_observation_fingerprint",
    "sidecar_content_sha256",
    "full_history_content_sha256",
    "fresh_raw_content_sha256",
    "frozen_v1_raw_content_sha256",
    "frozen_v1_raw_reproduced",
    "frozen_v1_full_history_content_sha256",
    "frozen_v1_full_history_reproduced",
)


def _json(path: Path) -> Mapping[str, object]:
    if path.is_symlink() or not path.is_file():
        raise T2ParityError(f"required manifest is unavailable: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise T2ParityError(f"manifest must be a mapping: {path}")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_digest(value: object) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _entries_by_identity(
    manifest: Mapping[str, object], *, stage_field: str
) -> dict[tuple[str, str, int], Mapping[str, object]]:
    entries = manifest.get("entries")
    if isinstance(entries, (str, bytes)) or not isinstance(entries, Sequence):
        raise T2ParityError("cache manifest entries must be a sequence")
    result: dict[tuple[str, str, int], Mapping[str, object]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("key"), Mapping):
            raise T2ParityError("cache manifest entry must contain a key")
        key = entry["key"]
        identity = (
            str(key["master_sequence_id"]),
            str(key["order_id"]),
            int(key[stage_field]),
        )
        if identity in result:
            raise T2ParityError("cache manifest contains duplicate logical cells")
        result[identity] = entry
    return result


def _local_prediction(sidecar: Mapping[str, object]) -> dict[str, torch.Tensor]:
    task = sidecar["task_prediction"]
    if not isinstance(task, Mapping):
        raise T2ParityError("task sidecar prediction must be a mapping")
    return {
        "pred_masks": task["pred_masks"],
        "pred_scores": task["pred_scores"],
        "pred_classes": task["pred_classes"],
    }


def _csv_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        record = dict(row)
        record["history_scan_ids"] = json.dumps(
            record["history_scan_ids"], ensure_ascii=True, separators=(",", ":")
        )
        writer.writerow(record)
    return stream.getvalue().encode("utf-8")


def _report_bytes(
    *,
    summary: Mapping[str, object],
    source_commit: str,
    aggregate_full_ap: float,
    aggregate_local_ap: float,
    rows_sha256: str,
    sidecar_manifest_sha256: str,
) -> bytes:
    aggregate_difference = abs(aggregate_full_ap - aggregate_local_ap)
    lines = [
        "# T2 Official-Task Current-Stage Parity",
        "",
        f"- Status: `{summary['status']}`",
        f"- Source commit: `{source_commit}`",
        "- Scope: `43 masters x 3 preregistered orders x T2 = 129 units`",
        f"- Units: `{summary['unit_count']}`",
        f"- Passed: `{summary['pass_count']}`",
        f"- Failed: `{summary['fail_count']}`",
        f"- Maximum score absolute difference: `{summary['max_score_abs_diff']}`",
        f"- Maximum per-unit AP absolute difference: `{summary['max_AP_abs_diff']}`",
        f"- Aggregate FullHistory current-stage AP: `{aggregate_full_ap}`",
        f"- Aggregate local-window current-stage AP: `{aggregate_local_ap}`",
        f"- Aggregate AP absolute difference: `{aggregate_difference}`",
        f"- Per-unit rows SHA256: `{rows_sha256}`",
        f"- Sidecar manifest SHA256: `{sidecar_manifest_sha256}`",
        "- Sidecar storage: `external:system_comparison_v2/task_sidecars/entries`",
        "- Matched raw storage: `external:system_comparison_v2/raw_predictions/entries`",
        f"- Frozen V1 local raw bytewise replays: `{summary['frozen_v1_raw_reproduced_count']}/{summary['unit_count']}`",
        f"- Frozen V1 FullHistory bytewise replays: `{summary['frozen_v1_full_history_reproduced_count']}/{summary['unit_count']}`",
        "",
        "## Gate Semantics",
        "",
        "Each local task prediction is produced from the same forward pass as its ",
        "content-addressed V2 raw observation. FullHistory and Local are replayed in ",
        "the same deterministic process, and the existing T2 observation fingerprint ",
        "regression must pass before task comparison. The gate then requires exact ",
        "candidate count, masks, and classes; scores use `rtol=0, atol=1e-7`; official ",
        "raw-local AP uses the same current-stage target on both sides.",
        "",
        "Frozen V1 content hashes are retained as diagnostics rather than replay gates: ",
        "the original audit established same-process repeatability, not cross-process ",
        "bitwise replay. This gate tests evaluator/input parity only. It does not ",
        "establish causal t-mAP parity or tune task post-processing/memory behavior.",
        "",
    ]
    return "\n".join(lines).encode("utf-8")


def run_t2_task_parity(
    *,
    metadata_path: Path,
    device_name: str,
    sidecar_entry_directory: Path,
    raw_entry_directory: Path,
    artifact_root: Path = ARTIFACT_ROOT,
) -> Mapping[str, object]:
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    binding = _json(REPRODUCIBILITY_BINDING)
    local_manifest = _json(LOCAL_CACHE_MANIFEST)
    full_manifest = _json(FULL_CACHE_MANIFEST)
    protocol_sha256 = _file_sha256(SOURCE_PROTOCOL)
    if protocol_sha256 != binding.get("protocol_sha256"):
        raise T2ParityError("Protocol-B manifest differs from frozen binding")
    if local_manifest.get("status") != "pass" or full_manifest.get("status") != "pass":
        raise T2ParityError("frozen cache manifests must pass")

    setup = _build_frozen_setup(
        binding=binding,
        metadata_path=metadata_path,
        device_name=device_name,
    )
    if dict(local_manifest.get("provenance", {})) != setup.local_provenance:
        raise T2ParityError("local cache provenance differs from frozen setup")
    if dict(full_manifest.get("provenance", {})) != setup.full_provenance:
        raise T2ParityError("FullHistory cache provenance differs from frozen setup")

    local_entries = _entries_by_identity(local_manifest, stage_field="stage_index")
    full_entries = _entries_by_identity(full_manifest, stage_field="horizon")
    t2_keys = [
        key for key in expected_cache_keys(setup.protocol) if key["stage_index"] == 1
    ]
    if len(t2_keys) != EXPECTED_T2_UNITS:
        raise T2ParityError("Protocol B does not contain 129 T2 units")

    producer = _local_producer(setup)
    producer.provenance = {
        **setup.local_provenance,
        "source_commit": source_commit,
    }
    full_producer = _full_producer(setup)
    class_mapper = build_rio_class_mapper(setup.dataset)
    full_accumulator = OfficialMetricAccumulator(mode="raw_local")
    local_accumulator = OfficialMetricAccumulator(mode="raw_local")
    rows: list[dict[str, object]] = []
    sidecar_entries: list[dict[str, object]] = []
    seed = int(setup.p6a_config["protocol_b"]["seed"])
    if setup.device is None:
        raise T2ParityError("CUDA setup did not expose a device")

    with deterministic_inference_runtime(seed, setup.device):
        for key in t2_keys:
            identity = (
                str(key["master_sequence_id"]),
                str(key["order_id"]),
                1,
            )
            if identity not in local_entries or identity not in full_entries:
                raise T2ParityError(f"frozen cache lacks T2 cell {identity}")
            local_entry = local_entries[identity]
            full_entry = full_entries[identity]
            produced = producer.produce_bundle(
                key,
                task_prediction_builder=extract_official_task_prediction,
                class_mapper=class_mapper,
            )
            if not isinstance(produced.task_prediction, OfficialTaskPrediction):
                raise T2ParityError("local forward did not produce official task output")
            full_produced = full_producer.produce_bundle(full_entry["key"])
            full_payload = full_produced.payload
            try:
                assert_t2_observation_regression(full_payload, produced.payload)
            except Exception as error:
                raise T2ParityError(
                    f"fresh local/full observation differs for T2 cell {identity}"
                ) from error

            sidecar = build_task_sidecar(
                raw_cache_payload=produced.payload,
                official_prediction=produced.task_prediction,
                protocol_manifest_sha256=protocol_sha256,
            )
            raw_entry = write_cache_entry(raw_entry_directory, produced.payload)
            sidecar_entry = write_task_sidecar(sidecar_entry_directory, sidecar)
            row = compare_t2_task_predictions(
                full_payload=full_payload,
                local_sidecar=sidecar,
                full_history_content_sha256=str(full_payload["content_sha256"]),
                sidecar_content_sha256=str(sidecar_entry["content_sha256"]),
            )
            row.update(
                {
                    "fresh_raw_content_sha256": cache_payload_digest(
                        produced.payload
                    ),
                    "frozen_v1_raw_content_sha256": local_entry[
                        "content_sha256"
                    ],
                    "frozen_v1_raw_reproduced": (
                        cache_payload_digest(produced.payload)
                        == local_entry["content_sha256"]
                    ),
                    "frozen_v1_full_history_content_sha256": full_entry[
                        "content_sha256"
                    ],
                    "frozen_v1_full_history_reproduced": (
                        full_payload["content_sha256"]
                        == full_entry["content_sha256"]
                    ),
                }
            )
            rows.append(row)
            sidecar_entries.append(
                {
                    **sidecar_entry,
                    "matched_raw_entry": raw_entry,
                    "source_raw_observation_fingerprint": row[
                        "raw_observation_fingerprint"
                    ],
                }
            )

            full_pair = current_stage_pair(
                causal_prefix_pair_from_payload(full_payload)
            )
            full_accumulator.update(full_pair.prediction, full_pair.target)
            local_accumulator.update(_local_prediction(sidecar), full_pair.target)

    summary = summarize_t2_rows(rows, expected_unit_count=EXPECTED_T2_UNITS)
    aggregate_full_ap = full_accumulator.compute()["raw_local_AP"]
    aggregate_local_ap = local_accumulator.compute()["raw_local_AP"]
    aggregate_difference = abs(aggregate_full_ap - aggregate_local_ap)
    if aggregate_difference > 1e-12:
        summary = {**summary, "status": "fail"}
    summary = {
        **summary,
        "frozen_v1_raw_reproduced_count": sum(
            row["frozen_v1_raw_reproduced"] is True for row in rows
        ),
        "frozen_v1_full_history_reproduced_count": sum(
            row["frozen_v1_full_history_reproduced"] is True for row in rows
        ),
    }
    rows_sha256 = _canonical_digest(rows)
    sidecar_manifest = {
        "schema_version": 1,
        "status": summary["status"],
        "source_commit": source_commit,
        "protocol_manifest_sha256": protocol_sha256,
        "checkpoint_sha256": binding["checkpoint_sha256"],
        "frozen_v1_raw_cache_entries_sha256": local_manifest["entries_sha256"],
        "frozen_v1_full_history_entries_sha256": full_manifest[
            "entries_sha256"
        ],
        "entry_count": len(sidecar_entries),
        "entries_sha256": _canonical_digest(sidecar_entries),
        "entries": sidecar_entries,
    }
    manifest_bytes = (
        json.dumps(
            sidecar_manifest,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    _publish_exact_bytes(
        artifact_root / "t2_task_sidecar_manifest.json", manifest_bytes
    )
    _publish_exact_bytes(
        artifact_root / "t2_task_parity_per_unit.csv", _csv_bytes(rows)
    )
    _publish_exact_bytes(
        artifact_root / "T2_TASK_PARITY.md",
        _report_bytes(
            summary=summary,
            source_commit=source_commit,
            aggregate_full_ap=aggregate_full_ap,
            aggregate_local_ap=aggregate_local_ap,
            rows_sha256=rows_sha256,
            sidecar_manifest_sha256=manifest_sha256,
        ),
    )
    if summary["status"] != "pass":
        raise T2ParityError("T2 official-task current-stage parity gate failed")
    return {
        **summary,
        "aggregate_full_current_stage_AP": aggregate_full_ap,
        "aggregate_local_current_stage_AP": aggregate_local_ap,
        "aggregate_AP_abs_diff": aggregate_difference,
        "rows_sha256": rows_sha256,
        "sidecar_manifest_sha256": manifest_sha256,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the blocking T2 current-stage official-task parity gate."
    )
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--sidecar-entry-directory",
        type=Path,
        default=DEFAULT_SIDECAR_ENTRIES,
    )
    parser.add_argument(
        "--raw-entry-directory",
        type=Path,
        default=DEFAULT_RAW_ENTRIES,
    )
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    result = run_t2_task_parity(
        metadata_path=arguments.metadata,
        device_name=arguments.device,
        sidecar_entry_directory=arguments.sidecar_entry_directory,
        raw_entry_directory=arguments.raw_entry_directory,
        artifact_root=arguments.artifact_root,
    )
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
