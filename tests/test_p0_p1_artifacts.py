import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts"
PROFILE_ROOT = ARTIFACT_ROOT / "profiling"

OFFICIAL_COMMIT = "fb2fe42eb8f1e926567c48eea9acb874e608ee10"
METADATA_SHA256 = "674a00f50f76b198b9de44efd86c390fea3da37ba8f12cf8ccd00045e265fa64"
CHECKPOINT_SHA256 = "845ec7dec97a5fabff8fadb5d9858ac6734347b612d1a4b574213419c139de07"

REQUIRED_ARTIFACTS = (
    "environment/source_manifest.json",
    "data_audit/3rscan_temporal_stats.json",
    "data_audit/3rscan_temporal_stats.md",
    "data_audit/sequence_db_manifest.json",
    "data_audit/temporal_loader_audit.json",
    "profiling/re_scene4d_scaling.csv",
    "profiling/re_scene4d_scaling.md",
    "profiling/peak_vram_vs_t.png",
    "profiling/latency_vs_t.png",
    "profiling/throughput_vs_t.png",
    "profiling/profile_manifest.json",
    "P0_P1_GO_NOGO_REPORT.md",
)

CSV_SCHEMA = (
    "run_id",
    "mode",
    "T",
    "batch_size",
    "trial",
    "precision",
    "seed",
    "sequence_names",
    "num_points",
    "num_voxels",
    "peak_gpu_memory_mb",
    "wall_time_ms",
    "samples_per_second",
    "forward_backward_ms",
    "max_batch_size_without_oom",
    "oom_observed",
    "gpu_name",
    "gpu_uuid",
    "driver_version",
    "torch_version",
    "cuda_version",
    "backbone",
    "voxel_size",
    "freeze_mode",
    "source_commit",
)

DATABASE_CONTRACT = {
    2: {
        "sequence_count": 1482,
        "count_by_split": {"test": 147, "train": 1178, "validation": 157},
        "sha256": "974299916db02d2ee0233564eb7b36e314dada93e997584d9dfef21ad70d0416",
        "unique_reference_scene_count": 478,
    },
    3: {
        "sequence_count": 1094,
        "count_by_split": {"test": 113, "train": 858, "validation": 123},
        "sha256": "20184ca27316bd668c084c39af72144d018e556b9562826b511a1413f4986893",
        "unique_reference_scene_count": 284,
    },
    4: {
        "sequence_count": 614,
        "count_by_split": {"test": 65, "train": 474, "validation": 75},
        "sha256": "18065940eccff3572dafb8408294363969e8674978b82a4d7b673837e8ff3832",
        "unique_reference_scene_count": 124,
    },
    5: {
        "sequence_count": 342,
        "count_by_split": {"test": 37, "train": 262, "validation": 43},
        "sha256": "252363f76524bb7eeff9f65b303aadda67dcd2646477daae1ac90f7f53398290",
        "unique_reference_scene_count": 56,
    },
}


def _load_json(relative_path: str) -> dict:
    return json.loads((ARTIFACT_ROOT / relative_path).read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_profile_csv() -> tuple[list[str], list[dict[str, str]]]:
    with (PROFILE_ROOT / "re_scene4d_scaling.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _assert_portable_reference(reference: str) -> None:
    assert reference.startswith(("repo:", "external:", "local_cache:")), reference


def test_all_required_p0_p1_artifacts_exist() -> None:
    missing = [
        path for path in REQUIRED_ARTIFACTS if not (ARTIFACT_ROOT / path).is_file()
    ]
    assert not missing, f"missing required artifacts: {missing}"


def test_p0_distribution_and_source_provenance_match_real_dataset() -> None:
    stats = _load_json("data_audit/3rscan_temporal_stats.json")
    sources = _load_json("environment/source_manifest.json")

    assert stats["source"] == {
        "path": "external:3RScan/3RScan.json",
        "sha256": METADATA_SHA256,
    }
    assert stats["source_scene_count"] == 478
    assert stats["audited_scene_count"] == 432
    assert stats["excluded_splits"] == {"test": 46}
    assert stats["scan_order_semantics"] == "metadata_order_only_no_timestamps"
    assert stats["global_distribution"] == {
        "T=1": 0,
        "T=2": 194,
        "T=3": 160,
        "T=4": 68,
        "T=5": 31,
        "T=6": 8,
        "T=7": 7,
        "T=8": 7,
        "T=9": 0,
        "T=10": 1,
        "T=11": 0,
        "T=12": 2,
        "T>=2": 478,
        "T>=3": 284,
        "T>=4": 124,
        "T>=5": 56,
        "T>=6": 25,
    }
    assert {
        split: {
            key: distribution[key] for key in ("T>=2", "T>=3", "T>=4", "T>=5", "T>=6")
        }
        for split, distribution in stats["split_distribution"].items()
    } == {
        "train": {"T>=2": 385, "T>=3": 225, "T>=4": 97, "T>=5": 44, "T>=6": 19},
        "val": {"T>=2": 47, "T>=3": 30, "T>=4": 14, "T>=5": 6, "T>=6": 3},
    }

    assert sources["official_source"]["commit"] == OFFICIAL_COMMIT
    assert sources["dataset_metadata"] == {
        "path": "external:3RScan/3RScan.json",
        "path_role": "runtime_evidence_only",
        "sha256": METADATA_SHA256,
    }
    checkpoint = sources["model_weights"]["concerto"]
    assert checkpoint["sha256"] == CHECKPOINT_SHA256
    assert checkpoint["byte_size"] == 433_987_358
    assert checkpoint["revision"] == "c31f993a56129f2ba9c5d06a35957e3f05bff710"
    assert (
        checkpoint["local_reference"]
        == "local_cache:persist4d/concerto/concerto_base.pth"
    )


def test_sequence_database_manifest_locks_counts_hashes_and_portable_paths() -> None:
    manifest = _load_json("data_audit/sequence_db_manifest.json")

    assert manifest["official_source_commit"] == OFFICIAL_COMMIT
    assert manifest["metadata"] == {
        "path": "external:3RScan/3RScan.json",
        "sha256": METADATA_SHA256,
    }
    assert manifest["data_dir"] == "external:3RScan/scans"
    assert manifest["processed_dir"] == "repo:data/processed/rio"
    assert manifest["official_seed"] == 45
    assert manifest["sequence_type"] == "sliding"
    assert manifest["supervised_splits"] == ["train", "validation"]
    assert manifest["scan_order_semantics"] == "metadata_order_only_no_timestamps"

    databases = {entry["sequence_length"]: entry for entry in manifest["databases"]}
    assert set(databases) == set(DATABASE_CONTRACT)
    for horizon, expected in DATABASE_CONTRACT.items():
        database = databases[horizon]
        for field, value in expected.items():
            assert database[field] == value
        assert database["path"] == (
            f"repo:data/processed/rio/sequence_database_sliding_{horizon}.yaml"
        )
        assert database["unresolved_filepath_count_by_split"] == {
            "test": expected["count_by_split"]["test"],
            "train": 0,
            "validation": 0,
        }


def test_loader_audit_covers_eight_split_horizon_pairs_and_forty_samples() -> None:
    audit = _load_json("data_audit/temporal_loader_audit.json")

    assert audit["status"] == "pass"
    assert audit["official_source_commit"] == OFFICIAL_COMMIT
    assert audit["totals"] == {
        "requested_audits": 8,
        "completed_audits": 8,
        "loaded_samples": 40,
        "failures": 0,
    }
    assert audit["configuration"]["processed_dir"] == "repo:data/processed/rio"
    assert audit["configuration"]["explicitly_excluded_splits"] == ["test"]

    records = {(entry["horizon"], entry["split"]): entry for entry in audit["audits"]}
    expected_pairs = {
        (horizon, split)
        for horizon in (2, 3, 4, 5)
        for split in ("train", "validation")
    }
    assert set(records) == expected_pairs
    assert sum(len(entry["samples"]) for entry in records.values()) == 40

    for (horizon, split), record in records.items():
        expected_count = DATABASE_CONTRACT[horizon]["count_by_split"][split]
        assert record["database_count"] == expected_count
        assert record["loader_sequence_count"] == expected_count
        assert record["database_sha256"] == DATABASE_CONTRACT[horizon]["sha256"]
        assert record["success_count"] == 5
        assert record["failure_count"] == 0
        assert record["exceptions"] == []
        assert record["validation_errors"] == []
        _assert_portable_reference(record["database_path"])
        for sample in record["samples"]:
            _assert_portable_reference(sample["change_filepath"])
            assert sample["temporal_stages"] == list(range(horizon))
            assert sample["projected_change_dim"] == 1
            assert sample["projection_matches_official_rule"] is True
            if horizon == 2:
                assert sample["raw_change_dim"] == 1
                assert len(sample["raw_change_shape"]) == 1
            else:
                assert sample["raw_change_dim"] == 2
                assert sample["raw_change_shape"][1] == horizon - 1


def test_profile_csv_has_strict_schema_complete_groups_and_finite_measurements() -> (
    None
):
    fieldnames, rows = _read_profile_csv()

    assert tuple(fieldnames) == CSV_SCHEMA
    assert len(rows) == 80
    assert all(set(row) == set(CSV_SCHEMA) for row in rows)
    groups: dict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        horizon = int(row["T"])
        mode = row["mode"]
        groups[(horizon, mode)].append(row)

        assert int(row["batch_size"]) == 1
        assert int(row["seed"]) == 45
        assert row["precision"] == "fp32"
        assert row["gpu_uuid"] == "redacted"
        assert row["backbone"] == "concerto_base"
        assert float(row["voxel_size"]) == 0.02
        assert row["freeze_mode"] == "backbone_encoder"
        assert row["source_commit"] == OFFICIAL_COMMIT
        assert int(row["num_points"]) > 0
        assert int(row["num_voxels"]) > 0
        for field in (
            "peak_gpu_memory_mb",
            "wall_time_ms",
            "samples_per_second",
            "forward_backward_ms",
        ):
            value = float(row[field])
            assert math.isfinite(value) and value > 0, (field, value)
        expected_throughput = (
            1000.0 * int(row["batch_size"]) / float(row["wall_time_ms"])
        )
        assert math.isclose(
            float(row["samples_per_second"]),
            expected_throughput,
            rel_tol=1e-9,
            abs_tol=1e-12,
        )

    expected_groups = {
        (horizon, mode)
        for horizon in (2, 3, 4, 5)
        for mode in ("inference", "training")
    }
    assert set(groups) == expected_groups
    windows_by_group = {}
    for (horizon, mode), group in groups.items():
        assert len(group) == 10
        assert sorted(int(row["trial"]) for row in group) == list(range(10))
        windows = []
        for row in group:
            sequence_names = json.loads(row["sequence_names"])
            assert len(sequence_names) == 1
            assert len(sequence_names[0].split("-")) == horizon
            windows.append(sequence_names[0])
        window_counts = Counter(windows)
        assert len(window_counts) == 5
        assert set(window_counts.values()) == {2}
        windows_by_group[(horizon, mode)] = set(window_counts)
    for horizon in (2, 3, 4, 5):
        assert (
            windows_by_group[(horizon, "inference")]
            == windows_by_group[(horizon, "training")]
        )


def test_profile_manifest_records_windows_and_batch_search_semantics() -> None:
    manifest = _load_json("profiling/profile_manifest.json")
    _, rows = _read_profile_csv()

    assert manifest["status"] == "pass"
    assert manifest["measurement_rows"] == 80
    assert manifest["official_source_commit"] == OFFICIAL_COMMIT
    assert manifest["configuration"]["horizons"] == [2, 3, 4, 5]
    assert manifest["configuration"]["measurement_iterations"] == 10
    assert manifest["configuration"]["profile_scenes"] == 5
    assert manifest["configuration"]["oom_safety_margin_mb"] == 512
    assert manifest["configuration"]["batch_search_skipped"] is False
    _assert_portable_reference(manifest["configuration"]["config_reference"])
    _assert_portable_reference(manifest["configuration"]["processed_dir_reference"])

    assert set(manifest["horizons"]) == {"2", "3", "4", "5"}
    sequence_names_by_horizon = defaultdict(set)
    for row in rows:
        sequence_names_by_horizon[int(row["T"])].update(
            json.loads(row["sequence_names"])
        )
    for horizon in (2, 3, 4, 5):
        result = manifest["horizons"][str(horizon)]
        selected = result["selected_windows"]
        assert len(selected) == 5
        assert {
            entry["sequence_name"] for entry in selected
        } == sequence_names_by_horizon[horizon]
        database = result["sequence_database"]
        assert database["sha256"] == DATABASE_CONTRACT[horizon]["sha256"]
        assert database["reference"] == (
            f"repo:data/processed/rio/sequence_database_sliding_{horizon}.yaml"
        )

    expected_search_keys = {
        f"{horizon}:{mode}"
        for horizon in (2, 4, 5)
        for mode in ("inference", "training")
    }
    assert set(manifest["batch_search"]) == expected_search_keys
    assert not any(key.startswith("3:") for key in manifest["batch_search"])

    for key, result in manifest["batch_search"].items():
        horizon, mode = key.split(":")
        maximum = int(result["max_batch_size_without_oom"])
        cap = int(result["configured_cap"])
        assert 0 <= maximum <= cap == manifest["configuration"]["max_batch_search"]
        assert result["oom_safety_margin_mb"] == 512
        assert result["safety_reserve_allocated_bytes"] == 512 * 1024 * 1024
        assert result["attempts"]
        outcomes = {attempt["outcome"] for attempt in result["attempts"]}
        assert outcomes <= {"success", "oom"}
        if result["oom_observed"]:
            assert result["stop_reason"] == "cuda_oom"
            assert "oom" in outcomes
        else:
            assert result["stop_reason"] == "configured_cap_right_censored"
            assert maximum == cap
            assert outcomes == {"success"}

        matching_rows = [
            row for row in rows if row["T"] == horizon and row["mode"] == mode
        ]
        assert len(matching_rows) == 10
        assert {int(row["max_batch_size_without_oom"]) for row in matching_rows} == {
            maximum
        }
        assert {
            row["oom_observed"].strip().lower() == "true" for row in matching_rows
        } == {bool(result["oom_observed"])}

    t3_rows = [row for row in rows if int(row["T"]) == 3]
    assert len(t3_rows) == 20
    assert all(row["max_batch_size_without_oom"] == "" for row in t3_rows)
    assert all(row["oom_observed"] == "" for row in t3_rows)


def test_profile_artifact_hashes_and_png_payloads_are_valid() -> None:
    manifest = _load_json("profiling/profile_manifest.json")
    expected_names = {
        "re_scene4d_scaling.csv",
        "re_scene4d_scaling.md",
        "peak_vram_vs_t.png",
        "latency_vs_t.png",
        "throughput_vs_t.png",
    }

    assert set(manifest["artifacts"]) == expected_names
    for filename, provenance in manifest["artifacts"].items():
        path = PROFILE_ROOT / filename
        assert provenance["byte_size"] == path.stat().st_size
        assert provenance["sha256"] == _sha256(path)

    for filename in sorted(name for name in expected_names if name.endswith(".png")):
        payload = (PROFILE_ROOT / filename).read_bytes()
        assert len(payload) > 1024
        assert payload.startswith(b"\x89PNG\r\n\x1a\n")
        assert payload[12:16] == b"IHDR"


def test_report_answers_all_stage_gate_questions_with_one_decision() -> None:
    report_path = ARTIFACT_ROOT / "P0_P1_GO_NOGO_REPORT.md"
    assert report_path.is_file(), "P0/P1 gate report has not been written"
    report = report_path.read_text(encoding="utf-8")
    for heading in (
        "A. T=3/4/5 direct execution",
        "B. Resource scaling curve",
        "C. Maximum T=4/5 batch size",
        "D. T>2 bugs and assumptions",
        "E. Strength of the scalability limitation",
        "F. Recommendation",
    ):
        assert heading in report

    decisions = re.findall(r"\bDecision:\s*(GO|NO-GO)\b", report)
    assert len(decisions) == 1, f"expected exactly one Decision, found {decisions}"
    assert "P2 method design" not in report
    assert "P2 T=2 baseline reproduction" in report
    assert "G2/G3" in report


def test_report_quantitative_claims_match_profile_evidence() -> None:
    report = (ARTIFACT_ROOT / "P0_P1_GO_NOGO_REPORT.md").read_text(encoding="utf-8")
    manifest = _load_json("profiling/profile_manifest.json")
    _, rows = _read_profile_csv()
    grouped: dict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["T"]), row["mode"])].append(row)

    summaries = {}
    for key, group in grouped.items():
        summaries[key] = {
            field: median(float(row[field]) for row in group)
            for field in (
                "num_points",
                "num_voxels",
                "peak_gpu_memory_mb",
                "wall_time_ms",
                "samples_per_second",
            )
        }

    for horizon in (2, 3, 4, 5):
        for mode, label in (
            ("inference", "inference"),
            ("training", "forward/backward"),
        ):
            summary = summaries[(horizon, mode)]
            expected_row = (
                f"| {label} | {horizon} | "
                f"{summary['peak_gpu_memory_mb']:,.1f} | "
                f"{summary['wall_time_ms']:,.1f} | "
                f"{summary['samples_per_second']:.4f} |"
            )
            assert expected_row in report

    points_ratio = (
        summaries[(5, "inference")]["num_points"]
        / summaries[(2, "inference")]["num_points"]
    )
    voxels_ratio = (
        summaries[(5, "inference")]["num_voxels"]
        / summaries[(2, "inference")]["num_voxels"]
    )
    assert f"increase by {points_ratio:.3f}x and {voxels_ratio:.3f}x" in report

    for mode, prefix in (("inference", "Inference"), ("training", "Forward/backward")):
        t2 = summaries[(2, mode)]
        t5 = summaries[(5, mode)]
        memory_ratio = t5["peak_gpu_memory_mb"] / t2["peak_gpu_memory_mb"]
        latency_ratio = t5["wall_time_ms"] / t2["wall_time_ms"]
        throughput_ratio = t5["samples_per_second"] / t2["samples_per_second"]
        expected = (
            f"{prefix} VRAM increases {memory_ratio:.3f}x "
            f"(+{(memory_ratio - 1) * 100:.1f}%), latency {latency_ratio:.3f}x "
            f"(+{(latency_ratio - 1) * 100:.1f}%), and throughput falls to "
            f"{throughput_ratio:.3f}x (-{(1 - throughput_ratio) * 100:.1f}%)."
        )
        assert expected in report

    for horizon in (4, 5):
        for mode, label in (
            ("inference", "inference"),
            ("training", "forward/backward"),
        ):
            search = manifest["batch_search"][f"{horizon}:{mode}"]
            nearest_oom = min(
                attempt["batch_size"]
                for attempt in search["attempts"]
                if attempt["outcome"] == "oom"
            )
            assert (
                f"| {label} | {horizon} | "
                f"{search['max_batch_size_without_oom']} | {nearest_oom} | "
                "observed CUDA OOM |"
            ) in report

    t2_inference = manifest["batch_search"]["2:inference"]
    t2_training = manifest["batch_search"]["2:training"]
    t5_inference = manifest["batch_search"]["5:inference"]
    t5_training = manifest["batch_search"]["5:training"]
    assert t2_inference["stop_reason"] == "configured_cap_right_censored"
    assert (
        f"`<={t5_inference['max_batch_size_without_oom']}/"
        f"{t2_inference['max_batch_size_without_oom']} = "
        f"{t5_inference['max_batch_size_without_oom'] / t2_inference['max_batch_size_without_oom']:.3f}`"
    ) in report
    assert (
        f"`{t5_training['max_batch_size_without_oom']}/"
        f"{t2_training['max_batch_size_without_oom']} = "
        f"{t5_training['max_batch_size_without_oom'] / t2_training['max_batch_size_without_oom']:.3f}`"
    ) in report

    assert (
        summaries[(5, "training")]["peak_gpu_memory_mb"]
        / summaries[(2, "training")]["peak_gpu_memory_mb"]
        > 2.0
    )
    assert (
        t5_training["max_batch_size_without_oom"]
        < t2_training["max_batch_size_without_oom"]
    )
    assert "**Decision: GO**" in report


def test_release_text_artifacts_contain_no_personal_absolute_paths() -> None:
    text_artifacts = sorted(
        path
        for path in ARTIFACT_ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in {".json", ".md", ".csv"}
    )
    assert text_artifacts
    linux_home = "ho" + "me"
    user_home = "Us" + "ers"
    forbidden = (
        re.compile(r"/" + linux_home + r"/[^/\s]+/"),
        re.compile(r"/" + user_home + r"/[^/\s]+/"),
        re.compile(r"[A-Za-z]:\\" + user_home + r"\\[^\\\s]+\\"),
    )
    for path in text_artifacts:
        payload = path.read_text(encoding="utf-8")
        for pattern in forbidden:
            assert not pattern.search(payload), (
                f"personal path in {path}: {pattern.pattern}"
            )
