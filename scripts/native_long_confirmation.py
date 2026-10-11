"""Separate, immutable formal populations for the locked native-long checkpoints."""

import argparse
import collections
import hashlib
import json
from pathlib import Path

import yaml

from scripts.native_long_budget import write_json_atomic
from scripts.native_long_postselection import read_json, sha256


def has_supervised_instance(points, taxonomy):
    import numpy as np

    if points.ndim != 2 or points.shape[1] < 12:
        raise ValueError("formal point array lacks semantic/global-instance columns")
    return bool(np.any(np.isin(points[:, 10], taxonomy) & (points[:, 11] >= 0)))


def bind_record(role, unit, scans, aliases, train_references, *, ambiguities=None):
    if not 2 <= len(scans) <= 5 or len(set(scans)) != len(scans):
        raise ValueError("formal input must contain two to five distinct scans")
    rows = [aliases[scan] for scan in scans]
    refs = {row["reference_id"] for row in rows}
    if len(refs) != 1:
        raise ValueError("formal input crosses a physical reference")
    ref = next(iter(refs))
    if ref in train_references:
        raise ValueError("formal reference overlaps new TRAIN population")
    physical = [row["physical_scan_id"] for row in rows]
    if len(set(physical)) != len(scans):
        raise ValueError("formal input aliases do not denote distinct physical scans")
    identity = {"role": role, "unit": unit, "physical_scan_ids": physical}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return {"domain": "rio", "horizon": len(scans), "reference_id": ref,
            "scan_indices": [row["index"] for row in rows], "scan_ids": list(scans),
            "physical_scan_ids": physical, "input_id": f"native-long-formal-v1:{digest}",
            "source_unit": unit, "role": role, "ambiguities": ambiguities}


def freeze_json(path, payload):
    if path.exists():
        if read_json(path) != payload:
            raise ValueError(f"immutable formal manifest already differs: {path}")
    else:
        write_json_atomic(path, payload)


def materialize(root, project, output):
    import numpy as np

    plan_path = project / "artifacts/native_long_postselection_v1/FORMAL_PLAN.json"
    plan = read_json(plan_path)
    cal_lock_path = root / "selection/CAL_LOCK.json"
    if sha256(cal_lock_path) != plan["CAL_LOCK_sha256"]:
        raise ValueError("CAL lock changed after module selection")
    source = read_json(root / "SOURCE_AND_INITIALIZATION.json")
    data_root = Path(source["assets"]["data_root"])
    folder = data_root / "processed/rio"
    paths = {
        "plan": plan_path, "cal_lock": cal_lock_path, "population": root / "POPULATION.json",
        "PB": project / "artifacts/P6A/protocol_b_manifest.json",
        "roles": project / "artifacts/perception_gain_v2/DATA_ROLES.json",
        "LOCAL-T2": project / "artifacts/perception_gain_v2/foundation/LOCAL_POPULATION_AUDIT.json",
        "metadata": Path(source["assets"]["rio_metadata"]),
        "metric_spec": folder / "rio.yaml",
        "historical_supervision_filter": project / "datasets/semseg.py",
        "validation_database": folder / "validation_database.yaml",
        **{f"sequences_{h}": folder / f"sequence_database_sliding_{h}.yaml" for h in range(2, 6)}}
    bindings = {k: {"path": str(p), "sha256": sha256(p)} for k, p in paths.items()}
    metadata = read_json(paths["metadata"])
    reference_by_scan = {}
    for room in metadata:
        for scan in (room["reference"], *(r["reference"] for r in room["scans"])):
            if scan in reference_by_scan:
                raise ValueError("metadata repeats a physical scan")
            reference_by_scan[scan] = room["reference"]
    database = yaml.safe_load(paths["validation_database"].read_text())
    aliases = {}
    supervised = {}
    taxonomy = yaml.safe_load(paths["metric_spec"].read_text())["valid_class_ids"]
    for index, row in enumerate(database):
        alias = Path(row["instance_gt_filepath"]).stem
        physical = Path(row["raw_filepath"]).parent.name
        if alias in aliases:
            raise ValueError("validation database repeats an alias")
        aliases[alias] = {"index": index, "physical_scan_id": physical,
                          "reference_id": reference_by_scan[physical]}
        point_path = Path(row["filepath"].replace("../../", ""))
        if not point_path.is_absolute():
            point_path = data_root / point_path.relative_to("data")
        points = np.load(point_path, mmap_mode="r", allow_pickle=False)
        supervised[alias] = has_supervised_instance(points, taxonomy)
    population = read_json(paths["population"])
    train = {r["reference_id"] for r in population["references"] if r["role"] == "TRAIN"}
    roles = read_json(paths["roles"])["roles"]
    sequences = {h: yaml.safe_load(paths[f"sequences_{h}"].read_text()) for h in range(2, 6)}
    records = {role: [] for role in ("PB", "LOCAL-T2", "ADDITIONAL")}
    for master in read_json(paths["PB"])["masters"]:
        unit = master["master_sequence_id"]
        original = sequences[5][unit]
        if original["type"] != "validation":
            raise ValueError("PB master is not validation data")
        for order, value in master["orders"].items():
            for h in range(2, 6):
                prefix = value["prefixes"][str(h)]
                row = bind_record("PB", f"{unit}:{order}", prefix["scan_ids"], aliases, train,
                                  ambiguities=original.get("ambiguities"))
                if (row["scan_indices"] != prefix["scan_indices"]
                        or row["reference_id"] != master["reference_scene_id"]):
                    raise ValueError("PB manifest indices/reference differ from validation database")
                records["PB"].append(row)
    local = read_json(paths["LOCAL-T2"])
    for unit in local["sequence_names"]:
        original = sequences[2][unit]
        if original["type"] != "validation":
            raise ValueError("LOCAL-T2 source sequence is not validation data")
        records["LOCAL-T2"].append(bind_record("LOCAL-T2", unit, unit.split("-"), aliases, train,
                                                ambiguities=original.get("ambiguities")))
    excluded = []
    for h in (2, 3, 4):
        for unit, original in sequences[h].items():
            if original["type"] != "validation":
                continue
            scans = unit.split("-")
            if aliases[scans[0]]["reference_id"] in roles["ADDITIONAL"]:
                # Preserve the historical sequence-level filter: a partly empty
                # sequence stays; only sequences with no valid instance anywhere go.
                if not any(supervised[scan] for scan in scans):
                    excluded.append({"role": "ADDITIONAL", "unit": unit, "horizon": h,
                                     "reason": "HISTORICAL_NO_VALID_INSTANCE_IN_ENTIRE_SEQUENCE"})
                    continue
                records["ADDITIONAL"].append(bind_record("ADDITIONAL", unit, scans, aliases, train,
                                                        ambiguities=original.get("ambiguities")))
    expected = {"PB": {str(h): 129 for h in range(2, 6)},
                "LOCAL-T2": {"2": 154}, "ADDITIONAL": {"2": 111, "3": 77, "4": 32}}
    reference_counts = {"PB": {str(h): 6 for h in range(2, 6)},
                        "LOCAL-T2": {"2": 46}, "ADDITIONAL": {"2": 40, "3": 23, "4": 8}}
    observed, refs, input_ids = {}, {}, set()
    for role, rows in records.items():
        records[role] = sorted(rows, key=lambda r: (r["horizon"], r["input_id"]))
        counts = dict(collections.Counter(str(r["horizon"]) for r in rows))
        grouped = collections.defaultdict(set)
        for row in rows:
            if row["input_id"] in input_ids:
                raise ValueError("formal input identity duplicated")
            input_ids.add(row["input_id"])
            grouped[str(row["horizon"])].add(row["reference_id"])
        if counts != expected[role] or {h: len(v) for h, v in grouped.items()} != reference_counts[role]:
            raise ValueError(f"{role} historical population coverage differs")
        refs[role] = set.union(*grouped.values())
        observed[role] = {"inputs_by_horizon": counts, "references_by_horizon": reference_counts[role],
                          "reference_ids": sorted(refs[role]), "TRAIN_overlap": sorted(refs[role] & train)}
    if refs["PB"] != set(roles["PB"]) or refs["LOCAL-T2"] != set(local["validation_reference_ids"]):
        raise ValueError("formal physical references differ from their audited manifest")
    filenames = {Path("processed/rio") / p.name for p in folder.glob("*.yaml")}
    for row in database:
        for key in ("filepath", "instance_gt_filepath"):
            path = Path(row[key].replace("../../", ""))
            relative = path.relative_to(data_root) if path.is_absolute() else path.relative_to("data")
            filenames.add(relative)
    data_files = []
    for relative in sorted(filenames):
        path = data_root / relative
        if not path.is_file():
            raise ValueError(f"formal input file is unavailable: {path}")
        data_files.append({"relative_path": str(relative), "bytes": path.stat().st_size, "sha256": sha256(path)})
    if any(sha256(paths[k]) != row["sha256"] for k, row in bindings.items()):
        raise ValueError("formal source changed during materialization")
    manifest = {"schema": 1, "status": "MATERIALIZED_NO_MODEL_EVALUATION", "training_seed": 45,
                "evaluation_seed": 9001, "native_process_protocol": "fresh process per role; sorted horizon,input_id",
                "source_bindings": bindings, "source_data_root": str(data_root),
                "records": records, "coverage": observed, "data_files": data_files,
                "data_bytes": sum(r["bytes"] for r in data_files),
                "validation_aliases": aliases,
                "historical_supervision_filter": {"taxonomy": taxonomy,
                    "scan_has_valid_instance": supervised, "excluded_units": excluded,
                    "rule": "retain if any scan has a valid semantic label and global instance >=0"},
                "role_overlaps": {f"{a}__{b}": sorted(refs[a] & refs[b])
                                  for a, b in (("PB", "LOCAL-T2"), ("PB", "ADDITIONAL"),
                                               ("LOCAL-T2", "ADDITIONAL"))},
                "claim_boundary": "historically used validation populations; not an untouched external test set"}
    path = output / "FORMAL_POPULATION.json"
    freeze_json(path, manifest)
    lock = {"schema": 1, "status": "LOCKED_PENDING_EXECUTION", "rows": plan["rows"],
            "selected_module": plan["selected_module"], "training_seed": 45, "evaluation_seed": 9001,
            "CAL_LOCK_sha256": plan["CAL_LOCK_sha256"], "FORMAL_POPULATION_sha256": sha256(path),
            "source_plan_sha256": sha256(plan_path), "formal_results_used_for_selection": False}
    freeze_json(output / "FORMAL_LOCK.json", lock)
    return {"status": lock["status"], "population": str(path), "coverage": observed,
            "input_count": len(input_ids), "data_files": len(data_files), "data_bytes": manifest["data_bytes"],
            "selected_rows": plan["rows"], "TRAIN_overlap": [],
            "overlap_counts": {k: len(v) for k, v in manifest["role_overlaps"].items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps(materialize(args.root, args.project, args.output), ensure_ascii=False))


if __name__ == "__main__":
    main()
