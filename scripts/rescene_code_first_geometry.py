"""Separate, conditional M0/M1 control using one expanded assignment per input."""

from pathlib import Path

import torch

from scripts.rescene_code_first_audit import OLD_ROOT, PUBLIC, ROOT
from scripts.short_module_data import low_segment_targets
from scripts.short_module_identity import assert_sources_current, identity_digest
from scripts.short_module_screen import read_json, sha256, write_json
from scripts.system_comparison_inference import unpack_bool_matrix

M_ROOT = ROOT / "expanded_geometry"


def prepare():
    index = read_json(ROOT / "EXPORT_INDEX.json")
    assert_sources_current(index["label_identity"])
    audit = read_json(PUBLIC / "RELABEL_AUDIT.json")
    affected = sum(row["new_quality_eligible"] for row in audit["groups"] if row["role"] == "TRAIN")
    if affected == 0:
        write_json(PUBLIC / "M_CONTROL_STATUS.json", {"status": "NOT_NEEDED", "affected": 0})
        return
    identity = {**index["label_identity"], "geometry_policy": "any-scored-threshold-v2",
                "sources": {**index["label_identity"]["sources"], str(Path(__file__)): sha256(Path(__file__))}}
    digest = identity_digest(identity)
    directory = M_ROOT / "labels" / digest
    write_json(directory / "IDENTITY.json", identity)
    entries, changed_columns = [], 0
    for entry in index["entries"]:
        if entry["record"]["role"] != "TRAIN":
            entries.append(entry)
            continue
        target_path = Path(index["cache"]) / entry["targets"]["file"]
        if sha256(target_path) != entry["targets"]["sha256"]:
            raise ValueError("Q-repair targets changed")
        targets = torch.load(target_path, map_location="cpu", weights_only=False)
        targets["quality"]["geometry_valid"] = targets["quality"]["valid"].clone()
        previous = targets["assignment"]
        targets["legacy_assignment"] = previous
        targets["assignment"] = targets["expanded_assignment"]
        changed = torch.where(previous != targets["assignment"])[0]
        if changed.numel():
            prediction_path = Path(index["cache"]) / entry["prediction"]["file"]
            if sha256(prediction_path) != entry["prediction"]["sha256"]:
                raise ValueError("frozen prediction changed")
            saved = torch.load(prediction_path, map_location="cpu", weights_only=False)
            soft = saved["parent"]["soft"]
            masks = unpack_bool_matrix(targets["target"]["masks"])
            for candidate in changed.tolist():
                entity = targets["assignment"][candidate]
                mask = (masks[targets["target"]["ids"] == entity].any(0) if entity >= 0
                        else torch.zeros_like(targets["semantic_labels"], dtype=torch.bool))
                values, weights = low_segment_targets(
                    low_point2segment=soft["low_point2segment"], voxel_inverse=soft["voxel_inverse"],
                    gt_mask=mask, semantic_labels=targets["semantic_labels"],
                    segment_count=soft["segment_logits"].shape[0])
                if not torch.equal(weights, targets["segment_weights"]):
                    raise ValueError("known-vertex weights changed")
                targets["segment_targets"][:, candidate] = values
            changed_columns += len(changed)
        destination = directory / f"{entry['input_id']}.targets.pt"
        torch.save(targets, destination)
        entries.append({**entry, "targets": {"file": str(destination), "sha256": sha256(destination),
                                              "bytes": destination.stat().st_size}})
    write_json(M_ROOT / "EXPORT_INDEX.json", {**index, "entries": entries,
                                               "label_identity": identity, "label_identity_sha256": digest})
    write_json(M_ROOT / "assets.local.json", read_json(ROOT / "assets.local.json"))
    write_json(PUBLIC / "M_CONTROL_STATUS.json", {"status": "PREPARED", "affected_eligibility": affected,
               "changed_assignment_columns": changed_columns, "label_identity_sha256": digest,
               "arms": ["M0", "M1"], "shared_assignment": True,
               "unchanged": ["parent", "prediction", "sampling", "steps", "loss", "weights", "old_locked_steps"]})


def train(device):
    from scripts.short_module_training import load_training_records, train_arm

    index = read_json(M_ROOT / "EXPORT_INDEX.json")
    assert_sources_current(index["label_identity"])
    records = load_training_records(M_ROOT, index)
    results = {}
    for arm in ("M0", "M1"):
        results[arm] = train_arm(arm, seed=45, updates=1500, root=M_ROOT, device=device, training_records=records)
        write_json(PUBLIC / "M_TRAINING_STATUS.json", results)
    return results


def evaluate(device):
    from scripts.short_module_evaluation import evaluate_point

    lock = read_json(PUBLIC / "selection/CAL_LOCK.json")["selected_updates"]
    results = []
    for role in ("CAL", "SEL"):
        for arm in ("M0", "M1"):
            old = evaluate_point(ROOT, module=arm, step=lock[arm], seed=45, role=role,
                                 device=device, checkpoint_root=OLD_ROOT)
            new = evaluate_point(ROOT, module=arm, step=lock[arm], seed=45, role=role,
                                 device=device, checkpoint_root=M_ROOT)
            results.append({"role": role, "arm": arm, "locked_step": lock[arm], "original": old,
                            "expanded_eligibility": new,
                            "delta": {h: new["metrics"][h] - old["metrics"][h] for h in ("T1", "T2")}})
            write_json(PUBLIC / "M_MATCHED_COMPARISON.json", results)
    return results


if __name__ == "__main__":
    torch.set_num_threads(2)
    prepare()
