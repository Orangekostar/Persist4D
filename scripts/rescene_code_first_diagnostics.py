"""Read-only original locked-Q score audit on the immutable baseline cache."""

from collections import defaultdict
from pathlib import Path

import torch
from torch.nn import functional as F

from models.short_module_heads import build_head, quality_inputs
from scripts.short_module_screen import read_json, sha256, write_json

PROJECT = Path(__file__).resolve().parents[1]
ROOT = Path("/home/ww/persist4d_runs/short_module_screen_v1")
PUBLIC = PROJECT / "artifacts/rescene_code_first_audit_v1"


def main():
    torch.set_num_threads(2)
    index = read_json(ROOT / "EXPORT_INDEX.json")
    cache = Path(index["cache"])
    lock = read_json(PROJECT / "artifacts/short_module_screen_v1/selection/CAL_LOCK.json")
    heads, identities = {}, {}
    for arm in ("Q1", "Q2", "Q3"):
        step = lock["selected_updates"][arm]
        path = ROOT / "training" / arm / "seed45" / f"update={step:04d}.pt"
        saved = torch.load(path, map_location="cpu", weights_only=False)
        meta = saved["metadata"]
        head = build_head(arm, **meta["dimensions"], thresholds=tuple(meta["thresholds"]), seed=45)
        head.load_state_dict(saved["head"], strict=True)
        heads[arm] = head.eval()
        identities[arm] = {"path": str(path), "sha256": sha256(path), "step": step}
    rows = defaultdict(lambda: {"count": 0, "score_raised": 0, "parent_score_sum": 0.,
                                "head_score_sum": 0., "head_score_above_0_5": 0})
    with torch.inference_mode():
        for number, entry in enumerate(index["entries"]):
            for field in ("prediction", "targets"):
                if sha256(cache / entry[field]["file"]) != entry[field]["sha256"]:
                    raise ValueError("historical shard content mismatch")
            saved = torch.load(cache / entry["prediction"]["file"], map_location="cpu", weights_only=False)
            labels = torch.load(cache / entry["targets"]["file"], map_location="cpu", weights_only=False)["quality"]
            soft, scores, desc = saved["parent"]["soft"], saved["parent"]["scores"], saved["descriptor"]
            x = quality_inputs(soft["query_features"], desc["h"], soft["class_probabilities"],
                               F.one_hot(soft["source_class_ids"], saved["dimensions"]["classes"]).float(),
                               scores, desc["support"], desc["foreground_probability"])
            categories = [(status, bool(status == "IGNORE" and fraction is not None and .5 < fraction <= .9))
                          for status, fraction in zip(labels["status"], labels["ignore_proportion"])]
            for arm, head in heads.items():
                output = head(x)
                for i, (status, affected) in enumerate(categories):
                    row = rows[entry["record"]["role"], entry["record"]["horizon"], arm, status, affected]
                    row["count"] += 1
                    row["score_raised"] += int(output[i] > scores[i])
                    row["parent_score_sum"] += float(scores[i])
                    row["head_score_sum"] += float(output[i])
                    row["head_score_above_0_5"] += int(output[i] > .5)
            if (number + 1) % 100 == 0:
                print(f"Original locked-Q score audit {number + 1}/{len(index['entries'])}", flush=True)
    result = {"status": "COMPLETE", "heads": identities,
              "historical_index_sha256": sha256(ROOT / "EXPORT_INDEX.json"),
              "scope": "score uplift cross-tab; not actual ranked-FP or AP causal attribution",
              "rows": [{"role": role, "H": h, "arm": arm, "original_status": status,
                        "cross_threshold_gap": affected, **values,
                        "mean_parent_score": values["parent_score_sum"] / values["count"],
                        "mean_head_score": values["head_score_sum"] / values["count"]}
                       for (role, h, arm, status, affected), values in sorted(rows.items())]}
    write_json(PUBLIC / "ORIGINAL_Q_SCORE_CROSSTAB.json", result)
    print("Score audit complete", flush=True)


if __name__ == "__main__":
    main()
