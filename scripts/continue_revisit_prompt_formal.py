"""Run the authorized formal matrix, then the existing locked evaluation campaign."""

import fcntl
import math
import os
from pathlib import Path
import time

from scripts.native_long_budget import write_json_atomic
from scripts.revisit_prompt_assets import (
    ARTIFACTS, local_assets, numeric_sources, public_error, read_json, read_run_plan, sha256,
)
from scripts.revisit_prompt_campaign import (
    child_command, ensure_job, evaluate_endpoint, evaluation_folder, run_campaign,
)


def fresh_m0_admission(assets, *, report_name="FRESH_M0_ADMISSION.json"):
    """Check fresh results without changing the inherited immutable RUN_LOCK."""
    expected = read_json(ARTIFACTS / "DATA_MANIFEST.json")["counts"]["CAL"]
    endpoint = dict(variant="M0", adapter_seed=146, step=0)
    repeats, evidence, request_identity = [], [], None
    for repeat in range(3):
        folder = evaluation_folder(assets, endpoint, "CAL", repeat)
        status, result = read_json(folder / "STATUS.json"), read_json(folder / "RESULTS.json")
        request = result["request"]
        if (status["status"] != "COMPLETE" or result["status"] != "COMPLETE"
                or status["completed_inputs"] != expected or request["expected_inputs"] != expected
                or request["input_count"] != sum(expected.values())):
            raise ValueError("fresh baseline lacks the complete frozen CAL population")
        if (request["numeric_sources"] != numeric_sources()
                or request["parent_sha256"] != assets["hashes"]["parent"]
                or any(request[k] != v for k, v in endpoint.items())
                or request["role"] != "CAL" or request["repeat_index"] != repeat):
            raise ValueError("fresh baseline numerical or endpoint identity differs")
        identity = {k:v for k,v in request.items() if k != "repeat_index"}
        if request_identity is not None and identity != request_identity:
            raise ValueError("fresh baseline requests differ beyond repeat index")
        request_identity = identity
        rows = result["rows"]
        if len(rows) != 5 or {row["horizon"] for row in rows} != set(range(1, 6)):
            raise ValueError("fresh baseline lacks the complete frozen CAL population")
        aps = {}
        for row in rows:
            h = str(row["horizon"])
            if row["expected_inputs"] != expected[h] or row["completed_inputs"] != expected[h]:
                raise ValueError("fresh baseline lacks the complete frozen CAL population")
            ap = row["AP"]
            if ap is None or not math.isfinite(ap):
                raise ValueError("fresh baseline AP is undefined or nonfinite")
            aps[h] = ap
        repeats.append(aps)
        evidence.append(dict(repeat_index=repeat, results_sha256=sha256(folder / "RESULTS.json")))
    ranges = {h:100 * (max(r[h] for r in repeats) - min(r[h] for r in repeats)) for h in expected}
    epsilon = max(ranges.values())
    result = dict(status="ADMITTED" if epsilon <= .2 + 1e-12 else "BLOCKED_NUMERICAL_VARIANCE",
                  epsilon_run_pp=epsilon, limit_pp=.2, range_by_horizon_pp=ranges,
                  AP_by_repeat=repeats, evidence=evidence, updated_unix=time.time())
    write_json_atomic(Path(assets["run_root"]) / report_name, result)
    write_json_atomic(ARTIFACTS / report_name, result)
    return result


def main():
    assets = local_assets()
    root = Path(assets["run_root"])
    status_path = root / "CONTROLLER_STATUS.json"
    with (root / "CONTROLLER.lock").open("a") as ownership:
        fcntl.flock(ownership.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = read_run_plan(assets)
        status = dict(status="RUNNING", pid=os.getpid(), started_unix=time.time(),
                      N=plan["N"], stage="FORMAL_MAIN_TRAINING", current_variant=None)
        write_json_atomic(status_path, status)
        try:
            for variant in ("M1", "M2", "M3", "M4", "M5", "M6"):
                status.update(current_variant=variant, updated_unix=time.time())
                write_json_atomic(status_path, status)
                folder = root / "trajectories" / f"{variant}-s146"
                ensure_job(assets, child_command(assets, "scripts.train_revisit_prompt",
                           "--variant", variant, "--seed", 146),
                           status_path=folder / "STATUS.json", phase="P2",
                           job_key=assets.get("dispatch_job_keys", {}).get(variant, f"train:{variant}:s146"),
                           training=True,
                           complete_status="COMPLETE")
            status.update(stage="FRESH_M0_CAL", current_variant="M0", updated_unix=time.time())
            write_json_atomic(status_path, status)
            evaluate_endpoint(assets, dict(variant="M0", adapter_seed=146, step=0), "CAL")
            admission = fresh_m0_admission(assets)
            status["fresh_m0_admission"] = admission
            if admission["status"] != "ADMITTED":
                status.update(status=admission["status"], stage="FRESH_M0_CAL", updated_unix=time.time())
                write_json_atomic(status_path, status)
                return 2
            status.update(stage="LOCKED_EVALUATION_AND_SELECTION", current_variant=None,
                          updated_unix=time.time())
            write_json_atomic(status_path, status)
            result = run_campaign(assets, resume=True)
            status.update(status=result["experiment_status"], stage="FINISHED",
                          scientific_status=result["scientific_status"], updated_unix=time.time())
            write_json_atomic(status_path, status)
        except BaseException as error:
            status.update(status="FAILED", error_type=type(error).__name__,
                          error=public_error(error, assets), updated_unix=time.time())
            write_json_atomic(status_path, status)
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
