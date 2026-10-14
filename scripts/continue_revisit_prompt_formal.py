"""Run the authorized formal matrix, then the existing locked evaluation campaign."""

import fcntl
import os
from pathlib import Path
import time

from scripts.native_long_budget import write_json_atomic
from scripts.revisit_prompt_assets import local_assets, public_error, read_run_plan
from scripts.revisit_prompt_campaign import (
    child_command, ensure_job, evaluate_endpoint, run_campaign,
)


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
