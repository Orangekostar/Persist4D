"""Reuse C3 training, freshly admit policy baselines, then continue the matrix."""
import fcntl
import os
from pathlib import Path
import time

from scripts.native_long_budget import write_json_atomic
from scripts.revisit_prompt_assets import (
    ARTIFACTS, local_assets, numeric_sources, public_error, read_json, read_run_plan, sha256,
)
from scripts.revisit_prompt_evaluation_policy import (
    campaign_policy, load_policy, policy_identity, validate_namespace,
)


def main():
    assets = local_assets()
    root = Path(assets["run_root"])
    status_path = root / "CONTROLLER_STATUS.json"
    with (root / "CONTROLLER.lock").open("a") as ownership:
        fcntl.flock(ownership.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        policy = load_policy(assets)
        identity = policy_identity(assets, policy)
        plan = read_run_plan(assets)
        status = dict(status="RUNNING", pid=os.getpid(), controller_alive=True,
                      started_unix=time.time(), N=plan["N"], stage="POLICY_BASELINE_ADMISSION",
                      current_variant="M0", evaluation_policy=identity,
                      evaluation_namespace=policy["namespace"])
        write_json_atomic(status_path, status)
        try:
            for name in ("CAL_SELECTION_LOCK.json", "REPLICATION_LOCK.json"):
                path = ARTIFACTS / name
                if path.is_file():
                    old = read_json(path)
                    if old.get("status") == "COMPLETE" and old.get("evaluation_policy") != identity:
                        raise ValueError("completed selection evaluation policy differs")
            # Never retrain a completed main model to adopt an evaluation policy.
            verification = read_json(root / "MAIN_TRAINING_VERIFICATION.json")
            if verification["run_lock_sha256"] != sha256(root / "RUN_LOCK.json"):
                raise ValueError("main training verification does not bind the original RUN_LOCK")
            verified = {row["variant"]:row for row in verification["variants"]}
            for variant in ("M1", "M2", "M3", "M4", "M5", "M6"):
                recorded = read_json(root / "trajectories" / f"{variant}-s146" / "STATUS.json")
                if (recorded["status"] != "COMPLETE" or recorded["step"] != plan["N"]
                        or recorded["identity"]["numeric_sources"] != numeric_sources()
                        or not recorded["frozen_parent_unchanged"]):
                    raise ValueError("policy continuation requires the complete original main trajectories")
                for step in plan["checkpoint_steps"]:
                    checkpoint = root / "trajectories" / f"{variant}-s146" / f"delta-{step:06d}.pt"
                    if sha256(checkpoint) != verified[variant]["checkpoints"][str(step)]["sha256"]:
                        raise ValueError("completed main checkpoint differs from its verified original")
            from scripts import revisit_prompt_campaign as campaign
            from scripts.continue_revisit_prompt_formal import fresh_m0_admission
            with campaign_policy(assets, policy) as scoped:
                campaign.evaluate_endpoint(scoped, dict(variant="M0", adapter_seed=146, step=0), "CAL")
                validate_namespace(scoped, identity, policy)
                admission = fresh_m0_admission(scoped, report_name="EVALUATION_M0_ADMISSION.json")
                status["fresh_m0_admission"] = admission
                if admission["status"] != "ADMITTED":
                    status.update(status=admission["status"], controller_alive=False, updated_unix=time.time())
                    write_json_atomic(status_path, status)
                    return 2
                runtime_path = ARTIFACTS / "RUNTIME_SUMMARY.json"
                runtime = read_json(runtime_path)
                runtime.setdefault("training_preflight_epsilon_run_pp", runtime.get("epsilon_run_pp"))
                runtime.update(epsilon_run_pp=admission["epsilon_run_pp"], P1_admitted=True,
                               evaluation_policy=identity, original_fresh_epsilon_run_pp=2.888917922973633,
                               original_large_outlier_root_cause="NOT_ESTABLISHED",
                               fresh_policy_admission="EVALUATION_M0_ADMISSION.json")
                write_json_atomic(runtime_path, runtime)
                status.update(stage="LOCKED_EVALUATION_AND_SELECTION", current_variant=None,
                              updated_unix=time.time())
                write_json_atomic(status_path, status)
                result = campaign.run_campaign(scoped, resume=True)
                status.update(status=result["experiment_status"], stage="FINISHED", controller_alive=False,
                              scientific_status=result["scientific_status"], updated_unix=time.time())
                write_json_atomic(status_path, status)
        except BaseException as error:
            status.update(status="FAILED", controller_alive=False, error_type=type(error).__name__,
                          error=public_error(error, assets), updated_unix=time.time())
            write_json_atomic(status_path, status)
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
