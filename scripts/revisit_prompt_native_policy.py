"""Bind Native sparse kernels and stable mean separately from immutable C3 training."""
from contextlib import contextmanager
from pathlib import Path

from models.revisit_prompt_native_numerics import runtime_identity
from scripts.native_long_budget import write_json_atomic
from scripts.revisit_prompt_assets import (
    ARTIFACTS, PROJECT, numeric_sources, read_json, sha256,
)

POLICY_ID = "native-stable-mean-v2"
NAMESPACE = "policies/native-stable-mean-v2"
LOCK_NAME = "EVALUATION_POLICY_V2.json"
POLICY_FILES = (
    "models/revisit_prompt_native_numerics.py",
    "scripts/revisit_prompt_native_policy.py",
    "scripts/evaluate_revisit_prompt_native.py",
    "scripts/continue_revisit_prompt_native.py",
    "models/revisit_prompt_numerics.py",
    "scripts/revisit_prompt_evaluation_policy.py",
    "scripts/evaluate_revisit_prompt_policy.py",
    "scripts/continue_revisit_prompt_policy.py",
    "scripts/continue_revisit_prompt_formal.py",
    "models/scatter.py",
    "models/pointcept.py",
    "datasets/native_long_dataset.py",
    "datasets/pointcept_utils.py",
)


def load_policy(assets):
    root = Path(assets["run_root"])
    private, public = root / LOCK_NAME, ARTIFACTS / LOCK_NAME
    if not private.is_file() or not public.is_file() or sha256(private) != sha256(public):
        raise ValueError("identical runtime/public evaluation policy locks are required")
    policy = read_json(private)
    if (policy.get("status") != "LOCKED" or policy.get("policy_id") != POLICY_ID
            or policy.get("namespace") != NAMESPACE):
        raise ValueError("evaluation policy status or namespace differs")
    if policy.get("training_numeric_sources") != numeric_sources():
        raise ValueError("training numerical source identity differs; checkpoints stay immutable")
    if policy.get("training_run_lock_sha256") != sha256(root / "RUN_LOCK.json"):
        raise ValueError("original training RUN_LOCK identity differs")
    if policy.get("policy_sources") != {name:sha256(PROJECT / name) for name in POLICY_FILES}:
        raise ValueError("evaluation policy source identity differs")
    if policy.get("inference_runtime") != runtime_identity():
        raise ValueError("inference runtime differs from the frozen Native execution policy")
    return policy


def policy_identity(assets, policy):
    return dict(policy_id=policy["policy_id"], policy_sha256=sha256(Path(assets["run_root"]) / LOCK_NAME))


def evaluation_root(assets, policy):
    if policy["namespace"] != NAMESPACE:
        raise ValueError("evaluation namespace differs")
    return Path(assets["run_root"]) / NAMESPACE


def prepare_namespace(assets, policy):
    root, namespace = Path(assets["run_root"]), evaluation_root(assets, policy)
    namespace.mkdir(parents=True, exist_ok=True)
    for name in ("RUN_LOCK.json", "COST_LEDGER.jsonl", "trajectories", "preflight"):
        source, target = root / name, namespace / name
        if not source.exists():
            continue
        if target.is_symlink():
            if target.resolve() != source.resolve():
                raise ValueError("policy namespace link differs from the owned original run")
        elif target.exists():
            raise ValueError("policy namespace must not copy or replace original assets")
        else:
            target.symlink_to(source, target_is_directory=source.is_dir())
    # P5 outputs belong to this policy; only the previously fixed panel is reused.
    p5 = namespace / "p5"
    if p5.is_symlink():
        raise ValueError("Native P5 outputs must use their separate policy namespace")
    p5.mkdir(exist_ok=True)
    original_panel, panel = root / "p5/PANEL.json", p5 / "PANEL.json"
    if original_panel.is_file():
        if panel.exists() and panel.read_bytes() != original_panel.read_bytes():
            raise ValueError("fixed original P5 panel identity differs")
        if not panel.exists():
            panel.write_bytes(original_panel.read_bytes())
    return {**assets, "run_root":str(namespace)}


def validate_result(result, identity, policy):
    if (result.get("evaluation_policy") != identity
            or result.get("request", {}).get("evaluation_policy") != identity):
        raise ValueError("evaluation policy identity differs; original results cannot be reused")
    if result["request"].get("numeric_sources") != policy["training_numeric_sources"]:
        raise ValueError("evaluation training numerical source identity differs")


def annotated_writer(writer, identity, *, all_outputs=False):
    def write(path, payload):
        if (all_outputs or Path(path).name in {"STATUS.json", "RESULTS.json", "EVAL_ENTRY_STATUS.json"}) and isinstance(payload, dict):
            payload = {**payload, "evaluation_policy":identity}
            if "request" in payload:
                payload["request"] = {**payload["request"], "evaluation_policy":identity}
        return writer(path, payload)
    return write


def validate_namespace(scoped_assets, identity, policy):
    for path in sorted((Path(scoped_assets["run_root"]) / "evaluation").glob("*/*/repeat*/RESULTS.json")):
        result = read_json(path)
        validate_result(result, identity, policy)
        status = read_json(path.with_name("STATUS.json"))
        if (result.get("status") != "COMPLETE" or status.get("status") != "COMPLETE"
                or status.get("evaluation_policy") != identity):
            raise ValueError("policy evaluation lacks its annotated complete status")


@contextmanager
def campaign_policy(assets, policy):
    """Route the existing scientific campaign through one audited namespace."""
    from scripts import revisit_prompt_campaign as campaign
    from scripts import evaluate_revisit_prompt as evaluation
    from scripts import select_revisit_prompt as selector

    scoped = prepare_namespace(assets, policy)
    identity = policy_identity(assets, policy)
    saved = dict(charged_job=campaign.charged_job, ensure_job=campaign.ensure_job,
                 child_command=campaign.child_command, collect_evaluations=campaign.collect_evaluations,
                 bootstrap_locked=evaluation.bootstrap_locked, select_cal_rows=selector.select_cal_rows)
    prefix = "policy:" + POLICY_ID + ":"

    def unchanged_policy():
        current = load_policy(assets)
        if policy_identity(assets, current) != identity:
            raise ValueError("evaluation policy lock changed during the owned campaign")

    def charged(root, command, **kwargs):
        unchanged_policy()
        if Path(root).resolve() != Path(scoped["run_root"]).resolve():
            raise ValueError("policy child must use its owned namespace")
        if not kwargs["job_key"].startswith(prefix):
            raise ValueError("policy job journal identity is missing")
        return saved["charged_job"](assets["run_root"], command, **kwargs)

    def ensure(current_assets, command, **kwargs):
        kwargs["job_key"] = prefix + kwargs["job_key"]
        # Completed training is reused with the original exact numerical checks.
        # Completed evaluations additionally require this execution policy.
        if not kwargs.get("training", False) and Path(kwargs["status_path"]).is_file():
            old = read_json(kwargs["status_path"])
            if old.get("status") == kwargs["complete_status"] and old.get("evaluation_policy") != identity:
                raise ValueError("completed child evaluation policy identity differs")
        result = saved["ensure_job"](current_assets, command, **kwargs)
        if not kwargs.get("training", False) and result.get("evaluation_policy") != identity:
            raise ValueError("child did not record its required evaluation policy identity")
        return result

    def child(current_assets, module, *args):
        if module == "scripts.evaluate_revisit_prompt":
            module = "scripts.evaluate_revisit_prompt_native"
        elif module == "scripts.revisit_prompt_campaign" and args and args[0] == "p5":
            module = "scripts.evaluate_revisit_prompt_native"
        return saved["child_command"](current_assets, module, *args)

    def collect(current_assets):
        unchanged_policy()
        validate_namespace(current_assets, identity, policy)
        return saved["collect_evaluations"](current_assets)

    def select(rows, run_lock, **kwargs):
        unchanged_policy()
        validate_namespace(scoped, identity, policy)
        result = saved["select_cal_rows"](rows, run_lock, **kwargs)
        result["evaluation_policy"] = identity
        return result

    def bootstrap(current_assets, **kwargs):
        unchanged_policy()
        validate_namespace(current_assets, identity, policy)
        return saved["bootstrap_locked"](current_assets, **kwargs)

    campaign.charged_job, campaign.ensure_job = charged, ensure
    campaign.child_command, campaign.collect_evaluations = child, collect
    evaluation.bootstrap_locked, selector.select_cal_rows = bootstrap, select
    try:
        yield scoped
    finally:
        for name in ("charged_job", "ensure_job", "child_command", "collect_evaluations"):
            setattr(campaign, name, saved[name])
        evaluation.bootstrap_locked, selector.select_cal_rows = saved["bootstrap_locked"], saved["select_cal_rows"]
