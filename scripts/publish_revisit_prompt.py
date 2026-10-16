"""Publish MPR commits A/B, selected LFS bytes, and an external verification receipt."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from scripts.native_long_budget import ledger_events, merge_events, write_json_atomic
from scripts.revisit_prompt_assets import ARTIFACTS, PROJECT, NUMERIC_FILES, local_assets, read_json, sha256, public_error

BRANCH = "research/persist4d-mpr-v1-20261013"
TAG = "persist4d-mpr-v1-20261013-v1"
B_FILES = {"HANDOFF.md","COMMANDS.md","PROMPT_AUDIT.md","PUBLICATION_RECEIPT.json"}


def git(*args, cwd=PROJECT, env=None):
    return subprocess.check_output(["git",*args],cwd=cwd,text=True,
        stderr=subprocess.STDOUT,env={**os.environ,**(env or {})}).strip()


def phase_a_files():
    files = {PROJECT/name for name in NUMERIC_FILES}
    files.update(PROJECT/name for name in (
        "scripts/publish_revisit_prompt.py","scripts/revisit_prompt_cpu_statistics.py",
        "scripts/continue_revisit_prompt_statistics.py",
        "scripts/revisit_prompt_pool_cache.py",
        "scripts/revisit_prompt_cooccurrence.py",
        "tests/test_revisit_prompt.py",
        "tests/revisit_prompt_selection_cases.py",
        "docs/superpowers/specs/2026-10-13-revisit-prompt-design.md",
        "docs/superpowers/plans/2026-10-13-revisit-prompt.md"))
    files.update(path for path in ARTIFACTS.rglob("*") if path.is_file()
                 and path.name not in B_FILES and path.name != "ARTIFACT_MANIFEST.json")
    identity = read_json(ARTIFACTS/"STATUS.json").get("evaluation_policy")
    if identity:
        locks = [path for path in ARTIFACTS.glob("EVALUATION_POLICY*.json")
                 if sha256(path) == identity["policy_sha256"]]
        if len(locks) != 1:
            raise ValueError("active evaluation policy lock identity differs")
        policy = read_json(locks[0])
        if policy["policy_id"] != identity["policy_id"]:
            raise ValueError("active evaluation policy identity differs")
        for name, expected in policy["policy_sources"].items():
            path = PROJECT/name
            if not path.resolve().is_relative_to(PROJECT.resolve()) or sha256(path) != expected:
                raise ValueError("evaluation policy source identity differs: "+name)
            files.add(path)
    files.update(PROJECT.glob("tests/test_revisit_prompt*.py"))
    return sorted(files)


def check_assets(manifest):
    rows = manifest.get("assets",[])
    for row in rows:
        path = PROJECT/row["path"]
        if not path.resolve().is_relative_to(ARTIFACTS.resolve()):
            raise ValueError("new publication asset must belong to the MPR public directory")
        if not path.is_file() or path.stat().st_size != row["bytes"] or sha256(path) != row["sha256"]:
            raise ValueError("publication asset bytes differ: "+row["path"])
    return rows


def write_handoff(assets, *, commit_a, status, models, branch=BRANCH, tag=TAG):
    events = merge_events([ledger_events(Path(assets["run_root"])/"COST_LEDGER.jsonl")])
    new = sum(event["gpu_hours"] for event in events)
    total = assets["budget"]["prior_gpu_hours"]+new
    selection = read_json(ARTIFACTS/"CAL_SELECTION_LOCK.json") if (ARTIFACTS/"CAL_SELECTION_LOCK.json").exists() else {}
    replication = read_json(ARTIFACTS/"REPLICATION_LOCK.json") if (ARTIFACTS/"REPLICATION_LOCK.json").exists() else {}
    policy_id = status.get("evaluation_policy",{}).get("policy_id")
    policy_modules = {
        "stable-mean-v1":("scripts.revisit_prompt_evaluation_policy","scripts.continue_revisit_prompt_policy"),
        "native-stable-mean-v2":("scripts.revisit_prompt_native_policy","scripts.continue_revisit_prompt_native"),
        "ordered-sparse-reference-v4":("scripts.revisit_prompt_sparse_policy","scripts.continue_revisit_prompt_sparse"),
    }
    if policy_id:
        policy_module, resume_module = policy_modules[policy_id]
        summary_command = "\n".join([
            "python - <<'PY'",
            "from scripts.revisit_prompt_assets import local_assets",
            f"from {policy_module} import campaign_policy, load_policy",
            "from scripts.revisit_prompt_campaign import summarize",
            "assets = local_assets()",
            "with campaign_policy(assets, load_policy(assets)) as scoped:",
            "    summarize(scoped)",
            "PY",
        ])
        resume_command = f"python -m {resume_module}"
    else:
        summary_command = "python -m scripts.revisit_prompt_campaign summarize"
        resume_command = "python -m scripts.revisit_prompt_campaign run --resume"
    publish_command = f"python -m scripts.publish_revisit_prompt --execute --branch {branch} --tag {tag}"
    commands = [
        "# MPR-V1 实际命令与复现入口", "",
        "在原仓库环境中绑定合法取得的数据、固定父包及编码器。已冻结的选点和结果保持原身份；以下汇总命令只读取对应评测命名空间。", "",
        "```sh", "python -m scripts.revisit_prompt_campaign resolve --auto",
        "python -m scripts.revisit_prompt_campaign plan --dry-run", "```", "",
        "尚未完成的工作使用原任务恢复入口；只恢复身份一致的项目，不重新训练已完成轨迹：", "",
        "```sh", resume_command, "```", "",
        "冻结发布前从实际状态汇总；已发布 A/B 的产物不原地改写：", "",
        "```sh", summary_command, "```", "",
        "真实发布使用本轮独立引用：", "", "```sh", publish_command, "```", "",
        "父包实体取得：", "", "```sh",
        "git lfs pull --include=artifacts/native_long_postselection_v1/deployment/E1-23760.pt --exclude=", "```", "",
        "完整的依赖取得、严格加载和官方状态离线复算命令见 [DEPLOYMENT_PROTOCOL.md](DEPLOYMENT_PROTOCOL.md)，固定来源和哈希见 [DEPENDENCY_ACQUISITION.json](DEPENDENCY_ACQUISITION.json)。加载和前向均须保持其中的 V4 推理上下文。",
        "模型文件与实际 optimizer 恢复文件分别见 MODEL_MANIFEST.json；delta 不含 optimizer，最终 resume 的进度为原 N 边界。官方状态离线复算不运行 GPU 推理。", "",
    ]
    (ARTIFACTS/"COMMANDS.md").write_text("\n".join(commands))
    endpoints = {name:selection.get(name) for name in
                 ("control_star","candidate_star","candidate_descriptive_star")}
    selected = {name:{key:value.get(key) for key in ("variant","adapter_seed","step")}
                for name,value in selection.get("selected_by_variant",{}).items() if value}
    lines = [
        "# MPR-V1 实际交接", "",
        "本提交发布状态：PENDING_EXTERNAL_VERIFICATION。发布事实以该 tag 的 Release 外部回执为准；上传失败时回执保存在运行根 publication/PUBLICATION_RECEIPT.json，状态与错误如实记录。", "",
        "任务：固定 E1-23760 seed45，比较首掩码前历史检索与 M0–M6 强对照。",
        "固定 base：6b1b779172afdeb51e2719a737cd05e7eeeca982。",
        f"分支：`{branch}`；tag：`{tag}`；冻结实验提交 A：`{commit_a}`。", "",
        f"代码：{status.get('code_status')}；实验：{status.get('experiment_status')}；科学：{status.get('scientific_status')}。",
        f"正式 N：{status.get('formal_N')}；选模状态：{selection.get('status','NOT_EXECUTED')}；seed147：{replication.get('status','NOT_EXECUTED')}。",
        f"复验未执行原因：{json.dumps(replication.get('reason',[]),ensure_ascii=False)}。",
        f"评测政策：{json.dumps(status.get('evaluation_policy'),sort_keys=True)}。",
        f"实际逐变体最佳点：{json.dumps(selected,sort_keys=True)}。",
        f"实际锁定端点：{json.dumps(endpoints,sort_keys=True)}。", "",
        "选点只使用完整 CAL 四点、三次独立进程，按 S_long、S_current、T2、较少参数和变体名排序；家族内最后以较早步数打破并列。SEL/PB 不改模型或阈值。seed146/147 适配同一个 seed45 父模型，不是两个独立父训练。", "",
        f"新增 GPUh：{new:.9f}；累计：{total:.9f}；累计余额：{assets['budget']['lifetime_cap_gpu_hours']-total:.9f}；保留预留：{assets['budget']['reserve_gpu_hours']:.9f}。",
        "发布准备要求本任务 ACTIVE_JOB.json 不存在；最终活动进程检查、各阶段完成范围和部署回载证据见 STATUS.json 与 PROMPT_AUDIT.md。", "",
        f"父包：artifacts/native_long_postselection_v1/deployment/E1-23760.pt，SHA256 `{assets['hashes']['parent']}`。",
        f"编码器 SHA256 `{assets['hashes']['encoder']}`；按原 Concerto 许可单独取得，未重新发布。",
        "3RScan 按原许可取得，使用仓库 README 的预处理和 resolver；固定父包 Git/LFS、编码器及数据来源见 [DEPENDENCY_ACQUISITION.json](DEPENDENCY_ACQUISITION.json)。原始点云、逐点 GT、编码器和本地绝对路径不在本次公开资产中。", "",
        f"实际模型及指标包数量：{len(models)}；每项实体字节、SHA256、变体、种子、步数和 optimizer 标记见 MODEL_MANIFEST.json。",
        "部署必须同时具备固定父包、指定编码器和 delta；严格加载、V4 推理上下文及离线复算命令见 [DEPLOYMENT_PROTOCOL.md](DEPLOYMENT_PROTOCOL.md)。optimizer 恢复使用 manifest 实际列出的 resume 文件与原 RUN_LOCK，其进度为原 N 边界。",
        "evidence/official_states.zip 保存可重新池化的官方充分状态。TRAINING/CAL_ALL/SEL_LOCKED/PB_LOCKED/BY_REFERENCE/IDENTITY_EVENTS_SUMMARY 保留原始行；机制与资源见 MECHANISM/PROFILE，实际命令见 COMMANDS.md。", "",
        "REPORT.md 与 PROMPT_AUDIT.md 记录所有失败、分母、历史面板暴露、数值根因限制以及 H1–H3 支持范围。有限重复一致不证明全局逐张量确定性；内部面板结果不等于 ReScene 原论文完整测试指标，也不自动表示 CVPR 投稿就绪。", "",
        "ARTIFACT_MANIFEST.json 冻结 A 中的代码、配置和实验资产，排除自身、B 交接及外部回执。HANDOFF 只引用 A，不嵌入自己的提交 SHA。", "",
        "唯一后续判断："+status.get("next_direction", "以 REPORT 的实际结论为准；不追加未预先约定的涨点搜索。"), "",
    ]
    (ARTIFACTS/"HANDOFF.md").write_text("\n".join(lines))


def publish(assets, *, execute, branch=BRANCH, tag=TAG):
    root = Path(assets["run_root"])/"publication"
    root.mkdir(parents=True,exist_ok=True)
    status_path = ARTIFACTS/"STATUS.json"
    status = read_json(status_path)
    missing = [name for name in ("REPORT.md","MODEL_MANIFEST.json","PROMPT_AUDIT.md")
               if not (ARTIFACTS/name).is_file()]
    preview = dict(status="DRY_RUN_NOT_PUBLISHED",branch=branch,tag=tag,
                   publication_ready=status.get("publication_ready",False),missing=missing)
    if not execute:
        return preview
    if missing or not status.get("publication_ready") or not status.get("prompt_audit_execution_complete"):
        raise ValueError("publication requires the completed primary audit, report and asset manifest")
    if git("branch","--show-current") != branch:
        raise ValueError("publication must use the bound new research branch")
    origin = git("remote","get-url","origin")
    if origin not in ("git@github.com:Orangekostar/Persist4D.git",
                      "https://github.com/Orangekostar/Persist4D.git",
                      "https://github.com/Orangekostar/Persist4D",
                      "ssh://git@github.com/Orangekostar/Persist4D.git"):
        raise ValueError("publication origin must be Orangekostar/Persist4D")
    if (Path(assets["run_root"])/"ACTIVE_JOB.json").exists():
        raise ValueError("MPR GPU accounting reservation is still active")
    rows = check_assets(read_json(ARTIFACTS/"MODEL_MANIFEST.json"))
    git("lfs","version")
    pending_path = root/"PUBLICATION_PENDING.json"
    if pending_path.exists():
        pending = read_json(pending_path)
        expected_head = pending.get("commit_b",pending["commit_a"])
        if pending["branch"] != branch or pending["tag"] != tag or git("rev-parse","HEAD") != expected_head:
            raise ValueError("publication retry identity differs from the prepared A/B commits")
    else:
        git("lfs","install","--local")
        status = {**status,"publication_status":"PENDING_EXTERNAL_VERIFICATION"}
        write_json_atomic(status_path,status)
        patterns = ("artifacts/revisit_prompt_v1/deployment/*.pt",
                    "artifacts/revisit_prompt_v1/resume/*.pt",
                    "artifacts/revisit_prompt_v1/evidence/official_states.zip")
        for pattern in patterns:
            git("lfs","track",pattern)
        files = phase_a_files()
        files.append(PROJECT/".gitattributes")
        frozen = [dict(path=path.relative_to(PROJECT).as_posix(),bytes=path.stat().st_size,sha256=sha256(path))
                  for path in sorted(files)]
        manifest = dict(schema=1,status="PENDING_EXTERNAL_VERIFICATION",files=frozen,
            excluded=["ARTIFACT_MANIFEST.json",*sorted(B_FILES)],
            scope="frozen experiment/code/model assets in commit A; B and external receipt are separate")
        write_json_atomic(ARTIFACTS/"ARTIFACT_MANIFEST.json",manifest)
        git("add","-f","--",*[p.relative_to(PROJECT).as_posix() for p in files],
            "artifacts/revisit_prompt_v1/ARTIFACT_MANIFEST.json")
        git("diff","--cached","--check")
        git("commit","-m","Publish frozen MPR-V1 experiments and bound assets")
        commit_a = git("rev-parse","HEAD")
        pending = dict(branch=branch,tag=tag,commit_a=commit_a)
        write_json_atomic(pending_path,pending)
    if "commit_b" not in pending:
        write_handoff(assets,commit_a=pending["commit_a"],status=status,models=rows,branch=branch,tag=tag)
        git("add","--",*[f"artifacts/revisit_prompt_v1/{name}" for name in
                           ("HANDOFF.md","COMMANDS.md","PROMPT_AUDIT.md")])
        git("commit","-m","Add MPR-V1 final handoff and execution audit")
        commit_b = git("rev-parse","HEAD")
        pending["commit_b"] = commit_b
        write_json_atomic(pending_path,pending)
    tags = git("tag","--list",tag)
    if not tags:
        git("tag",tag,pending["commit_b"])
    elif git("rev-parse",tag) != pending["commit_b"]:
        raise ValueError("existing publication tag points to a different commit")
    receipt = {**pending,"status":"PUSH_FAILED","assets":rows}
    try:
        for row in rows:
            git("lfs","push","--object-id","origin",row["sha256"])
        git("push","origin",branch,f"refs/tags/{tag}",env={"GIT_LFS_SKIP_PUSH":"1"})
        remote = git("ls-remote","origin",f"refs/heads/{branch}",f"refs/tags/{tag}")
        references = {line.split()[1]:line.split()[0] for line in remote.splitlines()}
        if any(references.get(ref) != pending["commit_b"] for ref in (f"refs/heads/{branch}",f"refs/tags/{tag}")):
            raise ValueError("remote branch/tag do not point to commit B")
        with tempfile.TemporaryDirectory(prefix="mpr-remote-") as temporary:
            checkout = Path(temporary)
            git("init",cwd=checkout)
            git("remote","add","origin",git("remote","get-url","origin"),cwd=checkout)
            git("fetch","--depth=1","origin",branch,cwd=checkout,env={"GIT_LFS_SKIP_SMUDGE":"1"})
            git("checkout","--detach",pending["commit_b"],cwd=checkout,env={"GIT_LFS_SKIP_SMUDGE":"1"})
            if rows:
                storage = str(checkout/"fresh-lfs-cache")
                git("-c",f"lfs.storage={storage}","lfs","fetch","origin",pending["commit_b"],
                    "--include="+",".join(row["path"] for row in rows),"--exclude=",cwd=checkout)
                git("-c",f"lfs.storage={storage}","lfs","checkout",*[row["path"] for row in rows],cwd=checkout)
            manifest = read_json(checkout/"artifacts/revisit_prompt_v1/ARTIFACT_MANIFEST.json")
            for row in manifest["files"]:
                path = checkout/row["path"]
                if path.stat().st_size != row["bytes"] or sha256(path) != row["sha256"]:
                    raise ValueError("fresh remote asset differs: "+row["path"])
            for name in ("REPORT.md","HANDOFF.md","COMMANDS.md","PROMPT_AUDIT.md"):
                if sha256(checkout/"artifacts/revisit_prompt_v1"/name) != sha256(ARTIFACTS/name):
                    raise ValueError("remote handoff/report bytes differ")
        receipt.update(status="CODE_AND_LFS_PUSH_VERIFIED_RELEASE_RECEIPT_LOCAL_ONLY",
            remote_references=references,remote_byte_verification="FRESH_EMPTY_LFS_CACHE_ALL_A_ASSETS_AND_HANDOFF",
            verified_utc=datetime.now(timezone.utc).isoformat())
        write_json_atomic(root/"PUBLICATION_RECEIPT.json",receipt)
        if shutil.which("gh") is None:
            capability = dict(gh_cli_available=False,
                environment_api_token_available=any(bool(os.environ.get(key)) for key in ("GH_TOKEN","GITHUB_TOKEN")),
                noninteractive_git_credential_available=False)
            try:
                check = subprocess.run(["git","credential","fill"],
                    input="protocol=https\nhost=github.com\n\n",cwd=PROJECT,
                    env={**os.environ,"GIT_TERMINAL_PROMPT":"0","GIT_ASKPASS":"false","SSH_ASKPASS":"false"},
                    capture_output=True,text=True,timeout=10)
                credentials = dict(line.split("=",1) for line in check.stdout.splitlines() if "=" in line)
                capability["noninteractive_git_credential_available"] = bool(credentials.get("password"))
                capability["credential_helper_exit_code"] = check.returncode
            except subprocess.TimeoutExpired:
                capability["credential_helper_status"] = "TIMEOUT"
            receipt["release_capability"] = capability
            receipt["release_error"] = (
                "gh CLI unavailable and no API credential found in the environment or noninteractive git helper; Release receipt remains local"
                if not capability["environment_api_token_available"] and not capability["noninteractive_git_credential_available"]
                else "gh CLI unavailable; API authentication is present but Release upload has not been performed")
        else:
            try:
                subprocess.run(["gh","release","create",tag,"--title","Persist4D MPR-V1",
                    "--notes-file",str(ARTIFACTS/"REPORT.md"),str(root/"PUBLICATION_RECEIPT.json")],
                    cwd=PROJECT,check=True,capture_output=True,text=True)
                receipt["status"] = "CODE_AND_LFS_PUSH_VERIFIED_RELEASE_RECEIPT_UPLOADED"
                write_json_atomic(root/"PUBLICATION_RECEIPT.json",receipt)
                subprocess.run(["gh","release","upload",tag,str(root/"PUBLICATION_RECEIPT.json"),"--clobber"],
                    cwd=PROJECT,check=True,capture_output=True,text=True)
            except subprocess.CalledProcessError as error:
                receipt.update(status="CODE_AND_LFS_PUSH_VERIFIED_RELEASE_RECEIPT_LOCAL_ONLY",
                    release_error=public_error(Exception(error.stderr),assets))
    except Exception as error:
        receipt.update(status="PUSH_FAILED",error_type=type(error).__name__,error=public_error(error,assets))
    write_json_atomic(root/"PUBLICATION_RECEIPT.json",receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("--dry-run",action="store_true")
    parser.add_argument("--branch",default=BRANCH)
    parser.add_argument("--tag",default=TAG)
    args = parser.parse_args(argv)
    assets = None
    try:
        assets = local_assets()
        result = publish(assets,execute=args.execute and not args.dry_run,branch=args.branch,tag=args.tag)
        print(json.dumps(result,sort_keys=True,allow_nan=False))
        return 1 if result["status"] == "PUSH_FAILED" else 0
    except Exception as error:
        write_json_atomic(ARTIFACTS/"PUBLISH_ENTRY_STATUS.json",dict(status="FAILED",error_type=type(error).__name__,
            error=public_error(error,assets)))
        print(f"{type(error).__name__}: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
