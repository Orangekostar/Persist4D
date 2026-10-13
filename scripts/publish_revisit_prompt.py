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
        "scripts/publish_revisit_prompt.py","tests/test_revisit_prompt.py",
        "tests/revisit_prompt_selection_cases.py",
        "docs/superpowers/specs/2026-10-13-revisit-prompt-design.md",
        "docs/superpowers/plans/2026-10-13-revisit-prompt.md"))
    files.update(path for path in ARTIFACTS.rglob("*") if path.is_file()
                 and path.name not in B_FILES and path.name != "ARTIFACT_MANIFEST.json")
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


def write_handoff(assets, *, commit_a, status, models):
    events = merge_events([ledger_events(Path(assets["run_root"])/"COST_LEDGER.jsonl")])
    new = sum(event["gpu_hours"] for event in events)
    total = assets["budget"]["prior_gpu_hours"]+new
    selection = read_json(ARTIFACTS/"CAL_SELECTION_LOCK.json") if (ARTIFACTS/"CAL_SELECTION_LOCK.json").exists() else {}
    commands = [
        "python -m scripts.revisit_prompt_campaign resolve --auto",
        "python -m scripts.revisit_prompt_campaign preflight",
        "python -m scripts.revisit_prompt_campaign plan",
        "python -m scripts.revisit_prompt_campaign run --resume",
        "python -m scripts.revisit_prompt_campaign summarize",
        "python -m scripts.publish_revisit_prompt --execute",
    ]
    (ARTIFACTS/"COMMANDS.md").write_text("# MPR-V1 commands\n\nRun in the bound original environment.\n\n```sh\n"+
        "\n".join(commands)+"\n```\n\nThe resolver binds local licensed data and the original encoder; these assets are not redistributed.\n")
    text = f"""# MPR-V1 final handoff

Publication state in this commit: PENDING_EXTERNAL_VERIFICATION.
The external receipt is attached to the tag Release when available, otherwise
retained in the bound run root's `publication/PUBLICATION_RECEIPT.json`.

- Task: fixed E1-23760 seed45 historical retrieval prompts and M0–M6 controls.
- Base: `6b1b779172afdeb51e2719a737cd05e7eeeca982`.
- Branch: `{BRANCH}`; tag: `{TAG}`; frozen experiment commit A: `{commit_a}`.
- Code: `{status.get('code_status')}`; experiment: `{status.get('experiment_status')}`;
  science: `{status.get('scientific_status')}`.
- New GPU hours: {new:.9f}; lifetime total: {total:.9f};
  lifetime balance: {assets['budget']['lifetime_cap_gpu_hours']-total:.9f};
  retained reserve: {assets['budget']['reserve_gpu_hours']:.9f}.
- No MPR GPU reservation remains active at publication preparation.

Parent dependency: `artifacts/native_long_postselection_v1/deployment/E1-23760.pt`,
SHA256 `{assets['hashes']['parent']}`. The Concerto encoder is separately obtained
under its original terms, SHA256 `{assets['hashes']['encoder']}`. Raw 3RScan data,
point-level GT and the encoder are not part of this release. Bind available data,
encoder and original environment with the resolver before model construction.

`MODEL_MANIFEST.json` lists each actually produced delta/resume and its bytes,
hash, variant, seed and step. Reconstruct via
`scripts.revisit_prompt_assets.load_revisit_bundle(delta_path, assets=bound_assets)`;
the delta depends on the exact parent and encoder. Resume only from listed resume
files with the frozen RUN_LOCK and the same training identity; a deployment delta
does not include optimizer state. Seeds146/147 adapt the same seed45 parent.

Selection uses complete three-repeat CAL quarter checkpoints, then S_long,
S_current, T2, fewer parameters and variant name; within a family, earlier steps
break ties. Seed147 families are fixed by the main selection. SEL/PB do not select
models. Actual endpoints: `{json.dumps({k:selection.get(k) for k in ('control_star','candidate_star','candidate_descriptive_star')},sort_keys=True)}`.

Read `REPORT.md`, `STATUS.json`, `PROMPT_AUDIT.md` and the locked result CSVs for
all incomplete stages, denominators, negative findings and numerical limitations.
`DATA_MANIFEST.json` records historical exposure. Official state files can be
recomputed with `scripts.evaluate_revisit_prompt.pool_official_states` and the
bound taxonomy; the compressed state archive is identified in MODEL_MANIFEST.
`ARTIFACT_MANIFEST.json` freezes A; this handoff does not embed its own commit SHA.
See `COMMANDS.md` for the actual commands and MODEL_MANIFEST for asset locations.

Unique continuation decision from the completed report:
{status.get('next_direction','No further research direction is justified by the available evidence.')}
"""
    (ARTIFACTS/"HANDOFF.md").write_text(text)


def publish(assets, *, execute):
    root = Path(assets["run_root"])/"publication"
    root.mkdir(parents=True,exist_ok=True)
    status_path = ARTIFACTS/"STATUS.json"
    status = read_json(status_path)
    missing = [name for name in ("REPORT.md","MODEL_MANIFEST.json","PROMPT_AUDIT.md")
               if not (ARTIFACTS/name).is_file()]
    preview = dict(status="DRY_RUN_NOT_PUBLISHED",branch=BRANCH,tag=TAG,
                   publication_ready=status.get("publication_ready",False),missing=missing)
    if not execute:
        return preview
    if missing or not status.get("publication_ready") or not status.get("prompt_audit_execution_complete"):
        raise ValueError("publication requires the completed primary audit, report and asset manifest")
    if git("branch","--show-current") != BRANCH:
        raise ValueError("publication must use the bound new research branch")
    if (Path(assets["run_root"])/"ACTIVE_JOB.json").exists():
        raise ValueError("MPR GPU accounting reservation is still active")
    rows = check_assets(read_json(ARTIFACTS/"MODEL_MANIFEST.json"))
    git("lfs","version")
    pending_path = root/"PUBLICATION_PENDING.json"
    if pending_path.exists():
        pending = read_json(pending_path)
        expected_head = pending.get("commit_b",pending["commit_a"])
        if pending["branch"] != BRANCH or pending["tag"] != TAG or git("rev-parse","HEAD") != expected_head:
            raise ValueError("publication retry identity differs from the prepared A/B commits")
    else:
        git("lfs","install","--local")
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
        pending = dict(branch=BRANCH,tag=TAG,commit_a=commit_a)
        write_json_atomic(pending_path,pending)
    if "commit_b" not in pending:
        write_handoff(assets,commit_a=pending["commit_a"],status=status,models=rows)
        git("add","--",*[f"artifacts/revisit_prompt_v1/{name}" for name in
                           ("HANDOFF.md","COMMANDS.md","PROMPT_AUDIT.md")])
        git("commit","-m","Add MPR-V1 final handoff and execution audit")
        commit_b = git("rev-parse","HEAD")
        pending["commit_b"] = commit_b
        write_json_atomic(pending_path,pending)
    tags = git("tag","--list",TAG)
    if not tags:
        git("tag",TAG,pending["commit_b"])
    elif git("rev-parse",TAG) != pending["commit_b"]:
        raise ValueError("existing publication tag points to a different commit")
    receipt = {**pending,"status":"PUSH_FAILED","assets":rows}
    try:
        for row in rows:
            git("lfs","push","--object-id","origin",row["sha256"])
        git("push","origin",BRANCH,f"refs/tags/{TAG}",env={"GIT_LFS_SKIP_PUSH":"1"})
        remote = git("ls-remote","origin",f"refs/heads/{BRANCH}",f"refs/tags/{TAG}")
        references = {line.split()[1]:line.split()[0] for line in remote.splitlines()}
        if any(references.get(ref) != pending["commit_b"] for ref in (f"refs/heads/{BRANCH}",f"refs/tags/{TAG}")):
            raise ValueError("remote branch/tag do not point to commit B")
        with tempfile.TemporaryDirectory(prefix="mpr-remote-") as temporary:
            checkout = Path(temporary)
            git("init",cwd=checkout)
            git("remote","add","origin",git("remote","get-url","origin"),cwd=checkout)
            git("fetch","--depth=1","origin",BRANCH,cwd=checkout,env={"GIT_LFS_SKIP_SMUDGE":"1"})
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
            receipt["release_error"] = "gh CLI unavailable; no Release receipt upload was performed"
        else:
            try:
                subprocess.run(["gh","release","create",TAG,"--title","Persist4D MPR-V1",
                    "--notes-file",str(ARTIFACTS/"REPORT.md"),str(root/"PUBLICATION_RECEIPT.json")],
                    cwd=PROJECT,check=True,capture_output=True,text=True)
                receipt["status"] = "CODE_AND_LFS_PUSH_VERIFIED_RELEASE_RECEIPT_UPLOADED"
                write_json_atomic(root/"PUBLICATION_RECEIPT.json",receipt)
                subprocess.run(["gh","release","upload",TAG,str(root/"PUBLICATION_RECEIPT.json"),"--clobber"],
                    cwd=PROJECT,check=True,capture_output=True,text=True)
            except subprocess.CalledProcessError as error:
                receipt.update(status="CODE_AND_LFS_PUSH_VERIFIED_RELEASE_RECEIPT_LOCAL_ONLY",release_error=error.stderr)
    except Exception as error:
        receipt.update(status="PUSH_FAILED",error_type=type(error).__name__,error=str(error))
    write_json_atomic(root/"PUBLICATION_RECEIPT.json",receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("--dry-run",action="store_true")
    args = parser.parse_args(argv)
    assets = None
    try:
        assets = local_assets()
        result = publish(assets,execute=args.execute and not args.dry_run)
        print(json.dumps(result,sort_keys=True,allow_nan=False))
        return 1 if result["status"] == "PUSH_FAILED" else 0
    except Exception as error:
        write_json_atomic(ARTIFACTS/"PUBLISH_ENTRY_STATUS.json",dict(status="FAILED",error_type=type(error).__name__,
            error=public_error(error,assets)))
        print(f"{type(error).__name__}: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
