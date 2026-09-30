"""Scoped A/B Git delivery with external, byte-verified publication receipt."""

import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from scripts.native_long_campaign import ARTIFACTS, BASE, PROJECT, code_identity
from scripts.qp_mn_publication import artifact_manifest
from scripts.short_module_screen import read_json, sha256, write_json


def git(*args, binary=False):
    return subprocess.check_output(["git", *args], cwd=PROJECT, text=not binary)


def scope_paths():
    paths = ["models/rescene.py", "models/pointcept.py", "models/native_long_modules.py",
             "datasets/pointcept_utils.py", "datasets/native_long_dataset.py", "trainer/native_long_trainer.py",
             "conf/config_native_long_retrain.yaml", "conf/callbacks/native_long.yaml",
             "docs/superpowers/plans/2026-09-30-native-long-retrain.md", str(ARTIFACTS.relative_to(PROJECT))]
    paths += [str(p.relative_to(PROJECT)) for p in sorted((PROJECT / "scripts").glob("native_long_*.py"))]
    paths += [str(p.relative_to(PROJECT)) for p in sorted((PROJECT / "tests").glob("test_native_long_*.py"))]
    return paths


def verify_delivery(root, receipt):
    expected = receipt["commit_B"]
    refs = dict(line.split()[::-1] for line in git("ls-remote", "origin",
                f"refs/heads/{receipt['branch']}", f"refs/tags/{receipt['tag']}").splitlines())
    if any(refs.get(ref) != expected for ref in (f"refs/heads/{receipt['branch']}", f"refs/tags/{receipt['tag']}")):
        raise ValueError("remote branch/tag do not match full delivery SHA")
    git("fetch", "--no-tags", "origin", f"refs/heads/{receipt['branch']}")
    manifest = read_json(ARTIFACTS / "ARTIFACT_MANIFEST.json")
    directory = ARTIFACTS.relative_to(PROJECT)
    manifest_blob = git("cat-file", "blob", f"{expected}:{directory}/ARTIFACT_MANIFEST.json", binary=True)
    if hashlib.sha256(manifest_blob).hexdigest() != receipt["artifact_manifest_sha256"]:
        raise ValueError("remote fetched artifact manifest differs from receipt")
    assets = []
    for row in manifest["files"]:
        content = git("cat-file", "blob", f"{expected}:{directory / row['path']}", binary=True)
        if hashlib.sha256(content).hexdigest() != row["sha256"] or len(content) != row["bytes"]:
            raise ValueError("remote fetched Git asset differs from committed byte manifest")
        assets.append({**row, "url": f"https://github.com/Orangekostar/Persist4D/blob/{expected}/{directory / row['path']}"})
    git("merge-base", "--is-ancestor", BASE, expected)
    git("merge-base", "--is-ancestor", receipt["commit_A"], expected)
    receipt.update(remote_refs=refs, verified_assets=assets, git_status="GIT_COMPLETE",
                   publication_status="CODE_ONLY" if receipt.get("missing_assets") else receipt.get("publication_status", "CODE_ONLY"),
                   verified_utc=datetime.now(timezone.utc).isoformat())
    write_json(root / "publication/PUBLICATION_RECEIPT.json", receipt)
    return receipt


def upload_missing_initialization(root, receipt):
    from scripts.perception_gain_v2_publication import GitHubReleaseClient

    client = GitHubReleaseClient.authorized()
    if client is None:
        raise RuntimeError("explicit asset upload has no existing GitHub Release authorization")
    try:
        release = client.create_draft(tag=receipt["tag"], commit=receipt["commit_B"],
            notes="Budget-limited native development; initialization assets only; no trained accuracy claim.")
        uploaded = []
        for row in receipt["missing_assets"]:
            path = Path(row["path"])
            if sha256(path) != row["sha256"] or path.stat().st_size != row["bytes"]:
                raise ValueError("asset differs from reviewed missing-asset manifest")
            name = path.name
            client.upload(release, {**row, "planned_name": name})
            remote = next(r for r in client.list_assets(release["id"]) if r["name"] == name)
            digest, size = hashlib.sha256(), 0
            for chunk in client.download(remote):
                digest.update(chunk)
                size += len(chunk)
            if size != row["bytes"] or digest.hexdigest() != row["sha256"]:
                raise ValueError("remote uploaded asset bytes/SHA differ")
            uploaded.append({**row, "status": "UPLOADED_VERIFIED",
                             "download_url": remote["browser_download_url"]})
        client.publish(release["id"])
        receipt.update(uploaded_initialization_assets=uploaded, missing_assets=[],
                       publication_status="GIT_WITH_INITIALIZATION_ASSET_NO_TRAINED_MODEL")
        write_json(root / "publication/PUBLICATION_RECEIPT.json", receipt)
        return receipt
    finally:
        client.session.close()


def publish(root, *, upload_assets=False):
    branch = git("branch", "--show-current").strip()
    if branch != "research/rescene-native-long-retrain-v1":
        raise ValueError("refusing a different publication branch")
    receipt_path = root / "publication/PUBLICATION_RECEIPT.json"
    scopes = scope_paths()
    if receipt_path.exists() and not git("status", "--porcelain", "--", *scopes).strip():
        receipt = verify_delivery(root, read_json(receipt_path))
        return upload_missing_initialization(root, receipt) if upload_assets else receipt
    review = read_json(ARTIFACTS / "PRIMARY_REVIEW.json")
    if review["code"] != code_identity() or review["decision"] != "ACCEPT_BUDGET_LIMITED_DEVELOPMENT_DELIVERY":
        raise ValueError("current numerical code lacks the primary-agent evidence review")
    if any(sha256(PROJECT / p) != digest for p, digest in review["evidence"].items()):
        raise ValueError("primary-reviewed evidence changed")
    staged = git("diff", "--cached", "--name-only").splitlines()
    if any(not any(p == scope or p.startswith(scope + "/") for scope in scopes) for p in staged):
        raise ValueError("unrelated staged changes cannot enter this publication")
    artifact_manifest(ARTIFACTS)
    git("add", "--", *scopes)
    if git("diff", "--cached", "--name-only").strip():
        git("commit", "-m", "Implement native long retraining and record budget-limited real preflight")
    commit_a = git("rev-parse", "HEAD").strip()
    tag_base = "rescene-native-long-retrain-v1"
    occupied = set(git("tag", "--list").splitlines())
    occupied.update(line.split()[1].removeprefix("refs/tags/") for line in git("ls-remote", "--tags", "origin").splitlines())
    tag, counter = tag_base, 2
    while tag in occupied:
        tag = f"{tag_base}-r{counter}"
        counter += 1
    reload_audit = read_json(ARTIFACTS / "initialization/TASK_BUNDLE_RELOAD_AUDIT.json")
    missing = [{"role": "INITIALIZATION_ONLY_NONENCODER_TASK", "path": reload_audit["path"],
                "sha256": reload_audit["sha256"], "bytes": reload_audit["bytes"], "download_url": None,
                "status": "LOCAL_ONLY_RELEASE_UNAVAILABLE"}]
    handoff = (
        "# Native Long Retrain Handoff\n\n"
        f"Experiment A: `{commit_a}`. Branch: `{branch}`. Delivery tag: `{tag}`. Base: `{BASE}`.\n\n"
        "Scientific status is PARTIAL_FULL_RETRAIN_BUDGET; publication is CODE_ONLY. All four arms have zero "
        "actual training updates. No main AP, CAL/SEL lock, seed46, formal confirmation, trained profile or gain exists. "
        "The measured full E0 forecast already exceeds the remaining lifetime budget. "
        "Do not shrink U or convert temporary preflight optimizer updates into a training result.\n\n"
        f"Runtime: `{root}`. Read FINAL_REPORT.md, REQUIREMENT_REVIEW.md, CODE_BINDINGS.md and RUNTIME_NOTE.md. "
        "The encoder is the separately obtained fixed Concerto SHA in SOURCE_AND_INITIALIZATION.json. "
        "Initialization contains no old task warm start. Current encoder has zero registered buffers; "
        "the loader explicitly preserves every nonencoder tensor and any registered encoder buffers.\n\n"
        "Git includes untrained F initialization parameters and official initialized TRAIN-A sufficient statistics. "
        f"Nonencoder task initialization bundle remains local: `{reload_audit['path']}`, "
        f"{reload_audit['bytes']} bytes, SHA `{reload_audit['sha256']}`. It is not a trained deployment model. "
        "No Release authorization was available; no trained weights were generated. "
        "Raw data, raw GT and original pretrained files are not redistributed.\n\n"
        "Commands with the existing persist4d environment, from this worktree:\n\n```bash\n"
        f"export RESCENE_NATIVE_LONG_ROOT='{root}'\n"
        "python -m scripts.native_long_campaign status --root \"$RESCENE_NATIVE_LONG_ROOT\"\n"
        "python -m scripts.native_long_campaign run --root \"$RESCENE_NATIVE_LONG_ROOT\" --resume\n"
        "python -m scripts.native_long_campaign report --root \"$RESCENE_NATIVE_LONG_ROOT\"\n"
        "python -m scripts.native_long_campaign publish --root \"$RESCENE_NATIVE_LONG_ROOT\" --resume\n```\n\n"
        "The run command revalidates bindings and retains the empty budget-authorized trajectory set. "
        "A new training budget needs an explicit new plan; current authorization cannot sustain full retraining. "
        "Seed45 training/CAL/SEL code has no production-run validation; positive-plan replication/formal/profile "
        "and trained-result reporting remain outstanding. These are not certified as complete.\n\n"
        "Once Release authorization is explicitly available, the existing GitHubReleaseClient can upload the "
        "single task initialization package with its manifest bytes/SHA; an empty Release or an LFS pointer "
        "is not delivery. Current missing-asset details are in the external receipt.\n\n```bash\n"
        "python -m scripts.native_long_campaign publish --root \"$RESCENE_NATIVE_LONG_ROOT\" --resume --upload-assets\n```\n\n"
        "B is recorded only in the external publication/PUBLICATION_RECEIPT.json after actual push. "
        "ARTIFACT_MANIFEST.json excludes itself; this handoff does not contain its own commit SHA.\n"
    )
    (ARTIFACTS / "HANDOFF.md").write_text(handoff)
    write_json(ARTIFACTS / "ARTIFACT_MANIFEST.json", artifact_manifest(ARTIFACTS))
    git("add", "--", str((ARTIFACTS / "HANDOFF.md").relative_to(PROJECT)),
        str((ARTIFACTS / "ARTIFACT_MANIFEST.json").relative_to(PROJECT)))
    git("commit", "-m", "Record native long retraining handoff and artifact manifest")
    commit_b = git("rev-parse", "HEAD").strip()
    git("push", "origin", f"HEAD:refs/heads/{branch}")
    git("tag", tag, commit_b)
    git("push", "origin", f"refs/tags/{tag}")
    receipt = {"commit_A": commit_a, "commit_B": commit_b, "branch": branch, "tag": tag,
               "scientific_status": "PARTIAL_FULL_RETRAIN_BUDGET", "missing_assets": missing,
               "trained_weight_assets": "NOT_CREATED", "release_authorized_at_startup": False,
               "artifact_manifest_sha256": sha256(ARTIFACTS / "ARTIFACT_MANIFEST.json")}
    verified = verify_delivery(root, receipt)
    return upload_missing_initialization(root, verified) if upload_assets else verified
