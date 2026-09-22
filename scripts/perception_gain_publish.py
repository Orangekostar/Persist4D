#!/usr/bin/env python3
"""Commit, push, tag, and release the Perception Gain V1 delivery."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts/perception_gain_v1"
DEFAULT_EXTERNAL_ROOT = Path("/home/ww/persist4d_runs/perception_gain_v1")
BASE_TAG = "persist4d-perception-gain-v1"
BRANCH = "research/persist4d-perception-gain-v1"
REPOSITORY = "Orangekostar/Persist4D"


class PublicationError(RuntimeError):
    """Raised when the publication transaction cannot be verified."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PublicationError(f"cannot read JSON: {path}") from error
    if not isinstance(value, dict):
        raise PublicationError(f"JSON root must be an object: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _run(
    command: Sequence[str],
    *,
    cwd: Path = PROJECT_ROOT,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        list(command), cwd=cwd, capture_output=True, text=True, check=False
    )
    if check and result.returncode != 0:
        reason = (result.stderr or result.stdout).strip()
        raise PublicationError(f"command failed: {' '.join(command)}: {reason}")
    return result


def choose_release_tag(existing_tags: set[str], *, base_tag: str = BASE_TAG) -> str:
    if base_tag not in existing_tags:
        return base_tag
    suffix = 2
    while f"{base_tag}-r{suffix}" in existing_tags:
        suffix += 1
    return f"{base_tag}-r{suffix}"


def validate_release_assets(
    records: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    output = []
    names = set()
    for record in records:
        name = record.get("planned_name")
        raw_path = record.get("path")
        expected_bytes = record.get("bytes")
        expected_sha = record.get("sha256")
        if (
            not isinstance(name, str)
            or not name
            or name in names
            or not isinstance(raw_path, (str, Path))
            or isinstance(expected_bytes, bool)
            or not isinstance(expected_bytes, int)
            or expected_bytes <= 0
            or not isinstance(expected_sha, str)
            or len(expected_sha) != 64
        ):
            raise PublicationError("release asset record differs")
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file() or path.stat().st_size != expected_bytes:
            raise PublicationError(f"release asset bytes differ: {name}")
        if _sha256(path) != expected_sha:
            raise PublicationError(f"release asset SHA256 differs: {name}")
        if expected_bytes > 1024**3:
            raise PublicationError(f"release asset exceeds 1 GiB: {name}")
        names.add(name)
        output.append({**dict(record), "path": path})
    return tuple(output)


def build_publication_receipt(
    *,
    publication_commit: str,
    remote_branch_sha: str,
    tag: str,
    remote_tag_sha: str,
    branch_url: str,
    commit_url: str,
    handoff_url: str,
    report_url: str,
    release: Mapping[str, object] | None,
    assets: Sequence[Mapping[str, object]],
    verified_remote_files: Sequence[str],
) -> dict[str, object]:
    if (
        len(publication_commit) != 40
        or remote_branch_sha != publication_commit
        or remote_tag_sha != publication_commit
        or not tag
        or not all(
            isinstance(value, str) and value.startswith("https://")
            for value in (branch_url, commit_url, handoff_url, report_url)
        )
    ):
        raise PublicationError("publication receipt identity differs")
    released = release is not None
    return {
        "schema_version": "perception-gain-publication-receipt-v1",
        "publication_status": "VERIFIED" if released else "CODE_ONLY",
        "publication_commit": publication_commit,
        "remote_branch_sha": remote_branch_sha,
        "tag": tag,
        "remote_tag_sha": remote_tag_sha,
        "branch_url": branch_url,
        "commit_url": commit_url,
        "handoff_url": handoff_url,
        "final_report_url": report_url,
        "release_url": release.get("url") if released else None,
        "release_id": release.get("id") if released else None,
        "assets": [dict(value) for value in assets],
        "verified_remote_files": list(verified_remote_files),
        "verification_level": "REMOTE_METADATA" if released else "REMOTE_GIT_CONTENT",
        "verified_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }


def _existing_tags() -> set[str]:
    local = set(_run(("git", "tag", "--list")).stdout.splitlines())
    remote_result = _run(("git", "ls-remote", "--tags", "origin"))
    remote = {
        line.split("refs/tags/", 1)[1].removesuffix("^{}")
        for line in remote_result.stdout.splitlines()
        if "refs/tags/" in line
    }
    return local | remote


def _git_commit(message: str) -> str:
    staged = _run(("git", "diff", "--cached", "--quiet"), check=False)
    if staged.returncode == 0:
        return _run(("git", "rev-parse", "HEAD")).stdout.strip()
    if staged.returncode != 1:
        raise PublicationError("cannot inspect staged publication changes")
    _run(("git", "commit", "-m", message))
    commit = _run(("git", "rev-parse", "HEAD")).stdout.strip()
    if len(commit) != 40:
        raise PublicationError("publication commit SHA is invalid")
    return commit


def _stage_experiment_snapshot(artifact_root: Path) -> None:
    relative_artifact = artifact_root.relative_to(PROJECT_ROOT).as_posix()
    paths = (
        "configs/perception_gain_v1.yaml",
        "conf/perception_gain_v1",
        "docs/superpowers/plans/2026-09-21-persist4d-perception-gain-v1.md",
        "models",
        "scripts",
        "tests",
        "trainer",
        relative_artifact,
        f":(exclude){relative_artifact}/HANDOFF.md",
        f":(exclude){relative_artifact}/ARTIFACT_MANIFEST.json",
        f":(exclude){relative_artifact}/RELEASE_PLAN.json",
    )
    _run(("git", "add", "-A", "--", *paths))


def _resolve_release_assets(
    *, release_plan: Mapping[str, object], external_root: Path
) -> tuple[dict[str, object], ...]:
    records = release_plan.get("assets")
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise PublicationError("release plan assets differ")
    resolved = []
    for record in records:
        if not isinstance(record, Mapping) or record.get("status") != "READY":
            continue
        reference = record.get("logical_reference")
        if not isinstance(reference, str) or not reference.startswith("external:"):
            raise PublicationError("release asset reference differs")
        path = (external_root / reference.removeprefix("external:")).resolve()
        try:
            path.relative_to(external_root)
        except ValueError as error:
            raise PublicationError("release asset escapes external root") from error
        resolved.append({**dict(record), "path": path})
    return validate_release_assets(resolved)


def _create_release(
    *,
    tag: str,
    assets: Sequence[Mapping[str, object]],
    external_root: Path,
) -> tuple[dict[str, object] | None, tuple[dict[str, object], ...]]:
    if (
        shutil.which("gh") is None
        or _run(("gh", "auth", "status"), check=False).returncode != 0
    ):
        return None, ()
    publication_root = external_root / "publication"
    upload_root = publication_root / "assets"
    upload_root.mkdir(parents=True, exist_ok=True)
    uploads = []
    for record in assets:
        target = upload_root / str(record["planned_name"])
        source = Path(record["path"])
        if (
            not target.is_file()
            or target.stat().st_size != source.stat().st_size
            or _sha256(target) != record["sha256"]
        ):
            shutil.copyfile(source, target)
        uploads.append(target)
    notes = publication_root / "RELEASE_NOTES.md"
    notes.write_text(
        "# Persist4D Perception Gain V1\n\n"
        "This prerelease reports the locked measured result, including negative or "
        "incomplete findings exactly as recorded in FINAL_REPORT.md.\n",
        encoding="utf-8",
    )
    command = [
        "gh",
        "release",
        "create",
        tag,
        "--repo",
        REPOSITORY,
        "--prerelease",
        "--title",
        "Persist4D Perception Gain V1",
        "--notes-file",
        str(notes),
        *[str(path) for path in uploads],
    ]
    created = _run(command, check=False)
    if created.returncode != 0:
        existing = _run(
            ("gh", "release", "view", tag, "--repo", REPOSITORY), check=False
        )
        if existing.returncode != 0:
            created = _run(command, check=False)
            if created.returncode != 0:
                return None, ()
    viewed = _run(
        (
            "gh",
            "release",
            "view",
            tag,
            "--repo",
            REPOSITORY,
            "--json",
            "url,databaseId,assets",
        )
    )
    value = json.loads(viewed.stdout)
    remote_assets = value.get("assets", ())
    by_name = {
        item.get("name"): item for item in remote_assets if isinstance(item, Mapping)
    }
    verified = []
    for record in assets:
        remote = by_name.get(record["planned_name"])
        if not isinstance(remote, Mapping) or remote.get("size") != record["bytes"]:
            raise PublicationError(
                f"release asset metadata differs: {record['planned_name']}"
            )
        verified.append(
            {
                "name": record["planned_name"],
                "bytes": record["bytes"],
                "sha256": record["sha256"],
                "url": remote.get("url"),
            }
        )
    return {"url": value.get("url"), "id": value.get("databaseId")}, tuple(verified)


def publish(
    *,
    artifact_root: Path = ARTIFACT_ROOT,
    external_root: Path | None = None,
    version: str = "v1",
    config: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if version == "v2":
        from scripts.perception_gain_v2 import load_config
        from scripts.perception_gain_v2_publication import publish_v2

        config = config or load_config(PROJECT_ROOT / "configs/perception_gain_v2.yaml")
        external_root = external_root or Path(
            os.environ.get(
                "PERSIST4D_GAIN_V2_ROOT", Path.home() / "persist4d_runs/perception_gain_v2"
            )
        )
        return publish_v2(config, external_root=external_root)
    if version != "v1":
        raise PublicationError("Unknown publication version")
    from scripts.perception_gain_reporting import generate_delivery

    artifact_root = artifact_root.resolve()
    external_root = external_root or DEFAULT_EXTERNAL_ROOT
    external_root = external_root.expanduser().resolve(strict=True)
    branch = _run(("git", "branch", "--show-current")).stdout.strip()
    if branch != BRANCH:
        raise PublicationError(f"publication branch differs: {branch}")
    tag = choose_release_tag(_existing_tags())
    _stage_experiment_snapshot(artifact_root)
    experiment_commit = _git_commit("exp: freeze perception gain v1 results")

    state_path = artifact_root / "RUN_STATE.json"
    state = _read_json(state_path)
    state.update(
        {
            "publication_phase": "READY",
            "publication_status": None,
            "publication_receipt": "external:publication/PUBLICATION_RECEIPT.json",
        }
    )
    _atomic_json(state_path, state)
    generate_delivery(
        project_root=PROJECT_ROOT,
        artifact_root=artifact_root,
        external_root=external_root,
        experiment_commit=experiment_commit,
        publication_phase="READY",
        release_tag=tag,
    )
    _run(
        (
            "git",
            "add",
            "--",
            str((artifact_root / "FINAL_REPORT.md").relative_to(PROJECT_ROOT)),
            str((artifact_root / "HANDOFF.md").relative_to(PROJECT_ROOT)),
            str((artifact_root / "ARTIFACT_MANIFEST.json").relative_to(PROJECT_ROOT)),
            str((artifact_root / "RELEASE_PLAN.json").relative_to(PROJECT_ROOT)),
            str(state_path.relative_to(PROJECT_ROOT)),
        )
    )
    publication_commit = _git_commit("docs: publish perception gain v1 handoff")
    _run(("git", "push", "-u", "origin", BRANCH))
    local_tag = _run(("git", "tag", "--list", tag)).stdout.strip()
    if local_tag:
        raise PublicationError("newly selected release tag unexpectedly exists locally")
    _run(("git", "tag", tag, publication_commit))
    _run(("git", "push", "origin", f"refs/tags/{tag}"))
    remote_branch = _run(
        ("git", "ls-remote", "origin", f"refs/heads/{BRANCH}")
    ).stdout.split()
    remote_tag = _run(("git", "ls-remote", "origin", f"refs/tags/{tag}")).stdout.split()
    if len(remote_branch) != 2 or len(remote_tag) != 2:
        raise PublicationError("remote branch or tag is unavailable after push")
    remote_branch_sha, remote_tag_sha = remote_branch[0], remote_tag[0]
    if remote_branch_sha != publication_commit or remote_tag_sha != publication_commit:
        raise PublicationError("remote branch/tag SHA differs from publication commit")
    _run(("git", "fetch", "origin", BRANCH))
    verified_files = ["artifacts/perception_gain_v1/HANDOFF.md"]
    if (artifact_root / "confirmation/CONFIRMATION_SUMMARY.json").is_file():
        verified_files.append(
            "artifacts/perception_gain_v1/confirmation/CONFIRMATION_SUMMARY.json"
        )
    else:
        verified_files.append("artifacts/perception_gain_v1/FINAL_REPORT.md")
    for path in verified_files:
        shown = _run(("git", "show", f"{remote_branch_sha}:{path}"))
        if not shown.stdout.strip():
            raise PublicationError(f"remote publication file is empty: {path}")
    plan = _read_json(artifact_root / "RELEASE_PLAN.json")
    assets = _resolve_release_assets(release_plan=plan, external_root=external_root)
    release, remote_assets = _create_release(
        tag=tag, assets=assets, external_root=external_root
    )
    base_url = f"https://github.com/{REPOSITORY}"
    receipt = build_publication_receipt(
        publication_commit=publication_commit,
        remote_branch_sha=remote_branch_sha,
        tag=tag,
        remote_tag_sha=remote_tag_sha,
        branch_url=f"{base_url}/tree/{BRANCH}",
        commit_url=f"{base_url}/commit/{publication_commit}",
        handoff_url=f"{base_url}/blob/{publication_commit}/artifacts/perception_gain_v1/HANDOFF.md",
        report_url=f"{base_url}/blob/{publication_commit}/artifacts/perception_gain_v1/FINAL_REPORT.md",
        release=release,
        assets=remote_assets,
        verified_remote_files=verified_files,
    )
    receipt_path = external_root / "publication/PUBLICATION_RECEIPT.json"
    _atomic_json(receipt_path, receipt)
    if release is not None:
        uploaded = _run(
            (
                "gh",
                "release",
                "upload",
                tag,
                str(receipt_path),
                "--repo",
                REPOSITORY,
                "--clobber",
            ),
            check=False,
        )
        if uploaded.returncode != 0:
            raise PublicationError("publication receipt upload failed")
    return {
        "schema_version": "perception-gain-publication-v1",
        "status": "PASS",
        "publication_status": receipt["publication_status"],
        "experiment_commit": experiment_commit,
        "publication_commit": publication_commit,
        "tag": tag,
        "branch_url": receipt["branch_url"],
        "commit_url": receipt["commit_url"],
        "handoff_url": receipt["handoff_url"],
        "final_report_url": receipt["final_report_url"],
        "release_url": receipt["release_url"],
        "receipt": "external:publication/PUBLICATION_RECEIPT.json",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--external-root", type=Path)
    parser.add_argument("--version", choices=("v1", "v2"), default="v1")
    return parser


def main() -> int:
    args = _parser().parse_args()
    print(
        json.dumps(
            publish(
                artifact_root=args.artifact_root,
                external_root=args.external_root,
                version=args.version,
            ),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
