"""V2 draft-release transaction with exact required-asset verification.

GitHub contract: https://docs.github.com/en/rest/releases/releases
Asset digest contract: https://docs.github.com/en/rest/releases/assets
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

from scripts.perception_gain_publish import (
    REPOSITORY,
    PublicationError,
    _existing_tags,
    _git_commit,
    _run,
    choose_release_tag,
    validate_release_assets,
)
from scripts.perception_gain_v2 import (
    PROJECT_ROOT,
    file_hash,
    read_json,
    utc_now,
    write_json,
)

BASE_TAG = "persist4d-perception-gain-v2"
BRANCH = "research/persist4d-perception-gain-v2"


def selected_release_methods(lock):
    """Freeze weights required for the final method and its named controls."""
    labels = ("FINAL", "P-D0", "C0-best-D0", "FINAL-NEW-control")
    methods = {}
    for label in labels:
        name = lock.get("aliases", {}).get(label)
        if name is not None:
            method = lock["methods"][name]
            methods.setdefault(method["inference_identity"], method)
    if not methods or lock["final_method_id"] not in {
        row["method_id"] for row in methods.values()
    }:
        raise PublicationError(
            "Final publication inventory lacks the locked final method"
        )
    return list(methods.values())


def exploratory_release_methods(artifacts, external_root):
    from scripts.perception_gain_v2_lock import describe
    from scripts.perception_gain_v2_perception import resolve_checkpoint

    methods = []
    for recipe_path in sorted((artifacts / "training").glob("*/recipe.json")):
        recipe = read_json(recipe_path)
        manifest = recipe_path.parent / "checkpoint_manifest.json"
        if not manifest.is_file():
            continue
        updates = [
            int(match.group(1))
            for row in read_json(manifest)["checkpoints"]
            if (match := re.fullmatch(r"update=(\d+)\.ckpt", row["name"]))
        ]
        if not updates:
            continue
        step = max(updates)
        checkpoint = resolve_checkpoint(
            recipe["recipe_id"], step, external_root=external_root
        )
        candidate = {
            "method_id": recipe["recipe_id"] + f"-u{step}",
            "recipe_id": recipe["recipe_id"],
            "optimizer_update": step,
            "checkpoint_sha256": file_hash(checkpoint),
        }
        method = describe(
            candidate, parent=None, artifacts=artifacts, external_root=external_root
        )
        method["publication_selection"] = "EXPLORATORY_NOT_SELECTED"
        methods.append(method)
    return methods


def _asset(path, *, external_root, role, name=None):
    return {
        "role": role,
        "planned_name": name or path.name,
        "logical_reference": "external:" + str(path.relative_to(external_root)),
        "path": path,
        "bytes": path.stat().st_size,
        "sha256": file_hash(path),
        "status": "READY",
    }


def prepare_model_assets(artifacts, external_root):
    from scripts.perception_gain_v2_bundle import build_method_bundle, split_asset
    from scripts.perception_gain_v2_lock import external_path

    lock_path = artifacts / "selection/FINAL_LOCK.json"
    methods = (
        selected_release_methods(read_json(lock_path))
        if lock_path.is_file()
        else exploratory_release_methods(artifacts, external_root)
    )
    assets, requirements = [], []
    for method in methods:
        name = re.sub(r"[^A-Za-z0-9_.-]", "_", method["method_id"])
        try:
            bundle = build_method_bundle(
                method, external_root=external_root, artifacts=artifacts
            )
            source = external_path(bundle["logical_reference"], external_root)
            parts = split_asset(source)
            for part in parts["ordered_parts"]:
                assets.append(
                    _asset(
                        source.parent / part["name"],
                        external_root=external_root,
                        role="deployment_bundle",
                    )
                )
            assets.append(
                _asset(
                    source.with_suffix(source.suffix + ".manifest.json"),
                    external_root=external_root,
                    role="bundle_parts_manifest",
                )
            )
            if bundle.get("live_panel_reload") != "PASS":
                requirements.append(
                    {"name": f"live-panel-reload:{name}", "status": "NOT_VERIFIED"}
                )
            if source.stat().st_size <= 20 * 1024**2:
                destination = artifacts / "publication/small-models" / source.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists() and file_hash(destination) != bundle["sha256"]:
                    raise PublicationError("Previously prepared small model differs")
                if not destination.exists():
                    shutil.copyfile(source, destination)
        except (OSError, ValueError, RuntimeError, KeyError) as error:
            requirements.append(
                {"name": name + ".pt", "status": "MISSING", "reason": str(error)}
            )
    # A partial run may have trained only repair/scorer components. Their own
    # parameter files can be delivered without distributing any parent weights.
    if not lock_path.is_file():
        for training in sorted((artifacts / "refiner").glob("*/[NP]*/TRAINING.json")):
            row = read_json(training)
            if row.get("completed_updates", 0) <= 0:
                continue
            parent, label = training.parent.parent.name, training.parent.name
            source = (
                external_root
                / f"training/refiner/{parent}/{label}-s45"
                / row["checkpoint"]
            )
            name = f"exploratory-{parent}-{label}-s45.ckpt"
            if not source.is_file() or file_hash(source) != row["checkpoint_sha256"]:
                requirements.append({"name": name, "status": "MISSING"})
                continue
            assets.append(
                _asset(
                    source,
                    external_root=external_root,
                    role="EXPLORATORY_NOT_SELECTED",
                    name=name,
                )
            )
        local_assets = read_json(external_root / "assets.local.json")
        if local_assets.get("scorer_checkpoint"):
            source = Path(local_assets["scorer_checkpoint"])
            if source.is_file():
                # Copy only the explicitly bound owned scorer, never its base model.
                destination = external_root / "publication/components/scorer.ckpt"
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists() and file_hash(destination) != file_hash(source):
                    raise PublicationError("Previously prepared scorer differs")
                if not destination.exists():
                    shutil.copyfile(source, destination)
                assets.append(
                    _asset(
                        destination,
                        external_root=external_root,
                        role="EXPLORATORY_NOT_SELECTED",
                    )
                )
    return assets, requirements


def snapshot_paths(artifacts, project_root):
    """Enumerate explicit owned files; raw caches/checkpoints are never staged."""
    deferred = {
        "HANDOFF.md",
        "ARTIFACT_MANIFEST.json",
        "RELEASE_PLAN.json",
        "PUBLICATION_SNAPSHOT.json",
    }
    paths = []
    for path in sorted(artifacts.rglob("*")):
        if not path.is_file() or path.name in deferred:
            continue
        relative = path.relative_to(artifacts)
        small_model = relative.parts[:2] == ("publication", "small-models")
        if path.is_symlink() or not path.resolve().is_relative_to(artifacts.resolve()):
            raise PublicationError("Public artifact may not follow an external symlink")
        if path.name in {
            "assets.local.json",
            "PUBLICATION_RECEIPT.json",
            "PREPUBLICATION_RECEIPT.json",
        }:
            raise PublicationError(
                "Private or external receipt found in Git artifact scope"
            )
        if small_model:
            if path.suffix != ".pt" or path.stat().st_size > 20 * 1024**2:
                raise PublicationError(
                    "Git small-model artifact exceeds its explicit scope"
                )
        elif path.suffix not in {
            ".json",
            ".jsonl",
            ".csv",
            ".yaml",
            ".md",
            ".txt",
            ".svg",
            ".png",
            ".pdf",
        }:
            raise PublicationError(f"Unapproved public artifact type: {relative}")
        paths.append(path.relative_to(project_root).as_posix())
    return paths


def artifact_manifest(paths, *, project_root, experiment_commit):
    excluded = {
        "ARTIFACT_MANIFEST.json",
        "RUN_STATE.json",
        "EXECUTION_LOG.jsonl",
        "RECOVERY_EVENTS.jsonl",
        "PUBLICATION_RECEIPT.json",
        "PREPUBLICATION_RECEIPT.json",
    }
    rows = []
    for name in sorted(set(paths)):
        path = project_root / name
        if path.name in excluded or "budget" in path.parts:
            continue
        rows.append(
            {"path": name, "bytes": path.stat().st_size, "sha256": file_hash(path)}
        )
    return {
        "schema_version": "perception-gain-artifact-manifest-v2",
        "experiment_commit": experiment_commit,
        "publication_phase": "READY",
        "excludes": sorted(excluded | {"budget/*"}),
        "files": rows,
    }


def _remote_sha(ref):
    fields = _run(("git", "ls-remote", "origin", ref)).stdout.split()
    if len(fields) != 2 or fields[1] != ref:
        raise PublicationError(f"Remote ref is unavailable: {ref}")
    return fields[0]


def publish_v2(config, *, external_root):
    """Freeze A then B, push exact refs, and publish assets using existing auth."""
    from scripts.perception_gain_v2_report import run_report

    artifacts = PROJECT_ROOT / config["artifact_root"]
    receipt_path = external_root / "publication/PUBLICATION_RECEIPT.json"
    previous = read_json(receipt_path) if receipt_path.is_file() else {}
    if previous:
        archive = receipt_path.with_name(
            "PUBLICATION_RECEIPT-"
            + previous.get("publication_commit", "blocked")[:12]
            + ".json"
        )
        if not archive.exists():
            shutil.copyfile(receipt_path, archive)
    receipt = {
        "schema_version": "perception-gain-publication-receipt-v2",
        "publication_status": "BLOCKED",
        "assets": [],
        "missing_assets": [],
        "verified_at_utc": utc_now(),
    }
    try:
        state = read_json(artifacts / "RUN_STATE.json")
        active = [
            key
            for key, value in state["tasks"].items()
            if value["status"] == "RUNNING" and key != "PUBLISH"
        ]
        if active:
            raise PublicationError(
                "Publication requires a stable experiment snapshot; active tasks: "
                + ", ".join(active)
            )
        if (
            _run(("git", "branch", "--show-current")).stdout.strip() != BRANCH
            or config["branch"] != BRANCH
        ):
            raise PublicationError("V2 publication is on another branch")
        if _run(("git", "diff", "--cached", "--name-only")).stdout.strip():
            raise PublicationError(
                "Pre-existing staged changes require their own review and commit"
            )
        # Source changes must already have passed the primary agent's review.
        dirty = _run(("git", "diff", "--name-only")).stdout.splitlines()
        prefix = artifacts.relative_to(PROJECT_ROOT).as_posix() + "/"
        if any(not path.startswith(prefix) for path in dirty):
            raise PublicationError(
                "Commit reviewed source changes before preparing publication"
            )
        assets, missing = prepare_model_assets(artifacts, external_root)
        run_report(config, external_root=external_root)
        owned = snapshot_paths(artifacts, PROJECT_ROOT)
        if owned:
            _run(("git", "add", "-f", "--", *owned))
        experiment_commit = _git_commit(
            "Record actual perception gain V2 experiments and deployment tensors"
        )
        receipt["experiment_commit"] = experiment_commit
        tag = choose_release_tag(_existing_tags(), base_tag=BASE_TAG)
        results = external_root / f"publication/results-{experiment_commit[:12]}.zip"
        results.parent.mkdir(parents=True, exist_ok=True)
        _run(
            (
                "git",
                "archive",
                "--format=zip",
                "--output",
                str(results),
                experiment_commit,
                "--",
                prefix.rstrip("/"),
            )
        )
        from scripts.perception_gain_v2_bundle import split_asset

        parts = split_asset(results)
        assets.extend(
            _asset(
                results.parent / part["name"],
                external_root=external_root,
                role="scientific_results",
            )
            for part in parts["ordered_parts"]
        )
        assets.append(
            _asset(
                results.with_suffix(results.suffix + ".manifest.json"),
                external_root=external_root,
                role="results_parts_manifest",
            )
        )
        # Prediction manifests are produced by the real evaluation path. A
        # missing export remains a missing requirement, even with a results ZIP.
        predictions = artifacts / "publication/PREDICTION_ASSETS.json"
        if predictions.is_file():
            for row in read_json(predictions).get("assets", []):
                reference = row["logical_reference"]
                from scripts.perception_gain_v2_lock import external_path

                path = external_path(reference, external_root)
                assets.append({**row, "path": path})
        else:
            missing.append({"name": "sanitized-predictions", "status": "MISSING"})
        assets = list(validate_release_assets(assets))
        plan = {
            "schema_version": "perception-gain-release-plan-v2",
            "publication_phase": "READY",
            "experiment_commit": experiment_commit,
            "tag": tag,
            "branch": BRANCH,
            "assets": [
                {key: value for key, value in row.items() if key != "path"}
                for row in assets
            ],
            "required_asset_names": [row["planned_name"] for row in assets]
            + ["ARTIFACT_MANIFEST.json"],
            "unfulfilled_requirements": missing,
        }
        write_json(artifacts / "RELEASE_PLAN.json", plan)
        write_json(
            artifacts / "PUBLICATION_SNAPSHOT.json",
            {
                key: plan[key]
                for key in ("publication_phase", "experiment_commit", "tag", "branch")
            },
        )
        handoff = artifacts / "HANDOFF.md"
        handoff.write_text(
            handoff.read_text()
            + f"\n\nPublication snapshot: experiment A `{experiment_commit}`; phase `READY`; branch `{BRANCH}`; planned tag `{tag}`.\nFinal receipt: `$PERSIST4D_GAIN_V2_ROOT/publication/PUBLICATION_RECEIPT.json` (external, written only after remote verification).\n"
        )
        delivery = [
            prefix + name
            for name in ("HANDOFF.md", "RELEASE_PLAN.json", "PUBLICATION_SNAPSHOT.json")
        ]
        write_json(
            artifacts / "ARTIFACT_MANIFEST.json",
            artifact_manifest(
                owned + delivery,
                project_root=PROJECT_ROOT,
                experiment_commit=experiment_commit,
            ),
        )
        delivery.append(prefix + "ARTIFACT_MANIFEST.json")
        _run(("git", "add", "--", *delivery))
        commit = _git_commit("Freeze perception gain V2 delivery snapshot READY")
        receipt.update(publication_commit=commit, tag=tag, missing_assets=missing)
        _run(("git", "push", "-u", "origin", f"{commit}:refs/heads/{BRANCH}"))
        if _remote_sha(f"refs/heads/{BRANCH}") != commit:
            raise PublicationError("Remote branch differs from snapshot B")
        _run(("git", "tag", tag, commit))
        _run(("git", "push", "origin", f"refs/tags/{tag}"))
        if _remote_sha(f"refs/tags/{tag}") != commit:
            raise PublicationError("Remote tag differs from snapshot B")
        _run(("git", "fetch", "origin", f"refs/heads/{BRANCH}"))
        if _run(("git", "rev-parse", "FETCH_HEAD")).stdout.strip() != commit:
            raise PublicationError("Fetched remote branch differs from snapshot B")
        verified_files = [*delivery, prefix + "FINAL_REPORT.md"]
        for name in verified_files:
            expected = _run(("git", "rev-parse", f"{commit}:{name}")).stdout.strip()
            observed = _run(("git", "rev-parse", f"FETCH_HEAD:{name}")).stdout.strip()
            if expected != observed:
                raise PublicationError("Remote delivery blob differs: " + name)
        base = f"https://github.com/{REPOSITORY}"
        receipt.update(
            publication_status="CODE_ONLY",
            remote_branch_sha=commit,
            remote_tag_sha=commit,
            branch_url=f"{base}/tree/{BRANCH}",
            commit_url=f"{base}/commit/{commit}",
            handoff_url=f"{base}/blob/{commit}/{prefix}HANDOFF.md",
            final_report_url=f"{base}/blob/{commit}/{prefix}FINAL_REPORT.md",
            verified_remote_files=verified_files,
            verification_level="REMOTE_GIT_CONTENT",
        )
        client = GitHubReleaseClient.authorized()
        if client is None:
            receipt["reason"] = (
                "No existing authorized GitHub Release credential is available; Git was pushed and verified."
            )
            receipt["missing_assets"] += [
                {"name": row["planned_name"], "status": "NOT_UPLOADED"}
                for row in assets
            ]
            receipt["missing_assets"].append(
                {"name": "ARTIFACT_MANIFEST.json", "status": "NOT_UPLOADED"}
            )
        else:
            manifest_asset = external_root / "publication/ARTIFACT_MANIFEST.json"
            shutil.copyfile(artifacts / "ARTIFACT_MANIFEST.json", manifest_asset)
            all_assets = [
                *assets,
                _asset(
                    manifest_asset,
                    external_root=external_root,
                    role="delivery_manifest",
                ),
            ]
            published = publish_draft_assets(
                client,
                tag=tag,
                commit=commit,
                notes=f"Actual Persist4D Perception Gain V2 results.\n\nSnapshot B: {commit}\nExperiment A: {experiment_commit}\n\nReport: {receipt['final_report_url']}\n\nUnfulfilled requirements: {json.dumps(missing, ensure_ascii=False)}",
                assets=all_assets,
            )
            receipt.update(
                release_id=published["release"]["id"],
                release_url=published["release"]["html_url"],
                assets=published["assets"],
                verification_level="REMOTE_GIT_AND_ASSET_SHA256",
                publication_status="CODE_ONLY" if missing else "VERIFIED",
            )
            client.session.close()
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        receipt["reason"] = str(error)
        # Successful Git verification remains valid if a later release step fails.
        if receipt["publication_status"] == "CODE_ONLY":
            receipt["missing_assets"] += [
                {"name": row["planned_name"], "status": "NOT_VERIFIED"}
                for row in locals().get("assets", [])
            ]
    receipt["verified_at_utc"] = utc_now()
    write_json(receipt_path, receipt)
    return {
        "status": "BLOCKED"
        if receipt["publication_status"] == "BLOCKED"
        else "COMPLETE",
        **receipt,
        "receipt": "external:publication/PUBLICATION_RECEIPT.json",
    }


def verify_required_assets(required, remote, *, download, previous=()):
    """Metadata alone never establishes the content hash of a required asset."""
    if not required:
        raise PublicationError("Required release asset inventory is empty")
    expected_names = [row["planned_name"] for row in required]
    remote_names = [row["name"] for row in remote]
    if len(set(expected_names)) != len(expected_names) or len(set(remote_names)) != len(
        remote_names
    ):
        raise PublicationError("Release has duplicate asset names")
    missing = sorted(set(expected_names) - set(remote_names))
    if missing:
        raise PublicationError(
            "Required release assets are missing: " + ", ".join(missing)
        )
    by_name = {row["name"]: row for row in remote}
    already = {row["name"]: row for row in previous}
    verified = []
    for required_row in required:
        name = required_row["planned_name"]
        observed = by_name[name]
        if observed.get("size") != required_row["bytes"]:
            raise PublicationError(f"Remote asset bytes differ: {name}")
        digest = observed.get("digest")
        if digest is not None:
            if digest != "sha256:" + required_row["sha256"]:
                raise PublicationError(f"Remote asset digest differs: {name}")
            level = "REMOTE_API_SHA256"
        elif (
            name in already
            and already[name]["id"] == observed["id"]
            and already[name]["bytes"] == required_row["bytes"]
            and already[name]["sha256"] == required_row["sha256"]
            and already[name]["verification_level"]
            in {"REMOTE_DOWNLOAD_SHA256", "REMOTE_API_SHA256"}
        ):
            level = already[name]["verification_level"]
        else:
            content = download(observed)
            blocks = (content,) if isinstance(content, bytes) else content
            hasher, size = hashlib.sha256(), 0
            for block in blocks:
                hasher.update(block)
                size += len(block)
            if (
                size != required_row["bytes"]
                or hasher.hexdigest() != required_row["sha256"]
            ):
                raise PublicationError(f"Downloaded asset digest differs: {name}")
            level = "REMOTE_DOWNLOAD_SHA256"
        verified.append(
            {
                "name": name,
                "id": observed["id"],
                "bytes": observed["size"],
                "sha256": required_row["sha256"],
                "verification_level": level,
                "url": observed.get("browser_download_url"),
                "status": "VERIFIED",
                "role": required_row.get("role"),
            }
        )
    return verified


def publish_draft_assets(client, *, tag, commit, notes, assets):
    """Upload and verify a new draft before publication; never overwrite assets."""
    assets = validate_release_assets(assets)
    if not assets:
        raise PublicationError("Required release asset inventory is empty")
    release = client.create_draft(tag=tag, commit=commit, notes=notes)
    if (
        release.get("draft") is not True
        or release.get("prerelease") is not True
        or release.get("tag_name") != tag
        or release.get("target_commitish") != commit
    ):
        raise PublicationError("New draft release identity differs")
    for row in assets:
        try:
            client.upload(release, row)
        except PublicationError:
            # A failed response can follow a successful upload. Check before a
            # single retry, never clobber an asset or delete an unrelated draft.
            existing = [
                item
                for item in client.list_assets(release["id"])
                if item["name"] == row["planned_name"]
            ]
            if existing:
                verify_required_assets([row], existing, download=client.download)
            else:
                client.upload(release, row)
    verified = verify_required_assets(
        assets, client.list_assets(release["id"]), download=client.download
    )
    client.publish(release["id"])
    final = client.get_release(release["id"])
    if (
        final.get("draft") is not False
        or final.get("tag_name") != tag
        or final.get("target_commitish") != commit
    ):
        raise PublicationError("Published release identity differs")
    verified = verify_required_assets(
        assets,
        client.list_assets(release["id"]),
        download=client.download,
        previous=verified,
    )
    return {"release": final, "assets": verified}


class GitHubReleaseClient:
    """Existing GitHub authorization only; credentials never enter argv or logs."""

    def __init__(self, token: str):
        import requests

        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": "Bearer " + token,
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            }
        )
        self.root = f"https://api.github.com/repos/{REPOSITORY}"

    @classmethod
    def authorized(cls):
        token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
        if not token and shutil.which("gh"):
            result = subprocess.run(
                ["gh", "auth", "token", "--hostname", "github.com"],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                token = result.stdout.strip()
        if not token:
            return None
        client = cls(token)
        response = client.session.get(client.root, timeout=60)
        if response.status_code != 200 or not response.json().get(
            "permissions", {}
        ).get("push"):
            client.session.close()
            return None
        return client

    def _json(self, method, path, **kwargs):
        import requests

        try:
            response = self.session.request(
                method, self.root + path, timeout=300, **kwargs
            )
        except requests.RequestException as error:
            raise PublicationError(
                f"GitHub {method} {path} transport failed ({type(error).__name__})"
            ) from None
        if not response.ok:
            raise PublicationError(
                f"GitHub {method} {path} returned HTTP {response.status_code}"
            )
        return response.json()

    def create_draft(self, *, tag, commit, notes):
        # The caller has already verified the remote tag points to commit.
        return self._json(
            "POST",
            "/releases",
            json={
                "tag_name": tag,
                "target_commitish": commit,
                "name": tag,
                "body": notes,
                "draft": True,
                "prerelease": True,
                "make_latest": "false",
            },
        )

    def list_assets(self, release_id):
        rows, page = [], 1
        while True:
            chunk = self._json(
                "GET",
                f"/releases/{release_id}/assets",
                params={"per_page": 100, "page": page},
            )
            rows.extend(chunk)
            if len(chunk) < 100:
                return rows
            page += 1

    def upload(self, release, row):
        import requests

        url = release["upload_url"].split("{", 1)[0]
        if not url.startswith(
            "https://uploads.github.com/repos/" + REPOSITORY + "/releases/"
        ):
            raise PublicationError("GitHub upload endpoint differs")
        try:
            with Path(row["path"]).open("rb") as content:
                response = self.session.post(
                    url,
                    params={"name": row["planned_name"]},
                    data=content,
                    headers={"Content-Type": "application/octet-stream"},
                    timeout=1200,
                )
        except requests.RequestException as error:
            raise PublicationError(
                f"GitHub asset upload transport failed ({type(error).__name__})"
            ) from None
        if response.status_code != 201:
            raise PublicationError(
                f"GitHub asset upload returned HTTP {response.status_code}"
            )

    def download(self, row):
        response = self.session.get(
            self.root + f"/releases/assets/{row['id']}",
            headers={"Accept": "application/octet-stream"},
            stream=True,
            timeout=300,
        )
        if not response.ok:
            response.close()
            raise PublicationError("GitHub asset download failed")
        try:
            yield from response.iter_content(chunk_size=8 * 1024**2)
        finally:
            response.close()

    def publish(self, release_id):
        return self._json(
            "PATCH",
            f"/releases/{release_id}",
            json={"draft": False, "prerelease": True, "make_latest": "false"},
        )

    def get_release(self, release_id):
        return self._json("GET", f"/releases/{release_id}")
