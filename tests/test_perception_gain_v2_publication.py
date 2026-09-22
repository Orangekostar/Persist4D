import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from scripts.perception_gain_publish import choose_release_tag
from scripts.perception_gain_v2_publication import (
    PublicationError,
    artifact_manifest,
    publish_draft_assets,
    selected_release_methods,
    snapshot_paths,
    verify_required_assets,
)


def record(tmp_path, name, value):
    path = tmp_path / name
    path.write_bytes(value)
    return {
        "planned_name": name,
        "path": path,
        "bytes": len(value),
        "sha256": hashlib.sha256(value).hexdigest(),
    }


def test_v2_tag_never_moves_v1_or_occupied_v2_tags():
    assert (
        choose_release_tag(
            {"persist4d-perception-gain-v1"}, base_tag="persist4d-perception-gain-v2"
        )
        == "persist4d-perception-gain-v2"
    )
    assert (
        choose_release_tag(
            {"persist4d-perception-gain-v2", "persist4d-perception-gain-v2-r3"},
            base_tag="persist4d-perception-gain-v2",
        )
        == "persist4d-perception-gain-v2-r2"
    )


def test_existing_release_with_empty_or_incomplete_assets_is_not_verified(tmp_path):
    required = [
        record(tmp_path, "weights.pt", b"weights"),
        record(tmp_path, "results.zip", b"results"),
    ]
    with pytest.raises(PublicationError, match="missing"):
        verify_required_assets(required, [], download=lambda _: b"")
    remote = [
        {
            "name": "weights.pt",
            "size": 7,
            "digest": "sha256:" + required[0]["sha256"],
            "id": 1,
        }
    ]
    with pytest.raises(PublicationError, match="missing"):
        verify_required_assets(required, remote, download=lambda _: b"")
    remote.append(
        {"name": "results.zip", "size": 7, "digest": "sha256:" + "0" * 64, "id": 2}
    )
    with pytest.raises(PublicationError, match="digest"):
        verify_required_assets(required, remote, download=lambda _: b"")


class FakeRelease:
    def __init__(self, *, corrupt=False):
        self.events, self.assets, self.data = [], [], {}
        self.corrupt = corrupt
        self.release = {
            "id": 7,
            "tag_name": "t",
            "target_commitish": "a" * 40,
            "draft": True,
            "prerelease": True,
            "html_url": "https://github.com/o/r/releases/tag/t",
        }

    def create_draft(self, **kwargs):
        self.events.append(("create", kwargs))
        return dict(self.release)

    def list_assets(self, release_id):
        self.events.append(("list", release_id))
        return list(self.assets)

    def upload(self, release, row):
        self.events.append(("upload", row["planned_name"]))
        asset_id = len(self.assets) + 1
        self.data[asset_id] = Path(row["path"]).read_bytes()
        self.assets.append(
            {
                "id": asset_id,
                "name": row["planned_name"],
                "size": row["bytes"],
                "digest": None,
                "browser_download_url": "https://example.org/" + row["planned_name"],
            }
        )

    def download(self, row):
        self.events.append(("download", row["id"]))
        return b"broken" if self.corrupt else self.data[row["id"]]

    def publish(self, release_id):
        self.events.append(("publish", release_id))
        self.release["draft"] = False

    def get_release(self, release_id):
        self.events.append(("get", release_id))
        return dict(self.release)


def test_draft_is_published_only_after_roundtrip_hashes_and_final_read(tmp_path):
    client = FakeRelease()
    required = [
        record(tmp_path, "weights.pt", b"weights"),
        record(tmp_path, "results.zip", b"results"),
    ]
    result = publish_draft_assets(
        client,
        tag="t",
        commit="a" * 40,
        notes="Actual negative result",
        assets=required,
    )
    events = [item[0] for item in client.events]
    assert events.index("publish") > max(
        i for i, event in enumerate(events) if event == "download"
    )
    assert events[-2:] == ["get", "list"]
    assert result["release"]["draft"] is False
    assert all(
        row["verification_level"] == "REMOTE_DOWNLOAD_SHA256"
        for row in result["assets"]
    )
    assert not any(
        item[1] == "PUBLICATION_RECEIPT.json"
        for item in client.events
        if item[0] == "upload"
    )


def test_corrupt_download_keeps_release_in_draft(tmp_path):
    client = FakeRelease(corrupt=True)
    with pytest.raises(PublicationError, match="digest"):
        publish_draft_assets(
            client,
            tag="t",
            commit="a" * 40,
            notes="Evidence",
            assets=[record(tmp_path, "weights.pt", b"weights")],
        )
    assert client.release["draft"] is True
    assert "publish" not in [event[0] for event in client.events]


def test_locked_pair_inventory_includes_matched_new_and_c0():
    methods = {
        name: {"method_id": name, "inference_identity": name}
        for name in ("pair", "new", "parent", "c0")
    }
    lock = {
        "final_method_id": "pair",
        "aliases": {
            "FINAL": "pair",
            "P-D0": "parent",
            "C0-best-D0": "c0",
            "FINAL-NEW-control": "new",
        },
        "methods": methods,
    }
    assert {row["method_id"] for row in selected_release_methods(lock)} == {
        "pair",
        "new",
        "parent",
        "c0",
    }


def test_snapshot_manifest_excludes_mutable_state_and_self_but_keeps_results(tmp_path):
    artifacts = tmp_path / "artifacts/perception_gain_v2"
    artifacts.mkdir(parents=True)
    for name in (
        "RUN_STATE.json",
        "FINAL_REPORT.md",
        "EXECUTION_LOG.jsonl",
        "ARTIFACT_MANIFEST.json",
    ):
        (artifacts / name).write_text("{}")
    unrelated = tmp_path / "unrelated.txt"
    unrelated.write_text("user data")
    paths = snapshot_paths(artifacts, tmp_path)
    assert all("unrelated" not in path for path in paths)
    result = artifact_manifest(paths, project_root=tmp_path, experiment_commit="a" * 40)
    assert [row["path"] for row in result["files"]] == [
        "artifacts/perception_gain_v2/FINAL_REPORT.md"
    ]
    (artifacts / "raw.ckpt").write_bytes(b"base")
    with pytest.raises(PublicationError, match="Unapproved"):
        snapshot_paths(artifacts, tmp_path)


def test_code_only_push_freezes_a_then_b_without_rewriting_state(tmp_path, monkeypatch):
    import scripts.perception_gain_publish as legacy
    import scripts.perception_gain_v2_publication as publication
    import scripts.perception_gain_v2_report as reporting

    repo, remote, external = (
        tmp_path / "repo",
        tmp_path / "remote.git",
        tmp_path / "external",
    )
    repo.mkdir()
    external.mkdir()

    def run(command, *, check=True, **kwargs):
        result = subprocess.run(
            command, cwd=repo, capture_output=True, text=True, check=False
        )
        if check and result.returncode:
            raise PublicationError(result.stderr)
        return result

    run(["git", "init", "-q", "--initial-branch", publication.BRANCH])
    run(["git", "config", "user.name", "Publication test"])
    run(["git", "config", "user.email", "test@example.invalid"])
    run(["git", "init", "-q", "--bare", str(remote)])
    run(["git", "remote", "add", "origin", str(remote)])
    (repo / "source.py").write_text("pass\n")
    run(["git", "add", "--", "source.py"])
    run(["git", "commit", "-qm", "Reviewed source"])
    artifacts = repo / "artifacts/perception_gain_v2"
    artifacts.mkdir(parents=True)
    state = json.dumps({"tasks": {"REPORT": {"status": "COMPLETE"}}})
    (artifacts / "RUN_STATE.json").write_text(state)
    (artifacts / "FINAL_REPORT.md").write_text("Actual partial results\n")
    (artifacts / "HANDOFF.md").write_text("Actual recovery commands\n")
    (repo / "unrelated.txt").write_text("Preserve user file\n")
    monkeypatch.setattr(publication, "PROJECT_ROOT", repo)
    monkeypatch.setattr(publication, "_run", run)
    monkeypatch.setattr(legacy, "_run", run)
    monkeypatch.setattr(publication, "prepare_model_assets", lambda *args: ([], []))
    monkeypatch.setattr(publication.GitHubReleaseClient, "authorized", lambda: None)
    monkeypatch.setattr(reporting, "run_report", lambda *args, **kwargs: None)
    result = publication.publish_v2(
        {"artifact_root": "artifacts/perception_gain_v2", "branch": publication.BRANCH},
        external_root=external,
    )
    assert result["publication_status"] == "CODE_ONLY", result
    assert result["experiment_commit"] != result["publication_commit"]
    assert (
        result["remote_branch_sha"]
        == result["remote_tag_sha"]
        == result["publication_commit"]
    )
    assert (
        run(
            [
                "git",
                "show",
                result["publication_commit"]
                + ":artifacts/perception_gain_v2/RUN_STATE.json",
            ]
        ).stdout
        == state
    )
    assert run(["git", "ls-files", "--", "unrelated.txt"]).stdout == ""
    assert result["experiment_commit"] in (artifacts / "HANDOFF.md").read_text()
    assert "NOT_UPLOADED" in {row["status"] for row in result["missing_assets"]}
    assert not (artifacts / "PUBLICATION_RECEIPT.json").exists()
