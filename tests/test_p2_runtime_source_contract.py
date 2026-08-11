import hashlib
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from utils import p2_preflight

RUNTIME_REPOSITORIES = ("concerto", "sonata", "detectron2", "stmetrics")


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _runtime_repo_fixture(tmp_path: Path, monkeypatch) -> dict[str, str]:
    expected: dict[str, str] = {}
    origins: dict[str, Path] = {}
    for name in RUNTIME_REPOSITORIES:
        root = tmp_path / "third_party" / name
        package = root / name
        package.mkdir(parents=True)
        origin = package / "__init__.py"
        origin.write_text(f'VERSION = "{name}"\n', encoding="utf-8")
        _git(root, "init", "-q")
        _git(root, "config", "user.email", "p2-contract@example.invalid")
        _git(root, "config", "user.name", "P2 Contract")
        _git(root, "add", ".")
        _git(root, "commit", "-qm", "fixture")
        expected[name] = _git(root, "rev-parse", "HEAD")
        origins[name] = origin

    monkeypatch.setattr(
        p2_preflight,
        "P2_RUNTIME_SOURCE_REPOSITORIES",
        {
            name: {
                "relative_root": f"third_party/{name}",
                "module": name,
                "expected_commit": expected[name],
            }
            for name in RUNTIME_REPOSITORIES
        },
        raising=False,
    )
    monkeypatch.setattr(
        p2_preflight.importlib.util,
        "find_spec",
        lambda name: SimpleNamespace(origin=str(origins[name])),
    )
    return expected


def _restore_index_flagged_file(
    root: Path,
    relative_path: str,
    original: bytes,
    clear_flag: str,
) -> None:
    (root / relative_path).write_bytes(original)
    _git(root, "update-index", clear_flag, relative_path)


def test_runtime_source_contract_binds_clean_pinned_repositories_and_origins(
    tmp_path: Path,
    monkeypatch,
) -> None:
    expected = _runtime_repo_fixture(tmp_path, monkeypatch)

    contract = p2_preflight.build_p2_runtime_source_contract(repo_root=tmp_path)

    assert contract["status"] == "pass"
    assert contract["errors"] == []
    assert set(contract["repositories"]) == set(RUNTIME_REPOSITORIES)
    for name, record in contract["repositories"].items():
        expected_tree_sha256 = record.pop("expected_tracked_tree_sha256")
        observed_tree_sha256 = record.pop("observed_tracked_tree_sha256")
        assert observed_tree_sha256 == expected_tree_sha256
        assert len(expected_tree_sha256) == 64
        assert record == {
            "reference": f"repo:third_party/{name}",
            "module": name,
            "expected_commit": expected[name],
            "observed_commit": expected[name],
            "module_origin_ref": f"repo:third_party/{name}/{name}/__init__.py",
            "dirty_paths": [],
            "index_flag_paths": [],
            "native_extensions": {},
            "status": "pass",
            "errors": [],
        }


@pytest.mark.parametrize("change_kind", ["tracked", "untracked"])
def test_runtime_source_contract_rejects_nested_worktree_changes(
    tmp_path: Path,
    monkeypatch,
    change_kind: str,
) -> None:
    _runtime_repo_fixture(tmp_path, monkeypatch)
    root = tmp_path / "third_party" / "concerto"
    if change_kind == "tracked":
        (root / "concerto" / "__init__.py").write_text(
            'VERSION = "changed"\n', encoding="utf-8"
        )
    else:
        (root / "concerto" / "runtime_override.py").write_text(
            "ENABLED = True\n", encoding="utf-8"
        )

    contract = p2_preflight.build_p2_runtime_source_contract(repo_root=tmp_path)

    assert contract["status"] == "fail"
    assert contract["repositories"]["concerto"]["status"] == "fail"
    assert "worktree_not_clean" in contract["repositories"]["concerto"][
        "errors"
    ]


@pytest.mark.parametrize(
    ("set_flag", "clear_flag"),
    [
        ("--assume-unchanged", "--no-assume-unchanged"),
        ("--skip-worktree", "--no-skip-worktree"),
    ],
)
def test_runtime_source_contract_rejects_hidden_tracked_changes(
    tmp_path: Path,
    monkeypatch,
    set_flag: str,
    clear_flag: str,
) -> None:
    _runtime_repo_fixture(tmp_path, monkeypatch)
    root = tmp_path / "third_party" / "concerto"
    relative_path = "concerto/__init__.py"
    source = root / relative_path
    original = source.read_bytes()
    _git(root, "update-index", set_flag, relative_path)
    try:
        source.write_text('VERSION = "hidden-change"\n', encoding="utf-8")

        contract = p2_preflight.build_p2_runtime_source_contract(
            repo_root=tmp_path
        )
    finally:
        _restore_index_flagged_file(
            root,
            relative_path,
            original,
            clear_flag,
        )

    record = contract["repositories"]["concerto"]
    assert contract["status"] == "fail"
    assert record["status"] == "fail"
    assert relative_path in record["index_flag_paths"]
    assert "index_flags_not_clean" in record["errors"]
    assert "tracked_tree_mismatch" in record["errors"]


@pytest.mark.parametrize(
    ("set_flag", "clear_flag"),
    [
        ("--assume-unchanged", "--no-assume-unchanged"),
        ("--skip-worktree", "--no-skip-worktree"),
    ],
)
def test_main_source_contract_rejects_hidden_tracked_changes(
    tmp_path: Path,
    set_flag: str,
    clear_flag: str,
) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    source = root / "runtime.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "p2-contract@example.invalid")
    _git(root, "config", "user.name", "P2 Contract")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "fixture")
    commit = _git(root, "rev-parse", "HEAD")
    original = source.read_bytes()
    _git(root, "update-index", set_flag, "runtime.py")
    try:
        source.write_text("VALUE = 2\n", encoding="utf-8")

        contract = p2_preflight.build_p2_source_tree_contract(
            source_commit=commit,
            repo_root=root,
        )
    finally:
        _restore_index_flagged_file(
            root,
            "runtime.py",
            original,
            clear_flag,
        )

    assert contract["status"] == "fail"
    assert contract["index_flag_paths"] == ["runtime.py"]
    assert contract["expected_tracked_tree_sha256"] != contract[
        "observed_tracked_tree_sha256"
    ]
    assert "index_flags_not_clean" in contract["errors"]
    assert "tracked_tree_mismatch" in contract["errors"]


def test_runtime_source_contract_rejects_nested_head_drift(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _runtime_repo_fixture(tmp_path, monkeypatch)
    root = tmp_path / "third_party" / "sonata"
    (root / "sonata" / "revision.py").write_text("REVISION = 2\n", encoding="utf-8")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "drift")

    contract = p2_preflight.build_p2_runtime_source_contract(repo_root=tmp_path)

    assert contract["status"] == "fail"
    assert "commit_mismatch" in contract["repositories"]["sonata"]["errors"]


def test_runtime_source_contract_rejects_module_origin_outside_pinned_root(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _runtime_repo_fixture(tmp_path, monkeypatch)
    external_origin = tmp_path / "site-packages" / "stmetrics" / "__init__.py"
    external_origin.parent.mkdir(parents=True)
    external_origin.write_text("VERSION = 'external'\n", encoding="utf-8")
    original_find_spec = p2_preflight.importlib.util.find_spec
    monkeypatch.setattr(
        p2_preflight.importlib.util,
        "find_spec",
        lambda name: (
            SimpleNamespace(origin=str(external_origin))
            if name == "stmetrics"
            else original_find_spec(name)
        ),
    )

    contract = p2_preflight.build_p2_runtime_source_contract(repo_root=tmp_path)

    assert contract["status"] == "fail"
    assert "module_origin_mismatch" in contract["repositories"]["stmetrics"][
        "errors"
    ]


def test_runtime_source_contract_rejects_ignored_shadow_origin_inside_repo(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _runtime_repo_fixture(tmp_path, monkeypatch)
    root = tmp_path / "third_party" / "concerto"
    shadow_origin = root / "shadow" / "concerto" / "__init__.py"
    shadow_origin.parent.mkdir(parents=True)
    shadow_origin.write_text("VERSION = 'shadow'\n", encoding="utf-8")
    (root / ".git" / "info" / "exclude").write_text(
        "shadow/\n", encoding="utf-8"
    )
    original_find_spec = p2_preflight.importlib.util.find_spec
    monkeypatch.setattr(
        p2_preflight.importlib.util,
        "find_spec",
        lambda name: (
            SimpleNamespace(origin=str(shadow_origin))
            if name == "concerto"
            else original_find_spec(name)
        ),
    )

    contract = p2_preflight.build_p2_runtime_source_contract(repo_root=tmp_path)

    assert contract["status"] == "fail"
    assert "module_origin_mismatch" in contract["repositories"]["concerto"][
        "errors"
    ]


def test_runtime_source_contract_hashes_loaded_ignored_native_extensions(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _runtime_repo_fixture(tmp_path, monkeypatch)
    extension = (
        tmp_path
        / "third_party"
        / "detectron2"
        / "detectron2"
        / "_C.cpython-310-x86_64-linux-gnu.so"
    )
    (extension.parents[1] / ".git" / "info" / "exclude").write_text(
        "*.so\n", encoding="utf-8"
    )
    extension.write_bytes(b"pinned-native-extension")
    definitions = dict(p2_preflight.P2_RUNTIME_SOURCE_REPOSITORIES)
    definitions["detectron2"] = {
        **definitions["detectron2"],
        "native_extensions": {
            "detectron2._C": {
                "relative_path": extension.relative_to(
                    tmp_path / "third_party" / "detectron2"
                ).as_posix(),
                "expected_byte_size": extension.stat().st_size,
                "expected_sha256": hashlib.sha256(extension.read_bytes()).hexdigest(),
            }
        },
    }
    monkeypatch.setattr(
        p2_preflight,
        "P2_RUNTIME_SOURCE_REPOSITORIES",
        definitions,
    )
    original_find_spec = p2_preflight.importlib.util.find_spec
    monkeypatch.setattr(
        p2_preflight.importlib.util,
        "find_spec",
        lambda name: (
            SimpleNamespace(origin=str(extension))
            if name == "detectron2._C"
            else original_find_spec(name)
        ),
    )

    passing = p2_preflight.build_p2_runtime_source_contract(repo_root=tmp_path)
    extension.write_bytes(b"replaced-native-extension")
    failing = p2_preflight.build_p2_runtime_source_contract(repo_root=tmp_path)

    assert passing["status"] == "pass"
    native = passing["repositories"]["detectron2"]["native_extensions"][
        "detectron2._C"
    ]
    assert native["status"] == "pass"
    assert native["observed_sha256"] == native["expected_sha256"]
    assert failing["status"] == "fail"
    assert "native_extension_mismatch" in failing["repositories"]["detectron2"][
        "errors"
    ]
