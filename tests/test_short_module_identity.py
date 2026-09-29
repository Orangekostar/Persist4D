import pytest

from scripts.short_module_identity import (
    assert_sources_current,
    identity_digest,
    source_identity,
)


def test_completed_cache_cannot_hide_changed_direct_dependency(tmp_path):
    source = tmp_path / "apply_module.py"
    source.write_text("old implementation")
    identity = {"status": "COMPLETE", "sources": source_identity([source])}
    assert_sources_current(identity)
    source.write_text("new implementation")
    with pytest.raises(ValueError, match="stale dependency"):
        assert_sources_current(identity)


def test_label_change_does_not_change_prediction_identity(tmp_path):
    producer = tmp_path / "producer.py"
    labels = tmp_path / "labels.py"
    producer.write_text("frozen R1")
    labels.write_text("v1")
    prediction_id = identity_digest(source_identity([producer]))
    label_id = identity_digest(source_identity([labels]))
    labels.write_text("threshold validity v2")
    assert identity_digest(source_identity([producer])) == prediction_id
    assert identity_digest(source_identity([labels])) != label_id


def test_evaluation_digest_covers_real_dependency_files_weights_and_spec(tmp_path, monkeypatch):
    from scripts import short_module_identity as identities

    monkeypatch.setattr(identities, "PROJECT", tmp_path)
    files = ["scripts/short_module_identity.py", "scripts/short_module_evaluation.py",
             "scripts/short_module_native.py", "models/short_module_heads.py",
             "scripts/rescene_task_postprocess.py", "trainer/trainer.py",
             "scripts/p6a_metrics.py", "scripts/system_comparison_inference.py",
             "matcher.py", "evaluator.py", "dataset.yaml", "head.pt"]
    for name in files:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("original")
    monkeypatch.setattr(identities, "metric_sources", lambda: [tmp_path / "matcher.py", tmp_path / "evaluator.py"])

    def current(index=None):
        return identity_digest(identities.evaluation_identity(
            index=index or {"prediction": "frozen"}, head_path=tmp_path / "head.pt",
            dataset_spec=tmp_path / "dataset.yaml"))

    baseline = current()
    for name in files:
        path = tmp_path / name
        path.write_text("changed")
        assert current() != baseline, name
        path.write_text("original")
    assert current({"prediction": "changed"}) != baseline


def test_resume_checks_current_checkout_even_when_old_checkout_and_publish_state_exist(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from scripts import short_module_screen as screen

    old = tmp_path / "old/scripts/short_module_native.py"
    current = tmp_path / "new/scripts/short_module_native.py"
    for path in (old, current):
        path.parent.mkdir(parents=True, exist_ok=True)
    old.write_text("baseline")
    current.write_text("changed current producer")
    root = tmp_path / "run"
    cache = root / "cache"
    screen.write_json(root / "EXPORT_INDEX.json", {"status": "COMPLETE", "cache": str(cache)})
    screen.write_json(cache / "IDENTITY.json", {"sources": source_identity([old])})
    screen.write_json(root / "publication/PUBLISH_STATE.json", {"status": "pending"})
    monkeypatch.setattr(screen, "PROJECT", tmp_path / "new")

    def forbidden(*args, **kwargs):
        raise AssertionError("stale computation reached a child stage")

    monkeypatch.setattr(screen.subprocess, "run", forbidden)
    with pytest.raises(ValueError, match="stale dependency"):
        screen.run_pipeline(SimpleNamespace(root=root, resume=True))


def test_official_ap_math_state_and_dispatch_changes_invalidate_results(tmp_path, monkeypatch):
    from scripts import short_module_eval_identity as identity

    package = tmp_path / "metric_package"
    package.mkdir()
    init = package / "__init__.py"
    init.write_text("original package")
    monkeypatch.setattr(identity.stmetrics, "__file__", str(init))
    paths = [package / name for name in ("ap_math.py", "state.py", "instance_metrics.py")]
    for path in paths:
        path.write_text("original")
    spec = tmp_path / "dataset.yaml"
    spec.write_text("same taxonomy")

    def digest():
        return identity_digest(identity.evaluation_identity(index={"input": "same"},
                                                           head_path=None, dataset_spec=spec))

    original = digest()
    for path in paths:
        path.write_text("changed official helper")
        assert digest() != original
        path.write_text("original")
