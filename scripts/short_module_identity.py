"""Content identities for independently reusable prediction/label/result layers."""

import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def source_identity(paths):
    return {str(Path(path).resolve()): file_digest(path) for path in paths}


def assert_sources_current(identity):
    for path, expected in identity["sources"].items():
        if not Path(path).is_file() or file_digest(path) != expected:
            raise ValueError(f"stale dependency: {path}; use a new run/label version")


def metric_sources():
    import stmetrics.instances.evaluator
    import stmetrics.instances.matcher
    return [Path(inspect.getfile(module)) for module in
            (stmetrics.instances.evaluator, stmetrics.instances.matcher)]


def evaluation_identity(*, index, head_path, dataset_spec):
    paths = [PROJECT / name for name in (
        "scripts/short_module_identity.py", "scripts/short_module_evaluation.py",
        "scripts/short_module_native.py", "models/short_module_heads.py",
        "scripts/rescene_task_postprocess.py", "trainer/trainer.py",
        "scripts/p6a_metrics.py", "scripts/system_comparison_inference.py")]
    return {"schema": "short-module-evaluation-v2", "sources": source_identity(paths + metric_sources()),
            "versions": {name: importlib.metadata.version(name) for name in ("torch", "numpy", "scipy")},
            "dataset_spec_sha256": file_digest(dataset_spec),
            "input_index_sha256": identity_digest(index),
            "head_sha256": file_digest(head_path) if head_path else None}


def label_identity(*, prediction_identity, dataset_spec, geometry_policy="frozen-v1"):
    paths = [PROJECT / "scripts/short_module_data.py", PROJECT / "scripts/p6a_metrics.py",
             PROJECT / "scripts/short_module_identity.py"]
    return {"schema": "short-module-labels-v2", "sources": source_identity(paths + metric_sources()),
            "prediction_identity": prediction_identity, "geometry_policy": geometry_policy,
            "dataset_spec_sha256": file_digest(dataset_spec)}


def assert_export_current(root):
    root = Path(root)
    index = json.loads((root / "EXPORT_INDEX.json").read_text())
    if index["status"] != "COMPLETE":
        raise ValueError("export is incomplete")
    identity = json.loads((Path(index["cache"]) / "IDENTITY.json").read_text())
    assert_sources_current(identity)
    return index
