"""Evaluation dependency closure, including official AP math and metric state."""

import importlib.metadata
from pathlib import Path

import stmetrics

from scripts.short_module_identity import evaluation_identity as base_identity
from scripts.short_module_identity import source_identity


def evaluation_identity(**kwargs):
    identity = base_identity(**kwargs)
    package = Path(stmetrics.__file__).resolve().parent
    identity["sources"].update(source_identity([Path(__file__), *sorted(package.rglob("*.py"))]))
    for name in ("torchmetrics", "stmetrics"):
        try:
            identity["versions"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            identity["versions"][name] = "source-tree-pinned-by-SHA256"
    return identity
