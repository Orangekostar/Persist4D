"""Backbone registry. MinkowskiEngine and Pointcept (sonata/concerto) are both
optional; if a family's deps aren't installed, its class names become stubs
that raise ImportError with install instructions on instantiation.
"""

from __future__ import annotations

import importlib
import logging
import sys
from collections.abc import Iterable

logger = logging.getLogger(__name__)

_MODELS: dict[str, type] = {}
_DISABLED: dict[str, str] = {}


def _try_import(path: str):
    """Returns (module, None) or (None, error_msg). Catches only ImportError."""
    try:
        return importlib.import_module(path), None
    except ImportError as e:
        return None, str(e)


def _load_family(modules: Iterable[str], explicit: dict[str, str], hint: str) -> None:
    """Import modules in this family; on first ImportError, disable every class
    name listed in ``explicit`` with the given hint. ``explicit`` maps class
    name -> attribute path on one of the family's modules."""
    loaded: dict[str, object] = {}
    for path in modules:
        mod, err = _try_import(path)
        if mod is None:
            logger.info("%s unavailable: %s", path, err)
            _DISABLED.update({n: f"{hint} (import failed: {err})" for n in explicit})
            return
        loaded[path] = mod
        for attr in dir(mod):
            if "Net" in attr:
                _MODELS[attr] = getattr(mod, attr)
    for cls_name, dotted in explicit.items():
        mod_path, attr = dotted.rsplit(".", 1)
        _MODELS[cls_name] = getattr(loaded[mod_path], attr)


# Always-available
_load_family(
    modules=[
        "models.rescene",
        "models.persist4d_allt",
        "models.persist4d_task_memory",
    ],
    explicit={
        "Persist4DAllT": "models.persist4d_allt.Persist4DAllT",
        "Persist4DTaskMemory": (
            "models.persist4d_task_memory.Persist4DTaskMemory"
        ),
        "ReScene": "models.rescene.ReScene",
    },
    hint="",
)

# Pointcept (sonata / concerto): needs torch_scatter, addict, huggingface_hub.
_load_family(
    modules=["models.pointcept"],
    explicit={
        "PointceptBackbone": "models.pointcept.PointceptBackbone",
        "PointceptBackboneEncOnly": "models.pointcept.PointceptBackboneEncOnly",
    },
    hint="install addict / huggingface_hub / torch_scatter, or use the Minkowski backbone",
)

# Minkowski: needs the custom-CUDA MinkowskiEngine build.
_load_family(
    modules=["models.resunet", "models.minkowski"],
    explicit={},
    hint="install MinkowskiEngine (see README), or use a Pointcept backbone (sonata / concerto)",
)


# Expose every registered name (or stub) at the package level so
# Hydra's `_target_: models.<ClassName>` resolves uniformly.
def _stub(name: str, reason: str) -> type:
    def __init__(self, *a, **k):
        raise ImportError(f"backbone '{name}' is unavailable: {reason}")
    return type(name, (), {"__init__": __init__})

_pkg = sys.modules[__name__]
for _n, _c in _MODELS.items():
    setattr(_pkg, _n, _c)
for _n, _r in _DISABLED.items():
    if _n not in _MODELS:
        setattr(_pkg, _n, _stub(_n, _r))

import models.task_memory_criterion as _task_memory_criterion
import models.task_memory_routing as _task_memory_routing
import models.task_memory_state as _task_memory_state
import models.task_memory_supervision as _task_memory_supervision

CommitResult = _task_memory_routing.CommitResult
EntityRoute = _task_memory_routing.EntityRoute
PredictionObservation = _task_memory_routing.PredictionObservation
commit_entities = _task_memory_routing.commit_entities
route_entities = _task_memory_routing.route_entities
TaskMemoryConfig = _task_memory_state.TaskMemoryConfig
TaskMemoryState = _task_memory_state.TaskMemoryState
TaskMemoryCriterion = _task_memory_criterion.TaskMemoryCriterion
TrainingIdentityLedger = _task_memory_supervision.TrainingIdentityLedger
build_tala_assignment = _task_memory_supervision.build_tala_assignment


def get_models() -> list[type]:
    return list(_MODELS.values())


def available_backbones() -> list[str]:
    return sorted(_MODELS)


def disabled_backbones() -> dict[str, str]:
    return dict(_DISABLED)


def load_model(name: str):
    if name in _MODELS:
        return _MODELS[name]
    if name in _DISABLED:
        raise ImportError(f"model '{name}' is unavailable: {_DISABLED[name]}")
    raise ValueError(
        f"unknown model '{name}'. Available: {available_backbones()}. "
        f"Disabled: {sorted(_DISABLED)}"
    )
