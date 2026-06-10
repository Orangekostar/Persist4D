"""Backbone registry. MinkowskiEngine and Pointcept (sonata/concerto) are both
optional; if a family's deps aren't installed, its class names become stubs
that raise ImportError with install instructions on instantiation.
"""

from __future__ import annotations

import importlib
import logging
import sys
from typing import Dict, Iterable, List

logger = logging.getLogger(__name__)

_MODELS: Dict[str, type] = {}
_DISABLED: Dict[str, str] = {}


def _try_import(path: str):
    """Returns (module, None) or (None, error_msg). Catches only ImportError."""
    try:
        return importlib.import_module(path), None
    except ImportError as e:
        return None, str(e)


def _load_family(modules: Iterable[str], explicit: Dict[str, str], hint: str) -> None:
    """Import modules in this family; on first ImportError, disable every class
    name listed in ``explicit`` with the given hint. ``explicit`` maps class
    name -> attribute path on one of the family's modules."""
    loaded: Dict[str, object] = {}
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
    modules=["models.rescene"],
    explicit={"ReScene": "models.rescene.ReScene"},
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


def get_models() -> List[type]:
    return list(_MODELS.values())


def available_backbones() -> List[str]:
    return sorted(_MODELS)


def disabled_backbones() -> Dict[str, str]:
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
