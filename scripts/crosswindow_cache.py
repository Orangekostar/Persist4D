"""Asset and data-role contracts for the CrossWindow evidence campaign."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class CrossWindowCacheError(ValueError):
    """Raised when cache provenance or data roles violate the campaign contract."""


ASSET_KEYS = (
    "data_root",
    "rio_metadata",
    "r1_checkpoint",
    "concerto_pretrained",
    "metric_dataset_spec",
    "dev_base_cache_root",
    "dev_supplement_root",
    "pb_base_cache_root",
    "pb_supplement_root",
    "fh_native_cache_root",
    "train_observation_cache_root",
    "external_run_root",
)

ASSET_ENVIRONMENT = {
    "data_root": "PERSIST4D_DATA_ROOT",
    "rio_metadata": "PERSIST4D_RIO_METADATA",
    "r1_checkpoint": "PERSIST4D_R1_CHECKPOINT",
    "concerto_pretrained": "PERSIST4D_CONCERTO_PRETRAINED",
    "metric_dataset_spec": "PERSIST4D_METRIC_DATASET_SPEC",
    "dev_base_cache_root": "PERSIST4D_DEV_BASE_CACHE_ROOT",
    "dev_supplement_root": "PERSIST4D_DEV_SUPPLEMENT_ROOT",
    "pb_base_cache_root": "PERSIST4D_PB_BASE_CACHE_ROOT",
    "pb_supplement_root": "PERSIST4D_PB_SUPPLEMENT_ROOT",
    "fh_native_cache_root": "PERSIST4D_FH_NATIVE_CACHE_ROOT",
    "train_observation_cache_root": "PERSIST4D_TRAIN_OBSERVATION_CACHE_ROOT",
    "external_run_root": "PERSIST4D_RUN_ROOT",
}

_FALLBACK_ALIASES = {
    "data_root": ("data_root", "external:data_root"),
    "rio_metadata": ("rio_metadata", "external:rio_metadata"),
    "r1_checkpoint": ("r1_checkpoint", "external:r1_checkpoint"),
    "concerto_pretrained": (
        "concerto_pretrained",
        "external:concerto_pretrained",
    ),
    "metric_dataset_spec": ("metric_dataset_spec",),
    "dev_base_cache_root": ("dev_base_cache_root",),
    "dev_supplement_root": ("dev_supplement_root",),
    "pb_base_cache_root": ("pb_base_cache_root",),
    "pb_supplement_root": ("pb_supplement_root",),
    "fh_native_cache_root": ("fh_native_cache_root",),
    "train_observation_cache_root": ("train_observation_cache_root",),
    "external_run_root": ("external_run_root", "external:run_root"),
}


@dataclass(frozen=True)
class AssetResolution:
    values: dict[str, str | None]
    sources: dict[str, str]


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise CrossWindowCacheError("asset values must be non-empty strings or null")
    return value


def resolve_assets(
    *,
    explicit: Mapping[str, Any],
    environ: Mapping[str, str],
    fallback: Mapping[str, Any],
) -> AssetResolution:
    """Resolve asset values without probing paths or leaking them into Git artifacts."""
    unknown = set(explicit) - set(ASSET_KEYS)
    if unknown:
        raise CrossWindowCacheError(f"unknown explicit asset keys: {sorted(unknown)}")
    values: dict[str, str | None] = {}
    sources: dict[str, str] = {}
    for key in ASSET_KEYS:
        explicit_value = _optional_text(explicit.get(key))
        environment_value = _optional_text(environ.get(ASSET_ENVIRONMENT[key]))
        fallback_value = None
        for alias in _FALLBACK_ALIASES[key]:
            if alias in fallback and fallback[alias] is not None:
                fallback_value = _optional_text(fallback[alias])
                break
        if explicit_value is not None:
            values[key], sources[key] = explicit_value, "cli"
        elif environment_value is not None:
            values[key], sources[key] = environment_value, "environment"
        elif fallback_value is not None:
            values[key], sources[key] = fallback_value, "fallback"
        else:
            values[key], sources[key] = None, "unresolved"
    if values["metric_dataset_spec"] is None and values["data_root"] is not None:
        data_root = Path(values["data_root"])
        candidates = (
            data_root / "processed/rio/rio.yaml",
            data_root / "data/processed/rio/rio.yaml",
        )
        specification = next((path for path in candidates if path.is_file()), None)
        if specification is not None:
            values["metric_dataset_spec"] = str(specification)
            sources["metric_dataset_spec"] = "derived:data_root"
    return AssetResolution(values=values, sources=sources)


def build_data_roles(
    data_contract: Mapping[str, Any],
    *,
    available_references: Sequence[str],
) -> dict[str, list[str]]:
    """Build the preregistered reference split without splitting a reference."""
    inventory = data_contract.get("native_reference_inventory")
    if isinstance(inventory, (str, bytes)) or not isinstance(inventory, Sequence):
        raise CrossWindowCacheError("data contract lacks native_reference_inventory")
    available = set(available_references)
    if len(available) != len(available_references) or any(
        not isinstance(value, str) or not value for value in available_references
    ):
        raise CrossWindowCacheError(
            "available references must be unique non-empty strings"
        )
    by_role: dict[str, list[str]] = {
        "development": [],
        "adaptation": [],
        "additional_native_refs": [],
    }
    seen: set[str] = set()
    for record in inventory:
        if not isinstance(record, Mapping):
            raise CrossWindowCacheError("reference inventory entries must be mappings")
        reference = record.get("reference_id")
        role = record.get("role")
        if not isinstance(reference, str) or not reference:
            raise CrossWindowCacheError("reference inventory contains an invalid ID")
        if reference in seen:
            raise CrossWindowCacheError("reference inventory contains duplicate IDs")
        seen.add(reference)
        if role in by_role:
            by_role[str(role)].append(reference)
    development = [value for value in by_role["development"] if value in available]
    development.sort(
        key=lambda value: hashlib.sha256(f"crosswindow-v1:{value}".encode()).hexdigest()
    )
    boundary = len(development) // 2
    return {
        "DEV-CAL": development[:boundary],
        "DEV-SEL": development[boundary:],
        "adaptation": sorted(by_role["adaptation"]),
        "additional_native_refs": sorted(by_role["additional_native_refs"]),
    }


__all__ = [
    "ASSET_ENVIRONMENT",
    "ASSET_KEYS",
    "AssetResolution",
    "CrossWindowCacheError",
    "build_data_roles",
    "resolve_assets",
]
