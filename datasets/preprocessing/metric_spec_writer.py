"""Writes the metric spec yaml that ``stmetrics.instances`` consumes.

Called from the preprocessing entry points (``RScan_preprocessing.py``,
``scannet_preprocessing.py``) so the spec stays in sync with the data the
preprocessing actually produced — no risk of label-list drift between the
two.

The yaml format is intentionally minimal so adding a new dataset is just
producing this file at preprocessing time and loading it via
``load_dataset_spec(name)``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

import yaml


# Default location the metric loader looks in. Override at call time for
# custom output paths.
DEFAULT_SPECS_DIR = Path(__file__).resolve().parent / "metric_specs"


# The two canonical "small" label sets used by the rio / scannet (non-200)
# variants. Kept here, not duplicated in both preprocessing scripts, so a
# bugfix only needs to happen in one place.
NYU40_SUBSET_18_LABELS: List[str] = [
    "cabinet", "bed", "chair", "sofa", "table", "door", "window",
    "bookshelf", "picture", "counter", "desk", "curtain",
    "refrigerator", "shower curtain", "toilet", "sink", "bathtub",
    "otherfurniture",
]
NYU40_SUBSET_18_IDS: List[int] = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]


def write_metric_spec(
    name: str,
    class_labels: Iterable[str],
    valid_class_ids: Iterable[int],
    aux_labels: Iterable[str],
    valid_aux_ids: Iterable[int],
    aux: str = "changes",
    categories: Optional[Dict[str, Iterable[str]]] = None,
    out_dir: Optional[Path] = None,
) -> Path:
    """Write ``<out_dir>/<name>.yaml`` and return the path.

    The schema matches what ``stmetrics.instances.DatasetSpec`` reads:
    ``aux_labels`` / ``valid_aux_ids`` / ``aux`` (the target dict key).

    Category memberships are filtered to only include labels present in
    ``class_labels`` so the spec stays self-consistent (the metric loader
    asserts this).
    """
    out_dir = Path(out_dir) if out_dir is not None else DEFAULT_SPECS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    class_labels = list(class_labels)
    valid_class_ids = [int(i) for i in valid_class_ids]
    aux_labels = list(aux_labels)
    valid_aux_ids = [int(i) for i in valid_aux_ids]

    assert len(class_labels) == len(valid_class_ids), (
        f"write_metric_spec({name}): class_labels has {len(class_labels)} "
        f"entries but valid_class_ids has {len(valid_class_ids)}"
    )
    assert len(aux_labels) == len(valid_aux_ids), (
        f"write_metric_spec({name}): aux_labels has {len(aux_labels)} "
        f"entries but valid_aux_ids has {len(valid_aux_ids)}"
    )

    spec: Dict = {
        "name": name,
        "class_labels": class_labels,
        "valid_class_ids": valid_class_ids,
        "aux": aux,
        "aux_labels": aux_labels,
        "valid_aux_ids": valid_aux_ids,
    }

    if categories:
        cl_set = set(class_labels)
        filtered: Dict[str, List[str]] = {}
        for cat_name, members in categories.items():
            kept = [m for m in members if m in cl_set]
            if kept:
                filtered[cat_name] = kept
        if filtered:
            spec["categories"] = filtered

    path = out_dir / f"{name}.yaml"
    with path.open("w") as f:
        yaml.safe_dump(spec, f, sort_keys=False, default_flow_style=False)
    return path


def resolve_200_subset(
    full_labels: Iterable[str],
    full_ids: Iterable[int],
    candidate_labels: Iterable[str],
    panoptic_ids: Iterable[int],
):
    """Filter the validation label set to (label, id) pairs that exist in the
    full 200-class index AND are not panoptic-stuff classes.

    Mirrors the resolution previously done inline in the metric ``state.py``
    so preprocessing can produce the exact subset the metric expects.
    """
    label_to_id = dict(zip(full_labels, full_ids))
    panoptic = set(int(i) for i in panoptic_ids)
    out_labels: List[str] = []
    out_ids: List[int] = []
    for l in candidate_labels:
        if l in label_to_id and int(label_to_id[l]) not in panoptic:
            out_labels.append(l)
            out_ids.append(int(label_to_id[l]))
    return out_labels, out_ids
