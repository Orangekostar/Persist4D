"""Lossless, deterministic prediction exports with no ground-truth fields."""

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
import zipfile
from pathlib import Path

import numpy as np

from scripts.perception_gain_v2 import file_hash, read_json, write_json
from scripts.perception_gain_v2_config import content_hash

SCHEMA = "perception-gain-prediction-export-v2"
IDENTITY_FIELDS = {
    "method_id",
    "data_role",
    "logical_unit_id",
    "T",
    "eval_seed",
    "scan_ids",
    "inference_identity",
    "lock_sha256",
}
MASK_DTYPES = {"bool", "float32", "float64", "int32", "int64"}


def encode_prediction(prediction, *, identity, scan_vertex_offsets):
    import torch

    if set(identity) != IDENTITY_FIELDS or len(identity["scan_ids"]) != identity["T"]:
        raise ValueError("Prediction identity differs from the export contract")
    masks, scores, classes = (
        prediction[key].detach().cpu().contiguous()
        for key in ("pred_masks", "pred_scores", "pred_classes")
    )
    offsets = np.asarray(scan_vertex_offsets, dtype=np.int64)
    dtype = str(masks.dtype).removeprefix("torch.")
    if (
        dtype not in MASK_DTYPES
        or masks.ndim != 2
        or scores.shape != (masks.shape[1],)
        or classes.shape != scores.shape
        or offsets.shape != (identity["T"] + 1,)
        or offsets[0] != 0
        or offsets[-1] != masks.shape[0]
        or np.any(np.diff(offsets) < 0)
        or not torch.all((masks == 0) | (masks == 1))
        or not torch.isfinite(scores).all()
        or classes.dtype not in {torch.int32, torch.int64}
    ):
        raise ValueError("Prediction arrays or per-scan point counts differ")
    metadata = {
        "schema_version": SCHEMA,
        "identity": identity,
        "mask_shape": list(masks.shape),
        "mask_dtype": dtype,
        "mask_order": "C",
        "bitorder": "little",
        "point_order": "canonical_vertex_order_within_each_observed_scan",
        "contains_ground_truth": False,
    }
    arrays = {
        "metadata": np.frombuffer(
            json.dumps(metadata, sort_keys=True, allow_nan=False).encode(),
            dtype=np.uint8,
        ),
        "masks_packed": np.packbits(
            masks.numpy().astype(np.bool_).reshape(-1), bitorder="little"
        ),
        "scores": scores.numpy(),
        "classes": classes.numpy(),
        "scan_vertex_offsets": offsets,
    }
    output = io.BytesIO()
    with zipfile.ZipFile(
        output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for name, array in arrays.items():
            encoded = io.BytesIO()
            np.lib.format.write_array(encoded, array, allow_pickle=False)
            member = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            member.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(member, encoded.getvalue(), compresslevel=6)
    return output.getvalue()


def decode_prediction(source):
    import torch

    with np.load(
        io.BytesIO(source) if isinstance(source, bytes) else source, allow_pickle=False
    ) as archive:
        metadata = json.loads(archive["metadata"].tobytes())
        if (
            metadata.get("schema_version") != SCHEMA
            or metadata.get("mask_dtype") not in MASK_DTYPES
        ):
            raise ValueError("Prediction archive schema differs")
        shape = tuple(metadata["mask_shape"])
        masks = np.unpackbits(
            archive["masks_packed"], count=shape[0] * shape[1], bitorder="little"
        ).reshape(shape)
        prediction = {
            "pred_masks": torch.from_numpy(masks.copy()).to(
                getattr(torch, metadata["mask_dtype"])
            ),
            "pred_scores": torch.from_numpy(archive["scores"].copy()),
            "pred_classes": torch.from_numpy(archive["classes"].copy()),
        }
        return prediction, metadata, archive["scan_vertex_offsets"].copy()


class PredictionSink:
    """One method/role/seed export; coverage denominators survive partial runs."""

    def __init__(
        self,
        root: Path,
        *,
        binding: dict,
        expected_units_by_horizon: dict,
        cache_root: Path | None = None,
        cache_byte_limit: int = 32 * 1024**3,
    ):
        self.root = root
        self.cache_root = cache_root or root
        self.cache_byte_limit = cache_byte_limit
        if (
            not root.resolve().is_relative_to(self.cache_root.resolve())
            or cache_byte_limit <= 0
        ):
            raise ValueError("Prediction cache root or byte allowance differs")
        self.binding = dict(binding)
        self.expected = {
            str(key): int(value) for key, value in expected_units_by_horizon.items()
        }
        if any(value <= 0 for value in self.expected.values()):
            raise ValueError(
                "Prediction coverage requires its real positive denominators"
            )
        if binding["data_role"] in {"PB", "ADDITIONAL", "LOCAL-T2"} and not binding.get(
            "lock_sha256"
        ):
            raise ValueError("Confirmation prediction export requires the final lock")
        root.mkdir(parents=True, exist_ok=True)
        self.records = {}
        index = root / "INDEX.json"
        if index.exists():
            saved = read_json(index)
            if (
                saved["binding"] != binding
                or saved["expected_units_by_horizon"] != self.expected
            ):
                raise ValueError(
                    "Existing prediction export belongs to another frozen run"
                )
            for row in saved["records"]:
                if file_hash(root / row["file"]) != row["sha256"]:
                    raise ValueError("Existing prediction archive changed")
                self.records[(row["logical_unit_id"], row["T"])] = row

    def write(
        self, prediction, *, logical_unit_id, horizon, scan_ids, scan_vertex_offsets
    ):
        if str(horizon) not in self.expected:
            raise ValueError("Prediction export includes an unscored horizon")
        identity = {
            key: self.binding[key]
            for key in (
                "method_id",
                "data_role",
                "eval_seed",
                "inference_identity",
                "lock_sha256",
            )
        }
        identity.update(
            logical_unit_id=logical_unit_id, T=horizon, scan_ids=list(scan_ids)
        )
        encoded = encode_prediction(
            prediction, identity=identity, scan_vertex_offsets=scan_vertex_offsets
        )
        sha = hashlib.sha256(encoded).hexdigest()
        path = self.root / (content_hash(identity) + ".npz")
        if path.exists():
            if file_hash(path) != sha:
                raise ValueError("Repeated prediction identity has different output")
        else:
            used = sum(
                path.stat().st_size
                for path in self.cache_root.rglob("*")
                if path.is_file()
            )
            if used + len(encoded) + 1024**2 > self.cache_byte_limit:
                raise ValueError(
                    "Prediction export exceeds the global new-cache allowance"
                )
            descriptor, temporary = tempfile.mkstemp(
                prefix=".prediction-", dir=self.root
            )
            try:
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(encoded)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.link(temporary, path)
            finally:
                Path(temporary).unlink(missing_ok=True)
        self.records[(logical_unit_id, horizon)] = {
            "logical_unit_id": logical_unit_id,
            "T": horizon,
            "file": path.name,
            "sha256": sha,
            "bytes": len(encoded),
        }

    def finalize(self):
        counts = {
            key: sum(str(row["T"]) == key for row in self.records.values())
            for key in self.expected
        }
        if any(counts[key] > value for key, value in self.expected.items()):
            raise ValueError("Prediction export exceeds the frozen population")
        result = {
            "schema_version": SCHEMA,
            "status": "COMPLETE" if counts == self.expected else "PARTIAL",
            "binding": self.binding,
            "expected_units_by_horizon": self.expected,
            "completed_units_by_horizon": counts,
            "records": [self.records[key] for key in sorted(self.records)],
            "contains_ground_truth": False,
        }
        write_json(self.root / "INDEX.json", result)
        return result


def package_predictions(root: Path, *, destination: Path):
    """Freeze an indexed export, retaining partial coverage in the package."""
    index = read_json(root / "INDEX.json")
    expected = [
        *index["records"],
        {"file": "INDEX.json", "sha256": file_hash(root / "INDEX.json")},
    ]
    for row in expected:
        if file_hash(root / row["file"]) != row["sha256"]:
            raise ValueError("Prediction export changed before packaging")
    if destination.exists():
        with zipfile.ZipFile(destination) as archive:
            if set(archive.namelist()) != {row["file"] for row in expected}:
                raise ValueError(
                    "Existing prediction package has a different inventory"
                )
            for row in expected:
                with archive.open(row["file"]) as content:
                    digest = (
                        hashlib.file_digest(content, "sha256")
                        if hasattr(hashlib, "file_digest")
                        else None
                    )
                    if digest is None:
                        hasher = hashlib.sha256()
                        for block in iter(lambda: content.read(1024**2), b""):
                            hasher.update(block)
                        digest = hasher
                    if digest.hexdigest() != row["sha256"]:
                        raise ValueError("Existing packaged prediction differs")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(
            destination, "x", compression=zipfile.ZIP_STORED
        ) as archive:
            for row in expected:
                archive.write(root / row["file"], arcname=row["file"])
    return {
        "status": index["status"],
        "bytes": destination.stat().st_size,
        "sha256": file_hash(destination),
        "binding": index["binding"],
        "expected_units_by_horizon": index["expected_units_by_horizon"],
        "completed_units_by_horizon": index["completed_units_by_horizon"],
    }
