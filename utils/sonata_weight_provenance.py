"""Fail-closed provenance and load-key checks for the formal Sonata weight."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import stat
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


class SonataWeightProvenanceError(ValueError):
    """Raised when a Sonata weight cannot be bound to its official source."""


class SonataLoadKeyError(ValueError):
    """Raised when the encoder-only Sonata weight fails the load-key contract."""


@dataclass(frozen=True)
class SonataWeightSpec:
    repo_id: str
    revision: str
    filename: str
    sha256: str
    bytes: int
    license: str


OFFICIAL_SONATA_WEIGHT_SPEC = SonataWeightSpec(
    repo_id="facebook/sonata",
    revision="df99897472c09f91ba9288da0a034aacffc0b010",
    filename="sonata.pth",
    sha256="c5ced5acdae30d1c469713398073a866e25e6e414e23feed5dc025373657ac50",
    bytes=434_008_287,
    license="CC-BY-NC-4.0",
)


def validate_official_sonata_remote_metadata(
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate normalized Hugging Face metadata for the frozen weight object."""

    spec = OFFICIAL_SONATA_WEIGHT_SPEC
    if not isinstance(metadata, Mapping):
        raise SonataWeightProvenanceError("remote metadata must be an object")
    if metadata.get("id") != spec.repo_id:
        raise SonataWeightProvenanceError("remote repo_id mismatch")
    if metadata.get("sha") != spec.revision:
        raise SonataWeightProvenanceError("remote immutable revision mismatch")
    if metadata.get("private") is not False:
        raise SonataWeightProvenanceError("remote model repository is private")
    if metadata.get("gated") is not False:
        raise SonataWeightProvenanceError("remote model repository is gated")
    siblings = metadata.get("siblings")
    if not isinstance(siblings, list):
        raise SonataWeightProvenanceError("remote siblings list is missing")
    candidates = [
        sibling
        for sibling in siblings
        if isinstance(sibling, Mapping)
        and sibling.get("rfilename") == spec.filename
    ]
    if len(candidates) != 1:
        raise SonataWeightProvenanceError(
            "remote metadata must contain exactly one official filename"
        )
    candidate = candidates[0]
    lfs = candidate.get("lfs")
    if not isinstance(lfs, Mapping) or lfs.get("sha256") != spec.sha256:
        raise SonataWeightProvenanceError("remote LFS SHA256 mismatch")
    if candidate.get("size") != spec.bytes or lfs.get("size") != spec.bytes:
        raise SonataWeightProvenanceError("remote LFS byte-size mismatch")
    return {
        "repo_id": spec.repo_id,
        "revision": spec.revision,
        "filename": spec.filename,
        "sha256": spec.sha256,
        "bytes": spec.bytes,
        "gated": False,
        "private": False,
    }


def _lower_hex(value: object, *, name: str, length: int) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SonataWeightProvenanceError(
            f"{name} must be a {length}-character lowercase hexadecimal digest"
        )
    return value


def _validate_spec(spec: SonataWeightSpec, *, require_official: bool) -> None:
    if not isinstance(spec, SonataWeightSpec):
        raise SonataWeightProvenanceError("weight spec must be SonataWeightSpec")
    _lower_hex(spec.sha256, name="weight SHA256", length=64)
    if not isinstance(spec.bytes, int) or isinstance(spec.bytes, bool) or spec.bytes <= 0:
        raise SonataWeightProvenanceError("weight bytes must be a positive integer")
    if not isinstance(spec.revision, str) or spec.revision == "main":
        raise SonataWeightProvenanceError("weight requires an immutable revision")
    _lower_hex(spec.revision, name="immutable revision", length=40)
    if not require_official:
        return
    expected = OFFICIAL_SONATA_WEIGHT_SPEC
    if spec.repo_id != expected.repo_id:
        raise SonataWeightProvenanceError("official repo_id mismatch")
    if spec.revision != expected.revision:
        raise SonataWeightProvenanceError("official immutable revision mismatch")
    if spec.filename != expected.filename:
        raise SonataWeightProvenanceError("official filename mismatch")
    if spec.sha256 != expected.sha256:
        raise SonataWeightProvenanceError("official weight SHA256 mismatch")
    if spec.bytes != expected.bytes:
        raise SonataWeightProvenanceError("official weight byte-size mismatch")
    if spec.license != expected.license:
        raise SonataWeightProvenanceError("official weight license mismatch")


def _stable_file_identity(path: Path) -> tuple[int, str]:
    path = Path(path)
    try:
        metadata = path.lstat()
    except OSError as error:
        raise SonataWeightProvenanceError(
            f"weight is not a readable regular file: {path}"
        ) from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise SonataWeightProvenanceError(
            f"weight is not a readable regular file: {path}"
        )

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise SonataWeightProvenanceError(
            f"weight is not a readable regular file: {path}"
        ) from error
    try:
        before = os.fstat(descriptor)
        hasher = hashlib.sha256()
        observed_bytes = 0
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(block)
                observed_bytes += len(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)

    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if before_identity != after_identity:
        raise SonataWeightProvenanceError("weight changed while hashing")
    return observed_bytes, hasher.hexdigest()


def build_sonata_weight_manifest(
    weight_path: Path,
    *,
    spec: SonataWeightSpec = OFFICIAL_SONATA_WEIGHT_SPEC,
    acquired_at: str,
    download_source: str,
    require_official: bool = False,
) -> dict[str, Any]:
    """Build a portable manifest after verifying one immutable local weight."""

    _validate_spec(spec, require_official=require_official)
    if (
        not isinstance(acquired_at, str)
        or "T" not in acquired_at
        or not acquired_at.endswith("Z")
    ):
        raise SonataWeightProvenanceError(
            "acquisition timestamp must be UTC ISO-8601"
        )
    if not isinstance(download_source, str) or not download_source.startswith(
        "https://"
    ):
        raise SonataWeightProvenanceError("download source must be an HTTPS URL")

    observed_bytes, observed_sha256 = _stable_file_identity(Path(weight_path))
    if observed_sha256 != spec.sha256:
        raise SonataWeightProvenanceError(
            "weight SHA256 mismatch: "
            f"expected {spec.sha256}, got {observed_sha256}"
        )
    if observed_bytes != spec.bytes:
        raise SonataWeightProvenanceError(
            "weight byte-size mismatch: "
            f"expected {spec.bytes}, got {observed_bytes}"
        )

    return {
        "schema_version": 1,
        "status": "pass",
        "gate": "weight-provenance-pass",
        "acquired_at": acquired_at,
        "source": {
            "repo_id": spec.repo_id,
            "revision": spec.revision,
            "filename": spec.filename,
            "download_source": download_source,
            "license": spec.license,
        },
        "declared_remote_file": {
            "sha256": spec.sha256,
            "bytes": spec.bytes,
        },
        "local_file": {
            "reference": f"external:sonata_weight/{spec.sha256}",
            "sha256": observed_sha256,
            "bytes": observed_bytes,
            "regular_file": True,
            "symlink": False,
        },
    }


def validate_sonata_weight_manifest(
    manifest: Mapping[str, Any],
    weight_path: Path,
    *,
    spec: SonataWeightSpec = OFFICIAL_SONATA_WEIGHT_SPEC,
    require_official: bool = False,
) -> None:
    """Revalidate a manifest against the current immutable file bytes."""

    _validate_spec(spec, require_official=require_official)
    if not isinstance(manifest, Mapping):
        raise SonataWeightProvenanceError("weight manifest must be an object")
    expected_source = {
        "repo_id": spec.repo_id,
        "revision": spec.revision,
        "filename": spec.filename,
        "download_source": manifest.get("source", {}).get("download_source")
        if isinstance(manifest.get("source"), Mapping)
        else None,
        "license": spec.license,
    }
    if (
        manifest.get("schema_version") != 1
        or manifest.get("status") != "pass"
        or manifest.get("gate") != "weight-provenance-pass"
        or manifest.get("source") != expected_source
    ):
        raise SonataWeightProvenanceError("weight manifest source mismatch")

    observed_bytes, observed_sha256 = _stable_file_identity(Path(weight_path))
    local_file = manifest.get("local_file")
    if not isinstance(local_file, Mapping):
        raise SonataWeightProvenanceError("weight manifest local file is missing")
    if (
        observed_sha256 != spec.sha256
        or observed_bytes != spec.bytes
        or local_file.get("sha256") != observed_sha256
        or local_file.get("bytes") != observed_bytes
        or local_file.get("regular_file") is not True
        or local_file.get("symlink") is not False
    ):
        raise SonataWeightProvenanceError("weight manifest/file mismatch")


def _key_set(values: Iterable[str], *, name: str) -> set[str]:
    result = set(values)
    if any(not isinstance(value, str) or not value for value in result):
        raise SonataLoadKeyError(f"{name} must contain non-empty strings")
    return result


def build_sonata_load_key_audit(
    *,
    checkpoint_keys: Iterable[str],
    model_keys: Iterable[str],
    missing_keys: Iterable[str],
    unexpected_keys: Iterable[str],
    weight_sha256: str,
) -> dict[str, Any]:
    """Classify strict=False results and reject non-decoder initialization."""

    try:
        _lower_hex(weight_sha256, name="weight SHA256", length=64)
    except SonataWeightProvenanceError as error:
        raise SonataLoadKeyError(str(error)) from error
    checkpoint = _key_set(checkpoint_keys, name="checkpoint keys")
    model = _key_set(model_keys, name="model keys")
    missing = _key_set(missing_keys, name="missing keys")
    unexpected = _key_set(unexpected_keys, name="unexpected keys")

    encoder_model = {
        key for key in model if key.startswith(("embedding.", "enc."))
    }
    encoder_checkpoint = {
        key for key in checkpoint if key.startswith(("embedding.", "enc."))
    }
    if not encoder_model or not encoder_checkpoint:
        raise SonataLoadKeyError("no encoder or embedding keys are loadable")
    if missing != model - checkpoint:
        raise SonataLoadKeyError("reported missing keys disagree with state dictionaries")
    if unexpected != checkpoint - model:
        raise SonataLoadKeyError(
            "reported unexpected keys disagree with state dictionaries"
        )

    critical_missing = sorted(encoder_model - checkpoint | (missing & encoder_model))
    if critical_missing:
        raise SonataLoadKeyError(
            "critical encoder or embedding keys are missing: "
            + ", ".join(critical_missing[:5])
        )
    non_decoder_missing = sorted(
        key for key in missing if not key.startswith("dec.")
    )
    if non_decoder_missing:
        raise SonataLoadKeyError(
            "non-decoder missing keys are not allowlisted: "
            + ", ".join(non_decoder_missing[:5])
        )
    if unexpected:
        raise SonataLoadKeyError(
            "unexpected checkpoint keys are not allowlisted: "
            + ", ".join(sorted(unexpected)[:5])
        )

    loaded = model - missing
    loaded_encoder = sorted(loaded & encoder_model)
    return {
        "schema_version": 1,
        "status": "pass",
        "gate": "SW0-PASS",
        "weight_sha256": weight_sha256,
        "checkpoint_key_count": len(checkpoint),
        "model_key_count": len(model),
        "loaded_key_count": len(loaded),
        "loaded_encoder_key_count": len(loaded_encoder),
        "loaded_encoder_keys": loaded_encoder,
        "expected_decoder_missing_key_count": len(missing),
        "expected_decoder_missing_keys": sorted(missing),
        "missing_keys": sorted(missing),
        "unexpected_keys": [],
        "critical_encoder_missing_keys": [],
    }


def audit_sonata_checkpoint_load(
    weight_path: Path,
    *,
    expected_sha256: str = OFFICIAL_SONATA_WEIGHT_SPEC.sha256,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Instantiate Sonata with its decoder enabled and audit strict=False keys."""

    import sonata
    import torch

    observed_bytes, observed_sha256 = _stable_file_identity(Path(weight_path))
    if observed_sha256 != expected_sha256:
        raise SonataLoadKeyError(
            f"weight SHA256 mismatch before load: {observed_sha256}"
        )
    try:
        checkpoint = torch.load(weight_path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise SonataLoadKeyError("Sonata checkpoint could not be loaded safely") from error
    if not isinstance(checkpoint, Mapping):
        raise SonataLoadKeyError("Sonata checkpoint must contain a mapping")
    config = checkpoint.get("config")
    state_dict = checkpoint.get("state_dict")
    if not isinstance(config, Mapping) or not isinstance(state_dict, Mapping):
        raise SonataLoadKeyError("Sonata checkpoint lacks config/state_dict mappings")
    if any(not isinstance(key, str) for key in state_dict):
        raise SonataLoadKeyError("Sonata state_dict keys must be strings")

    resolved_config = copy.deepcopy(dict(config))
    resolved_config["enc_mode"] = False
    try:
        model = sonata.model.PointTransformerV3(**resolved_config)
        incompatible = model.load_state_dict(state_dict, strict=False)
    except Exception as error:
        raise SonataLoadKeyError("Sonata model instantiation/load failed") from error
    audit = build_sonata_load_key_audit(
        checkpoint_keys=state_dict.keys(),
        model_keys=model.state_dict().keys(),
        missing_keys=incompatible.missing_keys,
        unexpected_keys=incompatible.unexpected_keys,
        weight_sha256=observed_sha256,
    )
    config_payload = json.dumps(
        resolved_config,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    metadata = {
        "weight_bytes": observed_bytes,
        "checkpoint_config_enc_mode": config.get("enc_mode"),
        "resolved_config_enc_mode": resolved_config["enc_mode"],
        "resolved_config_sha256": hashlib.sha256(
            config_payload.encode("ascii")
        ).hexdigest(),
    }
    return audit, metadata


def official_spec_dict() -> dict[str, Any]:
    """Return a JSON-serializable copy of the frozen official weight spec."""

    return asdict(OFFICIAL_SONATA_WEIGHT_SPEC)
