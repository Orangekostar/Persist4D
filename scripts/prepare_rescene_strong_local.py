#!/usr/bin/env python3
"""Create one immutable, evidence-gated ReScene-Strong authorization."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.rescene_rootcause_preflight import (
    RootCauseContractError,
    canonical_sha256,
    validate_portable_payload,
)
from utils.rescene_strong_local import (
    STRONG_VARIANTS,
    choose_diagnostic_base_variant,
    materialize_strong_config,
    strong_variant_gate,
    strong_variant_state_dict_entries,
    validate_strong_variant_isolation,
)


def _validate_signed(payload: Mapping[str, Any], *, field: str, name: str) -> None:
    expected = payload.get(field)
    unsigned = dict(payload)
    unsigned.pop(field, None)
    if not isinstance(expected, str) or canonical_sha256(unsigned) != expected:
        raise RootCauseContractError(f"{name} content hash differs")


def _identity(value: Mapping[str, Any], *, name: str) -> dict[str, object]:
    byte_size = value.get("bytes")
    sha256 = value.get("sha256")
    if (
        isinstance(byte_size, bool)
        or not isinstance(byte_size, int)
        or byte_size < 0
        or not isinstance(sha256, str)
        or len(sha256) != 64
        or any(character not in "0123456789abcdef" for character in sha256)
    ):
        raise RootCauseContractError(f"{name} identity is invalid")
    return {"bytes": byte_size, "sha256": sha256}


def _validate_prior_variant_result(
    *,
    name: str,
    expected_variant: str,
    result: Mapping[str, Any] | None,
    authorization: Mapping[str, Any] | None,
) -> None:
    if result is None and authorization is None:
        return
    if result is None or authorization is None:
        raise RootCauseContractError(f"{name} result authorization is incomplete")
    _validate_signed(
        authorization,
        field="authorization_sha256",
        name=f"{name} authorization",
    )
    _validate_signed(result, field="content_sha256", name=f"{name} result")
    if (
        authorization.get("status") != "authorized"
        or authorization.get("selected_variants") != [expected_variant]
        or result.get("status") != "pass"
        or result.get("variant") != expected_variant
        or result.get("variant_authorization_sha256")
        != authorization["authorization_sha256"]
    ):
        raise RootCauseContractError(f"{name} result binding differs")


def build_strong_authorization(
    *,
    variant: str,
    root_authorization: Mapping[str, Any],
    short_decision: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    source_commit: str,
    common_identity: Mapping[str, Any],
    pretrained_identity: Mapping[str, Any],
    input_identities: Mapping[str, Mapping[str, Any]],
    a1_result: Mapping[str, Any] | None = None,
    a2_result: Mapping[str, Any] | None = None,
    a1_authorization: Mapping[str, Any] | None = None,
    a2_authorization: Mapping[str, Any] | None = None,
) -> dict[str, object]:
    """Bind a native structural variant to all upstream scientific gates."""

    if (
        not isinstance(source_commit, str)
        or len(source_commit) != 40
        or any(character not in "0123456789abcdef" for character in source_commit)
    ):
        raise RootCauseContractError("strong-local source commit is invalid")
    _validate_signed(
        root_authorization,
        field="authorization_sha256",
        name="root-cause authorization",
    )
    if root_authorization.get("status") != "authorized":
        raise RootCauseContractError("root-cause authorization is inactive")
    root_authorization_sha256 = root_authorization["authorization_sha256"]
    _validate_signed(
        short_decision, field="content_sha256", name="root-cause short decision"
    )
    _validate_signed(
        diagnostics, field="content_sha256", name="decoder diagnostic decision"
    )
    diagnostic_provenance = diagnostics.get("provenance")
    diagnostic_bindings = (
        diagnostic_provenance.get("bindings")
        if isinstance(diagnostic_provenance, Mapping)
        else None
    )
    if (
        short_decision.get("status") != "pass"
        or short_decision.get("experiment") != "rescene_task_learning_root_cause_v1"
        or short_decision.get("variant_authorization_sha256")
        != root_authorization_sha256
        or diagnostics.get("status") != "pass"
        or diagnostics.get("experiment") != "rescene_task_learning_root_cause_v1"
        or not isinstance(diagnostic_bindings, Mapping)
        or diagnostic_bindings.get("variant_authorization_sha256")
        != root_authorization_sha256
    ):
        raise RootCauseContractError("root-cause evidence authorization differs")
    _validate_prior_variant_result(
        name="A1",
        expected_variant="A1",
        result=a1_result,
        authorization=a1_authorization,
    )
    _validate_prior_variant_result(
        name="A2",
        expected_variant="A2",
        result=a2_result,
        authorization=a2_authorization,
    )
    base_variant = choose_diagnostic_base_variant(short_decision, diagnostics)
    gate = strong_variant_gate(
        variant,
        diagnostics=diagnostics,
        a1_result=a1_result,
        a2_result=a2_result,
    )

    initialization = root_authorization.get("initialization")
    variants = root_authorization.get("variants")
    base_record = variants.get(base_variant) if isinstance(variants, Mapping) else None
    if not isinstance(initialization, Mapping) or not isinstance(base_record, Mapping):
        raise RootCauseContractError("root-cause base authorization is incomplete")
    base_config = base_record.get("resolved_config")
    if not isinstance(base_config, Mapping) or canonical_sha256(
        base_config
    ) != base_record.get("config_sha256"):
        raise RootCauseContractError("root-cause base config hash differs")
    expected_common = initialization.get("common_state")
    expected_pretrained = initialization.get("pretrained")
    observed_common = _identity(common_identity, name="common initialization")
    observed_pretrained = _identity(
        pretrained_identity, name="Concerto pretrained encoder"
    )
    if not isinstance(expected_common, Mapping) or observed_common != _identity(
        expected_common, name="authorized common initialization"
    ):
        raise RootCauseContractError("common initialization identity differs")
    if not isinstance(expected_pretrained, Mapping) or observed_pretrained != _identity(
        expected_pretrained, name="authorized Concerto pretrained encoder"
    ):
        raise RootCauseContractError("Concerto pretrained identity differs")

    required_inputs = {
        "root_authorization",
        "short_decision",
        "diagnostics",
        "root_learning_curves",
        "root_official_like_epoch60",
        "root_official_like_epoch90",
    }
    if a1_result is not None:
        required_inputs.update(("a1_result", "a1_authorization"))
    if a2_result is not None:
        required_inputs.update(("a2_result", "a2_authorization"))
    if set(input_identities) != required_inputs:
        raise RootCauseContractError("strong-local evidence identities are incomplete")
    evidence = {
        name: _identity(identity, name=name)
        for name, identity in sorted(input_identities.items())
    }
    if not gate["authorized"]:
        skipped: dict[str, object] = {
            "schema_version": 1,
            "status": "gate_skipped",
            "experiment": "rescene_strong_local_v1",
            "variant": variant,
            "base_variant": base_variant,
            "gate": gate,
            "upstream_evidence": evidence,
        }
        skipped["content_sha256"] = canonical_sha256(skipped)
        return skipped

    output_reference = f"external:checkpoint/rescene_strong_local/{variant}"
    config = materialize_strong_config(
        base_config, variant=variant, output=output_reference
    )
    isolation = validate_strong_variant_isolation(base_config, config, variant=variant)
    tensor_state = initialization.get("tensor_state")
    common_entries = (
        tensor_state.get("tensor_count") if isinstance(tensor_state, Mapping) else None
    )
    expected_entries = strong_variant_state_dict_entries(common_entries, variant)
    payload: dict[str, object] = {
        "schema_version": 1,
        "status": "authorized",
        "experiment": "rescene_strong_local_v1",
        "source_commit": source_commit,
        "checkpoint_namespace": "rescene_strong_local",
        "selected_variants": [variant],
        "base_variant": base_variant,
        "gate": gate,
        "rootcause_authorization_sha256": root_authorization["authorization_sha256"],
        "rootcause_short_decision_sha256": short_decision["content_sha256"],
        "decoder_diagnostics_sha256": diagnostics["content_sha256"],
        "upstream_evidence": evidence,
        "initialization": copy_initialization(initialization),
        "schedule": root_authorization.get("schedule"),
        "variants": {
            variant: {
                "config_sha256": canonical_sha256(config),
                "resolved_config": config,
                "isolation": isolation,
                "expected_state_dict_entries": expected_entries,
            }
        },
    }
    validate_portable_payload(payload)
    payload["authorization_sha256"] = canonical_sha256(payload)
    return payload


def copy_initialization(value: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(value, allow_nan=False))


def _load_json(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RootCauseContractError(f"{name} is not readable JSON") from error
    if not isinstance(value, dict):
        raise RootCauseContractError(f"{name} must contain an object")
    return value


def _file_identity(path: Path) -> dict[str, object]:
    try:
        before = path.lstat()
        if path.is_symlink() or not stat.S_ISREG(before.st_mode):
            raise OSError
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                size += len(block)
                digest.update(block)
        after = path.lstat()
    except OSError as error:
        raise RootCauseContractError("strong-local input is unavailable") from error
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ) or size != after.st_size:
        raise RootCauseContractError("strong-local input changed while hashing")
    return {"bytes": size, "sha256": digest.hexdigest()}


def _publish(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(
            payload, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True
        )
        + "\n"
    ).encode("ascii")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_file() and not path.is_symlink() and path.read_bytes() == encoded:
            return
        raise RootCauseContractError("refusing to overwrite strong-local authorization")
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except subprocess.CalledProcessError as error:
        raise RootCauseContractError(
            "strong-local Git identity is unavailable"
        ) from error


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=STRONG_VARIANTS, required=True)
    parser.add_argument("--root-authorization", type=Path, required=True)
    parser.add_argument("--short-decision", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--root-learning-curves", type=Path, required=True)
    parser.add_argument("--root-official-like-epoch60", type=Path, required=True)
    parser.add_argument("--root-official-like-epoch90", type=Path, required=True)
    parser.add_argument("--common-state", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--a1-result", type=Path)
    parser.add_argument("--a2-result", type=Path)
    parser.add_argument("--a1-authorization", type=Path)
    parser.add_argument("--a2-authorization", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    paths = {
        "root_authorization": arguments.root_authorization,
        "short_decision": arguments.short_decision,
        "diagnostics": arguments.diagnostics,
        "root_learning_curves": arguments.root_learning_curves,
        "root_official_like_epoch60": arguments.root_official_like_epoch60,
        "root_official_like_epoch90": arguments.root_official_like_epoch90,
    }
    if arguments.a1_result:
        paths["a1_result"] = arguments.a1_result
    if arguments.a1_authorization:
        paths["a1_authorization"] = arguments.a1_authorization
    if arguments.a2_result:
        paths["a2_result"] = arguments.a2_result
    if arguments.a2_authorization:
        paths["a2_authorization"] = arguments.a2_authorization
    result = build_strong_authorization(
        variant=arguments.variant,
        root_authorization=_load_json(
            arguments.root_authorization, name="root-cause authorization"
        ),
        short_decision=_load_json(
            arguments.short_decision, name="root-cause short decision"
        ),
        diagnostics=_load_json(
            arguments.diagnostics, name="decoder diagnostic decision"
        ),
        source_commit=_git_head(),
        common_identity=_file_identity(arguments.common_state),
        pretrained_identity=_file_identity(arguments.pretrained),
        input_identities={name: _file_identity(path) for name, path in paths.items()},
        a1_result=(
            _load_json(arguments.a1_result, name="A1 result")
            if arguments.a1_result
            else None
        ),
        a2_result=(
            _load_json(arguments.a2_result, name="A2 result")
            if arguments.a2_result
            else None
        ),
        a1_authorization=(
            _load_json(arguments.a1_authorization, name="A1 authorization")
            if arguments.a1_authorization
            else None
        ),
        a2_authorization=(
            _load_json(arguments.a2_authorization, name="A2 authorization")
            if arguments.a2_authorization
            else None
        ),
    )
    _publish(arguments.output, result)
    print(json.dumps({"status": result["status"], "variant": arguments.variant}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
