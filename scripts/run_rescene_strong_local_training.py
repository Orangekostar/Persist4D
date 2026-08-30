#!/usr/bin/env python3
"""Validate and launch one formally authorized ReScene-Strong curve."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.rescene_rootcause_preflight import (
    ROOTCAUSE_CONFIG_NAME,
    _git,
    _load_json,
    _runtime_environment,
    _stable_file_identity,
    compose_variant_config,
    portable_variant_config,
    variant_overrides,
)
from scripts.run_rescene_rootcause_training import (
    RootCauseLaunchError,
    authorize_unique_candidate,
    parse_devices,
)
from utils.rescene_rootcause_preflight import (
    RootCauseContractError,
    canonical_sha256,
    validate_portable_payload,
)
from utils.rescene_strong_local import (
    STRONG_VARIANTS,
    materialize_strong_config,
    validate_strong_variant_isolation,
)

FORMAL_BRANCH = "research/persist4d-rescene-task-learning-root-cause-v1"


def _validate_authorization(payload: Mapping[str, Any]) -> None:
    expected = payload.get("authorization_sha256")
    unsigned = dict(payload)
    unsigned.pop("authorization_sha256", None)
    if (
        payload.get("status") != "authorized"
        or not isinstance(expected, str)
        or canonical_sha256(unsigned) != expected
    ):
        raise RootCauseLaunchError("strong-local authorization hash differs")
    try:
        validate_portable_payload(payload)
    except RootCauseContractError as error:
        raise RootCauseLaunchError(str(error)) from error


def _require_equal(observed: object, expected: object, *, name: str) -> None:
    if observed != expected:
        raise RootCauseLaunchError(f"{name} differs from formal authorization")


def _validate_source(authorization: Mapping[str, Any]) -> None:
    if Path.cwd().resolve() != PROJECT_ROOT.resolve():
        raise RootCauseLaunchError("strong-local launcher requires repository root")
    if _git("branch", "--show-current") != FORMAL_BRANCH:
        raise RootCauseLaunchError("formal strong-local branch differs")
    source_commit = authorization.get("source_commit")
    if not isinstance(source_commit, str):
        raise RootCauseLaunchError("strong-local source commit is missing")
    try:
        _git("merge-base", "--is-ancestor", source_commit, "HEAD")
    except RootCauseContractError as error:
        raise RootCauseLaunchError(
            "authorized strong-local source is not an ancestor of HEAD"
        ) from error
    changed = [
        path
        for path in _git("diff", "--name-only", f"{source_commit}..HEAD").splitlines()
        if path
    ]
    artifact_prefix = "artifacts/rescene_task_learning_root_cause_v1/"
    if any(not path.startswith(artifact_prefix) for path in changed):
        raise RootCauseLaunchError("strong-local source changed after authorization")
    if _git("status", "--porcelain", "--untracked-files=all"):
        raise RootCauseLaunchError("strong-local worktree is not clean")


def build_strong_launch_environment(
    *,
    authorization: Mapping[str, Any],
    pretrained: Path,
    common_state: Path,
    output_dir: Path,
    devices: tuple[int, int],
    inherited: Mapping[str, str] | None = None,
) -> dict[str, str]:
    variant = authorization["selected_variants"][0]
    base_variant = authorization["base_variant"]
    environment = dict(os.environ if inherited is None else inherited)
    python_paths = [
        str(PROJECT_ROOT / "third_party/concerto"),
        str(PROJECT_ROOT / "third_party/detectron2"),
        str(PROJECT_ROOT / "third_party/sonata"),
        str(PROJECT_ROOT / "third_party/stmetrics"),
    ]
    if environment.get("PYTHONPATH"):
        python_paths.append(environment["PYTHONPATH"])
    environment.update(
        {
            "PYTHONPATH": os.pathsep.join(python_paths),
            "CUDA_VISIBLE_DEVICES": ",".join(str(device) for device in devices),
            "CONCERTO_CHECKPOINT": str(Path(pretrained).resolve()),
            "RESCENE_ROOTCAUSE_VARIANT": variant,
            "RESCENE_ROOTCAUSE_OUTPUT_DIR": str(Path(output_dir).resolve()),
            "RESCENE_ROOTCAUSE_COMMON_STATE": str(Path(common_state).resolve()),
            "RESCENE_ROOTCAUSE_COMMON_SHA256": authorization["initialization"][
                "common_state"
            ]["sha256"],
            "RESCENE_ROOTCAUSE_OBJECTIVE_MODE": (
                "raw_sum" if base_variant == "R1" else "weighted"
            ),
        }
    )
    return environment


def build_strong_launch_command(authorization: Mapping[str, Any]) -> list[str]:
    variants = authorization.get("selected_variants")
    if not isinstance(variants, list) or len(variants) != 1:
        raise RootCauseLaunchError("strong-local authorization is not singular")
    variant = variants[0]
    if variant not in STRONG_VARIANTS:
        raise RootCauseLaunchError("strong-local variant is not registered")
    base_variant = authorization.get("base_variant")
    if not isinstance(base_variant, str):
        raise RootCauseLaunchError("strong-local base variant is missing")
    model = authorization["variants"][variant]["resolved_config"]["model"]
    return [
        sys.executable,
        str(PROJECT_ROOT / "main_instance_segmentation.py"),
        "--config-name",
        ROOTCAUSE_CONFIG_NAME,
        *variant_overrides(base_variant),
        "general.project_name=rescene_strong_local_v1",
        "rootcause_preflight.target=rescene_strong_local_v1",
        (
            "rootcause_preflight.variant_manifest="
            "artifacts/rescene_task_learning_root_cause_v1/strong_local/"
            f"{variant}/variant_manifest.json"
        ),
        f"model.use_np_features={str(model['use_np_features']).lower()}",
        f"model.scatter_type={model['scatter_type']}",
    ]


def _runtime_config(
    *,
    authorization: Mapping[str, Any],
    pretrained: Path,
    common_state: Path,
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    variant = authorization["selected_variants"][0]
    base_variant = authorization["base_variant"]
    common_sha256 = authorization["initialization"]["common_state"]["sha256"]
    base = compose_variant_config(
        base_variant,
        pretrained=pretrained,
        common_state=common_state,
        common_sha256=common_sha256,
        output=output_dir,
    )
    runtime = materialize_strong_config(
        base, variant=variant, output=str(output_dir.resolve())
    )
    initialization = authorization["initialization"]
    portable = portable_variant_config(
        runtime,
        variant=variant,
        pretrained_reference=initialization["pretrained"]["reference"],
        common_reference=initialization["common_state"]["reference"],
        output_namespace=authorization["checkpoint_namespace"],
    )
    return runtime, portable


def require_strong_authorization(
    *,
    authorization_path: Path,
    root_authorization_path: Path,
    evidence_paths: Mapping[str, Path],
    pretrained: Path,
    common_state: Path,
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    authorization = _load_json(authorization_path, name="strong-local authorization")
    _validate_authorization(authorization)
    _validate_source(authorization)
    upstream = authorization.get("upstream_evidence")
    if not isinstance(upstream, Mapping) or set(evidence_paths) != set(upstream):
        raise RootCauseLaunchError("strong-local evidence inputs differ")
    for name, path in evidence_paths.items():
        _require_equal(
            _stable_file_identity(path), upstream[name], name=f"{name} evidence"
        )
    root_authorization = _load_json(
        root_authorization_path, name="root-cause authorization"
    )
    unsigned_root = dict(root_authorization)
    root_hash = unsigned_root.pop("authorization_sha256", None)
    if (
        root_hash != authorization.get("rootcause_authorization_sha256")
        or canonical_sha256(unsigned_root) != root_hash
    ):
        raise RootCauseLaunchError("root-cause authorization binding differs")

    from scripts.run_rescene_rootcause_training import (
        _validate_data,
        _validate_file_bindings,
    )

    _validate_file_bindings(root_authorization)
    _validate_data(root_authorization)
    _require_equal(
        _runtime_environment(), root_authorization["runtime"], name="runtime"
    )
    for actual_path, expected, name in (
        (
            common_state,
            authorization["initialization"]["common_state"],
            "common initialization",
        ),
        (
            pretrained,
            authorization["initialization"]["pretrained"],
            "Concerto pretrained encoder",
        ),
    ):
        identity = _stable_file_identity(actual_path)
        for field in ("bytes", "sha256"):
            _require_equal(identity[field], expected[field], name=name)
    _, portable = _runtime_config(
        authorization=authorization,
        pretrained=pretrained,
        common_state=common_state,
        output_dir=output_dir,
    )
    variant = authorization["selected_variants"][0]
    expected_record = authorization["variants"][variant]
    _require_equal(
        portable, expected_record["resolved_config"], name="strong-local config"
    )
    _require_equal(
        canonical_sha256(portable),
        expected_record["config_sha256"],
        name="strong-local config hash",
    )
    base_config = root_authorization["variants"][authorization["base_variant"]][
        "resolved_config"
    ]
    _require_equal(
        validate_strong_variant_isolation(base_config, portable, variant=variant),
        expected_record["isolation"],
        name="strong-local isolation",
    )
    return authorization, portable


def build_strong_candidate_contract(
    *,
    authorization_path: Path,
    authorization: Mapping[str, Any],
    portable_config: Mapping[str, Any],
    devices: tuple[int, int],
) -> dict[str, object]:
    variant = authorization["selected_variants"][0]
    payload: dict[str, object] = {
        "schema_version": 1,
        "status": "active",
        "experiment": authorization["experiment"],
        "variant": variant,
        "base_variant": authorization["base_variant"],
        "devices": list(devices),
        "source_commit": authorization["source_commit"],
        "variant_authorization_sha256": authorization["authorization_sha256"],
        "variant_manifest_file_sha256": _stable_file_identity(authorization_path)[
            "sha256"
        ],
        "config_sha256": canonical_sha256(portable_config),
        "common_initialization_sha256": authorization["initialization"]["common_state"][
            "sha256"
        ],
        "pretrained_sha256": authorization["initialization"]["pretrained"]["sha256"],
        "decoder_diagnostics_sha256": authorization["decoder_diagnostics_sha256"],
        "schedule": authorization["schedule"],
    }
    payload["candidate_id"] = canonical_sha256(payload)
    validate_portable_payload(payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--root-authorization", type=Path, required=True)
    parser.add_argument("--short-decision", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--root-learning-curves", type=Path, required=True)
    parser.add_argument("--root-official-like-epoch60", type=Path, required=True)
    parser.add_argument("--root-official-like-epoch90", type=Path, required=True)
    parser.add_argument("--a1-result", type=Path)
    parser.add_argument("--a2-result", type=Path)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--common-state", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--devices", type=parse_devices, default=(0, 1))
    parser.add_argument("--execute", action="store_true")
    arguments = parser.parse_args(argv)
    evidence_paths = {
        "root_authorization": arguments.root_authorization,
        "short_decision": arguments.short_decision,
        "diagnostics": arguments.diagnostics,
        "root_learning_curves": arguments.root_learning_curves,
        "root_official_like_epoch60": arguments.root_official_like_epoch60,
        "root_official_like_epoch90": arguments.root_official_like_epoch90,
    }
    if arguments.a1_result:
        evidence_paths["a1_result"] = arguments.a1_result
    if arguments.a2_result:
        evidence_paths["a2_result"] = arguments.a2_result
    authorization, portable = require_strong_authorization(
        authorization_path=arguments.authorization,
        root_authorization_path=arguments.root_authorization,
        evidence_paths=evidence_paths,
        pretrained=arguments.pretrained,
        common_state=arguments.common_state,
        output_dir=arguments.output_dir,
    )
    if not arguments.execute:
        print(
            json.dumps(
                {
                    "gate": "SP0-PASS",
                    "variant": authorization["selected_variants"][0],
                    "training_launched": False,
                },
                sort_keys=True,
            )
        )
        return 0
    candidate = build_strong_candidate_contract(
        authorization_path=arguments.authorization,
        authorization=authorization,
        portable_config=portable,
        devices=arguments.devices,
    )
    launch_mode = authorize_unique_candidate(arguments.output_dir, candidate)
    environment = build_strong_launch_environment(
        authorization=authorization,
        pretrained=arguments.pretrained,
        common_state=arguments.common_state,
        output_dir=arguments.output_dir,
        devices=arguments.devices,
    )
    print(
        json.dumps(
            {
                "gate": "SP0-PASS",
                "launch_mode": launch_mode,
                "variant": candidate["variant"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    os.execve(
        sys.executable,
        build_strong_launch_command(authorization),
        environment,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
