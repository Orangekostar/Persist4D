import copy
import re

import pytest

from utils import p2_preflight

EXPECTED_COMPONENTS = {
    "cumm",
    "flash_attn",
    "hydra",
    "nvidia_cuda_libraries",
    "omegaconf",
    "pointnet2",
    "pytorch_lightning",
    "python",
    "spconv",
    "torch",
    "torch_scatter",
}


@pytest.fixture(scope="module")
def runtime_environment_contract() -> dict:
    return p2_preflight.build_p2_runtime_environment_contract()


def test_runtime_environment_contract_binds_versions_origins_and_native_code(
    runtime_environment_contract: dict,
) -> None:
    contract = runtime_environment_contract

    assert contract["status"] == "pass"
    assert contract["errors"] == []
    assert contract["versions"] == p2_preflight.P2_RUNTIME_ENVIRONMENT_VERSIONS
    assert set(contract["components"]) == EXPECTED_COMPONENTS
    assert contract["optional_modules"] == {
        "pointops": {"required": False, "status": "absent"}
    }
    for record in contract["components"].values():
        assert record["status"] == "pass"
        assert record["errors"] == []
        assert record["origin_refs"]
        manifest = record.get("native_manifest") or record.get(
            "python_source_manifest"
        )
        assert manifest["file_count"] >= 1
        assert manifest["total_bytes"] >= 1
        assert re.fullmatch(r"[0-9a-f]{64}", manifest["content_sha256"])


def test_runtime_environment_contract_validator_rejects_binary_drift(
    runtime_environment_contract: dict,
) -> None:
    drifted = copy.deepcopy(runtime_environment_contract)
    drifted["components"]["flash_attn"]["native_manifest"][
        "content_sha256"
    ] = "invalid"
    errors = []

    observed = p2_preflight._validate_runtime_environment_contract(
        {"runtime_environment_contract": drifted},
        errors,
    )

    assert observed is drifted
    assert errors == [
        "runtime_environment_contract.flash_attn.native_manifest invalid"
    ]


def test_runtime_environment_contract_validator_rejects_version_drift(
    runtime_environment_contract: dict,
) -> None:
    contract = copy.deepcopy(runtime_environment_contract)
    contract["versions"]["torch"] = "different"
    errors = []

    p2_preflight._validate_runtime_environment_contract(
        {"runtime_environment_contract": contract},
        errors,
    )

    assert errors == ["runtime_environment_contract.versions mismatch"]


def test_runtime_environment_contract_fails_closed_on_runtime_probe_oserror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable_runtime_probe() -> dict:
        raise OSError("runtime shared library unavailable")

    monkeypatch.setattr(
        p2_preflight,
        "_runtime_environment_versions",
        unavailable_runtime_probe,
    )

    contract = p2_preflight.build_p2_runtime_environment_contract()

    assert contract["status"] == "fail"
    assert "runtime_versions_unavailable" in contract["errors"]
    assert "runtime_versions_mismatch" in contract["errors"]


def test_runtime_environment_contract_validator_rejects_python_source_drift(
    runtime_environment_contract: dict,
) -> None:
    contract = copy.deepcopy(runtime_environment_contract)
    contract["components"]["pytorch_lightning"]["python_source_manifest"][
        "content_sha256"
    ] = "invalid"
    errors = []

    p2_preflight._validate_runtime_environment_contract(
        {"runtime_environment_contract": contract},
        errors,
    )

    assert errors == [
        "runtime_environment_contract.pytorch_lightning.python_source_manifest invalid"
    ]
