"""Code-level audit for frozen ReScene4D full-history evaluation."""

from __future__ import annotations

import argparse
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

from scripts.system_comparison_protocol import sha256_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_CHECKPOINT_SHA256 = (
    "85ed1aba60320cd19798536b71b91dbc156b7ea60f838832bc0bbbdba131546e"
)
_EVIDENCE_FIELDS = {
    "id",
    "file_path",
    "function_or_class",
    "line",
    "relevant_behavior",
    "scientific_implication",
}


class CodeAuditError(ValueError):
    """Raised when audit evidence cannot support the declared conclusions."""


def _get(value: object, *keys: object) -> object:
    current = value
    for key in keys:
        if isinstance(current, Mapping):
            if key not in current:
                raise CodeAuditError(f"checkpoint metadata is missing {key}")
            current = current[key]
            continue
        try:
            current = current[key]  # type: ignore[index]
        except (KeyError, IndexError, TypeError) as error:
            raise CodeAuditError(f"checkpoint metadata is missing {key}") from error
    return current


def _integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CodeAuditError(f"{name} must be a non-negative integer")
    return value


def _boolean(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise CodeAuditError(f"{name} must be boolean")
    return value


def _source_line(path: Path, token: str) -> int:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise CodeAuditError(f"cannot read audit source {path}") from error
    matches = [index for index, line in enumerate(lines, start=1) if token in line]
    if len(matches) != 1:
        raise CodeAuditError(
            f"audit token {token!r} must occur exactly once in {path.name}"
        )
    return matches[0]


def _evidence(
    repo_root: Path,
    *,
    evidence_id: str,
    file_path: str,
    function_or_class: str,
    token: str,
    behavior: str,
    implication: str,
) -> dict[str, object]:
    return {
        "id": evidence_id,
        "file_path": file_path,
        "function_or_class": function_or_class,
        "line": _source_line(repo_root / file_path, token),
        "relevant_behavior": behavior,
        "scientific_implication": implication,
    }


def _checkpoint_facts(payload: Mapping[str, object], digest: str) -> dict[str, object]:
    hyper = _get(payload, "hyper_parameters")
    datasets = _get(hyper, "data", "train_dataset", "datasets")
    if isinstance(datasets, (str, bytes)) or not isinstance(datasets, Sequence):
        raise CodeAuditError("checkpoint training datasets must be a sequence")
    if len(datasets) != 2:
        raise CodeAuditError("checkpoint must expose the frozen RIO/ScanNet mix")
    serializations = _get(hyper, "backbone", "decoder_serializations")
    if list(serializations) != ["standard", "temporal_overlay"]:
        raise CodeAuditError("checkpoint temporal decoder serializations differ")
    facts = {
        "sha256": digest,
        "epoch": _integer(_get(payload, "epoch"), name="checkpoint epoch"),
        "global_step": _integer(
            _get(payload, "global_step"), name="checkpoint global_step"
        ),
        "data_temporal_window": _integer(
            _get(hyper, "data", "temporal_window"), name="data.temporal_window"
        ),
        "rio_train_temporal_window": _integer(
            _get(datasets, 0, "temporal_window"), name="RIO train temporal_window"
        ),
        "scannet_train_temporal_window": _integer(
            _get(datasets, 1, "temporal_window"),
            name="ScanNet train temporal_window",
        ),
        "validation_temporal_window": _integer(
            _get(hyper, "data", "validation_dataset", "temporal_window"),
            name="validation temporal_window",
        ),
        "test_temporal_window": _integer(
            _get(hyper, "data", "test_dataset", "temporal_window"),
            name="test temporal_window",
        ),
        "num_queries": _integer(
            _get(hyper, "model", "num_queries"), name="model.num_queries"
        ),
        "topk_per_image": _integer(
            _get(hyper, "general", "topk_per_image"),
            name="general.topk_per_image",
        ),
        "non_parametric_queries": _boolean(
            _get(hyper, "model", "non_parametric_queries"),
            name="model.non_parametric_queries",
        ),
        "random_queries": _boolean(
            _get(hyper, "model", "random_queries"), name="model.random_queries"
        ),
        "random_query_both": _boolean(
            _get(hyper, "model", "random_query_both"),
            name="model.random_query_both",
        ),
        "temporal_masking": _boolean(
            _get(hyper, "model", "temporal_masking"),
            name="model.temporal_masking",
        ),
        "use_changes_loss": _boolean(
            _get(hyper, "model", "use_changes_loss"),
            name="model.use_changes_loss",
        ),
        "trainer_deterministic": _boolean(
            _get(hyper, "trainer", "deterministic"),
            name="trainer.deterministic",
        ),
        "decoder_serializations": list(serializations),
    }
    expected_twos = (
        "data_temporal_window",
        "rio_train_temporal_window",
        "validation_temporal_window",
        "test_temporal_window",
    )
    if any(facts[name] != 2 for name in expected_twos):
        raise CodeAuditError("checkpoint is not the frozen T2 ReScene configuration")
    if facts["scannet_train_temporal_window"] != 1:
        raise CodeAuditError("checkpoint ScanNet auxiliary horizon differs")
    if (
        facts["num_queries"] != 100
        or facts["topk_per_image"] != 100
        or facts["non_parametric_queries"] is not True
        or facts["random_queries"] is not False
        or facts["random_query_both"] is not False
        or facts["temporal_masking"] is not False
        or facts["use_changes_loss"] is not False
    ):
        raise CodeAuditError("checkpoint query/evaluation semantics differ")
    return facts


def _load_checkpoint(path: Path) -> Mapping[str, object]:
    if path.is_symlink() or not path.is_file():
        raise CodeAuditError("checkpoint must be a regular file")
    import torch

    try:
        payload = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise CodeAuditError("cannot load checkpoint metadata") from error
    if not isinstance(payload, Mapping):
        raise CodeAuditError("checkpoint must decode to a mapping")
    return payload


def build_code_audit(
    *,
    repo_root: str | Path = PROJECT_ROOT,
    checkpoint_path: str | Path | None = None,
    checkpoint_payload: Mapping[str, object] | None = None,
    checkpoint_sha256: str | None = None,
) -> dict[str, object]:
    repository = Path(repo_root).resolve()
    if checkpoint_payload is None:
        if checkpoint_path is None:
            raise CodeAuditError("checkpoint_path is required")
        path = Path(checkpoint_path)
        checkpoint_payload = _load_checkpoint(path)
        checkpoint_sha256 = sha256_file(path)
    if not isinstance(checkpoint_sha256, str) or len(checkpoint_sha256) != 64:
        raise CodeAuditError("checkpoint_sha256 must contain 64 characters")
    checkpoint = _checkpoint_facts(checkpoint_payload, checkpoint_sha256)

    evidence = [
        _evidence(
            repository,
            evidence_id="E1",
            file_path="datasets/semseg.py",
            function_or_class="SemanticSegmentationDataset.load_scan_indices",
            token="def load_scan_indices(",
            behavior="Validates an explicit non-empty unique scan-index list and delegates it without requiring a fixed length.",
            implication="The dataset path structurally accepts T1-T5 exact prefixes and does not itself impose T=2.",
        ),
        _evidence(
            repository,
            evidence_id="E2",
            file_path="datasets/semseg.py",
            function_or_class="SemanticSegmentationDataset._load_scan_sequence",
            token="def _load_scan_sequence(",
            behavior="Loads every requested scan and assigns a local temporal coordinate in request order.",
            implication="A T>2 prefix receives distinct causal stage coordinates for every observed visit.",
        ),
        _evidence(
            repository,
            evidence_id="E3",
            file_path="datasets/pointcept_utils.py",
            function_or_class="voxelize",
            token="def voxelize(",
            behavior="Voxelizes temporal stages separately, then preserves their stage coordinate in the collated sequence.",
            implication="Collation has no fixed two-stage tensor contract and retains stage membership for metrics.",
        ),
        _evidence(
            repository,
            evidence_id="E4",
            file_path="models/pointcept.py",
            function_or_class="PointceptBackbone.forward",
            token="def forward(self, x):",
            behavior="Runs the configured standard and temporal-overlay serializations on the runtime sparse input.",
            implication="Backbone feature extraction shares information across every stage present in the supplied prefix.",
        ),
        _evidence(
            repository,
            evidence_id="E5",
            file_path="models/pointcept.py",
            function_or_class="PointceptBackbone.temporal_overlay",
            token="def temporal_overlay(self, point):",
            behavior="Reassigns serialization batch identity to the true sequence batch while retaining temporal coordinates.",
            implication="Temporal sharing is joint over the observed prefix rather than a persistent state transition.",
        ),
        _evidence(
            repository,
            evidence_id="E6",
            file_path="models/rescene.py",
            function_or_class="ReScene.initialize_queries",
            token="def initialize_queries(self, pcd_features, coords):",
            behavior="Initializes 100 non-parametric queries by farthest-point sampling coordinates from the current complete input.",
            implication="Changing the prefix can change sampled query anchors; raw query index has no guaranteed cross-prefix semantic namespace.",
        ),
        _evidence(
            repository,
            evidence_id="E7",
            file_path="models/rescene.py",
            function_or_class="ReScene.forward",
            token="    def forward(",
            behavior="Uses one joint query set to decode all features supplied by the current forward and emits class/mask predictions without track IDs.",
            implication="Within-prefix joint reasoning is supported, but deployment identity is not persisted between forwards.",
        ),
        _evidence(
            repository,
            evidence_id="E8",
            file_path="models/rescene.py",
            function_or_class="ReScene.mask_module",
            token="def mask_module(self, query_feat, features):",
            behavior="Produces one class logit vector and one mask logit column per query.",
            implication="Raw query index is the only model-native identity candidate exposed to the evaluation adapter.",
        ),
        _evidence(
            repository,
            evidence_id="E9",
            file_path="trainer/trainer.py",
            function_or_class="InstanceSegmentation._get_mask_and_scores",
            token="def _get_mask_and_scores(",
            behavior="Selects top query-class pairs, converts them back to query indices for masks, then returns scores/classes/masks without those indices.",
            implication="Official task-quality postprocessing is valid, but a separate adapter must preserve query indices for deployment identity.",
        ),
        _evidence(
            repository,
            evidence_id="E10",
            file_path="scripts/evaluate_persist4d_p6a.py",
            function_or_class="RealPredictionCacheProducer.__call__",
            token="def __call__(self, logical_key: Mapping[str, object])",
            behavior="Loads only the request-resolved prefix/window and explicitly passes change_file=None before frozen inference.",
            implication="The system adapter can enforce no-future access and exclude change-label supervision without changing the model.",
        ),
    ]
    conclusions = {
        "full_history_accepts_T_gt_2": True,
        "training_horizon": "T2 for RIO; auxiliary ScanNet samples use T1",
        "T3_T5_semantics": "zero-shot temporal-horizon extension",
        "formal_method_name": "ReScene4D Full-History (Frozen T2 Checkpoint)",
        "issued_identity": "raw query index within each prefix forward",
        "cross_prefix_namespace": "not guaranteed stable",
        "determinism": "requires three-repeat empirical gate",
        "future_information": "forbidden by exact prefix loading",
        "change_labels": (
            "disabled with change_file=None; all-static placeholders are metric-only"
        ),
    }
    answers = [
        {
            "id": "Q1",
            "question": "Does the ReScene4D code path natively accept T>2?",
            "answer": "Yes structurally: explicit variable-length prefixes survive dataset loading, collation, backbone serialization, and joint decoding.",
            "evidence_ids": ["E1", "E2", "E3", "E4", "E5", "E7"],
        },
        {
            "id": "Q2",
            "question": "At what temporal horizon was the checkpoint trained?",
            "answer": "The RIO training, validation, and test horizon is T2; the mixed ScanNet auxiliary dataset uses T1.",
            "evidence_ids": [],
        },
        {
            "id": "Q3",
            "question": "What are the semantics of T3/T4/T5 evaluation?",
            "answer": "They are zero-shot temporal-horizon extension of a frozen T2 checkpoint, not trained long-horizon ReScene4D.",
            "evidence_ids": ["E1", "E4", "E5", "E7"],
        },
        {
            "id": "Q4",
            "question": "How is instance identity represented inside one full-history forward?",
            "answer": "Each output mask/class is indexed by one raw query in the joint prefix forward; the model emits no separate track ID.",
            "evidence_ids": ["E6", "E7", "E8", "E9"],
        },
        {
            "id": "Q5",
            "question": "Is the query or track namespace stable between S1:S4 and S1:S5?",
            "answer": "No stability is guaranteed: non-parametric FPS query anchors are recomputed from the changed full prefix.",
            "evidence_ids": ["E6", "E7"],
        },
        {
            "id": "Q6",
            "question": "Is inference deterministic?",
            "answer": "The checkpoint did not enable deterministic training; evaluation will force deterministic controls and requires a three-repeat empirical fingerprint gate.",
            "evidence_ids": ["E6", "E7"],
        },
        {
            "id": "Q7",
            "question": "Can the evaluator use future information?",
            "answer": "Only if given it. The new adapter must bind each output and target to the exact observed prefix and reject later scan IDs or stage coordinates.",
            "evidence_ids": ["E1", "E2", "E9", "E10"],
        },
        {
            "id": "Q8",
            "question": "Does the change-label path affect identity evaluation?",
            "answer": "No. Full-history and persistent evaluation load change_file=None, the checkpoint disables change loss, and placeholder changes are not used for identity matching.",
            "evidence_ids": ["E10"],
        },
    ]
    return validate_code_audit(
        {
            "schema_version": 1,
            "checkpoint": checkpoint,
            "evidence": evidence,
            "conclusions": conclusions,
            "answers": answers,
        }
    )


def validate_code_audit(value: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise CodeAuditError("audit must be a mapping")
    root = dict(value)
    if set(root) != {
        "schema_version",
        "checkpoint",
        "evidence",
        "conclusions",
        "answers",
    }:
        raise CodeAuditError("audit fields differ")
    if root["schema_version"] != 1:
        raise CodeAuditError("audit schema_version must be 1")
    evidence = root["evidence"]
    if isinstance(evidence, (str, bytes)) or not isinstance(evidence, Sequence):
        raise CodeAuditError("audit evidence must be a sequence")
    if len(evidence) < 10:
        raise CodeAuditError("audit evidence is incomplete")
    ids: list[str] = []
    for row in evidence:
        if not isinstance(row, Mapping) or set(row) != _EVIDENCE_FIELDS:
            raise CodeAuditError("audit evidence fields differ")
        if not isinstance(row["id"], str) or not row["id"]:
            raise CodeAuditError("audit evidence id is invalid")
        if not isinstance(row["line"], int) or row["line"] <= 0:
            raise CodeAuditError("audit evidence line is invalid")
        for field in (
            "file_path",
            "function_or_class",
            "relevant_behavior",
            "scientific_implication",
        ):
            if not isinstance(row[field], str) or not row[field]:
                raise CodeAuditError(f"audit evidence {field} is invalid")
        ids.append(row["id"])
    if len(ids) != len(set(ids)):
        raise CodeAuditError("audit evidence IDs must be unique")
    answers = root["answers"]
    if isinstance(answers, (str, bytes)) or not isinstance(answers, Sequence):
        raise CodeAuditError("audit answers must be a sequence")
    if [row.get("id") if isinstance(row, Mapping) else None for row in answers] != [
        f"Q{index}" for index in range(1, 9)
    ]:
        raise CodeAuditError("audit must answer Q1-Q8 exactly")
    for row in answers:
        if not isinstance(row, Mapping) or set(row) != {
            "id",
            "question",
            "answer",
            "evidence_ids",
        }:
            raise CodeAuditError("audit answer fields differ")
        if not isinstance(row["question"], str) or not isinstance(row["answer"], str):
            raise CodeAuditError("audit question and answer must be text")
        if not set(row["evidence_ids"]) <= set(ids):
            raise CodeAuditError("audit answer references unknown evidence")
    conclusions = root["conclusions"]
    if not isinstance(conclusions, Mapping) or not conclusions:
        raise CodeAuditError("audit conclusions are missing")
    return root


def render_code_audit_markdown(value: Mapping[str, object]) -> str:
    audit = validate_code_audit(value)
    checkpoint = audit["checkpoint"]
    lines = [
        "# ReScene4D Full-History Code Audit",
        "",
        "## Frozen Checkpoint",
        "",
        f"- SHA256: `{checkpoint['sha256']}`",
        f"- Epoch/global step: `{checkpoint['epoch']}` / `{checkpoint['global_step']}`",
        "- RIO train/validation/test temporal horizon: `T2` / `T2` / `T2`",
        "- Auxiliary ScanNet training horizon: `T1`",
        "- Formal system name: **ReScene4D Full-History (Frozen T2 Checkpoint)**",
        "- T3-T5 status: **zero-shot temporal-horizon extension**",
        "",
        "## Code Evidence",
        "",
        "| ID | File path | Function/class | Line | Relevant behavior | Scientific implication |",
        "|---|---|---|---:|---|---|",
    ]
    for row in audit["evidence"]:
        lines.append(
            f"| {row['id']} | `{row['file_path']}` | `{row['function_or_class']}` | "
            f"{row['line']} | {row['relevant_behavior']} | {row['scientific_implication']} |"
        )
    for row in audit["answers"]:
        evidence_ids = ", ".join(row["evidence_ids"]) or "checkpoint metadata"
        lines.extend(
            [
                "",
                f"## {row['id']}. {row['question']}",
                "",
                row["answer"],
                "",
                f"Evidence: `{evidence_ids}`.",
            ]
        )
    lines.extend(
        [
            "",
            "## Evaluation Consequences",
            "",
            "Full-History may reason jointly over the exact observed prefix, but it may not access a later prefix. Official task postprocessing and deployment identity are separate: task metrics keep the registered top-k path, while identity analysis preserves the raw query index without adding persistent memory. Change labels remain disabled. Determinism and T2 parity must pass empirical smoke gates before the full run.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_new(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite audit: {path}")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/rescene4d_concerto_t2_repro.ckpt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/system_comparison/REScene_FULL_HISTORY_CODE_AUDIT.md"
        ),
    )
    args = parser.parse_args(argv)
    checkpoint = args.checkpoint
    if not checkpoint.is_absolute():
        checkpoint = PROJECT_ROOT / checkpoint
    output = args.output
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    digest = sha256_file(checkpoint)
    if digest != EXPECTED_CHECKPOINT_SHA256:
        raise CodeAuditError("checkpoint SHA256 differs from the frozen binding")
    audit = build_code_audit(
        repo_root=PROJECT_ROOT,
        checkpoint_path=checkpoint,
        checkpoint_sha256=digest,
    )
    _write_new(output, render_code_audit_markdown(audit))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CodeAuditError",
    "build_code_audit",
    "render_code_audit_markdown",
    "validate_code_audit",
]
