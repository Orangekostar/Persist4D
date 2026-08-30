#!/usr/bin/env python3
"""Run the fixed-real-batch ReScene objective and EOS audit."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_p2_native_smoke import (
    DEFAULT_CHECKPOINT,
    TINY_SAMPLE_NAME,
    _compose_runtime,
    _forward_losses,
    _materialize_named_train_batch,
    seed_everything,
)
from utils.rescene_objective_audit import (
    compare_gradients,
    eos_gradient_gate,
    objective_contribution_rows,
    optimized_objective,
)
from utils.rescene_rootcause_preflight import canonical_sha256

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "artifacts/rescene_task_learning_root_cause_v1/audit"
)


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _gradient_vector(
    objective: torch.Tensor,
    named_parameters: list[tuple[str, torch.nn.Parameter]],
    *,
    retain_graph: bool,
) -> tuple[torch.Tensor, list[str]]:
    names = [name for name, _ in named_parameters]
    parameters = [parameter for _, parameter in named_parameters]
    gradients = torch.autograd.grad(
        objective,
        parameters,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    values = [
        (
            torch.zeros_like(parameter).detach().cpu().float().reshape(-1)
            if gradient is None
            else gradient.detach().cpu().float().reshape(-1)
        )
        for parameter, gradient in zip(parameters, gradients, strict=True)
    ]
    if not values:
        raise RuntimeError("objective audit selected no trainable parameters")
    return torch.cat(values), names


def _publish(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _json_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True)
        + "\n"
    ).encode("ascii")


def run_audit(*, device_index: int) -> dict[str, object]:
    if not torch.cuda.is_available() or not 0 <= device_index < torch.cuda.device_count():
        raise ValueError("objective audit device is unavailable")
    device = torch.device(f"cuda:{device_index}")
    seed_everything(45)
    config, system = _compose_runtime(DEFAULT_CHECKPOINT, device)
    data, targets, sample, provenance = _materialize_named_train_batch(
        config, TINY_SAMPLE_NAME, device
    )
    system.train()
    seed_everything(45)
    output, losses, _ = _forward_losses(system, data, targets)
    weights = system.criterion.weight_dict
    rows = objective_contribution_rows(losses, weights)
    weighted = optimized_objective(losses, weights, mode="weighted")
    raw_sum = optimized_objective(losses, weights, mode="raw_sum")

    objective_parameters = [
        (name, parameter)
        for name, parameter in system.named_parameters()
        if parameter.requires_grad and name.startswith("model.")
    ]
    weighted_gradient, objective_names = _gradient_vector(
        weighted, objective_parameters, retain_graph=True
    )
    raw_gradient, _ = _gradient_vector(
        raw_sum, objective_parameters, retain_graph=True
    )

    outputs_without_aux = {
        key: value for key, value in output.items() if key != "aux_outputs"
    }
    indices = system.criterion.matcher(
        outputs_without_aux, targets, system.mask_type
    )
    class_parameters = [
        (name, parameter)
        for name, parameter in system.named_parameters()
        if parameter.requires_grad and name.startswith("model.class_embed_head.")
    ]
    original_eos = float(system.criterion.empty_weight[-1].detach().cpu().item())
    eos_losses: dict[str, float] = {}
    eos_gradients: dict[str, torch.Tensor] = {}
    eos_names: list[str] = []
    try:
        for eos in (0.1, 0.2):
            system.criterion.empty_weight[-1] = eos
            classification = system.criterion.loss_labels(
                outputs_without_aux,
                targets,
                indices,
                num_masks=torch.tensor(1.0, device=device),
                mask_type=system.mask_type,
            )["loss_ce"]
            gradient, eos_names = _gradient_vector(
                classification, class_parameters, retain_graph=True
            )
            eos_losses[str(eos)] = float(classification.detach().cpu().item())
            eos_gradients[str(eos)] = gradient
    finally:
        system.criterion.empty_weight[-1] = original_eos
    eos_comparison = compare_gradients(eos_gradients["0.1"], eos_gradients["0.2"])
    objective_comparison = compare_gradients(weighted_gradient, raw_gradient)
    portable_config = {
        "config_name": "config_p2_rescene4d_concerto_t2",
        "seed": 45,
        "objective_modes": ["weighted", "raw_sum"],
        "eos_values": [0.1, 0.2],
        "sample_id": TINY_SAMPLE_NAME,
        "pretrained_checkpoint_sha256": (
            "845ec7dec97a5fabff8fadb5d9858ac6734347b612d1a4b574213419c139de07"
        ),
    }
    return {
        "schema_version": 1,
        "status": "pass",
        "source_commit": _git_head(),
        "official_rescene_commit": "fb2fe42eb8f1e926567c48eea9acb874e608ee10",
        "portable_config_sha256": canonical_sha256(portable_config),
        "sample": sample,
        "input_provenance": provenance,
        "loss_rows": rows,
        "objectives": {
            "weighted": float(weighted.detach().cpu().item()),
            "raw_sum": float(raw_sum.detach().cpu().item()),
        },
        "objective_gradient": {
            **objective_comparison,
            "left": "weighted",
            "right": "raw_sum",
            "parameter_names": objective_names,
        },
        "eos": {
            "classification_losses": eos_losses,
            "gradient_comparison_0.1_to_0.2": eos_comparison,
            "parameter_names": eos_names,
            "gate": eos_gradient_gate(eos_comparison),
        },
    }


def _loss_markdown(result: dict[str, object]) -> str:
    rows = result["loss_rows"]
    lines = [
        "# ReScene Loss Semantics",
        "",
        "Status: `PASS`",
        "",
        "The fixed real batch uses a seed-45 randomly initialized task decoder and the verified pretrained Concerto encoder. The public-code objective includes every returned value once; the local weighted objective excludes per-layer contrastive diagnostics and applies the criterion weight dictionary.",
        "",
        "| loss key | class | raw | upstream multiplier | local multiplier | upstream contribution | local contribution |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['loss_key']} | {row['classification']} | {row['raw_value']:.9g} | "
            f"{row['upstream_multiplier']:.9g} | {row['local_weighted_multiplier']:.9g} | "
            f"{row['upstream_contribution']:.9g} | {row['local_weighted_contribution']:.9g} |"
        )
    objectives = result["objectives"]
    eos = result["eos"]
    comparison = eos["gradient_comparison_0.1_to_0.2"]
    gate = eos["gate"]
    lines.extend(
        [
            "",
            f"Weighted objective: `{objectives['weighted']:.12g}`.",
            f"Raw released-code objective: `{objectives['raw_sum']:.12g}`.",
            "",
            "## EOS",
            "",
            f"EOS 0.1 versus 0.2 class-head gradient cosine is `{comparison['cosine']:.12g}` and relative norm difference is `{comparison['relative_norm_difference']:.12g}`. R5 authorization is `{str(gate['authorized']).lower()}` under the preregistered cosine 0.98 / relative-norm 0.10 gate.",
            "",
        ]
    )
    return "\n".join(lines)


def _diff_markdown(result: dict[str, object]) -> str:
    return "\n".join(
        [
            "# Upstream / Local Training Difference",
            "",
            "Status: `PASS`",
            "",
            "| Item | Pinned public source | Local control | Controlled action |",
            "| --- | --- | --- | --- |",
            "| final objective | raw sum of all returned losses | weight-dictionary reducer with per-layer contrastive diagnostics excluded | R1 changes only this reducer |",
            "| accumulation | 1 | 8 | physical-batch audit before any R2 curve |",
            "| EOS | public config 0.1; paper 0.2 | 0.2 | fixed-batch gradient audit before any R5 curve |",
            "| class filter | [0, 1] | [0, 1, 255] | full inventory before any R4 curve |",
            "",
            f"Numeric objective and EOS evidence is bound by portable config SHA-256 `{result['portable_config_sha256']}`.",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    result = run_audit(device_index=args.device)
    output_dir = args.output_dir
    _publish(output_dir / "upstream_local_diff.json", _json_bytes(result))
    _publish(
        output_dir / "LOSS_SEMANTICS.md",
        _loss_markdown(result).encode("ascii"),
    )
    _publish(
        output_dir / "UPSTREAM_LOCAL_DIFF.md",
        _diff_markdown(result).encode("ascii"),
    )
    print(json.dumps({"status": "pass", "output": "repo:artifacts/rootcause/audit"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
