"""V2 numerical recipes and provenance shared by training and live consumers."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

PARENT_COMMIT = "8b5e93795817682fe70864daa545db219e1443c9"
R1_SHA256 = "629ff7624dcac15e6022906e808e2e05b3ec61c60a1116ab0e278f0cfd2368dd"
VARIANTS = ("C0", "S-BAL", "S-WORST", "Q-SEM", "A-OPEN")


def content_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def resolve_recipe(
    variant: str,
    *,
    learning_rate: str,
    seed: int = 45,
    devices: int = 2,
    num_workers: int = 2,
    timeout_seconds: float = 180,
) -> dict[str, object]:
    if (
        variant not in VARIANTS
        or learning_rate not in {"H", "L"}
        or seed not in {45, 46}
        or devices not in {1, 2}
        or not 0 <= num_workers <= 4
        or not 180 <= timeout_seconds <= 600
    ):
        raise ValueError("V2 recipe is outside the fixed numerical protocol")
    recipe = {
        "schema_version": "perception-gain-v2-recipe-v1",
        "recipe_id": f"{variant}-{learning_rate}-s{seed}",
        "architecture_variant": variant,
        "learning_rate_label": learning_rate,
        "learning_rate": {"H": 5e-5, "L": 1e-5}[learning_rate],
        "train_seed": seed,
        "base_weight_sha256": R1_SHA256,
        "devices": devices,
        "batch_size_per_gpu": 2,
        "gradient_accumulation": 16 // devices,
        "num_workers": num_workers,
        "timeout_seconds": timeout_seconds,
        "optimizer_updates": 3000,
        "warmup_updates": 150,
        "minimum_lr_fraction": 0.1,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": 0.01,
        "precision": "32-true",
        "gradient_clip_norm": 1.0,
        "freeze": "backbone_encoder",
        "frozen_encoder_eval": False,
        "objective": "raw_sum",
        "nominal_dataset_weights": [1.0, 0.8],
        "evaluation_updates": [0, 250, 750, 1500, 2250, 3000],
        "stage_worst_alpha": 0.5,
        "stage_worst_beta": 0.25,
        "query_count": 100,
        "state_capacity": 100,
        "association": "D0/LAST",
        "publisher": "lag1/mean",
        "eval_seed": 45,
    }
    recipe["training_recipe_hash"] = content_hash(recipe)
    recipe["inference_recipe_hash"] = content_hash(
        {
            key: recipe[key]
            for key in (
                "architecture_variant",
                "query_count",
                "state_capacity",
                "association",
                "publisher",
                "eval_seed",
            )
        }
    )
    return recipe


def validate_recipe(recipe: Mapping[str, object]) -> dict[str, object]:
    expected = resolve_recipe(
        str(recipe["architecture_variant"]),
        learning_rate=str(recipe["learning_rate_label"]),
        seed=int(recipe["train_seed"]),
        devices=int(recipe["devices"]),
        num_workers=int(recipe["num_workers"]),
        timeout_seconds=float(recipe["timeout_seconds"]),
    )
    # JSON treats 180 and 180.0 differently; preserve the supplied numeric representation.
    expected["timeout_seconds"] = recipe["timeout_seconds"]
    expected["training_recipe_hash"] = content_hash(
        {
            key: value
            for key, value in expected.items()
            if key not in {"training_recipe_hash", "inference_recipe_hash"}
        }
    )
    if dict(recipe) != expected:
        raise ValueError("V2 recipe/hash differs from the fixed numerical protocol")
    return dict(recipe)


def apply_recipe(config, recipe: Mapping[str, object]) -> None:
    from omegaconf import open_dict

    recipe = validate_recipe(recipe)
    if str(config.perception_training.variant) != recipe["architecture_variant"]:
        raise ValueError("recipe architecture differs from composed variant")
    with open_dict(config):
        config.perception_recipe = recipe
        config.general.seed = recipe["train_seed"]
        config.general.project_name = "persist4d_perception_gain_v2"
        config.general.experiment_name = recipe["recipe_id"]
        config.data.num_workers = recipe["num_workers"]
        for key in (
            "devices",
            "batch_size_per_gpu",
            "gradient_accumulation",
            "optimizer_updates",
            "warmup_updates",
            "minimum_lr_fraction",
            "betas",
            "eps",
            "weight_decay",
            "precision",
            "gradient_clip_norm",
            "evaluation_updates",
            "timeout_seconds",
        ):
            config.perception_training[key] = recipe[key]
        config.perception_training.existing_lr = recipe["learning_rate"]
        config.trainer.strategy = (
            "ddp_find_unused_parameters_false" if recipe["devices"] == 2 else "auto"
        )


def validate_resume_recipe(
    checkpoint: Mapping[str, object],
    recipe: Mapping[str, object],
    *,
    legacy_config: Mapping[str, object] | None = None,
) -> None:
    recipe = validate_recipe(recipe)
    saved = checkpoint.get("perception_recipe")
    if saved is not None:
        if saved != recipe:
            raise ValueError("resume recipe differs; H/L and runtime cannot be swapped")
        return
    if legacy_config is None or recipe["learning_rate_label"] != "H":
        raise ValueError(
            "legacy resume requires explicit original high-LR configuration"
        )
    training = legacy_config.get("perception_training", {})
    general = legacy_config.get("general", {})
    if not isinstance(training, Mapping) or not isinstance(general, Mapping):
        raise ValueError("legacy resume recipe is unavailable")
    fields = {
        "variant": recipe["architecture_variant"],
        "existing_lr": recipe["learning_rate"],
        **{
            key: recipe[key]
            for key in (
                "devices",
                "batch_size_per_gpu",
                "gradient_accumulation",
                "optimizer_updates",
                "warmup_updates",
                "minimum_lr_fraction",
                "betas",
                "eps",
                "weight_decay",
                "precision",
                "gradient_clip_norm",
                "evaluation_updates",
            )
        },
    }
    if (
        any(training.get(key) != value for key, value in fields.items())
        or general.get("seed") != recipe["train_seed"]
        or general.get("freeze") != recipe["freeze"]
        or general.get("frozen_encoder_eval") != recipe["frozen_encoder_eval"]
        or general.get("rootcause_objective_mode") != recipe["objective"]
    ):
        raise ValueError("legacy resume recipe does not match the imported run")
    resume = checkpoint.get("perception_resume", {})
    schedulers = checkpoint.get("lr_schedulers", [])
    optimizers = checkpoint.get("optimizer_states", [])
    if (
        resume.get("world_size") != recipe["devices"]
        or len(schedulers) != 1
        or len(optimizers) != 1
        or schedulers[0].get("base_lrs") != [recipe["learning_rate"]]
        or any(
            group.get("initial_lr") != recipe["learning_rate"]
            for group in optimizers[0].get("param_groups", [])
        )
    ):
        raise ValueError("legacy resume optimizer/scheduler/runtime recipe differs")


def remaining_gpu_hours(
    cap: float,
    prior: float,
    events: Sequence[Mapping[str, object]],
) -> float:
    if (
        not math.isfinite(cap)
        or not 0 < cap <= 192
        or not math.isfinite(prior)
        or prior < 0
    ):
        raise ValueError("cumulative cap or prior GPU usage is invalid")
    unique: dict[str, float] = {}
    for event in events:
        key, hours = str(event["event_id"]), float(event["gpu_hours"])
        if (
            not math.isfinite(hours)
            or hours < 0
            or (key in unique and unique[key] != hours)
        ):
            raise ValueError("budget event has invalid or conflicting GPU usage")
        unique[key] = hours
    return max(0.0, cap - prior - sum(unique.values()))


def executed_identity(project_root: Path, sources: Sequence[Path]) -> dict[str, object]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    hashes = {
        str(path.relative_to(project_root)): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sources
    }
    return {
        "parent_commit": PARENT_COMMIT,
        "executed_code_commit": commit,
        "relevant_source_digest": content_hash(hashes),
        "source_files": hashes,
    }
