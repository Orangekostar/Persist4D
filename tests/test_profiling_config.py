from pathlib import Path

import yaml


def test_p0_p1_a40_profiling_config() -> None:
    config_path = (
        Path(__file__).resolve().parents[1] / "conf" / "profiling" / "p0_p1_a40.yaml"
    )

    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    assert config == {
        "seed": 45,
        "horizons": [2, 3, 4, 5],
        "sequence_type": "sliding",
        "precision": "fp32",
        "gpu_index": 0,
        "freeze_mode": "backbone_encoder",
        "voxel_size": 0.02,
        "warmup_iterations": 3,
        "measurement_iterations": 10,
        "profile_scenes": 5,
        "max_batch_search": 32,
        "oom_safety_margin_mb": 512,
    }
