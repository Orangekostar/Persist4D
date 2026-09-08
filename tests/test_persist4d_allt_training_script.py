from pathlib import Path

from scripts.train_persist4d_allt import (
    _adapter_missing_prefixes,
    _checkpoint_interval,
    _compose_config,
    _summarize_gradient_steps,
    _workload_counts,
)


def test_formal_checkpoint_interval_matches_frozen_quarters() -> None:
    assert _checkpoint_interval(400, smoke=False) == 100
    assert _checkpoint_interval(2, smoke=True) == 1


def test_gradient_step_summary_counts_only_observed_nonzero_adapter_grads() -> None:
    summary = _summarize_gradient_steps(
        [
            {
                "nonzero_adapter_gradients": ["model.memory_read.output.weight"],
                "frozen_gradient_names": [],
                "nonfinite_gradient_names": [],
            },
            {
                "nonzero_adapter_gradients": [
                    "model.memory_read.output.weight",
                    "model.memory_read.query.weight",
                ],
                "frozen_gradient_names": [],
                "nonfinite_gradient_names": [],
            },
        ]
    )

    assert summary["optimizer_step_calls"] == 2
    assert summary["nonzero_gradient_step_counts"] == {
        "model.memory_read.output.weight": 2,
        "model.memory_read.query.weight": 1,
    }
    assert summary["frozen_gradient_names"] == []
    assert summary["nonfinite_gradient_names"] == []


def test_workload_counts_distinguish_local_and_full_history_encoder_scans() -> None:
    horizon_counts = {2: 0, 3: 1, 4: 0, 5: 0}

    assert _workload_counts(horizon_counts, 1, window_mode="local_pair") == {
        "encoder_scan_count": 6,
        "supervised_stage_count": 4,
    }
    assert _workload_counts(horizon_counts, 1, window_mode="full_history") == {
        "encoder_scan_count": 7,
        "supervised_stage_count": 4,
    }


def test_c1_config_enables_only_the_frozen_qcl_inspired_adapter() -> None:
    config = _compose_config(
        variant="C1",
        pretrained=Path("/tmp/concerto.pth"),
        run_dir=Path("/tmp/C1"),
        optimizer_updates=400,
        devices=2,
        gradient_accumulation=4,
    )

    assert config.model.memory_read_enabled is False
    assert config.model.local_enhancement._target_ == (
        "models.query_competition_adapter.QueryCompetitionAdapter"
    )
    assert config.allt_training.window_mode == "local_pair"
    assert _adapter_missing_prefixes("C1") == ("model.local_enhancement.",)


def test_combined_and_full_history_variant_contracts_are_fixed() -> None:
    assert _adapter_missing_prefixes("C2") == ("model.memory_read.",)
    assert _adapter_missing_prefixes("C3") == (
        "model.local_enhancement.",
        "model.memory_read.",
    )
    assert _adapter_missing_prefixes("FH-L") == ("model.local_enhancement.",)
    assert _adapter_missing_prefixes("C0") == ()
    assert _adapter_missing_prefixes("FH-adapt") == ()
