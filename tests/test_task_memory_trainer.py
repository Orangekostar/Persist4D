from __future__ import annotations

import random
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from pytorch_lightning import LightningModule

from datasets.task_memory_episode import (
    StageMeta,
    TaskMemoryEpisodeBatch,
    TaskMemoryEpisodeSpec,
    TaskMemoryStageBatch,
)
from models.task_memory_criterion import TaskMemoryLossResult
from models.task_memory_routing import PredictionObservation, route_entities
from models.task_memory_state import TaskMemoryConfig, TaskMemoryState
from models.task_memory_supervision import TaskMemoryAssignment, TrainingIdentityLedger
from trainer.task_memory_trainer import (
    TaskMemoryProgress,
    TaskMemoryTrainer,
    bind_training_births,
    build_final_prediction_observation,
    capture_task_memory_rng_state,
    commit_task_memory_stage,
    restore_task_memory_rng_state,
    stage_mean_coefficients,
    task_memory_lr_multiplier,
    task_read_gradient_snapshot,
    tbptt_stage_chunks,
    training_stage_chunks,
)


def _empty_assignment() -> TaskMemoryAssignment:
    pair = (torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))
    queries = torch.empty(0, dtype=torch.long)
    return TaskMemoryAssignment(
        matcher_mode="independent",
        indices=(pair,),
        fixed_indices=(pair,),
        residual_indices=(pair,),
        reserved_queries=(queries,),
        absent_inherited_queries=(queries,),
        duplicate_queries=(queries,),
        empty_current_queries=(queries,),
        diagnostics={},
    )


class _ScalarCriterion(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight_dict: dict[str, float] = {}

    def compute_with_assignments(self, output, *_args, **_kwargs):
        return TaskMemoryLossResult(
            losses={"loss_scalar": output["loss"]},
            final_assignment=_empty_assignment(),
            aux_assignments=(),
            diagnostics={},
        )


class _CountingSGD(torch.optim.SGD):
    def __init__(self, parameters) -> None:
        super().__init__(parameters, lr=0.1)
        self.step_calls = 0

    def step(self, closure=None):
        self.step_calls += 1
        return super().step(closure)


class _TrainingStepHarness(TaskMemoryTrainer):
    def __init__(self) -> None:
        LightningModule.__init__(self)
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.config = OmegaConf.create(
            {
                "general": {
                    "rootcause_objective_mode": "raw_sum",
                    "rootcause_fail_closed_runtime": True,
                },
                "task_memory_training": {
                    "gradient_accumulation": 2,
                    "state_enabled": False,
                    "tbptt_steps": 2,
                    "window_mode": "local_pair",
                },
                "trainer": {"gradient_clip_val": 1.0},
            }
        )
        self.criterion = _ScalarCriterion()
        self.mask_type = "masks"
        self.progress = TaskMemoryProgress.initial()
        self.training_audit = []
        self.optimizer = _CountingSGD((self.weight,))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda _step: 1.0
        )
        self.logged = []
        self.backward_calls = 0
        self._trainer = SimpleNamespace(global_rank=0, world_size=1)

    @property
    def device(self) -> torch.device:
        return self.weight.device

    def optimizers(self):
        return self.optimizer

    def lr_schedulers(self):
        return self.scheduler

    def manual_backward(self, loss: torch.Tensor) -> None:
        self.backward_calls += 1
        loss.backward()

    def clip_gradients(self, *_args, **_kwargs) -> None:
        return None

    def log_dict(self, values, **_kwargs) -> None:
        self.logged.append(values)

    def _backward_context(self, _synchronize: bool):
        return nullcontext()

    def _forward_stage(self, stage, _state):
        value, targets, _names = stage.model_batch
        return {"loss": self.weight * value}, targets


def _training_batch(draw_index: int) -> TaskMemoryEpisodeBatch:
    scan_ids = ("scene0001_00", "scene0001_01")
    spec = TaskMemoryEpisodeSpec(
        reference_id="reference",
        source_sequence_id="-".join(scan_ids),
        episode_id=f"episode-{draw_index}",
        scan_ids=scan_ids,
        scan_indices=(0, 1),
        role="adaptation",
        context_index=0,
        horizon=2,
        augmentation_seed=45,
        draw_index=draw_index,
        bucket="T2",
    )
    targets = [{"labels": torch.tensor([0]), "point2segment": torch.tensor([0])}]
    stages = tuple(
        TaskMemoryStageBatch(
            model_batch=(value, targets, ("sample",)),
            stage_meta=_meta(stage),
            training_identity_keys=((('reference', 1),),),
        )
        for stage, value in enumerate((1.0, 3.0))
    )
    return TaskMemoryEpisodeBatch(
        specs=(spec,),
        stage_batches=stages,
        ambiguity_metadata=([],),
    )


def _meta(stage: int, *, rows: int = 2) -> tuple[StageMeta, ...]:
    single_scan = stage == 0
    return (
        StageMeta(
            reference_id="reference",
            episode_id="episode",
            scan_ids_in_window=("scan-0",) if stage == 0 else ("scan-0", "scan-1"),
            absolute_stage_index=stage,
            local_stage_ids=(
                torch.zeros(rows, dtype=torch.long)
                if single_scan
                else torch.arange(rows)
            ),
            original_vertex_ids=(
                (torch.arange(rows),)
                if single_scan
                else tuple(torch.tensor([0]) for _ in range(2))
            ),
            scan_vertex_offsets=(
                torch.tensor([0, rows]) if single_scan else torch.arange(3)
            ),
            point2segment=torch.arange(rows),
            segment_stage_ids=torch.arange(rows),
            augmentation_transform_id="identity-v1",
            coordinate_frame_id="reference",
            voxel_inverse=torch.arange(rows),
            full_resolution_point2segment=torch.arange(rows),
        ),
    )


def _empty_state() -> TaskMemoryState:
    return TaskMemoryState.empty(
        batch_size=1,
        capacity=2,
        feature_dim=2,
        class_count=3,
        device="cpu",
        dtype=torch.float32,
        config=TaskMemoryConfig(update_mode="last"),
    )


def _route_observation(feature: torch.Tensor) -> PredictionObservation:
    return PredictionObservation(
        features=feature,
        class_prob=torch.tensor([[[0.9, 0.05, 0.05], [0.05, 0.05, 0.9]]]),
        confidence=torch.tensor([[0.9, 0.05]]),
        valid=torch.tensor([[True, False]]),
        current_supported=torch.tensor([[True, False]]),
        previous_supported=torch.tensor([[False, False]]),
    )


@pytest.mark.parametrize("horizon", range(1, 6))
def test_stage_loss_is_an_equal_mean(horizon: int) -> None:
    coefficients = stage_mean_coefficients(horizon)

    assert coefficients == pytest.approx((1.0 / horizon,) * horizon)
    assert sum(coefficients) == pytest.approx(1.0)


def test_two_stage_tbptt_chunks_cover_each_stage_once() -> None:
    assert tbptt_stage_chunks(5, chunk_size=2) == ((0, 1), (2, 3), (4,))
    assert tbptt_stage_chunks(1, chunk_size=2) == ((0,),)


def test_stateless_full_history_releases_each_stage_graph_before_next_forward() -> None:
    assert training_stage_chunks(
        5,
        state_enabled=False,
        window_mode="full_history",
        tbptt_steps=2,
    ) == ((0,), (1,), (2,), (3,), (4,))
    assert training_stage_chunks(
        5,
        state_enabled=True,
        window_mode="local_pair",
        tbptt_steps=2,
    ) == ((0, 1), (2, 3), (4,))


def test_scheduler_covers_frozen_3000_updates_and_reaches_ten_percent() -> None:
    values = [task_memory_lr_multiplier(step, total_steps=3000) for step in range(3000)]

    assert len(values) == 3000
    assert all(np.isfinite(value) and 0.0 < value <= 1.0 for value in values)
    assert all(value >= 0.1 for value in values[149:])
    assert values[149] == pytest.approx(1.0)
    assert values[-1] == pytest.approx(0.1)


def test_task_read_gradient_snapshot_separates_zero_init_boundary() -> None:
    module = torch.nn.Module()
    module.model = torch.nn.Module()
    module.model.task_read = torch.nn.Module()
    module.model.task_read.output_projection = torch.nn.Linear(2, 2)
    module.model.task_read.query_projection = torch.nn.Linear(2, 2)
    for name, parameter in module.named_parameters():
        parameter.grad = torch.ones_like(parameter) if "output_projection" in name else torch.zeros_like(parameter)

    snapshot = task_read_gradient_snapshot(module.named_parameters())

    assert snapshot["output_projection_gradient_norm"] > 0.0
    assert snapshot["upstream_gradient_norm"] == 0.0
    assert snapshot["missing_gradient_names"] == []
    assert snapshot["nonfinite_gradient_names"] == []


def test_commit_has_equal_runtime_and_graph_shadow_with_one_writer_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _empty_state()
    feature = torch.tensor([[[3.0, 4.0], [0.0, 1.0]]], requires_grad=True)
    route = route_entities(_route_observation(feature.detach()), state, _meta(0))
    calls = 0

    from trainer import task_memory_trainer as module

    original_commit = module.commit_entities

    def counted_commit(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_commit(*args, **kwargs)

    monkeypatch.setattr(module, "commit_entities", counted_commit)
    transition = commit_task_memory_stage(
        observation=_route_observation(feature),
        route=route,
        state=state,
        stage_meta=_meta(0),
    )

    assert calls == 1
    assert transition.commit.births.tolist() == [[True, False]]
    assert transition.parity_max_abs == 0.0
    assert transition.graph_state.embedding.grad_fn is not None
    assert transition.runtime_state.embedding.grad_fn is None
    transition.graph_state.embedding.sum().backward()
    assert feature.grad is not None
    assert torch.count_nonzero(feature.grad).item() > 0


def test_detached_chunk_boundary_blocks_earlier_state_gradient() -> None:
    state = _empty_state()
    first_feature = torch.tensor([[[3.0, 4.0], [0.0, 1.0]]], requires_grad=True)
    first_route = route_entities(
        _route_observation(first_feature.detach()), state, _meta(0)
    )
    first = commit_task_memory_stage(
        observation=_route_observation(first_feature),
        route=first_route,
        state=state,
        stage_meta=_meta(0),
    )
    next_state = replace(
        first.runtime_state,
        embedding=first.runtime_state.embedding * 2.0,
    )
    current_stage = torch.tensor(1.0, requires_grad=True)

    (next_state.embedding.sum() * current_stage).backward()

    assert first_feature.grad is None
    assert current_stage.grad is not None


def test_predicted_births_feed_training_ledger_only_through_assignment() -> None:
    state = _empty_state()
    feature = torch.tensor([[[3.0, 4.0], [0.0, 1.0]]], requires_grad=True)
    route = route_entities(_route_observation(feature.detach()), state, _meta(0))
    transition = commit_task_memory_stage(
        observation=_route_observation(feature),
        route=route,
        state=state,
        stage_meta=_meta(0),
    )
    ledger = TrainingIdentityLedger()

    diagnostics = bind_training_births(
        commit=transition.commit,
        ledgers=(ledger,),
        assigned_indices=((torch.tensor([0]), torch.tensor([0])),),
        identity_keys=((("reference", 17),),),
        target_labels=(torch.tensor([2]),),
        ambiguity_metadata=([],),
    )

    assert diagnostics["bound_births"] == 1
    assert ledger.resolve(0, 0) == ("reference", 17)


def test_final_prediction_observation_uses_current_stage_support() -> None:
    features = torch.randn(1, 2, 2, requires_grad=True)
    output = {
        "query_features": features,
        "pred_logits": torch.tensor([[[8.0, -8.0, -8.0], [-8.0, -8.0, 8.0]]]),
        "pred_masks": [torch.tensor([[8.0, -8.0], [-8.0, 8.0]])],
    }

    observation = build_final_prediction_observation(
        output,
        _meta(1),
        background_class=2,
        confidence_threshold=0.5,
        mask_threshold=0.5,
        minimum_mask_support=1,
    )

    assert observation.current_supported.tolist() == [[False, True]]
    assert observation.previous_supported.tolist() == [[True, False]]
    assert observation.valid.tolist() == [[True, False]]
    assert observation.features is features
    assert observation.class_prob.grad_fn is None


def test_progress_requires_rank_synchronous_next_draw_and_counts_global_exposure() -> (
    None
):
    progress = TaskMemoryProgress.initial(next_draw_index=40)

    updated = progress.advance(
        local_draw_indices=(41,),
        horizon=5,
        world_size=2,
        global_rank=1,
    )

    assert updated.completed_local_episodes == 1
    assert updated.completed_global_episodes == 2
    assert updated.completed_global_stages == 10
    assert updated.next_draw_index == 42
    assert TaskMemoryProgress.from_state_dict(updated.state_dict()) == updated


def test_rng_capture_and_restore_replays_python_numpy_and_torch() -> None:
    random.seed(45)
    np.random.seed(45)
    torch.manual_seed(45)
    state = capture_task_memory_rng_state()
    expected = (random.random(), float(np.random.rand()), torch.rand(3))

    random.seed(99)
    np.random.seed(99)
    torch.manual_seed(99)
    restore_task_memory_rng_state(state)
    actual = (random.random(), float(np.random.rand()), torch.rand(3))

    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])


def test_training_step_means_stages_and_steps_once_per_effective_batch() -> None:
    trainer = _TrainingStepHarness()

    first_loss = trainer.training_step(_training_batch(0), 0)
    second_loss = trainer.training_step(_training_batch(1), 1)

    assert first_loss.item() == pytest.approx(2.0)
    assert second_loss.item() == pytest.approx(2.0)
    assert trainer.backward_calls == 2
    assert trainer.optimizer.step_calls == 1
    assert trainer.scheduler.last_epoch == 1
    assert trainer.weight.item() == pytest.approx(0.8)
    assert [entry["optimizer_step"] for entry in trainer.training_audit] == [
        False,
        True,
    ]
    assert trainer.progress.completed_global_episodes == 2
    assert trainer.progress.completed_global_stages == 4
    assert trainer.progress.next_draw_index == 2


def test_stateless_full_history_backpropagates_after_each_stage() -> None:
    trainer = _TrainingStepHarness()
    trainer.config.task_memory_training.window_mode = "full_history"

    trainer.training_step(_training_batch(0), 0)

    assert trainer.backward_calls == 2
    assert trainer.optimizer.step_calls == 0


def test_training_step_reads_ddp_world_size_from_attached_trainer() -> None:
    trainer = _TrainingStepHarness()
    trainer._trainer = SimpleNamespace(global_rank=1, world_size=2)

    trainer.training_step(_training_batch(1), 0)

    assert trainer.progress.completed_local_episodes == 1
    assert trainer.progress.completed_global_episodes == 2
    assert trainer.progress.completed_global_stages == 4
    assert trainer.progress.next_draw_index == 2


def test_forward_stage_sets_sparse_batch_device_before_model_call() -> None:
    data = SimpleNamespace()

    class Model:
        def __call__(self, observed, *_args, **_kwargs):
            assert observed.device == torch.device("cpu")
            return {"ok": True}

    owner = SimpleNamespace(
        config=OmegaConf.create(
            {"task_memory_training": {"state_enabled": False}}
        ),
        device=torch.device("cpu"),
        model=Model(),
        _process_raw_coordinates=lambda observed: observed,
    )
    stage = TaskMemoryStageBatch(
        model_batch=(
            data,
            [{"point2segment": torch.tensor([0])}],
            ("sample",),
        ),
        stage_meta=_meta(0),
        training_identity_keys=((('reference', 1),),),
    )

    output, _ = TaskMemoryTrainer._forward_stage(owner, stage, None)

    assert output == {"ok": True}


def test_fit_setup_does_not_instantiate_legacy_datasets() -> None:
    trainer = _TrainingStepHarness()

    trainer.setup("fit")

    assert not hasattr(trainer, "train_dataset")
    assert not hasattr(trainer, "validation_dataset")


def test_exact_resume_payload_preserves_lightning_states_progress_and_rng() -> None:
    trainer = _TrainingStepHarness()
    trainer.progress = TaskMemoryProgress(
        completed_local_episodes=8,
        completed_global_episodes=16,
        completed_global_stages=48,
        next_draw_index=16,
    )
    random.seed(45)
    np.random.seed(45)
    torch.manual_seed(45)
    checkpoint = {
        "optimizer_states": [{"sentinel": "optimizer"}],
        "lr_schedulers": [{"sentinel": "scheduler"}],
    }

    trainer.on_save_checkpoint(checkpoint)
    expected = (random.random(), float(np.random.rand()), torch.rand(3))
    resumed = _TrainingStepHarness()
    random.seed(99)
    np.random.seed(99)
    torch.manual_seed(99)
    resumed.on_load_checkpoint(checkpoint)
    resumed.on_train_start()
    actual = (random.random(), float(np.random.rand()), torch.rand(3))

    assert checkpoint["optimizer_states"] == [{"sentinel": "optimizer"}]
    assert checkpoint["lr_schedulers"] == [{"sentinel": "scheduler"}]
    assert resumed.progress == trainer.progress
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])


def test_exact_resume_resets_only_lightning_batch_fast_forward_state() -> None:
    trainer = _TrainingStepHarness()
    trainer.progress = TaskMemoryProgress(
        completed_local_episodes=8,
        completed_global_episodes=16,
        completed_global_stages=48,
        next_draw_index=16,
    )
    checkpoint = {
        "optimizer_states": [{"sentinel": "optimizer"}],
        "lr_schedulers": [{"sentinel": "scheduler"}],
        "loops": {
            "fit_loop": {
                "epoch_loop.batch_progress": {
                    "current": {
                        "completed": 7,
                        "processed": 8,
                        "ready": 8,
                        "started": 8,
                    },
                    "is_last_batch": False,
                    "total": {
                        "completed": 7,
                        "processed": 8,
                        "ready": 8,
                        "started": 8,
                    },
                },
                "epoch_loop.manual_optimization.optim_step_progress": {
                    "current": {"completed": 2, "ready": 2},
                    "total": {"completed": 2, "ready": 2},
                },
            }
        },
    }
    trainer.on_save_checkpoint(checkpoint)

    resumed = _TrainingStepHarness()
    resumed.on_load_checkpoint(checkpoint)

    batch_progress = checkpoint["loops"]["fit_loop"][
        "epoch_loop.batch_progress"
    ]
    assert batch_progress == {
        "current": {"completed": 0, "processed": 0, "ready": 0, "started": 0},
        "is_last_batch": False,
        "total": {"completed": 0, "processed": 0, "ready": 0, "started": 0},
    }
    assert checkpoint["loops"]["fit_loop"][
        "epoch_loop.manual_optimization.optim_step_progress"
    ] == {
        "current": {"completed": 2, "ready": 2},
        "total": {"completed": 2, "ready": 2},
    }
    assert resumed.progress == trainer.progress
