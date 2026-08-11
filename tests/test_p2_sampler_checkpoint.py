import importlib
import hashlib
import json
import sys
import types
from contextlib import contextmanager

import pytest
import torch
from lightning_fabric.utilities.distributed import DistributedSamplerWrapper
from torch.utils.data import DataLoader, Dataset

from datasets.multi_dataset import MultiDataset

CHECKPOINT_KEY = "p2_train_sampler_generator"
OPTIMIZER_CONTRACT_KEY = "p2_optimizer_parameter_contract"


class SizedDataset(Dataset):
    def __init__(self, size: int) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> int:
        return index


@contextmanager
def _load_trainer_module(monkeypatch: pytest.MonkeyPatch):
    torch_scatter = types.ModuleType("torch_scatter")
    torch_scatter.scatter_mean = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "torch_scatter", torch_scatter)

    lightning = types.ModuleType("pytorch_lightning")
    lightning.LightningModule = object
    lightning.utilities = types.SimpleNamespace(grad_norm=lambda *args, **kwargs: {})
    monkeypatch.setitem(sys.modules, "pytorch_lightning", lightning)

    previous_module = sys.modules.pop("trainer.trainer", None)
    trainer_package = sys.modules.get("trainer")
    missing = object()
    previous_attribute = (
        getattr(trainer_package, "trainer", missing)
        if trainer_package is not None
        else missing
    )
    loaded_module = importlib.import_module("trainer.trainer")
    try:
        yield loaded_module
    finally:
        sys.modules.pop("trainer.trainer", None)
        trainer_package = sys.modules.get("trainer")
        if previous_module is not None:
            sys.modules["trainer.trainer"] = previous_module
        if trainer_package is not None:
            if previous_attribute is missing:
                if getattr(trainer_package, "trainer", None) is loaded_module:
                    delattr(trainer_package, "trainer")
            else:
                trainer_package.trainer = previous_attribute


def _make_dataset(*, sampler_seed=45) -> MultiDataset:
    return MultiDataset(
        [SizedDataset(10), SizedDataset(12)],
        weights=[1.0, 0.8],
        sampler_seed=sampler_seed,
        fail_closed=True,
    )


def _make_module(trainer_module, *, p2_fail_closed_runtime: bool):
    train_config = object()
    validation_config = object()
    module = object.__new__(trainer_module.InstanceSegmentation)
    module.config = types.SimpleNamespace(
        general=types.SimpleNamespace(
            p2_fail_closed_runtime=p2_fail_closed_runtime,
        ),
        data=types.SimpleNamespace(
            train_dataset=train_config,
            validation_dataset=validation_config,
        ),
    )
    return module, train_config, validation_config


def _patch_dataset_instantiation(
    monkeypatch: pytest.MonkeyPatch,
    trainer_module,
    train_config,
    validation_config,
) -> None:
    def instantiate(config):
        if config is train_config:
            return _make_dataset()
        if config is validation_config:
            return types.SimpleNamespace()
        raise AssertionError(f"unexpected config: {config!r}")

    monkeypatch.setattr(trainer_module.hydra.utils, "instantiate", instantiate)


def _checkpoint_after_complete_epochs(trainer_module, module, epochs: int):
    for _ in range(epochs):
        list(module.train_dataset.sampler)
    checkpoint = _checkpoint_payload_for_module(module)
    trainer_module.InstanceSegmentation.on_save_checkpoint(module, checkpoint)
    return checkpoint


def _checkpoint_payload_for_module(module) -> dict:
    if not module.config.general.p2_fail_closed_runtime:
        return {}
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([parameter], lr=1e-3)
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    module._trainer = types.SimpleNamespace(optimizers=[optimizer])
    module.named_parameters = lambda: iter([("model.weight", parameter)])
    return {
        "state_dict": {"model.weight": parameter.detach().clone()},
        "optimizer_states": [optimizer.state_dict()],
    }


def test_sampler_checkpoint_restores_next_epoch_after_load_then_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _load_trainer_module(monkeypatch) as trainer_module:
        running, _, _ = _make_module(
            trainer_module,
            p2_fail_closed_runtime=False,
        )
        running.train_dataset = _make_dataset()
        checkpoint = _checkpoint_after_complete_epochs(
            trainer_module,
            running,
            epochs=3,
        )
        uninterrupted_next_epoch = list(running.train_dataset.sampler)

        payload = checkpoint[CHECKPOINT_KEY]
        assert payload["schema_version"] == 1
        assert payload["resume_scope"] == "completed_epoch_boundary_only"
        assert payload["mid_epoch_resume_supported"] is False
        assert payload["dataloader_prefetch_state_checkpointed"] is False
        assert isinstance(payload["generator_state"], torch.Tensor)

        resumed, train_config, validation_config = _make_module(
            trainer_module,
            p2_fail_closed_runtime=False,
        )
        trainer_module.InstanceSegmentation.on_load_checkpoint(resumed, checkpoint)
        _patch_dataset_instantiation(
            monkeypatch,
            trainer_module,
            train_config,
            validation_config,
        )
        trainer_module.InstanceSegmentation.setup(resumed, "fit")

        assert list(resumed.train_dataset.sampler) == uninterrupted_next_epoch


def test_formal_checkpoint_writes_optimizer_parameter_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _load_trainer_module(monkeypatch) as trainer_module:
        module, _, _ = _make_module(
            trainer_module,
            p2_fail_closed_runtime=True,
        )
        module.train_dataset = _make_dataset()
        parameter = torch.nn.Parameter(torch.ones(2))
        optimizer = torch.optim.AdamW([parameter], lr=1e-3)
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        module._trainer = types.SimpleNamespace(optimizers=[optimizer])
        module.named_parameters = lambda: iter([("model.weight", parameter)])
        checkpoint = {
            "state_dict": {"model.weight": parameter.detach().clone()},
            "optimizer_states": [optimizer.state_dict()],
        }

        trainer_module.InstanceSegmentation.on_save_checkpoint(module, checkpoint)

        parameter_id = checkpoint["optimizer_states"][0]["param_groups"][0][
            "params"
        ][0]
        assert checkpoint[OPTIMIZER_CONTRACT_KEY] == {
            "schema_version": 1,
            "state_dict": {
                "model.weight": {
                    "shape": [2],
                    "dtype": "torch.float32",
                }
            },
            "state_dict_schema_sha256": hashlib.sha256(
                json.dumps(
                    [["model.weight", [2], "torch.float32"]],
                    ensure_ascii=True,
                    separators=(",", ":"),
                ).encode("ascii")
                ).hexdigest(),
            "param_groups": [[parameter_id]],
            "parameters": {
                parameter_id: {
                    "name": "model.weight",
                    "shape": [2],
                    "dtype": "torch.float32",
                }
            },
            "trainable_parameters": [
                ["model.weight", [2], "torch.float32"],
            ],
            "trainable_parameter_schema_sha256": hashlib.sha256(
                json.dumps(
                    [["model.weight", [2], "torch.float32"]],
                    ensure_ascii=True,
                    separators=(",", ":"),
                ).encode("ascii")
            ).hexdigest(),
        }


def test_sampler_checkpoint_restores_when_lightning_setup_precedes_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _load_trainer_module(monkeypatch) as trainer_module:
        running, _, _ = _make_module(
            trainer_module,
            p2_fail_closed_runtime=True,
        )
        running.train_dataset = _make_dataset()
        checkpoint = _checkpoint_after_complete_epochs(
            trainer_module,
            running,
            epochs=2,
        )
        uninterrupted_next_epoch = list(running.train_dataset.sampler)

        resumed, train_config, validation_config = _make_module(
            trainer_module,
            p2_fail_closed_runtime=True,
        )
        _patch_dataset_instantiation(
            monkeypatch,
            trainer_module,
            train_config,
            validation_config,
        )
        trainer_module.InstanceSegmentation.setup(resumed, "fit")
        trainer_module.InstanceSegmentation.on_load_checkpoint(resumed, checkpoint)

        assert list(resumed.train_dataset.sampler) == uninterrupted_next_epoch


def test_non_p2_sampler_checkpoint_load_accepts_legacy_checkpoint_without_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _load_trainer_module(monkeypatch) as trainer_module:
        resumed, train_config, validation_config = _make_module(
            trainer_module,
            p2_fail_closed_runtime=False,
        )
        trainer_module.InstanceSegmentation.on_load_checkpoint(resumed, {})
        _patch_dataset_instantiation(
            monkeypatch,
            trainer_module,
            train_config,
            validation_config,
        )
        trainer_module.InstanceSegmentation.setup(resumed, "fit")

        fresh_first_epoch = list(_make_dataset().sampler)
        assert list(resumed.train_dataset.sampler) == fresh_first_epoch


def test_default_sampler_without_generator_does_not_write_checkpoint_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _load_trainer_module(monkeypatch) as trainer_module:
        module, _, _ = _make_module(
            trainer_module,
            p2_fail_closed_runtime=False,
        )
        module.train_dataset = _make_dataset(sampler_seed=None)
        checkpoint = {"existing": "value"}

        trainer_module.InstanceSegmentation.on_save_checkpoint(module, checkpoint)

        assert checkpoint == {"existing": "value"}


def test_p2_runtime_rejects_checkpoint_without_explicit_sampler_generator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _load_trainer_module(monkeypatch) as trainer_module:
        module, _, _ = _make_module(
            trainer_module,
            p2_fail_closed_runtime=True,
        )
        module.train_dataset = _make_dataset(sampler_seed=None)

        with pytest.raises(RuntimeError, match="sampler.*explicit generator"):
            trainer_module.InstanceSegmentation.on_save_checkpoint(module, {})


def test_p2_runtime_rejects_fit_setup_without_explicit_sampler_generator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _load_trainer_module(monkeypatch) as trainer_module:
        module, train_config, validation_config = _make_module(
            trainer_module,
            p2_fail_closed_runtime=True,
        )

        def instantiate(config):
            if config is train_config:
                return _make_dataset(sampler_seed=None)
            if config is validation_config:
                return types.SimpleNamespace()
            raise AssertionError(f"unexpected config: {config!r}")

        monkeypatch.setattr(
            trainer_module.hydra.utils,
            "instantiate",
            instantiate,
        )

        with pytest.raises(RuntimeError, match="sampler.*explicit generator"):
            trainer_module.InstanceSegmentation.setup(module, "fit")


def test_distributed_sampler_rank_streams_resume_from_checkpoint_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _load_trainer_module(monkeypatch) as trainer_module:
        running_modules = []
        running_wrappers = []
        for rank in range(2):
            module, _, _ = _make_module(
                trainer_module,
                p2_fail_closed_runtime=True,
            )
            module.train_dataset = _make_dataset()
            running_modules.append(module)
            running_wrappers.append(
                DistributedSamplerWrapper(
                    module.train_dataset.sampler,
                    num_replicas=2,
                    rank=rank,
                    shuffle=True,
                    seed=45,
                    drop_last=False,
                )
            )

        for epoch in range(3):
            for wrapper in running_wrappers:
                wrapper.set_epoch(epoch)
                list(wrapper)

        rank_states = [
            module.train_dataset.sampler.generator.get_state()
            for module in running_modules
        ]
        assert torch.equal(rank_states[0], rank_states[1])
        checkpoint = _checkpoint_payload_for_module(running_modules[0])
        trainer_module.InstanceSegmentation.on_save_checkpoint(
            running_modules[0],
            checkpoint,
        )

        uninterrupted_next_rank_streams = []
        for wrapper in running_wrappers:
            wrapper.set_epoch(3)
            uninterrupted_next_rank_streams.append(list(wrapper))

        resumed_rank_streams = []
        for rank in range(2):
            resumed, _, _ = _make_module(
                trainer_module,
                p2_fail_closed_runtime=True,
            )
            resumed.train_dataset = _make_dataset()
            trainer_module.InstanceSegmentation.on_load_checkpoint(
                resumed,
                checkpoint,
            )
            wrapper = DistributedSamplerWrapper(
                resumed.train_dataset.sampler,
                num_replicas=2,
                rank=rank,
                shuffle=True,
                seed=45,
                drop_last=False,
            )
            wrapper.set_epoch(3)
            resumed_rank_streams.append(list(wrapper))

        assert resumed_rank_streams == uninterrupted_next_rank_streams


def test_lightning_checkpoint_resume_restores_next_epoch_sampler_stream(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint

    from trainer.trainer import InstanceSegmentation

    train_config = object()
    validation_config = object()

    def instantiate(config):
        if config is train_config:
            return _make_dataset()
        if config is validation_config:
            return SizedDataset(1)
        raise AssertionError(f"unexpected config: {config!r}")

    monkeypatch.setattr("trainer.trainer.hydra.utils.instantiate", instantiate)

    class SamplerCheckpointHarness(InstanceSegmentation):
        def __init__(self) -> None:
            pl.LightningModule.__init__(self)
            self.config = types.SimpleNamespace(
                general=types.SimpleNamespace(p2_fail_closed_runtime=True),
                data=types.SimpleNamespace(
                    train_dataset=train_config,
                    validation_dataset=validation_config,
                ),
            )
            self._pending_train_sampler_generator_state = None
            self.scale = torch.nn.Parameter(torch.tensor(1.0))
            self.samples_by_epoch = {}
            self.load_saw_train_dataset = None

        def training_step(self, batch, batch_idx):
            self.samples_by_epoch.setdefault(int(self.current_epoch), []).extend(
                batch.tolist()
            )
            return self.scale * 0.0

        def train_dataloader(self):
            return DataLoader(
                self.train_dataset,
                batch_size=4,
                sampler=self.train_dataset.sampler,
                num_workers=0,
            )

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.1)

        def on_train_batch_end(self, outputs, batch, batch_idx):
            return None

        def on_load_checkpoint(self, checkpoint):
            self.load_saw_train_dataset = hasattr(self, "train_dataset")
            super().on_load_checkpoint(checkpoint)

    checkpoint_callback = ModelCheckpoint(
        dirpath=tmp_path / "checkpoints",
        filename="{epoch:03d}",
        auto_insert_metric_name=False,
        every_n_epochs=1,
        save_top_k=-1,
        save_on_train_epoch_end=True,
    )
    trainer_kwargs = {
        "accelerator": "cpu",
        "devices": 1,
        "logger": False,
        "enable_model_summary": False,
        "enable_progress_bar": False,
        "limit_val_batches": 0,
        "num_sanity_val_steps": 0,
    }
    running = SamplerCheckpointHarness()
    trainer = pl.Trainer(
        max_epochs=2,
        callbacks=[checkpoint_callback],
        **trainer_kwargs,
    )
    trainer.fit(running)
    epoch_zero_checkpoint = tmp_path / "checkpoints" / "000.ckpt"
    assert epoch_zero_checkpoint.is_file()

    resumed = SamplerCheckpointHarness()
    resume_trainer = pl.Trainer(
        max_epochs=2,
        enable_checkpointing=False,
        **trainer_kwargs,
    )
    resume_trainer.fit(resumed, ckpt_path=epoch_zero_checkpoint)

    assert resumed.load_saw_train_dataset is True
    assert resumed.samples_by_epoch[1] == running.samples_by_epoch[1]
