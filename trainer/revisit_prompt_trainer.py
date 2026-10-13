"""Frozen-parent MPR training: four causal episodes, stagewise gradients and resumes."""

from __future__ import annotations

import math
import os
import random
from pathlib import Path

import numpy as np
import torch

from models.revisit_prompt_loss import TrainingLabelLedger, auxiliary_losses, bind_training_births
from scripts.revisit_prompt_assets import NEW_PREFIXES
from trainer.trainer import InstanceSegmentation, _configured_objective_loss


def lr_scale(cursor, total_updates):
    warmup = max(1, math.ceil(.05*total_updates))
    if cursor < warmup:
        return (cursor+1)/warmup
    progress = min(1., (cursor-warmup)/max(1, total_updates-warmup-1))
    return .1+.9*(1+math.cos(math.pi*progress))/2


def adapter_state(system):
    return {k: v.detach().cpu().clone() for k, v in system.state_dict().items()
            if k.startswith(NEW_PREFIXES)}


def load_adapter_state(system, state):
    current = system.state_dict()
    expected = {k for k in current if k.startswith(NEW_PREFIXES)}
    if set(state) != expected:
        raise ValueError("adapter parameter/buffer coverage differs")
    for key, tensor in state.items():
        if tensor.shape != current[key].shape or tensor.dtype != current[key].dtype:
            raise ValueError("adapter tensor shape/dtype differs: "+key)
        current[key] = tensor
    system.load_state_dict(current, strict=True)


def _save_torch_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+".tmp")
    with temporary.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def save_resume(path, system, optimizer, scheduler, *, step, identity):
    if step < 0 or scheduler.last_epoch != step:
        raise ValueError("resume cursor must follow an actual optimizer/scheduler boundary")
    value = dict(schema=1, kind="revisit-prompt-resume", step=step, next_draw=step*4,
                 identity=identity, state_dict=adapter_state(system),
                 optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                 rng=dict(python=random.getstate(), numpy=np.random.get_state(),
                          torch_cpu=torch.get_rng_state(),
                          torch_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []))
    _save_torch_atomic(path, value)
    return value


def restore_resume(path, system, optimizer, scheduler, *, identity):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if (saved.get("schema") != 1 or saved.get("kind") != "revisit-prompt-resume"
            or saved["identity"] != identity or saved["next_draw"] != saved["step"]*4
            or saved["scheduler"]["last_epoch"] != saved["step"]):
        raise ValueError("resume identity or optimizer/draw cursor differs")
    load_adapter_state(system, saved["state_dict"])
    optimizer.load_state_dict(saved["optimizer"])
    scheduler.load_state_dict(saved["scheduler"])
    random.setstate(saved["rng"]["python"])
    np.random.set_state(saved["rng"]["numpy"])
    torch.set_rng_state(saved["rng"]["torch_cpu"])
    if saved["rng"]["torch_cuda"]:
        if not torch.cuda.is_available() or len(saved["rng"]["torch_cuda"]) != torch.cuda.device_count():
            raise ValueError("resume CUDA RNG device identity differs")
        torch.cuda.set_rng_state_all(saved["rng"]["torch_cuda"])
    return saved


class RevisitPromptTrainer(InstanceSegmentation):
    def __init__(self, config):
        super().__init__(config)
        self.automatic_optimization = False
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name.startswith(NEW_PREFIXES))
        if config.general.rootcause_objective_mode != "raw_sum":
            raise ValueError("bound E1 parent objective must remain its actual raw_sum recipe")
        self.train(False)

    def train(self, mode=True):
        super().train(False)
        self.training = mode
        self.model.train(mode)
        return self

    def new_parameters(self):
        return [p for name, p in self.named_parameters() if name.startswith(NEW_PREFIXES) and p.requires_grad]

    def make_optimizer(self, total_updates):
        if not self.new_parameters():
            raise ValueError("M0 is a frozen step0 baseline and must not be trained")
        optimizer = torch.optim.AdamW(self.new_parameters(), lr=1e-4, betas=(.9, .999),
                                     eps=1e-8, weight_decay=.01)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda cursor: lr_scale(cursor, total_updates))
        return optimizer, scheduler

    def task_and_auxiliary_loss(self, output, stage, runtime, ledger):
        losses = self.criterion(output, stage.batch[1], mask_type=self.mask_type)
        base = _configured_objective_loss(self, losses)
        ret, support, diagnostics = auxiliary_losses(output, stage, runtime.state, ledger,
                                                     variant=self.model.mpr_variant)
        total = base+.1*ret+.1*support
        if not torch.isfinite(total):
            raise FloatingPointError("MPR stage task/auxiliary objective is not finite")
        components = {k: float(v.detach()) for k, v in losses.items()}
        components.update(base=float(base.detach()), retrieval=float(ret.detach()),
                          support=float(support.detach()), total=float(total.detach()))
        return total, components, diagnostics

    def optimize_update(self, loader, records, optimizer, scheduler, *, device,
                        capture_gradients=False):
        from datasets.revisit_prompt_episode import RevisitEpisodeSpec
        from scripts.evaluate_persist4d_p6a import build_rio_class_mapper
        from scripts.revisit_prompt_runtime import RevisitEpisodeRuntime

        if [r["horizon"] for r in records] != [2, 3, 4, 5]:
            raise ValueError("every optimizer update must include the four fixed horizons")
        self.train(True)
        optimizer.zero_grad(set_to_none=True)
        mapper = build_rio_class_mapper(loader.dataset("train"))
        stages, accumulated = [], 0.
        lr = float(optimizer.param_groups[0]["lr"])
        for record in records:
            spec = RevisitEpisodeSpec.from_record(record)
            runtime = RevisitEpisodeRuntime(spec, class_count=self.model.num_classes, device=device)
            ledger = TrainingLabelLedger()
            for cpu_stage in loader.stages(spec):
                stage = cpu_stage.to(device)
                output = runtime.forward(self, stage)
                loss, components, labels = self.task_and_auxiliary_loss(output, stage, runtime, ledger)
                normalized = loss/(4*spec.horizon)
                backward = bool(normalized.requires_grad)
                if backward:
                    normalized.backward()
                finished = runtime.finish(self, output, stage, class_mapper=mapper)
                births = bind_training_births(ledger, finished["commit"], output, stage)
                accumulated += float(normalized.detach())
                stages.append(dict(horizon=spec.horizon, stage=stage.stage_meta[0].absolute_stage_index,
                    backward=backward, selected_slots=int(output["mpr_selected_slots"][0].numel()),
                    occupied_slots=int(runtime.state.occupied.sum()), losses=components,
                    auxiliary_labels=labels, birth_labels=births))
                del output, finished, loss, normalized, stage, cpu_stage
        if len(stages) != 14:
            raise ValueError("an optimizer update did not execute all fourteen stages")
        parameters = self.new_parameters()
        layer_gradients = {}
        if capture_gradients:
            layer_gradients = {name: float(p.grad.detach().norm()) if p.grad is not None else None
                              for name, p in self.named_parameters() if name.startswith(NEW_PREFIXES)}
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        if any(p.requires_grad or p.grad is not None for name, p in self.named_parameters()
               if not name.startswith(NEW_PREFIXES)):
            raise ValueError("an original parent parameter participated in optimization")
        return dict(episode_count=4, stage_count=14, loss=accumulated, lr=lr,
                    gradient_norm=float(gradient_norm), stages=stages,
                    layer_gradients=layer_gradients)

    def save_delta(self, path, *, assets, step, numeric_sources, training_identity):
        value = dict(schema=1, kind="revisit-prompt-delta", variant=self.model.mpr_variant,
            parent_seed=45, adapter_seed=self.model.mpr_seed, step=step,
            parent_reference="artifacts/native_long_postselection_v1/deployment/E1-23760.pt",
            parent_sha256=assets["hashes"]["parent"], encoder_sha256=assets["hashes"]["encoder"],
            taxonomy_sha256=assets["hashes"]["taxonomy"], numeric_sources=numeric_sources,
            training_identity=training_identity, state_dict=adapter_state(self),
            resume_optimizer_included=False)
        _save_torch_atomic(path, value)
        return value
