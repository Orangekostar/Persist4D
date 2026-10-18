"""OGR stagewise raw-sum training and exact optimizer-boundary recovery."""

import random

import numpy as np
import torch

from models.observation_grounded_loss import (
    ConstrainedTaskCriterion, constrained_assignment, identity_loss_for_assignment, supervision_context,
)
from models.revisit_prompt_loss import TrainingLabelLedger, bind_training_births
from scripts.ogr_assets import NEW_PREFIXES, frozen_numeric_policy
from trainer.revisit_prompt_trainer import RevisitPromptTrainer, _save_torch_atomic
from trainer.trainer import InstanceSegmentation, _configured_objective_loss


def adapter_state(system):
    return {name: value.detach().cpu().clone() for name, value in system.state_dict().items()
            if name.startswith(NEW_PREFIXES)}


def load_adapter_state(system, tensors):
    state = system.state_dict()
    expected = {name for name in state if name.startswith(NEW_PREFIXES)}
    if set(tensors) != expected:
        raise ValueError('OGR adapter parameter/buffer coverage differs')
    for name, value in tensors.items():
        if value.shape != state[name].shape or value.dtype != state[name].dtype:
            raise ValueError('OGR adapter shape/dtype differs: ' + name)
        state[name] = value
    system.load_state_dict(state, strict=True)


def save_resume(path, system, optimizer, scheduler, *, step, identity):
    if step < 0 or scheduler.last_epoch != step:
        raise ValueError('OGR resume must follow an actual optimizer/scheduler boundary')
    value = dict(schema=1, kind='observation-grounded-resume', step=step, next_draw=4 * step,
        identity=identity, state_dict=adapter_state(system), optimizer=optimizer.state_dict(),
        scheduler=scheduler.state_dict(), rng=dict(python=random.getstate(), numpy=np.random.get_state(),
            torch_cpu=torch.get_rng_state(),
            torch_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []))
    _save_torch_atomic(path, value)
    return value


def restore_resume(path, system, optimizer, scheduler, *, identity):
    value = torch.load(path, map_location='cpu', weights_only=False)
    if (value.get('schema') != 1 or value.get('kind') != 'observation-grounded-resume'
            or value['identity'] != identity or value['next_draw'] != value['step'] * 4
            or value['scheduler']['last_epoch'] != value['step']):
        raise ValueError('OGR resume identity, optimizer or draw cursor differs')
    load_adapter_state(system, value['state_dict'])
    optimizer.load_state_dict(value['optimizer'])
    scheduler.load_state_dict(value['scheduler'])
    random.setstate(value['rng']['python'])
    np.random.set_state(value['rng']['numpy'])
    torch.set_rng_state(value['rng']['torch_cpu'])
    if value['rng']['torch_cuda']:
        if not torch.cuda.is_available() or len(value['rng']['torch_cuda']) != torch.cuda.device_count():
            raise ValueError('OGR resume CUDA RNG device identity differs')
        torch.cuda.set_rng_state_all(value['rng']['torch_cuda'])
    return value


class ObservationGroundedTrainer(RevisitPromptTrainer):
    def __init__(self, config):
        InstanceSegmentation.__init__(self, config)
        self.automatic_optimization = False
        self.criterion = ConstrainedTaskCriterion.from_parent(self.criterion)
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name.startswith(NEW_PREFIXES))
        if config.general.rootcause_objective_mode != 'raw_sum':
            raise ValueError('OGR must preserve the parent raw_sum task objective')
        self.train(False)

    def new_parameters(self):
        return [parameter for name, parameter in self.named_parameters()
                if name.startswith(NEW_PREFIXES) and parameter.requires_grad]

    def task_and_identity_loss(self, output, stage, runtime, ledger, *, lambda_id, gradient_probe=False):
        context = supervision_context(output, stage, runtime.state, ledger)
        losses = self.criterion(output, stage.batch[1], self.mask_type, constraints=context['constraints'])
        task = _configured_objective_loss(self, losses)
        identity = output['pred_logits'].sum() * 0.
        labels = dict(forced=len(context['constraints'].forced), duplicate_sources=len(context['constraints'].ignored),
                      reliable_current_targets=len(context['reliable']))
        if self.model.ogr_identity is not None:
            if output['ogr_mid'] is None:
                raise ValueError('OGR identity arm did not execute its pre-feedback hook')
            final_assignment = self.criterion.assignments[0]
            final, labels['final'] = identity_loss_for_assignment(output['ogr_final'], final_assignment,
                                                                 context, runtime.state, ledger)
            mid = output['ogr_mid']
            mid_assignment = constrained_assignment(self.criterion.matcher, mid, stage.batch[1],
                                                    self.mask_type, context['constraints'])
            intermediate, labels['mid'] = identity_loss_for_assignment(mid['regional'], mid_assignment,
                                                                       context, runtime.state, ledger)
            identity = .5 * (intermediate + final)
        gradient_norms = None
        if gradient_probe and self.model.ogr_identity is not None:
            probe_parameter = self.model.mpr_prompt.output_layer.weight
            if task.requires_grad and identity.requires_grad:
                task_grad, = torch.autograd.grad(task, probe_parameter, retain_graph=True, allow_unused=True)
                id_grad, = torch.autograd.grad(identity, probe_parameter, retain_graph=True, allow_unused=True)
                if task_grad is not None and id_grad is not None:
                    gradient_norms = dict(task=float(task_grad.detach().norm()),
                                          identity=float(id_grad.detach().norm()))
        total = task + lambda_id * identity
        if not torch.isfinite(total):
            raise FloatingPointError('OGR constrained task/identity objective is non-finite')
        components = {name: float(value.detach()) for name, value in losses.items()}
        components.update(task=float(task.detach()), identity=float(identity.detach()), total=float(total.detach()))
        return total, components, labels, gradient_norms

    def optimize_update(self, loader, records, optimizer, scheduler, *, device, lambda_id=.1,
                        gradient_probe=False, capture_gradients=False):
        from datasets.observation_grounded_episode import ObservationEpisodeSpec
        from scripts.ogr_runtime import ObservationEpisodeRuntime

        if [record['horizon'] for record in records] != [2, 3, 4, 5]:
            raise ValueError('OGR optimizer update must include exactly H2/H3/H4/H5')
        self.train(True)
        optimizer.zero_grad(set_to_none=True)
        stages, accumulated = [], 0.
        lr = float(optimizer.param_groups[0]['lr'])
        with frozen_numeric_policy():
            for record in records:
                spec = ObservationEpisodeSpec.from_record(record)
                runtime = ObservationEpisodeRuntime(spec, class_count=self.model.num_classes, device=device)
                ledger = TrainingLabelLedger()
                for cpu_stage in loader.stages(spec):
                    stage = cpu_stage.to(device)
                    output = runtime.forward(self, stage)
                    loss, components, coverage, norms = self.task_and_identity_loss(output, stage,
                        runtime, ledger, lambda_id=lambda_id, gradient_probe=gradient_probe)
                    normalized = loss / (4 * spec.horizon)
                    backward = bool(normalized.requires_grad)
                    if backward:
                        normalized.backward()
                    # Training needs the prediction commit, not an unused NMS run.
                    # Evaluation calls the identical commit and publishes raw official candidates.
                    finished = runtime.finish(self, output, stage, publish=False)
                    births = bind_training_births(ledger, finished['commit'], output, stage)
                    accumulated += float(normalized.detach())
                    stages.append(dict(horizon=spec.horizon, stage=stage.stage_meta[0].absolute_stage_index,
                        backward=backward, selected_slots=int(output['mpr_selected_slots'][0].numel()),
                        occupied_slots=int(runtime.state.occupied.sum()), losses=components,
                        supervision=coverage, birth_labels=births, gradient_probe=norms))
                    del stage, cpu_stage, output, loss, normalized, finished
        if len(stages) != 14:
            raise ValueError('OGR update did not execute fourteen causal stage forwards')
        layer_gradients = {}
        if capture_gradients:
            layer_gradients = {name: float(parameter.grad.detach().norm()) if parameter.grad is not None else None
                              for name, parameter in self.named_parameters() if name.startswith(NEW_PREFIXES)}
        norm = torch.nn.utils.clip_grad_norm_(self.new_parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        if any(parameter.requires_grad or parameter.grad is not None for name, parameter in self.named_parameters()
               if not name.startswith(NEW_PREFIXES)):
            raise ValueError('frozen parent parameter participated in OGR optimization')
        return dict(episode_count=4, stage_count=14, loss=accumulated, lr=lr,
                    gradient_norm=float(norm), stages=stages, layer_gradients=layer_gradients)

    def save_delta(self, path, *, assets, step, numeric_sources, training_identity):
        value = dict(schema=1, kind='observation-grounded-delta', variant=self.model.ogr_variant,
            parent_seed=45, adapter_seed=self.model.mpr_seed, step=step,
            parent_reference='artifacts/native_long_postselection_v1/deployment/E1-23760.pt',
            parent_sha256=assets['hashes']['parent'], encoder_sha256=assets['hashes']['encoder'],
            taxonomy_sha256=assets['hashes']['taxonomy'], numeric_sources=numeric_sources,
            training_identity=training_identity, state_dict=adapter_state(self), resume_optimizer_included=False)
        _save_torch_atomic(path, value)
        return value
