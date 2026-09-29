"""Prediction and fixed-record adapters without historical campaign globals."""
from pathlib import Path

import torch
from torch.nn import functional as F

from models.qp_mn_heads import (
    Module,
    RawQualityHead,
    mask_loss,
    quality_loss,
    shape_terms,
)
from models.short_module_heads import apply_module as legacy_apply_module
from models.short_module_heads import quality_inputs


def validate_output_paths(root, artifacts, protected):
    outputs = [Path(root).resolve(), Path(artifacts).resolve()]
    for output in outputs:
        for path in (Path(p).resolve() for p in protected):
            if output == path or output.is_relative_to(path) or path.is_relative_to(output):
                raise ValueError(f'output overlaps protected historical path: {output}')
    if outputs[0] == outputs[1]:
        raise ValueError('runtime and public artifact roots must differ')


@torch.no_grad()
def apply_module(module, head, parent, descriptor, *, system):
    module = Module(module)
    if module not in (Module.Q_A0, Module.Q_P):
        legacy = {Module.B0: 'B0', Module.Q2_F: 'Q2', Module.M_C: 'M1', Module.M_N: 'M1'}
        return legacy_apply_module(legacy[module], head, parent, descriptor, system=system)
    if not isinstance(head, RawQualityHead) or head.parent_skip != (module is Module.Q_P):
        raise ValueError('quality mode and parent-score skip differ')
    device = next(head.parameters()).device
    evidence = parent.soft_evidence
    if evidence is None:
        raise ValueError('quality head requires native soft evidence')
    scores = parent.pred_scores.to(device)
    x = quality_inputs(evidence.query_features.to(device), descriptor['h'].to(device),
                       evidence.class_probabilities.to(device),
                       F.one_hot(evidence.source_class_ids.to(device), descriptor['classes']).float(),
                       scores, descriptor['support'].to(device),
                       descriptor['foreground_probability'].to(device))
    result = parent.prediction()
    result['pred_scores'] = head(x, scores).cpu()
    if not torch.isfinite(result['pred_scores']).all():
        raise ValueError('nonfinite quality scores')
    return result


def segment_samples(record, draw, device):
    generator = torch.Generator().manual_seed(draw['segment_seed'])
    stages, weights = record['descriptor']['segment_stages'], record['weights']
    samples = []
    for stage in range(draw['horizon']):
        eligible = torch.where((stages == stage) & (weights > 0))[0]
        samples.append(eligible[torch.randperm(len(eligible), generator=generator)[:2048]].to(device))
    return samples


def batch_loss(module, head, records, draws, *, classes, device):
    """Preserve the paired plan; aggregate Q and M over all candidate slots."""
    module = Module(module)
    is_quality = module in (Module.Q2_F, Module.Q_A0, Module.Q_P)
    if module is Module.B0:
        raise ValueError('B0 has no training objective')
    xs, ys, weights, scores, geometry = [], [], [], [], []
    for draw in draws:
        record = records[draw['input_id']]
        selected = torch.tensor(draw['candidates'], dtype=torch.long)
        desc = record['descriptor']
        query, h = record['query'][selected].to(device), desc['h'][selected].to(device)
        one_hot = F.one_hot(record['class_ids'][selected].to(device), classes).float()
        if is_quality:
            parent_scores = record['scores'][selected].to(device)
            xs.append(quality_inputs(query, h, record['class_probabilities'][selected].to(device),
                                     one_hot, parent_scores, desc['support'][selected].to(device),
                                     desc['foreground_probability'][selected].to(device)))
            scores.append(parent_scores)
            ys.append(record['quality']['temporal'][selected].float().to(device))
            weights.append(record['quality']['valid'][selected].float().to(device))
        else:
            stages = desc['segment_stages'].to(device)
            delta = head(record['features'].to(device), query, h, one_hot, stages)
            geometry.append(shape_terms(logits=record['logits'][:, selected].to(device) + delta,
                delta=delta, targets=record['targets'][:, selected].to(device),
                weights=record['weights'].to(device), segment_stages=stages,
                matched=(record['assignment'][selected] >= 0).to(device),
                usable=record['quality']['geometry_valid'][selected].to(device),
                sample_indices=segment_samples(record, draw, device)))
    if is_quality:
        x, y, w, parent = torch.cat(xs), torch.cat(ys), torch.cat(weights), torch.cat(scores)
        prediction = head(x, parent) if isinstance(head, RawQualityHead) else head(x)
        loss = quality_loss(prediction, y, w)
        return loss, {'B': y.numel(), 'Npos': int((w > 0).sum()),
                      'quality_numerator': float(((prediction.detach() - y).square() * w).sum()),
                      'quality_weight': float(w.sum()), 'quality_loss': loss.detach().item()}, {
                          'shape': None, 'regularization': None}
    loss, detail = mask_loss(module, geometry)
    regularization = .01 * torch.stack([r['regularization'] for r in geometry]).mean()
    return loss, detail, {'shape': loss - regularization, 'regularization': regularization}
