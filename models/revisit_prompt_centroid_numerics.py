"""Evaluation-only fixed accumulation order for original segment centers."""
from contextlib import contextmanager

import torch

from models.revisit_prompt_modules import segment_context as original_segment_context
from models.revisit_prompt_numerics import stable_group_mean


def stable_segment_context(raw_coordinates, point2segment, features, padding_mask, stage_meta):
    context = original_segment_context(raw_coordinates, point2segment, features, padding_mask, stage_meta)
    raw = raw_coordinates.to(features.device)
    groups = point2segment.to(device=features.device, dtype=torch.long)
    context['centers'] = stable_group_mean(raw[:, :3], groups)
    return context


@contextmanager
def evaluation_centroids():
    from models import revisit_prompt as prompt

    original = prompt.segment_context
    prompt.segment_context = stable_segment_context
    try:
        yield
    finally:
        prompt.segment_context = original
