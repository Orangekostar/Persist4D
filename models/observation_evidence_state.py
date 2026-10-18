"""Detached regional prototypes synchronized with accepted common identities."""

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from models.task_memory_routing import _update_rates


@dataclass
class ObservationEvidenceState:
    prototype: torch.Tensor
    valid: torch.Tensor
    quality: torch.Tensor
    logical_ids: torch.Tensor
    generations: torch.Tensor
    last_updated: torch.Tensor

    @classmethod
    def empty(cls, state):
        return cls(torch.zeros_like(state.embedding, dtype=torch.float32),
                   torch.zeros_like(state.occupied),
                   torch.zeros_like(state.confidence, dtype=torch.float32),
                   state.logical_ids.clone(), state.generations.clone(),
                   torch.full_like(state.logical_ids, -1))

    @torch.no_grad()
    def update(self, commit, observation, regional, old_state, absolute_stage_index):
        self.assert_synchronized(old_state)
        if len(regional) != old_state.batch_size:
            raise ValueError("regional evidence batch differs from common state")
        for batch, evidence in enumerate(regional):
            queries = ((commit.query_to_slot[batch] >= 0)
                       & observation.valid[batch] & observation.current_supported[batch]
                       & evidence['valid']).nonzero(as_tuple=True)[0]
            slots = commit.query_to_slot[batch, queries]
            if (commit.state.last_seen[batch, slots] != absolute_stage_index).any():
                raise ValueError("regional update was not accepted by common commit")
            values = F.normalize(evidence['values'][queries].detach().float(), dim=-1, eps=1e-6)
            rates = _update_rates(observation, old_state, batch, queries).float()
            births = commit.births[batch, queries]
            mixed = F.normalize((1 - rates[:, None]) * self.prototype[batch, slots]
                                + rates[:, None] * values, dim=-1, eps=1e-6)
            self.prototype[batch, slots] = torch.where(births[:, None], values, mixed)
            confidence = observation.confidence[batch, queries].float()
            quality = (1 - rates) * self.quality[batch, slots] + rates * confidence
            self.quality[batch, slots] = torch.where(births, confidence, quality)
            self.valid[batch, slots] = True
            self.last_updated[batch, slots] = absolute_stage_index
        # Identity binding must advance even when a new object has invalid pool
        # mass; absence of an observation must not erase a retained prototype.
        self.logical_ids.copy_(commit.state.logical_ids)
        self.generations.copy_(commit.state.generations)
        self.assert_synchronized(commit.state)

    def tensors(self):
        return (self.prototype, self.valid, self.quality, self.logical_ids,
                self.generations, self.last_updated)

    def assert_synchronized(self, state):
        occupied = state.occupied
        if (self.prototype.shape != state.embedding.shape
                or any(value.shape != occupied.shape for value in self.tensors()[1:])
                or not torch.equal(self.logical_ids[occupied], state.logical_ids[occupied])
                or not torch.equal(self.generations[occupied], state.generations[occupied])
                or (self.valid & ~occupied).any()
                or any(value.requires_grad for value in self.tensors())):
            raise ValueError("observation evidence identity, shape or detached-state contract differs")

    def clone(self):
        return type(self)(*(value.clone() for value in self.tensors()))

    def nbytes(self):
        return sum(value.numel() * value.element_size() for value in self.tensors())
