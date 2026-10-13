"""Prediction-only latest geometry sidecar, synchronized with accepted identities."""

from dataclasses import dataclass

import torch

from models.revisit_prompt_modules import geometry6


@dataclass
class RevisitGeometryState:
    geometry: torch.Tensor
    centroid: torch.Tensor
    valid: torch.Tensor
    logical_ids: torch.Tensor
    generations: torch.Tensor
    support_counts: torch.Tensor

    @classmethod
    def empty(cls, state):
        shape = state.occupied.shape
        return cls(state.embedding.new_zeros((*shape, 6)),
                   state.embedding.new_zeros((*shape, 3)),
                   torch.zeros_like(state.occupied), state.logical_ids.clone(),
                   state.generations.clone(), torch.zeros_like(state.logical_ids))

    def update(self, commit, observation, masks, contexts, absolute_stage_index):
        """Update once, only when the shared commit accepted current support."""
        for batch, context in enumerate(contexts):
            indices = context["current_indices"]
            for query, slot in enumerate(commit.query_to_slot[batch].tolist()):
                if slot < 0 or not bool(observation.valid[batch, query]) or not bool(observation.current_supported[batch, query]):
                    continue
                if int(commit.state.last_seen[batch, slot]) != absolute_stage_index:
                    raise ValueError("geometry update was not accepted by the common commit")
                scores = masks[batch].detach().sigmoid()[indices, query]
                supported = indices[scores >= .5]
                # Stable index order is the secondary key for equal probabilities.
                order = masks[batch].detach().sigmoid()[supported, query].argsort(descending=True, stable=True)[:32]
                supported = supported[order]
                geometry, center, valid = geometry6(context["centers"][supported])
                self.geometry[batch, slot] = geometry.detach()
                self.centroid[batch, slot] = center.detach()
                self.valid[batch, slot] = valid
                self.logical_ids[batch, slot] = commit.query_to_logical_id[batch, query]
                self.generations[batch, slot] = commit.query_to_generation[batch, query]
                self.support_counts[batch, slot] = supported.numel()

    def tensors(self):
        return (self.geometry, self.centroid, self.valid, self.logical_ids,
                self.generations, self.support_counts)

    def assert_synchronized(self, state):
        occupied = state.occupied
        if (self.valid.shape != occupied.shape
                or self.geometry.shape != (*occupied.shape, 6)
                or self.centroid.shape != (*occupied.shape, 3)
                or not torch.equal(self.logical_ids[occupied], state.logical_ids[occupied])
                or not torch.equal(self.generations[occupied], state.generations[occupied])
                or (self.valid & ~occupied).any()):
            raise ValueError("geometry identity/generation differs from the accepted common state")

    def clone(self):
        return RevisitGeometryState(*(tensor.clone() for tensor in self.tensors()))

    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in self.tensors())
