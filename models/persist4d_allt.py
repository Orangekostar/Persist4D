"""ReScene extension with the single Persist4D All-T adapter insertion point."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from torch import Tensor, nn

from models.persistent_memory_read import DetachedMemoryReadState, PersistentMemoryRead
from models.rescene import ReScene


class Persist4DModelError(RuntimeError):
    """Raised when model construction or R1 subtree loading is unsafe."""


def strict_load_r1_with_named_adapters(
    module: nn.Module,
    state_dict: Mapping[str, Tensor],
    *,
    allowed_missing_prefixes: Sequence[str],
) -> dict[str, object]:
    if not isinstance(module, nn.Module):
        raise Persist4DModelError("R1 load target must be a module")
    if not isinstance(state_dict, Mapping):
        raise Persist4DModelError("R1 state_dict must be a mapping")
    if isinstance(allowed_missing_prefixes, (str, bytes)):
        raise Persist4DModelError("adapter missing-key prefixes are invalid")
    prefixes = tuple(allowed_missing_prefixes)
    if (
        not prefixes
        or any(not isinstance(prefix, str) or not prefix for prefix in prefixes)
        or len(set(prefixes)) != len(prefixes)
    ):
        raise Persist4DModelError("adapter missing-key prefixes are invalid")
    expected_missing = sorted(
        key for key in module.state_dict() if key.startswith(prefixes)
    )
    if not expected_missing:
        raise Persist4DModelError("named adapter prefixes match no model state")
    module_keys = set(module.state_dict())
    incoming_keys = set(state_dict)
    missing = sorted(module_keys - incoming_keys)
    unexpected = sorted(incoming_keys - module_keys)
    if unexpected:
        raise Persist4DModelError(f"R1 state has unexpected keys: {unexpected}")
    if missing != expected_missing:
        unapproved = sorted(set(missing) - set(expected_missing))
        raise Persist4DModelError(
            f"R1 state has unapproved missing keys: {unapproved or missing}"
        )
    incompatible = module.load_state_dict(state_dict, strict=False)
    if sorted(incompatible.missing_keys) != missing or incompatible.unexpected_keys:
        raise Persist4DModelError("R1 state changed during strict adapter load")
    return {
        "loaded_key_count": len(state_dict),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
    }


class Persist4DAllT(ReScene):
    """Read detached history once after the first complete decoder pass."""

    def __init__(
        self,
        *args: object,
        memory_read_enabled: bool = False,
        local_enhancement: nn.Module | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)
        if type(memory_read_enabled) is not bool:
            raise Persist4DModelError("memory_read_enabled must be a boolean")
        if local_enhancement is not None and not isinstance(local_enhancement, nn.Module):
            raise Persist4DModelError("local_enhancement must be a module or None")
        if memory_read_enabled and (
            self.mask_dim != PersistentMemoryRead.hidden_dim
            or self.num_queries != PersistentMemoryRead.num_queries
        ):
            raise Persist4DModelError("memory read requires Q=100 and D=128")
        self.memory_read_enabled = memory_read_enabled
        self.memory_read = PersistentMemoryRead() if memory_read_enabled else None
        self.local_enhancement = local_enhancement
        self._memory_read_state: DetachedMemoryReadState | None = None
        self._memory_read_call_count = 0
        self._local_enhancement_call_count = 0

    def after_decoder_stage(
        self,
        queries: Tensor,
        *,
        execution_stage_idx: int,
        shared_parameter_idx: int,
        decoder_features: Tensor | None = None,
        decoder_padding_mask: Tensor | None = None,
        point2segment: Sequence[Tensor] | None = None,
    ) -> Tensor:
        queries = super().after_decoder_stage(
            queries,
            execution_stage_idx=execution_stage_idx,
            shared_parameter_idx=shared_parameter_idx,
            decoder_features=decoder_features,
            decoder_padding_mask=decoder_padding_mask,
            point2segment=point2segment,
        )
        if execution_stage_idx != len(self.hlevels) - 1:
            return queries
        if self.local_enhancement is not None:
            if decoder_features is None or decoder_padding_mask is None:
                raise Persist4DModelError("local enhancement context is unavailable")
            class_logits, _, mask_logits = self.mask_module(
                queries, decoder_features
            )
            queries = self.local_enhancement(
                queries,
                class_logits=class_logits,
                mask_logits=mask_logits,
                padding_mask=decoder_padding_mask,
                point2segment=point2segment,
            )
            self._local_enhancement_call_count += 1
        if not self.memory_read_enabled:
            return queries
        if self.memory_read is None:
            raise Persist4DModelError("enabled memory read module is missing")
        state = self._memory_read_state
        if state is None:
            state = DetachedMemoryReadState.empty(
                queries.shape[0], device=queries.device, dtype=queries.dtype
            )
        queries = self.memory_read(queries, state)
        self._memory_read_call_count += 1
        return queries

    def forward(
        self,
        x: object,
        point2segment: object = None,
        raw_coordinates: object = None,
        is_eval: bool = False,
        *,
        memory_read_state: DetachedMemoryReadState | None = None,
    ) -> dict[str, object]:
        if memory_read_state is not None and not self.memory_read_enabled:
            raise Persist4DModelError("memory state was provided while read is disabled")
        if memory_read_state is not None and not isinstance(
            memory_read_state, DetachedMemoryReadState
        ):
            raise Persist4DModelError("memory state type differs")
        self._memory_read_state = memory_read_state
        self._memory_read_call_count = 0
        self._local_enhancement_call_count = 0
        try:
            output = super().forward(
                x,
                point2segment=point2segment,
                raw_coordinates=raw_coordinates,
                is_eval=is_eval,
            )
        finally:
            self._memory_read_state = None
        if self.memory_read_enabled:
            if self._memory_read_call_count != 1 or self.memory_read is None:
                raise Persist4DModelError("memory read did not execute exactly once")
            output = dict(output)
            output["memory_read_diagnostics"] = dict(
                self.memory_read.last_diagnostics
            )
        if self.local_enhancement is not None:
            if self._local_enhancement_call_count != 1:
                raise Persist4DModelError(
                    "local enhancement did not execute exactly once"
                )
            output = dict(output)
            diagnostics = getattr(self.local_enhancement, "last_diagnostics", {})
            output["local_enhancement_diagnostics"] = dict(diagnostics)
        return output


__all__ = [
    "Persist4DAllT",
    "Persist4DModelError",
    "strict_load_r1_with_named_adapters",
]
