# Persist4D Novelty Boundary

## Supported Scientific Statement

Perception horizon and identity horizon need not grow together. Persist4D
realizes this principle for sparse 4D scene understanding with bounded local
perception and a bounded causal entity state that preserves long-horizon
identity.

The evidence-supported claim remains:

> Persist4D provides a useful long-horizon accuracy-identity-compute Pareto
> operating point by decoupling bounded local perception from persistent entity
> identity.

## Unsafe Priority Claims

Do not claim `first persistent scene model`, `first long-term object memory`,
`first sparse-revisit object association`, or `first bounded memory`. The
historical and modern literature already contains persistent temporal scene
models, sparse object matching, and bounded state mechanisms.

## Boundary Against ReScan

ReScan demonstrates an inductive persistent temporal scene model for sparse
rescans. Persist4D instead asks how a learned sparse-4D perception system can
keep local perception bounded while maintaining causal entity identity across
successive deployment updates. The distinction is learned 4D semantic-instance
perception plus explicit horizon/compute scaling, not the invention of
persistent scene memory.

## Boundary Against ReScene4D

ReScene4D performs joint spatiotemporal reasoning over a finite observation
set. Persist4D maintains entity identity across successive bounded-window
updates without repeatedly expanding joint temporal context. Temporal context
and persistent entity state are therefore separate system contracts.

## Boundary Against LivingScenes

LivingScenes targets sparse object matching, relocalization, and reconstruction
with object shape representations. Persist4D targets causal persistent identity
inside learned 4D semantic-instance perception, with bounded historical state
and measured deployment scaling. This is a task and system distinction, not a
claim that either approach subsumes the other.

## Evidence Limits

The internal 3RScan evidence supports persistent identity, near-parity
long-horizon task quality, and better measured scaling against the evaluated
full-history alternatives. ReScan transfer is inconclusive and does not support
an external generalization claim. Persist4D does not claim improved local masks,
classes, or memory-conditioned perception.
