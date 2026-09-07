# Persist4D R1 Downstream Validation V1

This experiment replaces only the frozen ReScene task checkpoint with the
qualified R1 checkpoint. It does not train or fine-tune a model and does not
change Protocol-B, local candidate extraction, B2, B4, or official metric
semantics.

The evaluation covers 43 masters, three registered orders, five causal state
updates, and six `reference_scene_id` clusters. FullHistory, B2, and B4 are
reported at T2, T4, and T5. B2/B4 task quality uses the same official candidates
with mean, latest, and max trajectory score reducers. Direct local AP remains a
separate diagnostic.

The primary recovery result is `RECOVERY_SUPPORTED` only when pooled B4 minus B2
is strictly positive at both T4 and T5 and at least four of six valid clusters
are positive at each horizon. The rule, checkpoint, thresholds, and reducer set
cannot be changed after observing results.

Statistics use paired `reference_scene_id` bootstrap with seed 45 and 10,000
replicates. Profiling uses the first canonical master in each cluster at T2, T4,
and T5, with five warmups and ten measured repeats on one A40. Timing includes
model forward plus CPU tracking and excludes data loading, collation, H2D, and
metric scoring.

Large tensor caches are external. Git contains only contracts, compact
manifests, CSV summaries, reports, and provenance. A negative scientific result
does not make execution incomplete.
