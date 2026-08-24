# LivingScenes Related-Work And Baseline Audit

## Source Binding

The official repository is pinned at commit
`f290146d3a70ee8d7648ec3d841a58352331c27a` and tree
`ac707f3ddadde2b3b307069387aba3bef906f152`. The released
`LivingScenes_latest.pt` weight is 88,831,661 bytes with SHA256
`48b86e3cda90a066287aa6ae2acf3b0c92416afc6288b1da6d25f0a370013231`.
Exact source hashes are recorded in `external/livingscenes_source_manifest.json`.

## Paper Position

LivingScenes studies sparse reference/rescan object relocalization and
reconstruction using learned object-level shape codes. Its released 3RScan
evaluator encodes a fixed reference and each rescan, then applies the
`sequential` object matcher. This is a relevant long-term sparse object-matching
precedent, but it is not a causal multi-stage persistent entity-state update.

The default 3RScan configuration sets `use_gt_mask: True` and samples 1,024
points per object. Its 23 accepted 3RScan category aliases map to seven ShapeNet
priors: chair, table, bench, sofa, pillow, bed, and trash bin. The evaluator
also reads official scene/object transforms to categorize moving and static
objects and to evaluate registration. These annotations are valid in its native
protocol but are not inputs available to the frozen Persist4D comparison.

## Quantitative Decision

Decision: `NOT_RUN`.

The optional quantitative gate is not satisfied. Although official weights are
present, the default protocol uses GT object masks; matching predicted Mask3D
inputs are not provisioned; the supported category subset is restricted; and
the reference/rescan matching metric is not the common-prefix persistent-state
metric. Rewriting these interfaces would create a new method rather than a fair
reproduction. LivingScenes remains Related Work only and is not integrated into
Persist4D.
