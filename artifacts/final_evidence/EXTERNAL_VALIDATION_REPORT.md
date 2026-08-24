# Independent ReScan External Validation

## Decision

Classification: `EXTERNAL_INCONCLUSIVE`.

The preregistered gate fails all three minimum-evidence thresholds: 8 natural
visible-absent-visible gaps instead of 10, 2 gap-bearing physical scenes instead
of 3, and 0.055738 Persist4D Level-B scene-macro observation coverage instead
of 0.10. The classification is fixed before method effects are interpreted.

## Evaluated Population

- 13 independent physical scenes and 45 dataset-ordered captures.
- 365 stable scene-scoped identities; 333 Level-B object identities after
  excluding wall, floor, and ceiling classes.
- 8 natural gap opportunities, all in `scene_a` and `scene_b`.
- 22 official ambiguity files evaluated with the accepted-alternative identity
  semantics.
- One class-inconsistent identity (`scene_e`, identity 27) excluded from Level A
  and retained in class-agnostic Level B.

All four methods consume the same frozen local-observation cache. Model inputs
contain only XYZ, RGB, normals, and geometry-only voxel segments. Class labels,
instance IDs, and ambiguity alternatives are post-inference evaluator fields.

## Domain Shift And Task Transfer

The frozen local model transfers poorly. Scene-macro raw local AP/AP50/AP25 are
0.01627/0.02521/0.03834 and raw local recall is 0.02372. Online class-aware
t-mAP and t-REC are both 0.000204. These values are identical across trackers
because every method receives the same local predictions.

Level B contains 1,032 eligible identity observations, but Persist4D matches
only 42 observations and 14 identity transitions. It records 3 ID switches, 3
fragmentations, and 3 merges. No method makes a valid recovery attempt, so gap
recovery accuracy and recall are zero and cannot test the primary mechanism.

## Descriptive Tracker Effects

Against Pairwise Feature-Class Association, Persist4D's scene-macro normalized
ID-switch effect is -0.02564 (lower is better), with paired scene-bootstrap 95%
CI [-0.07692, 0]. Fragmentation changes by -0.07692, CI [-0.23077, 0], while
merges change by +0.07692, CI [0, 0.23077]. These boundary-touching intervals,
the worse merge direction, low coverage, and absent recovery attempts do not
support an external advantage claim.

## Claim Boundary

This experiment does not contradict the internal persistent-state result, but
it also does not support generalization beyond the frozen 3RScan protocol. The
paper may report the audited transfer attempt and domain-shift limitation. It
must not present a ReScan main result table, external Figure 4, or full-system
generality claim from this evidence.

Machine-readable sources: `external/rescan_raw.json`,
`external/rescan_results.csv`, `external/rescan_per_scene.csv`,
`external/rescan_scene_bootstrap.csv`, and `external_gate.json`.
