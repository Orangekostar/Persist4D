# ReScan Dataset Audit

## Binding

- Official project: <https://rescan.cs.princeton.edu/>
- Official code: <https://github.com/mhalber/Rescan>
- Pinned code commit: `f45283be31119e9bd955d40bc159b1774dfed092`
- Dataset object: `external:rescan/rescan_dataset.zip`
- Verified size: 26,239,218,080 bytes
- Archive SHA256: `6985096973085de6c3f9b1463022edbbddc19f5bcb2c6536fb42f8e74ba28f9a`
- Source manifest: `artifacts/final_evidence/external/rescan_source_manifest.json`
- Package manifest: `artifacts/final_evidence/external/rescan_dataset_manifest.json`

The extracted package verifies 45 captures grouped into 13 physical spaces,
with three to five captures per space and scene-scoped `instance_idx` values.

## Source-Level Contract

The official project page and parser declare binary little-endian PLY vertices
with `x`, `y`, `z`, `nx`, `ny`, `nz`, `red`, `green`, `blue`, `radius`,
`class_idx`, and `instance_idx`. The official semantic scripts define ids 1--40
using the NYU40/ScanNet-style names and reserve id 0 for `unlabelled` in the
instance-transfer script.

The official pipeline reads scene names from `scene_list.txt`, discovers PLY
captures under each scene's `gt_segmentation` directory, and sorts capture
basenames. The package contains contiguous numeric suffixes `0..n`; external
evaluation uses that dataset-provided order. No wall-clock acquisition times
are claimed.

The official instance-transfer evaluator optionally reads a same-basename text
file from `gt_segmentation` and permits listed equivalent instance ids by
choosing the best accepted correspondence. Package-level inventory must verify
whether these files are present and bind their exact syntax before external
identity evaluation.

## Verified Package Inventory

- 159 regular files, all content-hashed
- 13 scenes and 45 binary little-endian PLY captures
- 22 official ambiguity files, parsed with the official accepted-ID semantics
- 365 distinct valid scene identities in `[0,255]`
- 333 distinct non-structural object identities after excluding wall, floor,
  and ceiling source classes `1`, `2`, and `22`
- 8 natural visible-absent-visible gap opportunities across 2 scene clusters
- encountered source class IDs: `0,1,2,5,6,7,9,14,15,22,39`

The dataset contains one semantic inconsistency: `scene_e` identity `27` is
annotated with source classes `39` and `7` across captures. It is excluded from
Level A class-aware evaluation and retained in Level B class-agnostic identity
evaluation.

The natural gap count is below the preregistered minimum of 10. This fact is
bound before model results and forces the external gate to remain
`EXTERNAL_INCONCLUSIVE`, while still permitting descriptive full-population
evaluation.

## Evaluation Boundary

Level A includes only exact NYU40-to-frozen-ReScene mappings recorded in
`rescan_to_rescene_label_map.json`. Unsupported classes are excluded rather
than silently remapped. Level B includes valid object identities in `[0,255]`
and excludes structural classes `1`, `2`, and `22`.

The adapter exposes only XYZ, RGB, normals, and geometry-only voxel segments to
the model. `class_idx`, `instance_idx`, and ambiguity alternatives are loaded
only by the post-inference evaluator.
