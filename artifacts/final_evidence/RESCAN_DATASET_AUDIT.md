# ReScan Dataset Audit

## Binding

- Official project: <https://rescan.cs.princeton.edu/>
- Official code: <https://github.com/mhalber/Rescan>
- Pinned code commit: `f45283be31119e9bd955d40bc159b1774dfed092`
- Dataset object: `external:rescan/rescan_dataset.zip`
- Server-reported size: 26,239,218,080 bytes
- Source manifest: `artifacts/final_evidence/external/rescan_source_manifest.json`

The project page reports 45 captures grouped into 13 physical spaces, with
three to five captures per space and stable `instance_idx` values across
captures. Those are source-level expectations, not yet treated as verified
package contents below.

## Source-Level Contract

The official project page and parser declare binary little-endian PLY vertices
with `x`, `y`, `z`, `nx`, `ny`, `nz`, `red`, `green`, `blue`, `radius`,
`class_idx`, and `instance_idx`. The official semantic scripts define ids 1--40
using the NYU40/ScanNet-style names and reserve id 0 for `unlabelled` in the
instance-transfer script.

The official pipeline reads scene names from `scene_list.txt`, discovers PLY
captures under each scene's `gt_segmentation` directory, and sorts capture
basenames lexicographically. A sorted filename list is only deterministic
metadata order; it is not sufficient evidence of real acquisition chronology.

The official instance-transfer evaluator optionally reads a same-basename text
file from `gt_segmentation` and permits listed equivalent instance ids by
choosing the best accepted correspondence. Package-level inventory must verify
whether these files are present and bind their exact syntax before external
identity evaluation.

## Package Verification Status

`IN_PROGRESS`: the official archive is being acquired with resumable transfer.
No package-level claims, eligibility decisions, coordinate assumptions, label
mapping, or chronology claims are frozen until the archive size and SHA256 have
been verified and the extracted contents have been inventoried.
