# Official ReScan Code Audit

## Result

`RESCAN_METHOD_NOT_REPRODUCED`

This is an optional historical-method reproduction result. It does not affect
the independent dataset evaluation gate.

## Frozen Source

- Repository: <https://github.com/mhalber/Rescan>
- Commit: `f45283be31119e9bd955d40bc159b1774dfed092`
- Tree: `8825baa0b87135986160e1f820a8ed80d2d39930`
- License: MIT
- External checkout: `external:rescan/official-code`

## Bounded Build Attempt

The audit used CMake 3.28.3 and GCC 13.3.0 outside the Persist4D environment.
No official source file was modified.

| Target | Result | Evidence |
|---|---|---|
| `seg2rsdb` | built | binary SHA256 `36ff0bd58b26edd5a2b9f8ce4cb525f8bb1553c92e5a6c131d81a02ccb7c4b16` |
| `pose_proposal` | built | binary SHA256 `b24d27ab7a66dcdb908b81d987d8dc715a958725e9a30a65a79ebef3c7952784` |
| `create_eval_files` | built | binary SHA256 `b91d93ecb6264af7d9eeb390b99a3a83d37d94f398b775b105ffd0175f4c9636` |
| `segment_transfer` | not configured | missing official-required `lib/gco/GCOptimization.cpp` and `LinkedBlockList.cpp` |
| top-level build | not configured | unconditional `rsdb_viewer` requires unavailable OpenGL development libraries |

The repository README explicitly says GCO v3.0 cannot be distributed freely
and must be obtained separately. The full pipeline also requires external
PoissonRecon and SurfaceTrimmer executables. Since `segment_transfer` is the
core temporal update executable, the official method cannot run from the
released repository and official dataset without additional third-party assets.

## Compatibility Boundary

The native method consumes ground-truth segmented PLY input for its first
capture and transfers/updates an object database thereafter. Its native
instance-transfer and ScanNet-style semantic-instance scripts are useful
protocol references, but their values are not directly comparable with frozen
ReScene/Persist4D outputs until input modality, label taxonomy, and metric units
are matched. No native ReScan result is placed in a controlled-result table.
