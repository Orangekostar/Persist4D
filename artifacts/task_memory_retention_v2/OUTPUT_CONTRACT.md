# Output Contract

`commit0` is the compulsory control: stage t commits only the current scan and never revises older masks.

`lag1` is the primary policy: the single W=2 forward at stage t replaces exactly the previous scan's provisional output, freezes all older scans, and commits the current scan provisionally. The last scan remains provisional; no nonexistent future scan is used to flush it.

Route identity has priority over class-compatible same-scan IoU matching. The fallback threshold is 0.5 and exact score ties use source query index. One `(logical_track_id, generation, class_id)` candidate is selected per stage and class without GT-guided old/new selection.

The model may read only working state. The lag-one buffer is bounded to the previous scan; older outputs enter an append-only archive whose materialization and bytes are reported separately. Prediction route is read-only and state commit happens exactly once per stage.

FH-native and FH-lag1 are distinct comparator rows. No checkpoint, output policy, or reducer may be selected separately by horizon.
