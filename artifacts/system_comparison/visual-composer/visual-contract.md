# System Comparison Visual Contract

- Target format: editable full-width paper SVG and CSV source tables.
- Core claim: compare expanding joint history with bounded local perception plus persistent entity state without assuming either system wins.
- Reviewer questions: task quality, deployment identity, gap recovery, latency, VRAM, and accuracy-compute trade-off across T2-T5.
- Evidence layer: main system result and deployment-scaling evidence.
- Source data: `aggregate_results.csv`, `profile_results.csv`, `cluster_bootstrap.csv`, and frozen P6-A B3 rows.
- Statistics: six-reference-scene paired bootstrap is reported in CSV; figures show measured aggregates without invented error bars.
- Figure map: one line chart each for task, identity, gap, latency, and VRAM; one labeled accuracy-latency scatter.
- Table map: Table A compares Full-History, B3 EMA, and B4 Persist4D; Table B separates working VRAM, bounded state bytes, and explicit-history input bytes.
- Palette: Full-History `#0072B2`; Persist4D `#D55E00`; B3 `#666666`; markers and line styles remain distinct in grayscale.
- Captions: state the frozen T2 checkpoint and zero-shot T3-T5 status, units, metric direction, and source CSV.
- Placement: Table A and Figures 1-3 near system-quality results; Table B and Figures 4-6 near deployment scaling.
- Traceability: every SVG contains a title, description, and source filename; every displayed number is derived from a validated CSV row.
