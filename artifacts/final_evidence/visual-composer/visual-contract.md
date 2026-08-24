# Final-Evidence Capacity Visual Contract

Mode: standard

## Figure C1

Artifact: `figures/figure_c1_occupancy_vs_horizon.{svg,pdf,png}`
Target venue / format: paper-width robustness figure; editable SVG/PDF and 300 dpi PNG
Core claim: the observed entity state remains far below every preregistered capacity.
Reviewer question: is frozen K=100 actually saturated?
Evidence layer: mechanism and limitation
Source data: `capacity/capacity_aggregate.csv`
Statistics / uncertainty: median, IQR, and maximum over 129 sequences; six scene clusters
Figure prototype: horizon line with distribution band and registered capacity rules
Panel map: one occupancy panel, T2--T5
Caption role: identify headroom without implying sufficiency beyond T5.
Manuscript placement: bounded-state capacity sensitivity subsection.
Output formats: SVG, PDF, PNG
Traceability: source note in figure; result/figure manifests bind hashes.

## Figure C2

Artifact: `figures/figure_c2_performance_vs_capacity.{svg,pdf,png}`
Target venue / format: full-width two-column four-panel figure
Core claim: task and identity results are invariant over the tested K grid.
Reviewer question: does larger K robustly improve recall or gap recovery?
Evidence layer: robustness
Source data: `capacity/capacity_aggregate.csv`
Statistics / uncertainty: pooled official metrics; scene-paired 10,000-replicate effects are reported in the companion table, not drawn as redundant zero-width bands.
Figure prototype: small-multiple capacity lines
Panel map: (a) t-mAP, (b) t-REC, (c) ID-switch rate, (d) gap-recovery recall
Caption role: name pooling, missing T2 gap opportunities, and frozen K=100.
Manuscript placement: capacity sensitivity results.
Output formats: SVG, PDF, PNG
Traceability: exact K x T coverage is validated before render.

## Figure C3

Artifact: `figures/figure_c3_state_bytes_vs_capacity.{svg,pdf,png}`
Target venue / format: single-column mechanism figure
Core claim: persistent tensor state scales linearly and remains 59.6 KiB at K=100.
Reviewer question: what storage does bounded state actually require?
Evidence layer: mechanism
Source data: `capacity/capacity_aggregate.csv`
Statistics / uncertainty: exact allocated tensor bytes; no sampling uncertainty
Figure prototype: annotated line
Panel map: one K-to-KiB panel
Caption role: prohibit comparison with model weights, VRAM, or Full-History input bytes.
Manuscript placement: capacity resource paragraph.
Output formats: SVG, PDF, PNG
Traceability: exact values are validated as horizon-invariant for each K.

Palette and accessibility: Okabe-Ito colors with marker and line-style redundancy; neutral gray for registered capacity/reference rules; no red/green-only distinction.
No-fabrication status: pass; every plotted value is loaded from the published aggregate CSV.
