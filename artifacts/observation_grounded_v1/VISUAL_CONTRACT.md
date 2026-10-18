# Evidence visual contract

Mode: python-plotting; standard evidence preservation. Format: report SVG/PDF with PNG preview, no manuscript layout change.

Family 1, T2–T5 quality: NOT_GENERATED_BUDGET_BLOCKED. Formal AP is unavailable. Do not draw a zero or synthetic quality curve.

Family 2, target evidence: limitation/mechanism diagnostic. Reviewer question: does original M4-200's designated prompt query correspond to the bound current entity, and what happens under the fixed GT-assisted anchor reference? Source: TARGET_FLOW.csv (canonical_entity_event=True and current_target_evaluable=True) and ANCHOR_REFERENCE.csv (status=COMPLETE). Panel A: means of designated raw target IoU, same-class best IoU and source-query/class-bound published target IoU, with the common event denominator. Panel B: paired baseline/intervened means for raw and published IoU on the same fixed anchor interventions. Both axes start at zero, IoU range [0,1]. Labels explicitly state shared M0 history, M4-200 and GT-assisted reference. These are not new OGR formal scores or a theoretical upper bound. No significance or uncertainty claim; report raw event counts.

Family 3, resource evidence: partial resource evidence only; recovery/loss and seven-configuration P6 profile are unavailable without formal endpoints. Panel A: observed p90 of the final sixteen of twenty probe updates, by fixed order R-TARGET, R-ID, R-QFB, R-OBS. Panel B: measured (unscaled) score-free stage p90 for R-OBS and FH-COMMIT0 on CAL/PB. Timing processes used A40 cards; this is not a same-card P6 profile. Do not display forecast multipliers as observed latency. Source: evidence/PROBE_STATUS.json and evidence/TIMING_STATUS.json plus RUN_PLAN measured_costs. Units: seconds/update and seconds/stage; axes start at zero.

Palette: Okabe-Ito #0072B2/#D55E00 with neutral #666666; direct labels and hatches convey paired conditions in grayscale. Use DejaVu Sans, readable 9pt minimum at 10x4-inch export, constrained layout. Embed vector text and no raster-only figures. Save actual aggregation CSVs beside plots and bind source SHA256 in FIGURE_MANIFEST.json.

QA: validate every denominator and finite numeric source, render both PNGs, inspect both actual previews once for clipping, labels, legend, scale, and scope. Parent owns scientific interpretation and final visual review. No selective case search; qualitative case identifiers must follow fixed input/query order if exported.
