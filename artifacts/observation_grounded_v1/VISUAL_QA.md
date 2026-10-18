# Primary visual QA

Reviewed the actual target_evidence.png and resource_probe.png exports and their complete rendering script, CSV aggregates and source identities. Both families use readable labels, zero-based axes, explicit denominators and vector PDF/SVG outputs. No clipping or hidden axis truncation was found.

The provisional worker plot reversed the baseline and GT-assisted anchor data references. Primary review corrected baseline to paired_raw/paired_published and intervention to raw/published, rerendered and inspected the preview. The 67 paired events now show 0.028176→0.047126 raw IoU and 0→0.032166 published IoU. No source data cell was changed.

Probe p90 is calculated over 16 timed updates after four warmups, with 20 total updates per arm. The plot and TIMING_SUMMARY denominator now explicitly say 16. Structural p90 uses 18 actual stages from six T3 episodes per method/role; those source files are verified against RUN_PLAN hashes. Forecast safety/prefix multipliers are not plotted as measured latency.

Target scope is original M4-200, shared M0 history; the anchor is GT-assisted diagnostic. Resource scope is nonformal probes and structural timing, not the seven-configuration same-card P6 profile. The formal quality family and recovery/loss results remain unavailable. No uncertainty, significance, independent generalization or new OGR improvement is implied by the plots. FIXED_CASES.csv uses fixed input/query order and the declared 0.5 success threshold.

FIGURE_MANIFEST.json contains the actual source/aggregate/export hashes. Visual QA covers the final rendered previews; the original formal evidence requirements remain budget-blocked.
