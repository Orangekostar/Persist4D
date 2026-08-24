# Final-Evidence Capacity Visual QA Ledger

| Issue | Artifact | Page/section | Severity | Fix | Owner | Status |
| --- | --- | --- | --- | --- | --- | --- |
| C2 footer notes overlapped | `figure_c2_performance_vs_capacity.*` | bottom margin | Medium | Removed duplicate footer and placed the T2 gap note inside panel (d) | primary agent | resolved |
| Color could be lost in grayscale | all figures | all panels | High | Added marker, line-style, fill, and direct-label redundancy | primary agent | resolved |
| Capacity lines could imply observed saturation | `figure_c1_occupancy_vs_horizon.*` | occupancy panel | High | Kept full y range, direct K labels, and explicit observed maximum annotation | primary agent | resolved |
| Unlike memory quantities could be compared | `figure_c3_state_bytes_vs_capacity.*` | storage panel | High | Plotted tensor state only and stated exclusions in panel and caption | primary agent | resolved |
| Vector/raster export integrity | all nine files | all panels | Medium | Parsed SVG, checked nonblank 300 dpi PNG, verified single-page PDF and embedded TrueType fonts | primary agent | passed |
| Text clipping or incoherent overlap | all nine files | all panels | High | Inspected final rendered PNGs at original resolution | primary agent | passed |
