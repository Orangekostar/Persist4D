# Reviewer-Closure Visual QA Ledger

| Issue | Artifact | Page/section | Severity | Fix | Owner | Status |
| --- | --- | --- | --- | --- | --- | --- |
| Compact coverage labels collided in the composite | `performance_decomposition.*` | panel (b) | Medium | Use the associable fraction in the compact panel; retain full category stacks in the standalone figure | primary agent | resolved |
| T2 protocol note crossed the y axis | `strong_baseline_identity_scaling.*` | identity baseline | Medium | Move the note inside the plotting area and state that only Full-History/B2 initialize at T2 | primary agent | resolved |
| Method distinction could disappear in grayscale | all method comparisons | all panels | High | Preserve stable method colors and add marker, line-style, or hatch redundancy | primary agent | resolved |
| Unknown failures could be mistaken for an explained category | `failure_decomposition.*` | panel (c) | High | Keep an explicit black cross-hatched unknown/unresolved segment with direct percentage labels | primary agent | resolved |
| P6-A bars could be mistaken for an oracle upper bound | `oracle_association_gain.*`, `performance_decomposition.*` | panel (d) | High | Label the plot GT-ID-only and state that unmatched candidates remain | primary agent | resolved |
| Vector/raster export integrity | all 18 published files | all panels | Medium | Parse SVG, inspect 300 dpi PNG, verify single-page PDF and embedded TrueType fonts, compare published hashes to inspected renders | primary agent | passed |
