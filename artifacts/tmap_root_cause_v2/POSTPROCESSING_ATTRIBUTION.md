# Post-processing Attribution

- Status: `pass`
- Population: `43 masters x 3 orders`
- Horizons: `T2-T5`
- Primary P1 score reducer: `mean`

| Path | Horizon | Candidates / sequence | Current-stage candidates / sequence | Current AP | Causal t-mAP | t-mREC |
|---|---:|---:|---:|---:|---:|---:|
| F0 | T2 | 100.000 | 59.264 | 0.370851 | 0.190996 | 0.341656 |
| F0 | T3 | 100.000 | 55.612 | 0.366906 | 0.107897 | 0.247695 |
| F0 | T4 | 100.000 | 53.884 | 0.360500 | 0.068999 | 0.188359 |
| F0 | T5 | 100.000 | 49.519 | 0.363728 | 0.045340 | 0.136554 |
| L0 | T2 | 200.000 | 59.698 | 0.371692 | 0.068767 | 0.207362 |
| L0 | T3 | 300.000 | 60.992 | 0.384097 | 0.029016 | 0.130200 |
| L0 | T4 | 400.000 | 62.070 | 0.365923 | 0.009834 | 0.072727 |
| L0 | T5 | 500.000 | 61.791 | 0.382886 | 0.004091 | 0.051682 |
| L1 | T2 | 21.132 | 10.713 | 0.298467 | 0.037439 | 0.164214 |
| L1 | T3 | 31.791 | 10.659 | 0.298783 | 0.013969 | 0.099326 |
| L1 | T4 | 43.349 | 11.550 | 0.293197 | 0.004860 | 0.053637 |
| L1 | T5 | 54.450 | 11.101 | 0.303576 | 0.002539 | 0.035610 |
| P0 | T2 | 13.612 | 10.713 | 0.289467 | 0.156942 | 0.272481 |
| P0 | T3 | 15.070 | 10.659 | 0.299984 | 0.094784 | 0.191018 |
| P0 | T4 | 16.403 | 11.550 | 0.296944 | 0.058875 | 0.132821 |
| P0 | T5 | 17.116 | 11.101 | 0.305455 | 0.044663 | 0.104453 |
| P1 | T2 | 190.899 | 59.698 | 0.361744 | 0.207241 | 0.348681 |
| P1 | T3 | 280.039 | 60.992 | 0.369052 | 0.123102 | 0.252957 |
| P1 | T4 | 367.829 | 62.070 | 0.358928 | 0.070232 | 0.175815 |
| P1 | T5 | 455.186 | 61.791 | 0.371405 | 0.052503 | 0.138426 |

## Interpretation Boundary

L0/L1 and P0/P1 are controlled post-processing contrasts on the same
fresh local raw/sidecar cache. F0 uses the frozen FullHistory official
cache. Pairwise differences are diagnostics and are not assumed to add
up to the complete FullHistory-versus-Persist4D gap.
