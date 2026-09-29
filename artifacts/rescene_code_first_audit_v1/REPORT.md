# 0929 code-first audit

Fixed-input label repair and independent original-head input-condition verification.
No checkpoint reselection, Q+M combination, or method grid was performed.

## Fixed-input repair (AP percentage points)

| Role | Arm | Original T1 | Updated T1 | ΔT1 | Original T2 | Updated T2 | ΔT2 |
|---|---|---:|---:|---:|---:|---:|---:|
| CAL | Q1 | 29.7860 | 30.7552 | 0.9693 | 24.7055 | 24.8959 | 0.1904 |
| CAL | Q2 | 30.0276 | 30.9943 | 0.9666 | 23.9985 | 23.7285 | -0.2700 |
| CAL | Q3 | 26.2290 | 27.4500 | 1.2210 | 26.2562 | 27.2010 | 0.9449 |
| SEL | Q1 | 59.3538 | 60.9880 | 1.6342 | 56.1815 | 57.2126 | 1.0311 |
| SEL | Q2 | 57.0563 | 57.1474 | 0.0911 | 56.8443 | 57.0942 | 0.2499 |
| SEL | Q3 | 52.7758 | 53.1346 | 0.3588 | 53.2380 | 54.0947 | 0.8567 |
| CAL | M0 | 50.3696 | 50.4377 | 0.0681 | 37.5187 | 37.6818 | 0.1631 |
| CAL | M1 | 51.4627 | 51.2956 | -0.1671 | 38.6608 | 38.4652 | -0.1957 |
| SEL | M0 | 76.4583 | 76.3110 | -0.1473 | 66.4821 | 66.4798 | -0.0023 |
| SEL | M1 | 76.0948 | 76.1680 | 0.0732 | 66.8585 | 66.8516 | -0.0068 |

M0/M1 are a separate expanded-eligibility control; the Q comparison preserved all original M assignments.

## Original locked heads without training augmentation

| Role | Arm | T1 AP (%) | ΔT1 vs own B0 (pp) | T2 AP (%) | ΔT2 vs own B0 (pp) |
|---|---|---:|---:|---:|---:|
| CAL | B0 | 52.1017 | 0.0000 | 36.5049 | 0.0000 |
| CAL | Q1 | 32.7018 | -19.3999 | 21.0791 | -15.4258 |
| CAL | Q2 | 30.5448 | -21.5569 | 20.1992 | -16.3057 |
| CAL | Q3 | 30.9055 | -21.1961 | 22.9480 | -13.5569 |
| CAL | M0 | 53.3394 | 1.2377 | 36.5766 | 0.0717 |
| CAL | M1 | 52.3712 | 0.2695 | 36.4980 | -0.0068 |
| CAL | M2 | 52.0741 | -0.0276 | 36.4561 | -0.0488 |
| SEL | B0 | 78.5034 | 0.0000 | 70.2952 | 0.0000 |
| SEL | Q1 | 60.2975 | -18.2059 | 61.1832 | -9.1120 |
| SEL | Q2 | 60.8374 | -17.6660 | 62.1820 | -8.1132 |
| SEL | Q3 | 60.9372 | -17.5663 | 58.1392 | -12.1560 |
| SEL | M0 | 78.1144 | -0.3890 | 70.2337 | -0.0615 |
| SEL | M1 | 78.7090 | 0.2055 | 70.7267 | 0.4315 |
| SEL | M2 | 78.6978 | 0.1944 | 70.7989 | 0.5037 |

Historical results remain fixed-augmentation development screening. Neither condition is untouched test evidence.
Newly repaired heads were tested at seed45 only; these results do not establish replicated gains.
The score cross-tab demonstrates uplift of affected candidates, not sole causation of historical AP degradation.
Repair effects are the measured matched differences above; no claim of general method invalidity follows.

Measured campaign cost, including interrupted evaluation: 0.522891 GPU-hours; cumulative 80.505200.

Source artifacts: RELABEL_AUDIT.json, ORIGINAL_Q_SCORE_CROSSTAB.json, Q_MATCHED_COMPARISON.json,
M_MATCHED_COMPARISON.json, NOAUG_FIXED_HEADS.json, VERIFIED_RESULTS.json and TEST_NOTES.md.
