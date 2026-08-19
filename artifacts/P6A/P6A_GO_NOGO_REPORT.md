# Persist4D P6-A GO / NO-GO Report

## What was changed

Implemented the P6-A scientific evidence package with a frozen root payload.

## Why it was changed

To isolate cross-stage association and state maintenance from frozen local perception.

## Experimental protocol

Protocol B uses exactly 43 masters, 6 reference-scene clusters, 3 orders, and 645 cache entries at T=2/3/4/5.

## Reproducibility binding

P6-A source commit: `f8cb3c957bd0cdf8adc57ceadd99f2cf7219291a`; P5 source commit: `92bab01e93bacbc939606ec7c7f58d3f9b334fe6`; P5 artifact commit: `1380c4b9f37bec7933126ccc9bd70067de166f6f`; P5 checkpoint SHA256: `85ed1aba60320cd19798536b71b91dbc156b7ea60f838832bc0bbbdba131546e`.

## Main results

- G6A-1: PASS - Passed the preregistered threshold; see statistical_analysis.md.
- G6A-2: PASS - Passed the preregistered threshold; see statistical_analysis.md.
- G6A-3: PASS - Passed the preregistered threshold; see statistical_analysis.md.
- G6A-4: PASS - Passed the preregistered threshold; see statistical_analysis.md.
- G6A-5: PASS - Passed the preregistered threshold; see statistical_analysis.md.

## Statistical evidence

See `statistical_analysis.md`.

## Failure analysis

Association: `association_events.csv`; error: `error_breakdown.csv`; reactivation: `reactivation_audit.csv`.

## What claims are supported

- Exact common-prefix Protocol B evaluation completed.
- B4 reduces long-horizon identity switches against B3.
- B4 improves dormant-track reactivation against B3.
- All methods use exactly the same frozen local predictions.
- B4 preserves short-horizon quality and adds long-horizon utility.
- The preregistered failure taxonomy explains at least 90 percent.

## What claims are NOT supported

- Metadata order is not claimed to be real chronology.
- Native arbitrary-order change-label evidence is unavailable.
- P6-A does not claim to repair or reproduce an external benchmark score.

## GO / NO-GO decision

Decision: P6A_GO

## Exact next action

Exact next action: Stop after P6-A and await explicit continuation authorization.
