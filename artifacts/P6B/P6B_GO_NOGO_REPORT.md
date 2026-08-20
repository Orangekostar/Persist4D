# Persist4D P6-B GO / NO-GO Report

## 1. What was changed

Added threshold-aware assignment and quality-gated persistent memory without changing frozen local predictions.

## 2. Why it was changed

P6-A isolated association, reactivation, and birth quality as the actionable method bottlenecks.

## 3. Experimental protocol

Candidates used four tuning reference clusters; the selected config was frozen before one evaluation on two held-out clusters.

## 4. Reproducibility binding

Source `0036186d65aff6143f0f59713956ece767539c38`; selected config `ab4b4cfae20de56894a82b2bd19c41c895fa080c3b88a749b31ff689d4dea038`; split `80157a4f25d222d7a07757acbaa70e9a68b5d2e546ee4903bf755fda928689d6`.

## 5. Main results

Held-out P6B T5 t-mAP=0.017831, t-REC=0.050393, ID switches=125.

## 6. Statistical evidence

Paired per-sequence rows and deterministic eligibility/ranking evidence are included in the bundle.

## 7. Failure analysis

Failure categories are reported separately and remain bounded to frozen local predictions plus P6-B association decisions.

## 8. What claims are supported

- P6-B held-out evidence is reported without an improvement claim.

## 9. What claims are NOT supported

- P6-B does not establish SOTA, retraining gains, P7, or P8 claims.

## 10. GO / NO-GO decision

All five preregistered gates determine the terminal decision below.

## 11. Exact next action

Stop after P6-B and analyze failed held-out gates before any P7 work.

P6B_STOP