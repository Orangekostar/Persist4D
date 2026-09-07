# ReScene Task-Learning Root-Cause and Strong Local Final Report

Principal outcome: `TLRC-GREEN`

## Controlled Root-Cause Result

R1 changes only the optimized loss objective from the local weighted objective
to the public released-code raw sum. It passed every registered epoch-90 gate
against R0 and was therefore the only reproduction-compatible candidate resumed
to 450 epochs. The completed full result is `ROOTCAUSE-CONFIRMED`.

The selected R1 checkpoint is completed epoch `390`, optimizer step `25740`,
SHA256 `629ff7624dcac15e6022906e808e2e05b3ec61c60a1116ab0e278f0cfd2368dd`.
Its three-seed official-like means are t-mAP `0.3106326957543691`, overall mAP
`0.3963398337364197`, and SpatialStageMean `0.4399757186571757`; the paired
mean spatial delta from the frozen Concerto reimplementation is
`0.01450987160205841`.

## ReScene-Strong Result

A1 enables the existing ReScene-native feature-seeded FPS query switch. It
passed the short epoch-90 gate and was resumed to 450 epochs. The selected
completed-epoch-435 checkpoint has SHA256
`0bf2476a6fc53768f97aad7e14a4ac8779661f302725325993d5b37442a7864f`.
Its three-seed means are t-mAP `0.29871795574824017`, overall mAP
`0.3740434944629669`, and SpatialStageMean `0.43659768005212146`. The paired
mean spatial gain is `0.01113183299700419`, but seed 45 is negative by
`-0.004146277904510498`; the verdict is `STRONG-LOCAL-PARTIAL`.

A2 was not run because A1 passed its short gate and the registered STOP rule
closed architecture expansion. No query-competition or attention-relaxation
module was implemented.

## Interpretation

The controlled full-run evidence confirms that the public-code objective's
early advantage survives the registered 450-epoch budget: every registered
mean gate passes and all three paired spatial deltas are positive.
Feature-seeded FPS queries improve the mean local predictor but remain
seed-sensitive. These results are local-perception evidence only: no Persist4D,
Protocol-B, gap-recovery, identity-switch, latency, or memory result was used
for selection.

The locally trained checkpoints are not official ReScene4D checkpoints, and
the study does not establish reproduction of the paper-reported 34.8% t-mAP.
