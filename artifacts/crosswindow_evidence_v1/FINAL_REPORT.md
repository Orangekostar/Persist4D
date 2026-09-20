# Persist4D CrossWindow V1 final report

## Outcome

- Execution: complete for E0, E1, E2, E3, E4, E6 decision, lock, and cached Protocol-B confirmation.
- Scientific joint goal: **not confirmed** (`JOINT_GOAL_PASS=false`).
- Quality gate: `UNCONFIRMED` because native FH-R1 payloads were unavailable.
- Resource gate: `UNCONFIRMED` because comparable native real-forward profiling was unavailable.
- Frozen system: `D0` + `M-new` + `SYSTEM_SELECTED_SCORE`.

## Development evidence

- E1 decision: `MASK_OR_OTHER_DOMINANT`; GT-LAG1 long gain `-0.162826` and candidate-complete failure fraction `0.082707`.
- No 12-config search was authorized. A1/A2 defaults were evaluated on DEV-SEL only.
- Best exploratory connector: `A1-default` with status `FAIL_SELECTION_GATE`; no connector was promoted.
- E4 selected `M-new` with interpretation `NO_REVISION_GAIN` under its fixed parent.
- E6: `SKIPPED_CONDITION`; no learned head was created and R1 remained frozen.

## Protocol-B evidence

- Common cached cohort: 129 logical units across 6 references.
- Frozen candidate t-mAP: T2 `0.249229`, T3 `0.155346`, T4 `0.090114`, T5 `0.064331`.
- FH-R1-native rows are null with `BLOCKED_ASSET`; cached replay timing is not presented as network profiling.
- Capacity K=16/32/100 is reported for D0 and the exploratory connector in `final/resources.csv`.

## Boundaries

Development and Protocol-B caches share the frozen R1 producer and are historically exposed; this is not evidence on new independent scenes. GT-assisted rows are diagnostic only and never compete as deployable methods. The negative association result does not prove that all possible association mechanisms lack headroom.
