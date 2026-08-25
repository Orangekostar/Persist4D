# Baseline Evidence Contract

| ID | Method/source | t-mAP (%) | Evidence class | Locally rerun |
|---|---|---:|---|---|
| E0 | ReScene4D-C (paper-reported) | 34.8 | external_reference | false |
| E1 | ReScene4D-C (our reimplementation) | 27.939 | local_best_effort_reimplementation | true |
| E2 | FullHistory using shared frozen local reimplementation | n/a | controlled_internal_baseline | true |

## Official Repository Audit

- URL: `https://github.com/GradientSpaces/rescene4d`
- Revision: `fb2fe42eb8f1e926567c48eea9acb874e608ee10`
- Retrieved: `2026-08-25T13:34:12Z`
- README SHA256: `4550760cce90bc372175cc9638148c6cf6d581058b24c590bf0c88c27a31d070`
- Checkpoint section: `Coming soon.`
- Reported task checkpoint publicly available: `false`

## Allowed Claims

- ReScene4D reports 34.8% t-mAP.
- Our best-effort reimplementation reaches 27.939%.
- All controlled Persist4D-vs-FullHistory comparisons use the same frozen local model.

## Forbidden Claims

- Our ReScene4D reproduces the official 34.8 model.
- Persist4D beats official ReScene4D.
- ReScene4D obtains 27.939%.
- 34.8 -> Protocol-B t-mAP is a direct model degradation.

## Gate B0: PASS

Paper-reported, locally measured, and controlled internal evidence remain separate.
