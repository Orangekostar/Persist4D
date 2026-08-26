# Sonata Weight Provenance

- Gate: `SW0-PASS`
- Repository: `facebook/sonata`
- Immutable revision: `df99897472c09f91ba9288da0a034aacffc0b010`
- Filename: `sonata.pth`
- SHA-256: `c5ced5acdae30d1c469713398073a866e25e6e414e23feed5dc025373657ac50`
- Bytes: 434008287
- License: `CC-BY-NC-4.0`
- Acquired: `2026-08-26T08:08:44Z`
- Local reference: `external:sonata_weight/c5ced5acdae30d1c469713398073a866e25e6e414e23feed5dc025373657ac50`
- Sonata code revision: `18c09ff8d713494f78a8213792262b910977a65d`

## Load-Key Audit

- Checkpoint keys: 453
- Model keys: 701
- Loaded keys: 453
- Loaded encoder/embedding keys: 453
- Expected train-from-scratch decoder missing keys: 248
- Unexpected keys: 0
- Resolved `enc_mode`: `False`

The local file is an immutable regular-file snapshot. The official
weight is encoder-only; only `dec.*` parameters are initialized from
scratch. No critical `embedding.*` or `enc.*` parameter is missing.
