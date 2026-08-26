# Sonata Second-Perception Smoke

- Gate: `SSMOKE-PASS`
- Source tree SHA-256: `1d007d818eaf68cc137db6be2763219692599629985cc4d96bf11d7eb278d0e2`
- Physical/effective batch: 8 / 32
- Temporal-overlay calls: 20
- Temporal-mask calls: 48
- Contrastive loss: disabled; no contrastive objective term observed.
- Frozen encoder/embedding gradients: absent.
- Sonata decoder and ReScene task gradients: finite and nonzero.
- Query feature interface: `[1, 100, 128]`.
- Tiny optimization initial/minimum objective: 174.707535 / 160.620743

This preflight-only optimization is a runtime sanity check, not model-selection evidence.
