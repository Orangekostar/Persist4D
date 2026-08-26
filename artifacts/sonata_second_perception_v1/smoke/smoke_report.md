# Sonata Second-Perception Smoke

- Gate: `SSMOKE-PASS`
- Source tree SHA-256: `f8c4c71f4441d565c9935532ace88a1ee54a5fec3821880d4ef877e625a8a0bb`
- Physical/effective batch: 8 / 32
- Temporal-overlay calls: 20
- Temporal-mask calls: 48
- Contrastive loss: disabled; no contrastive objective term observed.
- Frozen encoder/embedding gradients: absent.
- Sonata decoder and ReScene task gradients: finite and nonzero.
- Query feature interface: `[1, 100, 128]`.
- Tiny optimization initial/minimum objective: 174.707535 / 160.625198

This preflight-only optimization is a runtime sanity check, not model-selection evidence.
