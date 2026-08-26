# Sonata Second-Perception Smoke

- Gate: `SSMOKE-PASS`
- Source tree SHA-256: `9aa27db8e8608c965f0ae7cfb380261ce86ed21120dbb077776a174c603155cc`
- Physical/effective batch: 4 / 32
- Temporal-overlay calls: 20
- Temporal-mask calls: 48
- Contrastive loss: disabled; no contrastive objective term observed.
- Frozen encoder/embedding gradients: absent.
- Sonata decoder and ReScene task gradients: finite and nonzero.
- Query feature interface: `[1, 100, 128]`.
- Tiny optimization initial/minimum objective: 174.707535 / 160.574432

This preflight-only optimization is a runtime sanity check, not model-selection evidence.
