# Upstream / Local Training Difference

Status: `PASS`

| Item | Pinned public source | Local control | Controlled action |
| --- | --- | --- | --- |
| final objective | raw sum of all returned losses | weight-dictionary reducer with per-layer contrastive diagnostics excluded | R1 changes only this reducer |
| accumulation | 1 | 8 | physical-batch audit before any R2 curve |
| EOS | public config 0.1; paper 0.2 | 0.2 | fixed-batch gradient audit before any R5 curve |
| class filter | [0, 1] | [0, 1, 255] | full inventory before any R4 curve |

Numeric objective and EOS evidence is bound by portable config SHA-256 `3a5a80f122787880595ac6044843bd4d5e477c8566703c4feae0db7f4f33df4c`.
