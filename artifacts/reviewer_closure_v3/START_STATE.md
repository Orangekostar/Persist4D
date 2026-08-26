# Reviewer Closure V3 Start State

Audit time: `2026-08-25T13:34:12Z`

| Field | Verified value |
|---|---|
| Repository root | `repo:.` |
| Branch | `research/persist4d-reviewer-closure-v3` |
| HEAD | `c2f1bcacff1ec244909426b57403965f679f08cc` |
| Expected start | `c2f1bcacff1ec244909426b57403965f679f08cc` |
| Initial worktree | clean |
| Submodules | none |
| Checkpoint SHA256 | `85ed1aba60320cd19798536b71b91dbc156b7ea60f838832bc0bbbdba131546e` |
| Checkpoint bytes | `754917862` |

## Runtime

Python `3.10.20`, PyTorch `2.6.0+cu126`, CUDA runtime `12.6`, and
`stmetrics 0.1.0` were loaded from the `persist4d` environment. CUDA exposed
three NVIDIA A40 GPUs with 46068 MiB each and driver `595.71.05`.

The required pre-change regression command passed: `59 passed in 11.07s`.

## Frozen Evidence

| Artifact | SHA256 |
|---|---|
| V2 manifest | `48b708f86097003faa2d8c64658e1676a6aaa82033e9e057811d995f9a22f8f4` |
| V2 cache manifest | `f525252cd306f29e7db80788e4d5c773d2a35c8767d5eaa63b189740ea212182` |
| V2 attribution manifest | `7e795aa3ea6e885fdc8e0ef002caf826416c726fa54b62bcc4a41aa2a57b3a15` |
| Protocol-B manifest | `246497165612699b103d0d79d5503025cb2cd14466aad3ab149d4fe82884ecbe` |
| P2 G2 report | `d891fb7fd53306d8ab65db81b9bb85f08664a9689de850ac7836143b238816bc` |
| P2R pilot manifest | `bed2743f42e16574c514778d6b3fa9f6864ca489fe88be13a72cbaafdc4f0e0c` |

The full machine-readable inventory, including all V2 result hashes and exact
third-party revisions, is in `START_STATE.json`. No frozen V1/V2 file was
modified.

## External Source Audit

The live official ReScene4D `main` revision was
`fb2fe42eb8f1e926567c48eea9acb874e608ee10`. The retrieved README SHA256 was
`4550760cce90bc372175cc9638148c6cf6d581058b24c590bf0c88c27a31d070`;
its Checkpoints section said `Coming soon.` This only establishes public asset
availability at retrieval time and does not alter the frozen local checkpoint.

Start-state gate: **PASS**.
