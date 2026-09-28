# ReScene round-one handoff

Experiment commit A: `68dfcd79d0155f7e158d74c24fac1fad35a76c2c`. Branch: `research/rescene-short-module-screen-v1`. Planned immutable tag: `rescene-short-module-screen-v1`.

Scientific results: [FINAL_REPORT.md](FINAL_REPORT.md). All small head checkpoints and selected deployment bundles are in [PUBLIC_HEADS.json](PUBLIC_HEADS.json). No combinations or default model replacement occurred.

This handoff/manifest is commit B. B is intentionally absent from its own content; the external publication receipt records A/B/tag and verifies actual remote references.

Publication availability: CODE_ONLY: Git carries code, complete result tables and small heads. No existing authorized Release credential was available.

Required external metric-evidence asset:

- `official-metric-evidence-68dfcd79d015.zip` — 4714905 bytes, SHA256 `a211ff532671ede3ac01b69ee954f2ba7e49d825c9d56ea418dc5fdc151baa75`; not available from Release; local verified copy exists.

After existing GitHub Release authorization is restored, rerun:

```bash
cd /home/ww/paper5/.worktrees/rescene-short-module-screen-v1
/home/ww/miniconda3/envs/persist4d/bin/python -m scripts.short_module_screen publish --root /home/ww/persist4d_runs/short_module_screen_v1 --resume
```

No token should be pasted into logs or supplied to another application. Parent weights, raw data and GT are not redistributed.
