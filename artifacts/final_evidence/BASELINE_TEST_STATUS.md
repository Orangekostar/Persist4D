# Baseline Test Status

## Source

- Branch: `research/persist4d-final-evidence`
- Baseline commit: `3323cba186479b7dd4c005bebd468415b7d07a3b`
- Baseline tree: `473f356b0d42810e63b5bd55d9081c7e166bf90d`
- Reviewer-closure artifact tree: `e521be7b038b9bfb84d8461380cc9cd228f70568`
- Initial worktree status: clean

## Environment Diagnosis

The unqualified `pytest` command is not a valid project runner on this host. It
resolves to `/home/ww/.local/bin/pytest`, whose shebang uses system Python 3.12
and a broken user-level PyTorch installation. The project runner is:

```text
/home/ww/miniconda3/envs/persist4d/bin/python -m pytest
```

That environment has Python 3.10.20, PyTorch 2.6.0+cu126, pytest 8.4.2, and
PyTorch Lightning 2.6.5.

## Baseline Result

Command:

```text
/home/ww/miniconda3/envs/persist4d/bin/python -m pytest -q -m 'not gpu'
```

Result: 1651 passed, 8 skipped, 2 deselected, and 111 failed in 825.77 seconds.

The failures are pre-existing worktree prerequisites rather than changes in this
branch. They fall into three observed classes:

1. ignored data, checkpoints, prediction entries, or sidecars are absent from a
   newly created Git worktree;
2. historical protocol tests bind older exact commits and source trees;
3. official metric tests depend on an environment-specific `stmetrics` build.

This task does not change frozen historical contracts to make those tests pass.
Completion requires all new final-evidence tests and every directly affected
regression with its declared external inputs to pass. The final report must keep
this baseline deviation visible.
