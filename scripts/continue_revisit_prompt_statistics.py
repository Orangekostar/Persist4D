"""Resume the locked V4 campaign with exact CPU bootstrap curve batching."""

import argparse
from contextlib import contextmanager
from functools import wraps
import json

from scripts.revisit_prompt_cpu_statistics import accelerated_precision_recall


@contextmanager
def accelerated_bootstrap():
    """Keep acceleration inside original bootstrap calls, restoring all bindings."""
    from scripts import evaluate_revisit_prompt as evaluation

    original = evaluation.bootstrap_locked

    @wraps(original)
    def bootstrap(*args, **kwargs):
        with accelerated_precision_recall():
            return original(*args, **kwargs)

    evaluation.bootstrap_locked = bootstrap
    try:
        yield
    finally:
        evaluation.bootstrap_locked = original


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(dict(status='DRY_RUN_NOT_STARTED',
                              controller='scripts.continue_revisit_prompt_sparse',
                              acceleration_scope='CPU_PRECISION_RECALL_INSIDE_ORIGINAL_BOOTSTRAP_ONLY')))
        return 0

    from scripts.continue_revisit_prompt_sparse import main as continue_locked_campaign

    with accelerated_bootstrap():
        return continue_locked_campaign()


if __name__ == '__main__':
    raise SystemExit(main())
