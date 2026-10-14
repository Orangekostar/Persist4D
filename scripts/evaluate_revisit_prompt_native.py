"""Dedicated inference/P5 child; no completed C3 training sources are changed."""
import sys

from scripts.revisit_prompt_assets import local_assets
from scripts.revisit_prompt_native_policy import (
    annotated_writer, load_policy, policy_identity, prepare_namespace,
)


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    assets = local_assets()
    policy = load_policy(assets)
    identity = policy_identity(assets, policy)
    scoped = prepare_namespace(assets, policy)
    from models.revisit_prompt_numerics import install_evaluation_policy
    from models.revisit_prompt_native_numerics import evaluation_builders
    install_evaluation_policy()
    if args and args[0] == "p5":
        from scripts import revisit_prompt_campaign as legacy
        writer = annotated_writer(legacy.write_json_atomic, identity, all_outputs=True)
    else:
        from scripts import evaluate_revisit_prompt as legacy
        writer = annotated_writer(legacy.write_json_atomic, identity)
    original_assets, original_writer = legacy.local_assets, legacy.write_json_atomic
    legacy.local_assets, legacy.write_json_atomic = lambda:scoped, writer
    try:
        with evaluation_builders():
            return legacy.main(args)
    finally:
        legacy.local_assets, legacy.write_json_atomic = original_assets, original_writer


if __name__ == "__main__":
    raise SystemExit(main())
