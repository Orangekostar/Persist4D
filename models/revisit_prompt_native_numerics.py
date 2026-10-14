"""Evaluation-only Native sparse kernels with the original KRSC weights."""
from contextlib import contextmanager
from pathlib import Path


def pin_native_convolutions(system):
    from spconv import constants
    from spconv.core import ConvAlgo
    from spconv.pytorch.conv import SparseConvolution

    if not constants.ALL_WEIGHT_IS_KRSC:
        raise ValueError("Native evaluation requires unchanged KRSC weight layout")
    if constants.SPCONV_ALLOW_TF32:
        raise ValueError("Native evaluation requires TF32 disabled")
    modules = [(name, module) for name, module in system.named_modules()
               if isinstance(module, SparseConvolution)]
    if not modules:
        raise ValueError("Expected frozen sparse convolution modules")
    for _, module in modules:
        module.algo = ConvAlgo.Native
    return [name for name, _ in modules]


def runtime_identity():
    import spconv
    from spconv import constants, core_cc
    from spconv.pytorch import conv
    import torch
    from scripts.revisit_prompt_assets import sha256

    return dict(spconv_version=spconv.__version__, torch_version=torch.__version__,
                torch_cuda=torch.version.cuda, all_weight_is_krsc=constants.ALL_WEIGHT_IS_KRSC,
                spconv_allow_tf32=constants.SPCONV_ALLOW_TF32,
                spconv_sources={name:sha256(Path(module.__file__)) for name,module in
                                (("conv.py",conv),("constants.py",constants),("core_cc",core_cc))})


@contextmanager
def evaluation_builders():
    """Pin after exact frozen loading, including P5's locally imported builders."""
    from scripts import revisit_prompt_assets as assets
    from scripts import evaluate_revisit_prompt as evaluation

    original_build, original_load = assets.build_system, assets.load_revisit_bundle
    original_evaluation_build = evaluation.build_system

    def build(*args, **kwargs):
        system, config, audit = original_build(*args, **kwargs)
        pin_native_convolutions(system)
        return system, config, audit

    def load(*args, **kwargs):
        system = original_load(*args, **kwargs)
        pin_native_convolutions(system)
        return system

    assets.build_system, assets.load_revisit_bundle = build, load
    evaluation.build_system = build
    try:
        yield
    finally:
        assets.build_system, assets.load_revisit_bundle = original_build, original_load
        evaluation.build_system = original_evaluation_build
