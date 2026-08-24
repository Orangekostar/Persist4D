from __future__ import annotations

from types import SimpleNamespace

import torch

from trainer.trainer import InstanceSegmentation


class _SparseExecution(torch.nn.Module):
    __module__ = "spconv.pytorch.conv"

    def forward(self, value):
        return value


class _ConcertoTree(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = torch.nn.Module()
        self.backbone.model = torch.nn.Module()
        self.backbone.model.embedding = torch.nn.Sequential(
            torch.nn.Linear(2, 2), torch.nn.Dropout(0.5)
        )
        self.backbone.model.enc = torch.nn.Sequential(
            torch.nn.Linear(2, 2), torch.nn.Dropout(0.5), _SparseExecution()
        )
        self.backbone.model.dec = torch.nn.Sequential(
            torch.nn.Linear(2, 2), torch.nn.Dropout(0.5)
        )
        self.class_embed_head = torch.nn.Linear(2, 2)


def _system(*, enabled: bool) -> InstanceSegmentation:
    system = InstanceSegmentation.__new__(InstanceSegmentation)
    torch.nn.Module.__init__(system)
    system.config = SimpleNamespace(
        general=SimpleNamespace(
            freeze="backbone_encoder",
            frozen_encoder_eval=enabled,
        )
    )
    system.model = _ConcertoTree()
    system._freeze_backbone_parameters()
    return system


def test_frozen_encoder_eval_flag_preserves_decoder_train_mode() -> None:
    system = _system(enabled=True)

    system.eval()
    system.train()

    assert system.training is True
    assert system.model.backbone.model.embedding.training is False
    assert system.model.backbone.model.enc.training is False
    assert system.model.backbone.model.enc[-1].training is True
    assert system.model.backbone.model.dec.training is True
    assert system.model.class_embed_head.training is True
    assert all(
        not parameter.requires_grad
        for parameter in system.model.backbone.model.embedding.parameters()
    )
    assert all(
        not parameter.requires_grad
        for parameter in system.model.backbone.model.enc.parameters()
    )
    assert all(
        parameter.requires_grad
        for parameter in system.model.backbone.model.dec.parameters()
    )
    assert all(
        parameter.requires_grad
        for parameter in system.model.class_embed_head.parameters()
    )


def test_frozen_encoder_eval_flag_false_preserves_legacy_mode() -> None:
    system = _system(enabled=False)

    system.eval()
    system.train()

    assert system.model.backbone.model.embedding.training is True
    assert system.model.backbone.model.enc.training is True
    assert system.model.backbone.model.dec.training is True
