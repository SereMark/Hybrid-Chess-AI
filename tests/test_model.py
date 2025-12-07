from __future__ import annotations

import pytest

pytest.importorskip("chesscore", reason="hiányzik a chesscore kiterjesztés")

import torch
from network import INPUT_PLANES, POLICY_OUTPUT, ChessNet, ResidualBlock


def test_residual_block_preserves_shape_and_gradients() -> None:
    block = ResidualBlock(32)
    block.train()

    x = torch.randn(3, 32, 8, 8, requires_grad=True)
    y = block(x)

    assert y.shape == x.shape
    loss = y.sum()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_chessnet_forward_emits_policy_and_value_outputs() -> None:
    net = ChessNet(num_blocks=1, channels=64)

    x = torch.randn(2, INPUT_PLANES, 8, 8)
    policy_logits, value = net(x)

    assert policy_logits.shape == (2, POLICY_OUTPUT)
    assert value.shape == (2,)
    assert torch.isfinite(policy_logits).all()
    assert torch.isfinite(value).all()

    loss = policy_logits.sum() + value.sum()
    loss.backward()
    assert all(p.grad is not None for p in net.parameters() if p.requires_grad)
