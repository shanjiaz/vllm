# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch
from torch import nn

from vllm.model_executor.models.interfaces import supports_eagle3
from vllm.models.glm5next.common.model import (
    Glm5NextForCausalLM,
    Glm5NextForConditionalGeneration,
    Glm5NextModel,
)


def test_glm5next_declares_eagle3_support():
    assert supports_eagle3(Glm5NextForCausalLM)
    assert supports_eagle3(Glm5NextForConditionalGeneration)


def test_glm5next_accepts_explicit_aux_hidden_state_layers():
    target = object.__new__(Glm5NextForCausalLM)
    nn.Module.__init__(target)
    target.model = object.__new__(Glm5NextModel)
    nn.Module.__init__(target.model)

    target.set_aux_hidden_state_layers((44, 4, 12, 4))

    assert target.model.aux_hidden_state_layers == (4, 12, 44)


def test_capture_aux_hidden_stream_returns_materialized_stream():
    hidden_states = torch.randn(4, 8)
    layer = SimpleNamespace(mhc=False, is_mtp_layer=False)

    captured = Glm5NextModel._capture_aux_hidden_stream(
        layer, hidden_states, torch.randn_like(hidden_states), None, None
    )

    assert captured is hidden_states


def test_capture_aux_hidden_stream_materializes_deferred_mhc_post():
    hidden_states = torch.randn(4, 2, 8)
    residual = torch.randn(4, 2, 8)
    post = torch.randn(4, 2, 8)
    comb = torch.randn(4, 2, 8)
    expanded = hidden_states + residual + post + comb
    layer = SimpleNamespace(
        mhc=True,
        is_mtp_layer=False,
        n=2,
        hc_post=lambda *_: expanded,
    )

    captured = Glm5NextModel._capture_aux_hidden_stream(
        layer, hidden_states, residual, post, comb
    )

    expected = expanded.mean(dim=1)
    torch.testing.assert_close(captured, expected)
