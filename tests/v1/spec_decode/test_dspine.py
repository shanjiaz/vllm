# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch
from torch import nn

from vllm.model_executor.models.qwen3_dspine import (
    AdjacentCausalInjection,
    _score_transitions,
)


def _injection(hidden_size: int, transfer_dim: int, block_size: int):
    return AdjacentCausalInjection(
        hidden_size=hidden_size,
        transfer_dim=transfer_dim,
        message_dim=transfer_dim,
        num_layers=1,
        block_size=block_size,
        params_dtype=torch.float32,
        quant_config=None,
        eps=1e-6,
        prefix="injection",
    )


def test_adjacent_injection_preserves_each_block_anchor(monkeypatch) -> None:
    from vllm.distributed import parallel_state

    monkeypatch.setattr(
        parallel_state, "_TP", SimpleNamespace(rank_in_group=0, world_size=1)
    )
    injection = _injection(hidden_size=2, transfer_dim=2, block_size=2)
    with torch.no_grad():
        injection.readouts[0].weight.copy_(torch.eye(2))
        injection.message.weight.copy_(torch.eye(2))
        injection.receiver_gate.weight.zero_()
        injection.predecessor_gate.weight.zero_()
        injection.gate_bias.fill_(20)
        injection.writes[0].weight.copy_(torch.eye(2))
    hidden = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.0, 2.0], [1.0, 0.0]]
    )

    output = injection(0, hidden)

    torch.testing.assert_close(output[::2], hidden[::2])
    assert not torch.equal(output[1], hidden[1])
    assert not torch.equal(output[3], hidden[3])


def test_transition_cache_matches_serial_last_write(monkeypatch) -> None:
    from vllm.distributed import parallel_state

    monkeypatch.setattr(
        parallel_state, "_TP", SimpleNamespace(rank_in_group=0, world_size=1)
    )
    torch.manual_seed(0)
    batch, steps, top_k, hidden_size, vocab = 2, 3, 2, 4, 11
    injection = _injection(hidden_size, transfer_dim=3, block_size=steps + 1)
    codebook = torch.nn.functional.normalize(torch.randn(vocab, 3), dim=-1)
    lm_head = torch.randn(vocab, hidden_size)
    candidates = torch.randint(vocab, (batch, steps, top_k))
    anchors = torch.randint(vocab, (batch,))
    receivers = torch.randn(batch, steps, hidden_size)

    actual = _score_transitions(
        injection,
        nn.Identity(),
        lm_head,
        codebook,
        candidates,
        receivers,
        anchors,
        0,
    )

    expected = torch.empty_like(actual)
    receiver_features = injection.project(0, receivers)
    for batch_idx in range(batch):
        for step in range(steps):
            predecessor_ids = (
                anchors[batch_idx].repeat(top_k)
                if step == 0
                else candidates[batch_idx, step - 1]
            )
            for predecessor_rank, predecessor_id in enumerate(predecessor_ids):
                predecessor = injection.token_features(codebook, predecessor_id)
                refined = injection.write(
                    0,
                    receivers[batch_idx, step],
                    receiver_features[batch_idx, step],
                    predecessor,
                )
                expected[batch_idx, step, predecessor_rank] = lm_head[
                    candidates[batch_idx, step]
                ] @ refined

    torch.testing.assert_close(actual, expected)
