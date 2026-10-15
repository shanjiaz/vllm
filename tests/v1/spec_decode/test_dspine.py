# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.model_executor.models.qwen3_dspine import (
    AdjacentCausalInjection,
    DSpineQwen3ForCausalLM,
    DSpineQwen3Model,
    _score_transitions,
)
from vllm.v1.worker.gpu.spec_decode.dspine.speculator import DSpineSpeculator


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
    hidden = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 2.0], [1.0, 0.0]])

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
                expected[batch_idx, step, predecessor_rank] = (
                    lm_head[candidates[batch_idx, step]] @ refined
                )

    torch.testing.assert_close(actual, expected)


def test_candidates_select_highest_processed_logits() -> None:
    """Native candidate selection preserves IDs, order, and processed scores."""
    logits = torch.tensor([[1.0, -2.0, 7.0, 3.0], [8.0, 2.0, 4.0, -1.0]])
    model = SimpleNamespace(
        model=SimpleNamespace(transition_top_k=2),
        lm_head=nn.Identity(),
        candidate_logits_processor=lambda head, states: logits,
    )
    ids, values = DSpineQwen3ForCausalLM.compute_candidates(model, torch.empty(2, 4))
    torch.testing.assert_close(ids, torch.tensor([[2, 3], [0, 2]]))
    torch.testing.assert_close(values, torch.tensor([[7.0, 3.0], [8.0, 4.0]]))


def test_known_anchor_injection_matches_paper_equations(monkeypatch) -> None:
    """Known anchors use token embeddings; other predecessors stay predictive."""
    from vllm.distributed import parallel_state

    monkeypatch.setattr(
        parallel_state, "_TP", SimpleNamespace(rank_in_group=0, world_size=1)
    )
    torch.manual_seed(19)
    injection = _injection(hidden_size=4, transfer_dim=2, block_size=3)
    with torch.no_grad():
        for parameter in injection.parameters():
            parameter.normal_(std=0.3)
    hidden = torch.randn(6, 4)
    blocks = hidden.reshape(2, 3, 4)
    codebook = torch.nn.functional.normalize(torch.randn(7, 2), dim=-1)
    anchors = torch.tensor([1, 3])
    features = (
        torch.nn.functional.normalize(blocks @ injection.readouts[0].weight.T, dim=-1)
        * 2**0.5
    )
    predecessor = torch.cat(
        (codebook[anchors, None] * 2**0.5, features[:, 1:-1]), dim=1
    )
    gate = torch.sigmoid(
        features[:, 1:] @ injection.receiver_gate.weight.T
        + predecessor @ injection.predecessor_gate.weight.T
        + injection.gate_bias
    )
    update = gate * (predecessor @ injection.message.weight.T)
    update = update @ injection.writes[0].weight.T
    rms = (blocks[:, 1:].square().mean(-1, keepdim=True) + injection.eps).sqrt()
    expected = torch.cat((blocks[:, :1], blocks[:, 1:] + rms * update), dim=1)

    actual = injection(
        0, hidden, anchor_features=injection.token_features(codebook, anchors)
    )

    torch.testing.assert_close(actual, expected.reshape_as(hidden))
    torch.testing.assert_close(actual[::3], hidden[::3])


@pytest.mark.parametrize("known_anchor", [False, True])
def test_backbone_passes_configured_anchor_to_every_layer(known_anchor) -> None:
    calls = []

    class Injection(nn.Module):
        block_size = 3

        def token_features(self, codebook, ids):
            return codebook[ids] * 2**0.5

        def forward(self, layer_idx, hidden, *, anchor_features):
            calls.append((layer_idx, anchor_features))
            return hidden

    codebook = torch.arange(14, dtype=torch.float32).reshape(7, 2)
    model = SimpleNamespace(
        injection=Injection(),
        known_anchor_injection=known_anchor,
        transfer_codebook=codebook,
        layers=[lambda **kw: (kw["hidden_states"], None)] * 2,
        norm=lambda hidden, residual: (hidden, residual),
    )
    ids = torch.tensor([1, 0, 0, 3, 0, 0])
    DSpineQwen3Model.forward(
        model, ids, torch.arange(6), input_embeds=torch.randn(6, 4)
    )
    assert [idx for idx, _ in calls] == [0, 1]
    for _, features in calls:
        if known_anchor:
            torch.testing.assert_close(
                features, codebook[torch.tensor([1, 3])] * 2**0.5
            )
        else:
            assert features is None


def test_speculator_routes_block_anchors_and_transition_scores() -> None:
    """The draft walk must score each block from its own known anchor."""
    hidden = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    pre_injection = hidden + 100
    candidate_ids = torch.arange(12).reshape(6, 2)
    transition_scores = torch.arange(24).reshape(2, 3, 2, 2).float()
    observed = {}

    def compute_candidates(states):
        torch.testing.assert_close(states, hidden[[1, 2, 3, 5, 6, 7]])
        return candidate_ids, None

    def compute_transition_scores(ids, states, anchors):
        torch.testing.assert_close(ids, candidate_ids.reshape(2, 3, 2))
        torch.testing.assert_close(
            states, pre_injection[[1, 2, 3, 5, 6, 7]].reshape(2, 3, 4)
        )
        torch.testing.assert_close(anchors, torch.tensor([11, 22]))
        return transition_scores

    speculator = SimpleNamespace(
        _run_model=lambda *args: (hidden, pre_injection),
        sample_indices=torch.tensor([1, 2, 3, 5, 6, 7]),
        num_speculative_steps=3,
        top_k=2,
        num_query_per_req=4,
        input_buffers=SimpleNamespace(
            input_ids=torch.tensor([11, 0, 0, 0, 22, 0, 0, 0])
        ),
        model=SimpleNamespace(
            compute_candidates=compute_candidates,
            compute_transition_scores=compute_transition_scores,
        ),
        candidate_sampler=SimpleNamespace(
            sample=lambda *args: observed.update(args=args)
        ),
        sample_pos=torch.arange(6),
        sample_idx_mapping=torch.arange(6),
        temperature=torch.tensor([0.0, 1.0]),
        seeds=torch.tensor([42, 43]),
        draft_tokens=torch.empty(2, 3, dtype=torch.long),
        draft_logits=torch.empty(2, 3, 32),
        use_fp64_gumbel=False,
    )

    DSpineSpeculator._generate_draft(speculator, 2, 8, None, None, None)

    torch.testing.assert_close(observed["args"][0], candidate_ids.reshape(2, 3, 2))
    assert observed["args"][1] is transition_scores
    assert observed["args"][2] == 2
    assert observed["args"][5] is speculator.temperature
    assert observed["args"][6] is speculator.seeds
    assert observed["args"][8] is speculator.draft_logits
