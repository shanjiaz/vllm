# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import torch
import torch.nn.functional as F
from torch import nn

from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

from .qwen3_dflash import DFlashQwen3ForCausalLM, DFlashQwen3Model
from .utils import maybe_prefix


class AdjacentCausalInjection(nn.Module):
    """Inject each draft position's predicted feature into its successor."""

    def __init__(
        self,
        *,
        hidden_size: int,
        transfer_dim: int,
        message_dim: int,
        num_layers: int,
        block_size: int,
        params_dtype: torch.dtype,
        quant_config: QuantizationConfig | None,
        eps: float,
        prefix: str,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.transfer_dim = transfer_dim
        self.num_layers = num_layers
        self.block_size = block_size
        self.eps = eps
        self.readouts = nn.ModuleList(
            ReplicatedLinear(
                hidden_size,
                transfer_dim,
                bias=False,
                params_dtype=params_dtype,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, f"readouts.{layer_idx}"),
                return_bias=False,
            )
            for layer_idx in range(num_layers)
        )
        self.writes = nn.ModuleList(
            ReplicatedLinear(
                message_dim,
                hidden_size,
                bias=False,
                params_dtype=params_dtype,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, f"writes.{layer_idx}"),
                return_bias=False,
            )
            for layer_idx in range(num_layers)
        )
        shared_args = dict(
            bias=False,
            params_dtype=params_dtype,
            quant_config=quant_config,
            return_bias=False,
        )
        self.message = ReplicatedLinear(
            transfer_dim,
            message_dim,
            prefix=maybe_prefix(prefix, "message"),
            **shared_args,
        )
        self.receiver_gate = ReplicatedLinear(
            transfer_dim,
            message_dim,
            prefix=maybe_prefix(prefix, "receiver_gate"),
            **shared_args,
        )
        self.predecessor_gate = ReplicatedLinear(
            transfer_dim,
            message_dim,
            prefix=maybe_prefix(prefix, "predecessor_gate"),
            **shared_args,
        )
        self.gate_bias = nn.Parameter(
            torch.empty(message_dim, dtype=params_dtype), requires_grad=False
        )

    def project(self, layer_idx: int, hidden_states: torch.Tensor) -> torch.Tensor:
        projected = self.readouts[layer_idx](hidden_states)
        projected = F.normalize(projected.float(), dim=-1, eps=self.eps)
        return projected.to(hidden_states.dtype) * math.sqrt(self.transfer_dim)

    def token_features(
        self, codebook: torch.Tensor, token_ids: torch.Tensor
    ) -> torch.Tensor:
        return codebook[token_ids.long()] * math.sqrt(self.transfer_dim)

    def write(
        self,
        layer_idx: int,
        receiver_states: torch.Tensor,
        receiver_features: torch.Tensor,
        predecessor_features: torch.Tensor,
    ) -> torch.Tensor:
        gate = torch.sigmoid(
            self.receiver_gate(receiver_features)
            + self.predecessor_gate(predecessor_features)
            + self.gate_bias
        )
        message = self.message(predecessor_features)
        update = self.writes[layer_idx](gate * message)
        rms = receiver_states.float().square().mean(dim=-1, keepdim=True)
        rms = rms.add(self.eps).sqrt().to(receiver_states.dtype)
        return receiver_states + rms * update.to(receiver_states.dtype)

    def forward(
        self, layer_idx: int, pre_injection_states: torch.Tensor
    ) -> torch.Tensor:
        blocks = pre_injection_states.reshape(
            -1, self.block_size, self.hidden_size
        )
        features = self.project(layer_idx, blocks)
        successors = self.write(
            layer_idx,
            blocks[:, 1:],
            features[:, 1:],
            features[:, :-1],
        )
        return torch.cat((blocks[:, :1], successors), dim=1).reshape_as(
            pre_injection_states
        )


def _score_transitions(
    injection: AdjacentCausalInjection,
    norm: nn.Module,
    lm_head_weight: torch.Tensor,
    transfer_codebook: torch.Tensor,
    candidate_ids: torch.Tensor,
    final_pre_injection: torch.Tensor,
    anchor_token_ids: torch.Tensor,
    final_layer_idx: int,
) -> torch.Tensor:
    top_k = candidate_ids.shape[-1]
    predecessor_ids = torch.cat(
        (
            anchor_token_ids[:, None, None].expand(-1, 1, top_k),
            candidate_ids[:, :-1],
        ),
        dim=1,
    )
    predecessor_features = injection.token_features(
        transfer_codebook, predecessor_ids
    )
    receiver_features = injection.project(final_layer_idx, final_pre_injection)
    receiver_states = final_pre_injection[:, :, None, :].expand(
        -1, -1, top_k, -1
    )
    refined = injection.write(
        final_layer_idx,
        receiver_states,
        receiver_features[:, :, None, :].expand(-1, -1, top_k, -1),
        predecessor_features,
    )
    normalized = norm(refined)
    successor_weights = lm_head_weight[candidate_ids.long()]
    return torch.einsum("bnkh,bnjh->bnkj", normalized, successor_weights)


class DSpineQwen3Model(DFlashQwen3Model):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        start_layer_id: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            start_layer_id=start_layer_id,
            prefix=prefix,
        )
        draft_config = self.config.dflash_config
        speculative_config = vllm_config.speculative_config
        assert speculative_config is not None
        block_size = 1 + speculative_config.num_speculative_tokens
        configured_block_size = int(draft_config.get("block_size", block_size))
        if configured_block_size != block_size:
            raise ValueError(
                "DSpine block_size must equal 1 + num_speculative_tokens: "
                f"{configured_block_size} != {block_size}."
            )
        transfer_dim = int(draft_config["transfer_dim"])
        if transfer_dim > self.config.hidden_size:
            raise ValueError("DSpine transfer_dim cannot exceed hidden_size.")
        self.transition_top_k = int(draft_config["transition_top_k"])
        self.injection = AdjacentCausalInjection(
            hidden_size=self.config.hidden_size,
            transfer_dim=transfer_dim,
            message_dim=int(draft_config["message_dim"]),
            num_layers=len(self.layers),
            block_size=block_size,
            params_dtype=vllm_config.model_config.dtype,
            quant_config=self.quant_config,
            eps=float(draft_config.get("transfer_space_eps", 1e-6)),
            prefix=maybe_prefix(prefix, "injection"),
        )
        self.transfer_codebook = nn.Parameter(
            torch.empty(
                self.config.vocab_size,
                transfer_dim,
                dtype=vllm_config.model_config.dtype,
            ),
            requires_grad=False,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        input_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if input_embeds is None:
            input_embeds = self.embed_input_ids(input_ids)
        hidden_states = input_embeds
        residual = None
        final_pre_injection = hidden_states
        for layer_idx, layer in enumerate(self.layers):
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )
            pre_injection = (
                hidden_states if residual is None else hidden_states + residual
            )
            if layer_idx == len(self.layers) - 1:
                final_pre_injection = pre_injection
            post_injection = self.injection(layer_idx, pre_injection)
            hidden_states = hidden_states + (post_injection - pre_injection)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states, final_pre_injection


class DSpineQwen3ForCausalLM(DFlashQwen3ForCausalLM):
    model_cls = DSpineQwen3Model

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        if get_tensor_model_parallel_world_size() != 1:
            raise NotImplementedError("DSpine currently supports tensor parallelism 1.")
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        target_vocab_size = vllm_config.model_config.get_vocab_size()
        if self.config.draft_vocab_size != target_vocab_size:
            raise ValueError("DSpine requires the full target vocabulary.")
        self.candidate_logits_processor = LogitsProcessor(
            self.config.draft_vocab_size,
            scale=float(getattr(self.config, "logit_scale", 1.0)),
        )

    def forward(  # type: ignore[override]
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(input_ids, positions, inputs_embeds)

    def compute_candidates(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.candidate_logits_processor.get_top_k_tokens(
            self.lm_head, hidden_states, self.model.transition_top_k
        )

    def compute_transition_scores(
        self,
        candidate_ids: torch.Tensor,
        final_pre_injection: torch.Tensor,
        anchor_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        return _score_transitions(
            self.model.injection,
            self.model.norm,
            self.lm_head.weight,
            self.model.transfer_codebook,
            candidate_ids,
            final_pre_injection,
            anchor_token_ids,
            len(self.model.layers) - 1,
        )


EntryClass = DSpineQwen3ForCausalLM
