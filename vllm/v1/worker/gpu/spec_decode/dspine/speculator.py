# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import DFlash2Speculator


class DSpineSpeculator(DFlash2Speculator):
    _speculator_name = "DSpine"
    _candidate_top_k_key = "transition_top_k"

    def __init__(self, vllm_config, device: torch.device):
        super().__init__(vllm_config, device)
        if self.enable_adaptive_verification:
            raise ValueError("DSpine does not support adaptive verification.")

    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        model_output = self._run_model(
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
        )
        hidden_states, final_pre_injection = model_output
        num_sample = num_reqs * self.num_speculative_steps
        sample_indices = self.sample_indices[:num_sample]
        hidden_states = hidden_states[sample_indices].view(
            num_reqs, self.num_speculative_steps, -1
        )
        final_pre_injection = final_pre_injection[sample_indices].view(
            num_reqs, self.num_speculative_steps, -1
        )
        candidate_ids, _ = self.model.compute_candidates(hidden_states.flatten(0, 1))
        candidate_ids = candidate_ids.view(
            num_reqs, self.num_speculative_steps, self.top_k
        )
        anchor_token_ids = self.input_buffers.input_ids[
            : num_reqs * self.num_query_per_req : self.num_query_per_req
        ]
        scores = self.model.compute_transition_scores(
            candidate_ids, final_pre_injection, anchor_token_ids
        )
        self.candidate_sampler.sample(
            candidate_ids,
            scores,
            num_reqs,
            self.sample_pos,
            self.sample_idx_mapping,
            self.temperature,
            self.seeds,
            self.draft_tokens,
            self.draft_logits,
            self.use_fp64_gumbel,
        )
