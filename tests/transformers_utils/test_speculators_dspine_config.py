# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.transformers_utils.configs.speculators.base import SpeculatorsConfig

pytestmark = pytest.mark.skip_global_cleanup


def _dspine_config() -> dict:
    return {
        "speculators_model_type": "dspine",
        "draft_vocab_size": 32,
        "mask_token_id": 31,
        "block_size": 4,
        "aux_hidden_state_layer_ids": [2, 5],
        "transfer_dim": 8,
        "message_dim": 6,
        "transition_top_k": 3,
        "transfer_space_eps": 1e-5,
        "transformer_layer_config": {
            "model_type": "qwen3",
            "hidden_size": 16,
        },
    }


def test_dspine_updater_builds_causal_dflash_contract() -> None:
    config = SpeculatorsConfig.extract_transformers_pre_trained_config(
        _dspine_config()
    )

    assert config["architectures"] == ["DSpineDraftModel"]
    assert config["eagle_aux_hidden_state_layer_ids"] == [2, 5]
    assert config["dflash_config"] == {
        "mask_token_id": 31,
        "target_layer_ids": [1, 4],
        "sample_from_anchor": False,
        "causal": True,
        "known_anchor_injection": False,
        "block_size": 4,
        "transfer_dim": 8,
        "message_dim": 6,
        "transition_top_k": 3,
        "transfer_space_eps": 1e-5,
    }


@pytest.mark.parametrize("enabled", [False, True])
def test_dspine_updater_preserves_explicit_anchor_mode(enabled) -> None:
    source = _dspine_config()
    source["known_anchor_injection"] = enabled
    config = SpeculatorsConfig.extract_transformers_pre_trained_config(source)
    assert config["dflash_config"]["known_anchor_injection"] is enabled


def test_dspine_speculative_config_reuses_dflash_runtime() -> None:
    config = _dspine_config()
    config["speculators_config"] = {
        "proposal_methods": [{"speculative_tokens": 3}],
        "verifier": {"name_or_path": "Qwen/Qwen3-4B"},
    }

    result = SpeculatorsConfig.extract_vllm_speculative_config(config)

    assert result == {"method": "dflash", "num_speculative_tokens": 3}
