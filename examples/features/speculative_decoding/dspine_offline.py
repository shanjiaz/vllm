# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run a DSpine draft checkpoint with its matching target model.

Example (the draft checkpoint must have block_size=16):

    .venv/bin/python examples/features/speculative_decoding/dspine_offline.py \
        --target Qwen/Qwen3.5-4B --draft /path/to/dspine-checkpoint \
        --num-speculative-tokens 15 --prompt "What is speculative decoding?"
"""

import argparse

from vllm import LLM, SamplingParams
from vllm.v1.metrics.reader import Counter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--num-speculative-tokens", type=int, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument(
        "--thinking", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()

    llm = LLM(
        model=args.target,
        tensor_parallel_size=1,
        enforce_eager=True,
        max_model_len=args.max_model_len,
        disable_log_stats=False,
        speculative_config={
            "method": "dflash",
            "model": args.draft,
            "num_speculative_tokens": args.num_speculative_tokens,
        },
    )
    sampling = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)
    result = llm.chat(
        [{"role": "user", "content": args.prompt}],
        sampling_params=sampling,
        chat_template_kwargs={"enable_thinking": args.thinking},
    )[0]
    print(result.outputs[0].text)

    counters = {
        metric.name: metric.value
        for metric in llm.get_metrics()
        if isinstance(metric, Counter)
    }
    rounds = counters.get("vllm:spec_decode_num_drafts", 0)
    accepted = counters.get("vllm:spec_decode_num_accepted_tokens", 0)
    if rounds <= 0:
        raise RuntimeError("No speculative draft rounds were recorded")
    print(f"Draft rounds: {rounds}")
    print(f"Accepted draft tokens: {accepted}")
    print(f"Acceptance EAL: {1 + accepted / rounds:.3f}")


if __name__ == "__main__":
    main()
