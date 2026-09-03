#!/usr/bin/env python
"""Measure the loss implementations on the Qwen 3.5 DroPE training shape."""

import argparse
import json
import time
import sys
from pathlib import Path

import torch

# Make ``python scripts/benchmark_memory_efficient_ce.py ...`` work as well
# as module invocation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from custom_models.drope import DroPEQwen3_5Config, DroPEQwen3_5ForCausalLM
from custom_models.memory_efficient_ce import apply_memory_efficient_ce


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("implementation", choices=("baseline", "cce", "liger"))
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--sequence-length", type=int, default=32768)
    args = parser.parse_args()

    config = DroPEQwen3_5Config.from_pretrained(
        "Qwen/Qwen3.5-0.8B-Base", attention_type="nope"
    )
    model = DroPEQwen3_5ForCausalLM.from_pretrained(
        "Qwen/Qwen3.5-0.8B-Base", config=config, dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).cuda()
    model.gradient_checkpointing_enable({"use_reentrant": False})
    model.train()
    apply_memory_efficient_ce(model, args.implementation)

    vocab_size = config.text_config.vocab_size
    input_ids = torch.randint(vocab_size, (1, args.sequence_length), device="cuda")
    labels = input_ids.clone()

    # Compile/autotune kernels outside the timed section.
    model(input_ids=input_ids, labels=labels).loss.backward()
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    elapsed = []
    for _ in range(args.steps):
        start = time.perf_counter()
        model(input_ids=input_ids, labels=labels).loss.backward()
        torch.cuda.synchronize()
        elapsed.append(time.perf_counter() - start)
        model.zero_grad(set_to_none=True)
    print(json.dumps({
        "implementation": args.implementation,
        "steps": args.steps,
        "mean_step_s": sum(elapsed) / len(elapsed),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
