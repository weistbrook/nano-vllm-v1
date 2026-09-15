"""Decode-only benchmark for nano-vllm.

The problem with serving_bench / bench is that most of the wall time is prefill
and queueing, so a decode-side optimization (e.g. Triton RMSNorm/Silu) is
invisible. This script submits `--batch-size` equal-length requests at once,
gives them a tiny prompt and a long output, then measures the steady-state cost
of one decode step (one token per sequence).

Run it twice, with and without --use-triton, to compare decode cost:

  python decode_bench.py --model ~/models/Qwen3-4B --batch-size 64 \
      --input-len 1 --output-len 256
  python decode_bench.py --model ~/models/Qwen3-4B --batch-size 64 \
      --input-len 1 --output-len 256 --use-triton

  # also compare with CUDA graphs disabled (isolates launch overhead)
  ... --enforce-eager

Reported metrics:
  step_ms      median wall time of one decode step (== TPOT, since each step
               emits exactly one token per running sequence)
  decode_tok/s batch_size / step_ms
"""

import argparse
import os
import statistics
import time
from random import randint, seed

import torch

from nanovllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser(description="Decode-only benchmark for nano-vllm.")
    parser.add_argument("--model", type=str, default=os.path.expanduser("~/models/Qwen3-4B"))
    parser.add_argument("--batch-size", type=int, default=64, help="Concurrent sequences (= decode batch).")
    parser.add_argument("--input-len", type=int, default=1, help="Prompt tokens per request; keep tiny.")
    parser.add_argument("--output-len", type=int, default=512, help="Tokens to generate per request.")
    parser.add_argument("--max-num-seqs", type=int, default=None, help="Default: batch-size.")
    parser.add_argument("--max-num-batched-tokens", type=int, default=None,
                        help="Default: large enough for the initial prefill plus a full decode step.")
    parser.add_argument("--chunked-prefill", action="store_true", default=False)
    parser.add_argument("--enforce-eager", action="store_true", default=False)
    parser.add_argument("--use-triton", action="store_true", default=False)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--warmup", type=int, default=10, help="Decode steps to discard before measuring.")
    parser.add_argument("--repeat", type=int, default=1, help="Run the whole measurement this many times.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    seed(args.seed)
    max_num_seqs = args.max_num_seqs or args.batch_size
    prefill_tokens = args.batch_size * max(args.input_len, 1)
    max_num_batched_tokens = args.max_num_batched_tokens or max(prefill_tokens, args.batch_size * 2, 1024)
    max_model_len = args.input_len + args.output_len + 8

    print(f"model={args.model} batch={args.batch_size} input_len={args.input_len} "
          f"output_len={args.output_len}")
    print(f"max_num_seqs={max_num_seqs} max_num_batched_tokens={max_num_batched_tokens} "
          f"max_model_len={max_model_len} chunked_prefill={args.chunked_prefill} "
          f"enforce_eager={args.enforce_eager} use_triton={args.use_triton}")

    llm = LLM(
        args.model,
        enforce_eager=args.enforce_eager,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
        chunked_prefill=args.chunked_prefill,
        gpu_memory_utilization=args.gpu_memory_utilization,
        use_triton=args.use_triton,
    )

    prompts = [[randint(0, 10000) for _ in range(args.input_len)] for _ in range(args.batch_size)]
    sampling = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=args.output_len)
                for _ in range(args.batch_size)]

    for run in range(args.repeat):
        for prompt, sp in zip(prompts, sampling):
            llm.add_request(prompt, sp)

        prefill_step_ms = None
        step_ms = []
        running = []
        finished = 0
        total_time_start = time.perf_counter()
        while not llm.is_finished():
            n_running = len(llm.scheduler.running)
            if n_running == 0 and llm.scheduler.waiting:
                n_running = min(len(llm.scheduler.waiting), max_num_seqs)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            outputs, _ = llm.step()
            torch.cuda.synchronize()
            dt_ms = (time.perf_counter() - t0) * 1000.0
            if prefill_step_ms is None:
                prefill_step_ms = dt_ms
            else:
                step_ms.append(dt_ms)
                running.append(n_running)
            finished += len(outputs)
        total_time = time.perf_counter() - total_time_start

        steady = [(c, t) for c, t in zip(running, step_ms) if c == args.batch_size]
        steady_times = [t for _, t in steady[args.warmup:]]
        if not steady_times:
            steady_times = step_ms[args.warmup:] or step_ms
        med = statistics.median(steady_times)
        mean = statistics.fmean(steady_times)
        p90 = sorted(steady_times)[min(len(steady_times) - 1, int(0.9 * len(steady_times)))]
        decode_tps = args.batch_size / (med / 1000.0)

        print(f"\n--- run {run + 1}/{args.repeat} ---")
        print(f"prefill/first step : {prefill_step_ms:9.2f} ms")
        print(f"decode steps timed : {len(steady_times)} (batch stayed at {args.batch_size})")
        print(f"step_ms            : median {med:8.2f}  mean {mean:8.2f}  p90 {p90:8.2f}")
        print(f"TPOT               : {med:8.2f} ms/token/seq")
        print(f"decode throughput  : {decode_tps:9.1f} tok/s")
        print(f"wall clock         : {total_time:8.2f} s ({finished} seqs x {args.output_len} tok)")


if __name__ == "__main__":
    main()
