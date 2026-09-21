"""End-to-end serving benchmark for hidden-size RMSNorm kernels.

Compares the normal PyTorch/torch.compile path with a path that uses Triton
only for hidden-size RMSNorm and Add+RMSNorm. Q/K head-dimension RMSNorm and
SiluAndMul remain on their normal path in both runs.

Example:
  python serving_bench_hidden_rms.py --model ~/models/Qwen3-4B \
      --num-requests 256 --request-rate 8 --random-input-len 128 \
      --random-output-len 128
"""

import argparse
import os
import re
import subprocess
import sys
import time
from random import randint, seed

import numpy as np
from tqdm.auto import tqdm

from nanovllm import LLM, SamplingParams


seed(100)
np.random.seed(100)


class RequestMetrics:

    def __init__(self, request_id, input_len):
        self.request_id = request_id
        self.input_len = input_len
        self.submission_time = -1
        self.first_token_time = -1
        self.completion_time = -1
        self.output_len = -1

    def record_first_token(self):
        if self.first_token_time == -1:
            self.first_token_time = time.perf_counter()

    def record_completion(self, output_ids):
        self.completion_time = time.perf_counter()
        self.output_len = len(output_ids)

    @property
    def ttft(self):
        return self.first_token_time - self.submission_time

    @property
    def tpot(self):
        if self.output_len > 1:
            return (self.completion_time - self.first_token_time) / (self.output_len - 1)
        return float("nan")

    @property
    def latency(self):
        return self.completion_time - self.submission_time


def make_requests(args):
    prompts = [
        [randint(0, 10000) for _ in range(randint(1, args.random_input_len))]
        for _ in range(args.num_requests)
    ]
    sampling_params = [
        SamplingParams(
            temperature=0.6,
            ignore_eos=True,
            max_tokens=args.random_output_len,
        )
        for _ in range(args.num_requests)
    ]
    request_intervals = np.random.exponential(1.0 / args.request_rate, args.num_requests)
    return prompts, sampling_params, np.cumsum(request_intervals)


def warm_up(engine, args):
    prompts = [
        [randint(0, 10000) for _ in range(randint(1, args.random_input_len))]
        for _ in range(50)
    ]
    sampling_params = [
        SamplingParams(
            temperature=0.6,
            ignore_eos=True,
            max_tokens=args.random_output_len,
        )
        for _ in range(50)
    ]
    engine.generate(prompts, sampling_params, use_tqdm=False)


def run_case(args, mode):
    prompts, sampling_params, arrival_times = make_requests(args)
    llm = LLM(
        args.model,
        enforce_eager=args.enforce_eager,
        max_model_len=40960,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        tensor_parallel_size=args.tensor_parallel_size,
        chunked_prefill=args.chunked_prefill,
        use_triton=False,
        use_triton_hidden_rmsnorm=(mode == "hidden-triton"),
    )
    warm_up(llm, args)

    metrics = {}
    requests_sent = 0
    start_time = time.perf_counter()
    completed_latencies = []

    with tqdm(total=args.num_requests, desc=mode) as pbar:
        while requests_sent < args.num_requests or not llm.is_finished():
            current_time = time.perf_counter()
            while (
                requests_sent < args.num_requests
                and current_time - start_time >= arrival_times[requests_sent]
            ):
                prompt = prompts[requests_sent]
                llm.add_request(prompt, sampling_params[requests_sent])
                new_seq = llm.scheduler.waiting[-1]
                metrics[new_seq.seq_id] = RequestMetrics(new_seq.seq_id, len(prompt))
                metrics[new_seq.seq_id].submission_time = (
                    start_time + arrival_times[requests_sent]
                )
                requests_sent += 1

            if not llm.is_finished():
                finished_outputs, _ = llm.step()
                for seq in list(llm.scheduler.running):
                    if (
                        seq.seq_id in metrics
                        and seq.num_cached_tokens == seq.num_prompt_tokens
                    ):
                        metrics[seq.seq_id].record_first_token()

                for seq_id, output_ids in finished_outputs:
                    if seq_id not in metrics:
                        continue
                    metrics[seq_id].record_first_token()
                    metrics[seq_id].record_completion(output_ids)
                    completed_latencies.append(metrics[seq_id].latency)
                    pbar.set_postfix({"avg_latency": f"{np.mean(completed_latencies):.2f}s"})
                    pbar.update(1)
            else:
                time.sleep(0.01)

    total_time = time.perf_counter() - start_time
    completed = [m for m in metrics.values() if m.completion_time != -1]
    total_input_tokens = sum(m.input_len for m in completed)
    total_output_tokens = sum(m.output_len for m in completed)
    avg_ttft = np.mean([m.ttft for m in completed])
    avg_tpot = np.mean([m.tpot for m in completed if not np.isnan(m.tpot)])
    avg_latency = np.mean([m.latency for m in completed])
    result = {
        "mode": mode,
        "total_time": total_time,
        "throughput": (total_input_tokens + total_output_tokens) / total_time,
        "avg_ttft": avg_ttft,
        "avg_tpot": avg_tpot,
        "avg_latency": avg_latency,
    }
    print(
        "RESULT "
        + " ".join(f"{key}={value:.9f}" if key != "mode" else f"{key}={value}"
                   for key, value in result.items())
    )


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--num-requests", type=int, default=256)
    parser.add_argument("--request-rate", type=float, default=8)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=2048)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--random-input-len", type=int, default=128)
    parser.add_argument("--random-output-len", type=int, default=128)
    parser.add_argument("--chunked-prefill", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--_worker-mode", choices=["baseline", "hidden-triton"])
    return parser


def run_worker(args):
    seed(100)
    np.random.seed(100)
    run_case(args, args._worker_mode)


def parse_result(output):
    match = re.search(r"^RESULT (.+)$", output, re.MULTILINE)
    if match is None:
        raise RuntimeError("benchmark worker did not emit a RESULT line")
    values = dict(item.split("=", 1) for item in match.group(1).split())
    return {key: value if key == "mode" else float(value) for key, value in values.items()}


def run_subprocess(args, mode):
    command = [
        sys.executable,
        os.path.abspath(__file__),
        "--model", args.model,
        "--num-requests", str(args.num_requests),
        "--request-rate", str(args.request_rate),
        "--max-num-batched-tokens", str(args.max_num_batched_tokens),
        "--max-num-seqs", str(args.max_num_seqs),
        "--tensor-parallel-size", str(args.tensor_parallel_size),
        "--random-input-len", str(args.random_input_len),
        "--random-output-len", str(args.random_output_len),
        "--_worker-mode", mode,
    ]
    if args.chunked_prefill:
        command.append("--chunked-prefill")
    if args.enforce_eager:
        command.append("--enforce-eager")
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    print(completed.stdout, end="")
    if completed.returncode != 0:
        print(completed.stderr, file=sys.stderr, end="")
        raise SystemExit(completed.returncode)
    return parse_result(completed.stdout)


def main():
    args = build_parser().parse_args()
    if args._worker_mode:
        run_worker(args)
        return

    baseline = run_subprocess(args, "baseline")
    hidden_triton = run_subprocess(args, "hidden-triton")
    print("--- End-to-end comparison ---")
    for metric in ("total_time", "throughput", "avg_ttft", "avg_tpot", "avg_latency"):
        baseline_value = baseline[metric]
        triton_value = hidden_triton[metric]
        if metric in {"total_time", "avg_ttft", "avg_tpot", "avg_latency"}:
            improvement = (baseline_value - triton_value) / baseline_value * 100
        else:
            improvement = (triton_value - baseline_value) / baseline_value * 100
        print(
            f"{metric}: baseline={baseline_value:.4f}, "
            f"hidden-triton={triton_value:.4f}, change={improvement:+.2f}%"
        )


if __name__ == "__main__":
    main()