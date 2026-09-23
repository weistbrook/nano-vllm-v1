"""Engine-level Poisson serving benchmark with reusable, mixed-length inputs.

Run one arrival rate at a time, for example:
  python serving_bench.py --model ~/models/Qwen3-4B --scheduler-mode both --request-rate 8
"""

import argparse
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from benchmarks.common import (
    TokenTracker, check_engine_structure, create_engine,
    model_vocab_size, read_model_config, reset_idle_cache, runtime_metadata,
    seed_everything, summarize_requests, workload_sha256, write_csv, write_json,
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--num-requests", type=int, default=256)
    parser.add_argument("--request-rate", type=float, default=8.0)
    parser.add_argument("--scheduler-mode", choices=("legacy", "chunked", "both"), default="chunked")
    parser.add_argument("--chunked-prefill", action="store_true", help="Compatibility alias for --scheduler-mode chunked.")
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=2048)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--random-input-len", type=int, default=None,
                        help="Compatibility mode: uniform input lengths from 1 to this value; overrides the mixture.")
    parser.add_argument("--random-output-len", type=int, default=128, help="Fixed output length for every request.")
    parser.add_argument("--short-request-ratio", type=float, default=0.9)
    parser.add_argument("--short-input-min", type=int, default=64)
    parser.add_argument("--short-input-max", type=int, default=256)
    parser.add_argument("--long-input-min", type=int, default=4096)
    parser.add_argument("--long-input-max", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--enforce-eager", action="store_true", default=True,
                        help="Eager mode is the default for controlled scheduler comparisons.")
    parser.add_argument("--cuda-graph", dest="enforce_eager", action="store_false")
    parser.add_argument("--use-triton", action="store_true", help="Optional kernel experiment; off by default.")
    parser.add_argument("--warmup-requests", type=int, default=4)
    parser.add_argument("--workload", type=Path, help="Reload an existing workload JSON, including its arrival times.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/online"))
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    return parser


def validate_args(args):
    for name in ("num_requests", "random_output_len", "max_num_batched_tokens", "max_num_seqs", "tensor_parallel_size"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if not math.isfinite(args.request_rate) or args.request_rate <= 0:
        raise ValueError("--request-rate must be finite and positive")
    if not 0 <= args.short_request_ratio <= 1:
        raise ValueError("--short-request-ratio must lie in [0, 1]")
    for group in ("short", "long"):
        low, high = getattr(args, f"{group}_input_min"), getattr(args, f"{group}_input_max")
        if not 1 <= low <= high:
            raise ValueError(f"invalid {group} input length range: {low}..{high}")
    if args.random_input_len is not None and args.random_input_len < 1:
        raise ValueError("--random-input-len must be positive")
    if args.warmup_requests < 0:
        raise ValueError("--warmup-requests cannot be negative")
    if not math.isfinite(args.temperature) or args.temperature <= 1e-10:
        raise ValueError("The current sampler requires --temperature > 1e-10; token identity is not required")
    if not 0 <= args.seed < 2 ** 32:
        raise ValueError("--seed must lie in [0, 2**32)")
    if args.max_model_len is not None and args.max_model_len < 1:
        raise ValueError("--max-model-len must be positive")
    if not 1 <= args.tensor_parallel_size <= 8:
        raise ValueError("--tensor-parallel-size must lie in [1, 8]")
    if not 0 < args.gpu_memory_utilization <= 1:
        raise ValueError("--gpu-memory-utilization must lie in (0, 1]")
    if args.chunked_prefill and args.scheduler_mode == "legacy":
        raise ValueError("--chunked-prefill conflicts with --scheduler-mode legacy")


def make_workload(args, vocab_size):
    """Generate once, before starting either scheduler; no global RNG dependency."""
    if vocab_size < 1:
        raise ValueError("vocab_size must be positive")
    rng = random.Random(args.seed)
    arrivals = np.cumsum(np.random.RandomState(args.seed).exponential(
        1.0 / args.request_rate, args.num_requests
    ))
    requests = []
    for i, arrival in enumerate(arrivals):
        if args.random_input_len is not None:
            group = "uniform"
            length = rng.randint(1, args.random_input_len)
        else:
            group = "short" if rng.random() < args.short_request_ratio else "long"
            length = rng.randint(getattr(args, f"{group}_input_min"), getattr(args, f"{group}_input_max"))
        requests.append({
            "request_id": str(i), "group": group, "arrival_time": float(arrival),
            "prompt_token_ids": [rng.randrange(vocab_size) for _ in range(length)],
            "max_tokens": args.random_output_len,
        })
    return {
        "kind": "online_poisson", "seed": args.seed, "request_rate": args.request_rate,
        "temperature": args.temperature, "ignore_eos": True, "requests": requests,
    }


def validate_workload(workload, vocab_size, max_model_len):
    if workload.get("kind") != "online_poisson" or not workload.get("requests"):
        raise ValueError("Expected a nonempty online_poisson workload JSON")
    if not workload.get("ignore_eos", False):
        raise ValueError("This benchmark requires ignore_eos=True")
    temperature = workload.get("temperature")
    if not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature <= 1e-10:
        raise ValueError("The current sampler requires positive workload temperature > 1e-10")
    if not isinstance(workload.get("seed"), int) or not 0 <= workload["seed"] < 2 ** 32:
        raise ValueError("Workload seed must be an integer in [0, 2**32)")
    rate = workload.get("request_rate")
    if not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate <= 0:
        raise ValueError("Workload request_rate must be finite and positive")
    previous = -1.0
    identifiers = set()
    for request in workload["requests"]:
        arrival = request["arrival_time"]
        if not math.isfinite(arrival) or arrival < 0 or arrival < previous:
            raise ValueError("Workload arrivals must be finite, nonnegative, and sorted")
        previous = arrival
        request_id = request["request_id"]
        if request_id in identifiers:
            raise ValueError(f"Duplicate request_id: {request_id}")
        identifiers.add(request_id)
        if request.get("group") not in ("short", "long", "uniform"):
            raise ValueError(f"Missing or invalid request group for {request_id}")
        prompt = request["prompt_token_ids"]
        if not prompt or any(not isinstance(token, int) or token < 0 or token >= vocab_size for token in prompt):
            raise ValueError(f"Invalid prompt token IDs for request {request_id}")
        if not isinstance(request["max_tokens"], int) or request["max_tokens"] <= 0:
            raise ValueError(f"Invalid output length for request {request_id}")
        if len(prompt) + request["max_tokens"] > max_model_len:
            raise ValueError(f"Request {request_id} exceeds max_model_len={max_model_len}")


def run_case(args, workload):
    from nanovllm import SamplingParams

    config = vars(args).copy()
    config["model"] = str(Path(args.model).expanduser().resolve())
    engine = create_engine(config, args.scheduler_mode)
    seed_everything(workload["seed"], include_torch=True)
    correctness = check_engine_structure(engine, temperature=workload["temperature"], seed=workload["seed"])
    requests = workload["requests"]
    # Exercise both length classes outside measurement, with identical inputs in both modes.
    sorted_requests = sorted(requests, key=lambda request: len(request["prompt_token_ids"]))
    warmup = [sorted_requests[0 if i % 2 == 0 else -1] for i in range(args.warmup_requests)]
    if warmup:
        engine.generate(
            [request["prompt_token_ids"] for request in warmup],
            [SamplingParams(temperature=workload["temperature"], ignore_eos=True,
                            max_tokens=min(16, request["max_tokens"])) for request in warmup],
            use_tqdm=False,
        )
    reset_idle_cache(engine)
    seed_everything(workload["seed"], include_torch=True)
    tracker = TokenTracker()
    submission_delays = []
    next_request = 0
    step_count = 0
    step_limit = 2 * sum(len(request["prompt_token_ids"]) + request["max_tokens"] for request in requests) + 64
    import torch
    torch.cuda.synchronize()  # Once outside measurement, after correctness checks and warmup.
    started = time.perf_counter()
    while next_request < len(requests) or not engine.is_finished():
        now = time.perf_counter()
        while next_request < len(requests) and now - started >= requests[next_request]["arrival_time"]:
            request = requests[next_request]
            engine.add_request(request["prompt_token_ids"], SamplingParams(
                temperature=workload["temperature"], ignore_eos=True, max_tokens=request["max_tokens"]
            ))
            sequence = engine.scheduler.waiting[-1]
            planned_arrival = started + request["arrival_time"]
            tracker.register(sequence, request_id=request["request_id"], submission_time=planned_arrival,
                             group=request["group"], max_tokens=request["max_tokens"])
            submission_delays.append(time.perf_counter() - planned_arrival)
            next_request += 1
        if not engine.is_finished():
            if step_count >= step_limit:
                raise RuntimeError("Workload exceeded its step bound; possible scheduling stall")
            tracker.step(engine)
            step_count += 1
        elif next_request < len(requests):
            remaining = started + requests[next_request]["arrival_time"] - time.perf_counter()
            time.sleep(max(0.0, min(remaining, 0.001)))
    duration = time.perf_counter() - started
    metrics = list(tracker.metrics.values())
    if any(metric.output_len != request["max_tokens"] for metric, request in zip(metrics, requests)):
        raise RuntimeError("A request did not produce its requested output length")
    reset_idle_cache(engine)  # Requires every finished request to release its KV blocks.
    summary = {
        "scheduler_mode": args.scheduler_mode, "seed": workload["seed"],
        "request_rate": workload["request_rate"], "num_requests": len(requests),
        "token_budget": args.max_num_batched_tokens, "total_runtime": duration,
        "throughput": sum(metric.input_len + metric.output_len for metric in metrics) / duration,
        "output_throughput": sum(metric.output_len for metric in metrics) / duration,
        "request_throughput": len(metrics) / duration,
        "submission_delay_mean": float(np.mean(submission_delays)),
        "submission_delay_max": max(submission_delays),
        **summarize_requests(metrics),
    }
    for group in ("short", "long", "uniform"):
        group_metrics = [metric for metric in metrics if metric.group == group]
        if group_metrics:
            summary.update({f"{group}_{key}": value for key, value in summarize_requests(group_metrics).items()})
    result = {
        "summary": summary,
        "metadata": {**runtime_metadata(engine, config), "workload_sha256": workload_sha256(workload),
                     "correctness_check": correctness,
                     "warmup": f"{args.warmup_requests} alternating shortest/longest prompts, up to 16 output tokens; reset cache and seed before timing",
                     "latency_origin": "scheduled Poisson arrival, including submission delay",
                     "timestamp_clock": "perf_counter immediately after engine.step(); no added per-step synchronize"},
        "requests": tracker.export(started),
    }
    write_json(args.output_dir / f"{args.scheduler_mode}.json", result)
    write_csv(args.output_dir / f"{args.scheduler_mode}.csv", [summary])
    print(f"\n{args.scheduler_mode}: {duration:.3f}s, {summary['throughput']:.2f} total tokens/s, "
          f"{summary['output_throughput']:.2f} output tokens/s")
    for name in ("ttft", "tpot", "itl", "latency"):
        fields = ("mean", "p50", "p95", "p99", "max") if name == "itl" else ("mean", "p50", "p95", "p99")
        print(f"{name.upper()} (ms): " + ", ".join(f"{field}={format_ms(summary[f'{name}_{field}'])}" for field in fields))
    for group in ("short", "long"):
        if f"{group}_ttft_mean" in summary:
            print(f"{group} TTFT (ms): " + ", ".join(
                f"{field}={format_ms(summary[f'{group}_ttft_{field}'])}" for field in ("mean", "p50", "p95", "p99")))
    return result


def format_ms(value):
    return "n/a" if value is None or not math.isfinite(value) else f"{value * 1000:.3f}"


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        validate_args(args)
        seed_everything(args.seed)
        vocab_size = model_vocab_size(args.model)
        workload = (json.loads(args.workload.expanduser().read_text(encoding="utf-8"))
                    if args.workload else make_workload(args, vocab_size))
        required_context = max(len(request["prompt_token_ids"]) + request["max_tokens"] for request in workload["requests"])
        model_context = read_model_config(args.model).get("max_position_embeddings", required_context)
        args.max_model_len = min(args.max_model_len or required_context, model_context)
        validate_workload(workload, vocab_size, args.max_model_len)
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))
    args.output_dir = args.output_dir.expanduser().resolve()
    if args._worker:
        if args.scheduler_mode == "both":
            parser.error("a worker needs one scheduler mode")
        run_case(args, workload)
        return
    workload_path = args.output_dir / "workload.json"
    write_json(workload_path, workload)
    modes = ("legacy", "chunked") if args.scheduler_mode == "both" else (args.scheduler_mode,)
    # Fresh processes isolate CUDA contexts / model allocations and consume the same saved workload.
    for mode in modes:
        command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:],
                   "--scheduler-mode", mode, "--workload", str(workload_path),
                   "--output-dir", str(args.output_dir), "--max-model-len", str(args.max_model_len), "--_worker"]
        if args.chunked_prefill:
            command = [item for item in command if item != "--chunked-prefill"]
        subprocess.run(command, check=True)
    results = [json.loads((args.output_dir / f"{mode}.json").read_text(encoding="utf-8")) for mode in modes]
    signatures = [{key: result["metadata"][key] for key in ("workload_sha256", "gpus", "num_kvcache_blocks", "versions")}
                  for result in results]
    if any(signature != signatures[0] for signature in signatures[1:]):
        raise RuntimeError("Modes differ in workload, GPUs, cache capacity, or package versions; results retained for diagnosis")
    rows = [result["summary"] for result in results]
    write_csv(args.output_dir / "summary.csv", rows)
    print(f"Saved workload, per-token records, and summaries in {args.output_dir}")


if __name__ == "__main__":
    main()
