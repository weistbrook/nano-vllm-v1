"""Shared measurements and controlled workloads for scheduler experiments.

All persisted durations and relative timestamps are in seconds.  Throughput is
input plus output tokens / measured wall time; output_throughput is separate.
Importing this module does not initialize CUDA or import the inference engine.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, field
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import random
import subprocess
import sys
from time import perf_counter
from typing import Any, Iterable

import numpy as np


@dataclass
class RequestMetrics:
    request_id: str
    input_len: int
    submission_time: float
    group: str = "short"
    max_tokens: int = 0
    token_timestamps: list[float] = field(default_factory=list)
    completion_time: float | None = None

    @property
    def output_len(self):
        return len(self.token_timestamps)

    @property
    def ttft(self):
        return self.token_timestamps[0] - self.submission_time if self.token_timestamps else math.nan

    @property
    def tpot(self):
        times = self.token_timestamps
        return (times[-1] - times[0]) / (len(times) - 1) if len(times) > 1 else math.nan

    @property
    def latency(self):
        return self.completion_time - self.submission_time if self.completion_time is not None else math.nan

    @property
    def itls(self):
        return [end - start for start, end in zip(self.token_timestamps, self.token_timestamps[1:])]


class TokenTracker:
    """Keep sequence references, including those removed from running on finish.

    A completion-count delta, rather than KV progress, identifies output tokens.
    Every token emitted by a batch gets the same step-completion timestamp.
    """

    def __init__(self):
        self.metrics: dict[int, RequestMetrics] = {}
        self.sequences: dict[int, Any] = {}

    def register(self, seq, request_id=None, submission_time=None, group="short", max_tokens=None):
        if seq.seq_id in self.sequences or seq.num_completion_tokens != 0:
            raise ValueError("Only new, unregistered requests can be tracked")
        metric = RequestMetrics(
            str(seq.seq_id) if request_id is None else str(request_id),
            seq.num_prompt_tokens,
            perf_counter() if submission_time is None else submission_time,
            group,
            seq.max_tokens if max_tokens is None else max_tokens,
        )
        self.sequences[seq.seq_id] = seq
        self.metrics[seq.seq_id] = metric
        return metric

    def step(self, engine):
        active = [(seq_id, seq, seq.num_completion_tokens)
                  for seq_id, seq in self.sequences.items() if not seq.is_finished]
        result = engine.step()
        completed_at = perf_counter()  # Before Python metric processing, shared by the batch.
        for seq_id, seq, previous in active:
            delta = seq.num_completion_tokens - previous
            if delta not in (0, 1):
                raise AssertionError(f"Request {seq_id}: completion count changed by {delta}, expected 0 or 1")
            metric = self.metrics[seq_id]
            if delta:
                metric.token_timestamps.append(completed_at)
            if seq.is_finished:
                if not delta:
                    raise AssertionError(f"Request {seq_id} finished without a final output token")
                metric.completion_time = completed_at
            if seq.num_completion_tokens != metric.output_len:
                raise AssertionError(f"Request {seq_id}: lost or duplicate output timestamp")
        return result, completed_at

    def export(self, origin):
        requests = []
        for seq_id, metric in self.metrics.items():
            row = asdict(metric)
            row["seq_id"] = seq_id
            row["submission_time"] -= origin
            row["token_timestamps"] = [t - origin for t in metric.token_timestamps]
            if row["completion_time"] is not None:
                row["completion_time"] -= origin
            requests.append(row)
        return requests


def stats(values: Iterable[float]):
    values = np.asarray([v for v in values if v is not None and math.isfinite(v)], dtype=float)
    result = {"count": int(values.size)}
    if values.size:
        result.update(mean=float(values.mean()), p50=float(np.percentile(values, 50)),
                      p95=float(np.percentile(values, 95)), p99=float(np.percentile(values, 99)),
                      max=float(values.max()))
    else:
        result.update({key: None for key in ("mean", "p50", "p95", "p99", "max")})
    return result


def prefix_stats(values, prefix):
    return {f"{prefix}_{key}": value for key, value in stats(values).items()}


def summarize_requests(metrics):
    metrics = list(metrics)
    return {
        **prefix_stats((m.ttft for m in metrics), "ttft"),
        **prefix_stats((m.tpot for m in metrics), "tpot"),
        **prefix_stats((itl for m in metrics for itl in m.itls), "itl"),
        **prefix_stats((m.latency for m in metrics), "latency"),
    }


def phase_itls(requests, arrival, prefill_completion):
    """Partition complete ITL intervals by overlap, never by endpoint alone.

    The legacy blocked interval commonly ENDS AFTER long prefill completes.
    It still belongs to interference.  An interval is never split or duplicated.
    """
    if prefill_completion < arrival:
        raise ValueError("Prefill completion must not precede arrival")
    phases = {"before": [], "interference": [], "after": []}
    for metric in requests:
        times = metric.token_timestamps
        for start, end in zip(times, times[1:]):
            if end <= arrival:
                phase = "before"
            elif start >= prefill_completion:
                phase = "after"
            else:
                phase = "interference"
            phases[phase].append(end - start)
    return phases


def seed_everything(seed, include_torch=False):
    random.seed(seed)
    np.random.seed(seed)
    if include_torch:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(value), indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path, rows):
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(_json_safe(rows))


def workload_sha256(workload):
    return hashlib.sha256(json.dumps(workload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_model_config(model):
    path = Path(model).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"Model must be a local model directory: {path}")
    try:
        return json.loads((path / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read model configuration at {path / 'config.json'}: {exc}") from exc


def model_vocab_size(model):
    size = int(read_model_config(model)["vocab_size"])
    if size < 2:
        raise ValueError("Model vocabulary must contain at least 2 token IDs")
    return size


def create_engine(config, mode):
    try:
        import torch
        from nanovllm import LLM
    except ImportError as exc:
        raise RuntimeError("GPU benchmark dependencies are unavailable. Install this project in a supported "
                           "Linux CUDA / Python 3.10-3.12 environment; no timings were produced.") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("A supported CUDA GPU is required; no timings were produced")
    return LLM(
        str(Path(config["model"]).expanduser().resolve()), scheduler_mode=mode,
        max_num_batched_tokens=config["max_num_batched_tokens"],
        max_num_seqs=config["max_num_seqs"], max_model_len=config["max_model_len"],
        tensor_parallel_size=config.get("tensor_parallel_size", 1),
        gpu_memory_utilization=config.get("gpu_memory_utilization", 0.9),
        enforce_eager=config.get("enforce_eager", True), use_triton=config.get("use_triton", False),
        use_triton_hidden_rmsnorm=config.get("use_triton_hidden_rmsnorm", False),
    )


def assert_idle_cache(engine):
    manager = engine.scheduler.block_manager
    if not engine.is_finished() or manager.used_block_ids:
        raise AssertionError("Finished workload left queued requests or allocated KV blocks")
    if (len(manager.free_block_ids) != len(manager.blocks)
            or set(manager.free_block_ids) != set(range(len(manager.blocks)))
            or any(b.ref_count for b in manager.blocks)):
        raise AssertionError("KV block leak after workload completion")


def reset_idle_cache(engine):
    """Discard prefix metadata, not GPU allocation, between untimed/timed runs."""
    assert_idle_cache(engine)
    old = engine.scheduler.block_manager
    engine.scheduler.block_manager = type(old)(len(old.blocks), old.block_size)


def check_engine_structure(engine, prompts=None, temperature=0.6, seed=100):
    """Small unmeasured real-engine check, including partial/shared-prefix blocks.

    Temporarily reduce only the scheduler budget to 17 to exercise partial blocks.
    Run a second shared-prefix request after completion to exercise cached reuse.
    Sampling remains unchanged and output identity is intentionally not required.
    """
    from nanovllm import SamplingParams
    scheduler = engine.scheduler
    reset_idle_cache(engine)
    original_budget = scheduler.max_num_batched_tokens
    block_size = scheduler.block_manager.block_size
    max_length = scheduler.max_model_len - 4
    length = min(block_size + 17, max_length)
    if length < 2:
        raise ValueError("max_model_len is too small for correctness checks")
    rng = random.Random(seed + 7919)
    vocab = int(engine.model_runner.config.hf_config.vocab_size)
    if prompts is None:
        shared = [rng.randrange(vocab) for _ in range(length)]
        sibling = shared.copy()
        sibling[-1] = (sibling[-1] + 1) % vocab
        prompts = [shared, sibling, [rng.randrange(vocab) for _ in range(min(31, length))]]
    checked = 0
    scheduler.max_num_batched_tokens = min(17, original_budget)
    try:
        for batch in (prompts, [prompts[0]]):
            tracker = TokenTracker()
            for prompt in batch:
                seq = engine.add_request(prompt, SamplingParams(temperature=temperature, ignore_eos=True, max_tokens=3))
                if seq is None:  # Compatibility with older add_request callers.
                    seq = scheduler.waiting[-1]
                tracker.register(seq)
            bound = sum(len(prompt) + 3 for prompt in batch) * 2 + 32
            steps = 0
            while not engine.is_finished():
                if steps >= bound:
                    raise AssertionError("Correctness check stalled / exceeded step bound")
                tracker.step(engine)
                steps += 1
                manager = scheduler.block_manager
                for block in manager.blocks:
                    if block.hash != -1 and len(block.token_ids) != block_size:
                        raise AssertionError("Partial block was inserted into the prefix cache")
                for seq in tracker.sequences.values():
                    if seq.is_finished and seq.block_table:
                        raise AssertionError("Finished sequence retained KV blocks")
                    if not seq.is_finished and not (0 <= seq.num_cached_tokens <= len(seq)):
                        raise AssertionError("Invalid incremental KV token count")
                    if any(i < 0 or i >= len(manager.blocks) for i in seq.block_table):
                        raise AssertionError("KV block index out of bounds")
            if any(m.output_len != 3 for m in tracker.metrics.values()):
                raise AssertionError("Correctness check returned wrong number of output tokens")
            assert_idle_cache(engine)
            checked += len(batch)
    finally:
        scheduler.max_num_batched_tokens = original_budget
    reset_idle_cache(engine)
    return {"passed": True, "requests": checked, "partial_block_check": True,
            "full_prefix_reuse_exercised": length > block_size, "sampling_identity_required": False}


def runtime_metadata(engine, config):
    import torch
    versions = {}
    for name in ("torch", "transformers", "triton", "flash-attn", "numpy"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                           cwd=Path(__file__).resolve().parents[1]).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True,
                                            cwd=Path(__file__).resolve().parents[1]).strip())
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    return {
        "python": sys.version, "platform": platform.platform(), "versions": versions,
        "torch_cuda": torch.version.cuda, "git_revision": revision, "git_dirty": dirty,
        "gpus": [{"name": torch.cuda.get_device_name(i),
                  "total_memory": torch.cuda.get_device_properties(i).total_memory,
                  "uuid": str(getattr(torch.cuda.get_device_properties(i), "uuid", "unavailable"))}
                 for i in range(config.get("tensor_parallel_size", 1))],
        "num_kvcache_blocks": len(engine.scheduler.block_manager.blocks),
        "block_size": engine.scheduler.block_manager.block_size,
        "effective_max_model_len": engine.scheduler.max_model_len,
        "settings": dict(config, enforce_eager=config.get("enforce_eager", True),
                         use_triton=config.get("use_triton", False),
                         use_triton_hidden_rmsnorm=config.get("use_triton_hidden_rmsnorm", False)),
        "latency_clock": "perf_counter immediately after output-producing engine step; sampling .tolist synchronizes outputs",
        "step_trace_clock": "host engine.step return times; zero-output prefill steps may still have GPU work pending",
        "timestamp_units": "seconds relative to measured run start",
        "long_prefill_completion_definition": "first long-request output token at step completion (includes sampling)",
        "throughput_definition": "(input tokens + output tokens) / measured runtime",
        "warmup": "full identical workload rehearsal; reset prefix metadata and reseed before timing",
        "kernel_note": "use_triton=False disables optional fused kernels; KV store and torch.compile may use Triton",
    }


def print_interference_summary(summary):
    def milliseconds(value):
        return "n/a" if value is None else f"{1000 * value:.3f}"

    print(f"\n{summary['scheduler_mode']} run {summary['run_id']}: "
          f"runtime {summary['total_runtime']:.3f} s, "
          f"throughput {summary['throughput']:.2f} total tokens/s, "
          f"{summary['output_throughput']:.2f} output tokens/s")
    columns = ("mean", "p50", "p95", "p99", "max")
    print("Latency statistics (ms): mean / P50 / P95 / P99 / max")
    for prefix, label in (("short_ttft", "Short TTFT"), ("decode_itl", "Short decode ITL"),
                          ("before_itl", "Before arrival ITL"), ("interference_itl", "Interference ITL"),
                          ("after_itl", "After prefill ITL")):
        values = " / ".join(milliseconds(summary[f"{prefix}_{key}"]) for key in columns)
        print(f"  {label}: {values} (n={summary[f'{prefix}_count']})")
    print(f"  Short TPOT mean: {milliseconds(summary['short_tpot_mean'])}")
    print(f"  Long TTFT: {milliseconds(summary['long_ttft'])}; latency: {milliseconds(summary['long_latency'])}")
    print(f"  Decode survivors at long prefill end: {summary['short_decode_survivors_at_long_prefill_completion']}; "
          f"valid interference window: {summary['interference_window_valid']}", flush=True)


def add_experiment_args(parser):
    parser.add_argument("--model", required=True, help="Local Qwen3-4B model directory")
    parser.add_argument("--num-decode-requests", type=int, default=32)
    parser.add_argument("--short-prompt-len", type=int, default=128)
    parser.add_argument("--short-output-len", type=int, default=256)
    parser.add_argument("--long-prompt-len", type=int, default=8192)
    parser.add_argument("--long-output-len", type=int, default=64)
    parser.add_argument("--inject-after-tokens", type=int, default=16)
    parser.add_argument("--max-num-batched-tokens", type=int, default=1024)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--enforce-eager", action="store_true", default=True,
                        help="Always enabled in these controlled experiments (CUDA Graph OFF)")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--no-plots", action="store_true", help="Save JSON/CSV without rendering figures")


def validate_experiment(config):
    for key in ("num_decode_requests", "short_prompt_len", "short_output_len", "long_prompt_len",
                "long_output_len", "inject_after_tokens", "max_num_batched_tokens", "max_num_seqs"):
        if config[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if config["max_num_batched_tokens"] <= config["num_decode_requests"]:
        raise ValueError("Token budget must exceed num_decode_requests: equality leaves no long-prefill budget")
    if config["max_num_seqs"] < config["num_decode_requests"] + 1:
        raise ValueError("max_num_seqs must hold all short requests plus the long request")
    if config["inject_after_tokens"] >= config["short_output_len"]:
        raise ValueError("inject_after_tokens must be less than short_output_len, leaving active decode requests")
    if config["temperature"] <= 1e-10 or not math.isfinite(config["temperature"]):
        raise ValueError("The current sampler requires positive temperature; structural checks do not compare token IDs")
    if not (1 <= config["tensor_parallel_size"] <= 8):
        raise ValueError("tensor_parallel_size must be between 1 and 8")
    if not (0 < config["gpu_memory_utilization"] <= 1):
        raise ValueError("gpu_memory_utilization must be in (0, 1]")
    needed = max(config["short_prompt_len"] + config["short_output_len"],
                 config["long_prompt_len"] + config["long_output_len"])
    if config["max_model_len"] is not None and config["max_model_len"] < needed:
        raise ValueError(f"max_model_len must be at least prompt + output length ({needed})")


def make_interference_workload(config, vocab_size, seed):
    rng = random.Random(seed)
    requests = [
        {"request_id": f"short_{i}", "group": "short", "arrival_time": 0.0,
         "prompt_token_ids": [rng.randrange(vocab_size) for _ in range(config["short_prompt_len"])],
         "max_tokens": config["short_output_len"]}
        for i in range(config["num_decode_requests"])
    ]
    requests.append({"request_id": "long", "group": "long", "arrival_time": None,
                     "prompt_token_ids": [rng.randrange(vocab_size) for _ in range(config["long_prompt_len"])],
                     "max_tokens": config["long_output_len"]})
    return {"schema_version": 1, "seed": seed, "vocab_size": vocab_size,
            "injection": {"all_short_requests_output_at_least": config["inject_after_tokens"]},
            "requests": requests}


def run_interference(engine, workload, config, mode, run_id, record_steps=True):
    from nanovllm import SamplingParams
    tracker = TokenTracker()
    short_seqs = []
    long_seq = None
    long_arrival = None
    long_prefill_end = None
    survivors = None
    step_records = []
    scheduler = engine.scheduler
    original_schedule = scheduler.schedule
    captured = []

    def recorded_schedule():
        nonlocal captured
        scheduled = original_schedule()
        captured = [(seq.seq_id, seq.num_new_tokens,
                     seq.num_completion_tokens > 0 and len(seq) - seq.num_cached_tokens == 1)
                    for seq in scheduled]
        return scheduled

    if record_steps:
        scheduler.schedule = recorded_schedule
    start = perf_counter()
    last_step_time = start
    bound = 2 * sum(len(r["prompt_token_ids"]) + r["max_tokens"] for r in workload["requests"]) + 64
    steps = 0

    def submit(request, arrival):
        seq = engine.add_request(request["prompt_token_ids"], SamplingParams(
            temperature=config["temperature"], ignore_eos=True, max_tokens=request["max_tokens"]))
        if seq is None:
            seq = engine.scheduler.waiting[-1]
        tracker.register(seq, request["request_id"], arrival, request["group"], request["max_tokens"])
        return seq

    try:
        for request in workload["requests"]:
            if request["group"] == "short":
                short_seqs.append(submit(request, start))
        long_request = next(r for r in workload["requests"] if r["group"] == "long")
        while not engine.is_finished():
            if long_seq is None and all(seq.num_completion_tokens >= config["inject_after_tokens"] for seq in short_seqs):
                if any(seq.is_finished for seq in short_seqs):
                    raise RuntimeError("A short request finished before the injection threshold was reached by all requests; "
                                       "increase short_output_len or reduce inject_after_tokens")
                long_arrival = perf_counter()
                long_seq = submit(long_request, long_arrival)
            if steps >= bound:
                raise RuntimeError("Workload exceeded bounded step count; possible scheduling stall")
            _, completed_at = tracker.step(engine)
            steps += 1
            if record_steps:
                step_records.append({"step": steps, "start_time": last_step_time - start,
                                     "completion_time": completed_at - start,
                                     "scheduled_tokens": sum(item[1] for item in captured),
                                     "decode_requests": sum(item[2] for item in captured),
                                     "prefill_tokens": sum(tokens for _, tokens, decode in captured if not decode),
                                     "long_prefill_tokens": sum(tokens for seq_id, tokens, decode in captured
                                                                 if long_seq is not None and seq_id == long_seq.seq_id and not decode)})
            last_step_time = completed_at
            if long_seq is not None and long_prefill_end is None and long_seq.num_completion_tokens:
                long_prefill_end = tracker.metrics[long_seq.seq_id].token_timestamps[0]
                survivors = sum(not seq.is_finished for seq in short_seqs)
    finally:
        if record_steps:
            scheduler.schedule = original_schedule
    duration = last_step_time - start
    if long_seq is None or long_prefill_end is None:
        raise RuntimeError("Long request was never injected or never produced a first token")
    if any(m.output_len != m.max_tokens for m in tracker.metrics.values()):
        raise AssertionError("Incorrect output token counts in measured workload")
    assert_idle_cache(engine)
    short_metrics = [tracker.metrics[seq.seq_id] for seq in short_seqs]
    long_metric = tracker.metrics[long_seq.seq_id]
    phases = phase_itls(short_metrics, long_arrival, long_prefill_end)
    inputs = sum(m.input_len for m in tracker.metrics.values())
    outputs = sum(m.output_len for m in tracker.metrics.values())
    short_summary = summarize_requests(short_metrics)
    summary = {
        "scheduler_mode": mode, "run_id": run_id, "seed": workload["seed"],
        "long_prompt_len": len(long_request["prompt_token_ids"]),
        "token_budget": config["max_num_batched_tokens"], "total_runtime": duration,
        "throughput": (inputs + outputs) / duration, "output_throughput": outputs / duration,
        "input_tokens": inputs, "output_tokens": outputs,
        **{f"short_{k}": v for k, v in short_summary.items() if k.startswith(("ttft_", "tpot_"))},
        **{f"decode_{k}": v for k, v in short_summary.items() if k.startswith("itl_")},
        **{f"{phase}_itl_{k}": v for phase, values in phases.items() for k, v in stats(values).items()},
        "long_ttft": long_metric.ttft, "long_latency": long_metric.latency,
        "long_arrival_time": long_arrival - start, "long_prefill_completion_time": long_prefill_end - start,
        "short_decode_survivors_at_long_prefill_completion": survivors,
        "interference_window_valid": bool(phases["interference"]) and survivors == len(short_seqs),
        "step_count": steps, "workload_sha256": workload_sha256(workload),
        "long_prefill_steps": sum(r["long_prefill_tokens"] > 0 for r in step_records),
        "interference_mean_step_prefill_tokens": stats(r["prefill_tokens"] for r in step_records
                                                         if r["long_prefill_tokens"] > 0)["mean"],
    }
    return {"schema_version": 1, "summary": summary, "requests": tracker.export(start),
            "long_arrival_time": long_arrival - start, "long_prefill_completion_time": long_prefill_end - start,
            "step_records": step_records}


def interference_worker(job_path):
    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    config, mode = job["config"], job["mode"]
    workload = json.loads(Path(job["workload_path"]).read_text(encoding="utf-8"))
    if workload_sha256(workload) != job["workload_sha256"]:
        raise ValueError("Saved workload checksum does not match the job")
    seed_everything(workload["seed"])
    engine = create_engine(config, mode)
    try:
        seed_everything(workload["seed"], include_torch=True)
        correctness = check_engine_structure(engine, temperature=config["temperature"], seed=workload["seed"])
        seed_everything(workload["seed"], include_torch=True)
        run_interference(engine, workload, config, mode, job["run_id"])
        reset_idle_cache(engine)
        seed_everything(workload["seed"], include_torch=True)
        import torch
        torch.cuda.synchronize()  # Once outside timing, after correctness/warmup.
        result = run_interference(engine, workload, config, mode, job["run_id"])
        result["metadata"] = runtime_metadata(engine, config)
        result["metadata"]["correctness"] = correctness
        result["metadata"]["workload_path"] = job["workload_path"]
        write_json(job["result_path"], result)
        print_interference_summary(result["summary"])
    finally:
        engine.exit()


def launch_worker(config, mode, run_id, workload_path, result_path):
    workload = json.loads(Path(workload_path).read_text(encoding="utf-8"))
    job_path = Path(result_path).with_suffix(".job.json")
    write_json(job_path, {"config": config, "mode": mode, "run_id": run_id,
                          "workload_path": str(Path(workload_path).resolve()),
                          "workload_sha256": workload_sha256(workload),
                          "result_path": str(Path(result_path).resolve())})
    script = Path(__file__).with_name("long_prompt_interference.py")
    completed = subprocess.run([sys.executable, str(script), "--worker-job", str(job_path.resolve())])
    if completed.returncode:
        raise RuntimeError(f"{mode} worker failed (exit {completed.returncode}); see worker output. "
                           "No substitute or fabricated timings were generated.")
    return json.loads(Path(result_path).read_text(encoding="utf-8"))


def verify_comparable(results):
    """Reject A/B pairs with distinct hardware or effective cache capacities."""
    groups = {}
    for result in results:
        s, m = result["summary"], result["metadata"]
        key = (s["run_id"], s["long_prompt_len"], s["token_budget"])
        signature = (s["workload_sha256"], m["num_kvcache_blocks"], m["gpus"], m["versions"], m["settings"])
        if key in groups and groups[key] != signature:
            raise RuntimeError("Paired modes differ in workload, GPU, settings, versions or KV capacity; "
                               "results were retained for diagnosis but are not a controlled comparison")
        groups[key] = signature


def prepare_config(args, largest_prompt=None):
    config = {key: value for key, value in vars(args).items()
              if key not in ("output_dir", "no_plots", "worker_job", "scheduler_mode", "repeats", "prompt_lengths", "budgets")}
    config["model"] = str(Path(config["model"]).expanduser().resolve())
    validate_experiment(config)
    required = max(config["short_prompt_len"] + config["short_output_len"],
                   (largest_prompt or config["long_prompt_len"]) + config["long_output_len"])
    config["max_model_len"] = config["max_model_len"] or required
    hf_config = read_model_config(config["model"])
    if required > config["max_model_len"] or config["max_model_len"] > int(hf_config["max_position_embeddings"]):
        raise ValueError("Requested prompt + output lengths / max_model_len exceed the model context limit")
    return config, int(hf_config["vocab_size"])
