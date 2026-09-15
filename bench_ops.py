"""Operator-level benchmark: eager PyTorch vs torch.compile vs Triton.

This isolates the three ops that --use-triton replaces (RMSNorm, Add+RMSNorm,
SiluAndMul) so you can see the kernel-level cost instead of end-to-end
throughput. Shapes are derived from the model config; `--tokens` is the number
of tokens in a batch (decode batch size, or chunked-prefill chunk size).

Examples:
  # decode-sized batches (1 token per sequence)
  python bench_ops.py --model ~/models/Qwen3-4B --tokens 1,16,64,128,256

  # prefill-sized chunks
  python bench_ops.py --model ~/models/Qwen3-4B --tokens 1024,2048

  # include launch latency (no CUDA graph) instead of pure kernel time
  python bench_ops.py --model ~/models/Qwen3-4B --mode loop
"""

import argparse
import os
import statistics

import torch
import torch.nn.functional as F
from transformers import AutoConfig

from nanovllm.layers import triton_ops


def load_dims(model: str):
    cfg = AutoConfig.from_pretrained(model)
    hidden = cfg.hidden_size
    inter = cfg.intermediate_size
    n_heads = cfg.num_attention_heads
    n_kv_heads = cfg.num_key_value_heads
    head_dim = getattr(cfg, "head_dim", hidden // n_heads)
    eps = getattr(cfg, "rms_norm_eps", 1e-6)
    return hidden, inter, n_heads, n_kv_heads, head_dim, cfg.torch_dtype, eps


# --- torch references: identical math to the @torch.compile methods in the
# --- repo's RMSNorm / SiluAndMul, so eager and compiled differ only by fusion.

def rms_torch(x, w, eps):
    orig = x.dtype
    y = x.float()
    var = y.pow(2).mean(dim=-1, keepdim=True)
    y = y * torch.rsqrt(var + eps)
    return y.to(orig) * w


def add_rms_torch(x, r, w, eps):
    orig = x.dtype
    y = x.float().add_(r.float())
    residual = y.to(orig)
    var = y.pow(2).mean(dim=-1, keepdim=True)
    y = y * torch.rsqrt(var + eps)
    return y.to(orig) * w, residual


def silu_torch(x):
    a, b = x.chunk(2, dim=-1)
    return F.silu(a) * b


RMS_COMPILED = torch.compile(rms_torch)
ADD_RMS_COMPILED = torch.compile(add_rms_torch)
SILU_COMPILED = torch.compile(silu_torch)


# ---------------------------------------------------------------- timing ----

def _measure_fallback(fn, reps, trials):
    times = []
    for _ in range(trials):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(reps):
            fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / reps * 1000.0)
    return statistics.median(times)


def measure(fn, warmup, reps, trials, use_graph):
    """Median per-call time in microseconds.

    With use_graph=True the op is captured `reps` times inside one CUDA graph
    and the graph is replayed `reps` times, so launch overhead is removed and
    the number is pure GPU kernel time (what the decode CUDA-graph path sees).
    Otherwise a plain back-to-back loop is timed, which includes launch latency.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    if use_graph:
        try:
            graph = torch.cuda.CUDAGraph()
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    fn()
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            with torch.cuda.graph(graph):
                for _ in range(reps):
                    fn()
            torch.cuda.synchronize()
            times = []
            for _ in range(trials):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(reps):
                    graph.replay()
                end.record()
                torch.cuda.synchronize()
                times.append(start.elapsed_time(end) / (reps * reps) * 1000.0)
            del graph
            return statistics.median(times)
        except Exception as exc:  # capture can fail (e.g. compiled graph breaks)
            print(f"    [graph capture failed, falling back to loop: {exc}]")

    return _measure_fallback(fn, reps, trials)


# ------------------------------------------------------------ op builders ----

def _as_tuple(v):
    return v if isinstance(v, tuple) else (v,)


def _rel_diff(a, b):
    """Max abs difference normalised by the max abs value of the reference.

    bf16 kernels legitimately differ by a few ULP depending on whether the
    weight multiply happens before or after the downcast, so an absolute
    tolerance produces false alarms. A real bug (wrong normalisation, etc.)
    shows up as a relative error near or above 1.
    """
    diff = 0.0
    scale = 1e-6
    for x, y in zip(_as_tuple(a), _as_tuple(b)):
        xf, yf = x.float(), y.float()
        diff = max(diff, (xf - yf).abs().max().item())
        scale = max(scale, xf.abs().max().item())
    return diff / scale


def _reps_for(numel, cap):
    # keep the captured allocation bounded (~64 MB of output per variant)
    return max(2, min(cap, int(32_000_000 // max(numel, 1))))


def build_cases(tokens, dims, device, dtype, reps_cap):
    hidden, inter, n_heads, n_kv_heads, head_dim, eps = dims
    cases = []

    def add_case(name, shape, eager, compile_fn, triton, ref, move_bytes, numel):
        cases.append(dict(name=name, shape=shape, eager=eager, compile=compile_fn,
                          triton=triton, ref=ref, move_bytes=move_bytes,
                          reps=_reps_for(numel, reps_cap)))

    T = tokens
    # Add + RMSNorm over the hidden dim (input_layernorm / post_attention_layernorm)
    x = torch.randn(T, hidden, device=device, dtype=dtype)
    r = torch.randn(T, hidden, device=device, dtype=dtype)
    w = torch.randn(hidden, device=device, dtype=dtype)
    add_case("add_rms_norm(hidden)",
             f"{T}x{hidden}",
             lambda: add_rms_torch(x, r, w, eps),
             lambda: ADD_RMS_COMPILED(x, r, w, eps),
             lambda: triton_ops.add_rms_norm(x, r, w, eps),
             lambda: add_rms_torch(x, r, w, eps),
             4 * T * hidden * dtype.itemsize, T * hidden)

    # Plain RMSNorm over the hidden dim (final norm)
    x2 = torch.randn(T, hidden, device=device, dtype=dtype)
    add_case("rms_norm(hidden)",
             f"{T}x{hidden}",
             lambda: rms_torch(x2, w, eps),
             lambda: RMS_COMPILED(x2, w, eps),
             lambda: triton_ops.rms_norm(x2, w, eps),
             lambda: rms_torch(x2, w, eps),
             2 * T * hidden * dtype.itemsize, T * hidden)

    # Plain RMSNorm over head_dim (q_norm / k_norm, applied per head)
    rows_q = T * n_heads
    rows_k = T * n_kv_heads
    xq = torch.randn(rows_q, head_dim, device=device, dtype=dtype)
    xk = torch.randn(rows_k, head_dim, device=device, dtype=dtype)
    wq = torch.randn(head_dim, device=device, dtype=dtype)
    add_case("rms_norm(head_dim) q+k",
             f"{rows_q + rows_k}x{head_dim}",
             lambda: (rms_torch(xq, wq, eps), rms_torch(xk, wq, eps)),
             lambda: (RMS_COMPILED(xq, wq, eps), RMS_COMPILED(xk, wq, eps)),
             lambda: (triton_ops.rms_norm(xq, wq, eps), triton_ops.rms_norm(xk, wq, eps)),
             lambda: (rms_torch(xq, wq, eps), rms_torch(xk, wq, eps)),
             4 * (rows_q + rows_k) * head_dim * dtype.itemsize,
             (rows_q + rows_k) * head_dim)

    # SiluAndMul over the gate/up projection output
    g = torch.randn(T, 2 * inter, device=device, dtype=dtype)
    add_case("silu_and_mul",
             f"{T}x{2 * inter}",
             lambda: silu_torch(g),
             lambda: SILU_COMPILED(g),
             lambda: triton_ops.silu_and_mul(g),
             lambda: silu_torch(g),
             3 * T * inter * dtype.itemsize, T * inter)

    return cases


def main():
    parser = argparse.ArgumentParser(description="Operator-level eager/compile/Triton benchmark.")
    parser.add_argument("--model", type=str, default=os.path.expanduser("~/models/Qwen3-4B"))
    parser.add_argument("--tokens", type=str, default="1,16,64,128,256,1024,2048",
                        help="Comma-separated token counts (decode batch size or prefill chunk size).")
    parser.add_argument("--dtype", type=str, default=None,
                        choices=["bfloat16", "float16", "float32"],
                        help="Default: the model's dtype.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--reps", type=int, default=20, help="Max in-graph repetitions (capped by size).")
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument("--mode", choices=["graph", "loop", "both"], default="graph",
                        help="graph: pure kernel time; loop: includes launch latency.")
    parser.add_argument("--no-check", action="store_true", help="Skip Triton-vs-torch correctness check.")
    args = parser.parse_args()

    hidden, inter, n_heads, n_kv_heads, head_dim, model_dtype, eps = load_dims(args.model)
    dtype = getattr(torch, args.dtype) if args.dtype else model_dtype
    if dtype is None:
        dtype = torch.bfloat16
    tokens = [int(t) for t in args.tokens.split(",")]
    dims = (hidden, inter, n_heads, n_kv_heads, head_dim, eps)

    print(f"model={args.model}")
    print(f"hidden={hidden} intermediate={inter} heads={n_heads}/{n_kv_heads} "
          f"head_dim={head_dim} eps={eps} dtype={dtype}")
    print(f"tokens={tokens} mode={args.mode} reps<={args.reps} trials={args.trials}")

    modes = ["graph", "loop"] if args.mode == "both" else [args.mode]
    for mode in modes:
        use_graph = mode == "graph"
        print(f"\n================ mode: {mode} "
              f"({'pure kernel time' if use_graph else 'includes launch latency'}) ================")
        for T in tokens:
            print(f"\n--- tokens={T} ---")
            for case in build_cases(T, dims, args.device, dtype, args.reps):
                label = f"{case['name']:<22} {case['shape']:>16}"
                if not args.no_check:
                    diff = _rel_diff(case["ref"](), case["triton"]())
                    tol = 0.05
                    flag = "ok" if diff <= tol else f"MISMATCH(rel={diff:.3g})"
                else:
                    flag = "skip"
                t_eager = measure(case["eager"], args.warmup, case["reps"], args.trials, use_graph)
                t_comp = measure(case["compile"], args.warmup, case["reps"], args.trials, use_graph)
                t_tri = measure(case["triton"], args.warmup, case["reps"], args.trials, use_graph)
                bw = case["move_bytes"] / (t_tri * 1e-6) / 1e9 if t_tri > 0 else 0.0
                print(f"{label}  eager {t_eager:8.2f}us  compile {t_comp:8.2f}us  "
                      f"triton {t_tri:8.2f}us  tri/comp {t_comp / t_tri:5.2f}x  "
                      f"tri/ea {t_eager / t_tri:5.2f}x  {bw:7.1f}GB/s  [{flag}]")
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
