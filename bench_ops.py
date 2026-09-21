"""Operator-level benchmark: eager PyTorch vs torch.compile vs Triton.

Uses triton.testing.perf_report + triton.testing.Benchmark so plots and an
HTML report are generated automatically (open the generated .html under
--out-dir, or view inline when running inside a notebook).

Timing uses triton.testing.do_bench (loop mode) which flushes the L2 cache
between iterations. This matters a lot on consumer GPUs such as the RTX 5060
Ti (roughly 448 GB/s specification peak): a naive back-to-back loop keeps
re-reading the same tensors from L2 and the "GB/s" figure can be several times
the real HBM bandwidth. The utilization report uses minimum logical tensor
traffic and a measured copy-bandwidth baseline by default.

Examples:
  python bench_ops.py --model ~/models/Qwen3-4B --mode loop
  python bench_ops.py --model ~/models/Qwen3-4B --mode graph
  python bench_ops.py --model ~/models/Qwen3-4B --no-show
"""

import argparse
import os

import torch
import torch.nn.functional as F
import triton
import triton.testing
from transformers import AutoConfig

from nanovllm.layers import triton_ops


# -------------------------------------------------------------- model dims --

def load_dims(model):
    cfg = AutoConfig.from_pretrained(model)
    hidden = cfg.hidden_size
    inter = cfg.intermediate_size
    n_heads = cfg.num_attention_heads
    n_kv_heads = cfg.num_key_value_heads
    head_dim = getattr(cfg, "head_dim", hidden // n_heads)
    eps = getattr(cfg, "rms_norm_eps", 1e-6)
    return hidden, inter, n_heads, n_kv_heads, head_dim, cfg.torch_dtype, eps


# -------------------------------------------------------- reference ops ----

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


# ----------------------------------------------------------- timing helper --

def _do_bench(fn, mode):
    """Return median per-call time in microseconds.

    mode == "loop"  -> triton.testing.do_bench, flushes a 256 MB buffer before
                       every timed iteration. Real HBM traffic.
    mode == "graph" -> triton.testing.do_bench_cudagraph, captures the op into
                       a CUDA graph and replays it. Launch overhead removed,
                       but L2 is not flushed inside the graph.
    """
    if mode == "graph":
        try:
            from triton.testing import do_bench_cudagraph
            return do_bench_cudagraph(fn, rep=100) * 1000.0
        except Exception:
            pass
    from triton.testing import do_bench
    return do_bench(fn, warmup=25, rep=100) * 1000.0


def _measure_peak_bandwidth(device):
    """Measure a copy bandwidth baseline in GB/s on the selected device."""
    size = 256 * 1024 * 1024
    source = torch.empty(size, dtype=torch.uint8, device=device)
    target = torch.empty_like(source)
    elapsed_ms = triton.testing.do_bench(
        lambda: target.copy_(source), warmup=25, rep=100)
    return 2 * source.numel() / (elapsed_ms * 1e-3) / 1e9


# ------------------------------------------------------- report builders ----

X_VALS = [1, 16, 64, 128, 256, 512, 1024, 2048]


def _bench_kwargs(plot_name, ylabel="time (us)"):
    return dict(
        x_names=["T"],
        x_vals=X_VALS,
        x_log=True,
        line_arg="backend",
        line_vals=["eager", "compile", "triton"],
        line_names=["eager", "torch.compile", "triton"],
        styles=[("gray", "-"), ("tab:blue", "-"), ("tab:red", "-")],
        ylabel=ylabel,
        plot_name=plot_name,
        args={},
    )


def _effective_bandwidth_gbps(byte_count, elapsed_us):
    return byte_count / (elapsed_us * 1e-6) / 1e9


def _bandwidth_report_kwargs(plot_name):
    return _bench_kwargs(plot_name, "bandwidth utilization (% of peak)")


def build_bandwidth_reports(dims, device, dtype, mode, peak_bandwidth):
    """Return reports for logical bandwidth and peak-bandwidth utilization."""
    hidden, inter, n_heads, n_kv_heads, head_dim, eps = dims
    itemsize = torch.empty((), dtype=dtype).element_size()
    reports = []

    def make_report(plot_name, traffic_bytes, make_fn):
        @triton.testing.perf_report(
            triton.testing.Benchmark(**_bandwidth_report_kwargs(plot_name)))
        def report(T, backend):
            fn = make_fn(T, backend)
            elapsed_us = _do_bench(fn, mode)
            bandwidth_gbps = _effective_bandwidth_gbps(
                traffic_bytes(T), elapsed_us)
            return 100.0 * bandwidth_gbps / peak_bandwidth
        return report

    def add_rms_fn(T, backend):
        x = torch.randn(T, hidden, device=device, dtype=dtype)
        r = torch.randn(T, hidden, device=device, dtype=dtype)
        w = torch.randn(hidden, device=device, dtype=dtype)
        if backend == "eager":
            return lambda: add_rms_torch(x, r, w, eps)
        if backend == "compile":
            return lambda: ADD_RMS_COMPILED(x, r, w, eps)
        return lambda: triton_ops.add_rms_norm(x, r, w, eps)

    reports.append(make_report(
        "add_rms_norm(hidden) bandwidth",
        lambda T: 4 * T * hidden * itemsize,
        add_rms_fn,
    ))

    def rms_hidden_fn(T, backend):
        x = torch.randn(T, hidden, device=device, dtype=dtype)
        w = torch.randn(hidden, device=device, dtype=dtype)
        if backend == "eager":
            return lambda: rms_torch(x, w, eps)
        if backend == "compile":
            return lambda: RMS_COMPILED(x, w, eps)
        return lambda: triton_ops.rms_norm(x, w, eps)

    reports.append(make_report(
        "rms_norm(hidden) bandwidth",
        lambda T: 2 * T * hidden * itemsize,
        rms_hidden_fn,
    ))

    def rms_head_fn(T, backend):
        rows_q = T * n_heads
        rows_k = T * n_kv_heads
        xq = torch.randn(rows_q, head_dim, device=device, dtype=dtype)
        xk = torch.randn(rows_k, head_dim, device=device, dtype=dtype)
        wq = torch.randn(head_dim, device=device, dtype=dtype)
        if backend == "eager":
            return lambda: (rms_torch(xq, wq, eps), rms_torch(xk, wq, eps))
        if backend == "compile":
            return lambda: (RMS_COMPILED(xq, wq, eps),
                            RMS_COMPILED(xk, wq, eps))
        return lambda: (triton_ops.rms_norm(xq, wq, eps),
                        triton_ops.rms_norm(xk, wq, eps))

    reports.append(make_report(
        "rms_norm(head_dim) q+k bandwidth",
        lambda T: 4 * T * (n_heads + n_kv_heads) * head_dim * itemsize,
        rms_head_fn,
    ))

    def silu_fn(T, backend):
        g = torch.randn(T, 2 * inter, device=device, dtype=dtype)
        if backend == "eager":
            return lambda: silu_torch(g)
        if backend == "compile":
            return lambda: SILU_COMPILED(g)
        return lambda: triton_ops.silu_and_mul(g)

    reports.append(make_report(
        "silu_and_mul bandwidth",
        lambda T: 3 * T * inter * itemsize,
        silu_fn,
    ))

    return reports


def build_reports(dims, device, dtype, mode):
    """Return a list of perf_report-decorated functions, one per op."""
    hidden, inter, n_heads, n_kv_heads, head_dim, eps = dims
    reports = []

    # ---- add_rms_norm(hidden) ----
    @triton.testing.perf_report(
        triton.testing.Benchmark(**_bench_kwargs("add_rms_norm(hidden)")))
    def bench_add_rms(T, backend):
        x = torch.randn(T, hidden, device=device, dtype=dtype)
        r = torch.randn(T, hidden, device=device, dtype=dtype)
        w = torch.randn(hidden, device=device, dtype=dtype)
        if backend == "eager":
            fn = lambda: add_rms_torch(x, r, w, eps)
        elif backend == "compile":
            fn = lambda: ADD_RMS_COMPILED(x, r, w, eps)
        else:
            fn = lambda: triton_ops.add_rms_norm(x, r, w, eps)
        return _do_bench(fn, mode)
    reports.append(bench_add_rms)

    # ---- rms_norm(hidden) ----
    @triton.testing.perf_report(
        triton.testing.Benchmark(**_bench_kwargs("rms_norm(hidden)")))
    def bench_rms_hidden(T, backend):
        x = torch.randn(T, hidden, device=device, dtype=dtype)
        w = torch.randn(hidden, device=device, dtype=dtype)
        if backend == "eager":
            fn = lambda: rms_torch(x, w, eps)
        elif backend == "compile":
            fn = lambda: RMS_COMPILED(x, w, eps)
        else:
            fn = lambda: triton_ops.rms_norm(x, w, eps)
        return _do_bench(fn, mode)
    reports.append(bench_rms_hidden)

    # ---- rms_norm(head_dim) q+k ----
    @triton.testing.perf_report(
        triton.testing.Benchmark(**_bench_kwargs("rms_norm(head_dim) q+k")))
    def bench_rms_head(T, backend):
        rows_q = T * n_heads
        rows_k = T * n_kv_heads
        xq = torch.randn(rows_q, head_dim, device=device, dtype=dtype)
        xk = torch.randn(rows_k, head_dim, device=device, dtype=dtype)
        wq = torch.randn(head_dim, device=device, dtype=dtype)
        if backend == "eager":
            fn = lambda: (rms_torch(xq, wq, eps), rms_torch(xk, wq, eps))
        elif backend == "compile":
            fn = lambda: (RMS_COMPILED(xq, wq, eps),
                          RMS_COMPILED(xk, wq, eps))
        else:
            fn = lambda: (triton_ops.rms_norm(xq, wq, eps),
                          triton_ops.rms_norm(xk, wq, eps))
        return _do_bench(fn, mode)
    reports.append(bench_rms_head)

    # ---- silu_and_mul ----
    @triton.testing.perf_report(
        triton.testing.Benchmark(**_bench_kwargs("silu_and_mul")))
    def bench_silu(T, backend):
        g = torch.randn(T, 2 * inter, device=device, dtype=dtype)
        if backend == "eager":
            fn = lambda: silu_torch(g)
        elif backend == "compile":
            fn = lambda: SILU_COMPILED(g)
        else:
            fn = lambda: triton_ops.silu_and_mul(g)
        return _do_bench(fn, mode)
    reports.append(bench_silu)

    return reports


# ------------------------------------------------------------ correctness ----

def _as_tuple(v):
    return v if isinstance(v, tuple) else (v,)


def _rel_diff(a, b):
    diff, scale = 0.0, 1e-6
    for x, y in zip(_as_tuple(a), _as_tuple(b)):
        xf, yf = x.float(), y.float()
        diff = max(diff, (xf - yf).abs().max().item())
        scale = max(scale, xf.abs().max().item())
    return diff / scale


def check_correctness(dims, device, dtype, T=64):
    hidden, inter, n_heads, n_kv_heads, head_dim, eps = dims

    x = torch.randn(T, hidden, device=device, dtype=dtype)
    r = torch.randn(T, hidden, device=device, dtype=dtype)
    w = torch.randn(hidden, device=device, dtype=dtype)
    d = _rel_diff(add_rms_torch(x, r, w, eps),
                  triton_ops.add_rms_norm(x, r, w, eps))
    print(f"add_rms_norm(hidden)      rel_diff = {d:.3g}  "
          f"[{'ok' if d <= 0.05 else 'MISMATCH'}]")

    x2 = torch.randn(T, hidden, device=device, dtype=dtype)
    d = _rel_diff(rms_torch(x2, w, eps), triton_ops.rms_norm(x2, w, eps))
    print(f"rms_norm(hidden)          rel_diff = {d:.3g}  "
          f"[{'ok' if d <= 0.05 else 'MISMATCH'}]")

    rows_q, rows_k = T * n_heads, T * n_kv_heads
    xq = torch.randn(rows_q, head_dim, device=device, dtype=dtype)
    xk = torch.randn(rows_k, head_dim, device=device, dtype=dtype)
    wq = torch.randn(head_dim, device=device, dtype=dtype)
    d = _rel_diff((rms_torch(xq, wq, eps), rms_torch(xk, wq, eps)),
                  (triton_ops.rms_norm(xq, wq, eps),
                   triton_ops.rms_norm(xk, wq, eps)))
    print(f"rms_norm(head_dim) q+k    rel_diff = {d:.3g}  "
          f"[{'ok' if d <= 0.05 else 'MISMATCH'}]")

    g = torch.randn(T, 2 * inter, device=device, dtype=dtype)
    d = _rel_diff(silu_torch(g), triton_ops.silu_and_mul(g))
    print(f"silu_and_mul              rel_diff = {d:.3g}  "
          f"[{'ok' if d <= 0.05 else 'MISMATCH'}]")


# ------------------------------------------------------------------ main ----

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,
                        default=os.path.expanduser("~/models/Qwen3-4B"))
    parser.add_argument("--dtype", type=str, default=None,
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--mode", choices=["loop", "graph"], default="loop",
                        help="loop: do_bench (L2 flushed, recommended). "
                             "graph: do_bench_cudagraph (launch overhead removed).")
    parser.add_argument("--peak-bandwidth", type=float, default=None,
                        help="Override measured copy bandwidth in GB/s. "
                            "By default it is measured on this run.")
    parser.add_argument("--no-check", action="store_true")
    parser.add_argument("--no-show", action="store_true",
                        help="Save plots but don't try to open them.")
    parser.add_argument("--out-dir", type=str, default="./bench_ops_plots")
    args = parser.parse_args()

    hidden, inter, n_heads, n_kv_heads, head_dim, model_dtype, eps = \
        load_dims(args.model)
    dtype = getattr(torch, args.dtype) if args.dtype else model_dtype
    if dtype is None:
        dtype = torch.bfloat16
    dims = (hidden, inter, n_heads, n_kv_heads, head_dim, eps)

    print(f"model={args.model}")
    print(f"hidden={hidden} intermediate={inter} heads={n_heads}/{n_kv_heads} "
          f"head_dim={head_dim} eps={eps} dtype={dtype}")
    print(f"mode={args.mode}")

    if args.peak_bandwidth is None:
        print("measuring copy bandwidth baseline...")
        peak_bandwidth = _measure_peak_bandwidth(args.device)
        peak_source = "measured"
    else:
        peak_bandwidth = args.peak_bandwidth
        peak_source = "manual override"
    if peak_bandwidth <= 0:
        parser.error("--peak-bandwidth must be greater than zero")
    print(f"peak_bandwidth={peak_bandwidth:.1f} GB/s ({peak_source})")

    if not args.no_check:
        print("\n--- correctness check (T=64) ---")
        check_correctness(dims, args.device, dtype)

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"\n--- running perf_report (mode={args.mode}) ---")
    for report in build_reports(dims, args.device, dtype, args.mode):
        report.run(
            show_plots=not args.no_show,
            print_data=True,
            save_path=args.out_dir,
        )

    print("\n--- running bandwidth utilization report ---")
    for report in build_bandwidth_reports(
            dims, args.device, dtype, args.mode, peak_bandwidth):
        report.run(
            show_plots=not args.no_show,
            print_data=True,
            save_path=args.out_dir,
        )

    print(f"\nPlots and HTML saved under: {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()