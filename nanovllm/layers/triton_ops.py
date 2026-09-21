import torch
import triton
import triton.language as tl


MAX_BLOCK_SIZE = 4096


@triton.jit
def _rms_norm_fwd_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    base = row * N  # 第 row 行的起始偏移
    var = 0.0
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
        var += tl.sum(x * x, axis=0)   # 求出每一行的方差
    rstd = tl.rsqrt(var / N + eps)   # 计算每一行的标准差的倒数
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + base + cols, x * rstd * w, mask=mask)


@triton.jit
def _add_rms_norm_fwd_kernel(
    x_ptr,
    res_ptr,
    w_ptr,
    out_ptr,
    res_out_ptr,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    base = row * N
    var = 0.0
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(res_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
        s = x + r
        tl.store(res_out_ptr + base + cols, s, mask=mask)
        var += tl.sum(s * s, axis=0)
    rstd = tl.rsqrt(var / N + eps)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(res_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        s = x + r
        tl.store(out_ptr + base + cols, s * rstd * w, mask=mask)


@triton.jit
def _silu_and_mul_kernel(
    x_ptr,
    out_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    base = row * 2 * N
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
        y = tl.load(x_ptr + base + N + cols, mask=mask, other=0.0).to(tl.float32)
        out = x * tl.sigmoid(x) * y
        tl.store(out_ptr + row * N + cols, out, mask=mask)


def _launch_config(n: int) -> tuple[int, int]:
    #block_size = min(triton.next_power_of_2(n), MAX_BLOCK_SIZE)
    block_size = min(512, triton.next_power_of_2(n))
    num_warps = 8 if block_size >= MAX_BLOCK_SIZE else 4
    return block_size, num_warps


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    x = x.contiguous()
    n = x.shape[-1]
    x2d = x.view(-1, n)
    out = torch.empty_like(x2d)
    if x2d.shape[0] == 0:
        return out.view_as(x)
    block_size, num_warps = _launch_config(n)
    _rms_norm_fwd_kernel[(x2d.shape[0],)](
        x2d, weight, out, n, eps,
        BLOCK_SIZE=block_size, num_warps=num_warps,
    )
    return out.view_as(x)


def add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = x.contiguous()
    residual = residual.contiguous()
    n = x.shape[-1]
    x2d = x.view(-1, n)
    res2d = residual.view(-1, n)
    out = torch.empty_like(x2d)
    res_out = torch.empty_like(res2d)
    if x2d.shape[0] == 0:
        return out.view_as(x), res_out.view_as(residual)
    block_size, num_warps = _launch_config(n)
    _add_rms_norm_fwd_kernel[(x2d.shape[0],)](
        x2d, res2d, weight, out, res_out, n, eps,
        BLOCK_SIZE=block_size, num_warps=num_warps,
    )
    return out.view_as(x), res_out.view_as(residual)


def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    x = x.contiguous()
    n = x.shape[-1] // 2
    x2d = x.view(-1, 2 * n)
    out = torch.empty(x2d.shape[0], n, dtype=x.dtype, device=x.device)
    if x2d.shape[0] == 0:
        return out.view(*x.shape[:-1], n)
    block_size, num_warps = _launch_config(n)
    _silu_and_mul_kernel[(x2d.shape[0],)](
        x2d, out, n,
        BLOCK_SIZE=block_size, num_warps=num_warps,
    )
    return out.view(*x.shape[:-1], n)
