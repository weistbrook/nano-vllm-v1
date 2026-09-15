import torch
from torch import nn
import torch.nn.functional as F

from nanovllm.layers import triton_ops


class SiluAndMul(nn.Module):

    def __init__(self, use_triton: bool = False):
        super().__init__()
        self.use_triton = use_triton

    @torch.compile
    def forward_torch(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)
        return F.silu(x) * y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_triton:
            return triton_ops.silu_and_mul(x)
        return self.forward_torch(x)
