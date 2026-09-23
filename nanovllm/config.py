import os
import warnings
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 40960
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    # None distinguishes an omitted mode from an explicit conflicting alias.
    # The resolved default, after __post_init__, is "chunked".
    scheduler_mode: str | None = None
    chunked_prefill: bool | None = None
    use_triton: bool = False
    use_triton_hidden_rmsnorm: bool = False

    def __post_init__(self):
        if self.scheduler_mode not in (None, "legacy", "chunked"):
            raise ValueError("scheduler_mode must be 'legacy' or 'chunked'")
        if self.chunked_prefill is not None:
            alias_mode = "chunked" if self.chunked_prefill else "legacy"
            if self.scheduler_mode is not None and self.scheduler_mode != alias_mode:
                raise ValueError("scheduler_mode conflicts with deprecated chunked_prefill")
            warnings.warn(
                "chunked_prefill is deprecated; use scheduler_mode='legacy' or 'chunked'. "
                "Both modes now enforce max_num_batched_tokens.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.scheduler_mode = alias_mode
        self.scheduler_mode = self.scheduler_mode or "chunked"
        if self.max_num_batched_tokens <= 0 or self.max_num_seqs <= 0:
            raise ValueError("max_num_batched_tokens and max_num_seqs must be positive")
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        # assert self.max_num_batched_tokens >= self.max_model_len
