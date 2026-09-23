from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:
    """Two policies sharing the same token budget, runner and KV allocator.

    Legacy is a controlled prefill-first baseline: long prefills are still
    split at the hard budget, but prefill and decode never share a step.
    Chunked reserves one token per ready decode before scheduling prefills.
    """

    def __init__(self, config: Config):
        self.scheduler_mode = config.scheduler_mode
        self.max_model_len = config.max_model_len
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        assert len(seq) <= self.max_model_len - 1, "Sequence length exceeds max_model_len"
        self.waiting.append(seq)

    @staticmethod
    def is_decode_ready(seq: Sequence) -> bool:
        # A preempted request can have output tokens but need a full recompute.
        return seq.num_completion_tokens > 0 and seq.num_cached_tokens == len(seq) - 1

    def schedule(self) -> list[Sequence]:
        scheduled: list[Sequence] = []
        protected: set[Sequence] = set()
        preempted: set[Sequence] = set()
        token_budget = self.max_num_batched_tokens
        decodes = [seq for seq in self.running if self.is_decode_ready(seq)]
        prefills = [seq for seq in self.running if not self.is_decode_ready(seq)]

        def schedule_running(candidates):
            nonlocal token_budget
            for seq in candidates:
                if token_budget <= 0:
                    break
                if seq in preempted:
                    continue
                num_new_tokens = min(
                    len(seq) - seq.num_cached_tokens,
                    token_budget,
                    self.max_model_len - 1 - seq.num_cached_tokens,
                )
                assert num_new_tokens > 0
                while not self.block_manager.can_append(seq, num_new_tokens):
                    victims = [
                        candidate for candidate in reversed(self.running)
                        if candidate not in protected and candidate is not seq
                    ]
                    if self.scheduler_mode == "chunked":
                        # Reclaim an unfinished prefill before an online decode.
                        victims.sort(key=self.is_decode_ready)
                    if not victims:
                        self.running.remove(seq)
                        self.preempt(seq)
                        preempted.add(seq)
                        break
                    victim = victims[0]
                    self.running.remove(victim)
                    self.preempt(victim)
                    preempted.add(victim)
                if seq in preempted:
                    continue
                seq.num_new_tokens = num_new_tokens
                self.block_manager.may_append(seq)
                scheduled.append(seq)
                protected.add(seq)
                token_budget -= num_new_tokens

        def schedule_waiting():
            nonlocal token_budget
            # Do not immediately readmit a request preempted by this same step.
            if preempted:
                return
            while self.waiting and token_budget > 0 and len(self.running) < self.max_num_seqs:
                seq = self.waiting[0]
                cached_used, cached_free, num_new_tokens = self.block_manager.get_token_layout(seq)
                num_new_tokens = min(num_new_tokens, token_budget)
                assert num_new_tokens > 0
                if not self.block_manager.can_allocate(cached_free + num_new_tokens):
                    break
                seq.num_new_tokens = num_new_tokens
                self.block_manager.allocate(seq)
                assert seq.num_cached_tokens == cached_used + cached_free
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
                scheduled.append(seq)
                protected.add(seq)
                token_budget -= num_new_tokens

        if self.scheduler_mode == "chunked":
            schedule_running(decodes)
            schedule_running(prefills)
            schedule_waiting()
        else:
            schedule_running(prefills)
            schedule_waiting()
            # Waiting requests can be blocked on slots/KV; let decode free them.
            if not scheduled:
                schedule_running(decodes)

        if not scheduled:
            raise RuntimeError(
                "Scheduler cannot make progress with the available KV blocks. "
                "Reduce the token budget or concurrent/context lengths, or increase KV capacity."
            )
        return scheduled


    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], seq_need_compute_logits) -> None:
        logits_indices = [int(index) for index in seq_need_compute_logits]
        assert len(token_ids) == len(logits_indices)
        assert len(logits_indices) == len(set(logits_indices)), "A sequence may emit at most one token per step"
        for seq in seqs:
            previous_cached_tokens = seq.num_cached_tokens
            seq.num_cached_tokens += seq.num_new_tokens
            seq.num_new_tokens = 0
            # KV is valid only after the runner completed this step. Publish
            # complete blocks now, including those belonging to a final token.
            self.block_manager.cache_full_blocks(seq, previous_cached_tokens)
        for seq_index, token_id in zip(logits_indices, token_ids):
            seq = seqs[seq_index]
            assert seq.num_cached_tokens == len(seq)
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or \
                seq.num_completion_tokens == seq.max_tokens or \
                    len(seq) >= self.max_model_len:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
