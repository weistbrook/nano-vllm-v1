"""CPU structural checks of the real scheduler and block allocator.

The GPU-heavy package initializer is bypassed; scheduling, allocation and
hashing are real. These tests do not validate GPU kernels or token content.
"""

from collections import Counter
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest


@pytest.fixture(scope="module")
def core():
    root = Path(__file__).resolve().parents[1]
    modules = {}
    for name, path in (("nanovllm", root / "nanovllm"), ("nanovllm.engine", root / "nanovllm/engine")):
        module = ModuleType(name)
        module.__path__ = [str(path)]
        modules[name] = module
    transformers = ModuleType("transformers")
    class AutoConfig:
        @staticmethod
        def from_pretrained(_):
            return SimpleNamespace(max_position_embeddings=32768)

    transformers.AutoConfig = AutoConfig
    modules["transformers"] = transformers
    pytest.importorskip("xxhash", reason="CPU scheduler checks require the real xxhash dependency")
    with patch.dict(sys.modules, modules):
        loaded = {}
        for name in ("config", "sampling_params", "engine.sequence", "engine.block_manager", "engine.scheduler"):
            module_name = f"nanovllm.{name}"
            spec = importlib.util.spec_from_file_location(module_name, root / "nanovllm" / (name.replace(".", "/") + ".py"))
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            loaded[name.rsplit(".", 1)[-1]] = module
        yield SimpleNamespace(**loaded)


@pytest.fixture
def factory(core):
    original_block_size = core.sequence.Sequence.block_size
    core.sequence.Sequence.block_size = 4

    def make(mode="chunked", budget=8, blocks=128, max_seqs=16, max_model_len=128, block_size=4):
        core.sequence.Sequence.block_size = block_size
        return core.scheduler.Scheduler(SimpleNamespace(
            scheduler_mode=mode, max_model_len=max_model_len,
            max_num_seqs=max_seqs, max_num_batched_tokens=budget,
            eos=-1, num_kvcache_blocks=blocks, kvcache_block_size=block_size,
        ))

    yield make
    core.sequence.Sequence.block_size = original_block_size


def sequence(core, prompt_len, salt=0, output_len=6):
    return core.sequence.Sequence(
        list(range(salt * 1000 + 1, salt * 1000 + prompt_len + 1)),
        core.sampling_params.SamplingParams(max_tokens=output_len, ignore_eos=True),
    )


def running(core, scheduler, prompt_len, cached, salt, completions=0, output_len=6):
    seq = sequence(core, prompt_len, salt, output_len)
    for _ in range(completions):
        seq.append_token(salt * 1000 + 999)
    seq.num_new_tokens = cached
    scheduler.block_manager.allocate(seq)
    previous_cached = seq.num_cached_tokens
    seq.num_cached_tokens += seq.num_new_tokens
    seq.num_new_tokens = 0
    scheduler.block_manager.cache_full_blocks(seq, previous_cached)
    seq.status = core.sequence.SequenceStatus.RUNNING
    scheduler.running.append(seq)
    return seq


def assert_blocks_consistent(scheduler):
    manager = scheduler.block_manager
    references = Counter(block for seq in scheduler.running for block in seq.block_table)
    assert references.keys() == manager.used_block_ids
    assert set(manager.free_block_ids).isdisjoint(manager.used_block_ids)
    assert len(manager.free_block_ids) + len(manager.used_block_ids) == len(manager.blocks)
    assert len(set(manager.free_block_ids)) == len(manager.free_block_ids)
    for block in manager.blocks:
        assert block.ref_count == references[block.block_id]
        if block.hash != -1:
            assert len(block.token_ids) == manager.block_size
    for block_hash, block_id in manager.hash_to_block_id.items():
        assert manager.blocks[block_id].hash == block_hash


def finish_step(scheduler, scheduled):
    # Exactly the runner's logits criterion, with deterministic synthetic tokens.
    indices = [i for i, seq in enumerate(scheduled) if seq.num_context_tokens == len(seq)]
    before = {seq: seq.num_completion_tokens for seq in scheduled}
    assert len(scheduled) == len(set(scheduled))
    assert sum(seq.num_new_tokens for seq in scheduled) <= scheduler.max_num_batched_tokens
    scheduler.postprocess(scheduled, [90001] * len(indices), indices)
    for i, seq in enumerate(scheduled):
        assert seq.num_completion_tokens == before[seq] + (i in indices)
        assert seq.num_new_tokens == 0
    assert_blocks_consistent(scheduler)


def drain(scheduler, limit=500):
    for _ in range(limit):
        if scheduler.is_finished():
            assert not scheduler.block_manager.used_block_ids
            assert_blocks_consistent(scheduler)
            return
        scheduled = scheduler.schedule()
        if scheduler.scheduler_mode == "legacy":
            assert len({scheduler.is_decode_ready(seq) for seq in scheduled}) == 1
        finish_step(scheduler, scheduled)
    pytest.fail("Scheduler made no bounded progress")


def test_config_default_and_deprecated_alias(core, tmp_path):
    assert core.config.Config(str(tmp_path)).scheduler_mode == "chunked"
    for alias, mode in ((False, "legacy"), (True, "chunked")):
        with pytest.warns(DeprecationWarning):
            assert core.config.Config(str(tmp_path), chunked_prefill=alias).scheduler_mode == mode
    with pytest.raises(ValueError, match="conflicts"):
        core.config.Config(str(tmp_path), scheduler_mode="chunked", chunked_prefill=False)
    with pytest.raises(ValueError, match="scheduler_mode"):
        core.config.Config(str(tmp_path), scheduler_mode="other")
    with pytest.raises(ValueError, match="positive"):
        core.config.Config(str(tmp_path), max_num_batched_tokens=0)


def test_decode_precedes_partial_prefill_even_at_queue_head(core, factory):
    scheduler = factory(budget=4)
    partial = running(core, scheduler, 10, 2, 1)
    first = running(core, scheduler, 2, 2, 2, completions=1)
    second = running(core, scheduler, 2, 2, 3, completions=1)
    waiting = sequence(core, 6, 4)
    scheduler.add(waiting)
    scheduled = scheduler.schedule()
    assert scheduled == [first, second, partial]
    assert [seq.num_new_tokens for seq in scheduled] == [1, 1, 2]
    assert list(scheduler.waiting) == [waiting]
    finish_step(scheduler, scheduled)
    drain(scheduler)


def test_decode_partial_and_new_prefill_share_remaining_budget(core, factory):
    scheduler = factory(budget=6)
    partial = running(core, scheduler, 3, 2, 1)
    decode = running(core, scheduler, 2, 2, 2, completions=1)
    waiting = sequence(core, 8, 3)
    scheduler.add(waiting)
    scheduled = scheduler.schedule()
    assert scheduled == [decode, partial, waiting]
    assert [seq.num_new_tokens for seq in scheduled] == [1, 1, 4]
    finish_step(scheduler, scheduled)
    drain(scheduler)


@pytest.mark.parametrize("mode", ["legacy", "chunked"])
def test_long_prompt_obeys_same_hard_budget(core, factory, mode):
    scheduler = factory(mode=mode, budget=1024, blocks=64, max_model_len=9000, block_size=256)
    decode = running(core, scheduler, 8, 8, 1, completions=1, output_len=20)
    long = sequence(core, 8192, 2, output_len=2)
    scheduler.add(long)
    prefill_steps = 0
    while long.num_completion_tokens == 0:
        scheduled = scheduler.schedule()
        assert sum(seq.num_new_tokens for seq in scheduled) <= 1024
        if mode == "legacy":
            assert scheduled == [long]
            assert decode.num_completion_tokens == 1
        else:
            assert scheduled[0] is decode
            assert decode.num_new_tokens == 1
            assert long in scheduled
        finish_step(scheduler, scheduled)
        prefill_steps += 1
    assert prefill_steps == (8 if mode == "legacy" else 9)
    drain(scheduler)


def test_recomputed_request_with_outputs_is_prefill_until_cache_catches_up(core, factory):
    scheduler = factory(budget=3)
    recompute = running(core, scheduler, 8, 2, 1, completions=2)
    decode = running(core, scheduler, 2, 2, 2, completions=1)
    scheduled = scheduler.schedule()
    assert scheduled == [decode, recompute]
    assert [seq.num_new_tokens for seq in scheduled] == [1, 2]
    finish_step(scheduler, scheduled)
    assert recompute.num_completion_tokens == 2
    drain(scheduler)


def test_scheduled_decode_never_preempted_by_partial_prefill(core, factory):
    scheduler = factory(budget=8, blocks=3)
    partial = running(core, scheduler, 8, 4, 1, output_len=2)
    decode = running(core, scheduler, 3, 4, 2, completions=2, output_len=3)
    # Decode needs a new block; the partial prefill can no longer extend after
    # this reservation, and must never reclaim that already scheduled block.
    scheduled = scheduler.schedule()
    assert scheduled == [decode]
    assert list(scheduler.waiting) == [partial]
    assert decode.block_table
    finish_step(scheduler, scheduled)
    drain(scheduler)


def test_decode_preempts_unscheduled_prefill_then_recompute_completes(core, factory):
    scheduler = factory(budget=4, blocks=3)
    decode = running(core, scheduler, 3, 4, 1, completions=2, output_len=3)
    partial = running(core, scheduler, 10, 8, 2, output_len=2)
    scheduled = scheduler.schedule()
    assert scheduled == [decode]
    assert list(scheduler.waiting) == [partial]
    assert partial.num_cached_tokens == 0 and partial.block_table == []
    finish_step(scheduler, scheduled)
    drain(scheduler)


def test_prefix_hashes_published_only_after_compute_and_full_block(core, factory):
    scheduler = factory(budget=3)
    seq = sequence(core, 9, 1, output_len=1)
    scheduler.add(seq)
    scheduled = scheduler.schedule()
    assert scheduler.block_manager.hash_to_block_id == {}
    finish_step(scheduler, scheduled)
    assert scheduler.block_manager.hash_to_block_id == {}
    scheduled = scheduler.schedule()
    assert scheduler.block_manager.hash_to_block_id == {}
    finish_step(scheduler, scheduled)
    assert len(scheduler.block_manager.hash_to_block_id) == 1
    first_block = seq.block_table[0]
    assert scheduler.block_manager.blocks[first_block].token_ids == seq[:4]
    # A second request can reuse the completed prefix, not the partial tail.
    repeated = sequence(core, 9, 1, output_len=1)
    assert scheduler.block_manager.get_token_layout(repeated) == (4, 0, 5)
    drain(scheduler)
    assert scheduler.block_manager.get_token_layout(repeated) == (0, 8, 1)
    scheduler.add(repeated)
    scheduled = scheduler.schedule()
    assert repeated.num_cached_tokens == 8
    finish_step(scheduler, scheduled)
    assert scheduler.is_finished()
    assert not scheduler.block_manager.used_block_ids


def test_live_shared_prefix_references_are_released(core, factory):
    scheduler = factory(budget=8)
    first = running(core, scheduler, 9, 8, 1, output_len=3)
    repeated = sequence(core, 9, 1, output_len=2)
    scheduler.add(repeated)
    scheduled = scheduler.schedule()
    assert repeated.num_cached_tokens == 8
    assert repeated.block_table[:2] == first.block_table[:2]
    assert all(scheduler.block_manager.blocks[i].ref_count == 2 for i in first.block_table[:2])
    finish_step(scheduler, scheduled)
    drain(scheduler)


def test_legacy_decodes_if_waiting_prefill_blocked_by_sequence_limit(core, factory):
    scheduler = factory(mode="legacy", max_seqs=1)
    decode = running(core, scheduler, 2, 2, 1, completions=1, output_len=2)
    waiting = sequence(core, 8, 2, output_len=2)
    scheduler.add(waiting)
    scheduled = scheduler.schedule()
    assert scheduled == [decode]
    finish_step(scheduler, scheduled)
    drain(scheduler)


def test_insufficient_kv_fails_clearly_instead_of_hanging(core, factory):
    scheduler = factory(budget=8, blocks=1)
    scheduler.add(sequence(core, 8, 1))
    with pytest.raises(RuntimeError, match="cannot make progress"):
        scheduler.schedule()
