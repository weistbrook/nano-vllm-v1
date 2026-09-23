"""CPU-only checks for token observation and benchmark accounting."""

import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from benchmarks.common import RequestMetrics, TokenTracker, phase_itls, stats, summarize_requests, write_json
from serving_bench import build_parser, format_ms, make_workload, validate_args, validate_workload


class TokenMeasurementTests(unittest.TestCase):
    def sequence(self, identifier=0, max_tokens=2):
        return SimpleNamespace(seq_id=identifier, num_prompt_tokens=64, num_completion_tokens=0,
                               num_cached_tokens=0, max_tokens=max_tokens, is_finished=False)

    def test_partial_prefill_not_recorded_and_final_token_retained(self):
        sequence = self.sequence()
        tracker = TokenTracker()
        metric = tracker.register(sequence, submission_time=1.0)
        scheduler = SimpleNamespace(running=[sequence])

        def partial():
            sequence.num_cached_tokens = 32
            return [], 0

        engine = SimpleNamespace(scheduler=scheduler, step=partial)
        with patch("benchmarks.common.perf_counter", return_value=2.0):
            tracker.step(engine)
        self.assertEqual(metric.token_timestamps, [])

        def first():
            sequence.num_cached_tokens = 64
            sequence.num_completion_tokens = 1
            return [], 0

        engine.step = first
        with patch("benchmarks.common.perf_counter", return_value=3.0):
            tracker.step(engine)
        self.assertEqual(metric.ttft, 2.0)

        def finish():
            sequence.num_completion_tokens = 2
            sequence.is_finished = True
            scheduler.running.clear()
            return [(sequence.seq_id, [10, 11])], 66

        engine.step = finish
        with patch("benchmarks.common.perf_counter", return_value=5.0):
            tracker.step(engine)
        self.assertEqual(metric.token_timestamps, [3.0, 5.0])
        self.assertEqual(metric.completion_time, 5.0)
        self.assertEqual(metric.itls, [2.0])
        self.assertEqual(metric.tpot, 2.0)
        self.assertEqual(metric.latency, 4.0)
        self.assertEqual(scheduler.running, [])
        self.assertEqual(tracker.export(1.0)[0]["token_timestamps"], [2.0, 4.0])

    def test_one_output_token_has_ttft_and_no_itl_or_tpot(self):
        sequence = self.sequence(max_tokens=1)
        tracker = TokenTracker()
        metric = tracker.register(sequence, submission_time=10.0)

        def finish():
            sequence.num_completion_tokens = 1
            sequence.is_finished = True
            return [(sequence.seq_id, [17])], 65

        with patch("benchmarks.common.perf_counter", return_value=12.0):
            tracker.step(SimpleNamespace(step=finish))
        self.assertEqual(metric.ttft, 2.0)
        self.assertEqual(metric.latency, 2.0)
        self.assertEqual(metric.itls, [])
        self.assertTrue(math.isnan(metric.tpot))
        summary = summarize_requests([metric])
        self.assertEqual(summary["itl_count"], 0)
        self.assertIsNone(summary["itl_p99"])
        self.assertIsNone(summary["tpot_mean"])
        self.assertEqual(format_ms(summary["tpot_mean"]), "n/a")

    def test_duplicate_append_is_rejected(self):
        sequence = self.sequence()
        tracker = TokenTracker()
        tracker.register(sequence, submission_time=0)

        def bad_step():
            sequence.num_completion_tokens += 2
            return [], 0

        with self.assertRaisesRegex(AssertionError, "completion count changed by 2"):
            tracker.step(SimpleNamespace(step=bad_step))

    def test_finished_sequences_do_not_get_duplicate_timestamps(self):
        sequence = self.sequence(max_tokens=1)
        tracker = TokenTracker()
        metric = tracker.register(sequence, submission_time=0)

        def finish():
            sequence.num_completion_tokens = 1
            sequence.is_finished = True
            return [], 0

        with patch("benchmarks.common.perf_counter", side_effect=[1.0, 2.0]):
            tracker.step(SimpleNamespace(step=finish))
            tracker.step(SimpleNamespace(step=lambda: ([], 0)))
        self.assertEqual(metric.token_timestamps, [1.0])
        self.assertEqual(metric.completion_time, 1.0)

    def test_legacy_stall_ending_after_prefill_remains_interference(self):
        metric = RequestMetrics("decode", 128, 0, token_timestamps=[0, 1, 2, 10, 11])
        phases = phase_itls([metric], arrival=2, prefill_completion=9)
        self.assertEqual(phases, {"before": [1, 1], "interference": [8], "after": [1]})
        self.assertEqual(stats(phases["interference"])["p99"], 8)

    def test_phase_boundary_intervals_are_never_split_or_duplicated(self):
        metric = RequestMetrics("decode", 128, 0, token_timestamps=[0, 1, 2, 3, 4])
        phases = phase_itls([metric], arrival=1, prefill_completion=3)
        self.assertEqual(phases, {"before": [1], "interference": [1, 1], "after": [1]})
        self.assertEqual(sum(map(len, phases.values())), len(metric.itls))
        with self.assertRaises(ValueError):
            phase_itls([metric], arrival=4, prefill_completion=1)

    def test_json_uses_null_for_missing_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            write_json(path, {"stats": stats([]), "tpot": math.nan})
            result = json.loads(path.read_text(encoding="utf-8"))
        self.assertIsNone(result["stats"]["p99"])
        self.assertIsNone(result["tpot"])


class OnlineWorkloadTests(unittest.TestCase):
    def args(self, *extra):
        return build_parser().parse_args(["--model", "unused-model", "--num-requests", "20", *extra])

    def test_mixed_workload_and_poisson_arrivals_are_reproducible(self):
        args = self.args("--request-rate", "2.5", "--short-request-ratio", "0.5")
        first = make_workload(args, 101)
        # Neither global RNG state nor the scheduler mode changes the saved inputs.
        np.random.seed(1234)
        args.scheduler_mode = "legacy"
        second = make_workload(args, 101)
        self.assertEqual(first, second)
        expected = np.cumsum(np.random.RandomState(args.seed).exponential(1 / 2.5, args.num_requests))
        self.assertEqual([request["arrival_time"] for request in first["requests"]], expected.tolist())
        self.assertEqual({request["group"] for request in first["requests"]}, {"short", "long"})
        for request in first["requests"]:
            group = request["group"]
            self.assertLessEqual(getattr(args, f"{group}_input_min"), len(request["prompt_token_ids"]))
            self.assertLessEqual(len(request["prompt_token_ids"]), getattr(args, f"{group}_input_max"))
            self.assertTrue(all(0 <= token < 101 for token in request["prompt_token_ids"]))
            self.assertEqual(request["max_tokens"], 128)
        validate_workload(first, vocab_size=101, max_model_len=8320)

    def test_uniform_compatibility_mode_and_single_output(self):
        args = self.args("--random-input-len", "5", "--random-output-len", "1")
        workload = make_workload(args, 32)
        for request in workload["requests"]:
            self.assertEqual(request["group"], "uniform")
            self.assertTrue(1 <= len(request["prompt_token_ids"]) <= 5)
            self.assertEqual(request["max_tokens"], 1)
        validate_workload(workload, vocab_size=32, max_model_len=6)

    def test_invalid_rates_and_context_are_rejected(self):
        for rate in ("0", "-1", "nan", "inf"):
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                validate_args(self.args("--request-rate", rate))
        workload = make_workload(self.args("--random-input-len", "5"), 32)
        with self.assertRaisesRegex(ValueError, "max_model_len"):
            validate_workload(workload, vocab_size=32, max_model_len=2)

    def test_default_controlled_settings(self):
        args = self.args()
        self.assertTrue(args.enforce_eager)
        self.assertFalse(args.use_triton)
        self.assertEqual(args.short_request_ratio, 0.9)
        self.assertIsNone(args.random_input_len)


if __name__ == "__main__":
    unittest.main()
