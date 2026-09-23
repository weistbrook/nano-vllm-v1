"""Measure the chunked-prefill token-budget / ITL / TTFT trade-off."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.common import (
    add_experiment_args, launch_worker, make_interference_workload, prepare_config,
    validate_experiment, write_csv, write_json,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_experiment_args(parser)
    parser.add_argument("--budgets", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    try:
        if args.repeats < 1:
            raise ValueError("repeats must be positive")
        if len(set(args.budgets)) != len(args.budgets):
            raise ValueError("budgets must be distinct")
        args.max_num_batched_tokens = args.budgets[0]
        config, vocab_size = prepare_config(args)
        for budget in args.budgets:
            validate_experiment(dict(config, max_num_batched_tokens=budget))
        directory = args.output_dir.resolve() / "token_budget_ablation"
        results = []
        for run_id in range(args.repeats):
            # The same file is used by every budget in this repetition.
            workload = make_interference_workload(config, vocab_size, args.seed + run_id)
            workload_path = directory / "workloads" / f"run_{run_id:03d}.json"
            write_json(workload_path, workload)
            budgets = args.budgets if run_id % 2 == 0 else list(reversed(args.budgets))
            for budget in budgets:
                variant = dict(config, max_num_batched_tokens=budget)
                result_path = directory / f"budget_{budget}_chunked_run_{run_id:03d}.json"
                results.append(launch_worker(variant, "chunked", run_id, workload_path, result_path))
        # Context and initialization shapes stay fixed. Refuse hidden GPU/cache changes.
        signatures = {(r["metadata"]["num_kvcache_blocks"], str(r["metadata"]["gpus"])) for r in results}
        if len(signatures) != 1:
            raise RuntimeError("GPU identity or KV capacity changed across budgets; raw results are retained "
                               "but cannot be presented as a controlled budget ablation")
        rows = [result["summary"] for result in results]
        csv_path = args.output_dir.resolve() / "token_budget_ablation.csv"
        write_csv(csv_path, rows)
        if not args.no_plots:
            from benchmarks.plot_results import plot_ablation
            plot_ablation(rows, directory)
        print(f"Budget ablation summary: {csv_path}\nRaw runs, step workloads and shared inputs: {directory}")
    except (ValueError, RuntimeError) as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    main()
