"""Compare interference P99 and long TTFT over a controlled prompt-length sweep."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.common import (
    add_experiment_args, launch_worker, make_interference_workload, prepare_config,
    validate_experiment, verify_comparable, write_csv, write_json,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_experiment_args(parser)
    parser.add_argument("--prompt-lengths", type=int, nargs="+", default=[2048, 4096, 8192])
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    try:
        if args.repeats < 3:
            raise ValueError("Length sweep requires at least 3 repetitions per configuration")
        if any(length < 1 for length in args.prompt_lengths) or len(set(args.prompt_lengths)) != len(args.prompt_lengths):
            raise ValueError("prompt-lengths must contain distinct positive lengths")
        # One initialization shape/context limit for the whole length sweep.
        args.long_prompt_len = max(args.prompt_lengths)
        config, vocab_size = prepare_config(args, largest_prompt=max(args.prompt_lengths))
        directory = args.output_dir.resolve() / "long_prompt_sweep"
        results = []
        for length in args.prompt_lengths:
            variant = dict(config, long_prompt_len=length)
            validate_experiment(variant)
            for run_id in range(args.repeats):
                workload = make_interference_workload(variant, vocab_size, args.seed + run_id)
                workload_path = directory / "workloads" / f"length_{length}_run_{run_id:03d}.json"
                write_json(workload_path, workload)
                modes = ["legacy", "chunked"] if run_id % 2 == 0 else ["chunked", "legacy"]
                for mode in modes:
                    result_path = directory / f"length_{length}_{mode}_run_{run_id:03d}.json"
                    results.append(launch_worker(variant, mode, run_id, workload_path, result_path))
        verify_comparable(results)
        rows = [result["summary"] for result in results]
        csv_path = args.output_dir.resolve() / "long_prompt_sweep.csv"
        write_csv(csv_path, rows)
        if not args.no_plots:
            from benchmarks.plot_results import plot_sweep
            plot_sweep(rows, directory)
        print(f"Length sweep summary: {csv_path}\nRaw runs and workloads: {directory}")
    except (ValueError, RuntimeError) as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    main()
