"""Inject a long prompt into stable decode; compare identical saved workloads."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.common import (
    add_experiment_args, interference_worker, launch_worker, make_interference_workload,
    prepare_config, verify_comparable, write_csv, write_json,
)


def main():
    if "--worker-job" in sys.argv:
        parser = argparse.ArgumentParser(description="Internal isolated benchmark worker")
        parser.add_argument("--worker-job", type=Path, required=True)
        interference_worker(parser.parse_args().worker_job)
        return
    parser = argparse.ArgumentParser(description=__doc__)
    add_experiment_args(parser)
    parser.add_argument("--scheduler-mode", choices=("legacy", "chunked", "both"), default="both")
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    try:
        if args.repeats < 1:
            raise ValueError("repeats must be positive")
        config, vocab_size = prepare_config(args)
        directory = args.output_dir.resolve() / "long_prompt_interference"
        results = []
        for run_id in range(args.repeats):
            workload = make_interference_workload(config, vocab_size, args.seed + run_id)
            workload_path = directory / "workloads" / f"run_{run_id:03d}.json"
            write_json(workload_path, workload)
            modes = ["legacy", "chunked"] if args.scheduler_mode == "both" else [args.scheduler_mode]
            if run_id % 2:
                modes.reverse()
            for mode in modes:
                result_path = directory / f"{mode}_run_{run_id:03d}.json"
                results.append(launch_worker(config, mode, run_id, workload_path, result_path))
        verify_comparable(results)
        csv_path = args.output_dir.resolve() / "long_prompt_interference.csv"
        write_csv(csv_path, [r["summary"] for r in results])
        if not args.no_plots:
            from benchmarks.plot_results import plot_interference
            plot_interference(results, directory)
        for result in results:
            if not result["summary"]["interference_window_valid"]:
                print("NOTE: a run lacks a complete stable-decode interference window; inspect survivor/sample counts.")
        print(f"Measurements (seconds), raw timestamps and workload: {directory}\nSummary: {csv_path}")
    except (ValueError, RuntimeError) as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    main()
