"""Plot measured scheduler results (all durations in source files are seconds).

python benchmarks/plot_results.py --kind interference --input results/interference
python benchmarks/plot_results.py --kind sweep --input results/long_prompt_sweep.csv
python benchmarks/plot_results.py --kind ablation --input results/token_budget_ablation.csv
"""

import argparse
import csv
import json
import math
import statistics
import warnings
from collections import defaultdict
from pathlib import Path


def _pyplot():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("Plotting requires matplotlib; install the benchmark dependencies first.") from error
    return plt


def _number(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _valid_window(row):
    return row.get("interference_window_valid", True) not in (False, "False", "false", "0", 0)


def _save(figure, output_dir, filename):
    path = Path(output_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    return path


def _labels(results):
    counts = defaultdict(int)
    for result in results:
        counts[result["summary"]["scheduler_mode"]] += 1
    labels = []
    for result in results:
        row = result["summary"]
        mode = row["scheduler_mode"]
        labels.append(mode.capitalize() + (f" run {row.get('run_id', '?')}" if counts[mode] > 1 else ""))
    return labels


def plot_interference(results, output_dir):
    """Maximum short-request ITL at each token-completion step, aligned at arrival.

    Every request's intervals contribute to the underlying summary distribution.
    The maximum per step keeps the timeline legible without hiding a stalled request.
    """
    if not results:
        raise ValueError("No measured interference JSON results were supplied")
    plt = _pyplot()
    labels = _labels(results)
    figure, axis = plt.subplots(figsize=(10, 5))
    has_observations = False
    for result, label in zip(results, labels):
        arrival = _number(result.get("long_arrival_time"))
        completion = _number(result.get("long_prefill_completion_time"))
        if arrival is None or completion is None:
            warnings.warn(f"{label}: missing long-prompt arrival / completion; timeline omitted")
            continue
        observations = defaultdict(list)
        for request in result.get("requests", []):
            if request.get("group") != "short":
                continue
            timestamps = request.get("token_timestamps", [])
            for start, end in zip(timestamps, timestamps[1:]):
                observations[end - arrival].append((end - start) * 1000)
        if not observations:
            warnings.warn(f"{label}: no short-request ITL observations")
            continue
        times = sorted(observations)
        axis.plot(times, [max(observations[time]) for time in times], label=label, linewidth=1)
        axis.axvline(completion - arrival, linestyle="--", linewidth=1,
                     label=f"{label} prefill complete")
        has_observations = True
    if not has_observations:
        plt.close(figure)
        raise ValueError("No usable token timestamps; no timeline was generated")
    axis.axvline(0, linestyle=":", label="Long prompt arrival")
    axis.set(xlabel="Time relative to long prompt arrival (s)", ylabel="Decode ITL (ms)",
             title="Maximum short-request ITL per token-completion step")
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize="small")
    paths = [_save(figure, output_dir, "interference_timeline.png")]
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 5))
    percentiles = ("p50", "p95", "p99", "max")
    width = 0.8 / len(results)
    observations = 0
    for index, (result, label) in enumerate(zip(results, labels)):
        row = result["summary"]
        values = [_number(row.get(f"interference_itl_{name}")) for name in percentiles]
        if not _valid_window(row):
            warnings.warn(f"{label}: interference window is incomplete; distribution omitted")
            continue
        positions, heights = [], []
        for tick, value in enumerate(values):
            if value is not None:
                positions.append(tick - 0.4 + width / 2 + index * width)
                heights.append(value * 1000)
        if heights:
            axis.bar(positions, heights, width=width, label=label)
            observations += len(heights)
        else:
            warnings.warn(f"{label}: missing interference-window ITL percentiles")
    if observations:
        axis.legend(fontsize="small")
    else:
        axis.text(0.5, 0.5, "No valid interference-window measurements", ha="center", transform=axis.transAxes)
    axis.set_xticks(range(len(percentiles)), ["P50", "P95", "P99", "Max"])
    axis.set(ylabel="Decode ITL (ms)", title="Intervals overlapping the long-prompt prefill window")
    axis.grid(True, axis="y", alpha=0.25)
    paths.append(_save(figure, output_dir, "interference_distribution.png"))
    plt.close(figure)
    return paths


def _plot_curve(rows, x_key, y_key, xlabel, ylabel, title, output_dir, filename, scale=1.0):
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(8, 5))
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if y_key.startswith("interference_") and not _valid_window(row):
            warnings.warn("Omitting an invalid interference window from the sweep plot")
            continue
        x, y = _number(row.get(x_key)), _number(row.get(y_key))
        if x is not None and y is not None:
            groups[row.get("scheduler_mode", "chunked")][x].append(y * scale)
    if not groups:
        plt.close(figure)
        raise ValueError(f"No finite {x_key} / {y_key} measurements; cannot produce {filename}")
    for mode, values in sorted(groups.items()):
        xs = sorted(values)
        means = [statistics.mean(values[x]) for x in xs]
        errors = [statistics.stdev(values[x]) if len(values[x]) > 1 else 0 for x in xs]
        counts = sorted({len(values[x]) for x in xs})
        count_label = str(counts[0]) if len(counts) == 1 else f"{counts[0]}–{counts[-1]}"
        axis.errorbar(xs, means, yerr=errors, marker="o", capsize=3,
                      label=f"{mode.capitalize()} (n={count_label})")
    axis.set(xlabel=xlabel, ylabel=ylabel, title=title + "\nMean across repeats; error bars = sample standard deviation")
    axis.set_xticks(sorted({x for values in groups.values() for x in values}))
    axis.grid(True, alpha=0.25)
    axis.legend()
    path = _save(figure, output_dir, filename)
    plt.close(figure)
    return path


def plot_sweep(rows, output_dir):
    return [
        _plot_curve(rows, "long_prompt_len", "interference_itl_p99", "Long prompt length (tokens)",
                    "Interference-window P99 decode ITL (ms)", "Long prompt length vs decode interference",
                    output_dir, "long_prompt_sweep_itl.png", scale=1000),
        _plot_curve(rows, "long_prompt_len", "long_ttft", "Long prompt length (tokens)", "Long request TTFT (s)",
                    "Long prompt length vs long request TTFT", output_dir, "long_prompt_sweep_ttft.png"),
    ]


def plot_ablation(rows, output_dir):
    if any(row.get("scheduler_mode", "chunked") != "chunked" for row in rows):
        raise ValueError("Token-budget ablation expects chunked scheduler results only")
    return [
        _plot_curve(rows, "token_budget", "decode_itl_p99", "Token budget (tokens / step)", "P99 decode ITL (ms)",
                    "Token budget vs decode latency", output_dir, "token_budget_itl.png", scale=1000),
        _plot_curve(rows, "token_budget", "long_ttft", "Token budget (tokens / step)", "Long request TTFT (s)",
                    "Token budget vs long request TTFT", output_dir, "token_budget_ttft.png"),
        _plot_curve(rows, "token_budget", "throughput", "Token budget (tokens / step)", "Total throughput (tokens / s)",
                    "Token budget vs throughput", output_dir, "token_budget_throughput.png"),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", required=True, choices=("interference", "sweep", "ablation"))
    parser.add_argument("--input", type=Path, nargs="+", required=True, help="Measured JSON / CSV files, or a result directory.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/figures"))
    args = parser.parse_args()
    files = []
    for path in args.input:
        if path.is_dir():
            files.extend(sorted(path.rglob("*.json" if args.kind == "interference" else "*.csv")))
        elif path.is_file():
            files.append(path)
        else:
            parser.error(f"Input does not exist: {path}")
    results, rows = [], []
    for path in dict.fromkeys(files):
        if path.suffix.lower() == ".json":
            result = json.loads(path.read_text(encoding="utf-8"))
            if "summary" in result:
                results.append(result)
                rows.append(result["summary"])
        elif path.suffix.lower() == ".csv":
            with path.open(encoding="utf-8", newline="") as stream:
                rows.extend(csv.DictReader(stream))
    try:
        if args.kind == "interference":
            paths = plot_interference(results, args.output_dir)
        else:
            paths = (plot_sweep if args.kind == "sweep" else plot_ablation)(rows, args.output_dir)
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))
    for path in paths:
        print(path.resolve())


if __name__ == "__main__":
    main()
