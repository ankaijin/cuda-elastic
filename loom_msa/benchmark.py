#!/usr/bin/env python3
"""Repeated baseline-vs-optimized comparison for every shape and input set.

Each repeat runs the baseline and optimized binaries back to back (interleaved,
so thermal or clock drift affects both), samples GPU clocks while each run
executes, and requires the run to pass the harness's reference check. Reported
values are the median across repeats with the min-max spread.

Metrics (all from the harness, see msa_harness.cuh):
  end_to_end_us          uninstrumented graph, back-to-back replays (mean)
  attention_us           attention kernel, median over synchronized replays
  through_statistics_us  attention start -> statistics end; for the optimized
                         split context this includes merging the partials
Effective throughput uses the same algorithmic FLOP count for both variants.
"""
import argparse
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

BUILD = Path(__file__).resolve().parent / "build"
SHAPES = ["4x4", "medium", "large"]
INPUTS = ["random", "repeat"]  # random is the primary performance input
VARIANTS = ["baseline", "optimized"]
METRICS = ["end_to_end_us", "attention_us", "through_statistics_us"]


def run_once(binary, inputs):
    sampler = subprocess.Popen(
        ["nvidia-smi",
         "--query-gpu=clocks.sm,pstate,temperature.gpu,utilization.gpu",
         "--format=csv,noheader,nounits", "-lms", "100"],
        stdout=subprocess.PIPE, text=True)
    try:
        result = subprocess.run([str(binary), f"--inputs={inputs}"],
                                capture_output=True, text=True)
    finally:
        sampler.terminate()
        samples, _ = sampler.communicate()
    if result.returncode != 0 or "PASSED" not in result.stdout:
        sys.exit(f"{binary.name} --inputs={inputs} failed:\n"
                 f"{result.stdout}{result.stderr}")
    line = next(l for l in result.stdout.splitlines()
                if l.startswith("metrics:"))
    fields = dict(re.findall(r"(\w+)=(\S+)", line))
    clocks, states, temperatures = [], set(), []
    for row in samples.splitlines():
        parts = [p.strip() for p in row.split(",")]
        # Only samples taken while the GPU was busy (not process startup or
        # the CPU reference check).
        if (len(parts) == 4 and parts[0].isdigit() and parts[3].isdigit()
                and int(parts[3]) > 0):
            clocks.append(int(parts[0]))
            states.add(parts[1])
            temperatures.append(int(parts[2]))
    return {
        "shape": fields["shape"],
        "replays": int(fields["replays"]),
        "flops": float(fields["attention_flops"]),
        **{m: float(fields[m]) for m in METRICS},
        "sm_clock_mhz": statistics.median(clocks) if clocks else None,
        "pstates": sorted(states),
        "max_temperature_c": max(temperatures) if temperatures else None,
    }


def summarize(values):
    return {"median": statistics.median(values), "min": min(values),
            "max": max(values)}


def fmt_us(summary):
    median = summary["median"]
    if median >= 10000:
        return (f"{median / 1000:.3f} ms "
                f"[{summary['min'] / 1000:.3f}-{summary['max'] / 1000:.3f}]")
    return f"{median:.2f} us [{summary['min']:.2f}-{summary['max']:.2f}]"


def fmt_saved(us):
    return f"{us / 1000:.3f} ms" if abs(us) >= 10000 else f"{us:.2f} us"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--shapes", nargs="+", default=SHAPES, choices=SHAPES)
    parser.add_argument("--inputs", nargs="+", default=INPUTS, choices=INPUTS)
    parser.add_argument("--json", default=str(BUILD / "benchmark.json"))
    args = parser.parse_args()

    # Discarded warm-up run so the first measured run does not start from an
    # idle GPU.
    run_once(BUILD / "msa_optimized_medium", "random")

    results = {}
    for shape in args.shapes:
        for inputs in args.inputs:
            runs = {v: [] for v in VARIANTS}
            for repeat in range(args.repeats):
                for variant in VARIANTS:
                    binary = BUILD / f"msa_{variant}_{shape}"
                    run = run_once(binary, inputs)
                    runs[variant].append(run)
                    print(f"  {shape:6s} {inputs:6s} {variant:9s} "
                          f"repeat {repeat + 1}/{args.repeats}: "
                          f"e2e {run['end_to_end_us']:.2f} us, "
                          f"attention {run['attention_us']:.2f} us, "
                          f"SM clock {run['sm_clock_mhz']} MHz",
                          file=sys.stderr)
            base, opt = runs["baseline"][0], runs["optimized"][0]
            if (base["shape"], base["replays"]) != (opt["shape"],
                                                   opt["replays"]):
                sys.exit(f"shape/replay mismatch: {base['shape']} x "
                         f"{base['replays']} vs {opt['shape']} x "
                         f"{opt['replays']}")
            results[(shape, inputs)] = runs

    report = []
    for (shape, inputs), runs in results.items():
        summary = {v: {m: summarize([r[m] for r in runs[v]]) for m in METRICS}
                   for v in VARIANTS}
        first = runs["baseline"][0]
        flops = first["flops"]
        clocks = [r["sm_clock_mhz"] for v in VARIANTS for r in runs[v]
                  if r["sm_clock_mhz"] is not None]
        pstates = sorted({p for v in VARIANTS for r in runs[v]
                          for p in r["pstates"]})
        temperature = max(r["max_temperature_c"] or 0 for v in VARIANTS
                          for r in runs[v])
        b, o = summary["baseline"], summary["optimized"]

        def line(label, value):
            print(f"{label:44s}{value}")

        print(f"\nshape: {first['shape']} (replays={first['replays']}, "
              f"inputs={inputs}, repeats={args.repeats}; "
              f"median [min-max])")
        for metric, label in [("attention_us", "attention kernel"),
                              ("through_statistics_us",
                               "attention through statistics"),
                              ("end_to_end_us", "end-to-end replay")]:
            line(f"baseline {label}:", fmt_us(b[metric]))
            line(f"optimized {label}:", fmt_us(o[metric]))
            speedup = b[metric]["median"] / o[metric]["median"]
            saved = b[metric]["median"] - o[metric]["median"]
            line(f"{label} speedup:", f"{speedup:.2f}x")
            line(f"{label} time saved:", fmt_saved(saved))
        for metric, label in [("attention_us", "kernel"),
                              ("through_statistics_us", "through statistics")]:
            line(f"effective TFLOP/s ({label}):",
                 f"baseline {flops / (b[metric]['median'] * 1e6):.4g}, "
                 f"optimized {flops / (o[metric]['median'] * 1e6):.4g}")
        line("attention FLOPs (algorithmic):", f"{flops:.4g}")
        line("GPU while busy:",
             f"SM clock median {statistics.median(clocks):.0f} MHz "
             f"[{min(clocks):.0f}-{max(clocks):.0f}], "
             f"pstates {','.join(pstates)}, max {temperature} C")
        report.append({"shape": first["shape"], "replays": first["replays"],
                       "inputs": inputs, "repeats": args.repeats,
                       "attention_flops": flops, "summary": summary,
                       "runs": runs})

    Path(args.json).write_text(json.dumps(report, indent=2))
    print(f"\nraw results: {args.json}")


if __name__ == "__main__":
    main()
