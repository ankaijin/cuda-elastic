#!/usr/bin/env python3
"""Repeated baseline / optimized / SOTA comparison for every shape and input.

Each repeat runs the baseline, optimized and SOTA binaries back to back
(interleaved, so thermal or clock drift affects all of them), samples GPU
clocks while each run executes, and requires the run to pass the harness's
reference check. Reported values are the median across repeats with the
min-max spread.

Metrics (all from the harness, see msa_harness.cuh):
  end_to_end_us          uninstrumented graph, back-to-back replays (mean)
  attention_us           attention kernel, median over synchronized replays
  through_statistics_us  attention start -> statistics end; for the optimized
                         split context this includes merging the partials
  The SOTA variant (msa_sota.cu) fuses the merge and statistics into its
  attention kernel(s) (a separate merge kernel at the medium shape), and its
  attention stage spans all of its attention launches, which overlap the
  chunked imu copy; see msa_sota.cu. It runs at its default rotation over
  3 input sets (msa_harness.cuh --rotate); baseline and optimized use 1.
Effective throughput uses the same algorithmic FLOP count for every variant.

--launch graph direct also runs every variant with --launch=direct (the same
work enqueued without a CUDA graph). Graph runs keep their variant name as the
key in benchmark.json; direct runs are stored as "<variant>_direct".
"""
import argparse
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Binaries live in <repo>/<gpu>/loom_msa/build (see gpu.sh).
GPU = subprocess.run([str(HERE.parent / "gpu.sh"), "name"], check=True,
                     capture_output=True, text=True).stdout.strip()
BUILD = HERE.parent / GPU / "loom_msa" / "build"
SHAPES = ["4x4", "medium", "large"]
INPUTS = ["random", "repeat"]  # random is the primary performance input
VARIANTS = ["baseline", "optimized", "sota"]
METRICS = ["end_to_end_us", "attention_us", "through_statistics_us"]
LAUNCHES = ["graph", "direct"]


def key(variant, launch):
    return variant if launch == "graph" else f"{variant}_{launch}"


def run_once(binary, inputs, launch="graph"):
    sampler = subprocess.Popen(
        ["nvidia-smi",
         "--query-gpu=clocks.sm,pstate,temperature.gpu,utilization.gpu",
         "--format=csv,noheader,nounits", "-lms", "100"],
        stdout=subprocess.PIPE, text=True)
    try:
        result = subprocess.run([str(binary), f"--inputs={inputs}",
                                 f"--launch={launch}"],
                                capture_output=True, text=True)
    finally:
        sampler.terminate()
        samples, _ = sampler.communicate()
    if result.returncode != 0 or "PASSED" not in result.stdout:
        sys.exit(f"{binary.name} --inputs={inputs} --launch={launch} failed:\n"
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
    parser.add_argument("--launch", nargs="+", default=["graph"],
                        choices=LAUNCHES)
    parser.add_argument("--json", default=str(BUILD / "benchmark.json"))
    args = parser.parse_args()

    # Discarded warm-up run so the first measured run does not start from an
    # idle GPU.
    run_once(BUILD / "msa_optimized_medium", "random")

    configs = [(v, l) for l in args.launch for v in VARIANTS]
    keys = [key(v, l) for v, l in configs]
    results = {}
    for shape in args.shapes:
        for inputs in args.inputs:
            runs = {k: [] for k in keys}
            for repeat in range(args.repeats):
                for variant, launch in configs:
                    binary = BUILD / f"msa_{variant}_{shape}"
                    run = run_once(binary, inputs, launch)
                    runs[key(variant, launch)].append(run)
                    print(f"  {shape:6s} {inputs:6s} {key(variant, launch):16s} "
                          f"repeat {repeat + 1}/{args.repeats}: "
                          f"e2e {run['end_to_end_us']:.2f} us, "
                          f"attention {run['attention_us']:.2f} us, "
                          f"SM clock {run['sm_clock_mhz']} MHz",
                          file=sys.stderr)
            first = runs[keys[0]][0]
            for k in keys:
                other = runs[k][0]
                if (first["shape"], first["replays"]) != (other["shape"],
                                                         other["replays"]):
                    sys.exit(f"shape/replay mismatch: {first['shape']} x "
                             f"{first['replays']} vs {other['shape']} x "
                             f"{other['replays']}")
            results[(shape, inputs)] = runs

    report = []
    for (shape, inputs), runs in results.items():
        summary = {k: {m: summarize([r[m] for r in runs[k]]) for m in METRICS}
                   for k in runs}
        first = next(iter(runs.values()))[0]
        flops = first["flops"]
        clocks = [r["sm_clock_mhz"] for k in runs for r in runs[k]
                  if r["sm_clock_mhz"] is not None]
        pstates = sorted({p for k in runs for r in runs[k]
                          for p in r["pstates"]})
        temperature = max(r["max_temperature_c"] or 0 for k in runs
                          for r in runs[k])

        def line(label, value):
            print(f"{label:44s}{value}")

        print(f"\nshape: {first['shape']} (replays={first['replays']}, "
              f"inputs={inputs}, repeats={args.repeats}; "
              f"median [min-max])")
        for launch in args.launch:
            b = summary[key("baseline", launch)]
            o = summary[key("optimized", launch)]
            t = summary[key("sota", launch)]
            prefix = "" if launch == "graph" else f"{launch} "
            for metric, label in [("attention_us", "attention kernel"),
                                  ("through_statistics_us",
                                   "attention through statistics"),
                                  ("end_to_end_us", "end-to-end replay")]:
                line(f"{prefix}baseline {label}:", fmt_us(b[metric]))
                line(f"{prefix}optimized {label}:", fmt_us(o[metric]))
                speedup = b[metric]["median"] / o[metric]["median"]
                saved = b[metric]["median"] - o[metric]["median"]
                line(f"{prefix}{label} speedup:", f"{speedup:.2f}x")
                line(f"{prefix}{label} time saved:", fmt_saved(saved))
                line(f"{prefix}sota {label}:", fmt_us(t[metric]))
                line(f"{prefix}sota {label} speedup vs optimized:",
                     f"{o[metric]['median'] / t[metric]['median']:.2f}x")
            for metric, label in [("attention_us", "kernel"),
                                  ("through_statistics_us",
                                   "through statistics")]:
                line(f"{prefix}effective TFLOP/s ({label}):",
                     f"baseline {flops / (b[metric]['median'] * 1e6):.4g}, "
                     f"optimized {flops / (o[metric]['median'] * 1e6):.4g}, "
                     f"sota {flops / (t[metric]['median'] * 1e6):.4g}")
        if "graph" in args.launch and "direct" in args.launch:
            for variant in VARIANTS:
                graph = summary[variant]["end_to_end_us"]["median"]
                direct = summary[key(variant, "direct")]["end_to_end_us"][
                    "median"]
                line(f"{variant} end-to-end graph speedup:",
                     f"{direct / graph:.2f}x (direct / graph)")
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
