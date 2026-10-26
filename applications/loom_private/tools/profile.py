#!/usr/bin/env python3
"""Profile the loom_private applications with Nsight Systems and Nsight Compute.

  profile.py [APP ...]        all six by default
  profile.py --no-ncu [APP]   Nsight Systems only (where counters are admin-only)

Each application is profiled in three versions: the baseline with direct
launches, the baseline as a CUDA Graph, and the optimized variant as a CUDA
Graph (the version the report's Optimized column times). Every capture runs
one bench of `--iterations 1 --warmup 0`: setup, then two invocations at
application scope and two at compute scope.

Per version, files go to <gpu>/loom_private/<app>/profiles/ as
<stamp>-<version>.nsys-rep with its kernel and memcpy summaries
(<stamp>-<version>_cuda_gpu_kern_sum.csv, _cuda_gpu_mem_time_sum.csv), and
<stamp>-<version>.ncu-rep (or .ncu-repz) with its raw metrics exported to
<stamp>-<version>-ncu-raw.csv, so the report never needs this machine's ncu
to read them. ncu (`--set basic`) profiles the setup kernels and the first
application-scope invocation: the launch count comes from the Nsight Systems
summary, where setup kernels are those whose instance count is not a
multiple of the four invocations. The stamp is $TIMESTAMP when set
(results/record_run.py sets it to the run id).

Binaries must be built first (run_suite.py builds both variants). Each
profiled process takes the GPU lock; do not wrap this script in flock.
"""

import argparse
import csv
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys

SUITE = Path(__file__).resolve().parents[1]
REPO = SUITE.parents[1]
APPS = ["pagerank", "attention", "decoder", "anomaly", "vecadd_memory", "vecadd_product"]
VERSIONS = [("baseline-direct", "baseline", "direct"), ("baseline-graph", "baseline", "graph"),
            ("optimized-graph", "optimized", "graph")]
INVOCATIONS = 4  # --iterations 1 --warmup 0: one batch and one single, per scope
LOCK = os.environ.get("GPU_LOCK", "/tmp/cuda-elastic-gpu.lock")


def run(command, **kwargs):
    print("$ " + " ".join(map(str, command)), flush=True)
    return subprocess.run([str(c) for c in command], check=True, **kwargs)


def launch_count(kernel_summary):
    """Setup kernels plus one invocation's kernels, from an nsys kernel summary."""
    with open(kernel_summary) as stream:
        instances = [int(row["Instances"]) for row in csv.DictReader(stream)]
    return sum(n if n % INVOCATIONS else n // INVOCATIONS for n in instances)


def profile(app, folder, stamp, ncu):
    out = REPO / folder / "loom_private" / app / "profiles"
    out.mkdir(parents=True, exist_ok=True)
    for name, variant, launch in VERSIONS:
        binary = REPO / folder / "loom_private" / app / "build" / (
            app if variant == "baseline" else f"{app}_optimized")
        if not binary.is_file():
            sys.exit(f"{binary.relative_to(REPO)} is missing; build both variants first")
        prefix = out / f"{stamp}-{name}"
        bench = [binary, "--action", "bench", "--launch", launch, "--iterations", "1",
                 "--warmup", "0"]
        run(["flock", LOCK, "nsys", "profile", "--trace", "cuda", "--cuda-graph-trace", "node",
             "--force-overwrite", "true", "--output", prefix, *bench], stdout=subprocess.DEVNULL)
        run(["nsys", "stats", "--report", "cuda_gpu_kern_sum,cuda_gpu_mem_time_sum",
             "--format", "csv", "--force-export", "true", "--output", prefix,
             prefix.with_suffix(".nsys-rep")], stdout=subprocess.DEVNULL)
        prefix.with_suffix(".sqlite").unlink(missing_ok=True)
        if not ncu:
            continue
        count = launch_count(f"{prefix}_cuda_gpu_kern_sum.csv")
        run(["flock", LOCK, "ncu", "--set", "basic", "--graph-profiling", "node",
             "--launch-count", count, "--target-processes", "application-only",
             "--force-overwrite", "--export", prefix, *bench], stdout=subprocess.DEVNULL)
        report = next(p for p in (prefix.with_suffix(".ncu-rep"), prefix.with_suffix(".ncu-repz"))
                      if p.is_file())
        with open(f"{prefix}-ncu-raw.csv", "w") as stream:
            run(["ncu", "--import", report, "--csv", "--page", "raw"], stdout=stream)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("apps", nargs="*", metavar="APP", help=f"default: {' '.join(APPS)}")
    parser.add_argument("--no-ncu", action="store_true", help="Nsight Systems only")
    args = parser.parse_args()
    unknown = [app for app in args.apps if app not in APPS]
    if unknown:
        parser.error(f"unknown application(s): {', '.join(unknown)}")
    folder = subprocess.check_output([str(REPO / "gpu.sh"), "name"], text=True).strip()
    stamp = os.environ.get("TIMESTAMP") or f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
    for app in args.apps or APPS:
        profile(app, folder, stamp, not args.no_ncu)
    return 0


if __name__ == "__main__":
    sys.exit(main())
