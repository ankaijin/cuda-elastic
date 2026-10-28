#!/usr/bin/env python3
"""Profile the loom_private applications with Nsight Systems and Nsight Compute.

  profile.py [APP ...]        all six by default
  profile.py --no-ncu [APP]   Nsight Systems only (where counters are admin-only)

Each application is profiled in three versions: the baseline with direct
launches, the baseline as a CUDA Graph, and the optimized variant as a CUDA
Graph (the version the report's Optimized column times). Every capture runs
the binary's `--action profile`: setup and one warmup invocation, then exactly
one end-to-end invocation between cudaProfilerStart and cudaProfilerStop. The
profilers capture only that range (nsys --capture-range cudaProfilerApi, ncu
--profile-from-start off), so each report holds one complete invocation and
nothing from setup.

Per version, files go to <gpu>/loom_private/<app>/profiles/ as
<stamp>-<version>.nsys-rep with its kernel and memcpy summaries
(<stamp>-<version>_cuda_gpu_kern_sum.csv, _cuda_gpu_mem_time_sum.csv), and
<stamp>-<version>.ncu-rep (or .ncu-repz) with its raw metrics exported to
<stamp>-<version>-ncu-raw.csv, so the report never needs this machine's Nsight
tools to read them. The stamp is $TIMESTAMP when set (results/record_run.py
sets it to the run id).

Both variants' binaries must be built first (`make build` with VARIANT=baseline
and VARIANT=optimized). Each profiled process takes the GPU lock; do not wrap
this script in flock.
"""

import argparse
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
LOCK = os.environ.get("GPU_LOCK", "/tmp/cuda-elastic-gpu.lock")


def named(prefix, suffix):
    """prefix + suffix, whatever dots the stamp contains."""
    return prefix.parent / (prefix.name + suffix)


def commands(binary, launch, prefix, ncu):
    """The capture and export commands for one version, in order."""
    run = [binary, "--action", "profile", "--launch", launch]
    steps = [
        ["flock", LOCK, "nsys", "profile", "--trace", "cuda", "--cuda-graph-trace", "node",
         "--capture-range", "cudaProfilerApi", "--capture-range-end", "stop",
         "--force-overwrite", "true", "--output", prefix, *run],
        ["nsys", "stats", "--report", "cuda_gpu_kern_sum,cuda_gpu_mem_time_sum",
         "--format", "csv", "--force-export", "true", "--output", prefix,
         named(prefix, ".nsys-rep")],
    ]
    if ncu:
        steps.append(["flock", LOCK, "ncu", "--set", "basic", "--graph-profiling", "node",
                      "--profile-from-start", "off", "--target-processes", "application-only",
                      "--force-overwrite", "--export", prefix, *run])
    return steps


def run(command, **kwargs):
    print("$ " + " ".join(map(str, command)), flush=True)
    result = subprocess.run([str(c) for c in command], **kwargs)
    if result.returncode:
        sys.exit(f"profile.py: command failed with exit {result.returncode}: "
                 + " ".join(map(str, command)))


def profile(app, folder, stamp, ncu):
    out = REPO / folder / "loom_private" / app / "profiles"
    out.mkdir(parents=True, exist_ok=True)
    for name, variant, launch in VERSIONS:
        binary = REPO / folder / "loom_private" / app / "build" / (
            app if variant == "baseline" else f"{app}_optimized")
        if not binary.is_file():
            sys.exit(f"{binary.relative_to(REPO)} is missing; build both variants first")
        prefix = out / f"{stamp}-{name}"
        for command in commands(binary, launch, prefix, ncu):
            run(command, stdout=subprocess.DEVNULL)
        named(prefix, ".sqlite").unlink(missing_ok=True)
        if not ncu:
            continue
        report = next((p for p in (named(prefix, ".ncu-rep"), named(prefix, ".ncu-repz"))
                       if p.is_file()), None)
        if report is None:
            sys.exit(f"profile.py: ncu wrote no report for {prefix.relative_to(REPO)}")
        with open(named(prefix, "-ncu-raw.csv"), "w") as stream:
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
