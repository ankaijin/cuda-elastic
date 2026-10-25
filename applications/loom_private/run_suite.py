#!/usr/bin/env python3
"""Build and run the loom_private suite: large correctness checks by default.

  run_suite.py [APP ...]                         build, then check-large each app
  run_suite.py --action bench --launch graph APP  time an app (explicit only)

APP names are directory names under applications/loom_private (default: all
six). Each run writes its result record and log to
<gpu>/loom_private/<app>/results/<stamp>-<action>[-<launch>].{json,log} and a
suite summary to <gpu>/loom_private/suite-<stamp>-<action>.json. The stamp is
$TIMESTAMP when set (results/record_run.py sets it to the run id, so these
files become that run's artifacts), otherwise the current UTC time.

The Makefile targets take the GPU lock themselves; do not wrap this script in
flock on the same lock file. Legacy workloads and their scripts are untouched.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

SUITE = Path(__file__).resolve().parent
REPO = SUITE.parents[1]
APPS = ["pagerank", "attention", "decoder", "anomaly", "vecadd_memory", "vecadd_product"]


def gpu_folder():
    return subprocess.check_output([str(REPO / "gpu.sh"), "name"], text=True).strip()


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_app(app, args, folder, stamp):
    out = REPO / folder / "loom_private" / app / "results"
    out.mkdir(parents=True, exist_ok=True)
    name = f"{stamp}-{args.action}" + (f"-{args.launch}" if args.action == "bench" else "")
    record, log = out / f"{name}.json", out / f"{name}.log"
    make = ["make", "-C", str(SUITE / app)]
    commands = [] if args.no_build else [make + ["build"]]
    if args.action == "check":
        commands.append(make + ["check-large", f"CHECK_JSON={record}"])
    else:
        bench = make + ["bench", f"LAUNCH={args.launch}", f"BENCH_JSON={record}"]
        bench += [f"ITERATIONS={args.iterations}"] if args.iterations else []
        bench += [f"WARMUP={args.warmup}"] if args.warmup is not None else []
        commands.append(bench)
    code = 0
    with log.open("w") as stream:
        for command in commands:
            stream.write("$ " + " ".join(command) + "\n")
            stream.flush()
            code = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT).returncode
            if code:
                break
    result = {"application": app, "exit_code": code,
              "log": str(log.relative_to(REPO)), "log_sha256": sha256(log)}
    if record.is_file():
        result.update(record=str(record.relative_to(REPO)), record_sha256=sha256(record),
                      status=json.loads(record.read_text())["status"])
    else:
        result["status"] = "no_record"
    if code and result["status"] == "pass":
        result["status"] = "fail"
    print(f"{app:15s} {result['status']:20s} exit {code}  {result['log']}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("apps", nargs="*", metavar="APP", help=f"default: {' '.join(APPS)}")
    parser.add_argument("--action", choices=["check", "bench"], default="check")
    parser.add_argument("--launch", choices=["direct", "graph"],
                        help="required with --action bench")
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--no-build", action="store_true", help="skip make build")
    args = parser.parse_args()
    apps = args.apps or APPS
    unknown = [app for app in apps if app not in APPS]
    if unknown:
        parser.error(f"unknown application(s): {', '.join(unknown)}; choose from {', '.join(APPS)}")
    if args.action == "bench" and not args.launch:
        parser.error("--action bench needs --launch direct or --launch graph")
    if args.action == "check" and (args.launch or args.iterations or args.warmup is not None):
        parser.error("--launch, --iterations and --warmup apply to bench only; "
                     "check always runs Direct and Graph")

    folder = gpu_folder()
    stamp = os.environ.get("TIMESTAMP") or f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
    commit = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    results = [run_app(app, args, folder, stamp) for app in apps]
    summary = REPO / folder / "loom_private" / f"suite-{stamp}-{args.action}.json"
    summary.write_text(json.dumps({
        "schema": "loom_private.suite/1", "suite_version": "v1", "stamp": stamp,
        "source_commit": commit, "gpu_folder": folder, "action": args.action,
        "launch": args.launch or "both", "preset": "large", "applications": results,
    }, indent=1) + "\n")
    passed = all(r["status"] == "pass" or (args.action == "bench" and r["exit_code"] == 0)
                 for r in results)
    print(f"{'PASS' if passed else 'FAIL'}: {summary.relative_to(REPO)}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
