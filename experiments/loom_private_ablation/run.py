#!/usr/bin/env python3
"""Check, time, and profile cumulative optimization stages of a loom_private app.

  run.py APP                           fresh builds, tests, large checks, SASS checks
  run.py APP --action bench [--repeat 3]  time every case (harness bench protocol)
  run.py APP --action profile [--no-ncu]  nsys and ncu of one invocation per case

APP is a directory under experiments/loom_private_ablation with a stages.json.
Its cases run in order: stage 0 is the suite's baseline (Direct and Graph),
the middle stages are the experiment's own binaries (APP/Makefile), and the
last stage is the suite's optimized binary. Every case passes the contract's
large check.

check (the default) forces fresh builds of the suite's baseline and optimized
binaries (applications/loom_private/APP/Makefile) and of the stage binaries,
runs the stages' tests (make test), the large check of every binary in Direct
and Graph mode, and the machine-code comparisons in stages.json "sass_equal".
Every binary's output digest must equal the baseline's.

bench never builds. Repetition r times every case once, in stages.json order,
with the suite's protocol and defaults (the harness's --action bench: 20
iterations after 3 warmups), exactly as run_suite.py --action bench times the
suite. profile captures each case with tools/profile.py's own commands: nsys
of one end-to-end invocation with its kernel and memcpy summaries, then ncu
--set basic with its raw CSV export.

Outputs go to <gpu>/loom_private_ablation/APP/{results,profiles}/ with names
that start with the stamp ($TIMESTAMP, which results/record_run.py sets to its
run id, else the current UTC time), plus a summary
<gpu>/loom_private_ablation/APP/<stamp>-<action>.json. Run it under
results/record_run.py. Each GPU process takes the GPU lock itself; do not wrap
this script in flock.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SUITE = REPO / "applications" / "loom_private"
LOCK = os.environ.get("GPU_LOCK", "/tmp/cuda-elastic-gpu.lock")


def load_profile_tool():
    spec = importlib.util.spec_from_file_location("lp_profile", SUITE / "tools" / "profile.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path):
    return str(Path(path).resolve().relative_to(REPO))


def gpu_folder():
    return subprocess.check_output([str(REPO / "gpu.sh"), "name"], text=True).strip()


def load_stages(app):
    config = json.loads((HERE / app / "stages.json").read_text())
    stages = [case["stage"] for case in config["cases"]]
    if stages != sorted(stages) or stages[0] != 0:
        raise ValueError(f"{app}: cases must start at stage 0 and be in stage order")
    for case in config["cases"]:
        if case["binary"] not in config["binaries"]:
            raise ValueError(f"{app}: case {case['id']} names unknown binary {case['binary']}")
    return config


def binary_path(app, folder, spec):
    if "suite_variant" in spec:
        name = app if spec["suite_variant"] == "baseline" else f"{app}_optimized"
        return REPO / folder / "loom_private" / app / "build" / name
    return REPO / folder / "loom_private_ablation" / app / "build" / spec["stage_binary"]


def logged(command, log):
    """Runs command with its output appended to log; returns the exit code."""
    with open(log, "a") as stream:
        stream.write("$ " + " ".join(map(str, command)) + "\n")
        stream.flush()
        return subprocess.run([str(c) for c in command], stdout=stream,
                              stderr=subprocess.STDOUT).returncode


def sass(binary, mangled):
    """The instruction text of the one kernel whose mangled name contains `mangled`."""
    text = subprocess.check_output(["cuobjdump", "-sass", str(binary)], text=True)
    found = []
    for block in re.split(r"\n\s*Function : ", text)[1:]:
        name, _, body = block.partition("\n")
        if mangled in name:
            found.append([m.group(1) for m in
                          re.finditer(r"/\*[0-9a-f]{4,}\*/\s+(.*?)\s*;", body)])
    if len(found) != 1:
        raise ValueError(f"{rel(binary)}: {len(found)} kernels match {mangled}")
    return found[0]


def check(app, config, folder, stamp, out):
    log = out / "results" / f"{stamp}-check.log"
    log.unlink(missing_ok=True)
    steps = []

    def step(name, command):
        code = logged(command, log)
        steps.append({"step": name, "exit_code": code})
        print(f"{name:40s} exit {code}", flush=True)
        if code:
            raise SystemExit(f"run.py: {name} failed; see {rel(log)}")

    for variant in ("baseline", "optimized"):
        step(f"build suite {variant}",
             ["make", "-B", "-C", SUITE / app, "build", f"VARIANT={variant}"])
    step("build stages", ["make", "-B", "-C", HERE / app, "build"])
    step("stage tests", ["make", "-C", HERE / app, "test"])

    binaries, digests = {}, {}
    for key, spec in config["binaries"].items():
        path = binary_path(app, folder, spec)
        record = out / "results" / f"{stamp}-check-{key}.json"
        record.unlink(missing_ok=True)
        step(f"large check {key}", ["flock", LOCK, path, "--preset", "large", "--action",
                                    "check", "--launch", "both", "--json-out", record])
        data = json.loads(record.read_text())
        passed = (data["status"] == "pass" and data["outputs_agree"]
                  and all(c["pass"] for c in data["checks"]))
        digest = {c["digest"] for c in data["checks"]}
        if not passed or len(digest) != 1:
            raise SystemExit(f"run.py: {key} failed its large check ({rel(record)})")
        digests[key] = digest.pop()
        binaries[key] = {"path": rel(path), "sha256": sha256(path), "variant": data["variant"],
                         "check_record": rel(record), "output_digest": digests[key]}
    if len(set(digests.values())) != 1:
        raise SystemExit(f"run.py: output digests differ: {digests}")

    comparisons = []
    for item in config.get("sass_equal", []):
        a = sass(binary_path(app, folder, config["binaries"][item["binary"]]), item["kernel"])
        b = sass(binary_path(app, folder, config["binaries"][item["reference"]]), item["kernel"])
        same = a == b
        comparisons.append({**item, "instructions": [len(a), len(b)], "identical": same})
        print(f"sass {item['binary']} vs {item['reference']} ({item['kernel']}): "
              f"{len(a)} and {len(b)} instructions, {'identical' if same else 'DIFFERENT'}",
              flush=True)
        if not same:
            raise SystemExit("run.py: machine code differs; see stages.json sass_equal")
    return {"steps": steps, "log": rel(log), "binaries": binaries, "sass_equal": comparisons}


def bench(app, config, folder, stamp, out, repeat):
    records = []
    for r in range(1, repeat + 1):
        for case in config["cases"]:
            path = binary_path(app, folder, config["binaries"][case["binary"]])
            if not path.is_file():
                raise SystemExit(f"run.py: {rel(path)} is missing; run the check first")
            name = f"{stamp}-bench-{case['id']}-r{r}"
            record, log = out / "results" / f"{name}.json", out / "results" / f"{name}.log"
            record.unlink(missing_ok=True)
            log.unlink(missing_ok=True)
            code = logged(["flock", LOCK, path, "--preset", "large", "--action", "bench",
                           "--launch", case["launch"], "--json-out", record], log)
            if code or not record.is_file():
                raise SystemExit(f"run.py: bench {case['id']} r{r} failed; see {rel(log)}")
            data = json.loads(record.read_text())
            times = {s: data["timing"][s]["batch_per_iteration_us"]
                     for s in ("application", "compute")}
            print(f"r{r} {case['id']:20s} application {times['application']:9.1f} us  "
                  f"compute {times['compute']:9.1f} us", flush=True)
            records.append({"case": case["id"], "repetition": r, "record": rel(record),
                            "log": rel(log), "binary_sha256": sha256(path),
                            "batch_per_iteration_us": times})
    return {"repeat": repeat, "records": records}


def profile(app, config, folder, stamp, out, ncu):
    tool = load_profile_tool()
    reports = []
    for case in config["cases"]:
        path = binary_path(app, folder, config["binaries"][case["binary"]])
        if not path.is_file():
            raise SystemExit(f"run.py: {rel(path)} is missing; run the check first")
        prefix = out / "profiles" / f"{stamp}-{case['id']}"
        for command in tool.commands(path, case["launch"], prefix, ncu):
            tool.run(command, stdout=subprocess.DEVNULL)
        tool.named(prefix, ".sqlite").unlink(missing_ok=True)
        entry = {"case": case["id"], "nsys": rel(tool.named(prefix, ".nsys-rep")),
                 "nsys_kernels": rel(tool.named(prefix, "_cuda_gpu_kern_sum.csv")),
                 "nsys_memcpy": rel(tool.named(prefix, "_cuda_gpu_mem_time_sum.csv"))}
        if ncu:
            report = next((p for p in (tool.named(prefix, ".ncu-rep"),
                                       tool.named(prefix, ".ncu-repz")) if p.is_file()), None)
            if report is None:
                raise SystemExit(f"run.py: ncu wrote no report for {rel(prefix)}")
            csv_path = tool.named(prefix, "-ncu-raw.csv")
            with open(csv_path, "w") as stream:
                tool.run(["ncu", "--import", report, "--csv", "--page", "raw"], stdout=stream)
            entry.update(ncu=rel(report), ncu_raw=rel(csv_path))
        reports.append(entry)
    return {"ncu": ncu, "profiles": reports}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    apps = sorted(p.parent.name for p in HERE.glob("*/stages.json"))
    parser.add_argument("app", choices=apps)
    parser.add_argument("--action", choices=["check", "bench", "profile"], default="check")
    parser.add_argument("--repeat", type=int, default=3, help="bench: timed runs per case")
    parser.add_argument("--no-ncu", action="store_true", help="profile: Nsight Systems only")
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be at least 1")
    config = load_stages(args.app)
    folder = gpu_folder()
    stamp = os.environ.get("TIMESTAMP") or f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
    out = REPO / folder / "loom_private_ablation" / args.app
    for sub in ("results", "profiles"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    query = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader", "-i",
         os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0] or "0"], text=True)
    name, uuid, driver = [s.strip() for s in query.strip().split(",")]
    summary = {"schema": "loom_private_ablation.summary/1", "application": args.app,
               "action": args.action, "stamp": stamp, "gpu_folder": folder,
               "device": {"name": name, "uuid": uuid, "driver": driver},
               "source_commit": subprocess.check_output(
                   ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
               "stages_sha256": sha256(HERE / args.app / "stages.json")}
    if args.action == "check":
        summary.update(check(args.app, config, folder, stamp, out))
    elif args.action == "bench":
        summary.update(bench(args.app, config, folder, stamp, out, args.repeat))
    else:
        summary.update(profile(args.app, config, folder, stamp, out, not args.no_ncu))
    path = out / f"{stamp}-{args.action}.json"
    path.write_text(json.dumps(summary, indent=1) + "\n")
    print(f"PASS: {rel(path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
