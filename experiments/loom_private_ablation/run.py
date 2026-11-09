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
It records each binary's hash, build command, and output digest, the nvcc
version, and that the production sources are unchanged (git status and
legacy_freeze.py verify). Digest rule: a stage listed in "arithmetic_changes"
(for this GPU, if the entry names "gpus") may change the output digest; every
other stage must reproduce the previous stage's digest exactly. Binaries with
"timed": false are references for "host_equal" (production built with its
test hooks, run with "env"); their digest must equal the stage they are
compared with.

bench and profile never build. They run only binaries the check validated:
before each run they hash the binary and compare it with the check summary of
the same stamp (else the newest check summary), and after the run they hash it
again. Both record the hash of every binary they ran and the check summary
they compared against. Repetition r times every case once in a rotated order (the
case list starting at offset (r - 1) mod n), and times the Graph baseline once
more before and after that rotation; the two bracket measurements give the
repetition's drift and are not used as case timings. Each measurement uses the
suite's protocol and defaults (the harness's --action bench: 20 iterations
after 3 warmups), exactly as run_suite.py --action bench times the suite.
profile captures each case with tools/profile.py's own commands: nsys of one
end-to-end invocation with its kernel and memcpy summaries, then ncu --set
basic with its raw CSV export. It also writes each capture's launch structure
(kernels with grid, block, and shared memory; copies with kind and bytes;
memsets with bytes) and checks every "host_equal" pair for identical
structures; untimed references are captured with nsys only.

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
import sqlite3
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SUITE = REPO / "applications" / "loom_private"
LOCK = os.environ.get("GPU_LOCK", "/tmp/cuda-elastic-gpu.lock")
SCOPES = ("application", "compute")


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
    if sorted(set(stages)) != list(range(max(stages) + 1)):
        raise ValueError(f"{app}: stages must be contiguous from 0")
    if config.get("scope", "compute") not in SCOPES:
        raise ValueError(f"{app}: scope must be one of {SCOPES}")
    binaries = config["binaries"]
    for case in config["cases"]:
        if case["binary"] not in binaries:
            raise ValueError(f"{app}: case {case['id']} names unknown binary {case['binary']}")
        if binaries[case["binary"]].get("timed", True) is False:
            raise ValueError(f"{app}: case {case['id']} uses untimed binary {case['binary']}")
    by_stage = {}
    for case in config["cases"]:
        if by_stage.setdefault(case["stage"], case["binary"]) != case["binary"]:
            raise ValueError(f"{app}: stage {case['stage']} uses more than one binary")
    for change in config.get("arithmetic_changes", []):
        if change["stage"] not in by_stage or change["stage"] == 0:
            raise ValueError(f"{app}: arithmetic_changes names unknown stage {change['stage']}")
    for item in config.get("sass_equal", []) + config.get("host_equal", []):
        for key in ("binary", "reference"):
            if item[key] not in binaries:
                raise ValueError(f"{app}: {item} names unknown binary {item[key]}")
    for item in config.get("host_equal", []):
        if item["binary"] not in by_stage.values():
            raise ValueError(f"{app}: host_equal binary {item['binary']} is not a stage")
    return config


def stage_binaries(config):
    """[(stage, binary key)] in stage order, one entry per stage."""
    seen = {}
    for case in config["cases"]:
        seen.setdefault(case["stage"], case["binary"])
    return sorted(seen.items())


def applies(item, folder):
    """Whether a stages.json entry with an optional "gpus" list applies here."""
    return "gpus" not in item or folder in item["gpus"]


def binary_env(spec):
    return {**os.environ, **{k: str(v) for k, v in spec.get("env", {}).items()}}


def binary_path(app, folder, spec):
    if "suite_variant" in spec:
        name = app if spec["suite_variant"] == "baseline" else f"{app}_optimized"
        return REPO / folder / "loom_private" / app / "build" / name
    return REPO / folder / "loom_private_ablation" / app / "build" / spec["stage_binary"]


def checked_binaries(out, stamp):
    """(summary path, {binary: sha256}) of the check that bench and profile
    must match: the check of this stamp, else the newest one."""
    path = out / f"{stamp}-check.json"
    if not path.is_file():
        found = sorted(out.glob("*-check.json"))
        if not found:
            raise SystemExit(f"run.py: no check summary in {rel(out)}; run the check first")
        path = found[-1]
    data = json.loads(path.read_text())
    return rel(path), {key: b["sha256"] for key, b in data["binaries"].items()}


def checked_run(path, expected, key, run):
    """Runs run() on the binary at path only if its hash is the checked one,
    and confirms the binary did not change during the run; returns the hash."""
    if not path.is_file():
        raise SystemExit(f"run.py: {rel(path)} is missing; run the check first")
    before = sha256(path)
    if key not in expected or before != expected[key]:
        raise SystemExit(f"run.py: {rel(path)} is not the binary the check validated; "
                         "run the check again")
    run()
    if sha256(path) != before:
        raise SystemExit(f"run.py: {rel(path)} changed while it ran")
    return before


def logged(command, log, env=None):
    """Runs command with its output appended to log; returns the exit code."""
    with open(log, "a") as stream:
        extra = {k: v for k, v in (env or {}).items() if os.environ.get(k) != v}
        stream.write("$ " + "".join(f"{k}={v} " for k, v in extra.items())
                     + " ".join(map(str, command)) + "\n")
        stream.flush()
        return subprocess.run([str(c) for c in command], stdout=stream,
                              stderr=subprocess.STDOUT, env=env).returncode


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


def nvcc_version():
    try:
        text = subprocess.check_output(["nvcc", "--version"], text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return text.strip().splitlines()[-1]


def production_unchanged(app):
    """git status of the production sources this app's endpoints build from."""
    paths = [f"applications/loom_private/{app}", "applications/loom_private/common",
             "applications/loom_private/common.mk"]
    status = subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain", "--",
                                      *paths], text=True)
    return {"paths": paths, "clean": not status.strip(), "status": status.splitlines()}


def digest_rule(config, digests, folder):
    """Per stage: its digest, whether stages.json declares an arithmetic change
    for it on this GPU, and whether it equals the previous stage's digest."""
    rule, previous = [], None
    for stage, key in stage_binaries(config):
        declared = any(c["stage"] == stage and applies(c, folder)
                       for c in config.get("arithmetic_changes", []))
        rule.append({"stage": stage, "binary": key, "digest": digests[key],
                     "declared_change": declared,
                     "equal_to_previous": previous is None or digests[key] == digests[previous]})
        previous = key
    return rule


def check(app, config, folder, stamp, out):
    log = out / "results" / f"{stamp}-check.log"
    log.unlink(missing_ok=True)
    steps = []

    def step(name, command, env=None):
        code = logged(command, log, env)
        steps.append({"step": name, "exit_code": code,
                      "command": " ".join(map(str, command))})
        print(f"{name:40s} exit {code}", flush=True)
        if code:
            raise SystemExit(f"run.py: {name} failed; see {rel(log)}")

    production = production_unchanged(app)
    if not production["clean"]:
        raise SystemExit(f"run.py: production sources have changes: {production['status']}")
    step("legacy freeze", [sys.executable, SUITE / "tools" / "legacy_freeze.py", "verify"])
    builds = {}
    for variant in ("baseline", "optimized"):
        command = ["make", "-B", "-C", SUITE / app, "build", f"VARIANT={variant}"]
        step(f"build suite {variant}", command)
        builds[variant] = " ".join(map(str, command))
    stage_build = ["make", "-B", "-C", HERE / app, "build"]
    step("build stages", stage_build)
    step("stage tests", ["make", "-C", HERE / app, "test"])

    binaries, digests = {}, {}
    for key, spec in config["binaries"].items():
        path = binary_path(app, folder, spec)
        env = binary_env(spec)
        record = out / "results" / f"{stamp}-check-{key}.json"
        record.unlink(missing_ok=True)
        step(f"large check {key}", ["flock", LOCK, path, "--preset", "large", "--action",
                                    "check", "--launch", "both", "--json-out", record], env)
        data = json.loads(record.read_text())
        passed = (data["status"] == "pass" and data["outputs_agree"]
                  and all(c["pass"] for c in data["checks"]))
        digest = {c["digest"] for c in data["checks"]}
        if not passed or len(digest) != 1:
            raise SystemExit(f"run.py: {key} failed its large check ({rel(record)})")
        digests[key] = digest.pop()
        binaries[key] = {"path": rel(path), "sha256": sha256(path), "variant": data["variant"],
                         "timed": spec.get("timed", True), "env": spec.get("env", {}),
                         "build_command": builds.get(spec.get("suite_variant"),
                                                     " ".join(map(str, stage_build))),
                         "check_record": rel(record), "output_digest": digests[key]}

    rule = digest_rule(config, digests, folder)
    for item in rule:
        print(f"stage {item['stage']} {item['binary']:12s} digest {item['digest']}"
              f"{' (declared arithmetic change)' if item['declared_change'] else ''}",
              flush=True)
    broken = [item for item in rule if not item["equal_to_previous"]
              and not item["declared_change"]]
    if broken:
        raise SystemExit(f"run.py: stage {broken[0]['stage']} ({broken[0]['binary']}) changes "
                         "the output digest, but stages.json does not declare an arithmetic "
                         "change for it on this GPU")
    for item in config.get("host_equal", []):
        if applies(item, folder) and digests[item["binary"]] != digests[item["reference"]]:
            raise SystemExit(f"run.py: host_equal pair {item['binary']} and "
                             f"{item['reference']} have different digests")

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
    return {"steps": steps, "log": rel(log), "nvcc": nvcc_version(),
            "production_sources": production, "binaries": binaries, "digest_rule": rule,
            "sass_equal": comparisons}


def bench_order(cases, repetition):
    """The rotated case order of a repetition (1-based)."""
    offset = (repetition - 1) % len(cases)
    return cases[offset:] + cases[:offset]


def bench(app, config, folder, stamp, out, repeat):
    graph = next(c for c in config["cases"] if c["stage"] == 0 and c["launch"] == "graph")
    records, orders, drift = [], [], []
    check_summary, expected = checked_binaries(out, stamp)

    def measure(case, r, role):
        path = binary_path(app, folder, config["binaries"][case["binary"]])
        tag = case["id"] if role == "case" else role
        name = f"{stamp}-bench-{tag}-r{r}"
        record, log = out / "results" / f"{name}.json", out / "results" / f"{name}.log"
        record.unlink(missing_ok=True)
        log.unlink(missing_ok=True)
        result = {}
        digest = checked_run(path, expected, case["binary"], lambda: result.update(
            code=logged(["flock", LOCK, path, "--preset", "large", "--action", "bench",
                         "--launch", case["launch"], "--json-out", record], log)))
        if result["code"] or not record.is_file():
            raise SystemExit(f"run.py: bench {tag} r{r} failed; see {rel(log)}")
        data = json.loads(record.read_text())
        times = {s: data["timing"][s]["batch_per_iteration_us"] for s in SCOPES}
        print(f"r{r} {tag:20s} application {times['application']:9.1f} us  "
              f"compute {times['compute']:9.1f} us", flush=True)
        entry = {"case": case["id"], "role": role, "repetition": r, "record": rel(record),
                 "log": rel(log), "binary_sha256": digest,
                 "batch_per_iteration_us": times}
        records.append(entry)
        return times

    for r in range(1, repeat + 1):
        order = bench_order(config["cases"], r)
        orders.append([c["id"] for c in order])
        start = measure(graph, r, "bracket-start")
        for case in order:
            measure(case, r, "case")
        end = measure(graph, r, "bracket-end")
        drift.append({"repetition": r, **{s: end[s] - start[s] for s in SCOPES}})

    ranges = {}
    for case in config["cases"]:
        values = [e["batch_per_iteration_us"] for e in records
                  if e["case"] == case["id"] and e["role"] == "case"]
        ranges[case["id"]] = {s: max(v[s] for v in values) - min(v[s] for v in values)
                              for s in SCOPES}
    return {"check_summary": check_summary, "repeat": repeat, "order": "rotated",
            "orders": orders, "records": records,
            "ranges_us": ranges, "drift_us": drift}


def launch_structure(database):
    """Kernels, copies, and memsets of one capture, as sorted [key..., count] lists."""
    connection = sqlite3.connect(database)

    def rows(query):
        try:
            return connection.execute(query).fetchall()
        except sqlite3.OperationalError:  # the capture has none of this activity
            return []

    def counted(items):
        counts = {}
        for item in items:
            counts[item] = counts.get(item, 0) + 1
        return [list(k) + [n] for k, n in sorted(counts.items())]

    kernels = [(re.sub(r"\(anonymous namespace\)::|<unnamed>::", "", name),
                gx, gy, gz, bx, by, bz, static, dynamic)
               for name, gx, gy, gz, bx, by, bz, static, dynamic in rows(
                   "SELECT s.value, gridX, gridY, gridZ, blockX, blockY, blockZ, "
                   "staticSharedMemory, dynamicSharedMemory FROM CUPTI_ACTIVITY_KIND_KERNEL k "
                   "JOIN StringIds s ON s.id = k.demangledName")]
    copies = rows("SELECT e.label, bytes FROM CUPTI_ACTIVITY_KIND_MEMCPY m "
                  "JOIN ENUM_CUDA_MEMCPY_OPER e ON e.id = m.copyKind")
    memsets = [(b,) for (b,) in rows("SELECT bytes FROM CUPTI_ACTIVITY_KIND_MEMSET")]
    connection.close()
    return {"kernel_fields": ["name", "grid_x", "grid_y", "grid_z", "block_x", "block_y",
                              "block_z", "static_shared", "dynamic_shared", "count"],
            "kernels": counted(kernels), "copies": counted(copies),
            "memsets": counted(memsets),
            "totals": {"kernels": len(kernels), "copies": len(copies),
                       "memsets": len(memsets)}}


def structure_difference(a, b):
    """A short description of how two launch structures differ ("" when equal)."""
    notes = []
    for part in ("kernels", "copies", "memsets"):
        only_a = [x for x in a[part] if x not in b[part]]
        only_b = [x for x in b[part] if x not in a[part]]
        if only_a or only_b:
            notes.append(f"{part}: only in binary {only_a[:3]}, only in reference {only_b[:3]}")
    return "; ".join(notes)


def profile(app, config, folder, stamp, out, ncu):
    tool = load_profile_tool()
    reports, structures = [], {}
    check_summary, expected = checked_binaries(out, stamp)

    def capture(key, launch, prefix, with_ncu):
        spec = config["binaries"][key]
        path = binary_path(app, folder, spec)

        def run():
            for command in tool.commands(path, launch, prefix, with_ncu):
                tool.run(command, stdout=subprocess.DEVNULL, env=binary_env(spec))
        digest = checked_run(path, expected, key, run)
        database = tool.named(prefix, ".sqlite")
        if not database.is_file():
            tool.run(["nsys", "export", "--type", "sqlite", "--force-overwrite", "true",
                      "--output", database, tool.named(prefix, ".nsys-rep")],
                     stdout=subprocess.DEVNULL)
        launches = launch_structure(database)
        database.unlink(missing_ok=True)
        launches_path = tool.named(prefix, "_launches.json")
        launches_path.write_text(json.dumps(launches, indent=1) + "\n")
        entry = {"binary": key, "binary_sha256": digest,
                 "nsys": rel(tool.named(prefix, ".nsys-rep")),
                 "nsys_kernels": rel(tool.named(prefix, "_cuda_gpu_kern_sum.csv")),
                 "nsys_memcpy": rel(tool.named(prefix, "_cuda_gpu_mem_time_sum.csv")),
                 "launches": rel(launches_path)}
        if with_ncu:
            report = next((p for p in (tool.named(prefix, ".ncu-rep"),
                                       tool.named(prefix, ".ncu-repz")) if p.is_file()), None)
            if report is None:
                raise SystemExit(f"run.py: ncu wrote no report for {rel(prefix)}")
            csv_path = tool.named(prefix, "-ncu-raw.csv")
            with open(csv_path, "w") as stream:
                tool.run(["ncu", "--import", report, "--csv", "--page", "raw"], stdout=stream)
            entry.update(ncu=rel(report), ncu_raw=rel(csv_path))
        return entry, launches

    for case in config["cases"]:
        prefix = out / "profiles" / f"{stamp}-{case['id']}"
        entry, launches = capture(case["binary"], case["launch"], prefix, ncu)
        structures[(case["binary"], case["launch"])] = launches
        reports.append({"case": case["id"], **entry})

    checks = []
    for item in config.get("host_equal", []):
        if not applies(item, folder):
            continue
        launch = next(c["launch"] for c in config["cases"] if c["binary"] == item["binary"])
        if (item["reference"], launch) not in structures:
            prefix = out / "profiles" / f"{stamp}-reference-{item['reference']}"
            entry, launches = capture(item["reference"], launch, prefix, False)
            structures[(item["reference"], launch)] = launches
            reports.append({"reference": item["reference"], **entry})
        difference = structure_difference(structures[(item["binary"], launch)],
                                          structures[(item["reference"], launch)])
        checks.append({**item, "launch": launch, "identical": not difference,
                       "difference": difference})
        print(f"host {item['binary']} vs {item['reference']}: "
              f"{'identical' if not difference else 'DIFFERENT: ' + difference}", flush=True)
        if difference:
            raise SystemExit("run.py: launch structures differ; see stages.json host_equal")
    return {"check_summary": check_summary, "ncu": ncu, "profiles": reports,
            "host_equal": checks}


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
