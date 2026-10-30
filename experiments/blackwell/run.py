#!/usr/bin/env python3
"""Build, check, time, and profile the RTX 5090 feature experiments.

  run.py EXPERIMENT                           fresh build, tests, large checks
  run.py EXPERIMENT --action bench --pairs 5  paired Direct and Graph timing
  run.py EXPERIMENT --action profile          nsys and ncu of one invocation
  run.py EXPERIMENT --case NAME ...           only the named cases

EXPERIMENT is a directory under experiments/blackwell with an
experiment.json listing its cases (the control first). Binaries are built by
the experiment's Makefile into <gpu>/blackwell_experiments/EXPERIMENT/build.

check (the default) forces a fresh build of every case (make -B), runs the
experiment's own tests (make test), then each case's large check in Direct and
Graph mode, its extra actions (experiment.json "extra_actions"), and a SASS
inspection of the kernels named in experiment.json "sass". When the
experiment requires it ("digests_must_match"), every case's output digest must
equal the control's. bench never builds: it requires the binaries the check
recorded, unchanged. Each pair times every case once per launch mode, in
experiment.json order on odd pairs and in reverse order on even pairs. A bench
record whose output digest differs from the same stamp's check digest fails.
profile captures one end-to-end invocation per case with Nsight Systems and
Nsight Compute (`--set full`), using the harness's `--action profile`.

Every record embeds the source revision, the build record (command, flags,
architecture, specification hash, source and binary hashes, external
dependencies), and the GPU's UUID and driver. File names start with the stamp:
$TIMESTAMP when set (results/record_run.py sets it to its run id), otherwise
the current UTC time. Run it under results/record_run.py for provenance.
Each GPU process takes the GPU lock; do not wrap this script in flock.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys

BW = Path(__file__).resolve().parent
REPO = BW.parents[1]
LOCK = os.environ.get("GPU_LOCK", "/tmp/cuda-elastic-gpu.lock")
LAUNCHES = ("direct", "graph")
SCOPES = ("application", "compute")
METRICS = ("batch_per_iteration_us", "single_median_us")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path):
    return str(Path(path).resolve().relative_to(REPO))


def experiments():
    return sorted(p.parent.name for p in BW.glob("*/experiment.json"))


def load_experiment(name):
    config = json.loads((BW / name / "experiment.json").read_text())
    roles = [case["role"] for case in config["cases"]]
    if roles.count("control") != 1 or roles[0] != "control":
        raise ValueError(f"{name}: the first case, and only it, must be the control")
    return config


def gpu_info():
    folder = subprocess.check_output([str(REPO / "gpu.sh"), "name"], text=True).strip()
    device = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0] or "0"
    query = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader",
         "-i", device], text=True).strip().split(", ")
    return {"folder": folder, "name": query[0], "uuid": query[1], "driver": query[2]}


def source_revision():
    commit = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                     text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain",
                                     "--untracked-files=no"], text=True).strip()
    return {"commit": commit, "dirty": bool(dirty)}


def stamp_now():
    return os.environ.get("TIMESTAMP") or f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"


class Runner:
    def __init__(self, name, args):
        self.name = name
        self.args = args
        self.config = load_experiment(name)
        known = [case["name"] for case in self.config["cases"]]
        unknown = [case for case in args.case if case not in known]
        if unknown:
            raise ValueError(f"unknown case(s) {', '.join(unknown)}; choose from {', '.join(known)}")
        self.gpu = gpu_info()
        self.revision = source_revision()
        self.stamp = stamp_now()
        self.out = REPO / self.gpu["folder"] / "blackwell_experiments" / name
        self.build = self.out / "build"
        self.results = self.out / "results"
        self.results.mkdir(parents=True, exist_ok=True)
        self.cases = [case for case in self.config["cases"]
                      if not args.case or case["name"] in args.case]
        self.records = []
        self.failed = False

    # -- helpers ---------------------------------------------------------
    def make(self, *targets, force=False):
        log = self.results / f"{self.stamp}-make-{'-'.join(targets)}.log"
        command = ["make", "-C", str(BW / self.name), *(["-B"] if force else []), *targets]
        with log.open("w") as stream:
            stream.write("$ " + " ".join(command) + "\n")
            stream.flush()
            code = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT).returncode
        print(f"{'make ' + ' '.join(targets):40s} exit {code}  {rel(log)}", flush=True)
        self.records.append({"kind": "make", "targets": list(targets), "force": force,
                             "exit_code": code, "log": rel(log), "log_sha256": sha256(log)})
        if code:
            self.failed = True
        return code

    def build_info(self, case):
        binary = self.build / case
        info_path = Path(str(binary) + ".build.json")
        if not binary.is_file() or not info_path.is_file():
            raise FileNotFoundError(f"{rel(binary)} or its build record is missing; run the check")
        info = json.loads(info_path.read_text())
        if sha256(binary) != info["binary_sha256"]:
            raise ValueError(f"{rel(binary)} changed after its build record was written")
        return binary, info

    def base_record(self, case, action, info):
        return {"schema": "blackwell_experiments.result/1", "experiment": self.name,
                "case": case["name"], "role": case["role"], "action": action,
                "stamp": self.stamp, "source": self.revision, "gpu": self.gpu,
                "build": info}

    def run_binary(self, case, action, arguments, name):
        """Runs one binary invocation under the GPU lock; returns its record."""
        binary, info = self.build_info(case["name"])
        record_path = self.results / f"{name}.json"
        log = self.results / f"{name}.log"
        raw = self.results / f"{name}.raw.json"
        for path in (record_path, log, raw):
            path.unlink(missing_ok=True)
        command = ["flock", LOCK, str(binary), *arguments, "--json-out", str(raw)]
        with log.open("w") as stream:
            stream.write("$ " + " ".join(command) + "\n")
            stream.flush()
            code = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT).returncode
        record = self.base_record(case, action, info)
        record.update(command=command[2:], exit_code=code, log=rel(log), log_sha256=sha256(log))
        if raw.is_file():
            record["result"] = json.loads(raw.read_text())
            raw.unlink()
        record_path.write_text(json.dumps(record, indent=1) + "\n")
        status = record.get("result", {}).get("status", "no_record")
        print(f"{case['name']:20s} {name.removeprefix(self.stamp + '-'):38s} {status:18s} "
              f"exit {code}", flush=True)
        self.records.append({"kind": action, "case": case["name"], "record": rel(record_path),
                             "record_sha256": sha256(record_path), "status": status,
                             "exit_code": code})
        return record

    # -- actions ---------------------------------------------------------
    def check(self):
        if not self.args.no_build:
            if self.make("build", force=True):
                return
        if not self.args.skip_tests and self.make("test"):
            return
        digests = {}
        for case in self.cases:
            record = self.run_binary(case, "check", self.config["check"]["args"],
                                     f"{self.stamp}-{case['name']}-check")
            result = record.get("result", {})
            if record["exit_code"] or result.get("status") != "pass":
                self.failed = True
            digests[case["name"]] = {c["digest"] for c in result.get("checks", [])}
            for extra in self.config.get("extra_actions", []):
                extra_record = self.run_binary(
                    case, extra["name"], extra["args"],
                    f"{self.stamp}-{case['name']}-{extra['name']}")
                if extra_record["exit_code"]:
                    self.failed = True
            self.inspect_sass(case)
        if self.config["check"].get("digests_must_match"):
            distinct = set().union(*digests.values())
            agree = len(distinct) == 1 and all(len(d) == 1 for d in digests.values())
            self.records.append({"kind": "digest_agreement", "digests":
                                 {k: sorted(v) for k, v in digests.items()}, "agree": agree})
            print(f"{'output digests agree across cases':60s} {agree}", flush=True)
            if not agree:
                self.failed = True

    def inspect_sass(self, case):
        """Opcode counts of the experiment's kernels, and their SASS."""
        sass = self.config.get("sass")
        if not sass:
            return
        binary, info = self.build_info(case["name"])
        dump = subprocess.run(["cuobjdump", "-sass", str(binary)], capture_output=True,
                              text=True, check=True).stdout
        kernels, current = {}, None
        for line in dump.splitlines():
            header = re.match(r"\s*Function : (\S+)", line)
            if header:
                current = header.group(1)
                kernels[current] = []
            elif current and re.match(r"\s+/\*[0-9a-f]{4,}\*/", line):
                kernels[current].append(line)
        names = list(kernels)
        demangled = subprocess.run(["cu++filt"], input="\n".join(names), capture_output=True,
                                   text=True).stdout.splitlines() if names else []
        pattern = re.compile(sass["kernels"])
        counts, text = {}, []
        for mangled, pretty in zip(names, demangled):
            if not pattern.search(pretty):
                continue
            body = kernels[mangled]
            opcodes = [re.sub(r"^\s*/\*[0-9a-f]+\*/\s*(@!?U?P\w+\s+)?", "", line).split(" ")[0]
                       .rstrip(";") for line in body]
            counts[pretty] = {
                "instructions": len(body),
                "features": {feature: sum(bool(re.search(regex, line)) for line in body)
                             for feature, regex in sass["features"].items()},
                "opcodes": dict(sorted(((op, opcodes.count(op)) for op in set(opcodes)),
                                       key=lambda item: -item[1])),
            }
            text.append(f"Function : {pretty}\n" + "\n".join(body))
        path = self.results / f"{self.stamp}-{case['name']}-sass.txt"
        path.write_text("\n\n".join(text) + "\n")
        record = self.base_record(case, "sass", info)
        record.update(sass_file=rel(path), sass_sha256=sha256(path), kernels=counts)
        record_path = self.results / f"{self.stamp}-{case['name']}-sass.json"
        record_path.write_text(json.dumps(record, indent=1) + "\n")
        summary = {feature: sum(k["features"][feature] for k in counts.values())
                   for feature in sass["features"]}
        print(f"{case['name']:20s} sass {len(counts)} kernel(s): "
              + ", ".join(f"{k} {v}" for k, v in summary.items()), flush=True)
        self.records.append({"kind": "sass", "case": case["name"], "record": rel(record_path),
                             "record_sha256": sha256(record_path), "features": summary})
        if not counts:
            self.failed = True

    def bench(self):
        checks = {}
        for case in self.cases:
            check = self.results / f"{self.stamp}-{case['name']}-check.json"
            if check.is_file():
                result = json.loads(check.read_text()).get("result", {})
                checks[case["name"]] = {c["digest"] for c in result.get("checks", [])}
        launches = LAUNCHES if self.args.launch == "both" else (self.args.launch,)
        for pair in range(1, self.args.pairs + 1):
            order = self.cases if pair % 2 else list(reversed(self.cases))
            for launch in launches:
                for position, case in enumerate(order):
                    record = self.run_binary(
                        case, "bench",
                        ["--preset", "large", "--action", "bench", "--launch", launch,
                         "--iterations", str(self.args.iterations), "--warmup",
                         str(self.args.warmup)],
                        f"{self.stamp}-{case['name']}-bench-{launch}-p{pair:02d}")
                    record.update(pair=pair, position=position, launch=launch)
                    digest = record.get("result", {}).get("output_digest")
                    expected = checks.get(case["name"])
                    record["digest_matches_check"] = (None if expected is None
                                                      else expected == {digest})
                    path = REPO / self.records[-1]["record"]
                    path.write_text(json.dumps(record, indent=1) + "\n")
                    self.records[-1].update(record_sha256=sha256(path), pair=pair,
                                            launch=launch, position=position)
                    if record["exit_code"] or record["digest_matches_check"] is False:
                        self.failed = True
        self.summary_extra = {"timing": summarize_bench(
            [json.loads((REPO / r["record"]).read_text()) for r in self.records
             if r["kind"] == "bench"])}

    def profile(self):
        profiles = self.out / "profiles"
        profiles.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        env.setdefault("NSYS_TMPDIR", str(Path.home() / ".cache" / "cuda-elastic" / "nsys-tmp"))
        Path(env["NSYS_TMPDIR"]).mkdir(parents=True, exist_ok=True)
        for case in self.cases:
            binary, info = self.build_info(case["name"])
            launch = self.args.launch if self.args.launch != "both" else self.config.get(
                "profile_launch", "graph")
            prefix = profiles / f"{self.stamp}-{case['name']}-{launch}"
            run = [str(binary), "--preset", "large", "--action", "profile", "--launch", launch]
            steps = [
                ["flock", LOCK, "nsys", "profile", "--trace", "cuda", "--cuda-graph-trace",
                 "node", "--capture-range", "cudaProfilerApi", "--capture-range-end", "stop",
                 "--force-overwrite", "true", "--output", str(prefix), *run],
                ["nsys", "stats", "--report", "cuda_gpu_kern_sum,cuda_gpu_mem_time_sum",
                 "--format", "csv", "--force-export", "true", "--output", str(prefix),
                 f"{prefix}.nsys-rep"],
            ]
            if not self.args.no_ncu:
                steps.append(["flock", LOCK, "ncu", "--set", "full", "--import-source", "no",
                              "--graph-profiling", "node", "--profile-from-start", "off",
                              "--target-processes", "application-only", "--force-overwrite",
                              "--export", str(prefix), *run])
            log = profiles / f"{prefix.name}.log"
            code = 0
            with log.open("w") as stream:
                for step in steps:
                    stream.write("$ " + " ".join(step) + "\n")
                    stream.flush()
                    code = subprocess.run(step, stdout=stream, stderr=subprocess.STDOUT,
                                          env=env).returncode
                    if code:
                        break
                Path(f"{prefix}.sqlite").unlink(missing_ok=True)
                report = next((p for p in (Path(f"{prefix}.ncu-rep"), Path(f"{prefix}.ncu-repz"))
                               if p.is_file()), None)
                if not code and not self.args.no_ncu and report is None:
                    stream.write("ncu wrote no report\n")
                    code = 1
                if not code and report is not None:
                    for page in ("raw", "details"):
                        with open(f"{prefix}-ncu-{page}.csv", "w") as csv:
                            code = subprocess.run(["ncu", "--import", str(report), "--csv",
                                                   "--page", page], stdout=csv,
                                                  stderr=stream).returncode
                            if code:
                                break
            files = sorted(rel(p) for p in profiles.glob(f"{prefix.name}*") if p.is_file())
            print(f"{case['name']:20s} profile {launch:6s} exit {code}  {len(files)} files",
                  flush=True)
            self.records.append({"kind": "profile", "case": case["name"], "launch": launch,
                                 "exit_code": code, "log": rel(log), "files": files,
                                 "build": info, "gpu": self.gpu})
            if code:
                self.failed = True

    def write_summary(self):
        path = self.out / f"{self.stamp}-{self.args.action}.json"
        summary = {"schema": "blackwell_experiments.summary/1", "experiment": self.name,
                   "question": self.config.get("question"), "stamp": self.stamp,
                   "action": self.args.action, "source": self.revision, "gpu": self.gpu,
                   "cases": [c["name"] for c in self.cases],
                   "settings": {"pairs": self.args.pairs, "iterations": self.args.iterations,
                                "warmup": self.args.warmup, "launch": self.args.launch}
                   if self.args.action == "bench" else None,
                   "status": "fail" if self.failed else "pass", "records": self.records}
        summary.update(getattr(self, "summary_extra", {}))
        path.write_text(json.dumps(summary, indent=1) + "\n")
        print(f"{'FAIL' if self.failed else 'PASS'}: {rel(path)}", flush=True)


def summarize_bench(records):
    """Medians, ranges, and paired candidate/control ratios per launch and scope."""
    control = next((r["case"] for r in records if r["role"] == "control"), None)
    cases = list(dict.fromkeys(r["case"] for r in records))
    table = {}
    for launch in LAUNCHES:
        for scope in SCOPES:
            for metric in METRICS:
                values = {case: {r["pair"]: r["result"]["timing"][scope][metric]
                                 for r in records if r["case"] == case and r["launch"] == launch
                                 and r.get("result", {}).get("timing")}
                          for case in cases}
                if not any(values.values()):
                    continue
                entry = {}
                for case, by_pair in values.items():
                    samples = list(by_pair.values())
                    stats = {"median": statistics.median(samples), "min": min(samples),
                             "max": max(samples), "n": len(samples)}
                    if case != control and values.get(control):
                        ratios = [by_pair[p] / values[control][p] for p in by_pair
                                  if p in values[control]]
                        stats["paired_ratio"] = {
                            "median": statistics.median(ratios), "min": min(ratios),
                            "max": max(ratios), "below_one": sum(r < 1 for r in ratios),
                            "n": len(ratios)}
                    entry[case] = stats
                table[f"{launch}/{scope}/{metric}"] = entry
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment", choices=experiments())
    parser.add_argument("--action", choices=["check", "bench", "profile"], default="check")
    parser.add_argument("--case", action="append", default=[], metavar="NAME")
    parser.add_argument("--launch", choices=["direct", "graph", "both"], default="both",
                        help="bench: modes to time (default both); profile: default "
                             "experiment.json profile_launch")
    parser.add_argument("--pairs", type=int, default=5, help="bench: paired repetitions")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--no-build", action="store_true", help="check: skip the fresh build")
    parser.add_argument("--skip-tests", action="store_true", help="check: skip make test")
    parser.add_argument("--no-ncu", action="store_true", help="profile: Nsight Systems only")
    args = parser.parse_args()
    if args.pairs < 1 or args.iterations < 1 or args.warmup < 0:
        parser.error("--pairs and --iterations must be positive, --warmup non-negative")
    try:
        runner = Runner(args.experiment, args)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(2, f"run.py: {error}\n")
    try:
        getattr(runner, args.action)()
    except (FileNotFoundError, ValueError) as error:
        print(f"run.py: {error}", file=sys.stderr)
        runner.failed = True
    runner.write_summary()
    return 1 if runner.failed else 0


if __name__ == "__main__":
    sys.exit(main())
