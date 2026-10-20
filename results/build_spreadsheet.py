#!/usr/bin/env python3
"""Builds results/cuda_performance.xlsx from the latest profiling results
committed in this repository. Nothing is re-run: every number is read from an
existing ncu report, nsys sqlite export, benchmark log or sweep CSV, and every
number links to the file it came from on GitHub.

One summary sheet per GPU (RTX 3090, RTX 4090), each read from that GPU's
output folder (rtx3090/, rtx4090/; see gpu.sh), plus each GPU's ncu metrics.
Only the latest run of each workload's current implementation is included, at
the largest problem size tested: the dominant kernel (when one dominates) and
the whole application.
Columns of each summary sheet, left to right:
  Direct launch      the workload's stream operations enqueued directly on
                     every iteration, without a graph (each harness's direct
                     mode: llama2c `direct`, loom_msa and mlperf
                     `--launch=direct`, pagerank PAGERANK_LAUNCH=direct)
  CUDA Graph         the same work captured once into a CUDA Graph and
                     replayed
  Agentic optimized  (formerly SOTA) the fastest implementation found
                     (<name>_sota.cu), free to change kernels and data
                     movement, launched whichever way is fastest
                     (SOTA_LAUNCH). It does the same work per iteration
                     (every timed iteration copies the full input from the
                     host and the full output back) and rotates over 3
                     different inputs, checking every output, so caching
                     across iterations cannot pass. llama2c is the exception:
                     its inputs stay device-resident in all versions.
Each version has three times:
  Full batch         one batch of back-to-back iterations between CUDA
                     events (time_all.sh: 50 iterations), in milliseconds
  Per iteration      full batch / iterations, in microseconds
  Single iteration   median of 20 iterations each launched alone on an idle
                     device, in microseconds. The device is drained before
                     each, so the launch cost is exposed instead of hiding
                     behind the previous iteration's work as in the batch.
Comparing the two shows how much launch overhead batching amortizes in each
launch mode (the Single / per-iteration columns). Whole-application rows read
them from time_all.sh's logs (median over its interleaved runs); kernel rows
have one ncu time, a single launch replayed in isolation, so they fill only
the Single iteration column.
Speedups: Graph = Direct / Graph; Agentic optimized = Graph / Agentic
optimized, where kernel rows compare the ncu time of all kernels doing that
row's work in one iteration.

Kernel rows use the ncu kernel duration (gpu__time_duration.sum, mean over
the profiled launches of that kernel). On a GPU without time_all.sh logs
(the RTX 4090 so far), whole-application rows fall back to a per-iteration
time only: the benchmark's own timing (cudaEvent per iteration / graph
replay) or, where only an nsys trace exists, the median GPU span of one
iteration in that trace: the work inside each NVTX range around a timed
iteration or, in traces from before the harnesses had NVTX ranges, each graph
replay. A cell whose measurement does not exist yet reads "pending".

Requires: ncu (Nsight Compute CLI) on PATH and openpyxl.
Usage:    python3 results/build_spreadsheet.py
"""

import bisect
import csv
import io
import json
import re
import sqlite3
import statistics
import subprocess
from pathlib import Path

from openpyxl import Workbook
from openpyxl.comments import Comment
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results" / "cuda_performance.xlsx"
MD_OUT = REPO / "results" / "README.md"
GITHUB = "https://github.com/ankaijin/cuda-elastic/blob"


# ------------------------------------------------------------------- GPUs ---

class GPU:
    """One GPU's results: the folder its outputs live in and, per input, the
    file to read: a glob, or a tuple of globs, whose match with the latest
    timestamp (YYYYMMDD-HHMMSS) in its name wins. Inputs not listed, or with no
    match, read as None."""

    def __init__(self, folder, label, toolkit, specs, files, notes):
        self.folder, self.label, self.toolkit = folder, label, toolkit
        self.specs, self.files, self.notes = specs, files, notes

    @property
    def sheet(self):
        return self.label.split(" (")[0]

    def file(self, key):
        patterns = self.files.get(key, ())
        if isinstance(patterns, str):
            patterns = (patterns,)
        matches = [m for p in patterns for m in (REPO / self.folder).glob(p)]
        if not matches:
            return None

        def stamp(path):
            found = re.findall(r"\d{8}-\d{6}", path.name)
            return (found[-1] if found else "", path.name)
        return str(max(matches, key=stamp).relative_to(REPO))


# Profile file name prefixes of the workloads with a --launch / PAGERANK_LAUNCH
# switch: (input key, tool, prefix). The profile scripts name their output
# <prefix><graph|direct>-<timestamp>.
LAUNCH_PROFILES = [
    ("loom", "ncu", "loom_msa/profiles/ncu-all-kernels-msa_optimized_large-"),
    ("loom", "nsys", "loom_msa/profiles/nsys-msa_optimized_large-"),
    ("pagerank", "ncu", "pagerank/profiles/ncu-pagerank-"),
    ("pagerank", "nsys", "pagerank/profiles/nsys-"),
    ("anomaly", "ncu", "mlperf-anomaly/profiles/ncu-anomaly_mse-"),
    ("anomaly", "nsys", "mlperf-anomaly/profiles/nsys-anomaly_mse-"),
    ("kws", "ncu", "mlperf-classification/profiles/ncu-kws_postprocess-"),
    ("kws", "nsys", "mlperf-classification/profiles/nsys-kws_postprocess-"),
    ("vww", "ncu", "mlperf-classification/profiles/ncu-vww_preprocess-"),
    ("vww", "nsys", "mlperf-classification/profiles/nsys-vww_preprocess-"),
]
EXTENSIONS = {"ncu": "ncu-rep", "nsys": "sqlite"}
MODES = ("graph", "direct")

# The SOTA binaries' launch mode (the faster one; see each <name>_sota.cu)
# and profile files: (input key, tool, file pattern with {stamp}).
SOTA_LAUNCH = {"llama": "graph", "loom": "direct", "pagerank": "direct",
               "anomaly": "graph", "kws": "direct", "vww": "direct"}
SOTA_PROFILES = [
    ("llama", "ncu", "llama2c/profiles/ncu-sota-large-b1024-graph-{stamp}.ncu-rep"),
    ("llama", "nsys", "llama2c/profiles/nsys-sota-b1024-graph-{stamp}.sqlite"),
    ("llama", "runs", "llama2c/profiles/sota-runs-b1024-{stamp}.txt"),
    ("loom", "ncu", "loom_msa/profiles/ncu-all-kernels-msa_sota_large_1chunk-direct-{stamp}.ncu-rep"),
    ("loom", "nsys", "loom_msa/profiles/nsys-msa_sota_large-direct-{stamp}.sqlite"),
    ("pagerank", "ncu", "pagerank/profiles/ncu-pagerank-sota-direct-{stamp}.ncu-rep"),
    ("pagerank", "nsys", "pagerank/profiles/nsys-sota-direct-{stamp}.sqlite"),
    ("anomaly", "ncu", "mlperf-anomaly/profiles/ncu-anomaly_mse_sota-graph-{stamp}.ncu-rep"),
    ("anomaly", "nsys", "mlperf-anomaly/profiles/nsys-anomaly_mse_sota-graph-{stamp}.sqlite"),
    ("kws", "ncu", "mlperf-classification/profiles/ncu-kws_postprocess_sota-direct-{stamp}.ncu-rep"),
    ("kws", "nsys", "mlperf-classification/profiles/nsys-kws_postprocess_sota-direct-{stamp}.sqlite"),
    ("vww", "ncu", "mlperf-classification/profiles/ncu-vww_preprocess_sota-direct-{stamp}.ncu-rep"),
    ("vww", "nsys", "mlperf-classification/profiles/nsys-vww_preprocess_sota-direct-{stamp}.sqlite"),
]


def sota_profiles(stamp):
    """Input keys <key>_sota_<tool> of SOTA_PROFILES; stamp None = the latest."""
    return {f"{key}_sota_{tool}": pattern.format(stamp=stamp or "*")
            for key, tool, pattern in SOTA_PROFILES}


def launch_profiles(stamp):
    """Input keys <key>_<tool>_<mode> of LAUNCH_PROFILES. stamp: one timestamp
    for every file, or None for the latest of each; profiles taken before the
    harnesses had a direct mode carry no mode in their names
    (<prefix>2026...) and count as graph runs."""
    files = {}
    for key, tool, prefix in LAUNCH_PROFILES:
        ext = EXTENSIONS[tool]
        for mode in MODES:
            if stamp:
                files[f"{key}_{tool}_{mode}"] = f"{prefix}{mode}-{stamp}.{ext}"
            else:
                files[f"{key}_{tool}_{mode}"] = (f"{prefix}{mode}-*.{ext}",) + (
                    (f"{prefix}2*.{ext}",) if mode == "graph" else ())
    return files


# time_all.sh's logs and the nsys traces of the same commands (time_all.sh
# nsys): <workload dir>/profiles/[nsys-]timing-<name>-<launch>-<stamp>.
TIMING_NAMES = {  # input key: (workload dir, baseline name, agentic optimized name)
    "llama": ("llama2c", "llama2c", "llama2c_sota"),
    "loom": ("loom_msa", "msa_optimized_large", "msa_sota_large"),
    "pagerank": ("pagerank", "pagerank", "pagerank_sota"),
    "anomaly": ("mlperf-anomaly", "anomaly_mse", "anomaly_mse_sota"),
    "kws": ("mlperf-classification", "kws_postprocess", "kws_postprocess_sota"),
    "vww": ("mlperf-classification", "vww_preprocess", "vww_preprocess_sota"),
}
SINGLE_ITERATIONS = 20  # each harness's kSingleIterations / kSingleReplays


def timing_files(stamp, nsys=True):
    """Input keys <key>_timing_<mode> and <key>_sota_timing (and, with nsys,
    the matching <key>_nsys_<mode> and <key>_sota_nsys traces, replacing the
    launch-mode profiles' traces); stamp None = the latest."""
    files = {}
    for key, (folder, name, sota) in TIMING_NAMES.items():
        runs = [(f"{key}_{{}}_{mode}", name, mode) for mode in MODES]
        runs.append((f"{key}_sota_{{}}", sota, SOTA_LAUNCH[key]))
        for pattern, binary, mode in runs:
            base = f"{folder}/profiles/{{}}timing-{binary}-{mode}-{stamp or '*'}"
            files[pattern.format("timing")] = base.format("") + ".log"
            if nsys:
                files[pattern.format("nsys")] = base.format("nsys-") + ".sqlite"
    return files


SOTA_STAMP_RTX3090 = "20261016-145900"
TIMING_STAMP_RTX3090 = "20261020-161213"

# Published specs. Both GPUs have a 384-bit GDDR6X bus and a PCIe 4.0 x16 link.
RTX3090 = GPU("rtx3090", "RTX 3090 (sm_86, 24 GB)", "CUDA 13.3",
              {"sms": 82, "boost_ghz": 1.695, "dram_gbps": 19.5}, {
    "llama_sweep": "llama2c/profiles/sweep-fill-20261002-152612-speedup.csv",
    "llama_nsys_graph": "llama2c/profiles/nsys-b1024-graph-20261014-215130.sqlite",
    "llama_nsys_direct": "llama2c/profiles/nsys-b1024-direct-20261014-215130.sqlite",
    "llama_ncu_graph": "llama2c/profiles/ncu-large-coalesced-b1024-graph-20261014-215130.ncu-rep",
    "llama_ncu_direct": "llama2c/profiles/ncu-large-coalesced-b1024-direct-20261014-215130.ncu-rep",
    "loom_benchmark": "loom_msa/build/benchmark.json",
    "loom_benchmark_txt": "loom_msa/build/benchmark.txt",
    **launch_profiles("20261015-224015"),
    **sota_profiles(SOTA_STAMP_RTX3090),
    **timing_files(TIMING_STAMP_RTX3090),
}, {})

RTX4090 = GPU("rtx4090", "RTX 4090 (sm_89, 24 GB)", "CUDA 12.5",
              {"sms": 128, "boost_ghz": 2.52, "dram_gbps": 21.0}, {
    "llama_sweep": "llama2c/profiles/sweep-fill-*-speedup.csv",
    "llama_nsys_graph": "llama2c/profiles/nsys-node-graph-*.sqlite",
    "llama_nsys_direct": "llama2c/profiles/nsys-node-direct-*.sqlite",
    "llama_ncu_graph": "llama2c/profiles/ncu-large-coalesced-b1024-graph-*.ncu-rep",
    "llama_ncu_direct": "llama2c/profiles/ncu-large-coalesced-b1024-direct-*.ncu-rep",
    "loom_benchmark": "loom_msa/build/benchmark.json",
    "loom_benchmark_txt": "loom_msa/build/benchmark.txt",
    **launch_profiles(None),
    **sota_profiles(None),
    **timing_files(None, nsys=False),
}, {
    "pagerank_timing": "A separate uninstrumented run of the binary "
                       "(profiles/bench-20261015-091815.txt): 5.714 ms with transfers, "
                       "2.651 ms compute only.",
})

GPUS = [RTX3090, RTX4090]
BASELINE = RTX3090  # the other GPUs' sheets compare against it


# ---------------------------------------------------------------- loaders ---

def git(*args):
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True)


def link(path):
    """GitHub URL for path, pinned to the latest commit on origin/main that
    has the file, under the name it had there (files moved since, e.g. into
    rtx3090/, keep their original links); files not merged yet link to main."""
    log = git("log", "--follow", "--name-only", "--format=%x00%H", "--", path).stdout
    for entry in log.split("\0")[1:]:
        commit, _, name = entry.strip().partition("\n")
        if git("merge-base", "--is-ancestor", commit, "origin/main").returncode == 0:
            return f"{GITHUB}/{commit}/{name.strip()}"
    return f"{GITHUB}/main/{path}"


_ncu_cache = {}


def ncu_rows(path):
    """All profiled launches in an ncu report, as {metric: (value, unit)} dicts."""
    if path not in _ncu_cache:
        text = subprocess.run(["ncu", "--import", str(REPO / path), "--csv", "--page", "raw"],
                              check=True, capture_output=True, text=True).stdout
        rows = list(csv.reader(io.StringIO(text)))
        header, units = rows[0], rows[1]
        _ncu_cache[path] = [
            {h: (v, u) for h, u, v in zip(header, units, row)} for row in rows[2:]
        ]
    return _ncu_cache[path]


def number(value):
    return float(value.replace(",", ""))


TIME_SCALE_TO_US = {"ns": 1e-3, "nsecond": 1e-3, "us": 1.0, "usecond": 1.0,
                    "ms": 1e3, "msecond": 1e3, "s": 1e6, "second": 1e6}


def kernel_name(row):
    return row["Kernel Name"][0].split("(")[0].split("::")[-1].removeprefix("void ")


def launch_us(row):
    v, unit = row["gpu__time_duration.sum"]
    return number(v) * TIME_SCALE_TO_US[unit]


def kernel_launches(path, kernel, grid=None):
    """Launches of `kernel` (name prefix after any namespace) in an ncu report."""
    return [r for r in ncu_rows(path)
            if kernel_name(r).startswith(kernel)
            and (grid is None or number(r["launch__grid_size"][0]) == grid)]


def ncu_kernel_us(path, kernel, grid=None):
    launches = kernel_launches(path, kernel, grid)
    if not launches:
        raise ValueError(f"{kernel} (grid={grid}) not found in {path}")
    times = [launch_us(r) for r in launches]
    return statistics.mean(times), len(times)


def ncu_time_shares(path):
    """Percent of the report's summed kernel time per kernel name."""
    totals = {}
    for r in ncu_rows(path):
        totals[kernel_name(r)] = totals.get(kernel_name(r), 0) + launch_us(r)
    return {k: 100 * v / sum(totals.values()) for k, v in totals.items()}


def nsys_node_replay_us(path):
    """Median span of CUDA Graph replays that include memcpy nodes, in an nsys
    trace taken with --cuda-graph-trace=node (no per-graph trace rows)."""
    with sqlite3.connect(REPO / path) as db:
        launches = {r[0] for r in db.execute(
            "SELECT r.correlationId FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
            "JOIN StringIds s ON s.id = r.nameId WHERE s.value LIKE 'cudaGraphLaunch%'")}
        spans = {}
        for table in ("CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_MEMCPY"):
            for corr, start, end in db.execute(
                    f"SELECT correlationId, start, end FROM {table} WHERE graphNodeId IS NOT NULL"):
                span = spans.setdefault(corr, [start, end, False])
                span[0], span[1] = min(span[0], start), max(span[1], end)
                span[2] |= table.endswith("MEMCPY")
    durations = [(end - start) / 1e3 for corr, (start, end, copies) in spans.items()
                 if corr in launches and copies]
    return statistics.median(durations), len(durations)


def nsys_nvtx_span_us(path, range_name):
    """Median GPU span (first start to last end) of the work enqueued inside
    each NVTX push/pop range named range_name, in an nsys trace taken with
    --trace=cuda,nvtx, and the number of ranges; None when the trace has no
    such range. A range's work is every kernel, memcpy and whole-graph
    execution whose correlationId belongs to a runtime API call made on the
    range's thread while it was open: the launches and copies of a direct
    iteration, or a cudaGraphLaunch (its nodes with --cuda-graph-trace=node,
    the whole graph with =graph)."""
    with sqlite3.connect(REPO / path) as db:
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        if "NVTX_EVENTS" not in tables:
            return None
        ranges = db.execute(
            "SELECT n.start, n.end, n.globalTid FROM NVTX_EVENTS n "
            "LEFT JOIN StringIds s ON s.id = n.textId "
            "WHERE n.eventType = 59 AND COALESCE(n.text, s.value) = ?", (range_name,)).fetchall()
        if not ranges:
            return None
        calls = {}
        for start, tid, corr in db.execute(
                "SELECT start, globalTid, correlationId FROM CUPTI_ACTIVITY_KIND_RUNTIME "
                "ORDER BY start"):
            starts, corrs = calls.setdefault(tid, ([], []))
            starts.append(start)
            corrs.append(corr)
        work = {}
        for table in ("CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_MEMCPY",
                      "CUPTI_ACTIVITY_KIND_GRAPH_TRACE"):
            if table in tables:
                for corr, start, end in db.execute(
                        f"SELECT correlationId, start, end FROM {table}"):
                    work.setdefault(corr, []).append((start, end))
    spans = []
    for start, end, tid in ranges:
        starts, corrs = calls.get(tid, ([], []))
        first, last = bisect.bisect_left(starts, start), bisect.bisect_right(starts, end)
        intervals = [w for corr in corrs[first:last] for w in work.get(corr, [])]
        if intervals:
            spans.append((max(e for _, e in intervals) - min(s for s, _ in intervals)) / 1e3)
    return statistics.median(spans), len(spans)


def nsys_iteration(path, range_name):
    """(median us, count, unit) of one end-to-end iteration in an nsys trace:
    the NVTX ranges named range_name where the trace has them, else (traces
    from before the harnesses had NVTX ranges) the graph replays with copies."""
    nvtx = nsys_nvtx_span_us(path, range_name)
    if nvtx:
        return nvtx + (f"{range_name} NVTX ranges",)
    return nsys_node_replay_us(path) + ("graph replays with copies",)


def iteration_note(measured):
    """Note for an end-to-end row from nsys_iteration's result per mode."""
    counts = " / ".join(str(measured[m][1]) for m in measured)
    modes = " (graph / direct)" if len(measured) > 1 else ""
    units = {measured[m][2] for m in measured}
    unit = units.pop() if len(units) == 1 else " / ".join(measured[m][2] for m in measured)
    return f"Median GPU span of the {counts}{modes} {unit} in the nsys traces."


def nsys_log_ms_per_iter(sqlite_path):
    """gpu_ms_per_iter the binary printed during an nsys run (the .log beside the trace)."""
    log = (REPO / sqlite_path).with_suffix(".log")
    return float(re.search(r"gpu_ms_per_iter=([\d.]+)", log.read_text()).group(1))


def loom_benchmark(path):
    runs = json.loads((REPO / path).read_text())
    return {(r["shape"], r["inputs"]): r["summary"] for r in runs}


def sweep_row(path, **match):
    with open(REPO / path) as f:
        for row in csv.DictReader(f):
            if all(row[k] == str(v) for k, v in match.items()):
                return row
    raise ValueError(f"{match} not in {path}")


# Per-run results in a time_all.sh log, by the line each harness prints:
# (regex, full batch scale to us, single scale to us). The groups are full
# batch, iterations in it, single iteration.
TIMING_LINES = [
    # llama2c: key=value fields
    (r"launches=(\d+) .*?gpu_ms=([\d.]+) .*?single_us=([\d.]+)", None, 1.0),
    # loom_msa: metrics: ... replays=N ... batch_ms=X single_us=Y
    (r"^metrics: .*?replays=(\d+) .*?batch_ms=([\d.]+) single_us=([\d.]+)", None, 1.0),
    # pagerank
    (r"^batch: ([\d.]+) ms for (\d+) back-to-back .*?single: ([\d.]+) ms", 1e3, 1e3),
    # mlperf: two lines
    (r"^  batch +([\d.]+) ms \((\d+) back-to-back[^\n]*\n  single +([\d.]+) us", 1e3, 1.0),
]


def timing_runs(path):
    """[(full batch us, iterations, single iteration us)] per run in a
    time_all.sh log."""
    text = (REPO / path).read_text()
    runs = []
    for run in text.split("### run ")[1:]:
        for pattern, batch_scale, single_scale in TIMING_LINES:
            m = re.search(pattern, run, re.M)
            if not m:
                continue
            if batch_scale is None:  # (iterations, batch ms, single)
                n, batch, single = int(m[1]), float(m[2]) * 1e3, float(m[3])
            else:
                batch, n, single = float(m[1]) * batch_scale, int(m[2]), float(m[3])
            runs.append((batch, n, single * single_scale))
            break
        else:
            raise ValueError(f"no batch / single timing in a run of {path}")
    if len({n for _, n, _ in runs}) != 1:
        raise ValueError(f"runs of {path} differ in batch size")
    return runs


# ------------------------------------------------------------------ cells ---

class Cell:
    """One measurement and the files behind it. Kernel cells hold one ncu time
    (us = single_us); whole-application cells hold the time per iteration in a
    batch (us) and, where time_all.sh ran, the full batch and the single-
    iteration time."""

    def __init__(self, us=None, ncu=None, nsys=None, source=None, empty="", predict=None,
                 kernel=False, batch_us=None, batch_n=None, single_us=None, runs=None):
        self.us, self.ncu, self.nsys, self.source = us, ncu, nsys, source
        self.empty, self.kernel = empty, kernel
        self.batch_us, self.batch_n, self.runs = batch_us, batch_n, runs
        self.single_us = us if kernel else single_us
        # Agentic optimized cells: the CUDA Graph version's cell (whole
        # application) or summed kernel time (kernel rows) for the same work.
        self.vs = None
        # How to predict this value on another GPU from this GPU's ncu report:
        # ("kernel", report, kernel, grid) or ("app", report, runs in report).
        self.predict = predict

    def value_file(self):
        """The file the value itself was read from."""
        return self.ncu or self.source or self.nsys

    def batched(self):
        """Per-iteration time in a timed batch, if this cell has one."""
        return self.us if self.batch_us is not None else None

    def overhead(self):
        """Single iteration / per iteration in the batch."""
        return ratio(self.single_us, self.batched())

    def agentic_speedups(self):
        """(full batch, single iteration) speedup over the CUDA Graph version."""
        if isinstance(self.vs, Cell):
            return ratio(self.vs.batched(), self.batched()), ratio(self.vs.single_us, self.single_us)
        return None, ratio(self.vs, self.single_us)


def ratio(a, b):
    return a / b if a is not None and b is not None else None


def graph_speedups(direct, graph):
    """(full batch, single iteration) speedup of the CUDA Graph over Direct."""
    return ratio(direct.batched(), graph.batched()), ratio(direct.single_us, graph.single_us)


NCU_PENDING = "pending (ncu not run yet)"
DIRECT_PENDING = "pending (direct mode not profiled yet)"
SOTA_PENDING = "pending (agentic optimized not profiled yet)"
TIMING_PENDING = "pending (time_all.sh not run yet)"
PENDING = (NCU_PENDING, DIRECT_PENDING, SOTA_PENDING, TIMING_PENDING)


def ncu_cell(path, kernel, grid=None, nsys=None, pending=NCU_PENDING):
    if path is None:
        return Cell(nsys=nsys, empty=pending, kernel=True)
    return Cell(ncu_kernel_us(path, kernel, grid)[0], ncu=path, nsys=nsys, kernel=True,
                predict=("kernel", path, kernel, grid))


def ncu_iteration_us(path, kernels, iterations=1, grid=None):
    """Summed ncu time of every launch of `kernels` (name prefixes) in a
    report covering `iterations` iterations, per iteration."""
    launches = [r for k in kernels for r in kernel_launches(path, k, grid)]
    if not launches:
        raise ValueError(f"{kernels} (grid={grid}) not found in {path}")
    return sum(launch_us(r) for r in launches) / iterations


def graph_kernels_us(cell, kernels):
    """CUDA Graph time of the baseline kernels an agentic optimized kernel row
    replaces: the sum of each kernel's mean launch time in the graph cell's
    report."""
    if cell.ncu is None:
        return None
    return sum(ncu_kernel_us(cell.ncu, k)[0] for k in kernels)


def sota_kernel_cell(gpu, key, kernels, vs, iterations=1, grid=None):
    path, nsys = gpu.file(f"{key}_sota_ncu"), gpu.file(f"{key}_sota_nsys")
    if path is None:
        cell = Cell(nsys=nsys, empty=SOTA_PENDING, kernel=True)
    else:
        cell = Cell(ncu_iteration_us(path, kernels, iterations, grid), ncu=path, nsys=nsys,
                    kernel=True)
    cell.vs = vs
    return cell


def timed_cell(log, nsys, predict=None):
    """Whole-application cell from a time_all.sh log: the medians over its
    runs of the full batch and the single-iteration time."""
    runs = timing_runs(log)
    batch = statistics.median(b for b, _, _ in runs)
    n = runs[0][1]
    return Cell(batch / n, source=log, nsys=nsys, predict=predict, batch_us=batch, batch_n=n,
                single_us=statistics.median(s for _, _, s in runs), runs=len(runs))


def app_cells(gpu, key, predict, fallback):
    """Whole-application cells per launch mode: from the GPU's time_all.sh
    logs where it has them, else fallback(mode) (a per-iteration time only)."""
    cells = {}
    for m in MODES:
        log, nsys = gpu.file(f"{key}_timing_{m}"), gpu.file(f"{key}_nsys_{m}")
        cells[m] = timed_cell(log, nsys, predict(m)) if log else fallback(m)
    return cells


def sota_app_cell(gpu, key, vs, fallback=None):
    """Agentic optimized whole-application cell: from its time_all.sh log, else
    fallback() if given, else pending."""
    log, nsys = gpu.file(f"{key}_sota_timing"), gpu.file(f"{key}_sota_nsys")
    if log:
        cell = timed_cell(log, nsys)
    elif fallback:
        cell = fallback()
    else:
        cell = Cell(nsys=nsys, empty=SOTA_PENDING)
    cell.vs = vs
    return cell


def timing_note(cell, what):
    """Note for a whole-application row timed by time_all.sh."""
    return (f"{what} Full batch: {cell.batch_n} back-to-back iterations between CUDA events "
            f"(per iteration = full batch / {cell.batch_n}). Single iteration: median of "
            f"{SINGLE_ITERATIONS} iterations, each launched alone on an idle device, so its "
            "launch cost is not hidden behind the previous iteration's work. Each time is the "
            f"median of {cell.runs} interleaved runs (time_all.sh).")


def sota_note(key, text):
    return f" Agentic optimized (launch={SOTA_LAUNCH[key]}): {text}"


# ------------------------------------------------------------------- rows ---
# Each row: (workload, kernel/stage, problem size, metric, direct, graph,
# agentic optimized, notes), the cells in column order. One problem size per
# workload (the largest tested): the dominant kernel, when one kernel
# dominates, and the whole application.

KERNEL_TIME = "Kernel time (ncu, one launch)"
WHOLE_APP = "Whole application"
TIMED = "Batch and single iteration (cudaEvent)"


def app_metric(cells, fallback):
    return TIMED if cells["graph"].batch_us is not None else fallback


def share(kernel, app, launches=1):
    if kernel.us is None or app.us is None:
        return ""
    return f"{100 * launches * kernel.us / app.us:.0f}% of the whole-application time per iteration."


def llama2c_rows(gpu):
    w = "llama2c (coalesced matmul)"
    size = "batch=1024, dim=16384, rows=8192"
    nsys = {m: gpu.file(f"llama_nsys_{m}") for m in MODES}
    ncu = {m: gpu.file(f"llama_ncu_{m}") for m in MODES}
    sweep = gpu.file("llama_sweep")

    def from_sweep(m):
        row = sweep_row(sweep, batch=1024, size=16384, rows=8192, repeats=1)
        # The ncu report covers 3 iterations.
        return Cell(float(row[f"{m}_gpu_ms"]) * 1e3, source=sweep, nsys=nsys[m],
                    predict=("app", ncu[m], 3))
    app = app_cells(gpu, "llama", lambda m: ("app", ncu[m], 3), from_sweep)
    matmul = {m: ncu_cell(ncu[m], "matmul_coalesced_kernel", nsys=nsys[m]) for m in MODES}
    if ncu["graph"]:
        s = ncu_time_shares(ncu["graph"])
        largest = (f"Largest kernel ({s['matmul_coalesced_kernel']:.0f}% of serialized kernel "
                   f"time under ncu; rmsnorm {s['rmsnorm_kernel']:.0f}%, softmax "
                   f"{s['softmax_kernel']:.0f}%). ")
    else:
        largest = "Largest kernel. "
    kernels = ("rmsnorm_kernel", "softmax_kernel", "matmul_coalesced_kernel")
    # The agentic optimized ncu report covers 3 iterations of one fused launch each.
    sota_matmul = sota_kernel_cell(gpu, "llama", ("llama2c_sota_kernel",),
                                   graph_kernels_us(matmul["graph"], kernels), iterations=3)
    runs = gpu.file("llama_sota_runs")

    def from_runs():
        times = [float(t) for t in re.findall(r"gpu_ms_per_iter=([\d.]+)", (REPO / runs).read_text())]
        return Cell(1e3 * statistics.mean(times), source=runs, nsys=gpu.file("llama_sota_nsys"))
    sota_app = sota_app_cell(gpu, "llama", app["graph"], from_runs if runs else None)
    if app["graph"].batch_us is not None:
        app_note = timing_note(app["graph"], "One iteration: rmsnorm + softmax + matmul on "
                               "device-resident data, no per-iteration copies.")
    else:
        reruns = " / ".join(f"{1e3 * nsys_log_ms_per_iter(nsys[m]):.1f}" for m in MODES)
        app_note = ("Mean of 3 interleaved trials from the GPU-filling sweep; no per-iteration "
                    f"copies. The nsys traces rerun this configuration ({reruns} us per "
                    "iteration, graph / direct).")
    return [
        (w, "matmul_coalesced_kernel", size, KERNEL_TIME, matmul["direct"], matmul["graph"],
         sota_matmul,
         largest + "It is a matrix-vector product, so its time does not grow with batch; the "
         "three kernels run concurrently on separate streams, so their times overlap. "
         + share(matmul["graph"], app["graph"])
         + sota_note("llama", "one fused kernel (llama2c_sota_kernel, one block per 64 KiB row, "
                     "each row read once from registers) computes rmsnorm, softmax and the "
                     "matmul, so its time is compared with the three graph kernels' summed ncu "
                     + (f"time ({sota_matmul.vs:,.1f} us)." if sota_matmul.vs else "time."))),
        (w, f"{WHOLE_APP} (rmsnorm + softmax + matmul iteration)", size,
         app_metric(app, "Iteration time (cudaEvent)"), app["direct"], app["graph"], sota_app,
         app_note + sota_note("llama", "llama2c_sota, the same device-resident nocopy "
                              "configuration, one fused kernel per iteration.")),
    ]


def loom_rows(gpu):
    w = "loom_msa (multi-sensor attention)"
    size = "25,000 queries x 250,000 context"
    nsys = {m: gpu.file(f"loom_nsys_{m}") for m in MODES}
    ncu = {m: gpu.file(f"loom_ncu_{m}") for m in MODES}
    bench = gpu.file("loom_benchmark")
    summary = loom_benchmark(bench)[("1x25000x250000", "random")] if bench else {}

    def from_benchmark(m):
        run = summary.get("optimized" if m == "graph" else f"optimized_{m}")
        return (Cell(run["end_to_end_us"]["median"], source=gpu.file("loom_benchmark_txt"),
                     nsys=nsys[m], predict=("app", ncu[m], 1)) if run
                else Cell(nsys=nsys[m], empty=DIRECT_PENDING))
    app = app_cells(gpu, "loom", lambda m: ("app", ncu[m], 1), from_benchmark)
    attention = {m: ncu_cell(ncu[m], "fuse_attention_kernel", nsys=nsys[m],
                             pending=NCU_PENDING if m == "graph" or nsys[m] else DIRECT_PENDING)
                 for m in MODES}
    kernels = ("project_audio_kernel", "project_imu_kernel", "fuse_attention_kernel",
               "reduce_statistics_kernel")
    sota_attention = sota_kernel_cell(gpu, "loom", ("msa_sota_attention_kernel",),
                                      graph_kernels_us(attention["graph"], kernels))
    launch = SOTA_LAUNCH["loom"]
    run = summary.get("sota" if launch == "graph" else f"sota_{launch}")
    sota_app = sota_app_cell(gpu, "loom", app["graph"], (lambda: Cell(
        run["end_to_end_us"]["median"], source=gpu.file("loom_benchmark_txt"),
        nsys=gpu.file("loom_sota_nsys"))) if run else None)
    if app["graph"].batch_us is not None:
        app_note = timing_note(app["graph"], "msa_optimized_large on random inputs, including "
                               "the H2D input and D2H output copies.")
    else:
        modes = "both launch modes" if app["direct"].us is not None else "CUDA graph only"
        app_note = (f"benchmark.py: median of 5 runs, random inputs, {modes}. Includes H2D input "
                    "and D2H output copies.")
    return [
        (w, "fuse_attention_kernel", size, KERNEL_TIME, attention["direct"], attention["graph"],
         sota_attention,
         "Single-pass online softmax, shared-memory context tiles, 8-way context split "
         "(FlashAttention-style). " + share(attention["graph"], app["graph"])
         + sota_note("loom", "msa_sota_attention_kernel (13-way context split, 8 queries per "
                     "thread) with the projections folded into its loads and the split merge and "
                     "statistics into its last block. msa_sota_large runs it as 13 concurrent "
                     "launches, one per imu copy chunk, which ncu would serialize, so ncu profiles "
                     "msa_sota_large_1chunk, the same kernel as one launch. Compared with the four "
                     "graph kernels' summed ncu "
                     + (f"time ({sota_attention.vs:,.1f} us)." if sota_attention.vs else "time."))),
        (w, f"{WHOLE_APP} (H2D + kernels + D2H)", size, app_metric(app, "Iteration time (cudaEvent)"),
         app["direct"], app["graph"], sota_app,
         app_note + sota_note("loom", "msa_sota_large, rotating over 3 random input sets; "
                              "output written straight to pinned host memory.")),
    ]


def pagerank_rows(gpu):
    w = "pagerank (GAPBS)"
    size = "2^20 vertices, 16 edges/vertex, 20 iterations"
    ncu = {m: gpu.file(f"pagerank_ncu_{m}") for m in MODES}
    nsys = {m: gpu.file(f"pagerank_nsys_{m}") for m in MODES}
    measured = {}

    def from_nsys(m):
        if not nsys[m]:
            return Cell(empty=DIRECT_PENDING)
        measured[m] = nsys_iteration(nsys[m], "pagerank.full")
        return Cell(measured[m][0], nsys=nsys[m], predict=("app", ncu[m], 1))
    app = app_cells(gpu, "pagerank", lambda m: ("app", ncu[m], 1), from_nsys)
    update = {m: ncu_cell(ncu[m], "pagerank_update", grid=32768, nsys=nsys[m],
                          pending=NCU_PENDING if m == "graph" or nsys[m] else DIRECT_PENDING)
              for m in MODES}
    if ncu["graph"]:
        chunks = [launch_us(r) for r in kernel_launches(ncu["graph"], "pagerank_update")
                  if number(r["launch__grid_size"][0]) != 32768]
        first = (f"the first iteration runs as {len(chunks)} upload-overlapped chunks of "
                 f"~{statistics.mean(chunks):.0f} us each. 19 full iterations = ")
    else:
        first = "the first iteration runs as 8 upload-overlapped chunks. "
    if gpu.file("pagerank_sota_ncu"):
        # Iterations 1-19 (the first runs on the CSR during the upload); the
        # last of them runs as two launches, one wave over half the parts each.
        sota_update = sota_kernel_cell(gpu, "pagerank", ("binned_update_kernel",),
                                       update["graph"].us, iterations=19)
    else:
        sota_update = Cell(empty=SOTA_PENDING, kernel=True)
    sota_nsys = gpu.file("pagerank_sota_nsys")

    def sota_from_nsys():
        if not sota_nsys:
            return Cell(empty=SOTA_PENDING)
        return Cell(nsys_iteration(sota_nsys, "pagerank.full")[0], nsys=sota_nsys)
    sota_app = sota_app_cell(gpu, "pagerank", app["graph"], sota_from_nsys)
    what = ("One iteration is one PageRank run: the CSR graph uploaded from pinned memory in 8 "
            "chunks overlapped with the first of 20 iterations, then the scores downloaded.")
    if app["graph"].batch_us is not None:
        app_note = timing_note(app["graph"], what)
    else:
        app_note = (iteration_note(measured) + " The CSR graph is uploaded from pinned memory in 8 "
                    "chunks overlapped with the first iteration. " + gpu.notes["pagerank_timing"])
    return [
        (w, "pagerank_update (one iteration)", size, KERNEL_TIME, update["direct"], update["graph"],
         sota_update,
         "Pull gather of precomputed contributions. Full-graph iterations only; "
         + first + share(update["graph"], app["graph"], 19)
         + sota_note("pagerank", "binned_update_kernel, summed over iterations 1-19 / 19: a "
                     "cooperative pull over destination parts that fit in shared memory, with "
                     "edges sorted by source sector on the GPU during the upload (5.4M instead "
                     "of 16.8M gathered sectors per iteration).")),
        (w, f"{WHOLE_APP} (H2D + 20 iterations + D2H)", size,
         app_metric(app, "Iteration time (nsys)"), app["direct"], app["graph"], sota_app,
         app_note + sota_note("pagerank", "rotating over 3 inputs (the benchmark graph and two "
                              "random graphs of the same size); includes the on-GPU edge sort.")),
    ]


def mlperf_rows(gpu):
    rows = []
    for w, kernel, size, key, replaced, sota_kernel, kernel_text, app_text in (
            ("mlperf-tiny anomaly", "mse_partial_kernel", "64M elements", "anomaly",
             ("mse_partial_kernel", "mse_finalize_kernel"), "mse_sota_kernel",
             "mse_sota_kernel, one single-pass reduction (the last block finishes the sum), "
             "compared with mse_partial_kernel + mse_finalize_kernel.",
             "the float input is copied in shrinking chunks while a second stream reduces each "
             "chunk as it lands; the result is written straight to pinned host memory."),
            ("mlperf-tiny KWS", "kws_postprocess_kernel", "12 classes x 4M rows", "kws",
             ("kws_postprocess_kernel",), "kws_postprocess_sota_kernel",
             "kws_postprocess_sota_kernel on device buffers (kernels-only), one row per thread "
             "with rows staged through shared memory for coalesced 16 B loads and stores.",
             "one zero-copy kernel reads the int8 rows from and writes the probabilities and "
             "argmax to mapped pinned host memory, so both PCIe directions stream at once."),
            ("mlperf-tiny VWW", "vww_preprocess_kernel", "256M bytes", "vww",
             ("vww_preprocess_kernel",), "vww_preprocess_sota_kernel",
             "vww_preprocess_sota_kernel on device buffers (kernels-only), four 16 B vectors "
             "per thread.",
             "64 chunks of 4 MiB on separate copy-in, compute and copy-out streams, so H2D and "
             "D2H overlap.")):
        ncu = {m: gpu.file(f"{key}_ncu_{m}") for m in MODES}
        nsys = {m: gpu.file(f"{key}_nsys_{m}") for m in MODES}
        measured = {}

        def from_nsys(m, key=key, ncu=ncu, nsys=nsys, measured=measured):
            if not nsys[m]:
                return Cell(empty=DIRECT_PENDING)
            measured[m] = nsys_iteration(nsys[m], f"{key}.end_to_end")
            return Cell(measured[m][0], nsys=nsys[m], predict=("app", ncu[m], 1))
        app = app_cells(gpu, key, lambda m, ncu=ncu: ("app", ncu[m], 1), from_nsys)
        k = {m: ncu_cell(ncu[m], kernel, nsys=nsys[m],
                         pending=NCU_PENDING if m == "graph" or nsys[m] else DIRECT_PENDING)
             for m in MODES}
        only = "" if kernel == "mse_partial_kernel" else "Only kernel in the application. "
        sota_k = sota_kernel_cell(gpu, key, (sota_kernel,), graph_kernels_us(k["graph"], replaced))
        sota_nsys = gpu.file(f"{key}_sota_nsys")

        def sota_from_nsys(key=key, sota_nsys=sota_nsys):
            if not sota_nsys:
                return Cell(empty=SOTA_PENDING)
            return Cell(nsys_iteration(sota_nsys, f"{key}.end_to_end")[0], nsys=sota_nsys)
        sota_app = sota_app_cell(gpu, key, app["graph"], sota_from_nsys)
        if app["graph"].batch_us is not None:
            app_note = timing_note(app["graph"], "Random inputs, including the H2D and D2H copies.")
        else:
            app_note = iteration_note(measured)
        rows += [(w, kernel, size, KERNEL_TIME, k["direct"], k["graph"], sota_k,
                  only + share(k["graph"], app["graph"]) + sota_note(key, kernel_text)),
                 (w, f"{WHOLE_APP} (H2D + kernels + D2H)", size,
                  app_metric(app, "Iteration time (nsys)"), app["direct"], app["graph"], sota_app,
                  app_note + sota_note(key, "rotating over 3 inputs; " + app_text))]
    return rows


def gpu_rows(gpu):
    rows = llama2c_rows(gpu) + loom_rows(gpu) + pagerank_rows(gpu) + mlperf_rows(gpu)
    # Drop the agentic optimized part of the notes where it has no measurement yet.
    return [(*r[:-1], r[-1].split(" Agentic optimized (launch=")[0].rstrip()
             if r[6].us is None else r[-1]) for r in rows]


# ------------------------------------------------------------ predictions ---
# Predicted speedup of a GPU over BASELINE, from BASELINE's ncu reports and the
# two GPUs' published specs; METHOD explains it.

BOUND_PCT = 60  # a kernel below this on every unit is latency-bound

METHOD = [
    "The predicted speedup uses only the RTX 3090's ncu reports and the two GPUs' published "
    "specs (RTX 3090: 82 SMs, 1.695 GHz boost, 19.5 Gbps GDDR6X; RTX 4090: 128 SMs, 2.52 GHz "
    "boost, 21 Gbps GDDR6X; both 384-bit, PCIe 4.0 x16), so it is fixed before any 4090 "
    "measurement. Each unit's peak scales by its hardware ratio: SM compute by SMs x SM clock, "
    "L1/L2 cache throughput assumed to scale the same way, DRAM by memory data rate "
    "(the 3090's own SM and memory clocks come from its ncu report: 1.69 GHz and 18.98 Gbps, "
    "so the ratios are about 2.32x and 1.11x).",
    f"Kernel rows: ncu gives each launch's SM, memory (L1/L2) and DRAM throughput as a percent "
    f"of the 3090's peak. Keeping the work per unit fixed, a unit at u% that is r times faster "
    f"needs u/r% of the old time, and the slowest unit sets the new time: speedup = max(u) / "
    f"max(u / r). So a DRAM-bound kernel gets ~1.11x and a compute- or cache-bound one ~2.32x. "
    f"A launch below {BOUND_PCT}% on every unit is latency-bound and gets the SM clock ratio "
    f"(1.49x). A kernel with several launches combines them weighted by time.",
    "Whole-application rows: the kernels in that workload's ncu report (one run of the "
    "application) scale by their predicted speedups; the rest of the 3090 time (H2D/D2H copies "
    "over the same PCIe 4.0 link, launch gaps) is assumed unchanged. Where the kernels overlap "
    "(llama2c's three streams; their summed time exceeds the iteration time), the whole time "
    "scales with the kernels.",
    "Not modelled: cache capacity (the 4090's L2 is 72 MB vs 6 MB, so data that spilled to DRAM "
    "on the 3090 can stay in L2), kernels overlapping copies (pagerank's first iteration), "
    "clocks above rated boost, and compiler differences (CUDA 13.3 vs 12.5). Measured = "
    "3090 time / 4090 time for the CUDA Graph column; kernel rows read pending until the 4090 "
    "ncu reports exist.",
]


def pct(row, metric):
    return number(row[metric][0])


def launch_speedup(row, target):
    """Predicted speedup on `target` of one launch in a BASELINE ncu report,
    and the reason."""
    base, new = BASELINE.specs, target.specs
    sm_ghz = number(row["gpc__cycles_elapsed.avg.per_second"][0])
    dram_gbps = 2 * number(row["dram__cycles_elapsed.avg.per_second"][0])  # GDDR6X: 2 bits/cycle
    compute = new["sms"] * new["boost_ghz"] / (base["sms"] * sm_ghz)
    units = {  # percent of the baseline's peak, and how much faster that peak is on target
        "SM": (pct(row, "sm__throughput.avg.pct_of_peak_sustained_elapsed"), compute),
        "L1/L2": (pct(row, "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"),
                  compute),
        "DRAM": (pct(row, "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed"),
                 new["dram_gbps"] / dram_gbps),
    }
    busiest = max(units, key=lambda k: units[k][0])
    peak = units[busiest][0]
    if peak < BOUND_PCT:
        clock = new["boost_ghz"] / sm_ghz
        return clock, (f"Latency-bound (busiest unit {busiest} {peak:.0f}% < {BOUND_PCT}%): "
                       f"SM clock {new['boost_ghz']} / {sm_ghz:.2f} GHz = {clock:.2f}x.")
    limit = max(units, key=lambda k: units[k][0] / units[k][1])
    u, r = units[limit]
    why = {
        "SM": f"SMs x clock {new['sms']} x {new['boost_ghz']} / ({base['sms']} x {sm_ghz:.2f}) "
              f"GHz = {r:.2f}x",
        "L1/L2": f"assumed to scale with SMs x clock, {r:.2f}x",
        "DRAM": f"data rate {new['dram_gbps']:g} / {dram_gbps:.2f} Gbps = {r:.2f}x",
    }[limit]
    others = ", ".join(f"{k} {v[0]:.0f}%" for k, v in units.items() if k != limit)
    text = f"{limit}-bound ({limit} {u:.0f}% busy; {others}): {why}"
    if u < peak:
        text += f"; {busiest} is busier but speeds up more, so {limit} limits"
    return peak / (u / r), text + "."


def predicted_speedup(cell, target):
    """(speedup, reason) of a BASELINE cell on `target`, or None."""
    if cell.predict is None or cell.predict[1] is None:
        return None
    kind, path, *rest = cell.predict
    launches = kernel_launches(path, *rest) if kind == "kernel" else ncu_rows(path)
    times = [launch_us(r) for r in launches]
    preds = [launch_speedup(r, target) for r in launches]
    kernels = sum(times)
    new_kernels = sum(t / s for t, (s, _) in zip(times, preds))
    if kind == "kernel":
        return kernels / new_kernels, max(zip(times, preds))[1][1]
    runs, = rest
    kernels, new_kernels = kernels / runs, new_kernels / runs
    k = kernels / new_kernels
    if kernels >= cell.us:
        return k, (f"Kernels overlap (their summed ncu time is {100 * kernels / cell.us:.0f}% of "
                   f"this time), so all of it scales with the kernels' {k:.2f}x.")
    other = cell.us - kernels
    return cell.us / (other + new_kernels), (
        f"Kernels are {100 * kernels / cell.us:.0f}% of the {BASELINE.sheet} time and predicted "
        f"{k:.2f}x faster; the other {100 * other / cell.us:.0f}% (copies, gaps) is unchanged.")


# ----------------------------------------------------------------- styling ---

BLUE = Font(color="0563C1", underline="single")
BOLD = Font(bold=True)
GREY = Font(italic=True, color="808080")
HEADER_FILL = PatternFill("solid", fgColor="1F3864")
HEADER_FONT = Font(bold=True, color="FFFFFF")
GROUP_FILLS = ["FCE4D6", "E2EFDA", "DDEBF7"]
THIN = Side(style="thin", color="BFBFBF")
BOX = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
WRAP = Alignment(wrap_text=True, vertical="top")
# Column order of the cells in every row: Direct, CUDA Graph, Agentic optimized.
GROUPS = ["Direct launch (no graph)", "CUDA Graph", "Agentic optimized (formerly SOTA)"]
SHORT = ["Direct", "Graph", "Agentic"]
TIMES = ["Full batch (ms)", "Per iteration in batch (us)", "Single iteration (us)"]
NOT_APPLICABLE = "—"


def put_link(ws, row, col, path, text):
    c = ws.cell(row, col, text)
    c.hyperlink = link(path)
    c.font = BLUE
    return c


def time_values(c):
    """The cell's (full batch ms, per iteration us, single iteration us) as
    numbers, or the text to show instead; None = blank."""
    if c.kernel:  # one ncu time: a single launch in isolation
        return NOT_APPLICABLE, NOT_APPLICABLE, c.us if c.us is not None else c.empty
    if c.us is None:
        return None, c.empty, None
    if c.batch_us is None:
        return TIMING_PENDING, c.us, TIMING_PENDING
    return c.batch_us / 1e3, c.us, c.single_us


def speedup_columns(baseline):
    """(group title, [(column title, value(cells, row number) -> number,
    formula or text)]) of the speedup columns. baseline: the GPU is compared
    with BASELINE (Predicted / Measured) instead of Direct / Graph."""
    agentic = ("Agentic optimized speedup\n(Graph / Agentic)", [
        ("Full batch", lambda cells, r: cells[2].agentic_speedups()[0]),
        ("Single iteration", lambda cells, r: cells[2].agentic_speedups()[1]),
    ])
    if baseline:
        return [(f"Speedup vs {BASELINE.sheet}\n(CUDA Graph)", [
            ("Predicted", None),  # filled from baseline
            (f"Measured\n({BASELINE.sheet} / {{gpu}})", None),
        ]), agentic]
    return [
        ("Graph speedup\n(Direct / Graph)", [
            ("Full batch", lambda cells, r: graph_speedups(cells[0], cells[1])[0]),
            ("Single iteration", lambda cells, r: graph_speedups(cells[0], cells[1])[1]),
        ]),
        agentic,
        ("Single iteration / per iteration in batch\n(launch cost a batch amortizes)", [
            (short, lambda cells, r, i=i: cells[i].overhead()) for i, short in enumerate(SHORT)
        ]),
    ]


def build_summary(ws, gpu, rows, baseline=None):
    """Writes one GPU's summary sheet and returns each row's sheet row number.
    baseline: per row, (its row number on the BASELINE sheet, predicted
    speedup or None); given, the first speedup columns compare against
    BASELINE instead of Direct / Graph."""
    ws.title = gpu.sheet
    ws["A1"] = f"CUDA performance results (latest runs): {gpu.sheet}"
    ws["A1"].font = Font(bold=True, size=14)
    ws["A2"] = (f"GPU: NVIDIA GeForce {gpu.label}, binaries built with {gpu.toolkit} "
                f"(outputs in {gpu.folder}/). Lower is better. Latest run of each workload's "
                "current implementation at the largest problem size tested: the dominant kernel "
                "(if one dominates) and the whole application, launched directly, as a CUDA "
                "Graph, and as the agentic optimized version (<name>_sota.cu, formerly SOTA: the "
                "fastest implementation found, same work per iteration, launched whichever way "
                "is fastest). Full batch = one batch of back-to-back iterations; per iteration = "
                "full batch / iterations; single iteration = median of iterations each launched "
                "alone on an idle device, so launch cost is exposed rather than overlapped with "
                "the previous iteration. Each value links to its source; the ncu-rep / nsys "
                "sqlite columns link the profile files on GitHub. ncu replays each kernel in "
                "isolation, so kernel rows have one single-launch time.")
    ws["A2"].alignment = Alignment(wrap_text=True)
    ws.row_dimensions[2].height = 75

    fixed = ["Workload", "Kernel / stage", "Problem size", "GPU", "Metric"]
    hr, sr = 4, 5
    for i, h in enumerate(fixed, 1):
        ws.cell(hr, i, h)
        ws.merge_cells(start_row=hr, start_column=i, end_row=sr, end_column=i)
    col = len(fixed) + 1
    group_cols = []
    for g in GROUPS:
        ws.cell(hr, col, g)
        ws.merge_cells(start_row=hr, start_column=col, end_row=hr, end_column=col + 4)
        for j, sub in enumerate(TIMES + ["ncu-rep", "nsys sqlite"]):
            ws.cell(sr, col + j, sub)
        group_cols.append(col)
        col += 5
    speedups = speedup_columns(baseline)
    speedup_cols = []
    for title, columns in speedups:
        ws.cell(hr, col, title)
        ws.merge_cells(start_row=hr, start_column=col, end_row=hr, end_column=col + len(columns) - 1)
        for j, (sub, _) in enumerate(columns):
            ws.cell(sr, col + j, sub.format(gpu=gpu.sheet))
        speedup_cols.append(col)
        col += len(columns)
    notes_col = col
    ws.cell(hr, notes_col, "Notes")
    ws.merge_cells(start_row=hr, start_column=notes_col, end_row=sr, end_column=notes_col)
    ws.merge_cells(start_row=2, start_column=1, end_row=2, end_column=notes_col)
    for r in (hr, sr):
        for c in range(1, notes_col + 1):
            cell = ws.cell(r, c)
            cell.font, cell.fill, cell.border = HEADER_FONT, HEADER_FILL, BOX
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    ws.row_dimensions[hr].height = 45
    ws.row_dimensions[sr].height = 45

    r = sr + 1
    previous = None
    positions = []
    graph_col = group_cols[1]
    for i, (workload, kernel, size, metric, *cells, notes) in enumerate(rows):
        if previous is not None and workload != previous:
            r += 1  # blank spacer between workloads
        previous = workload
        positions.append(r)
        for j, v in enumerate((workload, kernel, size, gpu.label, metric), 1):
            ws.cell(r, j, v).alignment = WRAP
        ws.cell(r, 1).font = BOLD
        for cell_spec, gc, fill in zip(cells, group_cols, GROUP_FILLS):
            for j, value in enumerate(time_values(cell_spec)):
                t = ws.cell(r, gc + j)
                if isinstance(value, float):
                    t.value = round(value, 3)
                    t.number_format = "#,##0.00"
                    t.hyperlink = link(cell_spec.value_file())
                    t.font = Font(color="0563C1")
                elif value is not None:
                    t.value, t.font = value, GREY
            if cell_spec.ncu:
                put_link(ws, r, gc + 3, cell_spec.ncu, Path(cell_spec.ncu).name)
            if cell_spec.nsys:
                put_link(ws, r, gc + 4, cell_spec.nsys, Path(cell_spec.nsys).name)
            for c in range(gc, gc + 5):
                ws.cell(r, c).fill = PatternFill("solid", fgColor=fill)
                ws.cell(r, c).alignment = WRAP
        written = []
        for (title, columns), sc in zip(speedups, speedup_cols):
            for j, (_, value) in enumerate(columns):
                if value is None:  # vs BASELINE: Predicted, then Measured
                    base_r, prediction = baseline[i]
                    if j == 0:
                        v = ws.cell(r, sc, round(prediction[0], 3) if prediction else None)
                        if prediction:
                            v.comment = Comment(prediction[1], "build_spreadsheet.py",
                                                width=320, height=140)
                    else:
                        # The primary time: per iteration, or the kernel's one launch.
                        t = get_column_letter(graph_col + (2 if cells[1].kernel else 1))
                        base = f"'{BASELINE.sheet}'!{t}{base_r}"
                        v = ws.cell(r, sc + j, f'=IF(AND(ISNUMBER({base}),ISNUMBER({t}{r})),'
                                               f'{base}/{t}{r},"pending")')
                else:
                    x = value(cells, r)
                    v = ws.cell(r, sc + j, round(x, 3) if x else None)
                written.append(v)
        for v in written:
            v.number_format = '0.00"x"'
            v.alignment = Alignment(horizontal="center", vertical="top")
        ws.cell(r, notes_col, notes).alignment = WRAP
        for c in range(1, notes_col + 1):
            ws.cell(r, c).border = BOX
        r += 1

    if baseline:
        r += 1
        ws.cell(r, 1, "How the predicted speedup is calculated "
                      "(hover a Predicted value for that row's reason)").font = BOLD
        for paragraph in METHOD:
            r += 1
            ws.cell(r, 1, paragraph).alignment = WRAP
            ws.merge_cells(start_row=r, start_column=1, end_row=r, end_column=notes_col)
            ws.row_dimensions[r].height = 48

    widths = [22, 30, 24, 14, 18] + [12, 13, 13, 22, 22] * 3
    widths += [11] * (notes_col - 1 - len(widths)) + [70]
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = w
    ws.freeze_panes = ws.cell(sr + 1, 3)
    return positions


NCU_DETAIL_METRICS = [
    ("gpu__time_duration.sum", "Duration"),
    ("sm__throughput.avg.pct_of_peak_sustained_elapsed", "SM throughput %"),
    ("gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed", "Memory throughput %"),
    ("gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed", "DRAM throughput %"),
    ("sm__warps_active.avg.pct_of_peak_sustained_active", "Achieved occupancy %"),
    ("sm__maximum_warps_per_active_cycle_pct", "Theoretical occupancy %"),
    ("launch__registers_per_thread", "Registers / thread"),
    ("launch__grid_size", "Grid"),
    ("launch__block_size", "Block"),
    ("launch__waves_per_multiprocessor", "Waves / SM"),
    ("dram__bytes_read.sum", "DRAM read"),
    ("dram__bytes_write.sum", "DRAM write"),
]


def build_ncu_details(wb, gpu, rows):
    """Every launch in every ncu report a summary sheet uses, with key metrics."""
    ws = wb.create_sheet(f"ncu metrics ({gpu.sheet})")
    reports = []
    for _, _, _, _, *cells, _ in rows:
        for c in cells:
            if c.ncu and c.ncu not in reports:
                reports.append(c.ncu)
    ws.append(["ncu report", "Launch #", "Kernel"] + [label for _, label in NCU_DETAIL_METRICS])
    if not reports:
        ws.append([NCU_PENDING])
        ws.cell(2, 1).font = GREY
    for path in reports:
        for i, launch in enumerate(ncu_rows(path)):
            values = []
            for metric, _ in NCU_DETAIL_METRICS:
                if metric not in launch:
                    values.append("")
                    continue
                v, unit = launch[metric]
                try:
                    v = number(v)
                except ValueError:
                    pass
                values.append(f"{v} {unit}".strip() if unit and unit != "%" else v)
            ws.append([path, i, launch["Kernel Name"][0].split("(")[0]] + values)
            put_link(ws, ws.max_row, 1, path, Path(path).name)
    for c in ws[1]:
        c.font, c.fill = HEADER_FONT, HEADER_FILL
        c.alignment = Alignment(wrap_text=True, vertical="center")
    for i, w in enumerate([52, 9, 40] + [16] * len(NCU_DETAIL_METRICS), 1):
        ws.column_dimensions[get_column_letter(i)].width = w
    ws.freeze_panes = "A2"


def write_markdown(results, predictions):
    """GitHub-readable copy of the summary sheets; values link to their source files.
    predictions: per non-BASELINE GPU, each row's predicted speedup or None."""
    def fmt(c, value):
        if isinstance(value, float):
            return f"[{value:,.2f}]({link(c.value_file())})"
        if value is None:
            return ""
        return value if value in PENDING or value == NOT_APPLICABLE else "—"

    def files(c):
        return " ".join(f"[{kind}]({link(path)})" for kind, path in
                        (("ncu", c.ncu), ("nsys", c.nsys)) if path)

    def times(x):
        return f"{x:.2f}x" if x else ""

    lines = [
        "# CUDA performance results",
        "",
        "Latest run of each workload's current implementation at the largest problem size tested "
        "(dominant kernel, if one dominates, and the whole application) on each GPU, generated by "
        "`build_spreadsheet.py` from measurements and profiles committed in this repo. Each GPU's "
        "binaries and profiles live in its own folder (`rtx3090/`, `rtx4090/`); the sources are "
        "shared. The spreadsheet [`cuda_performance.xlsx`](cuda_performance.xlsx) has one sheet "
        "per GPU plus the ncu metrics of every profiled launch. Lower times are better.",
        "",
        "Each workload runs three ways, left to right: **Direct** (the kernels and copies "
        "enqueued directly on every iteration, without a CUDA Graph), **CUDA Graph** (the same "
        "work captured once and replayed) and **Agentic optimized** (formerly SOTA; "
        "`<name>_sota.cu`): the fastest implementation found after researching published "
        "libraries and kernels, free to change kernels and data movement and launched whichever "
        "way is fastest. It does the same work per iteration: every timed iteration copies the "
        "full input from the host and the full output back, and it rotates over 3 different "
        "inputs and checks every output, so caching across iterations would fail (llama2c's "
        "inputs stay on the GPU in all versions). Outputs match the Direct and Graph versions "
        "(each workload's `compare_sota.sh`).",
        "",
        "Each version has three times:",
        "",
        "- **Full batch** (ms): one batch of back-to-back iterations between CUDA events "
        "(`time_all.sh`: 50 iterations).",
        "- **Per iteration** (us): full batch / iterations.",
        f"- **Single iteration** (us): median of {SINGLE_ITERATIONS} iterations, each launched "
        "alone on an idle device. The device is drained before each, so the iteration's launch "
        "cost is exposed instead of overlapping the previous iteration's work as it does in the "
        "batch.",
        "",
        "**Single / batch** = single iteration / per iteration in the batch: how much launch "
        "cost batching amortizes in each launch mode. **Graph speedup** = Direct / Graph and "
        "**Agentic speedup** = Graph / Agentic optimized, for the full batch and for a single "
        "iteration. Kernel rows have one ncu time per version (a single launch replayed in "
        "isolation), so they fill only the single-iteration columns; their agentic speedup "
        "compares the ncu time of all the kernels doing that row's work in one iteration (see "
        "the spreadsheet notes). — = not applicable.",
    ]
    base_rows = dict(results)[BASELINE]
    for gpu, rows in results:
        compare = gpu is not BASELINE
        heads = ["Workload", "Kernel / stage", "Problem size", "Metric"]
        for short in SHORT:
            heads += [f"{short}: full batch (ms)", f"{short}: per iteration (us)",
                      f"{short}: single iteration (us)", f"{short} profiles"]
        if compare:
            heads += [f"Predicted vs {BASELINE.sheet}", f"Measured vs {BASELINE.sheet}"]
        else:
            heads += ["Graph speedup: full batch", "Graph speedup: single iteration"]
        heads += ["Agentic speedup: full batch", "Agentic speedup: single iteration"]
        if not compare:
            heads += [f"Single / batch: {short}" for short in SHORT]
        align = ["---"] * 4 + ["--:", "--:", "--:", "---"] * 3 + ["--:"] * (len(heads) - 16)
        lines += [
            "",
            f"## {gpu.sheet}",
            "",
            f"GPU: NVIDIA GeForce {gpu.label}. Binaries built with {gpu.toolkit}.",
            "",
            "| " + " | ".join(heads) + " |",
            "|" + "|".join(align) + "|",
        ]
        for i, (workload, kernel, size, metric, *cells, _) in enumerate(rows):
            row = [workload, kernel, size, metric]
            for c in cells:
                row += [fmt(c, v) for v in time_values(c)] + [files(c)]
            direct, graph, sota = cells
            if compare:
                base, prediction = base_rows[i][5], predictions[gpu][i]
                row += [f"{prediction[0]:.2f}x" if prediction else "",
                        times(ratio(base.us, graph.us)) or "pending"]
            else:
                row += [times(x) for x in graph_speedups(direct, graph)]
            pending = "pending" if sota.us is None else ""
            row += [times(x) or pending for x in sota.agentic_speedups()]
            if not compare:
                row += [times(c.overhead()) for c in cells]
            lines.append("| " + " | ".join(row) + " |")
        if compare:
            lines += ["", "### How the predicted speedup is calculated", ""]
            lines += [x for paragraph in METHOD for x in (paragraph, "")]
            lines += ["Each row's prediction:", ""]
            lines += [f"- {w} / {k}: **{pr[0]:.2f}x**. {pr[1]}"
                      for (w, k, *_), pr in zip(rows, predictions[gpu]) if pr]
    lines += ["", "Regenerate (needs `ncu` on PATH and `pip install openpyxl`):", "",
              "```bash", "python3 results/build_spreadsheet.py", "```", ""]
    MD_OUT.write_text("\n".join(lines))


def main():
    results = [(gpu, gpu_rows(gpu)) for gpu in GPUS]
    base_rows = dict(results)[BASELINE]
    predictions = {}
    for gpu, rows in results:
        if gpu is not BASELINE:
            assert [r[:2] for r in rows] == [r[:2] for r in base_rows], "rows must line up"
            # r[5]: the CUDA Graph cell
            predictions[gpu] = [predicted_speedup(r[5], gpu) for r in base_rows]
    wb = Workbook()
    wb.remove(wb.active)
    positions = {}  # BASELINE comes first in GPUS, so its rows are placed first
    for gpu, rows in results:
        baseline = (list(zip(positions[BASELINE], predictions[gpu]))
                    if gpu is not BASELINE else None)
        positions[gpu] = build_summary(wb.create_sheet(), gpu, rows, baseline)
    for gpu, rows in results:
        build_ncu_details(wb, gpu, rows)
    wb.calculation.fullCalcOnLoad = True  # openpyxl does not evaluate the Measured formulas
    wb.save(OUT)
    write_markdown(results, predictions)
    print(f"wrote {OUT.relative_to(REPO)} and {MD_OUT.relative_to(REPO)} "
          f"({', '.join(f'{len(rows)} {gpu.sheet} rows' for gpu, rows in results)})")


if __name__ == "__main__":
    main()
