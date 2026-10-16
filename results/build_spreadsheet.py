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
Columns of each summary sheet:
  CUDA Graph     the workload launched as a CUDA Graph
  Direct launch  the same stream operations enqueued directly, without a graph
                 (each harness's direct mode: llama2c `direct`, loom_msa and
                 mlperf `--launch=direct`, pagerank PAGERANK_LAUNCH=direct)

Kernel rows use the ncu kernel duration (gpu__time_duration.sum, mean over
the profiled launches of that kernel). End-to-end rows use the benchmark's
own timing (cudaEvent per iteration / graph replay) or, where only an nsys
trace exists, the median GPU span of one iteration in that trace: the work
inside each NVTX range around a timed iteration or, in traces from before the
harnesses had NVTX ranges, each graph replay. A cell whose profile does not
exist yet (profile_ncu_all.sh or the direct-mode profiles not run on that GPU)
reads "pending".

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
}, {
    "pagerank_timing": "The binary's own cudaEvent timing in the same runs: 7.658 / 7.686 ms "
                       "(graph / direct) with transfers, 4.862 / 4.885 ms compute only.",
})

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


# ------------------------------------------------------------------ cells ---

class Cell:
    """One measurement: a value in microseconds and the files behind it."""

    def __init__(self, us=None, ncu=None, nsys=None, csv=None, empty="", predict=None):
        self.us, self.ncu, self.nsys, self.csv = us, ncu, nsys, csv
        self.empty = empty
        # How to predict this value on another GPU from this GPU's ncu report:
        # ("kernel", report, kernel, grid) or ("app", report, runs in report).
        self.predict = predict

    def value_file(self):
        """The file the value itself was read from."""
        return self.ncu or self.csv or self.nsys


NCU_PENDING = "pending (ncu not run yet)"
DIRECT_PENDING = "pending (direct mode not profiled yet)"


def ncu_cell(path, kernel, grid=None, nsys=None, pending=NCU_PENDING):
    if path is None:
        return Cell(nsys=nsys, empty=pending)
    return Cell(ncu_kernel_us(path, kernel, grid)[0], ncu=path, nsys=nsys,
                predict=("kernel", path, kernel, grid))


# ------------------------------------------------------------------- rows ---
# Each row: (workload, kernel/stage, problem size, metric, graph, direct, notes)
# One problem size per workload (the largest tested): the dominant kernel, when
# one kernel dominates, and the whole application.

KERNEL_TIME = "Kernel time (ncu)"
WHOLE_APP = "Whole application"


def share(kernel, app, launches=1):
    if kernel.us is None:
        return ""
    return f"{100 * launches * kernel.us / app.us:.0f}% of the whole-application time."


def llama2c_rows(gpu):
    w = "llama2c (coalesced matmul)"
    size = "batch=1024, dim=16384, rows=8192"
    sweep = gpu.file("llama_sweep")
    row = sweep_row(sweep, batch=1024, size=16384, rows=8192, repeats=1)
    nsys = {m: gpu.file(f"llama_nsys_{m}") for m in ("graph", "direct")}
    ncu = {m: gpu.file(f"llama_ncu_{m}") for m in nsys}
    # The ncu report covers 3 iterations.
    app = {m: Cell(float(row[f"{m}_gpu_ms"]) * 1e3, csv=sweep, nsys=nsys[m],
                   predict=("app", ncu[m], 3)) for m in nsys}
    matmul = {m: ncu_cell(ncu[m], "matmul_coalesced_kernel", nsys=nsys[m]) for m in nsys}
    if ncu["graph"]:
        s = ncu_time_shares(ncu["graph"])
        largest = (f"Largest kernel ({s['matmul_coalesced_kernel']:.0f}% of serialized kernel "
                   f"time under ncu; rmsnorm {s['rmsnorm_kernel']:.0f}%, softmax "
                   f"{s['softmax_kernel']:.0f}%). ")
    else:
        largest = "Largest kernel. "
    reruns = " / ".join(f"{1e3 * nsys_log_ms_per_iter(nsys[m]):.1f}" for m in nsys)
    return [
        (w, "matmul_coalesced_kernel", size, KERNEL_TIME, matmul["graph"], matmul["direct"],
         largest + "It is a matrix-vector product, so its time does not grow with batch; the "
         "three kernels run concurrently on separate streams, so their times overlap. "
         + share(matmul["graph"], app["graph"])),
        (w, f"{WHOLE_APP} (rmsnorm + softmax + matmul iteration)", size,
         "Iteration time (cudaEvent)", app["graph"], app["direct"],
         "Mean of 3 interleaved trials from the GPU-filling sweep; no per-iteration copies. "
         f"The nsys traces rerun this configuration ({reruns} us per iteration)."),
    ]


def loom_rows(gpu):
    w = "loom_msa (multi-sensor attention)"
    size = "25,000 queries x 250,000 context"
    nsys = {m: gpu.file(f"loom_nsys_{m}") for m in MODES}
    ncu = {m: gpu.file(f"loom_ncu_{m}") for m in MODES}
    summary = loom_benchmark(gpu.file("loom_benchmark"))[("1x25000x250000", "random")]
    app = {}
    for m in MODES:
        run = summary.get("optimized" if m == "graph" else f"optimized_{m}")
        app[m] = (Cell(run["end_to_end_us"]["median"], csv=gpu.file("loom_benchmark_txt"),
                       nsys=nsys[m], predict=("app", ncu[m], 1)) if run
                  else Cell(nsys=nsys[m], empty=DIRECT_PENDING))
    attention = {m: ncu_cell(ncu[m], "fuse_attention_kernel", nsys=nsys[m],
                             pending=NCU_PENDING if m == "graph" or nsys[m] else DIRECT_PENDING)
                 for m in MODES}
    modes = "both launch modes" if app["direct"].us is not None else "CUDA graph only"
    return [
        (w, "fuse_attention_kernel", size, KERNEL_TIME, attention["graph"], attention["direct"],
         "Single-pass online softmax, shared-memory context tiles, 8-way context split "
         "(FlashAttention-style). " + share(attention["graph"], app["graph"])),
        (w, f"{WHOLE_APP} (H2D + kernels + D2H)", size, "Iteration time (cudaEvent)",
         app["graph"], app["direct"],
         f"benchmark.py: median of 5 runs, random inputs, {modes}. Includes H2D input and D2H "
         "output copies."),
    ]


def pagerank_rows(gpu):
    w = "pagerank (GAPBS)"
    size = "2^20 vertices, 16 edges/vertex, 20 iterations"
    ncu = {m: gpu.file(f"pagerank_ncu_{m}") for m in MODES}
    nsys = {m: gpu.file(f"pagerank_nsys_{m}") for m in MODES}
    app, measured = {}, {}
    for m in MODES:
        if nsys[m]:
            measured[m] = nsys_iteration(nsys[m], "pagerank.full")
            app[m] = Cell(measured[m][0], nsys=nsys[m], predict=("app", ncu[m], 1))
        else:
            app[m] = Cell(empty=DIRECT_PENDING)
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
    return [
        (w, "pagerank_update (one iteration)", size, KERNEL_TIME, update["graph"], update["direct"],
         "Pull gather of precomputed contributions. Full-graph iterations only; "
         + first + share(update["graph"], app["graph"], 19)),
        (w, f"{WHOLE_APP} (H2D + 20 iterations + D2H)", size, "Iteration time (nsys)",
         app["graph"], app["direct"],
         iteration_note(measured) + " The CSR graph is uploaded from pinned memory in 8 chunks "
         "overlapped with the first iteration. " + gpu.notes["pagerank_timing"]),
    ]


def mlperf_rows(gpu):
    rows = []
    for w, kernel, size, key in (
            ("mlperf-tiny anomaly", "mse_partial_kernel", "64M elements", "anomaly"),
            ("mlperf-tiny KWS", "kws_postprocess_kernel", "12 classes x 4M rows", "kws"),
            ("mlperf-tiny VWW", "vww_preprocess_kernel", "256M bytes", "vww")):
        ncu = {m: gpu.file(f"{key}_ncu_{m}") for m in MODES}
        nsys = {m: gpu.file(f"{key}_nsys_{m}") for m in MODES}
        app, measured = {}, {}
        for m in MODES:
            if nsys[m]:
                measured[m] = nsys_iteration(nsys[m], f"{key}.end_to_end")
                app[m] = Cell(measured[m][0], nsys=nsys[m], predict=("app", ncu[m], 1))
            else:
                app[m] = Cell(empty=DIRECT_PENDING)
        k = {m: ncu_cell(ncu[m], kernel, nsys=nsys[m],
                         pending=NCU_PENDING if m == "graph" or nsys[m] else DIRECT_PENDING)
             for m in MODES}
        only = "" if kernel == "mse_partial_kernel" else "Only kernel in the application. "
        rows += [(w, kernel, size, KERNEL_TIME, k["graph"], k["direct"],
                  only + share(k["graph"], app["graph"])),
                 (w, f"{WHOLE_APP} (H2D + kernels + D2H)", size, "Iteration time (nsys)",
                  app["graph"], app["direct"], iteration_note(measured))]
    return rows


def gpu_rows(gpu):
    return llama2c_rows(gpu) + loom_rows(gpu) + pagerank_rows(gpu) + mlperf_rows(gpu)


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
GROUP_FILLS = ["E2EFDA", "FCE4D6"]
THIN = Side(style="thin", color="BFBFBF")
BOX = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
WRAP = Alignment(wrap_text=True, vertical="top")
GROUPS = ["CUDA Graph", "Direct launch (no graph)"]


def put_link(ws, row, col, path, text):
    c = ws.cell(row, col, text)
    c.hyperlink = link(path)
    c.font = BLUE
    return c


def build_summary(ws, gpu, rows, baseline=None):
    """Writes one GPU's summary sheet and returns each row's sheet row number.
    baseline: per row, (its row number on the BASELINE sheet, predicted
    speedup or None); given, the speedup columns compare against BASELINE
    instead of Direct / Graph."""
    ws.title = gpu.sheet
    ws["A1"] = f"CUDA performance results (latest runs): {gpu.sheet}"
    ws["A1"].font = Font(bold=True, size=14)
    ws["A2"] = (f"GPU: NVIDIA GeForce {gpu.label}, binaries built with {gpu.toolkit} "
                f"GPU (outputs in {gpu.folder}/). Times in microseconds (lower is better). "
                "Latest run of each workload's current implementation at the largest problem size "
                "tested: the dominant kernel (if one dominates) and the whole application. "
                "Each value links to its "
                "source; the ncu-rep / nsys sqlite columns link the profile files on GitHub. "
                "ncu replays each kernel in isolation, so ncu kernel times barely change with "
                "launch mode; the CUDA Graph benefit (launch overhead) shows in end-to-end rows.")
    ws["A2"].alignment = Alignment(wrap_text=True)
    ws.merge_cells("A2:M2")
    ws.row_dimensions[2].height = 45

    fixed = ["Workload", "Kernel / stage", "Problem size", "GPU", "Metric"]
    hr, sr = 4, 5
    for i, h in enumerate(fixed, 1):
        ws.cell(hr, i, h)
        ws.merge_cells(start_row=hr, start_column=i, end_row=sr, end_column=i)
    col = len(fixed) + 1
    group_cols = []
    for g in GROUPS:
        ws.cell(hr, col, g)
        ws.merge_cells(start_row=hr, start_column=col, end_row=hr, end_column=col + 2)
        for j, sub in enumerate(("Time (us)", "ncu-rep", "nsys sqlite")):
            ws.cell(sr, col + j, sub)
        group_cols.append(col)
        col += 3
    if baseline:
        ws.cell(hr, col, f"Speedup vs {BASELINE.sheet} (CUDA Graph time)")
        ws.merge_cells(start_row=hr, start_column=col, end_row=hr, end_column=col + 1)
        ws.cell(sr, col, "Predicted")
        ws.cell(sr, col + 1, f"Measured\n({BASELINE.sheet} / {gpu.sheet})")
        notes_col = col + 2
    else:
        ws.cell(hr, col, "Graph speedup\n(Direct / Graph)")
        ws.merge_cells(start_row=hr, start_column=col, end_row=sr, end_column=col)
        notes_col = col + 1
    ws.cell(hr, notes_col, "Notes")
    ws.merge_cells(start_row=hr, start_column=notes_col, end_row=sr, end_column=notes_col)
    last_col = notes_col
    for r in (hr, sr):
        for c in range(1, last_col + 1):
            cell = ws.cell(r, c)
            cell.font, cell.fill, cell.border = HEADER_FONT, HEADER_FILL, BOX
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    ws.row_dimensions[hr].height = 32

    r = sr + 1
    previous = None
    positions = []
    for i, (workload, kernel, size, metric, *cells, notes) in enumerate(rows):
        if previous is not None and workload != previous:
            r += 1  # blank spacer between workloads
        previous = workload
        positions.append(r)
        for j, v in enumerate((workload, kernel, size, gpu.label, metric), 1):
            ws.cell(r, j, v).alignment = WRAP
        ws.cell(r, 1).font = BOLD
        for cell_spec, gc, fill in zip(cells, group_cols, GROUP_FILLS):
            t = ws.cell(r, gc)
            if cell_spec.us is None:
                t.value, t.font = cell_spec.empty, GREY
            else:
                t.value = round(cell_spec.us, 3)
                t.number_format = "#,##0.00"
                t.hyperlink = link(cell_spec.value_file())
                t.font = Font(color="0563C1")
            if cell_spec.ncu:
                put_link(ws, r, gc + 1, cell_spec.ncu, Path(cell_spec.ncu).name)
            if cell_spec.nsys:
                put_link(ws, r, gc + 2, cell_spec.nsys, Path(cell_spec.nsys).name)
            for c in range(gc, gc + 3):
                ws.cell(r, c).fill = PatternFill("solid", fgColor=fill)
                ws.cell(r, c).alignment = WRAP
        g, d = (get_column_letter(c) for c in group_cols)
        if baseline:
            base_r, prediction = baseline[i]
            if prediction:
                p = ws.cell(r, col, round(prediction[0], 3))
                p.comment = Comment(prediction[1], "build_spreadsheet.py", width=320, height=140)
            base = f"'{BASELINE.sheet}'!{g}{base_r}"
            m = ws.cell(r, col + 1, f'=IF(AND(ISNUMBER({base}),ISNUMBER({g}{r})),'
                                    f'{base}/{g}{r},"pending")')
            speedups = (ws.cell(r, col), m)
        else:
            speedups = (ws.cell(r, col, f'=IF(AND(ISNUMBER({g}{r}),ISNUMBER({d}{r})),'
                                        f'{d}{r}/{g}{r},"")'),)
        for f in speedups:
            f.number_format = '0.00"x"'
            f.alignment = Alignment(horizontal="center", vertical="top")
        ws.cell(r, notes_col, notes).alignment = WRAP
        for c in range(1, last_col + 1):
            ws.cell(r, c).border = BOX
        r += 1

    if baseline:
        r += 1
        ws.cell(r, 1, "How the predicted speedup is calculated "
                      "(hover a Predicted value for that row's reason)").font = BOLD
        for paragraph in METHOD:
            r += 1
            ws.cell(r, 1, paragraph).alignment = WRAP
            ws.merge_cells(start_row=r, start_column=1, end_row=r, end_column=last_col)
            ws.row_dimensions[r].height = 48

    widths = ([22, 30, 24, 14, 16] + [13, 22, 22] * 2
              + ([11, 13] if baseline else [14]) + [60])
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
    def fmt(c):
        if c.us is None:
            return c.empty if c.empty in (NCU_PENDING, DIRECT_PENDING) else "—"
        return f"[{c.us:,.2f}]({link(c.value_file())})"

    def files(c):
        return " ".join(f"[{kind}]({link(path)})" for kind, path in
                        (("ncu", c.ncu), ("nsys", c.nsys)) if path)

    lines = [
        "# CUDA performance results",
        "",
        "Times in microseconds (lower is better). Latest run of each workload's current "
        "implementation at the largest problem size tested (dominant kernel, if one dominates, "
        "and the whole application) on each GPU, generated by `build_spreadsheet.py` from "
        "profiles committed in this repo. Each GPU's binaries and profiles live in its own "
        "folder (`rtx3090/`, `rtx4090/`); the sources are shared. The spreadsheet "
        "[`cuda_performance.xlsx`](cuda_performance.xlsx) has one sheet per GPU plus the ncu "
        "metrics of every profiled launch.",
        "",
        "**Direct** = the same kernels and copies enqueued directly, without a CUDA Graph "
        "(each harness's direct-launch mode). — = not available.",
    ]
    base_rows = dict(results)[BASELINE]
    for gpu, rows in results:
        compare = gpu is not BASELINE
        speedup_heads = (f"Predicted vs {BASELINE.sheet} | Measured vs {BASELINE.sheet}"
                         if compare else "Graph speedup")
        lines += [
            "",
            f"## {gpu.sheet}",
            "",
            f"GPU: NVIDIA GeForce {gpu.label}. Binaries built with {gpu.toolkit}.",
            "",
            "| Workload | Kernel / stage | Problem size | Metric | CUDA Graph | Graph profiles | "
            f"Direct | Direct profiles | {speedup_heads} |",
            "|---|---|---|---|--:|---|--:|---|--:|" + ("--:|" if compare else ""),
        ]
        for i, (workload, kernel, size, metric, graph, direct, _) in enumerate(rows):
            if compare:
                base, prediction = base_rows[i][4], predictions[gpu][i]
                speedup = " | ".join((
                    f"{prediction[0]:.2f}x" if prediction else "",
                    f"{base.us / graph.us:.2f}x" if graph.us is not None else "pending"))
            else:
                speedup = (f"{direct.us / graph.us:.2f}x"
                           if direct.us is not None and graph.us is not None else "")
            lines.append(f"| {workload} | {kernel} | {size} | {metric} | {fmt(graph)} | "
                         f"{files(graph)} | {fmt(direct)} | {files(direct)} | {speedup} |")
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
            predictions[gpu] = [predicted_speedup(r[4], gpu) for r in base_rows]
    wb = Workbook()
    wb.remove(wb.active)
    positions = {}  # BASELINE comes first in GPUS, so its rows are placed first
    for gpu, rows in results:
        baseline = (list(zip(positions[BASELINE], predictions[gpu]))
                    if gpu is not BASELINE else None)
        positions[gpu] = build_summary(wb.create_sheet(), gpu, rows, baseline)
    for gpu, rows in results:
        build_ncu_details(wb, gpu, rows)
    wb.calculation.fullCalcOnLoad = True  # openpyxl does not evaluate the speedup formulas
    wb.save(OUT)
    write_markdown(results, predictions)
    print(f"wrote {OUT.relative_to(REPO)} and {MD_OUT.relative_to(REPO)} "
          f"({', '.join(f'{len(rows)} {gpu.sheet} rows' for gpu, rows in results)})")


if __name__ == "__main__":
    main()
