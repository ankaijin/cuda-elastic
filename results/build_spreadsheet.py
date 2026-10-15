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
  Direct launch  the same kernels launched directly (only llama2c has this mode)

Kernel rows use the ncu kernel duration (gpu__time_duration.sum, mean over
the profiled launches of that kernel). End-to-end rows use the benchmark's
own timing (cudaEvent per iteration / graph replay) or, where only an nsys
trace exists, the median graph replay span from that trace. A kernel row whose
ncu report does not exist yet (profile_ncu_all.sh not run) reads "pending".

Requires: ncu (Nsight Compute CLI) on PATH and openpyxl.
Usage:    python3 results/build_spreadsheet.py
"""

import csv
import io
import json
import re
import sqlite3
import statistics
import subprocess
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results" / "cuda_performance.xlsx"
MD_OUT = REPO / "results" / "README.md"
GITHUB = "https://github.com/ankaijin/cuda-elastic/blob"


# ------------------------------------------------------------------- GPUs ---

class GPU:
    """One GPU's results: the folder its outputs live in and, per input, the
    file to read (a glob; the latest match by name, i.e. by timestamp, wins)."""

    def __init__(self, folder, label, toolkit, files, notes):
        self.folder, self.label, self.toolkit = folder, label, toolkit
        self.files, self.notes = files, notes

    @property
    def sheet(self):
        return self.label.split(" (")[0]

    def file(self, key):
        matches = sorted((REPO / self.folder).glob(self.files[key]))
        return str(matches[-1].relative_to(REPO)) if matches else None


RTX3090 = GPU("rtx3090", "RTX 3090 (sm_86, 24 GB)", "CUDA 13.3", {
    "llama_sweep": "llama2c/profiles/sweep-fill-20261002-152612-speedup.csv",
    "llama_nsys_graph": "llama2c/profiles/nsys-b1024-graph-20261014-215130.sqlite",
    "llama_nsys_direct": "llama2c/profiles/nsys-b1024-direct-20261014-215130.sqlite",
    "llama_ncu_graph": "llama2c/profiles/ncu-large-coalesced-b1024-graph-20261014-215130.ncu-rep",
    "llama_ncu_direct": "llama2c/profiles/ncu-large-coalesced-b1024-direct-20261014-215130.ncu-rep",
    "loom_benchmark": "loom_msa/build/benchmark.json",
    "loom_benchmark_txt": "loom_msa/build/benchmark.txt",
    "loom_ncu": "loom_msa/profiles/ncu-all-kernels-msa_optimized_large-20260925-142052.ncu-rep",
    "loom_nsys": "loom_msa/profiles/nsys-msa_optimized_large-20260925-142045.sqlite",
    "pagerank_ncu": "pagerank/profiles/ncu-pagerank-20261007-152417.ncu-rep",
    "pagerank_nsys": "pagerank/profiles/nsys-20261007-152346.sqlite",
    "anomaly_ncu": "mlperf-anomaly/profiles/ncu-anomaly_mse-20261007-144943.ncu-rep",
    "anomaly_nsys": "mlperf-anomaly/profiles/nsys-anomaly_mse-20261007-144939.sqlite",
    "kws_ncu": "mlperf-classification/profiles/ncu-kws_postprocess-20261007-144950.ncu-rep",
    "kws_nsys": "mlperf-classification/profiles/nsys-kws_postprocess-20261007-144947.sqlite",
    "vww_ncu": "mlperf-classification/profiles/ncu-vww_preprocess-20261007-144958.ncu-rep",
    "vww_nsys": "mlperf-classification/profiles/nsys-vww_preprocess-20261007-144953.sqlite",
}, {
    "pagerank_timing": "The binary's own cudaEvent timing in the same run: 7.684 ms with "
                       "transfers, 4.841 ms compute only.",
})

RTX4090 = GPU("rtx4090", "RTX 4090 (sm_89, 24 GB)", "CUDA 12.5", {
    "llama_sweep": "llama2c/profiles/sweep-fill-*-speedup.csv",
    "llama_nsys_graph": "llama2c/profiles/nsys-node-graph-*.sqlite",
    "llama_nsys_direct": "llama2c/profiles/nsys-node-direct-*.sqlite",
    "llama_ncu_graph": "llama2c/profiles/ncu-large-coalesced-b1024-graph-*.ncu-rep",
    "llama_ncu_direct": "llama2c/profiles/ncu-large-coalesced-b1024-direct-*.ncu-rep",
    "loom_benchmark": "loom_msa/build/benchmark.json",
    "loom_benchmark_txt": "loom_msa/build/benchmark.txt",
    "loom_ncu": "loom_msa/profiles/ncu-all-kernels-msa_optimized_large-*.ncu-rep",
    "loom_nsys": "loom_msa/profiles/nsys-msa_optimized_large-*.sqlite",
    "pagerank_ncu": "pagerank/profiles/ncu-pagerank-*.ncu-rep",
    "pagerank_nsys": "pagerank/profiles/nsys-*.sqlite",
    "anomaly_ncu": "mlperf-anomaly/profiles/ncu-anomaly_mse-*.ncu-rep",
    "anomaly_nsys": "mlperf-anomaly/profiles/nsys-anomaly_mse-*.sqlite",
    "kws_ncu": "mlperf-classification/profiles/ncu-kws_postprocess-*.ncu-rep",
    "kws_nsys": "mlperf-classification/profiles/nsys-kws_postprocess-*.sqlite",
    "vww_ncu": "mlperf-classification/profiles/ncu-vww_preprocess-*.ncu-rep",
    "vww_nsys": "mlperf-classification/profiles/nsys-vww_preprocess-*.sqlite",
}, {
    "pagerank_timing": "A separate uninstrumented run of the binary "
                       "(profiles/bench-20261015-091815.txt): 5.714 ms with transfers, "
                       "2.651 ms compute only.",
})

GPUS = [RTX3090, RTX4090]


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

    def __init__(self, us=None, ncu=None, nsys=None, csv=None, empty=""):
        self.us, self.ncu, self.nsys, self.csv = us, ncu, nsys, csv
        self.empty = empty

    def value_file(self):
        """The file the value itself was read from."""
        return self.ncu or self.csv or self.nsys


NCU_PENDING = "pending (ncu not run yet)"


def ncu_cell(path, kernel, grid=None, nsys=None):
    if path is None:
        return Cell(nsys=nsys, empty=NCU_PENDING)
    return Cell(ncu_kernel_us(path, kernel, grid)[0], ncu=path, nsys=nsys)


GRAPH_ONLY = Cell(empty="n/a (graph-only harness)")


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
    app = {m: Cell(float(row[f"{m}_gpu_ms"]) * 1e3, csv=sweep, nsys=nsys[m]) for m in nsys}
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
    nsys = gpu.file("loom_nsys")
    benchmark = loom_benchmark(gpu.file("loom_benchmark"))
    app = Cell(benchmark[("1x25000x250000", "random")]["optimized"]["end_to_end_us"]["median"],
               csv=gpu.file("loom_benchmark_txt"), nsys=nsys)
    attention = ncu_cell(gpu.file("loom_ncu"), "fuse_attention_kernel", nsys=nsys)
    return [
        (w, "fuse_attention_kernel", size, KERNEL_TIME, attention, GRAPH_ONLY,
         "Single-pass online softmax, shared-memory context tiles, 8-way context split "
         "(FlashAttention-style). " + share(attention, app)),
        (w, f"{WHOLE_APP} (graph replay)", size, "Replay time (cudaEvent)", app, GRAPH_ONLY,
         "benchmark.py: median of 5 runs, random inputs. Includes H2D input and D2H output copies."),
    ]


def pagerank_rows(gpu):
    w = "pagerank (GAPBS)"
    size = "2^20 vertices, 16 edges/vertex, 20 iterations"
    ncu = gpu.file("pagerank_ncu")
    nsys = gpu.file("pagerank_nsys")
    replay_us, n = nsys_node_replay_us(nsys)
    app = Cell(replay_us, nsys=nsys)
    update = ncu_cell(ncu, "pagerank_update", grid=32768, nsys=nsys)
    if ncu:
        chunks = [launch_us(r) for r in kernel_launches(ncu, "pagerank_update")
                  if number(r["launch__grid_size"][0]) != 32768]
        first = (f"the first iteration runs as {len(chunks)} upload-overlapped chunks of "
                 f"~{statistics.mean(chunks):.0f} us each. 19 full iterations = ")
    else:
        first = "the first iteration runs as 8 upload-overlapped chunks. "
    return [
        (w, "pagerank_update (one iteration)", size, KERNEL_TIME, update, GRAPH_ONLY,
         "Pull gather of precomputed contributions. Full-graph iterations only; "
         + first + share(update, app, 19)),
        (w, f"{WHOLE_APP} (graph replay: H2D + 20 iterations)", size, "Replay time (nsys)",
         app, GRAPH_ONLY,
         f"Median span of the {n} replays with transfers in the nsys node trace. The CSR graph "
         "is uploaded from pinned memory in 8 chunks overlapped with the first iteration. "
         + gpu.notes["pagerank_timing"]),
    ]


def mlperf_rows(gpu):
    rows = []
    for w, kernel, size, key in (
            ("mlperf-tiny anomaly", "mse_partial_kernel", "64M elements", "anomaly"),
            ("mlperf-tiny KWS", "kws_postprocess_kernel", "12 classes x 4M rows", "kws"),
            ("mlperf-tiny VWW", "vww_preprocess_kernel", "256M bytes", "vww")):
        nsys = gpu.file(f"{key}_nsys")
        replay_us, n = nsys_node_replay_us(nsys)
        app = Cell(replay_us, nsys=nsys)
        k = ncu_cell(gpu.file(f"{key}_ncu"), kernel, nsys=nsys)
        only = "" if kernel == "mse_partial_kernel" else "Only kernel in the application. "
        rows += [(w, kernel, size, KERNEL_TIME, k, GRAPH_ONLY, only + share(k, app)),
                 (w, f"{WHOLE_APP} (graph replay: H2D + kernels + D2H)", size, "Replay time (nsys)",
                  app, GRAPH_ONLY,
                  f"Median span of the {n} end-to-end replays in the nsys node trace.")]
    return rows


def gpu_rows(gpu):
    return llama2c_rows(gpu) + loom_rows(gpu) + pagerank_rows(gpu) + mlperf_rows(gpu)


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


def build_summary(ws, gpu, rows):
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
    tail = ["Graph speedup\n(Direct / Graph)", "Notes"]
    for j, h in enumerate(tail):
        ws.cell(hr, col + j, h)
        ws.merge_cells(start_row=hr, start_column=col + j, end_row=sr, end_column=col + j)
    last_col = col + len(tail) - 1
    for r in (hr, sr):
        for c in range(1, last_col + 1):
            cell = ws.cell(r, c)
            cell.font, cell.fill, cell.border = HEADER_FONT, HEADER_FILL, BOX
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    ws.row_dimensions[hr].height = 32

    r = sr + 1
    previous = None
    for workload, kernel, size, metric, *cells, notes in rows:
        if previous is not None and workload != previous:
            r += 1  # blank spacer between workloads
        previous = workload
        for i, v in enumerate((workload, kernel, size, gpu.label, metric), 1):
            ws.cell(r, i, v).alignment = WRAP
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
        f = ws.cell(r, col, f'=IF(AND(ISNUMBER({g}{r}),ISNUMBER({d}{r})),{d}{r}/{g}{r},"")')
        f.number_format = '0.00"x"'
        f.alignment = Alignment(horizontal="center", vertical="top")
        ws.cell(r, col + 1, notes).alignment = WRAP
        for c in range(1, last_col + 1):
            ws.cell(r, c).border = BOX
        r += 1

    widths = [22, 30, 24, 14, 16] + [13, 22, 22] * 2 + [14, 60]
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = w
    ws.freeze_panes = ws.cell(sr + 1, 3)


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


def write_markdown(results):
    """GitHub-readable copy of the summary sheets; values link to their source files."""
    def fmt(c):
        if c.us is None:
            return c.empty if c.empty == NCU_PENDING else "—"
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
        "**Direct** = the same kernels launched without a CUDA Graph (only llama2c has this "
        "mode; the other harnesses are graph-only). — = not available.",
    ]
    for gpu, rows in results:
        lines += [
            "",
            f"## {gpu.sheet}",
            "",
            f"GPU: NVIDIA GeForce {gpu.label}. Binaries built with {gpu.toolkit}.",
            "",
            "| Workload | Kernel / stage | Problem size | Metric | CUDA Graph | Graph profiles | "
            "Direct | Direct profiles | Graph speedup |",
            "|---|---|---|---|--:|---|--:|---|--:|",
        ]
        for workload, kernel, size, metric, graph, direct, _ in rows:
            speedup = (f"{direct.us / graph.us:.2f}x"
                       if direct.us is not None and graph.us is not None else "")
            lines.append(f"| {workload} | {kernel} | {size} | {metric} | {fmt(graph)} | "
                         f"{files(graph)} | {fmt(direct)} | {files(direct)} | {speedup} |")
    lines += ["", "Regenerate (needs `ncu` on PATH and `pip install openpyxl`):", "",
              "```bash", "python3 results/build_spreadsheet.py", "```", ""]
    MD_OUT.write_text("\n".join(lines))


def main():
    results = [(gpu, gpu_rows(gpu)) for gpu in GPUS]
    wb = Workbook()
    wb.remove(wb.active)
    for gpu, rows in results:
        build_summary(wb.create_sheet(), gpu, rows)
    for gpu, rows in results:
        build_ncu_details(wb, gpu, rows)
    wb.calculation.fullCalcOnLoad = True  # openpyxl does not evaluate the speedup formulas
    wb.save(OUT)
    write_markdown(results)
    print(f"wrote {OUT.relative_to(REPO)} and {MD_OUT.relative_to(REPO)} "
          f"({', '.join(f'{len(rows)} {gpu.sheet} rows' for gpu, rows in results)})")


if __name__ == "__main__":
    main()
