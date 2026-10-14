#!/usr/bin/env python3
"""Builds results/cuda_performance.xlsx from the profiling results committed
in this repository. Nothing is re-run: every number is read from an existing
ncu report, nsys sqlite export, benchmark log or sweep CSV, and every number
links to the file it came from on GitHub.

Columns of the Summary sheet:
  Baseline       naive / untiled kernels, launched directly (no CUDA Graph)
  CUDA Graph     the same naive kernels, launched as a CUDA Graph
  Optimized      our optimized kernels (graph-launched)
  SOTA library   a library reference implementation (none measured yet)

Kernel rows use the ncu kernel duration (gpu__time_duration.sum, mean over
the profiled launches of that kernel). End-to-end rows use the benchmark's
own timing (cudaEvent per iteration / graph replay) or, where only an nsys
trace exists, the median graph replay duration from that trace.

Requires: ncu (Nsight Compute CLI) on PATH and openpyxl.
Usage:    python3 results/build_spreadsheet.py
"""

import csv
import io
import json
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
GPU = "RTX 3090 (sm_86, 24 GB)"


# ---------------------------------------------------------------- loaders ---

def git(*args):
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True)


def link(path):
    """GitHub URL for path, pinned to the commit that last changed it when that
    commit is already on origin/main; files not merged yet link to main."""
    commit = git("log", "-1", "--format=%H", "--", path).stdout.strip()
    on_main = commit and git("merge-base", "--is-ancestor", commit, "origin/main").returncode == 0
    return f"{GITHUB}/{commit if on_main else 'main'}/{path}"


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


def kernel_launches(path, kernel, grid=None):
    """Launches of `kernel` (name prefix after any namespace) in an ncu report."""
    def name(row):
        return row["Kernel Name"][0].split("(")[0].split("::")[-1].removeprefix("void ")
    return [r for r in ncu_rows(path)
            if name(r).startswith(kernel)
            and (grid is None or number(r["launch__grid_size"][0]) == grid)]


def ncu_kernel_us(path, kernel, grid=None):
    launches = kernel_launches(path, kernel, grid)
    if not launches:
        raise ValueError(f"{kernel} (grid={grid}) not found in {path}")
    times = [number(r["gpu__time_duration.sum"][0]) * TIME_SCALE_TO_US[r["gpu__time_duration.sum"][1]]
             for r in launches]
    return statistics.mean(times), len(times)


def nsys_graph_replay_us(path):
    """Median CUDA Graph replay duration in an nsys trace (--cuda-graph-trace=graph)."""
    with sqlite3.connect(REPO / path) as db:
        durations = [r[0] / 1e3 for r in db.execute(
            "SELECT end - start FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE")]
    return statistics.median(durations), len(durations)


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


def loom_benchmark():
    runs = json.loads((REPO / "loom_msa/build/benchmark.json").read_text())
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

    def __init__(self, us=None, ncu=None, nsys=None, csv=None, source="", empty=""):
        self.us, self.ncu, self.nsys, self.csv = us, ncu, nsys, csv
        self.source, self.empty = source, empty

    def value_file(self):
        """The file the value itself was read from."""
        return self.ncu or self.csv or self.nsys


def ncu_cell(path, kernel, grid=None, nsys=None, note=""):
    us, n = ncu_kernel_us(path, kernel, grid)
    source = f"ncu gpu__time_duration, mean of {n} launch{'es' * (n > 1)} of {kernel}"
    if grid is not None:
        source += f" (grid={grid})"
    return Cell(us, ncu=path, nsys=nsys, source=source + (f"; {note}" if note else ""))


NOT_MEASURED = "not measured"
NO_DIRECT = "not measured (harness is graph-only)"


# ------------------------------------------------------------------- rows ---
# Each row: (workload, kernel/stage, problem size, metric, baseline, graph,
#            optimized, sota, notes)

def llama2c_rows():
    p = "llama2c/profiles/"
    small = "batch=1, dim=64, rows=64, 16 repeats/iter"
    large = "batch=512, dim=16384, rows=8192"
    sota_mm = "not measured (candidate: cuBLAS SGEMM)"
    rows = []
    for size, tag in ((small, "small"), (large, "large")):
        direct = f"{p}ncu-{tag}-direct-20261002-151558.ncu-rep"
        graph = f"{p}ncu-{tag}-graph-20261002-151558.ncu-rep"
        for kernel in ("rmsnorm_kernel", "softmax_kernel"):
            rows.append(("llama2c", kernel, size, "Kernel time (ncu)",
                         ncu_cell(direct, kernel), ncu_cell(graph, kernel),
                         Cell(empty="no optimized version (kernel unchanged)"),
                         Cell(empty=NOT_MEASURED), ""))
        if tag == "large":
            opt = ncu_cell(f"{p}ncu-large-coalesced-graph-20261002-160814.ncu-rep",
                           "matmul_coalesced_kernel")
            opt_direct_us, _ = ncu_kernel_us(
                f"{p}ncu-large-coalesced-direct-20261002-160814.ncu-rep", "matmul_coalesced_kernel")
            mm_note = (f"Optimized = warp-per-row coalesced matmul. Same kernel launched "
                       f"directly: {opt_direct_us:.1f} us (ncu-large-coalesced-direct).")
        else:
            opt = Cell(empty="not measured at this size")
            mm_note = ""
        rows.append(("llama2c", "matmul (naive matmul_kernel; optimized: matmul_coalesced_kernel)", size, "Kernel time (ncu)",
                     ncu_cell(direct, "matmul_kernel"), ncu_cell(graph, "matmul_kernel"),
                     opt, Cell(empty=sota_mm), mm_note))

    # End-to-end iteration time from the launch-mode sweep (naive matmul,
    # 3 interleaved trials each, mean).
    sweep = f"{p}sweep-20260929-160445-speedup.csv"
    for size, cfg, note in (
            (small, dict(batch=1, size=64, rows=64, repeats=16, copies="nocopy"), ""),
            (large, dict(batch=512, size=16384, rows=8192, repeats=1, copies="copy"),
             "Includes per-iteration H2D copies. nsys trace (graph cell) is the 09-16 build "
             "with the same naive kernels.")):
        row = sweep_row(sweep, **cfg)
        src = "sweep: GPU ms/iter (cudaEvent), mean of 3 trials"
        nsys = f"{p}nsys-20260916-132456.sqlite" if cfg["batch"] == 512 else None
        rows.append(("llama2c", "End-to-end iteration (rmsnorm+softmax+matmul)", size,
                     "Iteration time (cudaEvent)",
                     Cell(float(row["direct_gpu_ms"]) * 1e3, csv=sweep, source=src),
                     Cell(float(row["graph_gpu_ms"]) * 1e3, nsys=nsys, csv=sweep, source=src),
                     Cell(empty="not measured at this size"), Cell(empty=NOT_MEASURED), note))
    return rows


def loom_rows():
    p = "loom_msa/profiles/"
    bench = loom_benchmark()
    bench_src = "loom_msa/build/benchmark.txt"
    sota = "not measured (FlashAttention not applicable: head dim 4)"
    base_ncu = f"{p}ncu-all-kernels-20260911-022551.ncu-rep"
    base_nsys = f"{p}nsys-20260911-022545.sqlite"
    rows = []
    for shape, label, opt_ncu, opt_nsys in (
            ("1x1000x50000", "1,000 queries x 50,000 context",
             f"{p}ncu-all-kernels-msa_optimized-20260925-123922.ncu-rep",
             f"{p}nsys-msa_optimized-20260925-123919.sqlite"),
            ("1x25000x250000", "25,000 queries x 250,000 context",
             f"{p}ncu-all-kernels-msa_optimized_large-20260925-142052.ncu-rep",
             f"{p}nsys-msa_optimized_large-20260925-142045.sqlite")):
        medium = shape == "1x1000x50000"
        for kernel in ("project_audio_kernel", "project_imu_kernel",
                       "fuse_attention_kernel", "reduce_statistics_kernel"):
            base = (ncu_cell(base_ncu, kernel) if medium
                    else Cell(empty="not profiled with ncu at this size"))
            note = ""
            if kernel == "fuse_attention_kernel":
                note = ("Optimized = single-pass online softmax, shared-memory context tiles, "
                        "8-way context split (FlashAttention-style).")
            elif kernel == "reduce_statistics_kernel":
                note = "Optimized = multi-warp shuffle reduction that also merges the split partials."
            elif kernel.startswith("project"):
                note = "Projections intentionally unchanged in the optimized variant."
            rows.append(("loom_msa (multi-sensor attention)", kernel, label, "Kernel time (ncu)",
                         Cell(empty=NO_DIRECT), base, ncu_cell(opt_ncu, kernel),
                         Cell(empty=sota if kernel == "fuse_attention_kernel" else NOT_MEASURED),
                         note))
        s = bench[(shape, "random")]
        src = "benchmark.py: median of 5 interleaved runs, random inputs (cudaEvent in graph)"
        for metric_key, stage in (("attention_us", "Attention kernel (in-graph events)"),
                                  ("end_to_end_us", "End-to-end graph replay")):
            rows.append(("loom_msa (multi-sensor attention)", stage, label,
                         "Replay time (cudaEvent)",
                         Cell(empty=NO_DIRECT),
                         Cell(s["baseline"][metric_key]["median"],
                              nsys=base_nsys if medium and metric_key == "end_to_end_us" else None,
                              csv=bench_src, source=src),
                         Cell(s["optimized"][metric_key]["median"],
                              nsys=opt_nsys if metric_key == "end_to_end_us" else None,
                              csv=bench_src, source=src),
                         Cell(empty=NOT_MEASURED),
                         "Includes H2D input and D2H output copies." if metric_key == "end_to_end_us" else ""))
    return rows


def pagerank_rows():
    p = "pagerank/profiles/"
    size = "2^20 vertices, 16 edges/vertex, 20 iterations"
    base_ncu = f"{p}ncu-pagerank-20260916-134844.ncu-rep"
    base_nsys = f"{p}nsys-20260916-134844.sqlite"
    opt_ncu = f"{p}ncu-pagerank-20261007-152417.ncu-rep"
    opt_nsys = f"{p}nsys-20261007-152346.sqlite"
    w = "pagerank (GAPBS)"
    replay_us, n = nsys_graph_replay_us(base_nsys)
    opt_replay_us, opt_n = nsys_node_replay_us(opt_nsys)
    return [
        (w, "initialize_scores", size, "Kernel time (ncu)", Cell(empty=NO_DIRECT),
         ncu_cell(base_ncu, "initialize_scores", nsys=base_nsys),
         ncu_cell(opt_ncu, "initialize_scores", nsys=opt_nsys), Cell(empty=NOT_MEASURED), ""),
        (w, "pagerank_update (one iteration)", size, "Kernel time (ncu)", Cell(empty=NO_DIRECT),
         ncu_cell(base_ncu, "pagerank_update", nsys=base_nsys),
         ncu_cell(opt_ncu, "pagerank_update", grid=32768, nsys=opt_nsys),
         Cell(empty="not measured (candidate: Gunrock / cuGraph PageRank)"),
         "Baseline = pull gather reading score/out_degree per edge. Optimized = gathers "
         "precomputed contributions; full-graph iterations only (the first iteration runs "
         "as 8 upload-overlapped chunks of ~41 us each)."),
        (w, "End-to-end graph replay (H2D + 20 iterations)", size, "Replay time (nsys)",
         Cell(empty=NO_DIRECT),
         Cell(replay_us, nsys=base_nsys,
              source=f"nsys CUPTI_ACTIVITY_KIND_GRAPH_TRACE, median of {n} replays"),
         Cell(opt_replay_us, nsys=opt_nsys,
              source=f"nsys node trace: median span of the {opt_n} replays with transfers"),
         Cell(empty=NOT_MEASURED),
         "Both replays upload the CSR graph every replay: baseline from pageable memory, "
         "optimized from pinned memory in 8 chunks overlapped with the first iteration. "
         "The optimized binary's own cudaEvent timing in the same run: 7.684 ms with transfers, "
         "4.841 ms compute only."),
    ]


def mlperf_rows():
    single = "only implementation (no naive baseline exists)"
    note = ("Single graph-launched CUDA port, already vectorized with warp/block reductions, "
            "so it is filed under Optimized. ncu --set full.")
    a = "mlperf-anomaly/profiles/"
    c = "mlperf-classification/profiles/"
    anomaly = (f"{a}ncu-anomaly_mse-20261007-144943.ncu-rep",
               f"{a}nsys-anomaly_mse-20261007-144939.sqlite")
    kws = (f"{c}ncu-kws_postprocess-20261007-144950.ncu-rep",
           f"{c}nsys-kws_postprocess-20261007-144947.sqlite")
    vww = (f"{c}ncu-vww_preprocess-20261007-144958.ncu-rep",
           f"{c}nsys-vww_preprocess-20261007-144953.sqlite")
    rows = []
    for w, kernel, size, (ncu, nsys), sota in (
            ("mlperf-tiny anomaly", "mse_partial_kernel", "64M elements", anomaly,
             "not measured (candidate: CUB DeviceReduce)"),
            ("mlperf-tiny anomaly", "mse_finalize_kernel", "64M elements", anomaly, NOT_MEASURED),
            ("mlperf-tiny KWS", "kws_postprocess_kernel (argmax)", "12 classes x 4M rows", kws,
             "not measured (candidate: CUB segmented ArgMax)"),
            ("mlperf-tiny VWW", "vww_preprocess_kernel", "256M bytes", vww, NOT_MEASURED)):
        rows.append((w, kernel, size, "Kernel time (ncu)",
                     Cell(empty=single), Cell(empty=single),
                     ncu_cell(ncu, kernel.split(" ")[0], nsys=nsys), Cell(empty=sota), note))
    return rows


# ----------------------------------------------------------------- styling ---

BLUE = Font(color="0563C1", underline="single")
BOLD = Font(bold=True)
GREY = Font(italic=True, color="808080")
HEADER_FILL = PatternFill("solid", fgColor="1F3864")
HEADER_FONT = Font(bold=True, color="FFFFFF")
GROUP_FILLS = ["FCE4D6", "E2EFDA", "DDEBF7", "EDE4F7"]
THIN = Side(style="thin", color="BFBFBF")
BOX = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
WRAP = Alignment(wrap_text=True, vertical="top")
GROUPS = ["Baseline\n(naive, no graph)", "CUDA Graph\n(naive + graph)",
          "Optimized (ours)\n(+ graph)", "SOTA library"]


def put_link(ws, row, col, path, text):
    c = ws.cell(row, col, text)
    c.hyperlink = link(path)
    c.font = BLUE
    return c


def build_summary(wb, rows):
    ws = wb.active
    ws.title = "Summary"
    ws["A1"] = "CUDA performance results"
    ws["A1"].font = Font(bold=True, size=14)
    ws["A2"] = (f"GPU: NVIDIA GeForce {GPU}. Times in microseconds (lower is better). "
                "Each value links to its source; ncu-rep / nsys sqlite columns link the profile files "
                "on GitHub. Speedups are formulas over the time columns. ncu replays each kernel in "
                "isolation, so ncu kernel times barely change with launch mode; the CUDA Graph "
                "benefit (launch overhead) shows up in the end-to-end rows.")
    ws["A2"].alignment = Alignment(wrap_text=True)
    ws.merge_cells("A2:T2")
    ws.row_dimensions[2].height = 45

    fixed = ["Workload", "Kernel / stage", "Problem size", "GPU", "Metric"]
    hr, sr = 4, 5
    for i, h in enumerate(fixed, 1):
        ws.cell(hr, i, h)
        ws.merge_cells(start_row=hr, start_column=i, end_row=sr, end_column=i)
    col = len(fixed) + 1
    group_cols = []
    for g, fill in zip(GROUPS, GROUP_FILLS):
        ws.cell(hr, col, g)
        ws.merge_cells(start_row=hr, start_column=col, end_row=hr, end_column=col + 2)
        for j, sub in enumerate(("Time (us)", "ncu-rep", "nsys sqlite")):
            ws.cell(sr, col + j, sub)
        group_cols.append(col)
        col += 3
    tail = ["Graph speedup\n(Baseline / Graph)", "Optimized speedup\n(Graph / Optimized)", "Notes"]
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
        for i, v in enumerate((workload, kernel, size, GPU, metric), 1):
            ws.cell(r, i, v).alignment = WRAP
        ws.cell(r, 1).font = BOLD
        for cell_spec, gc, fill in zip(cells, group_cols, GROUP_FILLS):
            t = ws.cell(r, gc)
            if cell_spec.us is None:
                t.value, t.font = cell_spec.empty, GREY
            else:
                t.value = round(cell_spec.us, 3)
                t.number_format = "#,##0.00"
                if cell_spec.value_file():
                    t.hyperlink = link(cell_spec.value_file())
                    t.font = Font(color="0563C1")
            t.alignment = WRAP
            if cell_spec.ncu:
                put_link(ws, r, gc + 1, cell_spec.ncu, Path(cell_spec.ncu).name)
            if cell_spec.nsys:
                put_link(ws, r, gc + 2, cell_spec.nsys, Path(cell_spec.nsys).name)
            for c in range(gc, gc + 3):
                ws.cell(r, c).fill = PatternFill("solid", fgColor=fill)
                ws.cell(r, c).alignment = WRAP
        b, g, o = (get_column_letter(group_cols[i]) for i in range(3))
        for c, (num, den) in ((col, (b, g)), (col + 1, (g, o))):
            f = ws.cell(r, c, f'=IF(AND(ISNUMBER({num}{r}),ISNUMBER({den}{r})),{num}{r}/{den}{r},"")')
            f.number_format = '0.00"x"'
            f.alignment = Alignment(horizontal="center", vertical="top")
        ws.cell(r, col + 2, notes).alignment = WRAP
        for c in range(1, last_col + 1):
            ws.cell(r, c).border = BOX
        r += 1

    widths = [22, 30, 24, 14, 16] + [13, 22, 22] * 4 + [14, 15, 60]
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = w
    ws.freeze_panes = ws.cell(sr + 1, 3)
    return ws


def build_sources(wb, rows):
    ws = wb.create_sheet("Sources")
    header = ["Workload", "Kernel / stage", "Problem size", "Column", "Time (us)",
              "How the value was computed", "Value source file"]
    ws.append(header)
    for workload, kernel, size, _, *cells, _ in rows:
        for g, c in zip(GROUPS, cells):
            if c.us is None:
                continue
            src = c.value_file()
            ws.append([workload, kernel, size, g.replace("\n", " "), round(c.us, 3), c.source, src])
            put_link(ws, ws.max_row, 7, src, src)
    style_table(ws, [22, 34, 30, 30, 12, 70, 70])


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


def build_ncu_details(wb, rows):
    """Every launch in every ncu report the Summary uses, with key metrics."""
    ws = wb.create_sheet("ncu metrics")
    reports = []
    for _, _, _, _, *cells, _ in rows:
        for c in cells:
            if c.ncu and c.ncu not in reports:
                reports.append(c.ncu)
    ws.append(["ncu report", "Launch #", "Kernel"] + [label for _, label in NCU_DETAIL_METRICS])
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
    style_table(ws, [52, 9, 40] + [16] * len(NCU_DETAIL_METRICS))


# Every committed profile file, including ones the Summary does not use.
# GPU-metrics traces are committed as .sqlite.zst (the .sqlite is 120+ MB).
INVENTORY = [
    ("llama2c/profiles/ncu-small-direct-20261002-151558.ncu-rep", "llama2c", "small, naive, direct", "Summary"),
    ("llama2c/profiles/ncu-small-graph-20261002-151558.ncu-rep", "llama2c", "small, naive, graph", "Summary"),
    ("llama2c/profiles/ncu-large-direct-20261002-151558.ncu-rep", "llama2c", "large, naive, direct", "Summary"),
    ("llama2c/profiles/ncu-large-graph-20261002-151558.ncu-rep", "llama2c", "large, naive, graph", "Summary"),
    ("llama2c/profiles/ncu-large-coalesced-direct-20261002-160814.ncu-rep", "llama2c", "large, coalesced matmul, direct", "Notes"),
    ("llama2c/profiles/ncu-large-coalesced-graph-20261002-160814.ncu-rep", "llama2c", "large, coalesced matmul, graph", "Summary"),
    ("llama2c/profiles/nsys-20260916-132456.sqlite", "llama2c", "large (batch 512), naive, graph; 09-16 build", "Summary"),
    ("llama2c/profiles/ncu-llama2c-20260913-051736.ncu-rep", "llama2c", "1-block smoke shape; 09-13 build", "superseded"),
    ("llama2c/profiles/ncu-llama2c-20260916-112426.ncu-rep", "llama2c", "1-block smoke shape; 09-16 build", "superseded"),
    ("llama2c/profiles/ncu-llama2c-20260916-115531.ncu-rep", "llama2c", "pre-batch shape; 09-16 build", "superseded"),
    ("llama2c/profiles/ncu-llama2c-20260916-121319.ncu-rep", "llama2c", "pre-batch large dim; 09-16 build", "superseded"),
    ("llama2c/profiles/ncu-llama2c-20260916-132607.ncu-rep", "llama2c", "large (batch 512), naive, graph; 09-16 build", "superseded by ncu-large-graph"),
    ("llama2c/profiles/nsys-20260916-112418.sqlite", "llama2c", "1-block smoke shape; 09-16 build", "superseded"),
    ("llama2c/profiles/nsys-20260916-121340.sqlite", "llama2c", "pre-batch large dim; 09-16 build", "superseded"),
    ("llama2c/profiles/sweep-20260929-160445-speedup.csv", "llama2c", "launch-overhead sweep, naive matmul", "Summary"),
    ("llama2c/profiles/sweep-fill-20261002-152612-speedup.csv", "llama2c", "GPU-filling sweep, coalesced matmul", "Launch sweeps sheet"),
    ("loom_msa/profiles/ncu-all-kernels-20260911-022551.ncu-rep", "loom_msa", "1000x50000, baseline kernels (= msa_baseline.cu)", "Summary"),
    ("loom_msa/profiles/nsys-20260911-022545.sqlite", "loom_msa", "1000x50000, baseline kernels", "Summary"),
    ("loom_msa/profiles/ncu-all-kernels-msa_optimized-20260925-123922.ncu-rep", "loom_msa", "1000x50000, msa_optimized", "Summary"),
    ("loom_msa/profiles/ncu-all-kernels-msa_optimized_large-20260925-142052.ncu-rep", "loom_msa", "25000x250000, msa_optimized", "Summary"),
    ("loom_msa/profiles/nsys-msa_optimized-20260925-123919.sqlite", "loom_msa", "1000x50000, msa_optimized (repeat inputs)", "Summary"),
    ("loom_msa/profiles/nsys-msa_optimized_large-20260925-142045.sqlite", "loom_msa", "25000x250000, msa_optimized", "Summary"),
    ("loom_msa/build/benchmark.txt", "loom_msa", "baseline vs optimized benchmark, all shapes", "Summary"),
    ("loom_msa/profiles/ncu-all-kernels-20260909-065813.ncu-rep", "loom_msa", "1000x50000, pre-3a070a7 kernels (attention 1.6 ms)", "superseded"),
    ("loom_msa/profiles/nsys-20260909-062128.sqlite", "loom_msa", "1000x50000, pre-3a070a7 kernels", "superseded"),
    ("loom_msa/profiles/ncu-fuse-attention-20260909-062746.ncu-rep", "loom_msa", "attention only, pre-3a070a7", "superseded"),
    ("loom_msa/profiles/ncu-fuse-attention-20260909-064023.ncu-rep", "loom_msa", "mis-targeted (captured project_imu)", "superseded"),
    ("loom_msa/profiles/ncu-fuse-attention-20260909-064445.ncu-rep", "loom_msa", "mis-targeted (captured project_imu)", "superseded"),
    ("loom_msa/reports/attention_nodes.ncu-rep", "loom_msa", "batch 4096 x 4x4, early version", "superseded"),
    ("loom_msa/reports/attention_nsys.sqlite", "loom_msa", "4x4 smoke, early version", "superseded"),
    ("loom_msa/reports/batch/attention_nsys.sqlite", "loom_msa", "batch 4096 x 4x4, early version", "superseded"),
    ("loom_msa/reports/batch/reduce_statistics_parallel.ncu-rep", "loom_msa", "batch 4096, parallel reduction experiment", "superseded"),
    ("loom_msa/reports/large/attention_nodes.ncu-rep", "loom_msa", "1000x50000, early version", "superseded"),
    ("loom_msa/reports/large/attention_nsys.sqlite", "loom_msa", "1000x50000, early version", "superseded"),
    ("pagerank/profiles/ncu-pagerank-20260916-134844.ncu-rep", "pagerank", "2^20 vertices, baseline (commit ee1391b)", "Summary"),
    ("pagerank/profiles/nsys-20260916-134844.sqlite", "pagerank", "2^20 vertices, baseline (commit ee1391b)", "Summary"),
    ("pagerank/profiles/ncu-pagerank-20261007-152417.ncu-rep", "pagerank", "2^20 vertices, optimized (commit e9f064f)", "Summary"),
    ("pagerank/profiles/nsys-20261007-152346.sqlite", "pagerank", "2^20 vertices, optimized, node-level trace", "Summary"),
    ("pagerank/profiles/ncu-pagerank-20260916-122658.ncu-rep", "pagerank", "4-vertex smoke graph", "superseded"),
    ("pagerank/profiles/nsys-20260916-122658.sqlite", "pagerank", "4-vertex smoke graph", "superseded"),
    ("mlperf-anomaly/profiles/ncu-anomaly_mse-20261007-144943.ncu-rep", "mlperf-tiny anomaly", "64M elements, ncu --set full", "Summary"),
    ("mlperf-classification/profiles/ncu-kws_postprocess-20261007-144950.ncu-rep", "mlperf-tiny KWS", "12 x 4M rows, ncu --set full", "Summary"),
    ("mlperf-classification/profiles/ncu-vww_preprocess-20261007-144958.ncu-rep", "mlperf-tiny VWW", "256M bytes, ncu --set full", "Summary"),
    ("mlperf-anomaly/profiles/nsys-anomaly_mse-20261007-144939.sqlite", "mlperf-tiny anomaly", "64M elements, node-level trace, 20 replays", "Summary"),
    ("mlperf-classification/profiles/nsys-kws_postprocess-20261007-144947.sqlite", "mlperf-tiny KWS", "12 x 4M rows, node-level trace, 20 replays", "Summary"),
    ("mlperf-classification/profiles/nsys-vww_preprocess-20261007-144953.sqlite", "mlperf-tiny VWW", "256M bytes, node-level trace, 20 replays", "Summary"),
] + [
    (f"{d}/profiles/nsys-metrics-{name}-20261007-152610.{ext}", workload,
     f"{what}, nsys GPU-metrics sampling (full timing run)"
     + (", zstd-compressed (decompress: zstd -d)" if ext.endswith("zst") else ""),
     "GPU metrics (not in Summary)")
    for d, name, workload, what in (
        ("mlperf-anomaly", "anomaly_mse", "mlperf-tiny anomaly", "64M elements"),
        ("mlperf-classification", "kws_postprocess", "mlperf-tiny KWS", "12 x 4M rows"),
        ("mlperf-classification", "vww_preprocess", "mlperf-tiny VWW", "256M bytes"))
    for ext in ("sqlite.zst", "nsys-rep")
]


def build_inventory(wb):
    ws = wb.create_sheet("Profile files")
    ws.append(["File", "Workload", "What it profiles", "Used in"])
    for path, workload, what, used in INVENTORY:
        ws.append([path, workload, what, used])
        put_link(ws, ws.max_row, 1, path, path)
        if used.startswith("superseded"):
            for c in range(2, 5):
                ws.cell(ws.max_row, c).font = GREY
    style_table(ws, [78, 20, 52, 28])


def build_sweeps(wb):
    ws = wb.create_sheet("Launch sweeps")
    ws.append(["llama2c launch-mode sweeps: direct vs CUDA Graph per-iteration GPU time "
               "(mean of 3 interleaved trials). Speedup > 1 means the graph is faster."])
    ws["A1"].font = BOLD
    keep = ["batch", "size", "rows", "repeats", "copies", "matmul", "direct_gpu_ms", "graph_gpu_ms",
            "gpu_speedup", "direct_submit_us", "graph_submit_us", "submit_speedup",
            "graph_setup_ms", "break_even_replays"]
    for path in ("llama2c/profiles/sweep-20260929-160445-speedup.csv",
                 "llama2c/profiles/sweep-fill-20261002-152612-speedup.csv"):
        ws.append([])
        put_link(ws, ws.max_row + 1, 1, path, path).font = Font(bold=True, color="0563C1", underline="single")
        with open(REPO / path) as f:
            data = list(csv.DictReader(f))
        ws.append(keep)
        for c in range(1, len(keep) + 1):
            ws.cell(ws.max_row, c).font, ws.cell(ws.max_row, c).fill = HEADER_FONT, HEADER_FILL
        for row in data:
            row.setdefault("matmul", "naive")
            out = []
            for k in keep:
                v = row.get(k, "")
                try:
                    v = float(v)
                    v = int(v) if v.is_integer() else round(v, 4)
                except ValueError:
                    pass
                out.append(v)
            ws.append(out)
    for i in range(1, len(keep) + 1):
        ws.column_dimensions[get_column_letter(i)].width = 14


def style_table(ws, widths):
    for c in ws[1]:
        c.font, c.fill = HEADER_FONT, HEADER_FILL
        c.alignment = Alignment(wrap_text=True, vertical="center")
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = w
    ws.freeze_panes = "A2"


def write_markdown(rows):
    """GitHub-readable copy of the Summary sheet; values link to their source files."""
    def fmt(c):
        if c.us is None:
            return "—"
        text = f"{c.us:,.2f}"
        return f"[{text}]({link(c.value_file())})" if c.value_file() else text

    def speedup(a, b):
        return f"{a.us / b.us:.2f}x" if a.us is not None and b.us is not None else ""

    lines = [
        "# CUDA performance results",
        "",
        f"GPU: NVIDIA GeForce {GPU}. Times in microseconds (lower is better). "
        "Generated by `build_spreadsheet.py` from profiles already committed in this repo; "
        "the full table with per-cell ncu-rep / nsys sqlite links, value sources, every ncu "
        "metric and the launch sweeps is [`cuda_performance.xlsx`](cuda_performance.xlsx).",
        "",
        "Columns: **Baseline** = naive kernels, direct launches; **CUDA Graph** = the same "
        "kernels as a CUDA Graph; **Optimized** = our optimized kernels (graph-launched); "
        "**SOTA** = library reference (none measured yet). — = not measured.",
        "",
        "GPU-metrics nsys traces are committed as `.sqlite.zst` because the `.sqlite` exceeds "
        "GitHub's 100 MB limit; restore with `zstd -d <file>.sqlite.zst`.",
        "",
        "| Workload | Kernel / stage | Problem size | Metric | Baseline | CUDA Graph | Optimized | "
        "SOTA | Graph speedup | Optimized speedup |",
        "|---|---|---|---|--:|--:|--:|--:|--:|--:|",
    ]
    for workload, kernel, size, metric, base, graph, opt, sota, _ in rows:
        lines.append(f"| {workload} | {kernel} | {size} | {metric} | {fmt(base)} | {fmt(graph)} | "
                     f"{fmt(opt)} | {fmt(sota)} | {speedup(base, graph)} | {speedup(graph, opt)} |")
    lines += ["", "Regenerate (needs `ncu` on PATH and `pip install openpyxl`):", "",
              "```bash", "python3 results/build_spreadsheet.py", "```", ""]
    MD_OUT.write_text("\n".join(lines))


def main():
    rows = llama2c_rows() + loom_rows() + pagerank_rows() + mlperf_rows()
    wb = Workbook()
    build_summary(wb, rows)
    build_sources(wb, rows)
    build_ncu_details(wb, rows)
    build_inventory(wb)
    build_sweeps(wb)
    wb.calculation.fullCalcOnLoad = True  # openpyxl does not evaluate the speedup formulas
    wb.save(OUT)
    write_markdown(rows)
    print(f"wrote {OUT.relative_to(REPO)} and {MD_OUT.relative_to(REPO)} ({len(rows)} summary rows)")


if __name__ == "__main__":
    main()
