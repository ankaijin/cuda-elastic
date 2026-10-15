#!/usr/bin/env python3
"""Builds results/cuda_performance.xlsx from the latest profiling results
committed in this repository. Nothing is re-run: every number is read from an
existing ncu report, nsys sqlite export, benchmark log or sweep CSV, and every
number links to the file it came from on GitHub.

Only the latest run of each workload's current implementation is included.
Columns of the Summary sheet:
  CUDA Graph     the workload launched as a CUDA Graph
  Direct launch  the same kernels launched directly (only llama2c has this mode)

Kernel rows use the ncu kernel duration (gpu__time_duration.sum, mean over
the profiled launches of that kernel). End-to-end rows use the benchmark's
own timing (cudaEvent per iteration / graph replay) or, where only an nsys
trace exists, the median graph replay span from that trace.

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

    def __init__(self, us=None, ncu=None, nsys=None, csv=None, empty=""):
        self.us, self.ncu, self.nsys, self.csv = us, ncu, nsys, csv
        self.empty = empty

    def value_file(self):
        """The file the value itself was read from."""
        return self.ncu or self.csv or self.nsys


def ncu_cell(path, kernel, grid=None, nsys=None):
    return Cell(ncu_kernel_us(path, kernel, grid)[0], ncu=path, nsys=nsys)


GRAPH_ONLY = Cell(empty="n/a (graph-only harness)")


# ------------------------------------------------------------------- rows ---
# Each row: (workload, kernel/stage, problem size, metric, graph, direct, notes)

KERNEL_TIME = "Kernel time (ncu)"


def llama2c_rows():
    p = "llama2c/profiles/"
    w = "llama2c (coalesced matmul)"
    size = "batch=512, dim=16384, rows=8192"
    graph = f"{p}ncu-large-coalesced-graph-20261002-160814.ncu-rep"
    direct = f"{p}ncu-large-coalesced-direct-20261002-160814.ncu-rep"
    rows = [(w, kernel, size, KERNEL_TIME, ncu_cell(graph, kernel), ncu_cell(direct, kernel), note)
            for kernel, note in (
                ("rmsnorm_kernel", ""),
                ("softmax_kernel", ""),
                ("matmul_coalesced_kernel", "Warp-per-row matmul with coalesced weight reads."))]

    sweep = f"{p}sweep-fill-20261002-152612-speedup.csv"
    row = sweep_row(sweep, batch=1024, size=16384, rows=8192, repeats=1)
    rows.append((w, "End-to-end iteration (rmsnorm + softmax + matmul)",
                 "batch=1024, dim=16384, rows=8192", "Iteration time (cudaEvent)",
                 Cell(float(row["graph_gpu_ms"]) * 1e3, csv=sweep),
                 Cell(float(row["direct_gpu_ms"]) * 1e3, csv=sweep),
                 "Mean of 3 interleaved trials from the GPU-filling sweep; no per-iteration "
                 "copies. The sweep CSV has smaller sizes too."))
    return rows


def loom_rows():
    p = "loom_msa/profiles/"
    w = "loom_msa (multi-sensor attention)"
    bench = loom_benchmark()
    bench_src = "loom_msa/build/benchmark.txt"
    rows = []
    for shape, label, ncu, nsys in (
            ("1x1000x50000", "1,000 queries x 50,000 context",
             f"{p}ncu-all-kernels-msa_optimized-20260925-123922.ncu-rep",
             f"{p}nsys-msa_optimized-20260925-123919.sqlite"),
            ("1x25000x250000", "25,000 queries x 250,000 context",
             f"{p}ncu-all-kernels-msa_optimized_large-20260925-142052.ncu-rep",
             f"{p}nsys-msa_optimized_large-20260925-142045.sqlite")):
        for kernel, note in (
                ("project_audio_kernel", ""),
                ("project_imu_kernel", ""),
                ("fuse_attention_kernel", "Single-pass online softmax, shared-memory context "
                                          "tiles, 8-way context split (FlashAttention-style)."),
                ("reduce_statistics_kernel", "Multi-warp shuffle reduction that also merges "
                                             "the split partials.")):
            rows.append((w, kernel, label, KERNEL_TIME, ncu_cell(ncu, kernel, nsys=nsys),
                         GRAPH_ONLY, note))
        s = bench[(shape, "random")]["optimized"]
        for key, stage, note in (
                ("attention_us", "Attention kernel (in-graph events)", ""),
                ("end_to_end_us", "End-to-end graph replay",
                 "Includes H2D input and D2H output copies.")):
            rows.append((w, stage, label, "Replay time (cudaEvent)",
                         Cell(s[key]["median"], csv=bench_src,
                              nsys=nsys if key == "end_to_end_us" else None),
                         GRAPH_ONLY,
                         "benchmark.py: median of 5 runs, random inputs. " + note))
    return rows


def pagerank_rows():
    p = "pagerank/profiles/"
    w = "pagerank (GAPBS)"
    size = "2^20 vertices, 16 edges/vertex, 20 iterations"
    ncu = f"{p}ncu-pagerank-20261007-152417.ncu-rep"
    nsys = f"{p}nsys-20261007-152346.sqlite"
    replay_us, n = nsys_node_replay_us(nsys)
    return [
        (w, "initialize_scores", size, KERNEL_TIME, ncu_cell(ncu, "initialize_scores", nsys=nsys),
         GRAPH_ONLY, ""),
        (w, "pagerank_update (one iteration)", size, KERNEL_TIME,
         ncu_cell(ncu, "pagerank_update", grid=32768, nsys=nsys), GRAPH_ONLY,
         "Pull gather of precomputed contributions. Full-graph iterations only; the first "
         "iteration runs as 8 upload-overlapped chunks of ~41 us each."),
        (w, "End-to-end graph replay (H2D + 20 iterations)", size, "Replay time (nsys)",
         Cell(replay_us, nsys=nsys), GRAPH_ONLY,
         f"Median span of the {n} replays with transfers in the nsys node trace. The CSR graph "
         "is uploaded from pinned memory in 8 chunks overlapped with the first iteration. "
         "The binary's own cudaEvent timing in the same run: 7.684 ms with transfers, "
         "4.841 ms compute only."),
    ]


def mlperf_rows():
    a = "mlperf-anomaly/profiles/"
    c = "mlperf-classification/profiles/"
    anomaly = (f"{a}ncu-anomaly_mse-20261007-144943.ncu-rep",
               f"{a}nsys-anomaly_mse-20261007-144939.sqlite")
    kws = (f"{c}ncu-kws_postprocess-20261007-144950.ncu-rep",
           f"{c}nsys-kws_postprocess-20261007-144947.sqlite")
    vww = (f"{c}ncu-vww_preprocess-20261007-144958.ncu-rep",
           f"{c}nsys-vww_preprocess-20261007-144953.sqlite")
    return [(w, kernel, size, KERNEL_TIME, ncu_cell(ncu, kernel.split(" ")[0], nsys=nsys),
             GRAPH_ONLY, "")
            for w, kernel, size, (ncu, nsys) in (
                ("mlperf-tiny anomaly", "mse_partial_kernel", "64M elements", anomaly),
                ("mlperf-tiny anomaly", "mse_finalize_kernel", "64M elements", anomaly),
                ("mlperf-tiny KWS", "kws_postprocess_kernel (argmax)", "12 classes x 4M rows", kws),
                ("mlperf-tiny VWW", "vww_preprocess_kernel", "256M bytes", vww))]


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


def build_summary(wb, rows):
    ws = wb.active
    ws.title = "Summary"
    ws["A1"] = "CUDA performance results (latest runs)"
    ws["A1"].font = Font(bold=True, size=14)
    ws["A2"] = (f"GPU: NVIDIA GeForce {GPU}. Times in microseconds (lower is better). "
                "Latest run of each workload's current implementation. Each value links to its "
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
    for c in ws[1]:
        c.font, c.fill = HEADER_FONT, HEADER_FILL
        c.alignment = Alignment(wrap_text=True, vertical="center")
    for i, w in enumerate([52, 9, 40] + [16] * len(NCU_DETAIL_METRICS), 1):
        ws.column_dimensions[get_column_letter(i)].width = w
    ws.freeze_panes = "A2"


def write_markdown(rows):
    """GitHub-readable copy of the Summary sheet; values link to their source files."""
    def fmt(c):
        if c.us is None:
            return "—"
        return f"[{c.us:,.2f}]({link(c.value_file())})"

    def files(c):
        return " ".join(f"[{kind}]({link(path)})" for kind, path in
                        (("ncu", c.ncu), ("nsys", c.nsys)) if path)

    lines = [
        "# CUDA performance results",
        "",
        f"GPU: NVIDIA GeForce {GPU}. Times in microseconds (lower is better). "
        "Latest run of each workload's current implementation, generated by "
        "`build_spreadsheet.py` from profiles committed in this repo. The spreadsheet "
        "[`cuda_performance.xlsx`](cuda_performance.xlsx) also has the ncu metrics of every "
        "profiled launch.",
        "",
        "**Direct** = the same kernels launched without a CUDA Graph (only llama2c has this "
        "mode; the other harnesses are graph-only). — = not available.",
        "",
        "| Workload | Kernel / stage | Problem size | Metric | CUDA Graph | Graph profiles | "
        "Direct | Direct profiles | Graph speedup |",
        "|---|---|---|---|--:|---|--:|---|--:|",
    ]
    for workload, kernel, size, metric, graph, direct, _ in rows:
        speedup = f"{direct.us / graph.us:.2f}x" if direct.us is not None else ""
        lines.append(f"| {workload} | {kernel} | {size} | {metric} | {fmt(graph)} | {files(graph)} | "
                     f"{fmt(direct)} | {files(direct)} | {speedup} |")
    lines += ["", "Regenerate (needs `ncu` on PATH and `pip install openpyxl`):", "",
              "```bash", "python3 results/build_spreadsheet.py", "```", ""]
    MD_OUT.write_text("\n".join(lines))


def main():
    rows = llama2c_rows() + loom_rows() + pagerank_rows() + mlperf_rows()
    wb = Workbook()
    build_summary(wb, rows)
    build_ncu_details(wb, rows)
    wb.calculation.fullCalcOnLoad = True  # openpyxl does not evaluate the speedup formulas
    wb.save(OUT)
    write_markdown(rows)
    print(f"wrote {OUT.relative_to(REPO)} and {MD_OUT.relative_to(REPO)} ({len(rows)} summary rows)")


if __name__ == "__main__":
    main()
