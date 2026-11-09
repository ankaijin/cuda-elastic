#!/usr/bin/env python3
"""Write results/loom_private_ablation/opt_results.xlsx from the selected stage runs.

Each GPU's inputs/<gpu>.json selects, per application, the summaries that
experiments/loom_private_ablation/run.py wrote for one recorded run:

  {"folder": "rtx3090", "label": "RTX 3090 (sm_86, 24 GB)",
   "applications": {"pagerank": {
      "check": "loom_private_ablation/pagerank/<run id>-check.json",
      "bench": "loom_private_ablation/pagerank/<run id>-bench.json",
      "profile": "loom_private_ablation/pagerank/<run id>-profile.json"}}}

Paths are relative to the GPU folder; null means pending. The check, bench
and profile summaries must share a source revision and stages.json hash, and
the bench binaries must be the ones the check validated.

Sheets, per GPU and application: the stage summary (median timings, speedups,
each optimization's share of the saving), the nsys kernels and transfers of
one end-to-end invocation per stage, and the ncu metrics per kernel and launch
configuration. An Environment sheet lists every run. Only the timings come
from bench records; the profiles describe one invocation each.
"""

import csv
import hashlib
import json
import statistics
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HERE = REPO / "results" / "loom_private_ablation"
EXPERIMENTS = REPO / "experiments" / "loom_private_ablation"
OUT = HERE / "opt_results.xlsx"
GITHUB = "https://github.com/ankaijin/cuda-elastic/blob"
GPUS = ["rtx3090", "rtx4090", "rtx5090"]
SCOPES = ("application", "compute")
FIELD = "batch_per_iteration_us"
TIME_SCALE_TO_US = {"ns": 1e-3, "nsecond": 1e-3, "us": 1.0, "usecond": 1.0,
                    "ms": 1e3, "msecond": 1e3, "s": 1e6, "second": 1e6}
# ncu --set basic metrics shown per kernel configuration: (header, metric).
NCU_METRICS = [
    ("SM throughput (%)", "sm__throughput.avg.pct_of_peak_sustained_elapsed"),
    ("Memory throughput (%)", "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"),
    ("DRAM throughput (%)", "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed"),
    ("L2 throughput (%)", "lts__throughput.avg.pct_of_peak_sustained_elapsed"),
    ("L1 throughput (%)", "l1tex__throughput.avg.pct_of_peak_sustained_active"),
    ("Achieved occupancy (%)", "sm__warps_active.avg.pct_of_peak_sustained_active"),
]


def read_json(path):
    data = json.loads(Path(path).read_text())
    data["_path"] = str(Path(path).resolve().relative_to(REPO))
    return data


def resolve(folder, value):
    if value is None:
        return None
    path = (REPO / folder / value).resolve()
    if REPO / folder not in path.parents or not path.is_file():
        raise ValueError(f"inputs/{folder}.json: {value} is missing or outside {folder}/")
    return path


def short_kernel(name):
    """`void <unnamed>::foo<(bool)0>(int *, ...)` -> `foo<(bool)0>`."""
    for prefix in ("<unnamed>::", "(anonymous namespace)::"):
        name = name.replace(prefix, "")
    name = name.removeprefix("void ")
    depth = 0
    for i, c in enumerate(name):
        depth += {"<": 1, ">": -1}.get(c, 0)
        if c == "(" and depth == 0:
            return name[:i]
    return name


def number(text):
    return float(text.replace(",", ""))


def load_app(folder, app, selection):
    stages = json.loads((EXPERIMENTS / app / "stages.json").read_text())
    loaded = {key: read_json(p) for key in ("check", "bench", "profile")
              if (p := resolve(folder, selection.get(key)))}
    revisions = {s["source_commit"] for s in loaded.values()}
    hashes = {s["stages_sha256"] for s in loaded.values()}
    if len(revisions) > 1 or len(hashes) > 1:
        raise ValueError(f"{folder} {app}: the selected summaries come from different "
                         f"revisions or stages.json files")
    current = hashlib.sha256((EXPERIMENTS / app / "stages.json").read_bytes()).hexdigest()
    if hashes and hashes != {current}:
        raise ValueError(f"{folder} {app}: stages.json changed since the selected runs")
    check, bench, profile = (loaded.get(k) for k in ("check", "bench", "profile"))
    if bench and check:
        validated = {b["sha256"] for b in check["binaries"].values()}
        stray = {r["case"] for r in bench["records"] if r["binary_sha256"] not in validated}
        if stray:
            raise ValueError(f"{folder} {app}: bench binaries of {sorted(stray)} were not "
                             "validated by the selected check")
    cases = []
    for case in stages["cases"]:
        row = dict(case)
        if bench:
            records = [read_json(REPO / r["record"]) for r in bench["records"]
                       if r["case"] == case["id"]]
            row["records"] = records
            for scope in SCOPES:
                ranked = sorted(records, key=lambda r: r["timing"][scope][FIELD])
                row[scope] = statistics.median(r["timing"][scope][FIELD] for r in records)
                row[f"{scope}_record"] = ranked[(len(ranked) - 1) // 2]["_path"]
        if check:
            row["check"] = check["binaries"][case["binary"]]
        if profile:
            row["profile"] = next(p for p in profile["profiles"] if p["case"] == case["id"])
        cases.append(row)
    return {"app": app, "title": stages["title"], "stages": stages, "cases": cases,
            "check": check, "bench": bench, "profile": profile}


def load_gpu(folder):
    path = HERE / "inputs" / f"{folder}.json"
    if not path.is_file():
        return None
    manifest = json.loads(path.read_text())
    apps = [load_app(folder, app, sel) for app, sel in manifest["applications"].items()]
    return {"folder": folder, "label": manifest["label"], "short": manifest["short"],
            "apps": apps}


def run_id(summary):
    return summary["stamp"] if summary else None


def metadata(folder, stamp):
    path = REPO / folder / "runs" / stamp / "metadata.json"
    return (json.loads(path.read_text()), str(path.relative_to(REPO))) if path.is_file() \
        else (None, None)


def nsys_rows(entry):
    with open(REPO / entry["nsys_kernels"]) as stream:
        kernels = [(short_kernel(r["Name"]), int(r["Instances"]), number(r["Total Time (ns)"]) / 1e3,
                    number(r["Avg (ns)"]) / 1e3) for r in csv.DictReader(stream)]
    with open(REPO / entry["nsys_memcpy"]) as stream:
        copies = [(r["Operation"], int(r["Count"]), number(r["Total Time (ns)"]) / 1e3,
                   number(r["Avg (ns)"]) / 1e3) for r in csv.DictReader(stream)]
    return kernels, copies


def ncu_groups(entry):
    """[(kernel, grid, block, registers, launches, {metric: median}, durations)] per
    kernel and launch configuration, in first-launch order."""
    with open(REPO / entry["ncu_raw"], newline="") as stream:
        rows = list(csv.reader(stream))
    header, units = rows[0], rows[1]
    launches = [{h: (v, u) for h, u, v in zip(header, units, row)} for row in rows[2:]]
    groups = {}
    for launch in launches:
        key = (short_kernel(launch["Kernel Name"][0]), launch["launch__grid_size"][0],
               launch["launch__block_size"][0], launch["launch__registers_per_thread"][0])
        groups.setdefault(key, []).append(launch)
    out = []
    for (kernel, grid, block, registers), group in groups.items():
        durations = [number(l["gpu__time_duration.sum"][0])
                     * TIME_SCALE_TO_US[l["gpu__time_duration.sum"][1]] for l in group]
        metrics = {}
        for _, metric in NCU_METRICS:
            values = [l.get(metric, ("", ""))[0] for l in group]
            metrics[metric] = statistics.median(number(v) for v in values) \
                if all(values) else None
        out.append((kernel, int(number(grid)), int(number(block)), int(number(registers)),
                    len(group), metrics, durations))
    return out


def link(path):
    """GitHub URL for path, pinned to the latest origin/main commit that has it."""
    log = subprocess.run(["git", "-C", str(REPO), "log", "--format=%H", "-1", "origin/main",
                          "--", path], capture_output=True, text=True).stdout.strip()
    return f"{GITHUB}/{log or 'main'}/{path}"


def workbook(gpus):
    from openpyxl import Workbook
    from openpyxl.comments import Comment
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter

    blue = Font(color="0563C1", underline="single")
    bold = Font(bold=True)
    grey = Font(italic=True, color="808080")
    header_fill = PatternFill("solid", fgColor="1F3864")
    header_font = Font(bold=True, color="FFFFFF")
    # Direct, Graph, and Optimized colours of the loom_private workbook; the
    # intermediate stages get a neutral fill.
    fills = {"direct": PatternFill("solid", fgColor="FCE4D6"),
             "graph": PatternFill("solid", fgColor="E2EFDA"),
             "stage": PatternFill("solid", fgColor="F2F2F2"),
             "optimized": PatternFill("solid", fgColor="DDEBF7")}
    thin = Side(style="thin", color="BFBFBF")
    box = Border(left=thin, right=thin, top=thin, bottom=thin)
    wrap = Alignment(wrap_text=True, vertical="top")
    links = {}
    book = Workbook()
    book.remove(book.active)

    def github(path):
        if path not in links:
            links[path] = link(path)
        return links[path]

    def put_link(ws, row, col, path, text):
        c = ws.cell(row, col, text)
        c.hyperlink, c.font = github(path), blue
        return c

    def fill_of(case, last):
        if case["stage"] == 0:
            return fills[case["launch"]]
        return fills["optimized"] if case["stage"] == last else fills["stage"]

    def sheet(title, lines, headers, widths, header_row):
        ws = book.create_sheet(title)
        ws.cell(1, 1, title).font = Font(bold=True, size=14)
        for i, text in enumerate(lines, 2):
            ws.merge_cells(start_row=i, start_column=1, end_row=i, end_column=len(headers))
            ws.cell(i, 1, text).alignment = wrap
            ws.row_dimensions[i].height = 30 if len(text) > 160 else 16
        for i, (header, width) in enumerate(zip(headers, widths), 1):
            c = ws.cell(header_row, i, header)
            c.font, c.fill, c.border = header_font, header_fill, box
            c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            ws.column_dimensions[get_column_letter(i)].width = width
        ws.row_dimensions[header_row].height = 44
        ws.freeze_panes = ws.cell(header_row + 1, 3)
        ws.sheet_view.showGridLines = False
        return ws

    def cell(ws, row, col, value, fmt=None, fill=None):
        c = ws.cell(row, col, value)
        c.border, c.alignment = box, wrap
        if fmt and isinstance(value, (int, float)):
            c.number_format = fmt
        if fill:
            c.fill = fill
        return c

    env_runs = []
    for gpu in gpus:
        for data in gpu["apps"]:
            cases, last = data["cases"], max(c["stage"] for c in data["cases"])
            graph = next(c for c in cases if c["stage"] == 0 and c["launch"] == "graph")
            final = next(c for c in cases if c["stage"] == last)
            bench, check, profile = data["bench"], data["check"], data["profile"]
            for summary in (check, bench, profile):
                if summary and summary["stamp"] not in [r[1] for r in env_runs]:
                    env_runs.append((gpu, summary["stamp"]))
            digests = sorted({c["check"]["output_digest"] for c in cases if "check" in c})
            sass = "; ".join(f"{s['binary']} and {s['reference']} {s['kernel']}: "
                             f"{s['instructions'][0]} instructions, "
                             f"{'identical' if s['identical'] else 'DIFFERENT'}"
                             for s in (check or {}).get("sass_equal", []))
            dims = cases[0]["records"][0]["dimensions"] if bench else {}
            meta, _ = metadata(gpu["folder"], run_id(bench)) if bench else (None, None)
            toolkit = (meta or {}).get("installed_tools", {}).get("nvcc") or ""
            toolkit = toolkit.strip().splitlines()[-1] if toolkit else "unknown"
            lines = [
                f"loom-private/{data['app']} on the {gpu['label']}: "
                + ", ".join(f"{k} {v:,}" for k, v in dims.items()
                            if isinstance(v, int) and k != "epsilon")
                + f". Run {run_id(bench) or 'pending'} at source {bench['source_commit'][:7] if bench else '—'}, "
                f"driver {bench['device']['driver'] if bench else '—'}, {toolkit}.",
                "Each row adds one optimization to the row above it; stage 0 is the suite's baseline "
                "and the last stage is the suite's optimized binary. Times are the median over "
                f"{bench['repeat'] if bench else '—'} repetitions of the harness's batch time per "
                "invocation (20 invocations after 3 warmups), exactly as in the loom_private "
                "workbook. Application scope includes the uploads and the score download; compute "
                "scope is the kernels with resident inputs. Stages after the baseline run as CUDA "
                "Graphs.",
                f"Correctness: every binary passed the contract's large check in Direct and Graph "
                f"mode with output digest {', '.join(digests) or 'pending'} (bitwise identical to "
                f"the oracle). {sass + '.' if sass else ''}",
            ]
            headers = ["Stage", "Version", "Optimization added", "Launch",
                       "Application (us)", "Compute (us)",
                       "Application speedup vs Graph", "Compute speedup vs Graph",
                       "Compute speedup over previous stage", "Compute time saved by this step (us)",
                       "Share of Graph→Optimized compute saving", "Kernel launches per invocation",
                       "What changed"]
            widths = [7, 30, 28, 8, 13, 13, 13, 13, 14, 15, 15, 12, 90]
            ws = sheet(f"{data['title']} {gpu['short']}", lines, headers, widths, 6)
            total_saving = graph["compute"] - final["compute"] if bench else None
            previous = None
            for case in cases:
                r = ws.max_row + 1
                f = fill_of(case, last)
                cell(ws, r, 1, case["stage"], fill=f)
                cell(ws, r, 2, case["label"], fill=f).font = bold
                cell(ws, r, 3, case["adds"] if case["stage"] else "—", fill=f)
                cell(ws, r, 4, case["launch"].capitalize(), fill=f)
                if bench:
                    for col, scope in ((5, "application"), (6, "compute")):
                        c = cell(ws, r, col, round(case[scope], 1), "#,##0.0", f)
                        c.hyperlink, c.font = github(case[f"{scope}_record"]), blue
                    cell(ws, r, 7, graph["application"] / case["application"], '0.00"x"', f)
                    cell(ws, r, 8, graph["compute"] / case["compute"], '0.00"x"', f)
                    if previous is not None:
                        saved = previous["compute"] - case["compute"]
                        cell(ws, r, 9, previous["compute"] / case["compute"], '0.00"x"', f)
                        cell(ws, r, 10, round(saved, 1), "#,##0.0", f)
                        cell(ws, r, 11, saved / total_saving, "0.0%", f)
                    else:
                        for col in (9, 10, 11):
                            cell(ws, r, col, "—", fill=f)
                else:
                    for col in range(5, 12):
                        cell(ws, r, col, "pending", fill=f).font = grey
                if "profile" in case:
                    kernels, _ = nsys_rows(case["profile"])
                    c = cell(ws, r, 12, sum(k[1] for k in kernels), "#,##0", f)
                    c.hyperlink, c.font = github(case["profile"]["nsys"]), blue
                else:
                    cell(ws, r, 12, "pending", fill=f).font = grey
                cell(ws, r, 13, case["detail"], fill=f)
                ws.row_dimensions[r].height = 15 * max(2, len(case["detail"]) // 105 + 1)
                if case["stage"] >= 1 or case["launch"] == "graph":
                    previous = case
            r = ws.max_row + 2
            ws.cell(r, 1, "Notes").font = bold
            notes = [
                "Speedups divide the Graph baseline's (or the previous stage's) median time by this "
                "row's. The previous stage of stage 1 is the Graph baseline.",
                "The shares split the compute-scope saving from the Graph baseline to the optimized "
                "binary over the stages in this order. With another order the shares could differ, "
                "because optimizations interact (each stage depends on the one before it).",
                "Kernel launches come from the nsys trace of one end-to-end invocation (Nsys sheet). "
                "Every stage's binary hash and check record are on the Environment sheet's runs.",
            ]
            for i, text in enumerate(notes, 1):
                ws.merge_cells(start_row=r + i, start_column=1, end_row=r + i, end_column=13)
                ws.cell(r + i, 1, text).alignment = wrap
                ws.row_dimensions[r + i].height = 30

            # Nsys: kernels and transfers of one invocation per stage.
            ws = sheet(f"Nsys {data['title']} {gpu['short']}", [
                "Nsight Systems trace of exactly one end-to-end invocation per stage (the harness's "
                "--action profile, captured by tools/profile.py's nsys command). Kernels on the "
                "same stream run back to back, so their sum is the GPU kernel time of the "
                "invocation; transfers are listed below each stage's kernels."],
                ["Stage", "Version", "Kernel or transfer", "Count", "Total (us)", "Average (us)",
                 "nsys report"], [7, 30, 46, 9, 13, 13, 40], 4)
            for case in cases:
                if "profile" not in case:
                    continue
                kernels, copies = nsys_rows(case["profile"])
                f = fill_of(case, last)
                for name, count, total, avg in kernels + [(f"[sum of kernels]", sum(k[1] for k in kernels),
                                                          sum(k[2] for k in kernels), None)] + copies:
                    r = ws.max_row + 1
                    cell(ws, r, 1, case["stage"], fill=f)
                    cell(ws, r, 2, case["label"], fill=f)
                    cell(ws, r, 3, name, fill=f).font = bold if name.startswith("[sum") else Font()
                    cell(ws, r, 4, count, "#,##0", f)
                    cell(ws, r, 5, round(total, 1), "#,##0.0", f)
                    cell(ws, r, 6, round(avg, 2) if avg is not None else "", "#,##0.00", f)
                    put_link(ws, r, 7, case["profile"]["nsys"], Path(case["profile"]["nsys"]).name)

            # NCU: one row per kernel and launch configuration.
            ws = sheet(f"NCU {data['title']} {gpu['short']}", [
                "Nsight Compute --set basic of the same single invocation, one row per kernel and "
                "launch configuration (the baseline's update_level has a different grid per "
                "dependency level). ncu replays each launch alone, so durations exclude launch "
                "gaps and do not add up to the invocation time. Metrics are medians over the "
                "configuration's launches. A spinning persistent kernel keeps its warps active, "
                "so its achieved occupancy does not mean useful work. Stages 1-3 run 2 blocks of 512 "
                "threads per SM (at most 66.7% occupancy); the optimized binary runs 2 of 768.",
                "The basic set has no cache hit rates; the optimized variant's README cites them "
                "from separate full-set profiles."],
                ["Stage", "Version", "Kernel", "Grid x block", "Registers / thread", "Launches",
                 "Median duration (us)", "Total duration (us)"] + [h for h, _ in NCU_METRICS]
                + ["ncu report"],
                [7, 26, 24, 13, 10, 9, 12, 12] + [11] * len(NCU_METRICS) + [40], 5)
            for case in cases:
                if "ncu_raw" not in case.get("profile", {}):
                    continue
                f = fill_of(case, last)
                for kernel, grid, block, registers, count, metrics, durations in \
                        ncu_groups(case["profile"]):
                    r = ws.max_row + 1
                    cell(ws, r, 1, case["stage"], fill=f)
                    cell(ws, r, 2, case["label"], fill=f)
                    cell(ws, r, 3, kernel, fill=f)
                    cell(ws, r, 4, f"{grid:,} x {block:,}", fill=f)
                    cell(ws, r, 5, registers, "0", f)
                    cell(ws, r, 6, count, "#,##0", f)
                    cell(ws, r, 7, round(statistics.median(durations), 2), "#,##0.00", f)
                    cell(ws, r, 8, round(sum(durations), 1), "#,##0.0", f)
                    for i, (_, metric) in enumerate(NCU_METRICS):
                        value = metrics[metric]
                        cell(ws, r, 9 + i, round(value, 1) if value is not None else "", "0.0", f)
                    put_link(ws, r, 9 + len(NCU_METRICS), case["profile"]["ncu"],
                             Path(case["profile"]["ncu"]).name)

    ws = sheet("Environment", ["Every run that produced a selected summary. The metadata records "
                               "the source commit, GPU UUID, driver, tool versions, commands, and "
                               "binary hashes; the check summary lists each stage's binary hash."],
               ["GPU", "Run", "Status", "Host", "GPU UUID", "Driver", "nvcc", "Source commit",
                "Validation", "Measurement"], [16, 34, 12, 10, 44, 10, 40, 12, 50, 70], 3)
    for gpu, stamp in env_runs:
        meta, path = metadata(gpu["folder"], stamp)
        r = ws.max_row + 1
        cell(ws, r, 1, gpu["label"])
        if meta is None:
            cell(ws, r, 2, stamp)
            cell(ws, r, 3, "no recorder metadata").font = grey
            continue
        put_link(ws, r, 2, path, stamp).border = box
        nvcc = (meta["installed_tools"].get("nvcc") or "").strip().splitlines()
        for col, text in enumerate((meta["status"], meta["host"], meta["gpu"]["uuid"],
                                    meta["gpu"]["driver"], nvcc[-1] if nvcc else "",
                                    meta["source_commit"][:7], meta["validation"]["command"],
                                    " ".join(meta["measurement"]["command"])), 3):
            cell(ws, r, col, text)
    return book


def main():
    gpus = [g for g in (load_gpu(folder) for folder in GPUS) if g]
    workbook(gpus).save(OUT)
    contents = "; ".join(g["short"] + ": " + ", ".join(a["title"] for a in g["apps"])
                         for g in gpus)
    print(f"wrote {OUT.relative_to(REPO)} ({contents})")


if __name__ == "__main__":
    main()
