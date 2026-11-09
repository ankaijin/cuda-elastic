#!/usr/bin/env python3
"""Write results/loom_private_ablation/opt_results.xlsx from the selected stage runs.

Each GPU's inputs/<gpu>.json selects, per application, the summaries that
experiments/loom_private_ablation/run.py wrote for one recorded run:

  {"folder": "rtx3090", "label": "RTX 3090 (sm_86, 24 GB)",
   "applications": {"pagerank": {
      "check": "loom_private_ablation/pagerank/<run id>-check.json",
      "bench": "loom_private_ablation/pagerank/<run id>-bench.json",
      "profile": "loom_private_ablation/pagerank/<run id>-profile.json"}}}

Paths are relative to the GPU folder; null means pending. The selected
summaries must come from one source revision. The stages.json they used (read
from git at that revision, and matching the hash they record) must define the
same cases, binaries, and launches as the current one; only descriptions may
differ. The bench binaries must be the ones the check validated.

The workbook has one sheet per application. It holds one table per GPU, in
the order RTX 3090, RTX 4090, RTX 5090; a GPU without a selected run shows
its table with pending values. Timings come only from bench records; the
kernel launch count comes from the run's nsys trace of one invocation.
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
GPUS = {"rtx3090": "RTX 3090 (sm_86, 24 GB)", "rtx4090": "RTX 4090 (sm_89, 24 GB)",
        "rtx5090": "RTX 5090 (sm_120, 32 GB)"}
SCOPES = ("application", "compute")
FIELD = "batch_per_iteration_us"
PENDING = "pending"


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


def structure(stages):
    """What a run depends on: the binaries, the cases' ids, stages, binaries,
    and launches, and the machine-code comparisons. Labels and descriptions
    are free to change."""
    return {"binaries": stages["binaries"],
            "cases": [{k: c[k] for k in ("id", "stage", "binary", "launch")}
                      for c in stages["cases"]],
            "sass_equal": [{k: s[k] for k in ("binary", "reference", "kernel")}
                           for s in stages.get("sass_equal", [])]}


def stages_at(commit, app, expected_sha256):
    """stages.json as the selected run used it."""
    relative = f"experiments/loom_private_ablation/{app}/stages.json"
    text = subprocess.run(["git", "-C", str(REPO), "show", f"{commit}:{relative}"],
                          capture_output=True, check=True).stdout
    if hashlib.sha256(text).hexdigest() != expected_sha256:
        raise ValueError(f"{relative} at {commit[:7]} does not match the run's hash")
    return json.loads(text)


def load_app(folder, app, selection):
    stages = json.loads((EXPERIMENTS / app / "stages.json").read_text())
    loaded = {key: read_json(p) for key in ("check", "bench", "profile")
              if (p := resolve(folder, (selection or {}).get(key)))}
    revisions = {(s["source_commit"], s["stages_sha256"]) for s in loaded.values()}
    if len(revisions) > 1:
        raise ValueError(f"{folder} {app}: the selected summaries come from different "
                         "revisions or stages.json files")
    for commit, digest in revisions:
        if structure(stages_at(commit, app, digest)) != structure(stages):
            raise ValueError(f"{folder} {app}: stages.json now defines different cases "
                             "than the selected run")
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
            row["dimensions"] = records[0]["dimensions"]
            for scope in SCOPES:
                ranked = sorted(records, key=lambda r: r["timing"][scope][FIELD])
                row[scope] = statistics.median(r["timing"][scope][FIELD] for r in records)
                row[f"{scope}_record"] = ranked[(len(ranked) - 1) // 2]["_path"]
        if check:
            row["check"] = check["binaries"][case["binary"]]
        if profile:
            entry = next(p for p in profile["profiles"] if p["case"] == case["id"])
            with open(REPO / entry["nsys_kernels"]) as stream:
                row["launches"] = sum(int(r["Instances"]) for r in csv.DictReader(stream))
            row["nsys"] = entry["nsys"]
        cases.append(row)
    return {"stages": stages, "cases": cases, "check": check, "bench": bench}


def load_inputs():
    """{folder: manifest} for every GPU with an input manifest."""
    found = {}
    for folder in GPUS:
        path = HERE / "inputs" / f"{folder}.json"
        if path.is_file():
            found[folder] = json.loads(path.read_text())
    return found


def metadata(folder, stamp):
    path = REPO / folder / "runs" / stamp / "metadata.json"
    return (json.loads(path.read_text()), str(path.relative_to(REPO))) if path.is_file() \
        else (None, None)


def link(path):
    """GitHub URL for path, pinned to the latest origin/main commit that has it."""
    log = subprocess.run(["git", "-C", str(REPO), "log", "--format=%H", "-1", "origin/main",
                          "--", path], capture_output=True, text=True).stdout.strip()
    return f"{GITHUB}/{log or 'main'}/{path}"


def workbook(inputs):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter

    blue = Font(color="0563C1", underline="single")
    bold = Font(bold=True)
    grey = Font(italic=True, color="808080")
    header_fill = PatternFill("solid", fgColor="1F3864")
    header_font = Font(bold=True, color="FFFFFF")
    # The loom_private workbook's Direct, Graph, and Optimized colours; the
    # stages in between get a neutral fill.
    fills = {"direct": PatternFill("solid", fgColor="FCE4D6"),
             "graph": PatternFill("solid", fgColor="E2EFDA"),
             "stage": PatternFill("solid", fgColor="F2F2F2"),
             "optimized": PatternFill("solid", fgColor="DDEBF7")}
    thin = Side(style="thin", color="BFBFBF")
    box = Border(left=thin, right=thin, top=thin, bottom=thin)
    wrap = Alignment(wrap_text=True, vertical="top")
    headers = ["Stage", "Version", "What changed", "Launch", "Application (us)",
               "Compute (us)", "Application speedup vs Graph", "Compute speedup vs Graph",
               "Compute speedup over previous stage", "Compute time saved by this step (us)",
               "Share of Graph→Optimized compute saving", "Kernel launches per invocation"]
    widths = [7, 30, 62, 8, 13, 13, 13, 13, 14, 15, 15, 12]
    links = {}
    book = Workbook()
    book.remove(book.active)

    def github(path):
        if path not in links:
            links[path] = link(path)
        return links[path]

    def text_row(ws, row, text, height=30, font=None):
        ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=len(headers))
        c = ws.cell(row, 1, text)
        c.alignment = wrap
        if font:
            c.font = font
        ws.row_dimensions[row].height = height
        return c

    def cell(ws, row, col, value, fmt=None, fill=None, source=None):
        c = ws.cell(row, col, value)
        c.border, c.alignment = box, wrap
        if fill:
            c.fill = fill
        if value == PENDING:
            c.font = grey
        elif fmt and isinstance(value, (int, float)):
            c.number_format = fmt
            if source:
                c.hyperlink, c.font = github(source), blue
        return c

    def gpu_table(ws, row, folder, label, data):
        """One GPU's title, run line, and stage table from `row`; returns the
        next free row."""
        cases, bench, check = data["cases"], data["bench"], data["check"]
        last = max(c["stage"] for c in cases)
        graph = next(c for c in cases if c["stage"] == 0 and c["launch"] == "graph")
        final = next(c for c in cases if c["stage"] == last)
        ws.cell(row, 1, label).font = Font(bold=True, size=13)
        row += 1
        if bench:
            meta, meta_path = metadata(folder, bench["stamp"])
            nvcc = ((meta or {}).get("installed_tools", {}).get("nvcc") or "").strip()
            digests = sorted({c["check"]["output_digest"] for c in cases if "check" in c})
            sass = "; ".join(f"{s['binary']} matches {s['reference']} instruction for "
                             f"instruction ({s['instructions'][0]})"
                             for s in (check or {}).get("sass_equal", []) if s["identical"])
            c = text_row(ws, row,
                         f"Run {bench['stamp']} at source {bench['source_commit'][:7]}, driver "
                         f"{bench['device']['driver']}, "
                         f"{nvcc.splitlines()[-1] if nvcc else 'unknown toolkit'}. "
                         f"Median of {bench['repeat']} repetitions. Every binary passed the "
                         f"large check with digest {', '.join(digests) or PENDING}"
                         + (f"; {sass}." if sass else "."), 30)
            if meta_path:
                c.hyperlink, c.font = github(meta_path), blue
        else:
            text_row(ws, row, "Pending: no run selected in "
                     f"results/loom_private_ablation/inputs/{folder}.json.", 16, grey)
        row += 1
        for i, header in enumerate(headers, 1):
            c = ws.cell(row, i, header)
            c.font, c.fill, c.border = header_font, header_fill, box
            c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        ws.row_dimensions[row].height = 44
        total = graph["compute"] - final["compute"] if bench else None
        previous = None
        for case in cases:
            row += 1
            f = fills[case["launch"]] if case["stage"] == 0 else \
                fills["optimized"] if case["stage"] == last else fills["stage"]
            cell(ws, row, 1, case["stage"], fill=f)
            cell(ws, row, 2, case["label"], fill=f).font = bold
            cell(ws, row, 3, case["change"], fill=f)
            cell(ws, row, 4, case["launch"].capitalize(), fill=f)
            if bench:
                cell(ws, row, 5, round(case["application"], 1), "#,##0.0", f,
                     case["application_record"])
                cell(ws, row, 6, round(case["compute"], 1), "#,##0.0", f, case["compute_record"])
                cell(ws, row, 7, graph["application"] / case["application"], '0.00"x"', f)
                cell(ws, row, 8, graph["compute"] / case["compute"], '0.00"x"', f)
                if previous is None:
                    for col in (9, 10, 11):
                        cell(ws, row, col, "—", fill=f)
                else:
                    saved = previous["compute"] - case["compute"]
                    cell(ws, row, 9, previous["compute"] / case["compute"], '0.00"x"', f)
                    cell(ws, row, 10, round(saved, 1), "#,##0.0", f)
                    cell(ws, row, 11, saved / total, "0.0%", f)
            else:
                for col in range(5, 12):
                    cell(ws, row, col, PENDING, fill=f)
            cell(ws, row, 12, case.get("launches", PENDING), "#,##0", f, case.get("nsys"))
            ws.row_dimensions[row].height = 30
            if case["stage"] >= 1 or case["launch"] == "graph":
                previous = case
        return row + 2

    apps = sorted(p.parent.name for p in EXPERIMENTS.glob("*/stages.json"))
    for app in apps:
        stages = json.loads((EXPERIMENTS / app / "stages.json").read_text())
        tables = {}
        for folder in GPUS:
            selection = inputs.get(folder, {}).get("applications", {}).get(app)
            tables[folder] = load_app(folder, app, selection)
        ws = book.create_sheet(stages["title"])
        for i, width in enumerate(widths, 1):
            ws.column_dimensions[get_column_letter(i)].width = width
        ws.sheet_view.showGridLines = False
        ws.cell(1, 1, f"{stages['title']}: optimizations added one at a time").font = \
            Font(bold=True, size=14)
        dims = next((c["dimensions"] for t in tables.values() for c in t["cases"]
                     if "dimensions" in c), {})
        text_row(ws, 2,
                 f"loom-private/{app}"
                 + (" (" + ", ".join(f"{k} {v:,}" for k, v in dims.items()
                                     if isinstance(v, int) and k != "epsilon") + ")"
                    if dims else "")
                 + ". Each row adds one optimization to the row above it. Stage 0 is the "
                 "suite's baseline; the last stage is the suite's optimized binary. Times are "
                 "the median over repetitions of the harness's batch time per invocation (20 "
                 "invocations after 3 warmups), as in the loom_private workbook. Application "
                 "scope includes the transfers; compute scope is the kernels with resident "
                 "inputs. Stages after the baseline run as CUDA Graphs.", 44)
        text_row(ws, 3,
                 "Speedups divide the Graph baseline's (or the previous stage's) compute time by "
                 "this row's; stage 1's previous stage is the Graph baseline. The shares split "
                 "the Graph→Optimized compute saving in this order; the optimizations interact, "
                 "so another order could split it differently. Kernel launches come from an "
                 "nsys trace of one invocation. Design notes: "
                 f"experiments/loom_private_ablation/{app}/README.md.", 44)
        row = 5
        for folder, label in GPUS.items():
            row = gpu_table(ws, row, folder, inputs.get(folder, {}).get("label", label),
                            tables[folder])
    return book


def main():
    workbook(load_inputs()).save(OUT)
    print(f"wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
