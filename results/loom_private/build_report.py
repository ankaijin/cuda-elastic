#!/usr/bin/env python3
"""Build the loom_private v1 results page and workbook from selected records.

  python3 results/loom_private/build_report.py

Writes results/loom_private/README.md and results/loom_private/loom_private_v1.xlsx.
Inputs are pinned in results/loom_private/inputs/<gpu>.json, independently of
the legacy results/inputs manifests. Each application has three selections:
`check` (a `--action check` record), `bench_direct` and `bench_graph` (`--action
bench` records). Paths are relative to that GPU's folder. Null means pending.
Missing files, unsafe paths, records of another workload, records of another
action or launch (a check must cover both launch modes), records that
validated an older contract, records whose integer dimensions differ from the
contract's `large` values, and records from a GPU other than the folder's are
errors.

Timing cells come only from bench records, so a correctness check never
appears as a performance measurement. The workbook uses the layout and styling
of results/cuda_performance.xlsx: per-GPU summary and Latency sheets, then
Sources, Correctness, Environment, and Contracts. Its links point at GitHub,
pinned to the latest origin/main commit that has each file. The test-machine table comes from the
recorder metadata (<gpu>/runs/<run-id>/metadata.json) of the selected records,
except the source revision, which the manifest states (`source_revision`, a
full commit hash): bench runs start from commits that only add earlier runs'
results, so their recorded commits need not stay in the history.
Nothing is rerun. Requires openpyxl, and git for the workbook's links.
"""

import hashlib
import json
import re
from pathlib import Path
import subprocess

REPO = Path(__file__).resolve().parents[2]
HERE = REPO / "results" / "loom_private"
SUITE = REPO / "applications" / "loom_private"
VERSION = "v1"
MD_OUT = HERE / "README.md"
XLSX_OUT = HERE / f"loom_private_{VERSION}.xlsx"
GPUS = ["rtx3090", "rtx4090", "rtx5090"]
APPS = ["pagerank", "attention", "decoder", "anomaly", "vecadd_memory", "vecadd_product"]
SELECTIONS = {"check": ("check", "both"), "bench_direct": ("bench", "direct"),
              "bench_graph": ("bench", "graph")}
PENDING = "pending"
# Display name and a format string over the contract's `large` values.
PROBLEMS = {
    "pagerank": ("PageRank", "{nodes:,} vertices, {edges:,} edges, {iterations} ordered "
                 "Gauss–Seidel iterations"),
    "attention": ("Multisensor attention", "{query_tokens:,} queries × {context_tokens:,} "
                  "context tokens"),
    "decoder": ("llama2c decoder", "{batch:,} sequences × {steps} steps, width {width:,}, "
                "vocabulary {vocab:,}"),
    "anomaly": ("Anomaly detection", "{samples:,} samples × {sample_elements} values through "
                "the {layers}-layer int8 model"),
    "vecadd_memory": ("Vecadd memory", "{length:,} integer additions"),
    "vecadd_product": ("Vecadd product", "{records:,} pairs of {operand_bytes}-byte operands, "
                       "added byte by byte"),
}
# Integer `large` entries that are not problem extents (left out of the size column).
NON_EXTENT_KEYS = {"expected_checksum"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def contract(app):
    path = SUITE / app / "contract.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    data["_sha256"] = sha256(path)
    return data


def resolve(folder, value):
    """Exact path inside the GPU folder, or None when pending."""
    if value is None:
        return None
    if not isinstance(value, str) or not value or any(c in value for c in "*?["):
        raise ValueError(f"{folder}: selection must be an exact relative path: {value!r}")
    relative = Path(value)
    root = (REPO / folder).resolve()
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{folder}: selection escapes the GPU folder: {value}")
    path = (root / relative).resolve()
    if root not in path.parents:
        raise ValueError(f"{folder}: selection escapes the GPU folder: {value}")
    if not path.is_file():
        raise ValueError(f"{folder}: selected file is missing: {value}")
    return path


def is_integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def device_matches(folder, name):
    """True when the record's device name is the folder's GPU (rtx3090 -> RTX 3090)."""
    model = re.fullmatch(r"rtx(\d+)", folder)
    return (model is not None and isinstance(name, str)
            and re.fullmatch(rf"(NVIDIA )?(GeForce )?RTX {model.group(1)}", name) is not None)


def load_record(folder, app, key, value, spec):
    path = resolve(folder, value)
    if path is None:
        return None
    action, launch = SELECTIONS[key]
    record = json.loads(path.read_text())
    where = f"{folder}/{value}"
    if record.get("schema") != "loom_private.result/1":
        raise ValueError(f"{where}: not a loom_private result record")
    if spec is None or record.get("identity") != spec.get("identity"):
        raise ValueError(f"{where}: record identity {record.get('identity')!r} "
                         f"is not application {app}")
    if record.get("action") != action or record.get("launch") != launch:
        raise ValueError(f"{where}: selected as {key} but records "
                         f"{record.get('action')}/{record.get('launch')}")
    if record.get("contract_sha256") != spec["_sha256"]:
        raise ValueError(f"{where}: record validated contract {record.get('contract_sha256')}, "
                         f"current contract is {spec['_sha256']}")
    if record.get("preset") != "large" or record.get("overrides"):
        raise ValueError(f"{where}: only unmodified large-preset records can be selected")
    dimensions = record.get("dimensions")
    if not isinstance(dimensions, dict):
        raise ValueError(f"{where}: record has no dimensions")
    for name, expected in spec.get("large", {}).items():
        actual = dimensions.get(name)
        if is_integer(expected) and is_integer(actual) and actual != expected:
            raise ValueError(f"{where}: dimension {name}={actual} differs from the "
                             f"contract's large value {expected}")
    device = (record.get("device") or {}).get("name")
    if not device_matches(folder, device):
        raise ValueError(f"{where}: recorded on {device!r}, not the {folder} folder's GPU")
    record["_path"] = f"{folder}/{value}"
    return record


def load_gpu(folder):
    manifest = json.loads((HERE / "inputs" / f"{folder}.json").read_text())
    if manifest.get("folder") != folder or manifest.get("suite_version") != VERSION:
        raise ValueError(f"inputs/{folder}.json: folder or suite_version mismatch")
    if sorted(manifest["applications"]) != sorted(APPS):
        raise ValueError(f"inputs/{folder}.json: must list exactly {APPS}")
    rows = []
    for app in APPS:
        spec = contract(app)
        selected = manifest["applications"][app]
        if sorted(selected) != sorted(SELECTIONS):
            raise ValueError(f"inputs/{folder}.json: {app} needs keys {sorted(SELECTIONS)}")
        records = {key: load_record(folder, app, key, selected[key], spec)
                   for key in SELECTIONS}
        rows.append({"app": app, "contract": spec, **records})
    return manifest, rows


def timing(record, scope):
    """Batch per-iteration microseconds from a bench record, else pending."""
    if record is None:
        return PENDING
    return record["timing"][scope]["batch_per_iteration_us"]


def size_text(spec):
    if spec is None:
        return "contract pending"
    # Integer extents only; descriptive strings, nested notes, and non-extent
    # integers such as expected checksums stay in the contract.
    return ", ".join(f"{k}={v:,}" for k, v in spec.get("large", {}).items()
                     if is_integer(v) and k not in NON_EXTENT_KEYS and "checksum" not in k)


def check_cells(record):
    if record is None:
        return [PENDING] * 4
    def modes(launch):
        runs = [c for c in record["checks"] if c["launch"] == launch]
        return f"{sum(c['pass'] for c in runs)}/{len(runs)} pass" if runs else "not run"
    return [record["status"], modes("direct"), modes("graph"),
            "yes" if record["outputs_agree"] else "no"]


def fmt(value):
    if isinstance(value, str):
        return value
    return f"{value:,.0f}" if value >= 100 else f"{value:,.1f}"


def problem_text(app, spec):
    """One plain-language line describing the large problem, from the contract."""
    if spec is None:
        return "contract pending"
    return PROBLEMS[app][1].format(**spec["large"])


def check_summary(record):
    """pass only when every invocation passed and Direct and Graph agree."""
    if record is None:
        return PENDING
    passed = (record["status"] == "pass" and record["outputs_agree"]
              and all(c["pass"] for c in record["checks"]))
    return f"[{'pass' if passed else 'FAIL'}](../../{record['_path']})"


def run_metadata(record):
    """The recorder's metadata for the run that produced a selected record, if any."""
    folder, _, name = record["_path"].partition("/")
    run_id = re.sub(r"-(check|bench-direct|bench-graph)\.json$", "", Path(name).name)
    path = REPO / folder / "runs" / run_id / "metadata.json"
    return json.loads(path.read_text()) if path.is_file() else None


def environment(manifest, rows):
    """GPU, host, driver, CUDA release, and source revision of the selected runs."""
    runs = {}
    for row in rows:
        for key in SELECTIONS:
            if row[key] is not None:
                meta = run_metadata(row[key])
                runs[meta["run_id"] if meta else row[key]["_path"]] = meta
    if not runs:
        return [manifest["label"], PENDING, PENDING, PENDING, PENDING]
    def values(get):
        found = sorted({get(m) if m else "unrecorded" for m in runs.values()})
        return ", ".join(found)
    def cuda(meta):
        match = re.search(r"release ([\d.]+)", meta.get("installed_tools", {}).get("nvcc") or "")
        return f"CUDA {match.group(1)}" if match else "unknown"
    revision = manifest.get("source_revision")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError(f"inputs/{manifest['folder']}.json: selected runs need a full "
                         f"source_revision commit hash")
    return [manifest["label"], values(lambda m: m["host"]), values(lambda m: m["gpu"]["driver"]),
            values(cuda), revision[:7]]


def markdown(gpus):
    labels = [m["label"].split(" (")[0] for m, _ in gpus]
    lines = [
        f"# Private Loom CUDA workloads: results {VERSION}",
        "",
        "Baseline timings and correctness checks for the six workloads in "
        "[`applications/loom_private`](../../applications/loom_private), ported from private "
        "Loom commit `4a75f8d`. These are the first, unoptimized implementations. The older "
        "workloads have their own [results page](../README.md).",
        "",
        "- Times are microseconds per iteration, averaged over a back-to-back batch. "
        "Lower is better.",
        "- **Direct** launches every kernel on each iteration. **Graph** replays a CUDA Graph "
        "captured once.",
        "- *pending* means no run has been selected yet.",
        "",
        "## Workloads",
        "",
        "| Workload | Large problem |",
        "|---|---|",
    ]
    lines += [f"| {PROBLEMS[row['app']][0]} | {problem_text(row['app'], row['contract'])} |"
              for row in gpus[0][1]]
    scopes = [
        ("End-to-end time (µs)",
         "Includes copying inputs to the GPU and results back. The decoder keeps its weights on "
         "the GPU, and the anomaly model is loaded once, before timing. Each workload's "
         "`contract.json` defines the scope exactly.", "application"),
        ("Kernel-only time (µs)", "The same work without the copies.", "compute"),
    ]
    for title, note, scope in scopes:
        lines += ["", f"## {title}", "", note, "",
                  "| Workload | " + " | ".join(f"{g} Direct | {g} Graph" for g in labels) + " |",
                  "|---|" + "--:|--:|" * len(labels)]
        for index, app in enumerate(APPS):
            cells = []
            for _, rows in gpus:
                cells += [fmt(timing(rows[index]["bench_direct"], scope)),
                          fmt(timing(rows[index]["bench_graph"], scope))]
            lines.append(f"| {PROBLEMS[app][0]} | " + " | ".join(cells) + " |")
    lines += ["", "## Correctness", "",
              "Each check compares every output with an independent CPU reference. The GPU runs "
              "twice with direct launches and twice with a graph, each time starting from "
              "poisoned buffers, and all four outputs must be identical. Checks are not "
              "timed. Each cell links to its record.", "",
              "| Workload | " + " | ".join(labels) + " |",
              "|---|" + "---|" * len(labels)]
    for index, app in enumerate(APPS):
        lines.append(f"| {PROBLEMS[app][0]} | "
                     + " | ".join(check_summary(rows[index]["check"]) for _, rows in gpus) + " |")
    lines += ["", "## Test machines", "",
              "Taken from the recorder metadata (`<gpu>/runs/<run>/metadata.json`) of the "
              "selected runs. Source revision is the code the runs were built from, as stated in "
              "`inputs/<gpu>.json`.", "",
              "| GPU | Host | Driver | CUDA | Source revision |", "|---|---|---|---|---|"]
    lines += ["| " + " | ".join(environment(m, rows)) + " |" for m, rows in gpus]
    lines += ["", "Generated by `results/loom_private/build_report.py` from the runs selected in "
              "`results/loom_private/inputs/<gpu>.json`. The workbook "
              f"[`loom_private_{VERSION}.xlsx`](loom_private_{VERSION}.xlsx) has the same data "
              "plus contract hashes and per-invocation check results."]
    return "\n".join(lines)


# --------------------------------------------------------------- workbook ---
# The workbook uses the layout and styling of results/cuda_performance.xlsx
# (results/build_spreadsheet.py). The helpers are copied rather than imported so
# that this report never loads or validates the legacy inputs.

GITHUB = "https://github.com/ankaijin/cuda-elastic/blob"
BASELINE = "rtx3090"
VERSIONS = ["Direct", "Graph", "Optimized"]
STAGES = [("application", "Application (end to end)"), ("compute", "Kernel only (no transfers)")]
METRIC = "CUDA events (batched)"
BENCH_PENDING = "pending (no bench run selected)"
OPTIMIZED_PENDING = "pending (no optimized version yet)"
PROFILE_PENDING = "pending (not profiled yet)"
# Short visible note per application and stage; the contract's full timing
# scope goes in the cell comment.
SHORT_NOTES = {
    "application": {
        "decoder": "Weights stay on the GPU; copies prompts in and tokens out.",
        "anomaly": "Model loaded before timing; copies samples in and outputs out.",
    },
    "compute": {},
}
DEFAULT_NOTES = {"application": "Includes input upload and output download (pinned host memory).",
                 "compute": "Kernels only; inputs already on the GPU."}


def link(path):
    """GitHub URL for path, pinned to the latest commit on origin/main that has
    the file; files not merged yet link to main."""
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
    fills = [PatternFill("solid", fgColor=c) for c in ("FCE4D6", "E2EFDA", "DDEBF7")]
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

    def table_sheet(title, headers, widths, description):
        ws = book.create_sheet(title)
        ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=len(headers))
        ws.cell(1, 1, title).font = Font(bold=True, size=14)
        ws.merge_cells(start_row=2, start_column=1, end_row=2, end_column=len(headers))
        ws.cell(2, 1, description).alignment = wrap
        ws.row_dimensions[2].height = 42
        for i, (header, width) in enumerate(zip(headers, widths), 1):
            c = ws.cell(4, i, header)
            c.font, c.fill, c.border = header_font, header_fill, box
            c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            ws.column_dimensions[get_column_letter(i)].width = width
        ws.row_dimensions[4].height = 42
        ws.freeze_panes = "C5"
        ws.sheet_view.showGridLines = False
        ws.sheet_properties.pageSetUpPr.fitToPage = True
        ws.page_setup.orientation = "landscape"
        ws.page_setup.paperSize = ws.PAPERSIZE_A3
        ws.page_setup.fitToWidth, ws.page_setup.fitToHeight = 1, 0
        ws.print_title_rows = "1:4"
        return ws

    def finish_table(ws):
        ws.auto_filter.ref = f"A4:{get_column_letter(ws.max_column)}{ws.max_row}"
        for row in ws.iter_rows(min_row=5):
            for c in row:
                c.alignment = wrap
                c.border = box
            ws.row_dimensions[row[0].row].height = 58

    def value(ws, row, col, data, source=None, speedup=False):
        """A number (linked to its source) or grey pending text with its reason."""
        c = ws.cell(row, col)
        if isinstance(data, (int, float)):
            c.value = round(data, 3)
            c.number_format = '0.00"x"' if speedup else '#,##0.00'
            if source:
                c.hyperlink, c.font = github(source), blue
        elif data is not None:
            c.value = "pending" if data.startswith("pending") else data
            c.font = grey
            if data.startswith("pending"):
                c.comment = Comment(data, "build_report.py")
        return c

    def put_link(ws, row, col, path, text):
        c = ws.cell(row, col, text)
        c.hyperlink, c.font = github(path), blue
        return c

    def note(ws, row, col, short, full):
        c = ws.cell(row, col, short)
        c.comment = Comment(full, "build_report.py", width=480, height=280)
        return c

    def ratio(a, b):
        numbers = isinstance(a, (int, float)) and isinstance(b, (int, float))
        return a / b if numbers and b else None

    def bench(record, scope, field="batch_per_iteration_us"):
        return BENCH_PENDING if record is None else record["timing"][scope][field]

    def stage_note(row, scope):
        spec, records = row["contract"] or {}, [row["bench_direct"], row["bench_graph"]]
        short = SHORT_NOTES[scope].get(row["app"], DEFAULT_NOTES[scope])
        counts = {r["iterations"] for r in records if r}
        if len(counts) == 1:
            short += f" {counts.pop():,} iterations/batch. One run."
        full = spec.get("timing_scope", {}).get(scope, "Contract pending.")
        return short, full[0].upper() + full[1:] + "."

    names = {m["folder"]: m["label"].split(" (")[0] for m, _ in gpus}
    base_rows = dict((m["folder"], rows) for m, rows in gpus)[BASELINE]
    shared = ["Sources", "Correctness", "Environment", "Contracts"]

    for manifest, rows in gpus:
        folder, title = manifest["folder"], names[manifest["folder"]]
        compare = folder != BASELINE
        headers = ["Workload", "Stage", "Problem size", "Metric", "Direct (us)", "Graph (us)",
                   "Optimized (us)", "Graph speedup\nDirect / Graph",
                   "Optimized speedup\nGraph / Optimized"]
        if compare:
            headers += [f"GPU speedup\n{names[BASELINE].removeprefix('RTX ')} / "
                        f"{title.removeprefix('RTX ')}\n(Graph)"]
        headers += ["Notes"]
        ws = table_sheet(title, headers,
                         [24, 30, 30, 20, 15, 15, 15, 15, 17] + ([18] if compare else []) + [48],
                         f"{manifest['label']}; {manifest['toolkit']}. Large workloads: time per "
                         "iteration in a back-to-back batch; lower is better. Times link to the "
                         "bench records. Notes have full explanations in comments.")
        for col, target in enumerate([f"Latency {title}"] + shared, 1):
            c = ws.cell(3, col, target)
            c.hyperlink, c.font = f"#'{target}'!A1", blue
        for index, row in enumerate(rows):
            for scope, stage in STAGES:
                r = ws.max_row + 1
                for col, text in enumerate((PROBLEMS[row["app"]][0], stage,
                                            problem_text(row["app"], row["contract"]), METRIC), 1):
                    ws.cell(r, col, text)
                ws.cell(r, 1).font = bold
                direct, graph = bench(row["bench_direct"], scope), bench(row["bench_graph"], scope)
                sources = [(row["bench_direct"] or {}).get("_path"),
                           (row["bench_graph"] or {}).get("_path"), None]
                for col, (data, source, fill) in enumerate(
                        zip((direct, graph, OPTIMIZED_PENDING), sources, fills), 5):
                    value(ws, r, col, data, source).fill = fill
                value(ws, r, 8, ratio(direct, graph) or BENCH_PENDING, speedup=True)
                value(ws, r, 9, OPTIMIZED_PENDING, speedup=True)
                if compare:
                    base = bench(base_rows[index]["bench_graph"], scope)
                    value(ws, r, 10, ratio(base, graph) or BENCH_PENDING, speedup=True)
                note(ws, r, len(headers), *stage_note(row, scope))
        finish_table(ws)

    for manifest, rows in gpus:
        title = names[manifest["folder"]]
        headers = ["Workload", "Stage", "Problem size", "Version", "Batch iterations", "Runs",
                   "Full batch (ms)", "Batched (us/iteration)", "Single (us)", "Single / batched",
                   "Graph speedup (single)", "Optimized speedup (single)", "Notes"]
        ws = table_sheet(f"Latency {title}", headers,
                         [24, 24, 30, 16, 14, 9, 16, 17, 15, 15, 17, 18, 48],
                         f"{manifest['label']}. Single = median of the isolated iterations that "
                         "follow the batch (as many as the batch has). Single/batched reflects "
                         "scheduling and timing conditions, not pure launch overhead. Each "
                         "version comes from one bench run; missing measurements stay pending.")
        for row in rows:
            for scope, stage in STAGES:
                singles = {}
                for version, key, fill in zip(VERSIONS, ("bench_direct", "bench_graph", None),
                                              fills):
                    record = row.get(key) if key else None
                    pending = BENCH_PENDING if key else OPTIMIZED_PENDING
                    r = ws.max_row + 1
                    for col, text in enumerate((PROBLEMS[row["app"]][0], stage,
                                                problem_text(row["app"], row["contract"]),
                                                version), 1):
                        ws.cell(r, col, text)
                    ws.cell(r, 1).font = bold
                    if record is None:
                        for col in range(5, 11):
                            value(ws, r, col, pending, speedup=col == 10)
                    else:
                        timing_ = record["timing"][scope]
                        singles[version] = timing_["single_median_us"]
                        ws.cell(r, 5, record["iterations"])
                        ws.cell(r, 6, 1)
                        for col, data in enumerate((timing_["batch_total_ms"],
                                                    timing_["batch_per_iteration_us"],
                                                    timing_["single_median_us"]), 7):
                            value(ws, r, col, data, record["_path"])
                        value(ws, r, 10, ratio(timing_["single_median_us"],
                                               timing_["batch_per_iteration_us"]), speedup=True)
                    if version == "Graph":
                        value(ws, r, 11, ratio(singles.get("Direct"), singles.get("Graph"))
                              or pending, speedup=True)
                    if version == "Optimized":
                        value(ws, r, 12, OPTIMIZED_PENDING, speedup=True)
                    note(ws, r, 13, *stage_note(row, scope))
                    for col in range(1, len(headers) + 1):
                        ws.cell(r, col).fill = fill
        finish_table(ws)

    ws = table_sheet("Sources", ["GPU", "Workload", "Problem size", "Version", "Measurement",
                                 "Run metadata", "NCU report", "Nsight Systems", "Notes"],
                     [15, 24, 30, 15, 44, 44, 30, 30, 48],
                     "Source files for every timing. Filenames link to the original evidence. "
                     "Profiles are pending until the workloads are profiled.")
    for manifest, rows in gpus:
        for row in rows:
            for version, key in zip(VERSIONS, ("bench_direct", "bench_graph", None)):
                record = row.get(key) if key else None
                r = ws.max_row + 1
                for col, text in enumerate((names[manifest["folder"]], PROBLEMS[row["app"]][0],
                                            problem_text(row["app"], row["contract"]), version), 1):
                    ws.cell(r, col, text)
                if record is None:
                    value(ws, r, 5, BENCH_PENDING if key else OPTIMIZED_PENDING)
                else:
                    put_link(ws, r, 5, record["_path"], Path(record["_path"]).name)
                    meta = run_metadata(record)
                    if meta:
                        path = f"{manifest['folder']}/runs/{meta['run_id']}/metadata.json"
                        put_link(ws, r, 6, path, f"{meta['run_id']}/metadata.json")
                value(ws, r, 7, PROFILE_PENDING)
                value(ws, r, 8, PROFILE_PENDING)
                note(ws, r, 9, *stage_note(row, "application"))
    finish_table(ws)

    ws = table_sheet("Correctness", ["GPU", "Workload", "Contract SHA-256", "Status", "Direct",
                                     "Graph", "Direct = Graph", "Record"],
                     [15, 24, 30, 12, 12, 12, 14, 44],
                     "Large-workload checks, not timings. Each runs the CPU reference once, then "
                     "two Direct and two Graph invocations from poisoned buffers; every output "
                     "must match the reference and all four must be identical.")
    for manifest, rows in gpus:
        for row in rows:
            r = ws.max_row + 1
            spec, record = row["contract"] or {}, row["check"]
            for col, text in enumerate((names[manifest["folder"]], PROBLEMS[row["app"]][0],
                                        spec.get("_sha256", PENDING)), 1):
                ws.cell(r, col, text)
            for col, text in enumerate(check_cells(record), 4):
                if text == PENDING:
                    value(ws, r, col, "pending (no check run selected)")
                else:
                    ws.cell(r, col, text)
            if record:
                put_link(ws, r, 8, record["_path"], Path(record["_path"]).name)
    finish_table(ws)

    ws = table_sheet("Environment", ["GPU", "Run", "Measurement", "Host", "Driver", "CUDA",
                                     "Source revision"],
                     [15, 44, 20, 12, 14, 12, 16],
                     "Recorder metadata of every selected run. Source revision is the code the "
                     "runs were built from, as stated in results/loom_private/inputs/<gpu>.json.")
    for manifest, rows in gpus:
        runs = {}
        for row in rows:
            for key in SELECTIONS:
                if row[key] is not None and (meta := run_metadata(row[key])):
                    runs[meta["run_id"]] = (key.replace("_", " "), meta)
        for run_id, (kind, meta) in sorted(runs.items()):
            r = ws.max_row + 1
            ws.cell(r, 1, names[manifest["folder"]])
            put_link(ws, r, 2, f"{manifest['folder']}/runs/{run_id}/metadata.json", run_id)
            nvcc = meta.get("installed_tools", {}).get("nvcc") or ""
            release = re.search(r"release ([\d.]+)", nvcc)
            for col, text in enumerate((kind, meta["host"], meta["gpu"]["driver"],
                                        f"CUDA {release.group(1)}" if release else "unknown",
                                        manifest["source_revision"][:7]), 3):
                ws.cell(r, col, text)
    finish_table(ws)

    ws = table_sheet("Contracts", ["Workload", "Contract", "Contract SHA-256", "Upstream commit",
                                   "Extension of upstream"],
                     [24, 44, 30, 30, 60],
                     "Each workload's contract.json fixes its dimensions, inputs, outputs, "
                     "tolerances, and timing scopes. Every result record names the contract hash "
                     "it validated.")
    for app in APPS:
        spec = contract(app) or {}
        r = ws.max_row + 1
        ws.cell(r, 1, PROBLEMS[app][0]).font = bold
        put_link(ws, r, 2, f"applications/loom_private/{app}/contract.json",
                 f"{app}/contract.json")
        ws.cell(r, 3, spec.get("_sha256", PENDING))
        ws.cell(r, 4, spec.get("upstream", {}).get("commit", PENDING))
        extension = spec.get("extension_of_upstream", PENDING)
        ws.cell(r, 5, "No (direct port)" if extension is False else str(extension))
    finish_table(ws)
    return book


def main():
    gpus = [load_gpu(folder) for folder in GPUS]
    MD_OUT.write_text(markdown(gpus) + "\n")
    workbook(gpus).save(XLSX_OUT)
    print(f"wrote {MD_OUT.relative_to(REPO)} and {XLSX_OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
