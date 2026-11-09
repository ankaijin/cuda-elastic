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
same cases, binaries, launches, and checks as the current one; only
descriptions and the reporting scope may differ. Every binary the bench timed
and the profile captured must be one the check validated (by hash). A profile
summary written before run.py hashed profiled binaries (it has no
"check_summary") is accepted only when the recorder's binary hashes after
validation (<gpu>/runs/<stamp>/metadata.json) match the check for every
binary, and its run line says that the profile itself did not hash them.

The workbook has one sheet per application, in the loom_private workbook's
order. Each sheet holds one table per GPU, in the order RTX 3090, RTX 4090,
RTX 5090; a GPU without a selected run shows its table with pending values.
Timings come only from bench records (the median over repetitions; bracket
measurements of the Graph baseline only give the drift). The step columns use
the application's "scope" from stages.json. A step is flagged as within noise
when its saving is at most the larger repetition range (max - min) of its two
stages; shares are undefined when the Graph->Optimized saving is not positive
or is itself within noise. A step between equivalent settings (the profile
found the two stages' launch structures identical on this GPU, through a
"host_equal" pair, as for a tuning entry that equals the default) is not an
optimization: its time difference stays visible as variation, greyed, and it
gets no share. The kernel launch count comes from the run's nsys trace of one
invocation. Row heights follow the wrapped text, so no cell is clipped.
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
# The loom_private workbook's application order (results/loom_private/build_report.py).
APPS = ["pagerank", "attention", "decoder", "anomaly", "vecadd_memory", "vecadd_product"]
SCOPES = ("application", "compute")
FIELD = "batch_per_iteration_us"
PENDING = "pending"
UNDEFINED = "—"
CHANGE_LIMIT = 130
# Row height estimate: Calibri 11 fits about one character per unit of column
# width (fewer when bold), 15 points per line.
LINE_POINTS = 15


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


def scope_of(stages):
    return stages.get("scope", "compute")


def structure(stages):
    """What a run depends on: the binaries, the cases' ids, stages, binaries,
    and launches, and the checks. Labels, descriptions, and the reporting scope
    are free to change."""
    pick = lambda items, keys: [{k: i[k] for k in keys if k in i} for i in items]
    return {"binaries": stages["binaries"],
            "cases": pick(stages["cases"], ("id", "stage", "binary", "launch")),
            "sass_equal": pick(stages.get("sass_equal", []), ("binary", "reference", "kernel")),
            "host_equal": pick(stages.get("host_equal", []), ("binary", "reference", "gpus")),
            "arithmetic_changes": pick(stages.get("arithmetic_changes", []),
                                       ("stage", "gpus"))}


def stages_at(commit, app, expected_sha256):
    """stages.json as the selected run used it."""
    relative = f"experiments/loom_private_ablation/{app}/stages.json"
    text = subprocess.run(["git", "-C", str(REPO), "show", f"{commit}:{relative}"],
                          capture_output=True, check=True).stdout
    if hashlib.sha256(text).hexdigest() != expected_sha256:
        raise ValueError(f"{relative} at {commit[:7]} does not match the run's hash")
    return json.loads(text)


def timed_records(bench):
    """The bench entries that are case timings (not drift brackets)."""
    return [r for r in bench["records"] if r.get("role", "case") == "case"]


def digest_stages(stages, check):
    """[(first stage, digest, declared)] where each distinct digest begins."""
    if "digest_rule" in check:
        rule = check["digest_rule"]
    else:  # summaries before the digest rule: derive it from the binaries
        seen, rule = {}, []
        for case in stages["cases"]:
            if case["stage"] not in seen:
                seen[case["stage"]] = True
                rule.append({"stage": case["stage"], "declared_change": False,
                             "digest": check["binaries"][case["binary"]]["output_digest"]})
    starts, previous = [], None
    for item in rule:
        if item["digest"] != previous:
            starts.append((item["stage"], item["digest"], item["declared_change"]))
            previous = item["digest"]
    return starts


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
    if check:
        validated = {b["sha256"] for b in check["binaries"].values()}
        if bench:
            stray = {r["case"] for r in bench["records"] if r["binary_sha256"] not in validated}
            if stray:
                raise ValueError(f"{folder} {app}: bench binaries of {sorted(stray)} were not "
                                 "validated by the selected check")
        if profile and "check_summary" not in profile:
            meta, _ = metadata(folder, profile["stamp"])
            recorded = (meta or {}).get("binary_sha256", {})
            unmatched = sorted(k for k, b in check["binaries"].items()
                               if recorded.get(b["path"]) != b["sha256"])
            if unmatched:
                raise ValueError(f"{folder} {app}: the profile predates hash checks and the "
                                 f"run's recorded hashes do not match the check for "
                                 f"{unmatched}")
        elif profile:
            stray = {p.get("case") or p.get("reference") for p in profile["profiles"]
                     if p.get("binary_sha256") not in validated}
            if stray:
                raise ValueError(f"{folder} {app}: profiled binaries of {sorted(stray)} were "
                                 "not validated by the selected check (or the profile does "
                                 "not record their hashes)")
    cases = []
    for case in stages["cases"]:
        row = dict(case)
        if bench:
            records = [read_json(REPO / r["record"]) for r in timed_records(bench)
                       if r["case"] == case["id"]]
            row["dimensions"] = records[0]["dimensions"]
            for scope in SCOPES:
                values = [r["timing"][scope][FIELD] for r in records]
                ranked = sorted(records, key=lambda r: r["timing"][scope][FIELD])
                row[scope] = statistics.median(values)
                row[f"{scope}_values"] = values
                row[f"{scope}_range"] = max(values) - min(values)
                row[f"{scope}_record"] = ranked[(len(ranked) - 1) // 2]["_path"]
        if check:
            row["check"] = check["binaries"][case["binary"]]
        if profile:
            entry = next(p for p in profile["profiles"] if p.get("case") == case["id"])
            with open(REPO / entry["nsys_kernels"]) as stream:
                row["launches"] = sum(int(r["Instances"]) for r in csv.DictReader(stream))
            row["nsys"] = entry["nsys"]
        cases.append(row)
    equivalent = {frozenset((h["binary"], h["reference"]))
                  for h in (profile or {}).get("host_equal", []) if h["identical"]}
    previous = None
    for row in cases:
        row["equivalent_to_previous"] = bool(
            previous and previous["binary"] != row["binary"]
            and frozenset((previous["binary"], row["binary"])) in equivalent)
        if row["stage"] >= 1 or row["launch"] == "graph":
            previous = row
    return {"stages": stages, "cases": cases, "check": check, "bench": bench,
            "profile": profile}


def step_metrics(cases, scope):
    """Per case (in order): speedups vs the Graph baseline, and the step
    columns in `scope` against the previous stage. Stage 1's previous stage is
    the Graph baseline; both baseline rows have no step. Shares are None when
    undefined (the total saving is not positive or is within noise) and for a
    step between equivalent settings (case["equivalent_to_previous"]), whose
    difference is variation rather than an optimization's saving."""
    last = max(c["stage"] for c in cases)
    graph = next(c for c in cases if c["stage"] == 0 and c["launch"] == "graph")
    final = next(c for c in cases if c["stage"] == last)
    total = graph[scope] - final[scope]
    total_noise = abs(total) <= max(graph[f"{scope}_range"], final[f"{scope}_range"])
    shares = total > 0 and not total_noise
    out, previous = [], None
    for case in cases:
        m = {"application_vs_graph": graph["application"] / case["application"],
             "compute_vs_graph": graph["compute"] / case["compute"], "step": None}
        if previous is not None:
            saved = previous[scope] - case[scope]
            equivalent = case.get("equivalent_to_previous", False)
            m["step"] = {"speedup": previous[scope] / case[scope], "saved": saved,
                         "share": saved / total if shares and not equivalent else None,
                         "equivalent": equivalent,
                         "within_noise": abs(saved) <= max(previous[f"{scope}_range"],
                                                           case[f"{scope}_range"])}
        out.append(m)
        if case["stage"] >= 1 or case["launch"] == "graph":
            previous = case
    return {"rows": out, "total": total, "shares_defined": shares,
            "total_within_noise": total_noise}


def wrapped_lines(text, chars):
    """Lines that text takes when word-wrapped at `chars` characters."""
    lines = 0
    for paragraph in str(text).split("\n"):
        lines += 1
        used = 0
        for word in paragraph.split():
            if used and used + 1 + len(word) > chars:
                lines += 1
                used = 0
            used += (1 if used else 0) + len(word)
            while used > chars:  # a word longer than the line breaks anywhere
                lines += 1
                used -= chars
    return lines


def fitted_height(cells, minimum):
    """Row height (points) that shows every (text, width, bold) cell in full."""
    lines = max(wrapped_lines(text, max(1, int(width * (0.9 if bold else 1.0)) - 1))
                for text, width, bold in cells)
    return max(minimum, LINE_POINTS * lines + 4)


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


def headers(scope):
    word = scope.capitalize()
    return ["Stage", "Version", "What changed", "Launch", "Application (us)",
            "Compute (us)", "Application speedup vs Graph", "Compute speedup vs Graph",
            f"{word} speedup over previous stage", f"{word} time saved by this step (us)",
            f"Share of Graph→Optimized {scope} saving", "Kernel launches per invocation"]


WIDTHS = [7, 30, 62, 8, 13, 13, 13, 13, 14, 15, 15, 12]


def short_label(case):
    return case["launch"].capitalize() if case["stage"] == 0 else f"stage {case['stage']}"


def run_line(folder, data):
    """The run description above a GPU's table."""
    bench, check, profile, cases = data["bench"], data["check"], data["profile"], data["cases"]
    meta, meta_path = metadata(folder, bench["stamp"])
    nvcc = ((meta or {}).get("installed_tools", {}).get("nvcc") or "").strip()
    order = "a rotated order with a Graph baseline bracket" if bench.get("order") == "rotated" \
        else "a fixed stage order"
    text = (f"Run {bench['stamp']} at source {bench['source_commit'][:7]}, driver "
            f"{bench['device']['driver']}, "
            f"{nvcc.splitlines()[-1] if nvcc else 'unknown toolkit'}. "
            f"Median of {bench['repeat']} repetitions in {order}.")
    if check:
        starts = digest_stages(data["stages"], check)
        if len(starts) == 1:
            text += f" Every binary passed the large check with digest {starts[0][1]}"
        else:
            parts = [f"{d} from stage {s}" + (" (declared arithmetic change)" if declared
                                              else "") for s, d, declared in starts]
            text += " Every binary passed the large check; digests " + ", ".join(parts)
        same = [f"{s['binary']} matches {s['reference']} instruction for instruction in "
                f"{s['kernel']} ({s['instructions'][0]})"
                for s in check.get("sass_equal", []) if s["identical"]]
        text += ("; " + "; ".join(same) + "." if same else ".")
    else:
        text += " Check pending."
    host = [f"{h['binary']} launches exactly like {h['reference']}"
            for h in (profile or {}).get("host_equal", []) if h["identical"]]
    if host:
        text += " Host-side launch structure: " + "; ".join(host) + "."
    if profile and "check_summary" not in profile:
        text += (" The profile predates per-binary hash checks: the recorder's hashes after "
                 "validation match the check, but the profile did not hash what it ran.")
    return text, meta_path


def note_line(data, scope, metrics):
    bench, cases = data["bench"], data["cases"]
    ranges = ", ".join(f"{short_label(c)} {c[f'{scope}_range']:,.1f}" for c in cases)
    text = (f"Grey italic step cells are within noise: the step's saving is at most the larger "
            f"repetition range (max − min) of its two stages. This is a heuristic flag; "
            f"{bench['repeat']} repetitions do not establish statistical significance. "
            f"Ranges ({scope}, us): {ranges}. ")
    if bench.get("drift_us"):
        drift = ", ".join(f"{d[scope]:+,.1f}" for d in bench["drift_us"])
        text += (f"Drift of the Graph baseline ({scope}, end − start of each repetition, us): "
                 f"{drift}.")
    else:
        text += "Drift: not measured (this run timed the stages in a fixed order, without a "\
                "Graph baseline bracket)."
    if not metrics["shares_defined"]:
        text += (" Shares are undefined because the Graph→Optimized saving is not positive or "
                 "is within noise.")
    for case, m in zip(cases, metrics["rows"]):
        step = m["step"]
        if step and step["equivalent"]:
            text += (f" Stage {case['stage']} runs the same configuration as the stage before it "
                     f"on this GPU (identical launch structure in the profile): its "
                     f"{step['saved']:+,.1f} us is variation between equivalent settings, not a "
                     f"tuning gain, and gets no share")
            if metrics["shares_defined"]:
                text += (f" (the other shares add up to "
                         f"{1 - step['saved'] / metrics['total']:.1%})")
            text += "."
    return text


def render(sheets, labels=None):
    """The workbook for [(app, stages, {folder: data})]."""
    from openpyxl import Workbook
    from openpyxl.comments import Comment
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter

    labels = labels or {}
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
    links = {}
    book = Workbook()
    book.remove(book.active)

    def github(path):
        if path not in links:
            links[path] = link(path)
        return links[path]

    def text_row(ws, row, text, height=30, font=None):
        ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=len(WIDTHS))
        c = ws.cell(row, 1, text)
        c.alignment = wrap
        if font:
            c.font = font
        ws.row_dimensions[row].height = fitted_height([(text, sum(WIDTHS), False)], height)
        return c

    def cell(ws, row, col, value, fmt=None, fill=None, source=None, faint=False):
        c = ws.cell(row, col, value)
        c.border, c.alignment = box, wrap
        if fill:
            c.fill = fill
        if value == PENDING or faint:
            c.font = grey
        if fmt and isinstance(value, (int, float)):
            c.number_format = fmt
            if source and not faint:
                c.hyperlink, c.font = github(source), blue
        return c

    def gpu_table(ws, row, folder, label, data, scope):
        """One GPU's title, run line, stage table, and note from `row`; returns
        the next free row."""
        cases, bench = data["cases"], data["bench"]
        last = max(c["stage"] for c in cases)
        ws.cell(row, 1, label).font = Font(bold=True, size=13)
        row += 1
        metrics = step_metrics(cases, scope) if bench else None
        if bench:
            text, meta_path = run_line(folder, data)
            c = text_row(ws, row, text, 44)
            if meta_path:
                c.hyperlink, c.font = github(meta_path), blue
        else:
            text_row(ws, row, "Pending: no run selected in "
                     f"results/loom_private_ablation/inputs/{folder}.json.", 16, grey)
        row += 1
        for i, header in enumerate(headers(scope), 1):
            c = ws.cell(row, i, header)
            c.font, c.fill, c.border = header_font, header_fill, box
            c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        ws.row_dimensions[row].height = fitted_height(
            [(h, w, True) for h, w in zip(headers(scope), WIDTHS)], 44)
        for index, case in enumerate(cases):
            row += 1
            f = fills[case["launch"]] if case["stage"] == 0 else \
                fills["optimized"] if case["stage"] == last else fills["stage"]
            cell(ws, row, 1, case["stage"], fill=f)
            cell(ws, row, 2, case["label"], fill=f).font = bold
            cell(ws, row, 3, case["change"], fill=f)
            cell(ws, row, 4, case["launch"].capitalize(), fill=f)
            if bench:
                m = metrics["rows"][index]
                for col, scope_name in ((5, "application"), (6, "compute")):
                    c = cell(ws, row, col, round(case[scope_name], 1), "#,##0.0", f,
                             case[f"{scope_name}_record"])
                    c.comment = Comment("Repetitions (us): " + ", ".join(
                        f"{v:,.1f}" for v in case[f"{scope_name}_values"]), "build_report")
                cell(ws, row, 7, m["application_vs_graph"], '0.00"x"', f)
                cell(ws, row, 8, m["compute_vs_graph"], '0.00"x"', f)
                step = m["step"]
                if step is None:
                    for col in (9, 10, 11):
                        cell(ws, row, col, UNDEFINED, fill=f)
                else:
                    faint = step["within_noise"] or step["equivalent"]
                    cell(ws, row, 9, step["speedup"], '0.00"x"', f, faint=faint)
                    saved = cell(ws, row, 10, round(step["saved"], 1), "#,##0.0", f,
                                 faint=faint)
                    if step["equivalent"]:
                        saved.comment = Comment(
                            "Equivalent settings on this GPU: this stage launches exactly "
                            "like the previous one (host_equal in the profile). The "
                            "difference is variation between two runs of one configuration, "
                            "not a tuning gain, and gets no share.", "build_report")
                    if step["share"] is None:
                        cell(ws, row, 11, UNDEFINED, fill=f, faint=faint)
                    else:
                        cell(ws, row, 11, step["share"], "0.0%", f, faint=faint)
            else:
                for col in range(5, 12):
                    cell(ws, row, col, PENDING, fill=f)
            cell(ws, row, 12, case.get("launches", PENDING), "#,##0", f, case.get("nsys"))
            ws.row_dimensions[row].height = fitted_height(
                [(case["label"], WIDTHS[1], True), (case["change"], WIDTHS[2], False)], 30)
        row += 1
        if bench:
            text_row(ws, row, note_line(data, scope, metrics), 44, grey)
        else:
            text_row(ws, row, "Ranges and drift pending.", 16, grey)
        return row + 2

    for app, stages, tables in sheets:
        scope = scope_of(stages)
        ws = book.create_sheet(stages["title"])
        for i, width in enumerate(WIDTHS, 1):
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
                 "Speedups divide the Graph baseline's (or the previous stage's) time by this "
                 f"row's. The step columns use the {scope} scope; stage 1's previous stage is "
                 f"the Graph baseline. The shares split the Graph→Optimized {scope} saving in "
                 "this order: they are order-dependent cumulative contributions, so another "
                 "order could split it differently, and a step that slows the run shows a "
                 "negative saving. Kernel launches come from an nsys trace of one invocation. "
                 f"Design notes: experiments/loom_private_ablation/{app}/README.md.", 44)
        row = 5
        for folder, label in GPUS.items():
            row = gpu_table(ws, row, folder, labels.get(folder, label), tables[folder], scope)
    return book


def workbook(inputs):
    present = {p.parent.name for p in EXPERIMENTS.glob("*/stages.json")}
    unknown = present - set(APPS)
    if unknown:
        raise ValueError(f"stages.json for applications outside APPS: {sorted(unknown)}")
    sheets = []
    for app in (a for a in APPS if a in present):
        stages = json.loads((EXPERIMENTS / app / "stages.json").read_text())
        tables = {folder: load_app(folder, app,
                                   inputs.get(folder, {}).get("applications", {}).get(app))
                  for folder in GPUS}
        sheets.append((app, stages, tables))
    return render(sheets, {f: m["label"] for f, m in inputs.items() if "label" in m})


def main():
    workbook(load_inputs()).save(OUT)
    print(f"wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
