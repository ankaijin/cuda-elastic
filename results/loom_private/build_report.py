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
appears as a performance measurement. The test-machine table comes from the
recorder metadata (<gpu>/runs/<run-id>/metadata.json) of the selected records,
except the source revision, which the manifest states (`source_revision`, a
full commit hash): bench runs start from commits that only add earlier runs'
results, so their recorded commits need not stay in the history.
Nothing is rerun. Requires openpyxl.
"""

import hashlib
import json
import re
from pathlib import Path

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


def workbook(gpus):
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill

    bold, fill = Font(bold=True), PatternFill("solid", fgColor="DDEBF7")
    book = Workbook()
    book.remove(book.active)

    def sheet(title, header, rows):
        ws = book.create_sheet(title)
        ws.append(header)
        for cell in ws[1]:
            cell.font, cell.fill = bold, fill
        for row in rows:
            ws.append(row)
        for column in ws.columns:
            width = max(len(str(c.value or "")) for c in column)
            ws.column_dimensions[column[0].column_letter].width = min(max(width + 2, 10), 60)
        ws.freeze_panes = "A2"

    for manifest, rows in gpus:
        sheet(manifest["label"].split(" (")[0],
              ["Workload", "Large problem size", "Application scope", "Direct (us)",
               "Graph (us)", "Compute-only Direct (us)", "Compute-only Graph (us)",
               "Direct record", "Graph record"],
              [[(r["contract"] or {}).get("identity", r["app"]), size_text(r["contract"]),
                (r["contract"] or {}).get("timing_scope", {}).get("application", PENDING),
                timing(r["bench_direct"], "application"), timing(r["bench_graph"], "application"),
                timing(r["bench_direct"], "compute"), timing(r["bench_graph"], "compute"),
                (r["bench_direct"] or {}).get("_path", PENDING),
                (r["bench_graph"] or {}).get("_path", PENDING)] for r in rows])
    sheet("Correctness",
          ["GPU", "Workload", "Contract SHA-256", "Status", "Direct", "Graph",
           "Direct = Graph", "Record"],
          [[m["label"], (r["contract"] or {}).get("identity", r["app"]),
            (r["contract"] or {}).get("_sha256", PENDING), *check_cells(r["check"]),
            (r["check"] or {}).get("_path", PENDING)] for m, rows in gpus for r in rows])
    sheet("Environment", ["GPU", "Host", "Driver", "CUDA", "Source revision"],
          [environment(m, rows) for m, rows in gpus])
    sheet("Contracts",
          ["Workload", "Directory", "Contract SHA-256", "Upstream commit", "Extension"],
          [[(spec or {}).get("identity", app), f"applications/loom_private/{app}",
            (spec or {}).get("_sha256", PENDING),
            (spec or {}).get("upstream", {}).get("commit", PENDING),
            str((spec or {}).get("extension_of_upstream", PENDING))]
           for app, spec in ((a, contract(a)) for a in APPS)])
    return book


def main():
    gpus = [load_gpu(folder) for folder in GPUS]
    MD_OUT.write_text(markdown(gpus) + "\n")
    workbook(gpus).save(XLSX_OUT)
    print(f"wrote {MD_OUT.relative_to(REPO)} and {XLSX_OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
