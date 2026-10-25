#!/usr/bin/env python3
"""Build the loom_private v1 results page and workbook from selected records.

  python3 results/loom_private/build_report.py

Writes results/loom_private/README.md and results/loom_private/loom_private_v1.xlsx.
Inputs are pinned in results/loom_private/inputs/<gpu>.json, independently of
the legacy results/inputs manifests. Each application has three selections:
`check` (a `--action check` record), `bench_direct` and `bench_graph` (`--action
bench` records). Paths are relative to that GPU's folder. Null means pending.
Missing files, unsafe paths, records of another workload, records of another
action or launch, and records that validated an older contract are errors.

Timing cells come only from bench records, so a correctness check never
appears as a performance measurement. Nothing is rerun. Requires openpyxl.
"""

import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HERE = REPO / "results" / "loom_private"
SUITE = REPO / "applications" / "loom_private"
VERSION = "v1"
MD_OUT = HERE / "README.md"
XLSX_OUT = HERE / f"loom_private_{VERSION}.xlsx"
GPUS = ["rtx3090", "rtx4090", "rtx5090"]
APPS = ["pagerank", "attention", "decoder", "anomaly", "vecadd_memory", "vecadd_product"]
SELECTIONS = {"check": ("check", None), "bench_direct": ("bench", "direct"),
              "bench_graph": ("bench", "graph")}
PENDING = "pending"


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
    if record.get("action") != action or (launch and record.get("launch") != launch):
        raise ValueError(f"{where}: selected as {key} but records "
                         f"{record.get('action')}/{record.get('launch')}")
    if record.get("contract_sha256") != spec["_sha256"]:
        raise ValueError(f"{where}: record validated contract {record.get('contract_sha256')}, "
                         f"current contract is {spec['_sha256']}")
    if record.get("preset") != "large" or record.get("overrides"):
        raise ValueError(f"{where}: only unmodified large-preset records can be selected")
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
    return ", ".join(f"{k}={v:,}" if isinstance(v, int) else f"{k}={v}"
                     for k, v in spec.get("large", {}).items())


def check_cells(record):
    if record is None:
        return [PENDING] * 4
    def modes(launch):
        runs = [c for c in record["checks"] if c["launch"] == launch]
        return f"{sum(c['pass'] for c in runs)}/{len(runs)} pass" if runs else "not run"
    return [record["status"], modes("direct"), modes("graph"),
            "yes" if record["outputs_agree"] else "no"]


def fmt(value):
    return value if isinstance(value, str) else f"{value:,.2f}"


def markdown(gpus):
    lines = [
        f"# Private Loom CUDA workloads: results {VERSION}",
        "",
        "Generated by `results/loom_private/build_report.py` from the records selected in "
        "`results/loom_private/inputs/<gpu>.json`. This page is separate from the legacy "
        "[results](../README.md); the legacy workloads and their measurements are unchanged.",
        "",
        "The six applications port private Loom commit "
        "`4a75f8d743ef7fe755822bab0ea0fb78009c145b` (see "
        "[`applications/loom_private`](../../applications/loom_private)). Timings are "
        "microseconds per iteration in a back-to-back batch; pending means no bench record "
        "is selected. No optimized variants or cross-GPU comparisons exist yet.",
        "",
    ]
    for manifest, rows in gpus:
        lines += [f"## {manifest['label']}: timing", "",
                  f"Toolkit: {manifest['toolkit']}.", "",
                  "| Workload | Large problem size | Application scope | Direct (us) | Graph (us) "
                  "| Compute-only Direct (us) | Compute-only Graph (us) |",
                  "|---|---|---|--:|--:|--:|--:|"]
        for row in rows:
            spec = row["contract"]
            scope = (spec or {}).get("timing_scope", {}).get("application", PENDING)
            cells = [timing(row["bench_direct"], "application"),
                     timing(row["bench_graph"], "application"),
                     timing(row["bench_direct"], "compute"),
                     timing(row["bench_graph"], "compute")]
            identity = (spec or {}).get("identity", row["app"])
            lines.append(f"| `{identity}` | {size_text(spec)} | {scope} | "
                         + " | ".join(fmt(c) for c in cells) + " |")
        lines.append("")
    lines += ["## Correctness", "",
              "Large-workload correctness checks are validation evidence, not performance "
              "measurements. Each check runs the CPU oracle once, then two Direct and two CUDA "
              "Graph invocations from poisoned outputs; all four output digests must agree.", ""]
    for manifest, rows in gpus:
        lines += [f"### {manifest['label']}", "",
                  "| Workload | Contract SHA-256 | Status | Direct | Graph | Direct = Graph | Record |",
                  "|---|---|---|---|---|---|---|"]
        for row in rows:
            spec, record = row["contract"], row["check"]
            contract_hash = f"`{spec['_sha256'][:12]}`" if spec else PENDING
            link = f"[record](../../{record['_path']})" if record else PENDING
            identity = (spec or {}).get("identity", row["app"])
            lines.append(f"| `{identity}` | {contract_hash} | "
                         + " | ".join(check_cells(record)) + f" | {link} |")
        lines.append("")
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
