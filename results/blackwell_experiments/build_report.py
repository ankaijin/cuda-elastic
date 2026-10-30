#!/usr/bin/env python3
"""Build the RTX 5090 feature experiment report from selected runs.

  python3 results/blackwell_experiments/build_report.py [--output PATH]

Writes results/blackwell_experiments/README.md (or PATH) from the input
manifests results/blackwell_experiments/inputs/<gpu>.json. For each
experiment under experiments/blackwell, a manifest selects run summaries
written by experiments/blackwell/run.py, with paths relative to the GPU's
folder (null means pending):

- `check`: the check summary (`<stamp>-check.json`).
- `bench`: the five-pair bench summary.
- `confirm`: the ten-pair confirmation summary, or null when none was run.
- `profile`: the profile summary.
- `conclusions`: one of improved, no measurable benefit, regressed, or
  unsupported for each candidate case, or null while pending.
- `notes`: optional sentences shown under the conclusions.

A summary of another experiment or GPU, a failed summary, a listed record
whose SHA-256 differs from the summary's, benches with settings other than 20
iterations and 3 warmups, and a confirmation with fewer than ten pairs are
errors. The report keeps implementation correctness, numerical quality, and
performance in separate sections, and states the decision rule of
experiments/blackwell/README.md next to each timing table. Nothing is rerun;
it needs only the standard library.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys

REPO = Path(__file__).resolve().parents[2]
HERE = REPO / "results" / "blackwell_experiments"
EXPERIMENTS = REPO / "experiments" / "blackwell"
ORDER = ["packed", "tma", "lowprec"]
OUTCOMES = {"improved", "no measurable benefit", "regressed", "unsupported"}
LAUNCHES = ("direct", "graph")
SCOPES = ("compute", "application")
SCOPE_NAMES = {"compute": "kernels (compute scope)", "application": "application scope"}
# Kernels whose Nsight Compute metrics the report sums, per experiment.
PROFILE_KERNELS = {
    "packed": ("add_records",),
    "tma": ("attention_partial_kernel",),
    "lowprec": ("classifier_kernel", "cutlass", "quantize_", "to_bf16_kernel",
                "tensor_amax_kernel", "rmsnorm_kernel", "softmax_argmax_kernel"),
}
# Nsight Compute reports durations in the unit its second CSV row names.
TIME_UNITS = {"nsecond": 1e-3, "ns": 1e-3, "usecond": 1.0, "us": 1.0, "msecond": 1e3, "ms": 1e3}
NCU_METRICS = [("gpu__time_duration.sum", "kernel time (µs)", 1.0, "sum"),
               ("smsp__inst_executed.sum", "warp instructions", 1.0, "sum"),
               ("launch__registers_per_thread", "registers", 1.0, "max"),
               ("smsp__issue_active.avg.pct_of_peak_sustained_active", "issue active (%)", 1.0,
                "median")]


class ReportError(Exception):
    pass


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(folder, relative):
    path = (REPO / folder / relative).resolve()
    if not path.is_relative_to((REPO / folder).resolve()) or not path.is_file():
        raise ReportError(f"{folder}/{relative} is missing or outside {folder}/")
    return path, json.loads(path.read_text())


def checked_summary(folder, experiment, relative, action):
    path, summary = load(folder, relative)
    if summary.get("experiment") != experiment or summary.get("action") != action:
        raise ReportError(f"{path.relative_to(REPO)} is not a {experiment} {action} summary")
    if summary.get("status") != "pass":
        raise ReportError(f"{path.relative_to(REPO)} did not pass")
    if summary["gpu"]["folder"] != folder:
        raise ReportError(f"{path.relative_to(REPO)} comes from {summary['gpu']['folder']}")
    for record in summary["records"]:
        if "record_sha256" in record and sha256(REPO / record["record"]) != record["record_sha256"]:
            raise ReportError(f"{record['record']} changed after its summary was written")
    if action == "bench":
        settings = summary["settings"]
        if settings["iterations"] != 20 or settings["warmup"] != 3 or settings["launch"] != "both":
            raise ReportError(f"{path.relative_to(REPO)} does not use 20 iterations, 3 warmups, "
                              "and both launch modes")
    return path, summary


def run_metadata(folder, stamp):
    path = REPO / folder / "runs" / stamp / "metadata.json"
    return json.loads(path.read_text()) if path.is_file() else None


def fmt(value, digits=1):
    return f"{value:,.{digits}f}"


def classify(control, candidate, confirmation):
    """The decision rule of experiments/blackwell/README.md for one table cell."""
    ratio = candidate["paired_ratio"]
    half = (control["max"] - control["min"]) / 2 / control["median"]
    n, faster = ratio["n"], ratio["below_one"]
    slower = n - faster
    if confirmation:
        if ratio["median"] < 1 - half and faster >= 9 * n / 10:
            return "confirmed improvement"
        if ratio["median"] > 1 + half and slower >= 9 * n / 10:
            return "confirmed regression"
        return "within variation"
    if ratio["median"] < 1 - half and faster > n / 2:
        return "apparent improvement"
    if ratio["median"] > 1 + half and slower > n / 2:
        return "apparent regression"
    return "within variation"


def timing_table(summary, cases, confirmation):
    control = cases[0]
    lines = ["| Scope | Launch | " + " | ".join(
        [f"`{control}` µs (range)"] + [f"`{c}` µs, ratio (range), faster pairs, rule" for c in cases[1:]])
        + " |", "|---|---|" + "---|" * len(cases)]
    for scope in SCOPES:
        for launch in LAUNCHES:
            entry = summary["timing"].get(f"{launch}/{scope}/batch_per_iteration_us")
            if not entry:
                continue
            c = entry[control]
            row = [SCOPE_NAMES[scope], launch.capitalize(),
                   f"{fmt(c['median'])} ({fmt(c['min'])}–{fmt(c['max'])})"]
            for case in cases[1:]:
                t = entry[case]
                r = t["paired_ratio"]
                row.append(f"{fmt(t['median'])}, {r['median']:.4f} ({r['min']:.4f}–{r['max']:.4f}), "
                           f"{r['below_one']}/{r['n']}, {classify(c, t, confirmation)}")
            lines.append("| " + " | ".join(row) + " |")
    return lines


def ncu_rows(folder, profile, kernels):
    rows = []
    for record in profile["records"]:
        raw = next((REPO / f for f in record.get("files", []) if f.endswith("-ncu-raw.csv")), None)
        if raw is None:
            continue
        with raw.open() as stream:
            table = list(csv.reader(stream))
        index = {name: i for i, name in enumerate(table[0])}
        launches = [r for r in table[2:]
                    if any(k in r[index["Kernel Name"]] for k in kernels)]
        values = []
        for metric, _, scale, how in NCU_METRICS:
            if metric not in index or not launches:
                values.append("")
                continue
            unit = TIME_UNITS.get(table[1][index[metric]], 1.0) if metric.startswith("gpu__time") else 1.0
            numbers = [float(r[index[metric]].replace(",", "")) * scale * unit for r in launches]
            value = {"sum": sum, "max": max, "median": statistics.median}[how](numbers)
            values.append(fmt(value, 0 if how != "median" else 1))
        rows.append((record["case"], len(launches), values, raw.relative_to(REPO)))
    return rows


def accuracy_tables(check):
    records = [json.loads((REPO / r["record"]).read_text()) for r in check["records"]
               if r.get("kind") == "accuracy"]
    if not records:
        return []
    lines = ["Classifier error on fixed FP32 reference activations, and token agreement over a "
             "complete four-step decode:", "",
             "| Case | Dataset | Max abs error | Normalized RMS error | Max abs probability error "
             "| Mean total variation | Top-1 agree | Tokens agree | FP32 tolerance check |",
             "|---|---|--:|--:|--:|--:|--:|--:|---|"]
    weights = []
    for record in records:
        result = record["result"]
        weights.append((record["case"], result["weight_conversion_ms_median_of_5"]))
        for d in result["datasets"]:
            e, t = d["classifier_error"], d["tokens"]
            tolerance = d.get("production_fp32_tolerance_pass")
            lines.append(
                f"| `{record['case']}` | {d['dataset']} | {e['max_abs_error']:.3g} | "
                f"{e['normalized_rms_error']:.3g} | {e['max_abs_probability_error']:.3g} | "
                f"{e['mean_total_variation']:.3g} | {e['top1_agree']}/{e['rows']} | "
                f"{t['agree']}/{t['total']} | "
                + ("" if tolerance is None else "pass" if tolerance else "fail") + " |")
    lines += ["", "The FP32 tolerance column is the production contract check of the primary "
              "decode. A failure there is a precision result, not an implementation failure.", "",
              "Weight conversion at setup (median of five, outside every timed scope): "
              + ", ".join(f"`{case}` {ms:.3f} ms" for case, ms in weights) + "."]
    return lines


def experiment_section(folder, name, selection):
    config = json.loads((EXPERIMENTS / name / "experiment.json").read_text())
    cases = [c["name"] for c in config["cases"]]
    lines = [f"## {config['title']} (`{name}`)", "", f"**Question.** {config['question']}", "",
             f"**Workload.** {config['workload']}.", "",
             "| Case | Role | Description |", "|---|---|---|"]
    lines += [f"| `{c['name']}` | {c['role']} | {c['description']} |" for c in config["cases"]]
    lines.append("")
    if not selection.get("check"):
        return lines + ["Pending: no check selected.", ""]
    check_path, check = checked_summary(folder, name, selection["check"], "check")

    # Provenance.
    builds = {}
    for r in check["records"]:
        if r.get("kind") == "check":
            builds[r["case"]] = json.loads((REPO / r["record"]).read_text())["build"]
    first = builds[cases[0]]
    meta = run_metadata(folder, check["stamp"])
    lines += ["### Provenance", "",
              f"- GPU: {check['gpu']['name']} `{check['gpu']['uuid']}`, driver "
              f"{check['gpu']['driver']}.",
              f"- Compiler: {first['compiler_version'].splitlines()[-2] if first['compiler_version'] else 'unknown'}.",
              f"- Architecture: " + ", ".join(sorted({f'`{b['arch']}`' for b in builds.values()})) + ".",
              f"- Specification: `{first['spec']}` (SHA-256 `{first['spec_sha256'][:16]}…`)."]
    # Include directories outside the repository, one line per git checkout.
    externals = {}
    for b in builds.values():
        for path, info in b["external_includes"].items():
            externals[info["git_toplevel"] or path] = info
    for top, info in sorted(externals.items()):
        lines.append(f"- External dependency `{Path(top).name}`: {info['git_describe']} "
                     f"`{info['git_commit']}`" + (" (dirty)" if info["dirty"] else "") + ".")
    runs = [("check", selection["check"])] + [(k, selection.get(k)) for k in ("bench", "confirm", "profile")]
    for kind, relative in runs:
        if not relative:
            lines.append(f"- {kind.capitalize()}: pending.")
            continue
        _, summary = load(folder, relative)
        metadata = run_metadata(folder, summary["stamp"])
        commit = metadata["source_commit"][:7] if metadata else summary["source"]["commit"][:7]
        lines.append(f"- {kind.capitalize()}: run `{summary['stamp']}` at `{commit}`.")
    lines.append("")

    # Implementation correctness.
    lines += ["### Implementation correctness", ""]
    make = {tuple(r["targets"]): r for r in check["records"] if r.get("kind") == "make"}
    for targets, r in make.items():
        lines.append(f"- `make {' '.join(targets)}`{' (forced)' if r['force'] else ''}: "
                     f"exit {r['exit_code']} ([log](../../{r['log']})).")
    for r in check["records"]:
        if r.get("kind") == "check":
            lines.append(f"- `{r['case']}` large check, Direct and Graph: {r['status']}.")
    agreement = next((r for r in check["records"] if r.get("kind") == "digest_agreement"), None)
    if agreement:
        lines.append(f"- Output digests identical across cases: {'yes' if agreement['agree'] else 'no'}.")
    lines.append("")

    # Numerical quality.
    accuracy = accuracy_tables(check)
    if accuracy:
        lines += ["### Numerical quality", ""] + accuracy + [""]

    # Feature verification.
    sass = [r for r in check["records"] if r.get("kind") == "sass"]
    if sass:
        features = list(sass[0]["features"])
        lines += ["### Feature verification", "",
                  "SASS instruction counts in the experiment's kernels (`" +
                  config["sass"]["kernels"] + "`):", "",
                  "| Case | " + " | ".join(f"`{f}`" for f in features) + " |",
                  "|---|" + "--:|" * len(features)]
        lines += [f"| `{r['case']}` | " + " | ".join(str(r["features"][f]) for f in features) + " |"
                  for r in sass]
        lines.append("")

    # Performance.
    lines += ["### Performance", "",
              "Median batch time per iteration over the pairs (20 iterations, 3 warmups). The "
              "ratio is the median of candidate/control within each pair. The rule calls an "
              "improvement apparent when the median ratio is below 1 by more than the "
              "control's relative half range and most pairs improve; ten more pairs confirm it "
              "when at least nine improve and the median still clears the half range. "
              "Regressions are judged the same way.", ""]
    for kind, title, confirmation in (("bench", "Five pairs", False), ("confirm", "Ten confirmation pairs", True)):
        if not selection.get(kind):
            lines += [f"{title}: pending." if kind == "bench" else f"{title}: not run.", ""]
            continue
        _, bench = checked_summary(folder, name, selection[kind], "bench")
        if confirmation and bench["settings"]["pairs"] < 10:
            raise ReportError(f"{selection[kind]} has fewer than ten pairs")
        lines += [f"{title} (`{bench['stamp']}`):", ""] + timing_table(bench, cases, confirmation) + [""]

    # Profiles.
    if selection.get("profile"):
        _, profile = checked_summary(folder, name, selection["profile"], "profile")
        rows = ncu_rows(folder, profile, PROFILE_KERNELS[name])
        lines += ["### Profiles", "",
                  f"One Graph application-scope invocation per case (`{profile['stamp']}`); "
                  "profiled kernels run one at a time. Sums over the launches of the "
                  "experiment's kernels:", "",
                  "| Case | Launches | " + " | ".join(m[1] for m in NCU_METRICS) + " | Report |",
                  "|---|--:|" + "--:|" * len(NCU_METRICS) + "---|"]
        lines += [f"| `{case}` | {n} | " + " | ".join(values) + f" | [ncu](../../{path}) |"
                  for case, n, values, path in rows]
        lines.append("")

    # Conclusions.
    conclusions = selection.get("conclusions") or {}
    lines += ["### Conclusions", ""]
    for case in cases[1:]:
        outcome = conclusions.get(case)
        if outcome is not None and outcome not in OUTCOMES:
            raise ReportError(f"{name}/{case}: unknown conclusion {outcome!r}")
        lines.append(f"- `{case}`: **{outcome or 'pending'}**.")
    for note in selection.get("notes", []):
        lines.append(f"- {note}")
    lines.append("")
    return lines


def build(output):
    lines = ["# RTX 5090 feature experiments", "",
             "Generated by `results/blackwell_experiments/build_report.py` from "
             "`results/blackwell_experiments/inputs/`. Sources and design notes are in "
             "[`experiments/blackwell/`](../../experiments/blackwell). These experiments compare "
             "Blackwell features against the production optimized loom_private variants. Nothing "
             "here changes the published suite.", ""]
    for manifest_path in sorted((HERE / "inputs").glob("*.json")):
        manifest = json.loads(manifest_path.read_text())
        folder = manifest["folder"]
        lines += [f"Inputs: `results/blackwell_experiments/inputs/{manifest_path.name}` ({manifest['label']}).", ""]
        for name in ORDER:
            selection = manifest["experiments"].get(name)
            if selection is None:
                continue
            lines += experiment_section(folder, name, selection)
    output.write_text("\n".join(lines).rstrip() + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=HERE / "README.md")
    args = parser.parse_args()
    try:
        build(args.output)
    except ReportError as error:
        print(f"build_report.py: {error}", file=sys.stderr)
        return 1
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
