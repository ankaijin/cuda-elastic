"""Layout and arithmetic of results/loom_private_ablation/opt_results.xlsx, and
the pure helpers of experiments/loom_private_ablation/run.py."""

import importlib.util
import json
from pathlib import Path
import re
import sqlite3
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, REPO / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


report = load("opt_report", "results/loom_private_ablation/build_report.py")
runner = load("opt_runner", "experiments/loom_private_ablation/run.py")


def case(stage, launch, application, compute, cid=None, label=None, binary=None,
         app_values=None, compute_values=None):
    """A loaded case as load_app returns it, with explicit repetition values."""
    app_values = app_values or [application] * 3
    compute_values = compute_values or [compute] * 3
    return {"id": cid or f"s{stage}-{launch}", "stage": stage, "launch": launch,
            "binary": binary or ("baseline" if stage == 0 else f"stage{stage}"),
            "label": label or ({"direct": "Baseline, Direct", "graph": "Baseline, Graph"}
                               [launch] if stage == 0 else f"+ Step {stage}"),
            "change": f"Change {stage}.", "application": application, "compute": compute,
            "application_values": app_values, "compute_values": compute_values,
            "application_range": max(app_values) - min(app_values),
            "compute_range": max(compute_values) - min(compute_values),
            "application_record": "x.json", "compute_record": "x.json",
            "launches": 1, "dimensions": {"n": 10}}


def expected(cases, scope):
    """Independent recomputation of the step columns."""
    graph = [c for c in cases if c["stage"] == 0 and c["launch"] == "graph"][0]
    final = [c for c in cases if c["stage"] == max(x["stage"] for x in cases)][0]
    total = graph[scope] - final[scope]
    noise_total = max(graph[f"{scope}_range"], final[f"{scope}_range"])
    rows = []
    chain = [graph] + [c for c in cases if c["stage"] >= 1]
    for c in cases:
        if c["stage"] == 0:
            rows.append(None)
            continue
        prev = chain[chain.index(c) - 1]
        saved = prev[scope] - c[scope]
        share = saved / total if total > 0 and abs(total) > noise_total else None
        rows.append((prev[scope] / c[scope], saved, share,
                     abs(saved) <= max(prev[f"{scope}_range"], c[f"{scope}_range"])))
    return rows


def stages(title, scope, cases):
    return {"application": title.lower(), "title": title, "scope": scope,
            "cases": [{k: c[k] for k in ("id", "stage", "binary", "launch", "label", "change")}
                      for c in cases]}


def data(cases, stage_config, digests=None):
    digests = digests or {c["binary"]: "aaaa" for c in cases}
    rule, seen = [], set()
    for c in cases:
        if c["stage"] not in seen:
            seen.add(c["stage"])
            rule.append({"stage": c["stage"], "binary": c["binary"],
                         "digest": digests[c["binary"]], "declared_change": False})
    for c in cases:
        c["check"] = {"output_digest": digests[c["binary"]]}
    return {"stages": stage_config, "cases": cases,
            "check": {"digest_rule": rule, "sass_equal": []},
            "bench": {"stamp": "20990101-000000-test", "source_commit": "0" * 40,
                      "device": {"driver": "1.0"}, "repeat": 3, "order": "rotated",
                      "drift_us": [{"repetition": r, "application": 1.0, "compute": -0.5}
                                   for r in (1, 2, 3)]},
            "profile": {"host_equal": []}}


def pending(cases, stage_config):
    return {"stages": stage_config, "check": None, "bench": None, "profile": None,
            "cases": [{k: c[k] for k in ("id", "stage", "binary", "launch", "label", "change")}
                      for c in cases]}


class StepArithmeticTests(unittest.TestCase):
    def check(self, cases, scope):
        metrics = report.step_metrics(cases, scope)
        for m, e in zip(metrics["rows"], expected(cases, scope)):
            if e is None:
                self.assertIsNone(m["step"])
                continue
            speedup, saved, share, noise = e
            self.assertAlmostEqual(m["step"]["speedup"], speedup)
            self.assertAlmostEqual(m["step"]["saved"], saved)
            if share is None:
                self.assertIsNone(m["step"]["share"])
            else:
                self.assertAlmostEqual(m["step"]["share"], share)
            self.assertEqual(m["step"]["within_noise"], noise)
        graph = [c for c in cases if c["launch"] == "graph" and c["stage"] == 0][0]
        for m, c in zip(metrics["rows"], cases):
            self.assertAlmostEqual(m["application_vs_graph"], graph["application"] / c["application"])
            self.assertAlmostEqual(m["compute_vs_graph"], graph["compute"] / c["compute"])
        return metrics

    def test_slowdown_and_unchanged_steps(self):
        cases = [case(0, "direct", 120, 110), case(0, "graph", 100, 90,
                                                   compute_values=[89, 90, 91]),
                 case(1, "graph", 80, 70),
                 case(2, "graph", 85, 75),       # slowdown: negative saving and share
                 case(3, "graph", 85, 75.5, compute_values=[75, 75.5, 76]),  # within noise
                 case(4, "graph", 40, 30)]
        metrics = self.check(cases, "compute")
        steps = [m["step"] for m in metrics["rows"]]
        self.assertLess(steps[3]["saved"], 0)
        self.assertLess(steps[3]["share"], 0)
        self.assertFalse(steps[3]["within_noise"])
        self.assertTrue(steps[4]["within_noise"])
        self.assertAlmostEqual(sum(s["share"] for s in steps if s), 1.0)

    def test_zero_total_saving_leaves_shares_undefined(self):
        cases = [case(0, "direct", 110, 50), case(0, "graph", 100, 50),
                 case(1, "graph", 90, 50), case(2, "graph", 100, 50)]
        metrics = self.check(cases, "compute")
        self.assertFalse(metrics["shares_defined"])
        self.assertTrue(all(m["step"] is None or m["step"]["share"] is None
                            for m in metrics["rows"]))

    def test_total_within_noise_leaves_shares_undefined(self):
        cases = [case(0, "direct", 110, 50), case(0, "graph", 100, 50, app_values=[95, 100, 105]),
                 case(1, "graph", 98, 50)]
        metrics = self.check(cases, "application")
        self.assertTrue(metrics["total_within_noise"])
        self.assertIsNone(metrics["rows"][2]["step"]["share"])

    def test_application_scope_uses_application_times(self):
        cases = [case(0, "direct", 600, 20), case(0, "graph", 500, 20),
                 case(1, "graph", 400, 21), case(2, "graph", 300, 19)]
        metrics = self.check(cases, "application")
        self.assertAlmostEqual(metrics["total"], 200)
        self.assertAlmostEqual(metrics["rows"][2]["step"]["share"], 0.5)


class LayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        report.link = lambda path: f"https://example.invalid/{path}"
        compute = [case(0, "direct", 120, 110), case(0, "graph", 100, 90),
                   case(1, "graph", 80, 70), case(2, "graph", 40, 30,
                                                  label="+ Per-GPU tuning (= Optimized)")]
        application = [case(0, "direct", 600, 20), case(0, "graph", 500, 20),
                       case(1, "graph", 300, 20, label="+ Pipeline (= Optimized)")]
        cls.configs = [("pagerank", stages("PageRank", "compute", compute), compute),
                       ("vecadd_product", stages("Vecadd product", "application", application),
                        application)]
        sheets = []
        for app, config, cases in cls.configs:
            tables = {"rtx3090": data([dict(c) for c in cases], config)}
            tables.update({f: pending(cases, config) for f in ("rtx4090", "rtx5090")})
            sheets.append((app, config, tables))
        cls.book = report.render(sheets)

    def tables(self, ws, n):
        """Row numbers of each GPU table's label row."""
        starts, row = [], 5
        for _ in report.GPUS:
            starts.append(row)
            row += 3 + n + 1 + 1  # label, run, header, stages, note, blank
        return starts

    def test_sheets_rows_headers_and_formats(self):
        self.assertEqual(self.book.sheetnames, ["PageRank", "Vecadd product"])
        for (app, config, cases), ws in zip(self.configs, self.book.worksheets):
            scope = config["scope"]
            self.assertTrue(ws["A1"].value.startswith(config["title"]))
            self.assertIn(f"loom-private/{app}", ws["A2"].value)
            self.assertIn(f"use the {scope} scope", ws["A3"].value)
            for i, width in enumerate(report.WIDTHS):
                self.assertEqual(ws.column_dimensions[chr(65 + i)].width, width)
            for (folder, label), start in zip(report.GPUS.items(), self.tables(ws, len(cases))):
                self.assertEqual(ws.cell(start, 1).value, label)
                header = [ws.cell(start + 2, c).value for c in range(1, 13)]
                self.assertEqual(header, report.headers(scope))
                self.assertIn(scope.capitalize(), header[8])
                rows = range(start + 3, start + 3 + len(cases))
                self.assertEqual([ws.cell(r, 1).value for r in rows], [c["stage"] for c in cases])
                self.assertEqual(ws.cell(rows[0], 2).value, "Baseline, Direct")
                self.assertEqual(ws.cell(rows[1], 2).value, "Baseline, Graph")
                self.assertTrue(ws.cell(rows[-1], 2).value.endswith("(= Optimized)"))
                fills = [ws.cell(r, 1).fill.fgColor.rgb[-6:] for r in rows]
                self.assertEqual(fills[:2], ["FCE4D6", "E2EFDA"])
                self.assertEqual(fills[-1], "DDEBF7")
                self.assertTrue(all(f == "F2F2F2" for f in fills[2:-1]))
                note = ws.cell(rows[-1] + 1, 1).value
                if folder == "rtx3090":
                    self.assertEqual(ws.cell(rows[0], 5).number_format, "#,##0.0")
                    self.assertEqual(ws.cell(rows[0], 7).number_format, '0.00"x"')
                    self.assertEqual(ws.cell(rows[2], 11).number_format, "0.0%")
                    self.assertEqual([ws.cell(rows[1], c).value for c in (9, 10, 11)],
                                     [report.UNDEFINED] * 3)
                    self.assertIn("Repetitions", ws.cell(rows[2], 5).comment.text)
                    self.assertIn("heuristic", note)
                    self.assertIn("Drift of the Graph baseline", note)
                else:
                    self.assertTrue(all(ws.cell(r, c).value == report.PENDING
                                        for r in rows for c in range(5, 12)))
                    self.assertTrue(ws.cell(rows[0], 5).font.i)
                    self.assertIn("pending", note.lower())

    def test_within_noise_cells_are_grey(self):
        cases = [case(0, "direct", 120, 110), case(0, "graph", 100, 90, compute_values=[85, 90, 95]),
                 case(1, "graph", 98, 88), case(2, "graph", 40, 30,
                                                label="+ Per-GPU tuning (= Optimized)")]
        config = stages("PageRank", "compute", cases)
        tables = {"rtx3090": data(cases, config)}
        tables.update({f: pending(cases, config) for f in ("rtx4090", "rtx5090")})
        ws = report.render([("pagerank", config, tables)]).active
        self.assertTrue(ws.cell(10, 10).font.i)      # stage 1: 2 us saved, 10 us range
        self.assertFalse(ws.cell(11, 10).font.i)


class RealWorkbookTests(unittest.TestCase):
    def test_apps_match_the_loom_private_order(self):
        source = (REPO / "results/loom_private/build_report.py").read_text()
        apps = json.loads(re.search(r"^APPS = (\[.*\])$", source, re.M).group(1).replace("'", '"'))
        self.assertEqual(report.APPS, apps)

    def test_every_stages_json_follows_the_conventions(self):
        for path in sorted((REPO / "experiments/loom_private_ablation").glob("*/stages.json")):
            config = json.loads(path.read_text())
            with self.subTest(app=path.parent.name):
                self.assertIn(path.parent.name, report.APPS)
                self.assertIn(config.get("scope"), report.SCOPES)
                cases = config["cases"]
                self.assertEqual([c["id"] for c in cases[:2]], ["baseline-direct", "baseline-graph"])
                self.assertEqual(sorted({c["stage"] for c in cases}),
                                 list(range(max(c["stage"] for c in cases) + 1)))
                self.assertTrue(cases[-1]["label"].endswith("(= Optimized)"))
                self.assertEqual(config["binaries"][cases[-1]["binary"]],
                                 {"suite_variant": "optimized"})
                self.assertTrue(all(c["label"].startswith("+ ") for c in cases[2:]))
                self.assertTrue(all(len(c["change"]) <= report.CHANGE_LIMIT for c in cases))

    def test_generated_workbook_shares_add_up(self):
        inputs = report.load_inputs()
        for app in (a for a in report.APPS
                    if (report.EXPERIMENTS / a / "stages.json").is_file()):
            config = json.loads((report.EXPERIMENTS / app / "stages.json").read_text())
            for folder in report.GPUS:
                selection = inputs.get(folder, {}).get("applications", {}).get(app)
                loaded = report.load_app(folder, app, selection)
                if not loaded["bench"]:
                    continue
                metrics = report.step_metrics(loaded["cases"], report.scope_of(config))
                if metrics["shares_defined"]:
                    with self.subTest(app=app, gpu=folder):
                        self.assertAlmostEqual(sum(m["step"]["share"] for m in metrics["rows"]
                                                   if m["step"]), 1.0)


class RunnerTests(unittest.TestCase):
    config = {"cases": [{"id": "d", "stage": 0, "binary": "base", "launch": "direct"},
                        {"id": "g", "stage": 0, "binary": "base", "launch": "graph"},
                        {"id": "s1", "stage": 1, "binary": "s1", "launch": "graph"},
                        {"id": "s2", "stage": 2, "binary": "s2", "launch": "graph"},
                        {"id": "s3", "stage": 3, "binary": "opt", "launch": "graph"}],
              "arithmetic_changes": [{"stage": 1}, {"stage": 3, "gpus": ["rtx3090"]}]}

    def test_bench_order_rotates(self):
        cases = self.config["cases"]
        self.assertEqual([c["id"] for c in runner.bench_order(cases, 1)], ["d", "g", "s1", "s2", "s3"])
        self.assertEqual([c["id"] for c in runner.bench_order(cases, 2)], ["g", "s1", "s2", "s3", "d"])
        self.assertEqual([c["id"] for c in runner.bench_order(cases, 6)], ["d", "g", "s1", "s2", "s3"])

    def test_digest_rule_inherits_declared_changes(self):
        digests = {"base": "a", "s1": "b", "s2": "b", "opt": "c"}
        rule = runner.digest_rule(self.config, digests, "rtx3090")
        self.assertTrue(all(r["equal_to_previous"] or r["declared_change"] for r in rule))
        rule = runner.digest_rule(self.config, digests, "rtx4090")
        bad = [r["stage"] for r in rule if not r["equal_to_previous"] and not r["declared_change"]]
        self.assertEqual(bad, [3])
        rule = runner.digest_rule(self.config, {**digests, "s2": "z"}, "rtx3090")
        bad = [r["stage"] for r in rule if not r["equal_to_previous"] and not r["declared_change"]]
        self.assertEqual(bad, [2])

    def test_launch_structure_counts_and_compares(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "t.sqlite"
            c = sqlite3.connect(db)
            c.executescript("""
                CREATE TABLE StringIds (id INTEGER, value TEXT);
                CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (demangledName INTEGER, gridX, gridY,
                    gridZ, blockX, blockY, blockZ, staticSharedMemory, dynamicSharedMemory);
                CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (bytes, copyKind);
                CREATE TABLE ENUM_CUDA_MEMCPY_OPER (id, name, label);
                INSERT INTO StringIds VALUES (1, 'void (anonymous namespace)::k(int)');
                INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (1,4,1,1,256,1,1,0,0),(1,4,1,1,256,1,1,0,0);
                INSERT INTO ENUM_CUDA_MEMCPY_OPER VALUES (1,'H','Host-to-Device');
                INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (1024, 1);
            """)
            c.commit()
            c.close()
            s = runner.launch_structure(db)
        self.assertEqual(s["kernels"], [["void k(int)", 4, 1, 1, 256, 1, 1, 0, 0, 2]])
        self.assertEqual(s["copies"], [["Host-to-Device", 1024, 1]])
        self.assertEqual(s["memsets"], [])
        self.assertEqual(runner.structure_difference(s, s), "")
        other = json.loads(json.dumps(s))
        other["copies"] = [["Host-to-Device", 1024, 2]]
        self.assertIn("copies", runner.structure_difference(s, other))


if __name__ == "__main__":
    unittest.main()
