import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SUITE = REPO / "applications" / "loom_private"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LegacyFreezeTests(unittest.TestCase):
    def test_legacy_portfolio_matches_baseline(self):
        result = subprocess.run([sys.executable, str(SUITE / "tools/legacy_freeze.py"), "verify"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_freeze_covers_legacy_reports_and_inputs(self):
        files = json.loads((SUITE / "legacy-freeze.json").read_text())["files"]
        for path in ("results/README.md", "results/cuda_performance.xlsx",
                     "results/inputs/rtx3090.json", "pagerank/gapbs_pagerank_cuda_graph.cu"):
            self.assertIn(path, files)


class RunSuiteArgumentTests(unittest.TestCase):
    def run_suite(self, *args):
        return subprocess.run([sys.executable, str(SUITE / "run_suite.py"), *args],
                              capture_output=True, text=True)

    def test_unknown_application_is_rejected(self):
        result = self.run_suite("legacy-pagerank")
        self.assertEqual(result.returncode, 2)
        self.assertIn("unknown application", result.stderr)

    def test_bench_requires_explicit_launch(self):
        self.assertEqual(self.run_suite("--action", "bench", "pagerank").returncode, 2)

    def test_check_runs_both_launch_modes_only(self):
        self.assertEqual(self.run_suite("--launch", "graph", "pagerank").returncode, 2)

    def test_repeat_applies_to_bench_only(self):
        self.assertEqual(self.run_suite("--repeat", "3", "pagerank").returncode, 2)
        self.assertEqual(self.run_suite("--action", "bench", "--launch", "graph", "--repeat", "0",
                                        "pagerank").returncode, 2)

    def test_unknown_variant_is_rejected(self):
        self.assertEqual(self.run_suite("--variant", "fast", "pagerank").returncode, 2)


class ReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = load(REPO / "results/loom_private/build_report.py", "lp_report")

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        saved = {k: getattr(self.m, k) for k in ("REPO", "HERE", "SUITE")}
        self.addCleanup(lambda: [setattr(self.m, k, v) for k, v in saved.items()])
        self.m.REPO, self.m.HERE, self.m.SUITE = (
            root, root / "results/loom_private", root / "applications/loom_private")
        (self.m.HERE / "inputs").mkdir(parents=True)
        (root / "rtx3090").mkdir()
        contract = self.m.SUITE / "pagerank" / "contract.json"
        contract.parent.mkdir(parents=True)
        contract.write_text(json.dumps({"identity": "loom-private/pagerank",
                                        "large": {"nodes": 1048576, "edges": 2097151,
                                                  "iterations": 20},
                                        "timing_scope": {"application": "upload + compute + download"}}))
        self.contract_sha = hashlib.sha256(contract.read_bytes()).hexdigest()
        self.manifest = {"folder": "rtx3090", "label": "RTX 3090 (sm_86, 24 GB)",
                         "toolkit": "CUDA 13.3", "suite_version": "v1",
                         "source_revision": "296aa928542ce31d456c1b37ec473064f8bdc545",
                         "applications": {a: {"check": None, "check_optimized": None,
                                              "bench_direct": None, "bench_graph": None,
                                              "bench_optimized": None,
                                              "profiles": {kind: dict.fromkeys(
                                                  ("direct", "graph", "optimized"))
                                                  for kind in ("nsys", "ncu")}}
                                          for a in self.m.APPS}}

    def write_record(self, name, **fields):
        record = {"schema": "loom_private.result/1", "identity": "loom-private/pagerank",
                  "contract_sha256": self.contract_sha, "preset": "large", "overrides": {},
                  "action": "check", "launch": "both", "status": "pass",
                  "dimensions": {"nodes": 1048576, "levels": 16},
                  "device": {"name": "NVIDIA GeForce RTX 3090", "cc": "8.6"},
                  "outputs_agree": True, "timing": None,
                  "checks": [{"launch": "direct", "pass": True},
                             {"launch": "graph", "pass": True}]}
        record.update(fields)
        path = self.m.REPO / "rtx3090" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record))
        return name

    def write_bench(self, name, us, launch="graph", variant="baseline", single=None):
        timing = {scope: {"batch_total_ms": us * 20 / 1000, "batch_per_iteration_us": us * f,
                          "single_median_us": (single or us * 1.1) * f}
                  for scope, f in (("application", 1.0), ("compute", 0.8))}
        return self.write_record(name, action="bench", launch=launch, variant=variant,
                                 iterations=20, status="unvalidated_bench", timing=timing)

    def write_profile(self, stem, kernels=None, ncu=None):
        """An nsys report and its kernel summary, or an ncu report and its raw export,
        under loom_private/pagerank/profiles/."""
        folder = self.m.REPO / "rtx3090/loom_private/pagerank/profiles"
        folder.mkdir(parents=True, exist_ok=True)
        if kernels is not None:
            (folder / f"{stem}.nsys-rep").write_bytes(b"nsys")
            lines = ["Time (%),Total Time (ns),Instances,Avg (ns),Name"]
            lines += [f'1,{total},{count},1,"{name}"' for name, total, count in kernels]
            (folder / f"{stem}_cuda_gpu_kern_sum.csv").write_text("\n".join(lines) + "\n")
            return f"loom_private/pagerank/profiles/{stem}.nsys-rep"
        (folder / f"{stem}.ncu-rep").write_bytes(b"ncu")
        (folder / f"{stem}-ncu-raw.csv").write_text(ncu)
        return f"loom_private/pagerank/profiles/{stem}.ncu-rep"

    def select_profile(self, kind, version, value):
        self.manifest["applications"]["pagerank"]["profiles"][kind][version] = value
        (self.m.HERE / "inputs/rtx3090.json").write_text(json.dumps(self.manifest))
        return self.m.load_gpu("rtx3090")

    def select(self, key, value):
        self.manifest["applications"]["pagerank"][key] = value
        (self.m.HERE / "inputs/rtx3090.json").write_text(json.dumps(self.manifest))
        return self.m.load_gpu("rtx3090")

    def test_unselected_timing_is_pending(self):
        _, rows = self.select("check", None)
        self.assertEqual(self.m.timing(rows[0]["bench_direct"], "application"), "pending")
        self.assertEqual(self.m.timing(rows[0]["bench_optimized"], "application"), "pending")

    def test_check_record_never_becomes_a_timing(self):
        manifest, rows = self.select("check", self.write_record("check.json"))
        text = self.m.markdown([(manifest, rows)])
        pagerank = [line for line in text.splitlines() if line.startswith("| PageRank |")]
        # Workloads, two timing tables, the speedup table, correctness.
        self.assertEqual(len(pagerank), 5)
        self.assertIn("1,048,576 vertices, 2,097,151 edges, 20 ordered", pagerank[0])
        self.assertEqual([line.count("pending") for line in pagerank[1:4]], [3, 3, 2])
        self.assertEqual(pagerank[4], "| PageRank | [pass](../../rtx3090/check.json) | pending |")

    def test_disagreeing_outputs_fail_the_check_summary(self):
        _, rows = self.select("check", self.write_record("split.json", outputs_agree=False))
        self.assertEqual(self.m.check_summary(rows[0]["check"]), "[FAIL](../../rtx3090/split.json)")

    def test_environment_comes_from_run_metadata(self):
        run = "20261026-000000-mono-12345678"
        meta = self.m.REPO / "rtx3090/runs" / run / "metadata.json"
        meta.parent.mkdir(parents=True)
        meta.write_text(json.dumps({
            "run_id": run, "host": "mono", "source_commit": "296aa928542ce31d",
            "gpu": {"driver": "610.43.02"},
            "installed_tools": {"nvcc": "Cuda compilation tools, release 13.3, V13.3.73"}}))
        manifest, rows = self.select("check", self.write_record(f"{run}-check.json"))
        self.assertEqual(self.m.environment(manifest, rows),
                         ["RTX 3090 (sm_86, 24 GB)", "mono", "610.43.02", "CUDA 13.3", "296aa92"])

    def test_selected_runs_need_a_source_revision(self):
        del self.manifest["source_revision"]
        manifest, rows = self.select("check", self.write_record("check.json"))
        with self.assertRaisesRegex(ValueError, "source_revision"):
            self.m.environment(manifest, rows)

    def test_environment_without_selected_runs_is_pending(self):
        manifest, rows = self.select("check", None)
        self.assertEqual(self.m.environment(manifest, rows)[1:], ["pending"] * 4)

    def test_check_record_cannot_be_selected_as_bench(self):
        with self.assertRaisesRegex(ValueError, "selected as bench_direct"):
            self.select("bench_direct", self.write_record("check.json"))

    def test_stale_contract_is_rejected(self):
        name = self.write_record("old.json", contract_sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "older contract|current contract"):
            self.select("check", name)

    def test_overridden_dimensions_are_rejected(self):
        name = self.write_record("small.json", overrides={"nodes": 1024})
        with self.assertRaisesRegex(ValueError, "large-preset"):
            self.select("check", name)

    def test_unsafe_or_missing_paths_fail_visibly(self):
        for value in ("/etc/passwd", "../rtx4090/x.json", "*.json", "missing.json"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.select("check", value)

    def test_check_record_must_cover_both_launch_modes(self):
        name = self.write_record("direct-only.json", launch="direct")
        with self.assertRaisesRegex(ValueError, "selected as check"):
            self.select("check", name)

    def test_dimensions_must_match_the_contract(self):
        name = self.write_record("other-size.json", dimensions={"nodes": 1024})
        with self.assertRaisesRegex(ValueError, "dimension nodes=1024 differs"):
            self.select("check", name)

    def test_dimensions_without_an_integer_contract_value_are_not_compared(self):
        name = self.write_record("extra.json", dimensions={"nodes": 1048576, "levels": 3})
        _, rows = self.select("check", name)
        self.assertEqual(rows[0]["check"]["status"], "pass")

    def test_record_from_another_gpu_is_rejected(self):
        name = self.write_record("4090.json", device={"name": "NVIDIA GeForce RTX 4090"})
        with self.assertRaisesRegex(ValueError, "not the rtx3090 folder's GPU"):
            self.select("check", name)

    def test_device_name_matching(self):
        self.assertTrue(self.m.device_matches("rtx3090", "NVIDIA GeForce RTX 3090"))
        self.assertTrue(self.m.device_matches("rtx5090", "NVIDIA GeForce RTX 5090"))
        self.assertFalse(self.m.device_matches("rtx3090", "NVIDIA GeForce RTX 3090 Ti"))
        self.assertFalse(self.m.device_matches("rtx3090", "NVIDIA GeForce RTX 4090"))
        self.assertFalse(self.m.device_matches("rtx3090", None))

    def test_size_text_leaves_out_checksums(self):
        spec = {"large": {"length": 1048576, "expected_checksum": 1649265868800,
                          "output_checksum": 7, "element_type": "int32", "flag": True}}
        self.assertEqual(self.m.size_text(spec), "length=1,048,576")

    def test_repetitions_report_the_median_and_link_its_record(self):
        names = [self.write_bench(f"r{i}.json", us) for i, us in enumerate((30.0, 10.0, 20.0), 1)]
        manifest, rows = self.select("bench_graph", names)
        self.assertEqual(self.m.timing(rows[0]["bench_graph"], "application"), 20.0)
        self.assertEqual(self.m.median_record(rows[0]["bench_graph"], "application")["_path"],
                         "rtx3090/r3.json")
        self.assertIn("then the median of 3 recorded runs", self.m.markdown([(manifest, rows)]))

    def test_empty_repetition_list_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "empty list"):
            self.select("bench_graph", [])

    def test_optimized_selection_needs_an_optimized_graph_record(self):
        with self.assertRaisesRegex(ValueError, "selected as bench_optimized"):
            self.select("bench_optimized", [self.write_bench("base.json", 10.0)])
        with self.assertRaisesRegex(ValueError, "selected as bench_optimized"):
            self.select("bench_optimized", [self.write_bench("direct.json", 10.0, "direct",
                                                             "optimized")])
        with self.assertRaisesRegex(ValueError, "selected as check_optimized"):
            self.select("check_optimized", self.write_record("check.json"))
        self.manifest["applications"]["pagerank"]["check_optimized"] = None
        _, rows = self.select("bench_optimized", [self.write_bench("opt.json", 5.0,
                                                                   variant="optimized")])
        self.assertEqual(self.m.timing(rows[0]["bench_optimized"], "compute"), 4.0)

    def test_optimized_speedup_is_graph_over_optimized(self):
        self.select("bench_graph", [self.write_bench("graph.json", 30.0)])
        manifest, rows = self.select("bench_optimized", [self.write_bench(
            "opt.json", 10.0, variant="optimized")])
        text = self.m.markdown([(manifest, rows)])
        self.assertIn("| PageRank | 3.0× | 3.0× |", text)

    def test_run_metadata_follows_every_file_name(self):
        run = "20261026-000000-kilia.cs.ucla.edu-1234abcd"
        meta = self.m.REPO / "rtx3090/runs" / run / "metadata.json"
        meta.parent.mkdir(parents=True)
        meta.write_text(json.dumps({"run_id": run}))
        for name in (f"{run}-check.json", f"{run}-check-optimized.json",
                     f"{run}-bench-graph-optimized-r2.json",
                     f"{run}-baseline-direct.nsys-rep", f"{run}-optimized-graph.ncu-repz"):
            with self.subTest(name=name):
                self.assertEqual(self.m.run_metadata(f"rtx3090/x/{name}")["run_id"], run)
        self.assertIsNone(self.m.run_metadata("rtx3090/check.json"))

    def test_profiles_must_match_their_application_and_version(self):
        path = self.write_profile("run-baseline-graph", kernels=[("k", 400, 4)])
        with self.assertRaisesRegex(ValueError, "not a nsys report of pagerank baseline-direct"):
            self.select_profile("nsys", "direct", path)
        self.manifest["applications"]["pagerank"]["profiles"]["nsys"]["direct"] = None
        with self.assertRaisesRegex(ValueError, "not a ncu report"):
            self.select_profile("ncu", "graph", path)
        self.manifest["applications"]["pagerank"]["profiles"]["ncu"]["graph"] = None
        (self.m.REPO / "rtx3090/loom_private/pagerank/profiles/"
         "run-baseline-graph_cuda_gpu_kern_sum.csv").unlink()
        with self.assertRaisesRegex(ValueError, "export run-baseline-graph_cuda_gpu_kern_sum.csv"):
            self.select_profile("nsys", "graph", path)

    def test_kernel_time_per_invocation_leaves_out_setup_kernels(self):
        path = self.write_profile("run-optimized-graph", kernels=[
            ("<unnamed>::update<128>(const int *, float *)", 8000, 8),
            ("void (anonymous namespace)::init(int)", 5000, 1)])
        _, rows = self.select_profile("nsys", "optimized", path)
        kernels, setup = self.m.nsys_kernels(rows[0]["profiles"][("nsys", "optimized")])
        self.assertEqual(kernels, [("update<128>", 2.0, 2)])
        self.assertEqual(setup, ["init"])

    def test_bench_record_fills_timing(self):
        timing = {"application": {"batch_per_iteration_us": 12.5},
                  "compute": {"batch_per_iteration_us": 10.0}}
        name = self.write_record("bench.json", action="bench", launch="graph",
                                 status="unvalidated_bench", timing=timing)
        _, rows = self.select("bench_graph", name)
        self.assertEqual(self.m.timing(rows[0]["bench_graph"], "application"), 12.5)
        self.assertEqual(len(rows[0]["bench_graph"]), 1)


    def test_workbook_matches_the_legacy_layout(self):
        timing = {scope: {"batch_total_ms": 0.25, "batch_per_iteration_us": us,
                          "single_median_us": us * 1.1}
                  for scope, us in (("application", 12.5), ("compute", 10.0))}
        for launch in ("direct", "graph"):
            self.select(f"bench_{launch}", self.write_record(
                f"bench-{launch}.json", action="bench", launch=launch, iterations=20,
                status="unvalidated_bench", timing=timing))
        self.select("bench_optimized", [self.write_bench("opt.json", 5.0, variant="optimized",
                                                         single=6.0)])
        self.select_profile("nsys", "graph", self.write_profile(
            "run-baseline-graph", kernels=[("k", 400, 4)]))
        raw = ('"ID","Kernel Name","gpu__time_duration.sum","launch__grid_size",'
               '"launch__block_size","sm__throughput.avg.pct_of_peak_sustained_elapsed"\n'
               '"","","usecond","","","%"\n'
               '"0","update(int)","10.5","64","256","40"\n'
               '"1","update(int)","11.5","64","256","60"\n')
        self.select_profile("ncu", "graph", self.write_profile("run-baseline-graph", ncu=raw))
        manifest, rows = self.select("check", self.write_record("check.json"))
        book = self.m.workbook([(manifest, rows)])
        self.assertEqual(book.sheetnames, ["RTX 3090", "Latency RTX 3090", "Kernels RTX 3090",
                                           "Sources", "GPU analysis", "ncu metrics (RTX 3090)",
                                           "NCU launches RTX 3090", "Correctness",
                                           "Environment", "Contracts"])
        ws = book["RTX 3090"]
        self.assertEqual(ws["A1"].value, "RTX 3090")
        self.assertEqual(ws["A4"].value, "Workload")
        self.assertEqual(ws["A4"].fill.fgColor.rgb, "001F3864")
        self.assertEqual(ws.freeze_panes, "C5")
        # PageRank: an Application row and a Kernel-only row.
        self.assertEqual([ws.cell(r, 2).value for r in (5, 6)],
                         ["Application (end to end)", "Kernel only (no transfers)"])
        self.assertEqual(ws["E5"].value, 12.5)
        self.assertEqual(ws["E5"].number_format, "#,##0.00")
        self.assertTrue(ws["E5"].hyperlink.target.endswith("/rtx3090/bench-direct.json"))
        self.assertEqual(ws["G5"].value, 5.0)
        self.assertTrue(ws["G5"].hyperlink.target.endswith("/rtx3090/opt.json"))
        self.assertEqual(ws["H5"].value, 1.0)
        self.assertEqual(ws["H5"].number_format, '0.00"x"')
        self.assertEqual(ws["I5"].value, 2.5)  # Graph / Optimized
        self.assertEqual(ws.cell(7, 7).value, "pending")  # attention: nothing selected
        latency = book["Latency RTX 3090"]
        self.assertEqual(latency["I5"].value, 13.75)  # single-iteration median
        self.assertEqual(latency["F5"].value, 1)  # runs
        self.assertEqual(latency["L7"].value, 2.292)  # Graph single / Optimized single
        self.assertEqual(latency.max_row, 4 + len(self.m.APPS) * 2 * 3)
        kernels = book["Kernels RTX 3090"]
        self.assertEqual(kernels["F5"].value, 0.1)  # 400 ns over four invocations
        self.assertEqual(kernels["E5"].value, "pending")
        metrics = book["ncu metrics (RTX 3090)"]
        self.assertEqual([metrics.cell(5, c).value for c in (2, 3, 4, 5, 8)],
                         ["Graph", "update", 2, 11.0, 50.0])
        self.assertEqual(book["NCU launches RTX 3090"].max_row, 6)
        self.assertEqual(book["GPU analysis"]["A5"].value, "pending")


class ContractTests(unittest.TestCase):
    """Every application contract against the suite README schema and assets.json."""
    APPS = ["pagerank", "attention", "decoder", "anomaly", "vecadd_memory", "vecadd_product"]
    # The fields listed under "Workload contracts" in applications/loom_private/README.md.
    FIELDS = ["identity", "suite_version", "extension_of_upstream", "upstream", "large",
              "data_generation", "outputs", "tolerances", "timing_scope", "assets",
              "reference_only_upstream_configurations"]

    def test_contracts_have_every_schema_field(self):
        readme = (SUITE / "README.md").read_text()
        for field in self.FIELDS:
            self.assertIn(f"`{field}`", readme)
        for app in self.APPS:
            contract = json.loads((SUITE / app / "contract.json").read_text())
            with self.subTest(app=app):
                self.assertEqual([f for f in self.FIELDS if f not in contract], [])
                self.assertEqual(contract["suite_version"], "v1")

    def test_contract_assets_match_registered_hashes(self):
        registered = json.loads((SUITE / "assets.json").read_text())["assets"]
        for app in self.APPS:
            assets = json.loads((SUITE / app / "contract.json").read_text())["assets"]
            for name, digest in assets.items():
                with self.subTest(app=app, asset=name):
                    self.assertIn(name, registered)
                    self.assertEqual(digest, registered[name]["sha256"])


class ReportFreshnessTests(unittest.TestCase):
    # NOTE: this fails whenever a contract, the report generator, or a selected
    # record changes until results/loom_private/README.md is regenerated with
    # `python3 results/loom_private/build_report.py`. That is its purpose.
    def test_committed_report_is_current(self):
        report = load(REPO / "results/loom_private/build_report.py", "lp_report_fresh")
        expected = report.markdown([report.load_gpu(folder) for folder in report.GPUS]) + "\n"
        self.assertEqual((REPO / "results/loom_private/README.md").read_text(), expected,
                         "results/loom_private/README.md is stale: rerun "
                         "results/loom_private/build_report.py")


if __name__ == "__main__":
    unittest.main()
