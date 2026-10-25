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
                                        "large": {"vertices": 1048576},
                                        "timing_scope": {"application": "upload + compute + download"}}))
        self.contract_sha = hashlib.sha256(contract.read_bytes()).hexdigest()
        self.manifest = {"folder": "rtx3090", "label": "RTX 3090 (sm_86, 24 GB)",
                         "toolkit": "CUDA 13.3", "suite_version": "v1",
                         "applications": {a: {"check": None, "bench_direct": None,
                                              "bench_graph": None} for a in self.m.APPS}}

    def write_record(self, name, **fields):
        record = {"schema": "loom_private.result/1", "identity": "loom-private/pagerank",
                  "contract_sha256": self.contract_sha, "preset": "large", "overrides": {},
                  "action": "check", "launch": "both", "status": "pass",
                  "dimensions": {"vertices": 1048576, "levels": 16},
                  "device": {"name": "NVIDIA GeForce RTX 3090", "cc": "8.6"},
                  "outputs_agree": True, "timing": None,
                  "checks": [{"launch": "direct", "pass": True},
                             {"launch": "graph", "pass": True}]}
        record.update(fields)
        path = self.m.REPO / "rtx3090" / name
        path.write_text(json.dumps(record))
        return name

    def select(self, key, value):
        self.manifest["applications"]["pagerank"][key] = value
        (self.m.HERE / "inputs/rtx3090.json").write_text(json.dumps(self.manifest))
        return self.m.load_gpu("rtx3090")

    def test_unselected_timing_is_pending(self):
        _, rows = self.select("check", None)
        self.assertEqual(self.m.timing(rows[0]["bench_direct"], "application"), "pending")

    def test_check_record_never_becomes_a_timing(self):
        manifest, rows = self.select("check", self.write_record("check.json"))
        text = self.m.markdown([(manifest, rows)])
        timing_row = next(line for line in text.splitlines()
                          if line.startswith("| `loom-private/pagerank`") and "vertices" in line)
        self.assertEqual(timing_row.count("pending"), 4)
        self.assertIn("| pass | 1/1 pass | 1/1 pass | yes |", text)

    def test_check_record_cannot_be_selected_as_bench(self):
        with self.assertRaisesRegex(ValueError, "selected as bench_direct"):
            self.select("bench_direct", self.write_record("check.json"))

    def test_stale_contract_is_rejected(self):
        name = self.write_record("old.json", contract_sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "older contract|current contract"):
            self.select("check", name)

    def test_overridden_dimensions_are_rejected(self):
        name = self.write_record("small.json", overrides={"vertices": 1024})
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
        name = self.write_record("other-size.json", dimensions={"vertices": 1024})
        with self.assertRaisesRegex(ValueError, "dimension vertices=1024 differs"):
            self.select("check", name)

    def test_dimensions_without_an_integer_contract_value_are_not_compared(self):
        name = self.write_record("extra.json", dimensions={"vertices": 1048576, "levels": 3})
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

    def test_bench_record_fills_timing(self):
        timing = {"application": {"batch_per_iteration_us": 12.5},
                  "compute": {"batch_per_iteration_us": 10.0}}
        name = self.write_record("bench.json", action="bench", launch="graph",
                                 status="unvalidated_bench", timing=timing)
        _, rows = self.select("bench_graph", name)
        self.assertEqual(self.m.timing(rows[0]["bench_graph"], "application"), 12.5)


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
