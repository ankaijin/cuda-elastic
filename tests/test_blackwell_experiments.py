import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BW = REPO / "experiments" / "blackwell"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def experiment_dirs():
    return sorted(p.parent for p in BW.glob("*/experiment.json"))


class ExperimentConfigTests(unittest.TestCase):
    def test_every_experiment_has_one_control_first(self):
        self.assertTrue(experiment_dirs())
        for directory in experiment_dirs():
            config = json.loads((directory / "experiment.json").read_text())
            roles = [case["role"] for case in config["cases"]]
            self.assertEqual(roles[0], "control", directory.name)
            self.assertEqual(roles.count("control"), 1, directory.name)
            self.assertEqual(config["experiment"], directory.name)
            self.assertTrue(config["question"])

    def test_cases_match_the_makefile(self):
        for directory in experiment_dirs():
            config = json.loads((directory / "experiment.json").read_text())
            printed = subprocess.run(["make", "-s", "-C", str(directory), "print-cases"],
                                     capture_output=True, text=True,
                                     env=dict(os.environ, GPU="test"))
            self.assertEqual(printed.returncode, 0, printed.stderr)
            self.assertEqual(printed.stdout.split(), [c["name"] for c in config["cases"]])

    def test_committed_diffs_are_current(self):
        diffs = sorted(BW.glob("*/*.diff"))
        self.assertTrue(diffs)
        for path in diffs:
            lines = path.read_text().splitlines()
            old = lines[0].removeprefix("--- ").split("\t")[0]
            new = lines[1].removeprefix("+++ ").split("\t")[0]
            result = subprocess.run(["diff", "-u", "--label", old, "--label", new, old, new],
                                    cwd=REPO, capture_output=True, text=True)
            self.assertEqual(result.returncode, 1, f"{path.name}: the sources do not differ")
            self.assertEqual(result.stdout, path.read_text(),
                             f"{path.relative_to(REPO)} is stale; regenerate it with diff -u")

    def test_production_sources_are_not_copied_unchanged(self):
        # A candidate must differ from production only where its .diff says.
        for path in BW.glob("*/*.diff"):
            self.assertIn("@@", path.read_text())


class RunnerTests(unittest.TestCase):
    def test_unknown_case_is_a_usage_error(self):
        result = subprocess.run([sys.executable, str(BW / "run.py"), "packed", "--case",
                                 "fastest"], capture_output=True, text=True,
                                env=dict(os.environ, GPU="rtx5090"))
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("unknown case", result.stderr)

    def test_unknown_experiment_is_rejected(self):
        result = subprocess.run([sys.executable, str(BW / "run.py"), "fp8"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)

    def test_paired_ratios_compare_the_same_pair(self):
        run = load(BW / "run.py", "bw_run")

        def record(case, role, pair, launch, us):
            timing = {"batch_per_iteration_us": us, "single_median_us": us}
            return {"case": case, "role": role, "pair": pair, "launch": launch,
                    "result": {"timing": {"application": timing, "compute": timing}}}

        records = [record("control", "control", 1, "graph", 100.0),
                   record("fast", "candidate", 1, "graph", 90.0),
                   record("control", "control", 2, "graph", 200.0),
                   record("fast", "candidate", 2, "graph", 210.0),
                   record("control", "control", 3, "graph", 100.0),
                   record("fast", "candidate", 3, "graph", 80.0)]
        table = run.summarize_bench(records)["graph/compute/batch_per_iteration_us"]
        self.assertEqual(table["control"]["median"], 100.0)
        ratio = table["fast"]["paired_ratio"]
        self.assertAlmostEqual(ratio["median"], 0.9)
        self.assertAlmostEqual(ratio["max"], 1.05)
        self.assertEqual(ratio["below_one"], 2)
        self.assertNotIn("paired_ratio", table["control"])
        self.assertNotIn("direct/compute/batch_per_iteration_us", run.summarize_bench(records))


class BuildInfoTests(unittest.TestCase):
    def test_build_record_names_command_sources_and_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source, spec, binary = directory / "a.cu", directory / "spec.json", directory / "bin"
            source.write_text("int x;\n")
            spec.write_text("{}\n")
            command = [sys.executable, "-c",
                       f"open({str(binary)!r}, 'w').write('binary')", f"-I{directory}"]
            result = subprocess.run(
                [sys.executable, str(BW / "tools/build_info.py"), "--output", str(binary),
                 "--case", "c", "--experiment", "e", "--arch", "sm_120f", "--spec", str(spec),
                 "--source", str(source), "--", *command], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            info = json.loads((directory / "bin.build.json").read_text())
            self.assertEqual(info["arch"], "sm_120f")
            self.assertEqual(info["command"], command)
            self.assertEqual(info["binary_sha256"], hashlib.sha256(b"binary").hexdigest())
            self.assertEqual(info["spec_sha256"], hashlib.sha256(b"{}\n").hexdigest())
            self.assertIn(str(source), info["sources"])
            self.assertIn(str(directory.resolve()), info["external_includes"])

    def test_failed_compile_writes_no_record(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            spec = directory / "spec.json"
            spec.write_text("{}\n")
            result = subprocess.run(
                [sys.executable, str(BW / "tools/build_info.py"), "--output",
                 str(directory / "bin"), "--case", "c", "--experiment", "e", "--arch", "sm_120",
                 "--spec", str(spec), "--", "false"], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((directory / "bin.build.json").exists())


if __name__ == "__main__":
    unittest.main()
