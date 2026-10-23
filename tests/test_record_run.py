import importlib.util
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class RecordTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name)
        subprocess.run(["git", "init", "-q", str(self.repo)], check=True)
        self.git("config", "user.name", "Test")
        self.git("config", "user.email", "test@example.invalid")
        self.git("commit", "--allow-empty", "-qm", "source")
        self.bin = self.repo / ".git" / "test-bin"
        self.bin.mkdir()
        gpu = self.bin / "nvidia-smi"
        gpu.write_text("#!/bin/sh\nprintf 'NVIDIA GeForce RTX 4090, GPU-test-device, 555.58.02\\n'\n")
        gpu.chmod(0o755)
        self.env = patch.dict(os.environ, {"PATH": str(self.bin) + os.pathsep + os.environ["PATH"]})
        self.env.start()
        self.addCleanup(self.env.stop)
        path = Path(__file__).resolve().parents[1] / "results/record_run.py"
        self.assertTrue(path.exists(), "The experiment recorder must exist")
        spec = importlib.util.spec_from_file_location("record_run", path)
        self.m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.m)

    def git(self, *args):
        return subprocess.check_output(["git", "-C", str(self.repo), *args], text=True).strip()

    def manifests(self):
        return list((self.repo / "rtx4090/runs").glob("*/metadata.json"))

    def record(self, check="true", command=None):
        return self.m.record_run(self.repo, "0", check, command or [sys.executable, "-c", "print('timing')"])

    def test_records_success_and_device_environment(self):
        command = [sys.executable, "-c", "import os; print(os.environ['CUDA_VISIBLE_DEVICES']); print(os.environ['TIMESTAMP']); print(os.environ['GPU_LOCK'])"]
        self.assertEqual(self.record(command=command), 0)
        manifest, = self.manifests()
        d = json.loads(manifest.read_text())
        self.assertEqual(d["source_commit"], self.git("rev-parse", "HEAD"))
        self.assertEqual(d["gpu"]["uuid"], "GPU-test-device")
        self.assertEqual(d["validation"]["exit_code"], 0)
        self.assertEqual(d["measurement"]["exit_code"], 0)
        self.assertEqual(d["status"], "completed")
        log = manifest.with_name("measurement.log").read_text()
        self.assertIn("GPU-test-device", log)
        self.assertIn(manifest.parent.name, log)
        self.assertIn("/tmp/cuda-elastic-gpu.lock", log)

    def test_failed_validation_prevents_measurement_and_is_recorded(self):
        marker = self.repo / "should-not-exist"
        self.assertEqual(self.record(check="exit 7", command=["touch", str(marker)]), 7)
        self.assertFalse(marker.exists())
        manifest, = self.manifests()
        d = json.loads(manifest.read_text())
        self.assertEqual(d["status"], "validation_failed")
        self.assertIsNone(d["measurement"]["exit_code"])

    def test_measurement_failure_is_recorded(self):
        self.assertEqual(self.record(command=[sys.executable, "-c", "raise SystemExit(9)"]), 9)
        manifest, = self.manifests()
        d = json.loads(manifest.read_text())
        self.assertEqual(d["status"], "measurement_failed")
        self.assertEqual(d["measurement"]["exit_code"], 9)

    def test_dirty_source_is_rejected_before_validation(self):
        (self.repo / "new-source.cu").write_text("uncommitted")
        with self.assertRaisesRegex(ValueError, "clean"):
            self.record()
        self.assertFalse(self.manifests())

    def test_run_ids_do_not_collide(self):
        self.assertEqual(self.record(), 0)
        self.git("add", "rtx4090")
        self.git("commit", "-qm", "first result")
        self.assertEqual(self.record(), 0)
        records = self.manifests()
        self.assertEqual(len(records), 2)
        self.assertNotEqual(records[0].parent.name, records[1].parent.name)

    def test_workload_environment_and_validated_binary_are_recorded(self):
        check = "mkdir -p rtx4090/workload/build; printf '\\177ELFpayload' > rtx4090/workload/build/program; chmod +x rtx4090/workload/build/program"
        with patch.dict(os.environ, {"PAGERANK_ROTATE": "5", "NVCCFLAGS": "-O2"}):
            self.assertEqual(self.record(check=check), 0)
        manifest, = self.manifests()
        data = json.loads(manifest.read_text())
        self.assertEqual(data["environment"]["PAGERANK_ROTATE"], "5")
        self.assertEqual(data["environment"]["NVCCFLAGS"], "-O2")
        self.assertEqual(data["binary_sha256"]["rtx4090/workload/build/program"],
                         hashlib.sha256(b"\x7fELFpayload").hexdigest())

    def test_validation_source_change_blocks_measurement(self):
        marker = self.repo / "should-not-exist"
        self.assertEqual(self.record(check="touch changed.cu", command=["touch", str(marker)]), 2)
        self.assertFalse(marker.exists())
        manifest, = self.manifests()
        data = json.loads(manifest.read_text())
        self.assertEqual(data["status"], "source_changed")


if __name__ == "__main__":
    unittest.main()
