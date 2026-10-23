import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


def load_builder():
    spec = importlib.util.spec_from_file_location(
        "builder", Path(__file__).resolve().parents[1] / "results/build_spreadsheet.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class InputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = load_builder()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.old_repo = self.m.REPO
        self.m.REPO = Path(self.temp.name)
        self.addCleanup(setattr, self.m, "REPO", self.old_repo)
        (self.m.REPO / "rtx4090").mkdir()

    def gpu(self, path):
        return self.m.GPU("rtx4090", "RTX 4090", "test", {}, {"x": path}, {})

    def test_selection_stays_pinned_when_newer_result_arrives(self):
        root = self.m.REPO / "rtx4090"
        (root / "old.log").write_text("selected")
        (root / "new.log").write_text("new experiment")
        self.assertEqual(self.gpu("old.log").file("x"), "rtx4090/old.log")

    def test_unselected_input_is_explicitly_pending(self):
        self.assertIsNone(self.gpu(None).file("x"))

    def test_missing_selected_input_is_an_error(self):
        with self.assertRaises(FileNotFoundError):
            self.gpu("missing.log").file("x")

    def test_unused_selected_input_is_also_validated(self):
        gpu = self.m.GPU("rtx5090", "RTX 5090", "test", None,
                         {"loom_benchmark_txt": "missing-selected.txt"}, {})
        with self.assertRaises(FileNotFoundError):
            self.m.validate_inputs([gpu])

    def test_gpu_folder_cannot_redirect_to_another_gpu(self):
        with self.assertRaises(ValueError):
            self.m.GPU("rtx5090/../rtx4090", "RTX 5090", "test", None, {}, {})

    def test_manifest_folder_must_match_its_filename(self):
        folder = self.m.REPO / "results/inputs"
        folder.mkdir(parents=True)
        for name in ["rtx3090", "rtx4090", "rtx5090"]:
            data = dict(folder=name, label=name, toolkit="test", specs=None, files={}, notes={})
            if name == "rtx5090":
                data["folder"] = "rtx4090"
            (folder / f"{name}.json").write_text(json.dumps(data))
        with self.assertRaises(ValueError):
            self.m.load_gpus()

    def test_wildcards_and_escaping_paths_are_rejected(self):
        for name in ["*.log", "../other.log", "/tmp/elsewhere.log"]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.gpu(name).file("x")

    def test_symlink_cannot_escape_gpu_directory(self):
        (self.m.REPO / "outside.log").write_text("outside")
        (self.m.REPO / "rtx4090/link.log").symlink_to("../outside.log")
        with self.assertRaises(ValueError):
            self.gpu("link.log").file("x")

    def test_5090_has_no_measurements_or_borrowed_predictions(self):
        gpu = next((g for g in self.m.GPUS if g.folder == "rtx5090"), None)
        self.assertIsNotNone(gpu, "The combined report must include RTX 5090")
        self.assertTrue(all(value is None for value in gpu.files.values()))
        self.assertIsNone(self.m.predicted_speedup(object(), gpu))
        for row in self.m.gpu_rows(gpu):
            self.assertTrue(all(c.us is None for c in row[4:7]))


if __name__ == "__main__":
    unittest.main()
