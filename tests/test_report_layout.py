import importlib.util
import os
from pathlib import Path
import types
import unittest

from openpyxl import Workbook


class ReportLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / 'results/build_spreadsheet.py'
        source = Path(os.environ.get('REPORT_BUILDER_SOURCE', str(path))).read_text()
        cls.m = types.ModuleType('layout_builder')
        cls.m.__file__ = str(path)
        exec(compile(source, str(path), 'exec'), cls.m.__dict__)

    def test_summary_preserves_application_values_and_speedups(self):
        m = self.m
        row = ('mlperf-tiny VWW', 'Whole application', '16 bytes', m.TIMED,
               m.Cell(12.0), m.Cell(8.0), m.Cell(4.0), 'A measured example.')
        wb = Workbook()
        wb.remove(wb.active)
        sheet = m.build_summary(wb, m.RTX3090, [row])
        self.assertEqual([sheet.cell(5, c).value for c in range(5, 10)],
                         [12.0, 8.0, 4.0, 1.5, 2.0])
        self.assertEqual(sheet['J5'].comment.text, 'A measured example.')

    def test_5090_supporting_sheets_keep_missing_measurements_pending(self):
        m = self.m
        gpu = m.RTX5090
        rows = m.gpu_rows(gpu)
        wb = Workbook()
        wb.remove(wb.active)
        summary = m.build_summary(wb, gpu, rows, base_rows=rows)
        m.build_latency(wb, gpu, rows)
        m.build_summary(wb, gpu, rows, base_rows=rows, kernel=True)
        m.build_sources(wb, [(gpu, rows)])
        for row in range(5, 17):
            self.assertTrue(all(summary.cell(row, col).value == 'pending' for col in range(5, 8)))
            self.assertEqual(summary.cell(row, 10).value, 'pending')
        self.assertTrue({'RTX 5090', 'Latency RTX 5090', 'Kernels RTX 5090', 'Sources'} <= set(wb.sheetnames))

    def test_5090_analysis_does_not_claim_a_hardware_prediction(self):
        m = self.m
        gpu = m.RTX5090
        rows = m.gpu_rows(gpu)
        wb = Workbook()
        wb.remove(wb.active)
        m.build_gpu_analysis(wb, [(m.BASELINE, rows), (gpu, rows)], {gpu: [None] * len(rows)})
        sheet = wb['GPU analysis']
        for row in range(5, 23):
            self.assertIsNone(sheet.cell(row, 5).value)
            self.assertEqual(sheet.cell(row, 6).value, 'pending')
            self.assertIn('not enabled', sheet.cell(row, 7).value.lower())


if __name__ == '__main__':
    unittest.main()
