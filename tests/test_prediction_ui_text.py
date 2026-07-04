import unittest
from pathlib import Path


class PredictionUITextTest(unittest.TestCase):
    def test_legacy_optimizer_warning_and_candidate_backtest_section_exist(self):
        page_source = Path("pages/02_next_month_prediction.py").read_text()

        self.assertIn("Legacy Auto Backtesting & Parameter Optimization", page_source)
        self.assertIn("not calibrate Candidate Scoring V2.1", page_source)
        self.assertIn("Candidate Scoring V2.1 Backtest / Calibration", page_source)
        self.assertIn("Run Candidate Scoring Backtest", page_source)


if __name__ == "__main__":
    unittest.main()
