"""Unit tests for dashboard_data.py. Run: python -m unittest discover -s tests"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dashboard_data as dd  # noqa: E402


class SavedOutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metrics = dd.load_metrics()
        cls.pred = dd.load_predictions()

    def test_row_count_and_ids(self):
        self.assertEqual(len(self.pred), int(self.metrics["Test rows"]))
        self.assertTrue(self.pred["LoanID"].is_unique)

    def test_confusion_matches_saved_metrics(self):
        self.assertTrue(dd.metrics_consistent(self.pred, self.metrics))

    def test_prediction_matches_threshold(self):
        expected = (self.pred["default_probability"] >= self.metrics["Threshold"]).astype(int)
        self.assertTrue((expected == self.pred["prediction"]).all())

    def test_risk_bands(self):
        p = self.pred["default_probability"]
        expected = pd.cut(p, [-1, 0.30 - 1e-12, 0.60 - 1e-12, 2], labels=dd.RISK_ORDER)
        self.assertTrue((expected.astype(str) == self.pred["risk_category"].astype(str)).all())

    def test_kpis(self):
        k = dd.compute_kpis(self.pred, self.metrics)
        self.assertEqual(k["borrowers"], 1001)
        self.assertAlmostEqual(k["actual_default_rate"], 118 / 1001)
        self.assertAlmostEqual(k["predicted_default_rate"], 166 / 1001)

    def test_risk_summary_totals(self):
        s = dd.risk_summary(self.pred)
        self.assertEqual(int(s["borrowers"].sum()), len(self.pred))
        self.assertEqual(int(s["actual_defaults"].sum()), int(self.pred["Default"].sum()))

    def test_filters_and_export(self):
        first = self.pred["LoanID"].iloc[0]
        self.assertEqual(len(dd.filter_borrowers(self.pred, first.lower())), 1)
        self.assertEqual(len(dd.filter_borrowers(self.pred, "no-such-id")), 0)
        high = dd.filter_borrowers(self.pred, "", ["High"], "Actual defaults")
        self.assertTrue((high["risk_category"] == "High").all() and (high["Default"] == 1).all())
        csv = dd.export_frame(dd.filter_borrowers(self.pred)).to_csv(index=False)
        self.assertEqual(len(pd.read_csv(__import__("io").StringIO(csv))), len(self.pred))


class InvalidInputTests(unittest.TestCase):
    def test_missing_file(self):
        with self.assertRaises(dd.DataError):
            dd.load_predictions(Path("/nonexistent/x.csv"))

    def test_missing_column_and_bad_values(self):
        with tempfile.TemporaryDirectory() as d:
            bad = Path(d) / "p.csv"
            bad.write_text("LoanID,Default\nA,0\n")
            with self.assertRaises(dd.DataError):
                dd.load_predictions(bad)
            bad.write_text("LoanID,Default,prediction,default_probability,risk_category\nA,0,0,1.5,Low\n")
            with self.assertRaises(dd.DataError):
                dd.load_predictions(bad)

    def test_bad_metrics_json(self):
        with tempfile.TemporaryDirectory() as d:
            m = Path(d) / "m.json"
            m.write_text("{not json")
            with self.assertRaises(dd.DataError):
                dd.load_metrics(m)
            m.write_text(json.dumps({"Precision": 1}))
            with self.assertRaises(dd.DataError):
                dd.load_metrics(m)


if __name__ == "__main__":
    unittest.main()
