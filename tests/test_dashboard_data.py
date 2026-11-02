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

    def test_all_saved_metrics_match_predictions(self):
        calculated = dd.classification_metrics(self.pred)
        for key, value in calculated.items():
            self.assertAlmostEqual(float(self.metrics[key]), float(value), places=9)

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

    def test_metric_mismatch_is_rejected(self):
        metrics = dict(self.metrics)
        metrics["F1 Score"] = 0.0
        self.assertFalse(dd.metrics_consistent(self.pred, metrics))


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
            bad.write_text("LoanID,Default,prediction,default_probability,risk_category\nA,0,2,0.5,Low\n")
            with self.assertRaises(dd.DataError):
                dd.load_predictions(bad)

    def test_duplicate_loan_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "p.csv"
            path.write_text(
                "LoanID,Default,prediction,default_probability,risk_category\n"
                "A,0,0,0.2,Low\nA,0,0,0.2,Low\n"
            )
            with self.assertRaisesRegex(dd.DataError, "LoanID values must be unique"):
                dd.load_predictions(path)

    def test_bad_metrics_json(self):
        with tempfile.TemporaryDirectory() as d:
            m = Path(d) / "m.json"
            m.write_text("{not json")
            with self.assertRaises(dd.DataError):
                dd.load_metrics(m)
            m.write_text(json.dumps({"Precision": 1}))
            with self.assertRaises(dd.DataError):
                dd.load_metrics(m)

    def test_bad_comparison_csv(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "comparison.csv"
            path.write_text("Model,Threshold\nLogistic Regression,0.5\n")
            with self.assertRaisesRegex(dd.DataError, "missing columns"):
                dd.load_comparison(path)


if __name__ == "__main__":
    unittest.main()
