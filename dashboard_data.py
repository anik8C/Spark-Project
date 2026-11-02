"""Pure-pandas helpers for the dashboard (no Streamlit import, so they are unit-testable)."""

import json
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
TABLES = ROOT / "outputs" / "tables"
METRICS_PATH = TABLES / "final_test_metrics.json"
PREDICTIONS_PATH = TABLES / "borrower_predictions.csv"
COMPARISON_PATH = TABLES / "validation_model_comparison.csv"

RISK_ORDER = ["Low", "Medium", "High"]
RISK_COLORS = {"Low": "#2E8B57", "Medium": "#E0A100", "High": "#C0392B"}
RISK_BANDS = {"Low": "< 0.30", "Medium": "0.30 to < 0.60", "High": ">= 0.60"}
REQUIRED_PREDICTION_COLUMNS = ["LoanID", "Default", "prediction", "default_probability", "risk_category"]
REQUIRED_METRIC_KEYS = ["Selected model", "Threshold", "Accuracy", "Precision", "Recall",
                        "F1 Score", "ROC-AUC", "TN", "FP", "FN", "TP", "Test rows"]


class DataError(Exception):
    """Raised with a user-readable message when saved outputs are missing or invalid."""


def load_metrics(path: Path = METRICS_PATH) -> dict:
    if not path.exists():
        raise DataError(f"Missing file: {path.name}")
    try:
        metrics = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DataError(f"{path.name} is not valid JSON ({exc.msg}).") from exc
    missing = [key for key in REQUIRED_METRIC_KEYS if key not in metrics]
    if missing:
        raise DataError(f"{path.name} is missing keys: {', '.join(missing)}")
    if not isinstance(metrics["Selected model"], str) or not metrics["Selected model"].strip():
        raise DataError(f"{path.name} has an invalid Selected model.")
    for key in ("Threshold", "Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"):
        value = metrics[key]
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise DataError(f"{path.name} has a non-numeric {key}.")
        if not 0 <= float(value) <= 1:
            raise DataError(f"{path.name} has {key} outside the range 0 to 1.")
    for key in ("TN", "FP", "FN", "TP", "Test rows"):
        value = metrics[key]
        if not isinstance(value, (int, float)) or not math.isfinite(value) or int(value) != value:
            raise DataError(f"{path.name} has an invalid {key}.")
        if int(value) < 0:
            raise DataError(f"{path.name} has a negative {key}.")
    return metrics


def load_predictions(path: Path = PREDICTIONS_PATH) -> pd.DataFrame:
    if not path.exists():
        raise DataError(f"Missing file: {path.name}")
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # unreadable / malformed CSV
        raise DataError(f"{path.name} could not be read: {exc}") from exc
    missing = [c for c in REQUIRED_PREDICTION_COLUMNS if c not in df.columns]
    if missing:
        raise DataError(f"{path.name} is missing columns: {', '.join(missing)}")
    if df.empty:
        raise DataError(f"{path.name} contains no rows.")
    df = df[REQUIRED_PREDICTION_COLUMNS].copy()
    for column in ("Default", "prediction", "default_probability"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    if df[["Default", "prediction", "default_probability"]].isna().any().any():
        raise DataError(f"{path.name} contains non-numeric or missing values.")
    if not df["default_probability"].between(0, 1).all():
        raise DataError("default_probability values must lie between 0 and 1.")
    if not set(df["Default"].unique()) <= {0, 1}:
        raise DataError("Default must be 0 or 1.")
    if not set(df["prediction"].unique()) <= {0, 1}:
        raise DataError("prediction must be 0 or 1.")
    if df["LoanID"].isna().any() or (df["LoanID"].astype(str).str.strip() == "").any():
        raise DataError("LoanID must not be missing or blank.")
    if df["LoanID"].astype(str).duplicated().any():
        raise DataError("LoanID values must be unique.")
    if not set(df["risk_category"].unique()) <= set(RISK_ORDER):
        raise DataError("risk_category must be Low, Medium or High.")
    df["Default"] = df["Default"].astype(int)
    df["prediction"] = df["prediction"].astype(int)
    df["LoanID"] = df["LoanID"].astype(str)
    df["risk_category"] = pd.Categorical(df["risk_category"], categories=RISK_ORDER, ordered=True)
    expected_risk = pd.cut(
        df["default_probability"],
        [-1, 0.30 - 1e-12, 0.60 - 1e-12, 2],
        labels=RISK_ORDER,
    )
    if not (expected_risk.astype(str) == df["risk_category"].astype(str)).all():
        raise DataError("risk_category does not match the configured score bands.")
    df["Actual"] = df["Default"].map({0: "No default", 1: "Default"})
    df["Predicted"] = df["prediction"].map({0: "No default", 1: "Default"})
    return df


def load_comparison(path: Path = COMPARISON_PATH):
    """Validation comparison is optional; return None if unavailable."""
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise DataError(f"{path.name} could not be read: {exc}") from exc
    needed = ["Model", "Threshold", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    missing = [column for column in needed if column not in df.columns]
    if missing:
        raise DataError(f"{path.name} is missing columns: {', '.join(missing)}")
    if df.empty or df["Model"].isna().any() or (df["Model"].astype(str).str.strip() == "").any():
        raise DataError(f"{path.name} contains no valid model rows.")
    for column in needed[1:]:
        values = pd.to_numeric(df[column], errors="coerce")
        if values.isna().any() or (~values.between(0, 1)).any():
            raise DataError(f"{path.name} contains invalid values in {column}.")
    return df[needed]


def compute_kpis(predictions: pd.DataFrame, metrics: dict) -> dict:
    """KPIs: rates come from the saved predictions; Precision/Recall/ROC-AUC from final test metrics
    (ROC-AUC needs raw scores, and the saved metrics file is the authoritative test result)."""
    return {
        "borrowers": int(len(predictions)),
        "actual_default_rate": float(predictions["Default"].mean()),
        "predicted_default_rate": float(predictions["prediction"].mean()),
        "precision": float(metrics["Precision"]),
        "recall": float(metrics["Recall"]),
        "roc_auc": float(metrics["ROC-AUC"]),
    }


def confusion_from_predictions(predictions: pd.DataFrame) -> dict:
    a, p = predictions["Default"], predictions["prediction"]
    return {"TN": int(((a == 0) & (p == 0)).sum()), "FP": int(((a == 0) & (p == 1)).sum()),
            "FN": int(((a == 1) & (p == 0)).sum()), "TP": int(((a == 1) & (p == 1)).sum())}


def classification_metrics(predictions: pd.DataFrame) -> dict:
    """Recalculate classification metrics from labels, predictions and scores."""
    cm = confusion_from_predictions(predictions)
    tn, fp, fn, tp = (cm[key] for key in ("TN", "FP", "FN", "TP"))
    total = tn + fp + fn + tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "Accuracy": (tp + tn) / total if total else 0.0,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc_from_predictions(predictions),
        **cm,
    }


def roc_auc_from_predictions(predictions: pd.DataFrame) -> float:
    """Calculate ROC-AUC from continuous default_probability scores."""
    labels = predictions["Default"]
    if labels.nunique() != 2:
        raise DataError("ROC-AUC requires both default classes in the predictions.")
    positive = labels == 1
    negative = labels == 0
    ranks = predictions["default_probability"].rank(method="average")
    positive_count = int(positive.sum())
    negative_count = int(negative.sum())
    rank_sum = float(ranks[positive].sum())
    return (rank_sum - positive_count * (positive_count + 1) / 2) / (
        positive_count * negative_count
    )


def metrics_consistent(predictions: pd.DataFrame, metrics: dict) -> bool:
    """True when saved metrics agree with the complete prediction file."""
    try:
        calculated = classification_metrics(predictions)
    except DataError:
        return False
    if any(key not in metrics for key in REQUIRED_METRIC_KEYS):
        return False
    if int(metrics["Test rows"]) != len(predictions):
        return False
    threshold = float(metrics["Threshold"])
    expected = (predictions["default_probability"] >= threshold).astype(int)
    if not (expected == predictions["prediction"]).all():
        return False
    for key, value in calculated.items():
        if key in metrics and not math.isclose(float(metrics[key]), float(value), rel_tol=1e-9, abs_tol=1e-9):
            return False
    return True


def risk_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    grouped = predictions.groupby("risk_category", observed=False).agg(
        borrowers=("LoanID", "count"),
        actual_defaults=("Default", "sum"),
        predicted_defaults=("prediction", "sum"),
    ).reindex(RISK_ORDER)
    grouped["share"] = grouped["borrowers"] / grouped["borrowers"].sum()
    grouped["actual_default_rate"] = (grouped["actual_defaults"] / grouped["borrowers"]).where(
        grouped["borrowers"] > 0, 0.0)
    return grouped.reset_index()


def filter_borrowers(predictions: pd.DataFrame, loan_id: str = "", risks=None, status: str = "All") -> pd.DataFrame:
    out = predictions
    query = (loan_id or "").strip()
    if query:
        out = out[out["LoanID"].str.contains(query, case=False, regex=False)]
    if risks:
        out = out[out["risk_category"].isin(risks)]
    if status == "Actual defaults":
        out = out[out["Default"] == 1]
    elif status == "Actual non-defaults":
        out = out[out["Default"] == 0]
    elif status == "Predicted defaults":
        out = out[out["prediction"] == 1]
    return out.sort_values("default_probability", ascending=False)


def export_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Readable column names for the explorer table and CSV download."""
    return pd.DataFrame({
        "Loan ID": df["LoanID"],
        "Default score": df["default_probability"].round(4),
        "Risk category": df["risk_category"].astype(str),
        "Predicted": df["Predicted"],
        "Actual": df["Actual"],
    })
