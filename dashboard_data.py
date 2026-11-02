"""Pure-pandas helpers for the dashboard (no Streamlit import, so they are unit-testable)."""

import json
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
REQUIRED_METRIC_KEYS = ["Selected model", "Threshold", "Precision", "Recall", "F1 Score",
                        "ROC-AUC", "TN", "FP", "FN", "TP"]


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
    if not set(df["risk_category"].unique()) <= set(RISK_ORDER):
        raise DataError("risk_category must be Low, Medium or High.")
    df["Default"] = df["Default"].astype(int)
    df["prediction"] = df["prediction"].astype(int)
    df["LoanID"] = df["LoanID"].astype(str)
    df["risk_category"] = pd.Categorical(df["risk_category"], categories=RISK_ORDER, ordered=True)
    df["Actual"] = df["Default"].map({0: "No default", 1: "Default"})
    df["Predicted"] = df["prediction"].map({0: "No default", 1: "Default"})
    return df


def load_comparison(path: Path = COMPARISON_PATH):
    """Validation comparison is optional; return None if unavailable."""
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    needed = ["Model", "Threshold", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    return df[needed] if all(c in df.columns for c in needed) else None


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


def metrics_consistent(predictions: pd.DataFrame, metrics: dict) -> bool:
    """True when the saved metrics' confusion matrix matches the saved predictions."""
    cm = confusion_from_predictions(predictions)
    return all(cm[k] == int(metrics[k]) for k in cm)


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
