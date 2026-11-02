# Predicting Loan Default Risk Using Apache Spark and Machine Learning

INT315 Cluster Computing academic project. A PySpark ML pipeline predicts loan default from borrower and loan
characteristics, and a Streamlit dashboard presents the saved results.

> **Scope.** Spark runs in **local multi-core mode** (`local[2]`) on a single machine, not on a multi-node cluster.
> The model is a classroom prototype and is **not** production-ready or a validated lending policy.

## Problem statement
Lenders need to identify borrowers likely to default. Defaults are rare (about 12% here), so accuracy alone is misleading.
The project builds and evaluates classifiers that flag likely defaults and expose the trade-off between catching
defaults (recall) and avoiding false alarms (precision).

## Objectives
1. Load the loan dataset into a Spark DataFrame and handle missing/null values.
2. Encode categorical features and assemble numerical features.
3. Train Logistic Regression and Random Forest classifiers.
4. Evaluate precision, recall, F1-score, ROC-AUC and the confusion matrix.
5. Predict default risk per borrower.
6. Present default rates, actual vs predicted defaults and performance by risk category in a dashboard.

## Dataset
- **Name:** Loan Default Prediction Dataset (borrower, credit and loan characteristics with a binary `Default` label; it is *not* bank transaction or repayment-history data).
- **Source:** https://github.com/FGithaiga/LOAN-DEFAULT-PREDICTION (file `Data/Loan_default .csv`, original size 255,347 rows, 18 columns).
- **Sample used:** the **first 5,000 data records** (header retained). This is a sequential subset, **not a random sample**, so it may not represent the full dataset.
- **Sample profile:** 5,000 rows, 18 columns, 0 missing values, 0 duplicate rows, 5,000 unique Loan IDs; 4,414 non-defaults (88.28%) and 586 defaults (11.72%).
- **Columns:** LoanID, Age, Income, LoanAmount, CreditScore, MonthsEmployed, NumCreditLines, InterestRate, LoanTerm, DTIRatio, Education, EmploymentType, MaritalStatus, HasMortgage, HasDependents, LoanPurpose, HasCoSigner, Default (0 = no default, 1 = default).

### Dataset setup (required)
**The CSV is excluded from the project archive because of its size and must be downloaded separately.**
1. Download `Loan_default .csv` from the GitHub repository above (the filename contains a space before `.csv`).
2. Keep the header and the first 5,000 data records (for example `head -n 5001 "Loan_default .csv" > sample.csv`).
3. Save the result as `data/raw/Loan_default .csv` inside this project.

The saved model, metrics and predictions in `outputs/` are already included, so the **dashboard runs without the dataset**.
The dataset is only needed to re-run the workflow or the notebook.

## Technologies
Python 3.11, Apache Spark / PySpark 3.5.6 (local mode), Java 17, Pandas, NumPy, Matplotlib, Seaborn, Streamlit (Altair charts), Jupyter Notebook, VS Code. Developed on macOS (Apple Silicon M2).

## Project structure
```
app.py                       Streamlit dashboard (reads saved outputs only)
dashboard_data.py            Loading, validation, KPI and filter logic for the dashboard
scripts/run_workflow.py      Final workflow: split, train, tune, evaluate, save
Notebooks/loan_default_analysis.ipynb   Data inspection, EDA, preliminary experiments
Notebooks/testspark.py       Spark/Java installation smoke test
tests/test_dashboard_data.py Unit tests for the dashboard data layer
requirements.txt
data/raw/                    Place the dataset here (not included)
outputs/model/loan_default_pipeline/    Saved Spark PipelineModel
outputs/tables/              Metrics, validation comparison, borrower predictions
outputs/graphs/              EDA and evaluation figures
```

## Methodology
**Preprocessing.** The CSV is read with `inferSchema`. Missing values and duplicates were checked (none found in this sample). `LoanID` is excluded from model features. Imputation was therefore not needed; `handleInvalid="keep"` in the indexers, encoder and assembler guards against unseen categories or nulls at prediction time.

**EDA.** Class distribution; default rate by employment type, education, marital status and loan purpose; group means of numerical features by outcome; correlation of numerical features with `Default`. These are descriptive associations, not causal findings.

**Feature engineering.** 9 numerical features (Age, Income, LoanAmount, CreditScore, MonthsEmployed, NumCreditLines, InterestRate, LoanTerm, DTIRatio) and 7 categorical features (Education, EmploymentType, MaritalStatus, HasMortgage, HasDependents, LoanPurpose, HasCoSigner) processed with `StringIndexer`, `OneHotEncoder` and `VectorAssembler` into a 38-dimensional feature vector.

**Data partition (stratified by class, seed 42).**

| Partition | Records | Use |
|---|---:|---|
| Training | 3,199 | Fit candidate models |
| Validation | 800 | Model selection and threshold tuning |
| Test | 1,001 | Final evaluation only |

After selection, preprocessing and the chosen classifier are refit on training + validation (3,999 rows) and evaluated once on the test set.

**Model training.** Logistic Regression and Random Forest, each with and without class weights (four candidates).

**Class imbalance.** Defaults are about 12% of rows, so unweighted models predicted almost no defaults at the default 0.5 threshold. Weighted variants use inverse-frequency class weights during fitting only.

**Threshold selection.** For each candidate, thresholds from 0.10 to 0.80 (step 0.05) were scored on the **validation** set; the model/threshold with the best validation F1 (ties broken by recall, then precision) was selected: **Weighted Logistic Regression, threshold 0.65**.

### Validation comparison (model selection; not test results)

| Model | Threshold | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression (weighted) | 0.65 | 0.306 | 0.484 | 0.375 | 0.755 |
| Logistic Regression (unweighted) | 0.20 | 0.313 | 0.441 | 0.366 | 0.758 |
| Random Forest (unweighted) | 0.15 | 0.265 | 0.570 | 0.362 | 0.742 |
| Random Forest (weighted) | 0.35 | 0.234 | 0.688 | 0.349 | 0.745 |

The F1 margin between candidates is small on 800 validation rows, so the choice is not a strong statistical separation.

## Final test results (1,001 held-out borrowers)

| Metric | Result |
|---|---:|
| Precision | 0.3072 |
| Recall | 0.4322 |
| F1-score | 0.3592 |
| ROC-AUC | 0.7633 |
| Accuracy | 0.8182 |

| | Predicted no default | Predicted default |
|---|---:|---:|
| Actual no default | 768 | 115 |
| Actual default | 67 | 51 |

51 of 118 actual defaults were identified, 67 were missed, and 115 non-defaulters were flagged. Detection improves on the unweighted baselines, but precision is limited: roughly 7 in 10 flagged borrowers did not default. Source: `outputs/tables/final_test_metrics.json`.

## Risk categories (illustrative)
Low: score below 0.30. Medium: 0.30 to below 0.60. High: 0.60 or above. These bands are illustrative and are **not** a validated lending policy.

Limitations of the bands:
- Class weighting inflates the model's scores (mean test score about 0.41 versus an actual default rate of about 11.8%). The `default_probability` column is therefore a **risk score for ranking, not a calibrated probability**.
- The High band starts at 0.60, below the 0.65 decision threshold, so some High-risk borrowers are not predicted as defaults. The dashboard reports the *actual* default rate within each band instead of treating scores as probabilities.
- Bands were not tuned or validated for any business cost.

## Dashboard
Single-page, no sidebar: header (model and threshold), six KPIs (borrowers, actual default rate, predicted default rate, precision, recall, ROC-AUC), actual vs predicted defaults, borrowers by risk category, actual default rate by risk category, confusion matrix with test metrics, an optional validation comparison (clearly labelled as validation), and a borrower explorer (Loan ID search, risk/status filters, CSV download of the filtered rows). Rates are computed from the saved prediction file; the dashboard never loads or retrains Spark.

## Installation
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Java 17 is required only for Spark (the workflow and notebook). Set `JAVA_HOME` if it is not detected automatically.

## Execution
Commands work from any directory when you give the path to the project files; from the project root:
```bash
# 1. (Optional) Regenerate model, metrics, predictions and charts. Needs the dataset.
python scripts/run_workflow.py

# 2. Launch the dashboard (works with the included outputs, no dataset needed)
streamlit run app.py

# 3. Unit tests for the dashboard data layer
python -m unittest discover -s tests
```
Re-running the workflow overwrites files in `outputs/`. Results may differ slightly from those reported here if the Spark version or environment differs, and the metrics above should only be updated by re-running the workflow.

## Notebook
`Notebooks/loan_default_analysis.ipynb` contains inspection, EDA and feature-engineering steps reused by the final workflow, followed by a section marked **preliminary experiments** (80/20 random split, no weights, 0.5 threshold; Logistic Regression recall 0.039, Random Forest recall 0.0). Those numbers are superseded by the final workflow above.

## Limitations
- 5,000-row sequential subset of a 255,347-row dataset; not a random sample, so results may not generalize.
- Local single-machine Spark; no distributed performance evaluation was performed.
- Low precision (0.31) and moderate recall (0.43); many false alarms and missed defaults.
- Single stratified split with 118 test defaults; metrics carry notable sampling uncertainty, and no cross-validation or confidence intervals were computed.
- Scores are uncalibrated (see above); risk bands are illustrative.
- Demographic features (e.g. age, marital status, education) are in the model; no fairness or regulatory review was performed, and the model should not be used for real credit decisions.
- Findings are associations in a dataset of borrower attributes, not causal explanations.
