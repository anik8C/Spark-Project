# Predicting Loan Default Risk Using Apache Spark and Machine Learning

INT315 Cluster Computing academic project. The notebook is the single source of truth for the PySpark data pipeline, machine learning, evaluation, and saved dashboard outputs. Spark runs locally (`local[2]`) on one machine.

## Simplified architecture

```text
data/raw/Loan_default .csv
        |
        v
Notebooks/loan_default_analysis.ipynb  -->  outputs/
                                              |-- model/
                                              |-- tables/
                                              `-- graphs/
        |
        v
app.py + dashboard_data.py  (read-only Streamlit dashboard)
```

- `Notebooks/loan_default_analysis.ipynb`: complete runnable project: validation, EDA, preprocessing, stratified split, model training, model selection, threshold tuning, final test evaluation, predictions, and output generation.
- `app.py`: Streamlit dashboard. It never starts Spark or trains a model.
- `dashboard_data.py`: dashboard-only loading, validation, KPI, filtering, and export helpers.
- `outputs/`: saved model, metrics, predictions, comparison table, and graphs.
- `submission/Loan_Default_Prediction_Project_Report.docx`: editable academic report with verified results and screenshot placeholders.
- `data/raw/`: original 5,000-record sample.
- `tests/`: dashboard data-layer validation tests.

## Dataset

The dataset is the Loan Default Prediction Dataset from [FGithaiga/LOAN-DEFAULT-PREDICTION](https://github.com/FGithaiga/LOAN-DEFAULT-PREDICTION). The project uses the first 5,000 data records with the header retained. Save it as `data/raw/Loan_default .csv` (including the space before `.csv`). It contains 18 columns and the binary `Default` target.

## Methodology

The notebook validates required columns and types, nulls, duplicate rows, unique `LoanID` values, and binary labels. It processes nine numeric and seven categorical features with `StringIndexer`, `OneHotEncoder`, and `VectorAssembler`, fitted on training data only.

The class-stratified split is 64% training (3,199 rows), 16% validation (800 rows), and 20% test (1,001 rows). Logistic Regression and Random Forest are each trained in weighted and unweighted variants. Thresholds from 0.10 to 0.80 are selected using validation F1, with recall and precision tie-breakers. The selected model is refit on training plus validation rows, then evaluated once on the untouched test set. The notebook saves accuracy, precision, recall, F1-score, ROC-AUC, confusion-matrix counts, predictions, the Spark pipeline model, and graphs.

Risk bands are illustrative ranking categories: Low `< 0.30`, Medium `0.30` to `< 0.60`, and High `>= 0.60`. Scores are not calibrated probabilities or a lending policy.

## Installation and execution (macOS)

From the project root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Java 17 is required by Spark. Open `Notebooks/loan_default_analysis.ipynb` in Jupyter or VS Code and run all cells from beginning to end. This generates the required files in `outputs/`.

Only after the notebook finishes successfully, launch the dashboard:

```bash
python -m streamlit run app.py
```

The dashboard displays model KPIs, actual versus predicted defaults, confusion matrix, validation model comparison, risk categories, default rates by category, borrower search and filters, and CSV download. All predictions shown are explicitly from the held-out test set.

Run the existing automated checks with:

```bash
python -m unittest discover -s tests
```

Re-running the notebook overwrites generated files under `outputs/`. If another Spark installation is configured globally, unset `SPARK_HOME` before starting the notebook so the pinned PySpark 3.5.6 environment is used. The `.gitignore` excludes `.venv`, Python caches, notebook checkpoints, and macOS metadata.

## Limitations

- The 5,000-row sample is sequential rather than random and may not generalize.
- Spark is evaluated in local single-machine mode.
- The single stratified split has sampling uncertainty.
- Class-weighted scores are ranking scores, not calibrated probabilities.
- Demographic features are included; this academic prototype is not a production credit decision tool.
