"""Final loan-default workflow: split, tune, evaluate and save the Spark pipeline.

Run from any directory:  python scripts/run_workflow.py
Requires the dataset at <project>/data/raw/Loan_default .csv (not included in the archive).
Spark runs in local multi-core mode (local[2]), not on a multi-node cluster.
"""

import json
import os
import sys
from pathlib import Path

# Use the caller's JAVA_HOME if set; otherwise fall back to the Homebrew OpenJDK 17 path (macOS) when present.
_MAC_JAVA17 = "/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home"
if "JAVA_HOME" not in os.environ and Path(_MAC_JAVA17).exists():
    os.environ["JAVA_HOME"] = _MAC_JAVA17
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
os.environ.pop("SPARK_HOME", None)
os.environ.pop("SPARK_CONF_DIR", None)
os.environ.pop("PYSPARK_SUBMIT_ARGS", None)

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, lit, rand, row_number, when
from pyspark.sql.types import NumericType, StringType
from pyspark.sql.window import Window

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "Loan_default .csv"
OUTPUTS = ROOT / "outputs"
MODEL_DIR = OUTPUTS / "model"
TABLES_DIR = OUTPUTS / "tables"
GRAPHS_DIR = OUTPUTS / "graphs"

# Illustrative probability bands for dashboard risk categories (not a validated lending policy).
LOW_RISK_UPPER = 0.30
MEDIUM_RISK_UPPER = 0.60

NUMERIC_FEATURES = [
    "Age", "Income", "LoanAmount", "CreditScore", "MonthsEmployed",
    "NumCreditLines", "InterestRate", "LoanTerm", "DTIRatio",
]
CATEGORICAL_FEATURES = [
    "Education", "EmploymentType", "MaritalStatus", "HasMortgage",
    "HasDependents", "LoanPurpose", "HasCoSigner",
]
REQUIRED_COLUMNS = ["LoanID", *NUMERIC_FEATURES, *CATEGORICAL_FEATURES, "Default"]


def validate_input_dataframe(df: DataFrame) -> None:
    """Validate the input contract before any split or model fitting."""
    missing_columns = [name for name in REQUIRED_COLUMNS if name not in df.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing_columns)}")

    invalid_types = []
    for name in NUMERIC_FEATURES + ["Default"]:
        if not isinstance(df.schema[name].dataType, NumericType):
            invalid_types.append(f"{name} ({df.schema[name].dataType.simpleString()})")
    for name in ["LoanID", *CATEGORICAL_FEATURES]:
        if not isinstance(df.schema[name].dataType, StringType):
            invalid_types.append(f"{name} ({df.schema[name].dataType.simpleString()})")
    if invalid_types:
        raise ValueError(
            "Dataset has invalid column types; expected numeric features/Default and "
            f"string identifiers/categories: {', '.join(invalid_types)}"
        )

    missing_counts = (
        df.select([
            col(name).isNull().cast("long").alias(name)
            for name in REQUIRED_COLUMNS
        ]).groupBy().sum().first().asDict()
    )
    missing = {
        name: int(missing_counts.get(f"sum({name})") or 0)
        for name in REQUIRED_COLUMNS
        if int(missing_counts.get(f"sum({name})") or 0) > 0
    }
    if missing:
        details = ", ".join(f"{name}={count}" for name, count in missing.items())
        raise ValueError(f"Dataset contains missing values: {details}")

    invalid_labels = [
        row["Default"]
        for row in df.select("Default").distinct().collect()
        if row["Default"] not in (0, 1, 0.0, 1.0)
    ]
    if invalid_labels:
        raise ValueError(f"Default must contain only 0 and 1; found {invalid_labels}")

    duplicate_rows = df.count() - df.dropDuplicates().count()
    if duplicate_rows:
        raise ValueError(
            f"Dataset contains {duplicate_rows} duplicate row(s); remove them before training."
        )
    duplicate_ids = df.count() - df.select("LoanID").distinct().count()
    if duplicate_ids:
        raise ValueError(
            f"Dataset contains {duplicate_ids} duplicate LoanID row(s); "
            "LoanID values must be unique."
        )


def stratified_split(df: DataFrame, seed: int = 42) -> tuple[DataFrame, DataFrame, DataFrame]:
    """Create approximately 64:16:20 splits independently within each class."""
    ranked = df.withColumn(
        "_split_rank",
        row_number().over(Window.partitionBy("Default").orderBy(rand(seed))),
    )
    counts = {
        row["Default"]: row["count"]
        for row in df.groupBy("Default").count().collect()
    }
    train_cutoffs = {label: int(count * 0.64) for label, count in counts.items()}
    validation_cutoffs = {
        label: int(count * 0.80)
        for label, count in counts.items()
    }
    train_condition = lit(False)
    validation_condition = lit(False)
    for label in counts:
        train_condition = train_condition | (
            (col("Default") == label) & (col("_split_rank") <= train_cutoffs[label])
        )
        validation_condition = validation_condition | (
            (col("Default") == label)
            & (col("_split_rank") > train_cutoffs[label])
            & (col("_split_rank") <= validation_cutoffs[label])
        )
    train = ranked.filter(train_condition).drop("_split_rank")
    validation = ranked.filter(validation_condition).drop("_split_rank")
    test = ranked.filter(~train_condition & ~validation_condition).drop("_split_rank")
    return train, validation, test


def make_preprocessing_pipeline() -> Pipeline:
    indexers = [
        StringIndexer(inputCol=name, outputCol=f"{name}_Index", handleInvalid="keep")
        for name in CATEGORICAL_FEATURES
    ]
    encoder = OneHotEncoder(
        inputCols=[f"{name}_Index" for name in CATEGORICAL_FEATURES],
        outputCols=[f"{name}_Encoded" for name in CATEGORICAL_FEATURES],
        handleInvalid="keep",
    )
    assembler = VectorAssembler(
        inputCols=NUMERIC_FEATURES + [f"{name}_Encoded" for name in CATEGORICAL_FEATURES],
        outputCol="features",
        handleInvalid="keep",
    )
    return Pipeline(stages=indexers + [encoder, assembler])


def add_class_weights(df: DataFrame) -> DataFrame:
    counts = {row["Default"]: row["count"] for row in df.groupBy("Default").count().collect()}
    total = sum(counts.values())
    classes = len(counts)
    weights = {label: total / (classes * count) for label, count in counts.items()}
    return df.withColumn(
        "classWeight",
        when(col("Default") == 1, lit(weights[1])).otherwise(lit(weights[0])),
    )


def build_classifier(model_name: str, weighted: bool):
    if model_name == "Logistic Regression":
        classifier = LogisticRegression(
            featuresCol="features", labelCol="Default", maxIter=100, regParam=0.0,
        )
    else:
        classifier = RandomForestClassifier(
            featuresCol="features", labelCol="Default", numTrees=150, maxDepth=8, seed=42,
        )
    if weighted:
        classifier.setWeightCol("classWeight")
    return classifier


def with_threshold(predictions: DataFrame, threshold: float) -> DataFrame:
    return predictions.withColumn(
        "prediction",
        when(vector_to_array("probability")[1] >= lit(threshold), lit(1.0))
        .otherwise(lit(0.0)),
    )


def evaluate(predictions: DataFrame, model_name: str, threshold: float) -> dict:
    counts = predictions.select(
        ((col("Default") == 1) & (col("prediction") == 1)).cast("int").alias("tp"),
        ((col("Default") == 0) & (col("prediction") == 0)).cast("int").alias("tn"),
        ((col("Default") == 0) & (col("prediction") == 1)).cast("int").alias("fp"),
        ((col("Default") == 1) & (col("prediction") == 0)).cast("int").alias("fn"),
    ).groupBy().sum().first()
    tp, tn, fp, fn = [int(counts[f"sum({key})"] or 0) for key in ("tp", "tn", "fp", "fn")]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    auc = BinaryClassificationEvaluator(
        labelCol="Default", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
    ).evaluate(predictions)
    return {
        "Model": model_name, "Threshold": threshold, "Accuracy": accuracy,
        "Precision": precision, "Recall": recall, "F1 Score": f1,
        "ROC-AUC": auc, "TN": tn, "FP": fp, "FN": fn, "TP": tp,
    }


def select_threshold(predictions: DataFrame, model_name: str) -> tuple[float, dict]:
    candidates = [round(x / 100, 2) for x in range(10, 81, 5)]
    results = [evaluate(with_threshold(predictions, threshold), model_name, threshold)
               for threshold in candidates]
    best = max(results, key=lambda row: (row["F1 Score"], row["Recall"], row["Precision"]))
    return best["Threshold"], best


def save_confusion_matrix(result: dict, filename: str) -> None:
    matrix = [[result["TN"], result["FP"]], [result["FN"], result["TP"]]]
    plt.figure(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Predicted non-default", "Predicted default"],
                yticklabels=["Actual non-default", "Actual default"])
    plt.title(f"{result['Model']} confusion matrix")
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / filename, dpi=200)
    plt.close()


def main() -> None:
    for directory in (MODEL_DIR, TABLES_DIR, GRAPHS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Download 'Loan_default .csv' from https://github.com/FGithaiga/LOAN-DEFAULT-PREDICTION "
            "(Data folder), keep the first 5,000 records plus header, and place it at the path above."
        )

    spark = (
        SparkSession.builder.appName("LoanDefaultRiskPrediction")
        .master("local[2]").config("spark.sql.shuffle.partitions", "4")
        .config("spark.ui.enabled", "false").getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    df = spark.read.option("header", True).option("inferSchema", True).csv(str(DATA_PATH))
    validate_input_dataframe(df)
    train, validation, test = stratified_split(df)
    preprocessing_model = make_preprocessing_pipeline().fit(train)
    prepared_train = preprocessing_model.transform(train).cache()
    prepared_validation = preprocessing_model.transform(validation).cache()

    # Each candidate is fitted on training data only; validation selects the model and threshold.
    candidate_results = []
    fitted_candidates = {}
    weighted_train = add_class_weights(train)
    weighted_prepared_train = preprocessing_model.transform(weighted_train).cache()
    for model_name in ("Logistic Regression", "Random Forest"):
        for weighted, source in ((False, prepared_train), (True, weighted_prepared_train)):
            label = f"{model_name} ({'weighted' if weighted else 'unweighted'})"
            classifier = build_classifier(model_name, weighted)
            model = classifier.fit(source)
            validation_predictions = model.transform(prepared_validation)
            threshold, result = select_threshold(validation_predictions, label)
            candidate_results.append(result)
            fitted_candidates[label] = (classifier, model, threshold, weighted)

    validation_results = pd.DataFrame(candidate_results).sort_values(
        ["F1 Score", "Recall", "Precision", "ROC-AUC"], ascending=False
    )
    validation_results.to_csv(TABLES_DIR / "validation_model_comparison.csv", index=False)
    selected = validation_results.iloc[0].to_dict()
    selected_label = selected["Model"]
    model_name = selected_label.split(" (")[0]
    weighted = "weighted" in selected_label

    # Refit preprocessing and the selected classifier on all non-test rows only.
    full_train = train.unionByName(validation)
    full_source = add_class_weights(full_train) if weighted else full_train
    final_pipeline = Pipeline(
        stages=make_preprocessing_pipeline().getStages() + [build_classifier(model_name, weighted)]
    )
    final_model = final_pipeline.fit(full_source)
    final_predictions = with_threshold(final_model.transform(test), float(selected["Threshold"]))
    test_result = evaluate(final_predictions, selected_label, float(selected["Threshold"]))
    test_result["Training rows"] = full_train.count()
    test_result["Validation rows"] = validation.count()
    test_result["Test rows"] = test.count()
    test_result["Selected model"] = selected_label
    with open(TABLES_DIR / "final_test_metrics.json", "w", encoding="utf-8") as file:
        json.dump(test_result, file, indent=2)
    final_model.write().overwrite().save(str(MODEL_DIR / "loan_default_pipeline"))

    predictions = final_predictions.select(
        "LoanID", "Default", "probability", "prediction"
    ).withColumn(
        "default_probability", vector_to_array("probability")[1].cast("double")
    ).drop("probability")
    predictions = predictions.withColumn(
        "risk_category",
        when(col("default_probability") < LOW_RISK_UPPER, "Low")
        .when(col("default_probability") < MEDIUM_RISK_UPPER, "Medium")
        .otherwise("High"),
    )
    predictions.toPandas().to_csv(TABLES_DIR / "borrower_predictions.csv", index=False)
    pd.DataFrame([test_result]).to_csv(TABLES_DIR / "final_test_metrics.csv", index=False)
    save_confusion_matrix(test_result, "final_confusion_matrix.png")

    chart = validation_results[["Model", "Precision", "Recall", "F1 Score", "ROC-AUC"]].melt(
        id_vars="Model", var_name="Metric", value_name="Value"
    )
    plt.figure(figsize=(10, 5))
    sns.barplot(data=chart, x="Metric", y="Value", hue="Model")
    plt.ylim(0, 1)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / "model_comparison.png", dpi=200)
    plt.close()
    print("Validation comparison:")
    print(validation_results.to_string(index=False))
    print("\nUntouched test result:")
    print(json.dumps(test_result, indent=2))
    print(f"\nSaved model and outputs under {OUTPUTS}")
    spark.stop()


if __name__ == "__main__":
    main()
