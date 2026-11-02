"""Integration tests for workflow validation and leakage-safe splitting."""
import sys
import unittest
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, monotonically_increasing_id, when
from pyspark.sql.types import DoubleType, IntegerType, StringType, StructField, StructType

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_workflow as workflow  # noqa: E402


class WorkflowIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = (
            SparkSession.builder.appName("LoanWorkflowTests")
            .master("local[2]")
            .config("spark.ui.enabled", "false")
            .getOrCreate()
        )
        cls.spark.sparkContext.setLogLevel("ERROR")
        cls.schema = StructType(
            [
                StructField("LoanID", StringType(), False),
                *[
                    StructField(name, IntegerType(), False)
                    for name in [
                        "Age", "Income", "LoanAmount", "CreditScore",
                        "MonthsEmployed", "NumCreditLines", "LoanTerm",
                    ]
                ],
                StructField("InterestRate", DoubleType(), False),
                StructField("DTIRatio", DoubleType(), False),
                *[
                    StructField(name, StringType(), False)
                    for name in workflow.CATEGORICAL_FEATURES
                ],
                StructField("Default", IntegerType(), False),
            ]
        )

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def make_frame(self, rows=20):
        data = []
        for index in range(rows):
            data.append(
                (
                    f"ID-{index}",
                    30, 50000, 10000, 700, 24, 2, 36, 7.5, 0.3,
                    "Bachelor", "Employed", "Married", "Yes", "No", "Home", "Yes",
                    index % 2,
                )
            )
        return self.spark.createDataFrame(data, self.schema)

    def test_valid_frame_passes_input_validation(self):
        workflow.validate_input_dataframe(self.make_frame())

    def test_invalid_default_is_rejected(self):
        frame = self.make_frame().withColumn(
            "Default",
            lit(2),
        )
        with self.assertRaisesRegex(ValueError, "Default must contain only 0 and 1"):
            workflow.validate_input_dataframe(frame)

    def test_missing_required_column_is_rejected(self):
        frame = self.make_frame().drop("Income")
        with self.assertRaisesRegex(ValueError, "missing required columns"):
            workflow.validate_input_dataframe(frame)

    def test_invalid_type_is_rejected(self):
        frame = self.make_frame().withColumn("Age", col("LoanID"))
        with self.assertRaisesRegex(ValueError, "invalid column types"):
            workflow.validate_input_dataframe(frame)

    def test_missing_value_is_rejected(self):
        frame = self.make_frame().withColumn(
            "Income", when(col("LoanID") == "ID-0", None).otherwise(50000)
        )
        with self.assertRaisesRegex(ValueError, "missing values"):
            workflow.validate_input_dataframe(frame)

    def test_duplicate_rows_are_reported(self):
        frame = self.make_frame(19).unionByName(self.make_frame(1))
        with self.assertRaisesRegex(ValueError, "duplicate row"):
            workflow.validate_input_dataframe(frame)

    def test_duplicate_loan_ids_are_reported(self):
        frame = self.make_frame()
        duplicate = frame.withColumn("LoanID", lit("ID-0")).withColumn(
            "Income", monotonically_increasing_id()
        )
        with self.assertRaisesRegex(ValueError, "duplicate LoanID"):
            workflow.validate_input_dataframe(duplicate)

    def test_stratified_splits_are_disjoint_and_complete(self):
        frame = self.make_frame()
        train, validation, test = workflow.stratified_split(frame, seed=42)
        partitions = [set(part.select("LoanID").toPandas()["LoanID"]) for part in (train, validation, test)]
        self.assertEqual(sum(map(len, partitions)), frame.count())
        self.assertEqual(len(set.union(*partitions)), frame.count())
        self.assertEqual(set.intersection(*partitions), set())
        self.assertEqual([part.count() for part in (train, validation, test)], [12, 4, 4])


if __name__ == "__main__":
    unittest.main()
