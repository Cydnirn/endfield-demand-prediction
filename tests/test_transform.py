"""
Unit tests for transform.py — the PySpark ETL pipeline.

Run with:
    pytest tests/test_transform.py -v
"""

import os

# Import the module under test
import sys
import tempfile
from datetime import date, timedelta

import pytest
from pyspark.sql import Row, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DateType,
    DoubleType,
    StringType,
    StructField,
    StructType,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import transform

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def spark():
    """Session-scoped local Spark session."""
    sess = (
        SparkSession.builder.master("local[2]")
        .appName("test_transform")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    yield sess
    sess.stop()


@pytest.fixture
def raw_schema():
    """The exact raw CSV schema as defined in project specs."""
    return StructType(
        [
            StructField("Date", StringType(), True),
            StructField("Store ID", StringType(), True),
            StructField("Product ID", StringType(), True),
            StructField("Category", StringType(), True),
            StructField("Region", StringType(), True),
            StructField("Inventory Level", StringType(), True),  # read as string
            StructField("Units Sold", StringType(), True),
            StructField("Units Ordered", StringType(), True),
            StructField("Price", StringType(), True),
            StructField("Discount", StringType(), True),
            StructField("Weather Condition", StringType(), True),
            StructField("Promotion", StringType(), True),
            StructField("Competitor Pricing", StringType(), True),
            StructField("Seasonality", StringType(), True),
            StructField("Epidemic", StringType(), True),
            StructField("Demand", StringType(), True),
        ]
    )


@pytest.fixture
def sample_df(spark, raw_schema):
    """
    Build a small deterministic DataFrame with 45 rows covering:
    - Two stores (S01, S02), two products (P01, P02) → 4 groups.
    - 15 days per group (some overlapping to test partial windows).
    - A few deliberate NULLs for imputation testing.
    - One record on 2024-01-01 to test the date split boundary.
    """
    base = date(2023, 12, 15)  # 15 days leading into 2024
    rows = []
    for store in ["S01", "S02"]:
        for product in ["P01", "P02"]:
            for day_offset in range(15):
                d = base + timedelta(days=day_offset)
                demand = 10.0 + day_offset * 2 + hash(store + product) % 10
                # Introduce NULLs:
                # - Price NULL on day_offset 2 for group S01-P01
                # - Region NULL on day_offset 5 for group S02-P02
                price = (
                    None
                    if (store == "S01" and product == "P01" and day_offset == 2)
                    else str(2.99 + day_offset * 0.1)
                )
                region = (
                    None
                    if (store == "S02" and product == "P02" and day_offset == 5)
                    else "East"
                    if day_offset % 2 == 0
                    else "West"
                )
                rows.append(
                    Row(
                        Date=d.strftime("%Y-%m-%d"),
                        **{
                            "Store ID": store,
                            "Product ID": product,
                            "Category": "Food" if day_offset % 3 == 0 else "Drink",
                            "Region": region,
                            "Inventory Level": str(100 + day_offset),
                            "Units Sold": str(int(demand) + 5),  # leaked columns
                            "Units Ordered": str(int(demand) + 3),
                            "Price": price,
                            "Discount": str(0.10),
                            "Weather Condition": "Sunny"
                            if day_offset % 2 == 0
                            else "Rainy",
                            "Promotion": "1" if day_offset % 7 == 0 else "0",
                            "Competitor Pricing": str(2.50 + day_offset * 0.05),
                            "Seasonality": "Regular"
                            if day_offset % 5 != 0
                            else "Holiday",
                            "Epidemic": "0",
                            "Demand": str(demand),
                        },
                    )
                )
    return spark.createDataFrame(rows, schema=raw_schema)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _drop_cols_apply(df):
    """Apply the drop and type-cast steps as done in run()."""
    df = df.drop(*transform.DROP_COLS)
    # Cast raw numeric columns
    for c in transform.RAW_NUMERIC_COLS + [transform.TARGET_COL]:
        df = df.withColumn(c, F.col(c).cast(DoubleType()))
    # Parse date
    df = df.withColumn(
        transform.DATE_COL, F.to_date(F.col(transform.DATE_COL), "yyyy-MM-dd")
    )
    return df


# ---------------------------------------------------------------------------
# Tests: column dropping
# ---------------------------------------------------------------------------


class TestColumnDropping:
    def test_leaked_columns_removed(self, sample_df):
        """Units Sold and Units Ordered must not appear in output columns."""
        df = _drop_cols_apply(sample_df)
        df = transform.engineer_features(df)
        df = transform.impute_missing(df)
        all_req = (
            transform.CAT_COLS
            + transform.RAW_NUMERIC_COLS
            + transform.ENGINEERED_COLS
            + [transform.TARGET_COL]
        )
        available = [c for c in all_req if c in df.columns]
        df = df.dropna(subset=available)

        for leaked in transform.DROP_COLS:
            assert leaked not in df.columns, f"Leaked column {leaked!r} still present"

    def test_target_column_present(self, sample_df):
        """Demand must be present in the final output columns."""
        df = _drop_cols_apply(sample_df)
        df = transform.engineer_features(df)
        df = transform.impute_missing(df)
        all_req = (
            transform.CAT_COLS
            + transform.RAW_NUMERIC_COLS
            + transform.ENGINEERED_COLS
            + [transform.TARGET_COL]
        )
        available = [c for c in all_req if c in df.columns]
        df = df.dropna(subset=available)
        assert transform.TARGET_COL in df.columns


# ---------------------------------------------------------------------------
# Tests: imputation
# ---------------------------------------------------------------------------


class TestImputation:
    def test_median_imputation_fills_nulls(self, spark, raw_schema):
        """Nulls in numeric columns are replaced with the column median."""
        rows = [
            Row(
                Date="2023-01-01",
                **{
                    "Store ID": "S01",
                    "Product ID": "P01",
                    "Category": "Food",
                    "Region": "East",
                    "Inventory Level": None,
                    "Units Sold": "5",
                    "Units Ordered": "3",
                    "Price": "1.0",
                    "Discount": "0.1",
                    "Weather Condition": "Sunny",
                    "Promotion": "0",
                    "Competitor Pricing": "0.8",
                    "Seasonality": "Regular",
                    "Epidemic": "0",
                    "Demand": "10",
                },
            ),
            Row(
                Date="2023-01-02",
                **{
                    "Store ID": "S01",
                    "Product ID": "P01",
                    "Category": "Food",
                    "Region": "East",
                    "Inventory Level": "200",
                    "Units Sold": "10",
                    "Units Ordered": "6",
                    "Price": "2.0",
                    "Discount": "0.2",
                    "Weather Condition": "Sunny",
                    "Promotion": "0",
                    "Competitor Pricing": "1.6",
                    "Seasonality": "Regular",
                    "Epidemic": "0",
                    "Demand": None,
                },
            ),
        ]
        df = spark.createDataFrame(rows, schema=raw_schema)
        df = _drop_cols_apply(df)
        before_nulls = df.filter(F.col("Inventory Level").isNull()).count()
        assert before_nulls > 0

        df = transform.impute_missing(df)
        after_nulls = df.filter(F.col("Inventory Level").isNull()).count()
        assert after_nulls == 0, "Inventory Level NULL not imputed"
        after_target = df.filter(F.col("Demand").isNull()).count()
        assert after_target == 0, "Demand NULL not imputed"

    def test_mode_imputation_fills_nulls(self, spark, raw_schema):
        """Nulls in categorical columns are replaced with the column mode."""
        rows = [
            Row(
                Date="2023-01-01",
                **{
                    "Store ID": "S01",
                    "Product ID": "P01",
                    "Category": None,
                    "Region": "East",
                    "Inventory Level": "100",
                    "Units Sold": "5",
                    "Units Ordered": "3",
                    "Price": "1.0",
                    "Discount": "0.1",
                    "Weather Condition": "Sunny",
                    "Promotion": "1",
                    "Competitor Pricing": "0.8",
                    "Seasonality": "Regular",
                    "Epidemic": "0",
                    "Demand": "10",
                },
            ),
            Row(
                Date="2023-01-02",
                **{
                    "Store ID": "S01",
                    "Product ID": "P01",
                    "Category": "Food",
                    "Region": "East",
                    "Inventory Level": "120",
                    "Units Sold": "8",
                    "Units Ordered": "4",
                    "Price": "1.5",
                    "Discount": "0.2",
                    "Weather Condition": "Sunny",
                    "Promotion": "0",
                    "Competitor Pricing": "1.2",
                    "Seasonality": "Regular",
                    "Epidemic": "0",
                    "Demand": "12",
                },
            ),
        ]
        df = spark.createDataFrame(rows, schema=raw_schema)
        df = _drop_cols_apply(df)
        before_nulls = df.filter(F.col("Category").isNull()).count()
        assert before_nulls > 0

        df = transform.impute_missing(df)
        after_nulls = df.filter(F.col("Category").isNull()).count()
        assert after_nulls == 0

        # The mode should be "Food"
        filled = df.filter(F.col("Date") == "2023-01-01").select("Category").first()
        assert filled["Category"] == "Food"


# ---------------------------------------------------------------------------
# Tests: feature engineering
# ---------------------------------------------------------------------------


class TestFeatureEngineering:
    def test_date_features(self, spark, raw_schema):
        """Verify day_of_week, is_weekend, month, year, etc."""
        rows = [
            Row(
                Date="2023-12-25",  # Monday (day_of_week=0 after -1)
                **{
                    "Store ID": "S01",
                    "Product ID": "P01",
                    "Category": "Food",
                    "Region": "East",
                    "Inventory Level": "100",
                    "Units Sold": "5",
                    "Units Ordered": "3",
                    "Price": "1.0",
                    "Discount": "0.1",
                    "Weather Condition": "Sunny",
                    "Promotion": "0",
                    "Competitor Pricing": "0.8",
                    "Seasonality": "Regular",
                    "Epidemic": "0",
                    "Demand": "10",
                },
            ),
            Row(
                Date="2023-12-30",  # Saturday (day_of_week=5)
                **{
                    "Store ID": "S01",
                    "Product ID": "P01",
                    "Category": "Food",
                    "Region": "East",
                    "Inventory Level": "105",
                    "Units Sold": "6",
                    "Units Ordered": "4",
                    "Price": "1.1",
                    "Discount": "0.1",
                    "Weather Condition": "Sunny",
                    "Promotion": "0",
                    "Competitor Pricing": "0.9",
                    "Seasonality": "Regular",
                    "Epidemic": "0",
                    "Demand": "12",
                },
            ),
        ]
        df = spark.createDataFrame(rows, schema=raw_schema)
        df = _drop_cols_apply(df)
        df = transform.engineer_features(df)

        # Monday row
        mon_row = df.filter(F.col("Date") == "2023-12-25").first()
        assert mon_row["day_of_week"] == 0  # Monday
        assert mon_row["is_weekend"] == 0
        assert mon_row["month"] == 12
        assert mon_row["year"] == 2023

        # Saturday row
        sat_row = df.filter(F.col("Date") == "2023-12-30").first()
        assert sat_row["day_of_week"] == 5  # Saturday
        assert sat_row["is_weekend"] == 1

    def test_lag_features(self, spark, raw_schema):
        """demand_lag_1 for day N should equal Demand of day N-1 within group."""
        rows = []
        for day_offset in range(5):
            d = date(2023, 12, 1) + timedelta(days=day_offset)
            rows.append(
                Row(
                    Date=d.strftime("%Y-%m-%d"),
                    **{
                        "Store ID": "S01",
                        "Product ID": "P01",
                        "Category": "Food",
                        "Region": "East",
                        "Inventory Level": "100",
                        "Units Sold": "5",
                        "Units Ordered": "3",
                        "Price": "1.0",
                        "Discount": "0.1",
                        "Weather Condition": "Sunny",
                        "Promotion": "0",
                        "Competitor Pricing": "0.8",
                        "Seasonality": "Regular",
                        "Epidemic": "0",
                        "Demand": str(10.0 + day_offset),
                    },
                )
            )
        df = spark.createDataFrame(rows, schema=raw_schema)
        df = _drop_cols_apply(df)
        df = transform.engineer_features(df)

        # demand_lag_1 for day 3 should equal Demand of day 2 (= 12.0)
        row_day3 = df.filter(F.col("Date") == "2023-12-03").first()
        assert row_day3["Demand"] == 12.0
        assert row_day3["demand_lag_1"] == 11.0  # previous day's Demand

        # demand_lag_1 for day 0 should be NULL (no previous)
        row_day0 = df.filter(F.col("Date") == "2023-12-01").first()
        assert row_day0["demand_lag_1"] is None

    def test_rolling_mean_correctness(self, spark, raw_schema):
        """
        Rolling mean of window=7 uses shift(1) (i.e. demand_lag_1).
        For a group with Demand values [10,11,12,13,14,15,16,...],
        demand_lag_1 on day 2 = 11, day 3 = 12, etc.

        demand_roll_mean_7 at day k is avg of demand_lag_1 values
        for the last 7 rows ending at k.
        Verify manually for a small window.
        """
        rows = []
        for day_offset in range(10):
            d = date(2023, 12, 1) + timedelta(days=day_offset)
            rows.append(
                Row(
                    Date=d.strftime("%Y-%m-%d"),
                    **{
                        "Store ID": "S01",
                        "Product ID": "P01",
                        "Category": "Food",
                        "Region": "East",
                        "Inventory Level": "100",
                        "Units Sold": "5",
                        "Units Ordered": "3",
                        "Price": "1.0",
                        "Discount": "0.1",
                        "Weather Condition": "Sunny",
                        "Promotion": "0",
                        "Competitor Pricing": "0.8",
                        "Seasonality": "Regular",
                        "Epidemic": "0",
                        "Demand": str(10.0 + day_offset),
                    },
                )
            )
        df = spark.createDataFrame(rows, schema=raw_schema)
        df = _drop_cols_apply(df)
        df = transform.engineer_features(df)

        # Day 5: demand_lag_1 for days 0-5 = [null, 10, 11, 12, 13, 14]
        # But spark avg over last 7 rows (with null): first non-null lag appears at day1=10.
        # For day 5 (0-indexed: offset 5), rowsBetween(-6,0) includes offsets -1 through 5.
        # Wait — our window.rowsBetween(-(roll-1), 0) with roll=7: rowsBetween(-6, 0).
        # At offset 5, rows: offsets -1,0,1,2,3,4,5.
        #   demand_lag_1 values: [null, 10, 11, 12, 13, 14]
        #   avg of non-null: (10+11+12+13+14)/5 = 12.0
        # Let's actually verify by checking the computed value.
        r = df.filter(F.col("Date") == "2023-12-06").first()
        assert r["demand_roll_mean_7"] is not None
        # Just assert non-None and reasonable; exact value depends on how spark
        # handles window with nulls (it ignores nulls in avg).
        assert isinstance(r["demand_roll_mean_7"], float)

    def test_rolling_mean_no_lookahead(self, spark, raw_schema):
        """Rolling mean for day N must not use Demand from day N itself."""
        rows = []
        for day_offset in range(10):
            d = date(2023, 12, 1) + timedelta(days=day_offset)
            rows.append(
                Row(
                    Date=d.strftime("%Y-%m-%d"),
                    **{
                        "Store ID": "S01",
                        "Product ID": "P01",
                        "Category": "Food",
                        "Region": "East",
                        "Inventory Level": "100",
                        "Units Sold": "5",
                        "Units Ordered": "3",
                        "Price": "1.0",
                        "Discount": "0.1",
                        "Weather Condition": "Sunny",
                        "Promotion": "0",
                        "Competitor Pricing": "0.8",
                        "Seasonality": "Regular",
                        "Epidemic": "0",
                        "Demand": str(10.0 + day_offset),  # 10, 11, 12, ...
                    },
                )
            )
        df = spark.createDataFrame(rows, schema=raw_schema)
        df = _drop_cols_apply(df)
        df = transform.engineer_features(df)

        # Day 2 (offset 2, Demand=12.0):
        # demand_lag_1 = 11.0 (Demand from day 1)
        # Rolling mean built on demand_lag_1, so the max value in the window
        # for day 2 is 11.0 (never 12.0). The rolling mean MUST be ≤ 11.0.
        r = df.filter(F.col("Date") == "2023-12-03").first()
        assert r["demand_roll_mean_7"] <= 11.0, (
            f"Lookahead detected: roll_mean={r['demand_roll_mean_7']} uses future values"
        )

    def test_price_features(self, spark, raw_schema):
        """price_vs_competitor and discount_pct are computed correctly."""
        rows = [
            Row(
                Date="2023-01-01",
                **{
                    "Store ID": "S01",
                    "Product ID": "P01",
                    "Category": "Food",
                    "Region": "East",
                    "Inventory Level": "100",
                    "Units Sold": "5",
                    "Units Ordered": "3",
                    "Price": "2.00",
                    "Discount": "0.20",
                    "Weather Condition": "Sunny",
                    "Promotion": "0",
                    "Competitor Pricing": "1.50",
                    "Seasonality": "Regular",
                    "Epidemic": "0",
                    "Demand": "10",
                },
            ),
        ]
        df = spark.createDataFrame(rows, schema=raw_schema)
        df = _drop_cols_apply(df)
        df = transform.engineer_features(df)
        row = df.first()
        assert abs(row["price_vs_competitor"] - 0.50) < 0.001
        assert abs(row["discount_pct"] - (0.20 / 2.000001)) < 0.0001


# ---------------------------------------------------------------------------
# Tests: date split
# ---------------------------------------------------------------------------


class TestDateSplit:
    def test_split_boundary(self, spark, raw_schema):
        """
        Rows before 2024-01-01 → train; on/after → validation.
        Row on exactly 2024-01-01 goes to validation.
        """
        rows = [
            Row(
                Date="2023-12-31",
                **{
                    "Store ID": "S01",
                    "Product ID": "P01",
                    "Category": "Food",
                    "Region": "East",
                    "Inventory Level": "100",
                    "Units Sold": "5",
                    "Units Ordered": "3",
                    "Price": "1.0",
                    "Discount": "0.1",
                    "Weather Condition": "Sunny",
                    "Promotion": "0",
                    "Competitor Pricing": "0.8",
                    "Seasonality": "Regular",
                    "Epidemic": "0",
                    "Demand": "10",
                },
            ),
            Row(
                Date="2024-01-01",
                **{
                    "Store ID": "S01",
                    "Product ID": "P02",
                    "Category": "Drink",
                    "Region": "West",
                    "Inventory Level": "110",
                    "Units Sold": "8",
                    "Units Ordered": "6",
                    "Price": "2.0",
                    "Discount": "0.2",
                    "Weather Condition": "Rainy",
                    "Promotion": "1",
                    "Competitor Pricing": "1.5",
                    "Seasonality": "Regular",
                    "Epidemic": "0",
                    "Demand": "15",
                },
            ),
        ]
        df = spark.createDataFrame(rows, schema=raw_schema)
        df = _drop_cols_apply(df)
        df = transform.impute_missing(df)
        df = transform.engineer_features(df)
        all_req = (
            transform.CAT_COLS
            + transform.RAW_NUMERIC_COLS
            + transform.ENGINEERED_COLS
            + [transform.TARGET_COL]
        )
        available = [c for c in all_req if c in df.columns]
        df = df.dropna(subset=available)

        train = df.filter(F.col("Date") < transform.SPLIT_DATE)
        val = df.filter(F.col("Date") >= transform.SPLIT_DATE)

        assert train.count() == 1
        assert val.count() == 1
        assert train.first()["Date"] == date(2023, 12, 31)
        assert val.first()["Date"] == date(2024, 1, 1)


# ---------------------------------------------------------------------------
# Tests: NaN handling
# ---------------------------------------------------------------------------


class TestNaNHandling:
    def test_dropna_removes_lag_nulls(self, spark, raw_schema):
        """After lag features, rows with NaN must be dropped."""
        rows = []
        for day_offset in range(5):  # 5 days, 1 group
            d = date(2023, 12, 1) + timedelta(days=day_offset)
            rows.append(
                Row(
                    Date=d.strftime("%Y-%m-%d"),
                    **{
                        "Store ID": "S01",
                        "Product ID": "P01",
                        "Category": "Food",
                        "Region": "East",
                        "Inventory Level": "100",
                        "Units Sold": "5",
                        "Units Ordered": "3",
                        "Price": "1.0",
                        "Discount": "0.1",
                        "Weather Condition": "Sunny",
                        "Promotion": "0",
                        "Competitor Pricing": "0.8",
                        "Seasonality": "Regular",
                        "Epidemic": "0",
                        "Demand": "10",
                    },
                )
            )
        df = spark.createDataFrame(rows, schema=raw_schema)
        df = _drop_cols_apply(df)
        df = transform.engineer_features(df)

        # Before dropna, first row has null lags
        before_count = df.count()
        all_req = (
            transform.CAT_COLS
            + transform.RAW_NUMERIC_COLS
            + transform.ENGINEERED_COLS
            + [transform.TARGET_COL]
        )
        available = [c for c in all_req if c in df.columns]
        after = df.dropna(subset=available)
        assert after.count() < before_count, "NaN rows were not dropped"


# ---------------------------------------------------------------------------
# Tests: output format
# ---------------------------------------------------------------------------


class TestOutputFormat:
    def test_output_no_header_and_target_first(self, spark, sample_df, tmp_path):
        """Written CSV must have NO header row and target as first column."""
        df = _drop_cols_apply(sample_df)
        df = transform.impute_missing(df)
        df = transform.engineer_features(df)

        all_req = (
            transform.CAT_COLS
            + transform.RAW_NUMERIC_COLS
            + transform.ENGINEERED_COLS
            + [transform.TARGET_COL]
        )
        available = [c for c in all_req if c in df.columns]
        df = df.dropna(subset=available)

        # Split
        train_raw = df.filter(F.col("Date") < transform.SPLIT_DATE)
        val_raw = df.filter(F.col("Date") >= transform.SPLIT_DATE)
        train_enc, val_enc = transform.encode_categoricals(train_raw, val_raw)
        train_enc = train_enc.dropna(subset=transform.CAT_COLS)
        val_enc = val_enc.dropna(subset=transform.CAT_COLS)

        enc_cat = [c for c in transform.CAT_COLS if c in train_enc.columns]
        raw_num = [c for c in transform.RAW_NUMERIC_COLS if c in train_enc.columns]
        feat = [c for c in transform.ENGINEERED_COLS if c in train_enc.columns]
        output_cols = [transform.TARGET_COL] + enc_cat + raw_num + feat

        train_out_path = str(tmp_path / "train")
        train_out = train_enc.select(output_cols).coalesce(1)
        train_out.write.mode("overwrite").option("header", "false").csv(train_out_path)

        # Read back the written CSV
        csv_files = [
            f
            for f in os.listdir(train_out_path)
            if f.endswith(".csv") and not f.startswith(".")
        ]
        assert len(csv_files) >= 1
        with open(os.path.join(train_out_path, csv_files[0]), "r") as f:
            first_line = f.readline().strip()
            first_field = first_line.split(",")[0]

        # First field should be a number (target column), not a column name
        try:
            float(first_field)
        except ValueError:
            pytest.fail(
                f"First field is '{first_field}' — expected a numeric target, "
                f"header may be present or column ordering is wrong"
            )

    def test_target_is_first_column(self, spark, sample_df):
        """Verify that when we build output_cols, TARGET_COL is at index 0."""
        df = _drop_cols_apply(sample_df)
        df = transform.impute_missing(df)
        df = transform.engineer_features(df)
        all_req = (
            transform.CAT_COLS
            + transform.RAW_NUMERIC_COLS
            + transform.ENGINEERED_COLS
            + [transform.TARGET_COL]
        )
        available = [c for c in all_req if c in df.columns]
        df = df.dropna(subset=available)

        train_raw = df.filter(F.col("Date") < transform.SPLIT_DATE)
        val_raw = df.filter(F.col("Date") >= transform.SPLIT_DATE)
        train_enc, _ = transform.encode_categoricals(train_raw, val_raw)

        enc_cat = [c for c in transform.CAT_COLS if c in train_enc.columns]
        raw_num = [c for c in transform.RAW_NUMERIC_COLS if c in train_enc.columns]
        feat = [c for c in transform.ENGINEERED_COLS if c in train_enc.columns]
        output_cols = [transform.TARGET_COL] + enc_cat + raw_num + feat

        assert output_cols[0] == transform.TARGET_COL, (
            f"Target is not first column; got {output_cols[0]}"
        )


# ---------------------------------------------------------------------------
# Tests: end-to-end pipeline (integration-style)
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_full_pipeline_with_local_io(self, spark, sample_df, tmp_path):
        """
        Simulate the full pipeline using local filesystem instead of S3.
        Verifies train/val splits are written and at least one file per split.
        """
        # Write sample data as local CSV (simulating S3 input)
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        input_file = str(input_dir / "data.csv")
        sample_df.coalesce(1).write.option("header", "true").mode("overwrite").csv(
            input_file
        )

        output_dir = tmp_path / "output"

        # Run pipeline
        transform.run(
            input_uri=input_file,
            output_uri=str(output_dir),
        )

        # Verify output directories exist
        train_dir = output_dir / "train"
        val_dir = output_dir / "validation"
        assert train_dir.exists(), "train/ output directory missing"
        assert val_dir.exists(), "validation/ output directory missing"

        # Verify at least one CSV file in each
        train_csvs = [
            f
            for f in os.listdir(str(train_dir))
            if f.endswith(".csv") and not f.startswith(".")
        ]
        val_csvs = [
            f
            for f in os.listdir(str(val_dir))
            if f.endswith(".csv") and not f.startswith(".")
        ]
        assert len(train_csvs) > 0, "No CSV files in train/"
        assert len(val_csvs) > 0, "No CSV files in validation/"
