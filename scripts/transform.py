"""
PySpark ETL script for retail demand forecasting.

Reads raw CSV from S3, cleans missing values, engineers features
(lags, rolling stats, date features, price ratios), encodes categoricals
via StringIndexer, splits by date, and writes train/validation CSVs to S3.

Output format: CSV without header, target column first.
Compatible with SageMaker XGBoost built-in container 1.7-1.
"""

import argparse
import sys
from typing import List, Optional

from pyspark.ml.feature import StringIndexer
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import DateType, DoubleType

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Columns to DROP (target leakage)
DROP_COLS = ["Units Sold", "Units Ordered"]

# Categorical columns to label-encode
CAT_COLS = [
    "Store ID",
    "Product ID",
    "Category",
    "Region",
    "Weather Condition",
    "Seasonality",
]

# Numeric columns that exist in raw CSV (used for imputation + selection)
RAW_NUMERIC_COLS = [
    "Inventory Level",
    "Price",
    "Discount",
    "Promotion",
    "Competitor Pricing",
    "Epidemic",
]

TARGET_COL = "Demand"
DATE_COL = "Date"

# Engineered numeric feature columns (all lower-case / underscore naming)
ENGINEERED_COLS = [
    "price_vs_competitor",
    "discount_pct",
    "day_of_week",
    "day_of_month",
    "week_of_year",
    "month",
    "quarter",
    "year",
    "is_weekend",
    "demand_lag_1",
    "demand_lag_7",
    "demand_lag_14",
    "demand_lag_28",
    "demand_roll_mean_7",
    "demand_roll_std_7",
    "demand_roll_mean_30",
    "demand_roll_std_30",
]

SPLIT_DATE = "2024-01-01"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def log(msg: str) -> None:
    """Timestamped progress logger."""
    import datetime

    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Missing-value imputation
# ---------------------------------------------------------------------------


def _compute_medians(df: DataFrame, cols: List[str]) -> dict:
    """Return {col_name: median_value} for each numeric column."""
    medians = {}
    for c in cols:
        # approxQuantile returns a list; take the 0.5 quantile (median)
        q = df.approxQuantile(c, [0.5], 0.001)
        medians[c] = q[0] if q[0] is not None else 0.0
    return medians


def _compute_modes(df: DataFrame, cols: List[str]) -> dict:
    """Return {col_name: most_frequent_value} for each categorical column."""
    modes = {}
    for c in cols:
        row = (
            df.groupBy(c)
            .count()
            .orderBy(F.desc("count"))
            .filter(F.col(c).isNotNull())
            .first()
        )
        modes[c] = row[0] if row is not None else "UNKNOWN"
    return modes


def impute_missing(df: DataFrame) -> DataFrame:
    """
    Fill missing numeric values with the column median and missing
    categorical values with the column mode.
    """
    # -- numeric imputation --
    medians = _compute_medians(df, RAW_NUMERIC_COLS + [TARGET_COL])
    log(f"Numeric medians: { {k: round(v, 3) for k, v in medians.items()} }")

    for c, med in medians.items():
        df = df.withColumn(c, F.when(F.col(c).isNull(), F.lit(med)).otherwise(F.col(c)))

    # -- categorical imputation --
    modes = _compute_modes(df, CAT_COLS)
    log(f"Categorical modes: {modes}")

    for c, mode_val in modes.items():
        df = df.withColumn(
            c, F.when(F.col(c).isNull(), F.lit(mode_val)).otherwise(F.col(c))
        )

    return df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


def engineer_features(df: DataFrame) -> DataFrame:
    """Add date, lag, rolling, and price-derived features."""
    # --- Date features ---
    df = (
        df.withColumn("day_of_week", F.dayofweek(F.col(DATE_COL)).cast("int") - 1)
        .withColumn("day_of_month", F.dayofmonth(F.col(DATE_COL)).cast("int"))
        .withColumn("week_of_year", F.weekofyear(F.col(DATE_COL)).cast("int"))
        .withColumn("month", F.month(F.col(DATE_COL)).cast("int"))
        .withColumn("quarter", F.quarter(F.col(DATE_COL)).cast("int"))
        .withColumn("year", F.year(F.col(DATE_COL)).cast("int"))
        .withColumn(
            "is_weekend",
            F.when(F.col("day_of_week") >= 5, 1).otherwise(0).cast("int"),
        )
    )

    # --- Lag features (per Store ID + Product ID group) ---
    group_window = Window.partitionBy("Store ID", "Product ID").orderBy(DATE_COL)
    for lag in [1, 7, 14, 28]:
        df = df.withColumn(
            f"demand_lag_{lag}",
            F.lag(F.col(TARGET_COL), lag).over(group_window).cast(DoubleType()),
        )

    # --- Rolling features (shift-1 to avoid lookahead, then rolling window) ---
    # PySpark doesn't have a direct pandas-style .shift().rolling(), so we
    # use the lag trick: demand_lag_1 gives us the shifted series, then
    # we apply a rolling window over *that* column.
    for roll in [7, 30]:
        df = df.withColumn(
            f"demand_roll_mean_{roll}",
            F.avg(F.col(f"demand_lag_1")).over(
                group_window.rowsBetween(-(roll - 1), 0)
            ),
        ).withColumn(
            f"demand_roll_std_{roll}",
            F.stddev(F.col(f"demand_lag_1")).over(
                group_window.rowsBetween(-(roll - 1), 0)
            ),
        )

    # --- Price-derived features ---
    df = df.withColumn(
        "price_vs_competitor",
        F.col("Price").cast(DoubleType())
        - F.col("Competitor Pricing").cast(DoubleType()),
    ).withColumn(
        "discount_pct",
        F.col("Discount").cast(DoubleType())
        / (F.col("Price").cast(DoubleType()) + F.lit(1e-6)),
    )

    return df


# ---------------------------------------------------------------------------
# Categorical encoding
# ---------------------------------------------------------------------------


def encode_categoricals(train: DataFrame, val: DataFrame):
    """
    Fit StringIndexer on train, transform both train and val.

    Uses handleInvalid="keep" so unseen categories in val become NaN;
    those rows are dropped afterwards.
    """
    encoded_train = train
    encoded_val = val

    for col_name in CAT_COLS:
        indexer = StringIndexer(
            inputCol=col_name,
            outputCol=col_name + "_enc",
            handleInvalid="keep",
            stringOrderType="frequencyDesc",
        )
        model = indexer.fit(encoded_train)

        encoded_train = model.transform(encoded_train).drop(col_name)
        encoded_val = model.transform(encoded_val).drop(col_name)

        # Rename _enc → original name so downstream column references work
        encoded_train = encoded_train.withColumnRenamed(col_name + "_enc", col_name)
        encoded_val = encoded_val.withColumnRenamed(col_name + "_enc", col_name)

    return encoded_train, encoded_val


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------


def run(input_uri: str, output_uri: str, execution_date: Optional[str] = None):
    log("=== Starting transform pipeline ===")
    log(f"Input : {input_uri}")
    log(f"Output: {output_uri}")
    if execution_date:
        log(f"Exec date param: {execution_date}")

    spark = SparkSession.builder.appName("EndfieldDemandTransform").getOrCreate()

    # 1. Read raw CSV -----------------------------------------------------------
    log("Reading raw CSV from S3 …")
    raw_df = spark.read.option("header", "true").csv(input_uri)
    log(f"Raw rows: {raw_df.count()}, columns: {len(raw_df.columns)}")

    # 2. Drop leaky columns -----------------------------------------------------
    log(f"Dropping columns: {DROP_COLS}")
    df = raw_df.drop(*DROP_COLS)

    # 3. Cast numeric columns to double -----------------------------------------
    for c in RAW_NUMERIC_COLS + [TARGET_COL]:
        df = df.withColumn(c, F.col(c).cast(DoubleType()))

    # 4. Parse date and cast numeric columns ------------------------------------
    df = df.withColumn(DATE_COL, F.to_date(F.col(DATE_COL), "yyyy-MM-dd"))

    # 5. Impute missing values --------------------------------------------------
    log("Imputing missing values …")
    df = impute_missing(df)
    log(f"After imputation: {df.count()} rows")

    # 6. Feature engineering ----------------------------------------------------
    log("Engineering features …")
    df = engineer_features(df)

    # 7. Drop rows with NaN in any feature or target column ---------------------
    all_required_cols = CAT_COLS + RAW_NUMERIC_COLS + ENGINEERED_COLS + [TARGET_COL]
    # Ensure all required cols exist before dropna
    available = [c for c in all_required_cols if c in df.columns]
    before_drop = df.count()
    df = df.dropna(subset=available)
    after_drop = df.count()
    log(f"Dropped {before_drop - after_drop} NaN rows ({after_drop} remaining)")

    # 8. Split by date ----------------------------------------------------------
    log(f"Splitting at {SPLIT_DATE} …")
    train_df_raw = df.filter(F.col(DATE_COL) < SPLIT_DATE).orderBy(DATE_COL)
    val_df_raw = df.filter(F.col(DATE_COL) >= SPLIT_DATE).orderBy(DATE_COL)
    log(f"Train (pre-encode): {train_df_raw.count()} rows")
    log(f"Val   (pre-encode): {val_df_raw.count()} rows")

    # 9. Encode categoricals ----------------------------------------------------
    log("Encoding categorical columns …")
    train_df, val_df = encode_categoricals(train_df_raw, val_df_raw)

    # Drop any rows where StringIndexer produced NaN (unseen categories in val)
    val_df = val_df.dropna(subset=CAT_COLS)
    log(f"Train (post-encode): {train_df.count()} rows")
    log(f"Val   (post-encode): {val_df.count()} rows")

    # 10. Assemble output column order ------------------------------------------
    # Target first, then encoded categoricals, then raw numeric, then engineered
    enc_cat_cols = [c for c in CAT_COLS if c in train_df.columns]
    raw_num_cols = [c for c in RAW_NUMERIC_COLS if c in train_df.columns]
    feat_cols = [c for c in ENGINEERED_COLS if c in train_df.columns]
    # Drop date column (not used for training)
    output_cols = [TARGET_COL] + enc_cat_cols + raw_num_cols + feat_cols
    missing = [c for c in output_cols if c not in train_df.columns]
    if missing:
        log(f"WARNING: columns not found in DataFrame: {missing}")
    output_cols = [c for c in output_cols if c in train_df.columns]

    log(f"Output schema ({len(output_cols)} columns): {output_cols[:5]}...")

    # 11. Write train / validation ---------------------------------------------
    log("Writing train split …")
    train_out = train_df.select(output_cols).coalesce(1)
    train_out.write.mode("overwrite").option("header", "false").csv(
        f"{output_uri.rstrip('/')}/train/"
    )

    log("Writing validation split …")
    val_out = val_df.select(output_cols).coalesce(1)
    val_out.write.mode("overwrite").option("header", "false").csv(
        f"{output_uri.rstrip('/')}/validation/"
    )

    log("=== Pipeline complete ===")
    spark.stop()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Endfield demand-forecasting ETL pipeline (PySpark on EMR)"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="S3 URI for raw CSV input (e.g. s3://bucket/data/)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="S3 URI for processed output (e.g. s3://bucket/processed/)",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Execution date override (YYYY-MM-DD), optional",
    )
    args = parser.parse_args()

    run(args.input, args.output, args.date)
