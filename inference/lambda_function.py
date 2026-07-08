"""
Lambda inference handler for retail demand forecasting.

Receives a JSON payload from API Gateway, transforms it into a headerless CSV
row for the SageMaker XGBoost endpoint, and returns forecast predictions.
"""

import datetime
import json
import logging
import os

import boto3

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

sagemaker_runtime = boto3.client("sagemaker-runtime")
ENDPOINT_NAME = os.environ["ENDPOINT_NAME"]

# Feature schema (12 columns, no header, no target):
#   store_id_encoded, item_id_encoded, day_of_week, day_of_month, month, year,
#   is_weekend, lag_1_sales, lag_7_sales, rolling_mean_7, price, promo


def build_features(store_id, item_id, date_str, horizon):
    """
    Generate feature rows for each forecast day.

    Args:
        store_id: Store identifier (string, unused placeholder for encoding).
        item_id: Item identifier (string, unused placeholder for encoding).
        date_str: Reference date in YYYY-MM-DD format.
        horizon: Number of days to forecast ahead.

    Returns:
        list[list[float]]: One row per forecast day, each with 12 feature values.

    Notes:
        - store_id_encoded and item_id_encoded are set to 0.0 as placeholders;
          in production these require a mapping table from the training pipeline.
        - lag_1_sales, lag_7_sales, and rolling_mean_7 are set to 0.0 as
          placeholders; in production these must be populated from a historical
          data cache (e.g. DynamoDB or S3 lookup).
        - price is set to a constant 50.0 (placeholder average); promo is 0.
    """
    base_date = datetime.datetime.strptime(date_str, "%Y-%m-%d")
    rows = []

    for day_offset in range(1, horizon + 1):
        forecast_date = base_date + datetime.timedelta(days=day_offset)

        # Date-derived features
        day_of_week = float(forecast_date.weekday())  # 0=Monday .. 6=Sunday
        day_of_month = float(forecast_date.day)
        month = float(forecast_date.month)
        year = float(forecast_date.year)
        is_weekend = 1.0 if forecast_date.weekday() >= 5 else 0.0

        row = [
            0.0,  # store_id_encoded   – placeholder
            0.0,  # item_id_encoded    – placeholder
            day_of_week,
            day_of_month,
            month,
            year,
            is_weekend,
            0.0,  # lag_1_sales        – needs historical lookup
            0.0,  # lag_7_sales        – needs historical lookup
            0.0,  # rolling_mean_7     – needs historical lookup
            50.0,  # price              – placeholder average
            0.0,  # promo              – no promotion assumed
        ]
        rows.append(row)

    return rows


def get_forecast_dates(date_str, horizon):
    """Return the list of dates being forecast (one day after reference date)."""
    base_date = datetime.datetime.strptime(date_str, "%Y-%m-%d")
    return [base_date + datetime.timedelta(days=i + 1) for i in range(horizon)]


def _api_response(status_code, body_dict):
    """Build an API Gateway-compatible response dict."""
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
        },
        "body": json.dumps(body_dict),
    }


def lambda_handler(event, context):
    """
    API Gateway Lambda handler.

    Expects event['body'] to contain a JSON string with fields:
        store_id (str), item_id (str), date (str YYYY-MM-DD), horizon (int, default 1).

    Returns an API Gateway proxy response with forecast predictions.
    """
    try:
        # ── Parse request ──────────────────────────────────────
        try:
            body = json.loads(event["body"])
        except (KeyError, json.JSONDecodeError) as exc:
            logger.error("Failed to parse request body: %s", exc)
            return _api_response(400, {"error": "Invalid request body"})

        store_id = body.get("store_id", "")
        item_id = body.get("item_id", "")
        date_str = body.get("date", "")
        horizon = body.get("horizon", 1)

        # ── Validate inputs ────────────────────────────────────
        if not store_id:
            return _api_response(400, {"error": "store_id is required"})
        if not item_id:
            return _api_response(400, {"error": "item_id is required"})

        try:
            datetime.datetime.strptime(date_str, "%Y-%m-%d")
        except (ValueError, TypeError):
            return _api_response(400, {"error": "date must be in YYYY-MM-DD format"})

        if not isinstance(horizon, int) or horizon < 1 or horizon > 30:
            return _api_response(
                400, {"error": "horizon must be an integer between 1 and 30"}
            )

        # ── Build feature CSV ──────────────────────────────────
        feature_rows = build_features(store_id, item_id, date_str, horizon)
        csv_payload = "\n".join(
            ",".join(str(val) for val in row) for row in feature_rows
        )

        logger.info(
            "Invoking endpoint %s for store=%s item=%s date=%s horizon=%d",
            ENDPOINT_NAME,
            store_id,
            item_id,
            date_str,
            horizon,
        )

        # ── Call SageMaker endpoint ────────────────────────────
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="text/csv",
            Body=csv_payload,
        )

        result_str = response["Body"].read().decode("utf-8").strip()
        predictions = [float(val) for val in result_str.split("\n")]

        # ── Build response ─────────────────────────────────────
        forecast_dates = get_forecast_dates(date_str, horizon)
        predictions_list = [
            {"date": d.strftime("%Y-%m-%d"), "forecast": pred}
            for d, pred in zip(forecast_dates, predictions)
        ]

        return _api_response(
            200,
            {
                "store_id": store_id,
                "item_id": item_id,
                "predictions": predictions_list,
            },
        )

    except Exception as exc:
        logger.exception("Unhandled error during inference")
        return _api_response(
            500,
            {"error": f"Inference failed: {str(exc)}"},
        )
