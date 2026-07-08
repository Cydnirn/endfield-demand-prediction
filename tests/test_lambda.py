"""
Unit tests for the Lambda inference handler.

Uses pytest for test discovery and moto for mocking AWS SageMaker runtime.
"""

import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# Ensure the project root is on sys.path so we can import the inference module.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Set environment variable before importing the module-under-test.
os.environ["ENDPOINT_NAME"] = "test-endpoint"

import inference.lambda_function as lambda_module

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def ensure_endpoint_env():
    """Make sure ENDPOINT_NAME is always available."""
    os.environ["ENDPOINT_NAME"] = "test-endpoint"


@pytest.fixture
def valid_event():
    """A minimal valid request with horizon=1."""
    return {
        "body": json.dumps(
            {
                "store_id": "S001",
                "item_id": "I123",
                "date": "2023-01-15",
                "horizon": 1,
            }
        )
    }


@pytest.fixture
def multi_day_event():
    """A valid request with horizon=3."""
    return {
        "body": json.dumps(
            {
                "store_id": "S001",
                "item_id": "I123",
                "date": "2023-01-15",
                "horizon": 3,
            }
        )
    }


def _mock_invoke_response(predictions):
    """
    Build a fake SageMaker invoke_endpoint response.

    The Body is a StreamingBody-like object whose .read() returns CSV bytes.
    """
    body = MagicMock()
    body.read.return_value = "\n".join(str(p) for p in predictions).encode("utf-8")
    return {"Body": body}


# ---------------------------------------------------------------------------
# Test: build_features helper
# ---------------------------------------------------------------------------


def test_feature_csv_format():
    """Verify that build_features produces correct number of columns and no header."""
    rows = lambda_module.build_features(
        store_id="S001", item_id="I123", date_str="2023-01-15", horizon=2
    )

    # horizon=2 → 2 rows
    assert len(rows) == 2

    for row in rows:
        # 12 columns as defined in the feature schema
        assert len(row) == 12
        assert all(isinstance(val, float) for val in row)

    # Convert to CSV and verify format
    csv_payload = "\n".join(",".join(str(val) for val in row) for row in rows)
    lines = csv_payload.split("\n")
    assert len(lines) == 2

    # Verify no header row — first line should start with floats, not "store_id"
    first_val = lines[0].split(",")[0]
    assert first_val not in ("store_id_encoded", "store_id")
    # First two columns are the 0.0 placeholders
    assert first_val == "0.0"


# ---------------------------------------------------------------------------
# Test: valid requests
# ---------------------------------------------------------------------------


def test_valid_request_single_day(valid_event):
    """horizon=1 → single prediction with correct structure."""
    with patch.object(
        lambda_module.sagemaker_runtime, "invoke_endpoint"
    ) as mock_invoke:
        mock_invoke.return_value = _mock_invoke_response([120.5])

        response = lambda_module.lambda_handler(valid_event, None)

    assert response["statusCode"] == 200
    assert response["headers"]["Content-Type"] == "application/json"
    assert response["headers"]["Access-Control-Allow-Origin"] == "*"

    body = json.loads(response["body"])
    assert body["store_id"] == "S001"
    assert body["item_id"] == "I123"
    assert len(body["predictions"]) == 1
    assert body["predictions"][0]["date"] == "2023-01-16"
    assert body["predictions"][0]["forecast"] == 120.5

    # Verify the CSV payload sent to SageMaker has 12 columns and no header
    call_args = mock_invoke.call_args
    csv_sent = call_args[1]["Body"]
    lines = csv_sent.strip().split("\n")
    assert len(lines) == 1  # one row for horizon=1
    assert len(lines[0].split(",")) == 12


def test_valid_request_multi_day(multi_day_event):
    """horizon=3 → three predictions with sequential dates."""
    with patch.object(
        lambda_module.sagemaker_runtime, "invoke_endpoint"
    ) as mock_invoke:
        mock_invoke.return_value = _mock_invoke_response([98.2, 115.0, 102.7])

        response = lambda_module.lambda_handler(multi_day_event, None)

    assert response["statusCode"] == 200

    body = json.loads(response["body"])
    assert body["store_id"] == "S001"
    assert body["item_id"] == "I123"
    assert len(body["predictions"]) == 3

    # Dates should be 2023-01-16, 2023-01-17, 2023-01-18
    expected_dates = ["2023-01-16", "2023-01-17", "2023-01-18"]
    expected_forecasts = [98.2, 115.0, 102.7]
    for i, pred in enumerate(body["predictions"]):
        assert pred["date"] == expected_dates[i]
        assert pred["forecast"] == expected_forecasts[i]


# ---------------------------------------------------------------------------
# Test: input validation errors (400)
# ---------------------------------------------------------------------------


def test_missing_store_id():
    """Missing store_id returns 400."""
    event = {
        "body": json.dumps({"item_id": "I123", "date": "2023-01-15", "horizon": 1})
    }
    response = lambda_module.lambda_handler(event, None)
    assert response["statusCode"] == 400
    body = json.loads(response["body"])
    assert "store_id" in body["error"]


def test_missing_item_id():
    """Missing item_id returns 400."""
    event = {
        "body": json.dumps({"store_id": "S001", "date": "2023-01-15", "horizon": 1})
    }
    response = lambda_module.lambda_handler(event, None)
    assert response["statusCode"] == 400
    body = json.loads(response["body"])
    assert "item_id" in body["error"]


def test_invalid_date_format():
    """Non-YYYY-MM-DD date returns 400."""
    event = {
        "body": json.dumps(
            {
                "store_id": "S001",
                "item_id": "I123",
                "date": "15/01/2023",
                "horizon": 1,
            }
        )
    }
    response = lambda_module.lambda_handler(event, None)
    assert response["statusCode"] == 400
    body = json.loads(response["body"])
    assert "date" in body["error"].lower()


@pytest.mark.parametrize(
    "bad_horizon",
    [
        0,  # below range
        -1,  # negative
        31,  # above range
        100,  # way above range
        "abc",  # non-integer
    ],
)
def test_invalid_horizon(bad_horizon):
    """Out-of-range or non-integer horizon returns 400."""
    event = {
        "body": json.dumps(
            {
                "store_id": "S001",
                "item_id": "I123",
                "date": "2023-01-15",
                "horizon": bad_horizon,
            }
        )
    }
    response = lambda_module.lambda_handler(event, None)
    assert response["statusCode"] == 400
    body = json.loads(response["body"])
    assert "horizon" in body["error"]


# ---------------------------------------------------------------------------
# Test: SageMaker error (500)
# ---------------------------------------------------------------------------


def test_sagemaker_error(valid_event):
    """invoke_endpoint exception → 500."""
    with patch.object(
        lambda_module.sagemaker_runtime, "invoke_endpoint"
    ) as mock_invoke:
        mock_invoke.side_effect = Exception("SageMaker endpoint timeout")

        response = lambda_module.lambda_handler(valid_event, None)

    assert response["statusCode"] == 500
    body = json.loads(response["body"])
    assert "Inference failed" in body["error"]
    assert "SageMaker endpoint timeout" in body["error"]


# ---------------------------------------------------------------------------
# Test: default horizon
# ---------------------------------------------------------------------------


def test_default_horizon():
    """When horizon is omitted, default to 1."""
    event = {
        "body": json.dumps(
            {
                "store_id": "S001",
                "item_id": "I123",
                "date": "2023-01-15",
            }
        )
    }
    with patch.object(
        lambda_module.sagemaker_runtime, "invoke_endpoint"
    ) as mock_invoke:
        mock_invoke.return_value = _mock_invoke_response([42.0])

        response = lambda_module.lambda_handler(event, None)

    assert response["statusCode"] == 200
    body = json.loads(response["body"])
    assert len(body["predictions"]) == 1
