#!/usr/bin/env python3
"""
Standalone SageMaker training script.

Usage (manual / local invocation):
    python sagemaker_train.py \
        --train-data      s3://<bucket>/processed/train/ \
        --validation-data s3://<bucket>/processed/validation/ \
        --output          s3://<bucket>/output/ \
        --role-arn        arn:aws:iam::<account>:role/endfield-sagemaker-role-dev \
        --endpoint-name   endfield-forecast-endpoint \
        --region          us-east-1

This is the **alternative / backup** path to the Step Functions pipeline.
Step Functions creates training jobs directly via ASL, so this script is
useful for ad-hoc re-training, local testing, and debugging.
"""

import argparse
import logging
import sys
import time
from datetime import datetime

import boto3
import sagemaker
from sagemaker import image_uris

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
XGBOOST_VERSION = "1.7-1"
INSTANCE_TYPE = "ml.m5.xlarge"
INSTANCE_COUNT = 1
CONTENT_TYPE = "text/csv"
HYPERPARAMETERS = {
    "num_round": "100",
    "max_depth": "6",
    "eta": "0.2",
    "objective": "reg:linear",
    "eval_metric": "rmse",
}


# ---------------------------------------------------------------------------
# Parse CLI arguments
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an XGBoost model on SageMaker and update the real-time endpoint."
    )
    parser.add_argument(
        "--train-data",
        required=True,
        help="S3 URI for the training dataset (e.g. s3://bucket/processed/train/).",
    )
    parser.add_argument(
        "--validation-data",
        required=True,
        help="S3 URI for the validation dataset (e.g. s3://bucket/processed/validation/).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="S3 URI where model artifacts will be written (e.g. s3://bucket/output/).",
    )
    parser.add_argument(
        "--role-arn",
        required=True,
        help="IAM role ARN that SageMaker will assume during training.",
    )
    parser.add_argument(
        "--endpoint-name",
        required=True,
        help="Name of the SageMaker endpoint to update after training.",
    )
    parser.add_argument(
        "--region",
        default="us-east-1",
        help="AWS region (default: us-east-1).",
    )
    parser.add_argument(
        "--max-runtime",
        type=int,
        default=3600,
        help="Maximum training runtime in seconds (default: 3600).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Create a training job
# ---------------------------------------------------------------------------
def create_training_job(
    client: "boto3.client",
    job_name: str,
    role_arn: str,
    train_data: str,
    validation_data: str,
    output_path: str,
    max_runtime: int,
    region: str,
) -> str:
    """Launch an XGBoost training job and return the training job name."""
    image_uri = image_uris.retrieve(
        framework="xgboost",
        region=region,
        version=XGBOOST_VERSION,
    )
    log.info("Using training image: %s", image_uri)

    log.info("Creating training job: %s", job_name)
    client.create_training_job(
        TrainingJobName=job_name,
        AlgorithmSpecification={
            "TrainingImage": image_uri,
            "TrainingInputMode": "File",
        },
        HyperParameters=HYPERPARAMETERS,
        RoleArn=role_arn,
        InputDataConfig=[
            {
                "ChannelName": "train",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": train_data,
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
                "ContentType": CONTENT_TYPE,
            },
            {
                "ChannelName": "validation",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": validation_data,
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
                "ContentType": CONTENT_TYPE,
            },
        ],
        OutputDataConfig={"S3OutputPath": output_path},
        ResourceConfig={
            "InstanceType": INSTANCE_TYPE,
            "InstanceCount": INSTANCE_COUNT,
            "VolumeSizeInGB": 30,
        },
        StoppingCondition={"MaxRuntimeInSeconds": max_runtime},
    )
    log.info("Training job submitted.")
    return job_name


# ---------------------------------------------------------------------------
# Wait for the training job to complete
# ---------------------------------------------------------------------------
def wait_for_training(client: "boto3.client", job_name: str) -> str:
    """Poll training job status; return the S3 model artifact URI on success."""
    log.info("Waiting for training job '%s' to complete...", job_name)

    while True:
        resp = client.describe_training_job(TrainingJobName=job_name)
        status = resp["TrainingJobStatus"]
        log.info("  Status: %s", status)

        if status == "Completed":
            artifact = resp["ModelArtifacts"]["S3ModelArtifacts"]
            log.info("Training completed. Model artifact: %s", artifact)
            return artifact

        if status in ("Failed", "Stopped"):
            reason = resp.get("FailureReason", "Unknown")
            raise RuntimeError(f"Training job {job_name} {status}: {reason}")

        time.sleep(30)


# ---------------------------------------------------------------------------
# Create a SageMaker model
# ---------------------------------------------------------------------------
def create_model(
    client: "boto3.client",
    model_name: str,
    model_artifact: str,
    role_arn: str,
    image_uri: str,
) -> None:
    log.info("Creating model: %s", model_name)
    client.create_model(
        ModelName=model_name,
        PrimaryContainer={
            "Image": image_uri,
            "ModelDataUrl": model_artifact,
        },
        ExecutionRoleArn=role_arn,
    )
    log.info("Model created.")


# ---------------------------------------------------------------------------
# Create an endpoint configuration
# ---------------------------------------------------------------------------
def create_endpoint_config(
    client: "boto3.client",
    config_name: str,
    model_name: str,
) -> None:
    log.info("Creating endpoint configuration: %s", config_name)
    client.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": model_name,
                "InitialInstanceCount": INSTANCE_COUNT,
                "InstanceType": INSTANCE_TYPE,
            }
        ],
    )
    log.info("Endpoint configuration created.")


# ---------------------------------------------------------------------------
# Update (or create) the real-time endpoint
# ---------------------------------------------------------------------------
def deploy_endpoint(
    client: "boto3.client", endpoint_name: str, config_name: str
) -> None:
    """Create the endpoint if it does not exist, otherwise update it."""
    try:
        client.describe_endpoint(EndpointName=endpoint_name)
        log.info(
            "Endpoint '%s' exists. Updating to config '%s'...",
            endpoint_name,
            config_name,
        )
        client.update_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=config_name,
        )
    except client.exceptions.ClientError as e:
        if "Could not find endpoint" in str(e):
            log.info("Endpoint '%s' does not exist. Creating...", endpoint_name)
            client.create_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=config_name,
            )
        else:
            raise

    # Wait for endpoint to be ready
    while True:
        resp = client.describe_endpoint(EndpointName=endpoint_name)
        status = resp["EndpointStatus"]
        log.info("  Endpoint status: %s", status)
        if status == "InService":
            log.info("Endpoint ready.")
            return
        if status == "Failed":
            reason = resp.get("FailureReason", "Unknown")
            raise RuntimeError(f"Endpoint deployment failed: {reason}")
        time.sleep(15)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    job_name = f"endfield-train-{timestamp}"
    model_name = f"endfield-model-{timestamp}"
    config_name = f"endfield-config-{timestamp}"

    log.info("=== SageMaker Training Pipeline ===")
    log.info("  Train data:       %s", args.train_data)
    log.info("  Validation data:   %s", args.validation_data)
    log.info("  Output path:       %s", args.output)
    log.info("  Role ARN:          %s", args.role_arn)
    log.info("  Endpoint:          %s", args.endpoint_name)
    log.info("  Region:            %s", args.region)
    log.info("  Job name:          %s", job_name)
    log.info("===================================")

    # --- Clients ---
    sm_client = boto3.client("sagemaker", region_name=args.region)
    image_uri = image_uris.retrieve(
        framework="xgboost",
        region=args.region,
        version=XGBOOST_VERSION,
    )

    # --- Step 1: Train ---
    try:
        create_training_job(
            client=sm_client,
            job_name=job_name,
            role_arn=args.role_arn,
            train_data=args.train_data,
            validation_data=args.validation_data,
            output_path=args.output,
            max_runtime=args.max_runtime,
            region=args.region,
        )
        model_artifact = wait_for_training(sm_client, job_name)
    except Exception:
        log.exception("Training failed.")
        sys.exit(1)

    # --- Step 2: Create Model ---
    try:
        create_model(
            client=sm_client,
            model_name=model_name,
            model_artifact=model_artifact,
            role_arn=args.role_arn,
            image_uri=image_uri,
        )
    except Exception:
        log.exception("Model creation failed.")
        sys.exit(1)

    # --- Step 3: Create Endpoint Config ---
    try:
        create_endpoint_config(
            client=sm_client,
            config_name=config_name,
            model_name=model_name,
        )
    except Exception:
        log.exception("Endpoint config creation failed.")
        sys.exit(1)

    # --- Step 4: Deploy Endpoint (create or update) ---
    try:
        deploy_endpoint(
            client=sm_client,
            endpoint_name=args.endpoint_name,
            config_name=config_name,
        )
    except Exception:
        log.exception("Endpoint deployment failed.")
        sys.exit(1)

    log.info("=== Pipeline complete. Endpoint '%s' is live. ===", args.endpoint_name)


if __name__ == "__main__":
    main()
