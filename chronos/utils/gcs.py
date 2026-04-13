"""
GCS utilities for CHRONOS.

Used exclusively in production mode (ENVIRONMENT=production) to upload and
download champion model artifacts from Google Cloud Storage.

In local mode (ENVIRONMENT=local), these functions are never called, so GCP
credentials are not required for local development.

GCS bucket structure:
    gs://<GCS_BUCKET_NAME>/
    └── models/
        └── {ticker}/
            ├── champion.json         # Metadata: which model won and its MSE
            ├── ridge.pkl             # Serialized Ridge model
            ├── xgboost.pkl           # Serialized XGBoost model
            ├── lstm.keras            # Keras-format LSTM model
            └── latest_features.json  # Last training row (reference)
"""

import json
import os

from loguru import logger as log

GCS_BUCKET_NAME: str = os.getenv(
    "GCS_BUCKET_NAME", "chronos-mlflow-artifacts-evidentedesarrollo"
)
GCS_MODEL_PREFIX: str = os.getenv("GCS_MODEL_PREFIX", "models/")


def _get_client():
    """Lazy import of google-cloud-storage to keep local mode lightweight."""
    from google.cloud import storage  # noqa: PLC0415

    return storage.Client()


def upload_file_to_gcs(local_path: str, blob_name: str) -> str:
    """Upload a local file to GCS. Returns the full gs:// URI."""
    client = _get_client()
    bucket = client.bucket(GCS_BUCKET_NAME)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path)
    uri = f"gs://{GCS_BUCKET_NAME}/{blob_name}"
    log.info(f"GCS upload: {local_path} → {uri}")
    return uri


def upload_json_to_gcs(data: dict, blob_name: str) -> str:
    """Serialize a Python dict as JSON and upload it to GCS."""
    client = _get_client()
    bucket = client.bucket(GCS_BUCKET_NAME)
    blob = bucket.blob(blob_name)
    blob.upload_from_string(
        json.dumps(data, default=str), content_type="application/json"
    )
    uri = f"gs://{GCS_BUCKET_NAME}/{blob_name}"
    log.info(f"GCS JSON upload → {uri}")
    return uri


def download_bytes_from_gcs(blob_name: str) -> bytes:
    """Download a GCS blob and return its raw bytes."""
    client = _get_client()
    bucket = client.bucket(GCS_BUCKET_NAME)
    blob = bucket.blob(blob_name)
    return blob.download_as_bytes()


def download_json_from_gcs(blob_name: str) -> dict:
    """Download and deserialize a JSON blob from GCS."""
    return json.loads(download_bytes_from_gcs(blob_name))


def download_to_file(blob_name: str, local_path: str) -> None:
    """Download a GCS blob to a local file path."""
    client = _get_client()
    bucket = client.bucket(GCS_BUCKET_NAME)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(local_path)
    log.info(f"GCS download: gs://{GCS_BUCKET_NAME}/{blob_name} → {local_path}")
