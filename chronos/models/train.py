import os
import pickle
import tempfile
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import inspect
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import mlflow
import mlflow.sklearn
import mlflow.xgboost

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input

from chronos.utils.db import engine
from chronos.utils.logger import log

ENVIRONMENT = os.getenv("ENVIRONMENT", "local")

if os.environ.get("AIRFLOW_UID"):
    DEFAULT_TRACKING_URI = "http://mlflow:5000"
else:
    DEFAULT_TRACKING_URI = "http://localhost:5000"

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Feature columns — must match the PySpark feature pipeline exactly.
FEATURE_COLS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "daily_return",
    "sma_20",
    "volatility_20",
]


def load_training_data(ticker: str) -> pd.DataFrame:
    """Load features from Postgres and create the target variable."""
    log.info(f"Loading feature data for {ticker} from DB...")

    inspector = inspect(engine)
    tables = inspector.get_table_names()
    log.info(f"Tables found in database: {tables}")

    if "stock_features" not in tables:
        log.error("The table 'stock_features' was not found in the database")
        raise ValueError(
            "Feature Engineering task failed to create the table. Check PySpark logs."
        )

    query = f"SELECT * FROM stock_features WHERE ticker = '{ticker}' ORDER BY date"

    raw_conn = engine.raw_connection()
    try:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            df = pd.read_sql(query, raw_conn)

    finally:
        raw_conn.close()

    if df.empty:
        raise ValueError(f"No data found for ticker {ticker}")

    df["target_close"] = df["close"].shift(-1)
    df = df.dropna()
    return df


def build_lstm_model(input_shape):
    """Build and compile a basic LSTM neural network."""
    model = Sequential(
        [
            Input(shape=input_shape),
            LSTM(50, activation="relu"),
            Dense(25, activation="relu"),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    return model


def _upload_models_to_gcs(
    ticker: str,
    model_objects: dict,
    champion_name: str,
    champion_mse: float,
    training_df: pd.DataFrame,
) -> None:
    """
    Serialize all trained models and upload them to GCS for production serving.

    File layout in GCS:
        models/{ticker}/
          ├── champion.json         → metadata about the winning model
          ├── ridge.pkl             → serialized Ridge model
          ├── xgboost.pkl           → serialized XGBoost model
          ├── lstm.keras            → Keras-format LSTM model
          └── latest_features.json → last row of training features (reference)

    Note: All models are uploaded regardless of which one is champion. The API
    reads champion.json to decide which model(s) to use when serving.
    """
    from chronos.utils.gcs import (  # noqa: PLC0415
        GCS_BUCKET_NAME,
        upload_file_to_gcs,
        upload_json_to_gcs,
    )

    log.info(f"[GCS] Uploading models for {ticker} to GCS...")
    prefix = f"models/{ticker}"

    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Ridge ─────────────────────────────────────────────────────────
        ridge_path = os.path.join(tmpdir, "ridge.pkl")
        with open(ridge_path, "wb") as f:
            pickle.dump(model_objects["Ridge"], f)
        upload_file_to_gcs(ridge_path, f"{prefix}/ridge.pkl")

        # 2. XGBoost ───────────────────────────────────────────────────────
        xgb_path = os.path.join(tmpdir, "xgboost.pkl")
        with open(xgb_path, "wb") as f:
            pickle.dump(model_objects["XGBoost"], f)
        upload_file_to_gcs(xgb_path, f"{prefix}/xgboost.pkl")

        # 3. LSTM (.keras single-file format) ──────────────────────────────
        lstm_path = os.path.join(tmpdir, "lstm.keras")
        model_objects["LSTM"].save(lstm_path)
        upload_file_to_gcs(lstm_path, f"{prefix}/lstm.keras")

        # 4. champion.json ─────────────────────────────────────────────────
        champion_meta = {
            "ticker": ticker,
            "champion_model": champion_name,
            "mse": champion_mse,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "feature_columns": FEATURE_COLS,
            "gcs_bucket": GCS_BUCKET_NAME,
        }
        upload_json_to_gcs(champion_meta, f"{prefix}/champion.json")

        # 5. latest_features.json ──────────────────────────────────────────
        # Save the last feature row from training data as reference.
        # In production, the API computes live features via yfinance instead.
        last_row = training_df[FEATURE_COLS].iloc[-1].to_dict()
        last_date_col = (
            training_df["date"].iloc[-1] if "date" in training_df.columns else "unknown"
        )
        features_meta = {
            "ticker": ticker,
            "date": str(last_date_col),
            "features": {k: float(v) for k, v in last_row.items()},
        }
        upload_json_to_gcs(features_meta, f"{prefix}/latest_features.json")

    log.success(
        f"[GCS] All models for {ticker} uploaded. Champion: {champion_name} (MSE: {champion_mse:.4f})"
    )


def train_models(ticker: str):
    """
    Train Ridge, XGBoost, LSTM, and a Voting Ensemble. Log all runs to MLflow.
    The champion (lowest MSE) is selected and, if ENVIRONMENT=production,
    all models are serialized and uploaded to GCS for stateless Cloud Run serving.
    """
    mlflow.set_experiment("Chronos_Forecasting_V3")

    df = load_training_data(ticker)

    X = df[FEATURE_COLS].values
    y = df["target_close"].values

    split_idx = int(len(df) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Reshape for LSTM
    X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    log.info(f"Training set: {len(X_train)} rows | Test set: {len(X_test)} rows")
    log.info(
        f"GPU available for TensorFlow: {len(tf.config.list_physical_devices('GPU')) > 0}"
    )

    predictions = {}
    model_objects = {}  # Kept in memory for optional GCS export
    best_model_name = None
    best_mse = float("inf")

    # ── Sklearn / XGBoost models ───────────────────────────────────────────
    models = {
        "Ridge": Ridge(alpha=1.0),
        "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    }

    for model_name, model in models.items():
        with mlflow.start_run(run_name=f"{ticker}_{model_name}"):
            log.info(f"Training {model_name}...")

            model.fit(X_train, y_train)
            model_objects[model_name] = model

            preds = model.predict(X_test)
            predictions[model_name] = preds

            mse = mean_squared_error(y_test, preds)
            mae = mean_absolute_error(y_test, preds)

            mlflow.log_param("ticker", ticker)
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("mae", mae)

            if isinstance(model, Ridge):
                mlflow.sklearn.log_model(model, "model")
            else:
                mlflow.xgboost.log_model(model, "model")

            log.info(f"{model_name} -> MSE: {mse:.4f}, MAE: {mae:.4f}")

            if mse < best_mse:
                best_mse = mse
                best_model_name = model_name

    # ── LSTM ───────────────────────────────────────────────────────────────
    with mlflow.start_run(run_name=f"{ticker}_LSTM"):
        log.info("Training LSTM...")
        lstm = build_lstm_model((X_train_lstm.shape[1], X_train_lstm.shape[2]))
        lstm.fit(X_train_lstm, y_train, epochs=20, batch_size=32, verbose=0)
        model_objects["LSTM"] = lstm

        preds = lstm.predict(X_test_lstm, verbose=0).flatten()
        predictions["LSTM"] = preds

        mse = mean_squared_error(y_test, preds)
        mlflow.log_param("ticker", ticker)
        mlflow.log_metric("mse", mse)
        mlflow.tensorflow.log_model(lstm, "model")
        log.info(f"LSTM -> MSE: {mse:.4f}")
        if mse < best_mse:
            best_mse = mse
            best_model_name = "LSTM"

    # ── Voting Ensemble ────────────────────────────────────────────────────
    with mlflow.start_run(run_name=f"{ticker}_Ensemble"):
        log.info("Evaluating Voting Ensemble...")
        ensemble_preds = (
            predictions["Ridge"] + predictions["XGBoost"] + predictions["LSTM"]
        ) / 3.0

        mse = mean_squared_error(y_test, ensemble_preds)
        mae = mean_absolute_error(y_test, ensemble_preds)

        mlflow.log_param("ticker", ticker)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("mae", mae)
        log.info(f"Ensemble -> MSE: {mse:.4f}")
        if mse < best_mse:
            best_mse = mse
            best_model_name = "Ensemble"

    log.success(f"Champion for {ticker}: {best_model_name} (MSE: {best_mse:.4f})")

    # ── GCS Export (production mode or explicitly enabled) ─────────────────
    # Triggered by ENVIRONMENT=production OR by setting GCS_BUCKET_NAME manually.
    # This allows a recruiter to also trigger a GCS upload from local if desired.
    if ENVIRONMENT == "production" or os.getenv("GCS_BUCKET_NAME"):
        log.info("[GCS] Production mode detected — exporting models to GCS...")
        _upload_models_to_gcs(ticker, model_objects, best_model_name, best_mse, df)
    else:
        log.info(
            "[LOCAL] Skipping GCS upload. Set ENVIRONMENT=production or GCS_BUCKET_NAME to enable."
        )


if __name__ == "__main__":
    train_models("AAPL")
