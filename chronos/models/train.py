import os
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

if os.environ.get("AIRFLOW_UID"):
    DEFAULT_TRACKING_URI = "http://mlflow:5000"
else:
    DEFAULT_TRACKING_URI = "http://localhost:5000"

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def load_training_data(ticker: str) -> pd.DataFrame:
    """Load features from Postgres and create the target variable."""
    log.info(f"Loading feature data for {ticker} from DB...")

    inspector = inspect(engine)
    tables = inspector.get_table_names()
    log.info(f"Tables found in database: {tables}")

    if "stock_features" not in tables:
        log.error("The table 'stock_features' was not found in the database")
        raise ValueError("Feature Engineering task failed to create the table. Check PySpark logs.")

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


def train_models(ticker: str):
    """Train multiple models and log them to MLflow to select the champion."""
    mlflow.set_experiment("Chronos_Forecasting_V3")

    df = load_training_data(ticker)

    features = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "daily_return",
        "sma_20",
        "volatility_20",
    ]
    X = df[features].values
    y = df["target_close"].values

    split_idx = int(len(df) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Reshape for LSTM
    X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    log.info(f"Training set: {len(X_train)} rows | Test set: {len(X_test)} rows")
    log.info(
        f"GPU Available for TensorFlow: {len(tf.config.list_physical_devices('GPU')) > 0}"
    )

    predictions = {}
    best_model_name = None
    best_mse = float("inf")

    models = {
        "Ridge": Ridge(alpha=1.0),
        "XGBoost": XGBRegressor(n_estimators=100, learning_reate=0.1, random_state=42),
    }

    for model_name, model in models.items():
        with mlflow.start_run(run_name=f"{ticker}_{model_name}"):
            log.info(f"Training {model_name}...")

            model.fit(X_train, y_train)

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

    with mlflow.start_run(run_name=f"{ticker}_LSTM"):
        log.info("Training LSTM...")
        lstm = build_lstm_model((X_train_lstm.shape[1], X_train_lstm.shape[2]))
        lstm.fit(X_train_lstm, y_train, epochs=20, batch_size=32, verbose=0)

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

    log.success(f"Best model for {ticker}: {best_model_name} (MSE: {best_mse:.4f})")


if __name__ == "__main__":
    train_models("AAPL")
