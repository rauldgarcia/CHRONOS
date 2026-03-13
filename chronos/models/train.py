import os
import pandas as pd
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import mlflow
import mlflow.sklearn
import mlflow.xgboost

from chronos.utils.db import engine
from chronos.utils.logger import log

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def load_training_data(ticker: str) -> pd.DataFrame:
    """Load features from Postgres and create the target variable."""
    log.info(f"Loading feature data for {ticker} from DB...")
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


def train_models(ticker: str):
    """Train multiple models and log them to MLflow to select the champion."""
    mlflow.set_experiment("Chronos_Forecasting_V2")

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
    X = df[features]
    y = df["target_close"]

    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    log.info(f"Training set: {len(X_train)} rows | Test set: {len(X_test)} rows")

    models = {
        "Ridge_Regression": Ridge(alpha=1.0),
        "XGBoost": XGBRegressor(n_estimators=100, learning_reate=0.1, random_state=42),
    }

    best_model_name = None
    best_mse = float("inf")

    for model_name, model in models.items():
        with mlflow.start_run(run_name=f"{ticker}_{model_name}"):
            log.info(f"Training {model_name}...")

            model.fit(X_train, y_train)

            preds = model.predict(X_test)
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

    log.success(f"Best model for {ticker}: {best_model_name} (MSE: {best_mse:.4f})")


if __name__ == "__main__":
    train_models("AAPL")
