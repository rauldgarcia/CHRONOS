import os
import pandas as pd
import mlflow
from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy.orm import Session
from sqlalchemy import desc
from chronos.utils.db import get_db, engine
from chronos.models.sql import StockData
from chronos.schemas.stock import TickerResponse, ForecastResponse
from chronos.utils.logger import log

app = FastAPI(title="CHRONOS API", version="0.1.0")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


@app.get("/health")
def health_check():
    log.debug("Health check requested")
    return {"status": "healthy", "database": "connected"}


@app.get("/data/{ticker}", response_model=TickerResponse)
def get_ticker_data(ticker: str, limit: int = 10, db: Session = Depends(get_db)):
    """Get historical data for a ticker from Postgres."""
    log.info(f"Fetching data for {ticker} (limit={limit})")

    results = (
        db.query(StockData)
        .filter(StockData.ticker == ticker)
        .order_by(desc(StockData.date))
        .limit(limit)
        .all()
    )

    if not results:
        log.warning(f"Ticker {ticker} not found in DB")
        raise HTTPException(
            status_code=404, detail=f"Ticker {ticker} not found in DB. Run ingestion."
        )

    response = {
        "ticker": ticker,
        "rows_returned": len(results),
        "latest_price": results[0].close,
        "data": results,
    }

    return response


@app.get("/forecast/{ticker}", response_model=ForecastResponse)
def get_forecast(ticker: str):
    """
    Get the next-day forecast for a ticker using the Champion model from MLflow.
    """
    log.info(f"Requesting forecast for {ticker}")

    query = f"""
        SELECT date, open, high, low, close, volume, daily_return, sma_20, volatility_20
        FROM stock_features
        WHERE ticker = '{ticker}'
        ORDER BY date DESC LIMIT 1
    """
    raw_conn = engine.raw_connection()
    try:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            df_latest = pd.read_sql(query, raw_conn)
    finally:
        raw_conn.close()

    if df_latest.empty:
        raise HTTPException(status_code=404, detail=f"No features found for {ticker}")

    latest_date = df_latest.iloc[0]["date"]

    feature_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "daily_return",
        "sma_20",
        "volatility_20",
    ]
    X_input = df_latest[feature_cols]

    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("Chronos_Forecasting_V3")

    if not experiment:
        raise HTTPException(status_code=500, detail="Experiment not found in MLflow")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"params.ticker = '{ticker}'",
        order_by=["metrics.mse ASC"],
        max_results=1,
    )

    if not runs:
        raise HTTPException(
            status_code=404, detail=f"No trained models found for {ticker}"
        )

    best_run = runs[0]
    run_id = best_run.info.run_id
    model_name = best_run.data.tags.get("mlflow.runName", "Unknown_Model")

    log.info(f"Champion selected: {model_name} (Run ID: {run_id})")

    model_uri = f"runs:/{run_id}/model"

    try:
        model = mlflow.pyfunc.load_model(model_uri)

        if "LSTM" in model_name:
            X_input_array = X_input.values.reshape((1, 1, len(feature_cols)))
            prediction = model.predict(X_input_array)[0][0]
        else:
            prediction = model.predict(X_input)[0]

    except Exception as e:
        log.error(f"Error during inference: {e}")
        raise HTTPException(status_code=500, detail="Inference failed")

    from datetime import timedelta

    target_date = latest_date + timedelta(days=1)

    return {
        "ticker": ticker,
        "target_date": target_date,
        "predicted_close": float(prediction),
        "model_used": model_name,
        "model_run_id": run_id,
    }
