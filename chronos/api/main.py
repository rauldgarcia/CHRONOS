"""
CHRONOS API — Dual-Mode Forecasting Service
============================================
Operates in two modes controlled by the ENVIRONMENT environment variable:

  local (default):
    Uses local Postgres (via SQLAlchemy) for historical features and the local
    MLflow tracking server to resolve and load the champion model.
    Perfect for recruiters cloning the repo and running `docker-compose up`.

  production:
    100% stateless. No database required.
    - On first request for a ticker: downloads champion model files from GCS
      and caches them in memory for the lifetime of the container.
    - On /forecast: fetches live OHLCV data via yfinance, computes the same
      technical indicators as the PySpark training pipeline (daily_return,
      sma_20, volatility_20) in-memory, and runs inference.
    - Cost: $0. Cloud Run scales to zero when idle.
"""
import os
import pickle
import tempfile
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

import mlflow
import mlflow.pyfunc
import pandas as pd
import yfinance as yf
from fastapi import Depends, FastAPI, HTTPException
from loguru import logger as log
from sqlalchemy import desc
from sqlalchemy.orm import Session

from chronos.models.sql import StockData
from chronos.schemas.stock import ForecastResponse, TickerResponse
from chronos.utils.db import engine, get_db
from chronos.utils.logger import log  # noqa: F811

ENVIRONMENT: str = os.getenv("ENVIRONMENT", "local")
MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

# Feature columns — must exactly match the PySpark pipeline and train.py.
FEATURE_COLS: list[str] = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "daily_return",
    "sma_20",
    "volatility_20",
]


_model_cache: dict[str, dict] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if ENVIRONMENT == "production":
        log.info("🚀 PRODUCTION mode — CHRONOS API starting.")
        log.info("   Models: GCS (lazy-loaded per ticker on first request)")
        log.info("   Features: yfinance real-time")
    else:
        log.info(f"🔧 LOCAL mode — CHRONOS API starting. MLflow: {MLFLOW_TRACKING_URI}")
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    yield
    log.info("CHRONOS API shutting down.")


app = FastAPI(
    title="CHRONOS API",
    description=(
        "MLOps End-to-End Forecasting Platform. "
        "Dual-mode: `local` (Postgres + MLflow) or `production` (GCS + yfinance)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", tags=["ops"])
def health_check():
    """Returns API status and current serving mode."""
    return {
        "status": "healthy",
        "environment": ENVIRONMENT,
        "serving_mode": (
            "GCS artifact store + yfinance real-time features"
            if ENVIRONMENT == "production"
            else "local Postgres + MLflow tracking server"
        ),
    }


@app.get("/data/{ticker}", response_model=TickerResponse, tags=["data"])
def get_ticker_data(ticker: str, limit: int = 10, db: Session = Depends(get_db)):
    """
    [LOCAL MODE] Get historical OHLCV data for a ticker from the local Postgres DB.

    In production mode, this endpoint is disabled. Historical context is fetched
    in real-time via yfinance inside the /forecast endpoint.
    """
    if ENVIRONMENT == "production":
        raise HTTPException(
            status_code=503,
            detail=(
                "The /data endpoint is only available in local mode. "
                "In production, use /forecast which fetches live market data via yfinance."
            ),
        )

    log.info(f"[LOCAL] Fetching data for {ticker} (limit={limit})")
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
            status_code=404,
            detail=f"Ticker {ticker} not found in DB. Run the ingestion pipeline first.",
        )

    return {
        "ticker": ticker,
        "rows_returned": len(results),
        "latest_price": results[0].close,
        "data": results,
    }


@app.get("/forecast/{ticker}", response_model=ForecastResponse, tags=["inference"])
def get_forecast(ticker: str):
    """
    Get the next-day closing price forecast for a ticker.

    Behavior depends on ENVIRONMENT:
    - local:      Queries Postgres for the latest features, then loads the
                  champion model from the local MLflow tracking server.
    - production: Downloads live OHLCV data via yfinance, computes features
                  in-memory, and runs inference using the champion model
                  previously uploaded to GCS.
    """
    if ENVIRONMENT == "production":
        return _forecast_production(ticker)
    return _forecast_local(ticker)


# ──────────────────────────────────────────────────────────────────────────────
# Production helpers
# ──────────────────────────────────────────────────────────────────────────────


def _get_or_load_production_models(ticker: str) -> dict:
    """
    Lazy-load and cache all models for a given ticker from GCS.

    The first call for a ticker will download Ridge (.pkl), XGBoost (.pkl),
    and LSTM (.keras) from GCS. Subsequent calls hit the in-memory cache.
    """
    if ticker in _model_cache:
        log.debug(f"[CACHE HIT] Models for {ticker} already in memory.")
        return _model_cache[ticker]

    log.info(f"[PRODUCTION] Loading models for {ticker} from GCS (first request)...")

    from chronos.utils import gcs  # noqa: PLC0415
    import tensorflow as tf  # noqa: PLC0415

    prefix = f"models/{ticker}"
    tmpdir = tempfile.mkdtemp(prefix=f"chronos_{ticker}_")

    try:
        # Champion metadata
        champion_meta = gcs.download_json_from_gcs(f"{prefix}/champion.json")
        champion_type = champion_meta["champion_model"]
        log.info(
            f"[PRODUCTION] Champion for {ticker}: {champion_type} "
            f"(MSE: {champion_meta['mse']:.4f}, trained: {champion_meta['trained_at']})"
        )

        # Ridge
        ridge = pickle.loads(gcs.download_bytes_from_gcs(f"{prefix}/ridge.pkl"))

        # XGBoost
        xgboost_model = pickle.loads(
            gcs.download_bytes_from_gcs(f"{prefix}/xgboost.pkl")
        )

        # LSTM — download to temp file then load with Keras
        lstm_local_path = os.path.join(tmpdir, "lstm.keras")
        gcs.download_to_file(f"{prefix}/lstm.keras", lstm_local_path)
        lstm = tf.keras.models.load_model(lstm_local_path)

    except Exception as e:
        log.error(f"[PRODUCTION] Failed to load models for {ticker}: {e}")
        raise HTTPException(
            status_code=503,
            detail=(
                f"Could not load model for ticker '{ticker}' from GCS. "
                "Make sure you have run training with ENVIRONMENT=production first."
            ),
        ) from e

    cache_entry = {
        "champion_type": champion_type,
        "ridge": ridge,
        "xgboost": xgboost_model,
        "lstm": lstm,
        "champion_meta": champion_meta,
    }
    _model_cache[ticker] = cache_entry
    log.success(f"[PRODUCTION] Models for {ticker} loaded and cached in memory.")
    return cache_entry


def _compute_live_features(ticker: str) -> tuple[pd.DataFrame, datetime]:
    """
    Download the last 60 days of OHLCV data via yfinance and compute the same
    technical indicators as the PySpark training pipeline:
      - daily_return  : (close - close_prev) / close_prev   — pct_change()
      - sma_20        : 20-period rolling mean of close
      - volatility_20 : 20-period rolling std of close

    Returns:
        X_latest : DataFrame of shape (1, n_features) — the most recent row
        latest_date : datetime of that row
    """
    log.info(f"[PRODUCTION] Fetching live market data for {ticker} via yfinance...")

    raw = yf.download(
        ticker, period="60d", interval="1d", progress=False, auto_adjust=True
    )
    if raw.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No market data found for '{ticker}'. Verify the ticker symbol.",
        )

    # Normalize column names (handle both flat and MultiIndex columns)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.columns = [c.lower() for c in raw.columns]

    df = raw.reset_index()
    df.columns = [c.lower() for c in df.columns]

    # Compute technical indicators — mirrors the PySpark window functions
    df["daily_return"] = df["close"].pct_change()
    df["sma_20"] = df["close"].rolling(20).mean()
    df["volatility_20"] = df["close"].rolling(20).std()
    df = df.dropna()

    if df.empty:
        raise HTTPException(
            status_code=422,
            detail=f"Not enough data to compute features for '{ticker}' (need ≥20 trading days).",
        )

    latest = df.iloc[-1]
    latest_date: datetime = pd.Timestamp(latest["date"]).to_pydatetime()
    return df[FEATURE_COLS].iloc[[-1]], latest_date  # shape: (1, n_features)


def _forecast_production(ticker: str) -> dict:
    """Production inference: GCS champion model + yfinance real-time features."""
    models = _get_or_load_production_models(ticker)
    X_input, latest_date = _compute_live_features(ticker)
    X_arr = X_input.values  # shape: (1, n_features)

    champion_type = models["champion_type"]
    log.info(f"[PRODUCTION] Running {champion_type} inference for {ticker}")

    if champion_type == "Ridge":
        prediction = float(models["ridge"].predict(X_arr)[0])

    elif champion_type == "XGBoost":
        prediction = float(models["xgboost"].predict(X_arr)[0])

    elif champion_type == "LSTM":
        X_lstm = X_arr.reshape((1, 1, X_arr.shape[1]))
        prediction = float(models["lstm"].predict(X_lstm, verbose=0)[0][0])

    elif champion_type == "Ensemble":
        p_ridge = models["ridge"].predict(X_arr)
        p_xgb = models["xgboost"].predict(X_arr)
        X_lstm = X_arr.reshape((1, 1, X_arr.shape[1]))
        p_lstm = models["lstm"].predict(X_lstm, verbose=0).flatten()
        prediction = float((p_ridge + p_xgb + p_lstm)[0] / 3.0)

    else:
        raise HTTPException(
            status_code=500, detail=f"Unknown champion model type: '{champion_type}'"
        )

    meta = models["champion_meta"]
    target_date = (latest_date + timedelta(days=1)).replace(tzinfo=None)

    return {
        "ticker": ticker,
        "target_date": target_date,
        "predicted_close": prediction,
        "model_used": champion_type,
        "model_run_id": f"gs://{meta['gcs_bucket']}/models/{ticker}/champion.json",
        "environment": "production",
    }


def _forecast_local(ticker: str) -> dict:
    """Local inference: Postgres features + MLflow champion model."""
    log.info(f"[LOCAL] Requesting forecast for {ticker}")

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
        raise HTTPException(
            status_code=404,
            detail=f"No features found for '{ticker}'. Run the feature engineering pipeline first.",
        )

    latest_date = df_latest.iloc[0]["date"]
    X_input = df_latest[FEATURE_COLS]

    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("Chronos_Forecasting_V3")

    if not experiment:
        raise HTTPException(
            status_code=500,
            detail="MLflow experiment 'Chronos_Forecasting_V3' not found. Run training first.",
        )

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"params.ticker = '{ticker}'",
        order_by=["metrics.mse ASC"],
        max_results=1,
    )

    if not runs:
        raise HTTPException(
            status_code=404,
            detail=f"No trained models found for '{ticker}'. Run training first.",
        )

    best_run = runs[0]
    run_id = best_run.info.run_id
    model_name = best_run.data.tags.get("mlflow.runName", "Unknown_Model")
    log.info(f"[LOCAL] Champion selected: {model_name} (Run ID: {run_id})")

    model_uri = f"runs:/{run_id}/model"
    try:
        model = mlflow.pyfunc.load_model(model_uri)
        if "LSTM" in model_name:
            X_input_arr = X_input.values.reshape((1, 1, len(FEATURE_COLS)))
            prediction = model.predict(X_input_arr)[0][0]
        else:
            prediction = model.predict(X_input)[0]
    except Exception as e:
        log.error(f"[LOCAL] Inference failed: {e}")
        raise HTTPException(status_code=500, detail="Inference failed. Check MLflow server logs.")

    target_date = latest_date + timedelta(days=1)
    return {
        "ticker": ticker,
        "target_date": target_date,
        "predicted_close": float(prediction),
        "model_used": model_name,
        "model_run_id": run_id,
        "environment": "local",
    }
