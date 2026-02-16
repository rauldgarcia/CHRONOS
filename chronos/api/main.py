from fastapi import FastAPI, HTTPException
import pandas as pd
from pathlib import Path
import os

app = FastAPI(title="CHRONOS API")

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.get("/data/{ticker}")
def get_ticker_data(ticker: str):
    """Get historical data for a ticker."""
    project_root = Path(__file__).parent.parent.parent
    csv_path = project_root / "chronos" / "data" / "raw_stock_prices.csv"

    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="Data not found, Run ingestion first.")

    df = pd.read_csv(csv_path)
    ticker_data = df[df['ticker'] == ticker]

    if len(ticker_data) == 0:
        raise HTTPException(status_code=404, detail=f"Ticker {ticker} not found")

    ticker_data = ticker_data.fillna(0)

    return {
        "ticker": ticker,
        "rows": len(ticker_data),
        "latest_price": float(ticker_data.iloc[-1]['Close']),
        "data": ticker_data.tail(10).to_dict(orient="records")
    }