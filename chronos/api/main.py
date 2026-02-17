from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy.orm import Session
from sqlalchemy import desc
from chronos.utils.db import get_db
from chronos.models.sql import StockData

app = FastAPI(title="CHRONOS API")

@app.get("/health")
def health_check():
    return {"status": "healthy", "database": "connected"}

@app.get("/data/{ticker}")
def get_ticker_data(ticker: str, limit: int = 10, db: Session = Depends(get_db)):
    """Get historical data for a ticker from Postgres."""

    results = (
        db.query(StockData)
        .filter(StockData.ticker == ticker)
        .order_by(desc(StockData.date))
        .limit(limit)
        .all()
    )

    if not results:
        raise HTTPException(status_code=404, detail=f"Ticker {ticker} not found in DB. Run ingestion.")

    data = [
        {
            "date": r.date,
            "close": r.close,
            "volume": r.volume,
            "open": r.open,
            "high": r.high,
            "low": r.low,
        } for r in results
    ]

    return {
        "ticker": ticker,
        "rows_returned": len(data),
        "latest_price": data[0]["close"],
        "data": data
    }