from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy.orm import Session
from sqlalchemy import desc
from chronos.utils.db import get_db
from chronos.models.sql import StockData
from chronos.schemas.stock import TickerResponse
from chronos.utils.logger import log

app = FastAPI(title="CHRONOS API", version="0.1.0")

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
        raise HTTPException(status_code=404, detail=f"Ticker {ticker} not found in DB. Run ingestion.")

    response = {
        "ticker": ticker,
        "rows_returned": len(results),
        "latest_price": results[0].close,
        "data": results
    }

    return response