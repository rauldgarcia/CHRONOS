import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from tenacity import retry, stop_after_attempt, wait_exponential
from sqlalchemy.dialects.postgresql import insert
from chronos.utils.db import engine, Base, SessionLocal
from chronos.models.sql import StockData
from chronos.utils.logger import log

Base.metadata.create_all(bind=engine)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def download_stock_data(ticker: str, years: int = 5) -> pd.DataFrame:
    """Download historical stock data from Yahoo Finance with retries."""
    log.info(f"Downloading data for ticker: {ticker} (Last {years} years)")

    end_date = datetime.now()
    start_date = end_date - timedelta(days=years*365)

    try:
    # Set Auto_adjust=True to obtain adjusted prices (splits/dividends)
        df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, multi_level_index=False)

    except Exception as e:
        log.error(f"Failed to download from YFinance: {e}")
        raise

    if df.empty:
        log.warning(f"No data found for {ticker}")
        raise ValueError(f"No data found for {ticker}")

    df['ticker'] = ticker
    df.reset_index(inplace=True)

    df = df.rename(columns={
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume"
    })

    log.debug(f"Downloaded {len(df)} rows for {ticker}")
    return df[['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']]

def save_to_postgres(df: pd.DataFrame):
    """
    Upsert data tp PostgreSQL efficiently.
    If ticker+date exists, do nothing (or update).
    """
    log.info(f"Saving {len(df)} rows to DB...")
    
    data = df.to_dict(orient='records')
    try:
        with SessionLocal() as session:
            stmt = insert(StockData).values(data)
            stmt = stmt.on_conflict_do_nothing(index_elements=['ticker', 'date'])

            session.execute(stmt)
            session.commit()
        log.success("Data upserted successfully.")
    
    except Exception as e:
        log.critical(f"Database Transaction Failed: {e}")
        raise

if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "TSLA"]

    for ticker in tickers:
        try:
            df = download_stock_data(ticker)
            save_to_postgres(df)
        except Exception as e:
            log.exception(f"Process failed for {ticker}: {e}")