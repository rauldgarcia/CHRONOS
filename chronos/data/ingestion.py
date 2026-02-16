import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def download_stock_data(ticker: str, years: int = 5) -> pd.DataFrame:
    """Download historical stock data from Yahoo Finance."""

    end_date = datetime.now()
    start_date = end_date - timedelta(days=years*365)

    df = yf.download(ticker, start=start_date, end=end_date)
    df['ticker'] = ticker
    df.reset_index(inplace=True)

    return df

if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "TSLA"]
    data = pd.concat([download_stock_data(t) for t in tickers])

    print(f"Downloaded {len(data)} rows")
    print(data.head())

    data.to_csv("chronos/data/raw_stock_prices.csv", index=False)