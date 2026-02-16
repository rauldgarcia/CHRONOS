import pytest
from chronos.data.ingestion import download_stock_data

def test_download_stock_data_returns_dataframe():
    """Test that download returns a pandas DataFrame."""
    df = download_stock_data("AAPL", years=1)

    assert df is not None
    assert len(df) > 0


def test_download_stock_data_has_required_columns():
    """Test that DataFrame has expected columns."""
    df = download_stock_data("APPL", years=1)

    required_columns = ['Date', 'Close', 'ticker']
    for col in required_columns:
        assert col in df.columns


def test_download_stock_data_ticker_column():
    """Test that ticker column is correctly set."""
    df = download_stock_data("MSFT", years=1)

    assert df['ticker'].unique()[0] == "MSFT"


def test_download_stock_data_minimum_rows():
    """Test that at least 200 trading days are downloaded for 1 year."""
    df = download_stock_data("TSLA", years=1)

    assert len(df) >= 200