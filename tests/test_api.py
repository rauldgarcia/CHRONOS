import pytest
from fastapi.testclient import TestClient
from chronos.api.main import app

client = TestClient(app)

def test_health_endpoint():
    """Test that health endpoint returns 200."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_get_ticker_data_success():
    """Test getting data for existing ticker."""
    import os
    csv_path = "chronos/data/raw_stock_prices.csv"

    if not os.path.exists(csv_path):
        pytest.skip("CSV file not found. Run ingestion script first.")

    response = client.get("/data/AAPL")

    assert response.status_code == 200
    data = response.json()

    assert "ticker" in data
    assert "rows" in data
    assert "latest_price" in data
    assert data["ticker"] == "AAPL"


def test_get_ticker_data_not_found():
    """Test requesting data for non-existent ticker."""
    response = client.get("/data/INVALID")

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()