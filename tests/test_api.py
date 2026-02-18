import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock
from datetime import datetime
from chronos.api.main import app
from chronos.utils.db import get_db
from chronos.models.sql import StockData

mock_session = MagicMock()

def override_get_db():
    try:
        yield mock_session
    finally:
        pass

app.dependency_overrides[get_db] = override_get_db
client = TestClient(app)

def test_health_endpoint():
    """Test that health endpoint returns 200."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "database": "connected"}


def test_get_ticker_data_success():
    """Test getting data for existing ticker using Mocks (No DB needed)."""
    mock_row = StockData(
        ticker="AAPL",
        date=datetime(2024, 1, 1),
        open=150.0,
        high=155.0,
        low=149.0,
        close=153.0,
        volume=1000000.0
    )
    (mock_session.query.return_value
     .filter.return_value
     .order_by.return_value
     .limit.return_value
     .all.return_value) = [mock_row]
    
    response = client.get("/data/AAPL")

    assert response.status_code == 200
    data = response.json()

    assert data["ticker"] == "AAPL"
    assert data["rows_returned"] == 1
    assert data["latest_price"] == 153.0
    assert len(data["data"]) == 1
    assert data["data"][0]["open"] == 150.0


def test_get_ticker_data_not_found():
    """Test requesting data for non-existent ticker."""
    (mock_session.query.return_value
     .filter.return_value
     .order_by.return_value
     .limit.return_value
     .all.return_value) = []
    
    response = client.get("/data/INVALID")

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()