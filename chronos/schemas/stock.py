from pydantic import BaseModel, ConfigDict
from datetime import datetime
from typing import List

class StockDataBase(BaseModel):
    ticker: str
    date: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

class StockDataResponse(StockDataBase):
    model_config = ConfigDict(from_attributes=True)

class TickerResponse(BaseModel):
    ticker: str
    rows_returned: int
    latest_price: float
    data: List[StockDataResponse]