from sqlalchemy import Column, Integer, String, Float, DateTime, UniqueConstraint
from chronos.utils.db import Base

class StockData(Base):
    __tablename__ = "stock_data"

    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String, index=True, nullable=False)
    date = Column(DateTime, nullable=False, index=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)

    __table_args__ = (
        UniqueConstraint('ticker', 'date', name='uix_ticker_date'),
    )