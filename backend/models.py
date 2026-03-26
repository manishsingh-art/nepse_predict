from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime, Date
from sqlalchemy.orm import relationship
from .database import Base
import datetime

class Stock(Base):
    __tablename__ = "stocks"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, unique=True, index=True)
    name = Column(String)
    sector = Column(String)

    prices = relationship("Price", back_populates="stock")
    predictions = relationship("Prediction", back_populates="stock")

class Price(Base):
    __tablename__ = "prices"

    id = Column(Integer, primary_key=True, index=True)
    stock_id = Column(Integer, ForeignKey("stocks.id"))
    date = Column(Date, index=True)
    close = Column(Float)
    volume = Column(Float)

    stock = relationship("Stock", back_populates="prices")

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    stock_id = Column(Integer, ForeignKey("stocks.id"))
    date = Column(Date, index=True)
    predicted_close = Column(Float)
    direction_prob = Column(Float)
    confidence = Column(Float)
    signal = Column(String)
    
    # Store JSON strings for technical indicators, trend_info, etc.
    technical_json = Column(String) 
    trade_plan_json = Column(String)
    sentiment_json = Column(String)
    accuracy_json = Column(String)   # CV fold metrics
    features_json = Column(String)   # Feature importance
    anomalies_json = Column(String)  # Outlier detection
    full_result_json = Column(String) # Complete result (52W, scenarios, date_bs, regime, etc.)

    stock = relationship("Stock", back_populates="predictions")
