from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime, Date, Index
from sqlalchemy.orm import declarative_base, relationship
from datetime import datetime, timezone

Base = declarative_base()

class Stock(Base):
    __tablename__ = "stocks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String, unique=True, nullable=False)
    name = Column(String)

    # Relationships
    predictions = relationship("Prediction", back_populates="stock", order_by="desc(Prediction.date)")

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    stock_id = Column(Integer, ForeignKey("stocks.id"), nullable=False)
    date = Column(Date, nullable=False)  # Changed from created_at to date
    
    predicted_close = Column(Float)  # Renamed to match DB
    direction_prob = Column(Float)   # Renamed to match DB
    confidence = Column(Float)       # Renamed to match DB
    signal = Column(String)          # Renamed to match DB
    technical_json = Column(String)  # Renamed to match DB
    trade_plan_json = Column(String) # Renamed to match DB
    sentiment_json = Column(String)  # Renamed to match DB
    accuracy_json = Column(String)   # Renamed to match DB
    features_json = Column(String)   # Renamed to match DB
    anomalies_json = Column(String)  # Renamed to match DB
    full_result_json = Column(String) # Renamed to match DB

    # Relationships
    stock = relationship("Stock", back_populates="predictions")
    forecasts = relationship("Forecast", back_populates="prediction", cascade="all, delete-orphan")
    scenarios = relationship("Scenario", back_populates="prediction", cascade="all, delete-orphan")
    anomalies = relationship("Anomaly", back_populates="prediction", cascade="all, delete-orphan")
    trade_plans = relationship("TradePlan", back_populates="prediction", cascade="all, delete-orphan")

class Forecast(Base):
    __tablename__ = "forecasts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    prediction_id = Column(Integer, ForeignKey("predictions.id"), nullable=False)
    date = Column(Date, nullable=False)
    price = Column(Float)
    change_pct = Column(Float)
    prob_up = Column(Float)
    confidence = Column(Float)

    # Relationships
    prediction = relationship("Prediction", back_populates="forecasts")

class Scenario(Base):
    __tablename__ = "scenarios"

    id = Column(Integer, primary_key=True, autoincrement=True)
    prediction_id = Column(Integer, ForeignKey("predictions.id"), nullable=False)
    scenario_type = Column(String, nullable=False) # 'bull', 'base', 'bear'
    probability = Column(Float)
    target_price = Column(Float)

    # Relationships
    prediction = relationship("Prediction", back_populates="scenarios")

class Anomaly(Base):
    __tablename__ = "anomalies"

    id = Column(Integer, primary_key=True, autoincrement=True)
    prediction_id = Column(Integer, ForeignKey("predictions.id"), nullable=False)
    date = Column(Date, nullable=False)
    anomaly_type = Column(String) # 'spike', 'crash', etc.
    change_pct = Column(Float)

    # Relationships
    prediction = relationship("Prediction", back_populates="anomalies")

class TradePlan(Base):
    __tablename__ = "trade_plans"

    id = Column(Integer, primary_key=True, autoincrement=True)
    prediction_id = Column(Integer, ForeignKey("predictions.id"), nullable=False)
    buy_zone_low = Column(Float)
    buy_zone_high = Column(Float)
    target_price = Column(Float)
    stop_loss = Column(Float)
    rr_ratio = Column(Float)
    
    # Relationships
    prediction = relationship("Prediction", back_populates="trade_plans")

# Indexes explicitly defined as requested
Index('idx_stock_symbol', Stock.symbol)
Index('idx_prediction_stock_id', Prediction.stock_id)
Index('idx_prediction_date', Prediction.date.desc())  # Changed from created_at
