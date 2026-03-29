from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, Text, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime

DATABASE_URL = "sqlite:///./nepse_predictor.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class Stock(Base):
    __tablename__ = "stocks"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=False)

    predictions = relationship("Prediction", back_populates="stock")

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    stock_id = Column(Integer, ForeignKey("stocks.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    price = Column(Float, nullable=False)
    trend = Column(String, nullable=False)
    rsi = Column(Float, nullable=True)
    sentiment = Column(Float, nullable=True)
    ai_summary = Column(Text, nullable=True)

    stock = relationship("Stock", back_populates="predictions")
    forecasts = relationship("Forecast", back_populates="prediction")
    scenarios = relationship("Scenario", back_populates="prediction")
    anomalies = relationship("Anomaly", back_populates="prediction")

class Forecast(Base):
    __tablename__ = "forecasts"

    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(Integer, ForeignKey("predictions.id"), nullable=False)
    date = Column(DateTime, nullable=False)
    price = Column(Float, nullable=False)
    change_pct = Column(Float, nullable=False)
    prob_up = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)

    prediction = relationship("Prediction", back_populates="forecasts")

class Scenario(Base):
    __tablename__ = "scenarios"

    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(Integer, ForeignKey("predictions.id"), nullable=False)
    type = Column(String, nullable=False)  # bull/base/bear
    probability = Column(Float, nullable=False)
    target_price = Column(Float, nullable=False)

    prediction = relationship("Prediction", back_populates="scenarios")

class Anomaly(Base):
    __tablename__ = "anomalies"

    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(Integer, ForeignKey("predictions.id"), nullable=False)
    date = Column(DateTime, nullable=False)
    type = Column(String, nullable=False)
    change_pct = Column(Float, nullable=False)

    prediction = relationship("Prediction", back_populates="anomalies")

# Create indexes
Index('idx_prediction_stock_id', Prediction.stock_id)
Index('idx_prediction_created_at', Prediction.created_at.desc())

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()