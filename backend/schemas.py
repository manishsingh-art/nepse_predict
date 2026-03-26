from pydantic import BaseModel
from typing import List, Optional, Any
from datetime import date

class StockBase(BaseModel):
    symbol: str
    name: str
    sector: str

class StockCreate(StockBase):
    pass

class Stock(StockBase):
    id: int

    class Config:
        from_attributes = True

class PriceBase(BaseModel):
    date: date
    close: float
    volume: float

class Price(PriceBase):
    id: int
    stock_id: int

    class Config:
        from_attributes = True

class PredictionBase(BaseModel):
    date: date
    predicted_close: Optional[float] = None
    direction_prob: Optional[float] = None
    confidence: Optional[float] = None
    signal: Optional[str] = None
    # Extended forecast fields (from full_result_json)
    change_pct: Optional[float] = None
    date_bs: Optional[str] = None
    day_name: Optional[str] = None
    d_conf: Optional[float] = None
    trap: Optional[int] = None

class Prediction(PredictionBase):
    id: int
    stock_id: int

    class Config:
        from_attributes = True

class StockWithMetrics(Stock):
    latest_price: Optional[float] = None
    latest_change_pct: Optional[float] = None
    latest_signal: Optional[str] = None
    latest_confidence: Optional[float] = None
    week52_high: Optional[float] = None
    week52_low: Optional[float] = None

class StockDetail(StockWithMetrics):
    # Dataset summary
    records_count: Optional[int] = None
    date_range_start: Optional[str] = None
    date_range_end: Optional[str] = None
    avg_vol_20d: Optional[float] = None
    # Forecast
    forecast: List[PredictionBase] = []
    # Analytical sections
    technical_info: Optional[dict] = None
    trade_plan: Optional[dict] = None
    sentiment: Optional[dict] = None
    regime: Optional[dict] = None
    smart_money: Optional[dict] = None
    accuracy_metrics: Optional[dict] = None
    top_features: Optional[List[dict]] = None
    anomalies: Optional[List[dict]] = None
    scenarios: Optional[List[dict]] = None
    # Signal
    final_signal: Optional[str] = None
    final_confidence: Optional[float] = None
    model_reliability: Optional[float] = None
    signal_reason: Optional[str] = None
    # Risk & AI
    risk_summary: List[str] = []
    ai_summary: Optional[str] = None
