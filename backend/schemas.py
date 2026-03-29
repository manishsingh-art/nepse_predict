from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import date, datetime

# --- Anomalies ---
class AnomalyBase(BaseModel):
    date: str
    type: str # renamed from anomaly_type for JSON representation if needed, or mapped
    change_pct: float

class Anomaly(AnomalyBase):
    id: int
    prediction_id: int
    class Config:
        from_attributes = True

# --- Scenarios ---
class ScenarioData(BaseModel):
    prob: float
    target: float

class Scenarios(BaseModel):
    bull: Optional[ScenarioData] = None
    base: Optional[ScenarioData] = None
    bear: Optional[ScenarioData] = None

# --- Forecasts ---
class ForecastBase(BaseModel):
    date: str
    price: float
    change_pct: float
    prob_up: float
    confidence: float
    
    class Config:
        from_attributes = True

# --- Trade Plan ---
class TradePlan(BaseModel):
    buy_zone: List[float] = Field(default_factory=list)
    target: float
    stop_loss: float
    rr_ratio: float

# --- Prediction Master ---
class PredictionResponse(BaseModel):
    symbol: str
    predicted_close: float
    signal: str
    direction_prob: float
    confidence: float
    forecast: List[ForecastBase] = []
    scenarios: Scenarios
    trade_plan: TradePlan
    risks: List[str] = []
    anomalies: List[AnomalyBase] = []
    ai_summary: str

# --- Stock Overview (for GET /stocks) ---
class StockOverview(BaseModel):
    id: int
    symbol: str
    name: str
    predicted_close: Optional[float] = None
    signal: Optional[str] = None
    direction_prob: Optional[float] = None
    confidence: Optional[float] = None
    
# --- For Detail Page (GET /stocks/{symbol}) ---
# It matches the PredictionResponse practically, but wrapped just in case.
class StockDetailResponse(PredictionResponse):
    pass
