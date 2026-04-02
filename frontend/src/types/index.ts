// ── Domain enums ──────────────────────────────────────────────────────────────

export type Action = "BUY" | "SELL" | "HOLD" | "AVOID";

// ── Python report types (raw output from predict_api.py) ─────────────────────

export interface PythonForecastPoint {
  day: number;
  date: string;
  predicted_close: number;
  predicted_return: number;
  direction_prob: number;
  low_band: number;
  high_band: number;
  change_pct: number;
  confidence: "high" | "medium" | "low";
  direction_confidence: number;
  trap_score: number;
  is_trading_day: boolean;
  holiday_name: string | null;
  day_name: string;
}

export interface PythonReport {
  symbol: string;
  version?: string;
  generated_at: string;
  last_close: number;
  last_date: string;
  regime: {
    regime: string;
    confidence: number;
    volatility_pct: number;
    vol_ratio: number;
    trap_score: number;
  };
  strategy: {
    action: string;
    entry: number;
    stop_loss: number;
    take_profit: number;
    risk_reward_ratio: number;
    suggested_size_weight: number;
    reason: string;
  };
  sentiment: {
    score: number;
    reason: string;
    category: string;
  };
  forecast: PythonForecastPoint[];
  ml_metrics: {
    avg_mae: number;
    avg_dir_acc: number;
    avg_sharpe: number;
    models_used: string[];
  };
  error?: string;
}

// ── API response types ────────────────────────────────────────────────────────

export interface StockLatestPrediction {
  predictedClose: number;
  directionProb: number;
  action: Action;
  modelConfidence: number;
  predictionDate: string;
  regime: string;
}

export interface StockAccuracy {
  directionAccuracy: number;
  avgErrorPct: number;
  totalPredictions: number;
}

export interface StockSummary {
  id: string;
  symbol: string;
  name: string;
  sector: string | null;
  nepseId: number | null;       // merolagani company ID — higher = newer listing
  latestPrediction: StockLatestPrediction | null;
  accuracy: StockAccuracy | null;
}

export interface ForecastPoint {
  id: string;
  targetDate: string;
  predictedClose: number;
  predictedReturn: number;
  directionProb: number;
  lowerBound: number;
  upperBound: number;
  modelConfidence: number;
  trapScore: number;
  action: Action;
}

export interface HistoricalPrediction {
  id: string;
  predictionDate: string;
  targetDate: string;
  predictedClose: number;
  action: Action;
  directionProb: number;
  accuracy: {
    actualClose: number;
    errorPct: number;
    directionCorrect: boolean;
  } | null;
}

export interface ActualPricePoint {
  date: string;
  close: number;
  volume: number;
}

export interface StockDetail {
  id: string;
  symbol: string;
  name: string;
  sector: string | null;
  latestPrediction: StockLatestPrediction | null;
  accuracy: StockAccuracy | null;
  forecasts: ForecastPoint[];
  historicalPredictions: HistoricalPrediction[];
  actualPrices: ActualPricePoint[];
}

export interface AccuracyMetrics {
  directionAccuracy: number;
  avgErrorPct: number;
  totalPredictions: number;
  correctDirections: number;
  weeklyBreakdown: Array<{
    week: string;
    dirAccuracy: number;
    avgError: number;
    count: number;
  }>;
}

// ── API response wrappers ────────────────────────────────────────────────────

export interface ApiSuccess<T> {
  data: T;
  message?: string;
}

export interface ApiError {
  error: string;
  details?: string;
}
