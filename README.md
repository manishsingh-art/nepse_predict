# NEPSE ML Predictor Web v5.0

A professional-grade full-stack web application for NEPSE stock prediction using Ensemble Machine Learning.

## Features
- **FastAPI Backend**: High-performance async API for stock analysis.
- **SQLite Database**: Persistent storage for historical data and AI forecasts.
- **ML Engine Service**: Refactored ensemble model callable via API.
- **React Dashboard**: Modern, dark-themed UI with interactive charts (Recharts).
- **Unified Decision Engine**: Weighted consensus signals (BUY/SELL/HOLD).

## Quick Start
1. **Backend**:
   ```bash
   pip3 install fastapi uvicorn sqlalchemy pandas numpy lightgbm xgboost scikit-learn requests
   # From root directory:
   python3 -m uvicorn backend.main:app --reload
   ```
2. **Frontend**:
   ```bash
   cd frontend
   npm install
   npm run dev
   ```
3. Visit `http://localhost:5173` to view the dashboard.

## Technical Internals
- **Database**: `nepse_predictor.db` contains initialized stock metadata.
- **Prediction Logic**: Handles News Sentiment, Smart Money flow, and Technical Indicators (RSI, MACD, OBV, etc.).
- **Models**: Ensemble of LightGBM, XGBoost, and Ridge Regressor with recursive forecasting.

---
Developed as a senior full-stack and ML engineering project.
