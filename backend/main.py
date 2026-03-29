from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import exc
from typing import List, Optional, Set
import json
import logging
from datetime import datetime, date, timezone

from . import models, schemas, database
from .services.ml_engine import PredictorService

logger = logging.getLogger(__name__)

# Create tables based on the new explicit schema
models.Base.metadata.create_all(bind=database.engine)

app = FastAPI(title="NEPSE Predictor Architecture v5 API")

processing_stocks: Set[str] = set()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000", "http://localhost:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor_service = PredictorService()

def process_prediction_task(symbol: str):
    logger.info(f"Starting background prediction task for {symbol}")
    db = database.SessionLocal()
    try:
        sym = symbol.upper()
        
        # 1. Ensure Stock exists
        stock = db.query(models.Stock).filter(models.Stock.symbol == sym).first()
        if not stock:
            # Add basic dummy if missing, normally initialize scripts do this
            stock = models.Stock(symbol=sym, name=f"{sym} Company")
            db.add(stock)
            db.commit()
            db.refresh(stock)

        # 2. Run ML
        result = predictor_service.run_prediction(sym)
        if not result or not result.get("is_success"):
            logger.error(f"Failed to generate prediction for {sym}")
            return
            
        # 3. Transactional Insert
        try:
            db.begin_nested() # Create nested savepoint for atomic insertion
            
            # Predict
            pred = models.Prediction(
                stock_id=stock.id,
                predicted_close=result.get("predicted_close", 0.0),
                direction_prob=result.get("direction_prob", 0.5),
                confidence=result.get("confidence", 0.5),
                signal=result.get("signal", "HOLD"),
                technical_json=json.dumps(result.get("technical", {})),
                trade_plan_json=json.dumps(result.get("trade_plan", {})),
                sentiment_json=json.dumps(result.get("sentiment", {})),
                accuracy_json=json.dumps(result.get("accuracy", {})),
                features_json=json.dumps(result.get("features", {})),
                anomalies_json=json.dumps(result.get("anomalies", [])),
                full_result_json=json.dumps(result)
            )
            db.add(pred)
            db.flush() # flush to get prediction ID

            # Forecast
            forecasts = result.get("forecast", [])
            for f in forecasts:
                try:
                    f_date_str = f.get("date", "").split("T")[0]
                    f_date = datetime.strptime(f_date_str, "%Y-%m-%d").date()
                except:
                    f_date = date.today()
                
                db.add(models.Forecast(
                    prediction_id=pred.id,
                    date=f_date,
                    price=f.get("price", 0.0),
                    change_pct=f.get("change_pct", 0.0),
                    prob_up=f.get("prob_up", 50.0),
                    confidence=f.get("confidence", 5.0)
                ))

            # Scenarios
            scenarios = result.get("scenarios", {})
            for typ in ["bull", "base", "bear"]:
                s = scenarios.get(typ)
                if s:
                    db.add(models.Scenario(
                        prediction_id=pred.id,
                        scenario_type=typ,
                        probability=s.get("prob", 0.0),
                        target_price=s.get("target", 0.0)
                    ))

            # Trade Plan
            tp = result.get("trade_plan", {})
            bz = tp.get("buy_zone", [0, 0])
            db.add(models.TradePlan(
                prediction_id=pred.id,
                buy_zone_low=bz[0] if len(bz)>0 else 0.0,
                buy_zone_high=bz[1] if len(bz)>1 else 0.0,
                target_price=tp.get("target", 0.0),
                stop_loss=tp.get("stop_loss", 0.0),
                rr_ratio=tp.get("rr_ratio", 0.0)
            ))

            # Anomalies
            anomalies = result.get("anomalies", [])
            for a in anomalies:
                try:
                    a_date_str = a.get("date", "").split("T")[0]
                    a_date = datetime.strptime(a_date_str, "%Y-%m-%d").date()
                except:
                    a_date = date.today()
                    
                db.add(models.Anomaly(
                    prediction_id=pred.id,
                    date=a_date,
                    anomaly_type=a.get("type", "unknown"),
                    change_pct=a.get("change_pct", 0.0)
                ))

            db.commit() # Commit all successfully
            logger.info(f"✅ Stored comprehensive ML results for {sym} to database.")
            
        except exc.SQLAlchemyError as e:
            db.rollback()
            logger.error(f"Database transaction failed for {sym}: {e}")
            
    finally:
        db.close()
        if symbol in processing_stocks:
            processing_stocks.remove(symbol)
            

@app.post("/predict/{symbol}")
def run_prediction(symbol: str, background_tasks: BackgroundTasks):
    sym = symbol.upper()
    if sym in processing_stocks:
        return {"status": "already running", "message": f"Prediction for {sym} is already in progress"}
    
    processing_stocks.add(sym)
    background_tasks.add_task(process_prediction_task, sym)
    return {"status": "processing", "message": f"ML pipeline triggered for {sym}."}


@app.get("/stock/{slug}", response_model=schemas.StockDetailResponse)
def get_stock_detail(slug: str, db: Session = Depends(database.get_db)):
    """ Returns fully aggregated data for the Detail Page """
    sym = slug.upper()
    stock = db.query(models.Stock).filter(models.Stock.symbol == sym).first()
    if not stock:
        raise HTTPException(status_code=404, detail="Stock not found")
        
    pred = db.query(models.Prediction).filter(models.Prediction.stock_id == stock.id).order_by(models.Prediction.date.desc()).first()
    if not pred:
        raise HTTPException(status_code=404, detail="No predictions available for this stock")
        
    # Gather related models
    forecasts = db.query(models.Forecast).filter(models.Forecast.prediction_id == pred.id).order_by(models.Forecast.date.asc()).all()
    anomalies = db.query(models.Anomaly).filter(models.Anomaly.prediction_id == pred.id).order_by(models.Anomaly.date.desc()).all()
    scenarios_db = db.query(models.Scenario).filter(models.Scenario.prediction_id == pred.id).all()
    trade_plan_db = db.query(models.TradePlan).filter(models.TradePlan.prediction_id == pred.id).first()
    
    # Format forecasts
    fc = [schemas.ForecastBase(
        date=str(f.date),
        price=f.price,
        change_pct=f.change_pct,
        prob_up=f.prob_up,
        confidence=f.confidence
    ) for f in forecasts]
    
    # Format anomalies
    anols = [schemas.AnomalyBase(
        date=str(a.date),
        type=a.anomaly_type,
        change_pct=a.change_pct
    ) for a in anomalies]
    
    # Format Scenarios
    scens = schemas.Scenarios()
    for s in scenarios_db:
        if s.scenario_type == "bull": scens.bull = schemas.ScenarioData(prob=s.probability, target=s.target_price)
        if s.scenario_type == "base": scens.base = schemas.ScenarioData(prob=s.probability, target=s.target_price)
        if s.scenario_type == "bear": scens.bear = schemas.ScenarioData(prob=s.probability, target=s.target_price)
        
    # Format Trade Plan
    tp = schemas.TradePlan(
        buy_zone=[trade_plan_db.buy_zone_low, trade_plan_db.buy_zone_high] if trade_plan_db else [],
        target=trade_plan_db.target_price if trade_plan_db else 0.0,
        stop_loss=trade_plan_db.stop_loss if trade_plan_db else 0.0,
        rr_ratio=trade_plan_db.rr_ratio if trade_plan_db else 0.0
    )
    
    # Compute risks dynamically
    risks = []
    if pred.direction_prob and pred.direction_prob > 0.7: risks.append("High directional probability")
    if len(anols) > 0: risks.append(f"{len(anols)} recent price anomalies detected")

    return schemas.StockDetailResponse(
        symbol=stock.symbol,
        predicted_close=pred.predicted_close,
        signal=pred.signal,
        direction_prob=pred.direction_prob,
        confidence=pred.confidence,
        forecast=fc,
        scenarios=scens,
        trade_plan=tp,
        risks=risks,
        anomalies=anols,
        ai_summary=""  # Placeholder, since ai_summary is not in the new model
    )


@app.get("/stocks", response_model=List[schemas.StockOverview])
def get_stocks_overview(db: Session = Depends(database.get_db)):
    """ Returns data for the Dashboard """
    stocks = db.query(models.Stock).all()
    results = []
    
    for s in stocks:
        pred = db.query(models.Prediction).filter(models.Prediction.stock_id == s.id).order_by(models.Prediction.date.desc()).first()
        
        results.append(schemas.StockOverview(
            id=s.id,
            symbol=s.symbol,
            name=s.name,
            predicted_close=pred.predicted_close if pred else None,
            signal=pred.signal if pred else "HOLD",
            direction_prob=pred.direction_prob if pred else None,
            confidence=pred.confidence if pred else None
        ))
        
    return results
    
# Included to make init easier
@app.post("/init-stocks")
def init_stocks(db: Session = Depends(database.get_db)):
    companies = predictor_service.get_all_stocks()
    added = 0
    for _, row in companies.iterrows():
        existing = db.query(models.Stock).filter(models.Stock.symbol == row["symbol"]).first()
        if not existing:
            db.add(models.Stock(symbol=row["symbol"], name=row["name"]))
            added += 1
    db.commit()
    total = db.query(models.Stock).count()
    return {"message": f"Stock list initialized. Added {added} new stocks. Total: {total} stocks."}
