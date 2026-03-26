from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List, Optional, Set
import json
import logging
import urllib3
from datetime import datetime, date

from . import models, schemas, database
from .services.ml_engine import PredictorService

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
logger = logging.getLogger(__name__)

models.Base.metadata.create_all(bind=database.engine)

app = FastAPI(title="NEPSE Predictor API")

processing_stocks: Set[str] = set()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor_service = PredictorService()

@app.get("/stocks", response_model=List[schemas.StockWithMetrics])
def get_stocks(db: Session = Depends(database.get_db)):
    stocks = db.query(models.Stock).all()
    results = []
    for s in stocks:
        try:
            latest_price_obj = db.query(models.Price).filter(models.Price.stock_id == s.id).order_by(models.Price.date.desc()).first()
            prev_price_obj = db.query(models.Price).filter(models.Price.stock_id == s.id).order_by(models.Price.date.desc()).offset(1).first()
            latest_pred = db.query(models.Prediction).filter(models.Prediction.stock_id == s.id).order_by(models.Prediction.date.desc()).first()
            
            change_pct = 0.0
            if latest_price_obj and prev_price_obj and prev_price_obj.close:
                change_pct = (latest_price_obj.close - prev_price_obj.close) / (prev_price_obj.close + 1e-9) * 100

            # Grab 52W from full_result_json if available
            week52_high = None
            week52_low = None
            if latest_pred and latest_pred.full_result_json:
                try:
                    fr = json.loads(latest_pred.full_result_json)
                    week52_high = fr.get("week52_high")
                    week52_low = fr.get("week52_low")
                except: pass

            results.append(schemas.StockWithMetrics(
                id=s.id,
                symbol=s.symbol,
                name=s.name,
                sector=s.sector,
                latest_price=latest_price_obj.close if latest_price_obj else None,
                latest_change_pct=change_pct,
                latest_signal=latest_pred.signal if latest_pred else None,
                latest_confidence=latest_pred.confidence if latest_pred else None,
                week52_high=week52_high,
                week52_low=week52_low,
            ))
        except Exception as e:
            logger.error(f"Error processing stock {s.symbol}: {e}")
            continue
    return results

@app.get("/stock/{symbol}", response_model=schemas.StockDetail)
def get_stock_detail(symbol: str, db: Session = Depends(database.get_db)):
    stock = db.query(models.Stock).filter(models.Stock.symbol == symbol.upper()).first()
    if not stock:
        companies = predictor_service.get_all_stocks()
        match = companies[companies["symbol"].str.upper() == symbol.upper()]
        if match.empty:
            raise HTTPException(status_code=404, detail="Stock not found")
        row = match.iloc[0]
        stock = models.Stock(symbol=row["symbol"], name=row["name"], sector=row["sector"])
        db.add(stock)
        db.commit()
        db.refresh(stock)

    latest_price_obj = db.query(models.Price).filter(models.Price.stock_id == stock.id).order_by(models.Price.date.desc()).first()
    prev_price_obj = db.query(models.Price).filter(models.Price.stock_id == stock.id).order_by(models.Price.date.desc()).offset(1).first()
    # Get 7 predictions for forecast
    latest_preds = db.query(models.Prediction).filter(models.Prediction.stock_id == stock.id).order_by(models.Prediction.date.desc()).limit(7).all()
    # The most recent one holds all the analytics JSONs
    top_pred = latest_preds[0] if latest_preds else None

    change_pct = 0.0
    if latest_price_obj and prev_price_obj and prev_price_obj.close:
        change_pct = (latest_price_obj.close - prev_price_obj.close) / (prev_price_obj.close + 1e-9) * 100

    # Defaults
    forecast, technical_info, trade_plan, sentiment, accuracy_metrics = [], None, None, None, None
    top_features, anomalies, scenarios, ai_summary = None, None, None, None
    regime, smart_money = None, None
    final_signal, final_confidence, model_reliability, signal_reason = None, None, None, None
    week52_high, week52_low, records_count, date_range_start, date_range_end, avg_vol_20d = None, None, None, None, None, None

    if top_pred:
        # Load full result if available
        full_result = {}
        if top_pred.full_result_json:
            try:
                full_result = json.loads(top_pred.full_result_json)
            except: pass

        try:
            technical_info = json.loads(top_pred.technical_json) if top_pred.technical_json else full_result.get("trend_info")
            trade_plan = json.loads(top_pred.trade_plan_json) if top_pred.trade_plan_json else full_result.get("trade_plan")
            sentiment = json.loads(top_pred.sentiment_json) if top_pred.sentiment_json else full_result.get("sentiment")
            accuracy_metrics = json.loads(top_pred.accuracy_json) if top_pred.accuracy_json else full_result.get("accuracy_metrics")
            top_features = json.loads(top_pred.features_json) if top_pred.features_json else full_result.get("top_features")
            anomalies = json.loads(top_pred.anomalies_json) if top_pred.anomalies_json else full_result.get("anomalies")
        except: pass

        # Rich fields from full_result_json
        if full_result:
            if technical_info is None:
                technical_info = full_result.get("trend_info")
            scenarios = full_result.get("scenarios")
            regime = full_result.get("regime")
            smart_money = full_result.get("smart_money")
            final_signal = full_result.get("final_signal")
            final_confidence = full_result.get("final_confidence")
            model_reliability = full_result.get("model_reliability")
            signal_reason = full_result.get("signal_reason")
            week52_high = full_result.get("week52_high")
            week52_low = full_result.get("week52_low")
            records_count = full_result.get("records_count")
            date_range_start = full_result.get("date_range_start")
            date_range_end = full_result.get("date_range_end")
            avg_vol_20d = full_result.get("avg_vol_20d")
            ai_summary = full_result.get("ai_summary")

            # Build forecast from full_result["forecast"] for richer fields
            fr_forecast = full_result.get("forecast", [])
            if fr_forecast:
                for f in fr_forecast:
                    try:
                        pred_date = f.get("date", "")
                        if isinstance(pred_date, str) and "T" in pred_date:
                            pred_date = pred_date.split("T")[0]
                        forecast.append(schemas.PredictionBase(
                            date=datetime.strptime(pred_date, "%Y-%m-%d").date() if pred_date else date.today(),
                            predicted_close=f.get("predicted_close", 0),
                            direction_prob=f.get("direction_prob", 0),
                            confidence=f.get("confidence", f.get("direction_confidence", 0.5)),
                            signal=f.get("signal", top_pred.signal),
                            change_pct=f.get("change_pct"),
                            date_bs=f.get("date_bs") or f.get("bs_date"),
                            day_name=f.get("day_name") or f.get("day"),
                            d_conf=f.get("d_conf") or f.get("direction_confidence"),
                            trap=f.get("trap", 0),
                        ))
                    except Exception as e:
                        logger.warning(f"Forecast parse error: {e}")

        # Fallback: use DB rows if full_result had no forecast
        if not forecast:
            for p in latest_preds:
                forecast.append(schemas.PredictionBase(
                    date=p.date,
                    predicted_close=p.predicted_close,
                    direction_prob=p.direction_prob,
                    confidence=p.confidence,
                    signal=p.signal,
                ))

        if final_signal is None:
            final_signal = top_pred.signal
        if ai_summary is None and technical_info:
            ai_summary = technical_info.get("ai_summary")

    # Risk summary
    risk_summary = []
    if technical_info:
        vl = technical_info.get("volatility_label", "")
        if "HIGH" in str(vl).upper():
            risk_summary.append("🔥 High volatility — size positions carefully, set tight stop-losses.")
        rsi = technical_info.get("rsi")
        if rsi and rsi > 70:
            risk_summary.append("⚡ RSI overbought — consider reducing exposure.")
    if anomalies:
        risk_summary.append(f"⚡ {len(anomalies)} price anomalies — possible news-driven spikes/crashes.")
    if sentiment and sentiment.get("score", 0) < -0.3:
        risk_summary.append(f"📰 Negative news sentiment ({sentiment.get('score', 0):.2f}) — watch for downside.")

    return schemas.StockDetail(
        id=stock.id,
        symbol=stock.symbol,
        name=stock.name,
        sector=stock.sector,
        latest_price=latest_price_obj.close if latest_price_obj else None,
        latest_change_pct=change_pct,
        latest_signal=final_signal or (forecast[0].signal if forecast else None),
        latest_confidence=final_confidence or (forecast[0].confidence if forecast else None),
        week52_high=week52_high,
        week52_low=week52_low,
        records_count=records_count,
        date_range_start=date_range_start,
        date_range_end=date_range_end,
        avg_vol_20d=avg_vol_20d,
        forecast=forecast,
        technical_info=technical_info,
        trade_plan=trade_plan,
        sentiment=sentiment,
        regime=regime,
        smart_money=smart_money,
        accuracy_metrics=accuracy_metrics,
        top_features=top_features,
        anomalies=anomalies,
        scenarios=scenarios,
        final_signal=final_signal,
        final_confidence=final_confidence,
        model_reliability=model_reliability,
        signal_reason=signal_reason,
        risk_summary=risk_summary,
        ai_summary=ai_summary,
    )

@app.post("/run-prediction/{symbol}")
def run_prediction(symbol: str, background_tasks: BackgroundTasks, db: Session = Depends(database.get_db)):
    sym = symbol.upper()
    if sym in processing_stocks:
        return {"status": "already running", "message": f"Prediction for {sym} is already in progress"}
    processing_stocks.add(sym)
    background_tasks.add_task(process_prediction, sym, db)
    return {"message": f"Prediction for {sym} started in background"}

def process_prediction(symbol: str, db: Session):
    try:
        result = predictor_service.run_prediction(symbol)
        if not result or not result.get("is_success"):
            logger.error(f"Prediction failed for {symbol}: {result.get('error') if result else 'No result'}")
            return

        stock = db.query(models.Stock).filter(models.Stock.symbol == symbol.upper()).first()
        if not stock:
            return

        # 1. Update Price
        today = date.today()
        price = db.query(models.Price).filter(models.Price.stock_id == stock.id, models.Price.date == today).first()
        if not price:
            price = models.Price(stock_id=stock.id, date=today)
            db.add(price)
        price.close = result["last_price"]
        price.volume = result.get("avg_vol_20d", 0.0)

        # 2. Serialize the ENTIRE result for full_result_json
        forecast_list = result.get("forecast", [])
        full_result_for_storage = dict(result)
        # Make sure forecast is a plain list of dicts for JSON serialization
        if forecast_list and hasattr(forecast_list[0], '__dict__'):
            full_result_for_storage["forecast"] = [
                {k: v for k, v in f.__dict__.items() if not k.startswith('_')} for f in forecast_list
            ]
        
        try:
            full_result_json_str = json.dumps(full_result_for_storage, default=str)
        except Exception as e:
            logger.error(f"JSON serialization error: {e}")
            full_result_json_str = None

        # 3. Save each forecast day as a Prediction row
        for i, f in enumerate(forecast_list):
            try:
                # Handle both dict and object
                if isinstance(f, dict):
                    pred_date_str = f.get("date", "")
                    predicted_close = float(f.get("predicted_close", 0))
                    direction_prob = float(f.get("direction_prob", 0.5))
                    confidence_val = float(f.get("confidence", f.get("direction_confidence", 0.5)))
                else:
                    pred_date_str = str(getattr(f, "date", ""))
                    predicted_close = float(getattr(f, "predicted_close", 0))
                    direction_prob = float(getattr(f, "direction_prob", 0.5))
                    confidence_val = float(getattr(f, "confidence", getattr(f, "direction_confidence", 0.5)))

                if isinstance(pred_date_str, str) and "T" in pred_date_str:
                    pred_date_str = pred_date_str.split("T")[0]
                pred_date = datetime.strptime(pred_date_str, "%Y-%m-%d").date()

                db_pred = db.query(models.Prediction).filter(
                    models.Prediction.stock_id == stock.id,
                    models.Prediction.date == pred_date
                ).first()
                if not db_pred:
                    db_pred = models.Prediction(stock_id=stock.id, date=pred_date)
                    db.add(db_pred)

                db_pred.predicted_close = predicted_close
                db_pred.direction_prob = direction_prob
                db_pred.confidence = confidence_val
                db_pred.signal = result["final_signal"] if i == 0 else "HOLD"

                # For the first (today's) prediction, store all analytics JSONs
                if i == 0:
                    t_info = result.get("trend_info", {})
                    if isinstance(t_info, dict):
                        t_info["ai_summary"] = result.get("ai_summary")
                    db_pred.technical_json = json.dumps(t_info, default=str)
                    db_pred.trade_plan_json = json.dumps(result.get("trade_plan"), default=str)
                    db_pred.sentiment_json = json.dumps(result.get("sentiment"), default=str)
                    db_pred.accuracy_json = json.dumps(result.get("accuracy_metrics"), default=str)
                    db_pred.features_json = json.dumps(result.get("top_features"), default=str)
                    db_pred.anomalies_json = json.dumps(result.get("anomalies"), default=str)
                    db_pred.full_result_json = full_result_json_str

            except Exception as e:
                logger.error(f"Error saving prediction row for {symbol} day {i}: {e}", exc_info=True)

        db.commit()
        logger.info(f"Prediction for {symbol} saved successfully.")

    finally:
        if symbol in processing_stocks:
            processing_stocks.remove(symbol)

@app.post("/init-stocks")
def init_stocks(db: Session = Depends(database.get_db)):
    companies = predictor_service.get_all_stocks()
    added = 0
    for _, row in companies.iterrows():
        existing = db.query(models.Stock).filter(models.Stock.symbol == row["symbol"]).first()
        if not existing:
            db.add(models.Stock(symbol=row["symbol"], name=row["name"], sector=row.get("sector", "N/A")))
            added += 1
    db.commit()
    total = db.query(models.Stock).count()
    return {"message": f"Stock list initialized. Added {added} new stocks. Total: {total} stocks."}

@app.get("/stocks/count")
def stocks_count(db: Session = Depends(database.get_db)):
    return {"total": db.query(models.Stock).count()}
