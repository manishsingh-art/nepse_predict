import os
import sys
import json
import glob
import warnings
import contextlib
from datetime import datetime
import concurrent.futures

warnings.filterwarnings("ignore")

# Context manager to silence all textual logs from internal functions natively
@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

def load_latest_log():
    today_str = datetime.now().strftime("%Y-%m-%d")
    today_file = f"predictions_log-{today_str}.json"
    if os.path.exists(today_file):
        try:
            with open(today_file, "r") as f: return json.load(f)
        except Exception: return {}
        
    files = sorted(glob.glob("predictions_log-*.json"), reverse=True)
    if files:
        try:
            with open(files[0], "r") as f: return json.load(f)
        except Exception: pass
    return {}

def process_symbol(sym, log_data):
    from fetcher import fetch_history
    from analyze import predict_prices
    
    try:
        df = fetch_history(sym, years=1)
        if df is None or df.empty or len(df) < 5:
            return sym, None
            
        today_str = datetime.now().strftime("%Y-%m-%d")
        latest_dataset_date = df["date"].iloc[-1].strftime("%Y-%m-%d")
        actual_close = float(df["close"].iloc[-1])
        
        history = log_data.get(sym, [])
        cleaned_history = []
        
        for entry in history:
            date_val = entry.get("date", entry.get("prediction_for"))
            if not date_val: continue
            
            p_close = entry.get("predicted_close")
            if p_close is None: continue
            
            a_close = entry.get("actual_close")
            e_pct = entry.get("error_pct")
            
            # Evaluate an unevaluated past prediction if it matches the latest true closing price
            if date_val == latest_dataset_date and a_close is None:
                a_close = actual_close
                if p_close != 0:
                    e_pct = round(((a_close - p_close) / p_close) * 100.0, 2)
            
            cleaned_history.append({
                "date": date_val,
                "predicted_close": round(float(p_close), 2),
                "actual_close": round(float(a_close), 2) if a_close is not None else None,
                "error_pct": round(float(e_pct), 2) if e_pct is not None else None
            })
            
        # Backfill today's simulation if script hasn't been run for a day and evaluate directly
        has_today = any(e["date"] == today_str for e in cleaned_history)
        if latest_dataset_date == today_str and not has_today and len(df) > 20:
            bt_pred = predict_prices(df.iloc[:-1].copy(), 1)[0]
            bt_p_close = float(bt_pred["predicted_close"])
            bt_error = round(((actual_close - bt_p_close) / bt_p_close) * 100.0, 2)
            cleaned_history.append({
                "date": today_str,
                "predicted_close": round(bt_p_close, 2),
                "actual_close": round(actual_close, 2),
                "error_pct": bt_error
            })
            
        # Forecast actual upcoming day
        predictions = predict_prices(df, 1)
        next_day = predictions[0]
        next_date = next_day["date"]
        next_p_close = float(next_day["predicted_close"])
        
        has_tomorrow = any(e["date"] == next_date for e in cleaned_history)
        if not has_tomorrow:
            cleaned_history.append({
                "date": next_date,
                "predicted_close": round(next_p_close, 2),
                "actual_close": None,
                "error_pct": None
            })
            
        return sym, cleaned_history
    except Exception:
        return sym, None

def main():
    final_output = {}
    
    with suppress_stdout():
        from fetcher import fetch_company_list
        companies = fetch_company_list()
        
    if companies is None or companies.empty:
        print("{}")
        return
        
    symbols = companies["symbol"].tolist()
    symbols = [str(s) for s in symbols if s]
    
    with suppress_stdout():
        log_data = load_latest_log()
        
        # Threaded evaluation for maximum speed
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            fs = {executor.submit(process_symbol, sym, log_data): sym for sym in symbols}
            
            for future in concurrent.futures.as_completed(fs):
                sym, history = future.result()
                if history:
                    # Maintain chronological order
                    history.sort(key=lambda x: x["date"])
                    final_output[sym] = history

    # Output ONLY strictly formatted JSON to standard output (as requested)
    print(json.dumps(final_output, indent=2))
    
    # Store silently for subsequent continuity execution
    today_str = datetime.now().strftime("%Y-%m-%d")
    today_file = f"predictions_log-{today_str}.json"
    try:
        with open(today_file, "w", encoding="utf-8") as f:
            json.dump(final_output, f, indent=2)
    except Exception:
        pass

if __name__ == "__main__":
    main()
