import json
import logging
import requests
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

def is_ollama_available(ollama_url: str = "http://localhost:11434") -> bool:
    """
    Returns True if a local Ollama server appears reachable.
    Uses a quick tag endpoint probe to avoid long timeouts when Ollama isn't installed.
    """
    try:
        r = requests.get(f"{ollama_url.rstrip('/')}/api/tags", timeout=2)
        return r.status_code == 200
    except Exception:
        return False

def generate_ai_summary(
    symbol: str,
    analysis_data: Dict[str, Any],
    ollama_url: str = "http://localhost:11434/api/generate",
    model: str = "llama3"
) -> Optional[str]:
    """
    Sends structured market analysis data to a local Ollama instance 
    to generate a natural language summary.
    """
    base_url = ollama_url.split("/api/")[0] if "/api/" in ollama_url else "http://localhost:11434"
    if not is_ollama_available(base_url):
        return None

    prompt = f"""
Analyze the following NEPSE market data for {symbol} and provide a concise, 
professional summary (2-3 paragraphs) for a trader. 
Focus on trend direction, significant anomalies, and ML forecast sentiment.

Data:
- Current Price: {analysis_data.get('current_price')}
- Market Regime: {analysis_data.get('regime')}
- Smart Money: {analysis_data.get('smart_money')}
- Recommended Action: {analysis_data.get('strategy')}
- Sentiment: {analysis_data.get('sentiment_label')} ({analysis_data.get('sentiment_score')})
- Trend: {analysis_data.get('trend')} (Score: {analysis_data.get('trend_score')}/6)
- RSI: {analysis_data.get('rsi')} ({analysis_data.get('rsi_label')})
- ML Forecast: {analysis_data.get('forecast_summary')}
- Notable Anomalies: {analysis_data.get('anomalies')}

Instructions:
- Interpret the Smart Money and Regime context (e.g., bull market vs manipulation).
- Be objective and cautious.
- Keep it under 200 words.
"""
    
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "top_p": 0.9,
        }
    }
    
    try:
        response = requests.post(ollama_url, json=payload, timeout=30)
        if response.status_code == 200:
            return response.json().get("response")
        else:
            logger.warning(f"Ollama error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        logger.warning(f"Ollama connection failed: {e}")
        return None

def analyze_sentiment_headlines(
    headlines: list[str],
    ollama_url: str = "http://localhost:11434/api/generate",
    model: str = "llama3"
) -> Dict[str, Any]:
    """
    NLP-based sentiment analysis of news headlines using Ollama.
    Returns: {"score": float (-1 to 1), "reason": str, "category": str}
    """
    if not headlines:
        return {"score": 0.0, "reason": "No news available.", "category": "None"}

    base_url = ollama_url.split("/api/")[0] if "/api/" in ollama_url else "http://localhost:11434"
    if not is_ollama_available(base_url):
        return {"score": 0.0, "reason": "Ollama not available on this machine.", "category": "None"}
        
    text = "\n- ".join(headlines)
    prompt = f"""
You are a strict JSON generator for a NEPSE trading system.

Task:
- Read the headlines.
- Decide net market sentiment and the dominant driver category.
- Think step-by-step silently; output ONLY the final JSON object.

Output format (JSON only):
{{"score": <float>, "reason": <string>, "category": <string>}}

Rules:
- score range: -1.0 (very bearish) to 1.0 (very bullish)
- reason: max 18 words, mention the strongest driver
- category must be exactly one of: "Macro", "Corporate", "Sector"
- If mixed: pick the category that would move NEPSE the most this week.

Headlines:
- {text}
"""
    
    payload = {
        "model": model,
        "prompt": prompt,
        "format": "json",
        "stream": False,
        "options": {"temperature": 0.1}
    }
    
    try:
        response = requests.post(ollama_url, json=payload, timeout=30)
        if response.status_code == 200:
            res = json.loads(response.json().get("response", "{}"))
            return {
                "score": float(res.get("score", 0.0)),
                "reason": str(res.get("reason", "Parsed")),
                "category": str(res.get("category", "Macro"))
            }
        return {"score": 0.0, "reason": "Ollama error.", "category": "None"}
    except Exception:
        return {"score": 0.0, "reason": "NLP failed.", "category": "None"}

