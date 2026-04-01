from __future__ import annotations

"""
prediction_engine.py
--------------------
Shared, deterministic helpers to keep all run modes consistent.
"""

import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def set_global_determinism(seed: int = 42) -> None:
    """
    Best-effort determinism across numpy/python and common ML libs.
    """
    seed_i = int(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed_i))
    random.seed(seed_i)
    np.random.seed(seed_i)


@dataclass(frozen=True)
class SentimentResult:
    baseline_score: float
    final_score: float
    source: str  # "keyword" | "ollama" | "none"
    reason: str = ""
    category: str = "None"


def keyword_sentiment_score(headlines: List[str]) -> Tuple[float, str]:
    bull_words = ["surge", "profit", "dividend", "growth", "bull", "positive", "upgrade", "accumulate"]
    bear_words = ["loss", "decline", "crash", "bear", "negative", "penalty", "downgrade", "sell"]
    if not headlines:
        return 0.0, "No headlines"
    score = 0.0
    for t in headlines:
        tl = (t or "").lower()
        if any(w in tl for w in bull_words):
            score += 0.5
        if any(w in tl for w in bear_words):
            score -= 0.5
    score = float(max(-1.0, min(1.0, score / max(1, len(headlines)))))
    return score, "Keyword basic scoring"


def compute_sentiment(
    headlines: List[str],
    use_ollama: bool,
    ollama_model: str,
    analyze_sentiment_headlines_fn: Optional[Any] = None,
) -> SentimentResult:
    """
    Deterministic baseline always computed; optional Ollama replaces the score.
    """
    baseline, baseline_reason = keyword_sentiment_score(headlines)
    if not use_ollama or analyze_sentiment_headlines_fn is None:
        return SentimentResult(baseline_score=baseline, final_score=baseline, source="keyword", reason=baseline_reason)

    try:
        res: Dict[str, Any] = analyze_sentiment_headlines_fn(headlines, model=ollama_model)
        final = float(res.get("score", baseline))
        return SentimentResult(
            baseline_score=baseline,
            final_score=float(max(-1.0, min(1.0, final))),
            source="ollama",
            reason=str(res.get("reason", "")),
            category=str(res.get("category", "None")),
        )
    except Exception:
        return SentimentResult(baseline_score=baseline, final_score=baseline, source="keyword", reason=baseline_reason)

