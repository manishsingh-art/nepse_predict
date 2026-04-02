from __future__ import annotations

import hashlib
import json
import pickle
import time
from pathlib import Path
from typing import Any, Iterable, Optional


PROJECT_ROOT = Path(__file__).resolve().parent
CACHE_DIR = PROJECT_ROOT / ".cache"


def ensure_cache_dir(*parts: str) -> Path:
    path = CACHE_DIR.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_cache_path(*parts: str) -> Path:
    if not parts:
        ensure_cache_dir()
        return CACHE_DIR
    ensure_cache_dir(*parts[:-1])
    return CACHE_DIR.joinpath(*parts)


def file_age_seconds(path: Path) -> Optional[float]:
    try:
        if not path.exists():
            return None
        return max(0.0, time.time() - path.stat().st_mtime)
    except Exception:
        return None


def is_cache_fresh(path: Path, max_age_seconds: Optional[int]) -> bool:
    if max_age_seconds is None:
        return path.exists()
    age = file_age_seconds(path)
    return age is not None and age <= float(max_age_seconds)


def load_pickle(path: Path, max_age_seconds: Optional[int] = None) -> Any:
    try:
        if not path.exists():
            return None
        if max_age_seconds is not None and not is_cache_fresh(path, max_age_seconds):
            return None
        with open(path, "rb") as handle:
            return pickle.load(handle)
    except Exception:
        return None


def save_pickle(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(payload, handle)
    return path


def load_json(path: Path, max_age_seconds: Optional[int] = None) -> Any:
    try:
        if not path.exists():
            return None
        if max_age_seconds is not None and not is_cache_fresh(path, max_age_seconds):
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def fingerprint_parts(parts: Iterable[Any]) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(str(part).encode("utf-8", errors="ignore"))
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def dataframe_fingerprint(df: Any, columns: Optional[list[str]] = None) -> str:
    if df is None:
        return "none"
    try:
        import pandas as pd

        if not isinstance(df, pd.DataFrame) or df.empty:
            return "empty"
        use_cols = [c for c in (columns or list(df.columns)) if c in df.columns]
        if not use_cols:
            return "no_cols"
        subset = df[use_cols].copy()
        hashed = pd.util.hash_pandas_object(subset, index=True).values.tobytes()
        digest = hashlib.sha256()
        digest.update(hashed)
        return digest.hexdigest()[:16]
    except Exception:
        return fingerprint_parts([repr(df)])
