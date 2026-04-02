#!/usr/bin/env python3
"""
predict_api.py — JSON bridge for the Next.js dashboard.

Runs the full NEPSE ML pipeline for a given symbol and emits the
latest report JSON to stdout. Called by the Next.js API route via
child_process.exec.

Usage:
    python predict_api.py --symbol NABIL
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Force UTF-8 on Windows stdout so emoji in reports don't crash the pipe
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Run NEPSE prediction and return JSON")
    parser.add_argument("--symbol", required=True, help="NEPSE stock symbol (e.g. NABIL)")
    args = parser.parse_args()
    symbol = args.symbol.upper().strip()

    start_ts = datetime.now()

    cmd = [sys.executable, str(ROOT / "nepse_live.py"), "--symbol", symbol]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            timeout=600,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.TimeoutExpired:
        _fail(symbol, "Pipeline timed out after 600 seconds")
        return
    except Exception as exc:
        _fail(symbol, str(exc))
        return

    if proc.returncode != 0:
        # nepse_live.py prints user-facing "Error: ..." lines to stdout;
        # Python logging (INFO/DEBUG) goes to stderr.  Scan both streams for
        # the first meaningful error line.
        combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
        meaningful = ""
        for raw_line in combined.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            lower = line.lower()
            # Skip pure logging/banner lines
            if any(lower.startswith(p) for p in ("info:", "debug:", "warning:", "█", "╗", "╝", "╔", "╚", "╠", "║", "▶")):
                continue
            # Prefer lines that mention "error", "could not", "failed", "minimum", "insufficient"
            if any(kw in lower for kw in ("error", "could not", "failed", "minimum", "insufficient", "no data", "exception")):
                meaningful = line.lstrip("! ").strip()
                # Remove leading "Error:" prefix for clarity
                if meaningful.lower().startswith("error:"):
                    meaningful = meaningful[6:].strip()
                break
        detail = meaningful or (proc.stderr or "")[:300] or f"exit code {proc.returncode}"
        _fail(symbol, detail)
        return

    # Prefer reports created after the run started; fall back to any existing report
    pattern = str(ROOT / "reports" / f"{symbol}_*.json")
    fresh = [
        p for p in glob.glob(pattern)
        if datetime.fromtimestamp(Path(p).stat().st_mtime) >= start_ts
    ]
    candidates = fresh or glob.glob(pattern)

    if not candidates:
        _fail(symbol, f"No report file found for {symbol} after pipeline run")
        return

    latest = sorted(candidates)[-1]
    try:
        with open(latest, encoding="utf-8") as fh:
            report = json.load(fh)
        print(json.dumps(report, ensure_ascii=True, default=str))
    except Exception as exc:
        _fail(symbol, f"Failed to read report {latest}: {exc}")


def _fail(symbol: str, message: str) -> None:
    print(json.dumps({"error": message, "symbol": symbol}), file=sys.stdout)
    sys.exit(1)


if __name__ == "__main__":
    main()
