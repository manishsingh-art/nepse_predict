# NEPSE Predictor — Complete Project Documentation

> **Disclaimer:** This project is for **education and research only**. Not financial advice. Markets are inherently uncertain; treat every forecast as probabilistic, not deterministic.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture Diagram](#2-architecture-diagram)
3. [Module Reference](#3-module-reference)
4. [Data Pipeline](#4-data-pipeline)
5. [Feature Engineering](#5-feature-engineering)
6. [ML Ensemble & Training](#6-ml-ensemble--training)
7. [Market Regime Detection](#7-market-regime-detection)
8. [Smart Money / Floorsheet Analysis](#8-smart-money--floorsheet-analysis)
9. [Decision Engine](#9-decision-engine)
10. [Backtesting](#10-backtesting)
11. [Calendar & Nepal-Specific Logic](#11-calendar--nepal-specific-logic)
12. [Caching System](#12-caching-system)
13. [CLI Reference](#13-cli-reference)
14. [Configuration & Tuning](#14-configuration--tuning)
15. [Outputs & Reports](#15-outputs--reports)
16. [Testing](#16-testing)
17. [Dependency Map](#17-dependency-map)
18. [Bug Fix Changelog](#18-bug-fix-changelog)
19. [Development Guide](#19-development-guide)

---

## 1. Project Overview

**NEPSE Predictor** is a production-grade machine-learning pipeline purpose-built for the Nepal Stock Exchange (NEPSE). It integrates multi-source data fetching, 120+ technical/calendar features, a stacked ML ensemble, regime detection, smart-money floorsheet analysis, and optional LLM-powered narrative summaries — all designed around NEPSE's unique market structure (Sunday–Thursday sessions, Bikram Sambat calendar, circuit breakers, low liquidity).

### Key Capabilities

| Capability | Description |
|---|---|
| Multi-source data fetch | merolagani.com → nepalstock.com.np → sharesansar.com (priority cascade) |
| 120+ features | Price action, technical indicators, Nepal calendar effects, festival proximity |
| Stacked ensemble | LightGBM + XGBoost + GBM + RandomForest + Ridge → Ridge meta-learner |
| Regime detection | BULL / BEAR / SIDEWAYS / MANIPULATION with confidence score |
| Smart money signals | Floorsheet broker HHI, accumulation/distribution, trap probability |
| Walk-forward backtest | Purged k-fold, no look-ahead bias, ATR-based SL/TP, buy-and-hold benchmark |
| Decision engine | Composite score from model + technicals + regime + sentiment → BUY/SELL/HOLD/AVOID |
| Optional Ollama LLM | Local LLM (llama3) for headline sentiment + analyst narrative summary |
| Batch mode | `run_all.py` parallelises across all NEPSE symbols, outputs daily JSON log |

### Version History

| Version | Major additions |
|---|---|
| v6.0 | Smart-money floorsheet analysis, regime detection, FOMO/panic behavioral features |
| v6.1 | GARCH-style volatility, price-volume divergence, ATR-based SL/TP, NLP sentiment |
| v6.2 | Anomaly detection correlated with news categories, stronger model failure handling |
| v6.3 | Multi-step entry confirmation, drawdown-aware throttling, buy-and-hold alpha benchmark |
| Current (v6.4) | **Bug fixes:** infinite recursion in `nepse_market_calendar`, new-listing fetch (`pd.read_html` lxml crash, min-row threshold), auto-refresh symbol cache for unknown tickers |

---

## 2. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        nepse_live.py (CLI)                      │
│            Interactive menu + single-symbol analysis            │
└────────────────────┬────────────────────────────────────────────┘
                     │ orchestrates
         ┌───────────▼──────────────┐
         │       pipeline.py        │   PipelineResult dataclass
         │  prepare → train → eval  │
         └──┬────────┬─────────┬────┘
            │        │         │
     ┌──────▼──┐ ┌───▼───┐ ┌──▼────────────┐
     │fetcher.py│ │models │ │backtest_engine│
     │ (OHLCV + │ │  .py  │ │     .py       │
     │ sentiment│ │       │ │               │
     │ + floor) │ └───┬───┘ └───────────────┘
     └──────────┘     │
                 ┌────▼──────────────────────────────────────┐
                 │             features.py                    │
                 │   120+ features (price, TA, calendar,      │
                 │   smart money, sentiment injection)         │
                 └───────────────────────────────────────────┘
                          │             │
                   ┌──────▼──┐   ┌──────▼──────┐
                   │regime.py│   │smart_money.py│
                   └─────────┘   └─────────────┘
                          │             │
                   ┌──────▼─────────────▼──┐
                   │    decision_engine.py   │
                   │  BUY / SELL / HOLD /    │
                   │  AVOID + rationale      │
                   └────────────────────────┘

Support layers:
  cache.py            ← disk-based pickle/JSON cache with TTL
  nepal_calendar.py   ← BS↔AD conversion, holiday engine, 28 calendar features
  nepse_market_calendar.py ← authoritative NEPSE session rules
  ollama_ai.py        ← optional local LLM (llama3) integration
  analyze.py          ← standalone CSV-based technical analysis (backward compat)
  run_all.py          ← parallel batch runner across all symbols
```

---

## 3. Module Reference

### `nepse_live.py` — Main Entry Point
The primary CLI. Accepts `--symbol`, `--predict`, `--years`, `--backtest`, `--fast`, `--no-ml`, `--ollama`, `--ollama-model`, `--seed`, `--debug`, `--list`, `--allow-unknown-symbol`. Coordinates all other modules into a coloured terminal report.

**Auto-refresh for unknown symbols (v6.4):** When `--symbol <SYM>` is passed and the symbol is not in the local 24-hour cache, the CLI now automatically calls `fetch_nepse_symbols(force_refresh=True)` to pull a fresh list (500+ securities) from MeroLagani before deciding whether the symbol is valid. If the refresh still does not find the symbol, a fuzzy-match suggestion and a "possible new listing — attempting direct fetch" warning are shown and the run continues rather than exiting.

**Very-new-listing handling (v6.4):** A two-tier detection was added after history is loaded:

| Condition | Label | Behaviour |
|---|---|---|
| `len(df) < 30` | VERY NEW LISTING | ML training skipped; price snapshot + available technical indicators shown |
| `30 ≤ len(df) < 60` | NEW LISTING | ML training attempted with lighter settings (`opt=False`, `trials=3`) |
| `len(df) ≥ 100` | Normal | Full ML ensemble with Optuna tuning |

**`regime_info` key guard (v6.4):** `volatility_pct` is accessed via `.get()` and rendered only when present, preventing a `KeyError` crash when the regime detector has insufficient data (fewer than 20 rows).

### `pipeline.py` — Orchestration Core
Three public functions:
- `prepare_pipeline_frame(data, market_data)` → `(clean_data, feature_frame, feature_cols)`
- `train_model(...)` → `(NEPSEEnsemble, clean_data, feature_frame, feature_cols)` (cache-aware)
- `run_full_pipeline(...)` → `PipelineResult`

`PipelineResult` fields: `clean_data`, `feature_frame`, `feature_cols`, `model`, `forecast`, `predictions`, `signals`, `backtest`.

### `fetcher.py` — Data Acquisition
Multi-source OHLCV fetch with priority cascade:
1. `merolagani.com` TechnicalChartHandler (fastest, free)
2. `nepalstock.com.np` official REST API
3. `sharesansar.com` HTML scrape (fallback)

Also fetches:
- Floorsheet (intraday broker-level trade data) from NEPSE API
- News/sentiment from NewsAPI, RSS feeds (Himalayan Times, Republica, OnlineKhabar), Reddit r/investing
- Live market price and index data

Cache TTLs:
| Resource | TTL |
|---|---|
| Symbol list | 24 h |
| Company metadata | 24 h |
| Historical OHLCV | 1 h |
| Live price | 2 min |
| Floorsheet | 15 min |
| News | 2 h |
| Market live | 3 min |

**New-listing support (v6.4):** `fetch_history` previously rejected any response with fewer than 20 rows, causing a crash for stocks listed within the last few weeks. The logic was changed to:

| Constant | Value | Meaning |
|---|---|---|
| `MIN_ROWS_SUFFICIENT` | 20 | Prefer this source and stop searching immediately |
| `MIN_ROWS_ACCEPT` | 5 | Accept any source with at least this many rows |

All three sources are now tried, the one with the most rows is kept (up to `MIN_ROWS_SUFFICIENT`), and a `WARNING: NEW LISTING` log message is emitted when fewer than 20 rows are available. Previously the function would raise `RuntimeError` for any brand-new listing.

**`pd.read_html` lxml fix (v6.4):** All six `pd.read_html(r.text)` calls were updated to `pd.read_html(io.StringIO(r.text))`. When `lxml` is installed as the parser backend, passing raw HTML text directly caused an `OSError: Error reading file` crash because lxml's `etree.parse` treated the HTML string as a file path rather than in-memory content.

**`allow_company_lookup` always enabled:** `fetch_history` is now always called with `allow_company_lookup=True` so the NepalStock API company-ID lookup is attempted for every symbol, including symbols resolved via the auto-refresh path.

### `features.py` — Feature Engineering
`build_features(df, sentiment_score, smart_money_info, include_context_features)` adds 120+ columns to an OHLCV DataFrame:

| Group | Examples |
|---|---|
| Returns | `ret_1d` … `ret_20d`, `log_ret_1d` … `log_ret_20d` |
| Moving averages | `sma_5/10/20/50/100`, `ema_5/10/20/50/100`, `dist_sma_*`, `dist_ema_*` |
| MA crossovers | `sma_5_20_cross`, `sma_10_50_cross`, `ema_12_26_cross` |
| Bollinger Bands | `bb_upper_20/50`, `bb_lower_20/50`, `bb_width_20/50`, `bb_pos_20/50`, `bb_squeeze_20/50` |
| Momentum | RSI-7/14/21 (Wilder's), MACD, Stoch K/D, Williams %R, CCI, ROC |
| Volatility | ATR-7/14/21 (Wilder's), `atr_pct_*`, Garman-Klass vol, GARCH-style rolling std |
| Volume | `vol_sma_5/20`, `vol_ratio_5_20`, `vol_zscore`, OBV (vectorised) |
| Price action | `body_size`, `upper_shadow`, `lower_shadow`, `gap_up/down`, `doji`, `inside_bar` |
| Nepal calendar | BS month/day/year, weekday (Sun=0), fiscal quarter, festival proximity |
| Holiday proximity | `np_is_pre_holiday`, `np_is_post_holiday`, `np_days_to_next_holiday`, `np_festival_proximity` |
| Regime | `regime` label (-1/0/+1), `regime_bars` streak, `is_bull`, `is_bear`, `regime_volatility` |
| Behavioral | FOMO index, panic index, price-volume divergence, smart-money proximity |
| Smart money | HHI, `trap_score`, `accumulation_flag`, `broker_concentration` (context-only) |
| Sentiment | `sentiment_score` scalar injection from news analysis |

#### Timing Contract
This module operates as a **pure end-of-day feature engine**:
- Row *t* represents a NEPSE session that has **fully closed**.
- All features at row *t* use only data from sessions *t* and earlier.
- `target_ret_1d[t] = close[t+1] / close[t] − 1` (next session).
- Smart-money context (`include_context_features=True`) is for inference only — never set `True` during training.

#### Key Implementation Decisions
| Fix | Detail |
|---|---|
| **Wilder's RSI** | `ewm(com=period-1)` replacing SMA — matches TradingView/Bloomberg |
| **Wilder's ATR** | `ewm(com=period-1)` replacing `ewm(span=period)` — correct alpha = 1/period |
| **OBV** | `np.sign(diff) × volume → cumsum` — fully vectorised, ~50× faster |
| **Streak** | `ne().cumsum() + groupby.cumcount` — fully vectorised |
| **Noise ratio** | `std / abs().rolling().mean()` — vectorised, no rolling `.apply()` |
| **`illiquid_flag`** | `vol.shift(1).rolling(60).quantile(0.20)` — lagged quantile prevents self-inclusion |
| **`bb_squeeze`** | `bb_width.shift(1).rolling(250).quantile(0.1)` — same fix |

> **Cache invalidation notice:** the RSI and ATR smoothing changes produce different numerical values from previous versions. Delete `.cache/models/` to retrain all models after upgrading.

Targets added by `add_targets()`:
- `target_ret_1d` — next-session return (primary training label)
- `target_next_close` — next-session close
- `target_date` — next NEPSE trading date

`get_feature_cols(df, train_df=None)` — prunes by missingness (>50%) and variance (<1e-12). Pass `train_df` to compute pruning statistics on the training split only, preventing feature-selection leakage.

### `models.py` — ML Ensemble
`NEPSEEnsemble` implements a two-layer stacked ensemble:

**Layer 1 base models:**
| Model | Weight | Notes |
|---|---|---|
| LightGBM | 0.30 | Primary; handles NaN natively |
| XGBoost | 0.25 | Ensemble diversity |
| GradientBoosting | 0.20 | sklearn fallback |
| RandomForest | 0.15 | Variance reduction |
| Ridge | 0.10 | Linear baseline |

**Layer 2:** Ridge meta-learner on OOF (out-of-fold) predictions.

Key methods:
- `fit(feature_frame, feature_cols)` — trains with optional Optuna tuning
- `predict_next_session(X_row, prev_price, history, next_date, regime_info)` → `ForecastPoint`
- `forecast(clean_data, feature_cols, horizon)` → `List[ForecastPoint]`

`ForecastPoint` fields: `date`, `predicted_close`, `predicted_return`, `direction_prob`, `lower_bound`, `upper_bound`.

Training details:
- Purged walk-forward cross-validation (embargo gap between train/val folds)
- Optuna hyperparameter optimisation (when `optimise=True`, default 10 trials)
- NEPSE circuit-breaker clamping: ±10% per session
- SHAP feature importance (optional, when `shap` is installed)

### `regime.py` — Market Regime Detection
`MarketRegimeDetector.detect_regime(df, smart_money_info)` classifies into:
- **BULL** — EMA alignment up, moderate volatility, rising volume
- **BEAR** — EMA alignment down, high volatility, elevated volume
- **SIDEWAYS** — No clear EMA trend, low volatility
- **MANIPULATION/PUMP** — Extreme volume spike + broker concentration

Uses: EMA 10/20/50, ATR, 5d/20d volume profiles, EMA velocity (slope).

### `smart_money.py` — Floorsheet Analysis
`SmartMoneyAnalyst.analyze_floorsheet(floorsheet_df, recent_ohlcv)` returns:
- **Broker HHI** (Herfindahl-Hirschman Index): `<1500` retail, `1500–2500` moderate, `>2500` institutional
- **Accumulation flag** — top-5 buyer share > 40% with positive net flow
- **Trap score** (0–100) — likelihood of false breakout / bull/bear trap
- **Broker concentration** — net directional flow of top brokers
- **Wash-trade detection** — same broker on both buy/sell sides

### `decision_engine.py` — Signal Aggregator
`compute_final_decision(DecisionInputs)` → `FinalDecision`

Composite weighted score:
```
score = 0.38 * model_score
      + 0.22 * expected_ret_score
      + 0.18 * technical_score
      + 0.14 * regime_score
      + 0.08 * sentiment_score
      - 0.22 * trap_penalty
      - 0.10 * volatility_penalty
      - 0.08 * illiquidity_penalty
```

Action thresholds: `score ≥ 0.20` → BUY, `score ≤ -0.20` → SELL, `trap > 75% and conf < 0.55` → AVOID, otherwise HOLD.

### `backtest_engine.py` — Walk-Forward Backtest
`BacktestConfig` controls: initial capital, max position size (20%), fee rate (0.4%), slippage (0.1%), entry/exit thresholds, regime-adjusted multipliers, stop-loss (5%), max drawdown threshold (10%).

`run_backtest(signals, market_data, config)` → `BacktestResult`:
- `summary`: total return, Sharpe, Sortino, max drawdown, win rate, exposure, buy-and-hold alpha
- `equity_curve`: daily portfolio values
- `trades`: individual `TradeRecord` objects with entry/exit reason, regime, volatility, signal strength

### `cache.py` — Caching Utilities
File-based cache under `.cache/` directory:
- `load_pickle` / `save_pickle` — for model artifacts and DataFrames
- `load_json` / `save_json` — for symbol lists and metadata
- `is_cache_fresh(path, max_age_seconds)` — TTL-based freshness check
- `dataframe_fingerprint(df, columns)` — SHA-256 hash for cache keying

### `nepal_calendar.py` — Nepal Calendar Engine
- Bikram Sambat (BS) ↔ Gregorian (AD) conversion tables (years 2000–2090 BS)
- `NepalMarketCalendar.is_trading_day(date)` — NEPSE session rules (Sun–Thu, public holidays excluded)
- 28 ML-ready calendar feature columns (BS month, weekday, fiscal quarter, festival proximity, etc.)
- Nepal Bandh detection via news RSS feeds

### `nepse_market_calendar.py` — Authoritative Session Calendar
Preferred over `nepal_calendar.py` in `models.py` and `pipeline.py`. Uses `nepse_holidays.csv` and `nepse_holidays_overrides.csv` for precise holiday management.

**Recursion fix (v6.4):** The original `market_status → is_pre_holiday → is_trading_day → market_status` cycle caused a `RecursionError` at startup. Two private helpers were added to break it:

| Helper | Purpose |
|---|---|
| `_is_non_trading_basic(dd)` | Non-recursive closure check — weekend + overrides + API holidays + static public holidays only. Never calls `is_trading_day` / `market_status` / `is_pre_holiday`. |
| `_compute_pre_holiday(dd)` | Checks whether next `lookahead_days` calendar day(s) are non-trading using `_is_non_trading_basic`. Called by `market_status` instead of the public `is_pre_holiday`. |

The public `is_pre_holiday(d, lookahead_days)` now uses `_is_non_trading_basic` for its self-check, eliminating the cycle entirely.

### `analyze.py` — Standalone Technical Analysis
Backward-compatible standalone CSV analyser. Supports column aliasing for varied CSV schemas. Used for offline/file-based analysis without live fetching.

### `ollama_ai.py` — LLM Integration
- `is_ollama_available(ollama_url)` — probes `localhost:11434` with 2s timeout
- `generate_ai_summary(symbol, analysis_data, model)` — sends structured market data prompt to local Ollama instance, returns 2-3 paragraph analyst narrative

### `run_all.py` — Batch Runner
Parallel symbol processing via `ThreadPoolExecutor` (default 8 workers). Outputs `predictions_log-YYYY-MM-DD.json` with per-symbol: predicted close, direction probability, regime, action, error info.

### `enhanced_model.py` / `enhanced_model_demo.py` — Plugin Extension Point
Experimental extension API. `EnhancedModel.compute_features()`, `calculate_confidence()`, and `decision_engine` can be overridden without touching core `models.py`.

---

## 4. Data Pipeline

```
fetch_ohlcv(symbol, years)
    └── Source 1: merolagani TechnicalChartHandler
    └── Source 2: nepalstock.com.np REST API
    └── Source 3: sharesansar.com HTML scrape
         │
         ▼
clean_ohlcv_data(df)                  ← normalise columns, drop NaN, sort
         │
         ▼
add_market_features(df, market_df)    ← NEPSE index correlation features
         │
         ▼
build_features(df, sentiment, smart_money_info)
         │    ← 120+ columns added
         ▼
add_targets(df)                       ← target_ret_1d, target_next_close, target_date
         │
         ▼
dropna(subset=["target_ret_1d", "target_date"])
         │
         ▼
get_feature_cols(df)                  ← filter to numeric, non-NaN, non-target columns
```

---

## 5. Feature Engineering

### Nepal Calendar Features (28 features)
Derived from `nepal_calendar.py`:
- `bs_year`, `bs_month`, `bs_day` — Bikram Sambat date components
- `nepse_weekday` — 0=Sunday, 4=Thursday (NEPSE trading week)
- `fiscal_quarter` — Nepal fiscal year quarters (Shrawan–Ashadh)
- `days_to_dashain`, `days_to_tihar`, `days_to_holi`, `days_to_new_year` — proximity to major festivals
- `is_pre_holiday`, `is_post_holiday`, `bandh_risk` — operational calendar flags
- `month_start`, `month_end`, `quarter_start`, `quarter_end` — institutional rebalancing signals

### Target Construction
- Primary label: `target_ret_1d = (next_close - close) / close` (next NEPSE trading session)
- Forecast uses `next_nepse_trading_dates()` to skip weekends and holidays

---

## 6. ML Ensemble & Training

### Cross-Validation Strategy
Purged walk-forward splits (`purged_walk_forward_splits`):
- Embargo gap between train and validation folds prevents data leakage
- Default fold count: `min(5, max(3, len(data) // 180))`
- Minimum training set: `max(60, len(data) // 5)` rows

### Optuna Hyperparameter Search
When `optimise=True` and `optuna` is installed:
- LightGBM: searches `num_leaves`, `learning_rate`, `feature_fraction`, `min_child_samples`, `reg_alpha`, `reg_lambda`
- XGBoost: searches `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`
- Default: 10 trials per model (configurable with `--n-opt-trials` via nepse_live)

### Model Caching
Trained model artifacts are cached as `.pkl` files under `.cache/models/`. Cache key is a SHA-256 fingerprint of: symbol, OHLCV data hash, market data hash, optimise flag, fold count, trial count, random seed.

---

## 7. Market Regime Detection

```
Inputs: OHLCV DataFrame (≥20 rows), optional smart_money_info

Step 1: EMA alignment
  ├── EMA 10 > EMA 20 > EMA 50 → uptrend flag
  └── EMA 10 < EMA 20 < EMA 50 → downtrend flag
  EMA velocity (5-period slope of EMA20) for early transition

Step 2: ATR volatility
  └── ATR(20) / close × 100 = volatility_pct

Step 3: Volume profile
  ├── vol_ratio = today_vol / 20d_avg
  └── vol_momentum = 5d_avg / 20d_avg

Step 4: Scoring + classification
  → BULL: uptrend + vol_momentum > 1.1 + volatility_pct < 3%
  → BEAR: downtrend + vol_ratio > 1.5 + volatility_pct > 3%
  → MANIPULATION: vol_ratio > 3.0 OR smart_money trap_score > 70
  → SIDEWAYS: default
```

---

## 8. Smart Money / Floorsheet Analysis

Floorsheet data (intraday broker-level trades) is fetched from `nepalstock.com.np/api/nots/floorsheet`.

| Metric | Interpretation |
|---|---|
| HHI < 1500 | Retail-dominated, organic price discovery |
| HHI 1500–2500 | Moderate institutional presence |
| HHI > 2500 | Operator/institutional dominance |
| `accumulation_flag = 1` | Top-5 brokers net buying > 40% of volume |
| `trap_score > 70` | High probability of bull/bear trap |
| `net_flow_top10_norm > 0` | Smart money net accumulating |

Wash-trade detection: broker appears on both buy and sell sides of same stock in same session.

---

## 9. Decision Engine

The decision engine synthesises all signals into a single actionable output:

```python
DecisionInputs(
    direction_prob=0.68,        # from ML model
    expected_ret_5d_pct=3.2,    # from forecast
    regime="BULL",
    regime_confidence=0.75,
    sentiment_score=0.3,        # from Ollama/RSS
    trap_score=15.0,            # from SmartMoneyAnalyst
    volatility_pct=2.1,         # from regime detector
    illiquid_flag=0.0,
    technical_signals=[...],    # from analyze.py suggest_strategy()
)
# → FinalDecision(action="BUY", confidence=0.62, score=0.31, rationale="...")
```

---

## 10. Backtesting

### BacktestConfig Defaults
```python
initial_capital = 100_000        # NPR
max_position_size = 0.20         # 20% per trade
fee_rate = 0.004                 # 0.4% (SEBON + broker)
slippage_rate = 0.001            # 0.1%
entry_threshold = 0.003          # min predicted return to enter
min_direction_prob = 0.55        # min P(up)
min_hold_days = 3
cooldown_days = 2                # after exit, wait 2 sessions
stop_loss_pct = 0.05             # 5% hard stop
max_drawdown_threshold = 0.10    # 10% portfolio drawdown triggers scaling
```

### Regime-Adjusted Entry Rules
- **SIDEWAYS**: entry threshold × 1.15 (tighter), position scaled to 50%
- **BEAR**: entry threshold × 1.35, `min_direction_prob` raised to 0.65

### Output Metrics
`total_return_pct`, `sharpe_ratio`, `sortino_ratio`, `max_drawdown_pct`, `win_rate`, `total_trades`, `avg_hold_days`, `exposure_pct`, `buy_hold_return_pct`, `alpha_vs_buy_hold`.

---

## 11. Calendar & Nepal-Specific Logic

### NEPSE Trading Schedule
- **Days:** Sunday–Thursday (Friday and Saturday are always closed)
- **Hours:** 11:00–15:00 NST (UTC+5:45)
- **Circuit breaker:** ±10% per session hard limit

### Bikram Sambat Calendar
- Nepal's official calendar; ~56.7 years ahead of Gregorian
- Lookup tables for years 2000–2090 BS stored in `nepal_calendar.py`
- Conversion: `ad_to_bs(date)` / `bs_to_ad(year, month, day)`

### Holiday Sources
- `nepse_holidays.csv` — static annual holiday list
- `nepse_holidays_overrides.csv` — ad-hoc overrides
- Live NRB (Nepal Rastra Bank) RSS feed (when network available)

### Fiscal Year
Nepal fiscal year: **Shrawan to Ashadh** (mid-July to mid-July). Quarter boundaries are calendar features that capture institutional rebalancing effects.

---

## 12. Caching System

All cache files reside under `.cache/` (git-ignored):

```
.cache/
  companies_cache.csv          ← company metadata (24h TTL)
  symbols_cache.json           ← NEPSE symbol list (24h TTL)
  models/
    NABIL_d<hash>_m<hash>_opt1_f0_t10_r42.pkl   ← trained model
  <hash>.json                  ← price and floorsheet data
```

Cache key construction for models (`_model_cache_file`):
```
{SYMBOL}_d{ohlcv_sha256[:16]}_m{market_sha256[:16]}_opt{0|1}_f{folds}_t{trials}_r{seed}
```

---

## 13. CLI Reference

### `nepse_live.py`

```
python nepse_live.py [--symbol SYM] [--predict N] [--years N]
                     [--backtest] [--fast] [--no-ml]
                     [--ollama] [--ollama-model NAME]
                     [--seed N] [--debug] [--list]
                     [--allow-unknown-symbol]
```

| Flag | Default | Description |
|---|---|---|
| `--symbol` | interactive | NEPSE ticker (e.g. `NABIL`, `NTC`, `SKHL`) |
| `--predict` | 7 | Forecast horizon in trading sessions |
| `--years` | 2 | Historical data range |
| `--backtest` | off | Run walk-forward backtest |
| `--fast` | off | ~1 year history, lighter Optuna run |
| `--no-ml` | off | Deprecated — ignored; pipeline is ML-only |
| `--ollama` | off | Enable local LLM sentiment + summary |
| `--ollama-model` | `llama3` | Ollama model name |
| `--seed` | 42 | Reproducibility seed |
| `--debug` | off | Extra diagnostic output (feature snapshot, reasoning) |
| `--list` | — | Print all available symbols and exit |
| `--allow-unknown-symbol` | off | Bypass local symbol validation; attempt direct fetch (useful when symbols cache is stale and auto-refresh also fails) |

> **New-listing tip:** For stocks listed within the last few weeks, simply run `python nepse_live.py --symbol <SYM>`. The CLI will auto-refresh the symbol list from MeroLagani. The `--allow-unknown-symbol` flag is only needed if the auto-refresh itself fails (e.g. no internet).

### `run_all.py`

```
python run_all.py [--symbols SYM ...] [--workers N] [--no-ml]
```

| Flag | Default | Description |
|---|---|---|
| `--symbols` | all NEPSE | Space-separated ticker list |
| `--workers` | 8 | Thread pool size |
| `--no-ml` | off | (deprecated, ignored) |

### `analyze.py`

```
python analyze.py --old old_data.csv --new new_data.csv --symbol NABIL
python analyze.py --file combined_data.csv --symbol NABIL --predict 10
```

---

## 14. Configuration & Tuning

### Ensemble Weights (`models.py`)
```python
ENSEMBLE_WEIGHTS = {"lgb": 0.30, "xgb": 0.25, "gbm": 0.20, "rf": 0.15, "ridge": 0.10}
```
Modify these for different market regimes or data availability.

### Decision Engine Weights (`decision_engine.py`)
```python
score = 0.38*model + 0.22*ret + 0.18*tech + 0.14*regime + 0.08*sentiment
      - 0.22*trap_pen - 0.10*vol_pen - 0.08*illiq_pen
```
Tune the trap penalty weight (`0.22`) for noisier markets.

### Backtest Thresholds (`backtest_engine.py`)
Key parameters to tune per symbol liquidity class:
- `entry_threshold` — lower for liquid large-caps (NABIL, NTC), higher for illiquid small-caps
- `max_position_size` — reduce for illiquid stocks
- `stop_loss_pct` — widen for high-ATR stocks

---

## 15. Outputs & Reports

### Terminal Output
Colorized (via `colorama`) sections:
1. Data fetch summary
2. Regime and smart-money block
3. Technical signals and strategy
4. ML forecast table (next N sessions with predicted price and bounds)
5. Backtest summary (if `--backtest`)
6. Ollama AI narrative (if `--ollama`)

### `reports/` Directory
Per-run JSON with:
- `symbol`, `run_date`, `regime`, `direction_prob`
- `forecast_points` — list of `{date, predicted_close, lower, upper}`
- `backtest_summary` — Sharpe, drawdown, win rate, alpha
- `anomalies` — detected anomalous sessions with news correlation
- `smart_money` — HHI, trap_score, accumulation_flag
- `final_decision` — action, confidence, score, rationale

### `predictions_log-YYYY-MM-DD.json`
Rolling batch log for all symbols run that day:
```json
{
  "NABIL": {
    "predicted_close": 1234.56,
    "direction_prob": 0.68,
    "regime": "BULL",
    "action": "BUY",
    "confidence": 0.62
  }
}
```

---

## 16. Testing

```bash
# Feature leakage, correctness, and performance audit (pytest)
pytest test_features_leakage.py -v

# Primary test: calendar correctness + determinism across seeds
python test_calendar_and_determinism.py

# Pipeline consistency (walk-forward, cache invalidation)
python test_pipeline_consistency.py

# Ad-hoc fetch validation
python test_nepse_live.py
python test_live.py
```

### What `test_features_leakage.py` checks (pytest, 30+ tests):
- No future data used in any feature (perturb future rows → past features unchanged)
- Lag feature alignment (ret_lag_k[t] == ret_1d[t-k])
- Rolling correctness: SMA spot-check, RSI bounds [0,100], ATR ≥ 0
- OBV vectorised vs loop reference (must be identical)
- `_streak` vectorised vs loop reference (must be identical)
- `noise_20` vectorised vs manual calculation
- `illiquid_flag` threshold excludes current volume (LEAK-02)
- `bb_squeeze` threshold excludes current BB-width (LEAK-03)
- Wilder's RSI smoothness test (no abrupt SMA boundary jumps)
- ATR uses Wilder's alpha (diverges from span-based)
- `get_feature_cols` with `train_df` excludes columns sparse in training (LEAK-01)
- Target alignment (target_ret_1d[t] = close[t+1]/close[t] - 1)
- NaN levels: no BASE_FEATURE exceeds 5% NaN after row 100 on a 400-row series
- Determinism: same input → identical output on two separate calls

### What `test_calendar_and_determinism.py` checks:
- BS↔AD round-trip conversion accuracy
- `is_trading_day()` correctness for known holidays/weekends
- Pipeline output determinism given same seed and data snapshot

---

## 17. Dependency Map

```
nepse_live.py
├── fetcher.py
│   └── cache.py
├── pipeline.py
│   ├── features.py
│   │   └── nepal_calendar.py
│   ├── models.py
│   │   ├── nepal_calendar.py
│   │   └── nepse_market_calendar.py
│   ├── backtest_engine.py
│   ├── cache.py
│   └── regime.py
├── analyze.py
│   └── nepal_calendar.py
├── smart_money.py
├── decision_engine.py
├── ollama_ai.py
└── nepse_market_calendar.py
    ├── nepal_calendar.py
    └── nepse_holidays.csv / nepse_holidays_overrides.csv
```

---

## 18. Bug Fix Changelog

### v6.4 Fixes (2026-04-02)

#### 1. Infinite recursion in `nepse_market_calendar.py` (`RecursionError`)
**Symptom:** Running any `--symbol` command crashed with `RecursionError: maximum recursion depth exceeded` before producing any output.

**Root cause:** Mutual recursion cycle:
```
market_status(d)
  └─ is_pre_holiday(d)
       └─ is_trading_day(d)
            └─ market_status(d)   ← infinite loop
```

**Fix:** Added two private helpers that bypass the cycle:
- `_is_non_trading_basic(dd)` — checks weekend + overrides + API holidays + static public holidays directly, never calls `is_trading_day` / `market_status`.
- `_compute_pre_holiday(dd)` — uses `_is_non_trading_basic` to check next day(s). Called by `market_status` at the end of its happy path instead of the public `is_pre_holiday`.

The public `is_pre_holiday` was updated to use `_is_non_trading_basic` for its own-day guard as well.

---

#### 2. New / recently-listed stocks not fetchable (`RuntimeError`)
**Symptom:** `python nepse_live.py --symbol SKHL` (or any stock listed within the last few weeks) exited with:
```
Error: Could not fetch data for 'SKHL' from any source.
```

**Root causes (three separate issues):**

| # | Bug | Fix |
|---|---|---|
| 2a | `pd.read_html(r.text)` crashed with `OSError: Error reading file '<!DOCTYPE html>…'` when `lxml` is installed — lxml treats the HTML string as a file path | Changed all 6 calls to `pd.read_html(io.StringIO(r.text))` in `fetcher.py` |
| 2b | `fetch_history` rejected responses with < 20 rows, so brand-new listings (7–15 rows) were discarded from every source and the function raised `RuntimeError` | Added `MIN_ROWS_ACCEPT = 5` threshold; the best result across all sources is kept even for new listings, with a warning logged |
| 2c | `allow_company_lookup=bool(cid)` meant the NepalStock API ID lookup was skipped for all symbols not pre-loaded from the company list | Changed to `allow_company_lookup=True` always |

---

#### 3. Unknown symbol exits immediately without trying network
**Symptom:** A stock not yet in the 24-hour local symbol cache caused:
```
Error: Symbol XYZ not found in local symbol cache.
Tip: if this is a new listing, retry with --allow-unknown-symbol
```

**Fix:** When `is_known_symbol_local()` returns False, the CLI now automatically calls `fetch_nepse_symbols(force_refresh=True)` to refresh from MeroLagani (500+ symbols). Only if the symbol is still absent after the network refresh does the run proceed with a "possible new listing" warning instead of a hard exit. Fuzzy-match suggestions (top-3, cutoff 75%) are shown when available.

---

#### 4. `KeyError: 'volatility_pct'` crash for ultra-new listings
**Symptom:** Stocks with fewer than ~20 rows crashed at the regime display block with `KeyError: 'volatility_pct'` because the regime detector could not compute a 20-day rolling volatility.

**Fix:** `volatility_pct` is now read with `.get('volatility_pct')` and the line is skipped entirely when the value is `None`.

---

## 19. Development Guide

### Adding a New Feature
1. Open `features.py`
2. Add your computation inside `build_features()` — the DataFrame is sorted ascending by date
3. **Follow the timing contract**: feature at row *t* must only use data from rows ≤ *t* (end-of-day model — `close[t]` is observable)
4. Use vectorised pandas/numpy operations; avoid row-by-row Python loops
5. Add the new column name to `BASE_FEATURES` (or `NEPAL_FEATURES` for calendar columns) so `get_feature_cols()` picks it up
6. Add a corresponding test in `test_features_leakage.py` verifying no future data is used

### Adding a New Data Source
1. Implement a `fetch_*` function in `fetcher.py`
2. Add it as the next fallback in the priority cascade (after existing sources)
3. Return a DataFrame with normalised columns: `date`, `open`, `high`, `low`, `close`, `volume`
4. Respect the existing cache TTL pattern using `get_cache_path` and `save_json`/`load_json`

### Adding a New Model to the Ensemble
1. Define the model in `models.py` following the existing `_build_lgb` / `_build_xgb` pattern
2. Add Optuna search space in `_optuna_lgb` / `_optuna_xgb` equivalents
3. Add it to `ENSEMBLE_WEIGHTS` (weights must sum to 1.0)
4. Include a `HAS_*` guard for optional imports

### Extending the Decision Engine
Override `EnhancedModel` in `enhanced_model.py` rather than editing `decision_engine.py`. The plugin API allows `compute_features()`, `calculate_confidence()`, and custom decision logic without touching core modules.

### Environment Setup
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt

# Optional: Ollama local LLM
ollama pull llama3

# Optional: SHAP explainability
pip install shap
```

### Reproducibility
Always pass `--seed N` for reproducible runs. The seed propagates to: Optuna sampler, sklearn `random_state`, LightGBM/XGBoost seeds, and numpy global seed during training.
