# NEPSE Predictor — Backend & Frontend

> Research / education only. Not financial advice.

## Overview

This project extends the existing Python-based NEPSE prediction pipeline with a Next.js dashboard, Prisma-backed persistence, API routes, and interactive stock exploration.

Current runtime architecture:

```text
nepse_live.py / predict_api.py
        ↓ JSON via stdout
Next.js App Router (frontend/)
        ↓ service layer
Prisma ORM
        ↓
SQLite (current local schema)
```

Notes:

- The current checked-in Prisma schema uses `sqlite` for local development.
- The original design targeted PostgreSQL too, but the implementation was adapted so the app runs locally without requiring a Postgres instance.
- Python execution is still required for predictions; the dashboard does not replace the ML pipeline.

## Current structure

```text
nepse_predict/
├── predict_api.py
├── nepse_live.py
├── reports/
└── frontend/
    ├── prisma/
    │   └── schema.prisma
    ├── src/
    │   ├── app/
    │   │   ├── page.tsx
    │   │   ├── stocks/
    │   │   │   ├── page.tsx
    │   │   │   └── [symbol]/page.tsx
    │   │   └── api/
    │   │       ├── predict/route.ts
    │   │       └── stocks/
    │   │           ├── route.ts
    │   │           ├── sync/route.ts
    │   │           └── [symbol]/
    │   │               ├── route.ts
    │   │               └── accuracy/route.ts
    │   ├── components/
    │   │   ├── AccuracyMetrics.tsx
    │   │   ├── ActionBadge.tsx
    │   │   ├── ForecastTable.tsx
    │   │   ├── HistoricalTable.tsx
    │   │   ├── PriceChart.tsx
    │   │   ├── RunPredictionButton.tsx
    │   │   ├── SearchSort.tsx
    │   │   ├── SkeletonCard.tsx
    │   │   ├── StockCard.tsx
    │   │   ├── StockDetailClient.tsx
    │   │   └── SyncButton.tsx
    │   ├── lib/
    │   │   ├── nepse.ts
    │   │   ├── prisma.ts
    │   │   └── python-runner.ts
    │   ├── services/
    │   │   ├── accuracy.service.ts
    │   │   ├── prediction.service.ts
    │   │   └── stock.service.ts
    │   └── types/
    │       └── index.ts
    └── package.json
```

## What is implemented

### Backend

- `POST /api/predict` runs the Python pipeline, parses the emitted JSON report, stores forecast rows, and reconciles any newly verifiable accuracy rows.
- `GET /api/stocks` returns all companies with the latest prediction per symbol plus aggregated accuracy.
- `POST /api/stocks/sync` refreshes the company master list from `merolagani.com` and upserts it into `Stock`.
- `GET /api/stocks/[symbol]` returns stock metadata, latest prediction snapshot, latest 7 forecast rows, last 30 historical prediction rows, and up to 90 actual price rows.
- `GET /api/stocks/[symbol]/accuracy` returns overall accuracy plus a weekly breakdown.

### Frontend

- `/stocks` shows a server-rendered stock list with client-side search, sort, filter pills, clickable stat tiles, sync button, and bulk prediction workflow.
- `/stocks/[symbol]` shows a detail page with header metrics, chart, forecast table, historical table, and accuracy tab.
- Individual and bulk prediction actions surface pipeline errors to the UI instead of hiding them behind generic failures.
- Newly listed companies are marked using `nepseId` and exposed through both tile filters and inline badges.

## Quick start

### Prerequisites

- Node.js 20+
- Python environment with the NEPSE pipeline dependencies installed
- A working `.venv` for the root Python project

### Environment

From `frontend/`, create `.env.local`.

Typical Windows setup:

```env
DATABASE_URL="file:./dev.db"
NEPSE_ROOT="D:/new_py/nepse_predict"
PYTHON_EXEC="D:/new_py/nepse_predict/.venv/Scripts/python.exe"
```

Typical Linux/macOS setup:

```env
DATABASE_URL="file:./dev.db"
NEPSE_ROOT="/path/to/nepse_predict"
PYTHON_EXEC="/path/to/nepse_predict/.venv/bin/python"
```

Important:

- `schema.prisma` is currently configured for SQLite, so `DATABASE_URL` should match that unless you intentionally migrate the schema/provider.
- The checked-in `.env.local.example` is more PostgreSQL-oriented; use values that match the active Prisma provider.

### Database

```bash
cd frontend
npx prisma generate
npx prisma db push
```

Optional:

```bash
npx prisma studio
```

### Run

```bash
cd frontend
npm run dev
```

App entry:

- `http://localhost:3000/stocks`

## Data sources

### Company universe

Primary source:

- `https://www.merolagani.com/handlers/AutoSuggestHandler.ashx?type=Company`

`src/lib/nepse.ts` filters the raw autosuggest feed to keep ordinary listed equities only. It excludes promoter shares, many fixed-income instruments, mutual funds, and several non-equity institutional entries.

Fallback:

- A curated static seed list in `src/lib/nepse.ts`

Caching:

- In-memory cache for company list: 1 hour
- In-memory cache for single live price lookups: 5 minutes

### Prediction execution

`src/lib/python-runner.ts` calls:

```text
"<PYTHON_EXEC>" "<NEPSE_ROOT>/predict_api.py" --symbol SYMBOL
```

Key behaviors:

- Uses an 11-minute timeout
- Logs start/completion timing
- Parses Python-side JSON errors from `stdout`
- Falls back to `stderr` or Node exec errors if the subprocess exits non-zero

## API reference

### `POST /api/predict`

Runs the Python pipeline for one symbol.

Request:

```json
{ "symbol": "NABIL" }
```

Validation:

- Uppercase alphanumeric symbol
- Length 1 to 12
- Invalid JSON returns `400`

Success response shape:

```json
{
  "data": {
    "symbol": "NABIL",
    "generatedAt": "2026-04-02T16:42:13",
    "forecastCount": 7,
    "action": "SELL / REDUCE",
    "confidence": 1.1,
    "regime": "SIDEWAYS",
    "lastClose": 517.5,
    "mlMetrics": {
      "target_name": "target_ret_1d",
      "avg_mae": 0.736,
      "avg_dir_acc": 59.59,
      "avg_sharpe": 2.465,
      "models_used": "lgb xgb gbm rf ridge"
    }
  },
  "message": "Prediction complete for NABIL"
}
```

Failure response shape:

```json
{
  "error": "Prediction failed: Could not fetch data for 'ACE' from any source. Check the symbol and internet connection."
}
```

Implementation notes:

- `maxDuration = 660`
- Saves forecast rows through `savePredictions()`
- Immediately runs `reconcileAccuracy(symbol)`

### `GET /api/stocks`

Returns all stocks for the dashboard grid.

Fields returned per stock:

- `id`
- `symbol`
- `name`
- `sector`
- `nepseId`
- `latestPrediction`
- `accuracy`

Implementation notes:

- Calls `ensureStocksSeeded()` on demand if the DB is empty
- Uses a flat-query strategy in `stock.service.ts`
- Avoids Prisma SQLite crashes caused by nested `include` queries on large datasets

### `POST /api/stocks/sync`

Refreshes the stock master list from Merolagani and upserts records into `Stock`.

Response shape:

```json
{
  "data": {
    "fetched": 1084,
    "total": 1084,
    "message": "Sync complete — 1084 companies fetched from merolagani.com"
  }
}
```

### `GET /api/stocks/[symbol]`

Returns one stock detail payload:

- stock identity
- latest prediction snapshot
- aggregated accuracy summary
- latest forecast batch limited to 7 rows
- historical predictions limited to 30 rows
- actual prices limited to 90 rows

### `GET /api/stocks/[symbol]/accuracy`

Returns:

- `directionAccuracy`
- `avgErrorPct`
- `totalPredictions`
- `correctDirections`
- `weeklyBreakdown`

`weeklyBreakdown` is generated from `PredictionAccuracy.createdAt` and limited to the last 12 weeks.

## Prisma schema

The active schema is in `frontend/prisma/schema.prisma`.

### `Stock`

| Field | Type | Notes |
|-------|------|-------|
| `id` | `String` | UUID primary key |
| `symbol` | `String` | unique, indexed |
| `name` | `String` | company name |
| `sector` | `String?` | inferred/loaded sector |
| `nepseId` | `Int?` | Merolagani company ID, used as a proxy for newer listings |
| `createdAt` | `DateTime` | auto timestamp |

### `Prediction`

| Field | Type | Notes |
|-------|------|-------|
| `id` | `String` | UUID primary key |
| `symbol` | `String` | relation to `Stock.symbol` |
| `predictionDate` | `DateTime` | batch date |
| `targetDate` | `DateTime` | forecast target session |
| `predictedClose` | `Float` | forecast close |
| `predictedReturn` | `Float` | forecast return |
| `directionProb` | `Float` | directional probability |
| `lowerBound` | `Float` | forecast low band |
| `upperBound` | `Float` | forecast high band |
| `modelConfidence` | `Float` | direction confidence |
| `regime` | `String` | regime label |
| `sentimentScore` | `Float` | sentiment score |
| `trapScore` | `Float` | trap score |
| `action` | `String` | SQLite stores `BUY`, `SELL`, `HOLD`, `AVOID` as strings |
| `score` | `Float` | currently set from direction probability |
| `createdAt` | `DateTime` | auto timestamp |

### `ActualPrice`

| Field | Type | Notes |
|-------|------|-------|
| `id` | `String` | UUID primary key |
| `symbol` | `String` | relation to `Stock.symbol` |
| `date` | `DateTime` | unique with symbol |
| `close` | `Float` | close price |
| `volume` | `Float` | defaults to `0` |
| `createdAt` | `DateTime` | auto timestamp |

### `PredictionAccuracy`

| Field | Type | Notes |
|-------|------|-------|
| `id` | `String` | UUID primary key |
| `predictionId` | `String` | unique relation to `Prediction` |
| `actualClose` | `Float` | verified close |
| `errorPct` | `Float` | absolute percentage error |
| `directionCorrect` | `Boolean` | direction match flag |
| `createdAt` | `DateTime` | auto timestamp |

## Prediction persistence flow

`src/services/prediction.service.ts` is responsible for durable storage.

Current behavior:

1. Normalize the symbol and action
2. Filter out any forecast rows marked with `is_trading_day === false`
3. Ensure the `Stock` row exists
4. Insert each forecast row inside a Prisma transaction
5. Skip duplicates if the same `(symbol, targetDate, predictionDate)` already exists
6. Upsert `ActualPrice` using `last_close` and `last_date`
7. Reconcile missing `PredictionAccuracy` rows if actual prices are available

Normalization:

- Any action containing `BUY` becomes `BUY`
- Any action containing `SELL` or `SHORT` becomes `SELL`
- Any action containing `AVOID` becomes `AVOID`
- Everything else becomes `HOLD`

## Frontend behavior

### `/stocks`

Rendering:

- Server-rendered page
- `revalidate = 0` so counts stay fresh
- `SearchSort` handles client interactivity

Implemented features:

- Clickable stat tiles for:
  - All companies
  - With predictions
  - BUY signals
  - SELL signals
  - HOLD / AVOID
  - Newly listed
- Filter pills inside the client component:
  - All
  - Predicted
  - BUY
  - SELL
  - HOLD/AVOID
  - No Data
  - Newly Listed
- Search by symbol, company, or sector
- Sort by symbol, confidence, accuracy, or direction probability
- Per-card prediction button
- Bulk selection mode
- Bulk progress panel with success/error tracking
- Safe bulk-run warning for likely no-data stocks
- "Select established only" helper
- "Deselect visible" behavior limited to the currently filtered view

New listing heuristic:

- `nepseId >= 5000`

This is currently a practical UI heuristic, not an authoritative listing-date field.

### `/stocks/[symbol]`

Rendering:

- Server page loads stock detail and accuracy in parallel
- `StockDetailClient` handles tab switching
- `PriceChart` is dynamically imported with `ssr: false`

Implemented sections:

- Header metrics and action badge
- Inline rerun prediction button
- Price chart with actual series, forecast series, and forecast band
- 7-day forecast tab
- Historical prediction table
- Accuracy summary plus weekly breakdown

Empty-state handling:

- No forecast data: prompts user to run a prediction
- No historical data: shows explanatory text
- No verified accuracy: explains that actual target prices are required first

## Accuracy logic

Two accuracy layers are used:

- `getStocks()` returns quick per-symbol summary accuracy for the grid
- `getAccuracyMetrics(symbol)` returns richer detail-page metrics

`reconcileAccuracy(symbol)`:

- finds predictions without an accuracy row
- looks for matching `ActualPrice` rows on the same `targetDate`
- computes:
  - `errorPct = abs((actual - predicted) / predicted) * 100`
  - `directionCorrect` based on `directionProb >= 0.5`

## Known implementation details

- The current schema is SQLite-first.
- Prisma nested includes on large SQLite datasets caused engine instability, so `getStocks()` intentionally uses multiple flat queries and joins in application code.
- Some symbols, especially newly listed or thinly traded ones, can fail prediction because the Python pipeline cannot fetch enough historical data from its sources.
- Bulk prediction UI now warns about likely no-data selections before running.
- `SyncButton` reloads the page after a successful sync so the stock grid refreshes immediately.

## Deployment notes

- `POST /api/predict` depends on a real Python runtime plus the existing NEPSE project files.
- A pure static deployment is not enough.
- Vercel can host the Next.js app, but the Python execution path usually makes a VPS/custom server more practical unless prediction is moved to a separate backend service.

Checklist:

- `DATABASE_URL`
- `NEPSE_ROOT`
- `PYTHON_EXEC`
- Python dependencies installed in the selected environment

## Development commands

```bash
cd frontend
npx prisma generate
npx prisma db push
npx prisma studio
npx tsc --noEmit
```

Python bridge smoke test:

```bash
cd ..
python predict_api.py --symbol NABIL
```

Last updated: 2026-04-02
