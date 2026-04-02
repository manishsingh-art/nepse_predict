import { prisma } from "@/lib/prisma";
import { normalizeAction } from "@/services/stock.service";
import type { PythonReport, PythonForecastPoint } from "@/types";

// ── savePredictions ───────────────────────────────────────────────────────────
// Persists all forecast rows from a Python report inside a transaction.
// Idempotent: skips rows that already exist for (symbol, targetDate, predictionDate).

export async function savePredictions(report: PythonReport): Promise<void> {
  const symbol = report.symbol.toUpperCase();
  const predictionDate = new Date(report.generated_at.slice(0, 10));
  const action = normalizeAction(report.strategy?.action ?? "HOLD");
  const sentimentScore = report.sentiment?.score ?? 0;
  const regime = report.regime?.regime ?? "UNKNOWN";
  const modelConfidence = report.strategy?.suggested_size_weight ?? 0;
  const trapScore = report.regime?.trap_score ?? 0;

  const forecasts: PythonForecastPoint[] = (report.forecast ?? []).filter(
    (f) => f.is_trading_day !== false,
  );

  if (forecasts.length === 0) {
    console.warn(`[prediction.service] No trading-day forecasts for ${symbol}`);
    return;
  }

  // Ensure stock record exists
  await prisma.stock.upsert({
    where: { symbol },
    create: { symbol, name: symbol },
    update: {},
  });

  await prisma.$transaction(async (tx) => {
    for (const f of forecasts) {
      const targetDate = new Date(f.date);

      // Check for existing record to avoid duplicates
      const existing = await tx.prediction.findFirst({
        where: {
          symbol,
          targetDate,
          predictionDate,
        },
        select: { id: true },
      });

      if (existing) continue;

      await tx.prediction.create({
        data: {
          symbol,
          predictionDate,
          targetDate,
          predictedClose: safeFloat(f.predicted_close),
          predictedReturn: safeFloat(f.predicted_return),
          directionProb: safeFloat(f.direction_prob),
          lowerBound: safeFloat(f.low_band),
          upperBound: safeFloat(f.high_band),
          modelConfidence: safeFloat(f.direction_confidence ?? modelConfidence),
          regime,
          sentimentScore: safeFloat(sentimentScore),
          trapScore: safeFloat(f.trap_score ?? trapScore),
          action,
          score: safeFloat(f.direction_prob),
        },
      });
    }
  });

  // Also persist the last known close as an actual price for today
  if (report.last_close && report.last_date) {
    await prisma.actualPrice.upsert({
      where: {
        symbol_date: {
          symbol,
          date: new Date(report.last_date),
        },
      },
      create: {
        symbol,
        date: new Date(report.last_date),
        close: safeFloat(report.last_close),
        volume: 0,
      },
      update: {
        close: safeFloat(report.last_close),
      },
    });
  }
}

// ── reconcileAccuracy ─────────────────────────────────────────────────────────
// Called after saving predictions: checks if any pending predictions now have
// a matching actual price and creates PredictionAccuracy rows for them.

export async function reconcileAccuracy(symbol: string): Promise<void> {
  const upperSymbol = symbol.toUpperCase();

  // Find predictions for this symbol that don't yet have accuracy records
  const pending = await prisma.prediction.findMany({
    where: {
      symbol: upperSymbol,
      accuracy: null,
    },
    select: {
      id: true,
      targetDate: true,
      predictedClose: true,
      directionProb: true,
    },
  });

  if (pending.length === 0) return;

  // Find matching actual prices
  const targetDates = pending.map((p) => p.targetDate);
  const actuals = await prisma.actualPrice.findMany({
    where: {
      symbol: upperSymbol,
      date: { in: targetDates },
    },
    select: { date: true, close: true },
  });

  const actualMap = new Map(actuals.map((a) => [a.date.toISOString().slice(0, 10), a.close]));

  for (const pred of pending) {
    const key = pred.targetDate.toISOString().slice(0, 10);
    const actualClose = actualMap.get(key);
    if (actualClose === undefined) continue;

    const errorPct = Math.abs((actualClose - pred.predictedClose) / (pred.predictedClose + 1e-9)) * 100;
    const directionCorrect =
      (pred.directionProb >= 0.5 && actualClose >= pred.predictedClose) ||
      (pred.directionProb < 0.5 && actualClose < pred.predictedClose);

    await prisma.predictionAccuracy.upsert({
      where: { predictionId: pred.id },
      create: {
        predictionId: pred.id,
        actualClose,
        errorPct,
        directionCorrect,
      },
      update: { actualClose, errorPct, directionCorrect },
    });
  }
}

// ── helpers ───────────────────────────────────────────────────────────────────

function safeFloat(val: unknown, fallback = 0): number {
  const n = Number(val);
  if (!isFinite(n) || isNaN(n)) return fallback;
  return n;
}
