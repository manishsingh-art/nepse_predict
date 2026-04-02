import { prisma } from "@/lib/prisma";
import { fetchAllCompanies } from "@/lib/nepse";
import type { StockSummary, StockDetail } from "@/types";

// ── normalizeAction ───────────────────────────────────────────────────────────

function normalizeAction(raw: string): "BUY" | "SELL" | "HOLD" | "AVOID" {
  const upper = (raw ?? "").toUpperCase();
  if (upper.includes("BUY")) return "BUY";
  if (upper.includes("SELL") || upper.includes("SHORT")) return "SELL";
  if (upper.includes("AVOID")) return "AVOID";
  return "HOLD";
}

// ── ensureStocksSeeded ────────────────────────────────────────────────────────
// Guarantees the Stock table is populated; safe to call on every request.

export async function ensureStocksSeeded(): Promise<void> {
  const count = await prisma.stock.count();
  if (count > 0) return;

  const companies = await fetchAllCompanies();
  // createMany with skipDuplicates is not supported on SQLite;
  // use sequential upserts so the same code works on both SQLite and PostgreSQL.
  for (const c of companies) {
    await prisma.stock.upsert({
      where: { symbol: c.symbol },
      create: { symbol: c.symbol, name: c.name, sector: c.sector, nepseId: c.nepseId || null },
      update: { name: c.name, sector: c.sector, nepseId: c.nepseId || null },
    });
  }
}

// ── getStocks ─────────────────────────────────────────────────────────────────
// Returns all stocks with their latest prediction and aggregated accuracy.
//
// IMPORTANT: Do NOT use include:{predictions:{take:1}} on findMany when the
// Stock table is large — Prisma's SQLite engine panics ("no entry found for key").
// Instead we do two flat queries and join in JS.

export async function getStocks(): Promise<StockSummary[]> {
  await ensureStocksSeeded();

  // ── 1. All stocks (flat — no includes) ────────────────────────────────────
  const stocks = await prisma.stock.findMany({ orderBy: { symbol: "asc" } });
  if (stocks.length === 0) return [];

  // ── 2. Latest prediction per symbol (distinct + order = 1 query) ──────────
  const latestPreds = await prisma.prediction.findMany({
    orderBy: { createdAt: "desc" },
    distinct: ["symbol"],
    select: {
      symbol: true,
      predictedClose: true,
      directionProb: true,
      action: true,
      modelConfidence: true,
      predictionDate: true,
      regime: true,
    },
  });
  const predBySymbol = new Map(latestPreds.map((p) => [p.symbol, p]));

  // ── 3. Accuracy: grouped by symbol from PredictionAccuracy ──────────────
  // Use raw SQL to avoid groupBy limitations in SQLite + Prisma
  const accuracyRaw = await prisma.$queryRaw<
    Array<{ symbol: string; total: number; correct: number; avgErr: number }>
  >`
    SELECT p.symbol,
           COUNT(*)                           AS total,
           SUM(CASE WHEN pa.directionCorrect = 1 THEN 1 ELSE 0 END) AS correct,
           AVG(pa.errorPct)                   AS avgErr
    FROM   PredictionAccuracy pa
    JOIN   Prediction p ON p.id = pa.predictionId
    GROUP  BY p.symbol
  `;
  const accBySymbol = new Map(accuracyRaw.map((r) => [r.symbol, r]));

  return stocks.map((s) => {
    const pred = predBySymbol.get(s.symbol) ?? null;
    const acc = accBySymbol.get(s.symbol);

    return {
      id: s.id,
      symbol: s.symbol,
      name: s.name,
      sector: s.sector,
      nepseId: s.nepseId,
      latestPrediction: pred
        ? {
            predictedClose: pred.predictedClose,
            directionProb: pred.directionProb,
            action: pred.action as "BUY" | "SELL" | "HOLD" | "AVOID",
            modelConfidence: pred.modelConfidence,
            predictionDate: pred.predictionDate.toISOString().slice(0, 10),
            regime: pred.regime,
          }
        : null,
      accuracy: acc
        ? {
            directionAccuracy:
              Number(acc.total) > 0 ? (Number(acc.correct) / Number(acc.total)) * 100 : 0,
            avgErrorPct: Number(acc.avgErr) ?? 0,
            totalPredictions: Number(acc.total),
          }
        : null,
    };
  });
}

// ── getStock ──────────────────────────────────────────────────────────────────

export async function getStock(symbol: string): Promise<StockDetail | null> {
  const upperSymbol = symbol.toUpperCase();

  const stock = await prisma.stock.findUnique({
    where: { symbol: upperSymbol },
    include: {
      predictions: {
        orderBy: [{ predictionDate: "asc" }, { targetDate: "asc" }],
        include: { accuracy: true },
      },
      actualPrices: {
        orderBy: { date: "desc" },
        take: 90,
      },
    },
  });

  if (!stock) return null;

  // Latest prediction group = predictions made most recently
  const sortedByCreated = [...stock.predictions].sort(
    (a, b) => b.createdAt.getTime() - a.createdAt.getTime(),
  );
  const latestPred = sortedByCreated[0] ?? null;

  // 7-day forecasts from latest prediction batch (same predictionDate as latest)
  const latestPredDate = latestPred?.predictionDate;
  const forecasts = latestPredDate
    ? stock.predictions
        .filter(
          (p) =>
            p.predictionDate.toISOString().slice(0, 10) ===
            latestPredDate.toISOString().slice(0, 10),
        )
        .slice(0, 7)
    : [];

  // Historical = all unique targetDate predictions (for table)
  const historicalPredictions = sortedByCreated.slice(0, 30).map((p) => ({
    id: p.id,
    predictionDate: p.predictionDate.toISOString().slice(0, 10),
    targetDate: p.targetDate.toISOString().slice(0, 10),
    predictedClose: p.predictedClose,
    action: p.action as "BUY" | "SELL" | "HOLD" | "AVOID",
    directionProb: p.directionProb,
    accuracy: p.accuracy
      ? {
          actualClose: p.accuracy.actualClose,
          errorPct: p.accuracy.errorPct,
          directionCorrect: p.accuracy.directionCorrect,
        }
      : null,
  }));

  // Accuracy aggregation
  const accuracyRecords = stock.predictions
    .map((p) => p.accuracy)
    .filter(Boolean);

  const totalPredictions = accuracyRecords.length;
  const correctDirections = accuracyRecords.filter((a) => a?.directionCorrect).length;
  const avgErrorPct =
    totalPredictions > 0
      ? accuracyRecords.reduce((s, a) => s + (a?.errorPct ?? 0), 0) / totalPredictions
      : 0;

  return {
    id: stock.id,
    symbol: stock.symbol,
    name: stock.name,
    sector: stock.sector,
    latestPrediction: latestPred
      ? {
          predictedClose: latestPred.predictedClose,
          directionProb: latestPred.directionProb,
          action: latestPred.action as "BUY" | "SELL" | "HOLD" | "AVOID",
          modelConfidence: latestPred.modelConfidence,
          predictionDate: latestPred.predictionDate.toISOString().slice(0, 10),
          regime: latestPred.regime,
        }
      : null,
    accuracy:
      totalPredictions > 0
        ? {
            directionAccuracy: (correctDirections / totalPredictions) * 100,
            avgErrorPct,
            totalPredictions,
          }
        : null,
    forecasts: forecasts.map((p) => ({
      id: p.id,
      targetDate: p.targetDate.toISOString().slice(0, 10),
      predictedClose: p.predictedClose,
      predictedReturn: p.predictedReturn,
      directionProb: p.directionProb,
      lowerBound: p.lowerBound,
      upperBound: p.upperBound,
      modelConfidence: p.modelConfidence,
      trapScore: p.trapScore,
      action: p.action as "BUY" | "SELL" | "HOLD" | "AVOID",
    })),
    historicalPredictions,
    actualPrices: stock.actualPrices.map((ap) => ({
      date: ap.date.toISOString().slice(0, 10),
      close: ap.close,
      volume: ap.volume,
    })),
  };
}

export { normalizeAction };
