import { prisma } from "@/lib/prisma";
import { startOfWeek, format } from "date-fns";
import type { AccuracyMetrics } from "@/types";

// ── getAccuracyMetrics ────────────────────────────────────────────────────────

export async function getAccuracyMetrics(symbol: string): Promise<AccuracyMetrics> {
  const upperSymbol = symbol.toUpperCase();

  // Fetch all accuracy records for this stock in a single efficient query
  const records = await prisma.predictionAccuracy.findMany({
    where: {
      prediction: { symbol: upperSymbol },
    },
    select: {
      errorPct: true,
      directionCorrect: true,
      createdAt: true,
    },
    orderBy: { createdAt: "asc" },
  });

  const totalPredictions = records.length;
  const correctDirections = records.filter((r) => r.directionCorrect).length;

  const directionAccuracy =
    totalPredictions > 0 ? (correctDirections / totalPredictions) * 100 : 0;

  const avgErrorPct =
    totalPredictions > 0
      ? records.reduce((s, r) => s + r.errorPct, 0) / totalPredictions
      : 0;

  // Weekly breakdown
  const weeklyMap = new Map<
    string,
    { correct: number; total: number; errSum: number }
  >();

  for (const r of records) {
    const weekStart = format(startOfWeek(r.createdAt, { weekStartsOn: 0 }), "yyyy-MM-dd");
    const entry = weeklyMap.get(weekStart) ?? { correct: 0, total: 0, errSum: 0 };
    entry.total += 1;
    entry.errSum += r.errorPct;
    if (r.directionCorrect) entry.correct += 1;
    weeklyMap.set(weekStart, entry);
  }

  const weeklyBreakdown = Array.from(weeklyMap.entries())
    .map(([week, v]) => ({
      week,
      dirAccuracy: (v.correct / v.total) * 100,
      avgError: v.errSum / v.total,
      count: v.total,
    }))
    .slice(-12); // Last 12 weeks

  return {
    directionAccuracy,
    avgErrorPct,
    totalPredictions,
    correctDirections,
    weeklyBreakdown,
  };
}
