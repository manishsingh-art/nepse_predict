import { NextRequest, NextResponse } from "next/server";
import { runPrediction } from "@/lib/python-runner";
import { savePredictions, reconcileAccuracy } from "@/services/prediction.service";

export const maxDuration = 660; // 11 minutes — pipeline can be slow

export async function POST(req: NextRequest) {
  let symbol: string;

  try {
    const body = await req.json();
    symbol = (body?.symbol ?? "").toString().toUpperCase().trim();
  } catch {
    return NextResponse.json({ error: "Invalid JSON body" }, { status: 400 });
  }

  if (!symbol || !/^[A-Z0-9]{1,12}$/.test(symbol)) {
    return NextResponse.json(
      { error: "Missing or invalid symbol. Must be 1-12 uppercase alphanumeric characters." },
      { status: 400 },
    );
  }

  try {
    const report = await runPrediction(symbol);

    await savePredictions(report);
    await reconcileAccuracy(symbol);

    return NextResponse.json({
      data: {
        symbol: report.symbol,
        generatedAt: report.generated_at,
        forecastCount: report.forecast?.length ?? 0,
        action: report.strategy?.action,
        confidence: report.strategy?.suggested_size_weight,
        regime: report.regime?.regime,
        lastClose: report.last_close,
        mlMetrics: report.ml_metrics,
      },
      message: `Prediction complete for ${symbol}`,
    });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : "Unknown error";
    console.error(`[/api/predict] Error for ${symbol}:`, message);
    return NextResponse.json(
      { error: `Prediction failed: ${message}` },
      { status: 500 },
    );
  }
}
