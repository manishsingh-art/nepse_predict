import { NextRequest, NextResponse } from "next/server";
import { getAccuracyMetrics } from "@/services/accuracy.service";

export async function GET(
  _req: NextRequest,
  { params }: { params: Promise<{ symbol: string }> },
) {
  const { symbol } = await params;

  if (!symbol || !/^[A-Za-z0-9]{1,12}$/.test(symbol)) {
    return NextResponse.json({ error: "Invalid symbol" }, { status: 400 });
  }

  try {
    const metrics = await getAccuracyMetrics(symbol);
    return NextResponse.json({ data: metrics });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : "Unknown error";
    console.error(`[/api/stocks/${symbol}/accuracy] Error:`, message);
    return NextResponse.json({ error: `Failed to fetch accuracy: ${message}` }, { status: 500 });
  }
}
