import { NextRequest, NextResponse } from "next/server";
import { getStock } from "@/services/stock.service";

export async function GET(
  _req: NextRequest,
  { params }: { params: Promise<{ symbol: string }> },
) {
  const { symbol } = await params;

  if (!symbol || !/^[A-Za-z0-9]{1,12}$/.test(symbol)) {
    return NextResponse.json({ error: "Invalid symbol" }, { status: 400 });
  }

  try {
    const stock = await getStock(symbol);

    if (!stock) {
      return NextResponse.json(
        { error: `Stock ${symbol.toUpperCase()} not found` },
        { status: 404 },
      );
    }

    return NextResponse.json({ data: stock });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : "Unknown error";
    console.error(`[/api/stocks/${symbol}] Error:`, message);
    return NextResponse.json({ error: `Failed to fetch stock: ${message}` }, { status: 500 });
  }
}
