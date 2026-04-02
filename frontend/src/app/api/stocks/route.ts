import { NextResponse } from "next/server";
import { getStocks } from "@/services/stock.service";

export const revalidate = 60; // ISR: revalidate every 60 seconds

export async function GET() {
  try {
    const stocks = await getStocks();
    return NextResponse.json({ data: stocks });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : "Unknown error";
    console.error("[/api/stocks] Error:", message);
    return NextResponse.json({ error: `Failed to fetch stocks: ${message}` }, { status: 500 });
  }
}
