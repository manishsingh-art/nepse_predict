import { NextResponse } from "next/server";
import { prisma } from "@/lib/prisma";
import { fetchAllCompanies, invalidateCache } from "@/lib/nepse";

export async function POST() {
  try {
    // Bust the in-memory cache so we fetch a fresh list
    invalidateCache("nepse_companies");

    const companies = await fetchAllCompanies();

    let added = 0;
    let skipped = 0;

    for (const c of companies) {
      const result = await prisma.stock.upsert({
        where: { symbol: c.symbol },
        create: {
          symbol: c.symbol,
          name: c.name,
          sector: c.sector,
          nepseId: c.nepseId || null,
        },
        update: {
          name: c.name,
          sector: c.sector,
          nepseId: c.nepseId || null,
        },
      });

      // Prisma upsert doesn't tell us if it was a create or update,
      // so check against the pre-existing count
      if (result.createdAt.getTime() > Date.now() - 5000) {
        added++;
      } else {
        skipped++;
      }
    }

    const total = await prisma.stock.count();

    return NextResponse.json({
      data: {
        fetched: companies.length,
        total,
        message: `Sync complete — ${companies.length} companies fetched from merolagani.com`,
      },
    });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : "Unknown error";
    console.error("[/api/stocks/sync] Error:", message);
    return NextResponse.json({ error: `Sync failed: ${message}` }, { status: 500 });
  }
}
