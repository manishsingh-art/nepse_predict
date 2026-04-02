import { Suspense } from "react";
import Link from "next/link";
import { TrendingUp, Database, Activity, Building2, PauseCircle, Sparkles } from "lucide-react";
import { getStocks } from "@/services/stock.service";
import { prisma } from "@/lib/prisma";
import SearchSort, { StocksGridSkeleton } from "@/components/SearchSort";
import SyncButton from "@/components/SyncButton";
import type { Metadata } from "next";

export const revalidate = 0; // always fresh so filter counts stay accurate

export const metadata: Metadata = {
  title: "All Stocks — NEPSE Predictor",
};

// ── Stat tiles (server-rendered, URL-based filter) ────────────────────────────

function StatTile({
  icon,
  label,
  value,
  valueClass = "text-slate-100",
  href,
  active = false,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  valueClass?: string;
  href: string;
  active?: boolean;
}) {
  return (
    <Link href={href}>
      <div
        className={`card px-4 py-3 flex items-center gap-3 cursor-pointer transition-all select-none
          hover:border-blue-500/40 hover:bg-[#0d1829]
          ${active ? "border-blue-500/60 bg-[#0b1525] shadow-blue-500/10 shadow-lg" : ""}`}
      >
        <div className="w-8 h-8 rounded-lg bg-[#1e2a45] flex items-center justify-center shrink-0">
          {icon}
        </div>
        <div className="min-w-0">
          <p className="text-[10px] text-slate-600 uppercase tracking-wider whitespace-nowrap">{label}</p>
          <p className={`text-lg font-bold font-mono tabular-nums ${valueClass}`}>{value}</p>
        </div>
        {active && <div className="ml-auto w-1 h-8 rounded-full bg-blue-500 shrink-0" />}
      </div>
    </Link>
  );
}

// ── StocksContent (async — runs inside Suspense) ──────────────────────────────

// Threshold matching SearchSort.tsx
const NEW_LISTING_THRESHOLD = 5000;

async function StocksContent({ filter }: { filter: string }) {
  const [stocks, totalInDb] = await Promise.all([getStocks(), prisma.stock.count()]);

  const withPred    = stocks.filter((s) => s.latestPrediction !== null).length;
  const buyCount    = stocks.filter((s) => s.latestPrediction?.action === "BUY").length;
  const sellCount   = stocks.filter((s) => s.latestPrediction?.action === "SELL").length;
  const holdAvoid   = stocks.filter(
    (s) => s.latestPrediction?.action === "HOLD" || s.latestPrediction?.action === "AVOID",
  ).length;
  const newlyListed = stocks.filter((s) => (s.nepseId ?? 0) >= NEW_LISTING_THRESHOLD).length;

  const f = filter || "all";

  return (
    <div className="space-y-5">
      {/* Clickable filter tiles */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
        <StatTile
          icon={<Database size={16} className="text-blue-400" />}
          label="All Companies"
          value={totalInDb.toLocaleString()}
          href="/stocks?filter=all"
          active={f === "all"}
        />
        <StatTile
          icon={<Activity size={16} className="text-slate-300" />}
          label="With Predictions"
          value={withPred.toString()}
          href="/stocks?filter=predictions"
          active={f === "predictions"}
        />
        <StatTile
          icon={<TrendingUp size={16} className="text-emerald-400" />}
          label="BUY Signals"
          value={buyCount.toString()}
          valueClass="text-emerald-400"
          href="/stocks?filter=buy"
          active={f === "buy"}
        />
        <StatTile
          icon={<TrendingUp size={16} className="text-red-400 rotate-180" />}
          label="SELL Signals"
          value={sellCount.toString()}
          valueClass="text-red-400"
          href="/stocks?filter=sell"
          active={f === "sell"}
        />
        <StatTile
          icon={<PauseCircle size={16} className="text-amber-400" />}
          label="HOLD / AVOID"
          value={holdAvoid.toString()}
          valueClass="text-amber-400"
          href="/stocks?filter=hold"
          active={f === "hold"}
        />
        <StatTile
          icon={<Sparkles size={16} className="text-cyan-400" />}
          label="Newly Listed"
          value={newlyListed.toString()}
          valueClass="text-cyan-400"
          href="/stocks?filter=new"
          active={f === "new"}
        />
      </div>

      {/* Search + sort + bulk-select grid */}
      <SearchSort stocks={stocks} initialFilter={f} />
    </div>
  );
}

// ── Page ──────────────────────────────────────────────────────────────────────

export default async function StocksPage({
  searchParams,
}: {
  searchParams: Promise<{ filter?: string }>;
}) {
  const { filter = "all" } = await searchParams;

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-slate-100 flex items-center gap-2">
            <Building2 size={22} className="text-blue-400" />
            NEPSE Stocks
          </h1>
          <p className="mt-1 text-sm text-slate-500">
            Click a tile to filter · select stocks to bulk-run predictions · sync from{" "}
            <a
              href="https://www.merolagani.com"
              target="_blank"
              rel="noopener noreferrer"
              className="text-blue-400 underline underline-offset-2 hover:text-blue-300"
            >
              merolagani.com
            </a>
          </p>
        </div>
        <SyncButton />
      </div>

      <Suspense fallback={<StocksGridSkeleton />}>
        <StocksContent filter={filter} />
      </Suspense>
    </div>
  );
}
