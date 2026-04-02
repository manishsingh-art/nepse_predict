"use client";

import { useState, useMemo, useEffect, useRef, useCallback } from "react";
import {
  Search,
  ArrowUpDown,
  CheckSquare,
  Square,
  Play,
  X,
  CheckCircle2,
  XCircle,
  Loader2,
  Clock,
  ChevronDown,
  ChevronUp,
  AlertTriangle,
  ShieldCheck,
} from "lucide-react";
import StockCard from "./StockCard";
import SkeletonCard from "./SkeletonCard";
import type { StockSummary } from "@/types";

// ── Types ─────────────────────────────────────────────────────────────────────

type SortKey = "symbol" | "confidence" | "accuracy" | "dirProb";
// Stocks with merolagani nepseId >= this threshold are considered newly listed
const NEW_LISTING_THRESHOLD = 5000;

type FilterKey = "all" | "predictions" | "buy" | "sell" | "hold" | "no-data" | "new";

type BulkStatus = "idle" | "running" | "done";
type SymbolStatus = "pending" | "running" | "success" | "error";

interface BulkResult {
  symbol: string;
  status: SymbolStatus;
  error?: string;
}

interface Props {
  stocks: StockSummary[];
  initialFilter?: string;
}

// ── Component ─────────────────────────────────────────────────────────────────

export default function SearchSort({ stocks, initialFilter = "all" }: Props) {
  // ── Search / sort ──────────────────────────────────────────────────────────
  const [query, setQuery] = useState("");
  const [sortKey, setSortKey] = useState<SortKey>("symbol");
  const [sortAsc, setSortAsc] = useState(true);

  // ── Filter ─────────────────────────────────────────────────────────────────
  const [activeFilter, setActiveFilter] = useState<FilterKey>(
    (initialFilter as FilterKey) ?? "all",
  );

  // Sync when URL param changes (tile click causes full re-render)
  useEffect(() => {
    setActiveFilter((initialFilter as FilterKey) ?? "all");
  }, [initialFilter]);

  // ── Bulk select ────────────────────────────────────────────────────────────
  const [selectMode, setSelectMode] = useState(false);
  const [selected, setSelected] = useState<Set<string>>(new Set());

  // ── Bulk run ───────────────────────────────────────────────────────────────
  const [bulkStatus, setBulkStatus] = useState<BulkStatus>("idle");
  const [bulkResults, setBulkResults] = useState<BulkResult[]>([]);
  const [bulkExpanded, setBulkExpanded] = useState(false);
  const [showConfirm, setShowConfirm] = useState(false);
  const cancelRef = useRef(false);

  // Fast symbol → stock lookup
  const stockMap = useMemo(
    () => new Map(stocks.map((s) => [s.symbol, s])),
    [stocks],
  );

  // ── Filter counts ──────────────────────────────────────────────────────────
  const counts = useMemo(
    () => ({
      all: stocks.length,
      predictions: stocks.filter((s) => s.latestPrediction !== null).length,
      buy: stocks.filter((s) => s.latestPrediction?.action === "BUY").length,
      sell: stocks.filter((s) => s.latestPrediction?.action === "SELL").length,
      hold: stocks.filter(
        (s) =>
          s.latestPrediction?.action === "HOLD" || s.latestPrediction?.action === "AVOID",
      ).length,
      "no-data": stocks.filter((s) => s.latestPrediction === null).length,
      new: stocks.filter((s) => (s.nepseId ?? 0) >= NEW_LISTING_THRESHOLD).length,
    }),
    [stocks],
  );

  // ── Filtered + sorted list ─────────────────────────────────────────────────
  const filtered = useMemo(() => {
    let list = stocks;

    // Apply tile filter
    switch (activeFilter) {
      case "predictions":
        list = list.filter((s) => s.latestPrediction !== null);
        break;
      case "buy":
        list = list.filter((s) => s.latestPrediction?.action === "BUY");
        break;
      case "sell":
        list = list.filter((s) => s.latestPrediction?.action === "SELL");
        break;
      case "hold":
        list = list.filter(
          (s) =>
            s.latestPrediction?.action === "HOLD" || s.latestPrediction?.action === "AVOID",
        );
        break;
      case "no-data":
        list = list.filter((s) => s.latestPrediction === null);
        break;
      case "new":
        list = list.filter((s) => (s.nepseId ?? 0) >= NEW_LISTING_THRESHOLD);
        break;
    }

    // Apply text search
    const q = query.toLowerCase().trim();
    if (q) {
      list = list.filter(
        (s) =>
          s.symbol.toLowerCase().includes(q) ||
          s.name.toLowerCase().includes(q) ||
          (s.sector ?? "").toLowerCase().includes(q),
      );
    }

    // Sort
    return [...list].sort((a, b) => {
      let va: number | string = 0;
      let vb: number | string = 0;
      switch (sortKey) {
        case "symbol":
          va = a.symbol;
          vb = b.symbol;
          break;
        case "confidence":
          va = a.latestPrediction?.modelConfidence ?? -1;
          vb = b.latestPrediction?.modelConfidence ?? -1;
          break;
        case "accuracy":
          va = a.accuracy?.directionAccuracy ?? -1;
          vb = b.accuracy?.directionAccuracy ?? -1;
          break;
        case "dirProb":
          va = a.latestPrediction?.directionProb ?? -1;
          vb = b.latestPrediction?.directionProb ?? -1;
          break;
      }
      if (typeof va === "string")
        return sortAsc ? va.localeCompare(vb as string) : (vb as string).localeCompare(va);
      return sortAsc ? (va as number) - (vb as number) : (vb as number) - (va as number);
    });
  }, [stocks, query, sortKey, sortAsc, activeFilter]);

  // ── Handlers ───────────────────────────────────────────────────────────────

  function toggleSort(key: SortKey) {
    if (sortKey === key) setSortAsc((p) => !p);
    else { setSortKey(key); setSortAsc(key === "symbol"); }
  }

  function toggleSelectMode() {
    setSelectMode((p) => !p);
    setSelected(new Set());
  }

  function toggleSymbol(symbol: string) {
    setSelected((prev) => {
      const next = new Set(prev);
      next.has(symbol) ? next.delete(symbol) : next.add(symbol);
      return next;
    });
  }

  function selectAll() {
    // Add all currently-visible stocks to the selection (keeps other tabs' picks)
    setSelected((prev) => {
      const next = new Set(prev);
      filtered.forEach((s) => next.add(s.symbol));
      return next;
    });
  }

  function deselectAll() {
    // Remove only the currently-visible stocks from the selection
    const visibleSet = new Set(filtered.map((s) => s.symbol));
    setSelected((prev) => {
      const next = new Set(prev);
      visibleSet.forEach((sym) => next.delete(sym));
      return next;
    });
  }

  // "Established" = has latestPrediction OR nepseId < threshold (proven data source)
  function isEstablished(symbol: string): boolean {
    const s = stockMap.get(symbol);
    if (!s) return false;
    return s.latestPrediction !== null || (s.nepseId ?? 0) < NEW_LISTING_THRESHOLD;
  }

  // Count how many selected stocks are likely to fail
  const likelyFail = useMemo(
    () => [...selected].filter((sym) => !isEstablished(sym)).length,
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [selected, stockMap],
  );

  // Remove risky (newly-listed, no prior prediction) from selection
  function selectEstablishedOnly() {
    setSelected((prev) => {
      const next = new Set<string>();
      prev.forEach((sym) => { if (isEstablished(sym)) next.add(sym); });
      return next;
    });
  }

  // ── Bulk prediction runner ─────────────────────────────────────────────────

  const runBulk = useCallback(async (symbolsOverride?: string[]) => {
    const symbols = symbolsOverride ?? [...selected];
    if (symbols.length === 0) return;

    cancelRef.current = false;
    setBulkStatus("running");
    setBulkExpanded(true);

    const initial: BulkResult[] = symbols.map((s) => ({ symbol: s, status: "pending" }));
    setBulkResults(initial);

    const results = [...initial];

    for (let i = 0; i < symbols.length; i++) {
      if (cancelRef.current) break;

      const sym = symbols[i];
      results[i] = { ...results[i], status: "running" };
      setBulkResults([...results]);

      try {
        const res = await fetch("/api/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ symbol: sym }),
        });
        const json = await res.json();
        if (!res.ok) throw new Error(json.error ?? "Unknown error");
        results[i] = { symbol: sym, status: "success" };
      } catch (err: unknown) {
        results[i] = {
          symbol: sym,
          status: "error",
          error: err instanceof Error ? err.message : "Failed",
        };
      }

      setBulkResults([...results]);
    }

    setBulkStatus("done");
  }, [selected]);

  function cancelBulk() {
    cancelRef.current = true;
    setBulkStatus("done");
  }

  function closeBulkPanel() {
    setBulkStatus("idle");
    setBulkResults([]);
    setSelected(new Set());
    setSelectMode(false);
    setBulkExpanded(false);
    setShowConfirm(false);
  }

  function handleRunClick() {
    if (likelyFail > 0) {
      setShowConfirm(true); // show pre-run confirmation
    } else {
      void runBulk();
    }
  }

  // ── Derived bulk stats ────────────────────────────────────────────────────

  const bulkDone = bulkResults.filter((r) => r.status === "success" || r.status === "error").length;
  const bulkSuccess = bulkResults.filter((r) => r.status === "success").length;
  const bulkError = bulkResults.filter((r) => r.status === "error").length;
  const bulkRunning = bulkResults.find((r) => r.status === "running")?.symbol ?? "";
  const estMin = Math.ceil((selected.size * 45) / 60); // ~45 s per stock

  const sortOptions: { key: SortKey; label: string }[] = [
    { key: "symbol", label: "Symbol" },
    { key: "confidence", label: "Confidence" },
    { key: "accuracy", label: "Accuracy" },
    { key: "dirProb", label: "Dir Prob" },
  ];

  const filterPills: { key: FilterKey; label: string; count: number; color: string }[] = [
    { key: "all", label: "All", count: counts.all, color: "blue" },
    { key: "predictions", label: "Predicted", count: counts.predictions, color: "slate" },
    { key: "buy", label: "BUY", count: counts.buy, color: "emerald" },
    { key: "sell", label: "SELL", count: counts.sell, color: "red" },
    { key: "hold", label: "HOLD/AVOID", count: counts.hold, color: "amber" },
    { key: "no-data", label: "No Data", count: counts["no-data"], color: "slate" },
    { key: "new", label: "Newly Listed", count: counts.new, color: "cyan" },
  ];

  return (
    <div className="space-y-4">
      {/* ── Controls bar ──────────────────────────────────────────────────── */}
      <div className="flex flex-col gap-3">
        {/* Row 1: search + select toggle */}
        <div className="flex gap-3">
          <div className="relative flex-1">
            <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
            <input
              type="text"
              placeholder="Search symbol, company, sector…"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              className="w-full pl-9 pr-4 py-2.5 text-sm bg-[#111827] border border-[#1e2a45] rounded-xl
                text-slate-200 placeholder:text-slate-600 focus:outline-none focus:border-blue-500/60 transition-colors"
            />
          </div>

          {/* Select mode toggle */}
          <button
            onClick={toggleSelectMode}
            className={`flex items-center gap-2 text-xs font-medium px-4 py-2.5 rounded-xl border transition-all shrink-0 ${
              selectMode
                ? "bg-violet-600/20 border-violet-500/60 text-violet-300"
                : "bg-[#111827] border-[#1e2a45] text-slate-400 hover:border-violet-500/40 hover:text-violet-400"
            }`}
          >
            {selectMode ? <CheckSquare size={14} /> : <Square size={14} />}
            {selectMode ? `Select (${selected.size})` : "Select"}
          </button>
        </div>

        {/* Row 2: filter pills + sort pills */}
        <div className="flex flex-col sm:flex-row gap-2">
          {/* Filter pills */}
          <div className="flex items-center gap-1.5 flex-wrap">
            {filterPills.map((fp) => {
              const active = activeFilter === fp.key;
              const colorMap: Record<string, string> = {
                blue:    "bg-blue-600/20 border-blue-500/60 text-blue-400",
                emerald: "bg-emerald-600/20 border-emerald-500/60 text-emerald-400",
                red:     "bg-red-600/20 border-red-500/60 text-red-400",
                amber:   "bg-amber-600/20 border-amber-500/60 text-amber-400",
                slate:   "bg-slate-600/20 border-slate-500/40 text-slate-300",
                cyan:    "bg-cyan-600/20 border-cyan-500/60 text-cyan-400",
              };
              return (
                <button
                  key={fp.key}
                  onClick={() => setActiveFilter(fp.key)}
                  className={`text-xs px-2.5 py-1 rounded-lg border transition-all ${
                    active
                      ? colorMap[fp.color]
                      : "bg-[#111827] border-[#1e2a45] text-slate-600 hover:border-[#2a3a5c] hover:text-slate-400"
                  }`}
                >
                  {fp.label}
                  <span className={`ml-1.5 font-mono ${active ? "opacity-100" : "opacity-60"}`}>
                    {fp.count}
                  </span>
                </button>
              );
            })}
          </div>

          {/* Sort pills */}
          <div className="flex items-center gap-1.5 flex-wrap sm:ml-auto">
            <span className="text-xs text-slate-600 flex items-center gap-1">
              <ArrowUpDown size={11} /> Sort:
            </span>
            {sortOptions.map((opt) => (
              <button
                key={opt.key}
                onClick={() => toggleSort(opt.key)}
                className={`text-xs px-2.5 py-1 rounded-lg border transition-all ${
                  sortKey === opt.key
                    ? "bg-blue-600/20 border-blue-500/60 text-blue-400"
                    : "bg-[#111827] border-[#1e2a45] text-slate-500 hover:border-[#2a3a5c] hover:text-slate-400"
                }`}
              >
                {opt.label}
                {sortKey === opt.key && <span className="ml-1">{sortAsc ? "↑" : "↓"}</span>}
              </button>
            ))}
          </div>
        </div>

        {/* Row 3: select-mode toolbar (only when selectMode active) */}
        {selectMode && (
          <div className="flex flex-col gap-2">
            {/* ── Main selection toolbar ── */}
            <div className="flex items-center gap-2 flex-wrap px-3 py-2 rounded-xl bg-violet-600/10 border border-violet-500/30">
              {/* Counts */}
              {(() => {
                const visibleSelected = filtered.filter((s) => selected.has(s.symbol)).length;
                return (
                  <span className="text-xs text-violet-300 font-medium">
                    {visibleSelected} / {filtered.length} visible selected
                    {selected.size > visibleSelected && (
                      <span className="text-slate-500 ml-1">
                        ({selected.size} total across all tabs)
                      </span>
                    )}
                  </span>
                );
              })()}

              <button onClick={selectAll} className="text-xs text-violet-400 hover:text-violet-300 underline underline-offset-2">
                Select visible ({filtered.length})
              </button>
              <button onClick={deselectAll} className="text-xs text-slate-500 hover:text-slate-400 underline underline-offset-2">
                Deselect visible ({filtered.filter((s) => selected.has(s.symbol)).length})
              </button>

              {/* Established-only helper */}
              {likelyFail > 0 && selected.size > 0 && (
                <button
                  onClick={selectEstablishedOnly}
                  className="flex items-center gap-1 text-xs text-emerald-400 hover:text-emerald-300 underline underline-offset-2"
                >
                  <ShieldCheck size={11} />
                  Select established only ({selected.size - likelyFail})
                </button>
              )}

              {selected.size > 0 && (
                <>
                  <span className="text-slate-700">·</span>
                  <span className="text-xs text-slate-500 flex items-center gap-1">
                    <Clock size={11} /> ~{estMin} min
                  </span>

                  {/* Warning badge */}
                  {likelyFail > 0 && (
                    <span className="flex items-center gap-1 text-xs text-amber-400 bg-amber-900/20 border border-amber-700/40 px-2 py-0.5 rounded-full">
                      <AlertTriangle size={11} />
                      {likelyFail} likely no data
                    </span>
                  )}

                  <button
                    onClick={handleRunClick}
                    disabled={bulkStatus === "running"}
                    className="ml-auto flex items-center gap-2 text-xs font-semibold px-4 py-1.5 rounded-lg
                      bg-violet-600 hover:bg-violet-500 text-white transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    <Play size={12} />
                    Run {selected.size} Prediction{selected.size !== 1 ? "s" : ""}
                  </button>
                </>
              )}
            </div>

            {/* ── Pre-run confirmation (only when likelyFail > 0) ── */}
            {showConfirm && !bulkStatus.startsWith("run") && bulkStatus === "idle" && (
              <div className="px-4 py-3 rounded-xl bg-amber-900/20 border border-amber-600/40 space-y-2">
                <div className="flex items-start gap-2">
                  <AlertTriangle size={15} className="text-amber-400 shrink-0 mt-0.5" />
                  <div>
                    <p className="text-sm font-medium text-amber-300">
                      {likelyFail} of {selected.size} selected stocks may have no historical data
                    </p>
                    <p className="text-xs text-slate-500 mt-0.5">
                      Newly listed or thinly-traded stocks often fail with "Could not fetch data from any source".
                      Run only the <span className="text-emerald-400">{selected.size - likelyFail} established stocks</span>, or run all and skip failures.
                    </p>
                  </div>
                </div>
                <div className="flex gap-2 flex-wrap">
                  <button
                    onClick={() => {
                      setShowConfirm(false);
                      // Keep only established symbols
                      const safeSymbols = [...selected].filter((sym) => isEstablished(sym));
                      void runBulk(safeSymbols);
                    }}
                    className="flex items-center gap-1.5 text-xs font-semibold px-3 py-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white transition-colors"
                  >
                    <ShieldCheck size={12} />
                    Run {selected.size - likelyFail} established only
                  </button>
                  <button
                    onClick={() => { setShowConfirm(false); void runBulk(); }}
                    className="flex items-center gap-1.5 text-xs font-semibold px-3 py-1.5 rounded-lg bg-violet-600 hover:bg-violet-500 text-white transition-colors"
                  >
                    <Play size={12} />
                    Run all {selected.size} anyway
                  </button>
                  <button
                    onClick={() => setShowConfirm(false)}
                    className="text-xs text-slate-500 hover:text-slate-400 px-2 py-1.5 transition-colors"
                  >
                    Cancel
                  </button>
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* ── Results info ──────────────────────────────────────────────────── */}
      <p className="text-xs text-slate-600">
        {filtered.length} of {stocks.length} companies
        {query && ` matching "${query}"`}
        {activeFilter !== "all" && !query && ` — filter: ${activeFilter}`}
      </p>

      {/* ── Grid ──────────────────────────────────────────────────────────── */}
      {filtered.length === 0 ? (
        <div className="py-20 text-center">
          <p className="text-slate-500">No stocks found.</p>
          <button
            onClick={() => { setQuery(""); setActiveFilter("all"); }}
            className="mt-3 text-sm text-blue-400 hover:text-blue-300"
          >
            Clear filters
          </button>
        </div>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
          {filtered.map((s) => (
            <StockCard
              key={s.id}
              stock={s}
              selectMode={selectMode}
              selected={selected.has(s.symbol)}
              onToggleSelect={toggleSymbol}
            />
          ))}
        </div>
      )}

      {/* ── Bulk run progress panel (sticky bottom) ───────────────────────── */}
      {(bulkStatus === "running" || bulkStatus === "done") && (
        <div className="fixed bottom-6 left-1/2 -translate-x-1/2 z-50 w-full max-w-2xl px-4">
          <div className="bg-[#0b1525] border border-violet-500/40 rounded-2xl shadow-2xl shadow-black/60 overflow-hidden">
            {/* Header bar */}
            <div className="flex items-center gap-3 px-4 py-3 border-b border-[#1e2a45]">
              {bulkStatus === "running" ? (
                <Loader2 size={15} className="text-violet-400 animate-spin shrink-0" />
              ) : (
                <CheckCircle2 size={15} className="text-emerald-400 shrink-0" />
              )}

              <div className="flex-1 min-w-0">
                {bulkStatus === "running" ? (
                  <p className="text-sm font-medium text-slate-200">
                    Running{" "}
                    <span className="font-mono text-violet-300">{bulkRunning}</span>
                    {"  "}
                    <span className="text-slate-500 text-xs">
                      ({bulkDone}/{bulkResults.length})
                    </span>
                  </p>
                ) : (
                  <p className="text-sm font-medium text-slate-200">
                    Completed —{" "}
                    <span className="text-emerald-400">{bulkSuccess} succeeded</span>
                    {bulkError > 0 && (
                      <>, <span className="text-red-400">{bulkError} failed</span></>
                    )}
                  </p>
                )}

                {/* Progress bar */}
                <div className="mt-1.5 h-1 bg-[#1e2a45] rounded-full overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all duration-500 ${
                      bulkStatus === "done" ? "bg-emerald-500" : "bg-violet-500"
                    }`}
                    style={{
                      width: `${bulkResults.length > 0 ? (bulkDone / bulkResults.length) * 100 : 0}%`,
                    }}
                  />
                </div>
              </div>

              {/* Expand/collapse toggle */}
              <button
                onClick={() => setBulkExpanded((p) => !p)}
                className="text-slate-500 hover:text-slate-300 transition-colors shrink-0"
                title={bulkExpanded ? "Collapse" : "Expand"}
              >
                {bulkExpanded ? <ChevronDown size={16} /> : <ChevronUp size={16} />}
              </button>

              {/* Cancel / close */}
              {bulkStatus === "running" ? (
                <button
                  onClick={cancelBulk}
                  className="flex items-center gap-1.5 text-xs text-red-400 hover:text-red-300 border border-red-500/40 rounded-lg px-2.5 py-1 transition-colors shrink-0"
                >
                  <X size={12} /> Cancel
                </button>
              ) : (
                <button
                  onClick={closeBulkPanel}
                  className="flex items-center gap-1.5 text-xs text-slate-400 hover:text-slate-200 border border-[#2a3a5c] rounded-lg px-2.5 py-1 transition-colors shrink-0"
                >
                  <X size={12} /> Close
                </button>
              )}
            </div>

            {/* Expandable results list */}
            {bulkExpanded && (
              <div className="max-h-52 overflow-y-auto px-4 py-2 space-y-1.5 scrollbar-thin scrollbar-track-transparent scrollbar-thumb-[#2a3a5c]">
                {bulkResults.map((r) => (
                  <div key={r.symbol} className="flex items-center gap-2.5 py-0.5">
                    {r.status === "success" && (
                      <CheckCircle2 size={13} className="text-emerald-400 shrink-0" />
                    )}
                    {r.status === "error" && (
                      <XCircle size={13} className="text-red-400 shrink-0" />
                    )}
                    {r.status === "running" && (
                      <Loader2 size={13} className="text-violet-400 animate-spin shrink-0" />
                    )}
                    {r.status === "pending" && (
                      <div className="w-3.5 h-3.5 rounded-full border border-slate-700 shrink-0" />
                    )}
                    <span
                      className={`text-xs font-mono ${
                        r.status === "success"
                          ? "text-emerald-300"
                          : r.status === "error"
                            ? "text-red-300"
                            : r.status === "running"
                              ? "text-violet-300"
                              : "text-slate-600"
                      }`}
                    >
                      {r.symbol}
                    </span>
                    {r.status === "error" && r.error && (
                      <span
                        className="text-[10px] text-red-500/70 truncate max-w-xs"
                        title={r.error}
                      >
                        {r.error.slice(0, 90)}
                      </span>
                    )}
                    {r.status === "running" && (
                      <span className="text-[10px] text-violet-500 animate-pulse">
                        running ML pipeline…
                      </span>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

// ── Skeleton ──────────────────────────────────────────────────────────────────

export function StocksGridSkeleton() {
  return (
    <div className="space-y-4">
      <div className="h-9 w-full bg-[#111827] rounded-xl animate-pulse" />
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
        {Array.from({ length: 12 }).map((_, i) => (
          <SkeletonCard key={i} />
        ))}
      </div>
    </div>
  );
}
