import Link from "next/link";
import ActionBadge from "./ActionBadge";
import RunPredictionButton from "./RunPredictionButton";
import type { StockSummary } from "@/types";

interface Props {
  stock: StockSummary;
  selectMode?: boolean;
  selected?: boolean;
  onToggleSelect?: (symbol: string) => void;
}

export default function StockCard({ stock, selectMode, selected, onToggleSelect }: Props) {
  const pred = stock.latestPrediction;
  const acc = stock.accuracy;

  function handleCardClick() {
    if (selectMode && onToggleSelect) onToggleSelect(stock.symbol);
  }

  return (
    <div
      className={`card-hover p-5 flex flex-col gap-4 relative transition-all ${
        selectMode ? "cursor-pointer" : ""
      } ${selected ? "border-violet-500/60 bg-violet-900/10 shadow-violet-500/10 shadow-lg" : ""}`}
      onClick={selectMode ? handleCardClick : undefined}
    >
      {/* ── Select checkbox overlay ────────────────────────────────────── */}
      {selectMode && (
        <button
          onClick={(e) => { e.stopPropagation(); onToggleSelect?.(stock.symbol); }}
          className={`absolute top-3 right-3 z-10 w-5 h-5 rounded flex items-center justify-center border transition-all ${
            selected
              ? "bg-violet-600 border-violet-500 text-white"
              : "bg-[#111827] border-[#2a3a5c] text-transparent hover:border-violet-400"
          }`}
          aria-label={selected ? "Deselect" : "Select"}
        >
          {selected && (
            <svg viewBox="0 0 10 8" fill="none" className="w-3 h-3">
              <path
                d="M1 4L3.5 6.5L9 1.5"
                stroke="currentColor"
                strokeWidth="1.5"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          )}
        </button>
      )}

      {/* ── Header ────────────────────────────────────────────────────── */}
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0 flex-1">
          {selectMode ? (
            <p className="text-base font-bold text-slate-100 font-mono">{stock.symbol}</p>
          ) : (
            <Link
              href={`/stocks/${stock.symbol}`}
              className="text-base font-bold text-slate-100 hover:text-blue-400 transition-colors font-mono"
              onClick={(e) => e.stopPropagation()}
            >
              {stock.symbol}
            </Link>
          )}
          <p className="text-xs text-slate-500 mt-0.5 truncate" title={stock.name}>
            {stock.name}
          </p>
          <div className="flex items-center gap-1.5 mt-1 flex-wrap">
            {stock.sector && (
              <span className="text-[10px] text-slate-600 bg-[#1e2a45] px-1.5 py-0.5 rounded">
                {stock.sector}
              </span>
            )}
            {(stock.nepseId ?? 0) >= 5000 && (
              <span className="text-[10px] font-semibold text-cyan-400 bg-cyan-900/30 border border-cyan-500/30 px-1.5 py-0.5 rounded">
                NEW
              </span>
            )}
          </div>
        </div>
        {/* Action badge (hidden when checkbox is in top-right) */}
        {!selectMode && (
          <>
            {pred ? (
              <ActionBadge action={pred.action} />
            ) : (
              <span className="text-xs text-slate-600 bg-[#1e2a45] px-2.5 py-1 rounded-full shrink-0">
                No data
              </span>
            )}
          </>
        )}
        {selectMode && pred && (
          <ActionBadge action={pred.action} size="sm" />
        )}
      </div>

      {/* ── Stats grid ────────────────────────────────────────────────── */}
      <div className="grid grid-cols-2 gap-x-4 gap-y-3">
        <div>
          <p className="stat-label">Predicted Close</p>
          <p className="stat-value text-base">
            {pred ? `NPR ${pred.predictedClose.toFixed(2)}` : "—"}
          </p>
        </div>
        <div>
          <p className="stat-label">Confidence</p>
          <p className="stat-value text-base">
            {pred ? `${(pred.modelConfidence * 100).toFixed(0)}%` : "—"}
          </p>
        </div>
        <div>
          <p className="stat-label">Dir Probability</p>
          <p
            className={`stat-value text-base ${
              pred && pred.directionProb >= 0.6
                ? "text-emerald-400"
                : pred && pred.directionProb <= 0.4
                  ? "text-red-400"
                  : "text-amber-400"
            }`}
          >
            {pred ? `${(pred.directionProb * 100).toFixed(0)}%` : "—"}
          </p>
        </div>
        <div>
          <p className="stat-label">Dir Accuracy</p>
          <p
            className={`stat-value text-base ${
              acc && acc.directionAccuracy >= 55
                ? "text-emerald-400"
                : acc && acc.directionAccuracy < 45
                  ? "text-red-400"
                  : "text-slate-300"
            }`}
          >
            {acc ? `${acc.directionAccuracy.toFixed(0)}%` : "—"}
          </p>
        </div>
      </div>

      {/* ── Footer ────────────────────────────────────────────────────── */}
      <div className="mt-auto space-y-2">
        {pred && (
          <p className="text-[10px] text-slate-600 text-right font-mono">
            {pred.predictionDate} · {pred.regime}
          </p>
        )}
        {/* Hide run button in select mode — bulk runner replaces it */}
        {!selectMode && <RunPredictionButton symbol={stock.symbol} />}
      </div>
    </div>
  );
}
