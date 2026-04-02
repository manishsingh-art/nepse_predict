"use client";

import dynamic from "next/dynamic";
import { useState } from "react";
import ForecastTable from "./ForecastTable";
import HistoricalTable from "./HistoricalTable";
import AccuracyMetrics from "./AccuracyMetrics";
import RunPredictionButton from "./RunPredictionButton";
import ActionBadge from "./ActionBadge";
import type { StockDetail, AccuracyMetrics as AccuracyMetricsType } from "@/types";

// Dynamically import recharts chart to avoid SSR issues
const PriceChart = dynamic(() => import("./PriceChart"), {
  ssr: false,
  loading: () => (
    <div className="h-[300px] flex items-center justify-center text-slate-600 text-sm">
      Loading chart…
    </div>
  ),
});

interface Props {
  stock: StockDetail;
  accuracy: AccuracyMetricsType;
}

type Tab = "forecasts" | "history" | "accuracy";

export default function StockDetailClient({ stock, accuracy }: Props) {
  const [activeTab, setActiveTab] = useState<Tab>("forecasts");

  const pred = stock.latestPrediction;

  const tabs: { key: Tab; label: string; count?: number }[] = [
    { key: "forecasts", label: "7-Day Forecast", count: stock.forecasts.length },
    { key: "history", label: "History", count: stock.historicalPredictions.length },
    { key: "accuracy", label: "Accuracy", count: accuracy.totalPredictions },
  ];

  return (
    <div className="space-y-6">
      {/* Header card */}
      <div className="card p-6">
        <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-4">
          <div>
            <div className="flex items-center gap-3 flex-wrap">
              <h1 className="text-3xl font-bold font-mono text-slate-100">{stock.symbol}</h1>
              {pred && <ActionBadge action={pred.action} size="lg" />}
            </div>
            <p className="mt-1 text-slate-400">{stock.name}</p>
            {stock.sector && (
              <span className="inline-block mt-2 text-xs text-slate-500 bg-[#1e2a45] px-2 py-0.5 rounded">
                {stock.sector}
              </span>
            )}
          </div>

          <div className="flex flex-col gap-2 sm:items-end">
            {pred && (
              <div className="grid grid-cols-3 gap-3 sm:gap-4 text-right sm:text-right">
                <StatBlock
                  label="Predicted Close"
                  value={`NPR ${pred.predictedClose.toFixed(2)}`}
                />
                <StatBlock
                  label="Confidence"
                  value={`${(pred.modelConfidence * 100).toFixed(0)}%`}
                />
                <StatBlock
                  label="Dir Probability"
                  value={`${(pred.directionProb * 100).toFixed(0)}%`}
                  highlight={
                    pred.directionProb >= 0.6
                      ? "text-emerald-400"
                      : pred.directionProb <= 0.4
                        ? "text-red-400"
                        : "text-amber-400"
                  }
                />
              </div>
            )}
            <div className="w-full sm:w-48">
              <RunPredictionButton
                symbol={stock.symbol}
                onSuccess={() => window.location.reload()}
              />
            </div>
          </div>
        </div>

        {/* Regime + accuracy pills */}
        {pred && (
          <div className="mt-4 flex flex-wrap gap-2 text-xs">
            <span className="bg-[#1e2a45] text-slate-400 px-2.5 py-1 rounded-full">
              Regime: {pred.regime}
            </span>
            <span className="bg-[#1e2a45] text-slate-400 px-2.5 py-1 rounded-full">
              As of: {pred.predictionDate}
            </span>
            {stock.accuracy && (
              <span
                className={`px-2.5 py-1 rounded-full ${
                  stock.accuracy.directionAccuracy >= 55
                    ? "bg-emerald-900/30 text-emerald-400"
                    : "bg-[#1e2a45] text-slate-400"
                }`}
              >
                Historical dir acc: {stock.accuracy.directionAccuracy.toFixed(0)}%
              </span>
            )}
          </div>
        )}
      </div>

      {/* Chart section */}
      <div className="card p-6">
        <h2 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-4">
          Price Chart
        </h2>
        {stock.actualPrices.length === 0 && stock.forecasts.length === 0 ? (
          <div className="h-40 flex items-center justify-center text-slate-600 text-sm">
            Not enough data — run a prediction to populate the chart.
          </div>
        ) : (
          <PriceChart
            actualPrices={stock.actualPrices}
            forecasts={stock.forecasts}
            lastClose={pred?.predictedClose}
          />
        )}
      </div>

      {/* Tabbed sections */}
      <div className="card">
        {/* Tab bar */}
        <div className="flex border-b border-[#1e2a45] px-2">
          {tabs.map((t) => (
            <button
              key={t.key}
              onClick={() => setActiveTab(t.key)}
              className={`px-4 py-3.5 text-sm font-medium transition-all relative ${
                activeTab === t.key
                  ? "text-slate-200 after:absolute after:bottom-0 after:left-0 after:right-0 after:h-0.5 after:bg-blue-500"
                  : "text-slate-500 hover:text-slate-400"
              }`}
            >
              {t.label}
              {t.count !== undefined && t.count > 0 && (
                <span className="ml-2 text-xs bg-[#1e2a45] text-slate-500 px-1.5 py-0.5 rounded-full">
                  {t.count}
                </span>
              )}
            </button>
          ))}
        </div>

        <div className="p-6">
          {activeTab === "forecasts" && (
            <ForecastTable forecasts={stock.forecasts} lastClose={pred?.predictedClose} />
          )}
          {activeTab === "history" && (
            <HistoricalTable predictions={stock.historicalPredictions} />
          )}
          {activeTab === "accuracy" && <AccuracyMetrics metrics={accuracy} />}
        </div>
      </div>
    </div>
  );
}

function StatBlock({
  label,
  value,
  highlight = "text-slate-100",
}: {
  label: string;
  value: string;
  highlight?: string;
}) {
  return (
    <div>
      <p className="text-[10px] text-slate-600 uppercase tracking-wider">{label}</p>
      <p className={`text-sm font-bold font-mono tabular-nums mt-0.5 ${highlight}`}>{value}</p>
    </div>
  );
}
