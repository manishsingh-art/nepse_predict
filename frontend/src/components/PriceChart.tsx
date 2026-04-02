"use client";

import {
  ResponsiveContainer,
  ComposedChart,
  Line,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
} from "recharts";
import { format } from "date-fns";
import type { ActualPricePoint, ForecastPoint } from "@/types";

interface Props {
  actualPrices: ActualPricePoint[];
  forecasts: ForecastPoint[];
  lastClose?: number;
}

interface ChartPoint {
  date: string;
  actual?: number;
  forecast?: number;
  lower?: number;
  upper?: number;
  isForecast: boolean;
}

const CustomTooltip = ({
  active,
  payload,
  label,
}: {
  active?: boolean;
  payload?: Array<{ name: string; value: number; color: string }>;
  label?: string;
}) => {
  if (!active || !payload?.length) return null;

  return (
    <div className="bg-[#0f1629] border border-[#2a3a5c] rounded-lg p-3 shadow-xl text-xs space-y-1">
      <p className="text-slate-400 font-mono mb-2">{label}</p>
      {payload.map((p) => (
        <div key={p.name} className="flex items-center justify-between gap-4">
          <span className="text-slate-500">{p.name}</span>
          <span className="font-mono font-bold" style={{ color: p.color }}>
            NPR {Number(p.value).toFixed(2)}
          </span>
        </div>
      ))}
    </div>
  );
};

export default function PriceChart({ actualPrices, forecasts, lastClose }: Props) {
  // Build unified time series
  const actualSorted = [...actualPrices].sort((a, b) => a.date.localeCompare(b.date));
  const lastN = actualSorted.slice(-60); // Last 60 actual data points

  const chartData: ChartPoint[] = [
    ...lastN.map((p) => ({
      date: p.date,
      actual: p.close,
      isForecast: false,
    })),
    // Bridge point: last actual connects to first forecast
    ...(lastClose && forecasts.length > 0
      ? [{ date: actualSorted.at(-1)?.date ?? "", forecast: lastClose, isForecast: false }]
      : []),
    ...forecasts.map((f) => ({
      date: f.targetDate,
      forecast: f.predictedClose,
      lower: f.lowerBound,
      upper: f.upperBound,
      isForecast: true,
    })),
  ];

  if (chartData.length === 0) {
    return (
      <div className="h-64 flex items-center justify-center text-slate-600 text-sm">
        No price data available yet. Run a prediction to generate forecasts.
      </div>
    );
  }

  const allValues = chartData.flatMap((d) =>
    [d.actual, d.forecast, d.lower, d.upper].filter((v): v is number => v !== undefined),
  );
  const minVal = Math.min(...allValues) * 0.98;
  const maxVal = Math.max(...allValues) * 1.02;

  const formatTick = (v: string) => {
    try {
      return format(new Date(v), "MMM d");
    } catch {
      return v;
    }
  };

  return (
    <ResponsiveContainer width="100%" height={300}>
      <ComposedChart data={chartData} margin={{ top: 8, right: 16, bottom: 8, left: 8 }}>
        <defs>
          <linearGradient id="forecastGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.15} />
            <stop offset="95%" stopColor="#f59e0b" stopOpacity={0} />
          </linearGradient>
        </defs>

        <CartesianGrid strokeDasharray="3 3" stroke="#1e2a45" vertical={false} />

        <XAxis
          dataKey="date"
          tickFormatter={formatTick}
          tick={{ fill: "#64748b", fontSize: 11 }}
          axisLine={{ stroke: "#1e2a45" }}
          tickLine={false}
          interval="preserveStartEnd"
        />

        <YAxis
          domain={[minVal, maxVal]}
          tickFormatter={(v: number) => `${v.toFixed(0)}`}
          tick={{ fill: "#64748b", fontSize: 11 }}
          axisLine={false}
          tickLine={false}
          width={55}
        />

        <Tooltip content={<CustomTooltip />} />

        <Legend
          wrapperStyle={{ fontSize: "11px", paddingTop: "12px", color: "#64748b" }}
        />

        {/* Forecast confidence band */}
        <Area
          type="monotone"
          dataKey="upper"
          fill="url(#forecastGrad)"
          stroke="transparent"
          legendType="none"
          name="Upper Band"
        />
        <Area
          type="monotone"
          dataKey="lower"
          fill="transparent"
          stroke="transparent"
          legendType="none"
          name="Lower Band"
        />

        {/* Actual price */}
        <Line
          type="monotone"
          dataKey="actual"
          stroke="#10b981"
          strokeWidth={2}
          dot={false}
          activeDot={{ r: 4, fill: "#10b981" }}
          name="Actual Price"
          connectNulls
        />

        {/* Forecast price */}
        <Line
          type="monotone"
          dataKey="forecast"
          stroke="#f59e0b"
          strokeWidth={2}
          strokeDasharray="5 3"
          dot={{ r: 3, fill: "#f59e0b", strokeWidth: 0 }}
          activeDot={{ r: 5, fill: "#f59e0b" }}
          name="Forecast"
          connectNulls
        />

        {/* Divider line between actual and forecast */}
        {forecasts.length > 0 && actualSorted.length > 0 && (
          <ReferenceLine
            x={actualSorted.at(-1)?.date}
            stroke="#2a3a5c"
            strokeDasharray="3 3"
            label={{ value: "Today", fill: "#475569", fontSize: 10 }}
          />
        )}
      </ComposedChart>
    </ResponsiveContainer>
  );
}
