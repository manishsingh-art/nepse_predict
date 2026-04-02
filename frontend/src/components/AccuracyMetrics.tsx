import { TrendingUp, Target, BarChart2, CheckCircle } from "lucide-react";
import type { AccuracyMetrics as AccuracyMetricsType } from "@/types";

interface Props {
  metrics: AccuracyMetricsType;
}

export default function AccuracyMetrics({ metrics }: Props) {
  if (metrics.totalPredictions === 0) {
    return (
      <div className="py-10 text-center text-slate-500 text-sm">
        No verified predictions yet. Accuracy is calculated once target dates pass and actual
        prices are recorded.
      </div>
    );
  }

  const stats = [
    {
      icon: <CheckCircle size={18} className="text-emerald-400" />,
      label: "Direction Accuracy",
      value: `${metrics.directionAccuracy.toFixed(1)}%`,
      sub: `${metrics.correctDirections} of ${metrics.totalPredictions} correct`,
      color:
        metrics.directionAccuracy >= 55
          ? "text-emerald-400"
          : metrics.directionAccuracy < 45
            ? "text-red-400"
            : "text-amber-400",
      bg:
        metrics.directionAccuracy >= 55
          ? "bg-emerald-900/20 border-emerald-800/40"
          : metrics.directionAccuracy < 45
            ? "bg-red-900/20 border-red-800/40"
            : "bg-amber-900/20 border-amber-800/40",
    },
    {
      icon: <Target size={18} className="text-blue-400" />,
      label: "Avg Price Error",
      value: `${metrics.avgErrorPct.toFixed(2)}%`,
      sub: "Mean absolute percentage error",
      color:
        metrics.avgErrorPct <= 2
          ? "text-emerald-400"
          : metrics.avgErrorPct <= 5
            ? "text-amber-400"
            : "text-red-400",
      bg: "bg-[#111827] border-[#1e2a45]",
    },
    {
      icon: <BarChart2 size={18} className="text-purple-400" />,
      label: "Total Predictions",
      value: metrics.totalPredictions.toString(),
      sub: "Verified with actual data",
      color: "text-slate-200",
      bg: "bg-[#111827] border-[#1e2a45]",
    },
    {
      icon: <TrendingUp size={18} className="text-slate-400" />,
      label: "Correct Directions",
      value: metrics.correctDirections.toString(),
      sub: `${(metrics.totalPredictions - metrics.correctDirections)} incorrect`,
      color: "text-slate-200",
      bg: "bg-[#111827] border-[#1e2a45]",
    },
  ];

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        {stats.map((s) => (
          <div key={s.label} className={`border rounded-xl p-4 ${s.bg}`}>
            <div className="flex items-center gap-2 mb-2">{s.icon}</div>
            <p className="text-[10px] text-slate-500 uppercase tracking-wider">{s.label}</p>
            <p className={`text-2xl font-bold font-mono tabular-nums mt-0.5 ${s.color}`}>
              {s.value}
            </p>
            <p className="text-[10px] text-slate-600 mt-1">{s.sub}</p>
          </div>
        ))}
      </div>

      {/* Weekly breakdown */}
      {metrics.weeklyBreakdown.length > 0 && (
        <div>
          <h4 className="text-xs text-slate-500 uppercase tracking-wider mb-3">
            Weekly Breakdown (last 12 weeks)
          </h4>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-[#1e2a45]">
                  <th className="text-left px-3 py-2 text-slate-600 font-medium">Week</th>
                  <th className="text-right px-3 py-2 text-slate-600 font-medium">
                    Dir Accuracy
                  </th>
                  <th className="text-right px-3 py-2 text-slate-600 font-medium">
                    Avg Error
                  </th>
                  <th className="text-right px-3 py-2 text-slate-600 font-medium">Count</th>
                </tr>
              </thead>
              <tbody>
                {metrics.weeklyBreakdown.map((w) => (
                  <tr key={w.week} className="border-b border-[#1e2a45]/40 table-row-hover">
                    <td className="px-3 py-2 font-mono text-slate-500">{w.week}</td>
                    <td
                      className={`px-3 py-2 text-right font-mono ${
                        w.dirAccuracy >= 55
                          ? "text-emerald-400"
                          : w.dirAccuracy < 45
                            ? "text-red-400"
                            : "text-amber-400"
                      }`}
                    >
                      {w.dirAccuracy.toFixed(0)}%
                    </td>
                    <td
                      className={`px-3 py-2 text-right font-mono ${
                        w.avgError <= 2
                          ? "text-emerald-400"
                          : w.avgError <= 5
                            ? "text-amber-400"
                            : "text-red-400"
                      }`}
                    >
                      {w.avgError.toFixed(2)}%
                    </td>
                    <td className="px-3 py-2 text-right text-slate-500">{w.count}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
