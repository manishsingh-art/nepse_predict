import { format } from "date-fns";
import ActionBadge from "./ActionBadge";
import type { ForecastPoint } from "@/types";

interface Props {
  forecasts: ForecastPoint[];
  lastClose?: number;
}

export default function ForecastTable({ forecasts, lastClose }: Props) {
  if (forecasts.length === 0) {
    return (
      <div className="py-10 text-center text-slate-500 text-sm">
        No forecasts available. Run a prediction to generate the 7-day outlook.
      </div>
    );
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-[#1e2a45]">
            <th className="text-left px-3 py-2.5 text-xs text-slate-500 font-medium">Date</th>
            <th className="text-right px-3 py-2.5 text-xs text-slate-500 font-medium">
              Predicted
            </th>
            <th className="text-right px-3 py-2.5 text-xs text-slate-500 font-medium">Change</th>
            <th className="text-right px-3 py-2.5 text-xs text-slate-500 font-medium">Low Band</th>
            <th className="text-right px-3 py-2.5 text-xs text-slate-500 font-medium">High Band</th>
            <th className="text-right px-3 py-2.5 text-xs text-slate-500 font-medium">Dir Prob</th>
            <th className="text-center px-3 py-2.5 text-xs text-slate-500 font-medium">Signal</th>
          </tr>
        </thead>
        <tbody>
          {forecasts.map((f, i) => {
            const baseline = i === 0 ? (lastClose ?? f.predictedClose) : forecasts[i - 1].predictedClose;
            const changePct = ((f.predictedClose - baseline) / (baseline + 1e-9)) * 100;
            const isUp = changePct >= 0;

            return (
              <tr key={f.id} className="border-b border-[#1e2a45]/50 table-row-hover">
                <td className="px-3 py-2.5 font-mono text-slate-400 text-xs">
                  {format(new Date(f.targetDate), "EEE, MMM d")}
                </td>
                <td className="px-3 py-2.5 text-right font-mono font-semibold text-slate-200">
                  {f.predictedClose.toFixed(2)}
                </td>
                <td
                  className={`px-3 py-2.5 text-right font-mono text-xs ${
                    isUp ? "text-emerald-400" : "text-red-400"
                  }`}
                >
                  {isUp ? "+" : ""}
                  {changePct.toFixed(2)}%
                </td>
                <td className="px-3 py-2.5 text-right font-mono text-xs text-slate-500">
                  {f.lowerBound.toFixed(2)}
                </td>
                <td className="px-3 py-2.5 text-right font-mono text-xs text-slate-500">
                  {f.upperBound.toFixed(2)}
                </td>
                <td
                  className={`px-3 py-2.5 text-right font-mono text-xs ${
                    f.directionProb >= 0.6
                      ? "text-emerald-400"
                      : f.directionProb <= 0.4
                        ? "text-red-400"
                        : "text-amber-400"
                  }`}
                >
                  {(f.directionProb * 100).toFixed(0)}%
                </td>
                <td className="px-3 py-2.5 text-center">
                  <ActionBadge action={f.action} size="sm" />
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
