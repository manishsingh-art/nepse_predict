import { CheckCircle, XCircle, MinusCircle } from "lucide-react";
import ActionBadge from "./ActionBadge";
import type { HistoricalPrediction } from "@/types";

interface Props {
  predictions: HistoricalPrediction[];
}

export default function HistoricalTable({ predictions }: Props) {
  if (predictions.length === 0) {
    return (
      <div className="py-10 text-center text-slate-500 text-sm">
        No historical predictions yet. Run predictions over multiple sessions to build history.
      </div>
    );
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-[#1e2a45]">
            <th className="text-left px-3 py-2.5 text-xs text-slate-500 font-medium">
              Pred Date
            </th>
            <th className="text-left px-3 py-2.5 text-xs text-slate-500 font-medium">
              Target Date
            </th>
            <th className="text-right px-3 py-2.5 text-xs text-slate-500 font-medium">
              Predicted
            </th>
            <th className="text-right px-3 py-2.5 text-xs text-slate-500 font-medium">
              Actual
            </th>
            <th className="text-right px-3 py-2.5 text-xs text-slate-500 font-medium">
              Error %
            </th>
            <th className="text-center px-3 py-2.5 text-xs text-slate-500 font-medium">
              Direction
            </th>
            <th className="text-center px-3 py-2.5 text-xs text-slate-500 font-medium">
              Signal
            </th>
          </tr>
        </thead>
        <tbody>
          {predictions.map((p) => (
            <tr key={p.id} className="border-b border-[#1e2a45]/50 table-row-hover">
              <td className="px-3 py-2.5 font-mono text-slate-500 text-xs">
                {p.predictionDate}
              </td>
              <td className="px-3 py-2.5 font-mono text-slate-400 text-xs">
                {p.targetDate}
              </td>
              <td className="px-3 py-2.5 text-right font-mono text-slate-300 text-xs">
                {p.predictedClose.toFixed(2)}
              </td>
              <td className="px-3 py-2.5 text-right font-mono text-xs">
                {p.accuracy ? (
                  <span className="text-slate-300">{p.accuracy.actualClose.toFixed(2)}</span>
                ) : (
                  <span className="text-slate-600">Pending</span>
                )}
              </td>
              <td className="px-3 py-2.5 text-right font-mono text-xs">
                {p.accuracy ? (
                  <span
                    className={
                      p.accuracy.errorPct <= 2
                        ? "text-emerald-400"
                        : p.accuracy.errorPct <= 5
                          ? "text-amber-400"
                          : "text-red-400"
                    }
                  >
                    {p.accuracy.errorPct.toFixed(2)}%
                  </span>
                ) : (
                  <span className="text-slate-600">—</span>
                )}
              </td>
              <td className="px-3 py-2.5 text-center">
                {p.accuracy ? (
                  p.accuracy.directionCorrect ? (
                    <CheckCircle size={14} className="text-emerald-400 mx-auto" />
                  ) : (
                    <XCircle size={14} className="text-red-400 mx-auto" />
                  )
                ) : (
                  <MinusCircle size={14} className="text-slate-600 mx-auto" />
                )}
              </td>
              <td className="px-3 py-2.5 text-center">
                <ActionBadge action={p.action} size="sm" />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
