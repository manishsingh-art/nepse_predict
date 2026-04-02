"use client";

import { useState } from "react";
import { RefreshCw, CheckCircle, AlertCircle } from "lucide-react";

interface Props {
  onSuccess?: (total: number) => void;
}

type State = "idle" | "loading" | "success" | "error";

export default function SyncButton({ onSuccess }: Props) {
  const [state, setState] = useState<State>("idle");
  const [result, setResult] = useState<{ fetched: number; total: number } | null>(null);
  const [errorMsg, setErrorMsg] = useState("");

  async function handleSync() {
    if (state === "loading") return;
    setState("loading");
    setResult(null);
    setErrorMsg("");

    try {
      const res = await fetch("/api/stocks/sync", { method: "POST" });
      const json = await res.json();

      if (!res.ok) throw new Error(json.error ?? "Sync failed");

      setResult({ fetched: json.data.fetched, total: json.data.total });
      setState("success");
      onSuccess?.(json.data.total);

      // Reload after a short delay so the grid updates
      setTimeout(() => window.location.reload(), 1200);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Unknown error";
      setErrorMsg(msg);
      setState("error");
      setTimeout(() => setState("idle"), 6000);
    }
  }

  return (
    <div className="flex flex-col items-end gap-1">
      <button
        onClick={handleSync}
        disabled={state === "loading"}
        className={`flex items-center gap-2 text-xs font-medium px-3.5 py-2 rounded-lg border transition-all ${
          state === "loading"
            ? "bg-[#1e2a45] border-[#2a3a5c] text-slate-500 cursor-not-allowed"
            : state === "success"
              ? "bg-emerald-900/30 border-emerald-700/50 text-emerald-400"
              : state === "error"
                ? "bg-red-900/30 border-red-700/50 text-red-400"
                : "bg-[#111827] border-[#1e2a45] text-slate-400 hover:border-blue-500/50 hover:text-blue-400"
        }`}
        title="Fetch full company list from merolagani.com and sync into the database"
      >
        {state === "loading" ? (
          <RefreshCw size={13} className="animate-spin" />
        ) : state === "success" ? (
          <CheckCircle size={13} />
        ) : state === "error" ? (
          <AlertCircle size={13} />
        ) : (
          <RefreshCw size={13} />
        )}
        {state === "loading"
          ? "Syncing…"
          : state === "success"
            ? `Done — ${result?.total} companies`
            : state === "error"
              ? "Sync failed"
              : "Sync All Companies"}
      </button>

      {state === "success" && result && (
        <p className="text-[10px] text-slate-600">
          {result.fetched} fetched from merolagani.com · {result.total} total in DB
        </p>
      )}
      {state === "error" && errorMsg && (
        <p className="text-[10px] text-red-500 max-w-xs text-right" title={errorMsg}>
          {errorMsg.slice(0, 80)}
        </p>
      )}
    </div>
  );
}
