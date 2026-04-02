"use client";

import { useState } from "react";
import { RefreshCw, CheckCircle, AlertCircle } from "lucide-react";

interface Props {
  symbol: string;
  onSuccess?: () => void;
}

type State = "idle" | "loading" | "success" | "error";

export default function RunPredictionButton({ symbol, onSuccess }: Props) {
  const [state, setState] = useState<State>("idle");
  const [errorMsg, setErrorMsg] = useState("");

  async function handleClick() {
    if (state === "loading") return;
    setState("loading");
    setErrorMsg("");

    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ symbol }),
      });

      const json = await res.json();

      if (!res.ok) {
        throw new Error(json.error ?? "Prediction failed");
      }

      setState("success");
      onSuccess?.();

      // Reset to idle after 3 seconds
      setTimeout(() => setState("idle"), 3000);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Unknown error";
      setErrorMsg(msg);
      setState("error");
      setTimeout(() => setState("idle"), 5000);
    }
  }

  const configs = {
    idle: {
      label: "Run Prediction",
      className: "bg-blue-600 hover:bg-blue-500 text-white",
      icon: <RefreshCw size={13} />,
    },
    loading: {
      label: "Running…",
      className: "bg-blue-700 text-blue-200 cursor-not-allowed",
      icon: <RefreshCw size={13} className="animate-spin" />,
    },
    success: {
      label: "Updated!",
      className: "bg-emerald-700 text-emerald-200",
      icon: <CheckCircle size={13} />,
    },
    error: {
      label: "Failed",
      className: "bg-red-700 text-red-200",
      icon: <AlertCircle size={13} />,
    },
  };

  const cfg = configs[state];

  return (
    <div>
      <button
        onClick={handleClick}
        disabled={state === "loading"}
        className={`w-full flex items-center justify-center gap-1.5 text-xs font-medium px-3 py-2 rounded-lg transition-all duration-200 ${cfg.className}`}
        title={state === "loading" ? "ML pipeline is running, this may take a few minutes…" : `Run prediction for ${symbol}`}
      >
        {cfg.icon}
        {cfg.label}
      </button>
      {state === "error" && errorMsg && (
        <p className="mt-1 text-[10px] text-red-400 text-center truncate" title={errorMsg}>
          {errorMsg.slice(0, 60)}
        </p>
      )}
    </div>
  );
}
