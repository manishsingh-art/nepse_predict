import { AlertTriangle, RefreshCw } from "lucide-react";

interface Props {
  title?: string;
  message: string;
  onRetry?: () => void;
}

export default function ErrorState({
  title = "Something went wrong",
  message,
  onRetry,
}: Props) {
  return (
    <div className="py-16 flex flex-col items-center gap-4 text-center">
      <div className="w-12 h-12 rounded-full bg-red-900/30 border border-red-800/50 flex items-center justify-center">
        <AlertTriangle size={22} className="text-red-400" />
      </div>
      <div>
        <h3 className="text-sm font-semibold text-slate-300">{title}</h3>
        <p className="mt-1 text-xs text-slate-500 max-w-sm">{message}</p>
      </div>
      {onRetry && (
        <button
          onClick={onRetry}
          className="flex items-center gap-2 text-xs text-blue-400 hover:text-blue-300 transition-colors"
        >
          <RefreshCw size={12} />
          Try again
        </button>
      )}
    </div>
  );
}
