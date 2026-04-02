import Link from "next/link";
import { ArrowLeft, Search } from "lucide-react";

export default function StockNotFound() {
  return (
    <div className="py-20 flex flex-col items-center gap-4 text-center">
      <div className="w-14 h-14 rounded-full bg-[#1e2a45] flex items-center justify-center">
        <Search size={24} className="text-slate-500" />
      </div>
      <div>
        <h2 className="text-lg font-semibold text-slate-300">Stock Not Found</h2>
        <p className="mt-1 text-sm text-slate-500 max-w-sm">
          This symbol is not in the database yet. Run a prediction from the stocks list to add it.
        </p>
      </div>
      <Link
        href="/stocks"
        className="inline-flex items-center gap-2 text-sm text-blue-400 hover:text-blue-300 transition-colors"
      >
        <ArrowLeft size={14} />
        Back to Stocks
      </Link>
    </div>
  );
}
