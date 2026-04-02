import { notFound } from "next/navigation";
import { ArrowLeft } from "lucide-react";
import Link from "next/link";
import { getStock } from "@/services/stock.service";
import { getAccuracyMetrics } from "@/services/accuracy.service";
import StockDetailClient from "@/components/StockDetailClient";
import type { Metadata } from "next";

interface Props {
  params: Promise<{ symbol: string }>;
}

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { symbol } = await params;
  return {
    title: `${symbol.toUpperCase()} — NEPSE Predictor`,
    description: `ML ensemble predictions, 7-day forecast, and accuracy metrics for ${symbol.toUpperCase()}`,
  };
}

export default async function StockDetailPage({ params }: Props) {
  const { symbol } = await params;
  const upperSymbol = symbol.toUpperCase();

  const [stock, accuracy] = await Promise.all([
    getStock(upperSymbol),
    getAccuracyMetrics(upperSymbol),
  ]);

  if (!stock) {
    notFound();
  }

  return (
    <div className="space-y-4">
      {/* Breadcrumb */}
      <Link
        href="/stocks"
        className="inline-flex items-center gap-1.5 text-sm text-slate-500 hover:text-slate-300 transition-colors"
      >
        <ArrowLeft size={14} />
        All Stocks
      </Link>

      <StockDetailClient stock={stock} accuracy={accuracy} />
    </div>
  );
}
