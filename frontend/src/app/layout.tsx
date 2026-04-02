import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "NEPSE Predictor Dashboard",
  description:
    "ML-powered Nepal Stock Exchange prediction dashboard with ensemble models, regime detection, and smart money signals.",
  keywords: ["NEPSE", "Nepal Stock Exchange", "stock prediction", "ML", "trading"],
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap"
          rel="stylesheet"
        />
      </head>
      <body>
        {/* Top navigation bar */}
        <header className="sticky top-0 z-50 border-b border-[#1e2a45] bg-[#0a0e1a]/80 backdrop-blur-sm">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex items-center justify-between h-14">
              <a href="/stocks" className="flex items-center gap-2.5 group">
                <span className="w-7 h-7 rounded-lg bg-blue-600 flex items-center justify-center text-xs font-bold text-white">
                  N
                </span>
                <span className="font-semibold text-slate-200 group-hover:text-white transition-colors">
                  NEPSE Predictor
                </span>
                <span className="hidden sm:block text-xs text-slate-500 bg-[#1e2a45] px-2 py-0.5 rounded-full font-mono">
                  ML v5.0
                </span>
              </a>

              <nav className="flex items-center gap-4">
                <a
                  href="/stocks"
                  className="text-sm text-slate-400 hover:text-slate-200 transition-colors px-3 py-1.5 rounded-lg hover:bg-[#1e2a45]"
                >
                  Stocks
                </a>
                <span className="text-xs text-slate-600 hidden sm:block">
                  Research only · Not financial advice
                </span>
              </nav>
            </div>
          </div>
        </header>

        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          {children}
        </main>

        <footer className="mt-16 border-t border-[#1e2a45] py-6">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <p className="text-center text-xs text-slate-600">
              NEPSE Predictor · Ensemble ML Pipeline · For research and education only.
              Not financial advice. Past performance does not guarantee future results.
            </p>
          </div>
        </footer>
      </body>
    </html>
  );
}
