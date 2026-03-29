import React, { useState, useEffect, useCallback, useRef } from 'react';
import { getStockDetail, runPrediction } from '../services/api';
import {
  TrendingUp, TrendingDown, ArrowLeft, RefreshCw,
  ShieldCheck, AlertTriangle, Zap, Activity,
  LineChart, BarChart3, PieChart, Info, Target, Database
} from 'lucide-react';

const fmt = (v, d = 2) => (v != null ? Number(v).toFixed(d) : '—');
const fmtPct = (v) => (v != null ? `${v >= 0 ? '+' : ''}${Number(v).toFixed(2)}%` : '—');
const fmtNPR = (v) => (v != null ? `Rs. ${Number(v).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}` : '—');

const MetricCard = ({ label, value, valueClass = 'text-white' }) => (
  <div className="bg-[#21262d] rounded-xl p-3 border border-[#30363d]">
    <p className="text-[8px] text-gray-500 font-black uppercase tracking-widest mb-1">{label}</p>
    <p className={`text-sm font-bold font-mono ${valueClass}`}>{value}</p>
  </div>
);

const SectionHeader = ({ icon: Icon, title }) => (
  <div className="flex items-center gap-2 mb-4 pb-3 border-b border-gray-800">
    <Icon size={12} className="text-indigo-400 shrink-0" />
    <h3 className="text-[10px] font-black text-gray-500 uppercase tracking-[0.2em]">{title}</h3>
  </div>
);

const StockDetailPage = ({ symbol, onBack }) => {
  const [stock, setStock] = useState(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [refreshMsg, setRefreshMsg] = useState('');
  const initialized = useRef(false);

  const fetchDetail = useCallback(async (isRefresh = false) => {
    if (isRefresh) setRefreshing(true); else setLoading(true);
    try {
      const data = await getStockDetail(symbol);
      setStock(data);
    } catch (err) {
      console.error('Fetch error:', err);
    } finally {
      setLoading(false); setRefreshing(false);
    }
  }, [symbol]);

  useEffect(() => {
    if (!initialized.current) { fetchDetail(); initialized.current = true; }
  }, [fetchDetail]);

  const handleRefresh = async () => {
    setRefreshing(true);
    setRefreshMsg('Running ML Engine… please wait.');
    try {
      await runPrediction(symbol);
      setTimeout(() => { fetchDetail(true); setRefreshMsg(''); }, 15000);
    } catch (err) {
      console.error(err); setRefreshing(false); setRefreshMsg('');
    }
  };

  if (loading) return (
    <div className="flex flex-col items-center justify-center h-96 gap-4">
      <div className="relative">
        <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-indigo-500" />
        <div className="absolute inset-0 flex items-center justify-center"><Zap size={20} className="text-indigo-400" /></div>
      </div>
      <p className="text-gray-500 font-bold text-xs uppercase tracking-widest">Loading structural data…</p>
    </div>
  );

  if (!stock) return <div className="text-center py-20 text-red-400 font-bold">Failed to load data. API may be unavailable.</div>;

  const tp = stock.trade_plan || {};
  const sent = stock.sentiment || 0;
  const scenarios = stock.scenarios || {};
  const forecast = stock.forecast || [];
  const anomalies = stock.anomalies || [];
  const risks = stock.risks || [];

  // Derived properties for display
  const latestForecast = forecast.length > 0 ? forecast[0] : null;
  const signal = latestForecast ? (latestForecast.prob_up > 55 ? 'BUY' : latestForecast.prob_up < 45 ? 'SELL' : 'HOLD') : 'HOLD';
  const signalColor = signal === 'BUY' ? 'text-green-400' : signal === 'SELL' ? 'text-red-400' : 'text-amber-400';
  const signalBg = signal === 'BUY' ? 'bg-green-500/10 border-green-500/25' : signal === 'SELL' ? 'bg-red-500/10 border-red-500/25' : 'bg-amber-500/10 border-amber-500/25';

  return (
    <div className="space-y-4 pb-24 animate-fade-in">
      {/* ── Navigation ── */}
      <div className="flex items-center justify-between">
        <button onClick={onBack} className="flex items-center gap-2 px-4 py-2 bg-gray-900 border border-gray-800 rounded-xl text-gray-400 hover:text-white hover:border-indigo-500/50 transition-all text-xs font-bold uppercase tracking-widest">
          <ArrowLeft size={14} /> Back to Hub
        </button>
        <span className="text-[9px] text-gray-700 uppercase tracking-widest font-mono">NEPSE-ML-CORE v5.0</span>
      </div>

      {/* ── Hero Header ── */}
      <div className="bg-[#161b22] border border-[#30363d] rounded-2xl p-6 flex flex-col md:flex-row justify-between gap-6">
        <div className="space-y-3">
          <div className="flex flex-wrap items-center gap-3">
            <h1 className="text-4xl font-black text-white tracking-tighter">{stock.symbol}</h1>
            <span className={`text-[9px] font-black uppercase tracking-widest px-3 py-1 rounded-full border ${signalBg} ${signalColor}`}>
              {signal} Signal
            </span>
            <span className="text-[9px] font-black uppercase tracking-widest px-2 py-1 rounded border border-indigo-500/20 text-indigo-400 bg-indigo-500/5">
              {stock.trend || 'NEUTRAL'} Trend
            </span>
          </div>
          
          <div className="flex flex-wrap gap-x-6 gap-y-1 text-[10px] font-mono text-gray-500 mt-2">
            <span>RSI (14): <span className={stock.rsi > 70 ? 'text-red-400' : stock.rsi < 30 ? 'text-green-400' : 'text-white'}>{fmt(stock.rsi)}</span></span>
            <span>Sentiment: <span className={sent > 0 ? 'text-green-400' : sent < 0 ? 'text-red-400' : 'text-white'}>{fmt(sent)}</span></span>
          </div>
        </div>

        {/* Price block */}
        <div className="text-right space-y-1">
          <div className="text-4xl font-black font-mono text-[#3fb950] tracking-tighter">{fmtNPR(stock.price)}</div>
          
          <button onClick={handleRefresh} disabled={refreshing}
            className="mt-3 flex items-center gap-2 ml-auto px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white text-[10px] font-black uppercase tracking-widest rounded-xl transition-all disabled:opacity-50">
            <RefreshCw size={12} className={refreshing ? 'animate-spin' : ''} />
            {refreshing ? 'Running ML…' : 'Analyze'}
          </button>
          {refreshMsg && <p className="text-[9px] text-amber-400 text-right animate-pulse">{refreshMsg}</p>}
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Trade Plan */}
        <div className="bg-[#161b22] border border-[#30363d] rounded-2xl p-5 md:col-span-1">
          <SectionHeader icon={Target} title="Trade Strategy" />
          <div className="space-y-2 mb-4">
            <div className="flex justify-between items-center p-2.5 rounded-lg bg-blue-500/5 border border-blue-500/20">
              <span className="text-[8px] font-black uppercase tracking-widest text-blue-400">Buy Zone</span>
              <span className="text-sm font-bold font-mono text-blue-400">{tp.buy_zone && tp.buy_zone.length > 1 ? `${fmtNPR(tp.buy_zone[0])} - ${fmtNPR(tp.buy_zone[1])}` : '—'}</span>
            </div>
            <div className="flex justify-between items-center p-2.5 rounded-lg bg-green-500/5 border border-green-500/20">
              <span className="text-[8px] font-black uppercase tracking-widest text-green-400">Target (TP)</span>
              <span className="text-sm font-bold font-mono text-green-400">{fmtNPR(tp.target)}</span>
            </div>
            <div className="flex justify-between items-center p-2.5 rounded-lg bg-red-500/5 border border-red-500/20">
              <span className="text-[8px] font-black uppercase tracking-widest text-red-400">Stop Loss</span>
              <span className="text-sm font-bold font-mono text-red-400">{fmtNPR(tp.stop_loss)}</span>
            </div>
          </div>
          <div className="flex justify-between items-center border-t border-gray-800 pt-3">
            <span className="text-gray-500 uppercase font-black tracking-widest text-[9px]">Risk/Reward Ratio</span>
            <span className="font-bold text-white font-mono text-[11px]">{fmt(tp.rr_ratio)}x</span>
          </div>
        </div>

        {/* Forecast Table */}
        <div className="bg-[#161b22] border border-[#30363d] rounded-2xl overflow-hidden md:col-span-2">
          <div className="p-4 border-b border-gray-800 bg-[#1c2128]">
            <h3 className="text-[10px] font-black text-gray-500 uppercase tracking-[0.2em] flex items-center gap-2">
              <LineChart size={12} className="text-indigo-400" /> ML Forecast Vectors
            </h3>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="border-b border-gray-800/50 bg-[#0d1117]/30">
                  {['Date', 'Predicted', 'Δ%', 'P(Up)', 'Cnf'].map(h => (
                    <th key={h} className="px-4 py-3 text-[8px] font-black text-gray-500 uppercase tracking-widest whitespace-nowrap">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-800/30">
                {forecast.map((f, i) => (
                  <tr key={i} className="hover:bg-gray-800/20 transition-colors font-mono">
                    <td className="px-4 py-3 text-[11px] text-gray-300 whitespace-nowrap">{f.date}</td>
                    <td className="px-4 py-3 text-[12px] text-white font-bold">{fmtNPR(f.price)}</td>
                    <td className={`px-4 py-3 text-[11px] font-bold ${(f.change_pct ?? 0) >= 0 ? 'text-[#3fb950]' : 'text-[#f85149]'}`}>
                      {fmtPct(f.change_pct)}
                    </td>
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-1.5">
                        <div className="w-12 h-1.5 bg-[#30363d] rounded-full overflow-hidden">
                          <div className={`h-full rounded-full ${(f.prob_up ?? 0) >= 50 ? 'bg-[#3fb950]' : 'bg-[#f85149]'}`}
                            style={{ width: `${f.prob_up}%` }} />
                        </div>
                        <span className={`text-[10px] font-bold ${(f.prob_up ?? 0) >= 50 ? 'text-[#3fb950]' : 'text-[#f85149]'}`}>
                          {fmt(f.prob_up, 0)}%
                        </span>
                      </div>
                    </td>
                    <td className="px-4 py-3 text-[10px] text-gray-400">{fmt(f.confidence, 1)} / 10</td>
                  </tr>
                ))}
                {forecast.length === 0 && (
                   <tr><td colSpan="5" className="text-center py-6 text-gray-500 text-[10px] uppercase font-bold tracking-widest">No forecast data generated. Run prediction.</td></tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Scenario Analysis */}
        <div className="bg-[#161b22] border border-[#30363d] rounded-2xl p-5">
          <SectionHeader icon={PieChart} title="Scenario Analysis" />
          <div className="space-y-3">
            {['bull', 'base', 'bear'].map((k, i) => {
              const sc = scenarios[k];
              if (!sc) return null;
              const colors = ['text-green-400 bg-green-500/5 border-green-500/20', 'text-blue-400 bg-blue-500/5 border-blue-500/20', 'text-red-400 bg-red-500/5 border-red-500/20'];
              return (
                <div key={k} className={`p-3 rounded-xl border ${colors[i]}`}>
                  <div className="flex justify-between items-center mb-1">
                    <span className="text-[9px] font-black uppercase tracking-widest">{k} Case</span>
                    <span className="text-[9px] font-bold font-mono">{sc.prob}%</span>
                  </div>
                  <div className="flex justify-between items-baseline">
                    <span className="text-base font-bold font-mono">{fmtNPR(sc.target)}</span>
                    <span className="text-[10px] font-mono">{fmtPct(((sc.target/stock.price)-1)*100)}</span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Anomalies and Risks */}
        <div className="space-y-4">
          <div className="bg-[#161b22] border border-[#30363d] rounded-2xl p-5">
            <SectionHeader icon={AlertTriangle} title={`Anomalies (${anomalies.length})`} />
            <div className="space-y-2 max-h-40 overflow-y-auto pr-1">
              {anomalies.length === 0 ? (
                <div className="text-center py-4 text-gray-600 text-[10px] uppercase tracking-widest">No Anomalies</div>
              ) : anomalies.map((a, i) => (
                <div key={i} className="flex justify-between items-center p-2 rounded bg-[#21262d] border border-[#30363d]">
                  <span className="text-[9px] text-gray-500 font-mono">{a.date}</span>
                  <span className="text-[9px] font-black uppercase text-indigo-400">{a.type}</span>
                  <span className={`text-[10px] font-bold font-mono ${a.change_pct >= 0 ? 'text-green-400' : 'text-red-400'}`}>{fmtPct(a.change_pct)}</span>
                </div>
              ))}
            </div>
          </div>
          
          {risks.length > 0 && (
            <div className="bg-[#161b22] border border-amber-500/20 rounded-2xl p-5">
              <SectionHeader icon={ShieldCheck} title="Identified Risks" />
              <div className="space-y-2 text-[11px] text-gray-300">
                {risks.map((r, i) => (
                  <div key={i} className="flex gap-2 items-center"><AlertTriangle size={12} className="text-amber-400"/> {r}</div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* ── AI Analyst Summary ── */}
      <div className="bg-[#1c2128] border border-indigo-500/20 rounded-2xl p-8 relative overflow-hidden">
        <div className="absolute top-0 left-0 bottom-0 w-1.5 bg-indigo-500/50 rounded-l-2xl" />
        <div className="absolute -right-20 -bottom-20 w-64 h-64 bg-indigo-500/5 blur-[80px] rounded-full" />
        <div className="relative z-10">
          <h3 className="text-[11px] font-black text-indigo-400 uppercase tracking-[0.3em] mb-4 flex items-center gap-3">
            <Activity size={14} /> AI Synthesis
          </h3>
          <p className="text-[#e6edf3] text-sm leading-relaxed font-medium">
            {stock.ai_summary || 'Run prediction model to compute synthesis.'}
          </p>
        </div>
      </div>
    </div>
  );
};

export default StockDetailPage;
