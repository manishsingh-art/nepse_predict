import React, { useState, useEffect, useCallback, useRef } from 'react';
import { getStockDetail, runPrediction } from '../services/api';
import {
  TrendingUp, TrendingDown, ArrowLeft, RefreshCw,
  ShieldCheck, AlertTriangle, Zap, Activity,
  LineChart, BarChart3, PieChart, Info, Target, Database
} from 'lucide-react';

// ─── Helpers ────────────────────────────────────────────────────────────────
const fmt = (v, d = 2) => (v != null ? Number(v).toFixed(d) : '—');
const fmtPct = (v) => (v != null ? `${v >= 0 ? '+' : ''}${Number(v).toFixed(2)}%` : '—');
const fmtNPR = (v) => (v != null ? `Rs. ${Number(v).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}` : '—');
const signalColor = (s) => s === 'BUY' ? 'text-green-400' : s === 'SELL' ? 'text-red-400' : 'text-amber-400';
const signalBg   = (s) => s === 'BUY' ? 'bg-green-500/10 border-green-500/25' : s === 'SELL' ? 'bg-red-500/10 border-red-500/25' : 'bg-amber-500/10 border-amber-500/25';

// ─── Sub-components ─────────────────────────────────────────────────────────
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

// ─── Main Component ──────────────────────────────────────────────────────────
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
    setRefreshMsg('Running ML engine… this takes ~15–30 seconds.');
    try {
      await runPrediction(symbol);
      setTimeout(() => { fetchDetail(true); setRefreshMsg(''); }, 20000);
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
      <p className="text-gray-500 font-bold text-xs uppercase tracking-widest">Loading prediction data…</p>
    </div>
  );

  if (!stock) return <div className="text-center py-20 text-red-400 font-bold">Failed to load data.</div>;

  const tech   = stock.technical_info || {};
  const tp     = stock.trade_plan || {};
  const acc    = stock.accuracy_metrics || {};
  const reg    = stock.regime || {};
  const sm     = stock.smart_money || {};
  const sent   = stock.sentiment || {};

  const pctChange = stock.latest_change_pct ?? 0;

  return (
    <div className="space-y-4 pb-24 animate-fade-in">

      {/* ── Navigation ── */}
      <div className="flex items-center justify-between">
        <button onClick={onBack} className="flex items-center gap-2 px-4 py-2 bg-gray-900 border border-gray-800 rounded-xl text-gray-400 hover:text-white hover:border-indigo-500/50 transition-all text-xs font-bold uppercase tracking-widest">
          <ArrowLeft size={14} /> Back to Hub
        </button>
        <span className="text-[9px] text-gray-700 uppercase tracking-widest font-mono">NEPSE-ML-CORE v6.1</span>
      </div>

      {/* ── Hero Header ── */}
      <div className="bg-[#161b22] border border-[#30363d] rounded-2xl p-6 flex flex-col md:flex-row justify-between gap-6">
        <div className="space-y-3">
          <div className="flex flex-wrap items-center gap-3">
            <h1 className="text-4xl font-black text-white tracking-tighter">{stock.symbol}</h1>
            <span className={`text-[9px] font-black uppercase tracking-widest px-3 py-1 rounded-full border ${signalBg(stock.latest_signal)} ${signalColor(stock.latest_signal)}`}>
              {stock.latest_signal || 'HOLD'} Signal
            </span>
            <span className="text-[9px] font-black uppercase tracking-widest px-2 py-1 rounded border border-indigo-500/20 text-indigo-400 bg-indigo-500/5">
              {reg.regime || tech.trend || 'NEUTRAL'} Regime
            </span>
          </div>
          <p className="text-xs text-gray-400 font-bold">{stock.name} · {stock.sector}</p>

          {/* Dataset summary row */}
          <div className="flex flex-wrap gap-x-6 gap-y-1 text-[10px] font-mono text-gray-500">
            {stock.records_count && <span><span className="text-gray-400">{stock.records_count}</span> records</span>}
            {stock.date_range_start && <span>{stock.date_range_start} → {stock.date_range_end}</span>}
            {stock.avg_vol_20d && <span>Avg Vol 20d: <span className="text-gray-400">{Math.round(stock.avg_vol_20d).toLocaleString()}</span></span>}
          </div>

          {/* 52W bar */}
          {stock.week52_high && stock.week52_low && (
            <div className="space-y-1 max-w-xs">
              <div className="flex justify-between text-[9px] font-mono text-gray-500">
                <span className="text-red-400">52W L: {fmtNPR(stock.week52_low)}</span>
                <span className="text-green-400">52W H: {fmtNPR(stock.week52_high)}</span>
              </div>
              <div className="h-1.5 bg-[#30363d] rounded-full overflow-hidden">
                <div className="h-full bg-gradient-to-r from-red-500 to-green-500 rounded-full" style={{
                  width: `${Math.min(100, ((stock.latest_price - stock.week52_low) / (stock.week52_high - stock.week52_low + 1)) * 100)}%`
                }} />
              </div>
            </div>
          )}
        </div>

        {/* Price block */}
        <div className="text-right space-y-1">
          <div className="text-4xl font-black font-mono text-[#3fb950] tracking-tighter">{fmtNPR(stock.latest_price)}</div>
          <div className={`text-sm font-bold font-mono ${pctChange >= 0 ? 'text-[#3fb950]' : 'text-[#f85149]'}`}>{fmtPct(pctChange)}</div>
          <div className="text-[10px] text-gray-600 font-mono">Previous: {fmtNPR(stock.latest_price / (1 + pctChange / 100))}</div>
          <button onClick={handleRefresh} disabled={refreshing}
            className="mt-3 flex items-center gap-2 ml-auto px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white text-[10px] font-black uppercase tracking-widest rounded-xl transition-all disabled:opacity-50">
            <RefreshCw size={12} className={refreshing ? 'animate-spin' : ''} />
            {refreshing ? 'Running ML…' : 'Run Prediction'}
          </button>
          {refreshMsg && <p className="text-[9px] text-amber-400 text-right animate-pulse">{refreshMsg}</p>}
        </div>
      </div>

      {/* ── Row 1: Technicals · Accuracy · Trade Plan ── */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">

        {/* Technicals */}
        <div className="bg-[#161b22] border border-[#30363d] rounded-2xl p-5">
          <SectionHeader icon={Activity} title="Technical Indicators" />
          <div className="grid grid-cols-2 gap-y-3 gap-x-4 text-[11px]">
            {[
              { l: 'RSI (14)', v: fmt(tech.rsi), c: tech.rsi > 70 ? 'text-amber-400' : tech.rsi < 30 ? 'text-green-400' : 'text-white' },
              { l: 'CCI (20)', v: fmt(tech.cci) },
              { l: 'Williams %R', v: fmt(tech.williams_r) },
              { l: 'Stoch %K/%D', v: `${fmt(tech.stoch_k)} / ${fmt(tech.stoch_d)}` },
              { l: 'MACD', v: tech.macd_bullish ? 'BULLISH' : 'BEARISH', c: tech.macd_bullish ? 'text-green-400' : 'text-red-400' },
              { l: 'BB Position', v: `${fmt(tech.bb_position_pct)}%` },
              { l: 'OBV Trend', v: tech.obv_trend || '—' },
              { l: 'Vol Ratio', v: `${fmt(tech.vol_ratio)}x` },
              { l: 'ATR (14)', v: `${fmt(tech.volatility_pct)}%`, c: tech.volatility_pct > 4 ? 'text-amber-400' : 'text-white' },
              { l: 'Trend Score', v: `${tech.score ?? '—'}/6` },
            ].map(({ l, v, c = 'text-white' }) => (
              <div key={l}>
                <p className="text-[8px] text-gray-500 uppercase tracking-widest font-black">{l}</p>
                <p className={`font-bold font-mono ${c}`}>{v}</p>
              </div>
            ))}
          </div>
          <div className="mt-4 pt-3 border-t border-gray-800 flex justify-between text-[9px] font-mono font-black">
            <span>Support: <span className="text-green-400">{fmtNPR(tech.support)}</span></span>
            <span>Resistance: <span className="text-red-400">{fmtNPR(tech.resistance)}</span></span>
          </div>
        </div>

        {/* Model Accuracy */}
        <div className="bg-[#161b22] border border-[#30363d] rounded-2xl p-5">
          <SectionHeader icon={ShieldCheck} title="Model Reliability" />
          <div className="grid grid-cols-2 gap-2 mb-4">
            {[
              { l: 'MAE', v: fmt(acc.avg_mae), c: 'text-amber-400' },
              { l: 'RMSE', v: fmt(acc.avg_rmse), c: 'text-red-400' },
              { l: 'Dir Acc', v: `${fmt(acc.avg_dir_acc, 1)}%`, c: 'text-green-400' },
              { l: 'MAPE', v: `${fmt(acc.avg_mape, 1)}%`, c: 'text-indigo-400' },
            ].map(m => <MetricCard key={m.l} label={m.l} value={m.v} valueClass={m.c} />)}
          </div>
          <div className="space-y-2">
            <p className="text-[8px] text-gray-600 uppercase tracking-widest font-black">Fold Breakdown</p>
            {(acc.folds || []).map((fold, i) => (
              <div key={i} className="flex items-center gap-2">
                <span className="text-[8px] text-gray-500 w-14">Fold {fold.fold}</span>
                <div className="flex-1 h-1.5 bg-[#30363d] rounded-full overflow-hidden">
                  <div className="h-full bg-indigo-500/80 rounded-full" style={{ width: `${Math.min(100, fold.mae * 2)}%` }} />
                </div>
                <span className="text-[9px] font-mono text-gray-400 w-12 text-right">MAE {fold.mae}</span>
              </div>
            ))}
          </div>
          {stock.model_reliability != null && (
            <div className="mt-4 pt-3 border-t border-gray-800 text-[10px] text-gray-400">
              Historical Precision: <span className="text-green-400 font-bold">{stock.model_reliability}%</span>
            </div>
          )}
        </div>

        {/* Trade Plan */}
        <div className="bg-[#161b22] border border-[#30363d] rounded-2xl p-5">
          <SectionHeader icon={Target} title="AI Trade Thesis" />
          <div className="space-y-2 mb-4">
            {[
              { l: 'Buy Zone', v: tp.buy_zone ? `${fmtNPR(tp.buy_zone_low)} – ${fmtNPR(tp.buy_zone_high)}` : fmtNPR(tp.buy_zone), c: 'blue' },
              { l: 'Breakout Buy', v: fmtNPR(tp.breakout_buy), c: 'purple' },
              { l: 'Target (TP)', v: fmtNPR(tp.take_profit), c: 'green' },
              { l: 'Stop Loss (SL)', v: fmtNPR(tp.stop_loss), c: 'red' },
            ].map(({ l, v, c }) => (
              <div key={l} className={`flex justify-between items-center p-2.5 rounded-lg bg-${c}-500/5 border border-${c}-500/20`}>
                <span className={`text-[8px] font-black uppercase tracking-widest text-${c}-400`}>{l}</span>
                <span className={`text-sm font-bold font-mono text-${c}-400`}>{v}</span>
              </div>
            ))}
          </div>
          <div className="grid grid-cols-2 gap-3 text-[9px]">
            <div><p className="text-gray-500 uppercase font-black tracking-widest">Risk/Reward</p><p className="font-bold text-white font-mono">{fmt(tp.risk_reward_ratio)}x</p></div>
            <div><p className="text-gray-500 uppercase font-black tracking-widest">Position Wt</p><p className="font-bold text-amber-400 font-mono">{tp.position_size || tp.position_weight || '—'}%</p></div>
          </div>
          {stock.signal_reason && (
            <p className="mt-3 pt-3 border-t border-gray-800 text-[9px] text-gray-500 italic">{stock.signal_reason}</p>
          )}
        </div>
      </div>

      {/* ── Row 2: Forecast Table ── */}
      <div className="bg-[#161b22] border border-[#30363d] rounded-2xl overflow-hidden">
        <div className="p-4 border-b border-gray-800 bg-[#1c2128] flex justify-between items-center">
          <h3 className="text-[10px] font-black text-gray-500 uppercase tracking-[0.2em] flex items-center gap-2">
            <LineChart size={12} className="text-indigo-400" /> ML Forecast — Next 7 Nepal Trading Sessions
          </h3>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="border-b border-gray-800/50 bg-[#0d1117]/30">
                {['D', 'Date AD', 'Date BS', 'Day', 'Predicted', 'Δ%', 'P(Up)', 'D.Conf', 'Cnf', 'Signal'].map(h => (
                  <th key={h} className="px-4 py-3 text-[8px] font-black text-gray-500 uppercase tracking-widest whitespace-nowrap">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-800/30">
              {stock.forecast.map((f, i) => (
                <tr key={i} className="hover:bg-gray-800/20 transition-colors font-mono">
                  <td className="px-4 py-3 text-[10px] text-gray-600">{i + 1}</td>
                  <td className="px-4 py-3 text-[11px] text-gray-300 whitespace-nowrap">{String(f.date)}</td>
                  <td className="px-4 py-3 text-[10px] text-gray-500 whitespace-nowrap">{f.date_bs || '—'}</td>
                  <td className="px-4 py-3 text-[10px] text-gray-500">{f.day_name || '—'}</td>
                  <td className="px-4 py-3 text-[12px] text-white font-bold">{fmtNPR(f.predicted_close)}</td>
                  <td className={`px-4 py-3 text-[11px] font-bold ${(f.change_pct ?? 0) >= 0 ? 'text-[#3fb950]' : 'text-[#f85149]'}`}>
                    {fmtPct(f.change_pct)}
                  </td>
                  <td className="px-4 py-3">
                    <div className="flex items-center gap-1.5">
                      <div className="w-12 h-1.5 bg-[#30363d] rounded-full overflow-hidden">
                        <div className={`h-full rounded-full ${(f.direction_prob ?? 0) >= 0.5 ? 'bg-[#3fb950]' : 'bg-[#f85149]'}`}
                          style={{ width: `${(f.direction_prob ?? 0) * 100}%` }} />
                      </div>
                      <span className={`text-[10px] font-bold ${(f.direction_prob ?? 0) >= 0.5 ? 'text-[#3fb950]' : 'text-[#f85149]'}`}>
                        {Math.round((f.direction_prob ?? 0) * 100)}%
                      </span>
                    </div>
                  </td>
                  <td className="px-4 py-3 text-[10px] text-gray-400">{f.d_conf != null ? `${Math.round(f.d_conf * 100)}%` : '—'}</td>
                  <td className="px-4 py-3">
                    <span className={`text-[8px] font-black uppercase tracking-widest px-2 py-0.5 rounded border ${
                      String(f.confidence) === 'high' || (typeof f.confidence === 'number' && f.confidence > 0.6) ? 'text-green-400 bg-green-500/5 border-green-500/20' :
                      String(f.confidence) === 'medium' || (typeof f.confidence === 'number' && f.confidence > 0.4) ? 'text-amber-400 bg-amber-500/5 border-amber-500/20' :
                      'text-gray-500 bg-gray-800/40 border-gray-700'
                    }`}>{typeof f.confidence === 'number' ? (f.confidence > 0.6 ? 'HIGH' : f.confidence > 0.4 ? 'MED' : 'LOW') : f.confidence || '—'}</span>
                  </td>
                  <td className="px-4 py-3">
                    <span className={`text-[9px] font-black uppercase ${signalColor(f.signal)}`}>{f.signal || '—'}</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* ── Row 3: Scenarios · Sentiment · Regime ── */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">

        {/* Scenario Analysis */}
        <div className="bg-[#161b22] border border-[#30363d] rounded-2xl p-5">
          <SectionHeader icon={PieChart} title="Scenario Analysis" />
          <div className="space-y-3">
            {(stock.scenarios || []).map((sc, i) => {
              const colors = ['text-green-400 bg-green-500/5 border-green-500/20', 'text-blue-400 bg-blue-500/5 border-blue-500/20', 'text-red-400 bg-red-500/5 border-red-500/20'];
              return (
                <div key={i} className={`p-3 rounded-xl border ${colors[i]}`}>
                  <div className="flex justify-between items-center mb-1">
                    <span className="text-[9px] font-black uppercase tracking-widest">{sc.label}</span>
                    <span className="text-[9px] font-bold font-mono">{sc.probability}%</span>
                  </div>
                  <div className="flex justify-between items-baseline">
                    <span className="text-base font-bold font-mono">{fmtNPR(sc.target)}</span>
                    <span className="text-[10px] font-mono">{fmtPct(sc.change_pct)}</span>
                  </div>
                </div>
              );
            })}
            {(!stock.scenarios || stock.scenarios.length === 0) && (
              <p className="text-[10px] text-gray-600 text-center py-4">Run a prediction to see scenarios.</p>
            )}
          </div>
        </div>

        {/* Sentiment */}
        <div className="bg-[#161b22] border border-[#30363d] rounded-2xl p-5">
          <SectionHeader icon={Activity} title="Sentiment Analytics" />
          <div className="text-center space-y-3">
            <div className={`text-5xl font-black font-mono ${sent.score >= 0.2 ? 'text-green-400' : sent.score <= -0.2 ? 'text-red-400' : 'text-amber-400'}`}>
              {sent.score != null ? (sent.score > 0 ? '+' : '') + Number(sent.score).toFixed(2) : '—'}
            </div>
            <div className="h-2 w-full bg-[#30363d] rounded-full overflow-hidden relative">
              <div className="absolute inset-0 bg-gradient-to-r from-[#f85149] via-[#d29922] to-[#3fb950] opacity-40" />
              <div className="absolute top-0 bottom-0 w-1 bg-white shadow-glow rounded-full transition-all duration-700"
                style={{ left: `${((sent.score ?? 0) + 1) * 50}%` }} />
            </div>
            <div className="flex justify-between text-[7px] font-black text-gray-600 uppercase tracking-widest">
              <span>Bearish</span><span>Neutral</span><span>Bullish</span>
            </div>
            <p className="text-[11px] text-gray-400 italic">"{sent.summary || sent.analysis || 'Neutral market flow detected.'}"</p>
            {sent.label && <span className="text-[9px] font-black uppercase tracking-widest text-gray-500">{sent.label}</span>}
          </div>
        </div>

        {/* Regime */}
        <div className="bg-[#161b22] border border-[#30363d] rounded-2xl p-5">
          <SectionHeader icon={BarChart3} title="Market Regime" />
          <div className="space-y-3">
            <div className="text-center">
              <div className="text-2xl font-black text-indigo-400 tracking-tight">{reg.regime || 'UNKNOWN'}</div>
              {reg.confidence != null && (
                <div className="text-[10px] text-gray-500 mt-1">Confidence: <span className="text-white font-bold">{reg.confidence}%</span></div>
              )}
            </div>
            <div className="grid grid-cols-2 gap-2">
              {[
                { l: 'Regime Vol', v: reg.volatility ? `${fmt(reg.volatility)}%` : '—' },
                { l: 'Trend', v: tech.trend || tech.trend_label || '—' },
                { l: '20d Change', v: fmtPct(tech.change_20d) },
                { l: '5d Momentum', v: fmtPct(tech.recent_5d_momentum) },
              ].map(m => <MetricCard key={m.l} label={m.l} value={m.v} />)}
            </div>
            {sm && (sm.signal || sm.smart_signal) && (
              <div className="pt-3 border-t border-gray-800 text-[10px] text-gray-500">
                Smart Money: <span className="text-white font-bold">{sm.signal || sm.smart_signal}</span>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* ── Row 4: Feature Importance · Anomalies ── */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">

        {/* Feature Importance */}
        <div className="bg-[#161b22] border border-[#30363d] rounded-2xl p-5">
          <SectionHeader icon={BarChart3} title="Top Predictive Features (SHAP)" />
          <div className="space-y-2.5">
            {(stock.top_features || []).slice(0, 12).map((f, i) => {
              const maxVal = stock.top_features[0]?.value || 0.1;
              const pct = Math.min(100, (f.value / maxVal) * 100);
              return (
                <div key={i} className="space-y-0.5">
                  <div className="flex justify-between text-[9px] font-mono">
                    <span className="text-gray-400">{f.name}</span>
                    <span className="text-gray-600">{f.value.toFixed(4)}</span>
                  </div>
                  <div className="h-1.5 w-full bg-[#30363d] rounded-full overflow-hidden">
                    <div className="h-full bg-indigo-500/80 rounded-full" style={{ width: `${pct}%` }} />
                  </div>
                </div>
              );
            })}
            {(!stock.top_features || stock.top_features.length === 0) && (
              <p className="text-[10px] text-gray-600 text-center py-6">Run a prediction to see feature importance.</p>
            )}
          </div>
        </div>

        {/* Anomalies */}
        <div className="bg-[#161b22] border border-[#30363d] rounded-2xl p-5">
          <SectionHeader icon={AlertTriangle} title={`Outlier Engine — ${stock.anomalies?.length ?? 0} Anomalies`} />
          <div className="space-y-2 max-h-72 overflow-y-auto pr-1 custom-scroll">
            {(stock.anomalies || []).length === 0 ? (
              <div className="text-center py-8 text-gray-600 text-[10px] uppercase tracking-widest">Zero Anomalies Detected</div>
            ) : (stock.anomalies || []).map((a, i) => (
              <div key={i} className="flex justify-between items-center p-2.5 rounded-lg bg-[#21262d] border border-[#30363d] hover:border-gray-600 transition-colors font-mono">
                <div>
                  <p className="text-[8px] text-gray-500 uppercase font-black">{a.date}</p>
                  <p className="text-[11px] font-bold text-white">{fmtNPR(a.close)}</p>
                </div>
                <div className="text-center">
                  <p className={`text-[10px] font-bold ${a.label?.includes('SPIKE') || a.type?.includes('SPIKE') ? 'text-green-400' : 'text-red-400'}`}>
                    {a.change_pct != null ? fmtPct(a.change_pct) : '—'}
                  </p>
                  <p className="text-[8px] text-gray-600">{a.label?.includes('SPIKE') || a.type?.includes('SPIKE') ? '🔺 SPIKE' : '🔻 CRASH'}</p>
                </div>
                <span className="text-[9px] text-gray-500">{a.z_score != null ? `${Number(a.z_score).toFixed(1)}σ` : '—'}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* ── Risk Summary ── */}
      {stock.risk_summary?.length > 0 && (
        <div className="bg-[#161b22] border border-amber-500/20 rounded-2xl p-5">
          <SectionHeader icon={AlertTriangle} title="Risk Summary" />
          <div className="space-y-2">
            {stock.risk_summary.map((r, i) => (
              <div key={i} className="flex items-start gap-3 p-3 bg-amber-500/5 border border-amber-500/10 rounded-xl">
                <AlertTriangle size={14} className="text-amber-400 shrink-0 mt-0.5" />
                <span className="text-[11px] text-gray-300 leading-relaxed">{r}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ── AI Analyst Summary ── */}
      <div className="bg-[#1c2128] border border-indigo-500/20 rounded-2xl p-8 relative overflow-hidden">
        <div className="absolute top-0 left-0 bottom-0 w-1.5 bg-indigo-500/50 rounded-l-2xl" />
        <div className="absolute -right-20 -bottom-20 w-64 h-64 bg-indigo-500/5 blur-[80px] rounded-full" />
        <div className="relative z-10">
          <h3 className="text-[11px] font-black text-indigo-400 uppercase tracking-[0.3em] mb-4 flex items-center gap-3">
            <PieChart size={14} /> AI Analyst Diagnosis
          </h3>
          <p className="text-[#e6edf3] text-base leading-relaxed font-medium">
            {stock.ai_summary || 'AI summary will appear after the first prediction run.'}
          </p>
          <div className="flex flex-wrap gap-x-8 gap-y-2 mt-6 pt-6 border-t border-gray-800/50 text-[9px] uppercase tracking-widest text-gray-600">
            <span>Ensemble: LGB+XGB+GBM+RF+Ridge</span>
            <span>Features: 109+ Signals</span>
            <span>CV: Purged Walk-Forward</span>
            <span>Updated: {new Date().toLocaleString()}</span>
          </div>
        </div>
      </div>

    </div>
  );
};

export default StockDetailPage;
