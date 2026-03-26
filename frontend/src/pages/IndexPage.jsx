import React, { useState, useEffect, useCallback, useRef } from 'react';
import { getStocks, runPrediction } from '../services/api';
import { TrendingUp, TrendingDown, ArrowRight, Play, RefreshCw } from 'lucide-react';

const IndexPage = ({ onSelectStock }) => {
  const [stocks, setStocks] = useState([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [runningPredict, setRunningPredict] = useState(null);
  const initialized = useRef(false);

  const fetchStocks = useCallback(async (isManual = false) => {
    if (isManual) setRefreshing(true);
    else setLoading(true);
    
    try {
      const data = await getStocks();
      setStocks(data);
    } catch (error) {
      console.error('Error fetching stocks:', error);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    if (!initialized.current) {
      fetchStocks();
      initialized.current = true;
    }
  }, [fetchStocks]);

  const filteredStocks = stocks.filter(stock => 
    stock.symbol.toLowerCase().includes(searchTerm.toLowerCase()) ||
    stock.name.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const handleRunPredict = async (symbol) => {
    if (runningPredict) return;
    setRunningPredict(symbol);
    try {
      await runPrediction(symbol);
      alert(`Analysis for ${symbol} started. Results will appear shortly.`);
      // Refresh list after a delay to catch the update if it's fast
      setTimeout(() => fetchStocks(true), 10000);
    } catch (error) {
      console.error('Error starting prediction:', error);
    } finally {
      setRunningPredict(null);
    }
  };

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center h-64 space-y-4">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-500"></div>
        <p className="text-gray-400 animate-pulse">Fetching market data...</p>
      </div>
    );
  }

  return (
    <div className="space-y-8 animate-fade-in">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-6 bg-gray-900/50 p-6 rounded-[2rem] border border-gray-800">
        <div>
          <h1 className="text-3xl font-black text-white tracking-tight">Market Overview</h1>
          <p className="text-gray-400 mt-1">Real-time NEPSE signals and AI forecasts</p>
        </div>
        
        <div className="flex flex-1 max-w-md relative group">
          <input 
            type="text"
            placeholder="Search symbol or company..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full bg-gray-950 border border-gray-800 rounded-2xl px-5 py-3 text-white focus:outline-none focus:border-indigo-500 focus:ring-4 focus:ring-indigo-500/10 transition-all pl-12"
          />
          <div className="absolute left-4 top-1/2 -translate-y-1/2 text-gray-500 group-focus-within:text-indigo-400 transition-colors">
            <RefreshCw size={18} className={refreshing ? 'animate-spin' : ''} />
          </div>
        </div>

        <button 
          onClick={() => fetchStocks(true)}
          disabled={refreshing}
          className="flex items-center justify-center px-6 py-3 bg-indigo-600 text-white font-bold rounded-2xl hover:bg-indigo-500 transition-all shadow-lg shadow-indigo-900/20 disabled:opacity-50"
        >
          <RefreshCw size={18} className={`mr-2 ${refreshing ? 'animate-spin' : ''}`} />
          {refreshing ? 'Refreshing...' : 'Refresh Market'}
        </button>
      </div>

      {stocks.length === 0 ? (
        <div className="bg-gray-900 border border-gray-800 rounded-3xl p-12 text-center">
          <p className="text-gray-500 text-lg">No stocks found in database.</p>
          <button 
             onClick={() => fetchStocks(true)}
             className="mt-4 text-indigo-400 hover:text-indigo-300 font-bold"
          >
            Try reloading the market list
          </button>
        </div>
      ) : filteredStocks.length === 0 ? (
        <div className="bg-gray-900 border border-gray-800 rounded-3xl p-12 text-center">
          <p className="text-gray-400">No results matching "{searchTerm}"</p>
          <button 
             onClick={() => setSearchTerm('')}
             className="mt-2 text-indigo-400 hover:text-indigo-300 font-bold text-sm"
          >
            Clear search
          </button>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
          {filteredStocks.map((stock) => (
            <div 
              key={stock.id}
              onClick={() => onSelectStock(stock.symbol)}
              className="bg-gray-900/40 border border-gray-800/80 backdrop-blur-sm rounded-2xl p-6 hover:border-indigo-500/50 hover:bg-gray-800/50 transition-all cursor-pointer group relative overflow-hidden shadow-xl"
            >
              <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
                 <ArrowRight size={48} className="text-indigo-400 -rotate-45" />
              </div>

              <div className="flex justify-between items-start mb-6 relative z-10">
                <div>
                  <h3 className="text-xl font-black text-white group-hover:text-indigo-400 transition-colors uppercase tracking-tight">{stock.symbol}</h3>
                  <p className="text-[10px] text-gray-500 font-black truncate max-w-[150px] uppercase mt-0.5 tracking-wider">{stock.name}</p>
                </div>
                <div className={`px-3 py-1 rounded-full text-[10px] font-black uppercase tracking-widest shadow-sm ${
                  stock.latest_signal === 'BUY' ? 'bg-green-500/10 text-green-400 border border-green-500/20' :
                  stock.latest_signal === 'SELL' ? 'bg-red-500/10 text-red-400 border border-red-500/20' :
                  'bg-gray-800 text-gray-400 border border-gray-700'
                }`}>
                  {stock.latest_signal || 'NO SIGNAL'}
                </div>
              </div>

              <div className="flex justify-between items-end relative z-10">
                <div className="space-y-1">
                  <p className="text-2xl font-mono font-bold text-white tracking-tighter">
                    {stock.latest_price ? `Rs. ${stock.latest_price.toLocaleString()}` : 'N/A'}
                  </p>
                  <div className={`flex items-center text-[10px] font-black tracking-widest uppercase ${stock.latest_change_pct >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                    {stock.latest_change_pct >= 0 ? <TrendingUp size={12} className="mr-1" /> : <TrendingDown size={12} className="mr-1" />}
                    {Math.abs(stock.latest_change_pct).toFixed(2)}%
                  </div>
                </div>
                
                <div className="flex space-x-2">
                  <button 
                    onClick={(e) => { e.stopPropagation(); handleRunPredict(stock.symbol); }}
                    disabled={runningPredict === stock.symbol}
                    className="p-3 bg-gray-950 border border-gray-800 text-gray-500 rounded-xl hover:bg-indigo-600 hover:text-white hover:border-indigo-500 transition-all disabled:opacity-50 group/btn"
                    title="Run New Prediction"
                  >
                    <Play size={16} fill={runningPredict === stock.symbol ? "currentColor" : "none"} className={runningPredict === stock.symbol ? 'animate-pulse' : 'group-hover/btn:fill-current'} />
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default IndexPage;
