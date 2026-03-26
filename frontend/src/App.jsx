import React, { useState } from 'react';
import IndexPage from './pages/IndexPage';
import StockDetailPage from './pages/StockDetailPage';
import { Layout } from 'lucide-react';

function App() {
  const [selectedStock, setSelectedStock] = useState(null);

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 font-sans">
      <nav className="bg-gray-900 border-b border-gray-800 px-6 py-4 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center space-x-2 cursor-pointer" onClick={() => setSelectedStock(null)}>
            <div className="bg-indigo-600 p-1.5 rounded-lg">
              <Layout className="text-white" size={24} />
            </div>
            <span className="text-xl font-black tracking-tight text-white">NEPSE<span className="text-indigo-500">ML</span></span>
          </div>
          <div className="hidden md:flex space-x-8 text-sm font-medium text-gray-400">
            <a href="#" className="hover:text-white transition">Dashboard</a>
            <a href="#" className="hover:text-white transition">Top Gainers</a>
            <a href="#" className="hover:text-white transition">Sector Analysis</a>
          </div>
          <div className="flex items-center space-x-4">
            <div className="text-xs text-right hidden sm:block">
              <p className="text-gray-400">Market Status</p>
              <p className="text-green-500 font-bold">● OPEN</p>
            </div>
            <img 
              src="https://api.dicebear.com/7.x/avataaars/svg?seed=Manish" 
              alt="User" 
              className="w-10 h-10 rounded-full border border-gray-700"
            />
          </div>
        </div>
      </nav>

      <main className="max-w-7xl mx-auto px-6 py-8">
        {selectedStock ? (
          <StockDetailPage 
            symbol={selectedStock} 
            onBack={() => setSelectedStock(null)} 
          />
        ) : (
          <div className="space-y-12">
            {/* Hero Section */}
            <div className="relative bg-indigo-900/20 border border-indigo-500/10 rounded-[2.5rem] p-12 overflow-hidden shadow-2xl">
               <div className="absolute -top-24 -right-24 w-96 h-96 bg-indigo-500/10 blur-[100px] rounded-full"></div>
               <div className="absolute -bottom-24 -left-24 w-64 h-64 bg-indigo-600/5 blur-[80px] rounded-full"></div>
               
               <div className="relative z-10 max-w-2xl">
                  <div className="inline-flex items-center px-3 py-1 bg-indigo-600/10 border border-indigo-500/20 rounded-full text-[10px] font-black uppercase tracking-[0.2em] text-indigo-400 mb-6">
                    <span className="flex h-2 w-2 rounded-full bg-indigo-400 mr-2 animate-pulse"></span>
                    Advanced AI Trading Intelligence
                  </div>
                  <h1 className="text-5xl md:text-6xl font-black text-white tracking-tighter leading-none mb-6">
                    NEPSE <span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 to-indigo-600 font-black">MARKET</span> PREDICTOR
                  </h1>
                  <p className="text-gray-400 text-lg font-medium leading-relaxed max-w-lg mb-8">
                    Leveraging ensemble Machine Learning and real-time sentiment analysis to forecast 
                    Nepal Stock Exchange trends with institutional-grade accuracy.
                  </p>
               </div>
            </div>

            <IndexPage onSelectStock={(symbol) => setSelectedStock(symbol)} />
          </div>
        )}
      </main>

      <footer className="bg-gray-900 border-t border-gray-800 py-10 mt-20">
        <div className="max-w-7xl mx-auto px-6 flex flex-col md:flex-row justify-between items-center space-y-4 md:space-y-0">
          <div className="flex items-center space-x-2">
            <span className="text-sm font-bold text-gray-500">NEPSE ML Predictor v5.0</span>
          </div>
          <p className="text-gray-600 text-xs">
            &copy; 2026 Professional Trading-Grade AI. Not financial advice.
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;
