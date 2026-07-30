import React, { useState, useEffect } from "react";

export default function CausalTraceViewerPanel() {
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentLayer, setCurrentLayer] = useState(0);

  useEffect(() => {
    let timer;
    if (isPlaying) {
      timer = setInterval(() => {
        setCurrentLayer((prev) => (prev + 1) % 12);
      }, 500);
    }
    return () => clearInterval(timer);
  }, [isPlaying]);

  return (
    <div className="p-4 bg-slate-900 text-slate-100 h-full overflow-y-auto">
      <div className="flex justify-between items-center mb-4 pb-2 border-b border-slate-800">
        <div>
          <h2 className="text-lg font-bold text-indigo-400">🌊 Causal Trace Flow Viewer</h2>
          <p className="text-xs text-slate-400">Layer-by-Layer Activation Recovery Curve</p>
        </div>
        <div className="flex items-center space-x-2">
          <button
            onClick={() => setIsPlaying(!isPlaying)}
            className="px-3 py-1 bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold rounded"
          >
            {isPlaying ? "Pause ⏸" : "Play Flow ▶"}
          </button>
        </div>
      </div>

      <div className="flex space-x-1 items-end h-40 bg-slate-950 p-4 rounded border border-slate-800 mb-4">
        {Array.from({ length: 12 }).map((_, l) => {
          const isCurrent = l === currentLayer;
          const heightPct = 20 + ((l * 7 + (l === 8 ? 60 : 0)) % 80);
          return (
            <div
              key={l}
              onClick={() => setCurrentLayer(l)}
              className="flex-1 flex flex-col items-center cursor-pointer group"
            >
              <div
                style={{ height: `${heightPct}%` }}
                className={`w-full rounded-t transition-all ${
                  isCurrent
                    ? "bg-indigo-400 shadow-lg shadow-indigo-500/50"
                    : "bg-slate-700 group-hover:bg-slate-500"
                }`}
              />
              <span className="text-[10px] mt-2 text-slate-400 font-mono">L{l}</span>
            </div>
          );
        })}
      </div>

      <div className="p-3 bg-slate-800/80 rounded border border-slate-700 flex justify-between items-center">
        <div>
          <span className="text-xs text-slate-400 block">Active Inspection Layer</span>
          <span className="text-sm font-bold text-indigo-300">Layer {currentLayer}</span>
        </div>
        <div className="text-right">
          <span className="text-xs text-slate-400 block">Indirect Causal Score</span>
          <span className="text-sm font-bold text-emerald-400">
            {currentLayer === 8 ? "0.90 (Max Peak)" : `${(0.12 + currentLayer * 0.04).toFixed(2)}`}
          </span>
        </div>
      </div>
    </div>
  );
}
