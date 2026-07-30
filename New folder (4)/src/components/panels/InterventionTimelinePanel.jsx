import React from "react";

export default function InterventionTimelinePanel() {
  return (
    <div className="p-4 bg-slate-900 text-slate-100 h-full overflow-y-auto">
      <div className="mb-4 pb-2 border-b border-slate-800">
        <h2 className="text-lg font-bold text-rose-400">⏱ Intervention Timeline & Diff Viewer</h2>
        <p className="text-xs text-slate-400">Track Original State ➔ Patch Applied ➔ Output Prediction Change</p>
      </div>

      <div className="flex items-center justify-between space-x-3 mb-6">
        <div className="flex-1 p-3 bg-slate-800 rounded border border-slate-700">
          <span className="text-[10px] uppercase font-bold text-slate-400 block mb-1">1. Original State</span>
          <p className="text-xs text-slate-200 font-medium">Prompt: "John gave a book to Mary"</p>
          <span className="text-xs text-emerald-400 block mt-1">Top Token: " Mary" (84.2%)</span>
        </div>

        <div className="text-rose-400 text-xl font-bold">➔</div>

        <div className="flex-1 p-3 bg-rose-950/40 rounded border border-rose-800/60">
          <span className="text-[10px] uppercase font-bold text-rose-400 block mb-1">2. Activation Patch</span>
          <p className="text-xs text-rose-200 font-medium">Layer 8 MLP N402 = 0.0 (Zero Out)</p>
          <span className="text-xs text-rose-300 block mt-1">Intervention: Replacement</span>
        </div>

        <div className="text-rose-400 text-xl font-bold">➔</div>

        <div className="flex-1 p-3 bg-slate-800 rounded border border-slate-700">
          <span className="text-[10px] uppercase font-bold text-slate-400 block mb-1">3. Changed Prediction</span>
          <p className="text-xs text-slate-200 font-medium">Prompt: "John gave a book to Mary"</p>
          <span className="text-xs text-amber-400 block mt-1">Top Token: " John" (42.1%)</span>
        </div>
      </div>

      <div className="p-3 bg-slate-800/50 rounded border border-slate-700">
        <h3 className="text-xs font-bold text-slate-300 mb-2">Causal Logit Shift Delta</h3>
        <div className="w-full bg-slate-700 h-2 rounded overflow-hidden flex">
          <div style={{ width: "65%" }} className="bg-emerald-500 h-full" />
          <div style={{ width: "35%" }} className="bg-rose-500 h-full" />
        </div>
        <div className="flex justify-between text-[10px] text-slate-400 mt-1">
          <span>Mary: -4.2 logits</span>
          <span>John: +3.8 logits</span>
        </div>
      </div>
    </div>
  );
}
