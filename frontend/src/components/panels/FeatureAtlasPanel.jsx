import React, { useState } from "react";

export default function FeatureAtlasPanel() {
  const [searchTerm, setSearchTerm] = useState("");

  const sampleFeatures = [
    { id: 1402, label: "Indirect Object Identifier", freq: "0.04%", score: 0.94, cluster: "IOI" },
    { id: 1403, label: "Comma Separator Trigger", freq: "1.25%", score: 0.88, cluster: "Syntax" },
    { id: 789, label: "Geographic Capital Probe", freq: "0.12%", score: 0.91, cluster: "Geography" },
    { id: 812, label: "Country Name Entity", freq: "0.45%", score: 0.89, cluster: "Geography" },
    { id: 2104, label: "Base64 Decoding Pattern", freq: "0.01%", score: 0.96, cluster: "Code" },
  ];

  const filtered = sampleFeatures.filter(
    (f) =>
      f.label.toLowerCase().includes(searchTerm.toLowerCase()) ||
      f.id.toString().includes(searchTerm)
  );

  return (
    <div className="p-4 bg-slate-900 text-slate-100 h-full overflow-y-auto">
      <div className="flex justify-between items-center mb-4 pb-2 border-b border-slate-800">
        <div>
          <h2 className="text-lg font-bold text-amber-400">🗺 SAE Feature Atlas</h2>
          <p className="text-xs text-slate-400">Searchable Catalog of 16,384 Sparse Autoencoder Features</p>
        </div>
        <input
          type="text"
          placeholder="Filter features..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          className="px-3 py-1 bg-slate-800 border border-slate-700 rounded text-xs text-slate-200 focus:outline-none focus:border-amber-500 w-48"
        />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        {filtered.map((feature) => (
          <div
            key={feature.id}
            className="p-3 bg-slate-800/80 rounded border border-slate-700 hover:border-amber-500/50 transition"
          >
            <div className="flex justify-between items-start mb-1">
              <span className="text-xs font-bold text-amber-400">Feature #{feature.id}</span>
              <span className="px-2 py-0.5 bg-amber-500/10 text-amber-300 text-[10px] rounded font-mono">
                {feature.cluster}
              </span>
            </div>
            <h3 className="text-sm font-semibold text-slate-200 mb-2">{feature.label}</h3>
            <div className="flex justify-between text-xs text-slate-400 pt-2 border-t border-slate-700/50">
              <span>Firing Freq: {feature.freq}</span>
              <span className="text-emerald-400 font-medium">Interpretability: {feature.score}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
