import React, { useState } from "react";

export default function EmbeddingViewerPanel() {
  const [method, setMethod] = useState("PCA");

  const samplePoints = [
    { x: 12, y: 45, label: "IOI Features", color: "#38bdf8" },
    { x: 78, y: 22, label: "Geography Probes", color: "#f59e0b" },
    { x: 45, y: 88, label: "Syntax Patterns", color: "#10b981" },
    { x: 82, y: 75, label: "Code Decoders", color: "#ec4899" },
  ];

  return (
    <div className="p-4 bg-slate-900 text-slate-100 h-full overflow-y-auto">
      <div className="flex justify-between items-center mb-4 pb-2 border-b border-slate-800">
        <div>
          <h2 className="text-lg font-bold text-cyan-400">🌐 3D Embedding Manifold Viewer</h2>
          <p className="text-xs text-slate-400">Dimensionality Reduction (PCA / UMAP)</p>
        </div>
        <div className="flex space-x-1">
          {["PCA", "UMAP", "t-SNE"].map((m) => (
            <button
              key={m}
              onClick={() => setMethod(m)}
              className={`px-2.5 py-1 text-xs rounded font-semibold transition ${
                method === m
                  ? "bg-cyan-600 text-white"
                  : "bg-slate-800 text-slate-400 hover:text-slate-200"
              }`}
            >
              {m}
            </button>
          ))}
        </div>
      </div>

      <div className="relative h-64 bg-slate-950 rounded border border-slate-800 p-4 mb-4 flex items-center justify-center">
        {samplePoints.map((pt, idx) => (
          <div
            key={idx}
            style={{ left: `${pt.x}%`, top: `${pt.y}%`, backgroundColor: pt.color }}
            className="absolute w-4 h-4 rounded-full shadow-lg shadow-cyan-500/20 cursor-pointer transform -translate-x-1/2 -translate-y-1/2 group"
          >
            <span className="absolute left-5 top-0 text-[10px] whitespace-nowrap bg-slate-900 px-1.5 py-0.5 rounded border border-slate-700 text-slate-200 hidden group-hover:block z-10">
              {pt.label}
            </span>
          </div>
        ))}
        <span className="text-xs text-slate-600 font-mono">
          Projection: {method} (Component 1 vs Component 2)
        </span>
      </div>
    </div>
  );
}
