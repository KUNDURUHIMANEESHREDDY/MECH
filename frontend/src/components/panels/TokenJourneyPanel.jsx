import React from "react";
import { selectionManager } from "../../utils/selectionManager";

export default function TokenJourneyPanel() {
  const tokens = ["The", "capital", "of", "France", "is"];

  const handleSelectTokenLayer = (token, layer) => {
    selectionManager.selectToken(tokens.indexOf(token), token);
  };

  return (
    <div className="p-4 bg-slate-900 text-slate-100 h-full overflow-y-auto">
      <div className="mb-4 pb-2 border-b border-slate-800">
        <h2 className="text-lg font-bold text-teal-400">🚀 Token Representation Journey</h2>
        <p className="text-xs text-slate-400">Layer-by-Layer Representation Trajectory & Cosine Distance</p>
      </div>

      <div className="space-y-3">
        {tokens.map((tok, idx) => (
          <div key={idx} className="p-3 bg-slate-800/80 rounded border border-slate-700">
            <div className="flex justify-between items-center mb-2">
              <span className="text-sm font-bold text-teal-300">
                Token [{idx}]: "{tok}"
              </span>
              <span className="text-[10px] text-slate-400 font-mono">Embedding Dim: 768</span>
            </div>
            <div className="flex space-x-1">
              {Array.from({ length: 12 }).map((_, l) => (
                <button
                  key={l}
                  onClick={() => handleSelectTokenLayer(tok, l)}
                  className="flex-1 py-1 bg-slate-700 hover:bg-teal-600/50 text-[10px] rounded text-slate-300 font-mono transition"
                >
                  L{l}
                </button>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
