import React from "react";
import { artifactManagerService } from "../../services/artifactManagerService";

export default function ArtifactManagerPanel() {
  const artifacts = artifactManagerService.listArtifacts();

  return (
    <div className="p-4 bg-slate-900 text-slate-100 h-full overflow-y-auto">
      <div className="mb-4 pb-2 border-b border-slate-800">
        <h2 className="text-lg font-bold text-emerald-400">📦 Managed Research Artifacts</h2>
        <p className="text-xs text-slate-400">Centralized Store for Figures, Reports, Plots, and Notebooks</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        {artifacts.map((art) => (
          <div key={art.id} className="p-3 bg-slate-800/80 rounded border border-slate-700">
            <div className="flex justify-between items-start mb-1">
              <span className="text-xs font-bold text-emerald-300">{art.title}</span>
              <span className="px-1.5 py-0.5 bg-emerald-500/10 text-emerald-400 text-[10px] rounded font-mono">
                {art.type}
              </span>
            </div>
            <p className="text-[10px] text-slate-400 font-mono mb-2">ID: {art.id}</p>
            <div className="flex justify-between text-[10px] text-slate-400 border-t border-slate-700/50 pt-2">
              <span>Checksum: {art.checksum}</span>
              <span>Creator: {art.creator}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
