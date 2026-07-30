import React, { useState } from "react";

/**
 * ResearchNotebook
 * 
 * A block-based lite editor mixing Jupyter execution, Figma design notes, 
 * and W&B-style experiment tracking.
 */
export default function ResearchNotebook() {
  const [blocks, setBlocks] = useState([
    { id: "1", type: "markdown", content: "# Induction Heads Analysis\nInvestigating layer 5 attention patterns." },
    { id: "2", type: "code", content: "circuit = extract_circuit(model, 'induction_heads')\nprint(circuit.metrics)" },
    { id: "3", type: "visualization", content: "Attention Heatmap" }
  ]);

  return (
    <div className="p-8 max-w-4xl mx-auto">
      <header className="mb-8 flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-light text-slate-800">Notebook: L5 Analysis</h1>
          <p className="text-sm text-slate-400 mt-1">Last edited 2 mins ago</p>
        </div>
        <div className="flex gap-2">
          <button className="px-4 py-2 bg-slate-100 text-slate-700 rounded shadow-sm hover:bg-slate-200">Share</button>
          <button className="px-4 py-2 bg-indigo-600 text-white rounded shadow-sm hover:bg-indigo-700">Export</button>
        </div>
      </header>

      <div className="space-y-6">
        {blocks.map(block => (
          <div key={block.id} className="group relative">
            <div className="absolute -left-12 top-4 opacity-0 group-hover:opacity-100 transition">
              <button className="text-slate-300 hover:text-slate-600">☰</button>
            </div>
            
            {block.type === "markdown" && (
              <div className="p-4 border border-transparent hover:border-slate-200 rounded prose max-w-none text-slate-800">
                {block.content}
              </div>
            )}
            
            {block.type === "code" && (
              <div className="border border-slate-200 rounded overflow-hidden">
                <div className="bg-slate-50 px-4 py-2 text-xs text-slate-500 font-mono flex justify-between">
                  <span>Python (Interpretability SDK)</span>
                  <button className="text-indigo-600 hover:text-indigo-800">▶ Run</button>
                </div>
                <div className="p-4 bg-slate-900 text-emerald-400 font-mono text-sm whitespace-pre-wrap">
                  {block.content}
                </div>
              </div>
            )}

            {block.type === "visualization" && (
              <div className="border border-slate-200 rounded p-6 bg-slate-50 flex items-center justify-center min-h-[200px]">
                <div className="text-center text-slate-500">
                  <p className="font-medium mb-2">Interactive {block.content}</p>
                  <p className="text-xs">Widget embedded via Unified Registry</p>
                </div>
              </div>
            )}
          </div>
        ))}

        <div className="pt-8 text-center opacity-50 hover:opacity-100 transition">
          <span className="text-slate-400 text-sm cursor-pointer border border-dashed border-slate-300 rounded px-4 py-2">
            + Add Block
          </span>
        </div>
      </div>
    </div>
  );
}
