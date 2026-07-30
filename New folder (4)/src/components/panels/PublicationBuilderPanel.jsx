import React, { useState } from "react";
import { publicationPaperBuilder } from "../../services/publicationPaperBuilder";

export default function PublicationBuilderPanel() {
  const [paper, setPaper] = useState(publicationPaperBuilder.compilePaper());

  const handleRecompile = () => {
    setPaper(publicationPaperBuilder.compilePaper("Updated Mechanistic Paper"));
  };

  return (
    <div className="p-4 bg-slate-900 text-slate-100 h-full overflow-y-auto">
      <div className="flex justify-between items-center mb-4 pb-2 border-b border-slate-800">
        <div>
          <h2 className="text-lg font-bold text-fuchsia-400">📜 Publication Manuscript Builder</h2>
          <p className="text-xs text-slate-400">Automated LaTeX / Markdown Paper Generator</p>
        </div>
        <button
          onClick={handleRecompile}
          className="px-3 py-1 bg-fuchsia-600 hover:bg-fuchsia-500 text-white text-xs font-semibold rounded"
        >
          Recompile Paper ⚡
        </button>
      </div>

      <div className="p-4 bg-slate-950 rounded border border-slate-800 font-mono text-xs text-slate-300 whitespace-pre-wrap">
        {paper.manuscriptMarkdown}
      </div>
    </div>
  );
}
