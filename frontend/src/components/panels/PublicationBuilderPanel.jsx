import React, { useState } from "react";
import { ScrollText, Zap } from "lucide-react";
import { publicationPaperBuilder } from "../../services/publicationPaperBuilder";

export default function PublicationBuilderPanel() {
  const [paper, setPaper] = useState(publicationPaperBuilder.compilePaper());

  const handleRecompile = () => {
    setPaper(publicationPaperBuilder.compilePaper("Updated Mechanistic Paper"));
  };

  return (
    <div style={{ padding: 16, height: "100%", overflowY: "auto", color: "var(--text)" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16, paddingBottom: 8, borderBottom: "1px solid var(--border)" }}>
        <div>
          <h2 style={{ fontSize: 18, fontWeight: 700, color: "var(--accent)", margin: 0 }}>
            <ScrollText size={14} style={{ verticalAlign: "middle", marginRight: 6 }} /> Publication Manuscript Builder
          </h2>
          <p className="hint" style={{ margin: "2px 0 0 0" }}>Automated LaTeX / Markdown Paper Generator</p>
        </div>
        <button
          onClick={handleRecompile}
          className="btn btn-primary"
          style={{ fontSize: 12 }}
        >
          Recompile Paper <Zap size={12} style={{ verticalAlign: "middle", marginLeft: 4 }} />
        </button>
      </div>

      <div style={{ padding: 16, background: "var(--bg)", borderRadius: 8, border: "1px solid var(--border)", fontFamily: "var(--font-mono)", fontSize: 12, color: "var(--text-dim)", whiteSpace: "pre-wrap" }}>
        {paper.manuscriptMarkdown}
      </div>
    </div>
  );
}
