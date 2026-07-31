import React, { useState } from "react";
import { Globe } from "lucide-react";

export default function EmbeddingViewerPanel() {
  const [method, setMethod] = useState("PCA");

  const samplePoints = [
    { x: 12, y: 45, label: "IOI Features", color: "#38bdf8" },
    { x: 78, y: 22, label: "Geography Probes", color: "#f59e0b" },
    { x: 45, y: 88, label: "Syntax Patterns", color: "#10b981" },
    { x: 82, y: 75, label: "Code Decoders", color: "#ec4899" },
  ];

  return (
    <div style={{ padding: 16, height: "100%", overflowY: "auto", color: "var(--text)" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16, paddingBottom: 8, borderBottom: "1px solid var(--border)" }}>
        <div>
          <h2 style={{ fontSize: 18, fontWeight: 700, color: "var(--accent)", margin: 0 }}>
            <Globe size={14} style={{ verticalAlign: "middle", marginRight: 6 }} /> 3D Embedding Manifold Viewer
          </h2>
          <p className="hint" style={{ margin: "2px 0 0 0" }}>Dimensionality Reduction (PCA / UMAP)</p>
        </div>
        <div style={{ display: "flex", gap: 4 }}>
          {["PCA", "UMAP", "t-SNE"].map((m) => (
            <button
              key={m}
              onClick={() => setMethod(m)}
              className="btn"
              style={{
                padding: "4px 10px",
                fontSize: 12,
                borderRadius: 6,
                fontWeight: 600,
                transition: "all 0.2s",
                background: method === m ? "var(--accent-strong)" : "var(--bg-elev-2)",
                color: method === m ? "var(--text)" : "var(--text-dim)",
              }}
            >
              {m}
            </button>
          ))}
        </div>
      </div>

      <div style={{ position: "relative", height: 256, background: "var(--bg)", borderRadius: 8, border: "1px solid var(--border)", padding: 16, marginBottom: 16, display: "flex", alignItems: "center", justifyContent: "center" }}>
        {samplePoints.map((pt, idx) => (
          <div
            key={idx}
            style={{ left: `${pt.x}%`, top: `${pt.y}%`, backgroundColor: pt.color, position: "absolute", width: 16, height: 16, borderRadius: "50%", boxShadow: "0 2px 8px var(--accent-soft)", cursor: "pointer", transform: "translate(-50%, -50%)" }}
          >
            <span style={{ position: "absolute", left: 20, top: 0, fontSize: 10, whiteSpace: "nowrap", background: "var(--bg)", padding: "2px 6px", borderRadius: 4, border: "1px solid var(--border)", color: "var(--text)", zIndex: 10, display: "none" }}>
              {pt.label}
            </span>
          </div>
        ))}
        <span style={{ fontSize: 12, color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
          Projection: {method} (Component 1 vs Component 2)
        </span>
      </div>
    </div>
  );
}
