import React from "react";
import { Rocket } from "lucide-react";
import { selectionManager } from "../../utils/selectionManager";

export default function TokenJourneyPanel() {
  const tokens = ["The", "capital", "of", "France", "is"];

  const handleSelectTokenLayer = (token, layer) => {
    selectionManager.selectToken(tokens.indexOf(token), token);
  };

  return (
    <div style={{ padding: 16, height: "100%", overflowY: "auto", color: "var(--text)" }}>
      <div style={{ marginBottom: 16, paddingBottom: 8, borderBottom: "1px solid var(--border)" }}>
        <h2 style={{ fontSize: 18, fontWeight: 700, color: "var(--accent)", margin: 0 }}>
          <Rocket size={14} style={{ verticalAlign: "middle", marginRight: 6 }} /> Token Representation Journey
        </h2>
        <p className="hint" style={{ margin: "2px 0 0 0" }}>Layer-by-Layer Representation Trajectory & Cosine Distance</p>
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
        {tokens.map((tok, idx) => (
          <div key={idx} style={{ padding: 12, background: "var(--bg-elev-2)", borderRadius: 8, border: "1px solid var(--border)" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
              <span style={{ fontSize: 14, fontWeight: 700, color: "var(--accent-hover)" }}>
                Token [{idx}]: "{tok}"
              </span>
              <span style={{ fontSize: 10, color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>Embedding Dim: 768</span>
            </div>
            <div style={{ display: "flex", gap: 4 }}>
              {Array.from({ length: 12 }).map((_, l) => (
                <button
                  key={l}
                  onClick={() => handleSelectTokenLayer(tok, l)}
                  style={{ flex: 1, padding: "4px 0", background: "var(--bg-elev)", fontSize: 10, borderRadius: 4, color: "var(--text-dim)", fontFamily: "var(--font-mono)", transition: "background 0.2s" }}
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
