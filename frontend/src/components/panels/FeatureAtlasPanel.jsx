import React, { useState } from "react";
import { Map } from "lucide-react";

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
    <div style={{ padding: 16, height: "100%", overflowY: "auto", color: "var(--text)" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16, paddingBottom: 8, borderBottom: "1px solid var(--border)" }}>
        <div>
          <h2 style={{ fontSize: 18, fontWeight: 700, color: "var(--warning)", margin: 0 }}>
            <Map size={14} style={{ verticalAlign: "middle", marginRight: 6 }} /> SAE Feature Atlas
          </h2>
          <p className="hint" style={{ margin: "2px 0 0 0" }}>Searchable Catalog of 16,384 Sparse Autoencoder Features</p>
        </div>
        <input
          type="text"
          placeholder="Filter features..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          style={{ padding: "4px 12px", background: "var(--bg-elev-2)", border: "1px solid var(--border)", borderRadius: 6, fontSize: 12, color: "var(--text)", width: 192, outline: "none" }}
        />
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))", gap: 12 }}>
        {filtered.map((feature) => (
          <div
            key={feature.id}
            style={{ padding: 12, background: "var(--bg-elev-2)", borderRadius: 8, border: "1px solid var(--border)", transition: "border-color 0.2s" }}
          >
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 4 }}>
              <span style={{ fontSize: 12, fontWeight: 700, color: "var(--warning)" }}>Feature #{feature.id}</span>
              <span style={{ padding: "2px 8px", background: "rgba(249, 226, 175, 0.12)", color: "var(--warning)", fontSize: 10, borderRadius: 4, fontFamily: "var(--font-mono)" }}>
                {feature.cluster}
              </span>
            </div>
            <h3 style={{ fontSize: 14, fontWeight: 600, color: "var(--text)", marginBottom: 8 }}>{feature.label}</h3>
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, color: "var(--text-muted)", paddingTop: 8, borderTop: "1px solid var(--border)" }}>
              <span>Firing Freq: {feature.freq}</span>
              <span style={{ color: "var(--success)", fontWeight: 500 }}>Interpretability: {feature.score}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
