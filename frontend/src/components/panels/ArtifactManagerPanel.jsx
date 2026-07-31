import React from "react";
import { Package } from "lucide-react";
import { artifactManagerService } from "../../services/artifactManagerService";

export default function ArtifactManagerPanel() {
  const artifacts = artifactManagerService.listArtifacts();

  return (
    <div style={{ padding: 16, height: "100%", overflowY: "auto", color: "var(--text)" }}>
      <div style={{ marginBottom: 16, paddingBottom: 8, borderBottom: "1px solid var(--border)" }}>
        <h2 style={{ fontSize: 18, fontWeight: 700, color: "var(--success)", margin: 0 }}>
          <Package size={14} style={{ verticalAlign: "middle", marginRight: 6 }} /> Managed Research Artifacts
        </h2>
        <p className="hint" style={{ margin: "2px 0 0 0" }}>Centralized Store for Figures, Reports, Plots, and Notebooks</p>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))", gap: 12 }}>
        {artifacts.map((art) => (
          <div key={art.id} style={{ padding: 12, background: "var(--bg-elev-2)", borderRadius: 8, border: "1px solid var(--border)" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 4 }}>
              <span style={{ fontSize: 12, fontWeight: 700, color: "var(--success)" }}>{art.title}</span>
              <span style={{ padding: "2px 6px", background: "rgba(166, 227, 161, 0.12)", color: "var(--success)", fontSize: 10, borderRadius: 4, fontFamily: "var(--font-mono)" }}>
                {art.type}
              </span>
            </div>
            <p style={{ fontSize: 10, color: "var(--text-muted)", fontFamily: "var(--font-mono)", marginBottom: 8 }}>ID: {art.id}</p>
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, color: "var(--text-muted)", borderTop: "1px solid var(--border)", paddingTop: 8 }}>
              <span>Checksum: {art.checksum}</span>
              <span>Creator: {art.creator}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
