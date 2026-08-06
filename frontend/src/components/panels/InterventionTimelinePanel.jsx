import React from "react";
import { Timer, ArrowRight } from "lucide-react";

export default function InterventionTimelinePanel() {
  return (
    <div style={{ padding: 16, height: "100%", overflowY: "auto", color: "var(--text)" }}>
      <div style={{ marginBottom: 16, paddingBottom: 8, borderBottom: "1px solid var(--border)" }}>
        <h2 style={{ fontSize: 18, fontWeight: 700, color: "var(--danger)", margin: 0 }}>
          <Timer size={14} style={{ verticalAlign: "middle", marginRight: 6 }} /> Intervention Timeline & Diff Viewer
        </h2>
        <p className="hint" style={{ margin: "2px 0 0 0" }}>Track Original State {'→'} Patch Applied {'→'} Output Prediction Change</p>
      </div>

      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12, marginBottom: 24 }}>
        <div style={{ flex: 1, padding: 12, background: "var(--bg-elev-2)", borderRadius: 8, border: "1px solid var(--border)" }}>
          <span style={{ fontSize: 10, textTransform: "uppercase", fontWeight: 700, color: "var(--text-muted)", display: "block", marginBottom: 4 }}>1. Original State</span>
          <p style={{ fontSize: 12, color: "var(--text)", fontWeight: 500, margin: 0 }}>Prompt: "John gave a book to Mary"</p>
          <span style={{ fontSize: 12, color: "var(--success)", display: "block", marginTop: 4 }}>Top Token: " Mary" (84.2%)</span>
        </div>

        <div style={{ color: "var(--danger)", display: "flex", alignItems: "center" }}><ArrowRight size={18} /></div>

        <div style={{ flex: 1, padding: 12, background: "rgba(243, 139, 168, 0.08)", borderRadius: 8, border: "1px solid rgba(243, 139, 168, 0.25)" }}>
          <span style={{ fontSize: 10, textTransform: "uppercase", fontWeight: 700, color: "var(--danger)", display: "block", marginBottom: 4 }}>2. Activation Patch</span>
          <p style={{ fontSize: 12, color: "var(--text)", fontWeight: 500, margin: 0 }}>Layer 8 MLP N402 = 0.0 (Zero Out)</p>
          <span style={{ fontSize: 12, color: "var(--danger)", display: "block", marginTop: 4 }}>Intervention: Replacement</span>
        </div>

        <div style={{ color: "var(--danger)", display: "flex", alignItems: "center" }}><ArrowRight size={18} /></div>

        <div style={{ flex: 1, padding: 12, background: "var(--bg-elev-2)", borderRadius: 8, border: "1px solid var(--border)" }}>
          <span style={{ fontSize: 10, textTransform: "uppercase", fontWeight: 700, color: "var(--text-muted)", display: "block", marginBottom: 4 }}>3. Changed Prediction</span>
          <p style={{ fontSize: 12, color: "var(--text)", fontWeight: 500, margin: 0 }}>Prompt: "John gave a book to Mary"</p>
          <span style={{ fontSize: 12, color: "var(--warning)", display: "block", marginTop: 4 }}>Top Token: " John" (42.1%)</span>
        </div>
      </div>

      <div style={{ padding: 12, background: "var(--bg-elev-2)", borderRadius: 8, border: "1px solid var(--border)" }}>
        <h3 style={{ fontSize: 12, fontWeight: 700, color: "var(--text-dim)", marginBottom: 8 }}>Causal Logit Shift Delta</h3>
        <div style={{ width: "100%", background: "var(--bg-elev)", height: 8, borderRadius: 4, overflow: "hidden", display: "flex" }}>
          <div style={{ width: "65%", height: "100%", background: "var(--success)" }} />
          <div style={{ width: "35%", height: "100%", background: "var(--danger)" }} />
        </div>
        <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, color: "var(--text-muted)", marginTop: 4 }}>
          <span>Mary: -4.2 logits</span>
          <span>John: +3.8 logits</span>
        </div>
      </div>
    </div>
  );
}
