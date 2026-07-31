import React, { useState, useEffect } from "react";
import { Waves, Play, Pause } from "lucide-react";

export default function CausalTraceViewerPanel() {
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentLayer, setCurrentLayer] = useState(0);

  useEffect(() => {
    let timer;
    if (isPlaying) {
      timer = setInterval(() => {
        setCurrentLayer((prev) => (prev + 1) % 12);
      }, 500);
    }
    return () => clearInterval(timer);
  }, [isPlaying]);

  return (
    <div style={{ padding: 16, height: "100%", overflowY: "auto", color: "var(--text)" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16, paddingBottom: 8, borderBottom: "1px solid var(--border)" }}>
        <div>
          <h2 style={{ fontSize: 18, fontWeight: 700, color: "var(--accent)", margin: 0 }}>
            <Waves size={14} style={{ verticalAlign: "middle", marginRight: 6 }} /> Causal Trace Flow Viewer
          </h2>
          <p className="hint" style={{ margin: "2px 0 0 0" }}>Layer-by-Layer Activation Recovery Curve</p>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <button
            onClick={() => setIsPlaying(!isPlaying)}
            className="btn btn-primary"
            style={{ fontSize: 12 }}
          >
            {isPlaying ? (
              <>
                <Pause size={12} style={{ verticalAlign: "middle", marginRight: 4 }} /> Pause
              </>
            ) : (
              <>
                <Play size={12} style={{ verticalAlign: "middle", marginRight: 4 }} /> Play Flow
              </>
            )}
          </button>
        </div>
      </div>

      <div style={{ display: "flex", gap: 4, alignItems: "flex-end", height: 160, background: "var(--bg)", padding: 16, borderRadius: 8, border: "1px solid var(--border)", marginBottom: 16 }}>
        {Array.from({ length: 12 }).map((_, l) => {
          const isCurrent = l === currentLayer;
          const heightPct = 20 + ((l * 7 + (l === 8 ? 60 : 0)) % 80);
          return (
            <div
              key={l}
              onClick={() => setCurrentLayer(l)}
              style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", cursor: "pointer" }}
            >
              <div
                style={{
                  height: `${heightPct}%`,
                  width: "100%",
                  borderRadius: "4px 4px 0 0",
                  transition: "all 0.2s",
                  background: isCurrent ? "var(--accent-strong)" : "var(--bg-elev-2)",
                  boxShadow: isCurrent ? "0 0 12px var(--accent-soft)" : "none",
                }}
              />
              <span style={{ fontSize: 10, marginTop: 8, color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>L{l}</span>
            </div>
          );
        })}
      </div>

      <div style={{ padding: 12, background: "var(--bg-elev-2)", borderRadius: 8, border: "1px solid var(--border)", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div>
          <span style={{ fontSize: 12, color: "var(--text-muted)", display: "block" }}>Active Inspection Layer</span>
          <span style={{ fontSize: 14, fontWeight: 700, color: "var(--accent)" }}>Layer {currentLayer}</span>
        </div>
        <div style={{ textAlign: "right" }}>
          <span style={{ fontSize: 12, color: "var(--text-muted)", display: "block" }}>Indirect Causal Score</span>
          <span style={{ fontSize: 14, fontWeight: 700, color: "var(--success)" }}>
            {currentLayer === 8 ? "0.90 (Max Peak)" : `${(0.12 + currentLayer * 0.04).toFixed(2)}`}
          </span>
        </div>
      </div>
    </div>
  );
}
