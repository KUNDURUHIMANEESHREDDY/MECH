import React, { useState } from "react";
import { GitBranch } from "lucide-react";
import { experimentVersionControl } from "../../services/experimentVersionControl";

export default function VersionControlPanel() {
  const [commits, setCommits] = useState(experimentVersionControl.getHistory());
  const [msg, setMsg] = useState("");

  const handleCommit = () => {
    if (!msg.trim()) return;
    experimentVersionControl.commit(msg);
    setCommits(experimentVersionControl.getHistory());
    setMsg("");
  };

  return (
    <div style={{ padding: 16, height: "100%", overflowY: "auto", color: "var(--text)" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16, paddingBottom: 8, borderBottom: "1px solid var(--border)" }}>
        <div>
          <h2 style={{ fontSize: 18, fontWeight: 700, color: "var(--accent)", margin: 0 }}>
            <GitBranch size={14} style={{ verticalAlign: "middle", marginRight: 6 }} /> Experiment Version Control
          </h2>
          <p className="hint" style={{ margin: "2px 0 0 0" }}>Git-like State Snapshot Commit & Rollback History</p>
        </div>
      </div>

      <div style={{ display: "flex", gap: 8, marginBottom: 16 }}>
        <input
          type="text"
          placeholder="Commit message (e.g. Patched L8_N402)..."
          value={msg}
          onChange={(e) => setMsg(e.target.value)}
          style={{ flex: 1, padding: "6px 12px", background: "var(--bg-elev-2)", border: "1px solid var(--border)", borderRadius: 6, fontSize: 12, color: "var(--text)", outline: "none" }}
        />
        <button
          onClick={handleCommit}
          className="btn btn-primary"
          style={{ fontWeight: 600, fontSize: 12 }}
        >
          Commit Snapshot
        </button>
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        {commits.map((c) => (
          <div key={c.id} style={{ padding: 12, background: "var(--bg-elev-2)", borderRadius: 8, border: "1px solid var(--border)", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <div>
              <span style={{ fontSize: 12, fontWeight: 700, color: "var(--accent-hover)", display: "block" }}>{c.message}</span>
              <span style={{ fontSize: 10, color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
                {c.id} • {new Date(c.timestamp).toLocaleTimeString()}
              </span>
            </div>
            <button
              onClick={() => experimentVersionControl.rollback(c.id)}
              className="btn btn-secondary"
              style={{ fontSize: 10, padding: "4px 10px", fontWeight: 600 }}
            >
              Rollback
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}
