import React, { useState } from "react";
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
    <div className="p-4 bg-slate-900 text-slate-100 h-full overflow-y-auto">
      <div className="flex justify-between items-center mb-4 pb-2 border-b border-slate-800">
        <div>
          <h2 className="text-lg font-bold text-violet-400">🌿 Experiment Version Control</h2>
          <p className="text-xs text-slate-400">Git-like State Snapshot Commit & Rollback History</p>
        </div>
      </div>

      <div className="flex space-x-2 mb-4">
        <input
          type="text"
          placeholder="Commit message (e.g. Patched L8_N402)..."
          value={msg}
          onChange={(e) => setMsg(e.target.value)}
          className="flex-1 px-3 py-1.5 bg-slate-800 border border-slate-700 rounded text-xs text-slate-200 focus:outline-none focus:border-violet-500"
        />
        <button
          onClick={handleCommit}
          className="px-4 py-1.5 bg-violet-600 hover:bg-violet-500 text-white font-semibold text-xs rounded transition"
        >
          Commit Snapshot
        </button>
      </div>

      <div className="space-y-2">
        {commits.map((c) => (
          <div key={c.id} className="p-3 bg-slate-800/80 rounded border border-slate-700 flex justify-between items-center">
            <div>
              <span className="text-xs font-bold text-violet-300 block">{c.message}</span>
              <span className="text-[10px] text-slate-400 font-mono">
                {c.id} • {new Date(c.timestamp).toLocaleTimeString()}
              </span>
            </div>
            <button
              onClick={() => experimentVersionControl.rollback(c.id)}
              className="px-2.5 py-1 bg-slate-700 hover:bg-violet-900 text-[10px] text-slate-300 font-semibold rounded"
            >
              Rollback
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}
