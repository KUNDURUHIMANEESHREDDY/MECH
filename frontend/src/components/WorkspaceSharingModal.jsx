import React, { useState } from "react";
import { collaborationManager } from "../services/collaborationManager";

export default function WorkspaceSharingModal({ isOpen, onClose }) {
  const [shareInfo, setShareInfo] = useState(null);

  if (!isOpen) return null;

  const handleShare = () => {
    const info = collaborationManager.shareWorkspace("Shared GPT-2 Analysis");
    setShareInfo(info);
  };

  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-slate-900 border border-slate-700 rounded-xl max-w-md w-full p-5 text-slate-100 shadow-2xl">
        <div className="flex justify-between items-center mb-4 pb-2 border-b border-slate-800">
          <h3 className="text-base font-bold text-teal-400">🤝 Collaborative Workspace Sharing</h3>
          <button onClick={onClose} className="text-slate-400 hover:text-slate-200 text-sm">
            ✕
          </button>
        </div>

        <p className="text-xs text-slate-400 mb-4">
          Export full workspace manifest, experiment history, active layouts, and artifacts for peer research collaboration.
        </p>

        <button
          onClick={handleShare}
          className="w-full py-2 bg-teal-600 hover:bg-teal-500 text-white font-semibold text-xs rounded transition mb-4"
        >
          Generate Shareable Workspace URL & Manifest
        </button>

        {shareInfo && (
          <div className="p-3 bg-slate-950 rounded border border-slate-800">
            <span className="text-[10px] text-teal-400 font-bold block mb-1">
              ✓ Shared Workspace Link Ready
            </span>
            <input
              type="text"
              readOnly
              value={shareInfo.shareUrl}
              className="w-full px-2 py-1 bg-slate-900 border border-slate-700 rounded text-xs text-teal-300 font-mono"
            />
          </div>
        )}
      </div>
    </div>
  );
}
