import React, { useState } from "react";
import { X } from "lucide-react";
import { collaborationManager } from "../services/collaborationManager";

export default function WorkspaceSharingModal({ isOpen, onClose }) {
  const [shareInfo, setShareInfo] = useState(null);

  if (!isOpen) return null;

  const handleShare = () => {
    const info = collaborationManager.shareWorkspace("Shared GPT-2 Analysis");
    setShareInfo(info);
  };

  return (
    <div className="modal-overlay">
      <div className="modal-container modal-sm">
        <div className="modal-header">
          <h2>Collaborative Workspace Sharing</h2>
          <button onClick={onClose} className="modal-close-btn"><X size={14} /></button>
        </div>

        <div className="card">
          <p className="hint">Share your current workspace with collaborators via a unique share link.</p>
          {!shareInfo ? (
            <button className="btn" onClick={handleShare}>Generate Share Link</button>
          ) : (
            <div>
              <div className="field">
                <label>Share URL</label>
                <input className="input-text" value={shareInfo.shareUrl} readOnly />
              </div>
              <div className="field">
                <label>Shared At</label>
                <input className="input-text" value={shareInfo.sharedAt} readOnly />
              </div>
            </div>
          )}
        </div>

        <div className="modal-footer">
          <span />
          <button onClick={onClose} className="btn">Close</button>
        </div>
      </div>
    </div>
  );
}
