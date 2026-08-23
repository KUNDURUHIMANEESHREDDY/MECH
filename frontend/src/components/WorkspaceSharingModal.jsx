import React, { useState } from 'react';

export const WorkspaceSharingModal = ({
  isOpen = true,
  onClose,
}) => {
  const [accessLevel, setAccessLevel] = useState('view');
  const [signedBundle, setSignedBundle] = useState(true);
  const [shareLink, setShareLink] = useState('');
  const [copied, setCopied] = useState(false);

  if (!isOpen) return null;

  const handleGenerateLink = () => {
    const link = `mech://collab/workspace-session?id=ws_${Date.now()}&perm=${accessLevel}&signed=${signedBundle}`;
    setShareLink(link);
    setCopied(false);
  };

  const handleCopy = () => {
    if (shareLink) {
      navigator.clipboard?.writeText(shareLink);
      setCopied(true);
      setTimeout(() => setCopied(false), 2500);
    }
  };

  return (
    <div className="modal-backdrop" data-testid="workspace-sharing-modal">
      <div className="modal-container" style={{ maxWidth: '620px', background: '#18181b', color: '#f4f4f5', padding: '24px', borderRadius: '10px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
          <h2 style={{ margin: 0, fontSize: '18px', fontWeight: 600 }}>Share Research Workspace</h2>
          {onClose && (
            <button
              onClick={onClose}
              data-testid="modal-close-btn"
              style={{ background: 'transparent', border: 'none', color: '#a1a1aa', cursor: 'pointer', fontSize: '16px' }}
            >
              ✕
            </button>
          )}
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '14px', marginBottom: '18px' }}>
          <div>
            <label style={{ display: 'block', fontSize: '13px', marginBottom: '4px', color: '#a1a1aa' }}>Collaborator Access Rights</label>
            <select
              data-testid="access-rights-select"
              value={accessLevel}
              onChange={(e) => setAccessLevel(e.target.value)}
              style={{ width: '100%', padding: '8px 12px', background: '#27272a', border: '1px solid #3f3f46', borderRadius: '6px', color: '#fff' }}
            >
              <option value="view">Read-Only (Viewer / Reviewer)</option>
              <option value="edit">Interactive Editor (Run & Patch Activations)</option>
              <option value="admin">Full Administrator (Modify Weights & Schedulers)</option>
            </select>
          </div>

          <label style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '13px', cursor: 'pointer' }}>
            <input
              type="checkbox"
              checked={signedBundle}
              onChange={(e) => setSignedBundle(e.target.checked)}
            />
            <span>Cryptographically sign research artifacts with local developer key</span>
          </label>

          {shareLink && (
            <div>
              <label style={{ display: 'block', fontSize: '13px', marginBottom: '4px', color: '#10b981' }}>Active Session Link</label>
              <div style={{ display: 'flex', gap: '8px' }}>
                <input
                  data-testid="share-link-input"
                  type="text"
                  readOnly
                  value={shareLink}
                  style={{ flex: 1, padding: '8px 12px', background: '#27272a', border: '1px solid #3f3f46', borderRadius: '6px', color: '#a1a1aa', fontSize: '12px' }}
                />
                <button
                  data-testid="copy-link-btn"
                  onClick={handleCopy}
                  style={{ padding: '8px 14px', borderRadius: '6px', border: 'none', background: '#3f3f46', color: '#fff', cursor: 'pointer', fontSize: '12px' }}
                >
                  {copied ? 'Copied!' : 'Copy'}
                </button>
              </div>
            </div>
          )}
        </div>

        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '10px' }}>
          <button
            data-testid="generate-share-link-btn"
            onClick={handleGenerateLink}
            style={{
              padding: '8px 16px',
              borderRadius: '6px',
              border: 'none',
              fontWeight: 600,
              background: '#10b981',
              color: '#fff',
              cursor: 'pointer',
            }}
          >
            Generate Collaboration Token
          </button>
        </div>
      </div>
    </div>
  );
};
