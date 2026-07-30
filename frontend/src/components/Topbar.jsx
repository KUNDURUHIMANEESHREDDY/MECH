import React from 'react';

export default function Topbar({
  crumb,
  pythonStatus,
  onToggleDock,
  onToggleCmdPalette,
  onOpenShare,
  onOpenMarketplace,
  onOpenPublication,
  onOpenDiscoveryMemory,
}) {
  const statusLabel = pythonStatus === 'connected'
    ? 'Python connected'
    : pythonStatus === 'connecting'
      ? 'Connecting…'
      : 'Python offline';

  return (
    <header className="topbar">
      <div className="crumb" style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
        <span>{crumb}</span>
        <span style={{ fontSize: '0.75rem', opacity: 0.5 }}>|</span>
        <button
          onClick={onOpenDiscoveryMemory}
          className="btn btn-primary btn-sm"
          style={{ fontSize: '0.75rem', padding: '0.2rem 0.6rem', background: '#2a2a5a', border: '1px solid #5cd4c4', color: '#5cd4c4' }}
        >
          🧠 Discovery Memory
        </button>
        <button
          onClick={onToggleDock}
          className="btn btn-secondary btn-sm"
          style={{ fontSize: '0.75rem', padding: '0.2rem 0.5rem' }}
        >
          ⚡ Dock Panels
        </button>
        <button
          onClick={onOpenShare}
          className="btn btn-secondary btn-sm"
          style={{ fontSize: '0.75rem', padding: '0.2rem 0.5rem' }}
        >
          🤝 Share
        </button>
        <button
          onClick={onOpenMarketplace}
          className="btn btn-secondary btn-sm"
          style={{ fontSize: '0.75rem', padding: '0.2rem 0.5rem' }}
        >
          🧩 Marketplace
        </button>
        <button
          onClick={onOpenPublication}
          className="btn btn-secondary btn-sm"
          style={{ fontSize: '0.75rem', padding: '0.2rem 0.5rem' }}
        >
          📄 Export
        </button>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
        <button
          onClick={onToggleCmdPalette}
          className="btn btn-secondary btn-sm"
          style={{ fontSize: '0.75rem', padding: '0.2rem 0.5rem', fontFamily: 'monospace' }}
        >
          Ctrl+Shift+P
        </button>
        <div className="status" data-testid="python-status" data-status={pythonStatus}>
          <span className={'dot ' + (pythonStatus === 'connected' ? 'connected' : pythonStatus === 'connecting' ? 'connecting' : '')} />
          <span>{statusLabel}</span>
        </div>
      </div>
    </header>
  );
}
