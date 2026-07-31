import React from 'react';
import { Brain, PanelsTopLeft, Share2, Store, FileUp, Command, CircleDot } from 'lucide-react';

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
    ? 'Connected'
    : pythonStatus === 'connecting'
      ? '\u2026'
      : 'Offline';

  return (
    <header className="topbar">
      <div className="crumb-group">
        <span className="crumb">{crumb}</span>
      </div>

      <div className="topbar-actions">
        <button onClick={onOpenDiscoveryMemory} className="btn btn-ghost btn-sm" title="Discovery Memory">
          <Brain size={14} /> Mem
        </button>
        <button onClick={onToggleDock} className="btn btn-ghost btn-sm" title="Dock Panels">
          <PanelsTopLeft size={14} /> Dock
        </button>
        <button onClick={onOpenShare} className="btn btn-ghost btn-sm" title="Share">
          <Share2 size={14} /> Share
        </button>
        <button onClick={onOpenMarketplace} className="btn btn-ghost btn-sm" title="Marketplace">
          <Store size={14} /> Market
        </button>
        <button onClick={onOpenPublication} className="btn btn-ghost btn-sm" title="Export">
          <FileUp size={14} /> Export
        </button>
        <button onClick={onToggleCmdPalette} className="btn btn-ghost btn-sm" title="Command Palette">
          <Command size={13} /> P
        </button>
        <div className="status" data-testid="python-status" data-status={pythonStatus}>
          <span className={'dot ' + (pythonStatus === 'connected' ? 'connected' : pythonStatus === 'connecting' ? 'connecting' : '')} />
          <span>{statusLabel}</span>
        </div>
      </div>
    </header>
  );
}
