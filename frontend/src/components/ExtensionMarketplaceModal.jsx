import React, { useState } from "react";
import { X } from "lucide-react";
import { extensionMarketplaceService } from "../services/extensionMarketplaceService";

export default function ExtensionMarketplaceModal({ isOpen, onClose }) {
  const [plugins, setPlugins] = useState(extensionMarketplaceService.listAvailablePlugins());
  const [installed, setInstalled] = useState(extensionMarketplaceService.listInstalledPlugins());

  if (!isOpen) return null;

  const handleInstall = (pluginId) => {
    extensionMarketplaceService.installPlugin(pluginId);
    setInstalled(extensionMarketplaceService.listInstalledPlugins());
  };

  return (
    <div className="modal-overlay">
      <div className="modal-container">
        <div className="modal-header">
          <h2>Extension Marketplace</h2>
          <button onClick={onClose} className="modal-close-btn"><X size={14} /></button>
        </div>

        <div style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 12 }}>
          {plugins.length === 0 ? (
            <p className="hint">No plugins available.</p>
          ) : (
            plugins.map(p => {
              const isInstalled = installed.some(i => i.id === p.id);
              return (
                <div key={p.id} className="card" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <div>
                    <strong>{p.name}</strong>
                    <p className="hint">{p.description}</p>
                    <span style={{ fontSize: 11, color: 'var(--text-dim)' }}>v{p.version} by {p.author}</span>
                  </div>
                  <button
                    className={'btn' + (isInstalled ? ' btn-secondary' : '')}
                    onClick={() => handleInstall(p.id)}
                    disabled={isInstalled}
                  >
                    {isInstalled ? 'Installed' : 'Install'}
                  </button>
                </div>
              );
            })
          )}
        </div>

        <div className="modal-footer">
          <span>{plugins.length} plugins available</span>
          <button onClick={onClose} className="btn">Close</button>
        </div>
      </div>
    </div>
  );
}
