import React, { useState } from 'react';

export const ExtensionMarketplaceModal = ({
  isOpen = true,
  onClose,
}) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [installedMap, setInstalledMap] = useState({
    'sae-inspector': true,
    'circuit-tracer': true,
  });

  const extensions = [
    {
      id: 'sae-inspector',
      name: 'SAE Visual Feature Inspector',
      author: 'MECH Core Team',
      version: '1.4.0',
      description: 'Interactive high-dimensional feature cluster visualizer using UMAP and t-SNE projections.',
      downloads: '14.2k',
    },
    {
      id: 'circuit-tracer',
      name: 'Automated ACDC Circuit Discovery',
      author: 'Oxford AI Alignment',
      version: '2.1.0',
      description: 'Heuristic prune search algorithm for discovering minimal causal computational subgraphs.',
      downloads: '8.9k',
    },
    {
      id: 'hpc-slurm-driver',
      name: 'Slurm & Kubernetes Cluster Orchestrator',
      author: 'Supercomputing Labs',
      version: '1.0.2',
      description: 'Offload massive batch attribution jobs to Slurm partitions and GPU Kubernetes clusters.',
      downloads: '5.1k',
    },
    {
      id: 'latex-paper-gen',
      name: 'Auto-LaTeX Publication Formatter',
      author: 'DeepMind Research',
      version: '0.9.8',
      description: 'Compiles mechanistic discoveries into peer-review ready NeurIPS, ICLR, and ICML camera-ready PDFs.',
      downloads: '12.6k',
    },
  ];

  if (!isOpen) return null;

  const toggleInstall = (id) => {
    setInstalledMap((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  const filtered = extensions.filter((ext) =>
    ext.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    ext.description.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div className="modal-backdrop" data-testid="extension-marketplace-modal">
      <div className="modal-container" style={{ maxWidth: '780px', background: '#18181b', color: '#f4f4f5', padding: '24px', borderRadius: '10px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
          <h2 style={{ margin: 0, fontSize: '18px', fontWeight: 600 }}>Scientific Extension Marketplace</h2>
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

        <div style={{ marginBottom: '16px' }}>
          <input
            data-testid="marketplace-search-input"
            type="text"
            placeholder="Search extensions, visualizers, algorithms, and drivers..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            style={{ width: '100%', boxSizing: 'border-box', padding: '8px 12px', background: '#27272a', border: '1px solid #3f3f46', borderRadius: '6px', color: '#fff' }}
          />
        </div>

        <div style={{ maxHeight: '380px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '12px' }}>
          {filtered.map((ext) => {
            const isInstalled = !!installedMap[ext.id];
            return (
              <div
                key={ext.id}
                data-testid={`extension-item-${ext.id}`}
                style={{ padding: '14px', background: '#27272a', borderRadius: '8px', border: '1px solid #3f3f46', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}
              >
                <div style={{ flex: 1, paddingRight: '16px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
                    <strong style={{ fontSize: '15px' }}>{ext.name}</strong>
                    <span style={{ fontSize: '12px', color: '#71717a' }}>v{ext.version}</span>
                  </div>
                  <p style={{ margin: '0 0 6px 0', fontSize: '13px', color: '#d4d4d8' }}>{ext.description}</p>
                  <div style={{ fontSize: '11px', color: '#a1a1aa' }}>By {ext.author} • {ext.downloads} installs</div>
                </div>
                <button
                  data-testid={`extension-action-btn-${ext.id}`}
                  onClick={() => toggleInstall(ext.id)}
                  style={{
                    padding: '6px 14px',
                    borderRadius: '6px',
                    border: 'none',
                    fontWeight: 600,
                    fontSize: '12px',
                    cursor: 'pointer',
                    background: isInstalled ? '#3f3f46' : '#10b981',
                    color: isInstalled ? '#e4e4e7' : '#ffffff',
                  }}
                >
                  {isInstalled ? 'Uninstall' : 'Install'}
                </button>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};
