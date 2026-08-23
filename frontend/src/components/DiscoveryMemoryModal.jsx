import React, { useState } from 'react';

export const DiscoveryMemoryModal = ({
  isOpen = true,
  onClose,
}) => {
  const [filter, setFilter] = useState('');
  const [selectedDomain, setSelectedDomain] = useState('ALL');

  const [discoveries] = useState([
    {
      id: 'disc_001',
      title: 'Induction Heads Circuit in Layer 5 & 6',
      domain: 'Induction',
      confidence: 0.96,
      date: '2026-08-10',
      description: 'Identified precise token-matching attention heads attending to previous token positions.',
    },
    {
      id: 'disc_002',
      title: 'Indirect Object Identification (IOI) Name Mover Circuit',
      domain: 'IOI',
      confidence: 0.94,
      date: '2026-08-11',
      description: 'Causal tracing isolated Heads L9H6 and L9H9 transmitting target name logits to final residual.',
    },
    {
      id: 'disc_003',
      title: 'Sparse Autoencoder Monosemantic Feature 402',
      domain: 'SAE',
      confidence: 0.91,
      date: '2026-08-12',
      description: 'Feature activates specifically on geographical capital queries across multilingual prompts.',
    },
    {
      id: 'disc_004',
      title: 'Negative Name Mover Suppression Mechanism',
      domain: 'IOI',
      confidence: 0.88,
      date: '2026-08-13',
      description: 'Inhibition heads suppress duplicate candidate outputs in competitive lexical environments.',
    },
  ]);

  if (!isOpen) return null;

  const filtered = discoveries.filter((d) => {
    const matchesDomain = selectedDomain === 'ALL' || d.domain === selectedDomain;
    const matchesQuery = d.title.toLowerCase().includes(filter.toLowerCase()) ||
      d.description.toLowerCase().includes(filter.toLowerCase());
    return matchesDomain && matchesQuery;
  });

  return (
    <div className="modal-backdrop" data-testid="discovery-memory-modal">
      <div className="modal-container" style={{ maxWidth: '750px', background: '#18181b', color: '#f4f4f5', padding: '24px', borderRadius: '10px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
          <h2 style={{ margin: 0, fontSize: '18px', fontWeight: 600 }}>Scientific Discovery Memory</h2>
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

        <div style={{ display: 'flex', gap: '12px', marginBottom: '16px' }}>
          <input
            data-testid="discovery-search-input"
            type="text"
            placeholder="Search recalled circuits, neurons, and hypotheses..."
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            style={{ flex: 1, padding: '8px 12px', background: '#27272a', border: '1px solid #3f3f46', borderRadius: '6px', color: '#fff' }}
          />
          <select
            data-testid="discovery-domain-select"
            value={selectedDomain}
            onChange={(e) => setSelectedDomain(e.target.value)}
            style={{ padding: '8px 12px', background: '#27272a', border: '1px solid #3f3f46', borderRadius: '6px', color: '#fff' }}
          >
            <option value="ALL">All Domains</option>
            <option value="Induction">Induction</option>
            <option value="IOI">IOI Subcircuits</option>
            <option value="SAE">SAE Features</option>
          </select>
        </div>

        <div style={{ maxHeight: '360px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '12px' }}>
          {filtered.map((d) => (
            <div
              key={d.id}
              data-testid={`discovery-item-${d.id}`}
              style={{ padding: '12px', background: '#27272a', borderRadius: '8px', border: '1px solid #3f3f46' }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
                <strong style={{ fontSize: '14px', color: '#10b981' }}>{d.title}</strong>
                <span style={{ fontSize: '12px', background: '#064e3b', color: '#6ee7b7', padding: '2px 8px', borderRadius: '12px' }}>
                  {(d.confidence * 100).toFixed(0)}% Conf
                </span>
              </div>
              <p style={{ margin: '0 0 6px 0', fontSize: '13px', color: '#d4d4d8' }}>{d.description}</p>
              <div style={{ fontSize: '11px', color: '#a1a1aa' }}>Domain: {d.domain} • Recalled from {d.date}</div>
            </div>
          ))}
          {filtered.length === 0 && (
            <div style={{ textAlign: 'center', padding: '24px', color: '#71717a' }}>
              No discoveries matching query.
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
