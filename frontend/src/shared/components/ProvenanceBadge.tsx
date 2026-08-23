/**
 * ProvenanceBadge.tsx
 *
 * Prominent provenance indicator displayed in the shell Toolbar.
 * Shows three states:
 *   LIVE_PYTORCH    - green badge: backend reachable, live model weights
 *   BACKEND_OFFLINE - amber badge: backend offline, data is SYNTHETIC
 *   CHECKING        - neutral pulse while first check is in flight
 *
 * The amber DEMO MODE badge includes a prominent tooltip warning so
 * researchers cannot confuse synthetic mock activations with real model data.
 */
import React, { useState } from 'react';
import { Activity, AlertTriangle, Loader } from 'lucide-react';
import { useBackendStatus } from '../hooks/useBackendStatus';

const styles: Record<string, React.CSSProperties> = {
  badge: {
    display: 'flex',
    alignItems: 'center',
    gap: '5px',
    fontSize: '11px',
    fontWeight: 600,
    letterSpacing: '0.4px',
    padding: '4px 10px',
    borderRadius: '6px',
    cursor: 'default',
    userSelect: 'none',
    position: 'relative',
  },
  live: {
    background: 'rgba(30, 142, 62, 0.12)',
    color: '#1e8e3e',
    border: '1px solid rgba(30,142,62,0.35)',
  },
  demo: {
    background: 'rgba(251, 140, 0, 0.13)',
    color: '#e65100',
    border: '1px solid rgba(230,81,0,0.45)',
  },
  checking: {
    background: 'var(--bg-elev-2)',
    color: 'var(--text-muted)',
    border: '1px solid var(--border-light)',
  },
  tooltip: {
    position: 'absolute',
    top: 'calc(100% + 8px)',
    right: 0,
    zIndex: 9999,
    minWidth: '280px',
    background: '#1a1a2e',
    color: '#f5f5f5',
    fontSize: '11px',
    lineHeight: '1.5',
    borderRadius: '8px',
    padding: '10px 14px',
    boxShadow: '0 4px 20px rgba(0,0,0,0.45)',
    pointerEvents: 'none',
    whiteSpace: 'normal',
    border: '1px solid rgba(230,81,0,0.5)',
  },
};

export const ProvenanceBadge: React.FC = () => {
  const status = useBackendStatus();
  const [tooltipVisible, setTooltipVisible] = useState(false);

  if (status === 'CHECKING') {
    return (
      <div style={{ ...styles.badge, ...styles.checking }}>
        <Loader size={11} />
        <span>Connecting...</span>
      </div>
    );
  }

  if (status === 'LIVE_PYTORCH') {
    return (
      <div style={{ ...styles.badge, ...styles.live }}>
        <Activity size={11} />
        <span>LIVE PyTorch</span>
      </div>
    );
  }

  return (
    <div
      style={{ ...styles.badge, ...styles.demo }}
      onMouseEnter={() => setTooltipVisible(true)}
      onMouseLeave={() => setTooltipVisible(false)}
    >
      <AlertTriangle size={11} />
      <span>DEMO MODE (SYNTHETIC)</span>
      {tooltipVisible && (
        <div style={styles.tooltip}>
          <strong style={{ color: '#ffb74d' }}>Warning: Synthetic Data Active</strong>
          <br />
          The Python backend sidecar is unreachable. All displayed attention maps,
          neuron activations, and logit values are deterministic synthetic
          approximations — NOT live PyTorch model computations.
          <br /><br />
          Start the backend with: python -m backend.main
        </div>
      )}
    </div>
  );
};

export default ProvenanceBadge;
