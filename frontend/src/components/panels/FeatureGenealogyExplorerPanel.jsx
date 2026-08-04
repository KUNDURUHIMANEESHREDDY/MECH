import React from 'react';
import { TreeDeciduous } from 'lucide-react';

export default function FeatureGenealogyExplorerPanel() {
  return (
    <div className="panel feature-genealogy-explorer-panel" data-testid="feature-genealogy-explorer-panel">
      <div className="panel-header">
        <h3><TreeDeciduous size={13} style={{ verticalAlign: 'middle', marginRight: 6, color: 'var(--accent)' }} /> Feature Genealogy Explorer</h3>
      </div>
      <div className="panel-body" style={{ padding: '12px' }}>
        <div style={{ padding: '12px', background: 'var(--bg-elev-2)', borderRadius: '6px' }}>
          <h4 style={{ margin: '0 0 4px 0', color: 'var(--purple)', fontSize: '13px' }}>SAE Feature #1402 Lineage Tree</h4>
          <p style={{ margin: 0, fontSize: '11px', color: 'var(--text-muted)' }}>
            Parents: <strong>[Feature #310, Feature #402]</strong> -> Derived: <strong>[Feature #1402]</strong>
          </p>
        </div>
      </div>
    </div>
  );
}
