import React from 'react';
import { ClipboardPen } from 'lucide-react';

export default function PublicationStudioPanel() {
  return (
    <div className="panel publication-studio-panel" data-testid="publication-studio-panel">
      <div className="panel-header">
        <h3><ClipboardPen size={13} style={{ verticalAlign: 'middle', marginRight: 6, color: 'var(--accent)' }} /> Publication Studio 2.0</h3>
      </div>
      <div className="panel-body" style={{ padding: '12px' }}>
        <h4 style={{ margin: '0 0 6px 0', color: 'var(--text)', fontSize: '13px' }}>Live-Linked Manuscript Editor</h4>
        <p style={{ margin: 0, fontSize: '11px', color: 'var(--text-muted)' }}>
          Interactive LaTeX & Markdown paper editing. Figures updated automatically as underlying discoveries change.
        </p>
      </div>
    </div>
  );
}
