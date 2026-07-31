import React, { useState, useEffect } from 'react';
import { BarChart3, ClipboardList } from 'lucide-react';
import { notebookStore } from '../domain/notebook/notebookStore';
import { ReportGenerator } from '../domain/reports/reportGenerator';

export default function ExperimentNotebook() {
  const [cells, setCells] = useState(notebookStore.getCells());
  const [newContent, setNewContent] = useState('');

  const handleAddCell = () => {
    if (newContent.trim()) {
      notebookStore.addCell('markdown', newContent);
      setCells(notebookStore.getCells());
      setNewContent('');
    }
  };

  return (
    <div className="card experiment-notebook-card" data-testid="experiment-notebook">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
        <h3>Experiment Research Notebook</h3>
        <button
          className="btn btn-secondary btn-sm"
          onClick={() => {
            const report = ReportGenerator.generateMarkdownReport();
            alert('Generated Markdown Report:\n\n' + report.substring(0, 200) + '...');
          }}
        >
          Export Report
        </button>
      </div>

      <div className="notebook-cells" style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
        {cells.map((cell) => (
          <div key={cell.id} className="notebook-cell" style={{ background: 'var(--bg)', border: '1px solid var(--border)', padding: '10px', borderRadius: '6px' }}>
            <span className="kbd-shortcut" style={{ marginBottom: '6px', display: 'inline-block' }}>{cell.type.toUpperCase()}</span>
            {cell.type === 'markdown' && <div style={{ whiteSpace: 'pre-wrap', fontSize: '13px' }}>{cell.content}</div>}
            {cell.type === 'plot' && <div style={{ fontSize: '12px', color: 'var(--accent)' }}><BarChart3 size={12} style={{ verticalAlign: 'middle', marginRight: 4 }} />Plot: {cell.title}</div>}
            {cell.type === 'table' && <div style={{ fontSize: '12px' }}><ClipboardList size={12} style={{ verticalAlign: 'middle', marginRight: 4 }} />Table: {cell.headers.join(' | ')}</div>}
            {cell.type === 'json' && <pre style={{ fontSize: '11px', margin: 0 }}>{JSON.stringify(cell.data, null, 2)}</pre>}
          </div>
        ))}
      </div>

      <div style={{ marginTop: '12px', display: 'flex', gap: '8px' }}>
        <input
          type="text"
          placeholder="Add observation note..."
          value={newContent}
          onChange={(e) => setNewContent(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleAddCell()}
        />
        <button className="btn btn-primary btn-sm" onClick={handleAddCell}>+ Add Note</button>
      </div>
    </div>
  );
}

