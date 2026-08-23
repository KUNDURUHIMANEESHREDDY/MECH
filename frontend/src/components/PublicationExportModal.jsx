import React, { useState } from 'react';

export const PublicationExportModal = ({
  isOpen = true,
  onClose,
}) => {
  const [template, setTemplate] = useState('neurips');
  const [paperTitle, setPaperTitle] = useState('Mechanistic Circuit Analysis of Transformer In-Context Learning');
  const [includeVectorGraphics, setIncludeVectorGraphics] = useState(true);
  const [includeRawData, setIncludeRawData] = useState(false);
  const [isExporting, setIsExporting] = useState(false);
  const [exportResult, setExportResult] = useState('');

  if (!isOpen) return null;

  const handleExport = () => {
    setIsExporting(true);
    setExportResult('');
    setTimeout(() => {
      setIsExporting(false);
      setExportResult(`Successfully generated LaTeX preprint & PDF report: exports/${template}_paper.pdf`);
    }, 800);
  };

  return (
    <div className="modal-backdrop" data-testid="publication-export-modal">
      <div className="modal-container" style={{ maxWidth: '650px', background: '#18181b', color: '#f4f4f5', padding: '24px', borderRadius: '10px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
          <h2 style={{ margin: 0, fontSize: '18px', fontWeight: 600 }}>Export Scientific Publication</h2>
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
            <label style={{ display: 'block', fontSize: '13px', marginBottom: '4px', color: '#a1a1aa' }}>Paper Working Title</label>
            <input
              data-testid="paper-title-input"
              type="text"
              value={paperTitle}
              onChange={(e) => setPaperTitle(e.target.value)}
              style={{ width: '100%', boxSizing: 'border-box', padding: '8px 12px', background: '#27272a', border: '1px solid #3f3f46', borderRadius: '6px', color: '#fff' }}
            />
          </div>

          <div>
            <label style={{ display: 'block', fontSize: '13px', marginBottom: '4px', color: '#a1a1aa' }}>Conference / Journal Template</label>
            <select
              data-testid="paper-template-select"
              value={template}
              onChange={(e) => setTemplate(e.target.value)}
              style={{ width: '100%', padding: '8px 12px', background: '#27272a', border: '1px solid #3f3f46', borderRadius: '6px', color: '#fff' }}
            >
              <option value="neurips">NeurIPS 2026 Camera-Ready LaTeX</option>
              <option value="iclr">ICLR 2026 Double-Blind LaTeX</option>
              <option value="icml">ICML 2026 2-Column Template</option>
              <option value="pdf_summary">Executive Summary (PDF 4-Page)</option>
              <option value="interactive_html">Interactive HTML Research Report</option>
            </select>
          </div>

          <label style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '13px', cursor: 'pointer' }}>
            <input
              type="checkbox"
              checked={includeVectorGraphics}
              onChange={(e) => setIncludeVectorGraphics(e.target.checked)}
            />
            <span>Include vector SVG circuit diagrams and attention heatmaps</span>
          </label>

          <label style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '13px', cursor: 'pointer' }}>
            <input
              type="checkbox"
              checked={includeRawData}
              onChange={(e) => setIncludeRawData(e.target.checked)}
            />
            <span>Attach reproducibility data bundle (NumPy / Torch tensor checkpoints)</span>
          </label>
        </div>

        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '10px', alignItems: 'center' }}>
          {exportResult && <span style={{ fontSize: '12px', color: '#10b981' }}>{exportResult}</span>}
          <button
            data-testid="export-submit-btn"
            disabled={isExporting}
            onClick={handleExport}
            style={{
              padding: '8px 16px',
              borderRadius: '6px',
              border: 'none',
              fontWeight: 600,
              background: '#10b981',
              color: '#fff',
              cursor: isExporting ? 'not-allowed' : 'pointer',
            }}
          >
            {isExporting ? 'Compiling LaTeX...' : 'Export Publication'}
          </button>
        </div>
      </div>
    </div>
  );
};
