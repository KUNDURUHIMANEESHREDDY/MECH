import React, { useEffect, useState } from 'react';
import { useResearchStore } from '../../shared/stores/research';
import { colors } from '../../design/tokens/colors';
import { FileText, Download, Copy, Check, RefreshCw } from 'lucide-react';

export const ArtifactsReportView: React.FC = () => {
  const { activeInvestigation, generateReport } = useResearchStore();
  const [reportMarkdown, setReportMarkdown] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [copied, setCopied] = useState(false);

  const fetchReport = async () => {
    if (!activeInvestigation) return;
    setLoading(true);
    try {
      const md = await generateReport(activeInvestigation.id);
      setReportMarkdown(md);
    } catch (err) {
      console.warn('Report generation error:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchReport();
  }, [activeInvestigation?.id]);

  const handleCopy = () => {
    navigator.clipboard.writeText(reportMarkdown);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownload = () => {
    const blob = new Blob([reportMarkdown], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `MECH_Research_Report_${activeInvestigation?.id || 'investigation'}.md`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div style={{ padding: '20px', height: '100%', overflowY: 'auto', boxSizing: 'border-box', backgroundColor: colors.canvas }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
        <div>
          <div style={{ fontSize: 11, fontWeight: 700, color: colors.primary, textTransform: 'uppercase', letterSpacing: 0.8 }}>
            Scientific Publication Mode
          </div>
          <h2 style={{ margin: '4px 0 0', fontSize: 18, fontWeight: 700, color: colors.ink }}>
            11-Section Empirical Research Report
          </h2>
          <div style={{ fontSize: 12, color: colors.bodyMuted, marginTop: 2 }}>
            Generated directly from live experiment runs, causal intervention deltas, and evidence matrices.
          </div>
        </div>

        <div style={{ display: 'flex', gap: 8 }}>
          <button
            onClick={fetchReport}
            disabled={loading}
            style={{
              padding: '7px 12px',
              borderRadius: 6,
              border: `1px solid ${colors.border}`,
              backgroundColor: colors.surfaceTile1,
              color: colors.ink,
              fontSize: 12,
              fontWeight: 600,
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: 6,
            }}
          >
            <RefreshCw size={13} className={loading ? 'spinner' : ''} /> Refresh
          </button>
          <button
            onClick={handleCopy}
            style={{
              padding: '7px 12px',
              borderRadius: 6,
              border: `1px solid ${colors.border}`,
              backgroundColor: colors.surfaceTile1,
              color: colors.ink,
              fontSize: 12,
              fontWeight: 600,
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: 6,
            }}
          >
            {copied ? <Check size={13} style={{ color: colors.successText }} /> : <Copy size={13} />}
            {copied ? 'Copied!' : 'Copy Markdown'}
          </button>
          <button
            onClick={handleDownload}
            style={{
              padding: '7px 14px',
              borderRadius: 6,
              border: 'none',
              backgroundColor: colors.primary,
              color: colors.onPrimary,
              fontSize: 12,
              fontWeight: 600,
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: 6,
            }}
          >
            <Download size={13} /> Export Report (.md)
          </button>
        </div>
      </div>

      {/* Markdown Document Preview */}
      <div
        style={{
          border: `1px solid ${colors.border}`,
          borderRadius: 10,
          padding: '24px 32px',
          backgroundColor: colors.canvas,
          boxShadow: '0 1px 4px rgba(0,0,0,0.06)',
          fontFamily: 'var(--font-sans, system-ui, sans-serif)',
          fontSize: 13,
          lineHeight: 1.6,
          color: colors.ink,
          whiteSpace: 'pre-wrap',
        }}
      >
        {loading ? (
          <div style={{ color: colors.bodyMuted, textAlign: 'center', padding: '40px 0' }}>
            Compiling empirical evidence report…
          </div>
        ) : reportMarkdown ? (
          reportMarkdown
        ) : (
          <div style={{ color: colors.bodyMuted, textAlign: 'center', padding: '40px 0' }}>
            No report available. Ensure an active investigation exists with executed experiments.
          </div>
        )}
      </div>
    </div>
  );
};
