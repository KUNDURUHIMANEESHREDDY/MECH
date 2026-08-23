import React, { FC, useState } from 'react';
import { BarChart3, RefreshCw, AlertCircle, ArrowUp, ArrowRight, ArrowDown, Target, Beaker } from 'lucide-react';
import { colors } from '../../design/tokens/colors';

interface CandidatePriority {
  component: string;
  priority_score: number;
  reason: string;
  evidence_level: string;
  recommended_action: string;
}

export const CandidatePrioritizer: FC = () => {
  const [candidates, setCandidates] = useState('L9H9, L7H9, L8H1, L6H6');
  const [priorities, setPriorities] = useState<CandidatePriority[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const prioritizeCandidates = async () => {
    if (!candidates) {
      setError('Enter candidate components');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await fetch('http://localhost:8000/api/v1/reasoning/candidates/prioritize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          candidates: candidates.split(',').map(s => s.trim()),
          evidence_records: [],
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to prioritize candidates');
      }

      const data = await response.json();
      setPriorities(data.priorities);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to prioritize candidates');
    } finally {
      setLoading(false);
    }
  };

  const actionIcons: Record<string, React.ReactNode> = {
    INVESTIGATE: <Target size={14} />,
    REPLICATE: <RefreshCw size={14} />,
    FALSIFY: <Beaker size={14} />,
  };

  const actionColors: Record<string, string> = {
    INVESTIGATE: colors.primary,
    REPLICATE: colors.success,
    FALSIFY: colors.danger,
  };

  const levelIcons: Record<string, React.ReactNode> = {
    UNTESTED: <ArrowRight size={14} />,
    SUPPORTED: <ArrowUp size={14} />,
    CONTRADICTED: <ArrowDown size={14} />,
    INCONCLUSIVE: <ArrowRight size={14} />,
  };

  const levelColors: Record<string, string> = {
    UNTESTED: colors.inkMuted80,
    SUPPORTED: colors.success,
    CONTRADICTED: colors.danger,
    INCONCLUSIVE: colors.warning,
  };

  return (
    <div style={{ padding: '16px', height: '100%', overflow: 'auto' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '16px' }}>
        <BarChart3 size={20} style={{ color: colors.pink }} />
        <h2 style={{ fontSize: '16px', fontWeight: 600, color: colors.ink, margin: 0 }}>
          Candidate Prioritizer
        </h2>
      </div>

      <div style={{
        background: colors.surfacePearl,
        border: `1px solid ${colors.border}`,
        borderRadius: '8px',
        padding: '16px',
        marginBottom: '16px',
      }}>
        <h3 style={{ fontSize: '14px', color: colors.bodyMuted, marginBottom: '12px' }}>
          Enter Candidate Components
        </h3>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
          <div>
            <label style={{ fontSize: '12px', color: colors.bodyMuted, marginBottom: '4px', display: 'block' }}>
              Candidates (comma-separated)
            </label>
            <input
              value={candidates}
              onChange={(e) => setCandidates(e.target.value)}
              placeholder="L9H9, L7H9, L8H1, L6H6"
              style={{
                width: '100%',
                padding: '8px 12px',
                background: colors.canvas,
                border: `1px solid ${colors.border}`,
                borderRadius: '6px',
                color: colors.ink,
                fontSize: '13px',
              }}
            />
          </div>

          <button
            onClick={prioritizeCandidates}
            disabled={loading || !candidates}
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '8px',
              padding: '10px 16px',
              background: colors.pink,
              color: colors.onPrimary,
              border: 'none',
              borderRadius: '6px',
              cursor: loading || !candidates ? 'not-allowed' : 'pointer',
              opacity: loading || !candidates ? 0.6 : 1,
              fontSize: '13px',
              fontWeight: 500,
            }}
          >
            {loading ? <RefreshCw size={16} className="animate-spin" /> : <BarChart3 size={16} />}
            Prioritize Candidates
          </button>
        </div>
      </div>

      {error && (
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
          padding: '12px',
          background: `${colors.danger}15`,
          border: `1px solid ${colors.danger}40`,
          borderRadius: '6px',
          marginBottom: '16px',
        }}>
          <AlertCircle size={16} style={{ color: colors.danger }} />
          <span style={{ color: colors.danger, fontSize: '13px' }}>{error}</span>
        </div>
      )}

      {priorities.length > 0 && (
        <div>
          <h3 style={{ fontSize: '14px', color: colors.bodyMuted, marginBottom: '12px' }}>
            Prioritized Candidates ({priorities.length})
          </h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            {priorities.map((p, idx) => (
              <div
                key={p.component}
                style={{
                  background: colors.surfacePearl,
                  border: `1px solid ${colors.border}`,
                  borderRadius: '8px',
                  padding: '16px',
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '12px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span style={{
                      fontSize: '18px',
                      fontWeight: 700,
                      color: idx === 0 ? colors.warning : colors.inkMuted80,
                    }}>
                      #{idx + 1}
                    </span>
                    <h4 style={{ fontSize: '14px', color: colors.ink, margin: 0 }}>{p.component}</h4>
                  </div>
                  <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '6px',
                    padding: '4px 10px',
                    background: `${actionColors[p.recommended_action] || colors.inkMuted80}20`,
                    borderRadius: '4px',
                  }}>
                    <span style={{ color: actionColors[p.recommended_action] || colors.inkMuted80 }}>
                      {actionIcons[p.recommended_action]}
                    </span>
                    <span style={{
                      fontSize: '12px',
                      color: actionColors[p.recommended_action] || colors.inkMuted80,
                      fontWeight: 500,
                    }}>
                      {p.recommended_action}
                    </span>
                  </div>
                </div>

                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                  marginBottom: '12px',
                }}>
                  <div style={{
                    width: '100%',
                    height: '8px',
                    background: colors.canvas,
                    borderRadius: '4px',
                    overflow: 'hidden',
                  }}>
                    <div style={{
                      width: `${p.priority_score * 100}%`,
                      height: '100%',
                      background: actionColors[p.recommended_action] || colors.inkMuted80,
                      borderRadius: '4px',
                    }} />
                  </div>
                  <span style={{ fontSize: '12px', color: colors.inkMuted80, minWidth: '40px' }}>
                    {(p.priority_score * 100).toFixed(0)}%
                  </span>
                </div>

                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px',
                  padding: '8px 12px',
                  background: colors.canvas,
                  borderRadius: '4px',
                  border: `1px solid ${colors.hairline}`,
                }}>
                  <span style={{ color: levelColors[p.evidence_level] || colors.inkMuted80 }}>
                    {levelIcons[p.evidence_level]}
                  </span>
                  <span style={{
                    fontSize: '12px',
                    color: levelColors[p.evidence_level] || colors.inkMuted80,
                    fontWeight: 500,
                  }}>
                    {p.evidence_level}
                  </span>
                  <span style={{ fontSize: '12px', color: colors.inkMuted80, marginLeft: 'auto' }}>
                    {p.reason}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
