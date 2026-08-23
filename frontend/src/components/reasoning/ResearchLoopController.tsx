import React, { FC, useState, useEffect } from 'react';
import { Repeat, Play, Pause, RefreshCw, AlertCircle, CheckCircle, Clock, ArrowRight } from 'lucide-react';
import { colors } from '../../design/tokens/colors';

interface ResearchLoop {
  investigation_id: string;
  current_hypothesis_id: string | null;
  iteration: number;
  max_iterations: number;
  status: string;
}

export const ResearchLoopController: FC = () => {
  const [investigationId, setInvestigationId] = useState('');
  const [hypothesisId, setHypothesisId] = useState('');
  const [maxIterations, setMaxIterations] = useState(10);
  const [activeLoop, setActiveLoop] = useState<ResearchLoop | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const startLoop = async () => {
    if (!investigationId || !hypothesisId) {
      setError('Investigation ID and Hypothesis ID are required');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await fetch('http://localhost:8000/api/v1/reasoning/research-loop/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          investigation_id: investigationId,
          initial_hypothesis_id: hypothesisId,
          max_iterations: maxIterations,
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to start research loop');
      }

      const data = await response.json();
      setActiveLoop(data.loop);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to start research loop');
    } finally {
      setLoading(false);
    }
  };

  const statusColors: Record<string, string> = {
    RUNNING: colors.success,
    COMPLETED: colors.primary,
    PAUSED: colors.warning,
  };

  return (
    <div style={{ padding: '16px', height: '100%', overflow: 'auto' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '16px' }}>
        <Repeat size={20} style={{ color: colors.purple }} />
        <h2 style={{ fontSize: '16px', fontWeight: 600, color: colors.ink, margin: 0 }}>
          Research Loop Controller
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
          Start Research Loop
        </h3>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
          <div>
            <label style={{ fontSize: '12px', color: colors.bodyMuted, marginBottom: '4px', display: 'block' }}>
              Investigation ID
            </label>
            <input
              value={investigationId}
              onChange={(e) => setInvestigationId(e.target.value)}
              placeholder="inv_abc123"
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

          <div>
            <label style={{ fontSize: '12px', color: colors.bodyMuted, marginBottom: '4px', display: 'block' }}>
              Initial Hypothesis ID
            </label>
            <input
              value={hypothesisId}
              onChange={(e) => setHypothesisId(e.target.value)}
              placeholder="hyp_abc123"
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

          <div>
            <label style={{ fontSize: '12px', color: colors.bodyMuted, marginBottom: '4px', display: 'block' }}>
              Max Iterations
            </label>
            <input
              type="number"
              value={maxIterations}
              onChange={(e) => setMaxIterations(parseInt(e.target.value) || 10)}
              min={1}
              max={50}
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
            onClick={startLoop}
            disabled={loading || !investigationId || !hypothesisId}
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '8px',
              padding: '10px 16px',
              background: colors.purple,
              color: colors.onPrimary,
              border: 'none',
              borderRadius: '6px',
              cursor: loading || !investigationId || !hypothesisId ? 'not-allowed' : 'pointer',
              opacity: loading || !investigationId || !hypothesisId ? 0.6 : 1,
              fontSize: '13px',
              fontWeight: 500,
            }}
          >
            {loading ? <RefreshCw size={16} className="animate-spin" /> : <Play size={16} />}
            Start Research Loop
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

      {activeLoop && (
        <div style={{
          background: colors.surfacePearl,
          border: `1px solid ${colors.border}`,
          borderRadius: '8px',
          padding: '16px',
        }}>
          <h3 style={{ fontSize: '14px', color: colors.bodyMuted, marginBottom: '12px' }}>
            Active Research Loop
          </h3>

          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            padding: '12px',
            background: `${statusColors[activeLoop.status] || colors.inkMuted80}15`,
            border: `1px solid ${statusColors[activeLoop.status] || colors.inkMuted80}40`,
            borderRadius: '6px',
            marginBottom: '16px',
          }}>
            <span style={{ color: statusColors[activeLoop.status] || colors.inkMuted80 }}>
              {activeLoop.status === 'RUNNING' ? <Play size={16} /> : <CheckCircle size={16} />}
            </span>
            <span style={{
              fontSize: '14px',
              fontWeight: 600,
              color: statusColors[activeLoop.status] || colors.inkMuted80,
            }}>
              {activeLoop.status}
            </span>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '12px', marginBottom: '16px' }}>
            <div style={{
              background: colors.canvas,
              padding: '12px',
              borderRadius: '6px',
              border: `1px solid ${colors.hairline}`,
              textAlign: 'center',
            }}>
              <div style={{ fontSize: '24px', fontWeight: 700, color: colors.purple }}>
                {activeLoop.iteration}
              </div>
              <div style={{ fontSize: '11px', color: colors.inkMuted80, marginTop: '4px' }}>Iteration</div>
            </div>

            <div style={{
              background: colors.canvas,
              padding: '12px',
              borderRadius: '6px',
              border: `1px solid ${colors.hairline}`,
              textAlign: 'center',
            }}>
              <div style={{ fontSize: '24px', fontWeight: 700, color: colors.inkMuted80 }}>
                {activeLoop.max_iterations}
              </div>
              <div style={{ fontSize: '11px', color: colors.inkMuted80, marginTop: '4px' }}>Max</div>
            </div>

            <div style={{
              background: colors.canvas,
              padding: '12px',
              borderRadius: '6px',
              border: `1px solid ${colors.hairline}`,
              textAlign: 'center',
            }}>
              <div style={{ fontSize: '24px', fontWeight: 700, color: colors.infoSoft }}>
                {((activeLoop.iteration / activeLoop.max_iterations) * 100).toFixed(0)}%
              </div>
              <div style={{ fontSize: '11px', color: colors.inkMuted80, marginTop: '4px' }}>Complete</div>
            </div>
          </div>

          <div style={{
            background: colors.canvas,
            padding: '10px',
            borderRadius: '6px',
            border: `1px solid ${colors.hairline}`,
          }}>
            <div style={{ fontSize: '11px', color: colors.inkMuted80, marginBottom: '4px' }}>
              Investigation: {activeLoop.investigation_id}
            </div>
            <div style={{ fontSize: '11px', color: colors.inkMuted80 }}>
              Current Hypothesis: {activeLoop.current_hypothesis_id || 'None'}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
