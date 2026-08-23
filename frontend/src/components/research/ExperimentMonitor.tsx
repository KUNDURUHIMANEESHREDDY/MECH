import React, { useState, useEffect } from 'react';
import { useResearchStore } from '../../shared/stores/research';
import { colors } from '../../design/tokens/colors';
import {
  Clock,
  Play,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  Loader2,
  RefreshCw,
  ChevronDown,
  ChevronUp,
  Eye,
  EyeOff,
} from 'lucide-react';

type StatusFilter = 'ALL' | 'PENDING' | 'RUNNING' | 'COMPLETED' | 'FAILED' | 'NOT_EXECUTABLE';

const STATUS_CONFIG: Record<string, { color: string; bgColor: string; borderColor: string; icon: React.ReactNode; label: string }> = {
  PENDING: {
    color: colors.warningText,
    bgColor: colors.warningSoft,
    borderColor: colors.warningBorder,
    icon: <Clock size={14} />,
    label: 'PENDING',
  },
  RUNNING: {
    color: colors.infoText,
    bgColor: colors.infoSoft,
    borderColor: colors.infoBorder,
    icon: <Loader2 size={14} className="animate-spin" />,
    label: 'RUNNING',
  },
  COMPLETED: {
    color: colors.successText,
    bgColor: colors.successSoft,
    borderColor: colors.successBorder,
    icon: <CheckCircle2 size={14} />,
    label: 'COMPLETED',
  },
  FAILED: {
    color: colors.dangerText,
    bgColor: colors.dangerSoft,
    borderColor: colors.dangerBorder,
    icon: <XCircle size={14} />,
    label: 'FAILED',
  },
  NOT_EXECUTABLE: {
    color: colors.dangerText,
    bgColor: colors.dangerSoft,
    borderColor: colors.dangerBorder,
    icon: <AlertTriangle size={14} />,
    label: 'NOT EXECUTABLE',
  },
};

export const ExperimentMonitor: React.FC = () => {
  const { runs, loadRuns, loading, error } = useResearchStore();
  const [statusFilter, setStatusFilter] = useState<StatusFilter>('ALL');
  const [expandedRunId, setExpandedRunId] = useState<string | null>(null);
  const [showMockWarning, setShowMockWarning] = useState(true);

  useEffect(() => {
    loadRuns();
    const interval = setInterval(loadRuns, 5000);
    return () => clearInterval(interval);
  }, [loadRuns]);

  const filteredRuns = runs.filter((run) => {
    if (statusFilter === 'ALL') return true;
    return run.execution_status === statusFilter;
  });

  const statusCounts = {
    ALL: runs.length,
    PENDING: runs.filter((r) => r.execution_status === 'PENDING').length,
    RUNNING: runs.filter((r) => r.execution_status === 'RUNNING').length,
    COMPLETED: runs.filter((r) => r.execution_status === 'COMPLETED').length,
    FAILED: runs.filter((r) => r.execution_status === 'FAILED').length,
    NOT_EXECUTABLE: runs.filter((r) => r.execution_status === 'NOT_EXECUTABLE').length,
  };

  const toggleExpand = (id: string) => {
    setExpandedRunId(expandedRunId === id ? null : id);
  };

  return (
    <div style={{ padding: '20px', height: '100%', overflowY: 'auto', boxSizing: 'border-box', backgroundColor: colors.canvas }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 16 }}>
        <div>
          <div style={{ fontSize: 11, fontWeight: 700, color: colors.primary, textTransform: 'uppercase', letterSpacing: 0.8 }}>
            Experiment Monitoring
          </div>
          <h2 style={{ margin: '4px 0 0', fontSize: 18, fontWeight: 700, color: colors.ink }}>
            Experiment Monitor
          </h2>
          <div style={{ fontSize: 12, color: colors.bodyMuted, marginTop: 2 }}>
            Track experiment execution status and view results from Agent 1.
          </div>
        </div>

        <button
          onClick={() => loadRuns()}
          disabled={loading}
          style={{
            padding: '8px 12px',
            borderRadius: 6,
            border: `1px solid ${colors.border}`,
            backgroundColor: colors.canvas,
            color: colors.bodyMuted,
            fontSize: 12,
            fontWeight: 600,
            cursor: loading ? 'default' : 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: 4,
          }}
        >
          <RefreshCw size={13} className={loading ? 'animate-spin' : ''} /> Refresh
        </button>
      </div>

      {/* Error Display */}
      {error && (
        <div style={{
          padding: '10px 14px',
          borderRadius: 8,
          backgroundColor: colors.dangerSoft,
          border: `1px solid ${colors.dangerBorder}`,
          color: colors.dangerText,
          fontSize: 12,
          marginBottom: 16,
          display: 'flex',
          alignItems: 'center',
          gap: 8,
        }}>
          <AlertTriangle size={16} /> {error}
        </div>
      )}

      {/* Status Filter Tabs */}
      <div style={{ display: 'flex', gap: 6, marginBottom: 16, flexWrap: 'wrap' }}>
        {(Object.keys(STATUS_CONFIG) as StatusFilter[]).map((status) => {
          const config = STATUS_CONFIG[status] || STATUS_CONFIG.PENDING;
          const isActive = statusFilter === status;
          return (
            <button
              key={status}
              onClick={() => setStatusFilter(status)}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 6,
                padding: '6px 12px',
                borderRadius: 20,
                border: `1px solid ${isActive ? config.borderColor : colors.border}`,
                backgroundColor: isActive ? config.bgColor : colors.canvas,
                color: isActive ? config.color : colors.bodyMuted,
                fontSize: 11,
                fontWeight: 700,
                cursor: 'pointer',
                transition: 'all 0.15s ease',
              }}
            >
              {config.icon}
              {config.label}
              <span style={{
                padding: '1px 6px',
                borderRadius: 10,
                backgroundColor: isActive ? 'rgba(0,0,0,0.1)' : colors.surfacePearl,
                fontSize: 10,
                fontWeight: 800,
              }}>
                {statusCounts[status] || 0}
              </span>
            </button>
          );
        })}
      </div>

      {/* Mock Data Warning */}
      {showMockWarning && runs.some((r) => r.used_mock_data) && (
        <div style={{
          padding: '10px 14px',
          borderRadius: 8,
          backgroundColor: colors.warningSoft,
          border: `1px solid ${colors.warningBorder}`,
          color: colors.warningText,
          fontSize: 12,
          marginBottom: 16,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <AlertTriangle size={16} />
            <span><b>Mock Data Detected:</b> Some experiment results are from simulated data, not live model execution.</span>
          </div>
          <button
            onClick={() => setShowMockWarning(false)}
            style={{
              background: 'none',
              border: 'none',
              color: colors.warningText,
              cursor: 'pointer',
              padding: 4,
            }}
          >
            <XCircle size={14} />
          </button>
        </div>
      )}

      {/* Runs List */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
        {filteredRuns.length === 0 ? (
          <div style={{
            padding: 40,
            textAlign: 'center',
            color: colors.bodyMuted,
            backgroundColor: colors.surfacePearl,
            borderRadius: 10,
          }}>
            <Clock size={32} style={{ opacity: 0.3, marginBottom: 8 }} />
            <div style={{ fontSize: 13, fontWeight: 600, color: colors.ink }}>No Experiments Found</div>
            <div style={{ fontSize: 12, marginTop: 4 }}>
              {statusFilter === 'ALL'
                ? 'Run an experiment from the Experiment Builder to see results here.'
                : `No experiments with status "${statusFilter}".`}
            </div>
          </div>
        ) : (
          filteredRuns.map((run) => {
            const statusConfig = STATUS_CONFIG[run.execution_status] || STATUS_CONFIG.PENDING;
            const isExpanded = expandedRunId === run.id;

            return (
              <div
                key={run.id}
                style={{
                  border: `1px solid ${statusConfig.borderColor}`,
                  borderRadius: 10,
                  backgroundColor: colors.surfaceTile1,
                  overflow: 'hidden',
                  transition: 'all 0.15s ease',
                }}
              >
                {/* Run Header */}
                <div
                  onClick={() => toggleExpand(run.id)}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    padding: '12px 14px',
                    cursor: 'pointer',
                    backgroundColor: isExpanded ? statusConfig.bgColor : 'transparent',
                    transition: 'background-color 0.15s ease',
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                    <div style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: 4,
                      padding: '3px 8px',
                      borderRadius: 4,
                      backgroundColor: statusConfig.bgColor,
                      color: statusConfig.color,
                      fontSize: 10,
                      fontWeight: 700,
                    }}>
                      {statusConfig.icon}
                      {statusConfig.label}
                    </div>
                    <div>
                      <div style={{ fontSize: 13, fontWeight: 600, color: colors.ink }}>
                        {run.id.slice(0, 12)}...
                      </div>
                      <div style={{ fontSize: 11, color: colors.bodyMuted }}>
                        {new Date(run.timestamp * 1000).toLocaleString()} | {run.execution_time_ms?.toFixed(0) || '0'}ms
                      </div>
                    </div>
                  </div>

                  <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                    {run.used_mock_data && (
                      <span style={{
                        padding: '2px 6px',
                        borderRadius: 4,
                        backgroundColor: colors.warningSoft,
                        color: colors.warningText,
                        fontSize: 10,
                        fontWeight: 700,
                      }}>
                        MOCK DATA
                      </span>
                    )}
                    <span style={{ fontSize: 11, color: colors.bodyMuted }}>
                      Δ = {run.delta_logit?.toFixed(3) ?? 'N/A'}
                    </span>
                    {isExpanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                  </div>
                </div>

                {/* Expanded Details */}
                {isExpanded && (
                  <div style={{ padding: '0 14px 14px', borderTop: `1px solid ${colors.border}` }}>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: 10, marginTop: 12 }}>
                      <MetricDisplay label="Baseline Logit" value={run.baseline_logit?.toFixed(3) ?? 'N/A'} />
                      <MetricDisplay label="Intervened Logit" value={run.intervened_logit?.toFixed(3) ?? 'N/A'} />
                      <MetricDisplay label="Δ Logit" value={run.delta_logit?.toFixed(3) ?? 'N/A'} highlight={run.delta_logit > 1} />
                      <MetricDisplay label="Baseline Prob" value={`${((run.baseline_target_prob ?? 0) * 100).toFixed(1)}%`} />
                      <MetricDisplay label="Intervened Prob" value={`${((run.intervened_target_prob ?? 0) * 100).toFixed(1)}%`} />
                      <MetricDisplay label="Δ Probability" value={`${((run.delta_target_prob ?? 0) * 100).toFixed(1)}%`} />
                      <MetricDisplay label="Control Δ Logit" value={run.control_delta_logit?.toFixed(3) ?? 'N/A'} />
                      <MetricDisplay label="Effect Size (d)" value={run.effect_size_cohens_d?.toFixed(2) ?? 'N/A'} />
                    </div>

                    {/* Top Predictions */}
                    {run.top_predicted_tokens_clean?.length > 0 && (
                      <div style={{ marginTop: 12 }}>
                        <div style={{ fontSize: 11, fontWeight: 700, color: colors.bodyMuted, textTransform: 'uppercase', marginBottom: 6 }}>
                          Top Predictions
                        </div>
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                          <div style={{ padding: 8, borderRadius: 6, backgroundColor: colors.canvas, border: `1px solid ${colors.border}` }}>
                            <div style={{ fontSize: 10, fontWeight: 700, color: colors.bodyMuted, marginBottom: 4 }}>BASELINE</div>
                            {run.top_predicted_tokens_clean.slice(0, 3).map((t: any, i: number) => (
                              <div key={i} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, marginBottom: 2 }}>
                                <code>{t.token}</code>
                                <span>{(t.probability * 100).toFixed(1)}%</span>
                              </div>
                            ))}
                          </div>
                          <div style={{ padding: 8, borderRadius: 6, backgroundColor: colors.canvas, border: `1px solid ${colors.border}` }}>
                            <div style={{ fontSize: 10, fontWeight: 700, color: colors.bodyMuted, marginBottom: 4 }}>INTERVENED</div>
                            {run.top_predicted_tokens_intervened.slice(0, 3).map((t: any, i: number) => (
                              <div key={i} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, marginBottom: 2 }}>
                                <code>{t.token}</code>
                                <span>{(t.probability * 100).toFixed(1)}%</span>
                              </div>
                            ))}
                          </div>
                        </div>
                      </div>
                    )}

                    {/* Execution Info */}
                    <div style={{ marginTop: 12, padding: 8, borderRadius: 6, backgroundColor: colors.surfacePearl, fontSize: 11, color: colors.bodyMuted }}>
                      <div><b>Run ID:</b> {run.id}</div>
                      <div><b>Model:</b> {run.model_id}</div>
                      <div><b>Provenance:</b> {run.provenance_hash?.slice(0, 24) || 'N/A'}...</div>
                      <div><b>Reproducible:</b> {run.is_reproducible ? 'Yes' : 'No'}</div>
                    </div>
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>
    </div>
  );
};

/* ── Helper Component ──────────────────────────────────────────────── */

const MetricDisplay: React.FC<{ label: string; value: string; highlight?: boolean }> = ({ label, value, highlight }) => (
  <div style={{ padding: 8, borderRadius: 6, backgroundColor: colors.canvas, border: `1px solid ${colors.border}` }}>
    <div style={{ fontSize: 10, fontWeight: 700, color: colors.bodyMuted, textTransform: 'uppercase', marginBottom: 2 }}>
      {label}
    </div>
    <div style={{
      fontSize: 14,
      fontWeight: 700,
      color: highlight ? colors.success : colors.ink,
      fontVariantNumeric: 'tabular-nums',
    }}>
      {value}
    </div>
  </div>
);
