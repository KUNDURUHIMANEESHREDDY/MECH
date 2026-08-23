import React, { useState, useEffect, useMemo } from 'react';
import { useResearchStore } from '../../shared/stores/research';
import { colors } from '../../design/tokens/colors';
import {
  Search,
  Brain,
  ShieldCheck,
  ShieldAlert,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  ChevronDown,
  ChevronUp,
  Link,
  Unlink,
  BarChart3,
  Target,
  Layers,
  Clock,
  RefreshCw,
  Filter,
  Eye,
  EyeOff,
  ArrowRight,
  Zap,
} from 'lucide-react';

type EvidenceFilter = 'ALL' | 'SUPPORTED' | 'CONTRADICTING' | 'LINKED' | 'UNLINKED';

interface EvidenceItem {
  id: string;
  hypothesis_id?: string;
  hypothesis_title?: string;
  claim: string;
  evidence_level: string;
  supports_hypothesis: boolean;
  metric_name: string;
  metric_value: number;
  baseline_value?: number;
  control_value?: number;
  sample_size: number;
  provenance_chain: string[];
  created_at: number;
}

interface HypothesisWithEvidence {
  hypothesis: any;
  supporting: EvidenceItem[];
  contradicting: EvidenceItem[];
  totalEvidence: number;
}

const EVIDENCE_LEVEL_CONFIG: Record<string, { color: string; bgColor: string; borderColor: string; icon: React.ReactNode }> = {
  CAUSALLY_VERIFIED: {
    color: colors.successText,
    bgColor: colors.successSoft,
    borderColor: colors.successBorder,
    icon: <ShieldCheck size={14} />,
  },
  SUPPORTED: {
    color: colors.infoText,
    bgColor: colors.infoSoft,
    borderColor: colors.infoBorder,
    icon: <CheckCircle2 size={14} />,
  },
  CANDIDATE: {
    color: colors.warningText,
    bgColor: colors.warningSoft,
    borderColor: colors.warningBorder,
    icon: <AlertTriangle size={14} />,
  },
  OBSERVED: {
    color: colors.bodyMuted,
    bgColor: colors.surfacePearl,
    borderColor: colors.border,
    icon: <Eye size={14} />,
  },
  FALSIFIED: {
    color: colors.dangerText,
    bgColor: colors.dangerSoft,
    borderColor: colors.dangerBorder,
    icon: <ShieldAlert size={14} />,
  },
};

export const EvidenceExplorer: React.FC = () => {
  const {
    activeInvestigation,
    hypotheses,
    evidence,
    loading,
    error,
    loadEvidence,
    loadHypotheses,
  } = useResearchStore();

  const [filter, setFilter] = useState<EvidenceFilter>('ALL');
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [viewMode, setViewMode] = useState<'by_hypothesis' | 'by_evidence'>('by_hypothesis');

  useEffect(() => {
    if (activeInvestigation?.id) {
      loadEvidence(activeInvestigation.id);
      loadHypotheses(activeInvestigation.id);
    }
  }, [activeInvestigation?.id, loadEvidence, loadHypotheses]);

  const evidenceItems: EvidenceItem[] = useMemo(() => {
    return evidence.map((e: any) => ({
      id: e.id,
      hypothesis_id: e.hypothesis_id,
      hypothesis_title: hypotheses.find((h) => h.id === e.hypothesis_id)?.title,
      claim: e.claim,
      evidence_level: e.evidence_level,
      supports_hypothesis: e.supports_hypothesis,
      metric_name: e.metric_name,
      metric_value: e.metric_value,
      baseline_value: e.baseline_value,
      control_value: e.control_value,
      sample_size: e.sample_size,
      provenance_chain: e.provenance_chain || [],
      created_at: e.created_at,
    }));
  }, [evidence, hypotheses]);

  const hypothesisGroups: HypothesisWithEvidence[] = useMemo(() => {
    return hypotheses.map((hyp) => {
      const hypEvidence = evidenceItems.filter((e) => e.hypothesis_id === hyp.id);
      return {
        hypothesis: hyp,
        supporting: hypEvidence.filter((e) => e.supports_hypothesis),
        contradicting: hypEvidence.filter((e) => !e.supports_hypothesis),
        totalEvidence: hypEvidence.length,
      };
    });
  }, [hypotheses, evidenceItems]);

  const filteredEvidence = useMemo(() => {
    let items = evidenceItems;

    if (searchQuery) {
      const q = searchQuery.toLowerCase();
      items = items.filter(
        (e) =>
          e.claim.toLowerCase().includes(q) ||
          e.hypothesis_title?.toLowerCase().includes(q) ||
          e.metric_name.toLowerCase().includes(q)
      );
    }

    if (filter === 'SUPPORTED') {
      items = items.filter((e) => e.supports_hypothesis);
    } else if (filter === 'CONTRADICTING') {
      items = items.filter((e) => !e.supports_hypothesis);
    } else if (filter === 'LINKED') {
      items = items.filter((e) => e.hypothesis_id);
    } else if (filter === 'UNLINKED') {
      items = items.filter((e) => !e.hypothesis_id);
    }

    return items;
  }, [evidenceItems, filter, searchQuery]);

  const stats = useMemo(() => ({
    total: evidenceItems.length,
    linked: evidenceItems.filter((e) => e.hypothesis_id).length,
    supporting: evidenceItems.filter((e) => e.supports_hypothesis).length,
    contradicting: evidenceItems.filter((e) => !e.supports_hypothesis).length,
    byLevel: {
      CAUSALLY_VERIFIED: evidenceItems.filter((e) => e.evidence_level === 'CAUSALLY_VERIFIED').length,
      SUPPORTED: evidenceItems.filter((e) => e.evidence_level === 'SUPPORTED').length,
      CANDIDATE: evidenceItems.filter((e) => e.evidence_level === 'CANDIDATE').length,
      OBSERVED: evidenceItems.filter((e) => e.evidence_level === 'OBSERVED').length,
      FALSIFIED: evidenceItems.filter((e) => e.evidence_level === 'FALSIFIED').length,
    },
  }), [evidenceItems]);

  const toggleExpand = (id: string) => {
    setExpandedId(expandedId === id ? null : id);
  };

  const formatTimestamp = (ts: number) => {
    return new Date(ts * 1000).toLocaleString();
  };

  return (
    <div style={{ padding: '20px', height: '100%', overflowY: 'auto', boxSizing: 'border-box', backgroundColor: colors.canvas }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 16 }}>
        <div>
          <div style={{ fontSize: 11, fontWeight: 700, color: colors.primary, textTransform: 'uppercase', letterSpacing: 0.8 }}>
            Evidence Explorer
          </div>
          <h2 style={{ margin: '4px 0 0', fontSize: 18, fontWeight: 700, color: colors.ink }}>
            Hypothesis-Evidence Mapping
          </h2>
          <div style={{ fontSize: 12, color: colors.bodyMuted, marginTop: 2 }}>
            Inspect causal findings, track evidence levels, and map evidence to hypotheses.
          </div>
        </div>

        <button
          onClick={() => activeInvestigation?.id && loadEvidence(activeInvestigation.id)}
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

      {/* Stats Overview */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: 10, marginBottom: 16 }}>
        <StatCard label="Total Evidence" value={stats.total} color={colors.ink} />
        <StatCard label="Linked to Hypotheses" value={stats.linked} color={colors.primary} />
        <StatCard label="Supporting" value={stats.supporting} color={colors.success} />
        <StatCard label="Contradicting" value={stats.contradicting} color={colors.danger} />
        <StatCard label="Causally Verified" value={stats.byLevel.CAUSALLY_VERIFIED} color={colors.success} />
        <StatCard label="Falsified" value={stats.byLevel.FALSIFIED} color={colors.danger} />
      </div>

      {/* View Mode Toggle */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 16 }}>
        <button
          onClick={() => setViewMode('by_hypothesis')}
          style={{
            padding: '8px 16px',
            borderRadius: 6,
            border: `1px solid ${viewMode === 'by_hypothesis' ? colors.primary : colors.border}`,
            backgroundColor: viewMode === 'by_hypothesis' ? colors.accentSoft : colors.canvas,
            color: viewMode === 'by_hypothesis' ? colors.primary : colors.bodyMuted,
            fontSize: 12,
            fontWeight: 600,
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: 6,
          }}
        >
          <Brain size={14} /> By Hypothesis
        </button>
        <button
          onClick={() => setViewMode('by_evidence')}
          style={{
            padding: '8px 16px',
            borderRadius: 6,
            border: `1px solid ${viewMode === 'by_evidence' ? colors.primary : colors.border}`,
            backgroundColor: viewMode === 'by_evidence' ? colors.accentSoft : colors.canvas,
            color: viewMode === 'by_evidence' ? colors.primary : colors.bodyMuted,
            fontSize: 12,
            fontWeight: 600,
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: 6,
          }}
        >
          <BarChart3 size={14} /> By Evidence
        </button>
      </div>

      {/* Search and Filter */}
      <div style={{ display: 'flex', gap: 10, marginBottom: 16 }}>
        <div style={{ flex: 1, position: 'relative' }}>
          <Search size={14} style={{ position: 'absolute', left: 10, top: '50%', transform: 'translateY(-50%)', color: colors.bodyMuted }} />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search evidence by claim, hypothesis, or metric..."
            style={{
              width: '100%',
              padding: '8px 12px 8px 32px',
              borderRadius: 6,
              border: `1px solid ${colors.border}`,
              backgroundColor: colors.canvas,
              color: colors.ink,
              fontSize: 12,
              boxSizing: 'border-box',
              outline: 'none',
            }}
          />
        </div>

        <div style={{ display: 'flex', gap: 4 }}>
          {(['ALL', 'SUPPORTED', 'CONTRADICTING', 'LINKED', 'UNLINKED'] as EvidenceFilter[]).map((f) => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              style={{
                padding: '6px 12px',
                borderRadius: 6,
                border: `1px solid ${filter === f ? colors.primary : colors.border}`,
                backgroundColor: filter === f ? colors.accentSoft : colors.canvas,
                color: filter === f ? colors.primary : colors.bodyMuted,
                fontSize: 11,
                fontWeight: 600,
                cursor: 'pointer',
              }}
            >
              {f}
            </button>
          ))}
        </div>
      </div>

      {/* Content */}
      {viewMode === 'by_hypothesis' ? (
        /* Hypothesis-Evidence Mapping View */
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          {hypothesisGroups.length === 0 ? (
            <EmptyState message="No hypotheses found. Create a hypothesis to start mapping evidence." />
          ) : (
            hypothesisGroups.map((group) => (
              <div
                key={group.hypothesis.id}
                style={{
                  border: `1px solid ${colors.border}`,
                  borderRadius: 10,
                  backgroundColor: colors.surfaceTile1,
                  overflow: 'hidden',
                }}
              >
                {/* Hypothesis Header */}
                <div
                  onClick={() => toggleExpand(group.hypothesis.id)}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    padding: '12px 14px',
                    cursor: 'pointer',
                    backgroundColor: expandedId === group.hypothesis.id ? colors.surfacePearl : 'transparent',
                    transition: 'background-color 0.15s ease',
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                    <Brain size={16} style={{ color: colors.primary }} />
                    <div>
                      <div style={{ fontSize: 13, fontWeight: 700, color: colors.ink }}>{group.hypothesis.title}</div>
                      <div style={{ fontSize: 11, color: colors.bodyMuted }}>
                        Target: {group.hypothesis.target_component} | Status: {group.hypothesis.status}
                      </div>
                    </div>
                  </div>

                  <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                    <div style={{ display: 'flex', gap: 8, fontSize: 11 }}>
                      <span style={{ color: colors.successText }}>
                        <CheckCircle2 size={12} style={{ marginRight: 4 }} />
                        {group.supporting.length} supporting
                      </span>
                      <span style={{ color: colors.dangerText }}>
                        <XCircle size={12} style={{ marginRight: 4 }} />
                        {group.contradicting.length} contradicting
                      </span>
                    </div>
                    {expandedId === group.hypothesis.id ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                  </div>
                </div>

                {/* Expanded Evidence List */}
                {expandedId === group.hypothesis.id && (
                  <div style={{ padding: '0 14px 14px', borderTop: `1px solid ${colors.border}` }}>
                    <div style={{ marginTop: 12, marginBottom: 8, fontSize: 12, fontWeight: 600, color: colors.ink }}>
                      Evidence for this Hypothesis ({group.totalEvidence})
                    </div>

                    {group.totalEvidence === 0 ? (
                      <div style={{ padding: 16, textAlign: 'center', color: colors.bodyMuted, backgroundColor: colors.surfacePearl, borderRadius: 6, fontSize: 12 }}>
                        No evidence linked to this hypothesis yet.
                      </div>
                    ) : (
                      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                        {[...group.supporting, ...group.contradicting].map((item) => (
                          <EvidenceCard key={item.id} item={item} />
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>
            ))
          )}
        </div>
      ) : (
        /* Evidence List View */
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          {filteredEvidence.length === 0 ? (
            <EmptyState message="No evidence matches your search criteria." />
          ) : (
            filteredEvidence.map((item) => (
              <EvidenceCard key={item.id} item={item} expanded={expandedId === item.id} onToggle={() => toggleExpand(item.id)} />
            ))
          )}
        </div>
      )}
    </div>
  );
};

/* ── Evidence Card Component ──────────────────────────────────────── */

const EvidenceCard: React.FC<{
  item: EvidenceItem;
  expanded?: boolean;
  onToggle?: () => void;
}> = ({ item, expanded = false, onToggle }) => {
  const levelConfig = EVIDENCE_LEVEL_CONFIG[item.evidence_level] || EVIDENCE_LEVEL_CONFIG.OBSERVED;

  return (
    <div
      onClick={onToggle}
      style={{
        border: `1px solid ${levelConfig.borderColor}`,
        borderRadius: 8,
        backgroundColor: expanded ? levelConfig.bgColor : colors.surfaceTile1,
        padding: 12,
        cursor: onToggle ? 'pointer' : 'default',
        transition: 'all 0.15s ease',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 6 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          {levelConfig.icon}
          <div>
            <div style={{ fontSize: 12, fontWeight: 600, color: colors.ink }}>{item.claim}</div>
            {item.hypothesis_title && (
              <div style={{ fontSize: 11, color: colors.bodyMuted, display: 'flex', alignItems: 'center', gap: 4, marginTop: 2 }}>
                <Link size={10} /> {item.hypothesis_title}
              </div>
            )}
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span
            style={{
              fontSize: 10,
              fontWeight: 700,
              padding: '2px 6px',
              borderRadius: 4,
              backgroundColor: levelConfig.bgColor,
              color: levelConfig.color,
            }}
          >
            {item.evidence_level}
          </span>
          <span
            style={{
              fontSize: 10,
              fontWeight: 700,
              padding: '2px 6px',
              borderRadius: 4,
              backgroundColor: item.supports_hypothesis ? colors.successSoft : colors.dangerSoft,
              color: item.supports_hypothesis ? colors.successText : colors.dangerText,
            }}
          >
            {item.supports_hypothesis ? 'SUPPORTS' : 'CONTRADICTS'}
          </span>
        </div>
      </div>

      {/* Metrics Summary */}
      <div style={{ display: 'flex', gap: 12, fontSize: 11, color: colors.bodyMuted, marginTop: 6 }}>
        <span>
          <b>Metric:</b> {item.metric_name}
        </span>
        <span>
          <b>Value:</b> {item.metric_value.toFixed(4)}
        </span>
        {item.baseline_value !== undefined && (
          <span>
            <b>Baseline:</b> {item.baseline_value.toFixed(4)}
          </span>
        )}
        {item.control_value !== undefined && (
          <span>
            <b>Control:</b> {item.control_value.toFixed(4)}
          </span>
        )}
        <span>
          <b>N:</b> {item.sample_size}
        </span>
      </div>

      {/* Expanded Details */}
      {expanded && (
        <div style={{ marginTop: 10, paddingTop: 10, borderTop: `1px solid ${colors.border}` }}>
          <div style={{ fontSize: 11, fontWeight: 600, color: colors.ink, marginBottom: 6 }}>Provenance Chain</div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap' }}>
            {item.provenance_chain.length > 0 ? (
              item.provenance_chain.map((step, idx) => (
                <React.Fragment key={idx}>
                  <span style={{
                    padding: '3px 8px',
                    borderRadius: 4,
                    backgroundColor: colors.surfacePearl,
                    fontSize: 10,
                    fontWeight: 600,
                    color: colors.ink,
                  }}>
                    {step}
                  </span>
                  {idx < item.provenance_chain.length - 1 && <ArrowRight size={10} style={{ color: colors.bodyMuted }} />}
                </React.Fragment>
              ))
            ) : (
              <span style={{ fontSize: 11, color: colors.bodyMuted }}>No provenance chain recorded</span>
            )}
          </div>

          <div style={{ marginTop: 8, fontSize: 11, color: colors.bodyMuted }}>
            Recorded: {formatTimestamp(item.created_at)}
          </div>
        </div>
      )}
    </div>
  );
};

/* ── Stat Card Component ──────────────────────────────────────────── */

const StatCard: React.FC<{ label: string; value: number; color: string }> = ({ label, value, color }) => (
  <div style={{
    padding: 12,
    borderRadius: 8,
    backgroundColor: colors.surfaceTile1,
    border: `1px solid ${colors.border}`,
    textAlign: 'center',
  }}>
    <div style={{ fontSize: 20, fontWeight: 800, color, fontVariantNumeric: 'tabular-nums' }}>{value}</div>
    <div style={{ fontSize: 10, fontWeight: 600, color: colors.bodyMuted, textTransform: 'uppercase', marginTop: 2 }}>{label}</div>
  </div>
);

/* ── Empty State Component ────────────────────────────────────────── */

const EmptyState: React.FC<{ message: string }> = ({ message }) => (
  <div style={{
    padding: 40,
    textAlign: 'center',
    color: colors.bodyMuted,
    backgroundColor: colors.surfacePearl,
    borderRadius: 10,
  }}>
    <Search size={32} style={{ opacity: 0.3, marginBottom: 8 }} />
    <div style={{ fontSize: 13, fontWeight: 600, color: colors.ink }}>No Evidence Found</div>
    <div style={{ fontSize: 12, marginTop: 4 }}>{message}</div>
  </div>
);

/* ── Helper Function ──────────────────────────────────────────────── */

function formatTimestamp(ts: number): string {
  return new Date(ts * 1000).toLocaleString();
}
