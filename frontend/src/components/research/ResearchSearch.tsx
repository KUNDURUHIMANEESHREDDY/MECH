import React, { useState, useMemo, useEffect } from 'react';
import { useResearchStore } from '../../shared/stores/research';
import { colors } from '../../design/tokens/colors';
import {
  Search,
  X,
  Brain,
  FlaskConical,
  Layers,
  ShieldCheck,
  Database,
  Cpu,
  FileText,
  Clock,
  ArrowRight,
  Filter,
  Wrench,
  Package,
} from 'lucide-react';

type EntityType = 'ALL' | 'HYPOTHESIS' | 'EXPERIMENT' | 'COMPONENT' | 'EVIDENCE' | 'DATASET' | 'MODEL' | 'NOTE' | 'MECHANISM' | 'ARTIFACT';

interface SearchResult {
  id: string;
  title: string;
  type: EntityType;
  subtitle: string;
  status?: string;
  evidenceLevel?: string;
  timestamp?: number;
  source: string;
}

const ENTITY_CONFIG: Record<EntityType, { icon: React.ReactNode; color: string; bgColor: string }> = {
  ALL: { icon: <Search size={14} />, color: colors.bodyMuted, bgColor: colors.surfacePearl },
  HYPOTHESIS: { icon: <Brain size={14} />, color: colors.purpleText, bgColor: colors.purpleSoft },
  EXPERIMENT: { icon: <FlaskConical size={14} />, color: colors.primary, bgColor: colors.accentSoft },
  COMPONENT: { icon: <Layers size={14} />, color: colors.infoText, bgColor: colors.infoSoft },
  EVIDENCE: { icon: <ShieldCheck size={14} />, color: colors.successText, bgColor: colors.successSoft },
  DATASET: { icon: <Database size={14} />, color: colors.warningText, bgColor: colors.warningSoft },
  MODEL: { icon: <Cpu size={14} />, color: colors.dangerText, bgColor: colors.dangerSoft },
  NOTE: { icon: <FileText size={14} />, color: colors.bodyMuted, bgColor: colors.surfacePearl },
  MECHANISM: { icon: <Wrench size={14} />, color: colors.infoText, bgColor: colors.infoSoft },
  ARTIFACT: { icon: <Package size={14} />, color: colors.warningText, bgColor: colors.warningSoft },
};

interface Props {
  isOpen: boolean;
  onClose: () => void;
  onSelect: (result: SearchResult) => void;
}

export const ResearchSearch: React.FC<Props> = ({ isOpen, onClose, onSelect }) => {
  const {
    activeInvestigation,
    hypotheses,
    runs,
    evidence,
    mechanisms,
  } = useResearchStore();

  const [query, setQuery] = useState('');
  const [filterType, setFilterType] = useState<EntityType>('ALL');
  const [selectedIndex, setSelectedIndex] = useState(0);

  const searchIndex: SearchResult[] = useMemo(() => {
    const results: SearchResult[] = [];

    // Hypotheses
    hypotheses.forEach((h) => {
      results.push({
        id: h.id,
        title: h.title,
        type: 'HYPOTHESIS',
        subtitle: h.target_component,
        status: h.status,
        timestamp: h.created_at,
        source: 'research_store',
      });
    });

    // Experiments/Runs
    runs.forEach((r) => {
      results.push({
        id: r.id,
        title: `Experiment ${r.id.slice(-8)}`,
        type: 'EXPERIMENT',
        subtitle: `ΔL = ${r.delta_logit?.toFixed(3) ?? 'N/A'}`,
        status: r.execution_status,
        evidenceLevel: r.used_mock_data ? 'MOCK' : 'LIVE',
        timestamp: r.timestamp,
        source: 'research_store',
      });
    });

    // Components (extracted from hypotheses)
    const componentSet = new Set<string>();
    hypotheses.forEach((h) => {
      if (h.target_component && !componentSet.has(h.target_component)) {
        componentSet.add(h.target_component);
        results.push({
          id: `comp_${h.target_component}`,
          title: h.target_component,
          type: 'COMPONENT',
          subtitle: 'Model Component',
          source: 'extracted',
        });
      }
    });

    // Evidence
    evidence.forEach((e) => {
      results.push({
        id: e.id,
        title: e.claim.slice(0, 60),
        type: 'EVIDENCE',
        subtitle: `${e.metric_name}: ${e.metric_value.toFixed(4)}`,
        evidenceLevel: e.evidence_level,
        timestamp: e.created_at,
        source: 'research_store',
      });
    });

    // Mechanisms
    mechanisms.forEach((m) => {
      results.push({
        id: m.id,
        title: m.name,
        type: 'MECHANISM',
        subtitle: m.description.slice(0, 60),
        status: m.weakest_link_tier,
        timestamp: m.created_at,
        source: 'research_store',
      });
    });

    // Datasets (hardcoded for now, could be dynamic)
    const datasets = ['ioi', 'induction', 'factual', 'semantic', 'syntax'];
    datasets.forEach((ds) => {
      results.push({
        id: `dataset_${ds}`,
        title: ds.toUpperCase(),
        type: 'DATASET',
        subtitle: 'Probing Dataset',
        source: 'catalog',
      });
    });

    // Models (hardcoded for now)
    const models = ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'];
    models.forEach((m) => {
      results.push({
        id: `model_${m}`,
        title: m,
        type: 'MODEL',
        subtitle: 'Transformer Model',
        source: 'registry',
      });
    });

    return results;
  }, [hypotheses, runs, evidence, mechanisms]);

  const filteredResults = useMemo(() => {
    let results = searchIndex;

    if (filterType !== 'ALL') {
      results = results.filter((r) => r.type === filterType);
    }

    if (query) {
      const q = query.toLowerCase();
      results = results.filter(
        (r) =>
          r.title.toLowerCase().includes(q) ||
          r.subtitle.toLowerCase().includes(q) ||
          r.status?.toLowerCase().includes(q) ||
          r.evidenceLevel?.toLowerCase().includes(q)
      );
    }

    return results;
  }, [searchIndex, filterType, query]);

  useEffect(() => {
    setSelectedIndex(0);
  }, [filteredResults]);

  useEffect(() => {
    if (!isOpen) {
      setQuery('');
      setFilterType('ALL');
      setSelectedIndex(0);
    }
  }, [isOpen]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setSelectedIndex((prev) => Math.min(prev + 1, filteredResults.length - 1));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setSelectedIndex((prev) => Math.max(prev - 1, 0));
    } else if (e.key === 'Enter' && filteredResults[selectedIndex]) {
      onSelect(filteredResults[selectedIndex]);
      onClose();
    } else if (e.key === 'Escape') {
      onClose();
    }
  };

  const handleSelect = (result: SearchResult) => {
    onSelect(result);
    onClose();
  };

  const formatTimestamp = (ts?: number) => {
    if (!ts) return '';
    return new Date(ts * 1000).toLocaleDateString();
  };

  if (!isOpen) return null;

  return (
    <div
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        backgroundColor: 'rgba(0, 0, 0, 0.5)',
        display: 'flex',
        alignItems: 'flex-start',
        justifyContent: 'center',
        paddingTop: '15vh',
        zIndex: 1000,
      }}
      onClick={onClose}
    >
      <div
        style={{
          width: 600,
          maxHeight: 500,
          backgroundColor: colors.canvas,
          borderRadius: 12,
          border: `1px solid ${colors.border}`,
          boxShadow: '0 20px 60px rgba(0, 0, 0, 0.3)',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Search Input */}
        <div style={{ padding: 16, borderBottom: `1px solid ${colors.border}` }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <Search size={18} style={{ color: colors.bodyMuted }} />
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Search hypotheses, experiments, components, evidence..."
              autoFocus
              style={{
                flex: 1,
                border: 'none',
                outline: 'none',
                fontSize: 14,
                color: colors.ink,
                backgroundColor: 'transparent',
              }}
            />
            <button
              onClick={onClose}
              style={{
                background: 'none',
                border: 'none',
                cursor: 'pointer',
                color: colors.bodyMuted,
                padding: 4,
              }}
            >
              <X size={16} />
            </button>
          </div>
        </div>

        {/* Filter Tabs */}
        <div style={{ padding: '8px 16px', borderBottom: `1px solid ${colors.border}`, display: 'flex', gap: 4, flexWrap: 'wrap' }}>
          {(['ALL', 'HYPOTHESIS', 'EXPERIMENT', 'COMPONENT', 'EVIDENCE', 'MECHANISM', 'DATASET', 'MODEL'] as EntityType[]).map((type) => {
            const config = ENTITY_CONFIG[type];
            return (
              <button
                key={type}
                onClick={() => setFilterType(type)}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 4,
                  padding: '4px 10px',
                  borderRadius: 12,
                  border: `1px solid ${filterType === type ? config.color : colors.border}`,
                  backgroundColor: filterType === type ? config.bgColor : 'transparent',
                  color: filterType === type ? config.color : colors.bodyMuted,
                  fontSize: 11,
                  fontWeight: 600,
                  cursor: 'pointer',
                }}
              >
                {config.icon}
                {type}
              </button>
            );
          })}
        </div>

        {/* Results */}
        <div style={{ flex: 1, overflowY: 'auto', padding: 8 }}>
          {filteredResults.length === 0 ? (
            <div style={{ padding: 40, textAlign: 'center', color: colors.bodyMuted }}>
              <Search size={32} style={{ opacity: 0.3, marginBottom: 8 }} />
              <div style={{ fontSize: 13, fontWeight: 600, color: colors.ink }}>No Results Found</div>
              <div style={{ fontSize: 12, marginTop: 4 }}>Try a different search term or filter.</div>
            </div>
          ) : (
            filteredResults.map((result, idx) => {
              const config = ENTITY_CONFIG[result.type];
              const isSelected = idx === selectedIndex;

              return (
                <div
                  key={result.id}
                  onClick={() => handleSelect(result)}
                  onMouseEnter={() => setSelectedIndex(idx)}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 12,
                    padding: '10px 12px',
                    borderRadius: 8,
                    backgroundColor: isSelected ? config.bgColor : 'transparent',
                    cursor: 'pointer',
                    transition: 'background-color 0.1s ease',
                  }}
                >
                  <div style={{ color: config.color }}>{config.icon}</div>
                  <div style={{ flex: 1 }}>
                    <div style={{ fontSize: 13, fontWeight: 600, color: colors.ink }}>{result.title}</div>
                    <div style={{ fontSize: 11, color: colors.bodyMuted }}>
                      {result.subtitle}
                      {result.status && ` · ${result.status}`}
                      {result.evidenceLevel && ` · ${result.evidenceLevel}`}
                    </div>
                  </div>
                  {result.timestamp && (
                    <div style={{ fontSize: 10, color: colors.bodyMuted }}>
                      <Clock size={10} style={{ marginRight: 4 }} />
                      {formatTimestamp(result.timestamp)}
                    </div>
                  )}
                  <ArrowRight size={14} style={{ color: colors.bodyMuted }} />
                </div>
              );
            })
          )}
        </div>

        {/* Footer */}
        <div style={{ padding: '10px 16px', borderTop: `1px solid ${colors.border}`, display: 'flex', justifyContent: 'space-between', fontSize: 11, color: colors.bodyMuted }}>
          <span>{filteredResults.length} results</span>
          <span>↑↓ navigate · ↵ select · esc close</span>
        </div>
      </div>
    </div>
  );
};
