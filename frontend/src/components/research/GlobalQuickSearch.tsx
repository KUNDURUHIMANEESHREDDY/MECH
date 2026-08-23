import React, { useState, useEffect } from 'react';
import {
  Search,
  X,
  Target,
  FlaskConical,
  FileCheck,
  Layers,
  Award,
  ArrowRight,
  BookOpen,
} from 'lucide-react';
import { colors } from '../../design/tokens/colors';
import { useResearchStore } from '../../shared/stores/research';

interface SearchResult {
  id: string;
  title: string;
  type: 'COMPONENT' | 'HYPOTHESIS' | 'EXPERIMENT' | 'EVIDENCE' | 'MECHANISM' | 'BENCHMARK';
  category: string;
  meta: string;
}

interface Props {
  isOpen: boolean;
  onClose: () => void;
  onSelect: (result: SearchResult) => void;
}

export const GlobalQuickSearch: React.FC<Props> = ({ isOpen, onClose, onSelect }) => {
  const [query, setQuery] = useState('');

  const searchIndex: SearchResult[] = [
    {
      id: 'comp_l9h9',
      title: 'L9H9 (Name Mover Head)',
      type: 'COMPONENT',
      category: 'Attention Head',
      meta: 'GPT-2 Layer 9 Head 9 · Δlogit = +1.85 · Causal Mover',
    },
    {
      id: 'comp_l8h1',
      title: 'L8H1 (Duplicate Token Head)',
      type: 'COMPONENT',
      category: 'Attention Head',
      meta: 'GPT-2 Layer 8 Head 1 · Δlogit = +2.13 · Signal Inhibitor',
    },
    {
      id: 'comp_l5h5',
      title: 'L5H5 (Induction Head)',
      type: 'COMPONENT',
      category: 'Attention Head',
      meta: 'GPT-2 Layer 5 Head 5 · Prefix Attn = 0.62',
    },
    {
      id: 'hyp_1',
      title: 'H1: L9H9 Name Mover Mediation',
      type: 'HYPOTHESIS',
      category: 'Hypothesis',
      meta: 'Active Hypothesis · Status: SUPPORTED',
    },
    {
      id: 'exp_42',
      title: 'E42: L9H9 Zero-Ablation Treatment',
      type: 'EXPERIMENT',
      category: 'Causal Experiment',
      meta: 'Clean vs Corrupted · Target: Mary · Replicated (N=3)',
    },
    {
      id: 'evi_17',
      title: 'EVID-17: Causal Logit Degradation',
      type: 'EVIDENCE',
      category: 'Empirical Evidence',
      meta: 'Cohen\'s d = 3.42 · Negative Control L0H0 isolated',
    },
    {
      id: 'mech_ioi',
      title: 'M1: Dual Name-Mover & Duplicate Token Circuit',
      type: 'MECHANISM',
      category: 'Circuit Mechanism',
      meta: 'v2 · Components: L8H1, L9H9, Residual Stream',
    },
    {
      id: 'bench_ioi',
      title: 'Literature Benchmark: IOI Name Mover Circuit',
      type: 'BENCHMARK',
      category: 'Literature Reference',
      meta: 'Wang et al. (2022) · arXiv:2211.00593 · Status: CAUSAL_MATCH',
    },
    {
      id: 'bench_ind',
      title: 'Literature Benchmark: In-Context Induction Heads',
      type: 'BENCHMARK',
      category: 'Literature Reference',
      meta: 'Olsson et al. (2022) · Anthropic Circuits · Status: COMPONENT_MATCH',
    },
  ];

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        if (isOpen) onClose();
      }
      if (e.key === 'Escape' && isOpen) {
        onClose();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  const filtered = searchIndex.filter((item) =>
    item.title.toLowerCase().includes(query.toLowerCase()) ||
    item.meta.toLowerCase().includes(query.toLowerCase()) ||
    item.type.toLowerCase().includes(query.toLowerCase())
  );

  const getTypeIcon = (type: SearchResult['type']) => {
    switch (type) {
      case 'COMPONENT':
        return <Target size={14} color={colors.primary} />;
      case 'HYPOTHESIS':
        return <FlaskConical size={14} color={colors.warningText || colors.primary} />;
      case 'EXPERIMENT':
        return <Layers size={14} color={colors.secondary} />;
      case 'EVIDENCE':
        return <FileCheck size={14} color={colors.successText} />;
      case 'MECHANISM':
        return <Layers size={14} color={colors.primary} />;
      case 'BENCHMARK':
        return <Award size={14} color={colors.accent || colors.primary} />;
    }
  };

  return (
    <div
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        backgroundColor: 'rgba(0, 0, 0, 0.65)',
        display: 'flex',
        alignItems: 'flex-start',
        justifyContent: 'center',
        paddingTop: 80,
        zIndex: 2000,
      }}
      onClick={onClose}
    >
      <div
        style={{
          width: '100%',
          maxWidth: 620,
          backgroundColor: colors.canvas,
          borderRadius: 8,
          border: `1px solid ${colors.hairline}`,
          boxShadow: '0 16px 40px rgba(0,0,0,0.5)',
          overflow: 'hidden',
          display: 'flex',
          flexDirection: 'column',
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Search Input Bar */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 10,
            padding: '12px 16px',
            borderBottom: `1px solid ${colors.hairline}`,
          }}
        >
          <Search size={18} color={colors.bodyMuted} />
          <input
            type="text"
            placeholder="Search components (L9H9), hypotheses (H1), experiments (E42), evidence, mechanisms..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            autoFocus
            style={{
              flex: 1,
              backgroundColor: 'transparent',
              border: 'none',
              outline: 'none',
              color: colors.ink,
              fontSize: 13,
            }}
          />
          <div
            style={{
              padding: '2px 6px',
              borderRadius: 4,
              backgroundColor: colors.surfaceTile1,
              border: `1px solid ${colors.hairline}`,
              fontSize: 10,
              color: colors.bodyMuted,
              fontFamily: 'monospace',
            }}
          >
            ESC
          </div>
        </div>

        {/* Results List */}
        <div style={{ maxHeight: 380, overflowY: 'auto', padding: 8, display: 'flex', flexDirection: 'column', gap: 4 }}>
          {filtered.length === 0 ? (
            <div style={{ padding: 20, textAlign: 'center', color: colors.bodyMuted, fontSize: 12 }}>
              No matching scientific objects found for "{query}".
            </div>
          ) : (
            filtered.map((res) => (
              <div
                key={res.id}
                onClick={() => {
                  onSelect(res);
                  onClose();
                }}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  padding: '8px 12px',
                  borderRadius: 6,
                  cursor: 'pointer',
                  backgroundColor: colors.surfaceTile1,
                  border: `1px solid transparent`,
                  transition: 'background-color 0.15s ease',
                }}
                onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = colors.surfaceTile2)}
                onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = colors.surfaceTile1)}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                  <div
                    style={{
                      width: 26,
                      height: 26,
                      borderRadius: 4,
                      backgroundColor: colors.surfaceTile2,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                    }}
                  >
                    {getTypeIcon(res.type)}
                  </div>
                  <div>
                    <div style={{ fontSize: 12, fontWeight: 700, color: colors.ink }}>{res.title}</div>
                    <div style={{ fontSize: 11, color: colors.bodyMuted }}>{res.meta}</div>
                  </div>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <span
                    style={{
                      fontSize: 9,
                      fontWeight: 700,
                      padding: '2px 6px',
                      borderRadius: 3,
                      backgroundColor: colors.surfaceTile2,
                      color: colors.bodyMuted,
                    }}
                  >
                    {res.type}
                  </span>
                  <ArrowRight size={14} color={colors.bodyMuted} />
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
};
