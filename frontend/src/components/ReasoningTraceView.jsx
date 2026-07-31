import React, { useState, useMemo, useCallback } from 'react';
import { ArrowDown, ArrowRight, BarChart3, Check, CheckCircle2, ChevronDown, ChevronLeft, ChevronRight, Circle, CircleX, Clock3, Dna, Eye, FileText, FlaskConical, GitBranch, Globe, Lightbulb, ListChecks, Loader2, Network, Ruler, Scale, Search, Target, Upload, X, Zap } from 'lucide-react';

/**
 * ReasoningTraceView - Generic Reasoning Trace Viewer (v2)
 *
 * Features:
 *   1. Clickable nodes -> navigate to Neural Explorer, Circuit Explorer, etc.
 *   2. Time travel: step through reasoning with Prev / Next
 *   3. Live execution status indicators (check / Running / Waiting / Pending)
 *   4. Collapsible multi-level reasoning tree
 *   5. Full-text trace search
 *   6. Export (Markdown / JSON / LaTeX)
 *   7. Registry-based trace type system
 */

// ─── Trace Type Registry ────────────────────────────────────────────────────

const _traceRendererRegistry = {};

function registerTraceRenderer(type, label, icon, renderer) {
  _traceRendererRegistry[type] = { label, icon, renderer };
}

function getRegisteredTypes() {
  return Object.entries(_traceRendererRegistry).map(([id, r]) => ({ id, ...r }));
}

// lucide-react exports icons as forwardRef objects (not plain functions), so
// `typeof icon === 'function'` is false for them. Render any component-like
// value (function or forwardRef) as JSX; pass through strings/other children.
function renderIcon(Icon, size = 13, style) {
  if (typeof Icon === 'function' || (Icon && Icon.$$typeof)) {
    return <Icon size={size} style={style} />;
  }
  return Icon;
}

// ─── Styles ─────────────────────────────────────────────────────────────────

const S = {
  root: {
    padding: 24, background: '#0b0b1a', color: '#e0e0ff',
    height: '100%', overflowY: 'auto', fontFamily: "'Inter','Segoe UI',sans-serif",
  },
  panel: {
    background: '#12122a', borderRadius: 12, border: '1px solid #2a2a4a', padding: 18, minHeight: 100,
  },
  label: {
    fontSize: 11, color: '#888', textTransform: 'uppercase', letterSpacing: 1.2,
    marginBottom: 8, display: 'flex', alignItems: 'center', gap: 6,
  },
  card: (sel) => ({
    background: sel ? '#1a1a3a' : '#151528',
    border: sel ? '1px solid #5cd4c4' : '1px solid #2a2a4a',
    borderRadius: 8, padding: 12, cursor: 'pointer', transition: 'all 0.2s',
  }),
  clickable: {
    cursor: 'pointer', textDecoration: 'underline', textDecorationStyle: 'dotted',
    textUnderlineOffset: 3,
  },
  btn: (active) => ({
    padding: '6px 14px', borderRadius: 6, border: 'none', cursor: 'pointer', fontSize: 12, fontWeight: 600,
    background: active ? '#2a2a5a' : '#151528', color: active ? '#d0c0ff' : '#888', transition: 'all 0.2s',
  }),
  btnPrimary: {
    padding: '6px 14px', borderRadius: 6, border: '1px solid #5cd4c4', cursor: 'pointer',
    fontSize: 12, fontWeight: 600, background: 'transparent', color: '#5cd4c4',
  },
  badge: (color) => ({
    fontSize: 10, fontWeight: 700, padding: '2px 8px', borderRadius: 4,
    background: `${color}22`, color, textTransform: 'uppercase',
  }),
  searchInput: {
    width: '100%', padding: '8px 12px', background: '#1a1a2e', color: '#fff',
    border: '1px solid #3a3a5a', borderRadius: 8, fontSize: 13, outline: 'none',
  },
};

// ─── Confidence Bar ─────────────────────────────────────────────────────────

function ConfBar({ label, value, color = '#5cd4c4' }) {
  return (
    <div style={{ marginBottom: 8 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 3 }}>
        <span style={{ fontSize: 11, color: '#aaa' }}>{label}</span>
        <span style={{ fontSize: 11, color, fontWeight: 600 }}>{Math.round(value * 100)}%</span>
      </div>
      <div style={{ height: 5, borderRadius: 3, background: '#2a2a4a', position: 'relative', overflow: 'hidden' }}>
        <div style={{ position: 'absolute', top: 0, left: 0, height: '100%', width: `${Math.round(value * 100)}%`, background: color, borderRadius: 3, transition: 'width 0.4s' }} />
      </div>
    </div>
  );
}

// ─── Live Status Indicator ──────────────────────────────────────────────────

function StatusDot({ status }) {
  const map = {
    done: { color: '#2ea043', icon: Check, label: 'Done' },
    running: { color: '#feca57', icon: Loader2, label: 'Running...' },
    waiting: { color: '#888', icon: Clock3, label: 'Waiting...' },
    pending: { color: '#555', icon: Circle, label: 'Pending' },
    failed: { color: '#f85149', icon: X, label: 'Failed' },
  };
  const s = map[status] || map.pending;
  return (
    <span style={{ fontSize: 12, color: s.color, fontWeight: 600, display: 'inline-flex', alignItems: 'center', gap: 4 }}>
      <span style={{ fontSize: status === 'running' ? 14 : 12, animation: status === 'running' ? 'pulse 1.5s infinite' : 'none' }}>{renderIcon(s.icon, status === 'running' ? 14 : 12)}</span>
      {s.label}
      <style>{`@keyframes pulse { 0%,100% { opacity:1 } 50% { opacity:0.4 } }`}</style>
    </span>
  );
}

// ─── Clickable Node ─────────────────────────────────────────────────────────

function ClickableNode({ label, target, icon: Icon, onNavigate }) {
  return (
    <span
      onClick={() => onNavigate && onNavigate(target)}
      style={{ ...S.clickable, color: '#5cd4c4', fontSize: 12 }}
      title={`Open ${label}`}
    >
      {renderIcon(Icon, 13)} {label} <ArrowRight size={12} style={{ verticalAlign: 'middle' }} />
    </span>
  );
}

// ─── Collapsible Section ────────────────────────────────────────────────────

function Collapsible({ title, level = 0, defaultOpen = true, children }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div style={{ marginBottom: level === 0 ? 12 : 6, paddingLeft: level * 12 }}>
      <div
        onClick={() => setOpen(p => !p)}
        style={{ cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 6, padding: '4px 0' }}
      >
        <span style={{ color: '#555', display: 'inline-flex' }}>{open ? <ChevronDown size={12} /> : <ChevronRight size={12} />}</span>
        <span style={{ fontSize: 12 + Math.max(0, 2 - level), color: level === 0 ? '#d0c0ff' : '#aaa', fontWeight: level === 0 ? 600 : 400 }}>{title}</span>
      </div>
      {open && <div style={{ paddingLeft: 8, borderLeft: '1px solid #2a2a4a' }}>{children}</div>}
    </div>
  );
}

// ─── Agent Flow (with live status + clickable links) ─────────────────────────

function AgentFlow({ experiments, onNavigate, currentStep }) {
  const agents = [
    { role: 'Observer', icon: Eye, color: '#58a6ff', link: null },
    { role: 'Hypothesis Agent', icon: Lightbulb, color: '#d0c0ff', link: null },
    { role: 'Skeptic', icon: CircleX, color: '#ff6b6b', link: null },
    { role: 'Experiment Planner', icon: FlaskConical, color: '#feca57', link: null },
    { role: 'Statistician', icon: BarChart3, color: '#2ea043', link: 'neuralexplorer' },
    { role: 'Reviewer', icon: Scale, color: '#5cd4c4', link: null },
  ];

  const agentData = agents.map((a, idx) => {
    let description = a.description || '';
    let status = idx <= currentStep ? 'done' : idx === currentStep + 1 ? 'running' : 'pending';

    if (a.role === 'Skeptic') {
      const crit = experiments.find(e => e.type === 'skeptic_critique');
      description = crit?.critique || 'Awaiting...';
    } else if (a.role === 'Experiment Planner') {
      const ce = experiments.find(e => e.type === 'counterexample');
      description = ce ? `Designed: "${ce.prompt}"` : 'Awaiting...';
    } else if (a.role === 'Statistician') {
      const ce = experiments.find(e => e.type === 'counterexample');
      description = ce ? `Expected: ${ce.expected_firing}` : 'Awaiting...';
    } else if (a.role === 'Reviewer') {
      const rv = experiments.find(e => e.type === 'peer_review');
      description = rv?.rationale || 'Awaiting...';
    } else if (a.role === 'Observer') {
      description = 'Collected deterministic feature statistics.';
    } else if (a.role === 'Hypothesis Agent') {
      description = 'Proposed candidate explanations.';
    }
    return { ...a, description, status };
  });

  return (
    <div>
      {agentData.map((agent, i) => (
        <div key={agent.role}>
          <div style={{ background: '#151528', borderLeft: `3px solid ${agent.color}`, borderRadius: '0 8px 8px 0', padding: 12, marginBottom: 8 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
              <span style={{ fontSize: 13, fontWeight: 700, color: agent.color, display: 'inline-flex', alignItems: 'center', gap: 6 }}>{renderIcon(agent.icon, 13)} {agent.role}</span>
              <StatusDot status={agent.status} />
            </div>
            <div style={{ fontSize: 12, color: '#bbb', lineHeight: 1.5 }}>{agent.description}</div>
            {agent.link && onNavigate && (
              <div style={{ marginTop: 6 }}>
                <ClickableNode label="Open in Neural Explorer" target={agent.link} icon={Globe} onNavigate={onNavigate} />
              </div>
            )}
          </div>
          {i < agentData.length - 1 && <div style={{ textAlign: 'center', color: '#3a3a5a', fontSize: 12, marginBottom: 4, display: 'inline-flex', width: '100%', justifyContent: 'center' }}><ArrowDown size={12} /></div>}
        </div>
      ))}
    </div>
  );
}

// ─── Evidence Tree (Collapsible) ────────────────────────────────────────────

function EvidenceTree({ evidence, onNavigate }) {
  if (!evidence || evidence.length === 0) return <div style={{ color: '#666', fontSize: 13 }}>No evidence.</div>;

  const typeConfig = {
    skeptic_critique: { icon: CircleX, color: '#ff6b6b' },
    counterexample: { icon: Target, color: '#feca57' },
    peer_review: { icon: Scale, color: '#5cd4c4' },
    measurement: { icon: BarChart3, color: '#2ea043' },
  };

  return (
    <div style={{ fontSize: 13 }}>
      {evidence.map((ev, i) => {
        const tc = typeConfig[ev.type] || { icon: '•', color: '#888' };
        return (
          <Collapsible key={i} title={<span style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>{renderIcon(tc.icon, 12)} {ev.type.replace(/_/g, ' ')}</span>} level={1} defaultOpen={false}>
            {ev.critique && <div style={{ fontSize: 12, color: '#ccc', marginBottom: 4 }}><strong style={{ color: '#aaa' }}>Critique:</strong> {ev.critique}</div>}
            {ev.alternative && <div style={{ fontSize: 12, color: '#ccc', marginBottom: 4 }}><strong style={{ color: '#aaa' }}>Alt:</strong> {ev.alternative}</div>}
            {ev.prompt && (
              <div style={{ fontFamily: 'monospace', fontSize: 12, background: '#1a1a2e', padding: 6, borderRadius: 4, color: '#fff', marginBottom: 4 }}>{ev.prompt}</div>
            )}
            {ev.expected_firing !== undefined && (
              <div style={{ fontSize: 12, color: '#888' }}>Expected: <span style={{ color: ev.expected_firing ? '#2ea043' : '#f85149' }}>{String(ev.expected_firing)}</span></div>
            )}
            {ev.verdict && (
              <div style={{ fontSize: 12 }}>Verdict: <span style={{ color: ev.verdict === 'accepted' ? '#2ea043' : ev.verdict === 'rejected' ? '#f85149' : '#d29922', fontWeight: 700 }}>{ev.verdict.toUpperCase()}</span></div>
            )}
            {ev.rationale && <div style={{ fontSize: 12, fontStyle: 'italic', color: '#999', marginTop: 4 }}>{ev.rationale}</div>}
            {ev.type === 'counterexample' && onNavigate && (
              <div style={{ marginTop: 6 }}>
                <ClickableNode label="Open Patch Viewer" target="debugger" icon={Zap} onNavigate={onNavigate} />
              </div>
            )}
            {ev.type === 'peer_review' && onNavigate && (
              <div style={{ marginTop: 6 }}>
                <ClickableNode label="Open Circuit Explorer" target="circuitexplorer" icon={Search} onNavigate={onNavigate} />
              </div>
            )}
          </Collapsible>
        );
      })}
    </div>
  );
}

// ─── Counterexample Viewer ──────────────────────────────────────────────────

function CounterexampleViewer({ experiments, onNavigate }) {
  const ce = experiments.find(e => e.type === 'counterexample');
  const review = experiments.find(e => e.type === 'peer_review');
  if (!ce) return <div style={{ color: '#666', fontSize: 13 }}>No counterexamples tested.</div>;

  const passed = review?.verdict === 'accepted';
  const failed = review?.verdict === 'rejected';

  return (
    <div>
      <div style={{ fontFamily: 'monospace', fontSize: 13, background: '#1a1a2e', padding: 8, borderRadius: 4, color: '#fff', marginBottom: 8 }}>{ce.prompt}</div>
      <div style={{ fontSize: 12, color: '#888', marginBottom: 8 }}>
        Expected: <span style={{ color: ce.expected_firing ? '#2ea043' : '#f85149' }}>{String(ce.expected_firing)}</span>
      </div>
      {review && (
        <div style={{
          padding: '8px 12px', borderRadius: 6, fontSize: 12,
          background: failed ? '#2a1a1a' : passed ? '#1a2a1a' : '#2a2a1a',
          border: `1px solid ${failed ? '#f8514944' : passed ? '#2ea04344' : '#d2992244'}`}}>
          <span style={{ fontWeight: 700, color: failed ? '#f85149' : passed ? '#2ea043' : '#d29922', display: 'inline-flex', alignItems: 'center', gap: 4 }}>
            {failed ? <><X size={12} /> FALSIFIED</> : passed ? <><Check size={12} /> SURVIVED</> : '? REVISION NEEDED'}
          </span>
        </div>
      )}
      {onNavigate && (
        <div style={{ marginTop: 8, display: 'flex', gap: 12 }}>
          <ClickableNode label="Neural Explorer" target="neuralexplorer" icon={Globe} onNavigate={onNavigate} />
          <ClickableNode label="Patch Viewer" target="debugger" icon={Zap} onNavigate={onNavigate} />
        </div>
      )}
    </div>
  );
}

// ─── Export Functions ────────────────────────────────────────────────────────

function exportTrace(trace, format) {
  let content, filename, mime;

  if (format === 'json') {
    content = JSON.stringify(trace, null, 2);
    filename = `trace_${trace.id}.json`;
    mime = 'application/json';
  } else if (format === 'markdown') {
    const lines = [`# Reasoning Trace: ${trace.id}`, '', `**Type:** ${trace.type}`, `**Target:** ${trace.observation_id}`, `**Model:** ${trace.llm_model}`, `**Timestamp:** ${trace.timestamp}`, ''];
    lines.push('## Hypotheses', '');
    trace.hypotheses.forEach(h => { lines.push(`- **${h.text}** (Semantic: ${Math.round(h.semantic_confidence * 100)}%, Experimental: ${Math.round(h.experimental_confidence * 100)}%, Status: ${h.status})`); });
    lines.push('', '## Experiments', '');
    trace.experiments.forEach(e => { lines.push(`### ${e.type.replace(/_/g, ' ')}`, ''); if (e.critique) lines.push(`> ${e.critique}`); if (e.prompt) lines.push(`\`${e.prompt}\``); if (e.verdict) lines.push(`**Verdict:** ${e.verdict}`); if (e.rationale) lines.push(`*${e.rationale}*`); lines.push(''); });
    lines.push(`## Final Confidence: ${Math.round(trace.final_confidence * 100)}%`);
    content = lines.join('\n');
    filename = `trace_${trace.id}.md`;
    mime = 'text/markdown';
  } else if (format === 'latex') {
    const lines = [`\\section{Reasoning Trace: ${trace.id}}`, `\\textbf{Target:} ${trace.observation_id} \\\\`, `\\textbf{Model:} ${trace.llm_model} \\\\`, `\\textbf{Type:} ${trace.type} \\\\`, '', '\\subsection{Hypotheses}', '\\begin{itemize}'];
    trace.hypotheses.forEach(h => { lines.push(`  \\item \\textbf{${h.text}} -- Semantic: ${Math.round(h.semantic_confidence * 100)}\\%, Experimental: ${Math.round(h.experimental_confidence * 100)}\\%, Status: ${h.status}`); });
    lines.push('\\end{itemize}', '', '\\subsection{Experiments}');
    trace.experiments.forEach(e => { lines.push(`\\paragraph{${e.type.replace(/_/g, ' ')}}`); if (e.critique) lines.push(`\\textit{${e.critique}}`); if (e.prompt) lines.push(`\\texttt{${e.prompt}}`); if (e.verdict) lines.push(`\\textbf{Verdict:} ${e.verdict}`); });
    lines.push('', `\\subsection{Final Confidence: ${Math.round(trace.final_confidence * 100)}\\%}`);
    content = lines.join('\n');
    filename = `trace_${trace.id}.tex`;
    mime = 'text/x-latex';
  }

  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = filename; a.click();
  URL.revokeObjectURL(url);
}

// ─── Mock Traces ────────────────────────────────────────────────────────────

const MOCK_TRACES = {
  debate: {
    id: 'trace_001', type: 'debate', observation_id: 'Feature_1042',
    timestamp: '2026-07-28 13:42:01', llm_model: 'llama3',
    hypotheses: [
      { id: 'H1', text: 'Fires on French cities', semantic_confidence: 0.72, experimental_confidence: 0.25, replication_score: 0.20, status: 'rejected' },
      { id: 'H2', text: 'Capitalized words ending in "s"', semantic_confidence: 0.15, experimental_confidence: 0.68, replication_score: 0.90, status: 'active' },
      { id: 'H3', text: 'European geography', semantic_confidence: 0.11, experimental_confidence: 0.0, replication_score: 1.0, status: 'pending' },
    ],
    selected_hypothesis_id: 'H1',
    experiments: [
      { type: 'skeptic_critique', critique: '"French cities" is too specific. Many non-French cities end in "s" (Dallas, Athens, Memphis). It may just be syntax.', alternative: 'Capitalized nouns ending in "s".' },
      { type: 'counterexample', id: 'Exp_Adv_H1_0', prompt: 'I flew to Dallas for the weekend.', expected_firing: false },
      { type: 'peer_review', verdict: 'rejected', rationale: 'Model strongly fired on "Dallas" (activation 0.91), directly falsifying the French cities hypothesis.' },
    ],
    final_confidence: 0.14,
  },
  discovery: {
    id: 'trace_002', type: 'discovery', observation_id: 'ACDC Circuit #7',
    timestamp: '2026-07-28 14:01:33', llm_model: 'llama3',
    hypotheses: [
      { id: 'D1', text: 'IOI Name Mover circuit via L9H9 -> L10H0', semantic_confidence: 0.88, experimental_confidence: 0.92, replication_score: 0.95, status: 'accepted' },
    ],
    selected_hypothesis_id: 'D1',
    experiments: [
      { type: 'measurement', description: 'ACDC pruned 94% of edges; remaining circuit recovers 97.2% of clean logit diff.' },
    ],
    final_confidence: 0.93,
  },
  transcoders: {
    id: 'trace_003', type: 'transcoders', observation_id: 'MLP Transcoder L4->L5',
    timestamp: '2026-07-28 14:58:12', llm_model: 'llama3',
    hypotheses: [
      { id: 'TC1', text: 'Sparse feature dictionary explains 95% of MLP L4->L5 transformation', semantic_confidence: 0.92, experimental_confidence: 0.95, replication_score: 0.98, status: 'accepted' },
    ],
    selected_hypothesis_id: 'TC1',
    experiments: [
      { type: 'measurement', description: 'Transcoder dictionary size: 512, Active features: 8, FVE: 95.4%' },
    ],
    final_confidence: 0.95,
  },
  universality: {
    id: 'trace_004', type: 'universality', observation_id: 'GPT-2 vs Gemma Universality',
    timestamp: '2026-07-28 14:59:45', llm_model: 'gemma',
    hypotheses: [
      { id: 'U1', text: 'GPT-2 Feature #182 aligns with Gemma Feature #94 (French Cities)', semantic_confidence: 0.89, experimental_confidence: 0.94, replication_score: 0.96, status: 'accepted' },
    ],
    selected_hypothesis_id: 'U1',
    experiments: [
      { type: 'measurement', description: 'Bipartite cosine similarity score: 0.94 across 8 aligned semantic concepts.' },
    ],
    final_confidence: 0.94,
  },
};

// ─── Register Default Renderers ─────────────────────────────────────────────

registerTraceRenderer('debate', 'Debate', Scale);
registerTraceRenderer('discovery', 'Discovery', Search);
registerTraceRenderer('transcoders', 'Transcoders', Dna);
registerTraceRenderer('universality', 'Universality', Globe);
registerTraceRenderer('runtime', 'Runtime', Zap);
registerTraceRenderer('planner', 'Planner', ListChecks);
registerTraceRenderer('validation', 'Validation', CheckCircle2);
registerTraceRenderer('paper', 'Paper Repro', FileText);

// ─── Main Component ─────────────────────────────────────────────────────────

export default function ReasoningTraceView({ api, onNavigate }) {
  const [activeType, setActiveType] = useState('debate');
  const [selectedHypId, setSelectedHypId] = useState(null);
  const [currentStep, setCurrentStep] = useState(5); // 0-5 for 6 agents
  const [searchQuery, setSearchQuery] = useState('');
  const [showExport, setShowExport] = useState(false);

  const traceTypes = getRegisteredTypes();
  const trace = MOCK_TRACES[activeType] || MOCK_TRACES.debate;
  const selectedHyp = trace.hypotheses.find(h => h.id === (selectedHypId || trace.selected_hypothesis_id));
  const totalSteps = 5;

  const overallConfidence = selectedHyp
    ? Math.min(0.99, (selectedHyp.semantic_confidence * 0.2 + selectedHyp.experimental_confidence * 0.8) * selectedHyp.replication_score)
    : trace.final_confidence;

  // ── Search Filter ──
  const matchesSearch = useCallback((text) => {
    if (!searchQuery) return true;
    return text && text.toLowerCase().includes(searchQuery.toLowerCase());
  }, [searchQuery]);

  const filteredExperiments = useMemo(() => {
    if (!searchQuery) return trace.experiments;
    return trace.experiments.filter(e =>
      matchesSearch(e.critique) || matchesSearch(e.alternative) ||
      matchesSearch(e.prompt) || matchesSearch(e.rationale) ||
      matchesSearch(e.verdict) || matchesSearch(e.type) ||
      matchesSearch(e.description)
    );
  }, [trace.experiments, searchQuery, matchesSearch]);

  const filteredHypotheses = useMemo(() => {
    if (!searchQuery) return trace.hypotheses;
    return trace.hypotheses.filter(h => matchesSearch(h.text) || matchesSearch(h.status));
  }, [trace.hypotheses, searchQuery, matchesSearch]);

  return (
    <div style={S.root}>
      {/* ── Header ── */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
        <div>
          <h1 style={{ fontSize: 22, fontWeight: 700, margin: '0 0 4px 0', color: '#d0c0ff', display: 'flex', alignItems: 'center', gap: 8 }}><FlaskConical size={18} /> Scientific Reasoning</h1>
          <p style={{ margin: 0, color: '#888', fontSize: 13 }}>Auditable Reasoning Traces</p>
        </div>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          {/* Export */}
          <div style={{ position: 'relative' }}>
            <button style={S.btnPrimary} onClick={() => setShowExport(p => !p)}><Upload size={14} style={{ verticalAlign: 'middle', marginRight: 4 }} /> Export</button>
            {showExport && (
              <div style={{ position: 'absolute', top: '110%', right: 0, background: '#1a1a2e', border: '1px solid #3a3a5a', borderRadius: 8, padding: 8, zIndex: 10, display: 'flex', flexDirection: 'column', gap: 4 }}>
                {['json', 'markdown', 'latex'].map(fmt => (
                  <button key={fmt} style={{ ...S.btn(false), textAlign: 'left' }} onClick={() => { exportTrace(trace, fmt); setShowExport(false); }}>
                    {fmt === 'json' ? '{ } JSON' : fmt === 'markdown' ? <><FileText size={12} style={{ verticalAlign: 'middle', marginRight: 4 }} /> Markdown</> : <><Ruler size={12} style={{ verticalAlign: 'middle', marginRight: 4 }} /> LaTeX</>}
                  </button>
                ))}
              </div>
            )}
          </div>
          <div style={{ background: '#1a1a2e', padding: '5px 12px', borderRadius: 8, border: '1px solid #3a3a5a', fontSize: 12 }}>
            <span style={{ color: '#888' }}>LLM: </span><span style={{ color: '#5cd4c4', fontWeight: 600 }}>{trace.llm_model}</span>
          </div>
        </div>
      </div>

      {/* ── Trace Type Tabs (Registry-driven) ── */}
      <div style={{ display: 'flex', gap: 6, marginBottom: 12, flexWrap: 'wrap' }}>
        {traceTypes.map(tt => (
          <button key={tt.id} onClick={() => { setActiveType(tt.id); setSelectedHypId(null); setCurrentStep(5); }} style={S.btn(activeType === tt.id)}>
            {renderIcon(tt.icon, 13, { verticalAlign: 'middle', marginRight: 4 })} {tt.label}
          </button>
        ))}
      </div>

      {/* ── Search Bar ── */}
      <div style={{ marginBottom: 16 }}>
        <input
          style={S.searchInput}
          placeholder="Search traces: hypotheses, experiments, neurons, circuits..."
          value={searchQuery}
          onChange={e => setSearchQuery(e.target.value)}
        />
      </div>

      {/* ── Time Travel Controls ── */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16, padding: '8px 14px', background: '#12122a', borderRadius: 10, border: '1px solid #2a2a4a' }}>
        <button style={S.btn(false)} onClick={() => setCurrentStep(Math.max(0, currentStep - 1))} disabled={currentStep <= 0}><ChevronLeft size={14} style={{ verticalAlign: 'middle', marginRight: 4 }} />Prev</button>
        <div style={{ flex: 1, display: 'flex', gap: 4, alignItems: 'center' }}>
          {['Observer', 'Hypothesis', 'Skeptic', 'Planner', 'Statistician', 'Reviewer'].map((step, i) => (
            <div key={step} onClick={() => setCurrentStep(i)} style={{
              flex: 1, height: 6, borderRadius: 3, cursor: 'pointer',
              background: i <= currentStep ? '#5cd4c4' : '#2a2a4a', transition: 'background 0.3s'}} title={step} />
          ))}
        </div>
        <span style={{ fontSize: 12, color: '#888', minWidth: 80, textAlign: 'center' }}>
          Step {currentStep + 1}/6
        </span>
        <button style={S.btn(false)} onClick={() => setCurrentStep(Math.min(totalSteps, currentStep + 1))} disabled={currentStep >= totalSteps}>Next <ChevronRight size={14} style={{ verticalAlign: 'middle', marginLeft: 4 }} /></button>
      </div>

      {/* ── 4-Column Layout ── */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1.4fr 1fr 1fr', gap: 14, alignItems: 'start' }}>

        {/* Col 1: Observation */}
        <div style={S.panel}>
          <div style={S.label}><span><Eye size={13} /></span> Observation</div>
          <div style={{ fontSize: 17, fontWeight: 700, color: '#fff', marginBottom: 6 }}>{trace.observation_id}</div>
          <div style={{ fontSize: 12, color: '#888' }}>{trace.timestamp}</div>
          <div style={{ fontSize: 12, color: '#555', marginTop: 10 }}>Type: <span style={{ color: '#d0c0ff' }}>{trace.type}</span></div>
          {onNavigate && (
            <div style={{ marginTop: 12, display: 'flex', flexDirection: 'column', gap: 6 }}>
              <ClickableNode label="Neural Explorer" target="neuralexplorer" icon={Globe} onNavigate={onNavigate} />
              <ClickableNode label="Knowledge Graph" target="knowledgegraph" icon={Network} onNavigate={onNavigate} />
            </div>
          )}
        </div>

        {/* Col 2: Hypotheses */}
        <div style={S.panel}>
          <div style={S.label}><span><Lightbulb size={13} /></span> Hypotheses</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {filteredHypotheses.map(h => {
              const isSel = h.id === (selectedHypId || trace.selected_hypothesis_id);
              const statusColor = h.status === 'accepted' ? '#2ea043' : h.status === 'rejected' ? '#f85149' : h.status === 'active' ? '#feca57' : '#555';
              return (
                <div key={h.id} style={S.card(isSel)} onClick={() => setSelectedHypId(h.id)}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
                    <span style={{ fontSize: 13, color: '#fff', fontWeight: 600, flex: 1 }}>"{h.text}"</span>
                    <span style={S.badge(statusColor)}>{h.status}</span>
                  </div>
                  <ConfBar label="Semantic" value={h.semantic_confidence} color="#d0c0ff" />
                  <ConfBar label="Experimental" value={h.experimental_confidence} color="#5cd4c4" />
                  <ConfBar label="Replication" value={h.replication_score} color="#feca57" />
                  <Collapsible title="Evidence" level={1} defaultOpen={false}>
                    <EvidenceTree evidence={trace.experiments} onNavigate={onNavigate} />
                  </Collapsible>
                </div>
              );
            })}
          </div>
        </div>

        {/* Col 3: Agent Flow */}
        <div style={S.panel}>
          <div style={S.label}><span><FlaskConical size={13} /></span> Agents & Experiments</div>
          {trace.type === 'debate' ? (
            <AgentFlow experiments={filteredExperiments} onNavigate={onNavigate} currentStep={currentStep} />
          ) : (
            <EvidenceTree evidence={filteredExperiments} onNavigate={onNavigate} />
          )}
        </div>

        {/* Col 4: Verdict + Confidence + Counterexample */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          {/* Verdict */}
          <div style={S.panel}>
            <div style={S.label}><span><Scale size={13} /></span> Verdict</div>
            {trace.experiments.find(e => e.type === 'peer_review') ? (() => {
              const rv = trace.experiments.find(e => e.type === 'peer_review');
              const vc = rv.verdict === 'accepted' ? '#2ea043' : rv.verdict === 'rejected' ? '#f85149' : '#d29922';
              return (
                <div style={{ background: rv.verdict === 'accepted' ? '#1a2a1a' : rv.verdict === 'rejected' ? '#2a1a1a' : '#2a2a1a', border: `1px solid ${vc}44`, borderRadius: 10, padding: 14, textAlign: 'center' }}>
                  <div style={{ fontSize: 16, fontWeight: 700, marginBottom: 4, color: vc }}>{rv.verdict.toUpperCase()}</div>
                  <div style={{ fontSize: 12, color: '#bbb', lineHeight: 1.5 }}>{rv.rationale}</div>
                  {onNavigate && (
                    <div style={{ marginTop: 8 }}><ClickableNode label="Circuit Explorer" target="circuitexplorer" icon={Search} onNavigate={onNavigate} /></div>
                  )}
                </div>
              );
            })() : (
              <div style={{ background: '#1a2a1a', border: '1px solid #2ea04344', borderRadius: 10, padding: 14, textAlign: 'center' }}>
                <div style={{ fontSize: 16, fontWeight: 700, color: '#2ea043' }}>ACCEPTED</div>
                <div style={{ fontSize: 12, color: '#bbb' }}>No falsification attempted.</div>
              </div>
            )}
          </div>

          {/* Confidence Matrix */}
          <div style={S.panel}>
            <div style={S.label}><span><BarChart3 size={13} /></span> Confidence Matrix</div>
            {selectedHyp && (
              <>
                <ConfBar label="Semantic" value={selectedHyp.semantic_confidence} color="#d0c0ff" />
                <ConfBar label="Experimental" value={selectedHyp.experimental_confidence} color="#5cd4c4" />
                <ConfBar label="Replication" value={selectedHyp.replication_score} color="#feca57" />
                <div style={{ borderTop: '1px solid #2a2a4a', paddingTop: 8, marginTop: 4 }}>
                  <ConfBar label="Overall" value={overallConfidence} color={overallConfidence > 0.5 ? '#2ea043' : '#f85149'} />
                </div>
              </>
            )}
          </div>

          {/* Counterexample */}
          <div style={S.panel}>
            <div style={S.label}><span><Target size={13} /></span> Counterexamples</div>
            <CounterexampleViewer experiments={trace.experiments} onNavigate={onNavigate} />
          </div>
        </div>
      </div>

      {/* ── Full Evidence Tree (collapsible, full width) ── */}
      <div style={{ ...S.panel, marginTop: 14 }}>
        <Collapsible title={<span style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}><GitBranch size={13} /> Full Evidence Tree</span>} level={0} defaultOpen={false}>
          <Collapsible title={`Observation: ${trace.observation_id}`} level={1} defaultOpen={true}>
            {trace.hypotheses.map(h => (
              <Collapsible key={h.id} title={`Hypothesis: "${h.text}" [${h.status}]`} level={2} defaultOpen={false}>
                {trace.experiments.map((e, i) => (
                  <Collapsible key={i} title={`${e.type.replace(/_/g, ' ')} ${e.verdict ? `-> ${e.verdict.toUpperCase()}` : ''}`} level={3} defaultOpen={false}>
                    {e.critique && <div style={{ fontSize: 12, color: '#ccc' }}>{e.critique}</div>}
                    {e.prompt && <div style={{ fontFamily: 'monospace', fontSize: 12, color: '#fff', background: '#1a1a2e', padding: 4, borderRadius: 4, marginTop: 4 }}>{e.prompt}</div>}
                    {e.rationale && <div style={{ fontSize: 12, fontStyle: 'italic', color: '#999', marginTop: 4 }}>{e.rationale}</div>}
                  </Collapsible>
                ))}
              </Collapsible>
            ))}
          </Collapsible>
        </Collapsible>
      </div>
    </div>
  );
}

