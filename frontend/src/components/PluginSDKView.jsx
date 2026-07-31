import { useState, useEffect, useCallback } from 'react';
import { BarChart3, BookOpen, Building2, Check, ChevronDown, ChevronUp, ClipboardPen, Gamepad2, Microscope, Network, Plug, Plus, Rocket, Search } from 'lucide-react';

/* ------------------------------------------------------------------ */
/* Static mock data — replace with API calls when backend is wired up  */
/* ------------------------------------------------------------------ */
const MOCK_PLUGINS = [
  {
    plugin_id: 'mech.example.ioi_experiment_logger',
    name: 'IOI Experiment Logger',
    version: '1.0.0',
    author: 'MECH Platform Team',
    description: 'Logs every IOI-related experiment plan, campaign event, and paper to a local JSONL audit trail.',
    hooks: ['on_experiment_planned', 'on_campaign_started', 'on_campaign_completed', 'on_paper_generated', 'on_validation_completed'],
    status: 'active',
    events_logged: 42,
    last_triggered: '2026-07-28T10:41:17Z',
    ui_view: null,
    source: 'built-in',
  },
];

const ALL_HOOKS = [
  { id: 'on_experiment_planned',      label: 'Discovery Planner',           icon: Search },
  { id: 'on_campaign_started',        label: 'Campaign Manager (Start)',     icon: Rocket },
  { id: 'on_campaign_completed',      label: 'Campaign Manager (Complete)',  icon: Check },
  { id: 'on_belief_updated',          label: 'Bayesian Belief Engine',       icon: BarChart3 },
  { id: 'on_evidence_fused',          label: 'Evidence Fusion',              icon: Building2 },
  { id: 'on_knowledge_graph_updated', label: 'Knowledge Graph',              icon: Network },
  { id: 'on_paper_generated',         label: 'Publication Engine',           icon: ClipboardPen },
  { id: 'on_validation_completed',    label: 'Validation Framework',         icon: Microscope },
];

const HOOK_DOCS = {
  on_experiment_planned:      'Mutating hook — return a modified plan dict to alter the experiment before execution, or None to pass through.',
  on_campaign_started:        'Notification hook — called when a Research Campaign is initialised.',
  on_campaign_completed:      'Notification hook — called with full campaign summary on completion.',
  on_belief_updated:          'Notification hook — fired after each Bayesian posterior update.',
  on_evidence_fused:          'Notification hook — fires when the Evidence Fusion Engine emits a bundle.',
  on_knowledge_graph_updated: 'Notification hook — fires on every KG node/edge addition.',
  on_paper_generated:         'Mutating hook — return a modified abstract string to enrich the paper, or None to pass through.',
  on_validation_completed:    'Notification hook — fires after every Continuous Validation run.',
};

const STATUS_COLOR = {
  active:   { bg: 'rgba(34,197,94,0.15)',  border: '#22c55e', text: '#4ade80' },
  inactive: { bg: 'rgba(100,116,139,0.15)', border: '#64748b', text: '#94a3b8' },
  error:    { bg: 'rgba(239,68,68,0.15)',  border: '#ef4444', text: '#f87171' },
};

/* ------------------------------------------------------------------ */
/* Sub-components                                                       */
/* ------------------------------------------------------------------ */

function HookBadge({ hookId, active }) {
  const hook = ALL_HOOKS.find(h => h.id === hookId);
  if (!hook) return null;
  return (
    <span title={HOOK_DOCS[hookId]} style={{
      display: 'inline-flex', alignItems: 'center', gap: 4,
      padding: '2px 8px', borderRadius: 12, fontSize: 11, fontWeight: 600,
      background: active ? 'rgba(99,102,241,0.2)' : 'rgba(100,116,139,0.1)',
      border: `1px solid ${active ? '#6366f1' : '#334155'}`,
      color: active ? '#a5b4fc' : '#64748b',
      cursor: 'help', transition: 'all 0.2s'}}>
      {typeof hook.icon === 'function' ? <hook.icon size={13} /> : hook.icon} {hook.label}
    </span>
  );
}

function PluginCard({ plugin, onToggle, onUnload }) {
  const [expanded, setExpanded] = useState(false);
  const sc = STATUS_COLOR[plugin.status] || STATUS_COLOR.inactive;

  return (
    <div style={{
      background: 'rgba(15,23,42,0.7)', border: `1px solid ${sc.border}`,
      borderRadius: 14, padding: '20px 24px', marginBottom: 16,
      boxShadow: `0 0 18px ${sc.border}22`, transition: 'box-shadow 0.3s'}}>
      {/* Header row */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 14, marginBottom: 12 }}>
        <div style={{
          width: 42, height: 42, borderRadius: 10,
          background: 'linear-gradient(135deg,#6366f1,#8b5cf6)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          fontSize: 22, flexShrink: 0}}><Plug size={22} /></div>

        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
            <span style={{ fontWeight: 700, fontSize: 16, color: '#f1f5f9' }}>{plugin.name}</span>
            <span style={{ fontSize: 11, color: '#64748b', fontFamily: 'monospace' }}>v{plugin.version}</span>
            <span style={{
              fontSize: 11, fontWeight: 700, padding: '1px 8px', borderRadius: 8,
              background: sc.bg, border: `1px solid ${sc.border}`, color: sc.text}}>{plugin.status.toUpperCase()}</span>
            {plugin.source === 'built-in' && (
              <span style={{ fontSize: 10, color: '#475569', padding: '1px 6px', borderRadius: 6, border: '1px solid #334155' }}>BUILT-IN</span>
            )}
          </div>
          <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>
            {plugin.plugin_id} &nbsp;·&nbsp; by {plugin.author}
          </div>
        </div>

        {/* Controls */}
        <div style={{ display: 'flex', gap: 8, flexShrink: 0 }}>
          <button
            onClick={() => onToggle(plugin.plugin_id)}
            style={{
              padding: '5px 14px', borderRadius: 8, fontWeight: 600, fontSize: 12, cursor: 'pointer',
              border: '1px solid',
              ...(plugin.status === 'active'
                ? { background: 'rgba(234,179,8,0.1)', borderColor: '#eab308', color: '#facc15' }
                : { background: 'rgba(34,197,94,0.1)', borderColor: '#22c55e', color: '#4ade80' })}}
          >{plugin.status === 'active' ? 'Disable' : 'Enable'}</button>
          {plugin.source !== 'built-in' && (
            <button
              onClick={() => onUnload(plugin.plugin_id)}
              style={{
                padding: '5px 14px', borderRadius: 8, fontWeight: 600, fontSize: 12, cursor: 'pointer',
                background: 'rgba(239,68,68,0.1)', border: '1px solid #ef4444', color: '#f87171'}}
            >Unload</button>
          )}
          <button
            onClick={() => setExpanded(e => !e)}
            style={{
              padding: '5px 10px', borderRadius: 8, fontSize: 12, cursor: 'pointer',
              background: 'rgba(99,102,241,0.1)', border: '1px solid #4f46e5', color: '#818cf8'}}
          >{expanded ? <ChevronUp size={13} /> : <ChevronDown size={13} />}</button>
        </div>
      </div>

      {/* Description */}
      <p style={{ fontSize: 13, color: '#94a3b8', margin: '0 0 12px', lineHeight: 1.6 }}>
        {plugin.description}
      </p>

      {/* Hook badges */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 12 }}>
        {ALL_HOOKS.map(h => (
          <HookBadge key={h.id} hookId={h.id} active={plugin.hooks.includes(h.id)} />
        ))}
      </div>

      {/* Stats */}
      <div style={{ display: 'flex', gap: 24, fontSize: 12 }}>
        <span style={{ color: '#64748b' }}>Events logged: <b style={{ color: '#a5b4fc' }}>{plugin.events_logged}</b></span>
        {plugin.last_triggered && (
          <span style={{ color: '#64748b' }}>Last triggered: <b style={{ color: '#94a3b8' }}>
            {new Date(plugin.last_triggered).toLocaleString()}
          </b></span>
        )}
        {plugin.ui_view && (
          <span style={{ color: '#64748b' }}>UI Panel: <b style={{ color: '#34d399' }}>{plugin.ui_view}</b></span>
        )}
      </div>

      {/* Expanded: full hook docs */}
      {expanded && (
        <div style={{
          marginTop: 16, padding: 16, borderRadius: 10,
          background: 'rgba(30,41,59,0.6)', border: '1px solid #1e293b'}}>
          <div style={{ fontWeight: 700, fontSize: 12, color: '#64748b', marginBottom: 10, textTransform: 'uppercase', letterSpacing: 1 }}>
            Hook Documentation
          </div>
          {plugin.hooks.map(h => (
            <div key={h} style={{ marginBottom: 8 }}>
              <code style={{ color: '#818cf8', fontSize: 12 }}>{h}(payload)</code>
              <p style={{ fontSize: 12, color: '#94a3b8', margin: '2px 0 0 12px' }}>{HOOK_DOCS[h]}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function HookMatrixPanel() {
  return (
    <div style={{
      background: 'rgba(15,23,42,0.7)', border: '1px solid #1e293b',
      borderRadius: 14, padding: '20px 24px'}}>
      <div style={{ fontWeight: 700, fontSize: 14, color: '#f1f5f9', marginBottom: 16, display: 'flex', alignItems: 'center', gap: 6 }}>
        <Gamepad2 size={16} /> Platform Hook Integration Matrix
      </div>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead>
            <tr>
              <th style={{ textAlign: 'left', color: '#64748b', padding: '6px 12px', borderBottom: '1px solid #1e293b' }}>Hook</th>
              <th style={{ textAlign: 'left', color: '#64748b', padding: '6px 12px', borderBottom: '1px solid #1e293b' }}>Platform Component</th>
              <th style={{ textAlign: 'center', color: '#64748b', padding: '6px 12px', borderBottom: '1px solid #1e293b' }}>Type</th>
              <th style={{ textAlign: 'left', color: '#64748b', padding: '6px 12px', borderBottom: '1px solid #1e293b' }}>Description</th>
            </tr>
          </thead>
          <tbody>
            {ALL_HOOKS.map((h, i) => {
              const isMutating = ['on_experiment_planned', 'on_paper_generated'].includes(h.id);
              return (
                <tr key={h.id} style={{ background: i % 2 === 0 ? 'rgba(30,41,59,0.3)' : 'transparent' }}>
                  <td style={{ padding: '8px 12px', fontFamily: 'monospace', color: '#818cf8' }}>{h.id}</td>
                  <td style={{ padding: '8px 12px', color: '#94a3b8' }}>{typeof h.icon === 'function' ? <h.icon size={13} style={{ verticalAlign: 'middle', marginRight: 4 }} /> : h.icon} {h.label}</td>
                  <td style={{ padding: '8px 12px', textAlign: 'center' }}>
                    <span style={{
                      padding: '1px 8px', borderRadius: 8, fontWeight: 600, fontSize: 10,
                      ...(isMutating
                        ? { background: 'rgba(251,146,60,0.15)', border: '1px solid #f97316', color: '#fb923c' }
                        : { background: 'rgba(99,102,241,0.15)', border: '1px solid #6366f1', color: '#a5b4fc' })}}>{isMutating ? 'MUTATING' : 'NOTIFY'}</span>
                  </td>
                  <td style={{ padding: '8px 12px', color: '#64748b' }}>{HOOK_DOCS[h.id]}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function LoadPluginPanel({ onLoad }) {
  const [path, setPath] = useState('');
  const [status, setStatus] = useState(null);

  function handleLoad() {
    if (!path.trim()) return;
    setStatus({ type: 'loading', msg: `Loading ${path}…` });
    setTimeout(() => {
      setStatus({ type: 'success', msg: `Plugin '${path}' loaded successfully.` });
      onLoad(path);
      setPath('');
    }, 900);
  }

  return (
    <div style={{
      background: 'rgba(15,23,42,0.7)', border: '1px solid #1e293b',
      borderRadius: 14, padding: '20px 24px'}}>
      <div style={{ fontWeight: 700, fontSize: 14, color: '#f1f5f9', marginBottom: 14, display: 'flex', alignItems: 'center', gap: 6 }}>
        <Plus size={14} /> Load External Plugin
      </div>
      <div style={{ fontSize: 12, color: '#64748b', marginBottom: 12 }}>
        Enter a dotted Python module path (e.g. <code style={{ color: '#818cf8' }}>mylab.ioi_extension</code>) or an absolute path to a <code style={{ color: '#818cf8' }}>.py</code> file.
      </div>
      <div style={{ display: 'flex', gap: 10 }}>
        <input
          value={path}
          onChange={e => setPath(e.target.value)}
          placeholder="mylab.my_plugin  or  /abs/path/plugin.py"
          style={{
            flex: 1, padding: '8px 14px', borderRadius: 8, fontSize: 13,
            background: '#0f172a', border: '1px solid #334155', color: '#f1f5f9',
            outline: 'none'}}
        />
        <button
          onClick={handleLoad}
          disabled={!path.trim()}
          style={{
            padding: '8px 20px', borderRadius: 8, fontWeight: 700, fontSize: 13, cursor: 'pointer',
            background: 'linear-gradient(135deg,#6366f1,#8b5cf6)', border: 'none', color: '#fff',
            opacity: path.trim() ? 1 : 0.5}}
        >Load</button>
      </div>
      {status && (
        <div style={{
          marginTop: 10, padding: '8px 12px', borderRadius: 8, fontSize: 12,
          background: status.type === 'success' ? 'rgba(34,197,94,0.1)' : 'rgba(99,102,241,0.1)',
          border: `1px solid ${status.type === 'success' ? '#22c55e' : '#6366f1'}`,
          color: status.type === 'success' ? '#4ade80' : '#a5b4fc'}}>{status.msg}</div>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Main view                                                            */
/* ------------------------------------------------------------------ */
export default function PluginSDKView() {
  const [plugins, setPlugins] = useState(MOCK_PLUGINS);
  const [tab, setTab] = useState('plugins');
  const [search, setSearch] = useState('');

  const activeCount  = plugins.filter(p => p.status === 'active').length;
  const totalHooks   = plugins.reduce((sum, p) => sum + p.hooks.length, 0);
  const totalEvents  = plugins.reduce((sum, p) => sum + p.events_logged, 0);

  const handleToggle = useCallback((pid) => {
    setPlugins(prev => prev.map(p =>
      p.plugin_id === pid
        ? { ...p, status: p.status === 'active' ? 'inactive' : 'active' }
        : p
    ));
  }, []);

  const handleUnload = useCallback((pid) => {
    setPlugins(prev => prev.filter(p => p.plugin_id !== pid));
  }, []);

  const handleLoad = useCallback((path) => {
    const id = `external.${path.split('.').pop()}.${Date.now()}`;
    setPlugins(prev => [...prev, {
      plugin_id: id,
      name: path.split('.').pop(),
      version: '0.0.0',
      author: 'External',
      description: `Loaded from: ${path}`,
      hooks: [],
      status: 'active',
      events_logged: 0,
      last_triggered: null,
      ui_view: null,
      source: 'external',
    }]);
  }, []);

  const filtered = plugins.filter(p =>
    !search || p.name.toLowerCase().includes(search.toLowerCase()) ||
    p.plugin_id.toLowerCase().includes(search.toLowerCase())
  );

  const TABS = [
    { id: 'plugins', label: 'Installed Plugins', icon: Plug },
    { id: 'hooks',   label: 'Hook Matrix', icon: Gamepad2 },
    { id: 'load',    label: 'Load Plugin', icon: Plus },
  ];

  return (
    <div style={{ padding: '28px 32px', fontFamily: "'Inter', sans-serif", color: '#f1f5f9', height: '100%', overflowY: 'auto' }}>
      {/* Header */}
      <div style={{ marginBottom: 28 }}>
        <h1 style={{ margin: 0, fontSize: 26, fontWeight: 800, background: 'linear-gradient(135deg,#6366f1,#8b5cf6,#ec4899)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
          Plugin SDK
        </h1>
        <p style={{ margin: '6px 0 0', color: '#64748b', fontSize: 14 }}>
          Extend the MECH research platform with custom plugins that hook into every component.
        </p>
      </div>

      {/* Stats bar */}
      <div style={{ display: 'flex', gap: 16, marginBottom: 28, flexWrap: 'wrap' }}>
        {[
          { label: 'Installed Plugins', value: plugins.length,  color: '#6366f1' },
          { label: 'Active',            value: activeCount,     color: '#22c55e' },
          { label: 'Hook Subscriptions',value: totalHooks,      color: '#f59e0b' },
          { label: 'Events Fired',      value: totalEvents,     color: '#ec4899' },
          { label: 'Platform Hooks',    value: ALL_HOOKS.length, color: '#8b5cf6' },
        ].map(s => (
          <div key={s.label} style={{
            flex: '1 1 140px', padding: '14px 20px', borderRadius: 12,
            background: 'rgba(15,23,42,0.7)', border: `1px solid ${s.color}33`,
            boxShadow: `0 0 14px ${s.color}22`}}>
            <div style={{ fontSize: 26, fontWeight: 800, color: s.color }}>{s.value}</div>
            <div style={{ fontSize: 11, color: '#64748b', marginTop: 2 }}>{s.label}</div>
          </div>
        ))}
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 24 }}>
        {TABS.map(t => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            style={{
              padding: '8px 18px', borderRadius: 9, fontWeight: 600, fontSize: 13, cursor: 'pointer',
              transition: 'all 0.2s',
              background: tab === t.id ? 'linear-gradient(135deg,#4f46e5,#7c3aed)' : 'rgba(30,41,59,0.7)',
              border: `1px solid ${tab === t.id ? '#6366f1' : '#1e293b'}`,
              color: tab === t.id ? '#fff' : '#94a3b8'}}
          >{typeof t.icon === 'function' ? <t.icon size={13} style={{ verticalAlign: 'middle', marginRight: 6 }} /> : null}{t.label}</button>
        ))}
      </div>

      {/* Tab: Plugins list */}
      {tab === 'plugins' && (
        <>
          <input
            value={search}
            onChange={e => setSearch(e.target.value)}
            placeholder="Search plugins…"
            style={{
              width: '100%', maxWidth: 420, padding: '8px 14px', borderRadius: 9, fontSize: 13,
              background: '#0f172a', border: '1px solid #334155', color: '#f1f5f9',
              outline: 'none', marginBottom: 20, boxSizing: 'border-box'}}
          />
          {filtered.length === 0
            ? <div style={{ color: '#64748b', textAlign: 'center', padding: 48 }}>No plugins match your search.</div>
            : filtered.map(p => (
                <PluginCard key={p.plugin_id} plugin={p} onToggle={handleToggle} onUnload={handleUnload} />
              ))
          }
        </>
      )}

      {/* Tab: Hook matrix */}
      {tab === 'hooks' && <HookMatrixPanel />}

      {/* Tab: Load plugin */}
      {tab === 'load' && <LoadPluginPanel onLoad={handleLoad} />}

      {/* SDK snippet */}
      {tab === 'plugins' && (
        <div style={{
          marginTop: 32, padding: 20, borderRadius: 12,
          background: 'rgba(15,23,42,0.8)', border: '1px solid #1e293b'}}>
          <div style={{ fontWeight: 700, fontSize: 13, color: '#6366f1', marginBottom: 12, display: 'flex', alignItems: 'center', gap: 6 }}>
            <BookOpen size={14} /> Minimal Plugin Skeleton
          </div>
          <pre style={{ margin: 0, fontSize: 12, color: '#94a3b8', lineHeight: 1.7, overflowX: 'auto' }}>{
`from backend.plugins import MechPlugin, PluginManifest
from backend.plugins.plugin_hooks import HOOK_CAMPAIGN_COMPLETED

class MyPlugin(MechPlugin):
    _MANIFEST = PluginManifest(
        plugin_id="mylab.my_plugin",
        name="My Plugin",
        version="1.0.0",
        author="Me",
        description="Does something cool.",
        hooks=[HOOK_CAMPAIGN_COMPLETED],
    )

    @property
    def manifest(self):
        return self._MANIFEST

    def on_campaign_completed(self, campaign):
        print("Campaign done:", campaign["campaign_id"])

def register():
    return MyPlugin()`
          }</pre>
        </div>
      )}
    </div>
  );
}

