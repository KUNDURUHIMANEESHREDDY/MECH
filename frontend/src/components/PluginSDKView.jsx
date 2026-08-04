import { useState, useEffect, useCallback } from 'react';
import { BarChart3, BookOpen, Building2, Check, ChevronDown, ChevronUp, ClipboardPen, Gamepad2, Microscope, Network, Plug, Plus, Rocket, Search } from 'lucide-react';
import { colors, radii, spacing, typography } from '../design/tokens';

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
  active:   { bg: `${colors.success}26`,  border: colors.success, text: colors.successText },
  inactive: { bg: `${colors.inkMuted48}26`, border: colors.inkMuted48, text: colors.inkMuted48 },
  error:    { bg: `${colors.danger}26`,  border: colors.danger, text: colors.dangerText },
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
      background: active ? `${colors.primaryFocus}33` : `${colors.inkMuted48}1a`,
      border: `1px solid ${active ? colors.primaryFocus : colors.inkMuted80}`,
      color: active ? colors.purpleBorder : colors.inkMuted48,
      cursor: 'help', transition: 'all 0.2s'}}>
      {typeof hook.icon === 'string' ? hook.icon : hook.icon ? <hook.icon size={13} /> : null} {hook.label}
    </span>
  );
}

function PluginCard({ plugin, onToggle, onUnload }) {
  const [expanded, setExpanded] = useState(false);
  const sc = STATUS_COLOR[plugin.status] || STATUS_COLOR.inactive;

  return (
    <div style={{
      background: `${colors.canvas}b3`, border: `1px solid ${sc.border}`,
      borderRadius: 14, padding: '20px 24px', marginBottom: 16,
      boxShadow: `0 0 18px ${sc.border}22`, transition: 'box-shadow 0.3s'}}>
      {/* Header row */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 14, marginBottom: 12 }}>
        <div style={{
          width: 42, height: 42, borderRadius: 10,
          background: `linear-gradient(135deg, ${colors.primaryFocus}, ${colors.purple})`,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          fontSize: 22, flexShrink: 0}}><Plug size={22} /></div>

        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
            <span style={{ fontWeight: 700, fontSize: 16, color: colors.ink }}>{plugin.name}</span>
            <span style={{ fontSize: 11, color: colors.inkMuted48, fontFamily: 'monospace' }}>v{plugin.version}</span>
            <span style={{
              fontSize: 11, fontWeight: 700, padding: '1px 8px', borderRadius: 8,
              background: sc.bg, border: `1px solid ${sc.border}`, color: sc.text}}>{plugin.status.toUpperCase()}</span>
            {plugin.source === 'built-in' && (
              <span style={{ fontSize: 10, color: colors.inkMuted48, padding: '1px 6px', borderRadius: 6, border: `1px solid ${colors.hairline}` }}>BUILT-IN</span>
            )}
          </div>
          <div style={{ fontSize: 12, color: colors.inkMuted48, marginTop: 2 }}>
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
                ? { background: `${colors.warningBorder}1a`, borderColor: colors.warningBorder, color: colors.warningText }
                : { background: `${colors.success}1a`, borderColor: colors.success, color: colors.successText })}}
          >{plugin.status === 'active' ? 'Disable' : 'Enable'}</button>
          {plugin.source !== 'built-in' && (
            <button
              onClick={() => onUnload(plugin.plugin_id)}
              style={{
                padding: '5px 14px', borderRadius: 8, fontWeight: 600, fontSize: 12, cursor: 'pointer',
                background: `${colors.danger}1a`, border: `1px solid ${colors.danger}`, color: colors.dangerText}}
            >Unload</button>
          )}
          <button
            onClick={() => setExpanded(e => !e)}
            style={{
              padding: '5px 10px', borderRadius: 8, fontSize: 12, cursor: 'pointer',
              background: `${colors.primary}1a`, border: `1px solid ${colors.primary}`, color: colors.purpleBorder}}
          >{expanded ? <ChevronUp size={13} /> : <ChevronDown size={13} />}</button>
        </div>
      </div>

      {/* Description */}
      <p style={{ fontSize: 13, color: colors.inkMuted48, margin: '0 0 12px', lineHeight: 1.6 }}>
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
        <span style={{ color: colors.inkMuted48 }}>Events logged: <b style={{ color: colors.purpleBorder }}>{plugin.events_logged}</b></span>
        {plugin.last_triggered && (
          <span style={{ color: colors.inkMuted48 }}>Last triggered: <b style={{ color: colors.inkMuted48 }}>
            {new Date(plugin.last_triggered).toLocaleString()}
          </b></span>
        )}
        {plugin.ui_view && (
          <span style={{ color: colors.inkMuted48 }}>UI Panel: <b style={{ color: colors.success }}>{plugin.ui_view}</b></span>
        )}
      </div>

      {/* Expanded: full hook docs */}
      {expanded && (
        <div style={{
          marginTop: 16, padding: 16, borderRadius: 10,
          background: `${colors.surfacePearl}4d`, border: `1px solid ${colors.hairline}`}}>
          <div style={{ fontWeight: 700, fontSize: 12, color: colors.inkMuted48, marginBottom: 10, textTransform: 'uppercase', letterSpacing: 1 }}>
            Hook Documentation
          </div>
          {plugin.hooks.map(h => (
            <div key={h} style={{ marginBottom: 8 }}>
              <code style={{ color: colors.purpleBorder, fontSize: 12 }}>{h}(payload)</code>
              <p style={{ fontSize: 12, color: colors.inkMuted48, margin: '2px 0 0 12px' }}>{HOOK_DOCS[h]}</p>
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
      background: `${colors.canvas}b3`, border: `1px solid ${colors.hairline}`,
      borderRadius: 14, padding: '20px 24px'}}>
      <div style={{ fontWeight: 700, fontSize: 14, color: colors.ink, marginBottom: 16, display: 'flex', alignItems: 'center', gap: 6 }}>
        <Gamepad2 size={16} /> Platform Hook Integration Matrix
      </div>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead>
            <tr>
              <th style={{ textAlign: 'left', color: colors.inkMuted48, padding: '6px 12px', borderBottom: `1px solid ${colors.hairline}` }}>Hook</th>
              <th style={{ textAlign: 'left', color: colors.inkMuted48, padding: '6px 12px', borderBottom: `1px solid ${colors.hairline}` }}>Platform Component</th>
              <th style={{ textAlign: 'center', color: colors.inkMuted48, padding: '6px 12px', borderBottom: `1px solid ${colors.hairline}` }}>Type</th>
              <th style={{ textAlign: 'left', color: colors.inkMuted48, padding: '6px 12px', borderBottom: `1px solid ${colors.hairline}` }}>Description</th>
            </tr>
          </thead>
          <tbody>
            {ALL_HOOKS.map((h, i) => {
              const isMutating = ['on_experiment_planned', 'on_paper_generated'].includes(h.id);
              return (
                <tr key={h.id} style={{ background: i % 2 === 0 ? `${colors.surfacePearl}4d` : 'transparent' }}>
                  <td style={{ padding: '8px 12px', fontFamily: 'monospace', color: colors.purpleBorder }}>{h.id}</td>
                  <td style={{ padding: '8px 12px', color: colors.inkMuted48 }}>{typeof h.icon === 'string' ? h.icon : h.icon ? <h.icon size={13} style={{ verticalAlign: 'middle', marginRight: 4 }} /> : null} {h.label}</td>
                  <td style={{ padding: '8px 12px', textAlign: 'center' }}>
                    <span style={{
                      padding: '1px 8px', borderRadius: 8, fontWeight: 600, fontSize: 10,
                      ...(isMutating
                        ? { background: `${colors.warning}26`, border: `1px solid ${colors.warning}`, color: colors.warning }
                        : { background: `${colors.primaryFocus}26`, border: `1px solid ${colors.primaryFocus}`, color: colors.purpleBorder })}}>{isMutating ? 'MUTATING' : 'NOTIFY'}</span>
                  </td>
                  <td style={{ padding: '8px 12px', color: colors.inkMuted48 }}>{HOOK_DOCS[h.id]}</td>
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
      background: `${colors.canvas}b3`, border: `1px solid ${colors.hairline}`,
      borderRadius: 14, padding: '20px 24px'}}>
      <div style={{ fontWeight: 700, fontSize: 14, color: colors.ink, marginBottom: 14, display: 'flex', alignItems: 'center', gap: 6 }}>
        <Plus size={14} /> Load External Plugin
      </div>
      <div style={{ fontSize: 12, color: colors.inkMuted48, marginBottom: 12 }}>
        Enter a dotted Python module path (e.g. <code style={{ color: colors.purpleBorder }}>mylab.ioi_extension</code>) or an absolute path to a <code style={{ color: colors.purpleBorder }}>.py</code> file.
      </div>
      <div style={{ display: 'flex', gap: 10 }}>
        <input
          value={path}
          onChange={e => setPath(e.target.value)}
          placeholder="mylab.my_plugin  or  /abs/path/plugin.py"
          style={{
            flex: 1, padding: '8px 14px', borderRadius: 8, fontSize: 13,
            background: colors.canvas, border: `1px solid ${colors.hairline}`, color: colors.ink,
            outline: 'none'}}
        />
        <button
          onClick={handleLoad}
          disabled={!path.trim()}
          style={{
            padding: '8px 20px', borderRadius: 8, fontWeight: 700, fontSize: 13, cursor: 'pointer',
            background: `linear-gradient(135deg, ${colors.primaryFocus}, ${colors.purple})`, border: 'none', color: colors.onDark,
            opacity: path.trim() ? 1 : 0.5}}
        >Load</button>
      </div>
      {status && (
        <div style={{
          marginTop: 10, padding: '8px 12px', borderRadius: 8, fontSize: 12,
          background: status.type === 'success' ? `${colors.success}1a` : `${colors.primary}1a`,
          border: `1px solid ${status.type === 'success' ? colors.success : colors.primaryFocus}`,
          color: status.type === 'success' ? colors.successText : colors.purpleBorder}}>{status.msg}</div>
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
    <div style={{ padding: '28px 32px', fontFamily: "'Inter', sans-serif", color: colors.ink, height: '100%', overflowY: 'auto' }}>
      {/* Header */}
      <div style={{ marginBottom: 28 }}>
        <h1 style={{ margin: 0, fontSize: 26, fontWeight: 800, background: `linear-gradient(135deg, ${colors.primaryFocus}, ${colors.purple}, ${colors.pink})`, WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
          Plugin SDK
        </h1>
        <p style={{ margin: '6px 0 0', color: colors.inkMuted48, fontSize: 14 }}>
          Extend the MECH research platform with custom plugins that hook into every component.
        </p>
      </div>

      {/* Stats bar */}
      <div style={{ display: 'flex', gap: 16, marginBottom: 28, flexWrap: 'wrap' }}>
        {[
          { label: 'Installed Plugins', value: plugins.length,  color: colors.primaryFocus },
          { label: 'Active',            value: activeCount,     color: colors.success },
          { label: 'Hook Subscriptions',value: totalHooks,      color: colors.warning },
          { label: 'Events Fired',      value: totalEvents,     color: colors.pink },
          { label: 'Platform Hooks',    value: ALL_HOOKS.length, color: colors.purple },
        ].map(s => (
          <div key={s.label} style={{
            flex: '1 1 140px', padding: '14px 20px', borderRadius: 12,
            background: `${colors.canvas}b3`, border: `1px solid ${s.color}33`,
            boxShadow: `0 0 14px ${s.color}22`}}>
            <div style={{ fontSize: 26, fontWeight: 800, color: s.color }}>{s.value}</div>
            <div style={{ fontSize: 11, color: colors.inkMuted48, marginTop: 2 }}>{s.label}</div>
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
              background: tab === t.id ? `linear-gradient(135deg, ${colors.primary}, ${colors.purple})` : `${colors.surfacePearl}b3`,
              border: `1px solid ${tab === t.id ? colors.primaryFocus : colors.hairline}`,
              color: tab === t.id ? colors.onDark : colors.inkMuted48}}
          >{t.icon ? <t.icon size={13} style={{ verticalAlign: 'middle', marginRight: 6 }} /> : null}{t.label}</button>
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
              background: colors.canvas, border: `1px solid ${colors.hairline}`, color: colors.ink,
              outline: 'none', marginBottom: 20, boxSizing: 'border-box'}}
          />
          {filtered.length === 0
            ? <div style={{ color: colors.inkMuted48, textAlign: 'center', padding: 48 }}>No plugins match your search.</div>
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
          background: `${colors.canvas}cc`, border: `1px solid ${colors.hairline}`}}>
          <div style={{ fontWeight: 700, fontSize: 13, color: colors.primaryFocus, marginBottom: 12, display: 'flex', alignItems: 'center', gap: 6 }}>
            <BookOpen size={14} /> Minimal Plugin Skeleton
          </div>
          <pre style={{ margin: 0, fontSize: 12, color: colors.inkMuted48, lineHeight: 1.7, overflowX: 'auto' }}>{
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

