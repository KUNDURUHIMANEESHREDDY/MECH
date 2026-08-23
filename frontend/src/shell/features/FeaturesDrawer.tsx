import React, { useState, useRef, useCallback } from 'react';
import { ChevronRight, ChevronDown, Folder, FileText, Brain, Layers, Flame, Zap, Cpu, Activity, Code, BarChart, BookOpen, FlaskConical, Search, FolderTree, MessageSquare, GitBranch, Shield } from 'lucide-react';
import { useWorkspaceStore } from '../../shared/stores/workspace';
import { useUIStore } from '../../shared/stores/ui';
import { Resource, ResourceKind } from '../../shared/types';

interface NavTreeItem {
  id: string;
  label: string;
  kind: ResourceKind;
  icon: React.ReactNode;
  navKey?: string;
  children?: NavTreeItem[];
}

const RESOURCE_TREE: NavTreeItem[] = [
  {
    id: 'group_workspace',
    label: 'Workspace',
    kind: 'workspace',
    icon: <Folder size={15} />,
    children: [
      { id: 'active_investigation', label: 'Overview (Active Investigation)', kind: 'workspace', icon: <Brain size={14} />, navKey: 'active_investigation' },
      { id: 'models', label: 'Models', kind: 'model', icon: <Cpu size={14} />, navKey: 'models' },
      { id: 'datasets', label: 'Datasets & Probes', kind: 'dataset', icon: <Folder size={14} />, navKey: 'dataset_viewer' },
      { id: 'hypotheses', label: 'Hypotheses', kind: 'session', icon: <FlaskConical size={14} />, navKey: 'hypothesis_lab' },
      { id: 'mechanisms', label: 'Mechanisms', kind: 'circuit', icon: <Layers size={14} />, navKey: 'mechanism_builder' },
      { id: 'evidence_graph', label: 'Evidence Graph', kind: 'circuit', icon: <Activity size={14} />, navKey: 'evidence_graph' },
      { id: 'report_mode', label: 'Research Artifacts & Reports', kind: 'paper', icon: <BookOpen size={14} />, navKey: 'report_mode' },
    ],
  },
  {
    id: 'group_models',
    label: 'Model',
    kind: 'model',
    icon: <Brain size={15} />,
    children: [
      { id: 'model_explorer', label: 'Model Explorer', kind: 'model', icon: <Search size={14} />, navKey: 'model_explorer' },
      { id: 'transformer', label: 'Transformer Visualizer', kind: 'model', icon: <Layers size={14} />, navKey: 'transformer' },
      { id: 'attention_heatmap', label: 'Attention Lab', kind: 'model', icon: <Flame size={14} />, navKey: 'attention_heatmap' },
      { id: 'neuron_panel', label: 'Neuron & MLP Explorer', kind: 'neuron', icon: <Zap size={14} />, navKey: 'neuron_panel' },
      { id: 'sae_feature', label: 'SAE Feature Latents', kind: 'model', icon: <Layers size={14} />, navKey: 'sae_feature' },
      { id: 'circuit_explorer', label: 'Circuit Explorer', kind: 'circuit', icon: <Activity size={14} />, navKey: 'circuit_explorer' },
    ],
  },
  {
    id: 'group_analysis',
    label: 'Analysis',
    kind: 'circuit',
    icon: <Flame size={15} />,
    children: [
      { id: 'logit_lens', label: 'Logit Lens & Trajectory', kind: 'model', icon: <Search size={14} />, navKey: 'logit_lens' },
      { id: 'probing_lab', label: 'Probing Lab', kind: 'dataset', icon: <Folder size={14} />, navKey: 'dataset_viewer' },
      { id: 'attribution_dla', label: 'Attribution & DLA', kind: 'model', icon: <BarChart size={14} />, navKey: 'sae_feature' },
      { id: 'causal_tracing', label: 'Causal Tracing', kind: 'experiment', icon: <Activity size={14} />, navKey: 'intervention_lab' },
    ],
  },
  {
    id: 'group_research',
    label: 'Research',
    kind: 'session',
    icon: <FlaskConical size={15} />,
    children: [
      { id: 'experiment_builder', label: 'Experiment Builder', kind: 'experiment', icon: <FlaskConical size={14} />, navKey: 'experiment_builder' },
      { id: 'experiment_monitor', label: 'Experiment Monitor', kind: 'experiment', icon: <Activity size={14} />, navKey: 'experiment_monitor' },
      { id: 'evidence_explorer', label: 'Evidence Explorer', kind: 'evidence', icon: <Search size={14} />, navKey: 'evidence_explorer' },
      { id: 'research_graph', label: 'Research Graph', kind: 'research', icon: <GitBranch size={14} />, navKey: 'research_graph' },
      { id: 'research_notebook', label: 'Research Notebook', kind: 'research', icon: <FileText size={14} />, navKey: 'research_notebook' },
      { id: 'hypothesis_lab', label: 'Hypothesis Lab', kind: 'session', icon: <FlaskConical size={14} />, navKey: 'hypothesis_lab' },
      { id: 'intervention_lab', label: 'Causal Intervention Lab', kind: 'experiment', icon: <Zap size={14} />, navKey: 'intervention_lab' },
      { id: 'evidence_graph_res', label: 'Evidence Graph & Provenance', kind: 'circuit', icon: <Activity size={14} />, navKey: 'evidence_graph' },
      { id: 'mechanism_builder_res', label: 'Mechanism Builder', kind: 'circuit', icon: <Layers size={14} />, navKey: 'mechanism_builder' },
      { id: 'notebook', label: 'Research Notebook', kind: 'note', icon: <FileText size={14} />, navKey: 'experiment_notebook' },
      { id: 'health', label: 'Reproducibility & Validation', kind: 'session', icon: <Activity size={14} />, navKey: 'health' },
      { id: 'ui_adversarial_tester', label: 'UI Adversarial Tester', kind: 'session', icon: <Shield size={14} />, navKey: 'ui_adversarial_tester' },
      { id: 'acceptance_test_verifier', label: 'Acceptance Test Verifier', kind: 'session', icon: <Activity size={14} />, navKey: 'acceptance_test_verifier' },
    ],
  },
  {
    id: 'group_system',
    label: 'System',
    kind: 'workspace',
    icon: <Cpu size={15} />,
    children: [
      { id: 'compute_center', label: 'Compute Center & Jobs', kind: 'workspace', icon: <Cpu size={14} />, navKey: 'compute_center' },
      { id: 'real_time_dag', label: 'Storage & Telemetry DAG', kind: 'workspace', icon: <Activity size={14} />, navKey: 'real_time_dag' },
      { id: 'logging', label: 'Execution Logs', kind: 'session', icon: <Activity size={14} />, navKey: 'logging' },
      { id: 'projects', label: 'Projects', kind: 'workspace', icon: <Folder size={14} />, navKey: 'projects' },
      { id: 'settings', label: 'Configuration & Settings', kind: 'workspace', icon: <Folder size={14} />, navKey: 'settings' },
    ],
  },
];

export const FeaturesDrawer: React.FC = () => {
  const [expanded, setExpanded] = useState<Record<string, boolean>>({
    group_workspace: true,
    group_models: true,
    group_analysis: true,
    group_research: true,
    group_system: true,
  });

  const { openResource } = useWorkspaceStore();
  const setCrumb = useUIStore((s) => s.setCrumb);
  const open = useUIStore((s) => !s.sidebarCollapsed);

  const doubleClickTimerRef = useRef<Map<string, ReturnType<typeof setTimeout>>>(new Map());

  const toggleExpand = (id: string) => {
    setExpanded((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  const handleDoubleClick = useCallback((item: NavTreeItem) => {
    const existingTimer = doubleClickTimerRef.current.get(item.id);
    if (existingTimer) {
      clearTimeout(existingTimer);
      doubleClickTimerRef.current.delete(item.id);
    }

    const res: Resource = {
      id: item.id,
      kind: item.kind,
      label: item.label,
    };
    openResource(res, item.navKey);
  }, [openResource]);

  const handleNavClick = useCallback((item: NavTreeItem) => {
    if (item.navKey) {
      window.location.hash = `#${item.navKey}`;
    }
    if (item.label) {
      setCrumb(item.label);
    }

    if (item.navKey) {
      const existingTimer = doubleClickTimerRef.current.get(item.id);
      if (existingTimer) {
        clearTimeout(existingTimer);
        doubleClickTimerRef.current.delete(item.id);
      }

      const timer = setTimeout(() => {
        doubleClickTimerRef.current.delete(item.id);
        const res: Resource = {
          id: item.id,
          kind: item.kind,
          label: item.label,
        };
        openResource(res, item.navKey);
      }, 200);

      doubleClickTimerRef.current.set(item.id, timer);
    }
  }, [openResource, setCrumb]);

  if (!open) return null;

  return (
    <aside className="shell-navigator" data-testid="features-drawer" style={drawerStyle}>
      <div style={headerStyle}>
        <FolderTree size={14} style={{ color: 'var(--color-primary, #0066cc)' }} />
        <span style={{ fontWeight: 600, fontSize: '12px', letterSpacing: '-0.2px' }}>Features</span>
      </div>

      <div style={{ flex: 1, overflowY: 'auto', padding: '6px 4px' }}>
        {RESOURCE_TREE.map((group) => {
          const isExp = expanded[group.id];
          return (
            <div key={group.id} style={{ marginBottom: '4px' }}>
              <div onClick={() => toggleExpand(group.id)} style={groupItemStyle}>
                {isExp ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
                {group.icon}
                <span style={{ fontWeight: 600 }}>{group.label}</span>
              </div>

              {isExp && group.children && (
                <div style={{ paddingLeft: '16px', display: 'flex', flexDirection: 'column', gap: '2px' }}>
                  {group.children.map((child) => (
                    <div
                      key={child.id}
                      data-testid={child.navKey ? `nav-${child.navKey}` : undefined}
                      onClick={() => handleNavClick(child)}
                      onDoubleClick={() => handleDoubleClick(child)}
                      style={childItemStyle}
                      title={`Open ${child.label} in Canvas`}
                    >
                      {child.icon}
                      <span style={{ flex: 1, textOverflow: 'ellipsis', overflow: 'hidden', whiteSpace: 'nowrap' }}>
                        {child.label}
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </aside>
  );
};

const drawerStyle: React.CSSProperties = {
  position: 'fixed',
  top: 'var(--topbar-h, 44px)',
  left: 0,
  bottom: 0,
  width: 'var(--sidebar-width, 240px)',
  background: 'var(--bg-sidebar, #fafafa)',
  borderRight: '1px solid var(--border, #e0e0e0)',
  display: 'flex',
  flexDirection: 'column',
  userSelect: 'none',
  fontSize: '12px',
  zIndex: 30,
};

const headerStyle: React.CSSProperties = {
  height: '36px',
  display: 'flex',
  alignItems: 'center',
  gap: '8px',
  padding: '0 12px',
  borderBottom: '1px solid var(--border-light, #f0f0f0)',
  color: 'var(--text, #1d1d1f)',
};

const groupItemStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: '6px',
  padding: '6px 8px',
  borderRadius: '6px',
  color: 'var(--text-dim, #333333)',
  cursor: 'pointer',
  fontSize: '12px',
};

const childItemStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: '6px',
  padding: '5px 8px',
  borderRadius: '6px',
  color: 'var(--text, #1d1d1f)',
  cursor: 'pointer',
  fontSize: '12px',
};