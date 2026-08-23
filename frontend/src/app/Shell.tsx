import React, { useState, useEffect } from 'react';
import { Toolbar } from '../shell/toolbar/Toolbar';
import { FeaturesDrawer } from '../shell/features/FeaturesDrawer';
import { WorkspaceCanvas } from '../shell/workspace/WorkspaceCanvas';
import { Inspector } from '../shell/inspector/Inspector';
import { BottomWorkspace } from '../shell/bottom/BottomWorkspace';
import { CommandPalette } from '../shell/toolbar/CommandPalette';
import { DockManager } from '../layout/DockManager';
import { InvestigationHeader } from '../components/research/InvestigationHeader';
import { GlobalQuickSearch } from '../components/research/GlobalQuickSearch';
import { WelcomeOnboardingModal } from '../components/research/WelcomeOnboardingModal';
import { MethodologyDrawer } from '../components/research/MethodologyDrawer';
import { useUIStore } from '../shared/stores/ui';
import { useWorkspaceStore } from '../shared/stores/workspace';

interface ShellProps {
  children?: React.ReactNode;
  pythonStatus?: boolean;
  onOpenCommandPalette?: () => void;
  onLoadModel?: (name: string) => void;
  onRunPrompt?: () => void;
}

const PAGE_LABEL_MAP: Record<string, string> = {
  active_investigation: 'Overview (Active Investigation)',
  mechanism_graph: 'Mechanism Circuit Graph',
  research_queue: 'Research Queue (Next Experiments)',
  comparison_workspace: 'Comparison Workspace',
  ground_truth_benchmarks: 'Known Literature Benchmarks',
  mechanism_diff: 'Mechanism Version Diff',
  mechanism_critic: 'Mechanism Critic & Audit',
  hypothesis_lab: 'Hypothesis Lab & Falsification',
  intervention_lab: 'Causal Intervention Lab',
  evidence_graph: 'Evidence Graph & Provenance',
  mechanism_builder: 'Visual Mechanism Builder',
  model_explorer: 'Model Architecture Explorer',
  compute_center: 'Compute Center & Jobs',
  report_mode: 'Research Artifacts & Reports',
  attention_heatmap: 'Attention Lab',
  neuron_panel: 'Neuron & MLP Explorer',
  sae_feature: 'SAE Feature Explorer',
  circuit_explorer: 'Circuit Explorer',
  logit_lens: 'Logit Lens & Trajectory',
  dataset_viewer: 'Datasets & Probes',
  experiment_notebook: 'Research Notebook',
  experiment_builder: 'Experiment Builder',
  experiment_monitor: 'Experiment Monitor',
  evidence_explorer: 'Evidence Explorer',
  research_graph: 'Research Graph',
  research_notebook: 'Research Notebook',
  ui_adversarial_tester: 'UI Adversarial Tester',
  acceptance_test_verifier: 'Acceptance Test Verifier',
  real_time_dag: 'Storage & Telemetry DAG',
  gpt2explorer: 'GPT-2 Neuron Explorer',
  transformerExplorer: 'Transformer Explorer',
  aiassistant: 'AI Research Assistant',
  explorer: 'Model Explorer',
  gpt2: 'GPT-2 Live',
  transformer: 'Transformer Visualizer',
  workspace: 'Workspace',
  models: 'Models',
  prompts: 'Prompts',
  build: 'Build',
  experiments: 'Experiments',
  sessions: 'Sessions',
  reports: 'Reports',
  analytics: 'Analytics',
  health: 'Reproducibility & Validation',
  settings: 'Settings',
  logging: 'Execution Logs',
  projects: 'Projects',
};

export const Shell: React.FC<ShellProps> = ({
  children,
  pythonStatus = true,
  onOpenCommandPalette,
  onLoadModel,
  onRunPrompt,
}) => {
  const [quickSearchOpen, setQuickSearchOpen] = useState(false);
  const [onboardingOpen, setOnboardingOpen] = useState(false);
  const [methodologyDrawerOpen, setMethodologyDrawerOpen] = useState(false);

  const activityCollapsed = useUIStore((s) => s.activityCollapsed);
  const sidebarCollapsed = useUIStore((s) => s.sidebarCollapsed);
  const setCrumb = useUIStore((s) => s.setCrumb);
  const openPanel = useWorkspaceStore((s) => s.openPanel);

  useEffect(() => {
    const handleHashChange = () => {
      const hash = window.location.hash.replace(/^#/, '');
      if (hash && PAGE_LABEL_MAP[hash]) {
        setCrumb(PAGE_LABEL_MAP[hash]);
        openPanel(hash);
      }
    };
    handleHashChange();
    window.addEventListener('hashchange', handleHashChange);
    return () => window.removeEventListener('hashchange', handleHashChange);
  }, [setCrumb, openPanel]);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        setQuickSearchOpen((prev) => !prev);
      }
      if (e.key === 'Escape') {
        setQuickSearchOpen(false);
        setOnboardingOpen(false);
        setMethodologyDrawerOpen(false);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  const handleSelectSearchResult = (result: any) => {
    if (result.type === 'BENCHMARK') {
      openPanel('ground_truth_benchmarks');
    } else if (result.type === 'MECHANISM') {
      openPanel('mechanism_graph');
    } else if (result.type === 'EXPERIMENT') {
      openPanel('intervention_lab');
    } else if (result.type === 'HYPOTHESIS') {
      openPanel('hypothesis_lab');
    } else {
      openPanel('active_investigation');
    }
  };

  const handleSelectOnboardingAction = (action: string) => {
    if (action === 'NEW_INVESTIGATION') {
      openPanel('hypothesis_lab');
    } else if (action === 'EXPLORE_BENCHMARKS') {
      openPanel('ground_truth_benchmarks');
    } else if (action === 'OPEN_PACKAGE') {
      openPanel('report_mode');
    } else {
      openPanel('active_investigation');
    }
  };

  return (
    <div
      className={`shell-app-container app ${activityCollapsed ? 'activity-collapsed' : ''}`}
      style={{
        display: 'flex',
        flexDirection: 'column',
        height: '100vh',
        width: '100vw',
        overflow: 'hidden',
        background: 'var(--color-canvas-parchment, #f5f5f7)',
        color: 'var(--text, #1d1d1f)',
        fontFamily: 'var(--font, system-ui, sans-serif)',
      }}
    >
      <Toolbar
        onOpenCommandPalette={() => setQuickSearchOpen(true)}
        pythonStatus={pythonStatus}
      />
      <InvestigationHeader />
      <div style={{ flex: 1, display: 'flex', minHeight: 0, overflow: 'hidden' }}>
        <FeaturesDrawer />
        <WorkspaceCanvas>
          <DockManager />
        </WorkspaceCanvas>
        <Inspector />
      </div>
      <BottomWorkspace />

      {/* Global Quick Search (Ctrl+K) */}
      <GlobalQuickSearch
        isOpen={quickSearchOpen}
        onClose={() => setQuickSearchOpen(false)}
        onSelect={handleSelectSearchResult}
      />

      {/* Welcome Onboarding Modal */}
      <WelcomeOnboardingModal
        isOpen={onboardingOpen}
        onClose={() => setOnboardingOpen(false)}
        onSelectAction={handleSelectOnboardingAction}
      />

      {/* 1-Click Methodology Drawer */}
      <MethodologyDrawer
        isOpen={methodologyDrawerOpen}
        onClose={() => setMethodologyDrawerOpen(false)}
      />
    </div>
  );
};
